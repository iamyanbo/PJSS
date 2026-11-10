# Real data: Hudson's Bay lynx–hare pelts with the frozen model (2026-10-26)

**Bottom line.** On the 21 raw annual points, the frozen model (v1.0) fails: it raises an exception. The
bug-fix release v1.0.1 no longer raises, but it returns a degenerate model whose integration diverges. On a spline
resampling of the same points, it returns the Lotka–Volterra structure plus one small logistic term. Its
parameters are within about 10% of the published estimates, and its trajectory fits the data about as well as
the published fits do.

This was a one-dataset, time-boxed check. It is not a benchmark.

Files:
- runner: `benchmarks/real_lynx.py`
- data: `data/real/hudson-bay-lynx-hare.csv`, md5 `64390fbd546037d760409ccf3c878245`
- results: `results_real/lynx_hare.json`, `.log`
- diagnostic run: `results_real/lynx_hare_emptyfix.*`
- the run took under 1 min on one spare Spark core

## Data and reference model

**Data.** Pelts traded annually by the Hudson's Bay Company, 1900–1920, in thousands: 21 points for each of
hare and lynx. The file is from the Stan case study (stan-dev/example-models, `knitr/lotka-volterra`, commit
`e5b7d9e`), which in turn took it from Howard (2009).

**Why this dataset.** It is the standard real-data test for sparse ODE discovery:
- E-SINDy (Fasel et al. 2022, Proc. R. Soc. A) recovers Lotka–Volterra from exactly these 21 points;
- so do BINDy and Bayesian spline learning (arXiv 2210.08095).

**Accepted model.** Lotka–Volterra, with u = hare (prey) and v = lynx (predator):

- du/dt = α·u − β·u·v
- dv/dt = −γ·v + δ·u·v

Reference estimates, and the in-sample R² of each reference fit when integrated from the 1900 observation
(variance-weighted over both species):

| Source | α | β | γ | δ | R² vs the 21 observations |
| --- | --- | --- | --- | --- | --- |
| Stan posterior mean (Carpenter 2018) | 0.55 | 0.028 | 0.80 | 0.024 | 0.849 |
| Howard (2009) | 0.55 | 0.028 | 0.84 | 0.026 | 0.947 |

## Protocol

The protocol was fixed in the runner's docstring before the first run.

- **Model.** `eqdisc.discover(y, dt)` with every default. With D = 2, that means the degree-2 `joint_nd`
  library: 6 terms per equation, 12 candidate terms in total. Lotka–Volterra is inside this library.
- **Arms.**
  - `raw` (primary): the 21 points, dt = 1 year.
  - `spline` (sensitivity): a natural cubic spline through the 21 points, resampled at dt = 0.1 year
    (201 points). Interpolation correlates the errors, and the frozen model was never validated on
    interpolated data.

## Results

| Arm | Version | Outcome | Structure | α | β | γ | δ | R² of integrated model vs data |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw (21 points) | v1.0 (frozen) | **exception** (`IndexError` in `joint_nd.discover`) | – | – | – | – | – | – |
| raw (21 points) | **v1.0.1** | fitted; noise model "white"; not flagged ambiguous (gap 9.1) | **degenerate**: dx0/dt = 55.3·x0 − 3.33·x0·x1, dx1/dt = 0 | (55.3) | (3.33) | – | – | integration diverges (scored as failure) |
| spline (201 points) | v1.0 and v1.0.1 (identical) | fitted; noise model "white"; not flagged ambiguous (gap 57) | LV **+ one extra term** | 0.504 | 0.0256 | 0.892 | 0.0269 | 0.896 |

The v1.0.1 rows come from `results_real/lynx_hare_v101.json` and `.log` (rerun 2026-10-26 on the Spark). The raw
v1.0.1 answer is bit-identical to the earlier crash-patched diagnostic, and the spline answer is bit-identical to
v1.0's. v1.1 cannot change either arm: both are 2-D, and the AR(1) path exists only for D = 3.

The spline arm's model is:

- dx0/dt = +0.5036·x0 − 9.675e-05·x0² − 0.02558·x0·x1
- dx1/dt = −0.8918·x1 + 0.0269·x0·x1

How to read it:
- **Structure.** All four Lotka–Volterra terms are found. The only extra term is a weak logistic
  self-limitation of the hare, −9.7·10⁻⁵·u². At the peak hare count (u ≈ 77), this term is about 1.5% of the
  growth term αu. It means a carrying capacity of about 5,200, far above the data range. In exact-recovery
  terms, though, this is **not** an exact match.
- **Coefficients.** They are within 3–12% of the published estimates: α −8%, β −9%, γ +6 to +11%, δ +3 to +12%.
- **Fit.** The integrated discovered model fits the observations with R² = 0.896, between the Stan fit (0.849)
  and Howard's fit (0.947).

## What went wrong: the crash on short D ≠ 3 series

`joint_nd.discover` can propose an empty support for an equation. It then builds `idx = np.array(())`, which
is a float array, and indexing `A[:, idx]` raises `IndexError`.

- **Where it happens.** On all 21 raw points. ODEBench shows the same failure on every 1-D system (see
  `docs/reports/odebench.md`).
- **Fixed in v1.0.1.** The fix is `np.array(key, dtype=int)`; see `docs/reports/odebench.md` and
  `docs/reports/v101.md`. Frozen v1.0 is unchanged and still crashes here.
- **v1.0.1 on the raw arm.** It returns dx0/dt = 55.3·x0 − 3.33·x0·x1 and dx1/dt = 0. This is exactly what the
  earlier diagnostic patch (`benchmarks/_nd_emptyfix.py`) gave. The model is degenerate:
  - its integration diverges;
  - it is not flagged ambiguous (gap 9.1).

  So fixing the crash does not rescue the 21-point fit. 21 annual samples of a roughly 10-year cycle are
  about 2 points per oscillation phase, far below the N = 500–2,000 regime the model was designed and
  validated on.
- **Spline arm unaffected.** With the patch, the spline arm gives an identical answer, so the patch does
  nothing when no empty proposal occurs.

## Honest assessment

- **Raw data.** Neither version can be used on this dataset as published: v1.0 crashes, and v1.0.1 returns a
  degenerate model. E-SINDy, BINDy and spline
  learning do report Lotka–Volterra from these 21 points. They are built for the low-data regime (bagging,
  priors, smoothing); ours is not.
- **Interpolated data.** With one standard preprocessing step (cubic interpolation, which SINDy-family work
  also commonly uses on this series), the model is close to the accepted model. It gets the right four terms,
  credible parameters and a comparable fit, but adds one spurious weak term. One dataset is anecdotal and should
  be presented as such.
- **Not a test of the noise model.** Pelt counts are not a noisy observation of a true Lotka–Volterra state,
  and Lotka–Volterra itself is only an accepted approximation, not a ground truth.

## Candidates for a stronger real-data test (not obtained in the time box)

| Candidate | Accepted model | In our library? | What is needed |
| --- | --- | --- | --- |
| Electronic Lorenz / jerk / Sprott circuits (op-amp analog computers) | The circuit's design ODE, polynomial | Yes, poly3 D = 3 | A public, densely sampled scope recording. None found ready-made; it could be recorded with a ~$50 op-amp board and a USB scope |
| Chua circuit recordings | Piecewise-linear diode | No; needs an abs/piecewise term | A library extension |
| Double pendulum (IBM "double pendulum chaotic" video, marker tracks) | Lagrangian, with trig and rational mass-matrix terms | No | Angle extraction, plus trig3 with rational terms; or test trajectory prediction only |
| Chemostat predator–prey (Blasius et al. 2020, Nature; Fussmann et al. 2000) | Rosenzweig–MacArthur / chemostat model, rational | No | Rational library; long, dense series are available |
| Lab pendulum or spring–mass with tracked position (many teaching datasets) | Damped oscillator, linear or with sin | Linear: yes. sin: trig3 | A clean source with known physical constants |

The most valuable next step would be an analog-circuit recording of a polynomial chaotic system. It is the
only candidate whose accepted model is exactly in-library, densely sampled and chaotic, which matches the
regime the paper's claims are about.
