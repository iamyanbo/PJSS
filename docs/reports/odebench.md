# ODEBench: frozen model v1.0 and bug-fix release v1.0.1 (2026-10-26)

**Bottom line.** On ODEBench (63 ODEs, D = 1–4, from the ODEFormer paper), the frozen model **v1.0** scores
**37% reconstruction / 23% generalization accuracy** (R² > 0.9, mean over σ = 0–5%, ρ = 0). That is well below
ODEFormer's roughly 62% / 34%. Almost all of the gap has one cause: **v1.0 raises an exception on every 1-D
system** (23 of 63). An empty candidate support is used as a float index in `joint_nd.discover`.

**v1.0.1** fixes that crash and nothing else: `np.array(key, dtype=int)` in `joint_nd`, `joint_discover` and
`joint_colored` (see "The D = 1 crash" below). It was rerun under the same pre-registered protocol:

- **0 exceptions** in 2,016 runs (v1.0: 742).
- On the **1,274 runs that v1.0 completed, v1.0.1 is bit-identical**: every stored field matches, including ξ and
  both R².
- It also reproduces the earlier crash-patched diagnostic arm (`fit_ic0_emptyfix.json`) bit for bit on all 816 of
  its runs. The diagnostic numbers further down are therefore now the numbers of a released version.

| Primary set (fit IC 0, rep 0, 378 runs) | v1.0 (frozen) | **v1.0.1** |
| --- | --- | --- |
| Reconstruction accuracy, all 63 | 36.8 | **68.3** |
| Generalization accuracy, all 63 | 23.3 | **48.9** |
| Exact structure, in library (23) | 24.6 | **50.7** |
| Reconstruction / generalization, D = 1 (23 systems) | 0 / 0 (crash) | 86.2 / 70.3 |
| Reconstruction / generalization, D ≥ 2 (40 systems) | 57.9 / 36.7 | 57.9 / 36.7 (identical) |
| Model exceptions | 139 of 378 | 0 |

- **Reconstruction.** v1.0.1's reconstruction accuracy is in ODEFormer's range (≈ 52–72%).
- **Generalization.** Its generalization accuracy (49%) is above every method in the paper at ρ = 0, except PySR
  at σ = 0 (≈ 51%).
- **Structure.** Exact recovery on the 23 in-library systems doubles, to 50.7%. In the noise-free case, most of
  the remaining "misses" are numerically negligible extra terms.

Everything follows `results_odebench/PREREGISTRATION.md`, written and committed before the main runs. Where an
analysis was added afterwards, it is labelled post-hoc.

Files:
- runner: `benchmarks/odebench_bench.py`
- tables: `benchmarks/odebench_report.py`
- classification: `results_odebench/classes.json`
- per-run records, v1.0: `results_odebench/fit_ic0.json`, `fit_ic1.json`, `fit_ic0_emptyfix.json` (diagnostic)
- per-run records, v1.0.1: `results_odebench_v101/fit_ic0.json`, `fit_ic1.json`
- v1.0.1 tables: `results_odebench_v101/report.md`, from `python benchmarks/odebench_report.py --dir results_odebench_v101`
- run-for-run comparison: `python benchmarks/v101_compare.py all`
- source data, not committed (3.7 MB, md5 in the pre-registration):
  `data/odebench/strogatz_extended.json` from github.com/sdascoli/odeformer @ `c919301`

## Protocol (summary of the pre-registration)

- **Data.** The benchmark's own solutions: 512 points, t ∈ [0, 10], first parameter set. The model is fitted to
  initial condition (IC) 0; IC 1 is held out for generalization. A secondary run swaps the two.
- **Noise.** ODEFormer's multiplicative model, y = x(1 + ξ) with ξ ~ N(0, σ²), at σ ∈ {0, 1, 2, 3, 4, 5}%.
  Rep 0 is the primary realization; reps 1–2 are secondary.
- **Subsampling.** Only ρ = 0. The model needs a uniform grid, so ODEFormer's ρ = 0.5 setting is **not
  supported**.
- **Model.** `eqdisc.discover(y, dt)` with every default:
  - `library="auto"`: the degree-3 library for D = 3, and the degree-2 `joint_nd` library otherwise;
  - `noise="auto"`;
  - no tuning of any kind.
- **Metrics.**
  - **Reconstruction R².** Integrate from the first noisy observation; compare with the clean trajectory.
  - **Generalization R².** Integrate from the other clean IC; compare with its clean trajectory.
  - **R² definition.** Variance-weighted. Exceptions and failed or diverged integrations score 0, as in
    ODEFormer's `r2`.
  - **Accuracy.** The percentage of runs with R² > 0.9.
  - **Structure.** For in-library systems, exact support recovery.
- **Library classification.** Mechanical, by sympy expansion:

  | Class | Systems | By dimension |
  | --- | --- | --- |
  | In library | 23 | 8 with D = 1, 6 with D = 2, 8 with D = 3, 1 with D = 4 |
  | Polynomial but too high a degree | 14 | mostly cubic 2-D oscillators: van der Pol, Duffing, Brusselator… |
  | Non-polynomial | 26 | rational, exp, log, trig, fractional powers, abs |

- **Compute.** Spark, 5 spare cores: fit IC 0 took 14 min (1,008 runs), fit IC 1 15 min, the diagnostic 2 min.
  Median fit time per run: 1.0 s for D = 2, 15 s for D = 3, 7 s for D = 4.

## v1.0 and v1.0.1 side by side

v1.0.1 used the same protocol, data, seeds and code paths. It was rerun on 2026-10-26 on the Spark with 19
workers; fit IC 0 and fit IC 1 took about 2.5 min each. Accuracies are in %, as the mean over the six σ.

| Set | Metric | v1.0 | v1.0.1 |
| --- | --- | --- | --- |
| Fit IC 0, rep 0 (378 runs; primary) | Reconstruction / generalization | 36.8 / 23.3 | **68.3 / 48.9** |
| | Exact, in library | 24.6 | **50.7** |
| Fit IC 0, all 3 reps (1,008 runs) | Reconstruction / generalization | 35.0 / 22.8 | **66.4 / 48.9** |
| | Exact, in library | 22.2 | **48.8** |
| Fit IC 1, all 3 reps (1,008 runs) | Reconstruction / generalization | 39.2 / 22.0 | **71.5 / 41.1** |
| | Exact, in library | 21.3 | **42.0** |
| | D = 1 only: reconstruction / generalization | 0 / 0 | 88.4 / 52.4 |
| All sets | Model exceptions | 742 of 2,016 | **0** |

**v1.0.1, primary set (fit IC 0, rep 0), by σ**

| Subset | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reconstruction, all 63 | 73 | 65 | 70 | 70 | 67 | 65 | **68.3** |
| Reconstruction, in library (23) | 87 | 78 | 83 | 78 | 65 | 70 | 76.8 |
| Reconstruction, non-polynomial (26) | 81 | 69 | 77 | 81 | 81 | 77 | 77.6 |
| Generalization, all 63 | 54 | 49 | 52 | 51 | 43 | 44 | **48.9** |
| Generalization, in library (23) | 87 | 70 | 74 | 70 | 57 | 61 | 69.6 |
| Generalization, non-polynomial (26) | 42 | 38 | 42 | 42 | 35 | 38 | 39.7 |
| Exact, in library (23) | 43 | 48 | 52 | 61 | 52 | 48 | **50.7** |
| Exact, D = 1 in library (8) | 75 | 75 | 75 | 75 | 75 | 75 | 75.0 |

The D ≥ 2 rows are identical to the v1.0 tables below. The full v1.0.1 tables, with all subsets and every
system, are in `results_odebench_v101/report.md`.

**The D ≥ 2 runs that crashed under v1.0** now return an answer, but a poor one: every R² is below 0.9.
- **id 51** (2-D, non-polynomial), σ = 2%, fit IC 0, reps 0–2: a dense 10-term model whose integration leaves the
  data.
- **id 52** (Maxwell–Bloch), σ = 5%, three runs: dx/dt = dy/dt = 0, which are the all-zero equations the fix
  makes valid, plus a spurious z equation.

So the fix changes no D ≥ 2 accuracy figure.

## v1.0 (frozen) primary results: fit IC 0, rep 0 (63 systems × 6 σ = 378 runs)

**Reconstruction accuracy** (% of runs with R² > 0.9)

| Subset | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All 63** | 40 | 33 | 38 | 38 | 37 | 35 | **36.8** |
| D ≥ 2 (40) | 62 | 52 | 60 | 60 | 57 | 55 | 57.9 |
| In library (23) | 52 | 43 | 48 | 43 | 30 | 35 | 42.0 |
| In library, D ≥ 2 (15) | 80 | 67 | 73 | 67 | 47 | 53 | 64.4 |
| Polynomial, degree too high (14) | 29 | 29 | 29 | 29 | 36 | 29 | 29.8 |
| Non-polynomial (26) | 35 | 27 | 35 | 38 | 42 | 38 | 35.9 |

**Generalization accuracy** (held-out IC)

| Subset | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All 63** | 25 | 24 | 27 | 24 | 19 | 21 | **23.3** |
| D ≥ 2 (40) | 40 | 38 | 42 | 38 | 30 | 32 | 36.7 |
| In library (23) | 52 | 39 | 48 | 39 | 30 | 35 | 40.6 |
| In library, D ≥ 2 (15) | 80 | 60 | 73 | 60 | 47 | 53 | 62.2 |
| Polynomial, degree too high (14) | 14 | 21 | 21 | 21 | 21 | 14 | 19.0 |
| Non-polynomial (26) | 8 | 12 | 12 | 12 | 8 | 12 | 10.3 |

**Exact structure** (in-library systems; pre-registered metric)

| Subset | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| In library (23) | 17 | 22 | 26 | 35 | 26 | 22 | 24.6 |
| D = 1 (8) | 0 | 0 | 0 | 0 | 0 | 0 | 0 (crash) |
| D = 2 (6) | 0 | 50 | 33 | 67 | 33 | 17 | 33.3 |
| D = 3 (8) | 38 | 25 | 38 | 50 | 38 | 38 | 37.5 |
| D = 4 (1, SEIR) | 100 | 0 | 100 | 0 | 100 | 100 | 66.7 |

When the model is exact, the median coefficient error is 0.26%.

**Failures.**
- 139 of 378 runs raised exceptions: 138 are the 1-D crash, and 1 is a 2-D system (id 51 at σ = 2%).
- Among the fitted runs, 32 reconstructions and 33 generalizations diverged or failed to integrate.

**Robustness** (secondary sets; tables in `benchmarks/odebench_report.py` output):
- **All three noise realizations, fit IC 0** (1,008 runs): reconstruction 35.0%, generalization 22.8%,
  exact 22.2%.
- **Fit IC 1, generalize to IC 0** (1,008 runs): reconstruction 39.2%, generalization 22.0%, exact 21.3%.
  For D ≥ 2 the figures are 61.8% / 34.6%.
- **Reconstruction from the clean IC instead of the noisy y[0]:** 37.3% instead of 36.8%. The choice of y0
  barely matters.

## Why exact structure is low

**1. The D = 1 crash (v1.0 only).** Every 1-D call fails; see the next section. With v1.0.1, exact recovery on
the 8 in-library 1-D systems is 75% at every σ.

**2. Spurious negligible terms at low noise.** At σ = 0 the frozen model is exact on **0 of 6** in-library 2-D
systems, although all six reconstruct and generalize with R² ≈ 1.00. The extra terms are tiny:

```
id 24, true: dx/dt = y, dy/dt = −2.1x
found:  dx0/dt = −5.5e-07 − 3.3e-06·x0 + 1·x1;  dy/dt = −2.1·x0 + 1.4e-05·x0…
```

With zero noise, the self-calibrated σ collapses, so the selection penalty is negligible against the
integration-level errors in the benchmark solutions (LSODA, rtol 1e-5). Those errors get fitted with extra
terms.

*Post-hoc check, not pre-registered:* count as exact if the support is right after ignoring terms that
contribute less than 1% of the RMS of that equation's discovered right-hand side on the fitted trajectory.

| Subset | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| In library (23) | 57 | 39 | 30 | 35 | 26 | 39 | 37.7 |
| D = 2 (6) | **100** | 67 | 50 | 67 | 33 | 67 | 63.9 |
| D = 3 (8) | 75 | 62 | 38 | 50 | 38 | 50 | 52.1 |

The pre-registered 0% at σ = 0 is therefore a thresholding issue, not a search failure. The model was designed
and validated only at η ≥ 1% observation noise. With v1.0.1, the same post-hoc count gives 65.2% (rep 0, in
library, all D).

The 1% rule was found by looking at ODEBench results. It is therefore **not** part of v1.0.1 or v1.1, and no
headline number uses it. A small-term rule of this kind is future work: it would have to be chosen on dev
archives and confirmed on fresh data.

**3. Short, transient trajectories.** Several in-library systems are not identifiable from a 10-unit window:

| System | What the window shows | Outcome |
| --- | --- | --- |
| id 59, chaotic Rössler | the window covers about 1.5 loops | R² 0 everywhere |
| id 57, Rössler with a stable fixed point | a decaying transient | |
| id 55 / 56, Lorenz | chaotic | reconstruction R² < 0.9 even when the structure is exact, as expected for chaotic trajectories over this window |

## The D = 1 crash (bug in v1.0; fixed in v1.0.1)

- **Mechanism.** In `joint_nd.discover`, `idx = np.array(key)` with `key = ()` gives a float array, and
  `A[:, idx]` raises `IndexError`.
- **Where it occurs.**
  - every D = 1 call;
  - id 51 (a 2-D system) at σ = 2%, in all 3 reps when fitted on IC 0: 3 of 896 2-D runs;
  - the lynx–hare raw data (`docs/reports/real_data.md`).
- **D = 3 too.** The same pattern exists in `joint_discover.discover` (line 337), the poly3 path. It fired on
  3 of 320 D = 3 runs here, all Maxwell–Bloch (id 52) at σ = 5%. None of the ablation or dev reruns in
  `docs/reports/huber_dev.md` hit it: 1,921 runs, 0 errors in any arm.
- **The same pattern elsewhere.**
  - `joint_colored.discover` (the AR(1) path) has the identical line.
  - `joint_squeeze` skips proposals with an empty equation before that line.
  - `flow_discovery` and `joint_biglib` already used integer indices.
- **Fix (v1.0.1).** `idx = np.array(key, dtype=int)` in `joint_nd`, `joint_discover` and `joint_colored`.
  - The empty proposal is now scored by the profiled quadratic as the all-zero model (least squares with zero
    columns). As before, `add()` never fits an empty support.
  - A run that did not crash cannot change, because an empty key reaching that line always raised.
  - This was confirmed empirically (`docs/reports/v101.md`): answers are bit-identical on every completed
    ODEBench run and on the re-runs of the stored test archives.

### Diagnostic: crash patched (made under v1.0; v1.0.1 reproduces it exactly)

`benchmarks/_nd_emptyfix.py` makes `np.array(())` an integer array inside `joint_nd` only. Run on D = 1 and
D = 2, fit IC 0, 3 reps: 0 exceptions in 816 runs. On the D = 2 runs, the patched and frozen answers differ
only on the 3 runs that crashed.

| D = 1 (23 systems), patched | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reconstruction, all 23 | 91 | 88 | 88 | 86 | 81 | 81 | 86.0 |
| Reconstruction, in library (8) | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Generalization, all 23 | 78 | 75 | 71 | 71 | 68 | 67 | 71.7 |
| Exact, in library (8) | 75 | 83 | 75 | 75 | 75 | 75 | 76 |

**Composite.** The patched D = 1 runs are combined with the frozen D ≥ 2 runs, so all 63 systems are covered.
It was a diagnostic when it was made; v1.0.1 gives exactly these numbers.

| Rep 0 | σ = 0 | 1% | 2% | 3% | 4% | 5% | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reconstruction | 73 | 65 | 70 | 70 | 67 | 65 | **68.3** |
| Generalization | 54 | 49 | 52 | 51 | 43 | 44 | **48.9** |
| Exact, in library (23) | 43 | 48 | 52 | 61 | 52 | 48 | 50.7 |

## Comparison with the ODEFormer paper

Values are read off Fig. 4b (reconstruction) and Fig. 5 (generalization) of d'Ascoli et al., *ODEFormer*,
ICLR 2024 (arXiv:2310.05573), ρ = 0 panels. They are **approximate (±3 points)**, because the paper gives no
table. Each range runs from σ = 5% to σ = 0.

| Method | Reconstruction, ρ = 0 | Generalization, ρ = 0 |
| --- | --- | --- |
| ODEFormer | ≈ 52–72% | ≈ 32–36% |
| ODEFormer (opt; BFGS parameter refinement) | ≈ 55–76% | ≈ 35–42% |
| PySR | ≈ 60–82% | ≈ 19–51% |
| FFX | ≈ 49–72% | ≈ 12–33% |
| SINDy (poly) | ≈ 46–65% | ≈ 8–22% |
| Ours, v1.0 (frozen), all 63 | 33–40% (mean 36.8) | 19–27% (mean 23.3) |
| Ours, v1.0, D ≥ 2 only (40 systems; a different subset) | 52–62% (mean 57.9) | 30–42% (mean 36.7) |
| **Ours, v1.0.1, all 63** | **65–73% (mean 68.3)** | **43–54% (mean 48.9)** |

**Not like-for-like.** The comparison is uneven in both directions:

- **In the baselines' favour:**
  - Each baseline got a per-trajectory hyperparameter search: train on the first 70%, score on the last 30%.
  - Each picked its final equation from its Pareto front by accuracy.
  - ODEFormer itself picks among beam candidates by reconstruction R².
- **In ours:**
  - We return a single model with frozen defaults.
  - We use the 512 points per trajectory in `strogatz_extended.json`. ODEFormer's `evaluate.py` loads the same
    file; we did not check whether it is subsampled before inference. The paper's training trajectories have
    50–200 points.
  - We skip ρ = 0.5, which ODEFormer handles and we cannot.
- **Uncertain noise draw.** The noise realization differs, and whether ODEFormer's y0 is noisy is not certain.
  Our y0 choice moves the result by under 1 point.

The honest summary for the paper:

- **v1.0 as frozen.** The model is not competitive on ODEBench, because it crashes on 1-D systems.
- **v1.0.1 (crash fixed, otherwise bit-identical).**
  - Its reconstruction accuracy is in ODEFormer's range.
  - Its generalization accuracy (about 49%) is above every method in the paper except PySR at σ = 0 (about 51%).
    ODEFormer (opt) scores about 35–42%.
  - The generalization advantage comes from the in-library systems, where it recovers the actual equations.
  - The fix was made after the bug was found on this benchmark. But it is a crash fix with nothing to tune, and it
    changes no answer on any run that v1.0 completed.
- **Out of library** (non-polynomial systems), generalization is only 10–13%, as expected for a polynomial
  library. The ODEFormer comparison should therefore be presented as library-dependent.

## Caveats

- **Only one noise draw per cell in the primary set.** Reps 1–2 move the means by 1–2 points.
- **Coarse classification.** It does not credit a library that approximates a smooth non-polynomial field
  well. Non-polynomial reconstruction (36%) is close to the in-library rate, because reconstruction on a
  single trajectory is a weak test.
- **ODEFormer values are approximate**, read from a figure.
- **The post-hoc 1% threshold is not part of any version**, and no headline uses it. It is future work.
- **v1.0.1 was made after seeing ODEBench.** The pre-registered headline is v1.0's. v1.0.1's numbers come from the
  same pre-registered protocol and are reported next to v1.0's, not in place of them.
- **v1.1 was not rerun here.** v1.1 (Gaussian loss on the AR(1) path; `docs/reports/v11_coloured.md`) cannot
  change any ODEBench answer with D ≠ 3, because there is no AR(1) path there. D = 3 runs that route to AR(1)
  were not rerun.
