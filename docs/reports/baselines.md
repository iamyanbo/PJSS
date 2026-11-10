# Standard baselines vs `eqdisc.discover` (2026-10-25)

Every number here is exact support recovery (`eqdisc.core.metrics`: the estimated nonzero pattern equals the
true one in every equation). Each baseline ran on the same runs as our frozen final model, and we compare them
run by run. The final-model answers come from the frozen test passes (`results_dysts/test_final.json`,
`results_final/test_confirm.json`) and were not rerun. Reproduce with `benchmarks/baselines_bench.py` (runner)
and `benchmarks/baselines_report.py` (tables). Per-run records are in `results_baselines/`. The job files that
ran on the DGX Spark (6 cores) are `results_baselines/{tune,tune2,job,job2}.sh`.

## Headline

No standard baseline comes close to our frozen final model on any set. That holds even with an oracle,
per-run threshold. The strongest truth-free baselines are:
- **PySR:** 0.444 against our 0.929 on the same 126 dysts runs (61 wins / 0 losses).
- **SFD SINDy with STLSQ:** 0.306 against 0.891 on 880 runs (515/0).

The best *upper bound* among the baselines is MIOSR with the true number of terms in every equation (0.759).
Even that loses to our truth-free model, 123 wins / 7 losses. On the hard 40, every baseline under its
truth-free rule scores 0/40; ours scores 28/40.

## Paper table

Exact support recovery. **CV** and **AICc** are truth-free threshold rules. **tuned** (one threshold per set,
chosen with the truth) and **oracle** (best per run) are not, and are given as generous and upper-bound
references. "Ours" is `eqdisc.discover` at its frozen defaults on the *same* runs. W/L counts the runs that only
ours / only the baseline's CV pick got exactly right; for PySR and ODEFormer it uses their own default pick.
Runtime is the median wall time in seconds per run on one core, including the full threshold path and CV.

| Baseline | Set (runs) | CV | AICc | tuned | oracle | s / run | Ours | W/L vs ours (sign p) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SINDy STLSQ (SFD) | dysts7 (880) | 0.306 | 0.301 | 0.372 | 0.491 | 1.0 | **0.891** | 515/0 (2e-155) |
| SINDy SR3-L0 (SFD) | dysts7 (880) | 0.174 | 0.193 | 0.160 | 0.268 | 0.5 | **0.891** | 631/0 |
| Weak SINDy STLSQ | dysts7 (880) | 0.188 | 0.057 | 0.278 | 0.370 | 0.5 | **0.891** | 621/2 |
| Weak SINDy SR3-L0 | dysts7 (880) | 0.174 | 0.058 | 0.219 | 0.347 | 0.7 | **0.891** | 632/1 |
| Weak SINDy MIOSR (Gurobi) | dysts7 (880) | 0.073 | 0.020 | – | 0.759† | 4.0 | **0.891** | 720/0; vs oracle† 123/7 |
| E-SINDy (SFD) | dysts21 (294) | 0.293 | 0.231 | 0.374 | 0.483 | 13.8 | **0.874** | 171/0 |
| Weak E-SINDy | dysts21 (294) | 0.163 | 0.058 | 0.272 | 0.398 | 7–8 | **0.874** | 209/0 |
| PySR (60 s / equation) | dysts49 (126) | 0.444‡ | – | – | 0.778‡ | 180 | **0.929** | 61/0 (9e-19) |
| ODEFormer (pretrained) | dysts21 (294) | 0.007‡ | – | – | 0.020‡ | 63 | **0.874** | 255/0 |
| **eqdisc.discover (ours)** | dysts7 (880) | **0.891** | | | | 3.8 | | |

† MIOSR oracle = the true per-equation term count; this is the oracle-sparsity bound.
‡ PySR: its `model_selection="best"` pick / any expression on the Pareto front.
‡ ODEFormer: its top candidate / any beam candidate.

**By noise level and dimension**, for the best truth-free baseline of each family and for ours on the same runs:

| Method (rule) | Set | 1 % | 5 % | 10 % | 20 % | D = 3 | D = 4–6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SINDy STLSQ (SFD), CV | dysts7 | 0.473 | 0.364 | 0.236 | 0.150 | 0.334 | 0.197 |
| SINDy STLSQ (SFD), oracle | dysts7 | 0.736 | 0.550 | 0.400 | 0.277 | 0.531 | 0.339 |
| Weak MIOSR, true-k oracle | dysts7 | 0.923 | 0.841 | 0.714 | 0.559 | 0.782 | 0.672 |
| **Ours** | dysts7 | **0.964** | **0.936** | **0.886** | **0.777** | **0.897** | **0.869** |
| E-SINDy (SFD), CV | dysts21 | 0.486 | 0.338 | 0.233 | 0.110 | 0.329 | 0.150 |
| **Ours** | dysts21 | **0.946** | **0.959** | **0.849** | **0.740** | **0.880** | **0.850** |
| PySR, default | dysts49 | 0.656 | 0.531 | 0.355 | 0.226 | 0.485 | 0.296 |
| PySR, Pareto oracle | dysts49 | 0.906 | 0.906 | 0.710 | 0.581 | 0.788 | 0.741 |
| **Ours** | dysts49 | **1.000** | **0.938** | **0.935** | **0.839** | **0.929** | **0.926** |

**Hard 40** (20 % noise, N = 2000; exact out of 10 per system):

| Method | Lorenz | Chen | Rössler | Rabinovich–F. | All 40 |
| --- | --- | --- | --- | --- | --- |
| Every pysindy baseline (STLSQ, SR3, E-SINDy, MIOSR; SFD and weak), CV or AICc | 0 | 0 | 0 | 0 | 0 |
| Best pysindy oracle (weak STLSQ, raw, h 20) | 0 | 4 | 4 | 0 | 8 |
| Weak E-SINDy oracle | 0 | 4 | 2 | 0 | 6 |
| Weak MIOSR, true-k oracle | 0 | 0 | 0 | 0 | 0 |
| PySR default / Pareto oracle | 0 / 0 | 0 / 6 | 0 / 5 | 0 / 0 | 0 / 11 |
| ODEFormer default / beam oracle | 0 | 0 | 0 | 0 | 0 / 0 |
| DEDi (from `docs/WRITEUP.md`) | 8 | 7 | 6 | 0 | 21 |
| ODR-BINDy Python port, σ_y = 5×10⁻³ (from `docs/WRITEUP.md`) | 6 | 0* | 8 | 0 | 14 |
| **Ours (final)** | **10** | **10** | **8** | **0** | **28** |

\* The port's Chen score is 7/10 with a 1,000-iteration cap. See `docs/WRITEUP.md`.

## Reading

- **The bottleneck is threshold choice, not the features.** The per-run oracle is 1.4–2× the best truth-free rule
  (10× for MIOSR). AICc is close to useless on weak features (0.02–0.06), because overlapping test functions make
  the residuals correlated and it keeps adding terms. This is the same failure the writeup reported for BIC.
  Blocked CV is the better rule on weak features. On SFD features the two rules are roughly tied: AICc wins
  at 1 % noise and CV at higher noise.
- **MIOSR shows this most clearly.** With the true number of terms in every equation it reaches 0.759. With
  any truth-free choice of k it reaches 0.02–0.07. Our method is truth-free and still beats that oracle 123/7.
  The oracle has an edge only at 1 % noise (0.923 against 0.964 for ours), and it falls behind as noise grows.
- **Weak form alone does not rescue SINDy on dysts.** Under CV, weak STLSQ/SR3 are worse than SFD at every
  noise level. Only the weak STLSQ oracle edges ahead at 20 % noise (0.295 against 0.277).
- **PySR is the strongest truth-free competitor.** Its Pareto front often contains the right model (0.778).
  Its own complexity/accuracy pick finds it in only 0.444 of runs. It costs about 50× our runtime (3 min per
  run at a 60 s/equation budget), and a bigger budget might help it.
- **ODEFormer is not competitive for exact recovery on these systems.** Most top candidates contain
  non-polynomial pieces (sin, rational terms): 209 of 294 are outside the library, and the mean F1 of its
  picks is 0.45. As a sanity check, it also fails on *clean, noise-free* Lorenz and Rössler (200 samples), so
  this is the model's limit and not our wrapper. These 3-D chaotic systems are outside what it handles well.
- **Coverage caveat.** E-SINDy, PySR and ODEFormer ran on balanced subsamples (294 / 126 / 294 runs) for
  budget reasons, so their rows compare against ours on those runs. Our final model scores 0.874 (dysts21) and
  0.929 (dysts49) on them, against 0.891 on dysts7. dysts49 happens to be easier for both sides; the
  comparisons are paired run for run.
- **Incidents.** The Spark rebooted at about 21:03 (cause unknown) during the ODEFormer dysts21 run. 250 of 294 records were kept from the checkpoint, and the remaining 44 runs were
  redone with `--resume` (`results_baselines/job3.sh`). The answers are deterministic given the run index.

## Appendix: dev sweep (`data/instances.npz`, 160 runs)

Exact rate per config; this is what the per-rule configs were chosen from. "global" is the dev-set tuned
threshold.

| Config | AICc | CV | global | oracle |
| --- | --- | --- | --- | --- |
| fd_stlsq raw w5 / w11 / w21 | **0.431** / 0.081 / 0.000 | 0.338 / **0.444** / 0.350 | 0.487 / 0.556 / 0.512 | 0.581 / **0.644** / 0.588 |
| fd_stlsq rms w5 / w11 / w21 / w41 | 0.344 / 0.056 / 0 / 0 | 0.325 / 0.425 / 0.369 / 0.106 | 0.406 / 0.456 / 0.487 / 0.375 | 0.481 / 0.512 / 0.519 / 0.431 |
| fd_sr3 rms w5 / w11 / w21 / w41 | **0.325** / 0.125 / 0 / 0 | 0.244 / **0.275** / 0.188 / 0.062 | 0.244 / 0.269 / 0.300 / 0.244 | 0.338 / 0.394 / **0.463** / 0.294 |
| fd_sr3 raw w5 / w11 / w21 | 0.287 / 0.131 / 0 | 0.212 / 0.231 / 0.200 | 0.281 / 0.356 / 0.381 | 0.319 / 0.381 / 0.438 |
| weak_stlsq rms h10 / h20 / h40 | **0.075** / 0.025 / 0.006 | **0.287** / 0.263 / 0.125 | 0.456 / 0.475 / 0.444 | 0.469 / 0.500 / 0.481 |
| weak_stlsq raw h10 / h20 / h40 | 0.075 / 0.013 / 0.013 | 0.244 / 0.237 / 0.106 | 0.412 / 0.444 / 0.444 | 0.525 / **0.569** / 0.531 |
| weak_sr3 rms h10 / h20 / h40 | **0.056** / 0 / 0 | **0.219** / 0.212 / 0.062 | 0.419 / 0.412 / 0.356 | 0.469 / **0.487** / 0.438 |
| weak_sr3 raw h10 / h20 / h40 | 0.056 / 0.013 / 0 | 0.212 / 0.156 / 0.106 | 0.400 / 0.406 / 0.375 | 0.431 / 0.438 / 0.400 |
| weak_miosr rms h10 / h20 | 0.013 / 0 | **0.050** / 0.031 | – | 0.738 / 0.731 |

Full per-config test tables, including every config × rule that ran: `results_baselines/report_all.txt`. The
paper tables are generated by `python benchmarks/baselines_report.py --markdown` (→
`results_baselines/report.md`).

## Evaluation sets

- **dysts7.** Every 7th run of `data/dysts_test.npz`: 880 runs and all 77 systems. The archive cycles
  (noise, N) with period 8 (noise = index mod 4, N = (index div 4) mod 2). Stride 7 is coprime with 8, so each
  of the 8 cells has exactly 110 runs. D = 3 has 696 runs and D = 4–6 has 184.
- **dysts21 / dysts49.** Every 3rd / 7th run of dysts7, i.e. archive strides 21 and 49. Both are coprime with 8,
  so the cells stay balanced (294 and 126 runs). These are for the costly baselines: ODEFormer and E-SINDy use
  dysts21, PySR uses dysts49.
- **hard40.** `data/hybrid_confirm.npz` at the indices in `results_final/hard40_idx.json`: Lorenz, Chen,
  Rössler and Rabinovich–Fabrikant at 20 % noise, N = 2000, 10 runs each. This is the same set as the DEDi and
  ODR-BINDy comparisons in `docs/WRITEUP.md`. These systems were chosen *because* they are hard for us.
- **dev**, used only for tuning. Every 5th run of `data/instances.npz`: 160 runs, 2 per system × noise × N cell.
  This is our own dev archive, the same one our defaults were chosen on. No baseline setting was chosen on a
  test archive. The one exception is the "tuned" row, which is labelled as such below.

## How each baseline was configured

All the SINDy-family baselines use **pysindy 2.1.0's own components**: `PolynomialLibrary`,
`SmoothedFiniteDifference`, `WeakPDELibrary`, `STLSQ`, `SR3`, `EnsembleOptimizer` and `MIOSR` (Gurobi 13.0.3).
The library matches ours: all monomials of degree ≤ 3 (20 per equation) for D = 3, and degree ≤ 2 for D ≥ 4.
Feature columns are matched to the truth's terms by their exponent tuples.

**Features**
- *SFD (derivative-based).* `SmoothedFiniteDifference` (Savitzky–Golay, polyorder 3, window w) gives
  derivatives. The library is evaluated on the smoothed states, which is what `ps.SINDy` does with
  `save_smooth=True`.
- *Weak.* `WeakPDELibrary(PolynomialLibrary, K = max(3F, N/(h/2)), H_xt = h·dt, p = 4)`, with the target from
  `lib.convert_u_dot_integral`. This is the same construction as `ps.SINDy` with a weak library, and the same
  K/H rule as `benchmarks/bench.py`.
- *Preprocessing (`scale`).*
  - `raw`: original units, i.e. plain pysindy usage.
  - `rms`: each state divided by its RMS, with time rescaled so the derivative RMS is 1. Supports and
    coefficients are mapped back exactly.

  Both were swept on dev. `raw` matters for STLSQ: in raw units the high-degree monomials are large, so their
  coefficients are small and get thresholded first. With RMS scaling, STLSQ cannot recover Lorenz even at 1 %
  noise (dev oracle 0.51 against 0.64 raw). We found this on dev and fixed it before any test run.

**Optimisers along a threshold path**
- `STLSQ(threshold=th)` with pysindy defaults (alpha 0.05, unbias).
- `SR3(regularizer="L0", reg_weight_lam=th²/2, relax_coeff_nu=1, max_iter=200)`, which hard-thresholds at th.
- E-SINDy: `EnsembleOptimizer(STLSQ(threshold=th), bagging=True, n_models=20)`, with pysindy's default
  aggregation (median of the bagged coefficients). The seed is the run index.
- The grid is th ∈ geomspace(1e-3, 10, 41) for `rms` and geomspace(1e-4, 1e2, 55) for `raw`. One threshold is
  shared by all equations, as in pysindy.

**Threshold choice.** This is the crux. There are four rules:
- **CV** (truth-free): 5-fold contiguous-block cross-validation on the regression rows. Each fold refits least
  squares on the support found on the training folds and scores mean_j val-MSE_j / var(target_j). Ties go to
  the sparser model.
- **AICc** (truth-free): Σ_j [N log(RSS_j/N) + 2k_j + 2k_j(k_j+1)/(N−k_j−1)], computed on least-squares refits
  of each support, in the style of Mangan et al. (2017).
- **tuned** (not truth-free): one threshold for the whole evaluation set, chosen on that set with the truth.
  This is generous, and it is how `docs/WRITEUP.md` reported its earlier baselines.
- **oracle** (upper bound): the best threshold per run.

**MIOSR** (`weak_miosr`). `MIOSR(target_sparsity=k, alpha=1e-6, normalize_columns=True)` is fitted one equation
at a time for k = 1..8 on the weak features (`rms` scaling; with `normalize_columns` the scaling only affects
conditioning). k is chosen per equation by CV or AICc. The oracle is "the true support appears on the k-path of
every equation", which is the oracle-sparsity upper bound. h = 10 was picked by dev CV (h = 10 and h = 20 give
dev CV 0.050 and 0.031, oracle 0.74 and 0.73). Gurobi runs with 1 thread and a 10 s per-MIQP timeout (the
pysindy default).

**Config per rule, fixed in advance from dev.** Each truth-free rule uses the config (scale × window or h) that
scored best on dev *under that same rule*. "tuned" and "oracle" use the config with the best dev oracle.
E-SINDy reuses its STLSQ counterpart's configs, because it was too costly to sweep. The full dev sweep is
`results_baselines/dev/` (26 configs) and is summarised in the appendix.

| Method | CV config | AICc config | tuned / oracle config |
| --- | --- | --- | --- |
| SINDy STLSQ (SFD) | raw, w 11 | raw, w 5 | raw, w 11 |
| SINDy SR3-L0 (SFD) | rms, w 11 | rms, w 5 | rms, w 21 |
| E-SINDy (SFD) | raw, w 11 | raw, w 11 | raw, w 11 |
| Weak SINDy STLSQ | rms, h 10 | rms, h 10 | raw, h 20 |
| Weak SINDy SR3-L0 | rms, h 10 | rms, h 10 | rms, h 20 |
| Weak E-SINDy | rms, h 10 | rms, h 10 | raw, h 20 |
| Weak MIOSR | rms, h 10 | rms, h 10 | rms, h 10 |

**PySR** (2.7.2, SymbolicRegression.jl backend)
- **Input:** one regressor per equation, fitted on SFD-smoothed states and derivatives (w = 11, `rms` scaling
  so that the constants are O(1)). 1,000 evenly spaced rows.
- **Search:** operators `+ - *`, no unary operators, maxsize 35, 15 populations. Each equation gets a fixed
  wall budget of 60 s (`timeout_in_seconds`, with niterations unbounded), so a D = 3 run costs about 3 min.
  The search is serial and deterministic (seed = run index + equation).
- **Answer:** PySR's own `model_selection="best"`.
- **Mapping to a support:** `sympy.expand` of the selected expression, then `sympy.Poly` in the state symbols.
  Every monomial with a nonzero coefficient counts as a term, with no extra pruning. A monomial outside the
  library (degree > 3, or > 2 for D ≥ 4) makes the run non-exact.
- **Oracle:** "some expression on each equation's Pareto front has exactly the true support".

**ODEFormer** (`odeformer` 0.1.7 with torch 2.14 CPU, pretrained checkpoint downloaded by the package)
- **Install fixes:** the package pins sympy 1.11, which conflicts with torch 2.x, so it was installed
  `--no-deps` with numpy < 2 (the package uses `np.infty`). The checkpoint is loaded with
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, because it is a pickled model object from the authors' own download
  link.
- **Inference:** CPU, 1 thread, `beam_size=50` and `beam_temperature=0.1` (the README settings), with the
  package's own rescaling and candidate sorting.
- **Input:** 200 evenly spaced samples of the trajectory. ODEFormer was trained on short trajectories, and
  the package default (`max_input_points=10000`) would feed all N points in one bag: a 1,000-point probe took
  88 s against 20–35 s for 200 points. This choice was not tuned.
- **Answer:** the top-ranked candidate, mapped with `sympy.expand` as for PySR. Expressions with sin, exp,
  division and so on are out of the library and therefore non-exact.
- **Oracle:** "any beam candidate has the exact support".

**Fairness notes**
- **Baselines get more than we do in places.** They get the tuned and oracle rows; we report our method only
  at its frozen defaults. They also get per-rule config selection.
- **AICc and CV are standard and truth-free, but not optimal.** A different selection rule could do better.
  The oracle rows bound what any threshold-selection rule could achieve with these features. Note that MIOSR's
  per-equation oracle is more generous than the shared-threshold oracle of STLSQ and SR3.
- **Runtime is median wall seconds per run, on 1 core of the Spark.** For the path methods it covers the whole
  path plus CV. For our method it is the frozen run's time, which was single-threaded on the same machine.
- **PySR and ODEFormer are general-purpose.** Neither is restricted to the polynomial library. That makes exact
  support recovery a harsh metric for them, so their Pareto-front and beam oracles are reported too.

## ODR-BINDy: official MATLAB code

**Not run; skipped after about 1 h of effort. The Python port (`benchmarks/odr_bindy.py`) remains unvalidated
against MATLAB.**

- **No MATLAB on the Spark.** GNU Octave 9.4 was installed without root (micromamba, conda-forge
  linux-aarch64), together with the Forge packages struct, statistics 1.6.5 and optim. optim provides
  `lsqnonlin`.
- **The official code needs MATLAB-only features.** We shimmed `optimoptions`, `pagemtimes` and `bootstrp`
  (`results_baselines/odr_octave/shims/`) and patched `sum(·,'all')`. The next failure was sparse implicit
  expansion in `Jsq_xp.m` (`5982×6000 ./ 5982×1`); each fix exposed another incompatibility.
- **Even a working Octave run would not validate the port.** The official code calls `lsqnonlin` with
  `Algorithm='trust-region-reflective'` and analytic sparse Jacobians. Octave's optim `lsqnonlin` is its own
  Levenberg–Marquardt and ignores that option. A hard-40 run has 6,060 unknowns and about 1,700 `lsqnonlin`
  fits, and the port already needs about 3 h per run with a sparse solver. An Octave run would therefore
  substitute the solver, which is exactly the part of the port that is in question (see the cap-1000 note in
  `docs/WRITEUP.md`), and would be far slower.
- **What would settle it:** a MATLAB licence with the Optimization Toolbox (e.g. the university's), running
  `Lorenz.m`-style drivers on 5–10 hard-40 runs. The probe scripts and logs are in
  `results_baselines/odr_octave/`.
