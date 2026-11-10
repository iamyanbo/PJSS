# Accuracy vs runtime against fast baselines, on the 80 head-to-head runs (2026-10-28)

**Status: final.** This answers the reviewer's point: being 340× cheaper than DEDi, a slow method, does not show
that we sit on the speed/accuracy frontier against *fast* methods, WSINDy above all. Every method below ran on
exactly the 80 runs of the head-to-head sample (`results_h2h/order.json`, 16 per condition) with the same 20-term
degree-3 monomial library. Each picks its own sparsity **without the truth**: no oracle thresholds, and no
setting was tuned on these runs. Scores come from `eqdisc.core.metrics`. Numbers only, no significance tests.

Reproduce: `benchmarks/fast_baselines.py` (runner), `benchmarks/fast_baselines_report.py` (tables, which also
write `results_fast/report.md`). Per-method records are in `results_fast/*.json`, and the Spark job files are
`results_fast/job*.sh`.

## Bottom line

- **On the pooled frontier, every fast baseline gets less than half our exact count.** Exact recovery out of
  80, against median seconds per run on one core:

  | Method | Exact | Mean F1 | Median s/run | Exact per our 66 |
  | --- | --- | --- | --- | --- |
  | WSINDy + MSTLS (authors' code, default settings) | 24/80 | 0.728 | 0.27 | 0.36 |
  | In-repo weak χ² (`select_k`, chi2s) | 36/80 | 0.909 | 0.48 | 0.55 |
  | Best pysindy rule (SFD STLSQ or SFD E-SINDy, CV) | 17/80 | 0.67–0.68 | 3.1 / 60 | 0.26 |
  | ARGOS (R package, lasso, B = 200) | 17/80 | 0.654 | 126 | 0.26 |
  | **Ours v1.2** | **66/80** | **0.949** | **14.4** | 1 |
  | DEDi, full grid (BIC / val) | 61 / 55 | 0.86 / 0.85 | 5,930 (GPU for most runs) | 0.92 / 0.83 |

  Only three methods are not dominated, meaning no faster method has as many exact runs: WSINDy (0.27 s,
  24/80), the in-repo weak χ² baseline (0.48 s, 36/80) and ours (14.4 s, 66/80).
- **We are not faster than the fast methods.** WSINDy is about 50× cheaper per run (0.27 s against 14.4 s). Our
  claim has to be "far more accurate at a few seconds' extra cost", not "faster". No fast baseline comes close to
  our accuracy at any runtime. The pysindy E-SINDy variants and ARGOS are slower than ours *and* far less
  accurate.
- **By condition**, our count against the best fast baseline in each:
  - hard: 15/16 against 1/16 (WSINDy). Every pysindy rule, χ² and ARGOS get 0/16.
  - white: 14 against 9 (χ²).
  - coloured: 10 against 7 (χ²).
  - t3: 14 against 9 (χ²).
  - short: 13 against 11 (χ²) and 10 (WSINDy).

  The fast methods do best on short (N = 500, 5–10 % noise) and worst on hard (20 % noise).
- **WSINDy fails by keeping too many terms, not too few.** Over the 80 runs its median is 4 false-positive
  terms and 0 missed terms (median 11 terms; the true models have 5–12). The MSTLS loss stops too early on noisy
  chaotic data. Its chosen λ is inside the grid (median 0.027) on 239 of 240 equations, so this is not a grid-edge
  artefact.
- **Coefficient accuracy.** On the runs each method gets exact, our median relative coefficient error is 0.0077.
  WSINDy's is 0.019, χ²'s 0.017, and the pysindy rules' 0.014–0.025. DEDi's full grid is 0.008.

## Pooled table (80 runs)

Columns: exact = estimated support equals the true one in all three equations. Coef err = median relative
coefficient error ‖ξ̂ − ξ‖/‖ξ‖, over the runs *that method* got exact. s/run = median wall seconds per run on one
core, end to end, including the method's own model selection. A dash means the record does not hold that number:
`results_h2h/rows.json` keeps F1 and coefficient error only for DEDi's full grid.

| Method | Exact | Mean F1 | Coef err (exact runs) | s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 66/80 | 0.949 | 0.0077 | 15 |
| **ours v1.2, re-timed on cores 0–4** | 66/80 | 0.949 | 0.0077 | 14.4 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 24/80 | 0.728 | 0.0194 | 0.27 |
|   WSINDy variant: K = all centres | 24/80 | 0.725 | 0.0194 | 0.391 |
|   WSINDy variant: no pre-smoothing | 26/80 | 0.721 | 0.0127 | 0.26 |
| weak STLSQ, CV (pysindy) | 14/80 | 0.483 | 0.0247 | 1.61 |
| weak STLSQ, AICc (pysindy) | 3/80 | 0.362 | 0.0253 | 0.425 |
| weak E-SINDy, CV (pysindy) | 8/80 | 0.459 | 0.0172 | 27.6 |
| weak E-SINDy, AICc (pysindy) | 0/80 | 0.339 | – | 4.86 |
| SFD STLSQ, CV (pysindy) | 17/80 | 0.671 | 0.0179 | 3.06 |
| SFD STLSQ, AICc (pysindy) | 2/80 | 0.370 | 0.0089 | 0.539 |
| SFD E-SINDy, CV (pysindy) | 17/80 | 0.677 | 0.0142 | 59.8 |
| SFD E-SINDy, AICc (pysindy) | 3/80 | 0.390 | 0.0073 | 11 |
| in-repo weak chi2s (select_k) | 36/80 | 0.909 | 0.0166 | 0.476 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 17/80 | 0.654 | 0.0206 | 126 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 2/10 | 0.628 | 0.0179 | 1,140 |
| DEDi single pair (1e-3, 1), val | 44/80 | – | – | 120 |
| DEDi single pair (1e-3, 1), BIC | 44/80 | – | – | 120 |
| DEDi row lambda=1e-3, val | 53/80 | – | – | 569 |
| DEDi row lambda=1e-3, BIC | 56/80 | – | – | 569 |
| DEDi full grid, val | 55/80 | 0.854 | 0.0080 | 5,930 |
| DEDi full grid, BIC | 61/80 | 0.864 | 0.0084 | 5,930 |

## Per condition (16 runs each)

Same columns. For readability, the WSINDy variants, the duplicate timing row for ours and the ARGOS B = 2000 row
are left out here; they are in `results_fast/report.md`.

| Method | hard | white | coloured | t3 | short |
| --- | --- | --- | --- | --- | --- |
| **ours v1.2**: exact | **15** | **14** | **10** | **14** | **13** |
| mean F1 | 0.991 | 0.963 | 0.897 | 0.956 | 0.937 |
| coef err | 0.0087 | 0.0059 | 0.0160 | 0.0039 | 0.0068 |
| s/run (cores 0–4) | 17.5 | 13.7 | 31.4 | 13.1 | 3.83 |
| **WSINDy + MSTLS**: exact | 1 | 4 | 4 | 5 | 10 |
| mean F1 | 0.640 | 0.775 | 0.624 | 0.762 | 0.840 |
| coef err | 0.0607 | 0.0137 | 0.0308 | 0.0234 | 0.0158 |
| s/run | 0.278 | 0.267 | 0.270 | 0.275 | 0.126 |
| **in-repo weak χ²**: exact | 0 | 9 | 7 | 9 | 11 |
| mean F1 | 0.909 | 0.923 | 0.869 | 0.925 | 0.921 |
| coef err | – | 0.0098 | 0.0320 | 0.0180 | 0.0123 |
| s/run | 0.507 | 0.478 | 0.475 | 0.475 | 0.050 |
| **weak STLSQ, CV**: exact | 0 | 2 | 2 | 6 | 4 |
| mean F1 | 0.385 | 0.452 | 0.409 | 0.633 | 0.537 |
| s/run | 1.67 | 1.63 | 1.61 | 1.64 | 1.38 |
| **weak STLSQ, AICc**: exact | 0 | 0 | 0 | 1 | 2 |
| **weak E-SINDy, CV**: exact | 0 | 2 | 0 | 3 | 3 |
| mean F1 | 0.370 | 0.512 | 0.358 | 0.499 | 0.556 |
| s/run | 28.0 | 27.6 | 27.4 | 27.7 | 25.0 |
| **weak E-SINDy, AICc**: exact | 0 | 0 | 0 | 0 | 0 |
| **SFD STLSQ, CV**: exact | 0 | 4 | 3 | 6 | 4 |
| mean F1 | 0.584 | 0.678 | 0.623 | 0.683 | 0.786 |
| s/run | 3.42 | 3.14 | 3.14 | 3.09 | 2.01 |
| **SFD STLSQ, AICc**: exact | 0 | 0 | 0 | 0 | 2 |
| **SFD E-SINDy, CV**: exact | 0 | 5 | 3 | 4 | 5 |
| mean F1 | 0.594 | 0.676 | 0.628 | 0.677 | 0.810 |
| s/run | 66.8 | 61.6 | 59.9 | 59.7 | 41.5 |
| **SFD E-SINDy, AICc**: exact | 0 | 0 | 0 | 0 | 3 |
| **ARGOS, B = 200**: exact | 0 | 5 | 2 | 4 | 6 |
| mean F1 | 0.583 | 0.675 | 0.578 | 0.691 | 0.745 |
| s/run | 156 | 129 | 133 | 120 | 86 |
| **DEDi single pair**: exact val / BIC | 11 / 11 | 12 / 12 | 1 / 1 | 11 / 11 | 9 / 9 |
| s/run | 248 | 124 | 104 | 126 | 72 |
| **DEDi row λ = 10⁻³**: exact val / BIC | 14 / 15 | 14 / 14 | 0 / 1 | 12 / 13 | 13 / 13 |
| s/run | 1,400 | 591 | 408 | 573 | 321 |
| **DEDi full grid**: exact val / BIC | 13 / 15 | 14 / 14 | 5 / 5 | 12 / 13 | 11 / 14 |
| mean F1 val / BIC | 0.958 / 0.992 | 0.912 / 0.901 | 0.639 / 0.634 | 0.895 / 0.894 | 0.868 / 0.900 |
| s/run | 9,540 | 6,050 | 4,250 | 5,950 | 2,320 |

The full per-condition tables, with every row and the AICc F1, coefficient error and time, are in
`results_fast/report.md`.

## Methods and fidelity

| Method | Implementation | Fidelity | Settings we had to choose |
| --- | --- | --- | --- |
| **WSINDy + MSTLS** (Messenger & Bortz) | `benchmarks/wsindy_port.py`, a line-by-line Python port of the authors' current MATLAB **WSINDy_ODE** (github.com/MathBioCU/WSINDy_ODE, commit 72d8d6e, 2023-12-18): `wsindy_ode_fun`, `findcorners`/`findcornerpts`, `phi_int_weights`, `get_optimal_SMAF`, `wsindy_pde_RGLS_seq` (MSTLS λ-selection loss), modified `sparsifyDynamics` | **Verified identical to the official code.** The unmodified MATLAB, run in GNU Octave 9.4, gives the same support on **80/80** runs and the same test-function (m, p) on all of them. The largest relative coefficient difference is 4.5×10⁻¹³ (`results_fast/wsindy_octave_check.json`, `benchmarks/wsindy_octave_check.py`). Octave needs two one-line shims, `rms` and `range`. | **Library only:** polys = 0:3 instead of the script's 0:5. Everything else is as shipped in `wsindy_ode_script.m`: corner-point test functions (τ = 10⁻¹⁶, τ̂ = −1), K = 1000, uniform centres, λ = 10^linspace(−4, 0, 100), α = 0.8, scale_Theta = 2, optimal moving-average pre-smoothing, OLS. |
| WSINDy, other official code | **PyWSINDy_ODE** (the authors' Python repo) implements only the 2021 ODE paper, with a single user-fixed threshold `ld` and no MSTLS or automatic λ. pysindy 2.1.0 has `WeakPDELibrary` but no MSTLS. **No official Python MSTLS exists, so the port was needed.** | – | – |
| Weak STLSQ, weak E-SINDy, SFD STLSQ, SFD E-SINDy | pysindy 2.1.0 components (`WeakPDELibrary`, `SmoothedFiniteDifference`, `STLSQ`, `EnsembleOptimizer` with bagging and 20 models) through `benchmarks/baselines_bench.run_path`, unchanged from `docs/reports/baselines.md` | Official library | **Configs were frozen earlier on our dev archive** (`docs/reports/baselines.md`), not on these runs. Weak: unit-RMS rescaling, h = 10, 41 thresholds. SFD: original units, window 11, 55 thresholds. **Truth-free threshold rules:** CV = 5-fold contiguous-block CV of the LS-refit residual, ties to the sparser model; AICc = AICc of LS refits summed over equations. CV rows are timed for path + CV + AICc together, because one call computes both. AICc rows are timed separately for features + path + AICc. |
| In-repo weak χ² | `eqdisc.select_k.discover(criterion="chi2s")` | Our own code | Defaults |
| **ARGOS** (Egan, Li, Carvalho) | CRAN package **ARGOS 0.1.1**, unmodified R code (R 4.4, conda-forge), called by `benchmarks/argos_run.R` | Official package | Package defaults: Savitzky–Golay order 4 with the window chosen by `sg_optimal_combination`, `sr_method = "lasso"` (the package's own default), α = 0.05, `ols_ps = TRUE`. **We changed the bootstrap count:** B = 200 on all 80 runs. The default B = 2000 costs about 19 minutes per run (median 1,140 s), so it ran only on the first 10 runs (2 complete rounds). There it gives the same exact count (2/10) and mean F1 (0.628) as B = 200, with the same support on 7/10 runs. We did not run the adaptive-lasso option. |
| DEDi | Rows copied from `results_h2h/rows.json` (`docs/reports/h2h.md`) | Authors' code | 73/80 runs on GPU, 7 on CPU, so the medians mix devices. Single pair and λ row have no stored F1 or coefficient error. |

## Timing protocol

- **Hardware.** DGX Spark, cores 0–4: Cortex-A725 (2.8 GHz), one process per core, OMP/OPENBLAS/MKL threads = 1.
  Other sessions used other cores at the same time (cores 5 and 15).
- **What is timed.** Seconds are wall time of the method call. That includes feature or library building, the
  whole threshold or λ path, and the method's own model selection. Data loading is excluded, and package
  imports happen before the timer starts. For ARGOS the time is measured inside R and includes the design matrix
  and the three `argos()` calls. It excludes R start-up, about 1 s per worker, not per run.
- **Ours, re-timed.** We ran ours again on the same cores right after the baselines (`results_fast/job_ours.sh`).
  It reproduces the head-to-head answers exactly (80/80 identical exact flags and coefficient errors) at a median
  of 14.4 s/run. The h2h timing was 15.0 s on core 13, also an A725, while 18 DEDi workers loaded the machine.
- **Cores.** These are the Spark's slow A725 cores. On an idle X925 core, ours takes about 8–10 s and the
  baselines would speed up in proportion. The ratios hold, not the absolute seconds.

## Caveats

- **WSINDy is a port, not the official binary.** It was checked against the official MATLAB in Octave on all 80
  runs and agrees to round-off (support 80/80). We ran the authors' shipped defaults and changed only the
  library degree. Two fixed alternatives scored about the same: one test function at every valid centre (24/80),
  and no pre-smoothing (26/80). Neither changes the picture. They are sensitivity rows only and were not used to
  choose a headline.
- **Not tuned on this sample.** Settings other than the shipped defaults (K = 1000, the λ grid, α) might do better
  on some conditions. Tuning them on these runs would be oracle tuning, which this comparison excludes. In
  particular, α = 0.8 is a default the authors note may need to move toward 1 for small libraries. We left it as
  shipped.
- **The pysindy rules can reach the right model; the truth-free rules can't find it.** In 22–24 of 80 runs some
  threshold on the pysindy path is exact (path oracle, uses the truth, reference only). The truth-free CV picks
  reach 8–17 and AICc 0–3. This is the same threshold-choice failure seen on dysts (`docs/reports/baselines.md`).
- **ARGOS ran at B = 200, not its default B = 2000.** On the 10-run check, B = 2000 gave the same exact count and
  F1 at about 9× the cost. Even at B = 200 ARGOS is about 9× slower than ours. A small sanity check on *clean*
  Lorenz (N = 2000, B = 100) also missed the exact model: it kept spurious cubic terms in the ẏ equation. That
  suggests the failure is the method's lasso-plus-bootstrap-CI selection on this library, not our wrapper.
- **The in-repo χ² baseline is ours.** It is the weak-form stage our model builds on, not an outside method. It
  is listed because it is the strongest fast row, and it is not independent evidence.
- **The sample is small.** With 16 runs per condition, per-condition counts move by one or two runs. The pooled
  gaps are large: 66 against at most 36.
