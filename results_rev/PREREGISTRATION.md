# Revision experiments (pre-registered 2026-10-30, before any run)

**Why.** Two independent reviews of the paper found claims the existing data cannot support. Each experiment below
answers one of them. The method is unchanged: `eqdisc.discover` defaults, code as in this commit.

## E1. Does the profiling step matter? (ablation)
**Question.** The proposals are computed from the joint fit, so the ablation cannot say whether profiling the states
adds anything over best-subset search on the fixed states.

- **Arm `no_schur`:** the full method, with the proposal quadratic built from `A_cc` and `g_c` alone (states fixed, no
  Schur term), on both the white and the AR(1) path. Everything else is unchanged.
- **Arm `refit_only`:** weak start, no proposals, final settings. With the stored `full`, `no_proposals` and
  `single_start`, this completes the 2x2 of proposals by starts.
- **Sets:** the stored ablation subsamples, sealed (160), coloured (89), hard 40, dysts D=3 (287).
- **Analysis:** paired against stored `full`; two-sided sign test pooled over sets and per set. The stored `full` is
  re-run on 20 runs per set as a reproducibility check.
- **Primary verdict:** "profiling matters" iff pooled full wins exceed no_schur wins with p < 0.05. "No evidence"
  otherwise. Either result goes in the paper as it comes.

## E2. A DEDi comparison off our own benchmark
**Question.** The head-to-head sampled only our 10 core systems.

- **Sample:** `benchmarks/rev_make.py`, seed 20261030: 64 runs of the dysts test archive, restricted to the 3-D
  systems new to us, 8 per (noise in 1, 5, 10, 20%; N in 500, 2000) cell, balanced over systems, round-robin over cells.
- **DEDi:** as in `results_h2h`: authors' code, the 35-pair grid, every-third-sample mask, stand-in solver, 10 GPU
  workers. No CPU workers, so all runs share a device.
- **Ours:** `benchmarks/rev_ours.py`, defaults.
- **Analysis set:** longest prefix of complete rounds of 8 runs at the 30 h deadline. A run with an errored pair counts
  as a DEDi failure.
- **Primary:** DEDi by validation error, d = rate(DEDi) - rate(ours), Newcombe paired 95% CI, sign test.
  "Comparable" iff upper bound <= 0.10. "Ours better" iff the upper bound < 0 and sign p < 0.05. "DEDi better" iff
  lower bound > 0 and sign p < 0.05. Otherwise "inconclusive".
- **Secondary:** the same against DEDi by BIC, and the any-pair oracle.

## E3. Same-hardware cost
**Question.** The 340 ratio compares a loaded CPU core with a shared GPU, and the CPU ratio extrapolates from one run.

- **Sample:** the first three "white" runs of `results_h2h/order.json`.
- **CPU:** DEDi full grid with native CHOLMOD, one process per run on fast cores 15, 16, 17; ours 3 repeats on each same core.
- **GPU:** one DEDi process alone on the GPU, pinned to core 15, on the same runs.
- **Conditions:** nothing else of ours runs on the machine; starts after E1 and E2 finish. Load is logged.
- **Reported:** per-run core-seconds on the same core, the ratio to ours, and GPU process-seconds. No extrapolation.

## Stored-data analyses (no new runs)
- System-level bootstrap CIs (20,000 resamples of systems) for the dysts rates. This one was computed before this file
  was written: 0.890 [0.841, 0.931].
- DEDi accuracy against cost from `results_h2h/rows.json`, for the three pre-registered grid subsets.
