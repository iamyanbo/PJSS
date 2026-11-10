# Head-to-head: our model v1.2 vs DEDi with its full grid (pre-registered 2026-10-27, before any run)

**Why.** The paper's accuracy claim rests on 17 hard-120 runs that were deliberately enriched for disagreement
(results_dedi_fullgrid). A 17-run sample cannot rule out DEDi being ~20 points better. This sample is random,
covers 10 systems and five noise/data conditions, and fixes the decision rule before any result exists.

## Sample (`benchmarks/h2h_make.py`, seed 20261027 -> `order.json`)
Five conditions x 16 runs = 80 runs. Each condition is stratified over its systems and split 8/8 over two noise levels.
| cond | archive | N | noise |
| --- | --- | --- | --- |
| hard | hard120 (Lorenz/Chen/Rossler), excluding the 17 already analysed | 2000 | 0.2 |
| white | joint_sealed (10 systems) | 2000 | 0.1, 0.2 |
| coloured | sealed_colored2 (AR(1) noise) | 2000 | 0.1, 0.2 |
| t3 | sealed_student (Student-t, 3 dof) | 2000 | 0.1, 0.2 |
| short | joint_sealed | 500 | 0.05, 0.1 |
Order: round-robin over conditions. Coarse sampling (dt = 0.2) is not included (cost); this is stated as a gap.

## Methods
- **Ours:** `eqdisc.discover` v1.2 defaults (`final_bench.py --arms final_v12 --kind poly3`), no starts cache,
  one process pinned to core 13 of the Spark while DEDi runs on the other cores (same load).
- **DEDi:** authors' code unmodified, 20-term degree-3 library, every-third-sample validation mask, the paper's
  35-pair grid (lambda 1e-3..1e3 x R 1e-4..1), `benchmarks/dedi_h2h.py`. 10 GPU workers (cores 5-9, 15-19) and
  8 CPU workers (cores 0-4, 10-12). Each run is bound to one device; the device is recorded and reported.
  Sparse solves use the same exact scipy banded/Schur stand-in for CHOLMOD as the earlier runs (scikit-sparse
  is not installed on the Spark: no system SuiteSparse, no sudo). It is an exact factorisation, so it can change timing
  and roundoff, not the method. A separate native-CHOLMOD timing check is planned; timing is reported with this caveat.

## Analysis set
Deadline: 40 h after launch. Analysis set = the longest prefix of complete rounds (5 runs, one per condition) in
which all 35 pairs of every run finished. A run with an errored pair counts as a DEDi failure.

## Primary endpoint (exact support recovery, pooled over the analysis set)
- DEDi selection = argmin validation error over the 35 pairs (DEDi's default `output_sort`).
- Paired difference d = rate(DEDi) - rate(ours), 95% CI from the paired (Newcombe/Tango-style) score method;
  also the two-sided sign test on discordant runs.
- **"Comparable"** iff the upper CI bound <= 0.10. **"Ours better"** iff the lower CI bound > 0 and sign p < 0.05.
  **"DEDi better"** iff the lower bound of d > 0 and sign p < 0.05. Otherwise "inconclusive", reported as such.

## Secondary (all reported, none can change the primary verdict)
1. Same comparison with DEDi selecting by its BIC (minimum `bic` over the 35 pairs) and the oracle bound
   (any pair exact; uses truth, not a method).
2. Per-condition and per-device rows (descriptive).
3. F1 and relative coefficient error (on runs both get exact).
4. Cost: DEDi core-seconds (CPU runs) and GPU-process-seconds (GPU runs) per run, summed over pairs; ours
   seconds per run on the same machine and load. Ratio reported as medians, with the solver caveat.
5. DEDi accuracy vs compute for grid subsets fixed now: (a) the single pair (1e-3, 1) (chosen from the earlier
   hard-120 runs, a different sample), (b) the lambda = 1e-3 row (5 pairs), (c) the full grid; validation selection.
6. Hard condition pooled with the earlier 17 runs, labelled as partly enriched.
