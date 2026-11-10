# Revision experiments: results

Pre-registration: `results_rev/PREREGISTRATION.md`. Reports: `results_rev/e1_report.py`, `benchmarks/rev_report.py`.
All jobs ran on the Spark from 2026-10-30 20:38 to 2026-10-31 10:10, with no failed pairs and no errored runs.

Submission-audit note (2026-11-03): this report preserves the original analysis below. The manuscript
now uses the reproducible system-bootstrap intervals in `results_submission_audit/system_intervals.json`;
the E2 BIC interval is [-0.344, -0.044]. CPU totals below are sums of elapsed single-thread task times
across three concurrent workers, not observed serial full-grid wall times.

## E1. Does profiling matter?
Verdict: **profiling matters**, modestly. Reproducibility check: 80/80 stored `full` answers identical.

| Set | Full | no_schur | Full W/L |
| --- | --- | --- | --- |
| Sealed (160) | 139 | 135 | 4/0 |
| Coloured (89) | 65 | 66 | 2/3 |
| Hard 40 | 28 | 24 | 4/0 |
| dysts D=3 (287) | 258 | 255 | 5/2 |
| **Pooled (576)** | **490** | **480** | **15/5, p = 0.041** |

2x2 on the hard 40: refit weak 0, all starts refitted 19, proposals from weak 27, both 28. `refit_only` equals the weak
selector by construction, since a refit does not change a support.

## E2. DEDi on 64 runs of 35 new 3-D dysts systems
Verdict: **ours better**. All 64 runs complete, 0 errored.

| DEDi selection | Ours | DEDi | W/L | p | d [95% CI] | Bootstrap over systems (not pre-registered) |
| --- | --- | --- | --- | --- | --- | --- |
| Validation (primary) | 53 | 36 | 19/2 | 0.0002 | -0.266 [-0.384, -0.137] | [-0.413, -0.119] |
| BIC | 53 | 41 | 15/3 | 0.008 | -0.188 [-0.306, -0.064] | [-0.339, -0.034] |
| Oracle | 53 | 44 | 12/3 | 0.04 | -0.141 [-0.254, -0.026] | [-0.292, 0.000] |

By noise (ours / val / BIC / oracle): 1% 15/5/8/9, 5% 14/9/11/12, 10% 14/13/13/13, 20% 10/9/9/10.
Grid subsets, exact by val / BIC: one setting 26/26, one row 35/36, full grid 36/41.

## E3. Same-hardware cost
Ours: 9 repeats per run over cores 15-17, range within 0.4 s. Pairs with native CHOLMOD on the CPU, one process alone on the GPU.

| Run | Ours s (median) | DEDi CPU core-s | Ratio | DEDi GPU process-s | Ratio |
| --- | --- | --- | --- | --- | --- |
| white_374, Rabinovich-Fabrikant | 4.7 | 34,006 | 7,213 | 4,462 | 946 |
| white_175, Sprott A | 5.1 | 4,591 | 901 | 526 | 103 |
| white_186, Sprott A | 5.4 | 4,374 | 812 | 501 | 93 |

**Deviation from the pre-registration:** the sampled workers claim pairs from one queue, so the three CPU processes
shared each run's 35 pairs, instead of one process per run. Each pair was timed on one of the three fast cores, so the
sums are still core-seconds, with three processes running at once. The GPU timing ran as registered.

Earlier in the head-to-head, ours took 14-23 s on these runs on the loaded machine, about 3-5 times slower.

## Corrections made to the paper from these results
- Profiling is no longer called the main ingredient; exhaustive search around the joint states does most of the work.
- The 340 ratio is kept only as the loaded-machine median. The 1,100-1,900 estimate is replaced by measured values.
- 260,400 subsets are 12 screens: 6 on the profiled quadratic and 6 in the trajectory-segment start.
- Per-setting ratios of 8 (`tab:fast`) and 18 (`tab:anatomy`) differ through load and run sample; idle, one setting costs about 5 of our runs.
