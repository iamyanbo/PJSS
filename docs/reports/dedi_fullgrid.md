# DEDi with its full hyperparameter grid on hard-120 (2026-10-26/27)

**Status: final.** The run was pre-registered in `d7d72f3` before launch. It ran on the DGX Spark from 2026-10-26 14:54 to the 2026-10-27 13:00 deadline, on 10 single-core workers.

**Bottom line: the reviewer's objection holds.**
- With the authors' full grid, DEDi recovers 14 of the 17 analysed runs. With the reduced 2×2 grid it recovered 4 of the same 17, a gain of 10 to 0 (p = 0.002).
- Against the full-grid DEDi, our final model scores 3/2 (p = 1).
- Our hard-120 claim of 26/5 against DEDi (p = 2×10⁻⁴) came mostly from DEDi's reduced grid. It must not be reported as "better than DEDi".
- What survives is a claim about cost: comparable accuracy on these runs at roughly 5,000× less compute.

## Question

The hard-120 comparison (`docs/reports/hard120.md`) ran DEDi with λ ∈ {10⁻², 1} and R ∈ {10⁻⁴, 10⁻²}. Validation error picked λ = 10⁻² in 110 of 118 runs, and DEDi made 146 false positives.

Does DEDi with its own grid close the gap? Its grid is λ = 10⁻³…10³ and R = 10⁻⁴…10⁰, which is 35 pairs. The authors' example notebooks label this grid "Parameters used in paper".

## Protocol (pre-registered: `results_dedi_fullgrid/PREREGISTRATION.json`)

- **What stays the same as hard-120:**
  - archive `data/hard120.npz`;
  - DEDi code (unmodified);
  - the 20-term degree-3 library;
  - the every-third-sample validation mask;
  - the `FindBestModel` defaults.
- **What changes:** only the grid, now 35 pairs.
- **Execution:**
  - Each (run, λ, R) pair is a separate `FindBestModel([λ], [R])` call (`benchmarks/dedi_fullgrid.py`). This is equivalent to the joint call, because the authors' loop resets all optimisation state between pairs.
  - The work ran on 10 workers pinned to cores 10–19, one core each.
- **Selection:**
  - Primary: lowest validation error, which is DEDi's default `output_sort`.
  - Secondary: validation BIC, and an "oracle" that counts a run as exact if any of the 35 pairs is exact. The oracle uses the truth, so it is a bound, not a method.
- **Our model:** the stored final-model results (`results_hard120/ours_hard120.json`). It was not rerun.
- **Subset and order:** `results_dedi_fullgrid/order.json`, seed 20261026, built from the 114-run hard-120 analysis set. The strata are defined by (final exact, reduced-grid DEDi exact):

  | Stratum | Final model | Reduced-grid DEDi | Runs |
  | --- | --- | --- | --- |
  | A | exact | failed | 26 |
  | B | failed | exact | 5 |
  | C | exact | exact | 80 |
  | D | failed | failed | 3 |

  The order is blocks of three runs from a random permutation of A+B+D, each followed by one run from a random permutation of C.
- **Analysis set:** the longest prefix of that order in which all 35 pairs of every run finished by the deadline.

**The order deliberately puts disagreement runs first, so the subset is not representative of the 114 runs.**
- Three quarters of the early runs come from A, B or D, the strata where the head-to-head can change.
- The interim "4/5 vs 1/5" (3 runs from A, 1 from D, 1 from C) and the final 17 runs (11 from A, 2 from D, 4 from C, 0 from B) are therefore enriched for runs where reduced-grid DEDi failed.
- By chance, no B run (DEDi-only win) reached the analysis set: it was 0 of 5, against about 2 expected.
- So the raw counts on the 17 runs overstate how well DEDi did overall. Use the per-stratum rows and the projection below, not the raw totals.

## Results (analysis set: first 17 of 114 runs; `python benchmarks/dedi_fullgrid_report.py`)

### Primary: full grid, selection by validation error

| Subset | n | Final | DEDi reduced | **DEDi full** | Final vs full W/L (p) | Final vs reduced W/L (p) | Full vs reduced W/L (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lorenz | 3 | 2 | 0 | 2 | 1/1 (1) | 2/0 (0.5) | 2/0 (0.5) |
| Chen | 6 | 6 | 4 | 4 | 2/0 (0.5) | 2/0 (0.5) | 0/0 |
| Rössler | 8 | 7 | 0 | 8 | 0/1 (1) | 7/0 (0.016) | 8/0 (0.008) |
| A (final only) | 11 | 11 | 0 | 8 | 3/0 (0.25) | 11/0 (0.001) | 8/0 (0.008) |
| C (both exact) | 4 | 4 | 4 | 4 | 0/0 | 0/0 | 0/0 |
| D (neither) | 2 | 0 | 0 | 2 | 0/2 (0.5) | 0/0 | 2/0 (0.5) |
| **All** | **17** | **15** | **4** | **14** | **3/2 (p = 1)** | 11/0 (0.001) | **10/0 (0.002)** |

- **The full grid moves DEDi to λ = 10⁻³, a value not on the reduced grid:**
  - Validation error picks λ = 10⁻³ in 15 of 17 runs and λ = 10⁻² in 2.
  - False positives on these runs fall from 61 (reduced grid) to 4. The false negatives are 8 for the reduced grid and 0 for the full grid.
  - The final model makes 4 false positives and 2 false negatives.
- **In stratum A, which is where all of our hard-120 wins came from, the full grid turns 8 of 11 losses into ties.**
- **Coefficients:** where both methods are exact (n = 12), the median relative coefficient error is 0.78% for the final model and 0.74% for full-grid DEDi.

### Secondary selections (pre-registered)

| Selection over the 35 pairs | DEDi exact | Final vs DEDi W/L | p |
| --- | --- | --- | --- |
| Validation error (primary) | 14/17 | 3/2 | 1 |
| Validation BIC | 17/17 | 0/2 | 0.5 |
| Oracle (any pair exact; uses truth) | 17/17 | 0/2 | 0.5 |

- With BIC on the validation points, which is the other criterion DEDi computes and the one its README mentions, DEDi is exact on all 17 runs. That includes the 3 runs of stratum A that validation error still gets wrong.
- So on these runs the full grid always contains the correct model, and DEDi's remaining failures are selection failures.
- How often each grid point alone gives the exact model, out of 17:

  | λ \ R | 10⁻⁴ | 10⁻³ | 10⁻² | 10⁻¹ | 1 |
  | --- | --- | --- | --- | --- | --- |
  | 10⁻³ | 6 | 6 | 7 | 7 | 13 |
  | 10⁻² | 5 | 4 | 5 | 5 | 10 |
  | 10⁻¹ | 5 | 5 | 5 | 5 | 4 |
  | 1 | 3 | 3 | 3 | 3 | 3 |
  | 10 | 3 | 3 | 3 | 3 | 3 |
  | 10² | 2 | 2 | 2 | 2 | 2 |
  | 10³ | 2 | 2 | 2 | 2 | 2 |

  The reduced grid's four points (λ ∈ {10⁻², 1}) are among the weakest. The best single point is λ = 10⁻³, R = 1, outside the reduced grid.

### Projection to the 114-run set

The pre-registered estimate is stratum-weighted, using sampled strata only.
- It projects final against full-grid DEDi at about 7.1 wins and 3.0 losses on 114 runs, against 26/5 with the reduced grid.
- Stratum B was not sampled, so this treats B's 5 reduced-grid DEDi wins as ties.
- A post hoc variant keeps B at its reduced-grid outcome. It gives about **7.1 wins and 8.0 losses**.
- Either way, no advantage for our method can be claimed. Both projections rest on 11 of 26 A runs, 2 of 3 D runs and 4 of 80 C runs, so they are rough.

## Reproducibility check and a DEDi non-determinism

**The recomputed reduced grid matched the stored choice in only 14 of 17 runs.** Selecting from the 4 reduced-grid pairs recomputed in this run should reproduce the stored hard-120 choice and outcome. It did so in 14 of 17 runs. In the other three:

| Run | System | Stored hard-120 result | Recomputed here |
| --- | --- | --- | --- |
| 0 | Lorenz | failed | exact |
| 106 | Rössler | failed | exact |
| 82 | Rössler | failed | failed (different pair chosen) |

- **Within this setup DEDi is deterministic.** Index 0 at (λ, R) = (10⁻², 10⁻²) was run 4 times, on 2 slow and 2 fast cores, in fresh processes. All four gave bit-identical results (`determinism.log`), and they matched the main run.
- **The hard-120 run used a different setup.** It had 12–14 workers sharing a taskset of 12–14 cores, which lets DEDi's two internal optimisation threads run truly in parallel. Those threads write shared attributes of the model object. So the stored results probably depend on thread timing. This is a likely explanation; it was not proven.
- **This makes the stored reduced-grid results look slightly worse than reduced-grid DEDi under this run's conditions.**
  - Recomputed here, the reduced grid scores 6/17, against 4/17 stored.
  - The like-for-like comparison of grid effect alone, same numerics, is full against recomputed reduced: 8/0 (p = 0.008).
  - Final against recomputed reduced is 10/1 (p = 0.012).
  - The conclusion does not change: the grid, not this noise, accounts for most of the gap.

## Pilot (outside the analysis set)

Index 38 (Lorenz) was run first to measure cost and to check reproduction.
- The stored reduced-grid result (λ = 1, R = 10⁻², exact) was reproduced.
- Full-grid DEDi chose λ = 10⁻³, R = 1 and was exact.
- The final model was exact.

## Cost

- **Pairs:** 616 completed: 595 for the 17 analysed runs plus 21 for later runs. In total 208 core-hours, over about 21.7 h wall on 10 cores.
- **Per pair:** median 15 min, maximum 1.4 h.
- **Per run:** median 13.4 core-hours, range 6.5–16.8.
- **Against our model:** the final model takes a median 8.6 s per run, about 5,000× less.
- **Shortfall against plan:** the pre-registration expected 18–22 runs; 17 were achieved. Pairs with λ ≥ 10 were slower than the pilot suggested.

## What this means for the paper

1. **Withdraw "better than DEDi" on hard-120.** This covers the 26/5 headline and the pooled 33/5. On the 17 runs re-analysed with the authors' grid, the result is 3/2 (p = 1), and the projection to all 114 runs is roughly even. With BIC selection over the full grid, DEDi is exact on all 17.
2. **The defensible claim is:** comparable recovery to DEDi on these hard 20%-noise cases (93% for the final model on hard-120; full-grid DEDi is not measured on all 114), at about 3–4 orders of magnitude less compute and with no hyperparameter grid.
   - DEDi's accuracy depends on searching λ down to 10⁻³. Each grid point costs about 15 core-minutes.
3. **Report the reduced-grid result only as a sensitivity result.** DEDi with a reduced grid that omits λ = 10⁻³ is much worse.

## Caveats

- **n = 17** in a front-loaded order, with no stratum-B runs and only 4 of the 80 both-exact runs. The per-stratum results matter most:
  - in stratum A, full-grid DEDi fixes 8 of 11 runs;
  - D has only 2 runs.
- **Same archive and setting as hard-120:** three 3-D chaotic systems at 20% noise, N = 2000.
- **Logging bug in `job_main.sh`:** the "worker exit rc=" lines in the progress log print the exit status of `date`, not of `timeout`. Every worker was in fact killed at the 13:00 deadline, and the pairs in flight at that point were lost. Results are unaffected.

## Files

- `results_dedi_fullgrid/PREREGISTRATION.json`, `order.json`: committed before launch.
- `results_dedi_fullgrid/main/pairs/*.json`: one file per (run, λ, R), with validation error, BIC, coefficients, metrics and seconds.
- `results_dedi_fullgrid/main/`: worker logs and `progress.log`.
- `results_dedi_fullgrid/pilot/`: pilot results.
- `results_dedi_fullgrid/determinism.log`, `det.sh`: the 4-run determinism check.
- `results_dedi_fullgrid/report.txt`: full output of `python benchmarks/dedi_fullgrid_report.py`, which includes the post hoc rows marked as such.
- Code:
  - `benchmarks/dedi_fullgrid.py` (runner);
  - `benchmarks/dedi_fullgrid_report.py` (analysis);
  - `results_dedi_fullgrid/job.sh` (pilot) and `job_main.sh` (main).
