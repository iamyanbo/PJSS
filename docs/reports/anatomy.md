# Where the speed comes from, and how it scales (2026-10-28/29)

**Status: final.** Run on the Spark overnight (`results_anatomy/job.sh`). Tables come from `python benchmarks/anatomy_report.py` (`results_anatomy/report.txt`). Operations were counted by wrapping functions from `benchmarks/cost_anatomy.py`; neither method's code was changed.

## Bottom line
- **DEDi does about 50× more full joint solves than we do, and about 600× more matrix factorisations.** The time gap tracks the factorisation count. Both counts come from the structure of the algorithms, not from implementation details.
  - **DEDi** runs a backward-elimination loop for each of its 35 (λ, R) settings. Each round runs two full joint LM solves (with and without the sparsity penalty), drops the smallest coefficients, and repeats. That is a median of 15 rounds and 30 joint solves per setting, about 1,000 joint solves per run.
  - **Ours** runs about 21 joint LM fits per run. In place of elimination it scores about 260,000 candidate subsets with the profiled quadratic: one cheap linear solve per subset size, with no new joint fit. It needs no hyperparameter grid, because the noise level is estimated from the data and model size is chosen by BIC.
- **The gap grows with the number of state variables.**
  - With 3 variables, one DEDi setting costs about 18× a whole run of ours. That holds at both 20 and 35 terms per equation.
  - With 5–6 variables it costs 14–100×.
  - With 10 variables (Lorenz-96, 66 terms per equation) it costs about 310×: one setting took 5 hours on the GPU, while our whole run took 58 s on one CPU core.
  - The full 35-setting grid at D = 10 would take roughly 175 GPU-hours per run. We did not run it.
- **Accuracy holds.** Ours recovers all 10 runs with 4–10 variables exactly, and 5 of 6 with the degree-4 library (it misses Rabinovich–Fabrikant, as it does with the standard library). DEDi misses both hyperchaotic Rössler runs (D = 4) in its whole λ = 10⁻³ row.

## 1. Operation counts on 10 head-to-head runs (2 per condition; DEDi full grid on GPU)

| Run | DEDi joint LM solves | DEDi factorisations | DEDi s | Ours LM fits | Ours linear solves | Ours s | Solves ratio | Factorisations ratio | Time ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_47 | 1,026 | 119,658 | 4,938 | 21 | 144 | 5.7 | 49× | 831× | 861× |
| hard_107 | 890 | 406,612 | 11,547 | 21 | 160 | 7.5 | 42× | 2,541× | 1,534× |
| white_374 | 778 | 308,736 | 9,694 | 31 | 189 | 4.9 | 25× | 1,634× | 1,999× |
| white_175 | 1,050 | 33,936 | 1,583 | 20 | 140 | 5.5 | 52× | 242× | 286× |
| coloured_10 | 692 | 92,083 | 4,053 | 39 | 343 | 14.5 | 18× | 268× | 279× |
| coloured_643 | 538 | 275,444 | 8,162 | 29 | 252 | 9.8 | 19× | 1,093× | 836× |
| t3_194 | 1,058 | 130,416 | 4,857 | 21 | 147 | 5.4 | 50× | 887× | 907× |
| t3_271 | 1,050 | 44,490 | 1,951 | 20 | 133 | 4.4 | 52× | 335× | 444× |
| short_685 | 1,050 | 37,434 | 1,179 | 20 | 154 | 2.0 | 52× | 243× | 593× |
| short_546 | 988 | 48,615 | 1,617 | 20 | 138 | 1.1 | 49× | 352× | 1,415× |
| **Median** | | | | | | | **49×** | **592×** | **849×** |

- **What is counted:**
  - DEDi factorisations are sparse Cholesky factorisations, one per LM iteration attempt including damping retries.
  - Our linear solves are banded-Schur solves, one per LM iteration attempt.
  - Both are the dominant cost of an LM step.
- **Over all 80 runs, ours** has a median of 21 LM fits, 153 linear solves and 260,400 subsets scored (12 screens per run).
- **Timing conditions:** DEDi on GPU and ours on one CPU core (core 15), both on the Spark that night.
- **Why the time ratios differ from the head-to-head:** these are 10 runs, and the head-to-head medians include the machine being fully loaded.

## 2. Scaling: one DEDi setting vs one whole run of ours

DEDi ran the λ = 10⁻³ row (5 settings) on the GPU; ours ran on one CPU core. The ratio is DEDi's median seconds per setting divided by our seconds per run. The full grid multiplies DEDi's cost by about 35.

| Case | D | Terms/eq | Ours s | Ours exact | DEDi s/setting | DEDi joint solves/setting | DEDi exact at (10⁻³, 1) | DEDi exact anywhere in row | Ratio per setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D = 3, 20 terms (10 runs above) | 3 | 20 | 1–15 | – | – | 30 | – | – | 18× (median) |
| Rabinovich–Fabrikant | 3 | 35 | 5.7 | 0 | 108 | 28 | 0 | 0 | 19× |
| Sprott A | 3 | 35 | 5.7 | 1 | 87 | 48 | 1 | 1 | 15× |
| Sprott A | 3 | 35 | 5.8 | 1 | 124 | 48 | 1 | 1 | 21× |
| Sprott B | 3 | 35 | 7.8 | 1 | 226 | 48 | 1 | 1 | 29× |
| Sprott E | 3 | 35 | 10.0 | 1 | 185 | 48 | 1 | 1 | 19× |
| Rössler | 3 | 35 | 15.7 | 1 | 260 | 50 | 1 | 1 | 17× |
| Hyperchaotic Rössler | 4 | 15 | 32.9 | 1 | 83 | 30 | 0 | 0 | 3× |
| Hyperchaotic Rössler | 4 | 15 | 36.7 | 1 | 42 | 30 | 0 | 0 | 1× |
| Lorenz-96 | 5 | 21 | 7.6 | 1 | 105 | 42 | 1 | 1 | 14× |
| Lorenz-96 | 5 | 21 | 7.5 | 1 | 239 | 42 | 1 | 1 | 32× |
| Lorenz-96 | 6 | 28 | 28.7 | 1 | 408 | 68 | 0 | 1 | 14× |
| Lorenz-96 | 6 | 28 | 13.0 | 1 | 300 | 70 | 1 | 1 | 23× |
| Coupled Lorenz | 6 | 28 | 7.1 | 1 | 708 | 70 | 0 | 1 | 100× |
| Coupled Lorenz | 6 | 28 | 11.5 | 1 | 543 | 26 | 0 | 1 | 47× |
| Lorenz-96 | 10 | 66 | 57.5 | 1 | 18,156 (1 of 5 settings in 6 h) | 256 | – | 1 | 316× |
| Lorenz-96 | 10 | 66 | 57.9 | 1 | 17,904 (1 of 5 settings in 6 h) | 256 | – | 1 | 309× |

- **The D = 10 runs hit the 6-hour cap per run** after one setting each, (10⁻³, 10⁻⁴). Both of those settings were exact.
- **For D ≥ 4 ours uses its D-dimensional path** (degree-2 library). Only its banded solves are counted there; its joint fits and subset scores aren't.
- **DEDi needed a patch to run at D = 10.** Its library parser substitutes `x_1` before `x_10`, which mangles the 10th variable and raises a SyntaxError. Our runner swaps in the same function with longest-name-first substitution, for D ≥ 10 only. `results_anatomy/check_patch.py` shows it is identical to the original at D = 6 and exact at D = 10. The failed first attempt and the rerun are in `results_anatomy/progress.log`.
- **Hyperchaotic Rössler (D = 4) is the one case where DEDi is about as fast as us.** It fails there, and ours is at its slowest (33–37 s).

## What the paper can say
- **Mechanism:** DEDi's cost is (35 grid settings) × (about 15 elimination rounds) × (2 full joint solves). Ours replaces the grid with self-calibrated BIC selection, and the elimination rounds with one global screen of all candidate subsets on the profiled quadratic. Measured, that is about 50× fewer joint solves and about 600× fewer factorisations, the same order as the measured time gap.
- **Scaling:**
  - **More data (N):** the gap is constant (`docs/reports/timing.md`).
  - **Bigger library:** the gap is constant from 20 to 35 terms.
  - **More state variables:** the gap grows, from about 18× per setting at D = 3 to about 310× at D = 10. At D = 10 the full grid is impractical: about 175 GPU-hours per run, against our 58 s.
- **Present these as numbers only, with no significance tests.**
