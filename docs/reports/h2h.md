# Head-to-head on a random sample: our model v1.2 vs DEDi with its full grid (2026-10-27/28)

**Status: final.** Pre-registered in `e09e5bd` (`results_h2h/PREREGISTRATION.md`) before any run. Ran on the DGX Spark from 2026-10-27 22:32 to 2026-10-28 11:50. All 80 runs and all 2,800 DEDi (run, λ, R) pairs finished, with no errored pair. Tables: `python benchmarks/h2h_report.py` (`results_h2h/report.txt`, per-run rows in `results_h2h/rows.json`).

**Bottom line**
- **Pre-registered primary verdict: ours better.**
  - On 80 random runs, ours recovers the exact model in 66 and DEDi in 55, with DEDi selecting by its default validation error.
  - Discordant runs split 13/2 (sign p = 0.007). The difference d = DEDi − ours is −0.138, 95% CI [−0.229, −0.047].
- **The claim the paper can carry is "at least as accurate", not "more accurate".** Two of the pre-registered secondaries limit the primary result:
  - **DEDi with BIC selection** gets 61/80. Against it we have 66 vs 61, 7/2, p = 0.18, d CI [−0.140, +0.011]. This meets the pre-registered "comparable" bar, an upper bound ≤ 0.10.
  - **The gain is concentrated in coloured noise**, where DEDi has no noise model: 10 vs 5, 5/0. Without coloured noise we have 56/64 vs DEDi-val 50/64 and DEDi-BIC 56/64. With BIC, that is a tie.
- **No DEDi selection rule beats us.** DEDi's oracle, which counts a run as exact if any of its 35 pairs is exact and so uses the truth, reaches 64/80. Ours, with no oracle, reaches 66/80.
- **Cost: about 340× less than GPU DEDi, about 3,600× less than CPU DEDi (medians per run). This is not 5,000×.**
  - The full grid's accuracy needs its full cost. A single pair costs 9× ours and gets 44/80. The λ = 10⁻³ row costs 46× ours and gets 53/80.

## Sample and protocol (as pre-registered)
- **Sample:** 80 runs, 16 per condition, drawn at random with seed 20261027 (`benchmarks/h2h_make.py`). They cover 10 systems (3 for "hard"). Unlike the earlier 17-run comparison, the sample is not enriched for disagreement runs.

  | Condition | Archive | N | Noise |
  | --- | --- | --- | --- |
  | hard | hard-120 (Lorenz/Chen/Rössler), excluding the 17 earlier runs | 2000 | 20% |
  | white | joint_sealed | 2000 | 10%, 20% |
  | coloured | sealed_colored2 (AR(1)) | 2000 | 10%, 20% |
  | t3 | sealed_student | 2000 | 10%, 20% |
  | short | joint_sealed | 500 | 5%, 10% |

- **Ours:** `eqdisc.discover` v1.2 defaults, one process on Spark core 13 (an A725 core). Timed while DEDi ran on the other cores.
- **DEDi:**
  - The authors' code is unmodified: 20-term degree-3 library, every-third-sample validation mask, and the paper's 35-pair grid.
  - It ran on 10 GPU workers and 8 CPU workers. Each run was bound to one device: 73 runs on GPU, 7 on CPU.
  - Sparse solves used our exact scipy banded/Schur stand-in for CHOLMOD. A native-CHOLMOD timing check runs separately (`results_timing`).

## Primary: full grid, selection by validation error

| Subset | n | Ours exact | DEDi exact | Ours-only / DEDi-only | Sign p | d = DEDi − ours [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| **Pooled** | 80 | 66 | 55 | 13/2 | 0.007 | −0.138 [−0.229, −0.047] |
| hard | 16 | 15 | 13 | 3/1 | 0.63 | −0.125 [−0.379, +0.140] |
| white | 16 | 14 | 14 | 0/0 | 1 | 0.000 [−0.145, +0.145] |
| coloured | 16 | 10 | 5 | 5/0 | 0.063 | −0.312 [−0.490, −0.077] |
| t3 | 16 | 14 | 12 | 2/0 | 0.5 | −0.125 [−0.323, +0.053] |
| short | 16 | 13 | 11 | 3/1 | 0.63 | −0.125 [−0.354, +0.115] |

The CI is Newcombe's paired method 10. Per-condition rows are descriptive only: 16 runs per condition cannot settle a condition on its own.

## Secondary results

**DEDi by BIC, and its oracle bound**

| Subset | n | Ours | DEDi | Ours-only / DEDi-only | Sign p | d [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| Pooled, BIC | 80 | 66 | 61 | 7/2 | 0.18 | −0.062 [−0.140, +0.011] |
| hard, BIC | 16 | 15 | 15 | 1/1 | 1 | 0.000 |
| white, BIC | 16 | 14 | 14 | 0/0 | 1 | 0.000 |
| coloured, BIC | 16 | 10 | 5 | 5/0 | 0.063 | −0.312 |
| t3, BIC | 16 | 14 | 13 | 1/0 | 1 | −0.062 |
| short, BIC | 16 | 13 | 14 | 0/1 | 1 | +0.062 |
| Pooled, oracle (uses truth) | 80 | 66 | 64 | 5/3 | 0.73 | −0.025 [−0.099, +0.047] |

- **Coloured noise.** In 4 of the 5 coloured runs that only we get right, no DEDi pair is exact. DEDi's white-noise likelihood cannot get these runs, whatever its selection rule.
- **Short data.** This is the only condition where DEDi with BIC is ahead, by one run: 14 vs 13.

**F1 and coefficients**
- Mean F1: ours 0.949, DEDi (validation) 0.854, DEDi (BIC) 0.864.
- On the 53 runs both get exact, the median relative coefficient error is 0.0064 for ours and 0.0080 for DEDi. Ours is lower on 37 of the 53.

**Cost and accuracy vs compute** (medians per run; ours median 15.0 s)

| DEDi grid | Device (n) | Exact (val) | Exact (BIC) | Median s/run | Median ratio to ours |
| --- | --- | --- | --- | --- | --- |
| single pair (10⁻³, 1) | GPU (73) | 41/73 | 41/73 | 115 | 9× |
| λ = 10⁻³ row (5 pairs) | GPU (73) | 50/73 | 53/73 | 550 | 46× |
| full grid (35 pairs) | GPU (73) | 54/73 | 58/73 | 4,806 | 343× |
| full grid (35 pairs) | CPU (7) | 1/7 | 3/7 | 44,859 | 3,590× |

- **Pooled over 80 runs:**
  - single pair: 44/80 by both selection rules;
  - row: 53 by validation, 56 by BIC;
  - full grid: 55 by validation, 61 by BIC.
- **On cost, DEDi can't match us at the cheap end.** The cheapest DEDi setting that comes near our accuracy is the full grid, at about 340× our cost on the GPU.
- **Caveats on these timings:**
  - Our 15 s was timed on a slow core while the machine was fully loaded. The earlier 8.6 s median was on a less loaded machine.
  - DEDi's times use the solver stand-in.
  - The clean, like-for-like timing (native CHOLMOD, idle machine, same core) is in `results_timing`.

**Hard condition, pooled with the earlier 17 runs (those 17 were partly enriched)**
- The 16 new random runs: ours 15, DEDi 13 by validation and 15 by BIC.
- Pooled 33 runs: ours 30, DEDi 27 by validation and 32 by BIC.

## Caveats
- **Device was not randomised.** Runs went to whichever worker pool claimed them first, and the 7 CPU runs were the slower-to-claim ones. DEDi got 1 of those 7, and its BIC rule 3 of 7. Only 7 runs ran on the CPU, so this is no evidence either way of a device effect. The pooled result without them is 62 vs 54 (10/2, p = 0.039).
- **The GPU is not bit-reproducible.** One pair in 38 flipped support between GPU repeats in `dedi_gpu.md`. These are single GPU realisations.
- **Execution incidents (no effect on results).**
  - At 09:16 one GPU worker was killed by the kernel's out-of-memory killer. GPU worker memory grows to about 6.5 GB over 11 h.
  - At 09:42 three more workers were stopped deliberately to free memory.
  - Their unfinished pairs were rerun from scratch by fresh workers (`results_h2h/finish.sh`, `d4cca10`). Each pair is an independent `FindBestModel` call, so a rerun pair is the same computation.
  - Both incidents are logged in `results_h2h/progress.log`.
- **Not covered:** coarse sampling (dt = 0.2), our known weak spot. Also not covered: process noise, and baselines other than DEDi.

## What the paper should say
- "On a pre-registered random sample of 80 runs over 10 systems and five noise and data conditions, the method is at least as accurate as DEDi with its full 35-pair grid.
  - It recovers 66 of 80 exactly, against 55 with DEDi's default selection (p = 0.007) and 61 with DEDi's BIC selection (p = 0.18).
  - The gap comes mostly from coloured noise. Even an oracle choice among DEDi's 35 fits reaches only 64.
- It does this at about 340× less compute than DEDi on a GPU."
- Replace "5,000×" by the measured ratios.
- State the BIC result and the coloured-noise concentration up front.
