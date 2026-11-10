# Timing: native CHOLMOD, same core, runtime vs N and library size (2026-10-28)

**Status: final.** Run on the Spark after the head-to-head had finished (`results_timing/job.sh`, then `rerun_native.sh` for the native parts). Only one other job was running, on a different core: the dysts refresh on core 14. Raw data: `results_timing/*.json*`.

**Bottom line**
- **The CHOLMOD stand-in did not inflate DEDi's cost.** With native CHOLMOD a DEDi pair takes the same time or slightly longer: the stand-in/native time ratio ranges from 0.91 to 1.02. The sparse solve is only 10–20% of DEDi's time either way.
- **The solver doesn't change DEDi's answers.**
  - 3 of 4 pairs give the same support, with coefficients equal to 1e-11 or better.
  - In the 4th pair, (λ, R) = (1, 1), the support differs. That pair is the same one already known to be run-to-run non-deterministic (`dedi_gpu.md`).
- **Like-for-like cost on one CPU core:**
  - Ours takes 8.2–10.1 s per run on a fast core (X925) and 17–21 s on a slow one (A725). Repeats agree within 0.3 s.
  - One DEDi pair on the same run takes 430–840 s (fast) or 550–1,080 s (slow).
  - So a single DEDi grid setting costs about 30–100× a whole run of ours (50–100× on the fast core, 30–65× on the slow one).
  - **Full grid, like for like (correction, 2026-10-29):** in `dedi_fullgrid.md` the whole grid of run 43 cost 6.7× its 4 settings timed here. Scaled by that, the full grid costs about 1,800–1,900× ours on the fast core and 1,100–1,200× on the slow core. An earlier version of this report said 3,000–5,000×, using the 13 core-hour median of `dedi_fullgrid.md`; those runs were timed with 10 workers loading the machine, so they overstate the idle-core cost.
  - On the GPU the full grid costs about 340× ours (`h2h.md`).
- **Both methods scale about linearly in N.**

## Native CHOLMOD vs the stand-in (hard-120 run 43, Chen; CPU)

| Core | (λ, R) | Stand-in s (solver s) | Native s (solver s) | Stand-in / native | Same support |
| --- | --- | --- | --- | --- | --- |
| 0 (A725) | (10⁻³, 1) | 1073 (161) | 1082 (208) | 0.99 | yes |
| 0 | (1, 10⁻²) | 550 (102) | 607 (154) | 0.91 | yes |
| 0 | (0.1, 10⁻²) | 664 (131) | 726 (193) | 0.91 | yes |
| 0 | (1, 1) | 564 (104) | 591 (148) | 0.96 | no (known non-deterministic pair) |
| 15 (X925) | (10⁻³, 1) | 838 (106) | 821 (92) | 1.02 | yes |
| 15 | (1, 10⁻²) | 428 (67) | 455 (63) | 0.94 | yes |
| 15 | (0.1, 10⁻²) | 515 (85) | 545 (79) | 0.94 | yes |
| 15 | (1, 1) | 434 (68) | 454 (63) | 0.96 | no |

- **How native CHOLMOD was called:** scikit-sparse 0.5 from conda-forge, behind an adapter that exposes the 0.4 names DEDi imports (`benchmarks/dedi_native_shim`). DEDi's own code is unchanged.
- **The adapter needed three fixes before it ran:**
  1. It had been importing itself through a symlink.
  2. JAX arrays need converting to ndarrays.
  3. The converted array had to be made writable.
- These fixes affect only how the solver is called; DEDi's computation is unchanged. The failed attempts are logged in `results_timing/progress.log`.

## Runtime vs N (Lorenz, 10% noise, core 15 for ours, core 16 for DEDi; one DEDi pair (10⁻³, 1))

| N | Ours s (2 seeds) | Ours exact | DEDi, one pair, s | DEDi exact | Ratio, one pair / ours |
| --- | --- | --- | --- | --- | --- |
| 500 | 2.1, 2.1 | 0, 1 | 152 | 1 | 74× |
| 1000 | 4.0, 3.9 | 1, 1 | 279 | 1 | 71× |
| 2000 | 10.4, 8.1 | 1, 1 | 642 | 1 | 69× |
| 4000 | 16.2, 15.9 | 1, 1 | 1263 | 1 | 79× |
| 8000 | 36.6, 36.1 | 1, 1 | 2501 | 1 | 69× |

Both grow about linearly. The gap stays about 70× per DEDi pair, so about 2,500× for the full grid on CPU.

## Library size (N = 2000, ours)

| Library | Terms per equation | Seconds (2 seeds) | Exact |
| --- | --- | --- | --- |
| degree 3 | 20 | 10.4, 8.1 | 2/2 |
| degree 4 | 35 | 12.5, 11.9 | 2/2 |
| degree 5 | 56 | 18.2, 19.2 | 2/2 |

DEDi was not run with larger libraries. Its Hessian border grows with the library, so it would only get slower.

## What the paper should say

The paper's headline should be: "about 340× less compute than DEDi on a GPU and more than 1,000× less on one CPU core (about 1,100–1,900× on the same idle core)". The current "more than 5,000×" should go. The stand-in caveat can be dropped, because native CHOLMOD gives the same timing.
