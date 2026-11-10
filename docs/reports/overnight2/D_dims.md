# Track D: profiled joint flow beyond 3-D

## Method

`joint_nd.py` is a standalone D-dimensional copy of joint_discover (config C, unchanged: rho 0.03, automatic sub-steps, sigma/phi self-calibration, Huber 2, stiff final refit, BIC). Two things change with dimension:
- **Banded solve:** states are interleaved by time, so the state block stays banded (upper bandwidth 2D−1) and the Schur/banded solve is unchanged.
- **Subset search:** per-equation ranking on the profiled quadratic uses `joint_biglib.top_subsets`, which switches to a beam search for large libraries. At most 1,024 Cartesian combinations are tried.

The library is all monomials of degree ≤ 2 in D variables (15 terms per equation at D = 4, 66 at D = 10).

The starts are weak-form best-subset regressions per equation (size chosen by BIC), plus two runner-up variants (`weak_starts`). Truth is never used.

## Systems

All five are in `nd_systems.py`, checked bounded and chaotic; λ_max is the largest Lyapunov exponent, estimated from the separation of two nearby trajectories.

| System | D | Library terms per equation | λ_max |
| --- | --- | --- | --- |
| Hyperchaotic Rössler | 4 | 15 | 0.12 |
| Lorenz-96, F = 8 | 5 | 21 | 0.47 |
| Lorenz-96, F = 8 | 6 | 28 | 0.92 |
| Two diffusively coupled Lorenz systems (ε = 0.5, not synchronised) | 6 | 28 | 0.97 |
| Lorenz-96, F = 8 | 10 | 66 | 1.21 |

Archives use the benchmark noise protocol (1/5/10/20% of each component's standard deviation) at N = 500 and 2000, with 10 seeds each:
- dev: `data/nd_dev.npz`, seed 20261301;
- test: `data/nd_test.npz`, seed 20261302, 400 runs each.

## Baselines

These run on the same weak problem and library:
- **Weak BIC:** the joint method's own start.
- **WSINDy-style STLSQ:** the threshold is chosen on dev (λ = 0.2 in scaled units) and fixed for test. A per-run oracle threshold is also reported.

## Results

The configuration was frozen with no changes after dev (md5 in `results_nd/frozen.md5`). Single test pass:

| System | D | Weak BIC | STLSQ (λ = 0.2) | STLSQ, oracle λ | **Joint** | W/L vs best baseline | Flagged | Accuracy if unflagged | Coefficient error | Median s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hyperchaotic Rössler | 4 | 0.025 | 0.000 | 0.000 | **0.812** | 63/0 | 31% | 96.4% | 0.52% | 45 |
| Lorenz-96 | 5 | 0.025 | 0.575 | 0.662 | **1.000** | 34/0 | 2% | 100% | 0.33% | 13 |
| Coupled Lorenz | 6 | 0.075 | 0.087 | 0.113 | **0.675** | 42/0 | 42% | 93.5% | 0.23% | 19 |
| Lorenz-96 | 6 | 0.025 | 0.650 | 0.688 | **1.000** | 28/0 | 0% | 100% | 0.31% | 30 |
| Lorenz-96 | 10 | 0.000 | 0.400 | 0.400 | **1.000** | 48/0 | 1% | 100% | 0.34% | 136 |
| **All** | | 0.030 | 0.343 | 0.372 | **0.897** | 215/0 | | | | |

The dev results (0.890 overall) match the test results.

Test by noise level and N (STLSQ → joint):

| Noise | N = 500 | N = 2000 |
| --- | --- | --- |
| 1% | 0.40 → 1.00 | 0.74 → 1.00 |
| 5% | 0.18 → 0.94 | 0.60 → 1.00 |
| 10% | 0.00 → 0.72 | 0.58 → 1.00 |
| 20% | 0.00 → 0.66 | 0.24 → 0.86 |

## Honest notes

- **The baselines are my own minimal weak-form implementations,** not the 3-D χ² selector, pysindy, flow, DEDi or ODR-BINDy. Weak BIC badly overselects in D dimensions, which is why its numbers are near 0. STLSQ given an oracle threshold is the fairer reference.
- **Failures are mostly one-for-one term swaps at N = 500.** On coupled Lorenz the weak 0.5 coupling term is confused with correlated neighbours. Hyperchaotic Rössler failures are at 10–20% noise with N = 500. The flag catches most of them: unflagged accuracy is 93–96% on those two systems.
- **Only degree-2 libraries were tested.** Degree 3 at D = 10 would be 286 terms per equation and was not tried.
- **Runtime grows with D and N** (Lorenz-96 at D = 10, N = 2000: about 3–4 min). The profiled quadratic over all M·D coefficients dominates the cost, since it is a banded solve with M·D right-hand sides.
- **Compute:** everything ran on this PC with 3 workers. The Spark was fully loaded and was not used.

## Files

- Code: `joint_nd.py`, `nd_systems.py`, `nd_bench.py`, `nd_report.py`, `scratch_nd_smoke.py`.
- Data: `data/nd_dev.npz`, `data/nd_test.npz`.
- Results: `results_nd/dev.json`, `test.json`, logs, `frozen.md5`.

Reproduce with `python nd_bench.py --data data/nd_test.npz --output <new>.json`, then `python nd_report.py <new>.json 0.2`.
