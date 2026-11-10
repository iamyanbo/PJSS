## Integrity

- Sealed (white): 160 runs; variant errors 0; weak re-run equals cached seed 160/160; full = frozen test pass on 160/160 runs (exact flag and coefficient error)
- Coloured AR(1): 89 runs; variant errors 0; seeds computed here (final_model.default_starts); full = frozen test pass on 89/89 runs (exact flag and coefficient error)
- Hard 40: 40 runs; variant errors 0; weak re-run equals cached seed 40/40; full = frozen test pass on 40/40 runs (exact flag and coefficient error)
- dysts D=3: 287 runs; variant errors 0; seeds computed here (final_model.default_starts); full = frozen dysts run (results_dysts/test_final.json) on 287/287 runs; this subsample's full rate vs all 4,880 D=3 runs: 0.899 vs 0.898

## Leave-one-out from the full model

Cells: exact rate · full+/var+ (sign-test p) · median coef. error on jointly exact runs · median s/run (joint stage).

| Variant | Sealed (white) | Coloured AR(1) | Hard 40 | dysts D=3 | Pooled full+/var+ (p) |
| --- | --- | --- | --- | --- | --- |
| **Full final model** | **0.869** · – · 0.24% · 8.4s | **0.730** · – · 0.63% · 6.0s | **0.700** · – · 0.60% · 24.2s | **0.899** · – · 0.26% · 2.5s | – |
| Weak best-subset only (no joint states) | 0.769 · 18/2 (4e-4) · 0.58% | 0.640 · 8/0 (8e-3) · 1.46% | 0.000 · 28/0 (7e-9) · nan% | 0.829 · 20/0 (2e-6) · 0.62% | 74/2 (8e-20) |
| Hybrid (weak/flow, no joint states) | 0.812 · 10/1 (0.01) · 0.65% | 0.270 · 41/0 (9e-13) · 1.54% | 0.400 · 12/0 (5e-4) · 1.44% | 0.819 · 24/1 (2e-6) · 0.62% | 87/2 (1e-23) |
| − profiled-quadratic proposals (refit starts only) | 0.819 · 8/0 (8e-3) · 0.23% · 4.4s | 0.652 · 7/0 (0.02) · 0.58% · 3.2s | 0.475 · 9/0 (4e-3) · 0.83% · 10.3s | 0.871 · 8/0 (8e-3) · 0.25% · 1.3s | 32/0 (5e-10) |
| Greedy (stepwise) instead of best-subset proposals | 0.819 · 8/0 (8e-3) · 0.23% · 8.6s | 0.674 · 5/0 (0.06) · 0.62% · 6.9s | 0.475 · 9/0 (4e-3) · 0.83% · 23.7s | 0.878 · 6/0 (0.03) · 0.25% · 2.6s | 28/0 (7e-9) |
| − multiple starts (weak start only) | 0.869 · 0/0 (1) · 0.24% · 5.8s | 0.775 · 2/6 (0.29) · 0.62% · 3.5s | 0.675 · 1/0 (1) · 0.63% · 25.3s | 0.899 · 0/0 (1) · 0.26% · 1.8s | 3/6 (0.51) |
| − σ/φ self-calibration (HF σ, φ = 1) | 0.869 · 0/0 (1) · 0.23% · 8.5s | 0.730 · 0/0 (1) · 0.63% · 6.8s | 0.700 · 0/0 (1) · 0.60% · 21.8s | 0.899 · 0/0 (1) · 0.26% · 2.5s | 0/0 (1) |
| − φ autocorrelation inflation | 0.869 · 0/0 (1) · 0.24% · 8.5s | 0.742 · 0/1 (1) · 0.63% · 6.8s | 0.700 · 0/0 (1) · 0.60% · 22.6s | 0.899 · 0/0 (1) · 0.26% · 2.5s | 0/1 (1) |
| − Huber (Gaussian loss) | 0.875 · 0/1 (1) · 0.22% · 8.2s | 0.831 · 0/9 (4e-3) · 0.63% · 6.1s | 0.700 · 0/0 (1) · 0.59% · 21.2s | 0.895 · 1/0 (1) · 0.26% · 2.5s | 1/10 (0.01) |
| − auto sub-steps (s = 2) | 0.875 · 0/1 (1) · 0.23% · 6.9s | 0.730 · 1/1 (1) · 0.63% · 4.1s | 0.700 · 0/0 (1) · 0.56% · 11.7s | 0.885 · 6/2 (0.29) · 0.28% · 2.4s | 7/4 (0.55) |
| − complexity prior (degw = 0) | 0.856 · 2/0 (0.50) · 0.23% · 8.4s | 0.697 · 3/0 (0.25) · 0.62% · 5.8s | 0.625 · 3/0 (0.25) · 0.65% · 21.3s | 0.892 · 2/0 (0.50) · 0.25% · 2.6s | 10/0 (2e-3) |
| − AR(1) auto-switch (noise = white) | 0.869 · 0/0 (1) · 0.24% · 6.4s | 0.674 · 9/4 (0.27) · 0.73% · 4.4s | 0.700 · 0/0 (1) · 0.60% · 15.9s | 0.899 · 0/0 (1) · 0.26% · 1.9s | 9/4 (0.27) |
| Oracle noise level (not deployable) | 0.875 · 0/1 (1) · 0.23% · 8.3s | 0.820 · 0/8 (8e-3) · 0.60% · 4.8s | 0.700 · 0/0 (1) · 0.60% · 20.5s | 0.902 · 0/1 (1) · 0.25% · 2.6s | – |

## Cumulative ladder (weak best-subset → final)

Cells: exact rate · wins/losses vs the previous rung (sign-test p).

| Rung | Sealed (white) | Coloured AR(1) | Hard 40 | dysts D=3 |
| --- | --- | --- | --- | --- |
| Weak best-subset (χ²s) | 0.769 | 0.640 | 0.000 | 0.829 |
| + joint states (refit weak start; s=2, HF σ, Gaussian) | 0.769 · 0/0 (1) | 0.640 · 0/0 (1) | 0.000 · 0/0 (1) | 0.829 · 0/0 (1) |
| + profiled-quadratic best-subset proposals | 0.856 · 16/2 (1e-3) | 0.000 · 0/57 (1e-17) | 0.625 · 25/0 (6e-8) | 0.864 · 18/8 (0.08) |
| + multiple starts (hybrid, flow) | 0.856 · 0/0 (1) | 0.000 · 0/0 (1) | 0.650 · 1/0 (1) | 0.861 · 0/1 (1) |
| + σ/φ self-calibration | 0.856 · 1/1 (1) | 0.629 · 56/0 (3e-17) | 0.650 · 0/0 (1) | 0.868 · 2/0 (0.50) |
| + auto sub-steps | 0.856 · 0/0 (1) | 0.596 · 0/3 (0.25) | 0.625 · 0/1 (1) | 0.889 · 7/1 (0.07) |
| + Huber loss (= config C) | 0.856 · 0/0 (1) | 0.629 · 3/0 (0.25) | 0.625 · 1/1 (1) | 0.892 · 1/0 (1) |
| + complexity prior (= C + prior) | 0.869 · 2/0 (0.50) | 0.674 · 5/1 (0.22) | 0.700 · 3/0 (0.25) | 0.899 · 2/0 (0.50) |
| + AR(1) auto-switch (= final) | 0.869 · 0/0 (1) | 0.730 · 9/4 (0.27) | 0.700 · 0/0 (1) | 0.899 · 0/0 (1) |

## Detail

- Sealed (white): median weak-selector time 0.36s; AR(1) chosen by full model in 0%; median auto sub-steps 3
  - noise=0.01: n=40 full 0.97 / weak 0.95; noise=0.05: n=40 full 0.88 / weak 0.88; noise=0.1: n=40 full 0.85 / weak 0.65; noise=0.2: n=40 full 0.78 / weak 0.60
  - n=500: n=80 full 0.84 / weak 0.75; n=2000: n=80 full 0.90 / weak 0.79
- Coloured AR(1): median weak-selector time 0.14s, median all-seeds time 1.6s; AR(1) chosen by full model in 100%; median auto sub-steps 3
  - noise=0.01: n=23 full 0.87 / weak 0.83; noise=0.05: n=22 full 0.73 / weak 0.64; noise=0.1: n=22 full 0.68 / weak 0.55; noise=0.2: n=22 full 0.64 / weak 0.55
  - n=500: n=44 full 0.75 / weak 0.61; n=2000: n=45 full 0.71 / weak 0.67
- Hard 40: median weak-selector time 0.61s; AR(1) chosen by full model in 0%; median auto sub-steps 4
  - noise=0.2: n=40 full 0.70 / weak 0.00
  - n=2000: n=40 full 0.70 / weak 0.00
- dysts D=3: median weak-selector time 0.14s, median all-seeds time 1.1s; AR(1) chosen by full model in 1%; median auto sub-steps 3
  - noise=0.01: n=72 full 0.99 / weak 0.99; noise=0.05: n=72 full 0.88 / weak 0.85; noise=0.1: n=72 full 0.90 / weak 0.78; noise=0.2: n=71 full 0.83 / weak 0.70
  - n=500: n=143 full 0.86 / weak 0.78; n=2000: n=144 full 0.94 / weak 0.88
