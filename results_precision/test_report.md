# v1.2 test pass: tables

## Primary: sealed_clean.npz (200 runs, eta = 0)

v1.1 exact 0.265, v1.2 exact 1.000; **W/L 147/0, two-sided sign test p = 1.1e-44**; floor active in 200 runs; errors 0 []

- mean fp: v1.1 2.80, v1.2 0.00
- mean fn: v1.1 0.00, v1.2 0.00

| Cell | n | v1.1 | v1.2 | W/L |
| --- | --- | --- | --- | --- |
| N = 2000 | 100 | 0.220 | 1.000 | 78/0 |
| lorenz | 20 | 0.000 | 1.000 | 20/0 |
| N = 500 | 100 | 0.310 | 1.000 | 69/0 |
| rossler | 20 | 0.000 | 1.000 | 20/0 |
| chen | 20 | 0.100 | 1.000 | 18/0 |
| halvorsen | 20 | 1.000 | 1.000 | 0/0 |
| sprottA | 20 | 0.450 | 1.000 | 11/0 |
| sprottB | 20 | 0.200 | 1.000 | 16/0 |
| sprottC | 20 | 0.500 | 1.000 | 10/0 |
| sprottD | 20 | 0.050 | 1.000 | 19/0 |
| sprottE | 20 | 0.000 | 1.000 | 20/0 |
| rabinovich | 20 | 0.350 | 1.000 | 13/0 |

v1.2 non-exact runs: 0: 

## Guards

| Archive | Runs | floor active | coefficients changed | support changed | v1.1 exact | v1.2 exact | W/L | v1.1 arm = stored v1.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guard_white | 800 | 44 | 44 | 0 | 0.889 | 0.889 | 0/0 | 800/800 |
| guard_colored | 800 | 24 | 24 | 0 | 0.805 | 0.805 | 0/0 | 800/800 |
| guard_student | 800 | 75 | 75 | 0 | 0.905 | 0.905 | 0/0 | 800/800 |
| guard_nd | 400 | 35 | 35 | 0 | 0.917 | 0.917 | 0/0 | - |

## Secondary (NOT independent): ODEBench sigma = 0, fit IC 0

| Subset | n | v1.1 | v1.2 |
| --- | --- | --- | --- |
| exact, in library | 23 | 0.391 | 0.870 |
| reconstruction R2 > 0.9 | 63 | 0.746 | 0.746 |
| generalization R2 > 0.9 | 63 | 0.556 | 0.556 |

exact W/L (in library) 11/0; errors v1.1 0, v1.2 0; floor active 18/63
- D = 1: in-library exact v1.1 0.750, v1.2 0.875 (n = 8)
- D = 2: in-library exact v1.1 0.000, v1.2 1.000 (n = 6)
- D = 3: in-library exact v1.1 0.250, v1.2 0.750 (n = 8)
- D = 4: in-library exact v1.1 1.000, v1.2 1.000 (n = 1)

## pytest

```
equivalence: 42 runs, 0 differing, max |xi_frozen - xi_new| = 0
.

======================== 2 passed in 123.37s (0:02:03) =========================
```
