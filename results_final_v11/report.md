Repro on dev coloured (30 runs): v1.1 = stored ar1_gauss in 30/30; v1.0.1 = stored full in 30/30 (coef_err and gap, exact equality)

## Primary: `test_colored2.json` (800 runs)

| Arm | Exact | Mean F1 |
| --- | --- | --- |
| v1.0.1 (Huber in AR(1) likelihood) | 0.754 | 0.913 |
| v1.1 (Gaussian in AR(1) likelihood) | 0.805 | 0.931 |

**v1.1 vs v1.0.1: W/L = 43/2, two-sided sign test p = 5.9e-11**

Routing (v1.0.1, v1.1): {('ar1', 'ar1'): 800}; exceptions: 0; identical xi: 0/800; identical xi among white-routed: 0/0

Median coefficient error on runs both get exact (601): v1.0.1 0.63%, v1.1 0.63%; median seconds per run (joint stage): v1.0.1 6.4, v1.1 6.5

| Cell | n | v1.0.1 | v1.1 | W/L | p |
| --- | --- | --- | --- | --- | --- |
| η = 1% | 200 | 0.925 | 0.940 | 3/0 | 0.250 |
| η = 5% | 200 | 0.770 | 0.870 | 21/1 | 1.1e-05 |
| η = 10% | 200 | 0.740 | 0.785 | 10/1 | 0.012 |
| η = 20% | 200 | 0.580 | 0.625 | 9/0 | 0.0039 |
| N = 500 | 400 | 0.752 | 0.765 | 7/2 | 0.180 |
| N = 2000 | 400 | 0.755 | 0.845 | 36/0 | 2.9e-11 |

| System | n | v1.0.1 | v1.1 | W/L |
| --- | --- | --- | --- | --- |
| chen | 80 | 0.500 | 0.662 | 14/1 |
| halvorsen | 80 | 0.925 | 0.963 | 3/0 |
| lorenz | 80 | 0.512 | 0.738 | 18/0 |
| rabinovich | 80 | 0.062 | 0.100 | 3/0 |
| rossler | 80 | 0.550 | 0.600 | 4/0 |
| sprottA | 80 | 1.000 | 1.000 | 0/0 |
| sprottB | 80 | 1.000 | 1.000 | 0/0 |
| sprottC | 80 | 0.988 | 0.988 | 1/1 |
| sprottD | 80 | 1.000 | 1.000 | 0/0 |
| sprottE | 80 | 1.000 | 1.000 | 0/0 |

## Guards (v1.1 vs the v1.0.1 re-run of the same archive)

| Archive | Runs | Routed to AR(1) by v1.0.1 | Answers changed | v1.0.1 exact | v1.1 exact | W/L |
| --- | --- | --- | --- | --- | --- | --- |
| white, `joint_sealed.npz` (primary guard) | 800 | 0 | 0 | 0.889 | 0.889 | 0/0 |
| t₃, `sealed_student.npz` (primary guard) | 800 | 0 | 0 | 0.905 | 0.905 | 0/0 |
| coarse, AR(1)-routed runs only (secondary) | 12 | 12 | 12 | 0.583 | 0.833 | 3/0 | changed: [32, 48, 56, 60, 76, 188, 192, 207, 211, 224, 236, 610]
| held-out 2, AR(1)-routed runs only (secondary) | 22 | 22 | 22 | 0.045 | 0.091 | 2/1 | changed: [802, 806, 810, 814, 818, 819, 826, 833, 834, 841, 842, 846, 850, 851, 853, 854, 858, 861, 862, 865]
| old coloured test `sealed_colored.npz` (seen before; non-independent) | 800 | 799 | 799 | 0.761 | 0.812 | 42/1 | changed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
