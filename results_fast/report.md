
### Pooled (80 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 66/80 | 0.949 | 0.0077 | 15 |
| **ours v1.2, re-timed on cores 0-4** | 66/80 | 0.949 | 0.0077 | 14.4 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 24/80 | 0.728 | 0.0194 | 0.27 |
|   WSINDy variant: K = all centres | 24/80 | 0.725 | 0.0194 | 0.391 |
|   WSINDy variant: no pre-smoothing | 26/80 | 0.721 | 0.0127 | 0.26 |
| weak STLSQ, CV (pysindy) | 14/80 | 0.483 | 0.0247 | 1.61 |
| weak STLSQ, AICc (pysindy) | 3/80 | 0.362 | 0.0253 | 0.425 |
| weak E-SINDy, CV (pysindy) | 8/80 | 0.459 | 0.0172 | 27.6 |
| weak E-SINDy, AICc (pysindy) | 0/80 | 0.339 | – | 4.86 |
| SFD STLSQ, CV (pysindy) | 17/80 | 0.671 | 0.0179 | 3.06 |
| SFD STLSQ, AICc (pysindy) | 2/80 | 0.370 | 0.0089 | 0.539 |
| SFD E-SINDy, CV (pysindy) | 17/80 | 0.677 | 0.0142 | 59.8 |
| SFD E-SINDy, AICc (pysindy) | 3/80 | 0.390 | 0.0073 | 11 |
| in-repo weak chi2s (select_k) | 36/80 | 0.909 | 0.0166 | 0.476 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 17/80 | 0.654 | 0.0206 | 126 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 2/10 | 0.628 | 0.0179 | 1.14e+03 |
| DEDi single pair (1e-3, 1), val | 44/80 | – | – | 120 |
| DEDi single pair (1e-3, 1), BIC | 44/80 | – | – | 120 |
| DEDi row lambda=1e-3, val | 53/80 | – | – | 569 |
| DEDi row lambda=1e-3, BIC | 56/80 | – | – | 569 |
| DEDi full grid, val | 55/80 | 0.854 | 0.0080 | 5.93e+03 |
| DEDi full grid, BIC | 61/80 | 0.864 | 0.0084 | 5.93e+03 |

### hard (16 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 15/16 | 0.991 | 0.0087 | 18.2 |
| **ours v1.2, re-timed on cores 0-4** | 15/16 | 0.991 | 0.0087 | 17.5 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 1/16 | 0.640 | 0.0607 | 0.278 |
|   WSINDy variant: K = all centres | 1/16 | 0.637 | 0.0607 | 0.405 |
|   WSINDy variant: no pre-smoothing | 0/16 | 0.588 | – | 0.266 |
| weak STLSQ, CV (pysindy) | 0/16 | 0.385 | – | 1.67 |
| weak STLSQ, AICc (pysindy) | 0/16 | 0.312 | – | 0.437 |
| weak E-SINDy, CV (pysindy) | 0/16 | 0.370 | – | 28 |
| weak E-SINDy, AICc (pysindy) | 0/16 | 0.330 | – | 4.97 |
| SFD STLSQ, CV (pysindy) | 0/16 | 0.584 | – | 3.42 |
| SFD STLSQ, AICc (pysindy) | 0/16 | 0.276 | – | 0.589 |
| SFD E-SINDy, CV (pysindy) | 0/16 | 0.594 | – | 66.8 |
| SFD E-SINDy, AICc (pysindy) | 0/16 | 0.295 | – | 12.3 |
| in-repo weak chi2s (select_k) | 0/16 | 0.909 | – | 0.507 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 0/16 | 0.583 | – | 156 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 0/2 | 0.583 | – | 1.58e+03 |
| DEDi single pair (1e-3, 1), val | 11/16 | – | – | 248 |
| DEDi single pair (1e-3, 1), BIC | 11/16 | – | – | 248 |
| DEDi row lambda=1e-3, val | 14/16 | – | – | 1.4e+03 |
| DEDi row lambda=1e-3, BIC | 15/16 | – | – | 1.4e+03 |
| DEDi full grid, val | 13/16 | 0.958 | 0.0078 | 9.54e+03 |
| DEDi full grid, BIC | 15/16 | 0.992 | 0.0091 | 9.54e+03 |

### white (16 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 14/16 | 0.963 | 0.0059 | 15.6 |
| **ours v1.2, re-timed on cores 0-4** | 14/16 | 0.963 | 0.0059 | 13.7 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 4/16 | 0.775 | 0.0137 | 0.267 |
|   WSINDy variant: K = all centres | 4/16 | 0.776 | 0.0137 | 0.391 |
|   WSINDy variant: no pre-smoothing | 7/16 | 0.817 | 0.0098 | 0.263 |
| weak STLSQ, CV (pysindy) | 2/16 | 0.452 | 0.0176 | 1.63 |
| weak STLSQ, AICc (pysindy) | 0/16 | 0.332 | – | 0.426 |
| weak E-SINDy, CV (pysindy) | 2/16 | 0.512 | 0.0257 | 27.6 |
| weak E-SINDy, AICc (pysindy) | 0/16 | 0.368 | – | 4.89 |
| SFD STLSQ, CV (pysindy) | 4/16 | 0.678 | 0.0128 | 3.14 |
| SFD STLSQ, AICc (pysindy) | 0/16 | 0.373 | – | 0.544 |
| SFD E-SINDy, CV (pysindy) | 5/16 | 0.676 | 0.0128 | 61.6 |
| SFD E-SINDy, AICc (pysindy) | 0/16 | 0.439 | – | 11.4 |
| in-repo weak chi2s (select_k) | 9/16 | 0.923 | 0.0098 | 0.478 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 5/16 | 0.675 | 0.0131 | 129 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 1/2 | 0.500 | 0.0206 | 921 |
| DEDi single pair (1e-3, 1), val | 12/16 | – | – | 124 |
| DEDi single pair (1e-3, 1), BIC | 12/16 | – | – | 124 |
| DEDi row lambda=1e-3, val | 14/16 | – | – | 591 |
| DEDi row lambda=1e-3, BIC | 14/16 | – | – | 591 |
| DEDi full grid, val | 14/16 | 0.912 | 0.0074 | 6.05e+03 |
| DEDi full grid, BIC | 14/16 | 0.901 | 0.0074 | 6.05e+03 |

### coloured (16 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 10/16 | 0.897 | 0.0160 | 33.5 |
| **ours v1.2, re-timed on cores 0-4** | 10/16 | 0.897 | 0.0160 | 31.4 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 4/16 | 0.624 | 0.0308 | 0.27 |
|   WSINDy variant: K = all centres | 4/16 | 0.614 | 0.0308 | 0.391 |
|   WSINDy variant: no pre-smoothing | 3/16 | 0.585 | 0.0263 | 0.258 |
| weak STLSQ, CV (pysindy) | 2/16 | 0.409 | 0.0477 | 1.61 |
| weak STLSQ, AICc (pysindy) | 0/16 | 0.306 | – | 0.428 |
| weak E-SINDy, CV (pysindy) | 0/16 | 0.358 | – | 27.4 |
| weak E-SINDy, AICc (pysindy) | 0/16 | 0.285 | – | 4.86 |
| SFD STLSQ, CV (pysindy) | 3/16 | 0.623 | 0.0320 | 3.14 |
| SFD STLSQ, AICc (pysindy) | 0/16 | 0.261 | – | 0.553 |
| SFD E-SINDy, CV (pysindy) | 3/16 | 0.628 | 0.0331 | 59.9 |
| SFD E-SINDy, AICc (pysindy) | 0/16 | 0.250 | – | 11 |
| in-repo weak chi2s (select_k) | 7/16 | 0.869 | 0.0320 | 0.475 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 2/16 | 0.578 | 0.0637 | 133 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 0/2 | 0.498 | – | 1.65e+03 |
| DEDi single pair (1e-3, 1), val | 1/16 | – | – | 104 |
| DEDi single pair (1e-3, 1), BIC | 1/16 | – | – | 104 |
| DEDi row lambda=1e-3, val | 0/16 | – | – | 408 |
| DEDi row lambda=1e-3, BIC | 1/16 | – | – | 408 |
| DEDi full grid, val | 5/16 | 0.639 | 0.0296 | 4.25e+03 |
| DEDi full grid, BIC | 5/16 | 0.634 | 0.0296 | 4.25e+03 |

### t3 (16 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 14/16 | 0.956 | 0.0039 | 13.5 |
| **ours v1.2, re-timed on cores 0-4** | 14/16 | 0.956 | 0.0039 | 13.1 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 5/16 | 0.762 | 0.0234 | 0.275 |
|   WSINDy variant: K = all centres | 5/16 | 0.756 | 0.0234 | 0.402 |
|   WSINDy variant: no pre-smoothing | 8/16 | 0.833 | 0.0147 | 0.265 |
| weak STLSQ, CV (pysindy) | 6/16 | 0.633 | 0.0282 | 1.64 |
| weak STLSQ, AICc (pysindy) | 1/16 | 0.410 | 0.0253 | 0.426 |
| weak E-SINDy, CV (pysindy) | 3/16 | 0.499 | 0.0183 | 27.7 |
| weak E-SINDy, AICc (pysindy) | 0/16 | 0.317 | – | 4.93 |
| SFD STLSQ, CV (pysindy) | 6/16 | 0.683 | 0.0206 | 3.09 |
| SFD STLSQ, AICc (pysindy) | 0/16 | 0.386 | – | 0.549 |
| SFD E-SINDy, CV (pysindy) | 4/16 | 0.677 | 0.0224 | 59.7 |
| SFD E-SINDy, AICc (pysindy) | 0/16 | 0.411 | – | 10.9 |
| in-repo weak chi2s (select_k) | 9/16 | 0.925 | 0.0180 | 0.475 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 4/16 | 0.691 | 0.0246 | 120 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 0/2 | 0.679 | – | 1.24e+03 |
| DEDi single pair (1e-3, 1), val | 11/16 | – | – | 126 |
| DEDi single pair (1e-3, 1), BIC | 11/16 | – | – | 126 |
| DEDi row lambda=1e-3, val | 12/16 | – | – | 573 |
| DEDi row lambda=1e-3, BIC | 13/16 | – | – | 573 |
| DEDi full grid, val | 12/16 | 0.895 | 0.0098 | 5.95e+03 |
| DEDi full grid, BIC | 13/16 | 0.894 | 0.0096 | 5.95e+03 |

### short (16 runs)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 13/16 | 0.937 | 0.0068 | 3.86 |
| **ours v1.2, re-timed on cores 0-4** | 13/16 | 0.937 | 0.0068 | 3.83 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 10/16 | 0.840 | 0.0158 | 0.126 |
|   WSINDy variant: K = all centres | 10/16 | 0.840 | 0.0158 | 0.127 |
|   WSINDy variant: no pre-smoothing | 8/16 | 0.782 | 0.0122 | 0.124 |
| weak STLSQ, CV (pysindy) | 4/16 | 0.537 | 0.0263 | 1.38 |
| weak STLSQ, AICc (pysindy) | 2/16 | 0.452 | 0.0263 | 0.272 |
| weak E-SINDy, CV (pysindy) | 3/16 | 0.556 | 0.0114 | 25 |
| weak E-SINDy, AICc (pysindy) | 0/16 | 0.394 | – | 4.27 |
| SFD STLSQ, CV (pysindy) | 4/16 | 0.786 | 0.0140 | 2.01 |
| SFD STLSQ, AICc (pysindy) | 2/16 | 0.555 | 0.0089 | 0.331 |
| SFD E-SINDy, CV (pysindy) | 5/16 | 0.810 | 0.0136 | 41.5 |
| SFD E-SINDy, AICc (pysindy) | 3/16 | 0.558 | 0.0073 | 7.12 |
| in-repo weak chi2s (select_k) | 11/16 | 0.921 | 0.0123 | 0.0499 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 6/16 | 0.745 | 0.0157 | 86.1 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 1/2 | 0.881 | 0.0152 | 669 |
| DEDi single pair (1e-3, 1), val | 9/16 | – | – | 72.4 |
| DEDi single pair (1e-3, 1), BIC | 9/16 | – | – | 72.4 |
| DEDi row lambda=1e-3, val | 13/16 | – | – | 321 |
| DEDi row lambda=1e-3, BIC | 13/16 | – | – | 321 |
| DEDi full grid, val | 11/16 | 0.868 | 0.0054 | 2.32e+03 |
| DEDi full grid, BIC | 14/16 | 0.900 | 0.0056 | 2.32e+03 |

### Subset with ARGOS at its default B = 2000 (10 runs, first rounds of order.json)

| method | exact | mean F1 | median coef err (exact runs) | median s/run |
| --- | --- | --- | --- | --- |
| **ours v1.2** (h2h timing: core 13, loaded) | 9/10 | 0.970 | 0.0088 | 14.4 |
| **ours v1.2, re-timed on cores 0-4** | 9/10 | 0.970 | 0.0088 | 13.9 |
| WSINDy + MSTLS (authors' MATLAB, ported) | 5/10 | 0.741 | 0.0383 | 0.274 |
|   WSINDy variant: K = all centres | 5/10 | 0.741 | 0.0383 | 0.401 |
|   WSINDy variant: no pre-smoothing | 4/10 | 0.664 | 0.0208 | 0.263 |
| weak STLSQ, CV (pysindy) | 1/10 | 0.400 | 0.0201 | 1.62 |
| weak STLSQ, AICc (pysindy) | 0/10 | 0.331 | – | 0.413 |
| weak E-SINDy, CV (pysindy) | 1/10 | 0.474 | 0.0353 | 27.2 |
| weak E-SINDy, AICc (pysindy) | 0/10 | 0.318 | – | 4.77 |
| SFD STLSQ, CV (pysindy) | 2/10 | 0.678 | 0.0166 | 3.11 |
| SFD STLSQ, AICc (pysindy) | 0/10 | 0.411 | – | 0.545 |
| SFD E-SINDy, CV (pysindy) | 2/10 | 0.666 | 0.0153 | 59.5 |
| SFD E-SINDy, AICc (pysindy) | 1/10 | 0.505 | 0.0160 | 10.8 |
| in-repo weak chi2s (select_k) | 5/10 | 0.928 | 0.0281 | 0.479 |
| ARGOS lasso, B = 200 bootstraps (R pkg) | 2/10 | 0.628 | 0.0179 | 132 |
| ARGOS lasso, B = 2000 = pkg default (R pkg) | 2/10 | 0.628 | 0.0179 | 1.14e+03 |
| DEDi single pair (1e-3, 1), val | 4/10 | – | – | 127 |
| DEDi single pair (1e-3, 1), BIC | 4/10 | – | – | 127 |
| DEDi row lambda=1e-3, val | 7/10 | – | – | 642 |
| DEDi row lambda=1e-3, BIC | 7/10 | – | – | 642 |
| DEDi full grid, val | 6/10 | 0.765 | 0.0094 | 5.03e+03 |
| DEDi full grid, BIC | 7/10 | 0.769 | 0.0092 | 5.03e+03 |

### Pooled accuracy-runtime frontier (methods run on all 80 runs)

Methods not dominated by a faster method with at least as many exact runs:

- WSINDy + MSTLS (authors' MATLAB, ported): 24/80 exact at median 0.27 s/run
- in-repo weak chi2s (select_k): 36/80 exact at median 0.476 s/run
- **ours v1.2, re-timed on cores 0-4**: 66/80 exact at median 14.4 s/run

WSINDy port vs authors' MATLAB code in Octave: identical support on 80/80 runs, max relative coefficient difference 4.5e-13.
