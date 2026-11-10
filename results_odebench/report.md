## PRIMARY: fit on IC 0, rep 0 (378 runs)

**Reconstruction** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 40 | 33 | 38 | 38 | 37 | 35 | 36.8 |
| D >= 2 (40 systems) | 62 | 52 | 60 | 60 | 57 | 55 | 57.9 |
| in_library (23 systems) | 52 | 43 | 48 | 43 | 30 | 35 | 42.0 |
| in_library, D >= 2 (15 systems) | 80 | 67 | 73 | 67 | 47 | 53 | 64.4 |
| poly_high_degree (14 systems) | 29 | 29 | 29 | 29 | 36 | 29 | 29.8 |
| non_polynomial (26 systems) | 35 | 27 | 35 | 38 | 42 | 38 | 35.9 |

**Generalization (other IC)** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 25 | 24 | 27 | 24 | 19 | 21 | 23.3 |
| D >= 2 (40 systems) | 40 | 38 | 42 | 38 | 30 | 32 | 36.7 |
| in_library (23 systems) | 52 | 39 | 48 | 39 | 30 | 35 | 40.6 |
| in_library, D >= 2 (15 systems) | 80 | 60 | 73 | 60 | 47 | 53 | 62.2 |
| poly_high_degree (14 systems) | 14 | 21 | 21 | 21 | 21 | 14 | 19.0 |
| non_polynomial (26 systems) | 8 | 12 | 12 | 12 | 8 | 12 | 10.3 |

**Structure** (exact support recovery, in-library systems)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_library (23) | 17 | 22 | 26 | 35 | 26 | 22 | 24.6 |
| D = 1 (8) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| D = 2 (6) | 0 | 50 | 33 | 67 | 33 | 17 | 33.3 |
| D = 3 (8) | 38 | 25 | 38 | 50 | 38 | 38 | 37.5 |
| D = 4 (1) | 100 | 0 | 100 | 0 | 100 | 100 | 66.7 |

median coefficient error on exact runs: 0.26 %

**Structure, negligible terms ignored** (POST-HOC, not pre-registered: exact support ignoring terms contributing < 1% of the equation's RMS)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_library (23) | 57 | 39 | 30 | 35 | 26 | 39 | 37.7 |
| D = 2 (6) | 100 | 67 | 50 | 67 | 33 | 67 | 63.9 |
| D = 3 (8) | 75 | 62 | 38 | 50 | 38 | 50 | 52.1 |
| D = 4 (1) | 100 | 0 | 100 | 0 | 100 | 100 | 66.7 |

model exceptions: 139 of 378 runs  {(1, 'IndexError'): 138, (2, 'IndexError'): 1}
integration failures among fitted runs: reconstruction 32, generalization 33 of 239
median fit time: 0.7 s; by D: D=1: 0.0 s, D=2: 1.0 s, D=3: 15.2 s, D=4: 7.2 s

## Sensitivity: reconstruction from the clean IC (rep 0)

**Reconstruction, clean y0** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 40 | 33 | 38 | 40 | 38 | 35 | 37.3 |
| D >= 2 (40 systems) | 62 | 52 | 60 | 62 | 60 | 55 | 58.8 |
| in_library (23 systems) | 52 | 43 | 48 | 43 | 35 | 35 | 42.8 |
| in_library, D >= 2 (15 systems) | 80 | 67 | 73 | 67 | 53 | 53 | 65.6 |
| poly_high_degree (14 systems) | 29 | 29 | 29 | 29 | 36 | 29 | 29.8 |
| non_polynomial (26 systems) | 35 | 27 | 35 | 42 | 42 | 38 | 36.5 |

## All reps, fit on IC 0 (1008 runs)

**Reconstruction** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 40 | 35 | 35 | 34 | 33 | 33 | 35.0 |
| D >= 2 (40 systems) | 62 | 55 | 56 | 53 | 52 | 52 | 55.1 |
| in_library (23 systems) | 52 | 45 | 42 | 39 | 33 | 35 | 41.1 |
| in_library, D >= 2 (15 systems) | 80 | 69 | 64 | 60 | 51 | 53 | 63.0 |
| poly_high_degree (14 systems) | 29 | 29 | 29 | 26 | 33 | 26 | 28.6 |
| non_polynomial (26 systems) | 35 | 29 | 33 | 33 | 33 | 35 | 33.1 |

**Generalization** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 25 | 23 | 24 | 22 | 20 | 22 | 22.8 |
| D >= 2 (40 systems) | 40 | 37 | 38 | 35 | 31 | 34 | 35.8 |
| in_library (23 systems) | 52 | 38 | 41 | 35 | 32 | 32 | 38.2 |
| in_library, D >= 2 (15 systems) | 80 | 58 | 62 | 53 | 49 | 49 | 58.5 |
| poly_high_degree (14 systems) | 14 | 21 | 21 | 21 | 21 | 19 | 19.8 |
| non_polynomial (26 systems) | 8 | 12 | 12 | 12 | 8 | 14 | 10.7 |

**Structure** (exact support recovery, in-library systems)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_library (23) | 17 | 28 | 22 | 28 | 20 | 19 | 22.2 |
| D = 1 (8) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| D = 2 (6) | 0 | 56 | 33 | 44 | 33 | 33 | 33.3 |
| D = 3 (8) | 38 | 33 | 33 | 46 | 29 | 25 | 34.0 |
| D = 4 (1) | 100 | 33 | 33 | 0 | 33 | 33 | 38.9 |

median coefficient error on exact runs: 0.20 %

## Per system (fit IC 0, rep 0)

| id | D | class | exact σ=0/.01/.03/.05 | R²rec σ=0 | R²rec σ=.05 | R²gen σ=0 | R²gen σ=.05 | found (σ=0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 2 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 3 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 4 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 5 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 6 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 7 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 8 | 1 | poly_high_degree | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 9 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 10 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 11 | 1 | poly_high_degree | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 12 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 13 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 14 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 15 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 16 | 1 | poly_high_degree | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 17 | 1 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 18 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 19 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 20 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 21 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 22 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 23 | 1 | non_polynomial | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | IndexError: arrays used as indices must be of integer (or boolean) typ |
| 24 | 2 | in_library | 0/1/1/1 | 1.00 | 1.00 | 1.00 | 1.00 | dx0/dt = -5.475e-07 -3.346e-06*x0 +1*x1; dx1/dt = -2.1*x0 +1.427e-05*x0 |
| 25 | 2 | in_library | 0/1/1/0 | 1.00 | 1.00 | 1.00 | 1.00 | dx0/dt = +3.86e-07 -4.167e-05*x0 +1*x1 -0.0003193*x0^2 -0.0001725*x0*x |
| 26 | 2 | in_library | 0/1/0/0 | 1.00 | 0.00 | 1.00 | -2.61 | dx0/dt = +0.002501 +2.996*x0 -0.9967*x0^2 -2*x0*x1 -0.0006275*x1^2; dx1 |
| 27 | 2 | in_library | 0/0/1/0 | 1.00 | 1.00 | 1.00 | 1.00 | dx0/dt = +1.84*x0 -4.11e-05*x1 +4.835e-06*x0^2 -1.45*x0*x1 +1.29e-05*x |
| 28 | 2 | non_polynomial | -/-/-/- | 1.00 | 1.00 | 0.04 | 0.04 | dx0/dt = +1.006*x1; dx1/dt = -0.54*x0 |
| 29 | 2 | in_library | 0/0/1/0 | 1.00 | 0.00 | 1.00 | 0.00 | dx0/dt = +0.005526*x0 +0.003261*x1 -0.001634*x0^2 +0.6493*x0*x1 -7.945 |
| 30 | 2 | poly_high_degree | -/-/-/- | 1.00 | 0.00 | 0.00 | 0.00 | dx0/dt = -14.79*x1 +53.84*x0*x1 -29.93*x1^2; dx1/dt = -0.009195 -48.25* |
| 31 | 2 | in_library | 0/0/0/0 | 1.00 | 0.99 | 1.00 | 1.00 | dx0/dt = +0.0007466*x0 -8.359e-05*x0^2 -0.4001*x0*x1 -6.293e-07*x1^2; d |
| 32 | 2 | poly_high_degree | -/-/-/- | -1.13 | -44.42 | -0.22 | -0.12 | dx0/dt = +1.005*x1; dx1/dt = -0.9777*x0 |
| 33 | 2 | non_polynomial | -/-/-/- | 0.51 | 0.99 | -3.20 | 0.49 | dx0/dt = -0.1663*x0; dx1/dt = +1.038*x0 |
| 34 | 2 | non_polynomial | -/-/-/- | 1.00 | 0.99 | 0.08 | 0.08 | dx0/dt = +0.9945*x1; dx1/dt = -0.5696*x0 |
| 35 | 2 | non_polynomial | -/-/-/- | 0.93 | 0.94 | -36.04 | -35.99 | dx0/dt = -3.441 -2.189*x1; dx1/dt = +1.92*x0 |
| 36 | 2 | non_polynomial | -/-/-/- | 1.00 | 1.00 | 1.00 | 1.00 | dx0/dt = +1*x1; dx1/dt = -0.9929*x0 -1.074*x1 +0.116*x0^2 +0.04464*x0*x |
| 37 | 2 | poly_high_degree | -/-/-/- | 0.91 | 0.90 | 0.71 | 0.71 | dx0/dt = +1.004*x1; dx1/dt = -0.9698*x0 |
| 38 | 2 | poly_high_degree | -/-/-/- | 0.52 | 0.55 | 0.93 | 0.93 | dx0/dt = +1.453*x1; dx1/dt = -0.315*x0 |
| 39 | 2 | poly_high_degree | -/-/-/- | 1.00 | 1.00 | 1.00 | 0.99 | dx0/dt = -0.9987*x0 +2.406*x1 +0.04398*x0*x1 -0.04795*x1^2; dx1/dt = +0 |
| 40 | 2 | poly_high_degree | -/-/-/- | 0.48 | 0.50 | 0.28 | 0.28 | dx0/dt = +1.14*x1; dx1/dt = -0.8518*x0 |
| 41 | 2 | poly_high_degree | -/-/-/- | 0.00 | 0.00 | 0.00 | 0.00 | dx0/dt = -0.9523*x0 -0.4234*x1 -6.051*x0^2 +6.89*x0*x1; dx1/dt = +0.302 |
| 42 | 2 | non_polynomial | -/-/-/- | 0.89 | 0.79 | 0.51 | 0.71 | dx0/dt = +5.539 -1.189*x1; dx1/dt = -2.988 +2.645*x0 -0.2073*x0*x1 |
| 43 | 2 | non_polynomial | -/-/-/- | 0.52 | 1.00 | 0.77 | 0.95 | dx0/dt = +3.072 -7.773*x1 -0.006223*x0^2 -0.7288*x0*x1 +7.105*x1^2; dx1 |
| 44 | 2 | non_polynomial | -/-/-/- | 0.80 | 0.92 | 0.89 | 0.82 | dx0/dt = +1.003*x1; dx1/dt = +0.408 -0.0449*x0 |
| 45 | 2 | poly_high_degree | -/-/-/- | -0.30 | -0.60 | 0.00 | 0.00 | dx0/dt = +1.926*x0 +0.167*x1 -1.308*x0^2 -2.061*x0*x1 -0.03039*x1^2; dx |
| 46 | 2 | non_polynomial | -/-/-/- | 1.00 | 0.00 | 0.92 | -0.23 | dx0/dt = -0.6749*x0 -0.335*x1 +0.0844*x0^2 +0.2294*x0*x1 +0.1339*x1^2;  |
| 47 | 2 | non_polynomial | -/-/-/- | 1.00 | 0.95 | 0.00 | -9.22 | dx0/dt = +1.051 -3.052*x0 +2.275*x0^2 +0.5221*x0*x1 -1.995*x1^2; dx1/dt |
| 48 | 2 | non_polynomial | -/-/-/- | 0.00 | 0.00 | -0.84 | 0.96 | dx0/dt = +10.02 +0.9111*x0 -0.4611*x1 -0.1047*x0^2 +0.02495*x0*x1; dx1/ |
| 49 | 2 | poly_high_degree | -/-/-/- | 1.00 | 0.99 | 0.82 | 0.76 | dx0/dt = +1.812 -6.937*x0 -1.22*x1 +2.037*x0^2 +4.382*x0*x1; dx1/dt = - |
| 50 | 2 | poly_high_degree | -/-/-/- | 0.00 | 0.00 | 0.82 | 0.78 | dx0/dt = -2.777 +2.834*x1 +0.042*x0^2 +1.717*x0*x1 -0.8605*x1^2; dx1/dt |
| 51 | 2 | non_polynomial | -/-/-/- | 0.93 | 0.91 | 0.00 | 0.81 | dx0/dt = +0.8185*x1 -0.04555*x0*x1; dx1/dt = -0.9771 +0.458*x0 -0.02173 |
| 52 | 3 | in_library | 0/1/0/0 | 1.00 | -0.64 | 1.00 | -3.66 | dx/dt = -0.1*x +0.1*y; dy/dt = -0.21*y +0.21*x*z; dz/dt = +1.389 -0.34*z |
| 53 | 3 | non_polynomial | -/-/-/- | 1.00 | 0.99 | -7.87 | 0.00 | dx/dt = +0.04109 -0.8669*x +0.03461*y -1.723*x*y^2 +0.005139*z^3; dy/dt |
| 54 | 3 | in_library | 0/1/1/1 | 1.00 | 0.96 | 1.00 | 0.96 | dx/dt = -5.101*x +5.101*y +9.321e-06*x*y*z -1.148e-05*y^2*z; dy/dt = +1 |
| 55 | 3 | in_library | 1/0/1/1 | 0.10 | -0.44 | -0.39 | -0.42 | dx/dt = -9.998*x +10*y; dy/dt = +100*x -1.001*y -1*x*z; dz/dt = -2.666*z |
| 56 | 3 | in_library | 1/0/0/0 | 0.75 | -0.55 | 0.73 | -0.94 | dx/dt = -10*x +10*y; dy/dt = +28*x -0.9984*y -1*x*z; dz/dt = -2.667*z +1 |
| 57 | 3 | in_library | 0/0/0/0 | 1.00 | 1.00 | 1.00 | -0.06 | dx/dt = -8.757e-05 -5*y -4.998*z -1.671e-06*x^2; dy/dt = +5*x -1*y -2.6 |
| 58 | 3 | in_library | 1/0/1/1 | 1.00 | 1.00 | 1.00 | 1.00 | dx/dt = -5*y -5*z; dy/dt = +5*x +0.5*y; dz/dt = +0.9995 -28.5*z +5*x*z |
| 59 | 3 | in_library | 0/0/0/0 | 0.00 | 0.00 | 0.00 | 0.00 | dx/dt = -0.4462*y -0.1196*z^2 +0.03623*x*y*z -0.02102*y*z^2 +0.004178* |
| 60 | 3 | poly_high_degree | -/-/-/- | 0.71 | 1.00 | 0.67 | 0.85 | dx/dt = -0.9616*x -3.497*y +1.159*x*z -0.01362*y*z; dy/dt = +3.497*x -0 |
| 61 | 3 | in_library | 0/0/1/0 | 0.95 | -0.96 | 1.00 | 1.00 | dx/dt = +4.999*x -0.9988*y*z +9.237e-05*y*z^2; dy/dt = -9.999*y +1*x*z;  |
| 62 | 4 | non_polynomial | -/-/-/- | 1.00 | -2.31 | -2.43 | -51.88 | dx0/dt = +0.4948 -0.7671*x0 -0.04347*x0*x1 -0.07189*x2^2 +0.1056*x2*x3 |
| 63 | 4 | in_library | 1/0/0/1 | 1.00 | 0.99 | 1.00 | 1.00 | dx0/dt = -0.28*x0*x2; dx1/dt = -0.47*x1 +0.28*x0*x2; dx2/dt = +0.47*x1 - |

## Secondary: fit on IC 1, all reps (1008 runs)

**Reconstruction** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 48 | 44 | 38 | 35 | 35 | 35 | 39.2 |
| D >= 2 (40 systems) | 75 | 69 | 60 | 55 | 56 | 56 | 61.8 |
| in_library (23 systems) | 57 | 45 | 35 | 30 | 30 | 28 | 37.4 |
| in_library, D >= 2 (15 systems) | 87 | 69 | 53 | 47 | 47 | 42 | 57.4 |
| poly_high_degree (14 systems) | 43 | 43 | 43 | 43 | 40 | 48 | 43.3 |
| non_polynomial (26 systems) | 42 | 44 | 38 | 35 | 37 | 36 | 38.7 |

**Generalization (IC 0)** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (63 systems) | 35 | 22 | 20 | 20 | 19 | 16 | 22.0 |
| D >= 2 (40 systems) | 55 | 35 | 32 | 31 | 29 | 26 | 34.6 |
| in_library (23 systems) | 61 | 33 | 29 | 26 | 28 | 22 | 33.1 |
| in_library, D >= 2 (15 systems) | 93 | 51 | 44 | 40 | 42 | 33 | 50.7 |
| poly_high_degree (14 systems) | 21 | 21 | 21 | 21 | 19 | 19 | 20.6 |
| non_polynomial (26 systems) | 19 | 13 | 12 | 13 | 10 | 10 | 12.8 |

**Structure** (exact support recovery, in-library systems)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_library (23) | 17 | 25 | 23 | 20 | 20 | 22 | 21.3 |
| D = 1 (8) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| D = 2 (6) | 0 | 39 | 50 | 44 | 33 | 50 | 36.1 |
| D = 3 (8) | 50 | 42 | 29 | 25 | 33 | 25 | 34.0 |
| D = 4 (1) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

median coefficient error on exact runs: 0.17 %

**Structure, negligible terms ignored** (POST-HOC, not pre-registered: exact support ignoring terms contributing < 1% of the equation's RMS)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_library (23) | 57 | 32 | 32 | 23 | 26 | 22 | 31.9 |
| D = 2 (6) | 100 | 56 | 61 | 44 | 44 | 50 | 59.3 |
| D = 3 (8) | 88 | 50 | 46 | 33 | 42 | 25 | 47.2 |
| D = 4 (1) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

model exceptions: 370 of 1008 runs  {(1, 'IndexError'): 368, (3, 'IndexError'): 2}
integration failures among fitted runs: reconstruction 61, generalization 179 of 638
median fit time: 0.7 s; by D: D=1: 0.0 s, D=2: 1.0 s, D=3: 15.2 s, D=4: 7.4 s

## DIAGNOSTIC (not frozen): empty-support crash patched, D = 1, 2, fit IC 0, all reps (816 runs)

D = 1: 368 runs

**Reconstruction D = 1** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (23 systems) | 91 | 88 | 88 | 86 | 81 | 81 | 86.0 |
| in_library (8 systems) | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| poly_high_degree (3 systems) | 33 | 33 | 33 | 33 | 33 | 33 | 33.3 |
| non_polynomial (12 systems) | 100 | 94 | 94 | 89 | 81 | 81 | 89.8 |

**Generalization D = 1** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (23 systems) | 78 | 75 | 71 | 71 | 68 | 67 | 71.7 |
| in_library (8 systems) | 100 | 96 | 79 | 79 | 79 | 75 | 84.7 |
| poly_high_degree (3 systems) | 33 | 67 | 67 | 67 | 67 | 67 | 61.1 |
| non_polynomial (12 systems) | 75 | 64 | 67 | 67 | 61 | 61 | 65.7 |
  exact σ=0.0: 75% of 8
  exact σ=0.01: 83% of 24
  exact σ=0.02: 75% of 24
  exact σ=0.03: 75% of 24
  exact σ=0.04: 75% of 24
  exact σ=0.05: 75% of 24

D = 2: 448 runs

**Reconstruction D = 2** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (28 systems) | 61 | 55 | 54 | 51 | 52 | 52 | 54.2 |
| D >= 2 (28 systems) | 61 | 55 | 54 | 51 | 52 | 52 | 54.2 |
| in_library (6 systems) | 100 | 100 | 83 | 72 | 61 | 67 | 80.6 |
| in_library, D >= 2 (6 systems) | 100 | 100 | 83 | 72 | 61 | 67 | 80.6 |
| poly_high_degree (10 systems) | 40 | 30 | 30 | 27 | 37 | 27 | 31.7 |
| non_polynomial (12 systems) | 58 | 53 | 58 | 61 | 61 | 67 | 59.7 |

**Generalization D = 2** (% of runs with R² > 0.9)

| Subset | σ=0.0 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.04 | σ=0.05 | mean over σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (28 systems) | 36 | 36 | 37 | 32 | 29 | 33 | 33.7 |
| D >= 2 (28 systems) | 36 | 36 | 37 | 32 | 29 | 33 | 33.7 |
| in_library (6 systems) | 100 | 83 | 89 | 67 | 67 | 61 | 77.8 |
| in_library, D >= 2 (6 systems) | 100 | 83 | 89 | 67 | 67 | 61 | 77.8 |
| poly_high_degree (10 systems) | 20 | 20 | 20 | 20 | 20 | 20 | 20.0 |
| non_polynomial (12 systems) | 17 | 25 | 25 | 25 | 17 | 31 | 23.1 |
  exact σ=0.0: 0% of 6
  exact σ=0.01: 56% of 18
  exact σ=0.02: 33% of 18
  exact σ=0.03: 44% of 18
  exact σ=0.04: 33% of 18
  exact σ=0.05: 33% of 18

model exceptions: 0 of 816 runs  {}
integration failures among fitted runs: reconstruction 87, generalization 73 of 816
median fit time: 0.6 s; by D: D=1: 0.1 s, D=2: 0.9 s, D=3: nan s, D=4: nan s
