
## dysts21: 294 runs. W/L = runs only ours / only the baseline recovers exactly; sign test on discordant pairs
| method / rule | runs | exact | 1% | 5% | 10% | 20% | D=3 | D>=4 | median s | ours same runs | ours W/L | sign p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-SINDy (SFD) [raw_w11] CV | 294 | 0.293 | 0.486 | 0.338 | 0.233 | 0.110 | 0.329 | 0.150 | 13.8 | 0.874 | 171/0 | 7e-52 |
| E-SINDy (SFD) [raw_w11] AICc | 294 | 0.231 | 0.365 | 0.311 | 0.137 | 0.110 | 0.248 | 0.167 | 13.8 | 0.874 | 189/0 | 3e-57 |
| E-SINDy (SFD) [raw_w11] tuned | 294 | 0.374 | 0.581 | 0.405 | 0.274 | 0.233 | 0.415 | 0.217 | 13.8 | 0.874 | 147/0 | 1e-44 |
| E-SINDy (SFD) [raw_w11] oracle | 294 | 0.483 | 0.743 | 0.527 | 0.370 | 0.288 | 0.530 | 0.300 | 13.8 | 0.874 | 116/1 | 1e-33 |
| Weak E-SINDy [h10] CV | 294 | 0.163 | 0.243 | 0.203 | 0.137 | 0.068 | 0.188 | 0.067 | 7.2 | 0.874 | 209/0 | 2e-63 |
| Weak E-SINDy [h10] AICc | 294 | 0.058 | 0.176 | 0.014 | 0.041 | 0.000 | 0.073 | 0.000 | 7.2 | 0.874 | 240/0 | 1e-72 |
| Weak E-SINDy [raw_h20] tuned | 294 | 0.272 | 0.324 | 0.297 | 0.247 | 0.219 | 0.325 | 0.067 | 8.4 | 0.874 | 178/1 | 5e-52 |
| Weak E-SINDy [raw_h20] oracle | 294 | 0.398 | 0.446 | 0.405 | 0.384 | 0.356 | 0.444 | 0.217 | 8.4 | 0.874 | 141/1 | 5e-41 |
| ODEFormer [default] default | 294 | 0.007 | 0.014 | 0.000 | 0.014 | 0.000 | 0.009 | 0.000 | 63.2 | 0.874 | 255/0 | 3e-77 |
| ODEFormer [default] oracle | 294 | 0.020 | 0.014 | 0.041 | 0.027 | 0.000 | 0.026 | 0.000 | 63.2 | 0.874 | 251/0 | 6e-76 |
| eqdisc.discover (final, frozen) | 294 | 0.874 | 0.946 | 0.959 | 0.849 | 0.740 | 0.880 | 0.850 | 3.8 | 0.874 | 0/0 | 1e+00 |

## dysts49: 126 runs. W/L = runs only ours / only the baseline recovers exactly; sign test on discordant pairs
| method / rule | runs | exact | 1% | 5% | 10% | 20% | D=3 | D>=4 | median s | ours same runs | ours W/L | sign p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PySR [default] default | 126 | 0.444 | 0.656 | 0.531 | 0.355 | 0.226 | 0.485 | 0.296 | 180.4 | 0.929 | 61/0 | 9e-19 |
| PySR [default] oracle | 126 | 0.778 | 0.906 | 0.906 | 0.710 | 0.581 | 0.788 | 0.741 | 180.4 | 0.929 | 20/1 | 2e-05 |
| eqdisc.discover (final, frozen) | 126 | 0.929 | 1.000 | 0.938 | 0.935 | 0.839 | 0.929 | 0.926 | 3.9 | 0.929 | 0/0 | 1e+00 |

## dysts7: 880 runs. W/L = runs only ours / only the baseline recovers exactly; sign test on discordant pairs
| method / rule | runs | exact | 1% | 5% | 10% | 20% | D=3 | D>=4 | median s | ours same runs | ours W/L | sign p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SINDy STLSQ (SFD) [raw_w11] CV | 880 | 0.306 | 0.473 | 0.364 | 0.236 | 0.150 | 0.334 | 0.197 | 1.0 | 0.891 | 515/0 | 2e-155 |
| SINDy STLSQ (SFD) [raw_w5] AICc | 880 | 0.301 | 0.595 | 0.341 | 0.186 | 0.082 | 0.321 | 0.224 | 1.1 | 0.891 | 520/1 | 2e-154 |
| SINDy STLSQ (SFD) [raw_w11] tuned | 880 | 0.372 | 0.618 | 0.405 | 0.277 | 0.186 | 0.407 | 0.235 | 1.0 | 0.891 | 459/2 | 4e-134 |
| SINDy STLSQ (SFD) [raw_w11] oracle | 880 | 0.491 | 0.736 | 0.550 | 0.400 | 0.277 | 0.531 | 0.339 | 1.0 | 0.891 | 355/3 | 3e-101 |
| SINDy SR3-L0 (SFD) [w11] CV | 880 | 0.174 | 0.414 | 0.155 | 0.077 | 0.050 | 0.168 | 0.197 | 0.5 | 0.891 | 631/0 | 2e-190 |
| SINDy SR3-L0 (SFD) [w5] AICc | 880 | 0.193 | 0.541 | 0.141 | 0.059 | 0.032 | 0.198 | 0.175 | 0.5 | 0.891 | 614/0 | 3e-185 |
| SINDy SR3-L0 (SFD) [w21] tuned | 880 | 0.160 | 0.314 | 0.177 | 0.100 | 0.050 | 0.181 | 0.082 | 0.5 | 0.891 | 643/0 | 5e-194 |
| SINDy SR3-L0 (SFD) [w21] oracle | 880 | 0.268 | 0.523 | 0.305 | 0.159 | 0.086 | 0.291 | 0.180 | 0.5 | 0.891 | 549/1 | 3e-163 |
| Weak SINDy STLSQ [h10] CV | 880 | 0.188 | 0.245 | 0.195 | 0.177 | 0.132 | 0.218 | 0.071 | 0.5 | 0.891 | 621/2 | 1e-182 |
| Weak SINDy STLSQ [h10] AICc | 880 | 0.057 | 0.141 | 0.050 | 0.027 | 0.009 | 0.070 | 0.005 | 0.5 | 0.891 | 734/0 | 2e-221 |
| Weak SINDy STLSQ [raw_h20] tuned | 880 | 0.278 | 0.336 | 0.291 | 0.255 | 0.232 | 0.329 | 0.087 | 0.5 | 0.891 | 541/2 | 1e-158 |
| Weak SINDy STLSQ [raw_h20] oracle | 880 | 0.370 | 0.464 | 0.382 | 0.341 | 0.295 | 0.425 | 0.164 | 0.5 | 0.891 | 462/4 | 2e-131 |
| Weak SINDy SR3-L0 [h10] CV | 880 | 0.174 | 0.409 | 0.182 | 0.082 | 0.023 | 0.179 | 0.153 | 0.7 | 0.891 | 632/1 | 4e-188 |
| Weak SINDy SR3-L0 [h10] AICc | 880 | 0.058 | 0.118 | 0.077 | 0.032 | 0.005 | 0.059 | 0.055 | 0.7 | 0.891 | 733/0 | 4e-221 |
| Weak SINDy SR3-L0 [h20] tuned | 880 | 0.219 | 0.445 | 0.259 | 0.136 | 0.036 | 0.222 | 0.208 | 0.6 | 0.891 | 592/1 | 4e-176 |
| Weak SINDy SR3-L0 [h20] oracle | 880 | 0.347 | 0.645 | 0.373 | 0.223 | 0.145 | 0.360 | 0.295 | 0.6 | 0.891 | 480/1 | 2e-142 |
| Weak SINDy MIOSR [h10] CV | 880 | 0.073 | 0.100 | 0.064 | 0.068 | 0.059 | 0.082 | 0.038 | 4.0 | 0.891 | 720/0 | 4e-217 |
| Weak SINDy MIOSR [h10] AICc | 880 | 0.020 | 0.082 | 0.000 | 0.000 | 0.000 | 0.023 | 0.011 | 4.0 | 0.891 | 766/0 | 5e-231 |
| Weak SINDy MIOSR [h10] oracle | 880 | 0.759 | 0.923 | 0.841 | 0.714 | 0.559 | 0.782 | 0.672 | 4.0 | 0.891 | 123/7 | 2e-28 |
| eqdisc.discover (final, frozen) | 880 | 0.891 | 0.964 | 0.936 | 0.886 | 0.777 | 0.897 | 0.869 | 3.8 | 0.891 | 0/0 | 1e+00 |

## hard40: 40 runs. W/L = runs only ours / only the baseline recovers exactly; sign test on discordant pairs
| method / rule | runs | exact | 1% | 5% | 10% | 20% | D=3 | D>=4 | median s | ours same runs | ours W/L | sign p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SINDy STLSQ (SFD) [raw_w11] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 2.4 | 0.700 | 28/0 | 7e-09 |
| SINDy STLSQ (SFD) [raw_w5] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.3 | 0.700 | 28/0 | 7e-09 |
| SINDy STLSQ (SFD) [raw_w11] tuned | 40 | 0.025 | - | - | - | 0.025 | 0.025 | - | 2.4 | 0.700 | 27/0 | 1e-08 |
| SINDy STLSQ (SFD) [raw_w11] oracle | 40 | 0.025 | - | - | - | 0.025 | 0.025 | - | 2.4 | 0.700 | 27/0 | 1e-08 |
| SINDy SR3-L0 (SFD) [w11] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.3 | 0.700 | 28/0 | 7e-09 |
| SINDy SR3-L0 (SFD) [w5] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.2 | 0.700 | 28/0 | 7e-09 |
| SINDy SR3-L0 (SFD) [w21] tuned | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.0 | 0.700 | 28/0 | 7e-09 |
| SINDy SR3-L0 (SFD) [w21] oracle | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.0 | 0.700 | 28/0 | 7e-09 |
| E-SINDy (SFD) [raw_w11] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 52.7 | 0.700 | 28/0 | 7e-09 |
| E-SINDy (SFD) [raw_w11] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 52.7 | 0.700 | 28/0 | 7e-09 |
| E-SINDy (SFD) [raw_w11] tuned | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 52.7 | 0.700 | 28/0 | 7e-09 |
| E-SINDy (SFD) [raw_w11] oracle | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 52.7 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy STLSQ [h10] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.4 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy STLSQ [h10] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.4 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy STLSQ [raw_h20] tuned | 40 | 0.100 | - | - | - | 0.100 | 0.100 | - | 1.7 | 0.700 | 24/0 | 1e-07 |
| Weak SINDy STLSQ [raw_h20] oracle | 40 | 0.200 | - | - | - | 0.200 | 0.200 | - | 1.7 | 0.700 | 20/0 | 2e-06 |
| Weak SINDy SR3-L0 [h10] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.1 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy SR3-L0 [h10] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 1.1 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy SR3-L0 [h20] tuned | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 0.7 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy SR3-L0 [h20] oracle | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 0.7 | 0.700 | 28/0 | 7e-09 |
| Weak E-SINDy [h10] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 24.3 | 0.700 | 28/0 | 7e-09 |
| Weak E-SINDy [h10] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 24.3 | 0.700 | 28/0 | 7e-09 |
| Weak E-SINDy [raw_h20] tuned | 40 | 0.075 | - | - | - | 0.075 | 0.075 | - | 31.0 | 0.700 | 25/0 | 6e-08 |
| Weak E-SINDy [raw_h20] oracle | 40 | 0.150 | - | - | - | 0.150 | 0.150 | - | 31.0 | 0.700 | 22/0 | 5e-07 |
| Weak SINDy MIOSR [h10] CV | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 5.9 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy MIOSR [h10] AICc | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 5.9 | 0.700 | 28/0 | 7e-09 |
| Weak SINDy MIOSR [h10] oracle | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 5.9 | 0.700 | 28/0 | 7e-09 |
| PySR [default] default | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 180.9 | 0.700 | 28/0 | 7e-09 |
| PySR [default] oracle | 40 | 0.275 | - | - | - | 0.275 | 0.275 | - | 180.9 | 0.700 | 17/0 | 2e-05 |
| ODEFormer [default] default | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 61.6 | 0.700 | 28/0 | 7e-09 |
| ODEFormer [default] oracle | 40 | 0.000 | - | - | - | 0.000 | 0.000 | - | 61.6 | 0.700 | 28/0 | 7e-09 |
| eqdisc.discover (final, frozen) | 40 | 0.700 | - | - | - | 0.700 | 0.700 | - | 6.3 | 0.700 | 0/0 | 1e+00 |

hard 40 per system (exact / 10):
  SINDy STLSQ (SFD) [raw_w11] CV                   lorenz  0  chen  0  rossler  0  rabinovich  0
  SINDy STLSQ (SFD) [raw_w5] AICc                  lorenz  0  chen  0  rossler  0  rabinovich  0
  SINDy STLSQ (SFD) [raw_w11] tuned                lorenz  1  chen  0  rossler  0  rabinovich  0
  SINDy STLSQ (SFD) [raw_w11] oracle               lorenz  1  chen  0  rossler  0  rabinovich  0
  SINDy SR3-L0 (SFD) [w11] CV                      lorenz  0  chen  0  rossler  0  rabinovich  0
  SINDy SR3-L0 (SFD) [w5] AICc                     lorenz  0  chen  0  rossler  0  rabinovich  0
  SINDy SR3-L0 (SFD) [w21] tuned                   lorenz  0  chen  0  rossler  0  rabinovich  0
  SINDy SR3-L0 (SFD) [w21] oracle                  lorenz  0  chen  0  rossler  0  rabinovich  0
  E-SINDy (SFD) [raw_w11] CV                       lorenz  0  chen  0  rossler  0  rabinovich  0
  E-SINDy (SFD) [raw_w11] AICc                     lorenz  0  chen  0  rossler  0  rabinovich  0
  E-SINDy (SFD) [raw_w11] tuned                    lorenz  0  chen  0  rossler  0  rabinovich  0
  E-SINDy (SFD) [raw_w11] oracle                   lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy STLSQ [h10] CV                        lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy STLSQ [h10] AICc                      lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy STLSQ [raw_h20] tuned                 lorenz  0  chen  4  rossler  0  rabinovich  0
  Weak SINDy STLSQ [raw_h20] oracle                lorenz  0  chen  4  rossler  4  rabinovich  0
  Weak SINDy SR3-L0 [h10] CV                       lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy SR3-L0 [h10] AICc                     lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy SR3-L0 [h20] tuned                    lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy SR3-L0 [h20] oracle                   lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak E-SINDy [h10] CV                            lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak E-SINDy [h10] AICc                          lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak E-SINDy [raw_h20] tuned                     lorenz  0  chen  3  rossler  0  rabinovich  0
  Weak E-SINDy [raw_h20] oracle                    lorenz  0  chen  4  rossler  2  rabinovich  0
  Weak SINDy MIOSR [h10] CV                        lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy MIOSR [h10] AICc                      lorenz  0  chen  0  rossler  0  rabinovich  0
  Weak SINDy MIOSR [h10] oracle                    lorenz  0  chen  0  rossler  0  rabinovich  0
  PySR [default] default                           lorenz  0  chen  0  rossler  0  rabinovich  0
  PySR [default] oracle                            lorenz  0  chen  6  rossler  5  rabinovich  0
  ODEFormer [default] default                      lorenz  0  chen  0  rossler  0  rabinovich  0
  ODEFormer [default] oracle                       lorenz  0  chen  0  rossler  0  rabinovich  0
  eqdisc.discover (final)                          lorenz 10  chen 10  rossler  8  rabinovich  0
