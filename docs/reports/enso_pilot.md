# El Nino pilot (2026-10-30)

**Bottom line.** eqdisc did not find the recharge oscillator, and under the pre-registered rule the real-data result is
**uninformative** because the control gate failed. It recovered the core on 3/20 simulated recharge oscillators against
the required 16/20. The failure comes from eqdisc's noise model, not from the data: plain BIC regression got 20/20 on the
same simulations, and eqdisc got 10/10 exact on a deterministic system of the same length and sampling. eqdisc assumes
deterministic dynamics plus measurement noise. El Nino is driven by random forcing inside the dynamics.

Protocol: `results_enso/PREREGISTRATION.md`, committed in `cba8a5e` before any eqdisc fit to real indices. Runner:
`benchmarks/enso_pilot.py`. Results: `results_enso/control.json`, `results_enso/real.json`. `eqdisc/` was not modified.

## Controls (seeds not used in development)

| Control | Measure | Result |
| --- | --- | --- |
| Simulated recharge oscillator, eqdisc | core recovered | 3/20 (gate 16/20) |
| Simulated recharge oscillator, eqdisc | clock equations recovered | 0/20 |
| Simulated recharge oscillator, BIC regression | core recovered | 20/20 |
| Lotka-Volterra, deterministic, 10% measurement noise, eqdisc | exact support | 10/10 (coefficient error at most 4.5%) |

On the simulations, BIC regression's median F1 = 0.155 and F2 = 13.5, against true annual means of 0.164 and 14.07. The
core criterion allows extra terms, so 20/20 is about the two links, not exact recovery. eqdisc left the T equation with at
most one term on 13 of the 20 runs.

## Real data, trained on 1979-2011

- eqdisc E4 missed the core on all 5 jitter seeds, and the returned equations changed from seed to seed. Coefficients
  reached the thousands to about 158,000, on terms with no physical reading. All 5 lost the clock equations.
- eqdisc E2 returned an empty T equation, so its forecast is exactly persistence.
- BIC regression found the core with the expected signs: F1 = 0.187 and F2 = 12.4, against XRO-RO's annual means of
  0.164 and 14.07. Its equations also agree with XRO-RO on R (-0.84 against -0.74).

## Forecasts: 138 initial months, 2012-01 to 2023-06

| Model | Mean ACC, leads 1-18 | Lead 6 | Lead 12 | Lead 18 | RMSE lead 12 (deg C) | Gap to XRO-RO, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| XRO full (10 variables) | 0.680 | 0.821 | 0.601 | 0.340 | 0.76 | -0.002 to 0.205 |
| BIC regression (E4) | 0.659 | 0.789 | 0.527 | 0.407 | 0.76 | -0.002 to 0.110 |
| XRO-RO (2 variables) | 0.593 | 0.725 | 0.423 | 0.372 | 0.82 | reference |
| Persistence / eqdisc E2 | 0.272 | 0.406 | 0.037 | -0.081 | 1.20 | -0.500 to 0.021 |
| eqdisc E4, seeds 3 and 4 | 0.121 | 0.193 | 0.036 | -0.029 | 0.96 | -0.582 to -0.228 |
| eqdisc E4, seeds 0 to 2 | none | - | - | - | - | - |

eqdisc E4 diverged on 138/138 forecasts for seeds 0 to 2 and 115/138 for seeds 3 and 4. Several of those arms are
undefined at some or all leads, because a forecast that has diverged is held at 0. With the exact clock, all 5 seeds
diverged on all 138 forecasts.

## How far this goes

- BIC regression was a comparator, not a pre-registered claim. Its lead over XRO-RO is a single test period with 138
  overlapping starts, and the CI touches zero. It is a lead worth checking, not a finding.
- The anomalies were computed over the whole record, a small leak shared by every method.
- Both simulated controls are clean compared with reality, and the recharge-oscillator simulation is linear.
- The result does not show that sparse discovery fails for El Nino. It shows that this method, as frozen, should not be
  applied to systems driven by strong random forcing.
