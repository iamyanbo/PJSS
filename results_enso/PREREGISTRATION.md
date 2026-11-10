# El Nino pilot: pre-registration

Written 2026-10-30, before any eqdisc fit to real El Nino data and before any model touched the held-out months.
Runner: `benchmarks/enso_pilot.py`. Model: frozen v1.2, `eqdisc.discover(z, 1/12, confidence=False)`, defaults, with
`eqdisc/` unmodified.

## Question

Does the frozen model recover the recharge oscillator (Jin 1997) from monthly indices, and do the recovered
equations forecast El Nino out of sample as well as a fitted recharge oscillator?

## Data

`data/XRO_indices_oras5.nc` from github.com/senclimate/XRO (main, fetched 2026-10-30), sha256
`3d0fc489a3286aea6c2c411a800bf3f684939c1cc455cddf0b65b652a7a589e9`; read with xarray; 552 months, 1979-01 to 2024-12,
no gaps. T = Nino34 (deg C), h = WWV (m). The file holds anomalies, so its seasonal mean and any detrending used the
whole record: a small leak of held-out years into the training data, shared by every method.

Training: 1979-01 to 2011-12 (396 months). Held out: initial months 2012-01 to 2023-06 (138), leads 1 to 18.

## Done before this file was written (disclosed)

- XRO-RO was fitted to the training months to set up the simulations. Annual-mean operator: R = -0.743 per year,
  F1 = 0.164, F2 = 14.07, epsilon = 0.125.
- The forecast code was smoke-tested on a simulated record only. No forecast was made for a held-out month.
- Development on simulations from that fit, seeds 1 to 6:
  - One run with the exact clock returned clock equations wrong by about 100 times, because the noise estimate fails on
    states with no noise. So white noise of sd JITTER is added to c and s.
  - JITTER 0.02 and 0.1 were tried at forcing x1 and x0.25 (24 runs). Core recovered: 4/24, split 1/12 and 3/12.
    JITTER = 0.1 is fixed from that. The clock equations were mostly dropped entirely.
  - One run of E2 returned no recharge structure.
  - Forward-difference regression recovered the core on the one seed tried.
- These development results predict low power. That is why the control below is part of the verdict.

## Arms

- **E4 (primary):** (T, h, c, s), degree-2 library, JITTER 0.1, jitter seed 0. Seeds 1 to 4 are reported as stability.
- **E2:** (T, h), degree-2 library.
- **E4 with exact clock (secondary, forecasts only):** the returned clock equations are replaced by the exact ones.
- **BIC regression:** exhaustive best-subset regression of forward differences on the E4 library, up to 5 terms, BIC
  pick, exact clock.
- **XRO-RO:** two-variable linear seasonal fit. Reference for forecasts.
- **XRO full:** 10 variables, `maskb=["IOD"], maskNT=["T2","TH"]`, the published control configuration, fitted on
  training months only.
- **Persistence.**

## Controls (seeds 101 to 120, not used in development)

- **C-RO:** 33 years simulated from the training XRO-RO with its fitted red forcing. E4 and BIC regression on each.
  The recharge oscillator is true here and the record length matches, so this measures power on this problem.
- **C-LV:** a deterministic Lotka-Volterra cycle of 3.9 years, 396 months, 10% white measurement noise, E2, 10 seeds.
  The method's assumptions hold here, at the same length and sampling.

## Criteria

- **Core:** the dT equation contains +h and the dh equation contains -T. Seasonal modulation and extra terms are allowed.
- **Clock:** dc = -2 pi s and ds = 2 pi c within 10%, nothing else.
- **Forecast competitive:** mean ACC over leads 1 to 18 at least XRO-RO minus 0.05, and above persistence. The 95% CI
  of the gap to XRO-RO comes from a moving-block bootstrap (24 months, 2000 resamples). A forecast with |T| > 10 stops
  and reverts to 0; divergences are counted.

## Verdict

1. **Control gate:** eqdisc must get core on at least 16/20 C-RO runs. Otherwise the real-data result is
   **uninformative**: a pass could be chance and a failure says nothing about El Nino.
2. **Pass:** gate passed, core on E4 seed 0, and forecast competitive.
3. **Fail:** gate passed and either of the two missing.
4. If BIC regression reaches the gate and eqdisc does not, while C-LV succeeds, the conclusion is that the limit comes
   from eqdisc's noise model (no forcing in the dynamics), not from the data. The conclusion is reversed if BIC
   regression also misses the gate.

No settings change after this file is committed. Any later run is labelled exploratory.
