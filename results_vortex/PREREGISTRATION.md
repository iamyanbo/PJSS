# Vortex shedding pilot: pre-registration

Written 2026-10-30, before any eqdisc fit to these data. Runner: `benchmarks/vortex_pilot.py`. Model: current method,
`eqdisc.discover(y, 0.1, noise="auto", confidence=False)`, defaults, with `eqdisc/` unmodified.

## Question

From noisy reduced-order coefficients of a simulated cylinder wake, does the method recover the mean-field equations
without tuning, and does it do so more often than the fast truth-free baselines?

## Data

`examples/data/vonKarman_pod/vonKarman_a.dat` from github.com/dynamicslab/pysindy (master, fetched 2026-10-30), sha256
`8375a26377239ca8f64e3e9209e8e355e9379a8b13311f761f85e703e17a83bc`. Direct numerical simulation (Nek5000, Re=100,
`cyl.par` sha256 `09436faac26b4ac7342015c5dce9f0760b1989ea61539b57cd60460e8ab3a2df`) from near the unstable steady state
to the limit cycle. 3000 samples, dt 0.1, t = 0 to 299.9. Coefficients are stored to 6 decimals.

States: x, y = columns 1, 2 (shedding modes), z = column 9 (shift mode). Window: 30 <= t < 130, 1000 samples. It
contains the whole transient and about 30 time units of the limit cycle. This is simulated flow, not a measurement.

## Done before this file was written (disclosed)

- Amplitudes were inspected. Column 3 was first assumed to be the shift mode and was not: column 9 is (correlation 0.998
  with x^2 + y^2).
- The window and the target support below were chosen after that inspection, and after the least-squares fit below.
- `galerkin3.mat` was loaded and printed; it is not used.
- No sparse method, baseline or eqdisc, was run on these data.

## Target support

From the mean-field model with a frequency shift (Noack et al. 2003; Brunton et al. 2016), 11 terms:

    dx: x, y, xz, yz
    dy: x, y, xz, yz
    dz: z, x^2, y^2

Reference coefficients: least squares of central differences of the clean window on this support (`reference`).
R^2 on the derivatives: 0.9997, 0.9999, 0.5927. The dz equation is therefore only approximate: z is almost flat on the
limit cycle while x^2 + y^2 ripples. Integrated from the window start, the reference gives z 2.171, r^2 5.167,
omega 1.107 on the limit cycle, against 2.245, 5.309, 1.125 in the data.

## Runs

- **Real:** the window with white noise of eta = 1, 5, 10, 20% of each state's sd, seeds 1 to 10 per level (40 runs),
  plus one run with no added noise.
- **Control C-MF:** the reference model simulated over the same window with the same start, eta = 5%, seeds 101 to 120.
  The target support is true here and the data are otherwise alike, so this measures power on this problem.

All methods use the 20-term degree-3 library and pick their sparsity without the truth: eqdisc; the weak chi2s selector;
WSINDy with MSTLS; pysindy STLSQ on Savitzky-Golay differences and on weak features, each with CV and AICc, with the
configurations used in the paper. DEDi is not in this pilot because of its cost.

## Criteria

- **Exact:** the support equals the target.
- **Core:** all 11 target terms are present; extra terms are allowed.
- **Dynamics (secondary, real runs):** the model is integrated from the clean window start for 270 time units, and the
  last 100 give z, r^2 and omega. Good if none diverges and all three are within 10% of the data.

## Verdict

1. **Control gate:** eqdisc exact on at least 16 of 20 C-MF runs. Otherwise the real result is **uninformative**.
2. **Pass:** gate passed and eqdisc core on at least 8 of 10 real runs at eta = 5%.
3. **Fail:** gate passed and fewer than 8.
4. **Comparison (secondary):** paired exact sign test over the 40 noisy real runs, on core and on exact, against the
   baseline pick with the highest core count. That pick is chosen after the fact, and the table reports every method.

No settings change after this file is committed. Any later run is labelled exploratory.

## Amendments (no setting or criterion changed)

- 2026-10-30: after the control and before any result of the real runs was seen, the divergence check in `attractor` was
  made terminal: it was not stopping integrations, so diverging models ran very slowly. The real runs moved from this PC
  to the Spark (pysindy 2.1.0, numpy 2.5.4, scipy 1.18.1). The two interrupted attempts wrote no results. The control ran
  on this PC (numpy 2.4.4, scipy 1.17.1) with the earlier check, which does not affect its criteria.
