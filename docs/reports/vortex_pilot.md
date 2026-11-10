# Vortex shedding pilot

Protocol: `results_vortex/PREREGISTRATION.md`. Runner: `benchmarks/vortex_pilot.py`. Records: `results_vortex/control.json`,
`results_vortex/real.json`. Summary counts below were computed from those files.

## Verdict: fail

- **Control gate: passed.** eqdisc exact on 19 of 20 runs of the simulated mean-field model at 5% noise (core 20).
  Weak chi2s 14 exact; every pysindy pick at most 1; WSINDy 0.
- **Primary: failed.** eqdisc core on 0 of 10 real runs at 5% noise (needed 8). It is 0 of 40 over all noise levels and 0
  on the run with no added noise.
- **Comparison:** against the baseline pick with the highest core count (STLSQ on differences, AICc, 17 of 40), 0/17 on
  core ($p=1.5\times10^{-5}$). On exact, no method recovers the target on any run.

Core favours dense answers: STLSQ with AICc returns a median of about 20 to 28 terms and weak STLSQ about 30 to 37, of 60
possible, against 8 to 13 for eqdisc and 11 targets.

| Method | Core, of 40 | Exact, of 40 | Good dynamics at 1 / 5 / 10 / 20% (of 10 each) |
|---|---|---|---|
| eqdisc | 0 | 0 | 6 / 0 / 2 / 1 |
| Weak chi2s | 0 | 0 | 3 / 8 / 1 / 0 |
| WSINDy | 0 | 0 | 0 / 1 / 0 / 0 |
| STLSQ differences, AICc | 17 | 0 | 4 / 2 / 1 / 0 |
| STLSQ differences, CV | 9 | 0 | 6 / 2 / 4 / 0 |
| Weak STLSQ, AICc | 11 | 0 | 8 / 7 / 5 / 1 |
| Weak STLSQ, CV | 15 | 0 | 8 / 6 / 3 / 0 |

Good dynamics: no divergence, and z, r^2 and omega within 10% of the data on the limit cycle.

eqdisc diverged on 2 of 10 runs at 10% noise and on the noise-free run. Its AR(1) switch fired only on the noise-free
run. It flagged 19 of the 40 noisy runs as ambiguous.

## What it returned (exploratory from here on)

At 5% noise eqdisc returns a dz equation in z alone, dz = 0.27 z - 0.20 z^2 + 0.03 z^3, and adds xz^2 and yz^2 terms to dx
and dy. Refitting the clean window by least squares:

- dz on z, z^2, z^3: R^2 = 0.9958
- dz on the target z, x^2, y^2: R^2 = 0.5927

This does **not** show that eqdisc's equation is better physics, and an earlier version of this report suggested too
much. The window holds one transient on which z rises almost monotonically, so dz is close to a function of z by
construction, and a z-only fit will score well on any such curve. A ripple explanation was also tested and failed:
averaging x^2 and y^2 over one period raises the target's R^2 only from 0.55 to 0.61 (t = 40 to 120). Why the target's
dz equation fits this poorly is not established. This was found after the result and does not change the verdict.

It also does not rescue the method's answer. At 5% noise the returned models get z and omega right on the two runs inspected
(z 2.25, omega 1.13) but not the amplitude (r^2 22 to 31 against 5.31), and 0 of 10 pass the dynamics check. A z-only
equation also drops the physical link between the shift and the oscillation amplitude.

## Comparison with published work (checked after the result)

The test was not a faithful recreation of Brunton et al. (2016), and the pre-registration missed this:

- **Training data.** Their system identification includes off-attractor trajectories, started from the mean flow. They
  report that without those trajectories the method "incorrectly identifies the dynamics using cubic nonlinearities, and
  fails to correctly identify the dynamics associated with the shift mode". This pilot used one trajectory from near the
  unstable steady state, the setting they warn about, and eqdisc returned cubic terms and a wrong dz equation. The target
  may not be identifiable from this file, so the failure on exact and core recovery says little about the method.
- **Standard.** Their table is not an exact 11-term recovery, and the vortex section does not report added noise. Exact
  recovery of 11 terms at 1 to 20% noise is a stricter test than the published one.
- **Library and simulation.** Their table extends beyond degree 3 and they used an immersed-boundary code. This file comes
  from Nek5000 and has eight POD modes plus the shift mode, the basis of Noack et al. (2003).
- **Cubic terms.** Loiseau and Brunton (2018) report that cubic terms in a three-state model of this flow are stabilising
  and give a much better fit, so cubic terms are not wrong as such.

The dynamics result does not depend on the target and stands: at 5% noise eqdisc reproduces the limit cycle on 0 of 10
runs, and the weak chi2s selector on 8 of 10. A fair test of the target needs off-attractor trajectories, which this file
does not contain.

## Limits of this pilot

- The target support and window were chosen after inspecting the data and fitting the reference.
- This is simulated flow reduced to POD coefficients, not a measurement. DEDi was not run.
- The real runs ran on the Spark and the control on this PC, with different numpy and scipy versions.
