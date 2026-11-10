# Real data 2: the MultiArm pendulum (single and double) with the frozen model (2026-10-26)

**Bottom line.** No public measured dataset was found for a chaotic system whose accepted model lies in a frozen
library. So two measured systems were run from one public experimental rig.

- **Single pendulum** (in-library after the embedding (sin θ, cos θ, θ̇); not chaotic). The pre-registered
  primary test **fails**.
  - The model finds the gravity term exactly (A within 0.2%) and both kinematic terms.
  - It misses the viscous damping, adds 11 small extra terms, and so fails exact recovery.
  - Its trajectory fit looks excellent (median θ R² 0.9998, against 0.986 for the published model). But most
    of that fit comes from non-physical terms in the kinematic equations: the physics equation alone fits worse
    than the published model (0.67).
  - With θ̇ taken from finite differences of θ instead of the authors' wavelet-denoised channel (arm A3), the
    physics equation is A·sin θ − B·θ̇ with B within 12%, plus 3 small terms. It is not exact, but it is the
    best answer of any method.
  - No pysindy baseline gets the exact support in any arm, even at its oracle threshold.
- **Double pendulum** (chaotic; the accepted model is outside every frozen library). The frozen model **fails
  badly** and is **worse than the pysindy baselines**: forecast valid time 0.02–0.13 s, against 0.5–1.5 s for
  SINDy and ~0.13 s for doing nothing.
  - It does not even return the identity dθ/dt = ω. This happens on noise-free simulated data from the
    accepted model as well.
  - So under misspecification, the joint flow fit lets the wrong dynamic equations contaminate the kinematic
    ones.
  - The equation-wise SINDy baselines (all four on run 4, and weak SINDy on run 1 as well) return dθ/dt = ω
    almost exactly.

This is one rig and two systems. It should be presented as evidence of limits, not as a success.

Files:
- pre-registration (committed before any real-data fit): `results_real2/PREREGISTRATION.md`;
- sources and sha256: `results_real2/SOURCES.md`;
- runner: `benchmarks/real2_pendulum.py`;
- tables: `benchmarks/real2_report.py` writes `results_real2/tables.md`;
- results: `results_real2/{single,double}_*.json` and `.log`;
- exploratory runs: `results_real2/explore/`.

Everything ran on the Spark, 3 single-thread processes, in about 8 min.

## Data

Kaheman, Fasel, Bramburger, Strom, Kutz and Brunton, *The Experimental Multi-Arm Pendulum on a Cart* (arXiv
2205.06231, HardwareX 2023). The files are from github.com/dynamicslab/MultiArm-Pendulum, commit `e99a7e1`, MIT
licence, and the sha256 of every file is recorded. Raw `.mat` files are in `data/real2/` (gitignored, 38 MB).

**Single pendulum** (encoder; cart locked). The data are the authors' parameter-estimation set:
- θ, and θ̇ after the authors' median filter and wavelet denoising;
- 55 s of free decay at 1 kHz, from ±1.6 rad down to ±0.4 rad around the hanging position.

The accepted model is the authors' Lagrangian with viscous joint friction, θ̈ = A sin θ − B θ̇. From their
published estimates, A = 64.22 s⁻² and B = 0.0672 s⁻¹.

**Double pendulum** (cart locked): four free-swing runs of θ1, θ2, θ̇1, θ̇2 at 1 kHz, with large swings but no
flips. The accepted model is the authors' `DoublePendulumODE_Mounted.m` with their 10 estimated parameters. It
is rational in cos(θ1 − θ2). With the published parameters it stays within 0.5 rad of the data for ≥ 5 s
from most starts.

### Why not a chaotic circuit

Searches covered Zenodo, figshare, DataCite, GitHub and the literature. None found a public recording of a
polynomial chaotic circuit (Lorenz, jerk or Sprott type):
- The Lorenz-circuit papers (e.g. Restrepo et al., arXiv 2307.03690) give their data "on request".
- The public chaotic-circuit sets have non-polynomial accepted models:
  - Minati et al. on Zenodo: transistor and fractional-power jerk circuits;
  - the Guadalajara Rössler-like and multiscroll circuits: piecewise-linear.
- The ULiège electronic-Duffing sets on figshare are forced, periodic, control-based continuation data.

The best next step remains the one in `real_data.md`: record an op-amp/AD633 Lorenz or Sprott circuit
ourselves.

## Protocol (pre-registered)

**Single pendulum.**
- **Coordinates and library.** (x, y, z) = (sin θ, cos θ, θ̇). The accepted model becomes dx = yz, dy = −xz,
  dz = A x − B z: 4 terms, so the library is **poly3**.
- **Caveat.** x² + y² = 1 on the data, so the poly3 columns are exactly collinear (1 ~ x² + y², x ~ x³ + xy²,
  …). "Exact" therefore means the sparsest form.
- **Noise model.** `noise="auto"`.
- **Arms:**

  | Arm | Data | dt | N |
  | --- | --- | --- | --- |
  | **A0 (primary)** | 55 s, every 25th sample | 0.025 s | 2201 |
  | A1 | first 20 s | 0.01 s | 2000 |
  | A2 | 55 s, every 50th sample | 0.05 s | 1101 |
  | A3 | as A0, θ̇ from central differences of θ | 0.025 s | 2201 |
  | A4 | as A0 with library trig3 | 0.025 s | 2201 |

- **Success on A0 needs all three of:**
  - S1: exact support;
  - S2: A and the kinematic coefficients within 5%, and B > 0 within a factor of 2;
  - S3: median θ R² over the authors' 6 validation pieces (each integrated 9.167 s from its measured start) ≥
    the published model's 0.986 − 0.05.

**Double pendulum.**
- **Library.** No frozen library contains the model, so the rule gives `auto`: degree-2 `joint_nd`, D = 4, on
  the raw angles and rates.
- **Training data.** The first 20 s of each run, every 10th sample (dt = 0.01 s, N = 2000).
- **Score.** Forecast valid time: max angle error < 0.5 rad, 5 s horizon, 40 starts after or outside the
  training window.

**Baselines.** pysindy 2.1.0 through `baselines_bench.run_path`, with the configs fixed earlier in
`docs/reports/baselines.md`:
- SFD-STLSQ with the CV pick (raw, window 11) and the AICc pick (raw, window 5);
- weak-SINDy STLSQ with the CV and AICc picks (rms, h 10);
- plus "is the exact support anywhere on the threshold path" (oracle).

## Results: single pendulum

| Arm | Method | Exact | dz eq exact | Terms | A (64.22) | B (0.0672) | RHS rel. RMS vs accepted (dx, dy, dz) | Median θ R² (accepted 0.986) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A0** | **ours** | no | no | 15 | **64.08** | **0 (dropped)** | 0.008, 0.023, 0.014 | **0.9998** |
| A0 | SINDy STLSQ, CV | no | no | 45 | 54.4 | 0.76 | 0.032, 0.062, 0.013 | fails (diverges) |
| A0 | SINDy STLSQ, AICc | no | no | 52 | 42.9 | −0.05 | 0.010, 0.026, 0.012 | fails |
| A0 | weak SINDy, CV | no | no | 34 | 24.1* | 0.10 | 0.005, 0.010, 0.009 | 0.985 |
| A0 | weak SINDy, AICc | no | no | 33 | 24.1* | 0.06 | 0.005, 0.009, 0.009 | 0.917 |
| A1 | ours | no | no | 15 | 64.06 | 0 | 0.004, 0.008, 0.008 | 0.999 |
| A2 | ours | no | no | 11 | 64.08 | −724† | 0.005, 0.009, 0.010 | 0.496 |
| **A3** | **ours** | no | no | **9** | **63.98** | **0.059** | **0.0013, 0.0015, 0.0042** | **0.999** |
| A3 | weak SINDy, CV | no | no | 20 | 24.1* | 0.18 | 0.0003, 0.0002, 0.0037 | 0.999 |
| A3 | weak SINDy, AICc | no | no | 30 | 24.0* | 0.11 | 0.0008, 0.0003, 0.0037 | 0.999 |
| A3 | SINDy STLSQ, CV / AICc | no | no | 40 / 53 | 52 / 42 | ≈0 / 0.84 | 0.03, 0.06, 0.01 | 0.90 / 0.86 |
| A4 (trig3) | ours | no | no | 15 | 64.06 | 0.004 | 0.005, 0.008, 0.007 | fails (θ R² ≈ 0.02) |

Notes on the table:
- \* The baselines spread the gravity term over x, x³ and xy², which is equal to A·x on the circle. Their A
  column is therefore not comparable; the RHS column is.
- † A2 writes the damping as 723.6·z − 723.5·(x² + y²)·z, which is the same function on the circle. Off the
  circle it is unstable, and that is why the integrated fit fails.
- No baseline has the exact support anywhere on its threshold path, in any arm. All baseline rows are in
  `results_real2/tables.md`.

Our primary-arm (A0) model:
```
dx/dt = -0.04787*x +0.01393*x*y +0.9999*y*z -0.003734*x*y^2 -9.166e-05*x*z^2
dy/dt = -0.07799 -0.9995*x*z +0.07614*y^2 -0.004638*z^2 -0.005064*y*z^2
dz/dt = +64.08*x -0.04593*y -0.09621*x^2 -0.001492*y*z^2 +0.000654*z^3
```
Our A3 model (θ̇ from finite differences):
```
dx/dt = +1.001*y*z -0.001213*x*y*z
dy/dt = -1.001*x*z +0.001192*x^2*z
dz/dt = +63.98*x -0.05893*z +0.04954*y^2 -0.0001829*y*z^2 +5.32e-05*z^3
```
In A3 the kinematic extras are rewrites of the identity on the circle; for example, x²·z is equivalent to
z − y²·z. Per-term confidences for A3:
- x, z and the kinematic terms: 0.99–1.0;
- y²: 0.63;
- y·z²: 0.92;
- z³: 1.0.

**Pre-registered verdict on A0.**
- S1 (exact support) fails: 11 extra terms, and the damping term is missing.
- S2 fails on B, which was dropped, although A (−0.2%) and the kinematic coefficients (< 0.1%) pass.
- S3 passes, at 0.9998 against a threshold of 0.936.

The test is therefore **not passed**.

**Sensitivity.**
- Structure varies with the sampling (A0/A1/A2) and with the velocity source (A3).
- A at 64.0–64.1 is the only quantity that is stable across all arms.
- With the authors' denoised θ̇, the damping is never found; with finite-difference θ̇ (A3) it is.
- The superset library trig3 (A4) adds sin/cos terms and cancelling x/x³/xy² triples.

### Exploratory (post hoc): is the good fit physical?

The pre-registered fit (S3) integrates the 3-D model, and the kinematic equations can drift off the circle. So
afterwards we integrated only each model's dz/dt equation on the circle, with θ̇ = ω and ω̇ = dz/dt at
(sin θ, cos θ, ω). The table gives the median θ R²; the published model scores 0.986.

| Arm | ours | SINDy CV | SINDy AICc | weak CV | weak AICc |
| --- | --- | --- | --- | --- | --- |
| A0 | 0.668 | 0.730 | 0.805 | 0.878 | 0.872 |
| A1 | 0.897 | 0.838 | 0.839 | 0.822 | 0.863 |
| A2 | 0.965 | 0.695 | 0.738 | 0.853 | 0.891 |
| A3 | 0.999 | 0.987 | 0.985 | 0.999 | 1.000 |
| A4 | 0.908 | – | – | – | – |

How to read it:
- On the primary arm, our physics equation is the *worst* of the five. Its z³ term is anti-damping, so energy
  is gained at high speed. The 0.9998 fit came from the kinematic extras (−0.048x in dx/dt and
  −0.078 + 0.076y² in dy/dt), which shrink the radius and stand in for dissipation. That is a good fit for a
  non-physical reason.
- Only with finite-difference velocities (A3) does every method get a physics equation that beats the
  published fit.

## Results: double pendulum (out of library, chaotic)

The table gives the median forecast valid time over 40 starts. The published model gets 5.0 s (the horizon) and
persistence 0.13 s.

| Training run | ours | SINDy STLSQ CV | SINDy STLSQ AICc | weak CV | weak AICc |
| --- | --- | --- | --- | --- | --- |
| 1 | **0.037** | 1.46 | 1.47 | 0.79 | 0.69 |
| 2 | **0.041** | 1.04 | 0.83 | 0.96 | 0.87 |
| 3 | **0.024** | 0.60 | 0.59 | 0.56 | 0.47 |
| 4 | **0.127** | 0.57 | 0.51 | 0.48 | 0.59 |

Ours on run 1 (20 terms, not flagged ambiguous, gap 575):
```
dx0/dt = +10.28*x2 -0.2812*x0^2 +0.2119*x0*x1 -1.308*x1*x2 +0.007803*x3^2
dx1/dt = -10.32*x2 +1.207*x3 +2.524*x0*x2 -0.008216*x1*x3 -0.06457*x2*x3
...
```
- **Kinematic identities.** The data satisfy dθᵢ/dt = θ̇ᵢ to correlation 0.9999 with no lag. The
  equation-wise baselines recover it: weak SINDy CV gives dx0/dt = 0.9997·x2 and dx1/dt = 0.9986·x3. Ours
  never does, on any run.
- **Exploratory: centred angles (θ − π).** This changes nothing: ours gets 0.004–0.12 s, SFD-STLSQ
  0.57–1.46 s (`results_real2/explore/`).
- **Exploratory: noise-free data simulated from the accepted model**, same sampling and start. Ours still
  returns dx0/dt = 1.076·x2 − 0.379·x1 + … and dx1/dt = 0.9994·x3 + 0.32 + …, again with a large gap (74).
- **Interpretation (a hypothesis, not tested further).** When the ω-equations cannot be represented, the
  profiled joint-flow objective distorts the equations that *can* be represented to reduce the trajectory
  misfit, and its ambiguity flag does not fire. Equation-wise regression keeps a misspecified equation from
  contaminating the others.

## Honest assessment

- **In library, real noise (single pendulum).**
  - The frozen model reliably identifies the dominant physics: the gravity coefficient within 0.4% in every arm,
    and the kinematic terms within 0.2%.
  - It does not deliver the sparse accepted model. It misses the 1%-of-RHS viscous damping on the primary arm,
    adds 5–11 small terms, and is sensitive to sampling and to the velocity preprocessing.
  - The headline trajectory fit is excellent, but the exploratory check shows it is partly non-physical.
  - The exactly collinear library (sin² + cos² = 1) makes this harder than our synthetic benchmarks. The
    baselines fare no better: no exact support anywhere on their paths.
  - Before the run, a damping-only least-squares fit on Savitzky–Golay derivatives already disagreed with the
    published B (0.012 vs 0.067). So B is genuinely weakly identified in these data, and the published value is
    itself a trajectory fit, not ground truth.
- **Out of library, chaotic (double pendulum).** A clear negative result:
  - the model is worse than SINDy and no better than persistence;
  - it breaks exact kinematic identities;
  - it is not flagged ambiguous.

  This is a limitation the paper should state: guarantees and flags were validated only for in-library
  systems, and misspecification is not detected.
- **What a convincing positive real-data test still needs:** a measured polynomial chaotic system, i.e. a
  recorded analog Lorenz/Sprott circuit with all three state voltages. For pendulum-type data, a library with
  trig terms in D = 4, or a way to fix known kinematic equations, would also be needed.
- **Code:** no crash and no frozen-code change. One option is post hoc and marked exploratory in the runner:
  `--centre-angles`.
