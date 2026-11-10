# Real data 2 (MultiArm pendulum) with the frozen model: pre-registration

Written 2026-10-26, before any model (ours or a baseline) was fitted to these real data. Runner:
`benchmarks/real2_pendulum.py`. Model: the frozen default v1.1, `eqdisc.discover(y, dt, library=..., noise="auto")`;
`eqdisc/` is not modified.

## What was done before this file was written (disclosed)

- **Search.** No public measured data were found for a chaotic system whose accepted model lies in a frozen
  library (D = 3 polynomial ≤ degree 5, or trig3). Searches covered Zenodo, figshare, DataCite, GitHub and
  the literature: Lorenz/jerk/Chua/Rössler circuits, waterwheel, driven pendulum and Duffing. Every
  analog-Lorenz circuit paper found says the data are "available on request". The public circuit data that do
  exist (Minati et al., Zenodo; Guadalajara Rössler-like and multiscroll circuits) all have non-polynomial
  models: transistor, piecewise-linear or fractional-power.
- **Data inspection, without any model fit:**
  - checked that the 6 single-pendulum pieces are contiguous;
  - integrated the published single-pendulum model on each piece: θ R² 0.952, 0.963, 0.981, 0.991, 0.998,
    0.992;
  - least squares of a Savitzky–Golay derivative of θ̇ on [sin θ, θ̇] gives A = 64.06 and B = 0.012. So the
    damping is weakly identified: the published trajectory-fit value is 0.067;
  - for the double pendulum, the published model's forecast stays within 0.5 rad for 2.8 to 5 s;
  - persistence stays within 0.5 rad for about 0.1 s.
- **Code check on synthetic data only.** This used the published single-pendulum model integrated from the
  data's first state, with no noise, at the A0 sampling (arm `SYN`):
  - eqdisc returns the exact 4-term model, A = 64.220 and B = 0.06723;
  - pysindy SFD-STLSQ (CV pick) returns a dense model, and its exact support is never on the threshold path;
  - weak-STLSQ (AICc pick) returns a dense model that equals the true RHS on the circle sin² + cos² = 1;
  - both baseline failures come from the collinear library (see the caveat below). These synthetic numbers are
    not results.

## Data

Source: the MultiArm pendulum on a cart, from Kaheman, Fasel, Bramburger, Strom, Kutz and Brunton, *The
Experimental Multi-Arm Pendulum on a Cart: A Benchmark System for Chaos, Learning, and Control* (arXiv 2205.06231;
HardwareX 2023). Files are from github.com/dynamicslab/MultiArm-Pendulum, commit
`e99a7e13fe2adc8fc647321a156b6cb59a48a698`, raw URL
`https://raw.githubusercontent.com/dynamicslab/MultiArm-Pendulum/<commit>/<path>`.

The local copy is `data/real2/multiarm_pendulum/` (gitignored). Each file is read only with `scipy.io.loadmat`.

| File (repo path) | sha256 |
| --- | --- |
| ParameterEstimation/SinglePendulum/SinglePendulumDataForParameterEstimation.mat | f3b8cdda4a6e86405ec37ff32b98be76fd37806a609cba7870628c1e40377a98 |
| ParameterEstimation/SinglePendulum/EstimatedParameters_SinglePendulum.mat | 64108f0f69a202f8893a376bd87b69cd54ed13664aed33db085766639a5b0962 |
| Datas/DoublePendulum/DoubleDataFreeSwing_1_Dt_0_001.mat | 2af828048debc0ec33dcd9f46538b747a72a7bfaa3b333852cc474db5ada7633 |
| Datas/DoublePendulum/DoubleDataFreeSwing_2_Dt_0_001.mat | b0f94afdc6f1bb20285ca9fd416ddb249521ac678ac84f45c0881f4d9dcb8ff2 |
| Datas/DoublePendulum/DoubleDataFreeSwing_3_Dt_0_001.mat | 8e8369479e135b8bbd3fc292b051b314eec61e1ac98dc56cb871ceb8299978eb |
| Datas/DoublePendulum/DoubleDataFreeSwing_4_Dt_0_001.mat | 2876a0d76708725723bf382dce3e42a1c37327d88775671abb6c9ebd448c77c6 |
| ParameterEstimation/DoublePendulum/EstimatedParameters_DoublePendulum.mat | 786fb808d6ecfd770cf89d770778cb473a66223a5e63790bbc94277f2e5b3517 |

Other files were downloaded for reference only and are not used; their hashes are in `results_real2/SOURCES.md`.

**A. Single pendulum, free swing.** The data are the authors' processed parameter-estimation set:
- arm angle θ from the encoder;
- θ̇ from the authors' velocity channel, median-filtered and wavelet-denoised by them;
- 6 contiguous pieces of 9.167 s at dt = 1 ms (55.0 s in total), decaying from about ±1.6 rad around the
  hanging position θ = π.

`Datas/SinglePendulum/SingleDataFreeSwing_1` is not used. It has amplitude ±0.07 rad, which is the linear
regime, and it has no pendulum-velocity channel. The raw parameter-estimation file is a MATLAB opaque object
that scipy cannot read.

**Accepted model** (`SinglePendulumODE.m`, Lagrangian with viscous joint friction):
θ̈ = (a1·g·m1·sin θ − k1·θ̇)/(m1·a1² + I1). With the published estimates this is A = 64.219 s⁻² and
B = 0.06723 s⁻¹, so θ̈ = A sin θ − B θ̇.

**B. Double pendulum, free swing** (cart locked). Four runs of θ1, θ2, θ̇1, θ̇2 at dt = 1 ms, each 150–209 s
long. The swings are large but there are no flips. The accepted model is `DoublePendulumODE_Mounted.m`
with the published 10 parameters. It is rational in cos(θ1 − θ2).

## Coordinates and library (rule fixed in advance)

**Rule.**
- **Coordinates.** Use the measured states. Apply one fixed, model-independent change of variables only if it
  puts the accepted model inside a frozen library.
- **Library.** Use the smallest frozen library that contains the accepted model in those coordinates. If no
  frozen library contains it, use the frozen default (`auto`).
- **Noise model.** Always `noise="auto"`. Everything else is left at the frozen defaults.

**A.** The coordinates are (x, y, z) = (sin θ, cos θ, θ̇). The accepted model becomes:
- dx/dt = y·z
- dy/dt = −x·z
- dz/dt = A·x − B·z

That is 4 terms, all of degree ≤ 2, so the library is **poly3**. The first two equations are kinematic identities
of the embedding; only the dz/dt equation carries physics.

*Known caveat:* x² + y² = 1 on the data. The poly3 columns {1, x², y²}, {x, x³, xy²}, {y, x²y, y³} and
{z, x²z, y²z} are therefore exactly linearly dependent, and the accepted model is only the *sparsest* of
infinitely many equivalent forms. The test therefore also asks for the sparsest representation.

**B.** No frozen library contains the accepted model: it is rational, and D = 4. So the library is `auto`: the
degree-2 `joint_nd` library in the raw (θ1, θ2, θ̇1, θ̇2). Structural recovery is impossible by construction,
so this part is an **out-of-library, chaotic forecasting check**. The expected outcome is failure.

## Arms

| Arm | Data | dt | N |
| --- | --- | --- | --- |
| **A0 (primary)** | all 55 s, every 25th sample, authors' θ̇ | 0.025 s (~31 samples/period) | 2201 |
| A1 | first 20 s (largest amplitude), every 10th sample | 0.01 s | 2000 |
| A2 | all 55 s, every 50th sample | 0.05 s | 1101 |
| A3 | as A0, but θ̇ = central difference of the 1 kHz θ (no authors' denoising) | 0.025 s | 2201 |
| A4 | as A0, library trig3 (a superset; sensitivity to the library choice) | 0.025 s | 2201 |
| B-r (r = 1..4; primary r = 1) | run r, first 20 s, every 10th sample | 0.01 s | 2000 |

## Metrics and success criteria

**A, on the primary arm A0.** Success means all three of the following hold.
- **S1, structure:** the support equals the 4-term accepted support exactly. Also reported: the physics equation
  alone (dz/dt support = {x, z}), extra and missing terms, and per-term confidence.
- **S2, coefficients:**
  - A is within 5% of 64.22;
  - the y·z and −x·z coefficients are within 5% of ±1;
  - B > 0 and within a factor of 2 of 0.0672. B is weakly identified, as shown above.
- **S3, fit:** from the measured state at the start of each of the 6 pieces, integrate 9.167 s and compute θ = atan2(x, y)
  (unwrapped) and its R² against the measured θ. This is the authors' own validation protocol. The median R²
  over the 6 pieces must be ≥ the published model's median (0.986) − 0.05.

Also reported: the relative RMS difference between the discovered and accepted right-hand sides on the measured
states. This credits models that are equivalent modulo x² + y² = 1. The sensitivity arms A1–A4 are reported
as they come out; no arm replaces A0.

**B.** Forecast valid time is the first time max(|Δθ1|, |Δθ2|) > 0.5 rad, with a 5 s horizon and LSODA from the
measured state.
- **Starts:**
  - same run: 10 starts at t = 20, 24, …, 56 s, after the training window;
  - each other run: 10 starts at t = 0, 4, …, 36 s.
- **Comparisons:**
  - the published accepted model;
  - persistence (angles frozen).
- **Reported:** the median valid time over all 40 starts, and the number of discovered terms. There is no
  success claim; the model "beats persistence" only if its median valid time is > 2× persistence's.

## Baselines (same data, same coordinates, same polynomial library)

pysindy 2.1.0 via `benchmarks/baselines_bench.run_path`, with the configs fixed in `docs/reports/baselines.md`
(config per rule, one threshold path shared by all equations):

| Baseline | Config |
| --- | --- |
| SINDy STLSQ (SFD), CV pick | raw, window 11 |
| SINDy STLSQ (SFD), AICc pick | raw, window 5 |
| weak SINDy STLSQ, CV pick | rms, h 10 |
| weak SINDy STLSQ, AICc pick | rms, h 10 |

- Also recorded: whether the exact accepted support appears anywhere on the threshold path (oracle).
- The baselines are evaluated with the same metrics.
- They are not run on A4, because pysindy's library here is polynomial.
- For B they use the degree-2 library and the same forecast evaluation.

## Compute and outputs

- **Compute:** the Spark, at most 3 concurrent single-thread processes.
- **Outputs:**
  - `results_real2/single_<arm>_<method>.json`
  - `results_real2/double_r<k>_<method>.json`
  - logs
- **Report:** `docs/reports/real_data2.md`.
- **If the frozen model crashes:** the crash is recorded. A clear crash bug may be fixed in a separate commit and
  verified with `tests/test_equivalence.py`; results from before and after the fix are both reported.
