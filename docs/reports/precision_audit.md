# Precision audit: are some failures floating-point bugs? (2026-10-27)

**Bottom line.**
- **A. Spurious tiny terms on noise-free data: real, but not floating point.** The mechanism is the discretisation
  error of the implicit-midpoint latent grid. On noise-free data it is the only "noise" left, and it is larger than
  the model-residual scale the likelihood assumes. BIC then lets smooth extra terms absorb a deterministic error.
  The estimated noise scale never comes near float64 resolution: it is about 1e-5 of the signal in 3-D, and the
  1e-6 floor only binds in 2-D. A noise-scale floor derived from that discretisation error (**v1.2**) was designed
  on dev only, frozen and pre-registered.
  - **Fresh noise-free test archive:** exact recovery goes from **0.265 to 1.000**, **W/L 147/0, p = 1e-44**.
  - **Sealed white, coloured and t₃ guards:** **0** support changes in 2,400 runs. The v1.1 arm reproduces the
    stored v1.1 records bit for bit.
  - **nd guard:** 0 support changes in 400 runs. **Pytest** (frozen v1.0 equivalence) passes.
  - **Decision: v1.2 confirmed and now the default.** v1.1 and v1.0.1 stay callable with `version=`.
- **B. dysts tail: not numerical (clean negative).** The scaled-library condition numbers are moderate (10–10⁴)
  for every system but one, and they do not correlate with failure.
  - The profiled-quadratic matrices are badly conditioned: 1e10–1e18 on 7 of the 10 failing systems, with the
    1e-10 ridge above λ_min on 6 of them.
  - Two numerics-only variants of the joint stage were tried on 384 fresh dev instances (10 worst systems plus 6
    controls): backward-stable QR subset search with no ridge, and a square-root (QR) construction of the profiled
    quadratic. They change **1 of 384** selected supports and **0** exact outcomes (W/L 0/0).
  - The failures are genuine statistical or identifiability limits of the current method, not rounding.
- **Crash fix (separate commit `80993d1`).** `_discover_nd` crashed with `ValueError` when two candidates had
  exactly equal scores; this happened on 2 of 8 noise-free 2-D oscillator dev runs (plus 1 more in the halved-step arm). It is fixed with an
  order-preserving sort key, and the output is unchanged wherever the old code did not crash.

All design work used dev data of our own generation. No test archive, ODEBench run or dysts test run was used to
design anything.

Files:
- pre-registration: `results_precision/PREREGISTRATION_v12.md`, commit `fb1121e`, before the test archive existed
- archive sha256: commit `b4aac90`, before any run
- code: `eqdisc/joint_discover.py` (`midpoint_lte`, `disc_floor_sigma`), the `disc_floor=` option in
  `joint_discover` / `joint_colored` / `joint_nd`, and `final_model.VERSIONS["1.2"]`
- diagnostics: `benchmarks/precision_a_dev.py`, `precision_b_dev.py`, `precision_numerics.py`
  (instrumented / hardened / QR numerics)
- tables: `benchmarks/precision_report.py`, output in `results_precision/test_report.md`
- records and logs: `results_precision/`

## A. Noise floor for near-noise-free data

### A1. Mechanism (dev: `dev_clean.npz`, 80 noise-free 3-D runs + 32 noise-free 2-D runs of four degree-2 systems)

`benchmarks/precision_a_dev.py`, `results_precision/a1_dev.json`. All quantities are in the model's scaled units
(x / rms x).

| | 3-D (poly3, 80 runs) | 2-D (joint_nd, 30 runs*) |
| --- | --- | --- |
| v1.1 exact | 0.31 (all misses are extra terms; fn = 0) | 0.63 |
| Estimated σ (median, range) | 1.4e-5 (1e-6 – 1.2e-4) | 1e-6 (the 1e-6 floor binds in 30/30) |
| Runs where the 1e-6 floor binds | 5/80 | 30/30 |
| Noise model chosen by the router | AR(1) in 80/80 (residuals are smooth) | white (D ≠ 3) |
| ρσ / τ_rms (model-residual scale assumed / implicit-midpoint truncation error), median [10–90%] | **0.29 [0.10, 0.48]** | 0.27 [0.04, 1.85] |
| Extra terms: relative contribution, median (max) | 1.4e-4 (3.6e-3), 205 terms | 4.8e-6 (5.4e-5), 29 terms |
| Same runs, latent step halved (`substeps` × 2) | 0.58 | 0.79 |
| Same runs, forced white noise model | 0.10 | – |

\* Two of the 32 2-D runs crashed under v1.1, which is the tie-sort bug fixed in `80993d1`.

**Reading.**
- **The hypothesis as stated does not hold.** σ does not collapse to a 1e-12 floor. That floor
  (`flow_discovery.py:278`) is only used inside the flow-discovery starts, not in the selection. The selection's
  σ floor is 1e-6, and in 3-D the self-calibration keeps σ above it. Float64 rounding of the signal is ~1e-16, about
  10 orders below every σ seen. Nothing here is at machine precision.
- **What is happening instead.** The self-calibrated σ tracks the misfit that the discretisation induces at the
  samples. But the model residual is weighted by 1/(ρσ) with ρ = 0.03, and the implicit-midpoint local truncation
  error τ is about 3× larger than ρσ.
  - So even the true model and exact states carry a model-residual χ² of ~10 per latent step.
  - This error is smooth and deterministic, so a small extra term can absorb part of it and repay its BIC penalty.
- **Causal check.**
  - Halving the latent step cuts τ about 8× and doubles exact recovery (0.31 → 0.58 in 3-D).
  - The residual is not white, so forcing the white model makes it worse (0.10).

### A2. Design (dev only)

A floor on σ from an estimate of the scheme's discretisation error (`joint_discover.disc_floor_sigma`). It needs
no truth and has no tuned constant.

1. **Estimate τ.** Take the best start after self-calibration (fitted states u on the latent grid of step h,
   fitted coefficients C). Compute τ_k = z_k − u_k − h f((u_k+z_k)/2), where z_k is a 4-sub-step RK4 solution of
   the fitted model from u_k.
2. **Cap its χ².** Σ_k |τ_k/(ρσ)|² is deterministic approximation error, so it may not exceed the selection
   penalty of one term, pen = λ·φ·log(nD), split equally over the D components:
   **σ_i ← max(σ_i, sqrt(D Σ_k τ_ki² / pen) / ρ)**.
3. **Refit if needed.** If the floor raises σ, the starts are refitted, exactly as the existing self-calibration
   does.

This is a Richardson-type estimate of the scheme's own error, not a post-hoc rule on term sizes. The "<1%
contribution" count seen on ODEBench was not used. A relative-float-precision floor was rejected because it would
change nothing: σ is already 10 orders above it.

Dev results (`results_precision/a2_dev.json`, `guard_dev_*.json`):

| Dev set | v1.1 | v1.2 | W/L | Support changes | Floor active |
| --- | --- | --- | --- | --- | --- |
| Noise-free 3-D (80) | 0.31 | **1.00** | | | 80/80 (raises σ by 8–120×, median 38×) |
| Noise-free 2-D (32) | 0.66 | **0.97** | 65/0 over both | | 29/32 |
| White η 1–20% (`instances.npz`, 800) | 0.890 | 0.890 | 0/0 | **0** | 47 (η = 1%, mostly N = 2000; coefficients only) |
| Coloured (`instances_colored.npz`, 800) | 0.792 | 0.792 | 0/0 | **0** | 28 |
| t₃ (`instances_student.npz`, 800) | 0.915 | 0.915 | 0/0 | **0** | 74 |
| nd D = 5–10 (`nd_dev.npz`, 400) | 0.915 | 0.915 | 0/0 | **0** | 28 |

The one remaining 2-D dev miss is Lotka–Volterra at N = 2000, with two terms of relative size 5e-5. The median run
time is not higher (5.3 s → 4.5 s), because fewer candidates are fitted.

### A3. Frozen test (pre-registered; `results_precision/PREREGISTRATION_v12.md`)

**Primary: fresh noise-free archive.** `gen_variants.py --kind white --etas 0 --seed 20261127` →
`data/sealed_clean.npz`.
- Same generator family as the sealed archives: 10 systems × N ∈ {2000, 500} × 10 seeds, giving 200 runs.
- sha256 `5dc1f23a…5fb00d1`, committed before any run.
- No first sample is shared with the 22,341 trajectories of the other archives.

| | v1.1 | **v1.2** |
| --- | --- | --- |
| Exact | 0.265 | **1.000** |
| Mean extra terms (fp) | 2.80 | 0.00 |
| Mean missed terms (fn) | 0.00 | 0.00 |

**W/L 147/0, two-sided sign test p = 1.1e-44.**
- **By length.** N = 2000: 78/0; N = 500: 69/0.
- **By system.** v1.2 is exact on all 20 runs of every system. v1.1 was at 0.00 on Lorenz, Rössler and Sprott E,
  and at 1.00 on Halvorsen only.
- **The floor in use.** Active in 200/200 runs, raising σ by a median of 42× (10–90%: 15–120×).
- **Coefficients.** Median coefficient error on the 53 runs both versions get exact: 2.4e-4 (v1.1) vs 1.5e-4
  (v1.2).

**Guards (sealed test archives; both arms in the same record).**

| Archive | Runs | Floor active | Coefficients changed | **Support changed** | v1.1 exact | v1.2 exact | W/L | v1.1 arm = stored v1.1 record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| White `joint_sealed.npz` | 800 | 44 | 44 | **0** | 0.889 | 0.889 | 0/0 | 800/800 |
| Coloured `sealed_colored2.npz` | 800 | 24 | 24 | **0** | 0.805 | 0.805 | 0/0 | 800/800 |
| t₃ `sealed_student.npz` | 800 | 75 | 75 | **0** | 0.905 | 0.905 | 0/0 | 800/800 |
| nd `nd_test.npz` (D = 5–10) | 400 | 35 | 35 | **0** | 0.917 | 0.917 | 0/0 | – |

"v1.1 arm = stored v1.1 record" means bit-identical ξ against `results_final_v11/guard_sealed.json`,
`guard_student.json` and `test_colored2.json`. The v1.2 code path changed nothing for v1.1.

**Secondary (NOT independent; this is where the problem was first seen).** ODEBench σ = 0, fit IC 0, 63 systems:

| | v1.1 | v1.2 |
| --- | --- | --- |
| Exact, in library (23) | 0.391 | **0.870** (W/L 11/0) |
| D = 2 in library (6) | 0.000 | 1.000 |
| D = 3 in library (8) | 0.250 | 0.750 |
| D = 1 in library (8) | 0.750 | 0.875 |
| Reconstruction / generalization accuracy (R² > 0.9) | 74.6% / 55.6% | 74.6% / 55.6% |

- **R² is unchanged.** The extra terms were negligible for prediction anyway.
- **ODEBench's own solutions** carry LSODA rtol 1e-5 errors. The floor does not model these, and the floor was
  active in only 18/63 runs, yet the exact rate rose from 39% to 87%.
- **v1.1 vs v1.0.1 at σ = 0.** v1.1's 0.391 is one run below v1.0.1's 0.43 in `docs/reports/odebench.md`: noise-free
  3-D runs route to AR(1), where v1.1 differs from v1.0.1 by design.

**Pytest.** `tests/test_equivalence.py`: 2 passed. v1.0.1 vs frozen v1.0 on 42 runs, 0 differing, max |Δξ| = 0 (`results_precision/pytest.log`).

**Pre-registered decision.** Primary 147/0, p = 1.1e-44 ✔. Every guard has 0 support changes (≤ 1% allowed) and L − W = 0 (≤ 2 allowed) ✔. Pytest passes and the v1.1 arm reproduces the stored v1.1 records (2,400/2,400) ✔. **→ v1.2 confirmed.** `eqdisc.discover` now defaults to v1.2 (`__version__ == "1.2.0"`). `version="1.1"` and `version="1.0.1"` still give the earlier versions. Frozen md5s: `results_precision/frozen.md5`.

## B. Library conditioning and the dysts failures

### B1. Conditioning vs failure (all 77 in-library dysts systems; fresh clean trajectories, not test runs)

`benchmarks/precision_b_dev.py cond`, `results_precision/b_cond.json`. The metric is the 2-norm condition number of
the library matrix in the model's own scaling u = x / rms(x), N = 2000, columns equilibrated. Failure rate is from
the published dysts test results (`results_dysts/test_final.json`); it is used for correlation only.

| System (worst 10 by test exact) | Test exact | κ(library) | κ, centred basis | κ, true support | Offset \|mean\|/std |
| --- | --- | --- | --- | --- | --- |
| AtmosphericRegime | 0.000 | 5.5e2 | 2.0e2 | 11 | 1.7 |
| HyperYan | 0.175 | 3.9e2 | 3.8e2 | 10 | 1.9 |
| IsothermalChemical | 0.225 | **1.1e9** | 2.6e5 | 9.0e2 | **18.6** |
| GuckenheimerHolmes | 0.338 | 16 | 15 | 4.4 | 0.1 |
| QiChen | 0.487 | 3.7e3 | 92 | 25 | 7.0 |
| ChenLee | 0.500 | 1.2e3 | 1.6e2 | 3.0 | 2.7 |
| ZhouChen | 0.613 | 47 | 46 | 1.5 | 0.2 |
| HyperQi | 0.625 | 8.2 | 7.2 | 1.8 | 0.3 |
| Bouali | 0.700 | 1.8e2 | 66 | 4.6 | 2.1 |
| HyperBao | 0.713 | 2.8e2 | 1.9e2 | 8.4 | 3.7 |
| *Lorenz (z ≈ 24, control)* | 0.863 | 1.3e3 | 4.8e2 | 16 | 2.8 |
| *All 77: range* | | 8 – 1.6e4 (one 1.1e9) | | | |

**Spearman ρ with test exact.** κ(library): **+0.02 (p = 0.89)**; κ raw: −0.07 (p = 0.53); κ, centred: +0.07;
κ, true support: −0.31 (p = 0.006); offset: −0.31 (p = 0.007).
- **The full-library condition number is unrelated to failure.** AtmosphericRegime (0%) has κ = 550; SprottO
  (100%) has 1.1e4.
- **The weak correlations with offset and true-support κ** are what one expects from statistical collinearity
  (nearly collinear true terms are hard to tell apart from data). Whether rounding plays a part is what B2 tests.
- **Lorenz** has κ = 1.3e3: not ill-conditioned in float64 terms.

### B2. Numerics-only reruns on fresh dev instances

`dysts_generate.py --seed 20261777 --seeds 3` (new seed; sha256 in the manifest) for the 10 worst systems and 6
controls: 16 × 2 N × 3 seeds × 4 η = **384 runs**. Arms (`benchmarks/precision_numerics.py`):
- **v11.** The public `discover`, instrumented (behaviour unchanged).
- **hard.** Every subset RSS from Householder QR of the normalised columns, with no Gram matrix, no b'b − c'g
  cancellation and no ridge. The profiled quadratic is factored by eigendecomposition (clip at 1e-15 λ_max)
  instead of the 1e-10·trace/len ridge. This applies to the joint-stage proposals and the weak-form starts.
- **qr.** hard, plus a square-root construction of the profiled quadratic: the states are profiled out row by row,
  M = J_c − J_u X, and A = R comes from one QR of [M | r̃]. H = A'A is never formed, so its float64 error is
  ~ε·κ instead of ~ε·κ² (κ = ‖J_c‖/σ_min(M)). This is the "orthogonalised internal basis" option. It acts on the
  same monomial model class, and in exact arithmetic the quadratic is identical.
  - Checked against the original on a Lorenz run: H agrees to 2e-11 relative, and subset RSS differ only by the
    original ridge.

Instrumentation of the v11 arm (maximum over a system's 24 runs):

| System | v11 exact | hard | qr | W/L (hard, qr) | cond H | ridge / λ_min(H) | float64 bound ε·max diag A_cc / λ_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AtmosphericRegime | 0.000 | 0.000 | 0.000 | 0/0, 0/0 | 1.9e12 | 8.3 | 4.8e-3 |
| HyperYan | 0.167 | 0.167 | 0.167 | 0/0, 0/0 | 1.8e10 | 0.10 | 2.3e-5 |
| IsothermalChemical | 0.208 | 0.208 | 0.208 | 0/0, 0/0 | **1.1e18** | 2.4e6 | **93** |
| HyperQi | 0.375 | 0.375 | 0.375 | 0/0, 0/0 | 1.3e5 | 5e-7 | 3e-11 |
| GuckenheimerHolmes | 0.375 | 0.375 | 0.375 | 0/0, 0/0 | 9.6e13 | 2.5e2 | 3.8e-2 |
| ChenLee | 0.500 | 0.500 | 0.500 | 0/0, 0/0 | 6.7e11 | 1.9 | 2.3e-3 |
| QiChen | 0.500 | 0.500 | 0.500 | 0/0, 0/0 | 1.6e12 | 6.6 | 1.2e-3 |
| ZhouChen | 0.625 | 0.625 | 0.625 | 0/0, 0/0 | 7.9e14 | 2.2e3 | 4.8e-2 |
| HyperBao | 0.667 | 0.667 | 0.667 | 0/0, 0/0 | 2.6e7 | 1e-4 | 8e-9 |
| Bouali | 0.750 | 0.750 | 0.750 | 0/0, 0/0 | 3.5e7 | 1e-4 | 1e-8 |
| Lorenz / HyperLu / LorenzStenflo / Arneodo / Halvorsen / SprottA (controls) | 0.92–1.00 | same | same | 0/0 | 1e4–1.2e10 | ≤ 0.04 | ≤ 3e-6 |

**The numerical weak spots are real.**
- On 6 of the 10 failing systems, the profiled-quadratic matrix H is ill-conditioned enough that the 1e-10 ridge
  exceeds its smallest eigenvalue (cond H up to 1e18).
- For IsothermalChemical (offset 18.6), the float64 forward-error bound of H exceeds λ_min ~100-fold. Its smallest
  direction is pure rounding.
- In the subset search the b'b cancellation is harmless: ε·b'b ≤ 2e-9, against a penalty of ~8.
- The normal-equation and QR subset RSS never reorder the winners.

**But removing them changes nothing.**
- hard and qr choose the same support as v1.1 in **383 of 384** runs; the one change is a ChenLee run, the same
  one in both arms.
- The exact outcome changes in **0** runs (W/L 0/0 for both). The run-level coefficients differ in 180 runs, so
  the arms really did compute differently: qr took 2× the time.
- The poorly determined directions of H are directions the data cannot resolve. Whether they are computed
  accurately or not, they do not decide which terms enter.

**Conclusion for B.** The dysts tail is not a floating-point artefact. The dev failure rates match the test rates
(AtmosphericRegime 0.00, HyperYan 0.17, IsothermalChemical 0.21), and their profile is statistical:
- **AtmosphericRegime:** 10 extra and 7.8 missed terms per run. It is stiff (|λ|dt = 3.5) and its smallest term is
  0.6% of its equation.
- **HyperQi and ChenLee:** extra terms.
- **GuckenheimerHolmes, IsothermalChemical, QiChen, ZhouChen, HyperYan:** missed small terms (minimum relative
  contribution 1–3%), or a large offset that makes true terms nearly collinear.

Following the plan, no fix is designed, nothing is run on the dysts test archive, and the failures are reported as
genuine limits.

## What changed in the code

- `eqdisc/joint_discover.py`, `joint_colored.py`, `joint_nd.py`: a new keyword `disc_floor=False`. With the
  default no new code runs, so earlier versions are bit-identical. The guards reproduce 2,400 stored v1.1 records
  exactly.
- `eqdisc/final_model.py`:
  - `VERSIONS["1.2"]` and `discover(..., disc_floor=None, version=...)`;
  - `DEFAULTS["disc_floor"]`, set to True after the confirmed test;
  - the tie-sort crash fix (`80993d1`).
- Benchmarks:
  - `final_bench.py` arms `final_v11` and `final_v12`, and records σ and the floor;
  - `odebench_bench.py --version`;
  - `gen_variants.py --etas`.

## Caveats

- **Range of the floor.** It targets the latent-grid discretisation error only. Errors in the data themselves are
  not modelled: integration error in ODEBench, quantisation in real data. On ODEBench the floor was active in
  18/63 σ = 0 runs, though the exact rate still rose 39% → 87%.
- **Where the floor changes things on noisy data.** It is active on 2–9% of runs (mostly η = 1%, N = 2000) and
  changes coefficients there, but no support changed in 2,800 dev and 2,400 sealed noisy runs. The bound is
  conservative (it ignores how far the states can absorb τ), so at very low but non-zero noise with long series it
  may slightly over-regularise. That regime between η = 0 and η = 1% was not tested.
- **Ambiguity flag and confidences** were not re-calibrated for the inflated σ on noise-free data.
- **Process deviation.** `job_test.sh` gained one line after its md5 was recorded in the pre-registration: a wait
  for the B2 dev job to finish, so the two did not share cores. Nothing else changed.
