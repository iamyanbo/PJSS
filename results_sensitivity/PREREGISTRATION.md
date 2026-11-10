# Pre-registration: one-at-a-time sensitivity of the fixed constants (v1.2)

Written and committed 2026-10-27, before any sweep result exists. Only the plumbing check
(`identity_check.json`, default answers unchanged) and the sample file exist at commit time.

## Question

The paper calls `eqdisc.discover` (v1.2, `final_model.discover` with its frozen defaults) "tuning-free". Its fixed
constants were set on development data (Lorenz / Chen / Rössler and the dev archives). A reviewer asks whether
the results depend on those choices. We halve and double each constant, one at a time, on systems that were
never used in development, and compare with the default.

## Sample (fixed: `results_sensitivity/sample.json`)

* Archive `data/dysts_test.npz` (sidecar `data/dysts_test.json`, sha256 in `data/MANIFEST.sha256`; 77 dysts
  systems x noise {1, 5, 10, 20 %} x n {500, 2000} x 10 seeds).
* Systems: the 49 whose sidecar `overlap` is None, the "new systems" group of `benchmarks/dysts_report.py`
  (not identical to, a reparametrisation of, a support twin of, or a family member of any system used in
  development): 35 with D = 3 (20-term degree-3 library), 13 with D = 4 and 1 with D = 6 (joint_nd, degree-2
  library).
* 300 runs, seed 20261027 (`sensitivity_bench.py sample`): every one of the 49 x 4 (system, noise level)
  cells gets one run; 104 cells drawn without replacement get a second; inside a cell the run (length, seed) is
  uniform; the order is then shuffled. Result: 211 runs D = 3, 85 D = 4, 4 D = 6; 146 with n = 2000;
  noise 1 % / 5 % / 10 % / 20 %: 72 / 78 / 77 / 73; 4-8 runs per system.
* Every arm runs on exactly these 300 runs.

## Arms (19: default + 9 constants x 2)

| constant (where it lives) | frozen | x0.5 arm | x2 arm |
|---|---|---|---|
| complexity prior strength `degw` (final_model.DEFAULTS) | 0.2 | 0.1 | 0.4 |
| prior centre `c` (DEFAULTS["centre"]; poly3 and nd) | 1.5 | 0.75 | 3.0 |
| AR(1) switch threshold (DEFAULTS["ar_threshold"]) | 0.6 | 0.45 | 0.75 (as specified, not x0.5 / x2) |
| AR(1) penalty exponent `pen_exp` (DEFAULTS) | 1/3 | 1/6 | 2/3 |
| Huber constant, white path (`huber=2.0` of joint_discover / joint_nd and of the AR(1) pre-estimate `estimate_ar`; new kwarg `huber_white`) | 2 | 1 | 4 |
| `rho` (joint_discover / joint_colored / joint_nd and the AR(1) pre-estimate): the model-error scale; the dynamics residual is weighted 1/(rho sigma), i.e. how stiffly the latent states must follow the candidate ODE relative to the noise | 0.03 | 0.015 | 0.06 |
| `kmax` (same modules): maximum number of terms per equation in the proposal subset search | 5 | 3 (2.5 rounded up) | 10 |
| `finalists` (same modules): best-ranked proposals (by the profiled quadratic score) refitted per round | 10 | 5 | 20 |
| Savitzky-Golay window of the initial latent states (`_initial_states` of joint_discover, also used by joint_colored and estimate_ar, and of joint_nd; new module constant `SG_WINDOW`, kwarg `sg_window`) | 21 | 11 (10.5 -> nearest odd >= 5) | 41 (42 -> nearest odd; 41 and 43 tie, the smaller taken) |

Notes fixed in advance:
* `kmax = 10` on D = 3: the exhaustive proposal search (`flow_discovery._top_subsets`) refuses more than 250 000
  subsets, i.e. kmax > 7 on the 20-term library. That arm therefore uses the method's own large-library route,
  beam-search proposals (`joint_biglib.top_subsets`, width 60, as `final_model` uses for poly4 / poly5). joint_nd
  already switches to beam search by itself.
* The Huber arm changes the white-path constant everywhere it is 2 in v1.2 (white joint fit, joint_nd fit, AR(1)
  pre-estimate); the AR(1) likelihood keeps v1.2's Gaussian loss (`huber_ar1 = inf`).
* `rho` also changes the AR(1) pre-estimate's rho and the v1.2 discretisation floor (which uses the same rho);
  `final_rho = 0.001` (the stiff coefficient-only refit after selection) is not part of this constant and stays.
* The AR(1) threshold and pen_exp arms can only act on D = 3 runs (joint_nd has no AR(1) path), and pen_exp only on
  runs routed to the AR(1) model. On this white-noise archive few runs are expected near the switch, so these two
  constants are tested weakly here; this is a limitation, not a result.
* Copies: where a constant provably cannot act on a run (threshold arm: same routing decision as the default, since
  the AR estimate does not depend on the threshold, or D != 3; pen_exp arm: default routed white, or D != 3) the
  default record is copied instead of recomputed. The first 3 copy-eligible runs of each such arm are recomputed
  anyway and must be bit-identical (else copying is switched off for that arm). This changes compute, not results.
* Constants not covered (out of scope, listed for honesty): `final_rho` 0.001, `rounds` 2, `keep` 2, the latent
  sub-step rule (256 / 64 points per cycle, cap 12), held-out every third sample, the sigma-growth factor 1.1,
  the weight floor 0.1 of the prior, the ambiguity gap 2, the weak-form seed window (`half_width` 20) and the
  flow seeds' own constants, the AR clip 0.97, the v1.2 floor's one-term penalty.

## Plumbing check (before anything else)

Making the constants overridable must not change the default. `benchmarks/sensitivity_identity.py` compares, on 10
fixed runs (4 dysts D = 3 n = 500, 2 dysts D = 4 n = 500, 2 dysts D = 3 n = 2000, 2 sealed coloured AR(1) runs that
take the AR(1) route), the committed code (git HEAD e09e5bd), the new code with a default call, and the new code
with every swept constant passed explicitly at its frozen value: xi (np.array_equal), gap, flag and noise model must
be identical. Result in `results_sensitivity/identity_check.json`. The existing pytest suite is also run (light
tests locally; full suite including `test_equivalence.py` on the Spark, core 13, before the sweep; log in
`results_sensitivity/pytest.log`).

## Metrics

* Primary: exact support recovery (`core.metrics` "exact": the estimated support equals the true support in every
  equation). Errors and timeouts (wall limit 900 s per call) count as not exact.
* Per perturbed arm vs the default arm, on the same runs: exact rate, difference in percentage points, paired W/L
  (W = arm exact and default not; L = the reverse), two-sided exact sign test on W vs L, Holm correction over the 18
  arms.
* Secondary: mean support F1 (errors / timeouts F1 = 0); median relative coefficient error
  ||xi - xi_true||_F / ||xi_true||_F on the runs exact in both the arm and the default (arm vs default on the same
  runs). Subgroups (D = 3 / D >= 4, n, noise level) are descriptive only.

## Claim rule

* **Robust**: every one of the 18 perturbed arms has an exact rate within 5.0 percentage points of the default
  (|diff| <= 5.0, either direction). Then the paper may keep "no per-problem tuning" with this sweep as support,
  stating the range tested (x0.5 to x2 of each constant, one at a time).
* **Not robust**: otherwise. The arms outside +-5 points are named, the word "tuning-free" is replaced by "fixed
  constants set on development data", and the sweep is reported as a sensitivity table with the failing constants
  called out (an arm better than the default by > 5 points also counts as outside: it means the default is not at
  a flat spot).
* Sign tests with Holm p < 0.05 are reported as significant differences in either case; they do not change the
  verdict.
* Completeness: arms are run interleaved per run (default first), so every arm always covers the same prefix of the
  random order. If the sweep is stopped early, the analysis uses that common prefix; with fewer than 200 common
  runs no verdict is stated.
* Not allowed after seeing results: changing the sample, the arm values, the 5-point margin, or dropping arms.

## Compute

Spark, one worker on core 13 only (`taskset -c 13`), started only after `results_h2h/ours_short.json` exists.
Estimate: 300 runs at ~7-8.5 s (v1.0 timing on this sample from `results_dysts/test_final.json`, 0.59 h per arm,
+20 % for v1.2) x 19 arms, minus copied AR-arm records, plus extra cost of kmax 10 / finalists 20: about 11-13 h,
under the 16 h budget, so the full 300 runs are kept. Output per arm: `results_sensitivity/arms/<arm>.json`,
rewritten atomically after each run (resumable). Report: `benchmarks/sensitivity_report.py` ->
`docs/reports/sensitivity.md`.
