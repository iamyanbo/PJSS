# Model-discrepancy-aware joint discovery: PAUSED — work in progress (2026-10-27)

**Status: paused on request so the Spark can be used for other work. No results yet: no dev numbers, no
pre-registration and no test run.** Nothing here changes a frozen version. v1.1 is still the default, and
`tests/test_equivalence.py` is unaffected because no frozen module was edited.

## History

- The task began as a misspecification detector plus fallback. Before any code or run, it was redirected to a
  more principled approach: model error as part of the model (weak-constraint 4D-Var / Kennedy–O'Hagan
  discrepancy), followed by library growth (Phase 2). No detector code was written.
- Base: the latest tagged default, **v1.1** (HEAD 80993d1, including the v1.1.1 pool-sort crash fix). There was
  no v1.2 tag at the time. The precision-audit agent's `disc_floor` changes were uncommitted and were not used.

## What exists

### `eqdisc/discrepancy.py` (new module; written, not yet run or tested)

- **Model error per equation.** `WCSolver` is `joint_nd.Solver` with a per-equation stiffness vector ρ_i ≥ ρ0 = 0.03.
  - The model residual of equation i on the latent sub-grid is read as process noise η_i ~ N(0, (ρ_i σ_i)²).
  - With every ρ_i = ρ0, the objective is exactly the frozen one.
  - The library comes from `PolyLib`. `PolyLib(3,3)` has the same term order as `core.NAMES` (checked on the Spark).
- **Evidence and the per-equation test.** `neg2_log_evidence` is a Laplace-approximate −2 log p(y | support, ρ),
  with the states and the coefficients integrated out:
  - **Formula:** cost + logdet H_uu (banded Cholesky) + logdet S_cc (Schur complement) + 2(nf−1) Σ log ρ_i.
  - **`calibrate_rho`** searches each ρ_i over a grid of ρ0 × {1, 3, …, 3000}.
  - **Per-equation statistic:** Λ_i = ev(ρ_i = ρ0) − ev(ρ̂). This is the candidate "is equation i in the
    library?" test.
- **`search`.** The joint_nd proposal, refit and selection loop, run with the ρ vector. It also takes the
  final_model prior weights. The stiff final refit applies only to equations at ρ0; relaxed equations keep their ρ_i.
- **Intended wrapper (not written yet).** Run v1.1 → compute Λ_i at its answer → if max Λ_i > τ, alternate
  `search` and `calibrate_rho` (2 outer iterations); otherwise return v1.1 unchanged. Unchanged-by-construction
  is what makes the in-library guards ~0.

### `benchmarks/dc_generate.py` (simulation-only misspecification archive generator)

- **Misspecified systems** (at least one equation outside the library that `discover` would use):
  - pendulum, Duffing and van der Pol: D = 2, poly2;
  - double pendulum, coupled pendula and coupled double-well Duffing: D = 4, poly2;
  - Aizawa with its x³z term, Lorenz with a saturated (rational) xy term, and Thomas (sin): D = 3, poly3;
  - Moore–Spiegel and Arneodo: D = 3 with the truncated poly2 library.
- **Matched in-library controls:**
  - linear and quadratic oscillators and Lotka–Volterra (D = 2);
  - Hénon–Heiles, coupled linear oscillators and hyperchaotic Rössler (D = 4);
  - Aizawa without the degree-4 term, and Lorenz (D = 3);
  - a Sprott jerk and Rössler in poly2 (D = 3).
- **What each run records:** a per-equation `inlib` flag, true coefficients for the expressible equations, and a
  600-sample clean continuation for forecast valid time.
- **Design:** N ∈ {500, 2000}, η ∈ {1, 5, 10}%, 2 seeds per cell; the test set uses different parameters.
- **Seeds:** 2026102711 (dev) and 2026102793 (test). A grep found neither in the repo except in this file.
- **Not yet produced.** The one dev generation attempt (Spark, 1 core) hit its 15-minute timeout before writing
  anything. The likely cause is the long rtol = 1e-10 probe and burn-in runs, or a stiff system (van der Pol,
  Thomas). No archive exists and no sha256 is in `data/MANIFEST.sha256`.

## Next steps when resumed

1. Make generation fast:
   - shorter probe and burn-in, rtol 1e-9;
   - time each system with `--only`;
   - generate dev, add its sha256 to the manifest, and keep the test archive ungenerated until pre-registration.
2. Smoke-test `discrepancy.py` on a few dev runs: double pendulum, pendulum, Lorenz, Aizawa / Aizawa0.
3. Dev study:
   - distributions of Λ_i on in-library equations vs misspecified equations (per-equation AUC);
   - false-flag rate on the existing dev archives: `instances` (white), `instances_colored`,
     `instances_student`, `nd_dev`;
   - a known risk: midpoint discretisation error at low noise and AR(1) noise may inflate Λ in-library. The
     policy for AR(1)-routed runs (skip, or an AR(1) likelihood) is still to be decided on dev;
   - effect of the ρ-relaxed refit on exact recovery of the in-library equations and on forecast valid time,
     against v1.1 and per-equation weak best-subset.
4. Freeze τ and the wrapper as a new version on the latest tag, then pre-register. Then run the single test:
   the fresh `dc_test`, plus guards on the main test archives, plus the pendulum real data (secondary,
   non-independent).
5. Phase 2(a), implicit or rational growth g(x)·x′_i = f(x), only if Phase 1 works.

Spark working copy: `~/eqdisc_dc`. It is the committed HEAD plus these files, and `data` is a symlink to
`~/eqdisc_v3/data`. No jobs are running there.
