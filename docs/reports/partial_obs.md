# Partial observation: latent states in the profiled joint flow (2026-10-26/27)

**Status: final.**
- The pre-registration was committed in `09aef3e` before the test pass.
- The single test pass ran on the Spark (6 single-thread workers) from about 05:00 to 13:30 on 2026-10-27, and
  all 1200 runs finished.

## Summary

- **Primary result (fresh test archive, seed 20261702, 240 instance × configuration pairs):** the new method
  `po` recovers the equations exactly up to an affine latent gauge in **146/240 (61%; 95% CI 55–67%)**.
  - The three pre-registered baselines score: DAHSI-style annealing 21/240, SINDy on derivative coordinates
    2/240, and the frozen full-observation model on derivative coordinates 0/240.
  - Paired W/L for `po` vs each: **130/5, 144/0, 146/0**, with sign-test p = 2e-32, 9e-44 and 2e-44 (all
    significant after Holm).
- **By configuration:**
  - **One latent, where an observed derivative is affine in it:** works. xz is 60/60.
  - **One latent otherwise:** works partially. xy is 43/60 and yz 36/60.
  - **Two latents (x only):** fails, at 7/60.
- **The bottleneck is initialisation, not identifiability.**
  - Started from the true latent trajectory (diagnostic only), the same method gets 232/240, including 53/60
    on x-only.
  - `po` loses 88 pairs to that oracle and wins 2.
- **What made it work on dev:**
  - a projected scale gauge;
  - derivative and quadratic latent proxies as restarts;
  - a held-out-misfit selection score;
  - a sparsifying gauge search that moves a denser gauge-equivalent model to its minimal form (+15/0 on dev);
  - negligible-term pruning.
- **Cost:** about 3.5 min per run on one core, against about 8 min for the DAHSI-style baseline.

Files:
- method: `eqdisc/partial_obs.py` (new module; the frozen model is untouched and `tests/test_equivalence.py`
  still passes, 42 runs, 0 differing);
- benchmarks: `benchmarks/po_generate.py`, `po_bench.py`, `po_eval.py` (gauge-aware scoring),
  `po_baselines.py`, `po_report.py`;
- tests: `tests/test_partial_obs.py` (3 tests, about 1 min);
- pre-registration (committed before the test pass): `results_po/PREREGISTRATION.md`, `results_po/frozen.md5`;
- results: `results_po/dev_{D,E,F,G}.jsonl` (state arrays stripped), `results_po/test.jsonl`,
  `results_po/test_tables.md`.

## Problem and success criterion

Only some components of the state are measured: Lorenz, Rössler or Chen with x and z observed (`xz`, y latent),
`xy` (z latent), `yz` (x latent), or `x` alone (y and z latent). N = 2000 samples at the per-system dt, noise 1%
or 5% of each coordinate's std.

A latent coordinate is only defined up to a change of variables. For an affine change
x_L = A ξ_L + B x_O + c (A invertible), a cubic polynomial library stays closed, so the true system has an exact
polynomial form in (x_O, ξ_L), whose support can differ from the original one (offsets add constants, a shear
adds terms). The pre-registered success criterion is therefore **exact up to an affine latent gauge**
(`po_eval.gauge_match`):
- the discovered support equals the support of the true system transformed by some well-conditioned (A, B, c)
  (found by a two-stage least-squares search over the gauge, with 1e-6 relative tolerances and
  cond(A / rms x_L) < 1e4);
- and the estimated latent trajectory explains the true one (affine regression R² > 0.9 per latent).

"Strict" adds k ≤ k_true (the minimal representative, not a denser gauge-equivalent form).

The gauge can also be non-affine. For Lorenz or Chen with z latent, z → z + b x² keeps the library closed
(dz/dt picks up x² and xy terms, dy/dt an x³ term). The evaluator does **not** accept such forms as exact; they
show up as failures with a low latent R².

## Literature

- **DAHSI** (Ribera, Payne, Ward, Kutz, Brunton & Bramburger, Chaos 32, 063101, 2022; arXiv 2105.10068). Lorenz
  with x and z observed and y hidden, N = 501. Variational annealing of the model weight (Rf = Rf0 α^β) with a
  hard threshold λ after every step, 500 random latent initialisations, coefficients started at 0, a λ sweep, and
  model selection by AIC/BIC on validation data. About 80% of initialisations recover the model at noise
  σ = 0.01, ≤ 10% at σ = 0.1; about 4 h per sweep of initialisations. This is the closest prior method, and the
  `dahsi` baseline reimplements it (reduced budget, see below).
- **Delay / derivative embeddings with SINDy.** Bakarji, Champion, Kutz & Brunton (Proc. R. Soc. A 2023) learn a
  coordinate change from time-delay coordinates of one variable with an autoencoder and SINDy in the latent
  space (Lorenz, Rössler from x alone). They recover a dynamically similar model, not the original equations, and
  report no exact-support rate. Champion et al. (PNAS 2019) is the full-state autoencoder version. The baselines
  here use the explicit, interpretable variant: SINDy on [observed, d/dt observed (, d²/dt²)] coordinates, which
  under the affine gauge is a valid latent coordinate whenever the derivative is affine in the latent (e.g.
  Lorenz dx/dt = σ(y − x)).
- **Higher-order ODEs in one variable** (Somacal et al., PRE 2022): eliminate the latent and fit an n-th order
  ODE for the observed variable. The result is generally rational in the derivatives, not the original
  polynomial system.
- **Lu, Ariño & Soljačić** (arXiv 2107.10879): a neural encoder for the hidden states with sparse dynamics.
- **ODR-BINDy** (arXiv 2507.23426): errors-in-variables Bayesian SINDy on fully observed states; no hidden-state
  treatment.
- **DEDi** (Meissner & Glasner, arXiv 2406.06707): joint state/coefficient estimation with a `state_mask` for
  missing data. Its examples are randomly or contiguously missing samples (Lorenz, Van der Pol), not a
  permanently hidden component. A fully masked component is expressible, but nothing handles the gauge.
- **Bayesian / MCMC latent approaches** (e.g. the latent-SDE and Gaussian-process state-space literature) treat
  hidden states as random variables. They are much more expensive per model. In a non-exhaustive search I
  found none that reports exact-support recovery rates on these benchmarks, so none is used as a baseline.

## Method (`eqdisc/partial_obs.py`)

It uses the same objective, banded Levenberg–Marquardt solver, profiled-quadratic support search and
degree-weighted prior BIC as the full-observation joint flow, with these changes.

1. **Data rows only for observed components.** Latent states are constrained by the dynamics residuals alone,
   with σ_lat = mean observed σ in scaled units.
2. **Gauge fixing (scale).** Offsets and mixing change the support, so the sparse search chooses among them. The
   scale of a latent is a continuous symmetry of every support, though: without fixing it, shrinking the latent
   shrinks its own dynamics residual. The part of each latent orthogonal to span{1, observed states} is pinned
   to unit RMS by a stiff penalty. That is a rank-one Gauss–Newton term per latent, handled exactly with
   Woodbury on the banded Cholesky, so fits stay O(N).
3. **Latent restarts.** Each restart supplies a latent initialisation:
   - derivative proxies (Savitzky–Golay d/dt of each observed component; d/dt and d²/dt² for two latents);
   - quadratic products of observed components (x², xy; for two latents, mixed with derivatives);
   - two phase-randomised surrogates of an observed signal.

   Start coefficients come from weak χ² best-subset regression on the pseudo-full trajectory. The support search
   runs from every restart, and the best score over all restarts wins.
4. **Selection score "val":** held-out misfit + dynamics misfit + φ · prior BIC. The training misfit is left out,
   because a dense latent model can buy training fit by letting the latent chase noise.
5. **Sparsifying gauge search (regauge).** The selected model is mapped exactly through latent changes
   ξ_L → s(ξ_L + A_off ξ_L + B·mono_{≤2}(x_O)), which include mixing, shear, offset and polynomial shifts. Powell
   minimises the summed relative term contributions of the exactly transformed model; a least-squares polish then
   drives the small terms to zero. The resulting form is refitted jointly and searched again, and kept if it
   scores better.
6. **Negligible-term pruning.** Terms that contribute < 0.5% of their equation's RMS right-hand side are dropped
   and the model refitted. On dev, spurious extra terms were < 0.3% and true terms > 2%.

Truth is never used by the method. The `latent_init` argument (oracle latent trajectory) exists only for the
dev diagnostic `po_oracle`.

## Development (dev archive `data/po_dev.npz`, seed 20261701; 30 instances × 4 configurations)

All dev runs were on the Spark (6 single-thread workers), in chronological order. The arms marked "ora" add the
true latent as an extra restart. That is a diagnostic used to separate initialisation failures from
identifiability or selection failures.

| Stage | Change | Result |
|---|---|---|
| A | raw-RMS gauge, oracle init | Lorenz xz fails even from the truth. The latent shears to ξ ∝ y + 41.6x, which softens its own equation. |
| A | projected gauge (orthogonal to span{1, x_O}) | Fixes those cases. Oracle arm 112/120 (with the ridge bug below). |
| A | shear-invariant residual model (free regression β of latent on observed residuals) | **Negative.** β → ∞ absorbs the latent dynamics; the data misfit grows 40×. Kept off. |
| B | **bug:** a state ridge 1e-10·mean(diag) in the profiled quadratic | It penalised constant latent shifts by about 200 χ² units at 1% noise and hid the minimal-offset supports. Removed: q(8-term) = q(7-term) after the fix, as theory says. |
| B | **evaluator bug:** a near-singular A accepted a 6-term linear model for Rössler x-only | Added per-equation normalisation, the conditioning guard and the latent-R² check (the degenerate model now has cond 4e4, R² 0.04). |
| C/D | score "sum" (train + model + val) vs "pred" (val scaled by n/n_val) vs latent σ factor; gauge "proj" vs "poly" | With the oracle restart, 72 runs each: gpoly@pred 55, gpoly@sum 54, **gproj@pred 60**. "poly" (pin the part orthogonal to all observed monomials) breaks Chen x-only, so "proj" stays. |
| D | extra-term failures | 9 of the 12 gproj failures had one extra term contributing < 0.3% → negligible-term pruning (applied to the final stiff fit after a first version pruned too early). |
| E | unit test failure: an extra y·z term (contribution ≈ 0.5%) wins under "pred" because n/n_val = 3 triples the held-out noise | New rule "val" (no factor). 71 paired runs: val 42, pred 43 (2/3), so it is chosen on principle (and the unit test). |
| E | **hang:** one run never finished | The forecast evaluator's LSODA ran forever on a runaway model. `valid_time` now integrates in chunks, stops at the first failure, and caps evaluations and state size. |
| F | full dev, no oracle: **po 64/120** (xz 29, xy 19, yz 16, x 0; strict 44); delay 1/120; embed 1/120 | Most remaining failures are gauge-equivalent but denser forms (sheared or polynomially shifted latent), stuck as local optima of the support search. |
| G | + sparsifying gauge search (regauge) | **po 79/120** (xz 30, xy 26, yz 22, x 1; strict 76). Paired vs F: 15 wins, 0 losses. Median 207 s per run. |

Dev G by system and configuration (exact / 10):

| | xz | xy | yz | x |
|---|---|---|---|---|
| Lorenz | 10 | 9 | 6 | 1 |
| Rössler | 10 | 9 | 10 | 0 |
| Chen | 10 | 8 | 6 | 0 |

The `dahsi` baseline was 1/11 in the part of the dev run that finished (about 570 s per run, with dense
selections of k = 9–49), so the rest of its dev run was stopped. Its settings were not tuned.

What the dev runs say about **initialisation**:
- The derivative proxy works whenever an observed derivative is affine in the latent: Lorenz/Chen xz and yz,
  Rössler xz/xy/yz.
- The quadratic proxy (x²) recovers Lorenz/Chen with z latent.
- **Two latents (x only)** essentially never work from real initialisations (1/30). With the oracle init they
  mostly do (dev D: Lorenz x 6/6, Rössler 4/6, Chen 1/6). So this is an initialisation problem; for Chen it is
  identifiability or selection as well.
- ρ-annealing (DAHSI-style homotopy) of the start fits was implemented (`rho_path`), but it was not adopted:
  restarts already gave the basin, and annealing quadruples the cost.
- A homotopy from full observation needs the latent data, so it is only a dev diagnostic, equivalent to the
  oracle init. It was not pursued.

## Test (pre-registered; `results_po/test.jsonl`)

Analysis set: all 240 pairs for every arm (1200 runs).

**Evaluator bug (found only on the test pass).** In 20 `dahsi` runs on the x-only configuration (Lorenz 9,
Chen 11), the evaluator itself crashed. Stage 2 of `gauge_match` uses `least_squares(method="lm")`, which needs
at least as many off-support residuals as gauge parameters, and these near-full models have fewer.
- Those runs are counted as not exact, as the pre-registered runner does. A model that dense is not a minimal
  form.
- Even if all 20 counted as DAHSI wins, `po` vs `dahsi` would be 130/25 (p ≈ 1e-18).

### Table 1: exact recovery up to an affine latent gauge (test, 60 pairs per configuration)

| Arm | xz (y latent) | xy (z latent) | yz (x latent) | x (y, z latent) | All (95% CI) | Strict | Med. coef err | Med. VT (Lyap.) | Med. s |
|---|---|---|---|---|---|---|---|---|---|
| **po** | **60/60** | **43/60** | **36/60** | **7/60** | **146/240 (61%; 55–67%)** | 139 | 0.03% | 4.46 | 205 |
| dahsi | 9/60 | 5/60 | 7/60 | 0/60 | 21/240 (9%; 6–13%) | 2 | 0.03% | 1.58 | 461 |
| delay | 0/60 | 0/60 | 2/60 | 0/60 | 2/240 (1%; 0–3%) | 0 | 5.0% | 0.45 | <1 |
| embed | 0/60 | 0/60 | 0/60 | 0/60 | 0/240 (0%; 0–2%) | 0 | – | 0.12 | 70 |
| po_oracle (diagnostic) | 60/60 | 60/60 | 59/60 | 53/60 | 232/240 (97%; 94–98%) | 228 | 0.03% | 6.24 | 14 |

Key to the columns:
- "Strict": exact and k ≤ k_true.
- "Coef err": the median over exact runs, relative to the true model in the closest gauge.
- "VT": the median forecast valid time over all runs, from the method's own estimated state at the middle
  sample.

### Table 2: by system and configuration (exact / 20)

| Arm | Chen xz | xy | yz | x | Lorenz xz | xy | yz | x | Rössler xz | xy | yz | x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| po | 20 | 13 | 7 | 0 | 20 | 16 | 9 | 6 | 20 | 14 | 20 | 1 |
| dahsi | 7 | 5 | 4 | 0 | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| delay | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| embed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| po_oracle | 20 | 20 | 20 | 14 | 20 | 20 | 19 | 19 | 20 | 20 | 20 | 20 |

By noise, `po` gets 78/120 at 1% and 68/120 at 5%:

| System | 1% | 5% |
|---|---|---|
| Chen | 20/40 | 20/40 |
| Lorenz | 27/40 | 24/40 |
| Rössler | 31/40 | 24/40 |

### Table 3: pre-registered paired comparisons (240 pairs; W/L = pairs only `po` / only the baseline gets exact)

| `po` vs | W/L | Two-sided sign test p | Holm-adjusted |
|---|---|---|---|
| dahsi | 130/5 | 1.7×10⁻³² | 1.7×10⁻³² |
| delay | 144/0 | 9×10⁻⁴⁴ | 1.8×10⁻⁴³ |
| embed | 146/0 | 2.2×10⁻⁴⁴ | 6.6×10⁻⁴⁴ |

(Diagnostic, not a comparison: `po` vs `po_oracle` 2/88.)

By configuration, `po` vs `dahsi` is xz 51/0, xy 38/0, yz 34/5 and x 7/0, so all 5 `dahsi` wins are on yz.

Test against dev: dev G gave 79/120 (66%) and the test gives 146/240 (61%), with the same per-configuration
pattern (xz 100% in both; x 3% on dev, 12% on test).

### About the baselines (read before quoting the margins)

- **`delay` / `embed`.** On noisy data, derivative coordinates are themselves noisy (Savitzky–Golay
  differentiation amplifies noise). The models they give are close but not exact: `delay` has a median
  coefficient error of 5% and usually several small extra terms. Pruning those terms post hoc rescued no dev
  run. So these baselines show that "embed, then use a full-observation method" does not give exact recovery
  here. They do not show that a good delay-embedding method cannot.
- **`dahsi`** is a reduced-budget reimplementation on our solver (6 random initialisations and 5 λ values
  instead of 500 initialisations, IPOPT and a fine λ sweep). It uses our score, not AIC on a validation set.
  - It already costs 2.2× `po` per run (32 core-hours for the arm).
  - Its selections are dense (median k = 20).
  - Its 21 successes are mostly Chen (16), at 1% noise (19).
  - The published DAHSI per-initialisation success (about 80% on Lorenz xz at σ = 0.01, N = 501) uses absolute
    noise 0.01 on unscaled Lorenz, which is far smaller relative noise than 1% of the std. It also uses many
    more initialisations. A faithful DAHSI would do better than this arm, so the margin against it overstates
    the advantage over DAHSI itself.

## Failure modes

Post-hoc classification of the 94 `po` test failures (not pre-registered):

| Configuration | Latent right (R² > 0.9), support denser than minimal | Latent right, support wrong at k ≤ k_true | Latent wrong (R² < 0.9) | Of these, oracle init exact |
|---|---|---|---|---|
| xy | 6 | 3 | 8 | 17/17 |
| yz | 22 | 0 | 2 | 24/24 |
| x | 3 | 2 | 48 | 47/53 |

1. **Two latents (x only): initialisation.**
   - 48 of the 53 failures have a wrong latent, and the oracle init fixes 47 of the 53.
   - The derivative coordinates (d/dt, d²/dt² of x) are affine in y for Lorenz, but z enters only through
     x·z. A proxy that is a rational function of the derivatives would be needed.
   - Chen x-only also fails from the oracle 6/20 times. There it is identifiability or selection as well.
2. **yz on Lorenz and Chen (x latent): dense gauge-equivalent forms the gauge search does not reduce.**
   - The latent is right but small extra terms remain (y·z, y², x·z², …).
   - The regauge family (mixing, shear, offset, quadratic shifts by observed coordinates), or its Powell search,
     does not reach the minimal form.
   - The oracle init gives the minimal form, so it is a basin issue.
3. **xy (z latent): polynomial gauge and wrong basins.**
   - z → z + b·x² keeps the library closed for Lorenz and Chen. The fit can settle on such a representation,
     or on a mixed one, with low affine R².
   - The regauge step fixes many of these (dev xy went from 19/30 to 26/30). The remaining ones include models
     in a non-affine gauge, which the pre-registered affine criterion counts as failures.
   - The oracle init fixes all 17.

Regauging was used in 110 of the 240 test runs (74 of the 146 successes went through it), and pruning in 19.
Exact models forecast for 5.8 Lyapunov times (median), non-exact ones for 3.6.

## Runtime

On the Spark, single thread per run, with the machine shared with other jobs (load about 18):

| Arm | Median s per run (xz / xy / yz / x) | p90 / max s | Core-hours for 240 runs |
|---|---|---|---|
| po | 260 / 182 / 301 / 169 | 360 / 518 | 15.0 |
| po_oracle | 16 / 10 / 15 / 17 | 29 / 50 | 1.0 |
| dahsi | 461 (all configurations) | – | 31.6 |
| embed | 74 / 75 / 85 / 41 | 136 / 216 | 5.1 |
| delay | < 1 | – | < 0.1 |

The test pass took about 8.5 h of wall time on 6 cores. Most of the `po` time goes into about 7 restarts, each
with a support search. The oracle row shows that a single good start costs about 15 s.

## Notes

- The PC was out of commit memory during the session (about 97 GB committed, about 2 GB free, not held by these
  processes). Local single runs hit MemoryError and OpenBLAS allocation errors until OPENBLAS_NUM_THREADS=1, so
  all runs except a few single-run checks were on the Spark.
- Superseded dev records (stages A–C, before the evaluator and ridge fixes) are not included; their conclusions
  are summarised above.
