# ODEBench with the frozen final model: pre-registration

Written 2026-10-26 ~00:25 EDT, before the main runs (committed 00:32). One smoke test had been run before this file was written:
systems 1, 4, 26, 56, 63 at σ = 0 and 0.03, fit on IC 0, run only to check the code. It showed that the frozen
model raises an exception on every 1-D system (see "Known failure" below). Nothing was changed because of it.

## Benchmark

- **Source.** ODEBench, from d'Ascoli et al., *ODEFormer: Symbolic Regression of Dynamical Systems with
  Transformers* (ICLR 2024). Files are from github.com/sdascoli/odeformer, commit `c9193012ad07a97186290b98d8290d1a177f4609`
  (2024-08-15), `odeformer/odebench/strogatz_extended.json`. The link `facebookresearch/odeformer` returns 404.
- **Local copy.** `data/odebench/strogatz_extended.json`, md5 `4adf1895b211c8decf02dea1688b044f`.
- **Systems.** 63 ODEs: 23 with D = 1, 28 with D = 2, 10 with D = 3 and 2 with D = 4. Each has one parameter set
  and two initial conditions (ICs).
- **Trajectories.** The benchmark's own solutions: 512 points on t ∈ [0, 10] (dt = 10/511), integrated with
  LSODA at rtol 1e-5 and atol 1e-7.

## Protocol

- **Model.** `eqdisc.discover(y, dt)` with all defaults, exactly as frozen. Nothing was tuned, per system or
  globally.
- **Library choice rule.** The library is chosen by the frozen default `library="auto"`:
  - D = 3: the degree-3 polynomial library (20 terms per equation);
  - D ≠ 3: the degree-2 polynomial library in D variables (`joint_nd`).

  The noise model is chosen by `noise="auto"`. No other library (trig3, poly5) is tried.
- **Fitting.** The model is fitted to IC 0. IC 1 is used only for generalization. This matches ODEFormer, whose
  `evaluate.py` fits the first solution of each parameter set. As a secondary set, every run is repeated with
  the roles of IC 0 and IC 1 swapped.
- **Noise.** ODEFormer's multiplicative model: y = x·(1 + ξ), with ξ ~ N(0, σ²) independent per sample and
  component. Levels are σ ∈ {0, 0.01, 0.02, 0.03, 0.04, 0.05}, the six levels in the paper's figures 4 and 5.
  - Primary: one realization per (system, IC, σ), rep 0.
  - Secondary: reps 1 and 2.
  - Seed = 100000 + 1000·id + 100·fit_ic + 10·σ-index + rep.
- **Subsampling.** ρ = 0 only. The model needs a uniform time grid, so ODEFormer's ρ = 0.5 setting (irregular
  sampling) is not run. It is reported as not supported.

## Library classification

The classification is done mechanically, by sympy expansion of the substituted equations, before any fit.
Output: `results_odebench/classes.json`, md5 `038b05fc15d95d7edb4de179d9e3bb20`.

| Class | Rule |
| --- | --- |
| `in_library` | Every component is a polynomial in the states, of total degree ≤ the default library's degree for that D |
| `poly_high_degree` | Polynomial, but of higher degree (for example cubic terms when D = 2) |
| `non_polynomial` | Anything else: rational, exp, log, trig, non-integer powers, abs |

Counts by dimension:

| D | in_library | poly_high_degree | non_polynomial |
| --- | --- | --- | --- |
| 1 | 8 | 3 | 12 |
| 2 | 6 | 10 | 12 |
| 3 | 8 | 1 | 1 |
| 4 | 1 | 0 | 1 |
| **All** | **23** | **14** | **26** |

## Metrics

All metrics are fixed in advance, in `benchmarks/odebench_bench.py`.

1. **In-library systems.** Exact support recovery: the set of nonzero coefficients equals the support of the
   true expansion. Also reported: F1 on terms and the coefficient error ‖ξ̂ − ξ‖/‖ξ‖.
2. **All systems: ODEFormer's metrics.**
   - **Reconstruction R².** Integrate the discovered ODE (LSODA, same tolerances) from the first noisy
     observation y[0], on the same grid. Compare it with the clean IC-0 trajectory.
   - **Generalization R².** Integrate the discovered ODE from the clean IC 1. Compare it with the clean IC-1
     trajectory.
   - **R² definition.** Variance-weighted, as in sklearn's `r2_score(multioutput="variance_weighted")`.
   - **Failures.** A failed or diverged integration, or an exception in the model, scores R² = 0. This is
     ODEFormer's `r2` convention.
   - **Integration limits.** An integration counts as diverged when |x| > 10⁶·max(1, |y0|), or after more than
     2·10⁵ right-hand-side evaluations, or after more than 60 s.
   - **Headline.** Accuracy, the percentage of runs with R² > 0.9, per σ and averaged over σ. Results are
     reported for all 63 systems and separately for each class.
   - **Sensitivity check.** Reconstruction R² from the clean IC.
3. **Runtime** per fit.

## Comparison with ODEFormer

The ODEFormer paper reports ODEBench accuracy (R² > 0.9) only as box plots over σ (figures 4b and 5), at ρ = 0
and ρ = 0.5. Only approximate values read from those figures at ρ = 0 will be quoted, and they will be labelled
as approximate.

Differences in protocol that will be stated:
- The baselines in that paper had a per-trajectory hyperparameter search and chose the best Pareto candidate by
  accuracy.
- We use the frozen defaults and the single model we return.
- ODEFormer integrates from trajectory[0]. Whether that is the noisy or the clean point is not fully clear from
  the code, so both are reported.

## Known failure, found in the smoke test

Every D = 1 call raises `IndexError` inside `joint_nd.discover`. An empty candidate support gives
`np.array(())`, which is a float array, and it is then used as an index. The frozen model is not modified.
These runs count as failures (R² = 0, not exact). Results are reported both over all 63 systems and over the
40 systems with D ≥ 2.

## Code md5 at registration

```
a957580eeba2a0ef0bab9318e50db13f  benchmarks/odebench_bench.py
c1eae9e73bd7ada22a8a7a03805c2c18  eqdisc/final_model.py
a6d5be4efa081ddb9a3912351eaa1094  eqdisc/joint_nd.py
c4d3d8d04f0d85951c813e7f9711ea7c  eqdisc/joint_discover.py
bd452de5dc22dcb99b88ce74a3669b12  eqdisc/joint_colored.py
2e0276d87d0d5b550c08bb8476100db9  results_odebench/library_exponents.json
```

## Amendment (2026-10-26 ~00:30 EDT, before any main run started; committed 00:32)

`benchmarks/odebench_bench.py` gained two options. Neither changes the default (frozen) path; new md5
`05e0af9312acac4f768f43abf39e334d`.

- `--dims`: filter the systems by dimension.
- `--patch-empty-support`: a **diagnostic, not frozen** arm. It applies `benchmarks/_nd_emptyfix.py`, which
  works around the `joint_nd` empty-support crash. The arm runs on D = 1 and D = 2, fit on IC 0, with 3 reps,
  after the frozen runs. Its output goes to a separate file, `fit_ic0_emptyfix.json`, and it is reported only
  as "what fixing the crash would give".

The headline numbers remain the frozen runs.
