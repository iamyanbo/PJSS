# Partial observation: pre-registration of the single test pass (2026-10-27)

Written and committed after all development and before any method is run on the test archive. The test archive
was generated on the Spark right before this file was written; nothing has been run on it.

## Question

Can `eqdisc.partial_obs.discover` (profiled joint flow with data rows only for observed components, a
fixed latent gauge, derivative/quadratic/surrogate latent restarts, a held-out-misfit score, a sparsifying gauge
search and negligible-term pruning) recover Lorenz, Rössler and Chen equations from 1 or 2 of their 3
coordinates, exactly up to an affine change of the latent coordinates, and does it beat sensible baselines?

## Archive

- `data/po_test.npz`, sha256 `736677f3b44e1e7b6d16e3cdd2cd59e7adf9597f7bb3a89c8a66c04f72f70d4f`
  (also appended to `data/MANIFEST.sha256`), made with
  `benchmarks/po_generate.py --seed 20261702 --seeds 10 --output data/po_test.npz`.
- Seed 20261702 has never been used in this repository (dev used 20261701; both checked against every generator
  and the manifest).
- 60 instances: Lorenz, Rössler, Chen × noise 1% and 5% of each coordinate's std × 10; N = 2000, the
  per-system dt of `eqdisc.core.DT`.
- Every instance is run under all four observation configurations (paired): `xz` (y latent), `xy` (z latent),
  `yz` (x latent), `x` (y and z latent). 240 (instance, configuration) pairs per arm.

## Frozen code

md5 in `results_po/frozen.md5` (partial_obs, the five benchmark scripts, and the core modules they import).
The frozen full-observation model is untouched (no existing file was modified; `tests/test_equivalence.py` is
re-run on the Spark before the test pass).

## Arms (all run with `benchmarks/po_bench.py --data data/po_test.npz --workers 6`)

1. `po` (**primary method**): `eqdisc.partial_obs.discover` with its defaults: gauge "proj", score "val",
   restarts deriv + quad + 2 surrogates, regauge degree 2, prune 0.005, rho 0.03, final rho 0.001.
2. `delay` (baseline): weak χ² best-subset SINDy on derivative (Takens-style) coordinates.
3. `embed` (baseline): the frozen full-observation model (`final_model.discover`) on the same derivative
   coordinates.
4. `dahsi` (baseline): DAHSI-style variational annealing with hard thresholding, reimplemented on the same
   solver; 6 random latent initialisations × λ ∈ {0.2, 0.5, 1, 2, 4}, ρ annealed 1 → 0.03; the same score as `po`.
   (The original uses 500 initialisations and IPOPT; this is a reduced-budget reimplementation.)
5. `po_oracle` (**diagnostic only, not a comparison**): `po` started from the true latent trajectory.

Run order: po, delay, embed, po_oracle, dahsi (arm-major). Analysis set: every pair that finished. If the dahsi
arm cannot finish within 12 h of wall time, its analysis set is the finished pairs, and this is reported.

## Endpoints

- **Primary:** exact recovery up to an affine latent gauge (`po_eval.gauge_match` → `exact_gauge`: support
  equals the support of the true system transformed by some well-conditioned x_L = A ξ_L + B x_O + c, and the
  estimated latent explains the true latent with R² > 0.9).
- **Primary comparisons:** `po` vs each of `delay`, `embed`, `dahsi`, paired over the 240 pairs: wins/losses and
  the two-sided exact sign test, Holm-corrected over the three baselines at α = 0.05.
- **Secondary (descriptive):** per configuration, per system and per noise exact rates; strict rate (exact and
  k ≤ k_true, i.e. the minimal representative); median coefficient error over exact runs; median forecast valid
  time (Lyapunov times) from the estimated state at the middle sample; median seconds per run; `po_oracle`
  rates (an upper bound from initialisation).

## Expectations from dev (dev archive `data/po_dev.npz`, seed 20261701, 30 instances × 4 configurations)

Final dev pass (`results_po/dev_G.jsonl`, frozen code): `po` 79/120 exact (xz 30/30, xy 26/30, yz 22/30,
x 1/30; strict 76). `delay` 1/120, `embed` 1/120 (dev F). `dahsi` 1/11 in the part of the dev pass that ran
(≈ 570 s per run). The x-only configuration (two latents) essentially fails without the true latent init.

Dev result files in `results_po/` have the per-run state arrays (`u`) stripped to keep them small.
Development history, including negative results and bugs, is written up in `docs/reports/partial_obs.md`.
