# v1.2 (discretisation-error noise floor): pre-registration

Written 2026-10-27 and committed **before the test archive `data/sealed_clean.npz` exists**. No v1.2 run has been
made on any test archive, on ODEBench, or on any sealed archive. Every design decision below was made on dev data
of our own generation (`data/dev_clean.npz`, the 2-D dev systems in `benchmarks/precision_a_dev.py`, and the four
noisy dev archives `instances*.npz`, `nd_dev.npz`).

## Question

On noise-free trajectories the model adds tiny spurious terms. The precision audit (`docs/reports/precision_audit.md`,
part A) found on dev that this is **not** a floating-point effect (the estimated noise scale is ~1e-5 of the signal,
11 orders above float64 resolution, and the 1e-6 floor binds only in 2-D), but a **discretisation** effect: the
implicit-midpoint latent grid has a local truncation error τ that is larger than the model-residual scale ρσ the
likelihood assumes, and BIC lets smooth extra terms absorb this deterministic error. Does a noise-scale floor derived
from τ remove the spurious terms on a **fresh** noise-free archive, and does it leave the sealed noisy test archives
untouched?

## Versions compared

| Version | Definition | Code |
| --- | --- | --- |
| **v1.1** | current default | `eqdisc.discover(y, dt)` / `version="1.1"` |
| **v1.2** | v1.1 + `disc_floor=True` in `joint_discover` / `joint_colored` / `joint_nd` | `version="1.2"` |

**The floor (no tuned constant).** After the existing σ (and AR(1)) self-calibration, with the best start's fitted
states u and coefficients C on the latent grid of step h: τ_k = z_k − u_k − h f((u_k + z_k)/2), z_k = RK4 (4 sub-steps)
solution of the fitted model from u_k (`joint_discover.midpoint_lte`). The chi² that τ alone would produce,
Σ_k |τ_k/(ρσ)|², is deterministic approximation error, so it is capped at one term's selection penalty
pen = λ·φ·log(nD), split equally over the D components: σ_i ← max(σ_i, sqrt(D Σ_k τ_ki² / pen)/ρ). If the floor
raises any σ_i, the starts are refitted (as the existing self-calibration does). Everything else is unchanged.
`disc_floor=False` (v1.1 and earlier) is bit-identical to before (no code path changes).

**Dev evidence (all on dev; `docs/reports/precision_audit.md`).**
- Noise-free 3-D dev (`dev_clean.npz`, 80 runs) exact 0.31 → **1.00**; 2-D dev (32 runs) 0.66 → 0.97;
  paired 65/0.
- Noisy dev guards, v1.1 vs v1.2, support changes: white 0/800, coloured 0/800, t₃ 0/800, nd 0/400
  (the floor is active in 47/28/74/28 runs, mostly η = 1%, N = 2000, and changes only coefficients).

Frozen code md5 (v1.2; same files on the Spark):

```
9a9eb72382b605441efccdd99bbe557e  eqdisc/final_model.py
08affac51907736274304d6be31188c1  eqdisc/joint_discover.py
d7b53d5c47719ca39e5fe31011f6d713  eqdisc/joint_colored.py
64b0a26b0fb402299795a74746ddabe5  eqdisc/joint_nd.py
203dd5eb6ba6ed79876253abeff934cc  eqdisc/__init__.py
a49ec45af48bfe5d54e7c154dd69c8d5  benchmarks/final_bench.py      (arms final_v11, final_v12)
7ad14f0074ae133f60f831c67d36b021  benchmarks/odebench_bench.py   (--version)
727541db0f0b6a25b68e36aab8870106  benchmarks/gen_variants.py     (--etas)
7287bf1d4ac9ee04c40091bc404de048  results_precision/job_test.sh
```

## Fresh test archive

- **Generator.** `python benchmarks/gen_variants.py --kind white --etas 0 --seed 20261127 --output
  data/sealed_clean.npz`: the generator of the sealed archives (10 systems × N ∈ {2000, 500} × 10 seeds, random
  start on the attractor), η = 0 → **200 noise-free runs**.
- **Seed.** 20261127 does not appear anywhere in the repository. Independence is also checked after generation:
  no trajectory may equal or share its first sample with any trajectory of any archive in `data/`; if it does, the
  seed is incremented by 1 (reported).
- **Checksum.** sha256 added to `data/MANIFEST.sha256` and committed before any run; `job_test.sh` aborts otherwise.

## Runs (one pass; `results_precision/job_test.sh`, Spark, 18 workers)

1. **Primary.** `final_bench.py --arms final_v11 final_v12` on all 200 runs of `sealed_clean.npz` (shared starts).
2. **Secondary, NOT independent.** ODEBench σ = 0, fit IC 0 (63 systems), `--version 1.1` and `--version 1.2`.
   The spurious-term problem was first seen on ODEBench, so this is labelled non-independent and is not part of the
   decision. Also note ODEBench's own solutions carry LSODA rtol 1e-5 errors, which the floor does not model.
3. **Guards.** Both arms on all runs of `joint_sealed.npz` (800, with the cached starts), `sealed_colored2.npz`
   (800), `sealed_student.npz` (800), `nd_test.npz` (400).
4. **Old versions.** `tests/test_equivalence.py` (v1.0.1 vs frozen v1.0, 42 runs). Also: the `final_v11` arm must
   reproduce the stored v1.1 records bit for bit (ξ) on `joint_sealed` (`results_final_v11/guard_sealed.json`),
   `sealed_student` (`guard_student.json`) and `sealed_colored2` (`test_colored2.json`, arm `final`).

## Metric, test, decision

- Exact support recovery; W/L = runs that only one version gets exactly right; two-sided exact sign test.
- **v1.2 is confirmed (and becomes the default of `eqdisc.discover`) iff all of:**
  1. primary: W > L with p < 0.05;
  2. each guard archive: support changes ≤ 1% of its runs **and** L − W ≤ 2;
  3. pytest passes and `final_v11` reproduces the stored v1.1 records.
- Otherwise v1.2 stays available as `version="1.2"` only and the default stays v1.1. Either way the result is
  reported in `docs/reports/precision_audit.md`.
