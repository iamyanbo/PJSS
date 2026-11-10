# v1.1 (Gaussian loss on the AR(1) path): pre-registration

Written 2026-10-26 ~11:45 EDT and committed (9b88c9d, 11:47) **before the test archive exists**. No v1.1 run on any test archive has
been made. The only v1.1 runs so far are one dev coloured run on the laptop (code check: it reproduces the stored
`ar1_gauss` record of `results_huber_dev/`).

## Question

`docs/reports/huber_dev.md` found on the coloured DEV archive that the Gaussian loss inside the AR(1) likelihood
beats Huber (c = 2): path-specific variant 13/1 (p = 0.002) on the pre-registered subset, 36/2 on all 800 dev
runs; the t₃ and white dev runs never route to AR(1), so they cannot change. Does this replicate on a **fresh**
coloured test archive, and does it leave the white and t₃ test archives untouched?

## Versions compared

| Version | Definition | Code |
| --- | --- | --- |
| **v1.0.1** | frozen v1.0 + the empty-support crash fix (commit `e4b4b9c`); Huber c = 2 everywhere | `eqdisc.discover(y, dt, version="1.0.1")` |
| **v1.1** | v1.0.1, except `huber=inf` (Gaussian) passed to `joint_colored.discover` when the router picks AR(1). The AR(1) pre-estimate (`estimate_ar`) and the white path keep Huber c = 2, so routing is identical to v1.0.1 | `eqdisc.discover(y, dt)` / `version="1.1"`; `DEFAULTS["huber_ar1"] = inf` |

v1.1 is the derived `path_gauss` arm of the dev diagnostic, now a code path. Nothing else changes (prior, threshold,
λ(a), starts, libraries). Frozen code md5 (v1.1):

```
09295bbb76fb549ccead23d528f21374  eqdisc/final_model.py
24ba0897863aba1a67af92cd2d84df10  eqdisc/joint_colored.py      (= v1.0.1)
e7f563ae2080e3229097f9008acc90cd  eqdisc/joint_discover.py     (= v1.0.1)
4bc34d937c3b30b4b7f4c499120d760e  eqdisc/joint_nd.py           (= v1.0.1)
203dd5eb6ba6ed79876253abeff934cc  eqdisc/__init__.py
631ca07a77c90fe83f8e38d895f1df99  benchmarks/final_bench.py    (arms "final" = v1.1, "final_v101" = v1.0.1)
9dc794c551b8ce2fab33b45fa461cfa7  benchmarks/v11_report.py     (all tables and tests below)
df02fdeb05399fb3508fe7f8f2ff8b0b  benchmarks/gen_variants.py
40ee44a8d3525611b492b052ca95bcfc  results_final_v11/job.sh
```

All other `eqdisc/*.py` files are unchanged from v1.0 (md5 in `results_final/frozen.md5` / `results_v101/frozen.md5`).

## Fresh test archive

- **Generator.** `python benchmarks/gen_variants.py --kind colored --seed 20261126 --output data/sealed_colored2.npz`:
  the generator and settings of `sealed_colored.npz` (10 systems × N ∈ {2000, 500} × 10 seeds × η ∈ {1, 5, 10, 20}%,
  AR(1) noise with ρ = 0.8, unit variance) = **800 runs**.
- **Seed.** 20261126 appears in the repository only as the proposal in `docs/reports/huber_dev.md`; it is not the
  seed of any archive with a recorded seed (20261023, 20261025, 20261103/4, 20261301/2, 20261400/1, 20261501, and
  the `archive/overnight` seeds). The seeds of `sealed_colored.npz` and the other `gen_variants` archives were
  never recorded, so independence is also **checked empirically** right after generation: no trajectory of the new
  archive may equal (np.array_equal) or share its first sample with any trajectory of any archive in `data/`. If
  that check fails, the seed is incremented by 1 and the check repeated; this would be reported.
- **Checksum.** The archive's sha256 is added to `data/MANIFEST.sha256` and committed **before any method runs on
  it**; `job.sh` aborts if the sha256 is not in the manifest.
- Generated on the Spark (numpy 2.4.4).

## Runs (one pass; `results_final_v11/job.sh`, Spark, 19 workers)

1. **Reproduction check (dev, not part of the test).** Both arms on 30 coloured dev runs (indices 0, 3, …, 87 of
   `instances_colored.npz`): v1.1 must equal the stored `ar1_gauss` records and v1.0.1 the stored `full` records
   of `results_huber_dev/` (coef_err and gap, exact equality).
2. **Primary.** `final_bench.py --arms final final_v101` on all 800 runs of `sealed_colored2.npz`. Both arms are
   computed in the same record from the same seeds (starts) and the same AR(1) pre-estimate.
3. **Guards (primary).** v1.1 on all 800 runs of the white sealed test archive `joint_sealed.npz` (with the starts
   cache used by the frozen test pass) and of the t₃ test archive `sealed_student.npz`. Compared run for run with
   the v1.0.1 re-runs of the same archives made in the v1.0.1 verification (`results_v101/test_sealed.json`,
   `test_student.json`; those are themselves checked against the frozen v1.0 records).
4. **Guards (secondary).** v1.1 on every run of `sealed_coarse.npz` and `heldout2.npz` that v1.0.1 routes to
   AR(1) (index lists derived from `results_v101/` by `job.sh`); every other run of these archives is identical by
   construction, because the router and the white path are unchanged.
5. **Non-independent support.** v1.1 on all 800 runs of the old coloured test archive `sealed_colored.npz`.
   It was already used for this question (the ablation), so it is reported only as support, never as a test.

## Metrics and decision rule (fixed now)

- **Metric.** Exact support recovery of the 3 × 20 coefficient support (`core.metrics`).
- **Primary test.** On `sealed_colored2.npz`: W = runs only v1.1 gets exactly right, L = runs only v1.0.1 gets
  exactly right; two-sided exact sign test (binomial, p = 0.5) on W + L. Significance level 0.05.
- **Guards.**
  - white sealed and t₃: **0 changed answers** expected (xi, gap, noise model, exact flag identical run for run).
    Any change violates the guard and is investigated; any run that v1.0.1 routes to AR(1) there is reported.
  - coarse / held-out 2 AR(1)-routed runs: violated if L − W > 2 on either.
- **Decision.**
  - v1.1 is **confirmed** iff W > L with p < 0.05 on the primary test **and** all guards pass. Then v1.1 stays the
    default of `eqdisc.discover` (as committed here), with `version="1.0.1"` available.
  - Otherwise the default reverts to v1.0.1 (`DEFAULTS["huber_ar1"] = 2.0`), and v1.1 remains selectable with
    `version="1.1"`. The result is reported either way in `docs/reports/v11_coloured.md`.
- **Secondary (descriptive only).** Exact rate per arm; W/L by η, by N and by system; mean F1; median coefficient
  error on runs both versions get exact; runtime; routing counts.
- **Not changed after seeing results:** the archive, the arms, the metric, the test and the rule. No other
  variant (Huber c, ρ threshold, pre-estimate loss) is tried on this archive.

## Archive record (added after generation, before any method run)

- Generated 2026-10-26 11:47:48 EDT (Spark clock), after commit 9b88c9d, on the Spark in 12 s:
  `gen_variants.py --kind colored --seed 20261126 --output data/sealed_colored2.npz` → 800 runs.
- sha256 `281c613233899c585e18e6c47d0b513c8d2630f2be7c2a1306d9d31d2aab3bea`, added to `data/MANIFEST.sha256`.
- Independence check: 800 distinct first samples; compared with all 30 other archives on the Spark
  (`data/*.npz`, `overnight/**`; 20,987 trajectories): **0** equal trajectories, **0** shared first samples.
  Seed 20261126 is kept.
