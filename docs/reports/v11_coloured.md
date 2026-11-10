# v1.1: Gaussian loss on the AR(1) path, pre-registered test on a fresh coloured archive (2026-10-26)

**Bottom line.** On a **fresh** coloured-noise test archive (AR(1), ρ = 0.8; 800 runs; generated after the
pre-registration was committed), v1.1 recovers the exact support in **0.805** of runs against **0.754** for v1.0.1.
Paired, it wins **43** runs and loses **2**: two-sided sign test **p = 6 × 10⁻¹¹**.

The guards pass:
- **White sealed and t₃ test archives:** 0 of 1,600 answers changed. The router never sends these runs to AR(1).
- **AR(1)-routed coarse and held-out runs:** 3/0 and 2/1.

**Under the pre-registered rule, v1.1 is confirmed** and stays the default of `eqdisc.discover`. v1.0.1 remains
available with `discover(y, dt, version="1.0.1")`.

Files:
- pre-registration: `results_final_v11/PREREGISTRATION.md`, committed in `9b88c9d` before the archive existed;
  the archive sha256 went into `data/MANIFEST.sha256` in `c67625d`, before any run
- frozen v1.1 md5s: `results_final_v11/frozen.md5`. The code is unchanged since the pre-registration commit.
- runner: `benchmarks/final_bench.py` (arms `final` = v1.1, `final_v101` = v1.0.1)
- tables: `benchmarks/v11_report.py`, output in `results_final_v11/report.md`
- job and log: `results_final_v11/job.sh`, `launch.sh` and `progress.log`
- per-run records: `results_final_v11/*.json`

## What changed

**v1.1 vs v1.0.1.** v1.1 = v1.0.1 with `huber=inf` (least squares) passed to `joint_colored.discover`, and only
when the router picks the AR(1) noise model.
- Unchanged: Huber c = 2 on the white path and in the AR(1) pre-estimate (`estimate_ar`). Routing is therefore
  identical to v1.0.1, and so are the prior, threshold, λ(a), starts and libraries.
- In code, the switch is `DEFAULTS["huber_ar1"] = inf`.

**Where the idea came from.** The ablation (`docs/reports/ablation.md`) saw the effect on a subsample of the old
coloured TEST archive. The dev diagnostic (`docs/reports/huber_dev.md`) confirmed it on dev data (13/1; 36/2 on all
800 dev runs), and also showed that Huber must stay on the white path (t₃ dev 0/20 without it).

## Protocol (as pre-registered)

**Fresh archive.** `gen_variants.py --kind colored --seed 20261126` → `data/sealed_colored2.npz`.
- Same generator and settings as `sealed_colored.npz`: 10 systems × N ∈ {500, 2000} × 10 seeds × η ∈
  {1, 5, 10, 20}%, giving 800 runs.
- sha256 `281c6132…aab3bea`.
- No trajectory and no first sample is shared with any of the 30 other archives (20,987 trajectories).

**Arms.** Both arms run in the same record. They share the seeds (starts) and the AR(1) pre-estimate, so they can
differ only in the joint stage.

**Metric and test.**
- Exact support recovery.
- W/L counts the runs that only one version gets exactly right.
- Two-sided exact sign test, α = 0.05.

**Guards.**
- Primary: white sealed and t₃ test archives, all 800 runs each, compared run for run with the v1.0.1 re-runs
  from `docs/reports/v101.md`. Expected: 0 changes.
- Secondary: the runs of the coarse and held-out-2 test archives that v1.0.1 routes to AR(1). Violated if L − W > 2.

**Reproduction check (dev data, not part of the test).** On 30 coloured dev runs, v1.1 equals the stored
`ar1_gauss` records of `results_huber_dev/` in 30/30 runs, and v1.0.1 equals the stored `full` records in 30/30
(coefficient error and gap, exact equality).

**Compute.** Spark, 19 workers, 35 min in total; the primary set took 15 min.

## Primary result: `sealed_colored2.npz` (fresh, 800 runs)

| Version | Exact | Mean F1 |
| --- | --- | --- |
| v1.0.1 (Huber in the AR(1) likelihood) | 0.754 | 0.913 |
| **v1.1** (Gaussian in the AR(1) likelihood) | **0.805** | **0.931** |

**v1.1 vs v1.0.1: 43/2, p = 5.9 × 10⁻¹¹.**

- **Routing.** The router sent all 800 runs to AR(1) under both versions; the mean â per run ranged from 0.62 to
  0.86, median 0.76. No run raised an exception.
- **Coefficients.** Every run's coefficients differ numerically between the two versions, as expected, since the
  loss changes.
- **Coefficient error.** The median coefficient error on the 601 runs that both get exact is 0.63% for both.
- **Runtime.** Joint-stage time is unchanged: median 6.4 s vs 6.5 s.

| Cell | n | v1.0.1 | v1.1 | W/L | p |
| --- | --- | --- | --- | --- | --- |
| η = 1% | 200 | 0.925 | 0.940 | 3/0 | 0.25 |
| η = 5% | 200 | 0.770 | 0.870 | 21/1 | 1 × 10⁻⁵ |
| η = 10% | 200 | 0.740 | 0.785 | 10/1 | 0.012 |
| η = 20% | 200 | 0.580 | 0.625 | 9/0 | 0.004 |
| N = 500 | 400 | 0.752 | 0.765 | 7/2 | 0.18 |
| N = 2000 | 400 | 0.755 | 0.845 | 36/0 | 3 × 10⁻¹¹ |

| System | v1.0.1 | v1.1 | W/L |
| --- | --- | --- | --- |
| Lorenz | 0.512 | 0.738 | 18/0 |
| Chen | 0.500 | 0.662 | 14/1 |
| Rössler | 0.550 | 0.600 | 4/0 |
| Halvorsen | 0.925 | 0.963 | 3/0 |
| Rabinovich–Fabrikant | 0.062 | 0.100 | 3/0 |
| Sprott C | 0.988 | 0.988 | 1/1 |
| Sprott A, B, D, E | 1.000 | 1.000 | 0/0 |

(n = 80 runs per system.)

**Wins.** Of the 43 runs that v1.1 wins, v1.0.1 had added spurious terms in 31 and missed a true term in 36; these
overlap.

**Losses.** The 2 losses are Chen (η = 5%, N = 500), where v1.1 adds 3 terms, and Sprott C (η = 10%, N = 500).

**The pattern matches dev.** On dev the gains also came from Chen and Lorenz, from η = 5–10%, and mostly from N = 2000
(dev N = 500: 9/2; here 7/2). The size is similar too: +5.1 points here against +4.2 on all 800 dev runs.

**Ambiguity flags.** Flags are raised more often: 119 runs under v1.1 against 96 under v1.0.1. The flag threshold
is 2λ, but the Gaussian AR(1) score gives smaller gaps. The flag was calibrated on Huber scores. Its accuracy under
v1.1 has not been re-checked; this is a caveat, not something tested here.

## Guards

v1.1 is compared run for run with the v1.0.1 re-run of the same archive. "Changed" counts any difference in ξ,
gap, noise model or the exact flag.

| Archive | Runs | Routed to AR(1) | Changed | v1.0.1 exact | v1.1 exact | W/L |
| --- | --- | --- | --- | --- | --- | --- |
| White sealed `joint_sealed.npz` (primary guard) | 800 | 0 | **0** | 0.889 | 0.889 | 0/0 |
| t₃ `sealed_student.npz` (primary guard) | 800 | 0 | **0** | 0.905 | 0.905 | 0/0 |
| Coarse `sealed_coarse.npz`, AR(1)-routed only (secondary) | 12 of 800 | 12 | 12 | 0.583 | 0.833 | 3/0 |
| Held-out 2 `heldout2.npz`, AR(1)-routed only (secondary) | 22 of 880 | 22 | 22 | 0.045 | 0.091 | 2/1 |

- **Primary guards: pass.** 0 of 1,600 answers changed. Every white and t₃ test run stays on the white path, where
  v1.1 is v1.0.1 by construction, and the runs confirm it.
- **Secondary guards: pass** (L − W = −3 and −1).
  - **Coarse.** 12 runs are routed to AR(1) through model misfit (Lorenz, Chen and Sprott D; both N). The 3 wins
    are Lorenz at 1%, Chen at 1% and Sprott D at 10%, all at N = 500.
  - **Held-out 2.** All 22 AR(1)-routed runs are the held-out Chen–Lee system, which is almost never recovered by
    either version. v1.1 wins 2 and loses 1 (all at N = 500).
  - **Archive-level effect.** On the full archives, these changes move coarse from 0.7837 to 0.7875 and held-out
    2 from 0.9409 to 0.9420. The other runs are unchanged by construction.

## Non-independent support: the old coloured test archive

`sealed_colored.npz` was used for this question before, in the ablation's test subsample, so it is **not** a test.
For completeness: v1.0.1 0.761 → v1.1 **0.812**, W/L 42/1. 799 of 800 runs route to AR(1).

## Pre-registered decision

The rule was: confirmed iff W > L with p < 0.05 on the primary test, **and** 0 changed answers on white and t₃,
**and** L − W ≤ 2 on coarse and held-out 2.

- Primary: 43/2, p = 5.9 × 10⁻¹¹. ✔
- White: 0 changes. ✔
- t₃: 0 changes. ✔
- Coarse: 3/0. ✔
- Held-out 2: 2/1. ✔

**→ v1.1 confirmed.** `eqdisc.discover` defaults to v1.1 (`eqdisc.__version__ == "1.1.0"`). `version="1.0.1"`
reproduces v1.0.1.

## What v1.1 changes in the paper's numbers

Only runs that route to AR(1) can change. On the frozen test archives, these are:
- **Coloured** (`sealed_colored.npz`): 0.761 → 0.812. But this archive is non-independent for this question, so
  the fresh archive's 0.754 → 0.805 is the number to quote for the change.
- **Coarse:** 0.784 → 0.7875.
- **Held-out 2:** 0.941 → 0.942.

Unchanged: sealed white, t₃, confirmation, trig, the degree-4/5 libraries and D ≠ 3. They were not re-run for v1.1;
they stay at v1.0.1 = v1.0 because none of their runs routes to AR(1).
- This was **verified** for sealed white and t₃ (the guards above), and by the routing counts in `results_v101/`
  for coarse and held-out 2.
- For confirmation, trig, mixed-poly and degree-4/5 it rests on construction plus the stored routing: their
  v1.0.1 records show 0 AR(1) runs.
- dysts was not re-run with v1.1. Its 3-D runs that route to AR(1) could change. The dysts archive has white noise,
  so few such runs are expected, but this is not verified.
- ODEBench D = 3 was not re-run either.

## Caveats

- **Ten systems, one generator.** The fresh archive uses the same ten systems and the same noise generator as the
  dev and old test archives; only the seeds are new. The evidence is about AR(1) noise with ρ = 0.8 on these
  systems. Mid-ρ (0.3–0.55) never routes to AR(1), so v1.1 is untested there by construction.
- **Concentration.** 32 of the 43 wins are Lorenz and Chen.
- **Untested combination.** Noise that is both heavy-tailed and correlated, where Huber inside the AR(1)
  likelihood might matter, is still untested.
- **Ambiguity flag.** It fires more often under v1.1 (above); its calibration was not re-checked.
- **Process.** The first launch of the job aborted before any run, because the Spark's copy of
  `data/MANIFEST.sha256` had not been synced; `job.sh` checks the sha256. It was relaunched unchanged. Both events
  are in `results_final_v11/progress.log`.
