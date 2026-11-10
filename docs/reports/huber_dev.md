# Huber vs Gaussian loss on dev archives (diagnostic, 2026-10-26)

**Bottom line.** The ablation found, on the coloured TEST subsample, that the Gaussian loss beats Huber under
AR(1) noise (0.831 vs 0.730, 9/0). The same effect appears on the **coloured DEV archive**: Gaussian 0.805
against the frozen model's 0.760, **13/1, p = 0.002**.

Turning Huber off everywhere is not an option:
- on **t₃ dev** it costs **0/20 (p = 2×10⁻⁶)**, 0.918 → 0.843, so Huber does what it was added for;
- on white dev it is neutral (0/0).

The variant that passes every pre-registered guard is **path-specific**: Gaussian only inside the AR(1)
likelihood, Huber kept on the white path and in the AR(1) pre-estimate. It also gives 13/1 on coloured dev, and
it cannot change any t₃ or white answer here, because the router sends none of those runs to AR(1).

On **all 800** coloured dev runs it is 36/2 (p = 5×10⁻⁹), 0.750 → 0.792.

**Recommendation.** This is positive under the pre-registered rule. It is a candidate change for a future
version, to be confirmed on a **fresh** coloured test archive (proposal below; it was not run). The frozen
model is unchanged.

Files:
- plan, written before the runs: `results_huber_dev/PLAN.md`
- runner: `benchmarks/huber_dev.py`, a wrapper around `benchmarks/ablation.py`
- tables: `benchmarks/huber_dev_report.py`
- per-run records: `results_huber_dev/*.json`

## Setup

- **Arms.** All arms use the frozen final model (`final_model._discover3`, the same machinery as the
  ablation). They differ as follows:

  | Arm | Noise model | Loss |
  | --- | --- | --- |
  | `full` | frozen: noise = "auto" | Huber, c = 2 |
  | `gaussian` | auto | huber = ∞ in the joint fits **and** the AR(1) pre-estimate |
  | `ar1` | AR(1) likelihood forced | Huber |
  | `ar1_gauss` | AR(1) forced | Gaussian |

  A derived arm, `path_gauss`, equals `ar1_gauss` when `full` chose AR(1) and `full` otherwise. Its routing is
  identical to the frozen model's, because the pre-estimate keeps Huber.
- **Data.** Dev archives only. Every index ≡ 0 mod 3 is used; the stride 3 is coprime with the mod-4 noise
  cycle. That gives 267 runs per archive, 66–67 per noise level, 133–134 per N.
  - `instances_colored.npz`: AR(1), ρ = 0.8
  - `instances_student.npz`: t₃
  - `instances.npz`: white
- **Statistics.** Exact support recovery. W/L counts the runs only one arm gets exactly right. p is from a
  two-sided sign test.
- **Reproducibility.** `full` matches the stored frozen dev records run for run: coloured 134/134 against
  `results_final/dev_colored.json` arm ARp, t₃ 67/67 (Cp), white 267/267 (Cp). Both exact flags and
  coefficient errors agree to 1e-9.
- **Compute.** Spark, 5 workers on the cores not used by the DEDi job.
  - Primary: coloured 43 min (4 arms), t₃ 12 min, white 10 min.
  - Extensions: 2 h 0 min.
  - Total: 3 h 05 min wall.

## Primary results (index ≡ 0 mod 3)

| Archive (dev) | Runs | Frozen (Huber) | Gaussian everywhere | Gaussian vs frozen | Path-specific Gaussian | Path vs frozen |
| --- | --- | --- | --- | --- | --- | --- |
| Coloured AR(1), ρ = 0.8 | 267 | 0.760 | **0.805** | **13/1 (p = 0.002)** | **0.805** | **13/1 (p = 0.002)** |
| t₃ heavy tails | 267 | **0.918** | 0.843 | 0/20 (p = 2×10⁻⁶) | **0.918** | 0/0 (by construction) |
| White | 267 | 0.891 | 0.891 | 0/0 | 0.891 | 0/0 (by construction) |

The frozen router sent 267/267 coloured runs to AR(1), and 0/267 of the t₃ runs and 0/267 of the white runs.

**Coloured dev by cell** (Gaussian vs frozen; the forced-AR(1) pair `ar1_gauss` vs `ar1` is identical run for
run):

| Cell | n | Frozen | Gaussian | W/L |
| --- | --- | --- | --- | --- |
| η = 1% | 67 | 0.925 | 0.925 | 1/1 |
| η = 5% | 66 | 0.773 | 0.848 | 5/0 |
| η = 10% | 67 | 0.761 | 0.836 | 5/0 |
| η = 20% | 67 | 0.582 | 0.612 | 2/0 |
| N = 500 | 134 | 0.746 | 0.769 | 4/1 |
| N = 2000 | 133 | 0.774 | 0.842 | 9/0 |

- **Where the gains are.** The 13 wins are concentrated on Chen (7), then Lorenz (3) and one each of Rössler,
  Sprott B and Rabinovich–Fabrikant. In 7 of them the frozen model added terms; in the others it missed one.
  The single loss is Sprott A at 1%, N = 500.
- **Coefficients and runtime.** Median coefficient error on runs both get exact: 0.60% (Huber), 0.56%
  (Gaussian). Median joint-stage time: 9.3 s vs 8.8 s.

**t₃ guard.** The 20 losses are spread over 9 of the 10 systems and all four noise levels (5/4/6/5). This is
the regime Huber was designed for: on t₃ dev it was originally adopted at 0.78 → 0.905.

## Extensions (planned in `PLAN.md`, run after the primary sets)

| Archive (dev) | Runs | Frozen | Gaussian everywhere | Gaussian vs frozen | Path-specific vs frozen |
| --- | --- | --- | --- | --- | --- |
| Coloured, ρ = 0.8, **all 800** | 800 | 0.750 | **0.792** | **36/2 (p = 5×10⁻⁹)** | **36/2 (p = 5×10⁻⁹)** |
| t₃, indices 0 and 1 mod 3 | 534 | **0.918** | 0.830 | 0/47 (p = 1×10⁻¹⁴) | 0/0 (by construction) |
| Mid-ρ (ρ = 0.3 / 0.55), all | 320 | **0.841** | 0.825 | 1/6 (p = 0.12) | 0/0 (by construction) |

All 800 coloured runs, by cell (Gaussian vs frozen):

| Cell | Frozen | Gaussian | W/L |
| --- | --- | --- | --- |
| η = 1% | 0.915 | 0.930 | 4/1 |
| η = 5% | 0.785 | 0.855 | 14/0 |
| η = 10% | 0.725 | 0.775 | 11/1 |
| η = 20% | 0.575 | 0.610 | 7/0 |
| N = 500 | 0.738 | 0.755 | 9/2 |
| N = 2000 | 0.762 | 0.830 | 27/0 |

- **Coloured.** The 36 wins are spread over 7 of the 10 systems: Chen 15, Lorenz 8, Rössler 6,
  Rabinovich–Fabrikant 3, Halvorsen 2, Sprott A 1, Sprott B 1. The 2 losses are Sprott A (1%, N = 500) and
  Lorenz (10%, N = 500). The router sent 799 of 800 runs to AR(1); the exception is one Rabinovich–Fabrikant
  run at 20%, which both arms get wrong.
- **Mid-ρ.** The router never switches here, so the path-specific variant cannot change anything. Gaussian on
  the white path is mildly worse (1/6), which again says Huber should stay on the white path.
- **Reproduction.** Over all extension runs, `full` matches the stored frozen records exactly: coloured
  400/400, t₃ 134/134, mid-ρ 320/320.

## Interpretation

1. **The ablation's test-set finding replicates on dev data.** The size is similar: +4.5 points on dev against
   +10 on the test subsample. The coloured dev archive has the same generator as the test archive, with
   different seeds. So the effect is not an artefact of the test subsample.
2. **Mechanism.** The finding is consistent with the ablation's hypothesis:
   - the gain is entirely within the AR(1) likelihood (`ar1_gauss` vs `ar1` = `gaussian` vs `full`, run for run);
   - it is largest at moderate noise (5–10%) and at N = 2000.

   Huber at c = 2 on the whitened residuals down-weights large innovations, which the AR(1) likelihood treats
   as signal.
3. **Huber must stay on the white path.** Removing it there costs 20 t₃ runs and gains none. Gaussian
   everywhere is therefore rejected.
4. **The path-specific change is free on these archives, but only because of routing.** t₃ and white dev never
   trigger the AR(1) switch, so they cannot be hurt.
   - Untested: data that is both heavy-tailed and correlated, where Huber inside the AR(1) likelihood might
     matter.
   - Also untested: coarse or held-out runs that switch to AR(1) through model misfit (2% of them on dev in
     F_final).

## Pre-registered decision

The rule from `PLAN.md`:

> "positive" iff on coloured dev Gaussian (or path_gauss) beats full with p < 0.05, AND on t₃ it does not
> lose with p < 0.05 and has L − W ≤ 2, AND on white it has L − W ≤ 2.

- `gaussian` fails the t₃ guard: L − W = 20.
- `path_gauss` passes: 13/1 (p = 0.002), t₃ 0/0, white 0/0. **→ Positive (path-specific variant only).**

The frozen model is **not** changed. This is a recommendation for a possible v2, conditional on the test
below.

## Proposal (not run): fresh coloured test archive

- **Archive.** Generate with `python benchmarks/gen_variants.py --kind colored --seed 20261126 --output
  data/sealed_colored2.npz`. That is 800 runs: 10 systems × 2 N × 10 seeds × 4 noise levels, with the same
  generator as `sealed_colored.npz`.
  - Check first that the seed differs from every archive seed in `data/MANIFEST.sha256`'s provenance notes.
  - Record its sha256 in the manifest before any method runs.
- **Arms.**
  - `full`: the frozen model.
  - `path_gauss`: `noise="auto"`, then `huber=inf` passed only to `joint_colored.discover` when the router
    picks AR(1). In `benchmarks/huber_dev.py` this is the `ar1_gauss` arm read on the runs `full` routes to
    AR(1).
  - Freeze the runner's md5 in the pre-registration.
- **Primary test.** `path_gauss` vs `full` on all 800 runs: paired W/L with a two-sided sign test,
  significance at p < 0.05. No subsampling, so no stride issue.
- **Guards on existing test archives.**
  - Confirm that the router never selects AR(1) on `sealed_student.npz` / `joint_sealed.npz`. If it never does,
    `path_gauss` = `full` there by construction.
  - On `sealed_coarse.npz` and the held-out archives, report every run routed to AR(1) and whether the answer
    changes. A guard is violated if L − W > 2 on any of them.
- **Not part of the test.** `sealed_colored.npz` has already been seen with this question (in the ablation),
  so it can only be reported as non-independent support.
- **Cost.** 800 runs × 2 arms. That is about 1 h on 5 Spark cores, or about 20 min on 18.

## Caveats

- **Sample and independence.** The primary subsets have 267 runs. Runs within an archive share 10 systems. Chen contributes 7 of 13
  primary wins, and 15 of 36 over all 800 runs.
- **AR(1) path only tested at ρ = 0.8.** On mid-ρ the router stays white, so `path_gauss` is never exercised
  there (see Extensions).
- **Origin of the question.** It came from a test-archive observation. The dev check is clean, but the idea
  was not generated on dev.
