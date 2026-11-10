# Hard cases, enlarged: final model vs DEDi on a fresh archive (2026-10-25/26)

**Status: final.** The DEDi job ran on the DGX Spark in two sessions:
- **Session 1:** 17:40–21:02, 12 cores. The Spark crashed and rebooted at about 21:02, with 18 runs saved.
- **Session 2:** 23:01 to the 10:01 deadline, 14 cores. It resumed in the same pre-registered order and skipped the finished runs.

That is about 14.4 h of DEDi wall time in total. By the deadline 118 of 120 runs had finished, which gives a pre-registered analysis set of 114 (38 per system).

## Why

The earlier hard-40 comparison (Lorenz, Chen, Rössler and Rabinovich–Fabrikant at 20% noise, N = 2000, 10 runs each) gave final model 28/40 against DEDi 21/40, with 7/0 head to head (p = 0.016). That sample is small. It was also reused while earlier versions of the method were developed, because the hard-40 was reported after every change. This study uses a new archive that no method had seen before and a larger n.

## Protocol (pre-registered; committed in `dc8c325` before any method was run)

- **Archive:** `data/hard120.npz`, sha256 `0152457f…958096` (full hash in `data/MANIFEST.sha256`). Made with `benchmarks/flow_generate.py --seed 20261025 --seeds 40 --length 2000 --systems lorenz chen rossler --noise 0.2`.
  - This is the generator of the confirmation set: re-running it with seed 20261103 reproduces `hybrid_conf_2000.npz` bit for bit, and `hybrid_confirm.npz` is that file plus the N = 500 half.
  - So the protocol is unchanged: the same per-system dt, a reference orbit with 2000 burn-in steps, a random start on that orbit, and noise drawn as 0.2·std(clean)·N(0, 1) per coordinate.
  - Runs: Lorenz is indices 0–39, Chen 40–79, Rössler 80–119.
- **Rabinovich–Fabrikant is excluded.** Every method, DEDi included, scored 0/10 on it in the hard-40, and DEDi made 157 false positives there. This points to a data or identifiability limit at this noise level and length, so more runs would not change the comparison.
- **Our methods (all 120 runs):**
  - final model: `eqdisc.discover` with its frozen DEFAULTS (`final_bench.py`, arm `final`);
  - config C: arm `C`, white noise, no prior;
  - weak χ² best-subset: the first start, `starts[0]`.

  Starts were computed fresh. As a check, the same code and fresh starts reproduce the published hard-40 final result exactly: 28/40, with all 40 runs identical.
- **DEDi:**
  - Settings are identical to the hard-40 run, with no retuning: `benchmarks/dedi_bench.py --grid small`, i.e. λ ∈ {10⁻², 1} and R ∈ {10⁻⁴, 10⁻²}, chosen by validation error. The authors' code is used unmodified, with the same 20-term library and the every-third-sample validation mask.
  - Runs follow the pre-registered order `results_hard120/dedi_order.json`: Lorenz k, Chen k, Rössler k for k = 0…39, in draw order.
  - **Analysis set:** the longest prefix of that order in which every run has finished, truncated to whole (Lorenz, Chen, Rössler) triples.
  - Runner changes (plumbing only): `--index-file` for the order, and per-run saving with `as_completed`.
- **Tests:** exact support recovery; paired wins and losses with a two-sided exact sign test; 95% Wilson intervals.

## Results

### Table 1: our methods on all 120 new runs

| System (20% noise, N = 2000) | n | Weak χ² best-subset | Config C | **Final model** |
| --- | --- | --- | --- | --- |
| Lorenz | 40 | 0/40 (0%) | 31/40 (78%) | **35/40 (88%; 74–95%)** |
| Chen | 40 | 0/40 (0%) | 40/40 (100%) | **40/40 (100%; 91–100%)** |
| Rössler | 40 | 0/40 (0%) | 34/40 (85%) | **37/40 (92%; 80–97%)** |
| **All** | 120 | 0/120 (0%) | 105/120 (88%) | **112/120 (93%; 87–97%)** |

Final model against config C: 7 wins, 0 losses (p = 0.016). The complexity prior helps on Lorenz (+4) and Rössler (+3).

### Table 2: head to head with DEDi, pre-registered analysis set (new archive)

| System | n | Weak χ² | Config C | **Final** | DEDi | Final vs DEDi W/L | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lorenz | 38 | 0/38 | 29/38 (76%) | **33/38 (87%; 73–94%)** | 30/38 (79%; 64–89%) | 7/4 | 0.55 |
| Chen | 38 | 0/38 | 38/38 (100%) | **38/38 (100%; 91–100%)** | 31/38 (82%; 67–91%) | 7/0 | 0.016 |
| Rössler | 38 | 0/38 | 32/38 (84%) | **35/38 (92%; 79–97%)** | 24/38 (63%; 47–77%) | 12/1 | 0.0034 |
| **All** | 114 | 0/114 | 99/114 (87%) | **106/114 (93%; 87–96%)** | 85/114 (75%; 66–82%) | **26/5** | **1.9×10⁻⁴** |

### Table 3: pooled with the earlier hard-40 (Lorenz, Chen and Rössler only; 30 runs)

For the hard-40 runs, "Final" is the published frozen test pass (`results_final/test_confirm.json`), and DEDi is `results_hybrid/dedi_hard20*.json`.

| System (20%, N = 2000) | n | Weak χ² | Config C | **Final model** | DEDi | Final vs DEDi W/L | Sign-test p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lorenz | 48 | 0/48 | 36/48 (75%) | **43/48 (90%; 78–95%)** | 38/48 (79%; 66–88%) | 9/4 | 0.27 |
| Chen | 48 | 0/48 | 48/48 (100%) | **48/48 (100%; 93–100%)** | 38/48 (79%; 66–88%) | 10/0 | 0.002 |
| Rössler | 48 | 0/48 | 40/48 (83%) | **43/48 (90%; 78–95%)** | 30/48 (62%; 48–75%) | 14/1 | 9.8×10⁻⁴ |
| **All** | **144** | 0/144 | 124/144 (86%) | **134/144 (93%; 88–96%)** | 106/144 (74%; 66–80%) | **33/5** | **4.3×10⁻⁶** |

Four further DEDi runs finished outside the analysis set (the in-flight runs from the last triples); they are not in the headline. Both methods were exact on all four. Two runs were still unfinished at the deadline.

### Coefficients, false positives, cost

- **Coefficient error:** in the 80 runs where both methods are exact, the median relative coefficient error is 0.86% for the final model and 0.94% for DEDi. DEDi is lower in 28 of the 80, so the two are comparable.
- **Errors on the analysis set:** 10 false positives and 8 false negatives for the final model, against 146 and 13 for DEDi. DEDi's failures are mostly extra terms.
- **Cost:**
  - Final model: median 8.6 s per run on one core (max 24 s), including the weak starts.
  - DEDi: median 1.36 h wall per run (0.46–3.30 h), with one worker per pinned core. Roughly 500–600× slower, but the two figures were measured under different parallel loads, so treat the ratio as an order of magnitude.
  - All 120 runs of our method took 3 min on 11 cores. The DEDi runs took two sessions on 12 and then 14 cores, about 14.4 h of wall time in total.

## Reading, with caveats

- **On these three systems the final model beats DEDi:** pooled 134/144 (93%) against 106/144 (74%), 33/5, p ≈ 4×10⁻⁶. The fresh archive alone gives 106/114 against 85/114, 26/5 (p = 2×10⁻⁴), so the result does not depend on the reused hard-40.
- **Per system:**
  - Chen is clear (10/0, p = 0.002): the final model is 48/48.
  - Rössler is clear (14/1, p < 0.001).
  - Lorenz is not significant (9/4, p = 0.27): 90% against 79%, and DEDi wins 4 of these runs. The paper should say the advantage on Lorenz is not established.
- **DEDi may be under-tuned:**
  - It used the 4-pair grid, not the paper's 35 pairs, to keep the cost feasible (the full grid is about 9× slower).
  - Validation error picked λ = 10⁻² in 110 of 118 runs, even though λ = 1 was available, and most of its failures are false positives.
  - So selecting by validation error leans toward dense models here. Intermediate values on the paper's grid (λ = 0.1, 10) might trade false positives better, but that was not tested.
  - A paper must say this. The honest claim is "better than DEDi as run with a reduced 2×2 hyperparameter grid at matched library and validation protocol", not "better than DEDi".
  - A full-grid run on a subset (about 13 h per run) would close this gap but was not affordable here.
- **The weak χ² selector alone scores 0% here.** At 20% noise every win comes from the joint stage (config C plus the prior), which shows how much it adds. It also shows that this setting is outside the range where weak-form selection works.
- **Scope:**
  - The comparison covers three classical 3-D chaotic systems at a single setting: 20% noise and N = 2000.
  - Rabinovich–Fabrikant is excluded, and every method fails on it.
  - These systems were chosen because earlier versions of our method found them hard, so this is a stress test, not a representative average.
- **The crash did not bias the order:**
  - The 18 runs finished before the crash were kept.
  - The runs in flight at the crash restarted from scratch in the same pre-registered order, with the finished indices skipped (`dedi_order_resume.json`).
  - The analysis-set rule was applied to the merged results.

## Files

- `results_hard120/PREREGISTRATION.json` and `dedi_order.json`: the pre-registration and the run order (committed before launch). `dedi_order_resume.json` is the same order without the 18 runs finished before the crash.
- `results_hard120/ours_hard120.json` (final and C on 120) and `ours_hard40_repro.json` (the reproduction check).
- `results_hard120/dedi_hard120.json` (session 1, 18 runs) and `dedi_hard120_part2.json` (session 2, 100 runs). Logs: `dedi_*.log`; session times: `dedi_progress.log`.
- Job scripts: `ours.sh`, `dedi.sh`, `dedi_resume.sh`.
- Report: `python benchmarks/hard120_report.py` regenerates every table above.
