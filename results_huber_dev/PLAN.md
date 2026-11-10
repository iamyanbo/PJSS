# Huber vs Gaussian on dev archives: plan (written 2026-10-26 00:35, before any run)

Question (from docs/reports/ablation.md item 6): the Gaussian loss beat Huber on the coloured TEST subsample
(0.831 vs 0.730, 9/0). Is this visible on DEV data, and does it cost anything on heavy-tailed (t3) or white dev?
This is a diagnostic. The frozen model is not changed, whatever the outcome.

Runner: `benchmarks/huber_dev.py` (wrapper over `benchmarks/ablation.py`); job: `results_huber_dev/job.sh`.

Arms: `full` (frozen, noise auto, Huber c = 2), `gaussian` (auto, huber = inf in joint fits and AR pre-estimate),
on coloured also `ar1` / `ar1_gauss` (AR(1) likelihood forced, Huber / Gaussian).
Derived arm `path_gauss` = Gaussian only when the frozen router picks AR(1): equals `ar1_gauss` on runs where
`full` chose AR(1), else `full` (routing is unchanged because the pre-estimate keeps Huber).

Data, in run order (stride 3 is coprime with the mod-4 noise cycle):
1. `instances_colored.npz` (AR(1) rho = 0.8), indices = 0 mod 3 (267 runs): primary.
2. `instances_student.npz` (t3), 0 mod 3 (267): guard.
3. `instances.npz` (white), 0 mod 3 (267): guard.
4. Extensions if time allows: coloured 1 and 2 mod 3 (to all 800), `dev_colored_mid.npz` (320), t3 1 mod 3.

Metric: exact support recovery; paired W/L on runs where exactly one arm is exact; two-sided sign test.
Decision rule for the recommendation (fixed now):
- "positive" iff on the primary coloured set Gaussian (or path_gauss) beats full with p < 0.05, AND on the t3
  guard it does not lose with p < 0.05 and has L - W <= 2, AND on white it has L - W <= 2.
- Primary numbers are the 0 mod 3 subsets; extensions are reported separately and pooled.
- If positive: write (do not run) a proposal for a fresh coloured TEST archive.
Sanity check: `full` must reproduce the stored frozen dev records in results_final/dev_*.json (arm ARp on
coloured, Cp on t3 / white) run for run.
