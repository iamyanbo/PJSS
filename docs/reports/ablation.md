# Component ablation of the final model (2026-10-25)

Each component of `eqdisc.discover` (D = 3, poly3 library) is removed one at a time from the frozen final model
(leave-one-out). Separately, a cumulative ladder adds the components one by one, going from the weak best-subset
selector up to the final model. Only test archives are used, and nothing was tuned. The model and its defaults
are unchanged: every variant is an option flag or a temporary module patch in `benchmarks/ablation.py`. In all
four sets, the "full" arm reproduces the frozen test passes run by run, with identical exact flags and
coefficient errors (sealed 160/160, coloured 89/89, hard 40 40/40, dysts 287/287).

Runner: `benchmarks/ablation.py`; tables: `benchmarks/ablation_report.py` (full output in
`results_ablation/report.md`); raw per-run records: `results_ablation/*.json`.

## Evaluation sets (all subsampled; strides coprime with the noise cycle)

| Set | Archive | Runs used | Cell balance |
| --- | --- | --- | --- |
| Sealed (white) | `data/joint_sealed.npz`, every 5th | 160 / 800 | 40 per noise level, 80 per N, 16 per system |
| Coloured AR(1), ρ = 0.8 | `data/sealed_colored.npz`, every 9th | 89 / 800 | 11–12 per (noise, N) cell |
| Hard 40 | `data/hybrid_confirm.npz` at `results_final/hard40_idx.json` | 40 / 40 | 20 % noise, N = 2000 |
| dysts, D = 3 | `data/dysts_test.npz`, D = 3 runs with index ≡ 0 mod 17 | 287 / 4,880 | 35–37 per (noise, N) cell, all 61 systems |

The subsamples are smaller than originally planned. The PC had to be freed partway through, and the coloured
and dysts stages were moved to the busy Spark at 4–6 workers, where one hard stop made the dysts stage resume
from its checkpoint. The full model's rate on each subsample is close to its rate on the whole archive:
- sealed: 0.869 here, 0.889 on all 800 runs;
- coloured: 0.730 here, 0.761;
- dysts D = 3: 0.899 here, 0.898.

D = 4–6 dysts systems (the `joint_nd` path) and the coarse, t₃, held-out and big-library archives are not ablated.

## Leave-one-out table

Each cell gives the exact-support rate, then the paired count *full wins / variant wins* (runs that only one of
the two gets exactly right), then the two-sided sign-test p in parentheses. The pooled column sums the counts
over all four sets.

| Variant | Sealed (160) | Coloured (89) | Hard 40 | dysts D=3 (287) | Pooled full/var (p) |
| --- | --- | --- | --- | --- | --- |
| **Full final model** | **0.869** | **0.730** | **0.700** | **0.899** | – |
| No joint states: weak best-subset | 0.769 · 18/2 (4e-4) | 0.640 · 8/0 (8e-3) | 0.000 · 28/0 (7e-9) | 0.829 · 20/0 (2e-6) | 74/2 (8e-20) |
| No joint states: hybrid (weak / flow) | 0.812 · 10/1 (0.01) | 0.270 · 41/0 (9e-13) | 0.400 · 12/0 (5e-4) | 0.819 · 24/1 (2e-6) | 87/2 (1e-23) |
| − profiled-quadratic proposals (refit starts only) | 0.819 · 8/0 (8e-3) | 0.652 · 7/0 (0.02) | 0.475 · 9/0 (4e-3) | 0.871 · 8/0 (8e-3) | 32/0 (5e-10) |
| Greedy stepwise instead of best-subset proposals | 0.819 · 8/0 (8e-3) | 0.674 · 5/0 (0.06) | 0.475 · 9/0 (4e-3) | 0.878 · 6/0 (0.03) | 28/0 (7e-9) |
| − multiple starts (weak start only) | 0.869 · 0/0 | 0.775 · 2/6 (0.29) | 0.675 · 1/0 | 0.899 · 0/0 | 3/6 (0.51) |
| − σ/φ self-calibration (white path) | 0.869 · 0/0 | 0.730 · 0/0 | 0.700 · 0/0 | 0.899 · 0/0 | 0/0 |
| − φ autocorrelation inflation | 0.869 · 0/0 | 0.742 · 0/1 | 0.700 · 0/0 | 0.899 · 0/0 | 0/1 |
| − Huber loss (Gaussian) | 0.875 · 0/1 | **0.831 · 0/9 (4e-3)** | 0.700 · 0/0 | 0.895 · 1/0 | **1/10 (0.01)** |
| − auto sub-steps (s = 2) | 0.875 · 0/1 | 0.730 · 1/1 | 0.700 · 0/0 | 0.885 · 6/2 (0.29) | 7/4 (0.55) |
| − complexity prior (degw = 0) | 0.856 · 2/0 (0.50) | 0.697 · 3/0 (0.25) | 0.625 · 3/0 (0.25) | 0.892 · 2/0 (0.50) | 10/0 (2e-3) |
| − AR(1) auto-switch (noise = "white") | 0.869 · 0/0 | 0.674 · 9/4 (0.27) | 0.700 · 0/0 | 0.899 · 0/0 | 9/4 (0.27) |
| Oracle noise level (not deployable) | 0.875 · 0/1 | 0.820 · 0/8 (8e-3) | 0.700 · 0/0 | 0.902 · 0/1 | – |

**Coefficient error.** This is the median relative error ‖ξ̂ − ξ‖/‖ξ‖ over runs that both the variant and the
full model get exact. It barely moves under any ablation:

| | Sealed | Coloured | Hard 40 | dysts |
| --- | --- | --- | --- | --- |
| Full model | 0.24 % | 0.63 % | 0.60 % | 0.26 % |
| Range over variants | 0.22–0.24 % | 0.58–0.73 % | 0.56–0.83 % | 0.25–0.28 % |

Without joint states the error roughly doubles:
- weak selector: 0.58 % / 1.46 % / – / 0.62 %;
- hybrid: 0.65 % / 1.54 % / 1.44 % / 0.62 %.

**Runtime.** Median joint-stage seconds per run, with the seeds excluded because they are shared:

| | Sealed | Coloured | Hard 40 | dysts |
| --- | --- | --- | --- | --- |
| Full model | 8.4 | 6.0 | 24.2 | 2.5 |
| No proposals | 4.4 | 3.2 | 10.3 | 1.3 |
| s = 2 | 6.9 | 4.1 | 11.7 | 2.4 |
| White only | 6.4 | 4.4 | 15.9 | 1.9 |

The white-only variant is cheaper because it skips the AR(1) pre-estimate. Greedy proposals cost the same as
exhaustive ones (8.6 s on sealed). Seeds take a median of 1.1–1.6 s for all starts and 0.1–0.6 s for the weak
selector alone. Sealed and hard 40 were timed on the 12-core PC; coloured and dysts were timed on the loaded
Spark. Compare runtimes within a column only.

## Cumulative ladder

Each cell gives the exact rate, then the wins/losses against the previous rung, then the sign-test p in
parentheses. Until the last rung the white-noise likelihood is used. Every joint rung keeps the stiff final refit,
which changes coefficients only.

| Rung | Sealed | Coloured | Hard 40 | dysts D=3 |
| --- | --- | --- | --- | --- |
| Weak best-subset (χ²s) | 0.769 | 0.640 | 0.000 | 0.829 |
| + joint states (refit the weak start; s = 2, HF σ, φ = 1, Gaussian) | 0.769 · 0/0 | 0.640 · 0/0 | 0.000 · 0/0 | 0.829 · 0/0 |
| + profiled-quadratic best-subset proposals | 0.856 · 16/2 (1e-3) | 0.000 · 0/57 (1e-17) | 0.625 · 25/0 (6e-8) | 0.864 · 18/8 (0.08) |
| + multiple starts (hybrid, flow) | 0.856 · 0/0 | 0.000 · 0/0 | 0.650 · 1/0 | 0.861 · 0/1 |
| + σ/φ self-calibration | 0.856 · 1/1 | 0.629 · 56/0 (3e-17) | 0.650 · 0/0 | 0.868 · 2/0 (0.50) |
| + auto sub-steps | 0.856 · 0/0 | 0.596 · 0/3 (0.25) | 0.625 · 0/1 | 0.889 · 7/1 (0.07) |
| + Huber loss (= config C) | 0.856 · 0/0 | 0.629 · 3/0 (0.25) | 0.625 · 1/1 | 0.892 · 1/0 |
| + complexity prior (= C + prior) | 0.869 · 2/0 (0.50) | 0.674 · 5/1 (0.22) | 0.700 · 3/0 (0.25) | 0.899 · 2/0 (0.50) |
| + AR(1) auto-switch (= final) | 0.869 · 0/0 | 0.730 · 9/4 (0.27) | 0.700 · 0/0 | 0.899 · 0/0 |

## Interpretation

1. **The profiled-quadratic best-subset proposals are what matter.**
   - Joint state estimation on its own does nothing: refitting the weak answer with joint states changes the
     exact-recovery outcome of 0 of 576 runs.
   - Adding the proposals is the single largest step on white noise: +8.7 points on sealed, 0 → 25/40 on the
     hard 40, and +3.5 points on dysts.
   - Taking them out of the final model costs 32 runs and gains none (p = 5×10⁻¹⁰).
   - Exhaustive ranking matters, not just having proposals. Replacing it with forward stepwise selection on the
     same quadratic loses almost as much: 28/0 overall, with the same counts as "no proposals" on the hard 40 and
     on sealed. This supports the novelty claim in WRITEUP: the best-subset search on the profiled quadratic is new.
     The joint objective is not.
2. **Self-calibration is what makes the proposals safe under correlated noise.**
   - On the coloured archive, proposals with a fixed high-frequency σ and φ = 1 overfit every run: the exact
     rate falls 0.64 → 0.00.
   - Self-calibration restores it (56/0).
   - On white noise σ/φ self-calibration has no measurable effect (0/0 leave-one-out on sealed, hard 40 and
     dysts).
   - In the final model, coloured runs go to the AR(1) likelihood, which has its own calibration of a and σ. So
     the white-path leave-one-out also shows 0/0 on coloured, and that is expected.
   - Removing φ alone changes one coloured run.
3. **The complexity prior is small but consistent.** It never loses (10/0 pooled, p = 0.002), with gains spread
   across all four sets. This matches the dev evidence it was frozen on.
4. **The AR(1) auto-switch helps on coloured noise but not significantly at this sample size.**
   - Coloured: 0.674 → 0.730, 9/4 (p = 0.27). On all 800 coloured runs the frozen test found 23/1.
   - It is never triggered on sealed or hard 40, and on 1 % of dysts runs, where it changes no answer.
5. **Multiple starts, auto sub-steps and Huber are not supported by this ablation.**
   - **Multiple starts:** a single start changes nothing on white noise and is better on coloured, 6/2
     (p = 0.29).
   - **Auto sub-steps:** 7/4 overall. The only sign of benefit is on dysts (6/2 leave-one-out, 7/1 on the
     ladder); its design reason, discretisation error at low noise on stiff held-out systems (Chen–Lee), is not
     covered by these sets.
   - **Huber loss:** it was introduced for heavy-tailed t₃ noise (dev 0.78 → 0.905), which is not ablated here.
     On white noise it is neutral (1/1).
6. **Huber appears to hurt under AR(1) noise (negative result).** With the Gaussian loss, the final model scores
   0.831 against 0.730 on the coloured subsample. Gaussian wins 9 runs and loses none (p = 0.004).
   - The oracle noise variant (true a = 0.8 and true σ) reaches 0.820 with 8/0 wins, so most of the remaining
     coloured gap is noise mis-calibration, not search.
   - A plausible mechanism: Huber with c = 2 on the whitened residuals down-weights exactly the large innovations
     that the AR(1) likelihood needs.
   - This was found on a test archive. It must not be used to change the frozen model. The right follow-up is a
     dev-only check of `huber=inf` under `noise="ar1"` on `instances_colored.npz` / `dev_colored_mid.npz`,
     with t₃ dev as a guard, and then a fresh coloured test archive.
7. **The oracle noise level adds nothing on white noise** (sealed 1 run, dysts 1 run, hard 40 none), so the
   self-calibrated σ is as good as knowing σ. Under coloured noise the oracle gains 8/0, as described above.

**Summary for the paper.**

| Role | Components |
| --- | --- |
| Essential | Profiled-quadratic best-subset proposals; σ/φ self-calibration, but only under correlated noise |
| Small and consistent | Complexity prior |
| Helps in its regime, not significant here | AR(1) switch |
| No evidence on these test sets | Multiple starts; auto sub-steps (weak dysts signal only); Huber (its t₃ case was not ablated) |
| Counter-productive under AR(1) | Huber loss |

## Caveats

- **Sample sizes:** the subsamples are 89–287 runs, so effects under about 5 runs are not resolvable.
- **Independence:** the four sets are separate archives (hard 40 is part of the confirmation archive), but runs
  within an archive share systems.
- **The no-self-calibration variant** uses `selfcal=False` in `joint_discover` only. `joint_colored` has no such
  switch, because its (a, σ) calibration is the model itself.
- **Oracle σ** is η·std(clean), with std(clean) estimated as std(y)/√(1+η²). On the white path the
  self-calibration step may still raise it.
- **Where each set ran:** sealed and hard 40 on the PC, coloured and dysts on the Spark. The code and archives are
  identical (md5 / sha256 checked).
