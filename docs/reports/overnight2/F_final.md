# Track F: the final merged model (2026-10-25)

`final_model.discover(y, dt, library=..., noise=...)` is one entry point over config C (`joint_discover`), the big and trig libraries (`joint_biglib`), the AR(1) noise model (`joint_colored`), the D-dimensional version (`joint_nd`), the complexity prior (`joint_squeeze`) and per-term confidence (`confidence`). No existing file was edited. The defaults were chosen with three crossed experiments on dev archives only, frozen by md5, and then tested once on every test archive.

## 1. Construction

- **One prior mechanism for every variant.** The frozen fit classes expose `k`, the number of terms. It is used only in the selection penalty: φ·k·log 3n, times λ under AR(1). `final_model` swaps these classes for subclasses whose `k` is the weighted count Σ w(term), with w = max(0.1, 1 + 0.2·(degree − c)). With c = 1.5 this is exactly `joint_squeeze`'s degw prior. With the prior off, every variant is the frozen original.
- **Fidelity checks** (Spark against the PC's stored records, same seeds and starts). Every prior-off arm reproduces its frozen original **bit for bit**, with 0 changed answers in each case:
  - C on 400 white dev runs;
  - C + prior against `discover_sq` (`dev_deg2`), 400 runs, with identical score gaps;
  - degree-4 and degree-5 joint, 200 runs each;
  - `joint_nd`, 200 runs.

  So paired comparisons against the stored previous-best records are valid.
- **Libraries.** Degree 4/5 reuse `joint_biglib`'s beam search, which is also patched into `joint_colored`. Trig reuses `MixedLibrary`/`use_library`, also extended to `joint_colored`; each sin/cos term gets neutral weight 1. `core` fixes its library degree at import (`EQ_DEG`), so on a mismatch the public call reruns in a spawned process.
- **Automatic noise switch.** First, â (the lag-1 AR coefficient of the errors) is estimated as in `joint_colored`'s first calibration step: fit the best start with config C, then take the autocorrelation of its sample residuals. This adds about 1 s. If the mean of â over components exceeds the threshold, the AR(1) model is used; otherwise the white-noise model.
- **Outputs.** The fit returns the ambiguity flag (score gap < 2, or < 2λ under AR(1)) and per-term inclusion probabilities from the candidate pool. These are BIC weights; for poly3 they go through track C's frozen isotonic map (`results_final/confidence_isotonic.json`).

## 2. Protocol

- **Data.** Only dev archives were used:
  - `instances*.npz` (white, coloured, t₃, coarse);
  - the held-out design archive (`heldout.npz`, every 3rd run);
  - `trig_dev`, `nd_dev`;
  - a new dev-only archive `data/dev_colored_mid.npz` (AR(1) with ρ = 0.3 and 0.55, seed 20261501, 320 runs), made with `gen_colored_mid.py` to locate the switch threshold.
- **Decision rule, set before the runs.** A combination becomes the default only if it beats each component alone on dev: W > L in paired comparisons, with a two-sided sign test p < 0.05. Otherwise the simpler component is kept.
- **Two procedural notes.**
  - **Subset bug fixed early.** The archives cycle the noise level with index mod 4, so `--every 2/4` would silently drop noise levels. After the first set I stopped the campaign and switched to seed-block subsets that keep all four noise levels (`results_final/idx_*.json`). Only `dev_white` (even indices: 1% and 10%) kept the old subset, and `dev_white_odd` completes it to all 800 runs.
  - **nd dev extended once.** The first half of nd dev gave prior 5/0 (p = 0.06), short of the bar. I ran the other half once, with the decision on all 400 runs announced beforehand. That half also gave 5/0.

## 3. Dev crossed experiments

### (a) Prior × AR(1)

| Dev set | Runs | C | C + prior | AR(1) | **AR(1) + prior** | AR+p vs AR | AR+p vs C+p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White (all 800) | 800 | 0.881 | 0.890 | 0.881 | 0.891 | 9/1 (p = 0.02) | 1/0 |
| Coloured, ρ = 0.8 | 400 | 0.660 | 0.675 | 0.723 | **0.745** | **9/0 (p = 0.004)** | **36/8 (p = 3×10⁻⁵)** |
| Coloured, ρ = 0.55 (new) | 160 | 0.806 | 0.825 | 0.781 | 0.812 | | 5/7 |
| Coloured, ρ = 0.3 (new) | 160 | 0.819 | 0.856 | 0.825 | 0.856 | | 4/4 |
| All coloured | 720 | 0.728 | 0.749 | 0.758 | 0.785 | 19/0 (p = 4×10⁻⁶) | 45/19 (p = 0.002) |
| t₃ / coarse / held-out design | 200 / 200 / 294 | | 0.920 / 0.810 / 0.952 | | 0.925 / 0.800 / 0.949 | | 1/0, 1/3, 0/1 |

- **No double penalty.** On ρ = 0.8 dev, re-tuning λ's exponent with the prior on gives 1/3 (exponent ¼) and 2/2 (exponent ½) against the frozen ⅓. The prior reweights terms by degree with mean weight about 1 on true terms, while λ inflates every term uniformly, so they do not compound. λ is kept at exponent ⅓.
- **AR(1) only pays at strong correlation.** At ρ = 0.3 and 0.55, AR(1) gives nothing over the white model with prior (9/11 pooled). At ρ = 0.8 it is clearly better.

**â distributions and the switch.** Mean â over components:

| Set | Median | 95th percentile | Max |
| --- | --- | --- | --- |
| white | 0.000 | 0.021 | 0.135 |
| t₃ | 0.003 | 0.025 | 0.047 |
| coarse | 0.000 | 0.286 | **0.899** |
| held-out design | 0.000 | 0.122 | **0.781** |
| ρ = 0.3 | 0.269 | 0.295 | 0.324 |
| ρ = 0.55 | 0.508 | 0.534 | 0.576 |
| ρ = 0.8 | 0.757 (min 0.670) | 0.782 | 0.816 |

- **White-noise runs can still show high â.** These are model-misfit residuals: Chen–Lee, and coarse Chen/Lorenz at 1% noise. They are often high on all three components, so no per-component rule separates them from coloured noise.
- **Threshold.** Pooled dev exactness is flat at 0.858–0.860 for thresholds 0.05–0.6 and highest at **0.6**, which also sits in the â gap between ρ = 0.55 and ρ = 0.8. At 0.6 the switch fires on 100% of ρ = 0.8 runs, 0% of ρ ≤ 0.55 and white/t₃ runs, and 2% of coarse and held-out runs.
- **Pooled over all 2,214 3-D dev runs,** auto (threshold 0.6) against each component:

| Auto against | W/L | p |
| --- | --- | --- |
| C | 61/11 | 2×10⁻⁹ |
| AR(1) alone | 33/5 | 4×10⁻⁶ |
| C + prior | 36/10 | 2×10⁻⁴ |
| always AR(1) + prior | **13/12** | 1.0 |

  The switch is **not better than always using AR(1) + prior**; the two are tied. It was kept because it costs less, leaves white-noise answers identical to the frozen C + prior (the variant with the longest test record), and avoids AR(1) where it was, non-significantly, slightly worse (ρ = 0.55, coarse).

### (b) Prior × big library

The premise needed correcting first. In the 20-term library, the term-mean degree is 2.25, not 1.5; 1.5 is the midpoint of the degree range 0–3. The term-mean degree is 3.0 for degree 4 and 3.75 for degree 5. Four arms were compared (400 dev runs per library; the prior-off arm reproduces the frozen biglib runs exactly):

| Library | Off (frozen) | **c = 1.5 (uncentred)** | c = max degree / 2 | c = library mean | c = 1.5 vs off | Mean vs off |
| --- | --- | --- | --- | --- | --- | --- |
| Degree 4 | 0.845 | **0.868** | 0.863 | 0.845 | **9/0 (p = 0.004)** | 8/8 |
| Degree 5 | 0.840 | **0.860** | 0.848 | 0.810 | **9/1 (p = 0.02)** | **6/18 (p = 0.02)** |
| Deg 3 + trig (Thomas 160 + polynomial 200) | 0.981 / 0.830 | **0.988 / 0.860** | | | **8/1 (p = 0.04)** | |

- **The "known issue" doesn't hold up.** Centring at the library mean makes constant and linear terms nearly free (w ≈ 0.25–0.45) and over-selects low-degree terms; on degree 5 it hurts significantly. The uncentred prior wins because it penalises degree 4–5 terms more.
- **Caveat: the benchmark favours this by construction.** Every benchmark system has true degree ≤ 3, so a prior against degree 4–5 cannot lose here. A system with genuine quartic or quintic terms would be penalised, and none was tested.
- **Cost to the flag.** With the prior on, the flag fires less often (degree 5: 17% → 8%), so unflagged accuracy drops (0.964 → 0.910). The prior widens score gaps, so the fixed gap-2 flag catches fewer of the remaining errors.

### (c) Prior × D dimensions (degree-2 library)

| nd dev | Runs | Off (frozen) | c = 1.5 | c = 1.0 (degree-2 midpoint) | c = 1.5 vs off |
| --- | --- | --- | --- | --- | --- |
| First half | 200 | 0.890 | 0.915 | 0.915 | 5/0 (p = 0.06); 1.5 vs 1.0: 0/0 |
| Second half | 200 | 0.890 | 0.915 | — | 5/0 |
| **All** | 400 | 0.890 | **0.915** | | **10/0 (p = 0.002)** |

The gains are on coupled Lorenz (0.56 → 0.65) and hyperchaotic Rössler (0.91 → 0.94); Lorenz-96 is already near 1.0. c = 1.5 and c = 1.0 gave identical answers, so c = 1.5 was taken to keep one rule for every library.

## 4. Frozen defaults

| Setting | Value |
| --- | --- |
| Complexity prior | degw = 0.2, c = 1.5, on for every library (poly3/4/5, trig3, nd) and together with AR(1) |
| Noise model | `noise="auto"`: AR(1) iff mean â > **0.6** |
| AR(1) penalty | λ(a) = ((1+a)/(1−a))^{1/3}, unchanged |
| Sin/cos prior weight | 1 |
| Ambiguity flag | gap < 2 (2λ under AR(1)) |

md5 (`results_final/frozen.md5`, checked on the Spark by `test.sh` before the test runs; every file OK):

```
1e27d393ceae122e6f60d1048d24804d  final_model.py
96f155095590075af6693ae40fa21100  final_bench.py
91c85f7d0026b986ddbd2f1a7cc33b9a  joint_discover.py
a8d87059bf4e1624ee0c9596aac5afd7  joint_colored.py
daca8b9ab2d4a47d1fbeaac43e680906  joint_biglib.py
c07bbc84eb8ed52b683372e5213a3bfa  joint_nd.py
9e93185bf2c28a807d7be0555f594334  joint_squeeze.py   (reference; the prior is applied via final_model)
b10a1a33a607985de9f85b6115a70be5  confidence.py
f7ff1eec0338b10f3b706f5f61621c29  core.py
999078f84fa0595a551c9eaf912dc75b  select_k.py
3cd55e40f93604aae4ce36ab7f1d46cb  flow_discovery.py
cfb8ca29a3352dd860940bafd217c08c  results_final/confidence_isotonic.json
```

(It also covers subset.py, heldout_systems.py, gen_trig.py and nd_systems.py.)

## 5. Frozen test pass (one pass, public API, starts as in the earlier benches)

Paired by run index against each archive's previous best record. W/L is the final model against the previous best, with a two-sided sign test.

| Archive | Runs | Previous best | Prev. | **Final** | W/L | p | Flagged | Unflagged acc. | AR(1) chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sealed | 800 | C + prior | 0.889 | **0.889** | 0/0 | 1 | 12% | 97.3% | 0% |
| Confirmation | 800 | C + prior | 0.886 | **0.886** | 0/0 | 1 | 12% | 96.0% | 0% |
| Held-out systems, new seeds | 880 | C + prior | 0.943 | 0.941 | 1/3 | 0.62 | 3% | 95.3% | 2% |
| Hard 40 (20%, N = 2000) | 40 | C + prior | 28/40 | **28/40** | 0/0 | | | | DEDi 21/40; 7/0 against DEDi |
| **Coloured, new seeds** | 800 | AR(1) | 0.734 | **0.761** | **23/1** | **3×10⁻⁶** | 14% | 84.6% | 100% |
| **Heavy-tailed t₃, new seeds** | 800 | C | 0.889 | **0.905** | **14/1** | **0.001** | 10% | 97.5% | 0% |
| Coarse ×4, new seeds | 800 | C | 0.790 | 0.784 | 4/9 | 0.27 | 14% | 89.6% | 2% |
| Degree 4 (sealed, every 3rd) | 267 | biglib | 0.873 | 0.876 | 1/0 | 1 | 10% | 95.4% | 0% |
| Degree 5 (sealed, every 3rd) | 267 | biglib | 0.861 | 0.865 | 1/0 | 1 | 8% | 92.3% | 0% |
| Deg 3 + trig, polynomial systems | 134 | biglib | 0.881 | 0.896 | 2/0 | 0.5 | 13% | 98.3% | 0% |
| Deg 3 + trig, Thomas | 160 | biglib | 0.994 | 0.988 | 0/1 | 1 | 2% | 99.4% | 0% |
| **D = 4–10 (nd test)** | 400 | joint_nd (C) | 0.897 | **0.917** | **8/0** | **0.008** | 12% | 98.3% | 0% |
| **Pooled (11 archives, hard 40 not counted twice)** | 6,108 | | | | **54/15** | **3×10⁻⁶** | | | |

**Gains**
- **Coloured noise:** the prior now acts together with AR(1), mostly fixing Lorenz, Rössler and Chen at 5–10%. Against config C: 0.674 → 0.761.
- **Heavy-tailed noise:** the prior was never tested on this archive before; the gain is spread over Lorenz, Sprott and Rabinovich–Fabrikant.
- **D dimensions:** hyperchaotic Rössler 0.812 → 0.887, coupled Lorenz 0.675 → 0.700.

**Unchanged.** Sealed and confirmation, as designed: the switch never fired, and the answers equal the frozen C + prior. The big and trig libraries gain only +1/+1/+2 on these test subsets, which is not significant at n = 134–267, although dev showed 9/0 and 9/1.

## 6. Combinations that hurt (small, none significant)

- **Coarse sampling, 4/9 (p = 0.27).** Two losses come from the switch: Chen at 1% and Sprott D at 10%, where coarse discretisation misfit produces â > 0.6. The other seven come from the white-noise model with the prior, spread over Lorenz (2), Rössler, Sprott A/C/D and Rabinovich–Fabrikant, mostly at N = 500. The 4 wins are Rössler, Sprott D/E and Rabinovich–Fabrikant. The prior had never been checked against plain C on coarse data: track B's guards were white dev and held-out design, and in this track's coarse dev set it appeared in both arms (C + prior and AR + prior), so it was never compared with plain C there. This is the one place where the merged model is not at least the previous best.
- **Held-out systems, 1/3.** All four changed runs are Chen–Lee at N = 500, where misfit-induced â > 0.6 switched on AR(1). Dev had the same signature (0/1 on the held-out design archive).
- **Thomas, 0/1** (b = 0.18, 1% noise).
- **On dev only:** centring the prior at the library mean degree (degree 5: 6/18), and the flag becoming less informative with the prior in big libraries.

## 7. Caveats

- **The pooled 54/15 is not 6,108 independent trials.** The degree-4, degree-5 and mixed-library test archives reuse the sealed instances with other libraries, and most of the pooled signal comes from coloured (23/1), t₃ (14/1) and nd (8/0).
- **The mixed-polynomial test subset covers only 1% and 10% noise.** It is `joint_sealed` every 6th run, inherited unchanged from the earlier biglib test so that runs pair.
- **The switch threshold was set on synthetic AR(1) data** with ρ ∈ {0.3, 0.55, 0.8}. Real errors with â ≈ 0.6 fall on either side, and model misfit can push â above 0.6 on white data. On dev, the switch is only tied with always using AR(1) + prior. `info["ar_estimate"]` is returned so users can see it.
- **Coloured noise is still the weakest setting:** 0.761, with 84.6% unflagged accuracy. Coarse sampling is next.
- **Per-term confidence uses track C's isotonic map,** which was fitted on config C candidates (white, poly3). Applied to the final model's pools it still behaves on white data:

  | Test archive | Brier | "Sure" terms all true | Same, runs with a wrong answer |
  | --- | --- | --- | --- |
  | Sealed | 0.0086 | 98.4% | 85% |
  | Confirmation | 0.0088 | 98.5% | 87% |
  | Held-out | 0.0102 | 95.2% | 19% |
  | t₃ | 0.0082 | 97.6% | 75% |
  | Coarse | 0.0170 | 88.2% | 46% |
  | Coloured | 0.0199 | 84.5% | 35% |

  It is not calibrated for those last settings, or for big libraries and nd, where probabilities are reported raw.
- **The degree prior is the one structural assumption:** low-degree terms are preferred. Every benchmark is consistent with it, so the big-library gains are partly by construction.
- **Untested combinations:** AR(1) with the degree-4/5 or trig libraries, and coloured noise in D ≠ 3 (AR(1) is not implemented in `joint_nd`).
- **Compute.** Everything ran on the Spark, 18 workers, about 2.5 h wall time: dev about 1.5 h, test about 1 h. Spark results match the PC's stored records exactly wherever both exist.

## Files

- **Code (new):**
  - `final_model.py`: entry point, prior mechanism, switch, confidence;
  - `final_bench.py`: dev arms and the frozen `final` arm;
  - `final_dev_report.py`, `final_test_report.py`: tables;
  - `gen_colored_mid.py`: dev-only mid-ρ archive;
  - `README_final.md`: usage.
- **Data (new, dev only):** `data/dev_colored_mid.npz`.
- **Results (`results_final/`):**
  - dev: `dev_*.json` with logs, `dev_report.txt`;
  - test: `test_*.json` with logs, `test_report.txt`;
  - freeze: `frozen.md5`, `frozen_check.txt`;
  - job scripts: `dev.sh` (stopped run), `dev2.sh`, `dev3.sh`, `test.sh`;
  - subsets: `idx_*.json`, `hard40_idx.json`;
  - `confidence_isotonic.json`.
