# Equation discovery from noisy data — progress report

2026-10-22

Follow-up implementation (2026-10-23): [Profiled Flow Discovery](FLOW_DISCOVERY.md)
uses projected short-trajectory sensitivities to propose term replacements,
then verifies them with nonlinear fits. Its separate report includes the
derivation, executable code, comparisons, regressions and novelty limitations.

## Summary

A tuning-free selector — best-subset weak SINDy with a noise-calibrated chi-square stopping rule — recovers the exact equations in 88% of runs at 5% noise, against 60% for the best tuned SINDy-family baseline. It matches MIOSR given the oracle number of terms, and holds up under coloured noise, heavy-tailed noise, coarse sampling and a 56-term library.

The main lesson is diagnostic: on chaotic attractors, standard SINDy fails because true terms nearly cancel and the library is nearly collinear, not because of noise bias. Some cases (Rabinovich–Fabrikant, and Lorenz/Chen/Rössler at 20% noise) still fail even when the true term count is given.

The open question is whether those failures are a limit of the data or of the weak-form statistic. Five independent research lenses converged on the same one-day diagnostic to settle it. A follow-up trajectory-based method (profiled flow, 2026-10-23) answers it for Lorenz and Chen at 20% noise: the weak statistic is the bottleneck, and 17/40 hard cases become recoverable.

Against a joint-state baseline (DEDi, Meissner & Glasner) on the 40 hardest cases (four chaotic systems, 20% noise), the hybrid recovers 16/40 and DEDi 21/40 (9 wins / 4 losses, p = 0.27), at about 2,000× the cost. So the hybrid is not yet state of the art on these hard cases.

**Update 2026-10-24.** A new tuning-free joint-state method, profiled joint flow, beats the hybrid on all 11 archives, frozen before testing. Sealed: 0.881 against 0.796 (70 wins / 2 losses). Held-out systems: 0.933 against 0.901. Heavy-tailed noise: 0.889 against 0.749. On the hard 40 it scores 25/40 against DEDi's 21/40 (6/2, p = 0.29) at about 1/800 of the cost. Details are in the section on profiled joint flow.

**Update 2026-10-25.** A degree-weighted complexity prior (borrowed from symbolic regression) improves config C on every frozen archive (pooled 36 wins / 6 losses, p = 3×10⁻⁶) and lifts the hard 40 to 28/40, against DEDi's 21 (7/0). The same method scales unchanged to 4–10-dimensional systems (0.897 exact against 0.37 for tuned STLSQ). An identifiability map shows that Rabinovich–Fabrikant is a data limit: it is 10/10 with N = 8000. Symmetry priors, exact subset search and forecast-skill selection add nothing. See "Overnight campaign 2". All variants are now merged into one frozen `final_model.discover` (see "Final merged model"). It matches or beats each archive's previous best, pooled 54/15, with the largest gains on coloured noise (0.761) and heavy-tailed noise (0.905). On the external dysts benchmark (77 systems, 49 new, 6,160 frozen runs) it scores 0.890 against 0.879 for config C (101/33) and 0.675 for weak best-subset SINDy.

## Starting point

The [original project](https://github.com/iamyanbo/Data-Driven-Discovery-of-Governing-Equations-via-Reservoir-Computing-Data-Augmentation) trained an echo state network on masked, noisy data, generated Monte Carlo trajectories, filtered them by uncertainty, and fed the augmented set to SINDy. Its own README reports marginal gains: Lorenz coefficients were nearly identical with and without augmentation.

The reason is structural. A reservoir trained only on the observed data cannot add information about the equation that the data lacked; at best it smooths. The experiments below also show the real loss happened at term selection, which augmentation does not touch.

## Literature landscape (October 2026)

Sparse regression remains the most noise-robust family; pretrained and LLM-based methods are active but weaker under noise or crowded.

| Family | Representative work | Status for this project |
| --- | --- | --- |
| Weak-form SINDy | [WSINDy](https://arxiv.org/abs/2005.04339), [Fourier Weak SINDy](https://arxiv.org/pdf/2604.20141) | Standard noise-robust baseline |
| Ensemble / Bayesian SINDy | [E-SINDy](https://arxiv.org/abs/2111.10992), [UQ-SINDy](https://arxiv.org/pdf/2107.02107) | Baselines; posterior over supports exists |
| Mixed-integer best subset | MIOSR (Bertsimas & Gurney 2023) | Closest optimiser to our method |
| Weak-form parameter estimation | [WENDy](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002818/) | Source of our residual covariance; known model only |
| Preprocessing before SINDy | [Koopman/EDMD upsampling](https://arxiv.org/abs/2607.29036) | Occupies the old augmentation idea |
| Measurement-error correction | [PSI-SINDy](https://arxiv.org/html/2610.11486) | Hermite debiasing occupied |
| Active learning | [Larrañaga, Fasel, Brunton](https://arxiv.org/abs/2606.12182) | Occupied for forecast-variance sampling |
| Pretrained models | ODEFormer, [FIM-ODE](https://arxiv.org/abs/2602.08733) | Degrades at SNR ≤ 20 dB per [MDBench](https://arxiv.org/abs/2509.20529) |
| LLM-guided search | [LLM-SRBench](https://arxiv.org/pdf/2504.10415), [LLM-ODE](https://arxiv.org/abs/2603.20910) | Crowded; best 31.5% on memorisation-resistant set |
| Library learning | [AutoSINDy](https://arxiv.org/abs/2605.09696), [SINDy-KANs](https://arxiv.org/html/2603.18548v1) | Crowded |

Most of these were checked at abstract level only.

## Benchmark setup

Each run gives a method one noisy 3D trajectory and scores whether it recovers exactly the right terms; 800 runs per setting.

- **Systems (10):** Lorenz, Rössler, Chen, Halvorsen, Sprott A–E, Rabinovich–Fabrikant. All polynomial, degree ≤ 3, starting from random points on the attractor.
- **Noise:** 1, 5, 10, 20% of each coordinate's standard deviation. Variants: white Gaussian (base), AR(1) coloured (ρ = 0.8), Student-t with 3 degrees of freedom, and 4× coarser sampling.
- **Data length:** N = 500 or 2000 samples; 10 seeds per cell.
- **Library:** monomials up to degree 3 (20 terms), plus a degree-5 stress test (56 terms).
- **Metrics:** exact support recovery (main), F1 over terms, relative coefficient error.
- **Baselines:** SINDy with smoothed finite differences, weak SINDy, ensemble weak SINDy, pysindy's weak SINDy, pysindy MIOSR (Gurobi). Thresholds are tuned on the test set (generous to baselines), and we also report a per-run oracle threshold as an upper bound.

## Diagnosis: why SINDy-family methods fail

On Lorenz at 5% noise, every SINDy variant recovers the exact equations in 0% of runs, even with the best possible threshold for each run. Noise bias is not the cause; term selection is.

- **Noise bias ruled out.** Replacing monomials with Hermite polynomials makes every library term unbiased under Gaussian noise. It changed exact recovery by at most 0.03 in any cell.
- **Cancellation.** In Lorenz, each true term contributes 2–5× the size of the derivative itself (e.g. 4.9× for xz in dy/dt), so large terms cancel. Thresholding on coefficient size cannot tell true large terms from spurious large ones.
- **Collinearity.** On the attractor, the degree-3 library is nearly collinear, so many wrong combinations fit almost as well as the true one.
- **Search fixes most of it.** Exhaustive best-subset search, given the true number of terms, recovers Lorenz exactly in 90% of runs at 5% noise (0% for STLSQ). The problem therefore reduces to choosing the number of terms.

## Method: best-subset weak SINDy with a calibrated chi-square rule

The method picks the smallest set of terms whose leftover error is indistinguishable from noise, with no threshold to tune.

1. **Weak form.** Integrate against K overlapping compact test functions φ_k, so no derivatives of noisy data are needed:

   $$G_{km} = \int \phi_k(t)\,\theta_m(x(t))\,dt, \qquad b_k = -\int \dot\phi_k(t)\,x(t)\,dt$$

2. **Candidate supports.** For each equation and each size k ≤ 5, find the best k-term subset: exhaustive search for 20 terms, beam search with swap refinement for 56 terms (matches exhaustive in 96% of cases).
3. **Residual covariance.** For each candidate coefficient vector ξ, linearise how measurement noise enters the weak residual (the WENDy construction), giving L_l = −δ_jl Φ̇ − Φ diag(∂f_j/∂x_l):

   $$C(\xi) = \sum_l \sigma_l^2\, L_l L_l^\top, \qquad \chi^2(k) = r_k^\top C^{-1} r_k$$

4. **Self-calibrated scale.** Estimate the noise scale from the largest model, c = χ²(k_max)/(K − k_max), so a wrong noise estimate or non-white noise cancels out.
5. **Stopping rule.** Choose the smallest k with χ²(k)/c ≤ (K − k) + 3√(2(K − k)).

With the true term count, χ² sits at its degrees of freedom (about 200); one term short, it is 10–1000× larger. Standard BIC fails here (0.01–0.21 exact recovery) because overlapping test functions make residuals correlated, so it keeps adding terms.

## Results

At 5% noise the calibrated chi-square rule leads the best tuned baseline in all five settings, by 9 to 55 points, and ties MIOSR given the oracle term count.

Exact equation recovery, 5% noise, N = 2000, 100 runs per setting:

| Setting | Best tuned baseline | MIOSR, oracle sparsity | Calibrated chi-square |
| --- | --- | --- | --- |
| White Gaussian | 0.60 | 0.89 | **0.88** |
| Coloured AR(1) | 0.54 | 0.66 | **0.63** |
| Student-t (heavy tails) | 0.60 | not run | **0.89** |
| 4× coarser sampling | 0.50 | not run | **0.63** |
| Degree-5 library (56 terms) | 0.33 | not run | **0.88** |

The uncalibrated chi-square rule collapsed to 0–24% under coloured and heavy-tailed noise, which motivated the self-calibration. At 20% noise and N = 500 the new rule reaches 59% on white noise, against 17% for the tuned baseline and 58% for oracle MIOSR. In the 56-term library at N = 500, weak SINDy and pysindy recover nothing, while the new rule reaches 78%.

## Novelty and limitations

Every ingredient is known; the combination appears unpublished, so the honest rating is low–medium novelty as a method.

| Ingredient | Prior work |
| --- | --- |
| Weak-form regression | [WSINDy](https://arxiv.org/abs/2005.04339) |
| Residual covariance from test functions | WSINDy (as a regression weight), [WENDy](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002818/) (known-model estimation) |
| Best-subset search | MIOSR |
| Information criteria for SINDy | [Mangan et al. 2017](https://arxiv.org/pdf/1701.01773) (independent residuals) |
| Chi-square discrepancy, scale from full model | Morozov's principle, [Mallows' Cp](https://en.wikipedia.org/wiki/Mallows's_Cp) |

What appears new is using the WENDy covariance with a self-calibrated chi-square test to choose the number of terms. Searches of WSINDy, WENDy, MIOSR, PSI-SINDy and [projection-based selection](https://arxiv.org/pdf/2512.14404) found no match, but several were read at abstract level only. [Gallo et al.](https://arxiv.org/abs/2607.18490) already link identifiability to the library's moment matrix on the attractor, which overlaps the collinearity diagnosis.

**Limitations**

- Gains come from choosing model size well; with the oracle size, MIOSR finds the same equations.
- Exhaustive search caps at about 20 terms; beam search is approximate, and equations with more than 5–6 terms are unreachable.
- All results are synthetic; baselines are partly our own implementations, tuned on the test set.
- Rabinovich–Fabrikant and the 20%-noise chaotic cases still fail, even with the true term count.

## New perspectives from five research lenses

Five independent agents, each from a different field, converged on one reframing: equation discovery is hypothesis testing between candidate term sets, and the question is how much information each statistic keeps about which set is right.

| Lens | Main contribution | Novelty (agent's rating) |
| --- | --- | --- |
| Classical system ID | [Piroddi & Spinelli 2003](https://re.public.polimi.it/handle/11311/246502) showed derivative-matching criteria mis-rank terms on oversampled data; simulation error fixes it. On chaos, a coefficient error's effect grows like e^{λT}, so short simulated segments (up to ~2.8 time units for Lorenz at 20% noise) may carry far more information than 0.1–0.3 weak-form windows. | Low–medium |
| Information theory | Compare weak-form and shooting Fisher information before fitting anything. The chi-square noncentrality δ² between the true support and its rival gives a per-dataset identifiability certificate and a predicted data requirement N* ≈ N · 8 log K / δ². | Medium |
| Bayesian statistics | With about 6,000 supports per equation, exact enumeration of the evidence gives the best achievable recovery rate from weak-form statistics, and a calibrated set of plausible equations instead of one answer. | Low–medium |
| Algebraic geometry | On attractor data, equations are identifiable only modulo polynomials that nearly vanish on the attractor. If g ≈ 0 then x·g ≈ 0 too, which explains why larger libraries get worse. Symmetry restriction may fix Rabinovich–Fabrikant but is [already published](https://arxiv.org/abs/2405.16756). | Low–medium |
| Ergodic theory | Invariant-measure identities equal the time weak form with one whole-record window, so they add no new information. Global attractor statistics are useful only as a cheap reranker that rejects candidates that blow up or collapse. | Low |

**Ruled out**

- Orthogonalising the library on the attractor: a basis change adds no information and breaks sparsity.
- Instrumental variables: they fix bias, and bias is not the problem.
- Model-X knockoffs: no power under near-collinearity; [already done for PDEs](https://arxiv.org/abs/2605.26631).
- Replica / message-passing theory: assumes random designs, not attractor data.
- State-space (invariant-measure) weak form: a subset of the time weak form.

Related prior art on trajectory likelihoods: Kaheman et al. 2022, [JSINDy](https://arxiv.org/pdf/2511.18555), [Meissner–Glasner](https://arxiv.org/abs/2406.06707), [Galioto–Gorodetsky](https://arxiv.org/abs/2212.13902).

## Update, 2026-10-23: follow-up results

### Profiled flow discovery partly answers the data-or-statistic question

[Profiled Flow Discovery](FLOW_DISCOVERY.md) seeds from the chi-square selector, then proposes term replacements using the sensitivity of short (40-sample) simulated segments to every library coefficient, with segment start states projected out, and verifies them with nonlinear fits and held-out segments.

Fresh data (seed 20261023), N = 2000, exact recoveries out of 10:

| System | Weak χ² selector, 5% | Flow, 5% | Weak χ² selector, 20% | Flow, 20% |
| --- | --- | --- | --- | --- |
| Lorenz | 10 | 9 | 0 | 5 |
| Chen | 10 | 9 | 0 | 10 |
| Rössler | 9 | 5 | 0 | 2 |
| Rabinovich–Fabrikant | 0 | 0 | 0 | 0 |
| **Total /40** | **29** | **23** | **0** | **17** |

- **Lorenz and Chen at 20% noise are a statistic limit, not a data limit.** The same data recover the equations once trajectories are used; median coefficient error drops from about 10–14% to 2–5%.
- **The gain comes from the proposals.** Restricting flow to the weak-form candidate pool (same fitter and budget) gives 1/40 at 20% noise instead of 18/40 on the development set.
- **Flow is worse at 5% noise**, so it does not replace the weak selector; see the hybrid below.
- **Rössler failures are mostly numerical** (7 of 10 fresh Rössler runs failed to integrate).
- **Rabinovich–Fabrikant is a proposal failure:** the true support never entered the evaluated pool, so this says nothing about a data limit yet.

### Hybrid: keep the weak answer unless trajectories strongly reject it

Neither method dominates, so the hybrid (`archive/hybrid_bench.py`) runs both. The weak selector's answer is refitted on flow's training segments and scored on its held-out segments (`flow_discovery.discover(..., extra_models=[weak])`). It is kept unless its trajectory score (whitened RSS + k log n, a BIC) is worse than flow's best by more than 10, Kass and Raftery's "very strong evidence". If flow fails numerically, the weak answer stands.

The threshold was chosen after viewing the development runs; a first rule set before any results (keep the weak answer unless it falls outside flow's one-standard-error plausible set) gave 0.790 / 0.780 / 0.776. The confirmation set below was generated after the rule was frozen and not looked at before.

Exact recovery, all 10 systems, 100 runs per cell, confirmation set (seeds 20261103/4):

| Noise, N | Weak χ² selector | Flow alone | Hybrid (BIC 10) |
| --- | --- | --- | --- |
| 1%, 500 | 0.89 | 0.66 | 0.87 |
| 1%, 2000 | 0.96 | 0.79 | 0.96 |
| 5%, 500 | 0.76 | 0.60 | **0.81** |
| 5%, 2000 | 0.87 | 0.77 | 0.88 |
| 10%, 500 | 0.63 | 0.62 | **0.70** |
| 10%, 2000 | 0.73 | 0.78 | **0.82** |
| 20%, 500 | 0.60 | 0.51 | 0.62 |
| 20%, 2000 | 0.59 | 0.69 | **0.75** |
| **All 800** | **0.754** | **0.677** | **0.801** |

- **Overall:** dev 0.759 → 0.806, first fresh set 0.750 → 0.796, confirmation 0.754 → 0.801.
- **Paired on the confirmation set:** 42 runs won, 4 lost (sign test p = 5×10⁻⁹).
- **High noise, N = 2000 (10–20%):** 132 → 157 of 200.
- **Easy systems are preserved:** Halvorsen and Sprott A–E lose 1 run out of 480 in total.
- **Remaining gaps:** Rössler (flow fails to integrate in 15 of 80 runs) and Rabinovich–Fabrikant (7/80, no change).
- **Cost:** about 3 s per run on one core, against well under a second for the weak selector alone.

Reproduce: `python archive/hybrid_bench.py --data data/hybrid_confirm.npz --output results_hybrid/<new>.json` (results in `results_hybrid/`).

### Joint-state baseline: DEDi (Meissner & Glasner) on the hard cases

JSINDy has no public code, so the joint-state baseline is the authors' [DEDi](https://github.com/TeddyMeissner/DEDi). It is used unmodified except for two Windows-only substitutions: the sksparse Cholesky is replaced by an exact banded/Schur stand-in (error 4×10⁻¹⁶), and IPython display by a no-op. It used the same 20-term library and the authors' every-third-sample validation mask.

Cases: the four systems where the hybrid is weakest (Lorenz, Chen, Rössler, Rabinovich–Fabrikant), at 20% noise and N = 2000. All 10 confirmation-set runs per system were used, compared run-for-run with the same instances. These systems were picked *because* they are hard for our methods, so this is a stress comparison, not an average.

Hyperparameters: a 2×2 grid (λ ∈ {10⁻², 1}, R ∈ {10⁻⁴, 10⁻²}), chosen by validation error, instead of the paper's 35 pairs; the full grid would have cost about 9× more. DEDi chose λ = 10⁻², the smallest available value, in every run, so a wider grid might help it.

| System (20%, N = 2000) | Weak χ² | Flow | Hybrid | DEDi |
| --- | --- | --- | --- | --- |
| Lorenz | 0/10 | 5/10 | 5/10 | **8/10** |
| Chen | 0/10 | 9/10 | **9/10** | 7/10 |
| Rössler | 0/10 | 2/10 | 2/10 | **6/10** |
| Rabinovich–Fabrikant | 0/10 | 0/10 | 0/10 | 0/10 |
| **All 40** | **0** | **16** | **16** | **21** |

- **Paired against the hybrid:** DEDi wins 9 runs and loses 4 (sign test p = 0.27). It is better on Lorenz and Rössler and worse on Chen; the difference is not significant at n = 40.
- **Coefficients:** among the 12 runs both recover exactly, DEDi's coefficient error is lower in 8. Median error over each method's exact runs: 0.7% for DEDi, against 1.4% for the hybrid.
- **Rabinovich–Fabrikant defeats both:** DEDi produces 157 false positives over 10 runs, against 26 for the hybrid. This is consistent with a data or identifiability limit there, not just a proposal failure.
- **Cost:** about 1.5 h per run for DEDi (CPU, 4-pair grid; on the full 35-pair grid it would be about 13 h), against about 3 s for the hybrid, so DEDi is roughly 2,000× slower.

**Reading.** At the hardest setting, joint state estimation recovers more than our trajectory-segment flow, which is direct evidence that the remaining Lorenz/Rössler failures are a statistic limit, not a data limit. The hybrid is not state of the art on these cases. A cheap fix to try: run DEDi (or a joint-state refit) only on the hybrid's shortlist when the weak and flow answers disagree.

Reproduce: `python benchmarks/dedi_bench.py --ext <folder with shim/ and ext/DEDi> --systems lorenz chen rossler rabinovich --noise 0.2 --n 2000 --grid small --output results_hybrid/<new>.json`. Use `--offset` to split a run across machines. Results: `results_hybrid/dedi_hard20.json` (first 2 runs, this PC) and `dedi_hard20_rest.json` (38 runs, DGX Spark).

### Profiled joint flow: joint states and sparse coefficients (2026-10-24)

`eqdisc/joint_discover.py` estimates the denoised states u and the sparse coefficients C together, keeping the hybrid's fast proposal machinery.

**Objective.** Data misfit Σ|(u−y)/σ|² (Huber, c = 2) plus a model term Σ|(u_{k+1} − u_k − dt·f(m_k; C))/(ρσ)|² with midpoint m_k. Latent states live on a sub-grid with s steps per sample, so discretisation error stays below the noise. ρ = 0.03.

**Fit for a fixed support.** Levenberg–Marquardt. The normal matrix is banded in the states plus a dense coefficient border, so each step is one banded Cholesky plus a Schur complement. That is linear in N; a fit takes milliseconds.

**Proposals.** Profile the states out at the current fit to get a quadratic in all 60 coefficients. Rank the subsets of each equation in it, with the other equations' active columns projected out. Take the product of each equation's top two, then refit the ten best jointly; repeat for two rounds.

**Selection.** Every third sample is held out of the data term. The score is data + model + held-out misfit + φ·k·log(3n), a BIC whose penalty is inflated by φ = 1 + 2Σρ_k from the residual autocorrelation. Starts are the weak, hybrid and top-3 flow answers, never the truth.

**Self-calibration.** σ is taken as the larger of the high-frequency noise floor and 1.4826·MAD of the best start's residuals. The sub-step count is set automatically from the spectrum: s = max(⌈256 f_peak dt⌉, ⌈64 f99 dt⌉, 2), capped at 12. A final refit at ρ = 0.001 removes coefficient shrinkage.

**Ambiguity flag.** If the best and second-best scores are within 2, the answer is flagged.

**Protocol.** Every design choice was made on the dev archives and the first held-out-systems archive; these are marked "design" below. The test archives use new seeds, and the configuration was frozen before they were run. The 11 held-out systems are polynomial chaotic systems never used to develop any method: Lü, Shimizu, Rucklidge, Genesio, Dadras, Sprott F/G/H, Burke–Shaw, Rikitake and Chen–Lee. Each was checked chaotic by divergence of nearby trajectories. One caveat: the sealed set and held-out v1 had been looked at once with an earlier version (v1) before the final changes.

| Archive (exact-support rate) | Runs | Weak χ² | Hybrid | **Joint** | W/L vs hybrid | Flagged | Accuracy if unflagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev (design) | 800 | 0.759 | 0.806 | **0.881** | 63/3 | 14% | 97% |
| dev coloured AR(1) (design) | 800 | 0.650 | 0.221 | **0.679** | 376/10 | 8% | 73% |
| dev heavy-tailed t₃ (design) | 800 | 0.761 | 0.743 | **0.904** | 130/1 | 13% | 98% |
| dev coarse ×4 (design) | 800 | 0.671 | 0.490 | **0.797** | 259/13 | 17% | 91% |
| held-out systems (design) | 880 | 0.909 | 0.897 | **0.943** | 46/5 | 4% | 96% |
| **sealed (test)** | 800 | 0.752 | 0.796 | **0.881** | 70/2 (p = 10⁻¹⁸) | 15% | 98% |
| **confirmation (test)** | 800 | 0.754 | 0.801 | **0.868** | 62/9 (p = 10⁻¹⁰) | 18% | 99% |
| **held-out systems, new seeds (test)** | 880 | 0.907 | 0.901 | **0.933** | 37/9 (p = 4×10⁻⁵) | 4% | 95% |
| **coloured, new seeds (test)** | 800 | 0.645 | 0.224 | **0.674** | 373/13 | 9% | 73% |
| **heavy-tailed, new seeds (test)** | 800 | 0.755 | 0.749 | **0.889** | 118/6 | 13% | 98% |
| **coarse, new seeds (test)** | 800 | 0.654 | 0.472 | **0.790** | 266/12 | 17% | 91% |

Joint beats both the weak selector and the hybrid on every one of the 11 archives. The gains concentrate where expected. On the sealed set at 10% noise, joint scores 0.87–0.90 against the hybrid's 0.72–0.78; at 20% noise, 0.75–0.87 against 0.60–0.73. Per system on the sealed set (out of 80), hybrid → joint: Lorenz 49 → 72, Rössler 42 → 71, Chen 66 → 75. On held-out systems, Chen–Lee goes from 21 to 39 of 80; no held-out system gets worse by more than one run.

**Hard 40 against DEDi** (20% noise, N = 2000, same instances):

| System | Hybrid | DEDi | **Joint** |
| --- | --- | --- | --- |
| Lorenz | 5 | **8** | 7 |
| Chen | 9 | 7 | **10** |
| Rössler | 2 | 6 | **8** |
| Rabinovich–Fabrikant | 0 | 0 | 0 |
| **All 40** | 16 | 21 | **25** |

- **Paired against DEDi:** joint wins 6 runs and loses 2 (p = 0.29). That is better than DEDi, but not significantly at n = 40; the honest claim is "at least on par".
- **Coefficients:** over the 19 runs both recover exactly, median error is 0.81% for joint against 0.73% for DEDi; joint is lower in 13 of the 19.
- **Cost:** a median of 6.7 s per run for joint, including the weak and flow stages, against 5,499 s for DEDi. That is about 800× faster, which makes joint-state discovery cheap enough to run on every case rather than only on a shortlist.

**What changed between versions** (dev / held-out / coarse, design sets):

| Version | Change | Effect |
| --- | --- | --- |
| v1 | s = 2, fixed σ, BIC | 0.877 / 0.899 / –; lost to weak at 1% noise on held-out (discretisation error) |
| A | Automatic sub-steps + σ/φ self-calibration | 0.880 / 0.926 / 0.828; coloured 0.25 → 0.66 |
| B | + Huber loss | heavy-tailed 0.78 → 0.905 |
| C (frozen) | + f99 sub-step rule (Chen–Lee) + stiff final refit | 0.881 / 0.943 / 0.797; coefficient error on dev 0.42% → 0.30% |

C traded about 3 points on coarse data (0.828 → 0.797) for the held-out gain. Coarse sampling also costs about 3.5× more time (11 s per run), because the f99 rule asks for more sub-steps.

**Rabinovich–Fabrikant is an identifiability limit, not a search failure.** At ≥ 5% noise, the true support's held-out misfit is within ±8 (out of about 2,000) of the smaller wrong models the method finds. No selector based on fit can separate them. It is the one system where every method fails: weak, flow, hybrid, DEDi and joint.

**Limitations**

- **Coloured noise:** joint is only about 3 points ahead of weak, and unflagged accuracy is just 73%. φ inflation corrects the penalty but not the likelihood; proper AR(1) prewhitening of the data term is the obvious next step.
- **The flag** catches about 85% of wrong answers on white-noise sets, but much less under coloured noise.
- **Scope:** systems are 3-D only, with a fixed degree-2 library (20 terms per equation). The 56-term library has not been re-run with joint.
- **Novelty:** the joint objective itself is in the DEDi/JSINDy family. What is new is the profiled-quadratic proposal step on top of a banded LM. It turns a 1.5-hour method into a seconds-scale one, while self-calibration makes it tuning-free. This needs checking against the full texts of JSINDy and DEDi before it is claimed.

**Novelty check (2026-10-24).** I read DEDi, JSINDy, ODR-BINDy, SIDDS, DAHSI and Zhang–Cao–Carroll in full; the rest was checked from abstracts.

Already done by others:
- **Objective:** joint states with a soft dynamics penalty (ODR-BINDy [arXiv:2507.23426](https://arxiv.org/abs/2507.23426), DEDi, DAHSI, Ramsay et al. 2007).
- **Midpoint rule:** DEDi.
- **Banded-plus-border solver, linear in N:** SIDDS ([arXiv:2203.13837](https://arxiv.org/abs/2203.13837)), and DEDi's sparse Cholesky.
- **Profiling out the states to get a quadratic in the coefficients:** ODR-BINDy App. A.2 uses it to score models; Zhang, Cao & Carroll 2015 use it for one lasso pass.
- **Every-third-sample hold-out and BIC:** DEDi.
- **Held-out AIC/BIC:** Mangan 2017.
- **Score-gap ambiguity:** Burnham–Anderson ΔAIC; DAHSI reports runner-up models.

Not found in prior work on equation discovery:
- **Exhaustive best-subset proposals on the profiled quadratic, iterated with joint refits.** The ingredients are classical (Lawless & Singhal 1978 screened subsets on a quadratic for GLMs).
- **Automatic calibration:** σ, sub-steps from the spectrum, and a BIC penalty inflated by residual autocorrelation. ODR-BINDy is given the true σx and hand-sets σt.
- **Huber loss and an automated ambiguity flag inside the pipeline.**

Neither DEDi nor JSINDy reports end-to-end runtime. ODR-BINDy reports 40–90 s for Lorenz at N = 1000.

**Implication.** The paper would claim "the profiled quadratic, used so far only to score models, used instead to drive global subset proposals, plus calibration that removes σ and ρ tuning". The benchmark that decides it is ODR-BINDy at its own settings (given the true σx), not DEDi.

Reproduce: `python benchmarks/joint_bench.py --data data/joint_sealed.npz --workers 11 --output results_joint/<new>.json`, then `python benchmarks/final_report.py` (reads `results_joint/C_*.json`). Held-out systems are in `eqdisc/heldout_systems.py`; variant archives come from `benchmarks/gen_variants.py --kind white|colored|student|coarse [--heldout] --seed S`.

### ODR-BINDy baseline and four extensions (2026-10-24/25)

**ODR-BINDy on the hard 40.** The official code ([llfung/ODR-BINDy](https://github.com/llfung/ODR-BINDy)) is MATLAB only, so `benchmarks/odr_bindy.py` ports it to Python, keeping the control flow, option defaults and evidence formula. The one substitution is the solver: a sparse Levenberg–Marquardt with the same tolerances replaces MATLAB's lsqnonlin trust-region solver. The port's Jacobian matches finite differences to 6×10⁻¹⁰, and its evidence Hessian matches a numerical profile to 2×10⁻⁴. As in the paper's protocol, ODR-BINDy is given the true noise level; σ_y is set to the paper's Lorenz (10⁻³) and Rössler (5×10⁻³) values.

| System (20%, N = 2000) | ODR-BINDy σ_y = 10⁻³ | ODR-BINDy σ_y = 5×10⁻³ | ODR-BINDy 5×10⁻³, cap 1000 | Joint (C) | Joint C + prior | DEDi |
| --- | --- | --- | --- | --- | --- | --- |
| Lorenz | 1 | 6 | 8 | 7 | 10 | 8 |
| Chen | 0 | 0* | 7 | 10 | 10 | 7 |
| Rössler | 4 | 8 | not rerun | 8 | 8 | 6 |
| Rabinovich–Fabrikant | 0 | 0 | not rerun | 0 | 0 | 0 |

- **Chen\* was a port artifact.** At the MATLAB cap of 100 iterations, 7 of 10 Chen runs never removed a term, because my solver needs 400–2,000 iterations on those fits. With the cap at 1,000 (`results_odr/hard30_it1000_partial.log`), Chen goes from 0 to 7/10 and Lorenz from 6 to 8/10. Each run takes a median of about 8 h (6–12 h). The 10 Rössler runs were stopped unfinished, because Rössler had no cap failures in the first run.
- **Reading:** on Lorenz and Chen, ODR-BINDy at its best setting scores 15/20, against 17/20 for joint C and 20/20 for joint C + complexity prior, at roughly 8 h against 7 s per run. On Rössler, all methods score 8/10 except DEDi (6). ODR-BINDy's score also swings strongly with σ_y, which the user has to choose.

**Big libraries** (`eqdisc/joint_biglib.py`). The per-equation subset ranking on the profiled quadratic switches to a width-60 beam search when exhaustive search would exceed 250k subsets. Starts come from the weak selectors on the same library; flow starts are not available here. One frozen test pass:

| Library | Runs | Weak | **Joint** | W/L | Unflagged accuracy | Median s |
| --- | --- | --- | --- | --- | --- | --- |
| Degree 5 (56 terms/eq) | 267 | 0.730 | **0.861** | 36/1 | 97.7% | 7.4 |
| Degree 4 (35 terms/eq) | 267 | 0.745 | **0.873** | 34/0 | 98.2% | 2.8 |
| Degree 3 + sin/cos (26 terms/eq), polynomial systems | 134 | 0.769 | **0.881** | 15/0 | 97.4% | 2.2 |
| Same, Thomas attractor (true trig terms) | 160 | 0.988 | **0.994** | 2/1 | 99.3% | 2.1 |

Joint on the 20-term library scores 0.884 on the same sealed runs, so going to 56 terms per equation costs 1–2 points. The Rabinovich–Fabrikant score drops to 0 with big libraries, and Thomas is easy even for weak, so it shows only that trig terms are found, not a gain.

**AR(1) coloured-noise model** (`eqdisc/joint_colored.py`).
- **Noise model:** the exact AR(1) likelihood of the observation errors, per component. It stays banded, so the Schur solver still applies, and held-out samples are scored by their exact conditional likelihood given their neighbours.
- **Self-calibration:** the AR coefficient a is estimated from the best start's residuals (mean 0.77 against a true 0.8; 0.005 on white noise).
- **Penalty:** inflated by λ(a) = ((1+a)/(1−a))^{1/3}. This is a one-knob heuristic chosen on coloured dev data, not derived.

Frozen test:

| Set | Weak | Joint C | **Joint AR(1)** | vs C | Unflagged accuracy (C → AR) |
| --- | --- | --- | --- | --- | --- |
| Coloured, new seeds | 0.645 | 0.674 | **0.734** | 74/26 (p = 2×10⁻⁶) | 73% → 85% |
| White (sealed) | 0.752 | 0.881 | **0.881** | 0/0 | 97.8% |

There is no gain yet at N = 2000 with 10–20% coloured noise, and Lorenz and Chen stay around 0.5 under coloured noise.

**Attractor tiebreaker: negative.** For the 213 dev runs whose top candidates score within 4 points of each other, each candidate was simulated as an independent ensemble, run in parallel with a vectorised RK4. Three statistics compared each candidate with the data: energy distance on same-length runs, energy distance on the long-run attractor, and log-spectrum distance. Comparing the true model with a wrong one, the true model has the closer statistic 46%, 49% and 57% of the time respectively, which is chance level. No selection rule beat the score (0.573). Near-tied models differ by one or two small terms and produce indistinguishable attractors, which is the same identifiability wall as Rabinovich–Fabrikant.

**Experiment design: small positive.** For ambiguous runs, a quarter-length new noisy segment is collected from one of three restart points:
- a random point on the observed attractor ("record more");
- a random point in the expanded data box;
- the proposal, out of 256 restricted to those where every candidate stays bounded, that maximises the candidates' predicted disagreement.

Every candidate support is then refit on the new segment, and the summed score re-ranks them.

| Ambiguous runs | Before | Record more | Random box | **Designed** | Best possible |
| --- | --- | --- | --- | --- | --- |
| Dev (213) | 0.573 | 0.563 | 0.601 | 0.601 | 0.638 |
| **Sealed (238, frozen)** | 0.618 | 0.622 | 0.626 | **0.639** (fixed 9 / broke 4) | 0.672 |
| Sealed, flagged only (124) | 0.355 | 0.363 | 0.371 | **0.403** | 0.452 |

Recording more of the same trajectory does not help; leaving the attractor does. Design captures about 40% of the available headroom on the sealed set, but is not significantly better than a random restart (7/4). The headroom is small because the true model is often not among the near-tied candidates.

Reproduce: `benchmarks/odr_bench.py`, `benchmarks/biglib_bench.py` + `benchmarks/biglib_report.py`, `benchmarks/colored_bench.py` + `benchmarks/colored_report.py`, `archive/tiebreak_bench.py`, `archive/design_bench.py` (results in `results_odr/`, `results_biglib/`, `results_colored/`, `archive/results_tiebreak/`, `archive/results_design/`).

### Overnight campaign 2: five cross-field tracks (2026-10-25)

Each track had its own dev/test split and ran one frozen test pass after tuning on dev. Full reports are in `docs/reports/overnight2/`.

| Track | Idea borrowed from | Verdict | Headline (frozen test) |
| --- | --- | --- | --- |
| B. Complexity prior | Symbolic regression (PySR/Eureqa complexity costs) | **Positive** | Pooled 36 wins / 6 losses over 2,480 runs (p = 3×10⁻⁶). Hard 40: 28/40, against DEDi's 21 (7/0 head to head, p = 0.016) |
| D. Higher dimensions | — (scaling test) | **Positive** | D = 4–10: 0.897 exact against 0.37 for STLSQ with the best per-run threshold; 215 wins / 0 losses |
| E. Identifiability map | Experimental design / power analysis | **Positive** | Margin predicts exactness (AUC 0.985). Pre-registered: Rabinovich–Fabrikant 10/10 with N = 8000 at 5% noise |
| C. Certification | Optimisation (exact MIQP / branch-and-bound), conformal-style calibration | **Mixed** | Exact search changes no answers. Per-term "sure" labels are 97.8% all-true |
| A. Symmetry priors | Physics (equivariance) | **Diagnostic only** | Detection is reliable, but accuracy moves only 4 wins / 1 loss on 235 runs |
| E. Forecast-skill selection | Weather forecasting / ML validation | **Negative** | 0.866 against 0.881 (3/15, p = 0.008) |

**B. Where config C still loses, and the fix** (`eqdisc/joint_squeeze.py`).
- **Failure anatomy:** the true model was fitted directly on each of the 95 wrong dev runs. In 92 of them it scores worse than the winner even at its best fit; only 3 are search misses. So the remaining losses are about selection, not search.
- **Variants tried on dev:** ρ-annealing (0/1), add/drop/swap moves (1/0) and a Laplace-evidence score (1/1) do nothing.
- **What works:** a degree-weighted complexity prior. Each term's BIC cost becomes log(3n)·(1 + 0.2·(degree − 1.5)), so constant and linear terms get cheaper and cubic ones dearer, while the average penalty stays unchanged.
- **Frozen test** (strength 0.2, set on dev and guard-checked on all 800 dev runs and the held-out-systems design archive):

| Archive | Runs | C | **C + prior** | W/L |
| --- | --- | --- | --- | --- |
| Sealed | 800 | 0.881 | **0.889** | 8/2 |
| Confirmation | 800 | 0.868 | **0.886** | 19/4 |
| Held-out systems, new seeds | 880 | 0.933 | **0.943** | 9/0 |
| Hard 40 (20%, N = 2000) | 40 | 25/40 | **28/40** | Lorenz 7 → 10 |

It costs nothing extra. One caveat: a system made mostly of high-degree terms would be penalised, and none of the benchmarks has one. Rabinovich–Fabrikant, which has several cubic terms, is not hurt. The prior has not yet been combined with the big-library or AR(1) variants.

**D. Higher-dimensional systems** (`eqdisc/joint_nd.py`). This is config C unchanged in D dimensions. States are interleaved in time, so the banded Schur solve still applies, and the beam search from the big-library variant is used. The library is degree-2. One frozen pass over 400 test runs:

| System | D | Weak BIC | STLSQ (best λ per run) | **Joint** |
| --- | --- | --- | --- | --- |
| Hyperchaotic Rössler | 4 | 0.025 | 0.000 | **0.812** |
| Lorenz-96 | 5 | 0.025 | 0.662 | **1.000** |
| Coupled Lorenz | 6 | 0.075 | 0.113 | **0.675** |
| Lorenz-96 | 6 | 0.025 | 0.688 | **1.000** |
| Lorenz-96 | 10 | 0.000 | 0.400 | **1.000** |

Failures are one-for-one term swaps at N = 500 and 10–20% noise, mostly caught by the ambiguity flag. Runtime grows with D, to about 2 minutes per run at D = 10. The baselines are minimal re-implementations, not pysindy.

**E. Identifiability map** (`benchmarks/identifiability.py`). For each run, the margin between the true model and its nearest competitor on noise-free data predicts whether config C is exact: AUC 0.985 and 97.4% accuracy on sealed. The margin needs the truth, so it is a planning tool ("how much data do I need?"), not a selector.

Its pre-registered prediction for Rabinovich–Fabrikant held: 10/10 exact with N = 8000 at 5% noise and N = 16,000 at 10%. The predictions were about 2× too pessimistic. The 0/40 on the hard cases is therefore a data limit, not a method limit.

**C. Certification** (`benchmarks/bnb_subset.py`, `eqdisc/confidence.py`).
- **Exact search:** Gurobi and branch-and-bound subset search over 20 and 56 terms changed no answers, so the heuristic search is already optimal for its objective.
- **Per-term confidence:** BIC weights over the candidates, then recalibration. Terms labelled "sure" are all true in 97.8% of sealed runs (Brier 0.0095), and in 80.6% even when the overall answer is wrong.
- **Limit:** sure-plus-possible terms contain the full truth in only 12.9% of wrong runs.

**A. Symmetry priors** (`archive/symmetry.py`). Sign-flip symmetries become library masks.
- **Detection:** true symmetries are detected 135 out of 160 times, and false ones only 3 out of 1,709.
- **Why accuracy barely moves:** config C's wrong answers on Lorenz and Chen already respect the symmetry.
- **Results:** a gain on Chen–Lee (2/7 → 5/7), Rabinovich–Fabrikant stays at 0, and the cost is 4–6×. It is useful as a diagnostic ("these data support z → −z"), not as a default.

**E. Forecast-skill selection** (`archive/forecast_select.py`). Choosing among near-tied candidates by held-out forecast skill is worse than the score: 0.866 against 0.881. This agrees with the negative attractor-tiebreaker result, since near-tied models make indistinguishable forecasts.

**What this means.** The remaining errors are almost all model selection between near-collinear candidates. A cheap structural prior fixes a few of them, more data fixes the rest (the map gives the required N), and exact search and dynamical tiebreakers add nothing. The method now scales to D = 10 without changes.

### Final merged model (2026-10-25)

`final_model.discover(y, dt, library="poly3"|"poly4"|"poly5"|"trig3", noise="auto"|"white"|"ar1")` wraps config C, the big-library search, the AR(1) noise model and the D-dimensional code behind one entry point. Defaults were chosen on dev data only, with three crossed experiments, then frozen by md5 (`results_final/frozen.md5`) before one test pass. Each prior-off arm reproduced its frozen original exactly (0 changed answers).

**Defaults:**
- **Complexity prior:** on everywhere (degw 0.2, centred at degree 1.5). Against prior off on dev it wins 9/0 at degree 4, 9/1 at degree 5, 8/1 with trig terms and 10/0 in D dimensions. Re-centring at each library's mean degree hurts (degree 5: 6/18). Part of the gain is by construction, because no benchmark system has true terms above degree 3.
- **AR(1) switch:** the AR(1) noise model is used automatically when the estimated noise autocorrelation â exceeds 0.6. The threshold was set on a new dev archive with ρ = 0.3 and 0.55, where AR(1) adds nothing.
- **AR(1) and the prior together:** AR(1) + prior beats AR(1) alone 9/0. Re-tuning the λ(a) exponent did not help, so there is no double penalty.

**Frozen test pass against each archive's previous best:**

| Archive | Previous best | **Final** | W/L |
| --- | --- | --- | --- |
| Sealed | 0.889 | 0.889 | 0/0 |
| Confirmation | 0.886 | 0.886 | 0/0 |
| Held-out systems, new seeds | 0.943 | 0.941 | 1/3 |
| Hard 40 | 28/40 | 28/40 (DEDi 21) | 0/0 |
| Coloured | 0.734 | **0.761** | 23/1 |
| Heavy-tailed t₃ | 0.889 | **0.905** | 14/1 |
| Coarse sampling | 0.790 | 0.784 | 4/9 |
| Degree 4 / degree 5 | 0.873 / 0.861 | 0.876 / 0.865 | 1/0, 1/0 |
| Degree 3 + trig (polynomial systems / Thomas) | 0.881 / 0.994 | 0.896 / 0.988 | 2/0, 0/1 |
| D = 4–10 | 0.897 | **0.917** | 8/0 |

Pooled, that is 54 wins / 15 losses (p = 3×10⁻⁶). The archives are not fully independent: the big-library archives reuse the sealed runs.

**Regressions** (small, none significant):
- **Coarse sampling (4/9):** model misfit can push â above 0.6 and wrongly switch on AR(1). The prior was never compared with plain C on coarse data before.
- **Held-out (1/3):** all four changes are Chen–Lee at N = 500, from the same misfit-triggered switch.
- **Big libraries:** the prior makes the ambiguity flag fire less often (degree-5 dev: 17% → 8%), so unflagged accuracy drops from 0.964 to 0.910.
- **Per-term confidence:** it keeps the config C calibration. It holds on white noise ("sure" terms all true in 98.4% of runs) but is weaker under coloured (84.5%) and coarse (88.2%) noise.

Usage: `README.md`. Report: `docs/reports/overnight2/F_final.md`.

### dysts benchmark: first external test (2026-10-25)

The [dysts](https://github.com/williamgilpin/dysts) database (v0.96) has 135 continuous-time chaotic systems.
- **In the library (77):** 61 three-dimensional polynomial systems of degree ≤ 3, and 16 four- to six-dimensional systems of degree ≤ 2. For each one, our library reproduces dysts' right-hand side to ≤ 7×10⁻¹⁶.
- **New to us (49):** the rest overlap with our systems or have the same terms as one of them, and are reported separately.
- **Out of scope (58):** 53 non-polynomial (forced, rational, delay, trig, …), 4 polynomial of too high a degree, and 1 that collapses to a fixed point.

The frozen test archive has 77 systems × 10 seeds × N ∈ {500, 2000} × noise ∈ {1, 5, 10, 20}% = 6,160 runs. It was generated before any method was run, and the final model was md5-checked on the Spark before launch.

| Method | All (6,160) | New systems (3,920) | D = 3 | D = 4–6 | Median s/run |
| --- | --- | --- | --- | --- | --- |
| Weak best-subset | 0.675 | 0.591 | 0.815 | 0.142 | 0.1 |
| Joint C | 0.879 | 0.840 | 0.888 | 0.844 | 3.5 |
| **Final model** | **0.890** | **0.853** | **0.898** | **0.857** | 3.8 |

- **Final vs C:** 101 wins / 33 losses (p = 3×10⁻⁹); on new systems 79/28.
- **C vs weak:** 1,286 wins / 34 losses.
- **By cell (final):** 0.95–0.97 at 1% noise, down to 0.71 (N = 500) and 0.86 (N = 2000) at 20% noise.
- **Weakest systems (final < 0.5):**
  - AtmosphericRegime (0.00): stiff, with a true term contributing 0.6%.
  - HyperYan (0.18): smallest term contributes 1.7%.
  - IsothermalChemical (0.23), GuckenheimerHolmes (0.34), QiChen (0.49).
  - Chen–Lee (0.50) stays weak, as on our own archives.

So the method's accuracy carries over to a benchmark we did not design: 0.853 on 49 unseen systems against 0.940 on our held-out set. The main remaining failure mode is tiny or stiff terms. Reproduce with `benchmarks/dysts_bench.py` and `benchmarks/dysts_report.py`; results are in `results_dysts/`.

### Overnight screen: eleven further mechanisms

Full details are in `archive/overnight/MORNING_REPORT.md`. Two results are worth keeping:

- **Shared translation transport (multiple experiments).** When several runs carry unknown constant sensor offsets, fit one shared sparse polynomial field and profile the relative offsets, which act on the coefficients through an exact triangular polynomial map. With a 150-sample calibrated reference plus three 600-sample shifted runs, exact recovery was 108/160, against 115 with the true offsets supplied, 52 for the reference alone and 0 for naive pooling. This is a different (multi-run) setting and is not comparable with the single-run tables. Closest prior art: [Prokop et al. 2024](https://arxiv.org/abs/2402.03168) (abstract only checked).
- **Null-jet experiment design (controlled perturbations).** Terms in a direction normal to the observed data are invisible to passive data. Kicks along the estimated normal recovered 96/120 against 58 for random kicks and 79 for a D-optimal design, with equal measurement budgets. Only tested on planar data; [Active-SINDy](https://arxiv.org/html/2606.12182) is the closest prior work.

Minor or narrow: clock-free orbital one-forms (22/60 against 18/60), finite-exposure correction (better coefficients, same supports) and temporal polarisation features for heavy-tailed noise (all wins on the two easy systems).

**Negative results and rediscoveries (do not retry without a new idea)**

| Mechanism | Outcome |
| --- | --- |
| Exact Gaussian-chaos (Hermite) covariance | Already in PSI-SINDy (Appendix B.1); slower, no support gain |
| Omitted-term score-test witness | 30/40 against 29/40, lower F1; no valid post-selection guarantee |
| Observable-algebra calibration | 2/16 exact after reselection; noise fits hit bounds |
| Weak acceleration closure (x″ = Df·f) | 6/20 against the original weak selector's 9/20 |
| Sensor-clock (lag) estimation | 4/12 asynchronous, but synchronised data fall from 6/12 to 4/12 |
| Circulation (wedge) moments | 6/20 against 5/20, worse coefficients |

## Next step: data limit or statistic limit?

One day of diagnostics decides the direction: do the remaining failures come from the data, or from the weak-form statistic throwing information away? Profiled flow (above) answers this for Lorenz and Chen at 20% noise: the weak statistic is the bottleneck. The formal diagnostics below have not been run.

1. **Likelihood ratio.** On each failure case, fit the true support and the wrong winner by multiple-shooting maximum likelihood (segments up to about one Lyapunov time), across about 50 noise draws. Record how often the true support wins.
2. **Fisher information gap.** At the true parameters, compare the smallest eigenvalue of the weak-form Fisher matrix with that of the shooting Fisher matrix.
3. **Enumeration ceiling.** Enumerate all supports per equation with the whitened covariance; record the rank and posterior mass of the true support.

**Decision**

- **Shooting clearly separates them:** paper = "the weak form loses information on chaotic systems; here is how much, and a shortlist-then-shoot method that recovers it".
- **They are tied:** paper = a certified identifiability limit for the standard benchmarks, with a set-valued output and a predicted data requirement.

- [ ] Run the likelihood-ratio test on Lorenz, Chen, Rössler at 20% noise and on Rabinovich–Fabrikant
- [ ] Compute both Fisher matrices for the same cases
- [ ] Enumerate supports and record the true support's rank
- [ ] Read full texts of Gallo et al., PSI-SINDy, JSINDy and Meissner–Glasner before claiming novelty

## Code and reproduction

Paths are relative to the repository root. The package is `eqdisc/`, the scripts behind the tables are in `benchmarks/`, and superseded experiments are in `archive/` (see the repo map in `README.md`). Archived scripts import the pre-cleanup flat modules, so run them from a checkout of commit `ac7ab20`.

| File | Contents |
| --- | --- |
| `eqdisc/core.py` | Library, systems, weak/FD problems, STLSQ, E-SINDy, Hermite library, metrics |
| `eqdisc/subset.py` | Exhaustive and beam best-subset search |
| `eqdisc/select_k.py` | Weak-form matrices, WENDy covariance, selection rules (`chi2s`, `chi2`, `cv`, `bic`, `wbic`) |
| `benchmarks/bench.py` | Instance generation, all methods, parallel runner, report |
| `archive/quick_eval.py` | Fast comparison of selection rules on a 240-instance subsample |
| `eqdisc/flow_discovery.py`, `archive/hybrid_bench.py` | Profiled flow discovery; weak/flow hybrid benchmark |
| `benchmarks/dedi_bench.py` | DEDi (Meissner–Glasner) baseline runner |
| `eqdisc/joint_discover.py`, `benchmarks/joint_bench.py` | Profiled joint flow (joint states + sparse coefficients); benchmark |
| `benchmarks/joint_report.py`, `benchmarks/final_report.py` | Offline selection rules; summary tables over all archives |
| `eqdisc/heldout_systems.py`, `benchmarks/gen_variants.py` | 11 held-out chaotic systems; fresh noise-variant archives |
| `benchmarks/odr_bindy.py`, `benchmarks/odr_bench.py` | Python port of ODR-BINDy (Fung 2025); benchmark runner |
| `eqdisc/joint_biglib.py`, `benchmarks/biglib_bench.py`, `eqdisc/gen_trig.py` | Joint flow on degree-4/5 and trig libraries; Thomas archives |
| `eqdisc/joint_colored.py`, `benchmarks/colored_bench.py` | Joint flow with an AR(1) noise model |
| `archive/tiebreak_bench.py`, `archive/design_bench.py` | Attractor tiebreaker (negative); experiment design for ambiguous runs |
| `eqdisc/joint_squeeze.py`, `benchmarks/squeeze_bench.py`, `benchmarks/squeeze_report.py` | Config C + degree-weighted complexity prior (degw 0.2, frozen); results in `results_squeeze/` |
| `eqdisc/joint_nd.py`, `eqdisc/nd_systems.py`, `benchmarks/nd_bench.py`, `benchmarks/nd_report.py` | Profiled joint flow in D dimensions (Lorenz-96, hyperchaotic Rössler, coupled Lorenz); `results_nd/` |
| `benchmarks/identifiability.py`, `archive/forecast_select.py`, `archive/forecast_bench.py` | Identifiability map and required-N predictions; forecast-skill selection (negative); `results_forecast/` |
| `benchmarks/bnb_subset.py`, `eqdisc/confidence.py`, `benchmarks/certify_bench.py` | Exact subset search check; calibrated per-term confidence; `results_certify/` |
| `archive/symmetry.py`, `archive/symmetry_bench.py`, `archive/symmetry_report.py` | Sign-symmetry detection and masked search; `archive/results_symmetry/` |
| `eqdisc/final_model.py`, `benchmarks/final_bench.py`, `benchmarks/final_test_report.py` | **Final merged model** (frozen); `README.md`, `results_final/` |
| `benchmarks/dysts_classify.py`, `benchmarks/dysts_generate.py`, `benchmarks/dysts_bench.py`, `benchmarks/dysts_report.py` | dysts benchmark (77 in-library systems, 6,160 frozen runs); `results_dysts/` |

```
python benchmarks/bench.py gen                 # generate instances
python benchmarks/bench.py run bs_chi2s        # run a method
python benchmarks/bench.py report              # table: tuned and oracle results
```

Set `EQ_VARIANT=colored|student|coarse` for the noise variants and `EQ_DEG=5` for the 56-term library. pysindy MIOSR takes about 2 hours per 800 instances because of Gurobi timeouts.
