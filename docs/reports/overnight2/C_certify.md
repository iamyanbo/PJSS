# Track C: certified search and per-term confidence (2026-10-25)

## 1. Certified search on the profiled quadratic: negative, and informative

- **20-term library: the per-equation step is already exact.** `_top_subsets` searches every subset when there are at most 250k of them. The heuristic parts are the decoupling across equations (each equation searched with the others' active columns held fixed) and the top-2 product.
- **No cheap global certificate for the joint 60-coefficient problem.** The profiled quadratic is square (A = Lᵀ), so the full model fits exactly and RSS(all columns) = 0, which makes every RSS-of-superset bound vacuous. My branch-and-bound never certified (20k nodes). Gurobi MIQP (indicator constraints, at most 5 terms per equation, exact against brute force 10/10) certifies only 35–37% of rounds within 3 s. The Hessian's smallest eigenvalue is about 10⁻⁶, so the perspective relaxation doesn't help either. The collinearity that makes models ambiguous also makes them hard to certify.
- **Injection test** (`discover_bnb`: MIQP optimum plus its 2 next-best solutions added to each round's finalists), 229 dev runs:

| | Config C | C + global optimum injected |
|---|---|---|
| Exact | 0.882 | 0.882 (0 wins / 0 losses) |
| Rounds where the MIQP's best was already proposed or fitted | – | 96.9% |
| Median s per run | 3 | 10.5 |

  With no dev effect there was nothing to freeze, so no test pass was run. In failures such as Lorenz 6, the quadratic's optimum was the truth and was fitted, but the final nonlinear score preferred another model. **On the 20-term library the bottleneck is the score, not the search.**

- **Big libraries** (degree 5, 56 terms/eq): the beam missed the exact best subset of some size in 7 of 18 per-equation calls. Exact `bnb_topk` (certified, 0.2–54 s per call) on 30 dev runs at 10% noise:

| Search | Exact | Median s/run |
|---|---|---|
| Fork A beam (width 60) | 0.867 | 6.3 |
| Exact leaps-and-bounds (79/114 calls certified) | 0.867 (0 wins / 0 losses) | 139 |

  The beam is enough; its misses never changed a final answer. The sample contained no Rabinovich–Fabrikant runs.

## 2. Per-term confidence: positive

- **Method:** posterior inclusion probability p_j = Σ_m w_m [j ∈ S_m] with w ∝ exp(−Δscore/2) over the fitted candidates (BIC model averaging, T = 1). Then an isotonic recalibration fitted on dev (475 runs). Sets: sure p ≥ 0.9, possible 0.1–0.9.
- **Frozen before testing.** Test: `joint_sealed` every 3rd run (267 runs, all noise levels); this rerun reproduces config C's answers 100%.

| Sealed test | Raw | Recalibrated |
|---|---|---|
| Brier | 0.0100 | 0.0095 |
| Fraction true, p in 0.80–0.95 | 0.635 (mean p 0.875) | 0.844 (mean p 0.890) |
| Fraction true, p ≥ 0.95 | 0.990 | 0.997 |
| Every sure term is true (all runs) | 94.4% | **97.8%** |
| Same, runs where the point answer is wrong (31) | 51.6% | **80.6%** |
| Same, Rabinovich–Fabrikant (27) | 51.9% | 85.2% |
| Recall of the sure set (all runs) | 93.5% | 91.4% |
| Set contains the whole truth, point answer wrong | 12.9% | 12.9% |

- **What works:** recalibrated per-term probabilities tell the user which terms to trust, even when the point answer is wrong.
- **What doesn't:** a set-valued output does **not** recover the truth when the point answer fails. The true terms are usually missing from every candidate (1.8 true terms per wrong run appear in no candidate).
- **Caveats:**
  - Raw BIC weights are overconfident in the 0.8–0.95 range.
  - The dev calibration data under-represents 5% and 20% noise: part 2 was every 2nd run, and the noise level cycles with the index (mod 4).

## Files

- **Code:** `bnb_subset.py` (bnb_topk, bnb_penalized, miqp_penalized, discover_bnb, capture_quadratics), `confidence.py`, `certify_bench.py`. Scratch: `scratch_bnb_test.py`, `scratch_bnb_report.py`.
- **Results** (`results_certify/`): `dev_pool_part1/2.json`, `sealed_pool_every3.json`, `dev_bnb_e7.json`, `dev_bnb_e7o3.json`, `dev_deg5_exact.json`, `probe_audit.json`.
- **Frozen confidence spec:** `overnight2/C_frozen_confidence.txt`.
