# Track A: sign-flip symmetry priors for joint flow

**Method.**
- A sign flip g = diag(s) is a symmetry when f(gx) = g·f(x). Each such flip becomes a mask on the library terms each equation may use.
- **Pre-screen (truth-free):** compare a dense masked vs unmasked Savitzky–Golay derivative regression. At threshold 1, this keeps 97.5% of true symmetries and 16% of false ones on dev.
- **Constrained search:** for the top-2 screened flips, run joint flow restricted to the mask. Starts are the C starts projected onto the mask plus a dense least-squares fit of all allowed terms.
- **Evidence:** the constrained answers are added as starts to an unconstrained run. The evidence is D = (best score of any symmetry-breaking candidate) − (best score of any symmetric candidate).
- **Rule:** accept the symmetry when D ≥ −20 (τ = 20, frozen) and return the strongest accepted constrained answer.
- **Scope:** permutation symmetries (e.g. Halvorsen) are not covered.

**Detection works.** On dev (267 runs, all 7 flips tested):
- The true flip gets a median D of +6.2, against −50,000 for false flips.
- At |D| > 2: true flips are detected in 135/160 cases, and false flips are wrongly detected in only 3/1709 tests.
- **Runs with a false detection** at τ = 20: 23/267 on dev, almost all at 10–20% noise. These are mostly harmless, because the wrong winner already respects that flip.

**But accuracy barely moves.**
- Dev oracle (best of base / symstart / constrained): 0.880 against C's 0.876.
- **Why:** every failed Lorenz/Chen answer already respects the true symmetry, so the method swaps one symmetric term for another.
- **Rabinovich–Fabrikant** is identifiability again. Even started from the truth, the true 10-term support scores 3–22 points worse than smaller 6–9-term models. The dense symmetric start raises the constrained answer to 7/27 exact on dev (projected starts: 5/27; C: 4/27), but D stays mostly ≤ 0.

**Frozen test (one pass, subsamples):**

| Set | Runs | Config C | Symmetry | W/L | False-detection runs | Median s/run |
|---|---|---|---|---|---|---|
| Hard 40 (20%, N = 2000) | 40 | 0.625 | **0.650** | 1/0 | 7/40 | 74 |
| Sealed, every 7th (all 8 cells) | 115 | 0.896 | 0.887 | 0/1 | 8/115 | 12 |
| Held-out 2, every 11th | 80 | 0.938 | **0.975** | 3/0 | 0/80 | 14 |

- **What changed:** Chen–Lee goes from 2/7 to 5/7 (it has three sign symmetries). One Rössler run was gained via the extra starts, not the prior. One Sprott C run was lost. Rabinovich–Fabrikant stays 0.
- **Overall: 4 wins / 1 loss across 235 runs**, not significant. Cost is about 4–6× C (median 12–74 s, on a shared PC).

**Verdict.** It's a reliable symmetry detector, and a useful diagnostic: it says which symmetry the data support. But as a prior it helps only where the wrong answers break the symmetry, which on these benchmarks is mainly Chen–Lee. It isn't worth the 4–6× cost as a default. A possible cheap variant: run it only when a run is flagged.

**Protocol caveats:**
- τ = 20, the screen and the dense start were frozen without a full dev re-pass of the final v2 pipeline. The dense start was validated only on the 27 Rabinovich–Fabrikant dev runs, because of time and a shared CPU.
- The test sets are subsamples.

**Files:**
- Code: `symmetry.py`, `symmetry_bench.py`, `symmetry_report.py`, `scratch_sym_oracle.py`, `scratch_sym_dense.py`, `scratch_sym_final.py`
- Results: `results_symmetry/dev_all7.json`, `rf_oracle_dev.json`, `rf_dense_dev.json`, `test_hard40.json`, `test_sealed_e7.json`, `test_heldout2_e11.json`
- Frozen copies: `$TEMP/symmetry.frozen.py`, `$TEMP/symmetry_bench.frozen.py`
