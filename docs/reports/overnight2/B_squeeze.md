# Track B: squeezing config C

## 1. Failure anatomy (dev, 800 runs, `results_squeeze/dev_C_diag.json`)

For each wrong run, the true support was refitted with `fit_joint` from two inits: the method's own state estimate, and the true coefficients. The true coefficients were used only for this diagnostic, never in the method. Of 95 wrong runs:

| Cause | Runs |
| --- | --- |
| B, selection: the truth's best fit still scores worse than the winner | **92** |
| A, search: truth never proposed, but would win (by about 1.1) | 3 |
| C, local minimum | 0 |

Search is essentially solved, so every remaining point has to come from the score.

- **Rabinovich–Fabrikant (68 runs):** the winner is smaller than the truth (it misses 4.4 true terms on average), and the truth loses by a median of 8.7. This is the known identifiability limit.
- **The other 24 runs** lose by a median of 3.7, through near-collinear swaps:

| System | Failure | Runs |
| --- | --- | --- |
| Lorenz | y′: −y swapped for y·z (z stays near its mean on the attractor) | 8 |
| Lorenz | y′: −y swapped for x³ | 2 |
| Rössler | z′: constant term dropped | 9 |
| Chen | y′: x term dropped | 3 |

## 2. Variants (dev, every 3rd run = 267 runs; config C scores 0.876 on the same runs)

| Variant | Exact | W/L vs C | Median s |
| --- | --- | --- | --- |
| ρ continuation 0.3 → 0.1 → 0.03 | 0.873 | 0/1 | 18.8 (2×) |
| add/drop/swap moves | 0.880 | 1/0 | 7.6 |
| Laplace-evidence score (profiled Hessian, σp = 10) | 0.876 | 1/1 | 10.0 |
| **Complexity prior, degw = 0.2** | **0.891** | **4/0** | 8.4 |
| Complexity prior, degw = 0.4 | 0.880 | 4/3 | 8.9 |

The complexity prior is borrowed from symbolic regression (PySR and Eureqa complexity). The per-term cost becomes log(3n)·(1 + degw·(degree − 1.5)), so its mean stays at BIC's level while constant and linear terms get cheaper and cubic terms dearer.

Design guards before freezing:
- **Full dev, degw 0.2:** 0.881 → 0.890 (8/1, p = 0.04).
- **Held-out-systems design archive, every 3rd run:** degw 0.2 scores 0.946 → 0.952 (2/0). degw 0.3 goes 2/2, so it was rejected.

Frozen: degw = 0.2 (`joint_squeeze.py`, md5 9e93185b…; copy in `$TEMP/joint_squeeze.frozen_deg2.py`).

## 3. Frozen test (one pass)

| Archive | Runs | C | **C + complexity prior** | W/L | p |
| --- | --- | --- | --- | --- | --- |
| sealed | 800 | 0.881 | **0.889** | 8/2 | 0.11 |
| confirmation | 800 | 0.868 | **0.886** | 19/4 | 0.003 |
| held-out systems (new seeds) | 880 | 0.933 | **0.943** | 9/0 | 0.004 |
| **pooled** | 2480 | | | **36/6** | **3×10⁻⁶** |

- **Hard 40 (20% noise, N = 2000):** 25 → **28/40**, against DEDi's 21/40. Paired against DEDi, 7 wins and 0 losses (p = 0.016). Lorenz goes 7 → 10/10; Chen 10, Rössler 8 and Rabinovich–Fabrikant 0 are unchanged.
- **Wins:** mostly Lorenz (the y ↔ y·z swap), Rabinovich–Fabrikant at 1% noise, and Rikitake (5 runs).
- **Losses:** spread out (2 Rössler, 1 Sprott B, 2 Rabinovich–Fabrikant, 1 Lorenz).
- **Cost:** a median of 4–6 s per run (joint stage only, cached starts, this PC). That is the same as C, because the prior only changes the score.

## 4. Caveats

- The prior encodes a preference for low-degree terms. It would hurt a system whose true dynamics are dominated by high-degree terms; no such system is in these benchmarks. Rabinovich–Fabrikant (several cubic terms) was not hurt (net +5 across all archives).
- degw was picked from three values on dev, so it is one tuned knob.
- These runs did not include the big-library (degree 4/5) or AR(1) noise variants; how the prior combines with them is untested.
- A ceiling result for the paper: with search essentially solved on dev (3 search misses in 800 runs), what remains is identifiability, not search.

## Files

- Code: `joint_squeeze.py`, `squeeze_bench.py`, `squeeze_report.py`.
- Results: `results_squeeze/` (dev_C_diag, dev3_*, ho3_*, dev_deg2, test_*_deg2, starts_*.pkl).
