# Track E: forecast-skill selection and identifiability map

## 1. Forecast-skill selection: negative

The last 20% of each run is withheld and starts plus joint C are fitted on the first 80%. Candidates within a window of the best BIC are scored by rolling-origin ensemble forecasts inside the withheld block.

- **Origin states:** each origin's state comes from a causal states-only joint fit with coefficients frozen, using only observations up to the origin.
- **Forecasts:** a 16-member RK4 ensemble.
- **Lead times:** 0.1–1 Lyapunov time, with λ estimated by Benettin's method on the top candidate.
- **Proper scores:** Gaussian log score and ensemble CRPS.
- **Rule search:** 192 rules (score type × lead time × weight × window) were searched on dev; one was frozen.

| Exact recovery | Dev (800) | **Sealed (800, frozen)** |
| --- | --- | --- |
| Config C, all data | 0.881 | 0.881 |
| BIC, first 80% | 0.870 | 0.869 |
| Best forecast rule (log score, 0.1 T_L, weight 0.25, window 2) | 0.865 | **0.866** (7/9 vs BIC80; 3/15 vs C, p = 0.008) |
| Best possible pick from the candidate pool | 0.900 | 0.901 |

When the truth is in the pool but BIC misses it, forecast skill picks the truth only 27–46% of the time. Near-tied models forecast as alike as they fit, so withholding data for forecasting only costs accuracy.

## 2. Identifiability map: strong positive

Given the true support S (an oracle diagnostic), I fit joint C on S, take the profiled quadratic ‖Ac − b‖² in all 60 coefficients, and compute a noise-free BIC margin:

m* = min over all single-term drops and same-equation swaps T of ‖(I − P_T)A c_S‖² + log(3n)(|T| − |S|)

This is the misfit a competitor pays in the noise-free limit. The threshold τ was fit on dev and frozen.

| | Dev | **Sealed (frozen τ = 1.04)** | Fresh R–F, N up to 32k (frozen) |
| --- | --- | --- | --- |
| AUC of m* for predicting C exact | 0.978 | **0.985** | 0.98 |
| Accuracy of the rule m* > τ | 96.6% | **97.4%** (baseline 88.1%) | 91.4% (baseline 80%) |
| Exact rate when predicted identifiable / not | 97.6% / 11.4% | 98.4% / 10.6% | 91.7% / 10% |

- **Binding competitor:** almost always a same-equation swap, not a drop. Examples: Lorenz y→yz in ẏ; R–F x→x² in ẏ and y→xy in ẋ.
- **R–F collinearity on the clean attractor:** replacing x by x² in ẏ still explains 99.994% of ẏ's variance. The R–F failures are therefore an identifiability limit, not a search limit.
- **Scaling:** R–F's margin grows like N^1.0–1.5 and like 1/σ².

## 3. Data needed for Rabinovich–Fabrikant (prospective)

Predictions were written to `E_predictions.json` from dev margins before any new data was generated. The new archive (seed 20261301) was then generated and run through unchanged config C.

| Cell | Predicted P(identifiable) | Observed exact (C) | Median s/run |
| --- | --- | --- | --- |
| 1%, N = 1000 | 0.90–1.00 | 8/10 | 4 |
| 5%, N = 4000 | 0.30 | 4/10 | 16 |
| 5%, N = 8000 | 0.50–0.60 | **10/10** | 21 |
| 5%, N = 16000 | 1.00 | 10/10 | 84 |
| 10%, N = 8000 | 0.20 | 4/10 | 23 |
| 10%, N = 16000 | 0.30 | **10/10** | 85 |
| 10%, N = 32000 | 0.70 | 10/10 | 499 |

- **R–F is solvable with more data:** it goes from 0–10% at N = 2000 to 100% at N = 8000 (5% noise) and N = 16000 (10% noise).
- **The extrapolation is conservative:** about 2× pessimistic on N. Per-run margins on the new runs still classify 91% correctly.
- **Joint flow scales:** it handles N = 32000 in about 8 minutes on one core.

## Files

- Code: `forecast_select.py`, `forecast_bench.py`, `identifiability.py` (includes `gen_rf`).
- Analysis scripts: `scratch_E_forecast_report.py`, `scratch_E_ident_report.py`.
- Data: `data/rf_needed.npz`.
- Results in `results_forecast/`: `dev.json`, `sealed.json`, `ident_dev.json`, `ident_sealed.json`, `rf_needed_C.json`, `ident_rf_needed.json`.
- Frozen settings: `overnight2/E_predictions.json`, `overnight2/E_frozen_forecast_rule.json`.

Caveat: the ident_dev run briefly used 2 workers alongside the 3 forecast workers, so 5 cores in total, against the 3-core cap.
