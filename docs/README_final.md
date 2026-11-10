# Final merged model: `final_model.discover`

One entry point for the profiled-joint-flow family. It imports the frozen modules (`joint_discover`, `joint_biglib`, `joint_colored`, `joint_nd`, `confidence`) and edits none of them. It adds two things on top: the complexity prior, applied uniformly, and an automatic switch for the noise model. The defaults were chosen on dev data only and then frozen. The evidence is in `overnight2/F_final.md`.

```python
import numpy as np
from final_model import discover

res = discover(y, dt)                     # y: (n, 3) noisy trajectory, dt: sample step
print(res.equations())                    # dx/dt = -9.957*x +9.963*y ...
res.xi                                    # (n_terms, 3) coefficients, original units
res.names                                 # term names ("1", "x", "x*y", ..., "sin(x)")
res.ambiguous, res.gap                    # flag: best and runner-up within 2 (2*lam under AR(1))
res.confidence                            # {(term, equation): p}, per-term inclusion probability
res.sure_terms()                          # terms with p >= 0.9
res.info["noise_model"], res.info["ar_estimate"], res.info["prior"]
```

## Frozen defaults

- **Complexity prior:** per-term penalty log(3n)·max(0.1, 1 + 0.2·(degree − 1.5)), on for every library and also with AR(1).
- **Noise model:** AR(1) is used iff the mean estimated lag-1 error correlation â > 0.6, with λ(a) = ((1+a)/(1−a))^{1/3}.
- **Integrity:** md5 hashes are in three files:
  - `results_final/frozen.md5`: v1.0;
  - `results_v101/frozen.md5`: v1.0.1, the crash fix (`docs/reports/v101.md`);
  - `results_final_v11/frozen.md5`: v1.1, Gaussian loss inside the AR(1) likelihood (`docs/reports/v11_coloured.md`);
  - `results_precision/frozen.md5`: v1.2, discretisation-error noise floor (`docs/reports/precision_audit.md`).

  v1.2 is the default; `version="1.1"` and `version="1.0.1"` select the earlier versions.
- **Test results:**
  - coloured 0.761 (previous best 0.734);
  - heavy-tailed 0.905 (0.889);
  - D = 4–10: 0.917 (0.897);
  - sealed and confirmation unchanged at 0.889 / 0.886;
  - coarse 0.784 (0.790, the one small regression).

  The full tables are in `overnight2/F_final.md`.

## Options

| Argument | Values | Default |
| --- | --- | --- |
| `library` | `"poly3"` (20 terms/eq), `"poly4"` (35), `"poly5"` (56), `"trig3"` (deg-3 + sin/cos of each state, 26); `"poly2"`/`"auto"` for D ≠ 3 | `"auto"` = poly3 if D = 3, else poly2 (`joint_nd`) |
| `noise` | `"auto"`, `"white"` (config C likelihood), `"ar1"` (AR(1) likelihood + λ(a) penalty) | `"auto"`: AR(1) iff the estimated lag-1 error correlation exceeds `DEFAULTS["ar_threshold"]` |
| `starts` | list of coefficient arrays (original units) used as seeds | the frozen benches' seeds: weak χ² + hybrid + top-3 flow (poly3); weak chi2s/chi2/bic (big/trig libraries); `joint_nd.weak_starts` (D ≠ 3) |
| `prior`, `degw`, `centre`, `ar_threshold`, `pen_exp`, `huber_ar1` | overrides for experiments (`huber_ar1`: Huber c inside the AR(1) likelihood; `inf` = Gaussian) | `DEFAULTS` in `final_model.py` |
| `version` | `"1.0.1"` (frozen v1.0 + crash fix), `"1.1"` (Gaussian loss on the AR(1) path) or `"1.2"` (+ discretisation-error noise floor) | `None` = current defaults (v1.2) |
| `confidence` | per-term probabilities | `True`; isotonic-recalibrated (track C's frozen map) only for poly3 |
| `return_candidates` | adds the scored candidate pool to `info` | `False` |

Extra keyword arguments go to the underlying `discover` (for example `rounds`, `finalists`, `rho`).

## Notes

- **Big polynomial libraries need `EQ_DEG`.** `core` builds its library from the `EQ_DEG` environment variable at import time. If `library="poly4"`/`"poly5"` does not match the imported `core`, the call runs transparently in a spawned subprocess with `EQ_DEG` set, which costs an interpreter start. For batches, set `EQ_DEG=5` before starting Python.
- **D ≠ 3** uses `joint_nd` (degree-2 library). The AR(1) noise model is not implemented there, and `noise="ar1"` raises an error.
- **Untested combinations:** AR(1) with the degree-4/5 or trig libraries runs, but no dev or test archive covers it. The same goes for coloured noise in D ≠ 3.
- **Cost:** a few seconds per 3-D run (more for N = 2000 or coarse sampling), plus about 1 s for the AR(1) pre-estimate under `noise="auto"`. D = 10 takes minutes.
- **Reproduce the evaluation:** `final_bench.py` (dev arms and the frozen `final` arm), `final_dev_report.py` and `final_test_report.py`. Results are in `results_final/`; job scripts are `results_final/dev2.sh` and `results_final/test.sh`.
