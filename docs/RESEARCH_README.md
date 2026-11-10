# eqdisc: equation discovery from one noisy trajectory

`eqdisc` recovers a sparse polynomial (optionally polynomial + sin/cos) ODE dx/dt = f(x) from a single noisy,
uniformly sampled trajectory. Sparsity is selected automatically using fixed development-selected defaults. You get back the coefficients, an ambiguity
flag, and a per-term confidence.

**Method.** The method is *Profiled Joint Subset Search (PJSS)*. It estimates the denoised states u and the sparse
coefficients together. The objective is a robust (Huber) data misfit |(u − y)/σ|² plus a model misfit
|(u_{k+1} − u_k − dt·f(u_mid))/(ρσ)|² on a sub-sampled grid. For a fixed support, the fit is a
Levenberg–Marquardt solve whose normal matrix is banded in the states with a dense coefficient border, so it
costs linear time in N. Candidate supports come from profiling the states out at the current fit, which gives
one quadratic in all library coefficients. The search ranks subsets per equation in that quadratic, refits the
best combinations jointly, and repeats. Seeds come from a weak-form best-subset selector and from profiled flow
discovery, never from the truth. Supports are compared by a BIC-type score on held-out samples. Its penalty is
inflated for residual autocorrelation, and a degree-weighted complexity prior makes each term cost
log(3n)·max(0.1, 1 + 0.2·(degree − 1.5)).

**What the final model adds.** `eqdisc.discover` puts the variants behind one entry point: the 20-term cubic
library, degree-4/5 libraries with beam-search proposals, a cubic + trig library, an AR(1) observation-noise
likelihood, and D-dimensional systems with a degree-2 library. The AR(1) model is switched on automatically
when the estimated lag-1 error correlation exceeds 0.6. The defaults were chosen on dev archives only and
frozen (md5s in `results_final/frozen.md5`) before a single test pass.

**Results.** On the frozen test archives it recovers the exact support in:
- 0.889 of sealed white-noise runs;
- 0.941 of held-out systems;
- 0.761 of coloured-noise runs;
- 0.905 of heavy-tailed-noise runs;
- 0.917 of D = 4–10 runs;
- 28/40 of the hardest cases (20% noise), against 21/40 for DEDi.

The full story, including the negative results, is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Install

```bash
pip install -e .              # numpy, scipy, numba
pip install -e ".[bench,test]"   # + threadpoolctl, pytest for benchmarks/ and tests/
```

Optional, only for specific benchmark scripts:
- `pysindy` (+ `gurobipy` for MIOSR): `benchmarks/bench.py` baselines;
- `gurobipy`: the exact MIQP check in `benchmarks/bnb_subset.py`;
- `dysts` + `sympy`: `benchmarks/dysts_classify.py`, in its own venv;
- DEDi (external checkout): `benchmarks/dedi_bench.py --ext`.

## Quick start

```python
import numpy as np
from eqdisc import discover
from eqdisc.core import SYSTEMS, X0, DT, simulate

clean = simulate(SYSTEMS["lorenz"], X0["lorenz"], DT["lorenz"], 2000, burn=1000)
y = clean + 0.05 * clean.std(axis=0) * np.random.default_rng(0).standard_normal(clean.shape)

res = discover(y, DT["lorenz"])           # y: (n, D) noisy trajectory, dt: sample step
print(res.equations())                    # dx/dt = -9.957*x +9.963*y ...
res.xi                                    # (n_terms, D) coefficients, original units
res.names                                 # term names ("1", "x", "x*y", ..., "sin(x)")
res.ambiguous, res.gap                    # flag: best and runner-up within 2 (2*lam under AR(1))
res.confidence                            # {(term, equation): p}, per-term inclusion probability
res.sure_terms()                          # terms with p >= 0.9
res.info["noise_model"], res.info["ar_estimate"], res.info["prior"]
```

## The final model: `eqdisc.discover`

This section was `README_final.md`, which is kept as `docs/README_final.md`. `discover` is
`eqdisc.final_model.discover`.

There is one entry point for the profiled-joint-flow family. It imports the frozen modules (`joint_discover`,
`joint_biglib`, `joint_colored`, `joint_nd`, `confidence`) and edits none of them. On top it adds two things:
the complexity prior, applied uniformly, and an automatic switch for the noise model. The defaults were chosen
on dev data only and then frozen. The evidence is in `docs/reports/overnight2/F_final.md`.

### Frozen defaults

- **Complexity prior:** each term costs log(3n)·max(0.1, 1 + 0.2·(degree − 1.5)). It is on for every library,
  including with AR(1).
- **Noise model:** AR(1) is used iff the mean estimated lag-1 error correlation â > 0.6, with
  λ(a) = ((1+a)/(1−a))^{1/3}.
- **Integrity:** md5 hashes are in `results_final/frozen.md5`. They refer to the pre-cleanup flat files at
  commit `ac7ab20`, some hashed with CRLF line endings. `tests/test_equivalence.py` checks that the package
  gives identical answers.

### Versions

| Version | Change | md5 / evidence |
| --- | --- | --- |
| v1.0 | the frozen final model (all test numbers in this README) | `results_final/frozen.md5` |
| v1.0.1 | bug fix only: an empty candidate support was used as a float index (`IndexError` on every 1-D call, on some short 2-D series and on 3 of 320 ODEBench 3-D runs). Bit-identical to v1.0 on every run v1.0 completed | `results_v101/frozen.md5`; `docs/reports/v101.md` |
| v1.1 | Gaussian loss (Huber off) inside the AR(1) likelihood only; the white path and the AR(1) pre-estimate keep Huber, so routing is unchanged. Pre-registered fresh coloured test: 0.754 → 0.805, 43/2, p = 6e-11; white and t₃ test archives unchanged (0/1,600 answers) | `results_final_v11/frozen.md5`; `docs/reports/v11_coloured.md` |
| **v1.2** (default) | noise-scale floor from the latent grid's discretisation error (`disc_floor`): removes the tiny spurious terms on noise-free data. Pre-registered fresh noise-free test: 0.265 → 1.000, 147/0, p = 1e-44; sealed white, coloured, t₃ and nd archives: 0 support changes in 2,800 runs | `results_precision/frozen.md5`; `docs/reports/precision_audit.md` |

`discover(y, dt, version="1.0.1")` reproduces v1.0.1, `version="1.1"` v1.1; `version="1.2"` or no argument gives v1.2. Git tags `v1.0.1`, `v1.1`, `v1.2`.
- **Test results:**
  - coloured 0.761 (previous best 0.734);
  - heavy-tailed 0.905 (0.889);
  - D = 4–10: 0.917 (0.897);
  - sealed and confirmation unchanged at 0.889 / 0.886;
  - coarse 0.784 (0.790, the one small regression).

  The full tables are in `docs/reports/overnight2/F_final.md`.

### Options

| Argument | Values | Default |
| --- | --- | --- |
| `library` | `"poly3"` (20 terms/eq), `"poly4"` (35), `"poly5"` (56), `"trig3"` (deg-3 + sin/cos of each state, 26); `"poly2"`/`"auto"` for D ≠ 3 | `"auto"` = poly3 if D = 3, else poly2 (`joint_nd`) |
| `noise` | `"auto"`, `"white"` (config C likelihood), `"ar1"` (AR(1) likelihood + λ(a) penalty) | `"auto"`: AR(1) iff the estimated lag-1 error correlation exceeds `DEFAULTS["ar_threshold"]` |
| `starts` | list of coefficient arrays (original units) used as seeds | the frozen benches' seeds: weak χ² + hybrid + top-3 flow (poly3); weak chi2s/chi2/bic (big/trig libraries); `joint_nd.weak_starts` (D ≠ 3) |
| `prior`, `degw`, `centre`, `ar_threshold`, `pen_exp`, `huber_ar1` | overrides for experiments (`huber_ar1`: Huber c inside the AR(1) likelihood, `inf` = Gaussian) | `DEFAULTS` in `eqdisc/final_model.py` |
| `version` | `"1.0.1"`, `"1.1"` or `"1.2"` | `None` = current defaults (v1.2) |
| `confidence` | per-term probabilities | `True`; isotonic-recalibrated (track C's frozen map, `eqdisc/confidence_isotonic.json`) only for poly3 |
| `return_candidates` | adds the scored candidate pool to `info` | `False` |

Extra keyword arguments go to the underlying `discover` (for example `rounds`, `finalists`, `rho`).

### Notes

- **Big polynomial libraries need `EQ_DEG`.** `eqdisc.core` builds its library from the `EQ_DEG` environment
  variable at import time. If `library="poly4"`/`"poly5"` does not match the imported `core`, the call runs in a
  spawned subprocess with `EQ_DEG` set, which costs an interpreter start. For batches, set `EQ_DEG=5` before
  starting Python.
- **D ≠ 3** uses `joint_nd` (degree-2 library). The AR(1) noise model is not implemented there, and
  `noise="ar1"` raises an error.
- **Untested combinations:** AR(1) with the degree-4/5 or trig libraries runs, but no dev or test archive
  covers it. The same goes for coloured noise in D ≠ 3.
- **Cost:** a few seconds per 3-D run (more for N = 2000 or coarse sampling), plus about 1 s for the AR(1)
  pre-estimate under `noise="auto"`. D = 10 takes minutes.

## Tests

```bash
python -m pytest tests/test_smoke.py              # Lorenz, 5% noise -> exact (~40 s)
python -m pytest tests/test_equivalence.py -s     # package vs frozen ac7ab20 final_model, 42 archive runs
python -m pytest tests/test_flow_discovery.py     # unit tests of eqdisc.flow_discovery
```

`test_equivalence.py` extracts the frozen flat modules from commit `ac7ab20` with `git show` and checks them
against `results_final/frozen.md5`. It then runs both versions on 42 runs drawn with a fixed seed from the
sealed, coloured, t₃, coarse, trig, degree-4, degree-5 and D = 4–6 archives, and requires identical
coefficients (`np.array_equal`), gap, ambiguity flag, confidences and noise model. It uses 3 worker processes;
set `EQDISC_EQUIV_WORKERS` to change that.

## Reproducing the tables

The manuscript calls the method **PJSS (Profiled Joint Subset Search)**. Historical reports and
module names use "profiled joint flow"; this naming change does not change the algorithm or stored results.
The runtime and package metadata both identify the current implementation as version 1.2.0.

To check stored headline results and rebuild the manuscript:

```bash
python docs/paper/scripts/verify_numbers.py
python docs/paper/scripts/make_figures.py  # requires matplotlib
python docs/paper/scripts/build.py         # requires pdflatex and bibtex on PATH
```

The first two commands read saved results, not fresh method executions. When `data/instances.npz`
is available, the verifier also audits the Lorenz development cell (5% noise, N=2000) directly
from the saved threshold-path arrays: 0/10 for `fd9`, `weak20`, and `eweak20`, 1/10 for
`ps_weak`, and 9/10 for `bs_chi2s`. The original configurations and generation seed are in
`benchmarks/bench.py`; these diagnostics differ from the later official-package benchmark. The build script writes
LaTeX intermediates under `tmp/pdfs/build/` and updates `docs/paper/main.pdf` after successful compilation.

**Release status.** This checkout has no configured publication remote. A submission release still needs
a public or anonymous artifact URL, the exact source snapshot, the original benchmark environments,
and the required data archives. Minimum dependency versions in `pyproject.toml` are installation
requirements, not a record of the historical benchmark environment. Do not substitute a freeze of
the current machine for that record. In particular, `.npz` archives are excluded from Git, and some
original generation seeds are unrecorded (see the archive table below). Preserve and distribute those
archives with `data/MANIFEST.sha256`; checksums alone cannot regenerate them. Some external methods
also require separate installations, solver licenses or checkpoints, as documented in their reports.
Prospective protocols in `results_*/PREREGISTRATION*` are repository records, not independent registrations.

### Submission audit

The November 2026 audit is recorded in `results_submission_audit/`. It preserves historical outputs;
it does not relabel them as fresh runs of corrected code. Reusable checks (choose a new output path
for each run; the scripts refuse to overwrite an existing report):

```bash
python -m benchmarks.audit_stored_coefficients --output results_submission_audit/stored_coefficients_new.json
python -m benchmarks.audit_archive_evidence --output results_submission_audit/archive_evidence_new.json
python -m benchmarks.audit_system_intervals --output results_submission_audit/system_intervals_new.json
python -m benchmarks.audit_weak_seed --output results_submission_audit/weak_seed_new.json
python -m pytest tests/test_baseline_reproducibility.py -q
```

The first three checks use saved data only. The fourth runs a small, outcome-independent set of
weak baselines twice in different orders; it is not a full replacement benchmark. The baseline
runner now imports PySINDy and seeds the RNG before constructing weak domains, then separately
resets the optimizer seed. The original weak-baseline tables predate this fix. Their CV partitions
precomputed regression rows, not isolated time intervals; weak domains can overlap across folds.

The completed corrected-code diagnostic is `results_submission_audit/weak_seed_verified.json`:
all 14 runs (8 weak-STLSQ, 4 weak-SR3, 2 weak-E-SINDy) repeat identically under reversed execution
order and perturbed prior RNG state. Against the historical records, selected supports change in
12/14 CV and 13/14 AICc cases; pooled exact counts remain 2/14 for CV and change from 0/14 to 1/14
for AICc. These are small diagnostic samples, not full-archive estimates. Software-environment
differences may also contribute to the historical differences. The earlier `weak_seed.json`
records the failed intermediate check before optional-library imports were moved before seeding;
it is retained as a diagnostic trail, not the result of the final fix. Reproducing weak-baseline
tables with the corrected runner requires fresh runs of those affected configurations.

Checks found 44,424 coefficient arrays consistent with their saved metrics, and all 5,250 raw DEDi
fits complete and consistent. All 43 available manifest archives match SHA-256, including ten
documented relocated archives. Four originals are missing: `po_test.npz`, `dev_clean.npz`,
`sealed_clean.npz`, and `dysts_precdev.npz`. The documented generators were replayed into
`tmp/submission_audit/recovered/`; none reproduced the original hash, so none replaced an original.
`archive_recovery.json` records the commands and discrepancies. The central full-observation
benchmarks are available, but exact historical reproduction of the affected secondary experiments
still requires the original archives or explicitly reported fresh confirmations.

The system-bootstrap script documents the estimand, seed, cluster draws and percentile method.
The manuscript uses these reproducible, post-hoc endpoints instead of the undocumented historical
bootstrap endpoints; the paired Newcombe intervals are unchanged. The external PJSS comparison
and profiling variants retain summary metrics but not fitted coefficient arrays, which limits
independent rescoring of those particular records. Full threshold-path coefficients were also not
retained for the official baseline oracle paths.

Run everything from the repository root with the package installed. The scripts read `data/` and write
`results_*/` relative to the current directory. Benchmark runners refuse to overwrite an existing output, and
`--workers` sets the process count (the frozen runs used 18 on the DGX Spark). The archives (`data/*.npz`) are
gitignored. Their SHA-256 checksums are in `data/MANIFEST.sha256`, so regenerated archives can be checked with
`sha256sum -c data/MANIFEST.sha256`. The ten `overnight/data/...` entries there now live under
`archive/overnight/data/`.

### Archives

| Archive(s) | Generator |
| --- | --- |
| `instances.npz`, `instances_{colored,student,coarse}.npz` (dev) | `[EQ_VARIANT=colored\|student\|coarse] python benchmarks/bench.py gen` |
| `joint_sealed*.npz`, `hybrid_confirm.npz`, `hybrid_fresh*.npz`, `hybrid_conf_*.npz`, `sealed_{colored,student,coarse}.npz`, `heldout2.npz` | `python benchmarks/gen_variants.py --kind white\|colored\|student\|coarse [--heldout] --seed S --output ...` (the hybrid confirmation set is documented as seeds 20261103/4 in docs/WRITEUP.md; the other seeds are not recorded in the repo, so keep these archives and check them against the manifest) |
| `heldout.npz`, `heldout_{500,2000}.npz` | `eqdisc.heldout_systems.generate(seed, length, output)` |
| `flow_fresh.npz` | `python benchmarks/flow_generate.py --seed 20261023` |
| `dev_colored_mid.npz` | `python benchmarks/gen_colored_mid.py` (seed 20261501) |
| `trig_dev.npz`, `trig_test.npz` | `python -m eqdisc.gen_trig --seed S --output ...` |
| `nd_dev.npz`, `nd_test.npz` | `eqdisc.nd_systems.generate(...)` (seeds 20261301 / 20261302) |
| `rf_needed.npz` | `benchmarks/identifiability.py`, function `gen_rf(output, cells)` (seed 20261301) |
| `dysts_{smoke,test}.npz` + `.json` | `benchmarks/dysts_classify.py` (dysts venv) → `data/dysts_classes.json`, then `python benchmarks/dysts_generate.py --classes data/dysts_classes.json --output data/dysts_test.npz --seed 20261401` |

### Tables

| Table (docs/WRITEUP.md section) | Run | Report | Results |
| --- | --- | --- | --- |
| Weak best-subset vs SINDy baselines ("Results") | `python benchmarks/bench.py run METHOD` (EQ_VARIANT / EQ_DEG=5 for variants) | `python benchmarks/bench.py report` | `results/`, `results_deg5/` |
| Profiled joint flow, 11 archives ("Profiled joint flow") | `python benchmarks/joint_bench.py --data data/<archive>.npz --output results_joint/<name>.json` | `python benchmarks/final_report.py`, `benchmarks/joint_report.py` | `results_joint/` |
| Hard 40: DEDi | `python benchmarks/dedi_bench.py --ext <shim+DEDi> --systems lorenz chen rossler rabinovich --noise 0.2 --n 2000 --grid small --output ...` | in the run output | `results_hybrid/dedi_hard20*.json` |
| Hard 40: ODR-BINDy port | `python benchmarks/odr_bench.py --systems lorenz chen rossler rabinovich --noise 0.2 --n 2000 --output ...` (port in `benchmarks/odr_bindy.py`) | in the run output | `results_odr/hard40.json` |
| Big libraries / trig | `EQ_DEG=4\|5 python benchmarks/biglib_bench.py ...`, `--lib mixed` | `python benchmarks/biglib_report.py` | `results_biglib/` |
| AR(1) coloured noise | `python benchmarks/colored_bench.py ...` | `python benchmarks/colored_report.py` | `results_colored/` |
| B. Complexity prior | `python benchmarks/squeeze_bench.py starts ...` then `run --variant deg2` (see `results_squeeze/queue_test.sh`) | `python benchmarks/squeeze_report.py compare <json>...` | `results_squeeze/` |
| D. Higher dimensions | `python benchmarks/nd_bench.py --data data/nd_test.npz ...` | `python benchmarks/nd_report.py results_nd/test.json` | `results_nd/` |
| E. Identifiability map | `python benchmarks/identifiability.py --data data/<archive>.npz --output ...` | in the run output | `results_forecast/ident_*.json`, `rf_needed_C.json` |
| C. Certification / confidence | `python benchmarks/certify_bench.py audit\|bnb\|big ...` | `python -m eqdisc.confidence ...` | `results_certify/` (dev pools feed `eqdisc.final_model.build_calibration`) |
| Final merged model, dev + frozen test | `python benchmarks/final_bench.py --kind poly3\|poly4\|poly5\|trig\|nd --data ... --arms final ...` (job lists: `results_final/dev2.sh`, `results_final/test.sh`) | `python benchmarks/final_dev_report.py`, `python benchmarks/final_test_report.py` | `results_final/` |
| dysts (77 systems, 6,160 runs) | `python benchmarks/dysts_bench.py --method weak\|c\|c_prior\|final --data data/dysts_test.npz --output results_dysts/test_<m>.json` | `python benchmarks/dysts_report.py results_dysts/test_c.json results_dysts/test_c_prior.json ...` | `results_dysts/` |

The `.sh` job files in `results_*/` and `docs/reports/overnight2/G_dysts_job.sh` are records of what ran on the
Spark (`~/eqdisc_v2`, flat layout). Call the scripts as `benchmarks/<name>.py` here. `--help` on any runner
lists its options.

## Repository map

```
eqdisc/                 the package (final model and everything it imports)
  final_model.py          discover(), Result, DEFAULTS; complexity prior, AR(1) switch, confidence
  joint_discover.py       profiled joint flow (config C)
  joint_colored.py        AR(1) observation-noise likelihood
  joint_biglib.py         beam-search proposals for big libraries; MixedLibrary (poly + trig)
  joint_nd.py             D-dimensional version (degree-2 library), weak starts
  joint_squeeze.py        degree-weighted complexity prior as a standalone variant (discover_sq)
  flow_discovery.py       profiled flow discovery (seeds) and the exhaustive subset ranking
  select_k.py, subset.py  weak-form best-subset selector (chi2s etc.), subset search
  confidence.py           per-term inclusion probabilities, isotonic calibration
  core.py                 library, benchmark systems, simulation, metrics (EQ_DEG)
  heldout_systems.py, nd_systems.py, gen_trig.py   held-out, D-dim and Thomas systems (+ generators)
  confidence_isotonic.json  frozen calibration map (copy of results_final/confidence_isotonic.json)
benchmarks/             generators, runners and report scripts behind every table above
tests/                  equivalence vs ac7ab20, smoke test, flow_discovery unit tests
docs/                   WRITEUP.md (full report), FLOW_DISCOVERY.md, README_final.md, reports/overnight2/
data/                   archives (npz gitignored) + MANIFEST.sha256 + dysts sidecars
results*/               results behind the tables (results_dysts/ is filled from the Spark run)
archive/                superseded experiments and their results (see archive/README.md)
```
