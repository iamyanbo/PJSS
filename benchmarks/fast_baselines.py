"""Fast truth-free baselines on exactly the 80 head-to-head runs (results_h2h/order.json): accuracy vs runtime.

    python benchmarks/fast_baselines.py run --method wsindy --worker 0 --nworkers 5     # positions 0::5
    python benchmarks/fast_baselines.py argos-export --outdir results_fast/argos_csv     # CSV + manifests for R
    Rscript benchmarks/argos_run.R <manifest> results_fast/argos_<tag> lasso 200        # see results_fast/job.sh
    python benchmarks/fast_baselines.py collect                                         # -> results_fast/<method>.json
    tables: python benchmarks/fast_baselines_report.py      (doc: docs/reports/fast_baselines.md)

Every method uses the shared 20-term degree-3 monomial library (eqdisc.core.EXP_ARR) and picks its own
sparsity WITHOUT the truth; the truth (eqdisc.core.SYSTEMS) is used only by eqdisc.core.metrics. Seconds are
wall time of the method call (feature building + full threshold path + model selection) on one pinned core
with OMP/OPENBLAS/MKL threads = 1; data loading is excluded.

Methods
  wsindy        WSINDy + MSTLS, line-by-line port of the authors' MATLAB WSINDy_ODE (benchmarks/wsindy_port.py;
                verified identical to the MATLAB code run in Octave, benchmarks/wsindy_octave_check.py), the
                authors' shipped defaults (corner-point test functions, K = 1000, lambda = 10^linspace(-4,0,100),
                alpha_loss = 0.8, optimal moving-average pre-smoothing).
  weak_stlsq    pysindy 2.1.0 WeakPDELibrary + STLSQ along a 41-point threshold path, data rescaled to unit RMS
                ("rms"), test-function half-width h = 10 samples.            } configs = the dev-set choices of
  weak_esindy   same features, pysindy EnsembleOptimizer(STLSQ, bagging, 20)  } benchmarks/baselines_bench.py
  fd_stlsq      Savitzky-Golay-smoothed finite differences (window 11) + STLSQ, original units ("raw"), 55 thresholds
  fd_esindy     same features, E-SINDy (EnsembleOptimizer(STLSQ), bagging, 20 models, median)
                Truth-free picks recorded for all four: "cv" (5-fold regression-row CV of the LS-refit residual,
                ties -> sparser) and "aic" (AICc of LS refits, summed over equations). One run of the path
                computes both picks; the time is for both (CV dominates).
  chi2s         in-repo weak baseline eqdisc.select_k.discover(criterion="chi2s") (exhaustive best subsets
                k <= 5 + self-calibrated chi^2 stopping rule).
  argos         ARGOS (CRAN package 0.1.1, unmodified R code, benchmarks/argos_run.R), lasso, package defaults
                except num_samples (bootstrap count) which is a run-time argument.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results_fast")
PYSINDY = {"weak_stlsq": dict(method="weak_stlsq", scale="rms", window=11, h=10),
           "weak_esindy": dict(method="weak_esindy", scale="rms", window=11, h=10),
           "fd_stlsq": dict(method="fd_stlsq", scale="raw", window=11, h=10),
           "fd_esindy": dict(method="fd_esindy", scale="raw", window=11, h=10)}
METHODS = ["wsindy", "chi2s"] + list(PYSINDY)
# sensitivity variants of WSINDy (fixed, truth-free settings; reported, never used to pick a headline number):
#   wsindy_K1        K_frac = 1 => a test function at every valid centre (the "DEFAULT" in the script's comment)
#   wsindy_nosmooth  smoothing_window = 0 (no moving-average pre-filter)
WSINDY_VARIANTS = {"wsindy": {}, "wsindy_K1": dict(K_frac=1), "wsindy_nosmooth": dict(smooth=False)}


def load_runs(order_path="results_h2h/order.json"):
    from eqdisc.core import SYSTEMS
    order = json.loads(Path(order_path).read_text())
    cache, runs = {}, []
    for pos, r in enumerate(order):
        if r["data"] not in cache:
            with np.load(r["data"], allow_pickle=True) as d:
                cache[r["data"]] = (d["X"], d["dt"])
        X, dt = cache[r["data"]]
        runs.append(dict(pos=pos, key=f"{r['cond']}_{r['index']}", cond=r["cond"], index=int(r["index"]),
                         system=r["system"], noise=r["noise"], n=r["n"], x=np.asarray(X[r["index"]], float),
                         dt=float(dt[r["index"]]), truth=SYSTEMS[r["system"]]))
    return runs


def score(xi, truth):
    from eqdisc.core import metrics
    m = metrics(np.asarray(xi), np.asarray(truth))
    return {k: float(v) for k, v in m.items()}


def run_one(method, run):
    from eqdisc.core import EXP_ARR, NAMES
    rec = dict(key=run["key"], cond=run["cond"], system=run["system"], noise=run["noise"], n=run["n"])
    if method.startswith("wsindy"):
        from wsindy_port import wsindy_ode
        t0 = time.perf_counter()
        W, info = wsindy_ode(run["x"], np.arange(len(run["x"])) * run["dt"], EXP_ARR, return_info=True,
                             **WSINDY_VARIANTS[method])
        rec["seconds"] = time.perf_counter() - t0
        rec["mstls"] = score(W, run["truth"]) | {"xi": W.tolist()}
        rec["info"] = info
    elif method == "chi2s":
        from eqdisc.select_k import discover
        t0 = time.perf_counter()
        W = discover(run["x"], run["dt"], criterion="chi2s")
        rec["seconds"] = time.perf_counter() - t0
        rec["chi2s"] = score(W, run["truth"]) | {"xi": np.asarray(W).tolist()}
    else:
        from baselines_bench import run_path
        cfg = PYSINDY[method]
        r = dict(index=run["index"], x=run["x"], dt=run["dt"], truth=run["truth"].tolist(), names=list(NAMES))
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = run_path(r, cfg["method"], cfg)
        rec["seconds"] = time.perf_counter() - t0
        for pick in ("cv", "aic"):
            rec[pick] = p[pick] | {"xi": p[f"{pick}_xi"]}
        rec["oracle_path"] = p["oracle"]  # reference only: some threshold on the path was exact (uses truth)
        rec["seconds_aic"] = _time_path_aic(r, cfg)  # cost of the AICc rule alone (features + path + AICc)
    return rec


def _time_path_aic(r, cfg):
    """Re-runs exactly the features + threshold path + AICc part of baselines_bench.run_path, timed alone."""
    import baselines_bench as bb
    kind, opt = cfg["method"].split("_", 1)
    x, dt = r["x"], r["dt"]
    np.random.seed(r["index"] % (2 ** 31))
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, s, R = (x, np.ones(3), 1.0) if cfg["scale"] == "raw" else bb.scaling(x, dt)
        A, b, _ = bb.features(kind, u, dt * R, 3, cfg["window"], cfg["h"])
        path = [bb.fit_coef(opt, th, A, b) for th in bb.ths_for(cfg)]
        aic = [sum(bb.aicc(((A @ bb.ls_refit(A, b[:, j], c[:, j] != 0) - b[:, j]) ** 2).sum(), len(A),
                           int((c[:, j] != 0).sum())) for j in range(3)) for c in path]
        int(np.argmin(aic))
    return time.perf_counter() - t0


def cmd_run(a):
    runs = load_runs()
    import pysindy  # noqa: F401  (import cost kept out of the per-run timing)
    import baselines_bench  # noqa: F401
    import wsindy_port  # noqa: F401
    methods = METHODS if a.method == "all" else a.method.split(",")
    for method in methods:
        d = OUT / method
        d.mkdir(parents=True, exist_ok=True)
        for run in runs[a.worker::a.nworkers]:
            f = d / f"{run['key']}.json"
            if f.exists():
                continue
            try:
                rec = run_one(method, run)
            except Exception as e:  # recorded as a failure (non-exact) for that method
                rec = dict(key=run["key"], cond=run["cond"], system=run["system"], error=repr(e))
            f.write_text(json.dumps(rec))
            print(method, run["key"], rec.get("seconds"), flush=True)


def cmd_argos_export(a):
    runs = load_runs()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.nworkers):
        lines = []
        for run in runs[k::a.nworkers]:
            csv = out / f"{run['key']}.csv"
            np.savetxt(csv, run["x"], delimiter=",", fmt="%.17g")
            lines.append(f"{run['key']}\t{csv.resolve()}\t{run['dt']!r}")
        (out / f"manifest_{k}.tsv").write_text("\n".join(lines) + "\n")
    first = [r for r in runs if r["pos"] < a.subset]  # the first `subset` runs (complete rounds)
    lines = [f"{r['key']}\t{(out / (r['key'] + '.csv')).resolve()}\t{r['dt']!r}" for r in first]
    (out / "manifest_subset.tsv").write_text("\n".join(lines) + "\n")


def argos_xi(model):
    """ARGOS term names ('(Intercept)', 'x1', 'x1^2x3', ...) per equation -> (20, 3) over EXP_ARR."""
    from eqdisc.core import EXP_ARR
    rows = [tuple(e) for e in EXP_ARR.tolist()]
    xi = np.zeros((len(rows), 3))
    for j, eq in enumerate(model):
        for name, c in eq.items():
            e = [0, 0, 0]
            if name != "(Intercept)":
                for v, p in re.findall(r"x(\d)(?:\^(\d))?", name):
                    e[int(v) - 1] += int(p) if p else 1
            xi[rows.index(tuple(e)), j] = float(c) if c is not None else 0.0
    return xi


def cmd_collect(a):
    runs = load_runs()
    by_key = {r["key"]: r for r in runs}
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and not p.name.endswith("_csv")):
        recs = []
        for run in runs:
            f = d / f"{run['key']}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            if d.name.startswith("argos"):
                base = dict(key=run["key"], cond=run["cond"], system=run["system"], noise=run["noise"], n=run["n"],
                            seconds=rec["seconds"], num_samples=rec.get("num_samples"))
                if "error" in rec:
                    base["error"] = rec["error"]
                    base["argos"] = dict(exact=0.0, f1=0.0, coef_err=float("nan"))
                else:
                    xi = argos_xi(rec["model"])
                    base["argos"] = score(xi, by_key[run["key"]]["truth"]) | {"xi": xi.tolist()}
                rec = base
            rec["pos"] = run["pos"]
            recs.append(rec)
        (OUT / f"{d.name}.json").write_text(json.dumps(dict(method=d.name, n=len(recs), records=recs), indent=1))
        print(d.name, len(recs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", default="all")
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--nworkers", type=int, default=1)
    e = sub.add_parser("argos-export")
    e.add_argument("--outdir", default="results_fast/argos_csv")
    e.add_argument("--nworkers", type=int, default=5)
    e.add_argument("--subset", type=int, default=10)
    sub.add_parser("collect")
    a = ap.parse_args()
    {"run": cmd_run, "argos-export": cmd_argos_export, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
