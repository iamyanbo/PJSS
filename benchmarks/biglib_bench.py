"""Benchmark profiled joint flow on large libraries: polynomial (set EQ_DEG=4 or 5 in the environment) or
`--lib mixed` (degree-3 polynomials + sin/cos of each state, 26 terms; truth for polynomial systems is padded).

Starts for joint: weak-form selectors on the same big library (beam search inside select_k) with
criteria chi2s, chi2 and bic. Flow discovery does not support libraries this large (its exhaustive
guard raises), so the hybrid equals weak chi2s here and is not reported separately.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np


def _job(payload):
    index, x, dt, name, noise, width, lib = payload
    import contextlib
    from eqdisc.core import SYSTEMS, NLIB, metrics
    from eqdisc.heldout_systems import HELDOUT
    from eqdisc import select_k
    from eqdisc import joint_biglib as jb
    if lib == "mixed":
        from eqdisc.gen_trig import TRIG
        L = jb.MixedLibrary()
        if name in TRIG:
            truth = TRIG[name]
        else:
            truth = np.zeros((L.M, 3))
            truth[:NLIB] = (SYSTEMS | HELDOUT)[name]
        ctx, nlib = jb.use_library(L, x), L.M
    else:
        truth, ctx, nlib = (SYSTEMS | HELDOUT)[name], contextlib.nullcontext(), NLIB
    rec = {"index": index, "system": name, "noise": noise, "n": len(x), "nlib": nlib, "lib": lib}
    with ctx:
        _run(rec, x, dt, truth, width, select_k, jb, metrics)
    return rec


def _run(rec, x, dt, truth, width, select_k, jb, metrics):
    t = time.perf_counter()
    weak = select_k.discover(x, dt, criterion=["chi2s", "chi2", "bic"])
    rec["weak_seconds"] = time.perf_counter() - t
    for c, xi in weak.items():
        rec[f"weak_{c}"] = metrics(xi, truth)
    t = time.perf_counter()
    try:
        xi, info = jb.discover(x, dt, [weak["chi2s"], weak["chi2"], weak["bic"]], width=width, record=True)
        cands = []
        for c in info["candidates"]:
            m = metrics(np.array(c["xi"]), truth)
            cands.append({k: c[k] for k in ("k", "data", "model", "val", "converged")} |
                         {"exact": m["exact"], "fp": m["fp"], "fn": m["fn"]})
        rec["joint"] = metrics(xi, truth) | {"phi": info["phi"], "substeps": info["substeps"],
                                             "models_fitted": info["models_fitted"], "candidates": cands}
    except Exception as exc:
        rec["joint"] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    rec["joint_seconds"] = time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/instances.npz")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--noise", nargs="+", type=float)
    ap.add_argument("--n", nargs="+", type=int)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--lib", choices=["poly", "mixed"], default="poly",
                    help="poly: core library at EQ_DEG; mixed: degree-3 polynomials + sin/cos of each state")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    from eqdisc.core import NLIB
    with np.load(args.data, allow_pickle=True) as d:
        keep = np.ones(len(d["X"]), bool)
        if args.systems:
            keep &= np.isin(d["sys"], args.systems)
        if args.noise:
            keep &= np.isin(d["noise"], args.noise)
        if args.n:
            keep &= np.isin(d["n"], args.n)
        idx = np.flatnonzero(keep)[args.offset::args.every]
        jobs = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                 args.width, args.lib) for i in idx]
    print(f"{len(jobs)} runs, NLIB {NLIB}, EQ_DEG {os.environ.get('EQ_DEG')}", flush=True)
    recs, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} {r['noise']} {r['n']} weak "
                  f"{r['weak_chi2s']['exact']:.0f} joint {r['joint']['exact']:.0f} "
                  f"{r['weak_seconds']:.1f}+{r['joint_seconds']:.1f}s {r['joint'].get('error', '')}", flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "eq_deg": os.environ.get("EQ_DEG"),
                               "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
