"""Overnight squeeze track: cached starts + joint-flow variants.

  python squeeze_bench.py starts --data data/instances.npz --output results_squeeze/starts_dev.pkl
  python squeeze_bench.py run --data data/instances.npz --starts results_squeeze/starts_dev.pkl \
         --variant anneal --output results_squeeze/dev_anneal.json [--diag]

Starts (weak chi2s, hybrid, top-3 flow alternatives) are computed exactly as in joint_bench and cached, so
variants differ only in the joint stage. --diag fits the true support (diagnostic only) and classifies.
"""
import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

VARIANTS = {
    "C": {},
    "anneal": {"rho_path": [0.3, 0.1, 0.03]},
    "anneal2": {"rho_path": [0.1, 0.03]},
    "wide": {"rounds": 4, "finalists": 20, "keep": 3},
    "swaps": {"swaps": True},
    "smooth": {"smoothers": (11, 21, 41)},
    "ebic": {"ebic": 0.5},
    "deg2": {"degw": 0.2},
    "deg3": {"degw": 0.3},
    "deg4": {"degw": 0.4},
    "deg6": {"degw": 0.6},
    "ev3": {"evidence": 3.0},
    "ev10": {"evidence": 10.0},
    "ev30": {"evidence": 30.0},
}


def _starts_job(payload):
    index, x, dt = payload
    from threadpoolctl import threadpool_limits
    from eqdisc.select_k import discover as weak_discover
    from eqdisc.flow_discovery import discover
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        weak = weak_discover(x, dt, criterion="chi2s")
        starts, hyb = [weak], weak
        try:
            res = discover(x, dt, extra_models=[weak])
            alts = res.alternatives
            own = [a for a in alts if not a["extra_only"]]
            w = [a for a in alts if a["extra"] == 0]
            gap = w[0]["selection_score"] - alts[0]["selection_score"] if w else None
            hyb = weak if gap is not None and gap <= 10 else res.coefficients
            starts += [hyb] + [np.array(a["coefficients"]) for a in own[:3]]
        except Exception:
            pass
    return index, {"starts": [np.asarray(s) for s in starts], "weak": np.asarray(weak), "hyb": np.asarray(hyb),
                   "seconds": time.perf_counter() - t0}


def _run_job(payload):
    index, x, dt, name, noise, st, opts, diag = payload
    from threadpoolctl import threadpool_limits
    from eqdisc.core import SYSTEMS, metrics
    from eqdisc.heldout_systems import HELDOUT
    from eqdisc.joint_squeeze import discover_sq, classify
    truth = (SYSTEMS | HELDOUT)[name]
    rec = {"index": index, "system": name, "noise": noise, "n": len(x),
           "weak": metrics(st["weak"], truth)["exact"], "hyb": metrics(st["hyb"], truth)["exact"]}
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        try:
            xi, info = discover_sq(x, dt, st["starts"], truth=truth if diag else None, **opts)
        except Exception as exc:
            rec.update(error=f"{type(exc).__name__}: {exc}", final=metrics(st["weak"], truth))
            return rec
    rec["final"] = metrics(xi, truth)
    rec["seconds"] = time.perf_counter() - t0
    rec["fits"] = info["models_fitted"]
    rec["gap"] = float(info["gap"])
    if diag:
        rec["diag"] = info["diag"]
        rec["class"] = "ok" if rec["final"]["exact"] else classify(info["diag"])
    return rec


def _load(args):
    d = np.load(args.data, allow_pickle=True)
    idx = np.arange(len(d["X"]))[::args.every]
    return d, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["starts", "run"])
    ap.add_argument("--data", default="data/instances.npz")
    ap.add_argument("--starts")
    ap.add_argument("--variant", default="C")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(exist_ok=True)
    d, idx = _load(args)
    t0 = time.perf_counter()
    if args.mode == "starts":
        jobs = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i])) for i in idx]
        res = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for f in as_completed([ex.submit(_starts_job, j) for j in jobs]):
                i, r = f.result()
                res[i] = r
                if len(res) % 50 == 0:
                    print(f"starts {len(res)}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
        out.write_bytes(pickle.dumps(res))
        print("saved", out, flush=True)
        return
    S = pickle.loads(Path(args.starts).read_bytes())
    opts = VARIANTS[args.variant]
    jobs = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
             S[int(i)], opts, args.diag) for i in idx if int(i) in S]
    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_run_job, j) for j in jobs]):
            recs.append(f.result())
            if len(recs) % 50 == 0:
                print(f"{args.variant} {len(recs)}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    out.write_text(json.dumps({"args": vars(args), "opts": {k: list(v) if isinstance(v, tuple) else v
                                                            for k, v in opts.items()},
                               "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}))
    ex_ = np.mean([r["final"]["exact"] for r in recs])
    print(f"saved {out}: exact {ex_:.3f}, median s {np.median([r.get('seconds', 0) for r in recs]):.1f}", flush=True)


if __name__ == "__main__":
    main()
