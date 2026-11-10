"""Run the ODR-BINDy port on benchmark archives (same instances as joint_bench / dedi_bench).

ODR-BINDy is given the true per-component noise std (as in the paper's protocol); sigma_y and sigma_p are
the paper's Lorenz heatmap settings unless overridden.
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
    index, y, dt, name, noise, sigma_ys, sigma_p, max_seconds, removal_iters = payload
    from eqdisc.core import SYSTEMS, metrics
    from eqdisc.heldout_systems import HELDOUT
    from odr_bindy import odr_bindy
    truth = (SYSTEMS | HELDOUT)[name]
    sx = noise * y.std(0) / np.sqrt(1 + noise ** 2)
    rec = {"index": index, "system": name, "noise": noise, "n": len(y), "odr": {}}
    for sy in sigma_ys:
        try:
            xi, info = odr_bindy(y, dt, sx, sigma_y=sy, sigma_p=sigma_p, seed=index, max_seconds=max_seconds,
                                  removal_iters=removal_iters)
            rec["odr"][str(sy)] = metrics(xi, truth) | info
        except Exception as exc:
            rec["odr"][str(sy)] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/hybrid_confirm.npz")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--noise", nargs="+", type=float)
    ap.add_argument("--n", nargs="+", type=int)
    ap.add_argument("--indices", nargs="+", type=int)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th selected instance")
    ap.add_argument("--sigma-y", nargs="+", type=float, default=[1e-3])
    ap.add_argument("--sigma-p", type=float, default=100.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--removal-iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with np.load(args.data, allow_pickle=True) as d:
        keep = np.ones(len(d["X"]), bool)
        if args.systems:
            keep &= np.isin(d["sys"], args.systems)
        if args.noise:
            keep &= np.isin(d["noise"], args.noise)
        if args.n:
            keep &= np.isin(d["n"], args.n)
        idx = np.flatnonzero(keep)
        if args.indices:
            idx = np.array([i for i in args.indices if keep[i]])
        idx = idx[::args.every]
        jobs = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                 args.sigma_y, args.sigma_p, args.max_seconds, args.removal_iters) for i in idx]
    print(f"{len(jobs)} runs, sigma_y {args.sigma_y}", flush=True)
    recs, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            s = " ".join(f"{k}:{v.get('exact', 0):.0f}/{v.get('seconds', 0):.0f}s" for k, v in r["odr"].items())
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} {r['noise']} {r['n']} {s}", flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
