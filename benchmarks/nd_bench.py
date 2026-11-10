"""Benchmark joint_nd on higher-dimensional archives (data/nd_*.npz from nd_systems.generate).

Per run: weak-form best-subset BIC (the tuning-free start, reported as a baseline), WSINDy-style STLSQ on
the same weak problem over a threshold grid (reported per threshold, so a dev-chosen or oracle threshold
can be read off), and joint_nd seeded with the weak starts.
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

LAMS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _job(payload):
    index, x, dt, name, noise, kw = payload
    from eqdisc.core import metrics
    from eqdisc.nd_systems import system
    from eqdisc import joint_nd as jn
    lib, Xi = system(name)
    rec = {"index": index, "system": name, "noise": noise, "n": len(x), "D": lib.D}
    t = time.perf_counter()
    starts = jn.weak_starts(x, dt, lib)
    rec["weak_bic"] = metrics(starts[0], Xi)
    rec["weak_seconds"] = time.perf_counter() - t
    rec["stlsq"] = {}
    for lam in LAMS:
        try:
            rec["stlsq"][str(lam)] = metrics(jn.wsindy_stlsq(x, dt, lib, lam), Xi)
        except Exception as exc:
            rec["stlsq"][str(lam)] = {"exact": 0.0, "error": str(exc)}
    t = time.perf_counter()
    try:
        xi, info = jn.discover(x, dt, starts, lib, **kw)
        rec["joint"] = metrics(xi, Xi)
        rec["info"] = info
    except Exception as exc:
        rec["joint"] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    rec["joint_seconds"] = time.perf_counter() - t
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/nd_dev.npz")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--substep-cap", type=int, default=12)
    ap.add_argument("--finalists", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    kw = {"substep_cap": args.substep_cap, "finalists": args.finalists}
    with np.load(args.data, allow_pickle=True) as d:
        idx = [i for i in range(len(d["X"])) if not args.systems or d["sys"][i] in args.systems][::args.every]
        jobs = [(i, np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]), kw)
                for i in idx]
    # longest jobs first (D, N)
    jobs.sort(key=lambda j: -(j[1].shape[1] ** 2 * len(j[1])))
    print(f"{len(jobs)} runs", flush=True)
    recs, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} {r['noise']} {r['n']} "
                  f"weak {r['weak_bic']['exact']:.0f} joint {r['joint'].get('exact', 0):.0f} "
                  f"{r['joint_seconds']:.0f}s", flush=True)
            if len(recs) % 40 == 0:
                out.with_suffix(".partial.json").write_text(json.dumps({"records": recs}))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
