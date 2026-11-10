"""Benchmark the AR(1)-prewhitened joint flow (joint_colored) with the same starts as joint_bench.

Variants: 'ar' (AR(1) noise model, frozen v2), 'ar_conc' (with concentrated-likelihood score) and 'noar'
(a = 0; reproduces config C).
Every fitted candidate is stored with its scores; truth is used only for scoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


def _job(payload):
    index, x, dt, name, noise, variants = payload
    from eqdisc.core import SYSTEMS, metrics
    from eqdisc.select_k import discover as weak_discover
    from eqdisc.flow_discovery import discover
    from eqdisc import joint_colored as jc
    from eqdisc.heldout_systems import HELDOUT
    truth = (SYSTEMS | HELDOUT)[name]
    rec = {"index": index, "system": name, "noise": noise, "n": len(x)}
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        weak = weak_discover(x, dt, criterion="chi2s")
        rec["weak"] = metrics(weak, truth)
        starts = [weak]
        try:
            res = discover(x, dt, extra_models=[weak])
        except Exception as exc:
            rec["flow_error"] = f"{type(exc).__name__}: {exc}"
            rec["hyb_bic10"] = rec["weak"]
        else:
            alts = res.alternatives
            own = [a for a in alts if not a["extra_only"]]
            w = [a for a in alts if a["extra"] == 0]
            gap = w[0]["selection_score"] - alts[0]["selection_score"] if w else None
            hyb = weak if gap is not None and gap <= 10 else res.coefficients
            rec["hyb_bic10"] = metrics(hyb, truth)
            starts += [hyb] + [np.array(a["coefficients"]) for a in own[:3]]
        rec["base_seconds"] = time.perf_counter() - start
        rec["variants"] = {}
        for v in variants:
            t = time.perf_counter()
            try:
                kw = {"ar": v.startswith("ar"), "score_rule": "conc" if v.endswith("conc") else "chi2"}
                xi, info = jc.discover(x, dt, starts, record=True, **kw)
            except Exception as exc:
                rec["variants"][v] = {"error": f"{type(exc).__name__}: {exc}", "final": {"exact": 0.0}}
                continue
            cands = []
            for c in info["candidates"]:
                m = metrics(np.array(c["xi"]), truth)
                cands.append({k: c[k] for k in ("k", "data", "model", "val", "converged", "dcomp", "ncomp")} |
                             {"exact": m["exact"], "fp": m["fp"], "fn": m["fn"], "coef_err": m["coef_err"]})
            rec["variants"][v] = {"final": metrics(xi, truth), "candidates": cands, "phi": info["phi"],
                                  "ar": info["ar"], "lam": info.get("lam", 1.0), "substeps": info["substeps"],
                                  "seconds": time.perf_counter() - t}
    rec["seconds"] = time.perf_counter() - start
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/instances_colored.npz")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--noise", nargs="+", type=float)
    ap.add_argument("--n", nargs="+", type=int)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=["ar"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        ap.error(f"{out} exists; refusing to overwrite")
    with np.load(args.data, allow_pickle=True) as d:
        keep = np.ones(len(d["X"]), bool)
        if args.systems:
            keep &= np.isin(d["sys"], args.systems)
        if args.noise:
            keep &= np.isin(np.round(d["noise"].astype(float), 4), np.round(args.noise, 4))
        if args.n:
            keep &= np.isin(d["n"], args.n)
        tasks = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                  tuple(args.variants)) for i in np.flatnonzero(keep)[::args.every]]
    print(f"{len(tasks)} cases", flush=True)
    records, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(_job, tasks, chunksize=1):
            records.append(rec)
            if len(records) % 20 == 0 or len(records) == len(tasks):
                print(f"{len(records)}/{len(tasks)} {time.perf_counter() - start:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"data": args.data, "variants": args.variants, "records": records}, default=float),
                   encoding="utf-8")
    print(f"saved {out} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
