"""Track C bench: certified subset search on joint flow's profiled quadratic + per-term confidence.

Modes
  audit   config C (joint_discover.discover, unchanged) with the profiled quadratic of every round captured;
          for each, the exact global optimum of RSS(S) + log(nobs)|S| (bnb_subset.bnb_penalized, node-limited,
          certificate recorded) is compared with the heuristic's proposals/finalists. Also stores every fitted
          candidate's support and score (for posterior inclusion probabilities, confidence.py).
  bnb     discover_bnb: config C with the certified global optimum injected into each round's finalists.
  big     degree-4/5 libraries (EQ_DEG in env): joint_biglib with its beam replaced by exact bnb_topk
          (per-equation top-`keep` per size, certified); starts as in biglib_bench.
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


def starts_for(x, dt):
    from eqdisc.select_k import discover as weak_discover
    from eqdisc.flow_discovery import discover
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
    return starts, weak, hyb


def _cands(info, n, truth):
    from eqdisc.core import metrics
    out = []
    for c in info["candidates"]:
        s = c["data"] + c["model"] + c["val"] + info["phi"] * c["k"] * np.log(3 * n)
        if not np.isfinite(s):
            continue
        m = metrics(np.array(c["xi"]), truth)
        sup = np.flatnonzero(np.array(c["xi"]).ravel() != 0)
        out.append({"k": c["k"], "score": float(s), "exact": m["exact"], "fp": m["fp"], "fn": m["fn"],
                    "support": sup.tolist()})
    return sorted(out, key=lambda c: c["score"])


def _job(payload):
    index, x, dt, name, noise, mode, nodes = payload
    from threadpoolctl import threadpool_limits
    from eqdisc.core import SYSTEMS, metrics, NLIB
    from eqdisc.heldout_systems import HELDOUT
    from eqdisc import joint_discover as jd
    import bnb_subset as bs
    truth = (SYSTEMS | HELDOUT)[name]
    tkey = tuple(np.flatnonzero(truth.ravel() != 0).tolist())
    rec = {"index": index, "system": name, "noise": noise, "n": len(x), "truth": list(tkey)}
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        if mode == "big":
            from eqdisc import select_k
            from eqdisc import joint_biglib as jb
            weak = select_k.discover(x, dt, criterion=["chi2s", "chi2", "bic"])
            rec["weak"] = metrics(weak["chi2s"], truth)
            stats = {"calls": 0, "uncertified": 0, "nodes": 0}

            def exact_top(a, b, kmax, keep=2):
                out, cert, nn = bs.bnb_topk(a, b, kmax, keep, node_limit=nodes,
                                            incumbents=[s for s, _ in jb.beam_subsets(a, b, kmax, keep, 60)])
                stats["calls"] += 1; stats["uncertified"] += (not cert); stats["nodes"] += nn
                return out
            old = jd._top_subsets
            jd._top_subsets = exact_top
            try:
                xi, info = jd.discover(x, dt, [weak["chi2s"], weak["chi2"], weak["bic"]], record=True)
            finally:
                jd._top_subsets = old
            rec["joint"] = metrics(xi, truth) | {"phi": info["phi"]}
            rec["cands"] = _cands(info, len(x), truth)
            rec["bnb"] = stats
            rec["seconds"] = time.perf_counter() - t0
            return rec
        starts, weak, hyb = starts_for(x, dt)
        rec["weak"], rec["hyb"] = metrics(weak, truth), metrics(hyb, truth)
        rec["base_seconds"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        if mode == "audit":
            with bs.capture_quadratics() as qlog:
                xi, info = jd.discover(x, dt, starts, record=True)
            audits = []
            nobs = 3 * len(x)
            for q in (qlog if nodes > 0 else []):
                cur = tuple(np.flatnonzero(q["cur"].ravel()).tolist())
                ta = time.perf_counter()
                res, lb, cert, nn = bs.miqp_penalized(q["A"], q["b"], np.log(nobs), n_best=3,
                                                      time_limit=nodes / 1000, incumbents=[cur])
                G = bs._Gram(q["A"], q["b"])
                audits.append({"opt": [int(i) for i in res[0][0]], "opt_val": float(res[0][1]), "lower": float(lb),
                               "certified": bool(cert), "nodes": int(nn), "seconds": time.perf_counter() - ta,
                               "next": [[int(i) for i in r[0]] for r in res[1:]],
                               "next_val": [float(r[1]) for r in res[1:]],
                               "truth_val": G.rss(list(tkey)) + np.log(nobs) * len(tkey),
                               "cur": list(cur), "cur_val": G.rss(list(cur)) + np.log(nobs) * len(cur),
                               "opt_is_truth": tuple(int(i) for i in res[0][0]) == tkey})
            rec["audit"] = audits
        else:
            blog = []
            xi, info = bs.discover_bnb(x, dt, starts, record=True, bnb_log=blog, bnb_time=nodes / 1000, bnb_extra=3)
            rec["bnb_log"] = blog
        rec["joint"] = metrics(xi, truth) | {"phi": info["phi"]}
        rec["cands"] = _cands(info, len(x), truth)
        rec["joint_seconds"] = time.perf_counter() - t1
    return rec


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/instances.npz")
    ap.add_argument("--mode", choices=["audit", "bnb", "big"], default="audit")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--indices", nargs="+", type=int)
    ap.add_argument("--nodes", type=int, default=20000,
                    help="audit/bnb: MIQP time limit in ms per quadratic (0 = skip audit); big: bnb_topk node limit")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with np.load(args.data, allow_pickle=True) as d:
        idx = list(range(len(d["X"])))[args.offset::args.every]
        if args.indices:
            idx = args.indices
        jobs = [(i, np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                 args.mode, args.nodes) for i in idx]
    print(f"{len(jobs)} runs mode {args.mode}", flush=True)
    recs, t0 = [], time.perf_counter()
    part = out.with_suffix(".partial.json")
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            try:
                r = f.result()
            except Exception as exc:
                print("error", type(exc).__name__, exc, flush=True)
                continue
            recs.append(r)
            extra = ""
            if "audit" in r:
                extra = " audit " + " ".join(f"{int(a['certified'])}{int(a['opt_is_truth'])}" for a in r["audit"])
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} {r['noise']} {r['n']} "
                  f"joint {r['joint']['exact']:.0f}{extra}", flush=True)
            if len(recs) % 50 == 0:
                part.write_text(json.dumps({"args": vars(args), "records": recs}, default=_js))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}, default=_js))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
