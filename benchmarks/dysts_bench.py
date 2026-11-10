"""Run one discovery method on a dysts archive (data/dysts_*.npz from dysts_generate.py).

    python dysts_bench.py --method c       --data data/dysts_smoke.npz --workers 3 --output results_dysts/smoke_c.json
    python dysts_bench.py --method c_prior --data data/dysts_test.npz  --workers 16 --output results_dysts/test_c_prior.json

Truth comes from the sidecar <data>.json (never shown to the methods).
Methods
  weak     D = 3: select_k.discover(criterion="chi2s") (the weak chi2 selector);  D >= 4: joint_nd.weak_starts[0]
  c        D = 3: starts exactly as joint_bench/squeeze_bench (weak chi2s, hybrid, top-3 flow) then
                  joint_squeeze.discover_sq with no options (= frozen config C);
           D >= 4: joint_nd.discover seeded with joint_nd.weak_starts (config C in D dimensions)
  c_prior  D = 3: as c with discover_sq(degw=0.2) (degree-weighted complexity prior);
           D >= 4: not available outside final_model -> record status "unsupported" (excluded by the report)
  final    final_model.discover(x, dt) with its frozen defaults (imported lazily inside the worker)
Every record also stores the weak (and, for D = 3 c/c_prior, hybrid) exact flags as paired baselines.
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

METHODS = ("weak", "c", "c_prior", "final")


def _starts3(x, dt):
    """identical to squeeze_bench._starts_job"""
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
    return [np.asarray(s) for s in starts], np.asarray(weak), np.asarray(hyb)


def _job(payload):
    index, x, dt, name, noise, method, truth = payload
    from threadpoolctl import threadpool_limits
    from eqdisc.core import metrics
    truth = np.asarray(truth)
    D = x.shape[1]
    rec = {"index": index, "system": name, "noise": noise, "n": len(x), "D": D, "method": method}
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        try:
            if method == "final":
                from eqdisc import final_model
                res = final_model.discover(x, dt)
                xi = np.asarray(res.xi)
                rec["ambiguous"], rec["gap"] = bool(res.ambiguous), float(res.gap)
            elif D == 3:
                if method == "weak":
                    from eqdisc.select_k import discover as weak_discover
                    xi = weak_discover(x, dt, criterion="chi2s")
                else:
                    from eqdisc.joint_squeeze import discover_sq
                    starts, weak, hyb = _starts3(x, dt)
                    rec["weak"] = metrics(weak, truth)["exact"]
                    rec["hyb"] = metrics(hyb, truth)["exact"]
                    xi, info = discover_sq(x, dt, starts, **({"degw": 0.2} if method == "c_prior" else {}))
                    rec["gap"] = float(info["gap"])
                    rec["fits"] = info.get("models_fitted")
            else:
                from eqdisc import joint_nd as jn
                from eqdisc.joint_nd import PolyLib
                lib = PolyLib(D, 2)
                if method == "c_prior":
                    rec["status"] = "unsupported"
                    rec["seconds"] = 0.0
                    return rec
                starts = jn.weak_starts(x, dt, lib)
                rec["weak"] = metrics(starts[0], truth)["exact"]
                if method == "weak":
                    xi = starts[0]
                else:
                    xi, info = jn.discover(x, dt, starts, lib)
                    rec["gap"] = info.get("gap")
                    rec["fits"] = info.get("models_fitted")
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}", final={"exact": 0.0},
                       seconds=time.perf_counter() - t0)
            return rec
    xi = np.asarray(xi)
    if xi.shape != truth.shape:
        rec.update(status="error", error=f"shape {xi.shape} != truth {truth.shape}", final={"exact": 0.0})
    else:
        rec["status"] = "ok"
        rec["final"] = metrics(xi, truth)
        rec["xi"] = xi.tolist()
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--data", default="data/dysts_test.npz")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    side = json.loads(Path(args.data).with_suffix(".json").read_text())["systems"]
    with np.load(args.data, allow_pickle=True) as d:
        idx = [i for i in range(len(d["X"])) if not args.systems or d["sys"][i] in args.systems][::args.every]
        jobs = [(i, np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                 args.method, side[str(d["sys"][i])]["xi"]) for i in idx]
    jobs.sort(key=lambda j: -(j[1].shape[1] ** 2 * len(j[1])))  # longest (D, N) first
    print(f"{len(jobs)} runs, method {args.method}", flush=True)
    recs, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} D={r['D']} {r['noise']} {r['n']} "
                  f"{r['status']} exact {r.get('final', {}).get('exact', '-')} weak {r.get('weak', '-')} "
                  f"{r['seconds']:.1f}s {r.get('error', '')}", flush=True)
            if len(recs) % 100 == 0:
                out.with_suffix(".partial.json").write_text(json.dumps({"records": recs}))
    out.write_text(json.dumps({"args": vars(args), "method": args.method,
                               "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}))
    part = out.with_suffix(".partial.json")
    if part.exists():
        part.unlink()
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"saved {out}: {len(ok)}/{len(recs)} ok, exact {np.mean([r['final']['exact'] for r in ok]) if ok else 0:.3f}, "
          f"median s {np.median([r['seconds'] for r in recs]):.1f}, wall {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
