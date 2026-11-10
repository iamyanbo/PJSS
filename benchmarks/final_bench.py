"""Benchmark runner for the final merged model (final_model.py): dev crossed experiments and the frozen test pass.

  python final_bench.py --kind poly3 --data data/instances_colored.npz --arms C Cp AR ARp --every 2 \
         --output results_final/dev_colored.json [--starts results_squeeze/starts_dev.pkl]
  EQ_DEG=5 python final_bench.py --kind poly5 --data data/instances.npz --arms off p15 pmid pmean ...
  python final_bench.py --kind trig  --data data/trig_dev.npz --arms off p15 ...
  python final_bench.py --kind nd    --data data/nd_dev.npz --arms off p15 p10 ...
  python final_bench.py --kind poly3 --data data/joint_sealed.npz --arms final ...   (frozen public API)

Per run the seeds (starts) are computed once (or read from a squeeze_bench cache) and shared by all arms, so
arms differ only in the joint stage. The AR(1) pre-estimate a_hat (final_model.estimate_ar) is recorded for
every 3-D run. Arm "final" calls the public final_model.discover with its frozen DEFAULTS (only the seeds are
passed in). Truth is used only for scoring.
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

# arm -> options for final_model._discover3 / _discover_nd (None = public API with DEFAULTS)
ARMS = {
    # 3-D, noise x prior
    "C": dict(noise="white", prior=False),
    "Cp": dict(noise="white", prior=True, centre=1.5),
    "AR": dict(noise="ar1", prior=False),
    "ARp": dict(noise="ar1", prior=True, centre=1.5),
    "ARp_e14": dict(noise="ar1", prior=True, centre=1.5, pen_exp=0.25),
    "ARp_e12": dict(noise="ar1", prior=True, centre=1.5, pen_exp=0.5),
    "ARp_e0": dict(noise="ar1", prior=True, centre=1.5, pen_exp=0.0),
    # big / trig libraries and nd: prior centre
    "off": dict(noise="white", prior=False),
    "p15": dict(noise="white", prior=True, centre=1.5),
    "pmid": dict(noise="white", prior=True, centre="mid"),
    "pmean": dict(noise="white", prior=True, centre="mean"),
    "p10": dict(noise="white", prior=True, centre=1.0),
    "final": None,          # public API with the current DEFAULTS (v1.1 since 2026-10-26, v1.2 since 2026-10-27)
    "final_v101": "1.0.1",  # public API, version="1.0.1" (= frozen v1.0 + empty-support fix)
    "final_v11": "1.1",     # public API, version="1.1"
    "final_v12": "1.2",     # public API, version="1.2" (v1.1 + discretisation-error noise floor; precision audit)
}


def _truth(kind, name, L=None):
    from eqdisc.core import SYSTEMS, NLIB
    from eqdisc.heldout_systems import HELDOUT
    if kind == "trig":
        from eqdisc.gen_trig import TRIG
        if name in TRIG:
            return TRIG[name]
        t = np.zeros((L.M, 3))
        t[:NLIB] = (SYSTEMS | HELDOUT)[name]
        return t
    return (SYSTEMS | HELDOUT)[name]


def _arm_opts(opts):
    from eqdisc import final_model as fm
    d = fm.DEFAULTS
    return {"prior": opts.get("prior", False), "prior_ar1": opts.get("prior", False),
            "degw": opts.get("degw", d["degw"]), "centre": opts.get("centre", 1.5),
            "thr": opts.get("thr", d["ar_threshold"]), "pen_exp": opts.get("pen_exp", 1 / 3)}


def _summ(res, truth, t):
    from eqdisc.core import metrics
    m = metrics(res.xi, truth)
    i = res.info
    return m | {"gap": res.gap, "ambiguous": res.ambiguous, "seconds": time.perf_counter() - t,
                "noise_model": i.get("noise_model"), "lam": i.get("lam"), "ar": i.get("ar"),
                "phi": i.get("phi"), "prior": i.get("prior"), "centre": i.get("centre"),
                "fits": i.get("models_fitted"), "xi": np.asarray(res.xi).tolist(),
                "sigma": i.get("sigma"), "floor": i.get("sigma_floor_info"),
                "conf": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in res.confidence.items()}}


def _job(payload):
    index, x, dt, name, noise, kind, arms, starts = payload
    import contextlib
    from threadpoolctl import threadpool_limits
    from eqdisc import final_model as fm
    rec = {"index": index, "system": name, "noise": noise, "n": len(x), "arms": {}}
    with threadpool_limits(limits=1):
        if kind == "nd":
            return _job_nd(rec, x, dt, name, arms, fm)
        library = {"poly3": "poly3", "poly4": "poly4", "poly5": "poly5", "trig": "trig3"}[kind]
        ctx = fm._trig_library(x) if kind == "trig" else (fm._big_search() if kind in ("poly4", "poly5")
                                                         else contextlib.nullcontext())
        with ctx as L:
            truth = _truth(kind, name, L)
            t = time.perf_counter()
            if starts is None:
                starts = fm.default_starts(x, dt, library)
            rec["starts_seconds"] = time.perf_counter() - t
            from eqdisc.core import metrics
            rec["weak"] = metrics(starts[0], truth)["exact"]
            t = time.perf_counter()
            try:
                rec["a_hat"] = fm.estimate_ar(x, dt, starts).tolist()
            except Exception as exc:
                rec["a_hat_error"] = f"{type(exc).__name__}: {exc}"
            rec["a_seconds"] = time.perf_counter() - t
            for arm in arms:
                t = time.perf_counter()
                try:
                    if ARMS[arm] is None or isinstance(ARMS[arm], str):
                        hub = fm.VERSIONS[ARMS[arm]]["huber_ar1"] if ARMS[arm] else None
                        dfl = fm.VERSIONS[ARMS[arm]]["disc_floor"] if ARMS[arm] else fm.DEFAULTS["disc_floor"]
                        res = fm._discover3(x, dt, starts, library, "auto",
                                            fm._options(library, None, None, None, None, None, hub), L, True,
                                            False, {"disc_floor": True} if dfl else {})
                    else:
                        o = ARMS[arm]
                        res = fm._discover3(x, dt, starts, library, o["noise"], _arm_opts(o), L, True, False, {})
                    rec["arms"][arm] = _summ(res, truth, t)
                except Exception as exc:
                    rec["arms"][arm] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}",
                                        "seconds": time.perf_counter() - t}
    return rec


def _job_nd(rec, x, dt, name, arms, fm):
    from eqdisc.nd_systems import system
    from eqdisc import joint_nd as jn
    from eqdisc.core import metrics
    lib, truth = system(name)
    rec["D"] = lib.D
    t = time.perf_counter()
    starts = jn.weak_starts(x, dt, lib)
    rec["starts_seconds"] = time.perf_counter() - t
    rec["weak"] = metrics(starts[0], truth)["exact"]
    for arm in arms:
        t = time.perf_counter()
        try:
            if ARMS[arm] is None or isinstance(ARMS[arm], str):
                res = fm.discover(x, dt, starts=starts, version=ARMS[arm])
            else:
                o = ARMS[arm]
                res = fm._discover_nd(x, dt, "white", starts, o.get("prior", False), o.get("degw"),
                                      o.get("centre", 1.5), True, False, {})
            rec["arms"][arm] = _summ(res, truth, t)
        except Exception as exc:
            rec["arms"][arm] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}",
                                "seconds": time.perf_counter() - t}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["poly3", "poly4", "poly5", "trig", "nd"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--starts", help="squeeze_bench starts cache (poly3 only)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--index-file", help="json list of archive indices to run (overrides every/offset)")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    for a in args.arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}")
    want = {"poly4": "4", "poly5": "5"}.get(args.kind, "3")
    if os.environ.get("EQ_DEG", "3") != want:
        raise SystemExit(f"--kind {args.kind} needs EQ_DEG={want} in the environment")
    S = pickle.loads(Path(args.starts).read_bytes()) if args.starts else {}
    with np.load(args.data, allow_pickle=True) as d:
        if args.index_file:
            idx = json.loads(Path(args.index_file).read_text())
        else:
            idx = list(range(len(d["X"])))[args.offset::args.every]
        jobs = [(int(i), np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                 args.kind, tuple(args.arms), S[int(i)]["starts"] if int(i) in S else None) for i in idx]
    jobs.sort(key=lambda j: -(j[1].shape[1] ** 2 * len(j[1])))  # longest first
    print(f"{len(jobs)} runs, kind {args.kind}, arms {args.arms}, cached starts {sum(j[-1] is not None for j in jobs)}",
          flush=True)
    recs, t0 = [], time.perf_counter()
    out.parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            print(f"[{len(recs)}/{len(jobs)}] {r['index']} {r['system']} {r['noise']} {r['n']} " +
                  " ".join(f"{a}={r['arms'][a].get('exact', 0):.0f}" for a in args.arms) +
                  f" {sum(v.get('seconds', 0) for v in r['arms'].values()):.0f}s" +
                  "".join(f" ERR[{a}]{v['error'][:80]}" for a, v in r['arms'].items() if 'error' in v), flush=True)
            if len(recs) % 50 == 0:
                out.with_suffix(".partial.json").write_text(json.dumps({"records": recs}, default=float))
    out.write_text(json.dumps({"args": vars(args), "eq_deg": os.environ.get("EQ_DEG"),
                               "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}, default=float))
    print("saved", out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
