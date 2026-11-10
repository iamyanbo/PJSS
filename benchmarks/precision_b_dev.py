"""Precision audit, part B (dev only): is the dysts tail numerical?

  cond   per dysts system (all in-library systems of data/dysts_test.json; truth from the sidecar only, fresh
         clean trajectories from dysts_generate.prepare, i.e. NOT the test runs): condition numbers of the
         library matrix in the model's own scaling (u = x / rms(x)), column-equilibrated, and of a centred
         basis ((x - mean) / std) for comparison; per-equation condition of the true support.
  runs   on a dev archive generated with a NEW seed (dysts_generate.py --seed 20261777, failing + control
         systems): arm v11 (public discover, instrumented with precision_numerics.record()) and arm hard
         (the same call inside precision_numerics.hardened(): QR subset RSS, no ridges, eigen-factor of H) and
         arm qr (hardened + square-root/QR construction of the profiled quadratic, white path).

    python benchmarks/precision_b_dev.py cond --output results_precision/b_cond.json
    python benchmarks/precision_b_dev.py runs --data data/dysts_precdev.npz --output results_precision/b_dev.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _cond(M):
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def cond_stats(x, E, Xi):
    from dysts_generate import lib_eval
    u = x / np.sqrt(np.mean(x ** 2, axis=0))
    Th = lib_eval(u, E)
    Thn = Th / np.linalg.norm(Th, axis=0)
    uc = (x - x.mean(0)) / x.std(0)
    Thc = lib_eval(uc, E)
    Thc = Thc / np.linalg.norm(Thc, axis=0)
    sup = [float(_cond(Thn[:, np.flatnonzero(Xi[:, j])])) for j in range(x.shape[1]) if np.any(Xi[:, j])]
    return {"cond_raw": _cond(Th), "cond_eq": _cond(Thn), "cond_centred_eq": _cond(Thc),
            "cond_true_support_max": max(sup), "offset_max": float(np.max(np.abs(x.mean(0)) / x.std(0)))}


def _cond_job(name):
    from dysts_generate import prepare
    C = json.load(open("data/dysts_classes.json"))
    side = json.load(open("data/dysts_test.json"))["systems"][name]
    r = C["systems"][name]
    P = prepare(name, r)
    Xi = np.array(side["xi"])
    out = {"system": name, "D": side["D"]}
    for n in (500, 2000):
        out[str(n)] = cond_stats(P["ref"][:n], P["E"], Xi)
    return out


def _run_job(payload):
    idx, x, dt, name, noise, truth, arms = payload
    import precision_numerics as pn
    from eqdisc import discover
    from eqdisc.core import metrics
    rec = {"index": idx, "system": name, "noise": noise, "n": len(x), "D": x.shape[1], "arms": {}}
    for arm in arms:
        t = time.perf_counter()
        pn.LOG.clear()
        try:
            if arm == "v11":
                with pn.record():
                    res = discover(x, dt)
            elif arm == "hard":
                with pn.hardened():
                    res = discover(x, dt)
            elif arm == "qr":
                with pn.qr_basis():
                    res = discover(x, dt)
            else:
                raise ValueError(arm)
        except Exception as exc:
            rec["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        m = metrics(res.xi, truth)
        rec["arms"][arm] = m | {"seconds": time.perf_counter() - t, "xi": np.asarray(res.xi).tolist(),
                                "noise_model": res.info.get("noise_model"), "gap": res.gap}
        if arm == "v11":
            rec["arms"][arm]["numerics"] = pn.summarise(pn.LOG)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["cond", "runs"])
    ap.add_argument("--data", default="data/dysts_precdev.npz")
    ap.add_argument("--arms", nargs="+", default=["v11", "hard"])
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    recs = []
    if a.what == "cond":
        names = list(json.load(open("data/dysts_test.json"))["systems"])
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(_cond_job, n): n for n in names}
            for k, fu in enumerate(as_completed(futs)):
                try:
                    recs.append(fu.result())
                except Exception as exc:
                    recs.append({"system": futs[fu], "error": str(exc)})
                print(f"{k + 1}/{len(futs)}", flush=True)
        out.write_text(json.dumps(recs, indent=1))
        print("DONE", flush=True)
        return
    d = np.load(a.data, allow_pickle=True)
    side = json.load(open(Path(a.data).with_suffix(".json")))["systems"]
    jobs = [(k, d["X"][k], float(d["dt"][k]), str(d["sys"][k]), float(d["noise"][k]),
             np.array(side[str(d["sys"][k])]["xi"]), a.arms) for k in range(len(d["X"]))]
    jobs.sort(key=lambda j: -len(j[1]) * j[1].shape[1])
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_run_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            recs.append(fu.result())
            if k % 10 == 0 or k == len(futs) - 1:
                out.write_text(json.dumps({"arms": a.arms, "records": recs}))
                print(f"{k + 1}/{len(futs)}", flush=True)
    out.write_text(json.dumps({"arms": a.arms, "records": recs}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
