"""Partial-observation benchmark runner (eqdisc.partial_obs and baselines), resumable JSON-lines output.

    python benchmarks/po_bench.py --data data/po_dev.npz --arms po,delay --configs xz,xy,yz,x \
        --workers 6 --output results_po/dev_A.jsonl

Configurations: observed components (x = 0, y = 1, z = 2); "xz" observes x and z (y latent), "x" observes x only.
Arms:
  po          eqdisc.partial_obs.discover, default restarts (derivative + quadratic proxies + 2 surrogates)
  po_dr       derivative proxies + 2 surrogates (dev A used this as "po", before the quadratic proxies existed)
  po_deriv / po_quad / po_rand   one restart family only (po_rand: 4 surrogates)
  po_anneal   default restarts, start fits annealed over rho (1, .3, .1, .03)
  po_oracle   DIAGNOSTIC: oracle latent init (true latent trajectory, gauge-normalised); never a method
  <arm>@opt   options appended with "@": ls<f> (latent sigma factor), pred / sum / val (score rule), gproj / gpoly / grms / gauto (gauge), pr<f> (pruning threshold, pr0 = off), ora (adds the oracle latent init as an extra restart; diagnostic)
  delay       baseline: weak chi2s best-subset SINDy on derivative coordinates [observed, d/dt observed (, d2/dt2)]
  embed       baseline: frozen full-observation model (final_model.discover) on the same derivative coordinates
  dahsi       baseline: DAHSI-style variational annealing + hard thresholding (po_baselines.dahsi)
Each record: run index, system, noise, config, arm, exact (gauge-aware), coefficient error, valid time, seconds.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

CONFIGS = {"xz": [0, 2], "xy": [0, 1], "yz": [1, 2], "x": [0]}


def run_one(data, i, cfg, arm):
    from eqdisc.core import SYSTEMS
    from po_eval import gauge_match, valid_time
    import po_baselines as pb
    from eqdisc import partial_obs as po
    d = np.load(data, allow_pickle=True)
    X, clean = np.asarray(d["X"][i], float), np.asarray(d["clean"][i], float)
    sysn, eta, dt = str(d["sys"][i]), float(d["noise"][i]), float(d["dt"][i])
    obs = CONFIGS[cfg]
    lat = [j for j in range(3) if j not in obs]
    Xi = SYSTEMS[sysn]
    t0 = time.time()
    rec = {"i": i, "sys": sysn, "noise": eta, "cfg": cfg, "arm": arm}
    try:
        y = X[:, obs]
        if arm.startswith("po"):
            base, *opts = arm.split("@")
            kw = {"po": {}, "po_dr": {"kinds": ("deriv", "random")}, "po_deriv": {"kinds": ("deriv",)},
                  "po_quad": {"kinds": ("quad",)}, "po_rand": {"kinds": ("random",), "n_random": 4},
                  "po_anneal": {"rho_path": (1.0, 0.3, 0.1, 0.03)},
                  "po_oracle": {"kinds": (), "latent_init": clean[:, lat]}}[base]
            for o in opts:  # sweep options: ls<float> latent sigma factor, pred/sum score rule, ora oracle init
                if o.startswith("ls"):
                    kw["lat_sig"] = float(o[2:])
                elif o in ("pred", "sum", "val"):
                    kw["score_rule"] = o
                elif o.startswith("g") and o[1:] in ("proj", "poly", "rms", "auto"):
                    kw["gauge"] = o[1:]
                elif o.startswith("pr"):
                    kw["prune"] = float(o[2:])
                elif o == "ora":
                    kw["latent_init"] = clean[:, lat]  # oracle restart added to the arm's own restarts
                else:
                    raise ValueError(o)
            xi, info = po.discover(y, dt, obs, seed=i, **kw)
            u = np.array(info["u"])
            rec.update(init=info["init"], gap=info["gap"], ambiguous=info["ambiguous"], fits=info["models_fitted"],
                       substeps=info["substeps"], trace=info["trace"], pruned=info.get("pruned"), regauged=info.get("regauged"))
        elif arm == "delay":
            xi, u, extra = pb.delay_weak(y, dt, obs)
            rec.update(extra)
        elif arm == "embed":
            xi, u, extra = pb.embed_final(y, dt, obs)
            rec.update(extra)
        elif arm.startswith("dahsi"):
            pkw = {}
            for o in arm.split("@")[1:]:
                if o.startswith("ls"):
                    pkw["lat_sig"] = float(o[2:])
                elif o in ("pred", "sum", "val"):
                    pkw["score_rule"] = o
                elif o.startswith("g") and o[1:] in ("proj", "poly", "rms", "auto"):
                    pkw["gauge"] = o[1:]
                else:
                    raise ValueError(o)
            xi, u, extra = pb.dahsi(y, dt, obs, seed=i, **pkw)
            rec.update(extra)
        else:
            raise ValueError(arm)
        rec["sec"] = time.time() - t0
        g = gauge_match(xi, Xi, obs, u, clean)
        rec.update(exact=g["exact_gauge"], support_ok=g["support_ok"], cond=g["cond"], latent_r2=g["latent_r2"],
                   coef_err=g["coef_err"], offnorm=g["offnorm"], k=int((xi != 0).sum()),
                   k_true=int((Xi != 0).sum()), vt=valid_time(xi, u, clean, obs, dt, sysn), xi=np.asarray(xi).tolist(),
                   u=np.round(np.asarray(u, float), 5).tolist())
    except Exception as e:
        rec.update(error=repr(e), tb=traceback.format_exc()[-1500:], sec=time.time() - t0, exact=False)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--configs", default="xz,xy,yz,x")
    ap.add_argument("--systems", default="lorenz,rossler,chen")
    ap.add_argument("--noises", default="0.01,0.05")
    ap.add_argument("--limit", type=int, default=0, help="max instances per (system, noise); 0 = all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    systems, noises = a.systems.split(","), [float(v) for v in a.noises.split(",")]
    idx, seen = [], {}
    for i in range(len(d["X"])):
        key = (str(d["sys"][i]), float(d["noise"][i]))
        if key[0] in systems and key[1] in noises:
            seen[key] = seen.get(key, 0) + 1
            if not a.limit or seen[key] <= a.limit:
                idx.append(i)
    done = set()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["i"], r["cfg"], r["arm"]))
    jobs = [(i, c, arm) for arm in a.arms.split(",") for c in a.configs.split(",") for i in idx
            if (i, c, arm) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fh:
        futs = {ex.submit(run_one, a.data, i, c, arm): (i, c, arm) for i, c, arm in jobs}
        for n_done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(n_done, r["arm"], r["cfg"], r["sys"], r["noise"], r.get("exact"), round(r["sec"], 1),
                  r.get("error", ""), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
