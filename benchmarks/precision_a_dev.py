"""Precision audit, part A (dev only): what happens on noise-free trajectories?

Data (dev, our own generation, never test):
  3-D: data/dev_clean.npz = gen_variants.py --kind white --etas 0 --seed 5150001 --seeds 4
       (10 systems x N {2000, 500} x 4 seeds = 80 runs, eta = 0)
  2-D: generated here (seed 5150002): four degree-2 in-library systems x N {500, 2000} x 4 seeds (joint_nd path)

Arms (per run; seeds/starts computed by the public API each time):
  v11      eqdisc.discover(y, dt)                       (current default)
  white    noise="white"                                 (3-D only: routing effect)
  sub2x    substeps = 2 x the automatic choice          (halves the latent step: discretisation effect)
  v12      version="1.2" (v1.1 + the discretisation-error noise floor, joint_discover.disc_floor_sigma)
  v12_white  version="1.2", noise="white"

Per run and arm we log: estimated sigma (scaled units), the high-frequency estimate, whether the 1e-6 floor
was active, phi, lam, substeps, the noise model, exact / extra / missing terms and each extra term's relative
contribution (rms of xi*theta on the data / rms of that equation's fitted rhs). Per run (truth used only for
this diagnostic) we also log the scale of the local truncation error of the implicit-midpoint scheme on the
latent grid, tau = u(t+h) - u(t) - h f((u(t)+u(t+h))/2) along a fine reference solution, in scaled units,
next to rho * sigma (the model-residual scale the likelihood assumes).
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

RHO = 0.03


def systems2d():
    from eqdisc.joint_nd import PolyLib
    lib = PolyLib(2, 2)
    return lib, {
        "osc": (lib.coef([{"x1": 1.0}, {"x0": -2.0}]), 0.02),
        "damped": (lib.coef([{"x0": -0.1, "x1": 2.0}, {"x0": -2.0, "x1": -0.1}]), 0.02),
        "lv": (lib.coef([{"x0": 1.0, "x0*x1": -0.5}, {"x1": -0.75, "x0*x1": 0.25}]), 0.02),
        "quad": (lib.coef([{"x1": 1.0}, {"x0": -1.0, "x1": -0.2, "x0^2": 0.3}]), 0.02),
    }


def gen2d(seed=5150002, seeds=4):
    from eqdisc.nd_systems import simulate
    lib, S = systems2d()
    rng = np.random.default_rng(seed)
    out = []
    for name, (Xi, dt) in S.items():
        for n in (2000, 500):
            for _ in range(seeds):
                if name == "lv":
                    x0 = rng.uniform(1.0, 4.0, 2)
                else:
                    x0 = rng.uniform(-2, 2, 2)
                x = simulate(lib, Xi, x0, dt, n)
                out.append((name, n, dt, x))
    return out


def truncation(x, dt, s, f, scale):
    """rms over the latent grid of the implicit-midpoint local error along a fine reference solution."""
    from scipy.integrate import solve_ivp
    h = dt / s
    nf = (len(x) - 1) * s + 1
    t = np.arange(nf) * h
    sol = solve_ivp(lambda _, z: f(z[None])[0], (0, t[-1]), x[0], t_eval=t, method="DOP853", rtol=1e-12,
                    atol=1e-12)
    if not sol.success or sol.y.shape[1] != nf:
        return None
    u = sol.y.T
    m = 0.5 * (u[1:] + u[:-1])
    tau = (u[1:] - u[:-1] - h * f(m)) / scale
    return np.sqrt(np.mean(tau ** 2, axis=0)).tolist()


def contributions(xi, truth, x, lib_fn, names):
    th = lib_fn(x)
    rhs = th @ xi
    out = []
    for i, j in zip(*np.nonzero((xi != 0) & (truth == 0))):
        out.append({"term": names[i], "eq": int(j), "coef": float(xi[i, j]),
                    "rel": float(np.sqrt(np.mean((xi[i, j] * th[:, i]) ** 2)) / max(np.sqrt(np.mean(rhs[:, j] ** 2)), 1e-300))})
    miss = [{"term": names[i], "eq": int(j)} for i, j in zip(*np.nonzero((xi == 0) & (truth != 0)))]
    return out, miss


ARM_KW = {
    "v11": {},
    "white": {"noise": "white"},
    "v12": {"version": "1.2"},
    "v12_white": {"version": "1.2", "noise": "white"},
}


def _job(payload):
    idx, kind, name, n, dt, x, arms = payload
    from eqdisc import discover
    from eqdisc.core import metrics
    if kind == "3d":
        from eqdisc.core import SYSTEMS, NAMES, library
        truth, names, lib_fn = SYSTEMS[name], NAMES, library
    else:
        lib, S = systems2d()
        truth, names, lib_fn = S[name][0], lib.names, lib
    f = lambda z: lib_fn(z) @ truth
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    rec = {"index": idx, "kind": kind, "system": name, "n": n, "dt": dt, "arms": {}}
    from eqdisc.core import estimate_sigma
    rec["sigma_hf_scaled"] = estimate_sigma(x / scale).tolist()
    s_auto = None
    for arm in arms:
        kw = dict(ARM_KW.get(arm, {}))
        if arm in ("white", "v12_white") and kind != "3d":
            continue
        if arm == "sub2x":
            kw["substeps"] = 2 * s_auto if s_auto else None
            if kw["substeps"] is None:
                continue
        t = time.perf_counter()
        try:
            res = discover(x, dt, **kw)
        except Exception as exc:
            rec["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        m = metrics(res.xi, truth)
        extra, miss = contributions(res.xi, truth, x, lib_fn, names)
        i = res.info
        sig = np.asarray(i.get("sigma"), float) / scale
        rec["arms"][arm] = m | {"seconds": time.perf_counter() - t, "sigma_scaled": sig.tolist(),
                                "floor_active": bool(np.any(sig <= 1.0000001e-6)),
                                "phi": i.get("phi"), "lam": i.get("lam"), "substeps": i.get("substeps"),
                                "noise_model": i.get("noise_model"), "ar_estimate": i.get("ar_estimate"),
                                "extra": extra, "miss": miss, "gap": res.gap,
                                "sigma_floor_info": i.get("sigma_floor_info"),
                                "xi": np.asarray(res.xi).tolist()}
        if arm == "v11":
            s_auto = i.get("substeps")
            tr = truncation(x, dt, s_auto, f, scale)
            rec["tau_rms_scaled"] = tr
            rec["rho_sigma_scaled"] = (RHO * sig).tolist()
            tr2 = truncation(x, dt, 2 * s_auto, f, scale)
            rec["tau_rms_scaled_2x"] = tr2
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data3d", default="data/dev_clean.npz")
    ap.add_argument("--arms", nargs="+", default=["v11", "white", "sub2x"])
    ap.add_argument("--no2d", action="store_true")
    ap.add_argument("--no3d", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    jobs = []
    if not a.no3d:
        d = np.load(a.data3d, allow_pickle=True)
        for k in range(0, len(d["X"]), a.every):
            jobs.append((f"3d-{k}", "3d", str(d["sys"][k]), int(d["n"][k]), float(d["dt"][k]), d["X"][k], a.arms))
    if not a.no2d:
        for k, (name, n, dt, x) in enumerate(gen2d()):
            if k % a.every == 0:
                jobs.append((f"2d-{k}", "2d", name, n, dt, x, a.arms))
    # longest first
    jobs.sort(key=lambda j: -(j[3] * (3 if j[1] == "3d" else 1)))
    out = Path(a.output)
    recs = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            recs.append(fu.result())
            if k % 10 == 0 or k == len(futs) - 1:
                out.write_text(json.dumps({"arms": a.arms, "records": recs}))
                print(f"{k + 1}/{len(futs)}", flush=True)
    out.write_text(json.dumps({"arms": a.arms, "records": recs}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
