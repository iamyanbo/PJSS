"""Identifiability map from the profiled quadratic (sloppy-model / information-geometry practice).

For a run and its TRUE support S (oracle; this is a diagnostic, not a method), fit the joint objective on
S (config C settings), form the profiled quadratic in all library coefficients, ||A c - b||^2 (states
profiled out; H = A^T A is its Gauss-Newton curvature), and compute the noise-free selection margin

    m* = min over competitors T of  R(T) + pen (|T| - |S|),   R(T) = ||(I - P_T) A c_S||^2,

with pen = log(3n) (the BIC penalty) and competitors = every single-term drop and every same-equation
single-term swap of S. R(T) is the misfit the competitor would pay in the noise-free limit (a
non-centrality parameter): m* <= 0 means a smaller/equal model fits the true signal well enough that BIC
prefers it even without noise. R scales ~ linearly in N, so m*(N) ~ m*(N0) N/N0 + pen-terms gives a
"data needed" extrapolation. Also reported: the sloppiest eigen-direction of H_SS and the term pair of the
closest competitor.
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


def setup(x, dt):
    from eqdisc import joint_discover as jd
    from eqdisc.core import estimate_sigma
    n = len(x)
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    obs_train = np.ones(n, bool)
    obs_train[2::3] = False
    s_ = jd.auto_substeps(yo, dt)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    y = np.zeros((nf, 3)); y[oi] = yo
    train = np.zeros(nf, bool); train[oi[obs_train]] = True
    valmask = np.zeros(nf, bool); valmask[oi[~obs_train]] = True
    u0o = jd._initial_states(yo, obs_train)
    tf = np.arange(nf) / s_
    u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(3)])
    return dict(scale=scale, sig=sig, s_=s_, y=y, train=train, valmask=valmask, u0=u0, dt_f=dt / s_, n=n)


def margin(x, dt, truth, rho=0.03, huber=2.0, selfcal=True):
    from eqdisc import joint_discover as jd
    from eqdisc.core import xi_to_scaled, NAMES, NLIB, DIM
    S = setup(x, dt)
    C0 = xi_to_scaled(truth, S["scale"])
    sup = C0 != 0
    sig = S["sig"]
    fit = jd.fit_joint(S["y"], S["dt_f"], C0, sup, sig, rho, S["train"], S["u0"], valmask=S["valmask"], c=huber)
    if selfcal:  # sigma from the fit's residuals, as config C does
        r = ((fit.u - S["y"]) / 1.0)[S["train"]]
        sr = 1.4826 * np.median(np.abs(r - np.median(r, 0)), 0)
        sig = np.maximum(sig, sr)
        fit = jd.fit_joint(S["y"], S["dt_f"], fit.C, sup, sig, rho, S["train"], fit.u, valmask=S["valmask"], c=huber)
    A, _ = jd.profiled_quadratic(fit, S["y"], S["dt_f"], sig, rho, S["train"], c=huber)
    cS = fit.C.ravel()
    v = A @ cS
    Sidx = np.flatnonzero(sup.ravel())
    pen = np.log(3 * S["n"])

    def R(T):
        if len(T) == 0:
            return float(v @ v)
        AT = A[:, T]
        coef, *_ = np.linalg.lstsq(AT, v, rcond=None)
        r = v - AT @ coef
        return float(r @ r)

    best, best_desc, drops = np.inf, None, {}
    for i in Sidx:
        T = [t for t in Sidx if t != i]
        val = R(T) - pen
        drops[NAMES[i // DIM] + f"[{i % DIM}]"] = val + pen
        if val < best:
            best, best_desc = val, ("drop", int(i), None)
        e = i % DIM
        for j in range(e, NLIB * DIM, DIM):
            if j in Sidx:
                continue
            val = R(T + [j])
            if val < best:
                best, best_desc = val, ("swap", int(i), int(j))
    H = A.T @ A
    HSS = H[np.ix_(Sidx, Sidx)]
    w, V = np.linalg.eigh(HSS)
    load = V[:, 0] ** 2
    top = np.argsort(load)[::-1][:3]
    name = lambda k: NAMES[k // DIM] + f"[{'xyz'[k % DIM]}]"
    desc = {"kind": best_desc[0], "term": name(best_desc[1]),
            "with": name(best_desc[2]) if best_desc[2] is not None else None}
    zt = cS[Sidx] / np.sqrt(np.diag(np.linalg.inv(HSS)))
    return {"margin": best, "pen": pen, "sig": [float(v) for v in sig], "closest": desc, "min_drop": min(drops.values()),
            "min_z2": float((zt ** 2).min()), "eig_min": float(w[0]), "eig_max": float(w[-1]),
            "sloppy_terms": [name(Sidx[k]) for k in top], "sloppy_load": [float(load[k]) for k in top]}


def _job(payload):
    index, x, dt, name, noise = payload
    import warnings
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    from eqdisc.core import SYSTEMS
    from eqdisc.heldout_systems import HELDOUT
    truth = (SYSTEMS | HELDOUT)[name]
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        try:
            m = margin(x, dt, truth)
        except Exception as exc:
            m = {"error": f"{type(exc).__name__}: {exc}"}
    return {"index": index, "system": name, "noise": noise, "n": len(x), "seconds": time.perf_counter() - t0} | m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/instances.npz")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with np.load(args.data, allow_pickle=True) as d:
        jobs = [(i, np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]))
                for i in range(len(d["X"]))]
    print(f"{len(jobs)} runs", flush=True)
    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            if len(recs) % 25 == 0:
                print(f"[{len(recs)}/{len(jobs)}] {r['system']} {r['noise']} {r['n']} margin {r.get('margin')}",
                      flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"records": sorted(recs, key=lambda r: r["index"])}))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()


def gen_rf(output, cells, seeds=10, seed=20261301):
    """Fresh Rabinovich-Fabrikant runs (bench.py protocol: random attractor start, noise eta * std) for the
    data-needed validation. cells: list of (eta, N)."""
    import bench
    from eqdisc.core import SYSTEMS, simulate
    rng = np.random.default_rng(seed)
    Xi, dt = SYSTEMS["rabinovich"], bench.DT["rabinovich"]
    ref = simulate(Xi, bench.X0["rabinovich"], dt, 20000, burn=2000)
    out = {k: [] for k in ("X", "sys", "noise", "n", "dt")}
    for eta, N in cells:
        for _ in range(seeds):
            Xc = simulate(Xi, ref[rng.integers(len(ref))], dt, N)
            out["X"].append(Xc + eta * Xc.std(0) * rng.standard_normal(Xc.shape))
            out["sys"].append("rabinovich"); out["noise"].append(eta); out["n"].append(N); out["dt"].append(dt)
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    return len(X)
