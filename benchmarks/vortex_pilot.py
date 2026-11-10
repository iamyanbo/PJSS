"""Vortex shedding pilot: does the model find the mean-field equations of the Re=100 cylinder wake?
Protocol in results_vortex/PREREGISTRATION.md. eqdisc/ is not modified.

    python benchmarks/vortex_pilot.py reference
    python benchmarks/vortex_pilot.py control --output results_vortex/control.json
    python benchmarks/vortex_pilot.py real --output results_vortex/real.json

States: x, y = POD modes 1 and 2 (shedding), z = column 9 of the file (shift mode). Time is in convective units.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

DATA = HERE.parent / "data" / "vortex" / "vonKarman_a.dat"
DT = 0.1
WINDOW = (30.0, 130.0)
ETAS = (0.01, 0.05, 0.10, 0.20)
SEEDS = range(1, 11)
CONTROL_ETA = 0.05
CONTROL_SEEDS = range(101, 121)
# exponents (x, y, z) of the target support per equation: the mean-field form with a frequency shift
TARGET = {0: [(1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1)], 1: [(1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1)],
          2: [(0, 0, 1), (2, 0, 0), (0, 2, 0)]}
PYSINDY = {"fd_stlsq": dict(method="fd_stlsq", scale="raw", window=11, h=10),
           "weak_stlsq": dict(method="weak_stlsq", scale="rms", window=11, h=10)}


def load():
    a = np.loadtxt(DATA)
    return a[:, 0], a[:, [1, 2, 9]]


def window(t):
    return (t >= WINDOW[0]) & (t < WINDOW[1])


def terms(X, exps):
    return np.column_stack([np.prod(X ** np.array(e), axis=1) for e in exps])


def reference():
    """Least squares of central differences of the clean states on the target support; 20-term coefficient matrix."""
    from eqdisc.core import EXP_ARR
    t, X = load()
    w, dX = window(t), np.gradient(X, DT, axis=0)
    ex = [tuple(int(v) for v in e) for e in EXP_ARR]
    Xi, r2 = np.zeros((len(ex), 3)), []
    for j, sup in TARGET.items():
        A = terms(X[w], sup)
        c = np.linalg.lstsq(A, dX[w, j], rcond=None)[0]
        Xi[[ex.index(e) for e in sup], j] = c
        r2.append(float(1 - np.var(dX[w, j] - A @ c) / np.var(dX[w, j])))
    return Xi, r2


def rhs(Xi):
    from eqdisc.core import EXP_ARR
    ex = np.asarray(EXP_ARR, float)
    return lambda t, v: np.prod(np.asarray(v)[None] ** ex, axis=1) @ Xi


def attractor(Xi, x0, t_end=270.0):
    """Integrate from x0 and summarise the last 100 time units; None if the model diverges."""
    from scipy.integrate import solve_ivp
    tt = np.arange(0, t_end, DT)
    blowup = lambda t, v: 100 - np.max(np.abs(v))
    blowup.terminal = True
    s = solve_ivp(rhs(Xi), (0, tt[-1]), x0, t_eval=tt, rtol=1e-8, atol=1e-10, method="LSODA", events=blowup)
    if not s.success or s.y.shape[1] < len(tt):
        return None
    Y = s.y.T[-1000:]
    return dict(z=float(Y[:, 2].mean()), r2=float((Y[:, 0] ** 2 + Y[:, 1] ** 2).mean()),
                omega=float(abs(np.mean(np.gradient(np.unwrap(np.arctan2(Y[:, 1], Y[:, 0])), DT)))))


def truth_summary():
    t, X = load()
    Y = X[t >= 200]
    return dict(z=float(Y[:, 2].mean()), r2=float((Y[:, 0] ** 2 + Y[:, 1] ** 2).mean()),
                omega=float(abs(np.mean(np.gradient(np.unwrap(np.arctan2(Y[:, 1], Y[:, 0])), DT)))))


def clean_window():
    t, X = load()
    return X[window(t)]


def simulate(Xi, n):
    from scipy.integrate import solve_ivp
    x0 = clean_window()[0]
    return solve_ivp(rhs(Xi), (0, (n - 1) * DT), x0, t_eval=np.arange(n) * DT, rtol=1e-10, atol=1e-12,
                     method="LSODA").y.T


def noisy(X, eta, seed):
    return X + eta * X.std(0) * np.random.default_rng(seed).standard_normal(X.shape)


def core_exact(xi, truth):
    a, b = np.asarray(xi) != 0, np.asarray(truth) != 0
    tp = int((a & b).sum())
    return dict(core=bool((a | ~b).all()), exact=bool((a == b).all()), terms=int(a.sum()),
                f1=2 * tp / max(int(a.sum() + b.sum()), 1))


def fit(method, y):
    """Returns {pick: xi}. Every method uses the 20-term degree-3 library and picks its sparsity without the truth."""
    from eqdisc.core import NAMES
    if method == "eqdisc":
        from eqdisc import discover
        r = discover(y, DT, noise="auto", confidence=False)
        return {"eqdisc": np.asarray(r.xi, float)}, dict(ambiguous=bool(r.ambiguous),
                                                        noise_model=r.info.get("noise_model"))
    if method == "chi2s":
        from eqdisc.select_k import discover
        return {"chi2s": np.asarray(discover(y, DT, criterion="chi2s"), float)}, {}
    if method == "wsindy":
        from eqdisc.core import EXP_ARR
        from wsindy_port import wsindy_ode
        return {"wsindy": np.asarray(wsindy_ode(y, np.arange(len(y)) * DT, EXP_ARR), float)}, {}
    from baselines_bench import run_path
    cfg = PYSINDY[method]
    truth = reference()[0]
    r = dict(index=0, x=y, dt=DT, truth=truth.tolist(), names=list(NAMES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = run_path(r, cfg["method"], cfg)
    return {f"{method}_{k}": np.asarray(p[f"{k}_xi"], float) for k in ("cv", "aic")}, {}


def job(a):
    source, eta, seed, method = a
    truth, _ = reference()
    X = clean_window() if source == "real" else simulate(truth, len(clean_window()))
    y = noisy(X, eta, seed) if eta else X
    tic = time.time()
    try:
        picks, info = fit(method, y)
    except Exception as e:  # counted as not recovered
        return [dict(source=source, eta=eta, seed=seed, method=method, error=repr(e), core=False, exact=False)]
    sec = time.time() - tic
    rows = []
    for name, xi in picks.items():
        rows.append(dict(source=source, eta=eta, seed=seed, method=name, seconds=sec, xi=xi.tolist(), info=info,
                         dyn=attractor(xi, clean_window()[0]) if source == "real" else None,
                         **core_exact(xi, truth)))
    return rows


METHODS = ["eqdisc", "chi2s", "wsindy", "fd_stlsq", "weak_stlsq"]


def run(jobs, out, procs):
    from multiprocessing import Pool
    with Pool(procs) as p:
        rows = [r for b in p.imap_unordered(job, jobs, chunksize=1) for r in b]
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(dict(truth=truth_summary(), rows=rows), indent=1))
    keys = sorted({(r["source"], r["eta"], r["method"]) for r in rows})
    for k in keys:
        s = [r for r in rows if (r["source"], r["eta"], r["method"]) == k]
        print(k, "n", len(s), "core", sum(r["core"] for r in s), "exact", sum(r["exact"] for r in s),
              "errors", sum("error" in r for r in s))


def cmd_reference(a):
    Xi, r2 = reference()
    from eqdisc.core import NAMES
    for j in range(3):
        print(j, {str(NAMES[i]): round(float(Xi[i, j]), 4) for i in np.flatnonzero(Xi[:, j])}, "R2", round(r2[j], 4))
    print("data  ", truth_summary())
    print("fitted", attractor(Xi, clean_window()[0]))


def cmd_control(a):
    run([("control", CONTROL_ETA, s, m) for s in CONTROL_SEEDS for m in METHODS], a.output, a.procs)


def cmd_real(a):
    jobs = [("real", 0.0, 0, m) for m in METHODS]
    jobs += [("real", e, s, m) for e in ETAS for s in SEEDS for m in METHODS]
    run(jobs, a.output, a.procs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reference", "control", "real"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--output")
    a = ap.parse_args()
    dict(reference=cmd_reference, control=cmd_control, real=cmd_real)[a.cmd](a)
