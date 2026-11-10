"""Real data 2: the MultiArm pendulum (Kaheman et al. 2022) with the FROZEN final model, protocol in
results_real2/PREREGISTRATION.md (committed before any run of a model on these data).

    python -I benchmarks/real2_pendulum.py single --arm A0 --method eqdisc --output results_real2/single_A0_eqdisc.json
    python -I benchmarks/real2_pendulum.py double --run 1 --method fd_stlsq --output results_real2/double_r1_fd_stlsq.json

single  single pendulum free swing, states (x, y, z) = (sin th, cos th, dth/dt); accepted model
        dx = y z, dy = -x z, dz = A x - B z, A = a1 g m1 / J, B = k1 / J, J = m1 a1^2 + I1 (published estimates)
double  double pendulum free swing, raw states (th1, th2, dth1, dth2); accepted model (rational in cos(th1-th2))
        is outside every frozen library -> library "auto" (degree-2 joint_nd); forecast evaluation only
methods eqdisc (eqdisc.discover, library per the pre-registered rule, noise="auto"),
        fd_stlsq_cv, fd_stlsq_aic, weak_stlsq_cv, weak_stlsq_aic (pysindy 2.1.0 via baselines_bench.run_path with
        the configs fixed in docs/reports/baselines.md).
--root  directory with the downloaded .mat files (untrusted data; read with scipy.io.loadmat only).
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
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # baselines_bench, eqdisc (python -I drops script dir and cwd)

DT_RAW = 0.001
# arm: (decimation, t_max [s] or None, velocity source, library)
SINGLE_ARMS = {"A0": (25, None, "authors", "poly3"), "A1": (10, 20.0, "authors", "poly3"),
               "A2": (50, None, "authors", "poly3"), "A3": (25, None, "fd", "poly3"),
               "A4": (25, None, "authors", "trig3"), "SYN": (25, None, "synthetic", "poly3")}
# baseline name -> (run_path method, cfg) as fixed in docs/reports/baselines.md (config per rule)
BASELINES = {"fd_stlsq_cv": ("fd_stlsq", dict(scale="raw", window=11, h=20), "cv"),
             "fd_stlsq_aic": ("fd_stlsq", dict(scale="raw", window=5, h=20), "aic"),
             "weak_stlsq_cv": ("weak_stlsq", dict(scale="rms", window=21, h=10), "cv"),
             "weak_stlsq_aic": ("weak_stlsq", dict(scale="rms", window=21, h=10), "aic")}


# ------------------------------------------------------------------ data
def loadmat(p):
    import scipy.io as sio
    return sio.loadmat(str(p), squeeze_me=True)


def single_data(root):
    e = loadmat(Path(root) / "SinglePendulumDataForParameterEstimation.mat")
    pieces = [np.asarray(p, float).T for p in list(e["Y_id"]) + list(e["Y_vad"])]  # each (n, 2): th, dth
    lens = [len(p) for p in pieces]
    Y = np.concatenate(pieces)
    a1, m1, I1, k1, g = loadmat(Path(root) / "EstimatedParameters_SinglePendulum.mat")["EstimatedParameters"]
    J = m1 * a1 ** 2 + I1
    return Y[:, 0], Y[:, 1], lens, dict(A=float(a1 * g * m1 / J), B=float(k1 / J))


def double_params(root):
    return [float(v) for v in loadmat(Path(root) / "EstimatedParameters_DoublePendulum.mat")["EstimatedParameters"]]


def double_data(root, run):
    d = loadmat(Path(root) / f"DoubleDataFreeSwing_{run}_Dt_0_001.mat")
    return np.column_stack([np.asarray(d[k], float) for k in ("Theta1", "Theta2", "dTheta1", "dTheta2")])


def single_rhs(A, B):
    return lambda t, z: [z[1], A * np.sin(z[0]) - B * z[1]]


def double_rhs(P):
    m1, m2, a1, a2, L1, I1, I2, k1, k2, g = P  # order of DoublePendulumODE_Mounted.m

    def f(t, y):
        p1, p2, d1, d2 = y
        s12, c12 = np.sin(p1 - p2), np.cos(p1 - p2)
        den = (I1 * I2 + L1 ** 2 * a2 ** 2 * m2 ** 2 + I2 * L1 ** 2 * m2 + I2 * a1 ** 2 * m1 + I1 * a2 ** 2 * m2
               - L1 ** 2 * a2 ** 2 * m2 ** 2 * c12 ** 2 + a1 ** 2 * a2 ** 2 * m1 * m2)
        n3 = -(I2 * d1 * k1 + I2 * d1 * k2 - I2 * d2 * k2 + a2 ** 2 * d1 * k1 * m2 + a2 ** 2 * d1 * k2 * m2
               - a2 ** 2 * d2 * k2 * m2 + L1 * a2 ** 3 * d2 ** 2 * m2 ** 2 * s12 - (L1 * a2 ** 2 * g * m2 ** 2 * np.sin(p1)) / 2
               - I2 * L1 * g * m2 * np.sin(p1) - (L1 * a2 ** 2 * g * m2 ** 2 * np.sin(p1 - 2 * p2)) / 2
               - I2 * a1 * g * m1 * np.sin(p1) + (L1 ** 2 * a2 ** 2 * d1 ** 2 * m2 ** 2 * np.sin(2 * p1 - 2 * p2)) / 2
               + L1 * a2 * d1 * k2 * m2 * c12 - L1 * a2 * d2 * k2 * m2 * c12 + I2 * L1 * a2 * d2 ** 2 * m2 * s12
               - a1 * a2 ** 2 * g * m1 * m2 * np.sin(p1))
        n4 = (I1 * d1 * k2 - I1 * d2 * k2 + L1 ** 2 * d1 * k2 * m2 - L1 ** 2 * d2 * k2 * m2 + a1 ** 2 * d1 * k2 * m1
              - a1 ** 2 * d2 * k2 * m1 + L1 ** 3 * a2 * d1 ** 2 * m2 ** 2 * s12 + L1 ** 2 * a2 * g * m2 ** 2 * np.sin(p2)
              + I1 * a2 * g * m2 * np.sin(p2) + (L1 ** 2 * a2 ** 2 * d2 ** 2 * m2 ** 2 * np.sin(2 * p1 - 2 * p2)) / 2
              + L1 * a2 * d1 * k1 * m2 * c12 + L1 * a2 * d1 * k2 * m2 * c12 - L1 * a2 * d2 * k2 * m2 * c12
              - L1 ** 2 * a2 * g * m2 ** 2 * c12 * np.sin(p1) + I1 * L1 * a2 * d1 ** 2 * m2 * s12
              + a1 ** 2 * a2 * g * m1 * m2 * np.sin(p2) + L1 * a1 ** 2 * a2 * d1 ** 2 * m1 * m2 * s12
              - L1 * a1 * a2 * g * m1 * m2 * c12 * np.sin(p1))
        return [d1, d2, n3 / den, n4 / den]
    return f


# ------------------------------------------------------------------ term evaluation
def feat_fn(names, D):
    var = list("xyz") if D == 3 else [f"x{i}" for i in range(D)]

    def f(Z):
        Z = np.atleast_2d(Z)
        cols = []
        for nm in names:
            if nm == "1":
                cols.append(np.ones(len(Z)))
            elif nm.startswith(("sin(", "cos(")):
                u = Z[:, var.index(nm[4:-1])]
                cols.append(np.sin(u) if nm.startswith("sin") else np.cos(u))
            else:
                c = np.ones(len(Z))
                for tok in nm.split("*"):
                    v, _, p = tok.partition("^")
                    c = c * Z[:, var.index(v)] ** (int(p) if p else 1)
                cols.append(c)
        return np.column_stack(cols)
    return f


def simulate(rhs, z0, t, max_abs=1e6):
    def ev(tt, z):
        return max_abs - np.max(np.abs(z))
    ev.terminal = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = solve_ivp(rhs, (t[0], t[-1]), z0, t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10, events=ev)
        except Exception:
            return None
    return s.y.T  # may be shorter than t (blow-up / failure)


def r2(true, pred):
    if pred is None or len(pred) < len(true) or not np.isfinite(pred).all():
        return None
    return float(1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum())


# ------------------------------------------------------------------ single pendulum
def single_inputs(arm, root):
    dec, tmax, vel, lib = SINGLE_ARMS[arm]
    th, om, lens, ref = single_data(root)
    if vel == "fd":
        om = np.gradient(th, DT_RAW)
    elif vel == "synthetic":  # code check only: the accepted model's own trajectory (+ no noise)
        t = np.arange(len(th)) * DT_RAW
        s = solve_ivp(single_rhs(ref["A"], ref["B"]), (0, t[-1]), [th[0], om[0]], t_eval=t, rtol=1e-10, atol=1e-12)
        th, om = s.y
    n = len(th) if tmax is None else int(round(tmax / DT_RAW))
    idx = np.arange(0, n, dec)
    Z = np.column_stack([np.sin(th), np.cos(th), om])
    return Z[idx], dec * DT_RAW, lib, th, om, lens, ref


def single_truth(names, ref):
    T = np.zeros((len(names), 3))
    T[names.index("y*z"), 0] = 1.0
    T[names.index("x*z"), 1] = -1.0
    T[names.index("x"), 2] = ref["A"]
    T[names.index("z"), 2] = -ref["B"]
    return T


def single_eval(xi, names, th, om, lens, ref):
    """Integrate the 3-D model from the start of each of the 6 recorded pieces (as the authors' validation did);
    theta = atan2(x, y) unwrapped; R^2 vs the measured angle.  Also: RHS agreement on the data manifold."""
    D = 3
    F = feat_fn(names, D)
    rhs = lambda t, z: (F(z) @ xi)[0]
    out = {"r2_theta_pieces": [], "r2_theta_pieces_accepted": []}
    start = 0
    for L in lens:
        sl = slice(start, start + L)
        t = np.arange(L) * DT_RAW
        z0 = [np.sin(th[start]), np.cos(th[start]), om[start]]
        p = simulate(rhs, z0, t)
        if p is not None and len(p) == L and np.isfinite(p).all():
            ang = np.unwrap(np.arctan2(p[:, 0], p[:, 1]))
            ang += np.round((th[start] - ang[0]) / (2 * np.pi)) * 2 * np.pi
            out["r2_theta_pieces"].append(r2(th[sl], ang))
        else:
            out["r2_theta_pieces"].append(None)
        q = simulate(single_rhs(ref["A"], ref["B"]), [th[start], om[start]], t)
        out["r2_theta_pieces_accepted"].append(r2(th[sl], q[:, 0] if q is not None and len(q) == L else None))
        start += L
    # functional equivalence on the measured manifold (handles x^2 + y^2 = 1 rewrites)
    Z = np.column_stack([np.sin(th), np.cos(th), om])[::10]
    disc = F(Z) @ xi
    acc = np.column_stack([Z[:, 1] * Z[:, 2], -Z[:, 0] * Z[:, 2], ref["A"] * Z[:, 0] - ref["B"] * Z[:, 2]])
    out["rhs_rel_rms_vs_accepted"] = (np.sqrt(((disc - acc) ** 2).mean(0)) / np.sqrt((acc ** 2).mean(0))).tolist()
    # damping part only: the z-equation's response to z at fixed angle, d(dz)/dz averaged over the data
    h = 1e-3
    Zp, Zm = Z.copy(), Z.copy()
    Zp[:, 2] += h
    Zm[:, 2] -= h
    out["dz_dz_mean"] = float(((F(Zp) @ xi)[:, 2] - (F(Zm) @ xi)[:, 2]).mean() / (2 * h))
    out["accepted"] = ref
    rr = [v for v in out["r2_theta_pieces"]]
    out["r2_theta_median"] = None if any(v is None for v in rr) else float(np.median(rr))
    out["r2_theta_median_accepted"] = float(np.median([v for v in out["r2_theta_pieces_accepted"] if v is not None]))
    return out


def structure_single(xi, names, T):
    S, Tt = xi != 0, T != 0
    nm = lambda i, j: f"{names[i]} (d{'xyz'[j]})"
    return {"exact": bool((S == Tt).all()), "physics_eq_exact": bool((S[:, 2] == Tt[:, 2]).all()),
            "eq_exact": [bool((S[:, j] == Tt[:, j]).all()) for j in range(3)],
            "extra": [nm(i, j) for i, j in zip(*np.nonzero(S & ~Tt))],
            "missing": [nm(i, j) for i, j in zip(*np.nonzero(~S & Tt))],
            "A": float(xi[names.index("x"), 2]), "B": float(-xi[names.index("z"), 2]),
            "c_yz": float(xi[names.index("y*z"), 0]), "c_xz": float(xi[names.index("x*z"), 1])}


# ------------------------------------------------------------------ double pendulum
TRAIN_T, TRAIN_DEC = 20.0, 10
HORIZON, VT_TOL, STARTS = 5.0, 0.5, np.arange(0, 40, 4.0)


def valid_time(rhs, Y, i0):
    n = int(round(HORIZON / DT_RAW)) + 1
    true = Y[i0:i0 + n]
    t = np.arange(len(true)) * DT_RAW
    p = simulate(rhs, Y[i0], t, max_abs=1e3)
    if p is None or len(p) == 0:
        return 0.0
    m = min(len(p), len(true))
    bad = ~np.isfinite(p[:m]).all(1) | (np.abs(p[:m, :2] - true[:m, :2]).max(1) > VT_TOL)
    if bad.any():
        return float(t[np.argmax(bad)])
    return float(t[m - 1]) if m == len(true) else float(t[m - 1])


def double_eval(rhs_disc, root, train_run, P):
    out = {}
    acc = double_rhs(P)
    rest = lambda t, z: [0.0, 0.0, 0.0, 0.0]  # persistence: angles frozen at the initial value
    for r in (1, 2, 3, 4):
        Y = double_data(root, r)
        key = "same" if r == train_run else f"run{r}"
        starts = STARTS + (TRAIN_T if r == train_run else 0.0)  # same run: start after the training window
        rec = {"starts": starts.tolist(), "disc": [], "accepted": [], "persistence": []}
        for t0 in starts:
            i0 = int(round(t0 / DT_RAW))
            rec["disc"].append(None if rhs_disc is None else valid_time(rhs_disc, Y, i0))
            rec["accepted"].append(valid_time(acc, Y, i0))
            rec["persistence"].append(valid_time(rest, Y, i0))
        out[key] = rec
    allv = lambda k: [v for rec in out.values() for v in rec[k] if v is not None]
    out["median_valid_time"] = {k: (float(np.median(allv(k))) if allv(k) else None)
                                for k in ("disc", "accepted", "persistence")}
    return out


# ------------------------------------------------------------------ methods
def run_eqdisc(y, dt, library):
    from eqdisc import discover
    t0 = time.perf_counter()
    r = discover(y, dt, library=library, noise="auto")
    xi = np.asarray(r.xi, float)
    return xi, list(r.names), {"seconds": time.perf_counter() - t0, "equations": r.equations(),
                               "ambiguous": bool(r.ambiguous), "gap": float(r.gap),
                               "noise_model": r.info.get("noise_model"),
                               "ar_estimate": np.asarray(r.info.get("ar_estimate", np.nan)).tolist(),
                               "confidence": {f"{k[0]}|{k[1]}": float(v) for k, v in r.confidence.items()}}


def run_baseline(y, dt, method, names, truth):
    import baselines_bench as bb
    m, cfg, pick = BASELINES[method]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = bb.run_path(dict(x=y, dt=dt, truth=truth.tolist(), names=names, index=0), m, cfg)
    xi = np.asarray(rec[f"{pick}_xi"], float)
    return xi, list(names), {"seconds": time.perf_counter() - t0, "config": cfg, "pick": pick,
                             "oracle_exact_on_path": rec.get("oracle"), "oracle_eq_on_path": rec.get("oracle_eq"),
                             "nterms_path": rec.get("nterms_path")}


def eq_strings(xi, names, D):
    var = list("xyz") if D == 3 else [f"x{i}" for i in range(D)]
    return "\n".join(f"d{var[j]}/dt = " + (" ".join(f"{xi[m, j]:+.5g}" + ("" if names[m] == "1" else f"*{names[m]}")
                                                    for m in np.flatnonzero(xi[:, j])) or "0") for j in range(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("single", "double"))
    ap.add_argument("--arm", default="A0")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--method", required=True, choices=["eqdisc"] + list(BASELINES))
    ap.add_argument("--root", default="data/real2/multiarm_pendulum")
    ap.add_argument("--output", required=True)
    ap.add_argument("--centre-angles", action="store_true",
                    help="EXPLORATORY (not pre-registered, double only): fit on (th1 - pi, th2 - pi, dth1, dth2)")
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out} exists")
    res = {"part": a.part, "method": a.method}
    if a.part == "single":
        y, dt, lib, th, om, lens, ref = single_inputs(a.arm, a.root)
        res.update(arm=a.arm, n=len(y), dt=dt, library=lib if a.method == "eqdisc" else "poly3 (pysindy)")
        from eqdisc.core import NAMES
        if a.method == "eqdisc":
            xi, names, info = run_eqdisc(y, dt, lib)
        else:
            if lib != "poly3":
                raise SystemExit("baselines are polynomial-only (pre-registered: not run on the trig3 arm)")
            xi, names, info = run_baseline(y, dt, a.method, list(NAMES), single_truth(list(NAMES), ref))
        T = single_truth(names, ref)
        res.update(info=info, names=names, xi=xi.tolist(), eqs=eq_strings(xi, names, 3),
                   structure=structure_single(xi, names, T), fit=single_eval(xi, names, th, om, lens, ref))
        print(res["eqs"])
        print(json.dumps({"structure": res["structure"], "r2_med": res["fit"]["r2_theta_median"],
                          "r2_med_acc": res["fit"]["r2_theta_median_accepted"],
                          "rhs_rel": res["fit"]["rhs_rel_rms_vs_accepted"], "info": {k: info[k] for k in info
                          if k not in ("confidence", "nterms_path")}}, default=float))
    else:
        from eqdisc.joint_nd import PolyLib
        P = double_params(a.root)
        Y = double_data(a.root, a.run)
        shift = np.array([np.pi, np.pi, 0.0, 0.0]) if a.centre_angles else np.zeros(4)
        y = Y[:int(round(TRAIN_T / DT_RAW)):TRAIN_DEC] - shift
        dt = TRAIN_DEC * DT_RAW
        names = PolyLib(4, 2).names
        res.update(run=a.run, n=len(y), dt=dt, params=P, centre_angles=a.centre_angles)
        if a.method == "eqdisc":
            xi, names, info = run_eqdisc(y, dt, "auto")
        else:
            xi, names, info = run_baseline(y, dt, a.method, names, np.zeros((len(names), 4)))
        F = feat_fn(names, 4)
        rhs = lambda t, z: (F(np.asarray(z) - shift) @ xi)[0]
        res.update(info=info, names=names, xi=xi.tolist(), eqs=eq_strings(xi, names, 4), nterms=int((xi != 0).sum()),
                   forecast=double_eval(rhs, a.root, a.run, P))
        print(res["eqs"])
        print(json.dumps({"nterms": res["nterms"], "median_valid_time": res["forecast"]["median_valid_time"],
                          "info": {k: info[k] for k in info if k not in ("confidence", "nterms_path")}}, default=float))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
