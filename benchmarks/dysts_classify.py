"""Classify every dysts continuous-time system against our polynomial libraries and extract true coefficients.

Run with the separate venv that has dysts (it is NOT needed by the benchmark itself):
    .venv_dysts/Scripts/python -W ignore dysts_classify.py --output data/dysts_classes.json

Method
  1. Symbolic: call the system's rhs(X, t) with sympy symbols for the state and t (dysts falls back to plain
     Python when numba is absent, so _rhs is ordinary arithmetic). Parameters are dysts' default numeric values.
     If sympy succeeds and every component is a polynomial in the state (no t), read off the coefficients in
     the repo's library ordering (core.EXPS for D = 3, degree 3; joint_nd.PolyLib ordering for D >= 4).
     Any exception (np.sin on a symbol, branching on a symbol, ...) or a non-polynomial expression marks it
     non-polynomial; delay systems are classified from metadata before any call.
  2. Numerical cross-check for every system that is not delay: on 2000 points of a clean trajectory
     (integrated with dysts' own rhs from its initial condition after a burn-in), least-squares fit of a
     degree-<=4 polynomial library; residual ~ 0 means polynomial. This confirms the symbolic classification.
  3. Verification for in-library systems: integrate our library rhs (LSODA, tol 1e-10), evaluate dysts' rhs
     on those points numerically, and require max |dysts - library| / max|dysts| < 1e-12.

Classes
  in3        D = 3, polynomial, degree <= 3 (20-term library of core.py)
  innd       4 <= D <= 10, polynomial, degree <= 2 (joint_nd PolyLib(D, 2))
  poly_out   polynomial but outside both (D = 3 deg >= 4, D 4-10 deg 3+, D > 10, D = 2 ...)
  nonpoly    trig / exp / rational / abs / piecewise / delay / non-autonomous (forced)
"""
import argparse
import itertools
import json
import os
import warnings

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

import dysts
import dysts.flows as F
from dysts.systems import get_attractor_list


def exps(D, deg):
    E = [e for d in range(deg + 1) for e in itertools.product(range(d + 1), repeat=D) if sum(e) == d]
    return sorted(set(E), key=lambda e: (sum(e), [-v for v in e]))


def libnames(E):
    var = list("xyz") if len(E[0]) == 3 else None
    out = []
    for e in E:
        if var:
            t = [v + (f"^{p}" if p > 1 else "") for v, p in zip(var, e) if p]
        else:
            t = [f"x{i}" + (f"^{p}" if p > 1 else "") for i, p in enumerate(e) if p]
        out.append("*".join(t) if t else "1")
    return out


def lib_eval(X, E):
    return np.prod(X[..., None, :] ** np.array(E), axis=-1)


# our existing systems (name in dysts -> our name), used to flag overlap; verified below by support equality
OURS = {"Lorenz": "lorenz", "Rossler": "rossler", "Chen": "chen", "Halvorsen": "halvorsen",
        "SprottA": "sprottA", "SprottB": "sprottB", "SprottC": "sprottC", "SprottD": "sprottD",
        "SprottE": "sprottE", "RabinovichFabrikant": "rabinovich",
        "LuChen": "lu", "ShimizuMorioka": "shimizu", "Rucklidge": "rucklidge", "GenesioTesi": "genesio",
        "Dadras": "dadras", "SprottF": "sprottF", "SprottG": "sprottG", "SprottH": "sprottH",
        "Hadley": "hadley", "BurkeShaw": "burkeshaw", "RikitakeDynamo": "rikitake", "ChenLee": "chenlee",
        "HyperRossler": "hyper_rossler", "Lorenz96": "l96_*", "LorenzCoupled": "coupled_lorenz"}


def symbolic(model, D):
    xs = sp.symbols(f"s0:{D}")
    t = sp.Symbol("t")
    out = model.rhs(np.array(xs, dtype=object), t)
    out = [sp.expand(sp.sympify(o)) for o in list(out)]
    if len(out) != D:
        raise ValueError(f"rhs returned {len(out)} components")
    if any(o.has(t) for o in out):
        return "nonautonomous", None, None
    polys = []
    for o in out:
        if not o.free_symbols <= set(xs):
            raise ValueError(f"free symbols {o.free_symbols}")
        if not o.is_polynomial(*xs):
            return "nonpoly_expr", None, None
        polys.append(sp.Poly(o, *xs))
    deg = max((p.total_degree() for p in polys if not p.is_zero), default=0)
    terms = [{tuple(int(v) for v in m): float(c) for m, c in p.terms() if float(c) != 0.0} for p in polys]
    return "poly", deg, terms


def dysts_traj(model, D, n=2000, burn=4000, dt=None):
    period = model.period or 1.0
    dt = dt or period / 128
    f = lambda t, x: np.asarray(model.rhs(np.asarray(x), t), float)
    ic = np.asarray(model.ic, float).ravel()[:D]
    tt = np.arange(burn + n) * dt
    sol = solve_ivp(f, (0, tt[-1]), ic, t_eval=tt, method="LSODA", rtol=1e-9, atol=1e-9)
    if not sol.success or sol.y.shape[1] != len(tt):
        return None
    return sol.y.T[burn:]


def numeric_poly_degree(model, X, maxdeg=4):
    """smallest degree <= maxdeg whose library fits dysts' rhs on X to ~machine precision, else None"""
    dX = np.array([np.asarray(model.rhs(x, 0.0), float) for x in X])
    D = X.shape[1]
    sc = np.abs(X).max(0) + 1e-12
    for deg in range(1, maxdeg + 1):
        E = exps(D, deg)
        if len(E) > 0.8 * len(X):
            break
        A = lib_eval(X / sc, E)
        c, *_ = np.linalg.lstsq(A, dX, rcond=None)
        r = np.abs(A @ c - dX).max() / (np.abs(dX).max() + 1e-300)
        if r < 1e-8:
            return deg, float(r)
    return None, None


FEATS = {"trig": r"np\.(sin|cos|tan|arctan|arcsin|arccos)\b", "exp/log": r"np\.(exp|log)\b",
         "hyperbolic": r"np\.(tanh|cosh|sinh)\b", "sqrt/power": r"np\.sqrt|\*\*\s*\(?-|\*\*\s*0?\.\d",
         "piecewise": r"np\.(heaviside|sign|abs|where|maximum|minimum|clip)\b|\bif\b|\babs\(",
         "division": r"[^/]/[^/]"}


def source_features(cls):
    """crude text scan of the rhs source (for the report only; the classification itself is symbolic)"""
    import inspect
    import re
    src = ""
    for c in cls.__mro__:
        if c.__module__.startswith("dysts.flows"):
            for k in ("_rhs", "rhs"):
                if k in c.__dict__:
                    try:
                        src += inspect.getsource(c.__dict__[k].__func__ if hasattr(c.__dict__[k], "__func__")
                                                 else c.__dict__[k])
                    except (TypeError, OSError):
                        pass
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", "@", "def ")))
    return sorted(k for k, pat in FEATS.items() if re.search(pat, body))


def nice_dt(period, per=128):
    """our rule (heldout_systems.DT): dt ~ period/128 rounded to a 1-2-5 grid"""
    v = period / per
    e = np.floor(np.log10(v))
    m = v / 10 ** e
    g = min((1, 2, 5, 10), key=lambda q: abs(np.log(m / q)))
    return float(f"{g * 10 ** e:.3g}")


def lib_simulate(Xi, E, x0, dt, n, burn=0):
    Ea = np.array(E)
    f = lambda t, x: lib_eval(x[None], Ea)[0] @ Xi
    tt = np.arange(burn + n) * dt
    sol = solve_ivp(f, (0, tt[-1]), x0, t_eval=tt, method="LSODA", rtol=1e-10, atol=1e-10)
    if not sol.success or sol.y.shape[1] != len(tt):
        return None
    return sol.y.T[burn:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/dysts_classes.json")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    from eqdisc.core import EXPS as CORE_EXPS
    from eqdisc.core import SYSTEMS
    from eqdisc.heldout_systems import HELDOUT
    assert [tuple(e) for e in CORE_EXPS] == exps(3, 3), "core ordering changed"
    ours = SYSTEMS | HELDOUT
    res = {}
    for name in args.only or get_attractor_list():
        model = getattr(F, name)()
        md = model.metadata
        D = len(np.asarray(model.ic).ravel())
        r = {"name": name, "D": D, "period": md.get("period"), "dysts_dt": md.get("dt"),
             "lyap_max": md.get("maximum_lyapunov_estimated"), "citation": md.get("citation"),
             "params": {k: (np.asarray(v).tolist()) for k, v in model.params.items()},
             "ic": np.asarray(model.ic, float).ravel().tolist(),
             "postprocessing": hasattr(model, "_postprocessing"),
             "nonautonomous_meta": bool(md.get("nonautonomous")), "delay": bool(md.get("delay"))}
        r["source_features"] = source_features(type(model))
        if r["delay"]:
            r.update(cls="nonpoly", reason="delay")
            res[name] = r
            print(f"{name:26} D={D:2} nonpoly (delay)", flush=True)
            continue
        try:
            kind, deg, terms = symbolic(model, D)
            r["symbolic"] = kind
        except Exception as exc:
            kind, deg, terms = "fail", None, None
            r["symbolic"] = f"fail: {type(exc).__name__}: {str(exc)[:80]}"
        # numerical cross-check
        try:
            X = dysts_traj(model, D)
            ndeg = numeric_poly_degree(model, X) if X is not None and np.isfinite(X).all() else (None, None)
        except Exception as exc:
            ndeg = (None, None)
            r["numeric_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        r["numeric_degree"], r["numeric_resid"] = ndeg
        if kind == "nonautonomous" or r["nonautonomous_meta"]:
            r.update(cls="nonpoly", reason="nonautonomous")
        elif kind != "poly":
            r.update(cls="nonpoly", reason="non-polynomial rhs" if kind == "nonpoly_expr" else r["symbolic"])
        else:
            r["degree"] = deg
            r["n_terms"] = [len(t) for t in terms]
            if D == 3 and deg <= 3:
                r["cls"], E = "in3", exps(3, 3)
            elif 4 <= D <= 10 and deg <= 2:
                r["cls"], E = "innd", exps(D, 2)
            else:
                r["cls"], E = "poly_out", None
                r["reason"] = f"D={D}, degree {deg}"
            if ndeg[0] is not None and ndeg[0] != deg:
                r["numeric_mismatch"] = True
            if E is not None:
                idx = {e: i for i, e in enumerate(E)}
                Xi = np.zeros((len(E), D))
                for j, eq in enumerate(terms):
                    for m, c in eq.items():
                        Xi[idx[m], j] = c
                r["names"] = libnames(E)
                r["xi"] = Xi.tolist()
                nz = np.abs(Xi[Xi != 0])
                r["coef_range"] = [float(nz.min()), float(nz.max())]
                # verification on our own clean integration
                per = r["period"] or 1.0
                dt = nice_dt(per)
                r["dt"] = dt
                x0 = np.asarray(model.ic, float).ravel()
                Y = lib_simulate(Xi, E, x0, dt, 4000, burn=int(round(50 * per / dt)))
                if Y is None or not np.isfinite(Y).all() or np.abs(Y).max() > 1e6:
                    r["verify"] = "integration failed/diverged"
                else:
                    dd = np.array([np.asarray(model.rhs(y, 0.0), float) for y in Y[::4]])
                    dl = lib_eval(Y[::4], np.array(E)) @ Xi
                    r["verify_relerr"] = float(np.abs(dd - dl).max() / np.abs(dd).max())
                    r["verify"] = "ok" if r["verify_relerr"] < 1e-12 else "MISMATCH"
                    r["std"] = Y.std(0).tolist()
                    r["absmax"] = float(np.abs(Y).max())
                    # stiffness proxy: max |eig(J)| * dt over the trajectory
                    Ea = np.array(E)
                    ev = []
                    for y in Y[::40]:
                        J = np.zeros((D, D))
                        for l in range(D):
                            e2 = Ea.copy(); a = e2[:, l].astype(float); e2[:, l] = np.maximum(e2[:, l] - 1, 0)
                            J[:, l] = (a * np.prod(y[None] ** e2, axis=-1)) @ Xi
                        ev.append(np.abs(np.linalg.eigvals(J)).max())
                    r["stiff_eig_dt"] = float(max(ev) * dt)
                    s = Y[:, 0] - Y[:, 0].mean()
                    r["x_crossings"] = int((np.diff(np.sign(s)) > 0).sum())
                # overlap with our systems: name map + exact support equality
                ov = OURS.get(name)
                if D == 3:
                    S = Xi != 0
                    same = [k for k, v in ours.items() if (v != 0).shape == S.shape and ((v != 0) == S).all()]
                    if same:
                        r["same_support_as"] = same
                if ov:
                    r["overlap"] = ov
        res[name] = r
        print(f"{name:26} D={D:2} {r['cls']:8} deg={r.get('degree')} num={r['numeric_degree']} "
              f"{r.get('verify', '')} {r.get('overlap', '')} {r.get('reason', '')}", flush=True)
    from importlib.metadata import version
    out = {"dysts_version": version("dysts"), "systems": res}
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    from collections import Counter
    print(Counter(r["cls"] for r in res.values()))


if __name__ == "__main__":
    main()
