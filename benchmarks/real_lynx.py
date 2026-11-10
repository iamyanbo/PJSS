"""Real data: Hudson's Bay Company lynx / hare pelt counts 1900-1920 with the FROZEN final model, no tuning.

    python benchmarks/real_lynx.py --data data/real/hudson-bay-lynx-hare.csv --output results_real/lynx_hare.json

Accepted reference model: Lotka-Volterra, u = hare (prey), v = lynx (predator), thousands of pelts,
    du/dt = alpha u - beta u v,   dv/dt = -gamma v + delta u v
with point estimates alpha, beta, gamma, delta = 0.55, 0.028, 0.80, 0.024 (Stan case study posterior mean,
Carpenter 2018) and 0.55, 0.028, 0.84, 0.026 (Howard 2009). E-SINDy (Fasel et al. 2022), BINDy and Bayesian
spline learning report recovering this structure from the same 21 points.

Arms (fixed before running; see docs/reports/real_data.md):
  raw     the 21 annual points, dt = 1 year, columns (hare, lynx)  -- primary
  spline  natural cubic spline through the 21 points, resampled at dt = 0.1 year (201 points) -- sensitivity;
          interpolation makes the errors strongly correlated, so the noise model may switch to AR(1)
Model: eqdisc.discover(y, dt) with every default (D = 2 -> degree-2 joint_nd library). Truth is never used.
Reported: discovered equations, support vs LV, coefficients vs the reference values, and the R^2 of the
integrated discovered model (from the first observation) and of the integrated reference LV model against the
21 observations.
"""
import argparse
import json
import os
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

REF = {"stan": (0.55, 0.028, 0.80, 0.024), "howard": (0.55, 0.028, 0.84, 0.026)}


def load(path):
    rows = [l.split(",") for l in Path(path).read_text().splitlines() if l and not l.startswith(("#", "Year"))]
    a = np.array(rows, float)
    return a[:, 0], a[:, 2], a[:, 1]  # year, hare, lynx


def r2_vw(true, pred):
    sst = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ((true - pred) ** 2).sum() / sst)


def simulate(rhs, y0, t):
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="LSODA", rtol=1e-6, atol=1e-8)
    return sol.y.T if sol.success and sol.y.shape[1] == len(t) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/hudson-bay-lynx-hare.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--patch-empty-support", action="store_true",
                    help="DIAGNOSTIC (not frozen): benchmarks/_nd_emptyfix.py workaround for the joint_nd crash")
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out} exists")
    from eqdisc import discover
    from eqdisc.joint_nd import PolyLib
    if a.patch_empty_support:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _nd_emptyfix
        _nd_emptyfix.apply()
    year, hare, lynx = load(a.data)
    t_obs = year - year[0]
    Y = np.column_stack([hare, lynx])
    lib = PolyLib(2, 2)
    lv_support = np.zeros((lib.M, 2), bool)
    lv_support[lib.idx["x0"], 0] = lv_support[lib.idx["x0*x1"], 0] = True
    lv_support[lib.idx["x1"], 1] = lv_support[lib.idx["x0*x1"], 1] = True
    res_all = {"names": lib.names, "reference": REF, "arms": {}}
    for name, ref in REF.items():
        al, be, ga, de = ref
        p = simulate(lambda t, z: [al * z[0] - be * z[0] * z[1], -ga * z[1] + de * z[0] * z[1]], Y[0], t_obs)
        res_all[f"r2_reference_{name}"] = None if p is None else r2_vw(Y, p)
    arms = {"raw": (Y, 1.0)}
    tf = np.arange(0, t_obs[-1] + 1e-9, 0.1)
    cs = CubicSpline(t_obs, Y, bc_type="natural")
    arms["spline"] = (cs(tf), 0.1)
    for arm, (y, dt) in arms.items():
        t0 = time.perf_counter()
        try:
            r = discover(y, dt)
        except Exception as exc:  # frozen model failure: recorded, not worked around
            import traceback
            res_all["arms"][arm] = {"n": len(y), "dt": dt, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(limit=3)}
            print(arm, "ERROR", res_all["arms"][arm]["error"])
            continue
        xi = np.asarray(r.xi, float)
        rhs = lambda t, z, xi=xi: lib(np.asarray(z)[None, :])[0] @ xi
        p = simulate(rhs, Y[0], t_obs)
        sup = xi != 0
        rec = {"n": len(y), "dt": dt, "seconds": time.perf_counter() - t0, "equations": r.equations(),
               "xi": xi.tolist(), "support_is_lv": bool((sup == lv_support).all()),
               "extra_terms": [f"{lib.names[i]} (eq {j})" for i, j in zip(*np.nonzero(sup & ~lv_support))],
               "missing_terms": [f"{lib.names[i]} (eq {j})" for i, j in zip(*np.nonzero(~sup & lv_support))],
               "lv_coefficients": {"alpha": xi[lib.idx["x0"], 0], "beta": -xi[lib.idx["x0*x1"], 0],
                                   "gamma": -xi[lib.idx["x1"], 1], "delta": xi[lib.idx["x0*x1"], 1]},
               "r2_fit_vs_obs": None if p is None else r2_vw(Y, p), "ambiguous": r.ambiguous, "gap": r.gap,
               "noise_model": r.info.get("noise_model"), "sigma": np.asarray(r.info.get("sigma", [])).tolist(),
               "confidence": {f"{k[0]}|{k[1]}": v for k, v in r.confidence.items()}}
        res_all["arms"][arm] = rec
        print(arm, json.dumps({k: rec[k] for k in ("n", "support_is_lv", "extra_terms", "missing_terms",
                                                    "lv_coefficients", "r2_fit_vs_obs", "noise_model", "gap")},
                              default=float))
        print(rec["equations"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res_all, indent=1, default=float))
    print({k: v for k, v in res_all.items() if k.startswith("r2_")})


if __name__ == "__main__":
    main()
