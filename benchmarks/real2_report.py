"""Tables for docs/reports/real_data2.md from results_real2/*.json.

    python benchmarks/real2_report.py [results_real2]
"""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1] if len(sys.argv) > 1 else "results_real2")
METHODS = ["eqdisc", "fd_stlsq_cv", "fd_stlsq_aic", "weak_stlsq_cv", "weak_stlsq_aic"]
LABEL = {"eqdisc": "ours (v1.1)", "fd_stlsq_cv": "SINDy STLSQ, CV", "fd_stlsq_aic": "SINDy STLSQ, AICc",
         "weak_stlsq_cv": "weak SINDy, CV", "weak_stlsq_aic": "weak SINDy, AICc"}


def f(v, d=3):
    return "–" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))


def load(name):
    p = O / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


print("## Single pendulum\n")
print("| Arm | Method | exact (4 terms) | dz eq exact | #terms | A | B | c_yz | c_xz | RHS rel. RMS (x, y, z) |"
      " median R²(θ) | R² per piece | noise / gap / time |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for arm in ("A0", "A1", "A2", "A3", "A4"):
    for m in METHODS:
        r = load(f"single_{arm}_{m}")
        if r is None:
            continue
        s, fit, info = r["structure"], r["fit"], r["info"]
        nt = sum(v != 0 for row in r["xi"] for v in row)
        extra = (f"{info.get('noise_model')} / {f(info.get('gap'))} / {info['seconds']:.0f}s" if m == "eqdisc"
                 else f"oracle on path: {info.get('oracle_exact_on_path')}")
        print(f"| {arm} | {LABEL[m]} | {s['exact']} | {s['physics_eq_exact']} | {nt} | {f(s['A'], 4)} | {f(s['B'], 3)} | "
              f"{f(s['c_yz'], 4)} | {f(s['c_xz'], 4)} | {', '.join(f(v, 2) for v in fit['rhs_rel_rms_vs_accepted'])} | "
              f"{f(fit['r2_theta_median'])} | {' '.join(f(v, 3) for v in fit['r2_theta_pieces'])} | {extra} |")
r = load("single_A0_eqdisc")
if r:
    print("\nAccepted-model R² per piece:", " ".join(f(v) for v in r["fit"]["r2_theta_pieces_accepted"]),
          "median", f(r["fit"]["r2_theta_median_accepted"]))
print("\n### Discovered equations\n")
for arm in ("A0", "A1", "A2", "A3", "A4"):
    for m in METHODS:
        r = load(f"single_{arm}_{m}")
        if r is not None and (m == "eqdisc" or arm == "A0"):
            print(f"**{arm}, {LABEL[m]}**\n```\n{r['eqs']}\n```")
            if m == "eqdisc":
                c = {k: v for k, v in r["info"]["confidence"].items() if v >= 0.05}
                print("confidence (p ≥ 0.05):", json.dumps({k: round(v, 3) for k, v in c.items()}))
print("\n## Double pendulum\n")
print("| Run | Method | #terms | median valid time [s] (disc / accepted / persistence) | per-start (same run) |")
print("| --- | --- | --- | --- | --- |")
for k in (1, 2, 3, 4):
    for m in METHODS:
        r = load(f"double_r{k}_{m}")
        if r is None:
            continue
        mv = r["forecast"]["median_valid_time"]
        print(f"| {k} | {LABEL[m]} | {r['nterms']} | {f(mv['disc'])} / {f(mv['accepted'])} / {f(mv['persistence'])} | "
              f"{' '.join(f(v, 2) for v in r['forecast']['same']['disc'])} |")
r = load("double_r1_eqdisc")
if r:
    print(f"\n**run 1, ours**\n```\n{r['eqs']}\n```\nnoise {r['info'].get('noise_model')}, gap {f(r['info'].get('gap'))}, "
          f"ambiguous {r['info'].get('ambiguous')}, {r['info']['seconds']:.0f}s")

# ---------------------------------------------------------------- EXPLORATORY (post hoc, not pre-registered)
# Physics equation only, integrated on the circle: theta' = omega, omega' = (dz/dt of the discovered model)
# evaluated at (sin theta, cos theta, omega).  This removes the off-manifold drift that the x^2 + y^2 = 1
# cancellations cause, and scores only the dz/dt equation.
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import real2_pendulum as rp  # noqa: E402

root = Path("data/real2/multiarm_pendulum")
if root.exists():
    th, om, lens, ref = rp.single_data(root)
    print("\n## Exploratory: physics equation integrated on the circle (median R²(θ) over the 6 pieces)\n")
    print("| Arm | " + " | ".join(LABEL[m] for m in METHODS) + " |")
    print("| --- |" + " --- |" * len(METHODS))
    for arm in ("A0", "A1", "A2", "A3", "A4"):
        row = []
        for m in METHODS:
            r = load(f"single_{arm}_{m}")
            if r is None:
                row.append("–")
                continue
            xi = np.asarray(r["xi"])[:, 2]
            F = rp.feat_fn(r["names"], 3)
            rhs = lambda t, s, xi=xi, F=F: [s[1], float((F(np.array([[np.sin(s[0]), np.cos(s[0]), s[1]]])) @ xi)[0])]
            vals, start = [], 0
            for L in lens:
                p = rp.simulate(rhs, [th[start], om[start]], np.arange(L) * rp.DT_RAW)
                vals.append(rp.r2(th[start:start + L], None if p is None or len(p) < L else p[:, 0]))
                start += L
            row.append("fail" if any(v is None for v in vals) else f"{np.median(vals):.3f}")
        print(f"| {arm} | " + " | ".join(row) + " |")
