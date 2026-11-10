"""Misspecification archives (simulation only) for the model-discrepancy work (docs/reports/discrepancy.md).

    python benchmarks/dc_generate.py --seed 2026102711 --output data/dc_dev.npz            # dev
    python benchmarks/dc_generate.py --seed 2026102793 --output data/dc_test.npz --set test # frozen test

Each system has a library (the one eqdisc.discover would use: "auto" = poly3 for D = 3, poly2 otherwise; the
"truncated" category runs D = 3 systems with library="poly2"), a category ("misspec": at least one equation is
not expressible in that library; "control": every equation is), and per equation a flag `inlib` plus the
true coefficients of the expressible equations in the PolyLib basis of the run's library (truth for the
per-equation exact-recovery metric). Misspecified systems are paired with matched in-library controls of the
same structure (kinematic chains, mechanics, jerk systems, Aizawa with/without its degree-4 term).

Protocol: per system a reference trajectory (burn-in) gives random starts (energy-shell starts for the
Hamiltonian/damped mechanical systems); dt = 1/(100 f_peak) on a 1-2-5 grid; per seed and N in {500, 2000} one
clean trajectory of N + 600 samples: the first N get Gaussian noise eta * per-component std, eta in
{0.01, 0.05, 0.1}; the last 600 clean samples are kept for forecasting (`F`). Runs whose clean trajectory fails
or leaves 20x the reference range are redrawn. The dev and test sets use different seeds; the test set also
uses different parameter values for every system (`--set test`).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

NOISE = (0.01, 0.05, 0.1)
LENGTHS = (500, 2000)
NF = 600


# ------------------------------------------------------------------ systems
def S(name, D, f, inlib, truth, library, category, group, x0, params=None):
    return dict(name=name, D=D, f=f, inlib=inlib, truth=truth, library=library, category=category, group=group,
                x0=x0, params=params or {})


def systems(which="dev"):
    t = which == "test"
    out = []
    # ---- D = 2, poly2 (auto): kinematic identity x0' = x1 in library
    g, c = (4.0, 0.05) if not t else (6.0, 0.08)
    out.append(S("pendulum", 2, lambda s, p=(g, c): [s[1], -p[0] * np.sin(s[0]) - p[1] * s[1]], [1, 0],
                 [{"x1": 1}, None], "poly2", "misspec", "osc2", lambda r: [r.uniform(1.5, 2.8) * r.choice([-1, 1]), 0.0]))
    d, a = (0.05, 1.0) if not t else (0.08, 1.5)
    out.append(S("duffing", 2, lambda s, p=(d, a): [s[1], -p[0] * s[1] + p[1] * s[0] - s[0] ** 3], [1, 0],
                 [{"x1": 1}, None], "poly2", "misspec", "osc2", lambda r: [r.uniform(1.6, 2.2) * r.choice([-1, 1]), 0.0]))
    mu = 1.5 if not t else 1.0
    out.append(S("vanderpol", 2, lambda s, m=mu: [s[1], m * (1 - s[0] ** 2) * s[1] - s[0]], [1, 0],
                 [{"x1": 1}, None], "poly2", "misspec", "osc2", lambda r: [r.uniform(-2, 2), r.uniform(-2, 2)]))
    w2, cd = (4.0, 0.05) if not t else (6.0, 0.08)
    out.append(S("linosc", 2, lambda s, p=(w2, cd): [s[1], -p[0] * s[0] - p[1] * s[1]], [1, 1],
                 [{"x1": 1}, {"x0": -w2, "x1": -cd}], "poly2", "control", "osc2",
                 lambda r: [r.uniform(1.5, 2.8) * r.choice([-1, 1]), 0.0]))
    q = 0.5 if not t else 0.4
    out.append(S("quadosc", 2, lambda s, p=(q, cd): [s[1], -s[0] - p[0] * s[0] ** 2 - p[1] * s[1]], [1, 1],
                 [{"x1": 1}, {"x0": -1, "x0^2": -q, "x1": -cd}], "poly2", "control", "osc2",
                 lambda r: [r.uniform(0.6, 1.0), 0.0]))
    al, be = (1.0, 0.5) if not t else (0.8, 0.6)
    out.append(S("lotka_volterra", 2, lambda s, p=(al, be): [p[0] * s[0] - s[0] * s[1], -p[1] * s[1] + s[0] * s[1]],
                 [1, 1], [{"x0": al, "x0*x1": -1}, {"x1": -be, "x0*x1": 1}], "poly2", "control", "lv",
                 lambda r: [r.uniform(0.2, 0.4), r.uniform(0.5, 1.5)]))
    # ---- D = 4, poly2 (auto): mechanics with kinematic identities
    gd, kd = (9.81, 0.05) if not t else (9.81, 0.02)

    def dpend(s, g=gd, k=kd):  # point masses m1 = m2 = 1, l1 = l2 = 1, viscous joint damping k
        t1, t2, w1, w2 = s
        de = t1 - t2
        den = 2 - np.cos(de) ** 2
        a1 = (-np.sin(de) * (w2 ** 2 + w1 ** 2 * np.cos(de)) - g * (2 * np.sin(t1) - np.sin(t2) * np.cos(de))) / den
        a2 = (np.sin(de) * (2 * w1 ** 2 + w2 ** 2 * np.cos(de)) + g * (2 * np.sin(t1) * np.cos(de) - 2 * np.sin(t2))) / den
        return [w1, w2, a1 - k * w1, a2 - k * w2]
    out.append(S("double_pendulum", 4, dpend, [1, 1, 0, 0], [{"x2": 1}, {"x3": 1}, None, None], "poly2", "misspec",
                 "mech4", lambda r: [r.uniform(1.2, 2.2) * r.choice([-1, 1]), r.uniform(-2.0, 2.0), 0.0, 0.0]))
    kc = 0.5 if not t else 0.8
    out.append(S("coupled_pendula", 4, lambda s, k=kc, c=cd: [s[2], s[3], -4 * np.sin(s[0]) - c * s[2] + k * (s[1] - s[0]),
                                                             -4 * np.sin(s[1]) - c * s[3] + k * (s[0] - s[1])],
                 [1, 1, 0, 0], [{"x2": 1}, {"x3": 1}, None, None], "poly2", "misspec", "mech4",
                 lambda r: [r.uniform(1.5, 2.6) * r.choice([-1, 1]), r.uniform(-1.0, 1.0), 0.0, 0.0]))
    kk = 0.2 if not t else 0.3
    out.append(S("coupled_duffing", 4, lambda s, k=kk: [s[2], s[3], s[0] - s[0] ** 3 - 0.05 * s[2] + k * (s[1] - s[0]),
                                                       s[1] - s[1] ** 3 - 0.05 * s[3] + k * (s[0] - s[1])],
                 [1, 1, 0, 0], [{"x2": 1}, {"x3": 1}, None, None], "poly2", "misspec", "mech4",
                 lambda r: [r.uniform(1.6, 2.1) * r.choice([-1, 1]), r.uniform(-2.0, 2.0), 0.0, 0.0]))
    lam = 1.0 if not t else 0.8
    out.append(S("henon_heiles", 4, lambda s, l=lam: [s[2], s[3], -s[0] - 2 * l * s[0] * s[1],
                                                     -s[1] - l * s[0] ** 2 + l * s[1] ** 2],
                 [1, 1, 1, 1], [{"x2": 1}, {"x3": 1}, {"x0": -1, "x0*x1": -2 * lam}, {"x1": -1, "x0^2": -lam, "x1^2": lam}],
                 "poly2", "control", "mech4", "hh"))
    k2 = 0.5 if not t else 0.8
    out.append(S("coupled_linosc", 4, lambda s, k=k2, c=cd: [s[2], s[3], -4 * s[0] - c * s[2] + k * (s[1] - s[0]),
                                                            -2 * s[1] - c * s[3] + k * (s[0] - s[1])],
                 [1, 1, 1, 1], [{"x2": 1}, {"x3": 1}, {"x0": -4 - k2, "x1": k2, "x2": -cd},
                                {"x1": -2 - k2, "x0": k2, "x3": -cd}], "poly2", "control", "mech4",
                 lambda r: [r.uniform(1.0, 2.0), r.uniform(-2.0, 2.0), 0.0, 0.0]))
    ra, rb = (0.25, 0.05) if not t else (0.25, 0.04)
    out.append(S("hyper_rossler", 4, lambda s, p=(ra, rb): [-s[1] - s[2], s[0] + p[0] * s[1] + s[3], 3 + s[0] * s[2],
                                                           -0.5 * s[2] + p[1] * s[3]],
                 [1, 1, 1, 1], [{"x1": -1, "x2": -1}, {"x0": 1, "x1": ra, "x3": 1}, {"1": 3, "x0*x2": 1},
                                {"x2": -0.5, "x3": rb}], "poly2", "control", "hyper", [-10, -6, 0, 10]))
    # ---- D = 3, poly3 (auto)
    ae, af = (0.25, 0.1) if not t else (0.25, 0.15)

    def aiz(f):
        return lambda s, f=f: [(s[2] - 0.7) * s[0] - 3.5 * s[1], 3.5 * s[0] + (s[2] - 0.7) * s[1],
                               0.6 + 0.95 * s[2] - s[2] ** 3 / 3 - (s[0] ** 2 + s[1] ** 2) * (1 + ae * s[2]) + f * s[2] * s[0] ** 3]
    a_tr = [{"x0*x2": 1, "x0": -0.7, "x1": -3.5}, {"x0": 3.5, "x1*x2": 1, "x1": -0.7}]
    out.append(S("aizawa", 3, aiz(af), [1, 1, 0], a_tr + [None], "poly3", "misspec", "aizawa", [0.1, 0, 0]))
    out.append(S("aizawa0", 3, aiz(0.0), [1, 1, 1],
                 a_tr + [{"1": 0.6, "x2": 0.95, "x2^3": -1 / 3, "x0^2": -1, "x1^2": -1, "x0^2*x2": -ae, "x1^2*x2": -ae}],
                 "poly3", "control", "aizawa", [0.1, 0, 0]))
    sat = 10.0 if not t else 8.0
    out.append(S("lorenz_sat", 3, lambda s, k=sat: [10 * (s[1] - s[0]), 28 * s[0] - s[1] - s[0] * s[2],
                                                   s[0] * s[1] / (1 + (s[0] / k) ** 2) - 8 / 3 * s[2]],
                 [1, 1, 0], [{"x0": -10, "x1": 10}, {"x0": 28, "x1": -1, "x0*x2": -1}, None], "poly3", "misspec",
                 "lorenz", [1, 1, 20]))
    out.append(S("lorenz", 3, lambda s: [10 * (s[1] - s[0]), 28 * s[0] - s[1] - s[0] * s[2], s[0] * s[1] - 8 / 3 * s[2]],
                 [1, 1, 1], [{"x0": -10, "x1": 10}, {"x0": 28, "x1": -1, "x0*x2": -1}, {"x0*x1": 1, "x2": -8 / 3}],
                 "poly3", "control", "lorenz", [1, 1, 20]))
    tb = 0.208186 if not t else 0.18
    out.append(S("thomas", 3, lambda s, b=tb: [np.sin(s[1]) - b * s[0], np.sin(s[2]) - b * s[1], np.sin(s[0]) - b * s[2]],
                 [0, 0, 0], [None, None, None], "poly3", "misspec", "thomas", [1.0, 0.5, -0.5]))
    # ---- D = 3 jerk chains: truncated library poly2 (misspec) and poly2 in-library controls
    T_, R_ = (6.0, 20.0) if not t else (7.0, 20.0)
    out.append(S("moore_spiegel", 3, lambda s, p=(T_, R_): [s[1], s[2], -s[2] - (p[0] - p[1] + p[1] * s[0] ** 2) * s[1] - p[0] * s[0]],
                 [1, 1, 0], [{"x1": 1}, {"x2": 1}, None], "poly2", "misspec", "jerk", [0.1, 0, 0]))
    aa = 5.5 if not t else 5.2
    out.append(S("arneodo", 3, lambda s, a=aa: [s[1], s[2], a * s[0] - 3.5 * s[1] - s[2] - s[0] ** 3],
                 [1, 1, 0], [{"x1": 1}, {"x2": 1}, None], "poly2", "misspec", "jerk", [0.2, 0.2, 0]))
    jA = 2.017 if not t else 2.03
    out.append(S("sprott_jerk", 3, lambda s, a=jA: [s[1], s[2], -a * s[2] + s[1] ** 2 - s[0]],
                 [1, 1, 1], [{"x1": 1}, {"x2": 1}, {"x2": -jA, "x1^2": 1, "x0": -1}], "poly2", "control", "jerk",
                 [2.0, 0, 0]))
    out.append(S("rossler_p2", 3, lambda s: [-s[1] - s[2], s[0] + 0.2 * s[1], 0.2 + s[2] * (s[0] - 5.7)],
                 [1, 1, 1], [{"x1": -1, "x2": -1}, {"x0": 1, "x1": 0.2}, {"1": 0.2, "x2": -5.7, "x0*x2": 1}],
                 "poly2", "control", "jerk", [1, 1, 0]))
    return out


# ------------------------------------------------------------------ helpers
def sim(f, x0, dt, n, burn=0):
    t = np.arange(burn + n) * dt
    with np.errstate(all="ignore"):
        s = solve_ivp(lambda tt, z: f(z), (0, t[-1]), np.asarray(x0, float), t_eval=t, method="LSODA",
                      rtol=1e-10, atol=1e-10)
    if not s.success or s.y.shape[1] != len(t) or not np.isfinite(s.y).all():
        return None
    return s.y.T[burn:]


def grid125(v):
    e = np.floor(np.log10(v))
    m = v / 10 ** e
    return float(max(x for x in (1, 2, 5) if x <= m) * 10 ** e) if m >= 1 else float(10 ** e)


def f_peak(X, dt):
    out = []
    for i in range(X.shape[1]):
        s = X[:, i] - X[:, i].mean()
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        p[:3] = 0
        out.append(np.fft.rfftfreq(len(s), dt)[np.argmax(p)])
    return max(out)


def hh_start(r, lam):  # Henon-Heiles: random point at energy E in [0.10, 0.15]
    while True:
        E = r.uniform(0.10, 0.15)
        x, y = r.uniform(-0.5, 0.5, 2)
        V = 0.5 * (x * x + y * y) + lam * (x * x * y - y ** 3 / 3)
        if V < E:
            ang = r.uniform(0, 2 * np.pi)
            v = np.sqrt(2 * (E - V))
            return [x, y, v * np.cos(ang), v * np.sin(ang)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", choices=["dev", "test"], default="dev")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {k: [] for k in ("X", "F", "sys", "noise", "n", "dt", "library", "category")}
    meta = {}
    for s in systems(a.set):
        if a.only and s["name"] not in a.only:
            continue
        name, f, D = s["name"], s["f"], s["D"]
        draw = (lambda r: hh_start(r, 1.0 if a.set == "dev" else 0.8)) if s["x0"] == "hh" else \
            (s["x0"] if callable(s["x0"]) else (lambda r, x=s["x0"]: x))
        # dt from a long probe run
        probe = sim(f, draw(rng), 0.01, 20000)
        dt = grid125(1.0 / (100 * f_peak(probe, 0.01)))
        mechanical = callable(s["x0"]) or s["x0"] == "hh"
        if not mechanical:  # chaotic attractor: starts on a reference trajectory after burn-in
            ref = sim(f, s["x0"], dt, 10000, burn=int(round(200 / (f_peak(probe, 0.01) * dt))))
        rng_range = np.abs(probe).max(0)
        meta[name] = {k: s[k] for k in ("D", "inlib", "truth", "library", "category", "group")} | {"dt": dt}
        meta[name]["discarded"] = 0
        for _ in range(a.seeds):
            for L in LENGTHS:
                while True:
                    x0 = draw(rng) if mechanical else ref[rng.integers(len(ref))]
                    clean = sim(f, x0, dt, L + NF)
                    if clean is not None and (np.abs(clean).max(0) < 20 * rng_range + 1).all() and clean[:L].std(0).min() > 1e-6:
                        break
                    meta[name]["discarded"] += 1
                for eta in NOISE:
                    X = clean[:L]
                    out["X"].append(X + eta * X.std(0) * rng.standard_normal(X.shape))
                    out["F"].append(clean[L:])
                    out["sys"].append(name); out["noise"].append(eta); out["n"].append(L); out["dt"].append(dt)
                    out["library"].append(s["library"]); out["category"].append(s["category"])
        print(f"{name:16} D={D} dt={dt:g} lib={s['library']} {s['category']} inlib={s['inlib']} "
              f"std={clean[:L].std(0).round(3)} discarded={meta[name]['discarded']}", flush=True)
    arr = {}
    for k in ("X", "F"):
        o = np.empty(len(out[k]), object)
        o[:] = out[k]
        arr[k] = o
    np.savez(a.output, **arr, **{k: np.array(v) for k, v in out.items() if k not in ("X", "F")})
    Path(a.output).with_suffix(".json").write_text(json.dumps({"seed": a.seed, "set": a.set, "systems": meta}, indent=1))
    print(a.output, len(out["X"]))


if __name__ == "__main__":
    main()
