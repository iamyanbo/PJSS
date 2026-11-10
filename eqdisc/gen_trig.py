"""Archives of chaotic systems with trigonometric terms (for the mixed library in joint_biglib).

Thomas' cyclically symmetric attractor x' = sin y - b x (cyclic), b = 0.18 and b = 0.1 (both checked chaotic: nearby trajectories separate; b = 0.208186 was too close to periodic).
Same protocol as heldout_systems.generate: random start on the attractor, noise eta * std per component.
"""
import argparse

import numpy as np
from scipy.integrate import solve_ivp

from .joint_biglib import MixedLibrary

LIB = MixedLibrary()


def thomas(b):
    return LIB.coef([{"sin(y)": 1, "x": -b}, {"sin(z)": 1, "y": -b}, {"sin(x)": 1, "z": -b}])


TRIG = {"thomas18": thomas(0.18), "thomas10": thomas(0.1)}
DT = {"thomas18": 0.2, "thomas10": 0.2}


def simulate(Xi, x0, dt, n, burn=0):
    f = LIB.rhs(Xi)
    t = np.arange(burn + n) * dt
    s = solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-10, method="DOP853")
    return s.y.T[burn:]


def check():
    for name, Xi in TRIG.items():
        a = simulate(Xi, [0.1, 0.0, -0.1], DT[name], 5000, burn=2000)
        b = simulate(Xi, a[0] + 1e-8, DT[name], 5000)
        print(name, "range", a.min(0).round(2), a.max(0).round(2), "separation", np.abs(a - b).max().round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {k: [] for k in ("X", "sys", "noise", "n", "dt")}
    for name, Xi in TRIG.items():
        ref = simulate(Xi, [0.1, 0.0, -0.1], DT[name], 10000, burn=4000)
        for length in (500, 2000):
            for _ in range(args.seeds):
                clean = simulate(Xi, ref[rng.integers(len(ref))], DT[name], length)
                for eta in (.01, .05, .1, .2):
                    out["X"].append(clean + eta * clean.std(axis=0) * rng.standard_normal(clean.shape))
                    out["sys"].append(name); out["noise"].append(eta); out["n"].append(length)
                    out["dt"].append(DT[name])
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(args.output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    print(len(X), "runs ->", args.output)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        check()
    else:
        main()
