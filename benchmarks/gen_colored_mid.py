"""DEV-ONLY archive of AR(1) coloured noise at intermediate correlation (for choosing the AR(1) switch threshold).

Same protocol as gen_variants.py --kind colored (random start on the attractor, noise eta * std * e,
e unit-variance AR(1)), but with rho in {0.3, 0.55} instead of 0.8, and 2 seeds per cell (320 runs).
Seed 20261501 (new; never used for any test archive). Field `rho` records the true AR coefficient.
"""
import argparse
import numpy as np
from eqdisc.core import SYSTEMS, X0, DT, simulate


def ar1(shape, rho, rng):
    e = rng.standard_normal(shape)
    for i in range(1, len(e)):
        e[i] = rho * e[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261501)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.3, 0.55])
    ap.add_argument("--output", default="data/dev_colored_mid.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {k: [] for k in ("X", "sys", "noise", "n", "dt", "rho")}
    for name in DT:
        ref = simulate(SYSTEMS[name], X0[name], DT[name], 10000, burn=4000)
        for length in (2000, 500):
            for _ in range(a.seeds):
                clean = simulate(SYSTEMS[name], ref[rng.integers(len(ref))], DT[name], length)
                for rho in a.rhos:
                    for eta in (.01, .05, .1, .2):
                        out["X"].append(clean + eta * clean.std(0) * ar1(clean.shape, rho, rng))
                        out["sys"].append(name); out["noise"].append(eta); out["n"].append(length)
                        out["dt"].append(DT[name]); out["rho"].append(rho)
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(a.output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    print(a.output, len(X))


if __name__ == "__main__":
    main()
