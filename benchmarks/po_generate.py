"""Partial-observation archives: Lorenz, Rossler, Chen; N = 2000; noise 1% and 5% of each coordinate's std.

Each instance stores the full clean trajectory (for evaluation only) and the full noisy trajectory; a method
receives only the observed columns. Every instance is run under all observation configurations (paired).
Seeds must be new: dev 20261701, test 20261702 (neither appears anywhere else in the repository).

    python benchmarks/po_generate.py --seed 20261701 --seeds 5 --output data/po_dev.npz
"""
import argparse

import numpy as np

from eqdisc.core import SYSTEMS, X0, DT, simulate

PO_SYSTEMS = ("lorenz", "rossler", "chen")
NOISES = (0.01, 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {k: [] for k in ("X", "clean", "sys", "noise", "n", "dt")}
    for name in PO_SYSTEMS:
        ref = simulate(SYSTEMS[name], X0[name], DT[name], 10000, burn=4000)
        for _ in range(a.seeds):
            clean = None
            while clean is None:
                clean = simulate(SYSTEMS[name], ref[rng.integers(len(ref))], DT[name], a.n)
            for eta in NOISES:
                out["X"].append(clean + eta * clean.std(0) * rng.standard_normal(clean.shape))
                out["clean"].append(clean)
                out["sys"].append(name)
                out["noise"].append(eta)
                out["n"].append(a.n)
                out["dt"].append(DT[name])
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    Cl = np.empty(len(out["clean"]), object)
    Cl[:] = out["clean"]
    np.savez(a.output, X=X, clean=Cl, **{k: np.array(v) for k, v in out.items() if k not in ("X", "clean")})
    print(a.output, len(X))


if __name__ == "__main__":
    main()
