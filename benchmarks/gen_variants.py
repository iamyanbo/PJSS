"""Fresh test archives (new seeds) for noise variants and held-out systems.

Same protocol as flow_generate.py / bench.py: random start on the attractor, noise ~ eta * std.
colored = AR(1) rho 0.8 unit variance; student = t(3) unit variance; coarse = every 4th sample.
"""
import argparse
import numpy as np
from eqdisc.core import SYSTEMS, X0, DT, simulate
from eqdisc.heldout_systems import HELDOUT, X0 as HX0, DT as HDT


def noise(shape, kind, rng):
    if kind == "colored":
        e = rng.standard_normal(shape)
        for i in range(1, len(e)):
            e[i] = 0.8 * e[i - 1] + 0.6 * e[i]
        return e
    if kind == "student":
        return rng.standard_t(3, shape) / np.sqrt(3.0)
    return rng.standard_normal(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["white", "colored", "student", "coarse"], default="white")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--output", required=True)
    ap.add_argument("--etas", type=float, nargs="+", default=[.01, .05, .1, .2],
                    help="noise levels (eta = 0 gives noise-free trajectories; precision audit)")
    a = ap.parse_args()
    systems, x0s, dts = (HELDOUT, HX0, HDT) if a.heldout else (SYSTEMS, X0, DT)
    rng = np.random.default_rng(a.seed)
    stride = 4 if a.kind == "coarse" else 1
    out = {k: [] for k in ("X", "sys", "noise", "n", "dt")}
    for name in dts:
        ref = simulate(systems[name], x0s[name], dts[name], 10000, burn=4000)
        for length in (2000, 500):
            for _ in range(a.seeds):
                clean = simulate(systems[name], ref[rng.integers(len(ref))], dts[name], length * stride)[::stride]
                for eta in a.etas:
                    out["X"].append(clean + eta * clean.std(0) * noise(clean.shape, a.kind, rng))
                    out["sys"].append(name); out["noise"].append(eta); out["n"].append(length)
                    out["dt"].append(dts[name] * stride)
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(a.output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    print(a.output, len(X))


if __name__ == "__main__":
    main()
