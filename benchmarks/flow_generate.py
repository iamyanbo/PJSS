"""Generate a reproducible, independently seeded validation archive.

The output is for flow_bench.py. Generating data uses known equations; discovery
never receives system names, clean states, true coefficients, or noise levels.
"""
import argparse
from pathlib import Path
import numpy as np
from eqdisc.core import SYSTEMS, X0, DT, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--systems", nargs="+", default=["lorenz", "rossler", "chen", "rabinovich"])
    parser.add_argument("--noise", nargs="+", type=float, default=[.05, .2])
    parser.add_argument("--output", default="data/flow_fresh.npz")
    args = parser.parse_args()
    if args.seeds < 1 or args.length < 2 or any(e < 0 for e in args.noise):
        parser.error("Require positive seeds, length >= 2, and nonnegative noise")
    unknown = set(args.systems) - SYSTEMS.keys()
    if unknown:
        parser.error(f"Unknown systems: {sorted(unknown)}")
    rng = np.random.default_rng(args.seed)
    out = {key: [] for key in ("X", "sys", "noise", "n", "dt")}
    for name in args.systems:
        ref = simulate(SYSTEMS[name], X0[name], DT[name], 10000, burn=2000)
        if ref is None:
            raise RuntimeError(f"Reference simulation failed: {name}")
        for seed_index in range(args.seeds):
            x0 = ref[rng.integers(len(ref))]
            clean = simulate(SYSTEMS[name], x0, DT[name], args.length)
            if clean is None or not np.isfinite(clean).all():
                raise RuntimeError(f"Simulation failed: {name}, draw {seed_index}")
            for noise in args.noise:
                out["X"].append(clean + noise*clean.std(axis=0)*rng.standard_normal(clean.shape))
                out["sys"].append(name)
                out["noise"].append(noise)
                out["n"].append(args.length)
                out["dt"].append(DT[name])
        print(f"Generated {name}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.stack(out["X"]), sys=out["sys"], noise=out["noise"],
                        n=out["n"], dt=out["dt"], generator_seed=args.seed)
    print(f"Saved {len(out['X'])} instances to {output}")


if __name__ == "__main__":
    main()
