"""Archive for the timing/scaling study: Lorenz, 10% white noise, N in {500, 1000, 2000, 4000, 8000}, 2 seeds each.

  python benchmarks/timing_generate.py   # -> data/timing_scaling.npz (seed 20261028)
"""
import numpy as np

from eqdisc.core import SYSTEMS, DT, X0, simulate, add_noise

NS = [500, 1000, 2000, 4000, 8000]


def main():
    rng = np.random.default_rng(20261028)
    ref = simulate(SYSTEMS["lorenz"], X0["lorenz"], DT["lorenz"], 10000, burn=4000)  # as gen_variants.py
    xs, n = [], []
    for N in NS:
        for _ in range(2):
            clean = simulate(SYSTEMS["lorenz"], ref[rng.integers(len(ref))], DT["lorenz"], N)
            xs.append(add_noise(clean, 0.1, rng)); n.append(N)
    X = np.empty(len(xs), object)
    X[:] = xs
    np.savez("data/timing_scaling.npz", X=X, sys=np.array(["lorenz"] * len(xs)), noise=np.full(len(xs), 0.1),
             n=np.array(n), dt=np.full(len(xs), DT["lorenz"]))
    print(len(X), "runs")


if __name__ == "__main__":
    main()
