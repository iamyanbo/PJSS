"""Held-out polynomial chaotic systems (never used while developing any method).

Equations and parameters from the original papers / Sprott (1994) / the dysts collection.
`check()` simulates each and reports boundedness and a rough non-periodicity test;
`dt` is set to about 1/60 of the mean zero-crossing period of x.
"""
import numpy as np

from .core import coef, simulate

HELDOUT = {
    "lu": coef([{"x": -36, "y": 36}, {"y": 20, "x*z": -1}, {"z": -3, "x*y": 1}]),
    "shimizu": coef([{"y": 1}, {"x": 1, "y": -0.75, "x*z": -1}, {"z": -0.45, "x^2": 1}]),
    "rucklidge": coef([{"x": -2, "y": 6.7, "y*z": -1}, {"x": 1}, {"z": -1, "y^2": 1}]),
    "genesio": coef([{"y": 1}, {"z": 1}, {"x": -1, "y": -1.1, "z": -0.44, "x^2": 1}]),
    "dadras": coef([{"y": 1, "x": -3, "y*z": 2.7}, {"y": 1.7, "x*z": -1, "z": 1}, {"x*y": 2, "z": -9}]),
    "sprottF": coef([{"y": 1, "z": 1}, {"x": -1, "y": 0.5}, {"x^2": 1, "z": -1}]),
    "sprottG": coef([{"x": 0.4, "z": 1}, {"x*z": 1, "y": -1}, {"x": -1, "y": 1}]),
    "sprottH": coef([{"y": -1, "z^2": 1}, {"x": 1, "y": 0.5}, {"x": 1, "z": -1}]),
    "hadley": coef([{"y^2": -1, "z^2": -1, "x": -0.2, "1": 0.2 * 8}, {"x*y": 1, "x*z": -4, "y": -1, "1": 1},
                    {"x*y": 4, "x*z": 1, "z": -1}]),
    "burkeshaw": coef([{"x": -10, "y": -10}, {"y": -1, "x*z": -10}, {"x*y": 10, "1": 4.272}]),
    "rikitake": coef([{"x": -1, "y*z": 1}, {"y": -1, "x*z": 1, "x": -5}, {"1": 1, "x*y": -1}]),
    "chenlee": coef([{"x": 5, "y*z": -1}, {"y": -10, "x*z": 1}, {"z": -0.38, "x*y": 1 / 3}]),
}
X0 = {"lu": [1, 1, 1], "shimizu": [0.1, 0, 0], "rucklidge": [1, 0, 4.5], "genesio": [0.1, 0.1, 0.1],
      "dadras": [1, 1, 0], "sprottF": [0.1, 0.1, 0.1], "sprottG": [0.1, 0.1, 0.1], "sprottH": [0.1, 0.1, 0.1],
      "hadley": [0.1, 0.1, 0.1], "burkeshaw": [0.6, 0, 0], "rikitake": [0.1, 0.1, 0.1], "chenlee": [1, 1, 1]}
# dt ~ 1/(128 f_peak) on a 1-2-5 grid, 128 = median samples per dominant cycle of the original ten
DT = {"lu": 0.005, "shimizu": 0.05, "rucklidge": 0.02, "genesio": 0.05, "dadras": 0.01, "sprottF": 0.05,
      "sprottG": 0.05, "sprottH": 0.05, "burkeshaw": 0.01, "rikitake": 0.02, "chenlee": 0.05}


def check(n=40000, base_dt=0.005):
    """Return {name: dt} for systems whose trajectory is bounded and irregular."""
    ok = {}
    for name, Xi in HELDOUT.items():
        x = simulate(Xi, X0[name], base_dt, n, burn=n // 4)
        if x is None or not np.isfinite(x).all() or np.abs(x).max() > 1e4:
            print(f"{name:10} diverged")
            continue
        s = x[:, 0] - x[:, 0].mean()
        zc = np.flatnonzero(np.diff(np.sign(s)) > 0)
        if len(zc) < 20 or x.std(0).min() < 1e-3:
            print(f"{name:10} not oscillating (crossings {len(zc)})")
            continue
        periods = np.diff(zc) * base_dt
        cv = periods.std() / periods.mean()
        # irregularity: peak heights of x vary (a limit cycle gives near-constant peaks)
        peaks = [s[a:b].max() for a, b in zip(zc[:-1], zc[1:])]
        pcv = np.std(peaks) / (np.mean(np.abs(peaks)) + 1e-12)
        dt = float(f"{periods.mean() / 60:.1g}")
        print(f"{name:10} period {periods.mean():.3f} period-cv {cv:.2f} peak-cv {pcv:.2f} -> dt {dt}")
        if pcv > 0.02 or cv > 0.02:
            ok[name] = dt
        else:
            print(f"{name:10} looks periodic; excluded")
    return ok


if __name__ == "__main__":
    print(check())


def generate(seed, length, output, seeds=10, noise=(.01, .05, .1, .2)):
    """Same protocol as flow_generate.py: random start on the attractor, Gaussian noise ~ eta * std."""
    rng = np.random.default_rng(seed)
    out = {key: [] for key in ("X", "sys", "noise", "n", "dt")}
    for name, Xi in HELDOUT.items():
        if name not in DT:
            continue
        ref = simulate(Xi, X0[name], DT[name], 10000, burn=4000)
        for _ in range(seeds):
            clean = simulate(Xi, ref[rng.integers(len(ref))], DT[name], length)
            for eta in noise:
                out["X"].append(clean + eta * clean.std(axis=0) * rng.standard_normal(clean.shape))
                out["sys"].append(name); out["noise"].append(eta); out["n"].append(length); out["dt"].append(DT[name])
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    return len(X)
