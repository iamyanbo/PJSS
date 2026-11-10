"""Higher-dimensional polynomial chaotic systems for joint_nd (degree-2 library).

  l96_5, l96_6, l96_10   Lorenz-96, F = 8: x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
  hyper_rossler          4-D hyperchaotic Rossler (Rossler 1979): x'=-y-z, y'=x+0.25y+w, z'=3+xz, w'=-0.5z+0.05w
  coupled_lorenz         two Lorenz systems (10, 28, 8/3), diffusive x-coupling eps = 0.5

`check()` estimates the largest Lyapunov exponent from the divergence of two nearby trajectories
(renormalised every unit time); `generate()` writes an archive with the benchmark's noise protocol.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .joint_nd import PolyLib


def _l96(D, F=8.0):
    lib = PolyLib(D, 2)
    eqs = []
    for i in range(D):
        a, b, c = (i + 1) % D, (i - 2) % D, (i - 1) % D
        eq = {}

        def addt(names, v):
            key = "*".join(sorted(names, key=lambda s: int(s[1:]))) if len(names) == 2 and names[0] != names[1] \
                else (names[0] + "^2" if len(names) == 2 else names[0])
            eq[key] = eq.get(key, 0.0) + v
        addt([f"x{a}", f"x{c}"], 1.0)
        addt([f"x{b}", f"x{c}"], -1.0)
        eq[f"x{i}"] = -1.0
        eq["1"] = F
        eqs.append(eq)
    return lib, lib.coef(eqs)


def _hyper_rossler():
    lib = PolyLib(4, 2)
    return lib, lib.coef([{"x1": -1, "x2": -1}, {"x0": 1, "x1": 0.25, "x3": 1}, {"1": 3, "x0*x2": 1},
                          {"x2": -0.5, "x3": 0.05}])


def _coupled_lorenz(eps=0.5):
    lib = PolyLib(6, 2)
    eqs = []
    for o, other in ((0, 3), (3, 0)):
        x, y, z = (f"x{o}", f"x{o + 1}", f"x{o + 2}")
        xo = f"x{other}"
        eqs += [{x: -10 - eps, y: 10, xo: eps}, {x: 28, y: -1, f"{x}*{z}": -1}, {z: -8 / 3, f"{x}*{y}": 1}]
    return lib, lib.coef(eqs)


def _name_fix(lib, Xi):
    return Xi


BUILD = {"l96_5": lambda: _l96(5), "l96_6": lambda: _l96(6), "l96_10": lambda: _l96(10),
         "hyper_rossler": _hyper_rossler, "coupled_lorenz": _coupled_lorenz}
DT = {"l96_5": 0.01, "l96_6": 0.01, "l96_10": 0.01, "hyper_rossler": 0.05, "coupled_lorenz": 0.005}
X0 = {"l96_5": None, "l96_6": None, "l96_10": None, "hyper_rossler": [-10, -6, 0, 10],
      "coupled_lorenz": [1, 1, 1, -1, 2, 20]}


def system(name):
    return BUILD[name]()


def simulate(lib, Xi, x0, dt, n, burn=0):
    def f(t, x):
        return lib(x[None])[0] @ Xi
    t = np.arange(burn + n) * dt
    sol = solve_ivp(f, (0, t[-1]), np.asarray(x0, float), t_eval=t, method="LSODA", rtol=1e-10, atol=1e-10)
    if not sol.success or sol.y.shape[1] != len(t):
        return None
    return sol.y.T[burn:]


def start(name, rng):
    lib, Xi = system(name)
    if X0[name] is None:
        x0 = 8.0 + 0.01 * rng.standard_normal(lib.D)
    else:
        x0 = np.asarray(X0[name], float)
    return x0


def lyapunov(name, T=200.0, seed=0):
    lib, Xi = system(name)
    rng = np.random.default_rng(seed)
    dt = DT[name]
    ref = simulate(lib, Xi, start(name, rng), dt, 2, burn=int(50 / dt))
    a = ref[-1]
    b = a + 1e-8 * rng.standard_normal(len(a))
    s, unit = 0.0, int(round(1.0 / dt))
    for _ in range(int(T)):
        ta = simulate(lib, Xi, a, dt, unit + 1)
        tb = simulate(lib, Xi, b, dt, unit + 1)
        a, b = ta[-1], tb[-1]
        d = np.linalg.norm(b - a)
        s += np.log(d / 1e-8)
        b = a + (b - a) * (1e-8 / d)
    return s / T


def check():
    for name in BUILD:
        lib, Xi = system(name)
        x = simulate(lib, Xi, start(name, np.random.default_rng(0)), DT[name], 4000, burn=4000)
        ok = x is not None and np.isfinite(x).all()
        print(f"{name:15} D={lib.D} M={lib.M} terms/eq={[(Xi[:, i] != 0).sum() for i in range(lib.D)]} "
              f"bounded={ok} std={None if not ok else x.std(0).round(2)} lyap={lyapunov(name):.3f}")


def generate(seed, output, length=(500, 2000), seeds=10, noise=(.01, .05, .1, .2), names=None):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("X", "sys", "noise", "n", "dt")}
    for name in names or BUILD:
        lib, Xi = system(name)
        dt = DT[name]
        ref = simulate(lib, Xi, start(name, rng), dt, 10000, burn=int(100 / dt))
        for _ in range(seeds):
            for L in length:
                clean = simulate(lib, Xi, ref[rng.integers(len(ref))], dt, L)
                for eta in noise:
                    out["X"].append(clean + eta * clean.std(axis=0) * rng.standard_normal(clean.shape))
                    out["sys"].append(name); out["noise"].append(eta); out["n"].append(L); out["dt"].append(dt)
    X = np.empty(len(out["X"]), object)
    X[:] = out["X"]
    np.savez(output, X=X, **{k: np.array(v) for k, v in out.items() if k != "X"})
    return len(X)


if __name__ == "__main__":
    check()
