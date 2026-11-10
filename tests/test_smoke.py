"""Quick end-to-end check of the public API: Lorenz, N = 2000, 5% noise -> exact support."""
import numpy as np


def test_lorenz_5pct_exact():
    from eqdisc import discover
    from eqdisc.core import DT, SYSTEMS, X0, metrics, simulate

    rng = np.random.default_rng(0)
    clean = simulate(SYSTEMS["lorenz"], X0["lorenz"], DT["lorenz"], 2000, burn=1000)
    y = clean + 0.05 * clean.std(axis=0) * rng.standard_normal(clean.shape)
    res = discover(y, DT["lorenz"])
    assert metrics(res.xi, SYSTEMS["lorenz"])["exact"] == 1.0, res.equations()
    assert res.info["noise_model"] == "white"
    assert res.names[:4] == ["1", "x", "y", "z"]
    assert "dx/dt" in res.equations()


def test_1d_logistic_no_crash():
    """D = 1 (v1.0.1 fix): every 1-D call used to raise IndexError on the empty candidate support."""
    from scipy.integrate import solve_ivp
    from eqdisc import discover

    t = np.linspace(0, 10, 512)
    sol = solve_ivp(lambda _, x: 0.8 * x * (1 - x / 5), (0, 10), [0.2], t_eval=t, rtol=1e-10, atol=1e-12)
    x = sol.y.T
    y = x + 0.01 * x.std() * np.random.default_rng(0).standard_normal(x.shape)
    res = discover(y, t[1] - t[0])
    assert res.xi.shape == (3, 1) and res.names == ["1", "x0", "x0^2"]
    assert [res.names[i] for i in np.flatnonzero(res.xi[:, 0])] == ["x0", "x0^2"], res.equations()
    assert abs(res.xi[1, 0] - 0.8) < 0.05 and abs(res.xi[2, 0] + 0.16) < 0.01, res.equations()
