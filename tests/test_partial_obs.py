"""Smoke tests for eqdisc.partial_obs (partial observation): solver pieces and one end-to-end recovery.

    python -m pytest tests/test_partial_obs.py -s        # ~1-3 min on one core
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))


def test_fast_library_matches_core():
    from eqdisc import core, partial_obs as po, select_k
    X = np.random.default_rng(0).standard_normal((300, 3)) * 3
    assert np.allclose(po.library(X), core.library(X), rtol=1e-12, atol=1e-10)
    assert np.allclose(po.dlibrary(X), select_k.dlibrary(X), rtol=1e-12, atol=1e-10)


def test_gauge_evaluator():
    """Truth, an offset-shifted equivalent form, and a sheared form pass; a dropped term and a degenerate linear
    model fail."""
    from po_eval import gauge_match, transformed
    from eqdisc.core import SYSTEMS, X0, simulate
    L = SYSTEMS["lorenz"]
    X = simulate(L, X0["lorenz"], 0.01, 2000, burn=1000)
    assert gauge_match(L, L, [0, 1], X, X)["exact_gauge"]
    T = transformed(L, [0, 1], [2], np.eye(1) * 3, np.zeros((1, 2)), np.array([28.0]), X)
    T[np.abs(T) < 1e-9] = 0
    U = X.copy()
    U[:, 2] = (U[:, 2] - 28) / 3
    assert np.count_nonzero(T) == 7 and gauge_match(T, L, [0, 1], U, X)["exact_gauge"]
    Ty = transformed(L, [0, 2], [1], np.eye(1), np.array([[1.0, 0.0]]), np.zeros(1), X)
    Ty[np.abs(Ty) < 1e-9] = 0
    U = X.copy()
    U[:, 1] = X[:, 1] - X[:, 0]
    assert gauge_match(Ty, L, [0, 2], U, X)["exact_gauge"]
    W = L.copy()
    W[3, 2] = 0
    assert not gauge_match(W, L, [0, 1], X, X)["exact_gauge"]


def test_lorenz_y_latent_recovered():
    """Lorenz with x and z observed (y latent), 1% noise, N = 2000: exact up to an affine change of y."""
    from po_eval import gauge_match
    from eqdisc import partial_obs as po
    from eqdisc.core import DT, SYSTEMS, X0, simulate
    rng = np.random.default_rng(1)
    clean = simulate(SYSTEMS["lorenz"], X0["lorenz"], DT["lorenz"], 2000, burn=1000)
    y = clean + 0.01 * clean.std(axis=0) * rng.standard_normal(clean.shape)
    obs = [0, 2]
    xi, info = po.discover(y[:, obs], DT["lorenz"], obs, kinds=("deriv",))
    g = gauge_match(xi, SYSTEMS["lorenz"], obs, np.array(info["u"]), clean)
    assert g["exact_gauge"], (xi, g)
