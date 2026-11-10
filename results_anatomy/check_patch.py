"""Check cost_anatomy.prepare_vectorized_lambda_fixed: identical to DEDi's original for D = 6, correct for D = 10.

  PYTHONPATH=.:~/eqdisc_dedi/shim:~/eqdisc_dedi/ext/DEDi python results_anatomy/check_patch.py
"""
import sys

import numpy as np

sys.argv = ["x"]
sys.path.insert(0, "benchmarks")
import cost_anatomy as ca  # noqa: E402
from DEDi.ODE.OdeLib import prepare_vectorized_lambda as orig  # noqa: E402
from eqdisc.joint_nd import PolyLib  # noqa: E402


def term(e):
    parts = [f"x_{i + 1}" + (f"**{p}" if p > 1 else "") for i, p in enumerate(e) if p]
    return "*".join(parts) if parts else "1"


rng = np.random.default_rng(0)
for D in (6, 10):
    lib = PolyLib(D, 2)
    terms = [term(e) for e in lib.E]
    X = rng.standard_normal((50, D))
    fixed = np.asarray(ca.prepare_vectorized_lambda_fixed(X, terms)(X, np.zeros(0), 0.0))
    direct = lib.theta(X) if hasattr(lib, "theta") else np.prod(X[:, None, :] ** lib.E, axis=-1)
    print(D, "fixed vs direct monomials, max abs diff:", float(np.max(np.abs(fixed - direct))))
    if D < 10:
        o = np.asarray(orig(X, terms)(X, np.zeros(0), 0.0))
        print(D, "fixed vs DEDi original, max abs diff:", float(np.max(np.abs(fixed - o))))
