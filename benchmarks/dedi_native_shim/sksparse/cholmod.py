"""Adapter for timing only: the scikit-sparse 0.4 names DEDi imports (analyze / cholesky_inplace / solve_A), backed by
the native CHOLMOD of the installed scikit-sparse 0.5 (conda-forge SuiteSparse). Symbolic analysis happens on the
first factorisation and the factor object is refactorised afterwards, as with 0.4's analyze + cholesky_inplace."""
import importlib

import numpy as np
import sys
from pathlib import Path

_shim_pkg = sys.modules.pop("sksparse")
_me = sys.modules.pop("sksparse.cholmod")
_path = sys.path[:]
_here = Path(__file__).resolve().parents[1]  # the shim folder holding this sksparse package (symlinks resolved)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _here]
_native = importlib.import_module("sksparse.cholmod")
sys.path[:] = _path
sys.modules["sksparse"] = _shim_pkg
sys.modules["sksparse.cholmod"] = _me

CholmodError = _native.CholmodError
cholesky = _native.cholesky


class _Factor:
    def __init__(self):
        self.F = None

    def cholesky_inplace(self, H):
        if self.F is None:
            self.F = _native.cho_factor(H)
        else:
            self.F.factorize(H)

    def solve_A(self, b):
        return self.F.solve(np.array(b, dtype=np.float64))  # writable copy (jax arrays are read-only views)


def analyze(H):
    return _Factor()
