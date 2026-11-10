"""DIAGNOSTIC ONLY, NOT THE FROZEN MODEL. Works around the joint_nd crash on empty candidate supports.

joint_nd.discover builds `idx = np.array(key)` for each proposal; for the empty support key = () this gives a
float array and `A[:, idx]` raises IndexError (every D = 1 call, and some short D = 2 series). `apply()` swaps
the module's `np` for a proxy whose `array(())` returns an int array; nothing else changes. eqdisc/ is not
edited. Use only in clearly labelled diagnostic arms.
"""
import types

import numpy as np


class _NP(types.ModuleType):
    def __getattr__(self, k):
        return getattr(np, k)

    @staticmethod
    def array(x, *a, **k):
        if isinstance(x, tuple) and len(x) == 0 and not a and not k:
            return np.array(x, dtype=int)
        return np.array(x, *a, **k)


def apply():
    from eqdisc import joint_nd as jn
    jn.np = _NP("np_emptyfix_proxy")
