"""eqdisc: sparse ODE discovery from one noisy trajectory by profiled joint flow.

    from eqdisc import discover
    res = discover(y, dt)          # y: (n, D) noisy samples, dt: sample step
    print(res.equations())

`discover` is the frozen final model (eqdisc.final_model), v1.2 by default; `discover(y, dt, version="1.1")` gives
v1.1 and `version="1.0.1"` gives v1.0.1 (= frozen v1.0 + the empty-support crash fix). See README.md for options.
Polynomial libraries of degree 4/5 need EQ_DEG set before eqdisc.core is first imported (see README).
"""
from .final_model import DEFAULTS, VERSIONS, Result, discover

__all__ = ["discover", "Result", "DEFAULTS", "VERSIONS"]
__version__ = "1.2.0"
