"""Worker for test_equivalence: run one archive run through the frozen and the refactored model.

Imported in spawned processes; nothing heavy is imported at module level so EQ_DEG (read by `core` at import
time) can be set by the parent before the child starts.
"""
import sys


def run_one(payload):
    frozen_dir, archive, index, library = payload
    import numpy as np
    from threadpoolctl import threadpool_limits

    if frozen_dir not in sys.path:
        sys.path.insert(0, frozen_dir)
    import final_model as frozen  # flat modules extracted from commit ac7ab20
    from eqdisc import final_model as new

    assert frozen.__file__.startswith(frozen_dir), frozen.__file__
    with np.load(archive, allow_pickle=True) as d:
        x = np.asarray(d["X"][index], float)
        dt = float(d["dt"][index])
        meta = {"system": str(d["sys"][index]), "noise": float(d["noise"][index]), "n": len(x), "D": x.shape[1]}
    out = {"archive": archive, "index": index, "library": library, **meta}
    with threadpool_limits(limits=1):
        a = frozen.discover(x, dt, library=library)
        b = new.discover(x, dt, library=library, version="1.0.1")  # v1.1 differs by design on AR(1) runs
    out["xi_equal"] = bool(a.xi.shape == b.xi.shape and np.array_equal(a.xi, b.xi))
    out["max_abs_diff"] = float(np.max(np.abs(a.xi - b.xi))) if a.xi.shape == b.xi.shape else float("inf")
    out["names_equal"] = list(a.names) == list(b.names)
    out["gap_equal"] = bool(a.gap == b.gap)
    out["ambiguous_equal"] = a.ambiguous == b.ambiguous
    out["confidence_equal"] = a.confidence == b.confidence
    out["noise_model"] = (a.info.get("noise_model"), b.info.get("noise_model"))
    out["n_terms"] = int((a.xi != 0).sum())
    return out
