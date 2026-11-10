"""Time DEDi (authors' code, unmodified) on CPU vs GPU for chosen (run, lambda, R) pairs.

Same settings as benchmarks/dedi_fullgrid.py (20-term degree-3 library, every-third-sample validation
mask, FindBestModel defaults, float64); the only change is the `cpu_or_gpu` argument of `ode_find`,
which DEDi itself exposes. One JSON line per pair is appended to --out, with the selected coefficients,
validation error/BIC and wall time, plus the time spent inside the CPU sparse-Cholesky shim (summed
over DEDi's two optimisation threads).

GPU runs need a CUDA jax build (see docs/reports/dedi_gpu.md); set XLA_PYTHON_CLIENT_PREALLOCATE=false
so a process does not reserve most of the Spark's unified memory.

  taskset -c 10 python benchmarks/dedi_gpu_bench.py --device gpu --ext ~/eqdisc_dedi \
      --index 43 --pairs 1e-3,1 1,1e-2 --out results_dedi_gpu/gpu.jsonl
"""
import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


def term(e):
    parts = [f"x_{i + 1}" + (f"**{p}" if p > 1 else "") for i, p in enumerate(e) if p]
    return "*".join(parts) if parts else "1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/hard120.npz")
    ap.add_argument("--ext", required=True, help="folder holding shim/ and ext/DEDi")
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="lam,R pairs, e.g. 1e-3,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
                               + os.environ.get("DEDI_XLA_EXTRA", "")).strip()
    sys.path[:0] = [str(Path(args.ext) / "shim"), str(Path(args.ext) / "ext" / "DEDi")]
    import sksparse.cholmod as chol
    from DEDi import ode_find
    from eqdisc.core import SYSTEMS, EXP_ARR, metrics

    # Wall time spent in the CPU linear-algebra shim (DEDi's two threads run concurrently, so this sums both).
    shim_time = {"s": 0.0}
    lock = threading.Lock()

    def timed(fn):
        def wrap(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                with lock:
                    shim_time["s"] += time.perf_counter() - t0
        return wrap

    chol._Factor.cholesky_inplace = timed(chol._Factor.cholesky_inplace)
    chol._Factor.solve_A = timed(chol._Factor.solve_A)

    data = np.load(args.data, allow_pickle=True)
    x = np.asarray(data["X"][args.index], float)
    dt, system = float(data["dt"][args.index]), str(data["sys"][args.index])
    lib = [term(e) for e in EXP_ARR]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        vm = np.ones_like(x, dtype=bool)
        vm[2::3] = False
        model = ode_find(x, lib, [0.0, dt * (len(x) - 1)], validation_mask=vm, cpu_or_gpu=args.device)
    setup = time.perf_counter() - t0
    import jax
    dev = str(jax.devices()[0])
    for p in args.pairs:
        lam, r = (float(v) for v in p.split(","))
        shim_time["s"] = 0.0
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            model.FindBestModel([lam], [r])
        lam_, r_, bic, val, model_err, state_err = model.tested_models[0][:6]
        xi = np.array(model.best_model()[0], dtype=float)
        rec = {"index": args.index, "system": system, "lam": lam, "R": r, "device": args.device, "jax_device": dev,
               "tag": args.tag, "val_error": float(val), "bic": float(bic), "model_error": float(model_err),
               "state_error": float(state_err), "nonzero": int(np.count_nonzero(xi)), "xi": xi.tolist(),
               "dedi": metrics(xi, SYSTEMS[system]), "seconds": time.perf_counter() - start,
               "shim_seconds": shim_time["s"], "setup_seconds": setup, "x64": bool(jax.config.jax_enable_x64)}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"{time.strftime('%F %T')} {args.device} {args.index} lam={lam:g} R={r:g} "
              f"{rec['seconds']:.1f}s shim={rec['shim_seconds']:.1f}s exact={rec['dedi'].get('exact')}", flush=True)


if __name__ == "__main__":
    main()
