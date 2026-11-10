"""DEDi with the authors' full hyperparameter grid, one (run, lambda, R) pair per task.

Same method and settings as benchmarks/dedi_bench.py (authors' code unmodified, 20-term degree-3 library,
every-third-sample validation mask), but the grid is the paper's lambda in 10^{-3..3} x R in 10^{-4..0}
(35 pairs; "Parameters used in paper" in the authors' example notebooks) and each pair is run as its own
task, so a run's 35 pairs spread over several cores and every pair is saved as soon as it finishes.

Running FindBestModel([lam], [R]) once per pair is equivalent to FindBestModel(lambdas, Rs): the authors'
loop calls reset_variables() before every pair and the only state carried between pairs (MinLikely) is
never read. Selection (argmin validation error over the 35 pairs, as FindBestModel's default
output_sort='Validation') is done afterwards by benchmarks/dedi_fullgrid_report.py.

Workers: start N copies of this script, each pinned to its own core (taskset -c K); they share one task
list and claim tasks with exclusive lock files, so the job resumes after a kill (remove stale claims first,
see results_dedi_fullgrid/job.sh). Truth is used only for scoring.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LAMBDAS = [10.0**i for i in range(-3, 4)]
RS = [10.0**j for j in range(-4, 1)]


def term(e):
    parts = [f"x_{i + 1}" + (f"**{p}" if p > 1 else "") for i, p in enumerate(e) if p]
    return "*".join(parts) if parts else "1"


def tag(index, lam, r):
    return f"{index:04d}_l{np.log10(lam):+.0f}_r{np.log10(r):+.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/hard120.npz")
    ap.add_argument("--ext", required=True, help="folder holding shim/ and ext/DEDi")
    ap.add_argument("--order", required=True, help="json list of archive indices, run in this order")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()

    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path[:0] = [str(Path(args.ext) / "shim"), str(Path(args.ext) / "ext" / "DEDi")]
    from DEDi import ode_find
    from eqdisc.core import SYSTEMS, EXP_ARR, metrics

    out = Path(args.outdir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "claims").mkdir(parents=True, exist_ok=True)
    order = [int(i) for i in json.loads(Path(args.order).read_text())]
    tasks = [(i, lam, r) for i in order for lam in LAMBDAS for r in RS]  # run-major: runs finish in order
    lib = [term(e) for e in EXP_ARR]
    data = np.load(args.data, allow_pickle=True)
    cache = {}
    log = open(out / f"worker{args.worker:02d}.log", "a", buffering=1)
    for index, lam, r in tasks:
        name = tag(index, lam, r)
        res = out / "pairs" / f"{name}.json"
        if res.exists():
            continue
        try:
            os.close(os.open(out / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        x = np.asarray(data["X"][index], float)
        dt, system = float(data["dt"][index]), str(data["sys"][index])
        rec = {"index": index, "system": system, "lam": lam, "R": r, "worker": args.worker}
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if index not in cache:
                    cache.clear()
                    vm = np.ones_like(x, dtype=bool)
                    vm[2::3] = False
                    cache[index] = ode_find(x, lib, [0.0, dt * (len(x) - 1)], validation_mask=vm)
                model = cache[index]
                model.FindBestModel([lam], [r])
            lam_, r_, bic, val, model_err, state_err = model.tested_models[0][:6]
            xi = np.array(model.best_model()[0], dtype=float)
            rec.update(val_error=float(val), bic=float(bic), model_error=float(model_err),
                       state_error=float(state_err), nonzero=int(np.count_nonzero(xi)), xi=xi.tolist(),
                       dedi=metrics(xi, SYSTEMS[system]))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = time.perf_counter() - start
        tmp = res.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, default=float))
        tmp.replace(res)
        print(f"{time.strftime('%F %T')} {name} exact={rec.get('dedi', {}).get('exact')} "
              f"val={rec.get('val_error')} {rec['seconds']:.0f}s {rec.get('error', '')}", file=log)
    print(f"{time.strftime('%F %T')} worker {args.worker} no tasks left", file=log)


if __name__ == "__main__":
    main()
