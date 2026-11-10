"""Full-grid DEDi on the pre-registered head-to-head sample (results_h2h/order.json), one (run, lambda, R) pair per task.

Same as benchmarks/dedi_fullgrid.py (authors' code unmodified, 20-term degree-3 library, every-third-sample
validation mask, the paper's 35-pair grid), with two differences:
  * runs come from several archives: order.json is a list of {"cond", "data", "index"} and results are keyed by
    "<cond>_<index>";
  * --device cpu|gpu passes DEDi's own cpu_or_gpu argument. A run is bound to the device of the first worker that
    touches it (runs/<key>.<device>, created exclusively); workers of the other device skip it, so CPU and GPU
    numerics are never mixed within one run (docs/reports/dedi_gpu.md).

Workers: start N copies, each pinned to its own core (taskset -c K). Resumable: remove stale claims first
(results_h2h/job.sh). Truth is used only for scoring.
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


def tag(key, lam, r):
    return f"{key}_l{np.log10(lam):+.0f}_r{np.log10(r):+.0f}"


def bind(out, key, device):
    """True if this run belongs to `device` (binding it now if it is still unbound)."""
    mine = out / "runs" / f"{key}.{device}"
    if mine.exists():
        return True
    if any((out / "runs").glob(f"{key}.*")):
        return False
    try:
        os.close(os.open(mine, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return True
    other = [p for p in (out / "runs").glob(f"{key}.*") if p != mine]
    if other:  # lost a race with the other device: back off
        mine.unlink()
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext", required=True, help="folder holding shim/ and ext/DEDi")
    ap.add_argument("--order", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--truth-json", default=None, help="sidecar with systems[name]['xi'] (dysts); default: SYSTEMS")
    args = ap.parse_args()

    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path[:0] = [str(Path(args.ext) / "shim"), str(Path(args.ext) / "ext" / "DEDi")]
    from DEDi import ode_find
    from eqdisc.core import SYSTEMS, EXP_ARR, metrics
    if args.truth_json:
        side = json.loads(Path(args.truth_json).read_text())["systems"]
        SYSTEMS = {k: np.asarray(v["xi"], float) for k, v in side.items()}

    out = Path(args.outdir)
    for d in ("pairs", "claims", "runs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    order = json.loads(Path(args.order).read_text())
    lib = [term(e) for e in EXP_ARR]
    archives = {}
    cache = {}
    log = open(out / f"worker_{args.device}{args.worker:02d}.log", "a", buffering=1)
    for run in order:
        key = f"{run['cond']}_{run['index']}"
        for lam in LAMBDAS:
            for r in RS:
                name = tag(key, lam, r)
                res = out / "pairs" / f"{name}.json"
                if res.exists() or not bind(out, key, args.device):
                    continue
                try:
                    os.close(os.open(out / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
                if run["data"] not in archives:
                    archives[run["data"]] = np.load(run["data"], allow_pickle=True)
                data = archives[run["data"]]
                index = int(run["index"])
                x = np.asarray(data["X"][index], float)
                dt, system = float(data["dt"][index]), str(data["sys"][index])
                rec = {"key": key, "cond": run["cond"], "index": index, "system": system, "lam": lam, "R": r,
                       "device": args.device, "worker": args.worker}
                start = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        if key not in cache:
                            cache.clear()
                            vm = np.ones_like(x, dtype=bool)
                            vm[2::3] = False
                            cache[key] = ode_find(x, lib, [0.0, dt * (len(x) - 1)], validation_mask=vm,
                                                  cpu_or_gpu=args.device)
                        model = cache[key]
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
    print(f"{time.strftime('%F %T')} worker {args.device}{args.worker} no tasks left", file=log)


if __name__ == "__main__":
    main()
