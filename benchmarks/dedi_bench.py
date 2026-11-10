"""Meissner & Glasner's DEDi (github.com/TeddyMeissner/DEDi, arXiv 2406.06707v2) on our instances.

Joint state + sparse-coefficient estimation; the authors' code is used unmodified, except that
sksparse.cholmod (no Windows build) is replaced by an exact banded/Schur Cholesky shim and
IPython.display by a no-op (printing only). Same 20-term degree-3 library as everything else,
same every-third-sample validation mask as the authors' examples. Hyperparameters: the paper's
grid lambda in 10^{-3..3}, R in 10^{-4..0} (35 pairs), best by validation error, unless --grid small.
Truth is used only for scoring.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

GRIDS = {"paper": ([10.0**i for i in range(-3, 4)], [10.0**j for j in range(-4, 1)]),
         "small": ([1e-2, 1.0], [1e-4, 1e-2])}


def term(e):
    parts = [f"x_{i + 1}" + (f"**{p}" if p > 1 else "") for i, p in enumerate(e) if p]
    return "*".join(parts) if parts else "1"


def _init(ext_root):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path[:0] = [str(Path(ext_root) / "shim"), str(Path(ext_root) / "ext" / "DEDi")]


def _job(payload):
    index, x, dt, name, noise, grid = payload
    import contextlib, io
    from eqdisc.core import SYSTEMS, EXP_ARR, metrics
    from DEDi import ode_find
    lib = [term(e) for e in EXP_ARR]
    vm = np.ones_like(x, dtype=bool)
    vm[2::3] = False
    rec = {"index": index, "system": name, "noise": noise, "n": len(x)}
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = ode_find(x, lib, [0.0, dt * (len(x) - 1)], validation_mask=vm)
            model.FindBestModel(*GRIDS[grid])
        xi = np.array(model.best_model()[0], dtype=float)
        rec["dedi"] = metrics(xi, SYSTEMS[name])
        rec["chosen"] = [float(v) for v in model.tested_models[0][:2]]
    except Exception as exc:
        rec["dedi_error"] = f"{type(exc).__name__}: {exc}"
        rec["dedi"] = {"exact": 0.0}
    rec["seconds"] = time.perf_counter() - start
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/hybrid_confirm.npz")
    ap.add_argument("--ext", required=True, help="folder holding shim/ and ext/DEDi")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--noise", nargs="+", type=float)
    ap.add_argument("--n", nargs="+", type=int)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid", choices=GRIDS, default="paper")
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET selected cases (to split a run across machines)")
    ap.add_argument("--index-file", help="json list of archive indices, run in this order (overrides the filters)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        ap.error(f"{out} exists; refusing to overwrite")
    with np.load(args.data, allow_pickle=True) as d:
        if args.index_file:  # pre-registered order; filters ignored
            order = [int(i) for i in json.loads(Path(args.index_file).read_text())]
            tasks = [(i, np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i]), float(d["noise"][i]),
                      args.grid) for i in order]
        else:
            keep = np.ones(len(d["X"]), bool)
            if args.systems:
                keep &= np.isin(d["sys"], args.systems)
            if args.noise:
                keep &= np.isin(np.round(d["noise"], 4), np.round(args.noise, 4))
            if args.n:
                keep &= np.isin(d["n"], args.n)
            tasks, seen = [], {}
            for i in np.flatnonzero(keep):
                cell = (str(d["sys"][i]), float(d["noise"][i]), int(d["n"][i]))
                seen[cell] = seen.get(cell, 0) + 1
                if seen[cell] <= args.seeds:
                    tasks.append((int(i), d["X"][i], float(d["dt"][i]), cell[0], cell[1], args.grid))
    tasks = tasks[args.offset:]
    print(f"{len(tasks)} cases", flush=True)
    records, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.ext,)) as pool:
        # save every run as soon as it finishes (not in submission order), so a kill loses only runs in flight
        for fut in as_completed([pool.submit(_job, t) for t in tasks]):
            rec = fut.result()
            records.append(rec)
            print(f"{len(records)}/{len(tasks)} {rec['system']} {rec['noise']} {rec['n']} exact={rec['dedi']['exact']} "
                  f"{rec['seconds']:.0f}s {rec.get('dedi_error', '')}", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"data": args.data, "grid": args.grid, "index_file": args.index_file,
                                       "records": records}, default=float))
    print(f"saved {out} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
