"""Where the compute goes: count the expensive operations of our model (v1.2) and of DEDi on the same runs.

Neither method's code is modified; functions are wrapped from here and the wrappers only count.

  ours:  python benchmarks/cost_anatomy.py ours --order results_h2h/order.json --out results_anatomy/ours.jsonl
  DEDi:  python benchmarks/cost_anatomy.py dedi --ext ~/eqdisc_dedi --run hard_47 --device gpu \
             --order results_h2h/order.json --out results_anatomy/dedi.jsonl

Ours counts: joint LM fits (joint_discover.fit_joint / joint_colored.fit_joint), LM linear solves (_schur_solve,
one banded-Schur solve per LM iteration attempt), profiled-quadratic evaluations, and candidate subsets scored by
the exhaustive quadratic screen (flow_discovery._top_subsets: sum_k C(20, k) per call).
DEDi counts, per (lambda, R) pair: rounds of its backward-elimination loop (_FindParsimoniousModel's while loop,
one entry of track_info_criteria each), full joint LM solves (LM_Minimize calls; two per round, run in two
threads), and sparse Cholesky factorisations (one per LM iteration attempt, incl. damping retries).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

LAMBDAS = [10.0**i for i in range(-3, 4)]
RS = [10.0**j for j in range(-4, 1)]


def counting(counter, key, fn, extra=None, lock=None):
    def wrap(*a, **k):
        if lock:
            with lock:
                counter[key] += 1
        else:
            counter[key] += 1
        if extra:
            extra(counter, a, k)
        return fn(*a, **k)
    return wrap


def prepare_vectorized_lambda_fixed(X, terms, return_param_count=False):
    """DEDi.ODE.OdeLib.prepare_vectorized_lambda with the x_i substitutions done from i = N down to 1."""
    import re
    import jax.numpy as jnp
    N = X.shape[1]
    out = []
    for t_ in (t.replace("^", "**") for t in terms):
        if t_ in ("1", "c"):
            out.append("jnp.ones((x.shape[0], 1))")
            continue
        for i in range(N, 0, -1):
            t_ = t_.replace(f"x_{i}", f"x[:, {i - 1}:{i}]")
        for p_str in sorted(set(re.findall(r"p\d+", t_)), key=lambda s: int(s[1:])):
            t_ = t_.replace(p_str, f"p[{int(p_str[1:]) - 1}]")
        out.append(t_)
    env = {"jnp": jnp, "sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp, "tanh": jnp.tanh, "sqrt": jnp.sqrt,
           "log": jnp.log}
    return eval(f"lambda x, p, t: jnp.concatenate([{', '.join(out)}], axis=1)", env)


def pairs(spec):
    """'full' (the paper's 35-pair grid), 'row' (lambda = 1e-3 row, 5 pairs) or 'single' ((1e-3, 1))."""
    return {"full": [(l, r) for l in LAMBDAS for r in RS], "row": [(1e-3, r) for r in RS],
            "single": [(1e-3, 1.0)]}[spec]


def load_run(args, order):
    """(x, dt, system, exponent rows of the library, true coefficients, key) for --run or --data/--index."""
    if args.run:
        run = next(r for r in order if f"{r['cond']}_{r['index']}" == args.run)
        path, i = run["data"], int(run["index"])
    else:
        path, i = args.data, int(args.index)
    d = np.load(path, allow_pickle=True)
    x, dt, system = np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i])
    key = (args.run or f"{Path(path).stem}_{i}") + ("" if args.lib == "poly3" else f"_{args.lib}")
    if args.lib == "nd":
        from eqdisc.nd_systems import system as nd_system
        plib, xi = nd_system(system)
        return x, dt, system, plib.E, np.asarray(xi, float), key
    from eqdisc.core import SYSTEMS, EXP_ARR  # EQ_DEG=4 in the environment gives the degree-4 library
    return x, dt, system, EXP_ARR, SYSTEMS[system], key


def run_ours(args, order):
    from eqdisc import final_model as fm
    from eqdisc import joint_discover as jd, joint_colored as jc, joint_squeeze as js
    from eqdisc.core import SYSTEMS, metrics
    cnt = Counter()

    def subsets(c, a, k):
        A, kmax = a[0], a[2] if len(a) > 2 else k.get("kmax")
        c["subsets_scored"] += sum(comb(A.shape[1], j) for j in range(kmax + 1))

    jd.fit_joint = counting(cnt, "lm_fits", jd.fit_joint)
    jc.fit_joint = counting(cnt, "lm_fits", jc.fit_joint)
    jd._schur_solve = counting(cnt, "lm_solves", jd._schur_solve)
    jc._schur_solve = counting(cnt, "lm_solves", jc._schur_solve)
    jd.profiled_quadratic = counting(cnt, "profiled_quadratics", jd.profiled_quadratic)
    jc.profiled_quadratic = counting(cnt, "profiled_quadratics", jc.profiled_quadratic)
    from eqdisc import joint_nd as jn
    jn.solveh_banded = counting(cnt, "lm_solves", jn.solveh_banded)  # D > 3: banded solves (LM steps + profiling)
    from eqdisc import flow_discovery as fd
    for mod in (jd, js, jc, fd):  # each module holds its own reference to the exhaustive screen
        if hasattr(mod, "_top_subsets"):
            mod._top_subsets = counting(cnt, "screen_calls", mod._top_subsets, extra=subsets)
    if args.runs:  # scaling runs: [[archive, index], ...] with --lib
        items = [argparse.Namespace(run=None, data=p, index=i, lib=args.lib)
                 for p, i in json.loads(Path(args.runs).read_text())]
    else:
        items = [argparse.Namespace(run=f"{r['cond']}_{r['index']}", data=None, index=None, lib="poly3")
                 for r in order]
    library = {"poly3": "poly3", "poly4": "poly4", "nd": "auto"}[args.lib]
    with open(args.out, "a") as out:
        for item in items:
            x, dt, system, _, truth, key = load_run(item, order)
            cnt.clear()
            t = time.perf_counter()
            res = fm.discover(x, dt, library=library, version="1.2", confidence=False)
            rec = {"method": "ours", "key": key, "system": system, "D": x.shape[1], "n_terms": res.xi.shape[0],
                   "seconds": time.perf_counter() - t, **cnt, "noise_model": res.info.get("noise_model"),
                   **metrics(res.xi, truth)}
            out.write(json.dumps(rec, default=float) + "\n")
            out.flush()
            print(rec, flush=True)


def run_dedi(args, order):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path[:0] = [str(Path(args.ext) / "shim"), str(Path(args.ext) / "ext" / "DEDi")]
    import sksparse.cholmod as chol
    import DEDi.DEDi as dd
    from DEDi import ode_find
    from eqdisc.core import metrics
    cnt, lock = Counter(), threading.Lock()
    chol._Factor.cholesky_inplace = counting(cnt, "factorisations", chol._Factor.cholesky_inplace, lock=lock)
    dd.HeadDiffEqDiscovery.LM_Minimize = counting(cnt, "lm_solves", dd.HeadDiffEqDiscovery.LM_Minimize, lock=lock)

    def term(e):
        parts = [f"x_{i + 1}" + (f"**{p}" if p > 1 else "") for i, p in enumerate(e) if p]
        return "*".join(parts) if parts else "1"

    x, dt, system, exps, truth, key = load_run(args, order)
    lib = [term(e) for e in exps]
    if x.shape[1] >= 10:
        # DEDi's prepare_vectorized_lambda substitutes x_1 .. x_N in ascending order, so 'x_10' is consumed by the
        # 'x_1' replacement (SyntaxError for D >= 10). Same function with the substitutions longest-name first; only
        # used for D >= 10, so every D < 10 run uses the authors' code unchanged.
        import DEDi.ODE.OdeFind as odefind
        odefind.prepare_vectorized_lambda = prepare_vectorized_lambda_fixed
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        vm = np.ones_like(x, dtype=bool)
        vm[2::3] = False
        model = ode_find(x, lib, [0.0, dt * (len(x) - 1)], validation_mask=vm, cpu_or_gpu=args.device)
    setup = time.perf_counter() - t
    with open(args.out, "a") as out:
        for lam, r in pairs(args.pairs):
            cnt.clear()
            t = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                model.FindBestModel([lam], [r])
            xi = np.array(model.best_model()[0], dtype=float)
            rec = {"method": "dedi", "key": key, "system": system, "D": x.shape[1], "n_terms": len(lib),
                   "lam": lam, "R": r, "device": args.device, "seconds": time.perf_counter() - t,
                   "setup_seconds": setup, **cnt, **metrics(xi, truth)}
            out.write(json.dumps(rec, default=float) + "\n")
            out.flush()
            print(f"{key} lam={lam:g} R={r:g} {rec['seconds']:.0f}s lm_solves={cnt['lm_solves']} "
                  f"factorisations={cnt['factorisations']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["ours", "dedi"])
    ap.add_argument("--order", default="results_h2h/order.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext")
    ap.add_argument("--run")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--data")
    ap.add_argument("--index")
    ap.add_argument("--lib", default="poly3", choices=["poly3", "poly4", "nd"])
    ap.add_argument("--pairs", default="full", choices=["full", "row", "single"])
    ap.add_argument("--runs", help="ours: json list of [data, index] to run instead of --order")
    args = ap.parse_args()
    order = json.loads(Path(args.order).read_text())
    (run_ours if args.method == "ours" else run_dedi)(args, order)


if __name__ == "__main__":
    main()
