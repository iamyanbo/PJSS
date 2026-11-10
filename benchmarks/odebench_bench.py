"""ODEBench (d'Ascoli et al. 2024, ODEFormer paper; 63 ODEs, D = 1-4) with the FROZEN final model, no tuning.

    python benchmarks/odebench_bench.py classify --data data/odebench/strogatz_extended.json \
        --output results_odebench/classes.json
    python benchmarks/odebench_bench.py run --data data/odebench/strogatz_extended.json --fit-ic 0 \
        --workers 5 --output results_odebench/fit_ic0.json [--resume]

Protocol (pre-registered in results_odebench/PREREGISTRATION.md):
  * trajectories: the benchmark's own solutions (512 points, t in [0, 10], LSODA rtol 1e-5 / atol 1e-7), first
    parameter set, initial condition `fit_ic` is fitted, the other one is the generalization trajectory;
  * noise: ODEFormer's multiplicative model y = x (1 + xi), xi ~ N(0, sigma^2), sigma in {0, .01, ..., .05};
    `reps` realizations per (system, IC, sigma > 0), seed = 100000 + 1000 id + 100 fit_ic + 10 sigma_index + rep
    (rep 0 is the primary, ODEFormer-like single realization);
    no subsampling (the model needs a uniform grid);
  * model: eqdisc.discover(y, dt) with every default (library "auto": poly3 for D = 3, degree-2 joint_nd
    otherwise; noise "auto"). Nothing is tuned; the truth is used only for scoring;
  * scoring: (a) exact support recovery vs the true expansion for in-library systems; (b) ODEFormer's metrics:
    reconstruction R^2 (integrate the discovered ODE from the first NOISY observation over the same grid,
    compare with the clean trajectory) and generalization R^2 (integrate from the other clean initial condition,
    compare with its clean trajectory); variance-weighted R^2 as sklearn multioutput="variance_weighted";
    failed / diverged integrations score R^2 = 0 (ODEFormer's `r2` convention); accuracy = share with R^2 > 0.9.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
ODE_CFG = dict(method="LSODA", rtol=1e-5, atol=1e-7, first_step=1e-6, min_step=1e-10)


LIBFILE = Path(__file__).resolve().parent.parent / "results_odebench" / "library_exponents.json"


def exponents(D):
    """Exponent rows of the library the frozen default uses for dimension D (row order = Result.xi rows):
    eqdisc.core.EXP_ARR for D = 3, eqdisc.joint_nd.PolyLib(D, 2).E otherwise. Read from LIBFILE (dumped from
    the package, so `classify` runs without numba); `run` asserts it still matches the package."""
    L = json.loads(LIBFILE.read_text())[str(D)]
    return np.asarray(L["E"]), L["deg"]


def _check_exponents():
    from eqdisc.core import EXP_ARR
    from eqdisc.joint_nd import PolyLib
    assert np.array_equal(exponents(3)[0], np.asarray(EXP_ARR))
    for D in (1, 2, 4):
        assert np.array_equal(exponents(D)[0], PolyLib(D, 2).E)


# ------------------------------------------------------------------------------------------------ classify
def classify(entry):
    """in_library: every component is a polynomial in the states of total degree <= the default library's degree.
    Otherwise 'poly_high_degree' (polynomial, degree too high) or 'non_polynomial' (rational, exp, trig, ...)."""
    import sympy as sp
    D = entry["dim"]
    xs = sp.symbols([f"x_{i}" for i in range(D)])
    E, deg = exponents(D)
    row = {tuple(int(v) for v in e): i for i, e in enumerate(E)}
    xi = np.zeros((len(E), D))
    cls, maxdeg = "in_library", 0
    for j, s in enumerate(entry["substituted"][0]):
        ex = sp.expand(sp.sympify(s))
        try:
            P = sp.Poly(ex, *xs)
            ok = all(c.is_number for c in P.coeffs())
        except sp.PolynomialError:
            ok = False
        if not ok:
            cls = "non_polynomial"
            continue
        maxdeg = max(maxdeg, P.total_degree())
        for mon, c in P.terms():
            if mon in row:
                xi[row[mon], j] = float(c)
            elif cls == "in_library":
                cls = "poly_high_degree"
    return {"id": entry["id"], "dim": D, "class": cls, "lib_degree": deg, "max_degree": maxdeg,
            "eq": entry["substituted"][0], "description": entry["eq_description"],
            "xi_true": xi.tolist() if cls == "in_library" else None}


# ------------------------------------------------------------------------------------------------ scoring
def r2_vw(true, pred):
    """sklearn r2_score(..., multioutput='variance_weighted') = 1 - sum SSE / sum SST."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    sst = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ((true - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def integrate(xi, E, y0, t, max_nfev=200000, max_seconds=60.0):
    """Integrate dx/dt = monomials(x) @ xi on grid t. Returns (n, D) or None if it fails / blows up."""
    from scipy.integrate import solve_ivp
    xi = np.asarray(xi, float)
    E = np.asarray(E)
    bound = 1e6 * max(1.0, float(np.max(np.abs(y0))))
    cnt = [0]
    t0 = time.perf_counter()

    class _Stop(Exception):
        pass

    def f(_t, x):
        cnt[0] += 1
        if cnt[0] > max_nfev or time.perf_counter() - t0 > max_seconds:
            raise _Stop
        return np.prod(x[None, :] ** E, axis=1) @ xi

    def blow(_t, x):
        return bound - np.max(np.abs(x))
    blow.terminal = True
    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(f, (t[0], t[-1]), np.asarray(y0, float), t_eval=t, events=blow, **ODE_CFG)
    except _Stop:
        return None
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


def _job(payload):
    sid, D, cls, xi_true, t, Xfit, Xgen, sigma, seed, rep, patch = payload[:11]
    version = payload[11] if len(payload) > 11 else None
    if patch:  # diagnostic only, not the frozen model
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _nd_emptyfix
        _nd_emptyfix.apply()
    from threadpoolctl import threadpool_limits
    from eqdisc import discover
    from eqdisc.core import metrics
    t = np.asarray(t)
    dt = float(t[1] - t[0])
    rng = np.random.default_rng(seed)
    y = Xfit * (1 + sigma * rng.standard_normal(Xfit.shape))
    E, _ = exponents(D)
    rec = {"id": sid, "dim": D, "class": cls, "sigma": sigma, "seed": seed, "rep": rep, "version": version}
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            res = discover(y, dt) if version is None else discover(y, dt, version=version)
        xi = np.asarray(res.xi, float)
        rec.update(seconds=time.perf_counter() - t0, n_terms=int((xi != 0).sum()), xi=xi.tolist(),
                   equations=res.equations(), noise_model=res.info.get("noise_model"),
                   library=res.info.get("library"), ambiguous=bool(res.ambiguous),
                   floor=res.info.get("sigma_floor_info"))
    except Exception as exc:
        rec.update(seconds=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}", r2_rec=0.0, r2_gen=0.0,
                   rec_fail=True, gen_fail=True)
        if xi_true is not None:
            rec.update(exact=0.0)
        return rec
    if xi_true is not None:
        rec.update(metrics(xi, np.asarray(xi_true)))
    pr = integrate(xi, E, y[0], t)
    pg = integrate(xi, E, Xgen[0], t)
    rec["rec_fail"], rec["gen_fail"] = pr is None, pg is None
    rec["r2_rec"] = 0.0 if pr is None else r2_vw(Xfit, pr)
    rec["r2_gen"] = 0.0 if pg is None else r2_vw(Xgen, pg)
    # same quantities from the clean initial condition (sensitivity only)
    pc = integrate(xi, E, Xfit[0], t)
    rec["r2_rec_clean_y0"] = 0.0 if pc is None else r2_vw(Xfit, pc)
    rec["total_seconds"] = time.perf_counter() - t0
    return rec


def load(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["classify", "run"])
    ap.add_argument("--data", default="data/odebench/strogatz_extended.json")
    ap.add_argument("--fit-ic", type=int, default=0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--ids", type=int, nargs="+")
    ap.add_argument("--reps", type=int, default=1, help="noise realizations per sigma > 0 (rep 0 = primary)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--classes", default="results_odebench/classes.json")
    ap.add_argument("--patch-empty-support", action="store_true",
                    help="DIAGNOSTIC (not frozen): benchmarks/_nd_emptyfix.py workaround for the joint_nd crash")
    ap.add_argument("--dims", type=int, nargs="+", help="only systems of these dimensions")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--version", help="eqdisc.discover(version=...); default: the package default (precision audit)")
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    bench = load(a.data)
    if a.cmd == "classify":
        C = [classify(e) for e in bench]
        out.write_text(json.dumps(C, indent=1))
        from collections import Counter
        print(Counter((c["dim"], c["class"]) for c in C))
        return
    _check_exponents()
    C = {c["id"]: c for c in json.loads(Path(a.classes).read_text())}
    jobs = []
    for e in bench:
        if a.ids and e["id"] not in a.ids:
            continue
        if a.dims and e["dim"] not in a.dims:
            continue
        sols = e["solutions"][0]
        t = np.asarray(sols[a.fit_ic]["t"])
        Xf = np.asarray(sols[a.fit_ic]["y"], float).T
        Xg = np.asarray(sols[1 - a.fit_ic]["y"], float).T
        c = C[e["id"]]
        for s in a.sigmas:
            for rep in range(a.reps if s > 0 else 1):
                seed = 100000 + 1000 * e["id"] + 100 * a.fit_ic + 10 * SIGMAS.index(s) + rep
                jobs.append((e["id"], e["dim"], c["class"], c["xi_true"], t.tolist(), Xf, Xg, s, seed, rep,
                             a.patch_empty_support, a.version))
    part = out.with_suffix(".partial.json")
    recs = []
    if a.resume and part.exists():
        recs = json.loads(part.read_text())["records"]
        done = {(r["id"], r["sigma"], r["rep"]) for r in recs}
        jobs = [j for j in jobs if (j[0], j[7], j[9]) not in done]
    jobs.sort(key=lambda j: -j[1])  # D = 3/4 first (slowest)
    print(f"{len(jobs)} runs to do ({len(recs)} resumed)", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            print(f"[{len(recs)}] id={r['id']} D={r['dim']} {r['class'][:6]} s={r['sigma']} exact={r.get('exact', '-')} "
                  f"R2rec={r['r2_rec']:.3f} R2gen={r['r2_gen']:.3f} {r['seconds']:.0f}s t={time.perf_counter() - t0:.0f}"
                  + (f" ERR {r['error'][:100]}" if "error" in r else ""), flush=True)
            if len(recs) % 5 == 0:
                part.write_text(json.dumps({"records": recs}, default=float))
    part.write_text(json.dumps({"records": recs}, default=float))
    out.write_text(json.dumps({"args": vars(a), "records": sorted(recs, key=lambda r: (r["id"], r["sigma"], r["rep"])),
                               "seconds": time.perf_counter() - t0}, default=float))
    print("saved", out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
