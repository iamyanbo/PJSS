"""Standard / official baseline implementations, paired run-for-run with eqdisc.discover.

    python benchmarks/baselines_bench.py run --method fd_stlsq --set dysts7 --workers 6 --output results_baselines/dysts7_fd_stlsq.json
    python benchmarks/baselines_bench.py run --method pysr --set dysts7 --every 7 --workers 6 --output ...
    tables: python benchmarks/baselines_report.py      (job files: results_baselines/*.sh; doc: docs/reports/baselines.md)

Sets (truth is never shown to a method; it is only used to score):
  dysts7  data/dysts_test.npz, every 7th run (880 runs; stride 7 is coprime with the archive's period-8
          noise x N cycle, so every (noise, N) cell gets 110 runs, all 77 systems appear)
  hard40  data/hybrid_confirm.npz at results_final/hard40_idx.json (Lorenz, Rossler, Chen, Rabinovich-F.,
          20% noise, N = 2000)
  dev     data/instances.npz every 5th run (160 runs; our dev archive) -- used ONLY to choose each baseline's
          smoothing / test-function hyperparameters, never for the reported numbers
`--every k` thins the chosen set further (positions ::k), used for the costly baselines.

Methods (library: monomials of degree <= 3 for D = 3, degree <= 2 for D >= 4, exactly as eqdisc.discover):
  {fd,weak}_{stlsq,sr3,esindy}   pysindy optimizers on pysindy features along a threshold path
       fd   = ps.SmoothedFiniteDifference (Savitzky-Golay, polyorder 3, window --window) + ps.PolynomialLibrary
              evaluated on the smoothed states (what ps.SINDy does with save_smooth=True)
       weak = ps.WeakPDELibrary(PolynomialLibrary, K = max(3 F, N / (h/2)), H_xt = h dt, p = 4), target from
              lib.convert_u_dot_integral (what ps.SINDy does with a weak library)
       stlsq  = ps.STLSQ(threshold=th)  (pysindy defaults otherwise: alpha 0.05, unbias)
       sr3    = ps.SR3(regularizer="L0", reg_weight_lam=th^2/2, relax_coeff_nu=1, max_iter=200)  (hard threshold th)
       esindy = ps.EnsembleOptimizer(ps.STLSQ(threshold=th), bagging=True, n_models=20)  (median aggregation)
     Preprocessing --scale: "raw" = original units (plain pysindy usage), th in geomspace(1e-4, 1e2, 55);
     "rms" = states / RMS and time rescaled so the derivative RMS is 1 (support-preserving), th in
     geomspace(1e-3, 10, 41).  One threshold shared by all equations.  Recorded per run: the support at every th
     (=> per-run oracle and set-tuned threshold) and two truth-free picks:
       aic: AICc summed over equations, from least-squares refits on each support (Mangan et al. 2017 style)
       cv : 5-fold contiguous regression-row CV of the (refit) residual, mean over equations of
            val-MSE / var(target); ties -> sparser. Weak rows are randomly ordered, overlapping
            temporal domains, not disjoint time blocks. SFD features are precomputed before splitting.
  weak_miosr  ps.MIOSR (Gurobi) on the weak features, one equation at a time, target_sparsity k = 1..8
     (normalize_columns, alpha 1e-6 as in benchmarks/bench.py); picks per equation by AICc and 5-fold CV;
     oracle = true support among the k-path for every equation.
  pysr        PySR (SymbolicRegression.jl) per equation on SFD-smoothed states/derivatives (same scaling),
     operators + - *, maxsize 35, fixed wall budget --budget s per equation, serial deterministic search,
     PySR's own model_selection="best".  Support: sympy.expand -> Poly monomials (exact zeros dropped);
     any monomial outside the library makes the run non-exact.  Also recorded: Pareto-front oracle.
  odeformer   pretrained ODEFormer (d'Ascoli et al. 2024), CPU, beam 50 / temperature 0.1, its own rescaling and
     candidate sorting, on 200 evenly spaced samples (--rows); top candidate mapped as for PySR; beam oracle.
     Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 and odeformer.pt in the working directory.
"""
import argparse
import json
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

THS = np.geomspace(1e-3, 10, 41)       # scale="rms": states / RMS, time rescaled (derivative RMS 1)
THS_RAW = np.geomspace(1e-4, 1e2, 55)  # scale="raw": original units, as plain pysindy usage


def ths_for(cfg):
    return THS_RAW if cfg.get("scale") == "raw" else THS
KMAX = 8
METHODS = [f"{f}_{o}" for f in ("fd", "weak") for o in ("stlsq", "sr3", "esindy")] + ["weak_miosr", "pysr", "odeformer"]


# ---------------------------------------------------------------- data
def _exps_from_names(names, D):
    """'x*y^2' (D=3, xyz) or 'x0 x1^2' / 'x0*x1^2' -> exponent tuple"""
    out = []
    for nm in names:
        e = [0] * D
        if nm.strip() != "1":
            for tok in re.split(r"[ *]+", nm.strip()):
                v, _, p = tok.partition("^")
                j = "xyz".index(v) if v in ("x", "y", "z") and D == 3 else int(v[1:])
                e[j] += int(p) if p else 1
        out.append(tuple(e))
    return out


def load_set(name, every=1):
    from eqdisc.core import SYSTEMS, NAMES
    runs = []
    if name == "dysts7":
        side = json.loads(Path("data/dysts_test.json").read_text())["systems"]
        with np.load("data/dysts_test.npz", allow_pickle=True) as d:
            for i in range(0, len(d["X"]), 7):
                s = side[str(d["sys"][i])]
                runs.append(dict(index=i, x=np.asarray(d["X"][i], float), dt=float(d["dt"][i]), system=str(d["sys"][i]),
                                 noise=float(d["noise"][i]), n=int(d["n"][i]), truth=s["xi"], names=s["names"]))
    elif name in ("hard40", "dev"):
        arc = "data/hybrid_confirm.npz" if name == "hard40" else "data/instances.npz"
        with np.load(arc, allow_pickle=True) as d:
            idx = json.loads(Path("results_final/hard40_idx.json").read_text()) if name == "hard40" \
                else list(range(0, len(d["X"]), 5))
            for i in idx:
                runs.append(dict(index=int(i), x=np.asarray(d["X"][i], float), dt=float(d["dt"][i]),
                                 system=str(d["sys"][i]), noise=float(d["noise"][i]), n=int(d["n"][i]),
                                 truth=SYSTEMS[str(d["sys"][i])].tolist(), names=list(NAMES)))
    else:
        raise ValueError(name)
    return runs[::every]


# ---------------------------------------------------------------- helpers
def scaling(x, dt):
    """u = x / s (unit RMS); tau = R t with R the RMS of du/dt, so d u / d tau has RMS ~ 1."""
    from scipy.signal import savgol_filter
    s = np.sqrt((x ** 2).mean(0))
    u = x / s
    w = min(21, len(u) - 1 - (len(u) % 2 == 0))
    R = float(np.sqrt((savgol_filter(u, w, 3, deriv=1, delta=dt, axis=0) ** 2).mean()))
    return u, s, R


def to_original(xi_u, exps, s, R):
    """coefficients for d u/d tau (rows = exps, cols = equations) -> original units"""
    E = np.array(exps)
    return xi_u * R / np.prod(s[None, :] ** E, axis=1)[:, None] * s[None, :]


def score(xi, truth):
    from eqdisc.core import metrics
    return metrics(np.asarray(xi), np.asarray(truth))


def ls_refit(A, b, supp):
    c = np.zeros(A.shape[1])
    if supp.any():
        c[supp] = np.linalg.lstsq(A[:, supp], b, rcond=None)[0]
    return c


def aicc(rss, n, k):
    return n * np.log(max(rss, 1e-300) / n) + 2 * k + (2 * k * (k + 1) / max(n - k - 1, 1))


def folds(n, nf=5):
    edges = np.linspace(0, n, nf + 1).round().astype(int)
    return [(np.r_[0:edges[f], edges[f + 1]:n], np.arange(edges[f], edges[f + 1])) for f in range(nf)]


def features(kind, u, dt_tau, deg, window, h):
    import pysindy as ps
    n, D = u.shape
    t = np.arange(n) * dt_tau
    if kind == "fd":
        sfd = ps.SmoothedFiniteDifference(smoother_kws={"window_length": window, "polyorder": 3})
        b = np.asarray(sfd(u, t))
        us = np.asarray(sfd.smoothed_x_)
        lib = ps.PolynomialLibrary(degree=deg)
        A = np.asarray(lib.fit_transform(us))
    else:
        F = len(ps.PolynomialLibrary(degree=deg).fit(u).get_feature_names())
        lib = ps.WeakPDELibrary(function_library=ps.PolynomialLibrary(degree=deg), spatiotemporal_grid=t,
                                K=max(3 * F, n // max(h // 2, 1)), H_xt=[h * dt_tau], p=4)
        lib.fit(u)
        A = np.asarray(lib.transform(u))
        b = np.asarray(lib.convert_u_dot_integral(u))
    return A, b, lib.get_feature_names()


def make_opt(name, th):
    import pysindy as ps
    if name == "stlsq":
        return ps.STLSQ(threshold=th)
    if name == "sr3":
        return ps.SR3(regularizer="L0", reg_weight_lam=th ** 2 / 2, relax_coeff_nu=1.0, max_iter=200)
    if name == "esindy":
        return ps.EnsembleOptimizer(ps.STLSQ(threshold=th), bagging=True, n_models=20)
    raise ValueError(name)


def fit_coef(opt_name, th, A, b):
    o = make_opt(opt_name, th)
    o.fit(A, b)
    return np.asarray(o.coef_).T  # (F, D)


# ---------------------------------------------------------------- pysindy threshold path
def run_path(run, method, cfg):
    # Import before seeding: first-time optional-library imports can consume RNG.
    import pysindy  # noqa: F401
    kind, opt_name = method.split("_", 1)
    x, dt = run["x"], run["dt"]
    D = x.shape[1]
    deg = 3 if D == 3 else 2
    if cfg.get("scale") == "raw" and opt_name != "miosr":
        u, s, R = x, np.ones(D), 1.0
    else:
        u, s, R = scaling(x, dt)
    ths = ths_for(cfg)
    # WeakPDELibrary samples domains during construction. Seed before features,
    # independently of worker/task order. Older stored results predate this fix.
    feature_seed = run["index"] % (2 ** 31)
    np.random.seed(feature_seed)
    A, b, fnames = features(kind, u, dt * R, deg, cfg["window"], cfg["h"])
    exps_ps = _exps_from_names(fnames, D)
    exps_t = _exps_from_names(run["names"], D)
    perm = [exps_ps.index(e) for e in exps_t]  # pysindy column for each truth row
    A = A[:, perm]
    truth = np.asarray(run["truth"])
    T = truth != 0
    N = len(A)
    np.random.seed(run["index"] % (2 ** 31))
    rec = dict(n_rows=N, feature_seed=feature_seed, feature_seed_policy="before_features_v1",
               cv_split="contiguous_regression_rows", preprocessing="full_trajectory")
    if opt_name == "miosr":
        return _miosr_path(run, A, b, s, R, exps_t, truth, rec)
    path = [fit_coef(opt_name, th, A, b) for th in ths]
    supps = [c != 0 for c in path]
    rec["exact_path"] = [int((S == T).all()) for S in supps]
    rec["eq_exact_path"] = [[int((S[:, j] == T[:, j]).all()) for j in range(D)] for S in supps]
    rec["nterms_path"] = [int(S.sum()) for S in supps]
    # AICc on LS refits
    aic = []
    for S in supps:
        tot = 0.0
        for j in range(D):
            c = ls_refit(A, b[:, j], S[:, j])
            tot += aicc(((A @ c - b[:, j]) ** 2).sum(), N, int(S[:, j].sum()))
        aic.append(tot)
    rec["aic_idx"] = int(np.argmin(aic))
    # Row-fold CV: weak rows are not ordered, separated temporal intervals.
    var = b.var(0) + 1e-300
    cv = np.zeros(len(ths))
    for tr, va in folds(N):
        for k, th in enumerate(ths):
            C = fit_coef(opt_name, th, A[tr], b[tr])
            err = 0.0
            for j in range(D):
                c = ls_refit(A[tr], b[tr, j], C[:, j] != 0)
                err += ((A[va] @ c - b[va, j]) ** 2).mean() / var[j]
            cv[k] += err / D
    rec["cv_idx"] = int(len(ths) - 1 - np.argmin(cv[::-1]))  # ties -> larger threshold (sparser)
    for pick in ("aic", "cv"):
        xi = to_original(path[rec[f"{pick}_idx"]], exps_t, s, R)
        rec[pick] = score(xi, truth)
        rec[f"{pick}_xi"] = xi.tolist()
    rec["oracle"] = float(max(rec["exact_path"]))
    rec["oracle_eq"] = float(all(any(p[j] for p in rec["eq_exact_path"]) for j in range(D)))
    return rec


def _miosr_path(run, A, b, s, R, exps_t, truth, rec):
    import pysindy as ps
    import gurobipy as gp
    if not getattr(gp, "_patched", False):  # one thread, silent (pysindy calls gp.Model() with the default env)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        orig = gp.Model
        gp.Model = lambda *a, **k: orig(*a, env=env, **k)
        gp._patched = True
    N, F = A.shape
    D = b.shape[1]
    T = truth != 0

    def fit(Ar, br, k):
        o = ps.MIOSR(target_sparsity=k, alpha=1e-6, normalize_columns=True)
        o.fit(Ar, br[:, None])
        return np.asarray(o.coef_).ravel()

    ks = list(range(1, min(KMAX, F) + 1))
    var = b.var(0) + 1e-300
    pick = {"aic": np.zeros((F, D)), "cv": np.zeros((F, D))}
    eq_exact = []
    for j in range(D):
        cs = [fit(A, b[:, j], k) for k in ks]
        eq_exact.append([int(((c != 0) == T[:, j]).all()) for c in cs])
        a = [aicc(((A @ ls_refit(A, b[:, j], c != 0) - b[:, j]) ** 2).sum(), N, int((c != 0).sum())) for c in cs]
        cv = np.zeros(len(ks))
        for tr, va in folds(N):
            for i, k in enumerate(ks):
                c = ls_refit(A[tr], b[tr, j], fit(A[tr], b[tr, j], k) != 0)
                cv[i] += ((A[va] @ c - b[va, j]) ** 2).mean() / var[j]
        pick["aic"][:, j] = ls_refit(A, b[:, j], cs[int(np.argmin(a))] != 0)
        pick["cv"][:, j] = ls_refit(A, b[:, j], cs[int(np.argmin(cv))] != 0)
    rec["eq_exact_k"] = eq_exact
    for p in ("aic", "cv"):
        xi = to_original(pick[p], exps_t, s, R)
        rec[p] = score(xi, truth)
        rec[f"{p}_xi"] = xi.tolist()
    rec["oracle"] = rec["oracle_eq"] = float(all(any(e) for e in eq_exact))
    return rec


# ---------------------------------------------------------------- PySR
def _poly_support(expr, syms, exps_t):
    """expanded sympy expression -> coefficient vector over exps_t, or None if a monomial is outside the library"""
    import sympy as sp
    expr = sp.expand(sp.sympify(expr))
    if expr == 0:
        return np.zeros(len(exps_t))
    try:
        P = sp.Poly(expr, *syms)
    except sp.PolynomialError:
        return None
    c = np.zeros(len(exps_t))
    for mon, co in P.terms():
        if tuple(mon) not in exps_t:
            return None
        co = float(co)
        if co != 0.0:
            c[exps_t.index(tuple(mon))] = co
    return c


def run_pysr(run, cfg):
    import sympy as sp
    from pysr import PySRRegressor
    import pysindy as ps
    x, dt = run["x"], run["dt"]
    D = x.shape[1]
    u, s, R = scaling(x, dt)
    t = np.arange(len(u)) * dt * R
    sfd = ps.SmoothedFiniteDifference(smoother_kws={"window_length": cfg["window"], "polyorder": 3})
    du = np.asarray(sfd(u, t))
    us = np.asarray(sfd.smoothed_x_)
    rows = np.linspace(0, len(u) - 1, min(len(u), cfg["rows"])).round().astype(int)
    names = ["x", "y", "z"] if D == 3 else [f"x{j}" for j in range(D)]
    syms = sp.symbols(names)
    exps_t = _exps_from_names(run["names"], D)
    truth = np.asarray(run["truth"])
    T = truth != 0
    xi_u = np.zeros((len(exps_t), D))
    ok, front_ok, exprs = True, [], []
    for j in range(D):
        m = PySRRegressor(binary_operators=["+", "-", "*"], unary_operators=[], maxsize=35,
                          niterations=100000, timeout_in_seconds=cfg["budget"], populations=15,
                          parallelism="serial", deterministic=True, random_state=run["index"] % 100000 + j,
                          progress=False, verbosity=0, model_selection="best", temp_equation_file=True,
                          delete_tempfiles=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(us[rows], du[rows, j], variable_names=names)
        best = m.sympy()
        exprs.append(str(best))
        c = _poly_support(best, syms, exps_t)
        if c is None:
            ok = False
        else:
            xi_u[:, j] = c
        hit = 0
        for e in m.equations_["sympy_format"]:
            cc = _poly_support(e, syms, exps_t)
            if cc is not None and ((cc != 0) == T[:, j]).all():
                hit = 1
                break
        front_ok.append(hit)
    xi = to_original(xi_u, exps_t, s, R)
    sc = score(xi, truth)
    if not ok:
        sc["exact"] = 0.0
    rec = dict(best=sc, best_xi=xi.tolist(), exprs=exprs, out_of_library=not ok, front_eq=front_ok,
               oracle=float(all(front_ok)))
    rec["best"]["out_of_library"] = not ok
    return rec


# ---------------------------------------------------------------- ODEFormer
_ODEF = None


def run_odeformer(run, cfg):
    global _ODEF
    import sympy as sp
    if _ODEF is None:
        import torch
        torch.set_num_threads(1)
        from odeformer.model import SymbolicTransformerRegressor
        _ODEF = SymbolicTransformerRegressor(from_pretrained=True)
        _ODEF.set_model_args({"beam_size": cfg["beam"], "beam_temperature": 0.1})
    x, dt = run["x"], run["dt"]
    D = x.shape[1]
    rows = np.linspace(0, len(x) - 1, min(len(x), cfg["rows"])).round().astype(int)  # model's 200-point regime
    t = np.arange(len(x))[rows] * dt
    np.random.seed(run["index"] % (2 ** 31))
    _ODEF.fit(t, x[rows])
    cands = [c for c in _ODEF.predictions[0] if c is not None]
    syms = sp.symbols([f"x_{j}" for j in range(D)])
    exps_t = _exps_from_names(run["names"], D)
    truth = np.asarray(run["truth"])
    T = truth != 0

    def parse(tree):
        strs = str(tree).split("|")
        xi = np.zeros((len(exps_t), D))
        ok = len(strs) == D
        for j, st in enumerate(strs[:D]):
            try:
                c = _poly_support(sp.sympify(st.strip()), syms, exps_t)
            except Exception:
                c = None
            if c is None:
                ok = False
            else:
                xi[:, j] = c
        return xi, ok, strs

    if not cands:
        return dict(best=dict(exact=0.0, f1=0.0, out_of_library=True), exprs=[], oracle=0.0, n_cands=0)
    xi, ok, strs = parse(cands[0])
    sc = score(xi, truth)
    if not ok:
        sc["exact"] = 0.0
    sc["out_of_library"] = not ok
    beam_hit = 0.0
    for c in cands:
        x2, ok2, _ = parse(c)
        if ok2 and ((x2 != 0) == T).all():
            beam_hit = 1.0
            break
    return dict(best=sc, best_xi=xi.tolist(), exprs=strs, oracle=beam_hit, n_cands=len(cands))


# ---------------------------------------------------------------- driver
def _job(payload):
    run, method, cfg = payload
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore")
    rec = {k: run[k] for k in ("index", "system", "noise", "n")}
    rec.update(D=int(run["x"].shape[1]), method=method)
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            if method == "pysr":
                rec.update(run_pysr(run, cfg))
            elif method == "odeformer":
                rec.update(run_odeformer(run, cfg))
            else:
                rec.update(run_path(run, method, cfg))
        rec["status"] = "ok"
    except Exception as exc:
        import traceback
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}", tb=traceback.format_exc()[-1500:])
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", choices=METHODS, required=True)
    r.add_argument("--set", choices=("dysts7", "hard40", "dev"), required=True)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--window", type=int, default=21, help="SFD Savitzky-Golay window (fd_*, pysr)")
    r.add_argument("--scale", choices=("rms", "raw"), default="rms", help="pysindy path methods: preprocessing")
    r.add_argument("--h", type=int, default=20, help="weak test-function half width in samples (weak_*)")
    r.add_argument("--budget", type=float, default=60, help="PySR seconds per equation")
    r.add_argument("--rows", type=int, default=None, help="rows used, even subsample (PySR 1000, ODEFormer 200)")
    r.add_argument("--beam", type=int, default=50, help="ODEFormer beam size")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--resume", action="store_true", help="reuse records from <output>.partial.json")
    r.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = load_set(args.set, args.every)[:args.limit]
    cfg = dict(window=args.window, h=args.h, scale=args.scale, budget=args.budget, beam=args.beam,
               rows=args.rows or (200 if args.method == "odeformer" else 1000))
    print(f"{len(runs)} runs, {args.method}, cfg {cfg}", flush=True)
    recs, t0 = [], time.perf_counter()
    part = out.with_suffix(".partial.json")
    if args.resume and part.exists():
        recs = json.loads(part.read_text())["records"]
        print(f"resuming: {len(recs)} records from {part}", flush=True)
    done = {q["index"] for q in recs}
    jobs = sorted([q for q in runs if q["index"] not in done], key=lambda q: -(q["x"].shape[1] ** 2 * len(q["x"])))
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, (q, args.method, cfg)) for q in jobs]):
            rr = f.result()
            recs.append(rr)
            main_exact = rr.get("aic", rr.get("best", {})).get("exact", "-")
            print(f"[{len(recs)}/{len(runs)}] {rr['index']} {rr['system']} D={rr['D']} {rr['noise']} {rr['n']} "
                  f"{rr['status']} exact {main_exact} cv {rr.get('cv', {}).get('exact', '-')} "
                  f"oracle {rr.get('oracle', '-')} {rr['seconds']:.1f}s {rr.get('error', '')}", flush=True)
            if rr["status"] != "ok":
                print(rr.get("tb", ""), flush=True)
            if len(recs) % 50 == 0:
                out.with_suffix(".partial.json").write_text(json.dumps({"records": recs}))
    out.write_text(json.dumps({"args": vars(args), "cfg": cfg, "ths": ths_for(cfg).tolist(),
                               "records": sorted(recs, key=lambda q: q["index"]),
                               "seconds": time.perf_counter() - t0}))
    if part.exists():
        part.unlink()
    ok = [q for q in recs if q["status"] == "ok"]
    print(f"saved {out}: {len(ok)}/{len(recs)} ok, wall {time.perf_counter() - t0:.0f}s, "
          f"median s {np.median([q['seconds'] for q in recs]):.2f}", flush=True)


if __name__ == "__main__":
    main()
