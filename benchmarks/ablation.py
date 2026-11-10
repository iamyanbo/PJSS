"""Component ablation of the frozen final model (eqdisc.discover, poly3 / D = 3 path).

    python benchmarks/ablation.py --set sealed  --every 5 --workers 8 --output results_ablation/sealed.json
    python benchmarks/ablation.py --set colored --every 5 --workers 8 --output results_ablation/colored.json
    python benchmarks/ablation.py --set hard40          --workers 8 --output results_ablation/hard40.json
    python benchmarks/ablation.py --set dysts   --every 7 --workers 8 --output results_ablation/dysts.json

Nothing in eqdisc/ is edited. Every variant calls final_model._discover3 (the code path behind the public
discover for D = 3) with the frozen DEFAULTS except for the one thing it switches off. Switches are either
existing keyword arguments of joint_discover / joint_colored (rounds, huber, substeps, selfcal) or temporary
monkey-patches of module attributes inside a context manager (restored afterwards), which never change the
defaults seen by any other caller:

  full            frozen final model (must equal results_final/test_*.json arm "final")
  -- leave one out of the full model --
  no_proposals    rounds=0: only the starts are refitted jointly; best by score (no profiled-quadratic search)
  greedy          profiled-quadratic proposals from forward stepwise selection (one subset per size) instead
                  of exhaustive best-subset ranking (jd/jc._top_subsets patched)
  single_start    only the weak chi2s answer as seed (no hybrid / flow seeds)
  no_selfcal      selfcal=False on the white-noise path: sigma = high-frequency difference estimate, phi = 1.
                  Runs routed to AR(1) keep that model's own (a, sigma) calibration, which it cannot run without.
  no_phi          phi = 1 (no autocorrelation inflation of the penalty); sigma self-calibration kept
  gaussian        huber=inf (least squares) in the joint fits and the AR(1) pre-estimate
  s2              substeps=2 (the v1 value) instead of the spectral auto rule, incl. the AR(1) pre-estimate
  no_prior        degw = 0 (prior=False)
  white           noise="white": no AR(1) auto-switch
  oracle          known noise: white archives -> initial sigma = eta * std(clean) (std(clean) estimated as
                  std(y)/sqrt(1+eta^2)); the self-calibration may still raise it. Coloured archive ->
                  noise="ar1" with oracle_noise=(0.8, sigma). Not a deployable variant.
  -- cumulative ladder (white-noise likelihood until the last rung) --
  L1_refit        weak start only, joint refit, rounds=0, s=2, fixed HF sigma, phi=1, Gaussian, no prior
  L2_proposals    + profiled-quadratic proposals (rounds=2)
  L3_starts       + hybrid / flow starts
  L4_selfcal      + sigma / phi self-calibration
  L5_substeps     + automatic sub-steps
  L6_huber        + Huber loss                (= config C)
  (L7 = white = C + prior;  L8 = full = + AR(1) switch)
Weak (starts[0]) and hybrid (starts[1]) are recorded from the seeds. Truth is used only for scoring.
"""
import argparse
import contextlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "full": {},
    "no_proposals": {"rounds": 0},
    "greedy": {"greedy": True},
    "single_start": {"starts": "weak"},
    "no_selfcal": {"selfcal": False},
    "no_phi": {"phi": False},
    "gaussian": {"huber": np.inf},
    "s2": {"substeps": 2},
    "no_prior": {"prior": False},
    "white": {"noise": "white"},
    "oracle": {"oracle": True},
    "no_schur": {"no_schur": True},
    "refit_only": {"starts": "weak", "rounds": 0},
}
_L = dict(starts="weak", rounds=0, substeps=2, selfcal=False, huber=np.inf, prior=False, noise="white")
LADDER = {
    "L1_refit": dict(_L),
    "L2_proposals": dict(_L, rounds=2),
    "L3_starts": dict(_L, rounds=2, starts="all"),
    "L4_selfcal": dict(_L, rounds=2, starts="all", selfcal=True),
    "L5_substeps": dict(_L, rounds=2, starts="all", selfcal=True, substeps="auto"),
    "L6_huber": dict(_L, rounds=2, starts="all", selfcal=True, substeps="auto", huber=2.0),
}
VARIANTS.update(LADDER)

SETS = {
    "sealed": dict(data="data/joint_sealed.npz", starts="results_squeeze/starts_joint_sealed.pkl", white=True),
    "colored": dict(data="data/sealed_colored.npz", starts=None, white=False),
    "hard40": dict(data="data/hybrid_confirm.npz", starts="results_squeeze/starts_hybrid_confirm.pkl",
                   white=True, index_file="results_final/hard40_idx.json"),
    "dysts": dict(data="data/dysts_test.npz", starts=None, white=True),
}


# ------------------------------------------------------------------------------------------------
def _greedy_top_subsets(a, b, kmax, keep=2, **_):
    """Forward stepwise path on the (whitened) least-squares problem (a, b): one subset per size."""
    norm = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    an = a / norm
    out = [(np.empty(0, dtype=int), float(b @ b))]
    sel = []
    for _k in range(min(kmax, a.shape[1])):
        best = None
        for j in range(a.shape[1]):
            if j in sel:
                continue
            idx = sel + [j]
            c = np.linalg.lstsq(an[:, idx], b, rcond=None)[0]
            rss = float(np.sum((b - an[:, idx] @ c) ** 2))
            if best is None or rss < best[0]:
                best = (rss, j)
        sel.append(best[1])
        out.append((np.array(sorted(sel)), best[0]))
    return out


def _fixed_state_quadratic(module):
    """profiled_quadratic without the Schur term: H = Acc, g = gc, so the states stay fixed (results_rev preregistration)."""
    from scipy.linalg import solve as _solve

    def quad(fit, *args, **kw):
        params = np.arange(module.NLIB * module.DIM)
        if module.__name__.endswith("joint_colored"):
            obs, used, s, dt, sig, rho = args[:6]
            c = args[6] if len(args) > 6 else kw.get("c", np.inf)
            _, _, Acc, _, gc, _ = module._normal(fit.u, fit.C, params, obs, used, s, dt, sig, rho, c)
        else:
            y, dt, sig, rho, train = args[:5]
            c = args[5] if len(args) > 5 else kw.get("c", np.inf)
            _, _, Acc, _, gc, _ = module._normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
        H = 0.5 * (Acc + Acc.T) + 1e-10 * np.trace(Acc) / len(Acc) * np.eye(len(Acc))
        cstar = fit.C.ravel() - _solve(H, gc, assume_a="pos")
        L = np.linalg.cholesky(H)
        return L.T, L.T @ cstar
    return quad


@contextlib.contextmanager
def _patched(pairs):
    """pairs: [(module, attr, value)]; restores the originals on exit."""
    old = [(m, k, getattr(m, k)) for m, k, _ in pairs]
    try:
        for m, k, v in pairs:
            setattr(m, k, v)
        yield
    finally:
        for m, k, v in old:
            setattr(m, k, v)


def run_variant(x, dt, starts, spec, eta, colored):
    from eqdisc import final_model as fm
    from eqdisc import joint_discover as jd
    from eqdisc import joint_colored as jc
    st = list(starts) if spec.get("starts", "all") == "all" else list(starts[:1])
    noise = spec.get("noise", "auto")
    kw, pairs = {}, []
    if "rounds" in spec:
        kw["rounds"] = spec["rounds"]
    if "huber" in spec:
        kw["huber"] = spec["huber"]
        pairs.append((fm, "estimate_ar", partial(fm.estimate_ar, huber=spec["huber"])))
    if spec.get("substeps", "auto") != "auto":
        s = int(spec["substeps"])
        kw["substeps"] = s
        f = lambda *a, **k: s
        pairs += [(jd, "auto_substeps", f), (jc, "auto_substeps", f)]
    if spec.get("greedy"):
        pairs += [(jd, "_top_subsets", _greedy_top_subsets), (jc, "_top_subsets", _greedy_top_subsets)]
    if spec.get("no_schur"):
        pairs += [(jd, "profiled_quadratic", _fixed_state_quadratic(jd)),
                  (jc, "profiled_quadratic", _fixed_state_quadratic(jc))]
    if spec.get("phi", True) is False:
        one = lambda r, kmax=50: 1.0
        pairs += [(jd, "_phi", one), (jc, "_phi", one)]
    selfcal = spec.get("selfcal", True)
    if spec.get("oracle"):
        if colored:
            noise = "ar1"
            sig = eta * np.sqrt(x.var(0) / (1 + eta ** 2))
            kw["oracle_noise"] = (0.8, sig)
        else:
            orc = lambda X: np.maximum(eta * np.sqrt(np.asarray(X).var(0) / (1 + eta ** 2)), 1e-6)
            pairs += [(jd, "estimate_sigma", orc), (jc, "estimate_sigma", orc)]
    if not selfcal:
        # joint_discover only; joint_colored has no selfcal switch (its a/sigma calibration is the model)
        jd_orig = jd.discover
        pairs.append((jd, "discover", partial(jd_orig, selfcal=False)))
    o = fm._options("poly3", spec.get("prior"), None, None, None, None)
    with _patched(pairs):
        return fm._discover3(x, dt, st, "poly3", noise, o, None, True, False, kw)


def _summ(res, truth, t):
    from eqdisc.core import metrics
    i = res.info
    return metrics(res.xi, truth) | {
        "seconds": time.perf_counter() - t, "ambiguous": res.ambiguous, "gap": res.gap,
        "noise_model": i.get("noise_model"), "phi": i.get("phi"), "substeps": i.get("substeps"),
        "sigma": i.get("sigma"), "fits": i.get("models_fitted")}


def _starts(x, dt):
    from eqdisc import final_model as fm
    return [np.asarray(s, float) for s in fm.default_starts(x, dt, "poly3")]


def _job(payload):
    index, x, dt, name, eta, truth, starts, variants, colored = payload
    from threadpoolctl import threadpool_limits
    from eqdisc.core import metrics
    from eqdisc.select_k import discover as weak_discover
    truth = np.asarray(truth)
    rec = {"index": index, "system": name, "noise": eta, "n": len(x), "variants": {}}
    with threadpool_limits(limits=1):
        t = time.perf_counter()
        w = weak_discover(x, dt, criterion="chi2s")
        rec["weak_seconds"] = time.perf_counter() - t
        if starts is None:
            t = time.perf_counter()
            starts = _starts(x, dt)
            rec["starts_seconds"] = time.perf_counter() - t
            rec["starts_cached"] = False
        else:
            rec["starts_cached"] = True
        rec["weak_matches_cache"] = bool(np.array_equal(np.asarray(w), np.asarray(starts[0])))
        rec["weak"] = metrics(np.asarray(starts[0]), truth)
        rec["hybrid"] = metrics(np.asarray(starts[1]), truth) if len(starts) > 1 else rec["weak"]
        rec["n_starts"] = len(starts)
        for v in variants:
            t = time.perf_counter()
            try:
                rec["variants"][v] = _summ(run_variant(x, dt, starts, VARIANTS[v], eta, colored), truth, t)
            except Exception as exc:
                rec["variants"][v] = {"exact": 0.0, "error": f"{type(exc).__name__}: {exc}",
                                      "seconds": time.perf_counter() - t}
    return rec


def load_jobs(set_name, every, offset, variants, limit=None):
    cfg = SETS[set_name]
    data = ROOT / cfg["data"]
    S = pickle.loads((ROOT / cfg["starts"]).read_bytes()) if cfg["starts"] else {}
    with np.load(data, allow_pickle=True) as d:
        X, sys_, noise, dts = d["X"], d["sys"], d["noise"], d["dt"]
        if set_name == "dysts":
            side = json.loads(data.with_suffix(".json").read_text())["systems"]
            idx = [i for i in range(len(X)) if X[i].shape[1] == 3 and i % every == offset]
            truth = {i: side[str(sys_[i])]["xi"] for i in idx}
        else:
            from eqdisc.core import SYSTEMS
            from eqdisc.heldout_systems import HELDOUT
            if "index_file" in cfg:
                idx = json.loads((ROOT / cfg["index_file"]).read_text())
            else:
                idx = list(range(len(X)))[offset::every]
            truth = {i: (SYSTEMS | HELDOUT)[str(sys_[i])] for i in idx}
        if limit:
            idx = idx[:: max(1, len(idx) // limit)][:limit]
        jobs = [(int(i), np.asarray(X[i], float), float(dts[i]), str(sys_[i]), float(noise[i]),
                 np.asarray(truth[i], float).tolist(), S[int(i)]["starts"] if int(i) in S else None,
                 tuple(variants), not cfg["white"]) for i in idx]
    jobs.sort(key=lambda j: -len(j[1]))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, help="timing runs: evenly spaced subset of this many runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true", help="skip indices already in <output>.partial.json")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    if os.environ.get("EQ_DEG", "3") != "3":
        raise SystemExit("ablation runs the poly3 path; unset EQ_DEG")
    jobs = load_jobs(args.set, args.every, args.offset, args.variants, args.limit)
    part = out.with_suffix(".partial.json")
    recs = []
    if args.resume and part.exists():
        recs = json.loads(part.read_text())["records"]
        done = {r["index"] for r in recs}
        jobs = [j for j in jobs if j[0] not in done]
    print(f"{args.set}: {len(jobs)} runs to do ({len(recs)} resumed), {len(args.variants)} variants", flush=True)
    t0 = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in jobs]):
            r = f.result()
            recs.append(r)
            V = r["variants"]
            errs = "".join(f" ERR[{a}]{v['error'][:80]}" for a, v in V.items() if "error" in v)
            print(f"[{len(recs)}] {r['index']} {r['system']} {r['noise']} {r['n']} full={V.get('full', {}).get('exact', '-')} "
                  f"sum={sum(v['exact'] for v in V.values()):.0f}/{len(V)} "
                  f"{sum(v.get('seconds', 0) for v in V.values()) + r.get('starts_seconds', 0):.0f}s"
                  f" t={time.perf_counter() - t0:.0f}{errs}", flush=True)
            if len(recs) % 5 == 0:
                part.write_text(json.dumps({"records": recs}, default=float))
    part.write_text(json.dumps({"records": recs}, default=float))
    out.write_text(json.dumps({"args": vars(args), "records": sorted(recs, key=lambda r: r["index"]),
                               "seconds": time.perf_counter() - t0}, default=float))
    print("saved", out, f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
