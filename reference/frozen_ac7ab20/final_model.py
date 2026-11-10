"""Final merged model: one entry point over the profiled-joint-flow variants.

    from final_model import discover
    res = discover(y, dt)                       # 3-D, degree-3 library, noise model chosen automatically
    res = discover(y, dt, library="poly5")      # degree-5 library (56 terms per equation)
    res = discover(y, dt, library="trig3")      # degree-3 polynomials + sin/cos of each state
    res = discover(Y10, dt)                     # D != 3: degree-2 library in D variables (joint_nd)
    print(res.equations()); res.ambiguous; res.confidence

Nothing is re-implemented; the existing frozen modules are imported and combined:
  joint_discover (config C)      fit, proposals, selection on the 20-term library
  joint_biglib                   beam-search proposals for degree-4/5 and the trig library
  joint_colored                  AR(1) observation-noise likelihood + penalty inflation lam(a)
  joint_nd                       D-dimensional version (degree-2 library)
  joint_squeeze                  the degree-weighted complexity prior (degw), implemented here through the
                                 per-term weight it defines: cost of a term = log(3n) * max(0.1, 1 + degw*(deg - c))
  confidence                     per-term inclusion probabilities from the candidate pool (+ isotonic map)

The complexity prior is applied to every variant the same way: the fitted-model classes' `k` (number of
terms, used only in the selection score's penalty phi*k*log(3n), times lam for AR(1)) is replaced by the
weighted count sum_terms w(term). For the 20-term library with c = 1.5 this is exactly joint_squeeze's
degw prior; with degw = 0 every variant is bit-for-bit its frozen original.

Defaults (DEFAULTS below) were chosen on dev archives only; see overnight2/F_final.md.
Truth is never used.
"""
import contextlib
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---- frozen defaults (set from the dev crossed experiments; see overnight2/F_final.md) ----
# dev evidence (paired W/L, two-sided sign test):
#   poly3 white      prior vs off 8/1 (p 0.04, 800 runs; = joint_squeeze)
#   coloured rho .8  AR+prior vs AR 9/0 (p 0.004), vs C+prior 36/8 (p 3e-5); lam exponent 1/4, 1/2 no better
#   rho .3/.55       AR adds nothing (AR+prior vs C+prior 9/11)  -> threshold 0.6 (gap in a_hat: .58 | .67)
#   poly4 / poly5    prior c=1.5 vs off 9/0 (p .004) / 9/1 (p .02); c = library mean degree hurts (deg5 6/18)
#   trig3            prior vs off 8/1 (p .04, Thomas + polynomial systems)
DEFAULTS = {
    "degw": 0.2,            # prior strength (joint_squeeze, frozen)
    "centre": {             # prior centre c per library: number, or "mean" (term-mean degree), "mid" (max deg / 2)
        "poly3": 1.5, "poly4": 1.5, "poly5": 1.5, "trig3": 1.5, "nd": 1.5,
    },
    "prior_on": {"poly3": True, "poly4": True, "poly5": True, "trig3": True, "nd": True},
    "prior_on_ar1": True,   # prior also used together with the AR(1) noise model
    "ar_threshold": 0.6,    # noise="auto": AR(1) model iff mean estimated lag-1 coefficient a > threshold
    "pen_exp": 1 / 3,       # AR(1) penalty inflation lam(a) = mean ((1+a)/(1-a))^pen_exp
    "trig_weight": 1.0,     # prior weight of each sin/cos term (neutral)
}
FLAG_GAP = 2.0              # ambiguity flag: best and runner-up scores within 2 (times lam for AR(1))
CAL_FILE = HERE / "results_final" / "confidence_isotonic.json"
POLY_DEG = {"poly3": 3, "poly4": 4, "poly5": 5}


@dataclass
class Result:
    xi: np.ndarray                      # (n_terms, D) coefficients, original units
    names: list                         # term names
    ambiguous: bool                     # best/runner-up score gap below the flag threshold
    gap: float
    confidence: dict = field(default_factory=dict)   # {(term, eq): p}; calibrated if info["confidence_calibrated"]
    info: dict = field(default_factory=dict)

    def equations(self, digits=4):
        D = self.xi.shape[1]
        var = list("xyz") if D == 3 else [f"x{i}" for i in range(D)]
        out = []
        for j in range(D):
            terms = [f"{self.xi[m, j]:+.{digits}g}" + ("" if self.names[m] == "1" else f"*{self.names[m]}")
                     for m in np.flatnonzero(self.xi[:, j])]
            out.append(f"d{var[j]}/dt = " + (" ".join(terms) if terms else "0"))
        return "\n".join(out)

    def sure_terms(self, hi=0.9):
        return [k for k, p in self.confidence.items() if p >= hi]


# ------------------------------------------------------------------------------------------------
# prior: weighted term count replacing `k` in the selection penalty
# ------------------------------------------------------------------------------------------------
_W = [None]  # current per-term weights (n_terms,), or None for plain k


def _weighted_k(self):
    w = _W[0]
    if w is None:
        return int(self.support.sum())
    return float((self.support * w[:, None]).sum())


@contextlib.contextmanager
def _prior(w):
    """Patch the fit classes of joint_discover / joint_colored / joint_nd so that .k is the weighted count."""
    import joint_discover as jd
    import joint_colored as jc
    import joint_nd as jn
    old = [(jd, "JointFit", jd.JointFit), (jc, "JointFit", jc.JointFit), (jn, "Fit", jn.Fit)]
    WJ = type("WJointFit", (jd.JointFit,), {"k": property(_weighted_k)})
    WN = type("WFit", (jn.Fit,), {"k": property(_weighted_k)})
    _W[0] = None if w is None else np.asarray(w, float)
    jd.JointFit, jc.JointFit, jn.Fit = WJ, WJ, WN
    try:
        yield
    finally:
        for m, k, v in old:
            setattr(m, k, v)
        _W[0] = None


def prior_weights(degrees, degw, centre, extra=None):
    """w = max(0.1, 1 + degw*(deg - c)); `centre` number, "mean" or "mid"; extra = weights appended (trig)."""
    d = np.asarray(degrees, float)
    if centre == "mean":
        c = d.mean()
    elif centre == "mid":
        c = d.max() / 2
    else:
        c = float(centre)
    w = np.maximum(0.1, 1 + degw * (d - c))
    if extra is not None:
        w = np.concatenate([w, np.asarray(extra, float)])
    return w, c


# ------------------------------------------------------------------------------------------------
# library contexts (big polynomial libraries need EQ_DEG set before `core` is imported)
# ------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _big_search(width=60):
    """Beam-search proposals (joint_biglib.top_subsets) for joint_discover and joint_colored."""
    import joint_discover as jd
    import joint_colored as jc
    import joint_biglib as jb
    f = lambda a, b, kmax, keep=2: jb.top_subsets(a, b, kmax, keep, width=width)
    old = (jd._top_subsets, jc._top_subsets)
    jd._top_subsets = jc._top_subsets = f
    try:
        yield
    finally:
        jd._top_subsets, jc._top_subsets = old


@contextlib.contextmanager
def _trig_library(x):
    """joint_biglib.use_library (patches joint_discover + select_k) extended to joint_colored."""
    import joint_biglib as jb
    import joint_colored as jc
    L = jb.MixedLibrary()
    with jb.use_library(L, x) as Lb:
        saved = {k: getattr(jc, k) for k in ("NLIB", "library", "dlibrary", "xi_to_scaled", "xi_from_scaled")}
        jc.NLIB, jc.library, jc.dlibrary = Lb.M, Lb.library, Lb.dlibrary
        jc.xi_to_scaled, jc.xi_from_scaled = Lb.to_scaled, Lb.from_scaled
        try:
            with _big_search():
                yield Lb
        finally:
            for k, v in saved.items():
                setattr(jc, k, v)


def _core_deg():
    import core
    return core.DEG


def _spawn_call(deg, args, kwargs):
    """Run discover in a fresh process whose `core` is built at degree `deg` (EQ_DEG)."""
    old = os.environ.get("EQ_DEG")
    os.environ["EQ_DEG"] = str(deg)
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply(discover, args, kwargs)
    finally:
        if old is None:
            os.environ.pop("EQ_DEG", None)
        else:
            os.environ["EQ_DEG"] = old


# ------------------------------------------------------------------------------------------------
# starts (proposal seeds) and the AR(1) pre-estimate
# ------------------------------------------------------------------------------------------------
def default_starts(x, dt, library="poly3"):
    """Seeds for the joint search, exactly as in the frozen benches (truth never used).
    poly3: weak chi2s, the hybrid answer and the top-3 flow alternatives (squeeze_bench / joint_bench);
    poly4/poly5/trig3: weak selectors chi2s, chi2, bic on the same library (biglib_bench).
    For trig3 call inside the library context."""
    import select_k
    if library == "poly3":
        from flow_discovery import discover as flow
        weak = select_k.discover(x, dt, criterion="chi2s")
        starts = [weak]
        try:
            res = flow(x, dt, extra_models=[weak])
            alts = res.alternatives
            own = [a for a in alts if not a["extra_only"]]
            w = [a for a in alts if a["extra"] == 0]
            gap = w[0]["selection_score"] - alts[0]["selection_score"] if w else None
            hyb = weak if gap is not None and gap <= 10 else res.coefficients
            starts += [hyb] + [np.array(a["coefficients"]) for a in own[:3]]
        except Exception:
            pass
        return starts
    weak = select_k.discover(x, dt, criterion=["chi2s", "chi2", "bic"])
    return [weak["chi2s"], weak["chi2"], weak["bic"]]


def estimate_ar(x, dt, starts, rho=0.03, huber=2.0, max_iter=40):
    """Lag-1 AR coefficient of the observation errors, per component: the first self-calibration step of
    joint_colored (best start fitted by config C's white-noise joint fit, a from its sample residuals).
    Uses whatever library joint_discover currently points at."""
    import joint_discover as jd
    import joint_colored as jc
    from core import estimate_sigma
    x = np.asarray(x, float)
    n, D = x.shape
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    obs_train = np.ones(n, bool)
    obs_train[2::3] = False
    s_ = jd.auto_substeps(yo, dt)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    y = np.zeros((nf, D))
    y[oi] = yo
    train = np.zeros(nf, bool)
    train[oi[obs_train]] = True
    valmask = np.zeros(nf, bool)
    valmask[oi[~obs_train]] = True
    u0o = jd._initial_states(yo, obs_train)
    tf = np.arange(nf) / s_
    u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(D)])
    best, bs = None, np.inf
    for xi in starts:
        sup = np.asarray(xi) != 0
        if not sup.any():
            continue
        f = jd.fit_joint(y, dt / s_, jd.xi_to_scaled(np.asarray(xi, float), scale), sup, sig, rho, train, u0,
                         max_iter=max_iter, valmask=valmask, c=huber)
        s = f.data_chi2 + f.model_chi2 + f.val_chi2 + int(sup.sum()) * np.log(n * D)
        if np.isfinite(s) and s < bs:
            best, bs = f, s
    if best is None:
        return np.zeros(D)
    return jc._ar1(yo - best.u[oi])


# ------------------------------------------------------------------------------------------------
# candidate pool -> gap, per-term confidence
# ------------------------------------------------------------------------------------------------
_CAL = [None]


def _calibration():
    if _CAL[0] is None:
        import json
        if CAL_FILE.exists():
            d = json.loads(CAL_FILE.read_text())
            _CAL[0] = (np.array(d["knots"]), np.array(d["values"]))
        else:
            _CAL[0] = False
    return _CAL[0] or None


def build_calibration(pool_files=("results_certify/dev_pool_part1.json", "results_certify/dev_pool_part2.json")):
    """Frozen per-term recalibration of track C (isotonic, fitted on dev pools, T = 1) -> CAL_FILE."""
    import json
    import confidence as cf
    recs = [r for f in pool_files for r in json.loads((HERE / f).read_text())["records"]]
    P, Y = cf.raw_pips(recs, 1.0)
    knots, vals = cf.fit_isotonic(P[P > 0], Y[P > 0])
    CAL_FILE.parent.mkdir(exist_ok=True)
    CAL_FILE.write_text(json.dumps({"source": list(pool_files), "T": 1.0, "knots": knots.tolist(),
                                    "values": vals.tolist()}))
    _CAL[0] = None
    return knots, vals


def _pool(cands, pen_mult, nobs):
    """[(score, flat support indices)] from joint_discover / joint_colored record=True candidates."""
    out = []
    for c in cands:
        s = c["data"] + c["model"] + c["val"] + pen_mult * c["k"] * np.log(nobs)
        out.append((float(s), np.flatnonzero(np.asarray(c["support"]).ravel())))
    return sorted(out, key=lambda v: v[0])


def _confidence(pool, nt, calibrate):
    import confidence as cf
    p = cf.pips([{"score": s, "support": sup} for s, sup in pool], 1.0, nt)
    cal = _calibration() if calibrate else None
    if cal is not None:
        p = np.where(p > 0, cf.apply_isotonic(p, cal), 0.0)
    return p, cal is not None


# ------------------------------------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------------------------------------
def discover(y, dt, library="auto", noise="auto", *, starts=None, prior=None, degw=None, centre=None,
             ar_threshold=None, pen_exp=None, confidence=True, return_candidates=False, **kw):
    """Discover a sparse polynomial (or polynomial + trig) ODE from one noisy trajectory y (n, D), step dt.

    library  "poly3" (default for D = 3; 20 terms/eq), "poly4", "poly5" (beam-search proposals), "trig3"
             (degree-3 polynomials + sin/cos of each state), "poly2" / "auto" for D != 3 (joint_nd).
    noise    "auto" (AR(1) model iff the estimated lag-1 coefficient of the errors exceeds
             DEFAULTS["ar_threshold"]), "white" (config C likelihood), "ar1" (joint_colored).
    starts   optional seeds (coefficient arrays, original units); default: the frozen benches' seeds.
    prior    True/False to force the complexity prior; degw / centre / ar_threshold / pen_exp override
             DEFAULTS (for experiments). Extra keyword arguments go to the underlying discover.
    Returns a Result (xi, names, ambiguous, gap, confidence {(term, eq): p}, info).
    """
    y = np.asarray(y, float)
    D = y.shape[1]
    if library == "auto":
        library = "poly3" if D == 3 else "poly2"
    if D != 3 or library == "poly2":
        return _discover_nd(y, dt, noise, starts, prior, degw, centre, confidence, return_candidates, kw)
    if library not in ("poly3", "poly4", "poly5", "trig3"):
        raise ValueError(f"unknown library {library!r}")
    need = POLY_DEG.get(library, 3)
    if _core_deg() != need:
        return _spawn_call(need, (y, dt, library, noise),
                           dict(starts=starts, prior=prior, degw=degw, centre=centre, ar_threshold=ar_threshold,
                                pen_exp=pen_exp, confidence=confidence, return_candidates=return_candidates, **kw))
    o = _options(library, prior, degw, centre, ar_threshold, pen_exp)
    ctx = _trig_library(y) if library == "trig3" else (_big_search() if library != "poly3" else
                                                       contextlib.nullcontext())
    with ctx as L:
        if starts is None:
            starts = default_starts(y, dt, library)
        return _discover3(y, dt, starts, library, noise, o, L, confidence, return_candidates, kw)


def _options(library, prior, degw, centre, ar_threshold, pen_exp):
    on = DEFAULTS["prior_on"][library] if prior is None else bool(prior)
    return {"prior": on, "prior_ar1": DEFAULTS["prior_on_ar1"] if prior is None else bool(prior),
            "degw": DEFAULTS["degw"] if degw is None else degw,
            "centre": DEFAULTS["centre"][library] if centre is None else centre,
            "thr": DEFAULTS["ar_threshold"] if ar_threshold is None else ar_threshold,
            "pen_exp": DEFAULTS["pen_exp"] if pen_exp is None else pen_exp}


def _discover3(y, dt, starts, library, noise, o, L, want_conf, want_cands, kw):
    import joint_discover as jd
    import joint_colored as jc
    from core import EXP_ARR, NAMES
    if library == "trig3":
        names, w_extra = L.names, [DEFAULTS["trig_weight"]] * (L.M - L.npoly)
        degs = L.exps.sum(1)
    else:
        names, w_extra, degs = list(NAMES), None, EXP_ARR.sum(1)
    n = len(y)
    info = {"library": library, "n_terms": len(names)}
    a_hat = None
    if noise == "auto":
        a_hat = estimate_ar(y, dt, starts)
        model = "ar1" if float(np.mean(a_hat)) > o["thr"] else "white"
        info["ar_estimate"] = a_hat.tolist()
    elif noise in ("white", "ar1"):
        model = noise
    else:
        raise ValueError(f"unknown noise {noise!r}")
    use_prior = o["prior"] and (model == "white" or o["prior_ar1"])
    w, c = prior_weights(degs, o["degw"], o["centre"], w_extra) if use_prior else (None, None)
    info.update(noise_model=model, prior=bool(use_prior), degw=o["degw"] if use_prior else 0.0, centre=c)
    with _prior(w):
        if model == "white":
            xi, inf = jd.discover(y, dt, starts, record=True, **kw)
            mult, lam = inf["phi"], 1.0
        else:
            xi, inf = jc.discover(y, dt, starts, record=True, pen_exp=o["pen_exp"], **kw)
            mult, lam = inf["phi"], inf["lam"]  # joint_colored's info["phi"] already includes lam
    pool = _pool(inf.pop("candidates"), mult, n * 3)
    gap = pool[1][0] - pool[0][0] if len(pool) > 1 else np.inf
    info.update({k: v for k, v in inf.items()})
    info["lam"] = lam
    res = Result(xi, names, bool(gap < FLAG_GAP * lam), float(gap), info=info)
    if want_conf:
        p, calibrated = _confidence(pool, len(names) * 3, calibrate=(library == "poly3"))
        res.confidence = {(names[i // 3], "xyz"[i % 3]): float(p[i]) for i in np.flatnonzero(p > 0)}
        info["confidence_calibrated"] = calibrated
    if want_cands:
        info["candidates"] = [(s, sup.tolist()) for s, sup in pool]
    return res


def _discover_nd(y, dt, noise, starts, prior, degw, centre, want_conf, want_cands, kw):
    import joint_nd as jn
    if noise == "ar1":
        raise NotImplementedError("the AR(1) noise model is only implemented for D = 3")
    D = y.shape[1]
    lib = jn.PolyLib(D, 2)
    on = DEFAULTS["prior_on"]["nd"] if prior is None else bool(prior)
    dw = DEFAULTS["degw"] if degw is None else degw
    cen = DEFAULTS["centre"]["nd"] if centre is None else centre
    w, c = prior_weights(lib.E.sum(1), dw, cen) if on else (None, None)
    if starts is None:
        starts = jn.weak_starts(y, dt, lib)
    with _prior(w):
        xi, inf = jn.discover(y, dt, starts, lib, record=True, **kw)
    cands = inf.pop("candidates")
    pool = sorted((float(cc["score"]), np.flatnonzero(np.asarray(cc["xi"]).ravel() != 0)) for cc in cands)
    pool = sorted(pool, key=lambda v: v[0])
    gap = pool[1][0] - pool[0][0] if len(pool) > 1 else np.inf
    info = {"library": "poly2", "D": D, "noise_model": "white", "prior": bool(on), "degw": dw if on else 0.0,
            "centre": c, **inf}
    res = Result(xi, lib.names, bool(gap < FLAG_GAP), float(gap), info=info)
    if want_conf:
        p, _ = _confidence(pool, lib.M * D, calibrate=False)
        res.confidence = {(lib.names[i // D], f"x{i % D}"): float(p[i]) for i in np.flatnonzero(p > 0)}
        info["confidence_calibrated"] = False
    if want_cands:
        info["candidates"] = [(s, sup.tolist()) for s, sup in pool]
    return res
