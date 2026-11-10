"""Variants of profiled joint flow (joint_discover.discover, config C) for the overnight squeeze track.

`discover_sq` reproduces joint_discover.discover exactly with default options and adds:
  rho_path      continuation: solve the start/proposal loop at each rho in turn, warm-starting states
                (the last rho is the selection rho; None = [rho])
  rounds, finalists, keep, kmax   wider search
  swaps         add/drop/swap local search on the profiled quadratic around the current support
  ebic          extended-BIC gamma: penalty k log(3n) + 2 gamma log C(M, k_j) per equation (0 = BIC)
  smoothers     extra state initialisations (savgol windows) for the starts
  degw          complexity prior: per-term cost log(3n)*(1 + degw*(degree-1.5)), floored at 0.1
  evidence      prior std (scaled units) -> Laplace-evidence score replacing BIC (None = BIC)
  truth         DIAGNOSTIC ONLY: fit the true support and classify the run's outcome (never used to select)
"""
import itertools
from math import comb, log

import numpy as np
from scipy.signal import savgol_filter

from . import joint_discover as jd
from .core import NLIB, DIM, EXP_ARR, estimate_sigma, xi_to_scaled, xi_from_scaled
from .flow_discovery import _top_subsets


def _init_states(yo, obs_train, w):
    u = yo.copy()
    idx = np.arange(len(yo))
    for i in range(DIM):
        u[~obs_train, i] = np.interp(idx[~obs_train], idx[obs_train], yo[obs_train, i])
    w = min(w, len(yo) - 1 - (len(yo) % 2 == 0))
    if w % 2 == 0:
        w -= 1
    return savgol_filter(u, w, 3, axis=0)


def discover_sq(x, dt, starts, *, rho=0.03, rho_path=None, rounds=2, kmax=5, keep=2, finalists=10, max_iter=40,
                substeps="auto", huber=2.0, final_rho=0.001, swaps=False, ebic=0.0, smoothers=(21,),
                evidence=None, degw=0.0,
                truth=None, record=False):
    x = np.asarray(x, float)
    n = len(x)
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    obs_train = np.ones(n, bool)
    obs_train[2::3] = False
    nobs = n * DIM
    s_ = jd.auto_substeps(yo, dt) if substeps == "auto" else int(substeps)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    y = np.zeros((nf, DIM))
    y[oi] = yo
    train = np.zeros(nf, bool)
    train[oi[obs_train]] = True
    valmask = np.zeros(nf, bool)
    valmask[oi[~obs_train]] = True
    dt_f = dt / s_
    tf = np.arange(nf) / s_
    u0s = []
    for w in smoothers:
        u0o = _init_states(yo, obs_train, w)
        u0s.append(np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(DIM)]))
    allobs = np.zeros(nf, bool)
    allobs[oi] = True
    path = [rho] if rho_path is None else list(rho_path)
    phi = 1.0
    stats = {"fits": 0}

    deg = EXP_ARR.sum(1)  # monomial degree of each library term

    def pen(support):
        if degw:
            w = np.maximum(0.1, 1 + degw * (deg - 1.5))  # complexity-weighted per-term cost (mean ~ BIC)
            p = float((support * w[:, None]).sum()) * np.log(nobs)
        else:
            p = support.sum() * np.log(nobs)
        if ebic:
            p += 2 * ebic * sum(log(comb(NLIB, int(k))) for k in support.sum(0))
        return p

    ev_cache = {}

    def ev_pen(f):
        # Laplace evidence Occam term: log det of the profiled (states-out) Hessian on the support, with a
        # N(0, evidence^2) prior per coefficient (scaled units); H/phi for correlated residuals
        key = tuple(np.flatnonzero(f.support.ravel()))
        if key not in ev_cache:
            A, _ = jd.profiled_quadratic(f, y, dt_f, sig, r_now, train, huber)
            idx = np.array(key)
            H = A[:, idx].T @ A[:, idx]
            sgn, ld = np.linalg.slogdet(H)
            c = f.C.ravel()[idx]
            ev_cache[key] = (ld if sgn > 0 else np.inf, len(idx), float(c @ c))
        ld, k, cc = ev_cache[key]
        return ld - k * np.log(phi) + 2 * k * np.log(evidence) + cc / evidence ** 2

    def score(f):
        if evidence:
            return f.data_chi2 + f.model_chi2 + f.val_chi2 + phi * ev_pen(f)
        return f.data_chi2 + f.model_chi2 + f.val_chi2 + phi * pen(f.support)

    fits = {}
    r_now = path[0]

    def add(support, C0, u_init):
        key = tuple(np.flatnonzero(support.ravel()))
        if key in fits or not support.any():
            return fits.get(key)
        f = jd.fit_joint(y, dt_f, C0, support, sig, r_now, train, u_init, max_iter=max_iter, valmask=valmask,
                         c=huber)
        stats["fits"] += 1
        if np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2):
            fits[key] = f
        return fits.get(key)

    def fit_starts(u_inits):
        for xi in starts:
            xi = np.asarray(xi, float)
            sup = xi != 0
            if not sup.any():
                continue
            key = tuple(np.flatnonzero(sup.ravel()))
            best = None
            for u_init in u_inits:
                f = jd.fit_joint(y, dt_f, xi_to_scaled(xi, scale), sup, sig, r_now, train, u_init,
                                 max_iter=max_iter, valmask=valmask, c=huber)
                stats["fits"] += 1
                if np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2) and (best is None or score(f) < score(best)):
                    best = f
            if best is not None and (key not in fits or score(best) < score(fits[key])):
                fits[key] = best
        if not fits:
            raise RuntimeError("no start model fitted")
        return min(fits.values(), key=score)

    def propose(current):
        A, b = jd.profiled_quadratic(current, y, dt_f, sig, r_now, train, huber)
        cur = current.support
        proposals, choices = {}, []
        for j in range(DIM):
            other = cur.copy()
            other[:, j] = False
            cols = np.arange(j, NLIB * DIM, DIM)
            aj, bj = A[:, cols].copy(), b.copy()
            if other.any():
                q, s, _ = np.linalg.svd(A[:, other.ravel()], full_matrices=False)
                q = q[:, s > s[0] * 1e-12]
                aj -= q @ (q.T @ aj)
                bj -= q @ (q.T @ bj)
            sups = [sup for sup, _ in _top_subsets(aj, bj, kmax, keep)]
            choices.append(sups)
            for sup in sups:
                mask = cur.copy()
                mask[:, j] = False
                mask[sup, j] = True
                proposals[tuple(np.flatnonzero(mask.ravel()))] = mask
        ranked_eq = []
        for j, sups in enumerate(choices):
            rk = []
            for sup in sups:
                mask = cur.copy()
                mask[:, j] = False
                mask[sup, j] = True
                idx = np.flatnonzero(mask.ravel())
                c = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
                rk.append((np.sum((b - A[:, idx] @ c) ** 2) + len(idx) * np.log(nobs), sup))
            ranked_eq.append([s for _, s in sorted(rk, key=lambda v: v[0])[:2]])
        for combo in itertools.product(*ranked_eq):
            mask = np.zeros_like(cur)
            for j, sup in enumerate(combo):
                mask[sup, j] = True
            proposals[tuple(np.flatnonzero(mask.ravel()))] = mask
        if swaps:
            on = np.flatnonzero(cur.ravel())
            off = np.flatnonzero(~cur.ravel())
            moves = [m for m in on] + [m for m in off]
            for m in moves:  # single drop / add
                mask = cur.ravel().copy()
                mask[m] = ~mask[m]
                proposals[tuple(np.flatnonzero(mask))] = mask.reshape(cur.shape)
            for a in on:  # swap within the same equation
                for b_ in off:
                    if a % DIM == b_ % DIM:
                        mask = cur.ravel().copy()
                        mask[a], mask[b_] = False, True
                        proposals[tuple(np.flatnonzero(mask))] = mask.reshape(cur.shape)
        ranked = []
        for key, mask in proposals.items():
            if key in fits or not mask.any(0).all():
                continue
            idx = np.array(key)
            c = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            ranked.append((np.sum((b - A[:, idx] @ c) ** 2) + len(idx) * np.log(nobs), key, mask, c))
        return sorted(ranked, key=lambda v: v[0])[:finalists]

    current = fit_starts(u0s)
    sig_hf = sig.copy()
    r = yo - current.u[oi]
    sig_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
    phi = jd._phi(r / np.maximum(sig_r, 1e-12))
    if np.any(sig_r > 1.1 * sig):
        sig = np.maximum(sig, sig_r)
        fits.clear()
        current = fit_starts(u0s)
    for pi, r_now_ in enumerate(path):
        if pi > 0:
            # continuation: refit everything kept so far at the next rho, warm-started
            r_now = r_now_
            old = sorted(fits.values(), key=score)[:finalists]
            fits.clear()
            for f in old:
                add(f.support, f.C, f.u)
            current = min(fits.values(), key=score)
        for _ in range(rounds):
            for _, key, mask, c in propose(current):
                C0 = np.zeros(NLIB * DIM)
                C0[list(key)] = c
                add(mask, C0.reshape(NLIB, DIM), current.u)
            best = min(fits.values(), key=score)
            if best is current:
                break
            current = best
    info = {"models_fitted": stats["fits"], "sigma": (sig * scale).tolist(), "phi": phi, "substeps": s_}
    ranked_fits = sorted(fits.values(), key=score)
    info["gap"] = score(ranked_fits[1]) - score(ranked_fits[0]) if len(ranked_fits) > 1 else np.inf
    if truth is not None:
        ts = xi_to_scaled(truth, scale) != 0
        key = tuple(np.flatnonzero(ts.ravel()))
        wscore = score(current)
        d = {"winner_score": wscore, "proposed": key in fits,
             "truth_score_search": score(fits[key]) if key in fits else None}
        # truth support from the method's own state estimate (no truth coefficients)
        C0 = jd._coeffs_from_states(current.u, dt_f, ts)
        f1 = jd.fit_joint(y, dt_f, C0, ts, sig, r_now, train, current.u, max_iter=4 * max_iter, valmask=valmask,
                          c=huber)
        # truth support from the true coefficients and the method's states (diagnostic only)
        f2 = jd.fit_joint(y, dt_f, xi_to_scaled(truth, scale), ts, sig, r_now, train, current.u,
                          max_iter=4 * max_iter, valmask=valmask, c=huber)
        d["truth_score_own_init"] = score(f1) if np.isfinite(score(f1)) else None
        d["truth_score_true_init"] = score(f2) if np.isfinite(score(f2)) else None
        info["diag"] = d
    final = jd.fit_joint(y, dt_f, current.C, current.support, sig, r_now, allobs, current.u, max_iter=max_iter,
                         valmask=valmask, c=huber)
    if final_rho is not None:
        stiff = jd.fit_joint(y, dt_f, final.C, current.support, sig, final_rho, allobs, final.u,
                             max_iter=2 * max_iter, valmask=valmask, c=huber)
        if np.isfinite(stiff.data_chi2 + stiff.model_chi2):
            final = stiff
    if record:
        info["candidates"] = [{"k": f.k, "score": score(f), "xi": xi_from_scaled(f.C, scale).tolist()}
                              for f in ranked_fits[:10]]
    return xi_from_scaled(final.C, scale), info


def classify(diag, tol=1e-6):
    """A_search: truth would win but was never proposed; C_local: proposed/fitted worse than it can be and a
    better truth fit would win; B_select: truth support scores worse than the winner even at its best fit."""
    w = diag["winner_score"]
    best_t = min(v for v in (diag["truth_score_search"], diag["truth_score_own_init"],
                             diag["truth_score_true_init"]) if v is not None)
    if best_t >= w - tol:
        return "B_select"
    if not diag["proposed"]:
        return "A_search"
    return "C_local"
