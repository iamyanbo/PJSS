"""Profiled joint flow with an AR(1) observation-noise model (prewhitened data term).

Same method as joint_discover (config C), except the observation errors e_k = u(t_k) - y_k are
modelled per component as AR(1), e_k = a e_{k-1} + nu_k, with marginal std sigma:

* Data term = exact Gaussian likelihood of the used observations (train subset, or all in the final
  refit) under AR(1). AR(1) is Markov, so for used observations j_0 < j_1 < ... with gaps g:
      w_0 = e_{j_0} / sigma,   w_t = (e_{j_t} - a^g e_{j_{t-1}}) / (sigma sqrt(1 - a^{2g})).
  Every third sample is held out, so train gaps are 1 or 2. Each w couples two observation states, so
  the state block stays banded (upper bandwidth max(2 DIM - 1, 2 s DIM), s = sub-steps).
* Held-out score = exact conditional of each held-out error given ALL training errors, which by the
  Markov property is given its two (training) neighbours:
      mean a/(1+a^2) (e_{k-1} + e_{k+1}),  std sigma sqrt((1-a^2)/(1+a^2))
  (one-sided at the series end). Train likelihood + held-out conditionals = the full joint
  likelihood of all observations, so the score keeps its meaning; with a = 0 everything reduces
  exactly to joint_discover.
* Term penalty phi * lam * k log(3n) with lam = mean_i ((1+a_i)/(1-a_i))^(1/3): under the AR(1) data
  term alone, spurious terms that bend the latent trajectory toward slow noise still win (dev:
  oracle diagnostics), so the BIC penalty grows with the noise correlation time; lam = 1 when a = 0.
  The ambiguity flag threshold is 2 lam.
* Score (score_rule="conc", optional, not used): data + held-out misfit enter as the concentrated (profile)
  likelihood sum_i n_i log(S_i / n_i) over a free per-component noise scale, so that a few-% error in
  the calibrated sigma (common under correlated noise, where MAD is noisy) cannot buy spurious terms;
  default score_rule="chi2" keeps joint_discover's fixed-sigma chi^2 (conc did not help on dev).
* a and sigma are self-calibrated per component from the best start's residuals at the samples
  (lag-1 autocorrelation, clipped to [0, 0.97]; MAD for sigma), re-estimated once after the first
  refit. phi is computed from the whitened residuals.
Truth is never used.
"""
import itertools

import numpy as np
from scipy.linalg import solveh_banded

from core import NLIB, DIM, library, estimate_sigma, xi_to_scaled, xi_from_scaled
from select_k import dlibrary
from flow_discovery import _top_subsets
from joint_discover import JointFit, _hw, _hsum, _initial_states, auto_substeps, _phi

BW0 = 2 * DIM - 1


class Obs:
    """Observation structure on the latent grid."""

    def __init__(self, yo, oi, a, sig):
        self.yo, self.oi = yo, oi
        self.a = np.asarray(a, float)
        self.sig = np.asarray(sig, float)
        self.n = len(yo)

    def pairs(self, used):
        """Whitening coefficients for the used observations (bool over n).
        Returns cur, prev (obs indices; prev=-1 for the first), alpha (k, DIM), beta (k, DIM)."""
        idx = np.flatnonzero(used)
        prev = np.concatenate([[-1], idx[:-1]])
        g = np.where(prev >= 0, idx - prev, 0)[:, None]
        ag = np.where(g > 0, self.a[None, :] ** g, 0.0)
        alpha = 1.0 / (self.sig[None, :] * np.sqrt(np.maximum(1 - ag ** 2, 1e-12)))
        beta = ag * alpha
        return idx, prev, alpha, beta

    def bandwidth(self, used, s):
        idx = np.flatnonzero(used)
        gmax = int(np.max(np.diff(idx))) if len(idx) > 1 else 1
        return max(BW0, gmax * s * DIM) if np.any(self.a != 0) else BW0

    def white(self, u, used):
        cur, prev, al, be = self.pairs(used)
        e = u[self.oi] - self.yo
        pe = np.where(prev[:, None] >= 0, e[np.maximum(prev, 0)], 0.0)
        return al * e[cur] - be * pe

    def heldout(self, u, held):
        """Conditional residuals of held-out errors given neighbouring errors (sigma units of the conditional)."""
        e = u[self.oi] - self.yo
        k = np.flatnonzero(held)
        a, s = self.a[None, :], self.sig[None, :]
        left = e[np.maximum(k - 1, 0)]
        has_r = (k + 1 < self.n)[:, None]
        right = e[np.minimum(k + 1, self.n - 1)]
        two = (a / (1 + a ** 2)) * (left + right)
        sd_two = s * np.sqrt((1 - a ** 2) / (1 + a ** 2))
        one = a * left
        sd_one = s * np.sqrt(1 - a ** 2)
        mean = np.where(has_r, two, one)
        sd = np.where(has_r, sd_two, sd_one)
        return (e[k] - mean) / np.maximum(sd, 1e-12)


def _model_res(u, C, dt, sig, rho):
    m = 0.5 * (u[1:] + u[:-1])
    th = library(m)
    rm = (u[1:] - u[:-1] - dt * th @ C) / (rho * sig)
    return rm, m, th


def _normal(u, C, params, obs, used, s, dt, sig, rho, c=np.inf):
    n = len(u)
    bw = obs.bandwidth(used, s)
    rm, m, th = _model_res(u, C, dt, sig, rho)
    J = np.einsum("mi,kml->kil", C, dlibrary(m))
    eye = np.eye(DIM)
    w = 1.0 / (rho * sig)[None, :, None]
    B = np.concatenate([(-eye - 0.5 * dt * J) * w, (eye - 0.5 * dt * J) * w], axis=2)
    M = np.einsum("kap,kaq->kpq", B, B)
    ab = np.zeros((bw + 1, n * DIM))
    k3 = DIM * np.arange(n - 1)
    for p in range(2 * DIM):
        for q in range(p, 2 * DIM):
            np.add.at(ab[bw + p - q], k3 + q, M[:, p, q])
    gu = np.zeros(n * DIM)
    gb = np.einsum("kap,ka->kp", B, rm)
    for p in range(2 * DIM):
        np.add.at(gu, k3 + p, gb[:, p])
    # whitened data term
    cur, prev, al, be = obs.pairs(used)
    e = u[obs.oi] - obs.yo
    pe = np.where(prev[:, None] >= 0, e[np.maximum(prev, 0)], 0.0)
    rd = al * e[cur] - be * pe
    wd = _hw(rd, c)
    comp = np.arange(DIM)[None, :]
    fc = (obs.oi[cur][:, None] * DIM + comp)
    np.add.at(ab[bw], fc.ravel(), (wd * al ** 2).ravel())
    np.add.at(gu, fc.ravel(), (wd * al * rd).ravel())
    hp = prev >= 0
    if hp.any() and np.any(be[hp] != 0):
        fp = (obs.oi[prev[hp]][:, None] * DIM + comp)
        np.add.at(ab[bw], fp.ravel(), (wd[hp] * be[hp] ** 2).ravel())
        np.add.at(gu, fp.ravel(), (-wd[hp] * be[hp] * rd[hp]).ravel())
        off = fc[hp] - fp
        np.add.at(ab, (bw - off.ravel(), fc[hp].ravel()), (-wd[hp] * al[hp] * be[hp]).ravel())
    mi = params // DIM
    ei = params % DIM
    D = -dt * th[:, mi] / (rho * sig[ei])[None, :]
    P = len(params)
    Acc = np.zeros((P, P))
    for q in range(DIM):
        sl = np.flatnonzero(ei == q)
        Acc[np.ix_(sl, sl)] = D[:, sl].T @ D[:, sl]
    Auc = np.zeros((n * DIM, P))
    Bsel = B[:, ei, :] * D[:, :, None]
    for r in range(2 * DIM):
        Auc[k3 + r] += Bsel[:, :, r]
    gc = np.einsum("kp,kp->p", D, rm[:, ei])
    cost = _hsum(rd, c) + float((rm ** 2).sum())
    return ab, Auc, Acc, gu, gc, cost


def _schur_solve(ab, Auc, Acc, gu, gc, mu):
    ab = ab.copy()
    bw = ab.shape[0] - 1
    ab[bw] += mu * (1.0 + ab[bw])
    acc = Acc + mu * np.diag(1.0 + np.diag(Acc))
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    xg, xc = X[:, 0], X[:, 1:]
    S = acc - Auc.T @ xc
    dc = np.linalg.solve(S + 1e-12 * np.trace(S) / max(len(S), 1) * np.eye(len(S)), -(gc - Auc.T @ xg))
    return -xg - xc @ dc, dc


def _cost(u, C, obs, used, dt, sig, rho, c):
    rm, _, _ = _model_res(u, C, dt, sig, rho)
    rd = obs.white(u, used)
    return _hsum(rd, c), float((rm ** 2).sum())


def fit_joint(obs, used, held, s, dt, C0, support, sig, rho, u0, max_iter=40, tol=1e-7, c=np.inf):
    params = np.flatnonzero(support.ravel())
    u, C = u0.copy(), np.where(support, C0, 0.0)
    mu, converged, it = 1e-3, False, 0
    ab, Auc, Acc, gu, gc, cost = _normal(u, C, params, obs, used, s, dt, sig, rho, c)
    for it in range(1, max_iter + 1):
        try:
            du, dc = _schur_solve(ab, Auc, Acc, gu, gc, mu)
        except (np.linalg.LinAlgError, ValueError):
            mu *= 10
            continue
        un = u + du.reshape(u.shape)
        Cn = C.copy()
        Cn.ravel()[params] += dc
        new = sum(_cost(un, Cn, obs, used, dt, sig, rho, c))
        if np.isfinite(new) and new < cost:
            rel = (cost - new) / max(cost, 1e-300)
            u, C = un, Cn
            ab, Auc, Acc, gu, gc, cost = _normal(u, C, params, obs, used, s, dt, sig, rho, c)
            mu = max(mu / 3, 1e-9)
            if rel < tol:
                converged = True
                break
        else:
            mu *= 4
            if mu > 1e8:
                break
    d, mo = _cost(u, C, obs, used, dt, sig, rho, c)
    hv = obs.heldout(u, held)
    val = _hsum(hv, c)
    f = JointFit(u, C, support.copy(), d, mo, val, it, converged)
    rd = obs.white(u, used)
    f.dcomp = np.array([_hsum(rd[:, i], c) + _hsum(hv[:, i], c) for i in range(DIM)])
    f.ncomp = len(rd) + len(hv)
    return f


def profiled_quadratic(fit, obs, used, s, dt, sig, rho, c=np.inf):
    params = np.arange(NLIB * DIM)
    ab, Auc, Acc, gu, gc, _ = _normal(fit.u, fit.C, params, obs, used, s, dt, sig, rho, c)
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    H = Acc - Auc.T @ X[:, 1:]
    g = gc - Auc.T @ X[:, 0]
    H = 0.5 * (H + H.T) + 1e-10 * np.trace(H) / len(H) * np.eye(len(H))
    cstar = fit.C.ravel() - np.linalg.solve(H, g)
    L = np.linalg.cholesky(H)
    return L.T, L.T @ cstar


def _ar1(r, amax=0.97):
    out = []
    for i in range(r.shape[1]):
        e = r[:, i] - r[:, i].mean()
        out.append(np.clip((e[:-1] @ e[1:]) / max(e @ e, 1e-300), 0.0, amax))
    return np.array(out)


def discover(x, dt, starts, *, rho=0.03, rounds=2, kmax=5, keep=2, finalists=10, max_iter=40,
             substeps="auto", huber=2.0, ar=True, ar_iters=2, final_rho=0.001, record=False,
             oracle_noise=None, score_rule="chi2", pen_exp=1 / 3):
    """oracle_noise=(a, sigma in original units) fixes the noise model (diagnostics only)."""
    x = np.asarray(x, float)
    n = len(x)
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    train = np.ones(n, bool)
    train[2::3] = False
    held = ~train
    allobs = np.ones(n, bool)
    nobs = n * DIM
    s_ = auto_substeps(yo, dt) if substeps == "auto" else int(substeps)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    dt_f = dt / s_
    u0o = _initial_states(yo, train)
    tf = np.arange(nf) / s_
    u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(DIM)])
    obs = Obs(yo, oi, np.zeros(DIM), sig)
    phi = 1.0

    lam = 1.0

    def score(f):
        if score_rule == "conc":
            # profile likelihood over a free per-component noise scale: robust to sigma mis-calibration
            d = float(np.sum(f.ncomp * np.log(np.maximum(f.dcomp, 1e-300) / f.ncomp)))
        else:
            d = f.data_chi2 + f.val_chi2
        return d + f.model_chi2 + lam * phi * f.k * np.log(nobs)

    fits = {}

    def add(support, C0, u_init):
        key = tuple(np.flatnonzero(support.ravel()))
        if key in fits or not support.any():
            return fits.get(key)
        f = fit_joint(obs, train, held, s_, dt_f, C0, support, sig, rho, u_init, max_iter=max_iter, c=huber)
        if np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2):
            fits[key] = f
        return fits.get(key)

    def fit_starts():
        for xi in starts:
            xi = np.asarray(xi, float)
            sup = xi != 0
            if sup.any():
                add(sup, xi_to_scaled(xi, scale), u0)
        if not fits:
            raise RuntimeError("no start model fitted")
        return min(fits.values(), key=score)

    current = fit_starts()
    sig_hf = sig.copy()
    a = np.zeros(DIM)
    if oracle_noise is not None:
        a = np.broadcast_to(np.asarray(oracle_noise[0], float), (DIM,)).copy()
        sig = np.asarray(oracle_noise[1], float) / scale
        obs = Obs(yo, oi, a, sig)
        phi = _phi(obs.white(current.u, allobs))
        fits.clear()
        current = fit_starts()
    for _ in (range(ar_iters if ar else 1) if oracle_noise is None else []):
        r = yo - current.u[oi]
        sig_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        a_new = _ar1(r) if ar else np.zeros(DIM)
        # marginal std of AR(1) errors: MAD of raw residuals (the start's states absorb little of slow noise)
        grow = bool(np.any(sig_r > 1.1 * sig))  # same rule as joint_discover
        changed = grow or bool(np.any(np.abs(a_new - a) > 0.02))
        a = a_new
        if grow:
            sig = np.maximum(sig, sig_r)
        obs = Obs(yo, oi, a, sig)
        w = obs.white(current.u, allobs)
        phi = _phi(w)
        if not changed:
            break
        fits.clear()
        current = fit_starts()
    # AR(1) penalty multiplier ((1+a)/(1-a))^pen_exp (= 1 for white noise); pen_exp = 1/3 set on dev
    lam = float(np.mean(((1 + a) / (1 - a)) ** pen_exp))
    for _ in range(rounds):
        A, b = profiled_quadratic(current, obs, train, s_, dt_f, sig, rho, huber)
        cur = current.support
        proposals, choices = {}, []
        for j in range(DIM):
            other = cur.copy()
            other[:, j] = False
            cols = np.arange(j, NLIB * DIM, DIM)
            aj, bj = A[:, cols].copy(), b.copy()
            if other.any():
                q, sv, _ = np.linalg.svd(A[:, other.ravel()], full_matrices=False)
                q = q[:, sv > sv[0] * 1e-12]
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
                cc = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
                rk.append((np.sum((b - A[:, idx] @ cc) ** 2) + len(idx) * np.log(nobs), sup))
            ranked_eq.append([sp_ for _, sp_ in sorted(rk, key=lambda v: v[0])[:2]])
        for comb in itertools.product(*ranked_eq):
            mask = np.zeros_like(cur)
            for j, sup in enumerate(comb):
                mask[sup, j] = True
            proposals[tuple(np.flatnonzero(mask.ravel()))] = mask
        ranked = []
        for key, mask in proposals.items():
            if key in fits:
                continue
            idx = np.array(key)
            cc = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            ranked.append((np.sum((b - A[:, idx] @ cc) ** 2) + len(idx) * np.log(nobs), key, mask, cc))
        for _, key, mask, cc in sorted(ranked, key=lambda v: v[0])[:finalists]:
            C0 = np.zeros(NLIB * DIM)
            C0[list(key)] = cc
            add(mask, C0.reshape(NLIB, DIM), current.u)
        best = min(fits.values(), key=score)
        if best is current:
            break
        current = best
    final = fit_joint(obs, allobs, held, s_, dt_f, current.C, current.support, sig, rho, current.u,
                      max_iter=max_iter, c=huber)
    if final_rho is not None:
        stiff = fit_joint(obs, allobs, held, s_, dt_f, final.C, current.support, sig, final_rho, final.u,
                          max_iter=2 * max_iter, c=huber)
        if np.isfinite(stiff.data_chi2 + stiff.model_chi2):
            final = stiff
    xi = xi_from_scaled(final.C, scale)
    info = {"models_fitted": len(fits), "sigma": (sig * scale).tolist(), "sigma_hf": (sig_hf * scale).tolist(),
            "phi": phi * lam, "phi_raw": phi, "lam": lam, "substeps": s_, "ar": a.tolist()}
    if record:
        info["candidates"] = [{"support": f.support.tolist(), "k": f.k, "data": f.data_chi2, "model": f.model_chi2,
                               "val": f.val_chi2, "converged": f.converged, "iters": f.iters,
                               "dcomp": f.dcomp.tolist(), "ncomp": f.ncomp,
                               "xi": xi_from_scaled(f.C, scale).tolist()} for f in fits.values()]
    return xi, info
