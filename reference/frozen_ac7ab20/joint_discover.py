"""Fast joint-state support search ("profiled joint flow").

States u_k and sparse coefficients C are estimated together, as in DEDi
(Meissner & Glasner), by minimising

    sum_train |(u_k - y_k)/sigma|^2 + sum_k |(u_{k+1} - u_k - dt f(m_k; C))/(rho sigma)|^2,
    m_k = (u_k + u_{k+1})/2  (midpoint rule),

but with no L0 smoothing or hyperparameter grid: for a FIXED support the
problem is a small nonlinear least squares solved by Levenberg-Marquardt, whose
normal matrix is banded in the states plus a dense coefficient border (Schur
complement, solveh_banded). Supports are proposed from the profiled quadratic
model of the same objective in ALL library coefficients (states profiled out),
the joint-state analogue of flow_discovery's segment sensitivities. Every third
sample is held out of the data term; held-out misfit scores candidates.
Truth is never used.
"""
import itertools

import numpy as np
from scipy.linalg import solveh_banded, cho_factor, cho_solve
from scipy.signal import savgol_filter

from core import NLIB, DIM, library, estimate_sigma, xi_to_scaled, xi_from_scaled
from select_k import dlibrary
from flow_discovery import _top_subsets

BW = 2 * DIM - 1  # upper bandwidth of the state block


class JointFit:
    def __init__(self, u, C, support, data_chi2, model_chi2, val_chi2, iters, converged):
        self.u, self.C, self.support = u, C, support
        self.data_chi2, self.model_chi2, self.val_chi2 = data_chi2, model_chi2, val_chi2
        self.iters, self.converged = iters, converged

    @property
    def k(self):
        return int(self.support.sum())


def _hw(r, c):
    """Huber IRLS weights (r in sigma units); c = inf gives least squares."""
    a = np.abs(r)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))


def _hsum(r, c):
    """Sum of 2*Huber loss: r^2 inside c, 2c|r| - c^2 outside (equals chi^2 when c = inf)."""
    a = np.abs(r)
    if not np.isfinite(c):
        return float((a ** 2).sum())
    return float(np.where(a <= c, a ** 2, 2 * c * a - c ** 2).sum())


def _residuals(u, C, y, train, dt, sig, rho):
    m = 0.5 * (u[1:] + u[:-1])
    th = library(m)
    rm = (u[1:] - u[:-1] - dt * th @ C) / (rho * sig)
    rd = np.where(train[:, None], (u - y) / sig, 0.0)
    return rd, rm, m, th


def _normal(u, C, params, y, train, dt, sig, rho, c=np.inf):
    """Gauss-Newton pieces: banded A_uu, dense A_uc, A_cc, gradients, residuals."""
    n = len(u)
    rd, rm, m, th = _residuals(u, C, y, train, dt, sig, rho)
    J = np.einsum("mi,kml->kil", C, dlibrary(m))  # (n-1, i, l)
    eye = np.eye(DIM)
    w = 1.0 / (rho * sig)[None, :, None]
    B = np.concatenate([(-eye - 0.5 * dt * J) * w, (eye - 0.5 * dt * J) * w], axis=2)  # (n-1, DIM, 2DIM)
    M = np.einsum("kap,kaq->kpq", B, B)
    ab = np.zeros((BW + 1, n * DIM))
    k3 = DIM * np.arange(n - 1)
    for p in range(2 * DIM):
        for q in range(p, 2 * DIM):
            np.add.at(ab[BW + p - q], k3 + q, M[:, p, q])
    wd = _hw(rd, c)
    dd = (train[:, None] * wd / sig[None, :] ** 2).ravel()
    ab[BW] += dd
    gu = np.zeros(n * DIM)
    gb = np.einsum("kap,ka->kp", B, rm)
    for p in range(2 * DIM):
        np.add.at(gu, k3 + p, gb[:, p])
    gu += (wd * rd / sig).ravel()
    mi = params // DIM
    ei = params % DIM
    D = -dt * th[:, mi] / (rho * sig[ei])[None, :]  # (n-1, P)
    P = len(params)
    Acc = np.zeros((P, P))
    for e in range(DIM):
        s = np.flatnonzero(ei == e)
        Acc[np.ix_(s, s)] = D[:, s].T @ D[:, s]
    Auc = np.zeros((n * DIM, P))
    Bsel = B[:, ei, :] * D[:, :, None]  # (n-1, P, 2DIM)
    for r in range(2 * DIM):
        Auc[k3 + r] += Bsel[:, :, r]
    gc = np.einsum("kp,kp->p", D, rm[:, ei])
    cost = _hsum(rd, c) + float((rm ** 2).sum())
    return ab, Auc, Acc, gu, gc, cost


def _schur_solve(ab, Auc, Acc, gu, gc, mu):
    ab = ab.copy()
    ab[BW] += mu * (1.0 + ab[BW])
    acc = Acc + mu * np.diag(1.0 + np.diag(Acc))
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    xg, xc = X[:, 0], X[:, 1:]
    S = acc - Auc.T @ xc
    dc = np.linalg.solve(S + 1e-12 * np.trace(S) / max(len(S), 1) * np.eye(len(S)), -(gc - Auc.T @ xg))
    du = -xg - xc @ dc
    return du, dc


def fit_joint(y, dt, C0, support, sig, rho, train, u0, max_iter=40, tol=1e-7, valmask=None, c=np.inf):
    """LM on states + coefficients restricted to `support` (NLIB, DIM)."""
    params = np.flatnonzero(support.ravel())
    u, C = u0.copy(), np.where(support, C0, 0.0)
    mu, converged, it = 1e-3, False, 0
    ab, Auc, Acc, gu, gc, cost = _normal(u, C, params, y, train, dt, sig, rho, c)
    for it in range(1, max_iter + 1):
        try:
            du, dc = _schur_solve(ab, Auc, Acc, gu, gc, mu)
        except (np.linalg.LinAlgError, ValueError):
            mu *= 10
            continue
        un = u + du.reshape(u.shape)
        Cn = C.copy()
        Cn.ravel()[params] += dc
        rd, rm, _, _ = _residuals(un, Cn, y, train, dt, sig, rho)
        new = _hsum(rd, c) + float((rm ** 2).sum())
        if np.isfinite(new) and new < cost:
            rel = (cost - new) / max(cost, 1e-300)
            u, C = un, Cn
            ab, Auc, Acc, gu, gc, cost = _normal(u, C, params, y, train, dt, sig, rho, c)
            mu = max(mu / 3, 1e-9)
            if rel < tol:
                converged = True
                break
        else:
            mu *= 4
            if mu > 1e8:
                break
    rd, rm, _, _ = _residuals(u, C, y, train, dt, sig, rho)
    vm = ~train if valmask is None else valmask
    val = _hsum(((u - y) / sig)[vm], c)
    return JointFit(u, C, support.copy(), _hsum(rd, c), float((rm ** 2).sum()), val, it, converged)


def profiled_quadratic(fit, y, dt, sig, rho, train, c=np.inf):
    """Least-squares form (A, b) of the profiled quadratic model in ALL NLIB*DIM coefficients."""
    params = np.arange(NLIB * DIM)
    ab, Auc, Acc, gu, gc, _ = _normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    H = Acc - Auc.T @ X[:, 1:]
    g = gc - Auc.T @ X[:, 0]
    H = 0.5 * (H + H.T) + 1e-10 * np.trace(H) / len(H) * np.eye(len(H))
    c0 = fit.C.ravel()
    cstar = c0 - np.linalg.solve(H, g)
    L = np.linalg.cholesky(H)
    return L.T, L.T @ cstar


def _initial_states(y, train):
    u = y.copy()
    idx = np.arange(len(y))
    for i in range(DIM):
        u[~train, i] = np.interp(idx[~train], idx[train], y[train, i])
    w = min(21, len(y) - 1 - (len(y) % 2 == 0))
    return savgol_filter(u, w, 3, axis=0)


def auto_substeps(y, dt, target=256, target99=64, cap=12):
    """Sub-steps for the latent grid, from the data's spectrum: the larger of
    ~`target` points per dominant cycle and ~`target99` points per cycle of the frequency
    below which 99% of the above-noise power lies (bins under 10x the white-noise floor,
    taken from the top quarter of frequencies, are ignored). At least 2, at most `cap`."""
    f, f99 = [], []
    for i in range(y.shape[1]):
        s = y[:, i] - y[:, i].mean()
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), dt)
        p[:3] = 0  # ignore the window's lowest bins
        f.append(fr[np.argmax(p)])
        floor = np.mean(p[int(0.75 * len(p)):])
        ps = np.where(p > 10 * floor, p - floor, 0.0)
        f99.append(fr[np.searchsorted(np.cumsum(ps) / ps.sum(), 0.99)] if ps.sum() > 0 else f[-1])
    s_peak = np.ceil(target * max(f) * dt)
    s_99 = np.ceil(target99 * max(f99) * dt)
    return int(np.clip(max(s_peak, s_99, 2), 2, cap))


def _phi(r, kmax=50):
    """Variance inflation 1 + 2 sum rho_k of residual autocorrelation (>= 1), averaged over coordinates."""
    out = []
    for i in range(r.shape[1]):
        e = r[:, i] - r[:, i].mean()
        v = e @ e
        t = 1.0
        for k in range(1, min(kmax, len(e) // 4)):
            rk = (e[:-k] @ e[k:]) / v
            if rk < 0.05:
                break
            t += 2 * rk
        out.append(t)
    return max(1.0, float(np.mean(out)))


def _coeffs_from_states(u, dt, support):
    m = 0.5 * (u[1:] + u[:-1])
    th = library(m)
    du = (u[1:] - u[:-1]) / dt
    C = np.zeros((NLIB, DIM))
    for i in range(DIM):
        s = np.flatnonzero(support[:, i])
        if len(s):
            C[s, i] = np.linalg.lstsq(th[:, s], du[:, i], rcond=None)[0]
    return C


def discover(x, dt, starts, *, rho=0.03, rounds=2, kmax=5, keep=2, finalists=10, max_iter=40,
             penalty="bic", substeps="auto", huber=2.0, selfcal=True, final_rho=0.001, record=False):
    """starts: coefficient arrays (ORIGINAL units) from other methods, e.g. weak / flow / hybrid.

    Returns (Xi in original units, info). info["candidates"] lists every fitted support with
    its scores when record=True.
    """
    x = np.asarray(x, float)
    n = len(x)
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    obs_train = np.ones(n, bool)
    obs_train[2::3] = False
    nobs = n * DIM
    # latent sub-grid: `substeps` midpoint steps between consecutive samples
    s_ = auto_substeps(yo, dt) if substeps == "auto" else int(substeps)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    y = np.zeros((nf, DIM))
    y[oi] = yo
    train = np.zeros(nf, bool)
    train[oi[obs_train]] = True
    valmask = np.zeros(nf, bool)
    valmask[oi[~obs_train]] = True
    dt_f = dt / s_
    u0o = _initial_states(yo, obs_train)
    tf = np.arange(nf) / s_
    u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(DIM)])
    allobs = np.zeros(nf, bool)
    allobs[oi] = True

    phi = 1.0

    def score(f):
        pen = f.k * np.log(nobs) if penalty == "bic" else 2 * f.k
        return f.data_chi2 + f.model_chi2 + f.val_chi2 + phi * pen

    fits = {}

    def add(support, C0, u_init):
        key = tuple(np.flatnonzero(support.ravel()))
        if key in fits or not support.any():
            return fits.get(key)
        f = fit_joint(y, dt_f, C0, support, sig, rho, train, u_init, max_iter=max_iter, valmask=valmask,
                      c=huber)
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
    if selfcal:
        # Noise scale and correlation from the best start's own residuals at the samples
        # (high-order differences miss slowly correlated noise). Scores are then divided by
        # phi, equivalently the term penalty multiplied by it.
        r = yo - current.u[oi]
        sig_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        phi = _phi(r / np.maximum(sig_r, 1e-12))
        if np.any(sig_r > 1.1 * sig):
            sig = np.maximum(sig, sig_r)
            fits.clear()
            current = fit_starts()
    for _ in range(rounds):
        A, b = profiled_quadratic(current, y, dt_f, sig, rho, train, huber)
        cur = current.support
        proposals = {}
        choices = []
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
            c = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            ranked.append((np.sum((b - A[:, idx] @ c) ** 2) + len(idx) * np.log(nobs), key, mask, c))
        for _, key, mask, c in sorted(ranked, key=lambda v: v[0])[:finalists]:
            C0 = np.zeros(NLIB * DIM)
            C0[list(key)] = c
            add(mask, C0.reshape(NLIB, DIM), current.u)
        best = min(fits.values(), key=score)
        if best is current:
            break
        current = best
    final = fit_joint(y, dt_f, current.C, current.support, sig, rho, allobs, current.u, max_iter=max_iter,
                      valmask=valmask, c=huber)
    for r_ in ([] if final_rho is None else [final_rho]):
        # stiffer all-data refit of the chosen support (coefficients only; selection unchanged)
        stiff = fit_joint(y, dt_f, final.C, current.support, sig, r_, allobs, final.u, max_iter=2 * max_iter,
                          valmask=valmask, c=huber)
        if np.isfinite(stiff.data_chi2 + stiff.model_chi2):
            final = stiff
    xi = xi_from_scaled(final.C, scale)
    info = {"models_fitted": len(fits), "sigma": (sig * scale).tolist(), "sigma_hf": (sig_hf * scale).tolist(),
            "phi": phi, "substeps": s_}
    if record:
        info["candidates"] = [{"support": f.support.tolist(), "k": f.k, "data": f.data_chi2, "model": f.model_chi2,
                               "val": f.val_chi2, "converged": f.converged, "iters": f.iters,
                               "xi": xi_from_scaled(f.C, scale).tolist()} for f in fits.values()]
    return xi, info
