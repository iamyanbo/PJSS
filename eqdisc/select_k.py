"""Best-subset weak SINDy with model-size selection.

Criteria:
  bic      : naive BIC, iid residuals                      (baseline, MIOSR-style)
  cv       : blocked cross-validation over test functions  (baseline)
  wbic     : BIC on GLS likelihood with the weak-form residual covariance (new)
  chi2     : smallest k whose whitened residual passes a chi^2 test        (new)
"""
import numpy as np
from .core import EXP_ARR, NLIB, DIM, library, scale_factors, xi_from_scaled, estimate_sigma
from .subset import best_subsets, beam_subsets


def weak_mats(n, dt, half_width=20, p=4):
    m = min(half_width, (n - 1) // 2)
    K = max(3 * NLIB, n // max(m // 2, 1))
    centers = np.linspace(m, n - 1 - m, K).round().astype(int)
    s = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - s ** 2) ** p
    dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / (m * dt)
    w = np.full(2 * m + 1, dt)
    w[0] = w[-1] = dt / 2
    nrm = (phi * w).sum()
    Phi = np.zeros((K, n))
    dPhi = np.zeros((K, n))
    for k, c in enumerate(centers):
        Phi[k, c - m:c + m + 1] = phi * w / nrm
        dPhi[k, c - m:c + m + 1] = dphi * w / nrm
    return Phi, dPhi


def dlibrary(X):
    """d Theta_m / d x_l : (N, NLIB, DIM)"""
    out = np.zeros((len(X), NLIB, DIM))
    for l in range(DIM):
        e = EXP_ARR.copy()
        a = e[:, l].astype(float)
        e[:, l] = np.maximum(e[:, l] - 1, 0)
        out[:, :, l] = a * np.prod(X[:, None, :] ** e, axis=-1)
    return out


def resid_cov(Phi, dPhi, dTh, xi, j, sig):
    """Covariance of the weak residual of equation j for coefficient vector xi (linearised EIV)."""
    K = Phi.shape[0]
    C = np.zeros((K, K))
    for l in range(DIM):
        J = dTh[:, :, l] @ xi  # (N,)
        L = -Phi * J[None, :]
        if l == j:
            L = L - dPhi
        C += sig[l] ** 2 * (L @ L.T)
    return C + 1e-10 * np.trace(C) / K * np.eye(K)


def fit_eq(G, b, sup):
    xi = np.zeros(G.shape[1])
    if len(sup):
        xi[sup] = np.linalg.lstsq(G[:, sup], b, rcond=None)[0]
    return xi


def discover(X, dt, half_width=20, kmax=None, criterion="chi2", sig=None, return_all=False, search=None):
    if search is None:
        search = "exhaustive" if NLIB <= 20 else "beam"
    if kmax is None:
        kmax = 5 if search == "exhaustive" else 6
    subsets = best_subsets if search == "exhaustive" else beam_subsets
    s = scale_factors(X)
    Xn = X / s
    n = len(Xn)
    Phi, dPhi = weak_mats(n, dt, half_width)
    Th = library(Xn)
    G = Phi @ Th
    B = -dPhi @ Xn
    K = G.shape[0]
    if sig is None:
        sig = estimate_sigma(Xn)
    dTh = dlibrary(Xn)
    crits = [criterion] if isinstance(criterion, str) else list(criterion)
    Xis = {c: np.zeros((NLIB, DIM)) for c in crits}
    info = []
    for j in range(DIM):
        cands = subsets(G, B[:, j], kmax)
        xis = [fit_eq(G, B[:, j], sup) for sup, _ in cands]
        chi = ll = None
        if any(c not in ("bic", "cv") for c in crits):
            chi, ll = [], []
            for xi in xis:
                C = resid_cov(Phi, dPhi, dTh, xi, j, sig)
                Lc = np.linalg.cholesky(C)
                z = np.linalg.solve(Lc, B[:, j] - G @ xi)
                chi.append(z @ z)
                ll.append(z @ z + 2 * np.log(np.diag(Lc)).sum())
            chi, ll = np.array(chi), np.array(ll)
            info.append(chi)
        for c in crits:
            Xis[c][:, j] = xis[_choose(c, cands, chi, ll, K, G, B[:, j], kmax, subsets)]
    Xis = {c: xi_from_scaled(v, s) for c, v in Xis.items()}
    out = Xis[criterion] if isinstance(criterion, str) else Xis
    return (out, info) if return_all else out


def _choose(criterion, cands, chi, ll, K, G, b, kmax, subsets):
    n = len(cands)
    dof = K - np.arange(n)
    if criterion == "bic":
        return int(np.argmin([K * np.log(rss / K + 1e-300) + len(sup) * np.log(K) for sup, rss in cands]))
    if criterion == "cv":
        return _cv_select(G, b, kmax, subsets=subsets)
    if criterion == "wbic":
        return int(np.argmin(ll + np.arange(n) * np.log(K)))
    if criterion == "chi2":  # discrepancy: smallest k consistent with the estimated noise
        ok = chi <= dof + 3 * np.sqrt(2 * dof)
        return int(np.argmax(ok)) if ok.any() else int(np.argmin(chi / dof))
    if criterion == "chi2s":  # scale self-calibrated from the largest model (Mallows-Cp style)
        ok = chi / (chi[-1] / dof[-1]) <= dof + 3 * np.sqrt(2 * dof)
        return int(np.argmax(ok)) if ok.any() else n - 1
    if criterion.startswith("elbow"):  # scale-free: stop when the next term improves chi^2 by < tau-fold
        small = chi[:-1] / chi[1:] < float(criterion[5:] or 2.0)
        return int(np.argmax(small)) if small.any() else n - 1
    raise ValueError(criterion)


def _cv_select(G, b, kmax, folds=5, subsets=best_subsets):
    K = len(b)
    blocks = np.array_split(np.arange(K), folds)
    err = np.zeros(kmax + 1)
    for blk in blocks:
        tr = np.setdiff1d(np.arange(K), blk)
        cands = subsets(G[tr], b[tr], kmax)
        for k, (sup, _) in enumerate(cands):
            xi = fit_eq(G[tr], b[tr], sup)
            err[k] += ((b[blk] - G[blk] @ xi) ** 2).sum()
    # one-standard-error-ish rule: smallest k within 5% of best
    return int(np.argmax(err <= err.min() * 1.05))
