"""Core utilities: polynomial library, systems, weak-form / FD regression, STLSQ, metrics."""
import itertools
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

DIM = 3
DEG = int(os.environ.get("EQ_DEG", 3))
# exponent tuples for monomials in (x, y, z) up to DEG, ordered by degree
EXPS = [e for d in range(DEG + 1) for e in itertools.product(range(d + 1), repeat=DIM) if sum(e) == d]
EXPS = sorted(set(EXPS), key=lambda e: (sum(e), [-v for v in e]))
NLIB = len(EXPS)  # 20
EXP_ARR = np.array(EXPS)  # (NLIB, DIM)
NAMES = ["1" if sum(e) == 0 else "*".join(v + (f"^{p}" if p > 1 else "") for v, p in zip("xyz", e) if p) for e in EXPS]
IDX = {n: i for i, n in enumerate(NAMES)}


def library(X):
    """X: (..., N, DIM) -> (..., N, NLIB)"""
    return np.prod(X[..., None, :] ** EXP_ARR, axis=-1)


def coef(terms):
    """terms: list of 3 dicts {name: value} -> (NLIB, DIM) coefficient matrix"""
    Xi = np.zeros((NLIB, DIM))
    for i, eq in enumerate(terms):
        for n, v in eq.items():
            Xi[IDX[n], i] = v
    return Xi


# ---------------- benchmark systems (all polynomial, deg <= 3, 3D) ----------------
SYSTEMS = {
    "lorenz": coef([{"x": -10, "y": 10}, {"x": 28, "y": -1, "x*z": -1}, {"z": -8 / 3, "x*y": 1}]),
    "rossler": coef([{"y": -1, "z": -1}, {"x": 1, "y": 0.2}, {"1": 0.2, "z": -5.7, "x*z": 1}]),
    "chen": coef([{"x": -35, "y": 35}, {"x": -7, "y": 28, "x*z": -1}, {"z": -3, "x*y": 1}]),
    "halvorsen": coef([{"x": -1.4, "y": -4, "z": -4, "y^2": -1},
                       {"y": -1.4, "z": -4, "x": -4, "z^2": -1},
                       {"z": -1.4, "x": -4, "y": -4, "x^2": -1}]),
    "sprottA": coef([{"y": 1}, {"x": -1, "y*z": 1}, {"1": 1, "y^2": -1}]),
    "sprottB": coef([{"y*z": 1}, {"x": 1, "y": -1}, {"1": 1, "x*y": -1}]),
    "sprottC": coef([{"y*z": 1}, {"x": 1, "y": -1}, {"1": 1, "x^2": -1}]),
    "sprottD": coef([{"y": -1}, {"x": 1, "z": 1}, {"x*z": 1, "y^2": 3}]),
    "sprottE": coef([{"y*z": 1}, {"x^2": 1, "y": -1}, {"1": 1, "x": -4}]),
    "rabinovich": coef([{"y*z": 1, "y": -1, "x^2*y": 1, "x": 0.1},  # alpha=0.14 gamma=0.1
                        {"x*z": 3, "x": 1, "x^3": -1, "y": 0.1},
                        {"z": -2 * 0.14, "x*y*z": -2}]),
}
X0 = {"lorenz": [1, 1, 1], "rossler": [1, 1, 0], "chen": [-10, 0, 37], "halvorsen": [-5, 0, 0],
      "sprottA": [0, 5, 0], "sprottB": [0.1, 0.1, 0.1], "sprottC": [0.1, 0.1, 0.1], "sprottD": [0.1, 0.1, 0.1],
      "sprottE": [0.1, 0.1, 0.1], "rabinovich": [-1, 0, 0.5]}
DT = {"lorenz": 0.01, "rossler": 0.05, "chen": 0.005, "halvorsen": 0.01, "sprottA": 0.05, "sprottB": 0.05,
      "sprottC": 0.05, "sprottD": 0.05, "sprottE": 0.05, "rabinovich": 0.01}


def rhs(Xi):
    return lambda t, x: (library(x[None])[0] @ Xi)


def simulate(Xi, x0, dt, n, burn=0, rtol=1e-10, atol=1e-10):
    t = np.arange(burn + n) * dt
    sol = solve_ivp(rhs(Xi), (0, t[-1]), x0, t_eval=t, method="LSODA", rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t):
        return None
    return sol.y.T[burn:]


def add_noise(X, eta, rng):
    return X + eta * X.std(0) * rng.standard_normal(X.shape)


# ---------------- scaling (support-preserving: scale states only, no centring) ----------------
def scale_factors(X):
    return np.sqrt((X ** 2).mean(0))  # RMS per coordinate


def xi_to_scaled(Xi, s):
    # u = x / s ;  u_i' = sum_m Xi[m,i] * prod s^a / s_i * m(u)
    return Xi * np.prod(s ** EXP_ARR, axis=1)[:, None] / s[None, :]


def xi_from_scaled(Xu, s):
    return Xu / np.prod(s ** EXP_ARR, axis=1)[:, None] * s[None, :]


# ---------------- regression problems ----------------
def fd_problem(X, dt, window=None):
    """Savitzky-Golay smoothed states + derivatives -> (Theta, dX)"""
    n = len(X)
    if window is None:
        window = 15
    window = min(window, n - (1 - n % 2))
    if window % 2 == 0:
        window -= 1
    Xs = savgol_filter(X, window, 3, axis=0)
    dX = savgol_filter(X, window, 3, deriv=1, delta=dt, axis=0)
    return library(Xs), dX


def weak_problem(X, dt, half_width=20, n_test=None, p=4, sig=None):
    """Weak form with compact polynomial test functions phi=((t-a)(b-t))^p.
    G[k] = int phi_k Theta(X) dt ;  b[k] = -int phi_k' X dt"""
    n = len(X)
    m = min(half_width, (n - 1) // 2)
    if n_test is None:
        n_test = max(3 * NLIB, n // max(m // 2, 1))
    centers = np.linspace(m, n - 1 - m, n_test).round().astype(int)
    s = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - s ** 2) ** p
    dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / (m * dt)  # d/dt
    w = np.full(2 * m + 1, dt)
    w[0] = w[-1] = dt / 2
    nrm = (phi * w).sum()
    phi, dphi = phi / nrm, dphi / nrm
    Th = library(X) if sig is None else library_debiased(X, sig)
    idx = centers[:, None] + np.arange(-m, m + 1)[None, :]
    G = np.einsum("j,kjl->kl", phi * w, Th[idx])
    b = -np.einsum("j,kjd->kd", dphi * w, X[idx])
    return G, b


# ---------------- sparse regression ----------------
def stlsq(A, B, lam, alpha=1e-10, iters=20, mask=None, weights=None):
    """Sequentially thresholded least squares, columns normalised.
    lam: threshold on each term's relative contribution to the target. weights: per-entry multipliers on lam."""
    nrm = np.linalg.norm(A, axis=0) + 1e-12
    An = A / nrm
    nf, d = A.shape[1], B.shape[1]
    Xi = np.zeros((nf, d))
    thr = lam * (np.ones((nf, d)) if weights is None else weights)
    for j in range(d):
        act = np.ones(nf, bool) if mask is None else mask[:, j].copy()
        for _ in range(iters):
            if not act.any():
                break
            Aa = An[:, act]
            c = np.linalg.solve(Aa.T @ Aa + alpha * np.eye(act.sum()), Aa.T @ B[:, j])
            full = np.zeros(nf)
            full[act] = c
            # threshold on relative contribution: |c_k| * ||A_k|| / ||b||
            rel = np.abs(full) / (np.linalg.norm(B[:, j]) + 1e-12)
            new = act & (rel >= thr[:, j])
            if (new == act).all():
                break
            act = new
        Xi[:, j] = 0
        if act.any():
            Aa = An[:, act]
            c = np.linalg.solve(Aa.T @ Aa + alpha * np.eye(act.sum()), Aa.T @ B[:, j])
            Xi[act, j] = c / nrm[act]
    return Xi


def lstsq_support(A, B, support, alpha=1e-8):
    Xi = np.zeros(support.shape)
    for j in range(B.shape[1]):
        act = support[:, j]
        if act.any():
            Aa = A[:, act]
            Xi[act, j] = np.linalg.solve(Aa.T @ Aa + alpha * np.eye(act.sum()), Aa.T @ B[:, j])
    return Xi


def esindy(A, B, lam, n_boot=40, frac=0.6, pthr=0.6, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(A)
    inc = np.zeros((A.shape[1], B.shape[1]))
    for _ in range(n_boot):
        r = rng.choice(n, int(frac * n), replace=False)
        inc += stlsq(A[r], B[r], lam) != 0
    sup = inc / n_boot >= pthr
    return lstsq_support(A, B, sup)


# ---------------- metrics ----------------
def metrics(Xi_hat, Xi_true):
    S, T = Xi_hat != 0, Xi_true != 0
    tp = (S & T).sum()
    fp = (S & ~T).sum()
    fn = (~S & T).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    err = np.linalg.norm(Xi_hat - Xi_true) / np.linalg.norm(Xi_true)
    return dict(exact=float((S == T).all()), f1=f1, coef_err=err, fp=int(fp), fn=int(fn))


# ---------------- noise-debiased (Hermite) library ----------------
def estimate_sigma(X, order=4):
    """Noise std per coordinate from high-order finite differences (smooth signal ~ cancels).
    var(Delta^k e) = C(2k,k) sigma^2"""
    from math import comb
    D = np.diff(X, n=order, axis=0)
    mad = np.median(np.abs(D - np.median(D, 0)), 0) / 0.6745
    return mad / np.sqrt(comb(2 * order, order))


def hermite_powers(x, sig, kmax=DEG):
    """unbiased estimators of x_true^k given x = x_true + N(0, sig^2): probabilists' Hermite."""
    H = [np.ones_like(x), x]
    for k in range(1, kmax):
        H.append(x * H[k] - k * sig ** 2 * H[k - 1])
    return np.stack(H, -1)  # (..., kmax+1)


def library_debiased(X, sig):
    """X: (N, DIM); sig: (DIM,) -> (N, NLIB), E[Theta] = Theta(X_true) under independent gaussian noise"""
    P = [hermite_powers(X[:, j], sig[j]) for j in range(DIM)]
    return np.prod(np.stack([P[j][:, EXP_ARR[:, j]] for j in range(DIM)], 0), axis=0)
