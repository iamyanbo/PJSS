"""Gauge-aware scoring for partial observation (evaluation only; never used by a method).

A latent coordinate is identifiable only up to an affine change that keeps the observed coordinates fixed:
    x_L = A xi_L + B x_O + c        (A invertible n_L x n_L, B n_L x n_O, c in R^n_L).
Such maps keep the degree-3 polynomial library closed, so the true system has an exact polynomial form T(Xi; A,B,c)
in the coordinates (x_O, xi_L). Offsets and mixing change the support; scales do not.

exact-up-to-gauge: the discovered support S equals supp T(Xi; A,B,c) for SOME (A,B,c). Checked numerically:
  stage 1  (A,B,c) closest in coefficients: min ||T - Xi_hat||^2 / ||Xi_hat||^2, started from the regression of
           the true latent states on the method's estimated states (u_L, x_O, 1);
  stage 2  from there, min over (A,B,c) of the off-support coefficients of T, each relative to its equation's
           coefficient norm; support match iff every off-support entry is < 1e-6 of its equation's norm, every term
           of S is > 1e-6 of it, and the gauge is well conditioned (cond(diag(1/rms x_L) A) < 1e4).
exact-up-to-gauge = support match AND state consistency: the regression of the true latent trajectory on the
           method's estimated states (u_L, x_O, 1) has R^2 > 0.9 for every latent component.
coefficient error: ||T(stage 1) - Xi_hat||_F / ||T(stage 1)||_F.
forecast: observed-coordinate valid time (in Lyapunov times) of the discovered model started from the method's
          estimated state at the middle sample, against the clean trajectory (normalised error > 0.4 ends it).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from eqdisc.core import NLIB, DIM, library

LYAP = {"lorenz": 0.906, "rossler": 0.0714, "chen": 2.03}


def _unpack(th, nl, no):
    A = th[:nl * nl].reshape(nl, nl)
    B = th[nl * nl:nl * nl + nl * no].reshape(nl, no)
    c = th[nl * nl + nl * no:]
    return A, B, c


def transformed(Xi, obs, lat, A, B, c, pts):
    """Coefficients (NLIB, DIM) of the true model in coordinates z = (x_O, xi_L), by exact polynomial fit."""
    z = pts
    x = z.copy()
    x[:, lat] = z[:, lat] @ A.T + z[:, obs] @ B.T + c
    f = library(x) @ Xi
    g = f.copy()
    g[:, lat] = np.linalg.solve(A, (f[:, lat] - f[:, obs] @ B.T).T).T
    Th = library(z)
    return np.linalg.lstsq(Th, g, rcond=None)[0]


def gauge_match(xi_hat, Xi, obs, u_hat, x_true, seed=0):
    obs, lat = list(obs), [j for j in range(DIM) if j not in obs]
    nl, no = len(lat), len(obs)
    rng = np.random.default_rng(seed)
    u_hat = np.asarray(u_hat)
    lo, hi = u_hat.min(0), u_hat.max(0)
    pts = lo + (hi - lo) * rng.random((80, DIM))
    S = np.asarray(xi_hat) != 0
    # initial gauge: regress true latent on (estimated latent, observed, 1)
    R = np.column_stack([u_hat[:, lat], u_hat[:, obs], np.ones(len(u_hat))])
    W = np.linalg.lstsq(R, x_true[:, lat], rcond=None)[0]  # (nl+no+1, nl)
    th0 = np.concatenate([W[:nl].T.ravel(), W[nl:nl + no].T.ravel(), W[-1]])
    nrm = max(np.linalg.norm(xi_hat), 1e-12)

    def T(th):
        A, B, c = _unpack(th, nl, no)
        return transformed(Xi, obs, lat, A, B, c, pts)

    def r1(th):
        try:
            return ((T(th) - xi_hat) / nrm).ravel()
        except np.linalg.LinAlgError:
            return np.full(NLIB * DIM, 1e3)

    s1 = least_squares(r1, th0, method="lm", max_nfev=4000)
    T1 = T(s1.x)
    coef_err = float(np.linalg.norm(T1 - xi_hat) / max(np.linalg.norm(T1), 1e-12))

    def r2(th):
        try:
            t = T(th)
        except np.linalg.LinAlgError:
            return np.full(int((~S).sum()), 1e3)
        return t[~S] / max(np.linalg.norm(t), 1e-12)

    def colnorm(t):
        return np.maximum(np.linalg.norm(t, axis=0), 1e-300)

    def r2c(th):  # per-equation relative off-support coefficients
        try:
            t = T(th)
        except np.linalg.LinAlgError:
            return np.full(int((~S).sum()), 1e3)
        return (t / colnorm(t))[~S]

    best = None
    for st in (s1.x, th0):
        s2 = least_squares(r2c, st, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        t2 = T(s2.x)
        tn = np.abs(t2) / colnorm(t2)
        off = float(tn[~S].max()) if (~S).any() else 0.0
        on_min = float(tn[S].min()) if S.any() else 0.0
        if best is None or off < best[0]:
            best = (off, on_min, s2.x)
    off, on_min, th = best
    A, B, c = _unpack(th, nl, no)
    # gauge conditioning, in units of the true latent RMS (a legitimate change of coordinates is well conditioned;
    # a near-singular A can make any model look equivalent after normalisation)
    rms_true = np.sqrt(np.mean(x_true[:, lat] ** 2, axis=0))
    cond = float(np.linalg.cond(A / rms_true[:, None]))
    # state consistency: the method's latent states must map affinely onto the true latent states
    pred = R @ W
    r2 = 1 - np.sum((x_true[:, lat] - pred) ** 2, axis=0) / np.sum((x_true[:, lat] - x_true[:, lat].mean(0)) ** 2,
                                                                     axis=0)
    support_ok = bool(off < 1e-6 and on_min > 1e-6 and cond < 1e4)
    exact = bool(support_ok and r2.min() > 0.9)
    return {"exact_gauge": exact, "support_ok": support_ok, "offnorm": off, "on_min": on_min, "cond": cond,
            "latent_r2": float(r2.min()), "coef_err": coef_err,
            "gauge": {"A": A.tolist(), "B": B.tolist(), "c": c.tolist()}}


def valid_time(xi_hat, u_hat, clean, obs, dt, system, thr=0.4, i0=None):
    """Lyapunov-time horizon over which the model, started at the estimated state u_hat[i0], tracks the clean
    observed coordinates (normalised RMS error over observed coordinates below thr)."""
    clean = np.asarray(clean)
    n = len(clean)
    i0 = n // 2 if i0 is None else i0
    horizon = n - i0
    obs = list(obs)
    scl = np.sqrt(np.mean(clean[:, obs] ** 2, axis=0) - np.mean(clean[:, obs], axis=0) ** 2)
    big = 1e3 * np.abs(clean).max()
    nfev = [0]

    def f(_, x):
        nfev[0] += 1
        if nfev[0] > 200000 or not np.all(np.abs(x) < big):  # runaway / stiff model: stop (dev E hang)
            raise RuntimeError("runaway")
        return library(x[None])[0] @ xi_hat

    # integrate in chunks and stop at the first sample whose normalised error reaches thr
    x, k, chunk = np.asarray(u_hat, float)[i0], 0, 50
    with np.errstate(all="ignore"):
        while k < horizon - 1:
            m = min(chunk, horizon - 1 - k)
            t = np.arange(m + 1) * dt
            try:
                sol = solve_ivp(f, (0, t[-1]), x, t_eval=t, method="LSODA", rtol=1e-8, atol=1e-8)
            except Exception:
                break
            if not sol.success or sol.y.shape[1] < m + 1:
                break
            pred = sol.y.T[1:, obs]
            err = np.sqrt(np.mean(((pred - clean[i0 + k + 1:i0 + k + m + 1][:, obs]) / scl) ** 2, axis=1))
            bad = np.flatnonzero(~(err < thr))
            if len(bad):
                k += int(bad[0])
                break
            k += m
            x = sol.y[:, -1]
        else:
            k = horizon
    return float(k * dt * LYAP[system])
