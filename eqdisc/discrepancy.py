"""Model-discrepancy-aware profiled joint flow (weak-constraint 4D-Var with per-equation model error).

The frozen joint fit (joint_discover / joint_nd) treats the latent dynamics as exact up to a fixed stiffness:
the model residual of every equation on the latent sub-grid, u_{k+1} - u_k - h f(m_k), is weighted by
1/(rho*sigma_i) with rho = 0.03 for all equations. When an equation is not expressible in the library, its
misfit is pushed into the shared states and distorts the equations that are expressible.

Here every equation i gets its own model-error scale rho_i >= rho0 (rho0 = the frozen 0.03): the model
residual is read as a process-noise increment eta_i ~ N(0, (rho_i sigma_i)^2) per sub-step (white model error
x' = Theta(x) xi + eta(t), Kennedy-O'Hagan discrepancy / weak-constraint 4D-Var). rho_i is self-calibrated by
the Laplace-approximate marginal likelihood (evidence) of the data with the states u and the coefficients C
integrated out (flat prior on C, Gauss-Newton Hessian, banded Cholesky for the state block and a Schur
complement for the coefficients):

    -2 log p(y | support, rho) ~= cost(u*, C*) + logdet H_uu + logdet S_cc + 2 (nf - 1) sum_i log rho_i + const

The per-equation likelihood-ratio statistic  Lambda_i = ev(rho_i = rho0) - min_{rho_i} ev(rho_i)  is the
"is equation i in the library?" signal. With every rho_i = rho0 the objective is exactly the frozen one.

Public entry point: `discover(y, dt, ...)` = the frozen default (final_model.discover) followed by the
evidence test; only if some Lambda_i exceeds the frozen threshold is the joint search re-run with the
self-calibrated rho_i (equation-wise relaxed constraints). Otherwise the frozen answer is returned unchanged.
"""
import itertools

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

from . import joint_nd as jn
from .core import estimate_sigma
from .joint_biglib import top_subsets

RHO0 = 0.03
RHO_GRID = RHO0 * np.array([1, 3, 10, 30, 100, 300, 1000, 3000.0])


# ------------------------------------------------------------------------------------------------
# solver with a per-equation stiffness vector rho (D,)
# ------------------------------------------------------------------------------------------------
class WCSolver(jn.Solver):
    """joint_nd.Solver whose `rho` may be a vector (one model-error scale per equation)."""

    def normal(self, u, C, params, y, train, dt, sig, rho, c=np.inf):
        rho = np.broadcast_to(np.asarray(rho, float), sig.shape)
        D, BW = self.D, self.BW
        n = len(u)
        rd, rm, m, th = self.residuals(u, C, y, train, dt, sig, rho)
        J = np.einsum("mi,kml->kil", C, self.lib.d(m))
        eye = np.eye(D)
        w = 1.0 / (rho * sig)[None, :, None]
        B = np.concatenate([(-eye - 0.5 * dt * J) * w, (eye - 0.5 * dt * J) * w], axis=2)
        Mx = np.einsum("kap,kaq->kpq", B, B)
        ab = np.zeros((BW + 1, n * D))
        kD = D * np.arange(n - 1)
        for p in range(2 * D):
            for q in range(p, 2 * D):
                np.add.at(ab[BW + p - q], kD + q, Mx[:, p, q])
        wd = jn._hw(rd, c)
        ab[BW] += (train[:, None] * wd / sig[None, :] ** 2).ravel()
        gu = np.zeros(n * D)
        gb = np.einsum("kap,ka->kp", B, rm)
        for p in range(2 * D):
            np.add.at(gu, kD + p, gb[:, p])
        gu += (wd * rd / sig).ravel()
        mi, ei = params // D, params % D
        Dm = -dt * th[:, mi] / (rho[ei] * sig[ei])[None, :]
        P = len(params)
        Acc = np.zeros((P, P))
        for e in range(D):
            s = np.flatnonzero(ei == e)
            Acc[np.ix_(s, s)] = Dm[:, s].T @ Dm[:, s]
        Auc = np.zeros((n * D, P))
        Bsel = B[:, ei, :] * Dm[:, :, None]
        for r in range(2 * D):
            Auc[kD + r] += Bsel[:, :, r]
        gc = np.einsum("kp,kp->p", Dm, rm[:, ei])
        cost = jn._hsum(rd, c) + float((rm ** 2).sum())
        return ab, Auc, Acc, gu, gc, cost


def library_for(D, library):
    """PolyLib for the run's library: poly3 for D = 3 (same term order as core.NAMES), else degree 2."""
    if library in ("poly3",) and D == 3:
        return jn.PolyLib(3, 3)
    if library in ("poly2", "nd"):
        return jn.PolyLib(D, 2)
    raise ValueError(f"discrepancy model not implemented for library {library!r} (D = {D})")


# ------------------------------------------------------------------------------------------------
# problem setup (identical to joint_nd.discover / joint_discover.discover)
# ------------------------------------------------------------------------------------------------
class Problem:
    def __init__(self, x, dt, lib, sig=None, substeps=None, substep_cap=12):
        x = np.asarray(x, float)
        self.n, self.D = x.shape
        n, D = self.n, self.D
        self.lib, self.dt = lib, dt
        self.scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
        self.yo = x / self.scale
        self.sig_hf = np.maximum(estimate_sigma(self.yo), 1e-6)
        self.sig = self.sig_hf.copy() if sig is None else np.asarray(sig, float) / self.scale
        obs_train = np.ones(n, bool)
        obs_train[2::3] = False
        self.obs_train = obs_train
        self.s_ = jn.auto_substeps(self.yo, dt, cap=substep_cap) if substeps is None else int(substeps)
        s_ = self.s_
        self.nf = nf = (n - 1) * s_ + 1
        self.oi = oi = np.arange(n) * s_
        self.y = np.zeros((nf, D))
        self.y[oi] = self.yo
        self.train = np.zeros(nf, bool)
        self.train[oi[obs_train]] = True
        self.valmask = np.zeros(nf, bool)
        self.valmask[oi[~obs_train]] = True
        self.allobs = np.zeros(nf, bool)
        self.allobs[oi] = True
        self.dt_f = dt / s_
        u0o = jn._initial_states(self.yo, obs_train)
        tf = np.arange(nf) / s_
        self.u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(D)])
        self.nobs = n * D
        self.sv = WCSolver(lib)

    def fit(self, Cs, support, rho, u_init=None, train=None, max_iter=40, huber=2.0):
        u_init = self.u0 if u_init is None else u_init
        train = self.train if train is None else train
        return self.sv.fit(self.y, self.dt_f, Cs, support, self.sig, np.asarray(rho, float), train, u_init,
                           max_iter=max_iter, valmask=self.valmask, c=huber)


# ------------------------------------------------------------------------------------------------
# Laplace evidence and per-equation model-error calibration
# ------------------------------------------------------------------------------------------------
def neg2_log_evidence(P, f, rho, train=None, huber=2.0):
    """-2 log p(y | support, rho) up to rho-independent constants (Laplace over states and coefficients)."""
    rho = np.broadcast_to(np.asarray(rho, float), (P.D,))
    train = P.allobs if train is None else train
    params = np.flatnonzero(f.support.ravel())
    ab, Auc, Acc, gu, gc, cost = P.sv.normal(f.u, f.C, params, P.y, train, P.dt_f, P.sig, rho, huber)
    try:
        cb = cholesky_banded(ab, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf
    ld_u = 2.0 * float(np.log(cb[-1]).sum())
    ld_c = 0.0
    if len(params):
        X = cho_solve_banded((cb, False), Auc, check_finite=False)
        S = Acc - Auc.T @ X
        S = 0.5 * (S + S.T)
        sgn, ld_c = np.linalg.slogdet(S + 1e-12 * np.trace(S) / len(S) * np.eye(len(S)))
        if sgn <= 0:
            return np.inf
    return float(cost + ld_u + ld_c + 2.0 * (P.nf - 1) * np.log(rho).sum())


def _ev_fit(P, support, Cs, rho, u_init, huber):
    f = P.fit(Cs, support, rho, u_init=u_init, train=P.allobs, huber=huber)
    return neg2_log_evidence(P, f, rho, huber=huber), f


def calibrate_rho(P, support, Cs, u_init=None, huber=2.0, grid=RHO_GRID, passes=1, rho_start=None):
    """Coordinate search of rho_i on `grid` by the Laplace evidence (each step refits states + coefficients,
    warm-started). Returns (rho, ev, Lambda, fit, ev0): Lambda_i = ev(rho_i = rho0, others at the optimum) -
    ev(optimum), recomputed at the end; ev0 = evidence with every rho_i = rho0."""
    D = P.D
    rho = np.full(D, grid[0]) if rho_start is None else np.asarray(rho_start, float).copy()
    ev, f = _ev_fit(P, support, Cs, rho, u_init, huber)
    ev0 = ev if np.all(rho == grid[0]) else _ev_fit(P, support, Cs, np.full(D, grid[0]), u_init, huber)[0]
    for _ in range(passes):
        changed = False
        for i in range(D):
            k0 = int(np.argmin(np.abs(grid - rho[i])))
            for direction in (1, -1):  # walk up the grid while the evidence improves, then down
                k = k0
                while 0 <= k + direction < len(grid):
                    r = rho.copy()
                    r[i] = grid[k + direction]
                    e, g = _ev_fit(P, support, f.C, r, f.u, huber)
                    if e < ev - 1e-9:
                        rho, ev, f, k = r, e, g, k + direction
                        changed = True
                    else:
                        break
                if k != k0:
                    break
        if not changed:
            break
    lam = np.zeros(D)
    for i in range(D):
        if rho[i] > grid[0]:
            r = rho.copy()
            r[i] = grid[0]
            lam[i] = _ev_fit(P, support, f.C, r, f.u, huber)[0] - ev
    return rho, ev, lam, f, ev0


# ------------------------------------------------------------------------------------------------
# weak-constraint joint search (joint_nd.discover with a per-equation rho vector)
# ------------------------------------------------------------------------------------------------
def search(P, starts, rho, *, w=None, rounds=2, kmax=5, keep=2, finalists=10, max_iter=40, huber=2.0,
           phi=1.0, width=60, final_rho=0.001, record=False):
    """Same proposals / refits / BIC-phi selection as joint_nd.discover (sigma fixed in P), with the
    stiffness rho (D,) per equation. `w`: per-term prior weights (weighted term count, final_model prior).
    Final refit on all observations; the stiff final refit uses final_rho only for equations at rho0."""
    lib, D, NL = P.lib, P.D, P.lib.M
    rho = np.asarray(rho, float)
    sv, nobs = P.sv, P.nobs

    def kk(f):
        return float((f.support * (1.0 if w is None else w[:, None])).sum())

    def score(f):
        return f.data_chi2 + f.model_chi2 + f.val_chi2 + phi * kk(f) * np.log(nobs)

    fits = {}

    def add(support, C0, u_init):
        key = tuple(np.flatnonzero(support.ravel()))
        if key in fits or not support.any():
            return fits.get(key)
        f = P.fit(C0, support, rho, u_init=u_init, max_iter=max_iter, huber=huber)
        if np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2):
            fits[key] = f
        return fits.get(key)

    for xi in starts:
        xi = np.asarray(xi, float)
        if (xi != 0).any():
            add(xi != 0, lib.to_scaled(xi, P.scale), P.u0)
    if not fits:
        raise RuntimeError("no start model fitted")
    current = min(fits.values(), key=score)
    for _ in range(rounds):
        A, b = sv.profiled_quadratic(current, P.y, P.dt_f, P.sig, rho, P.train, huber)
        cur = current.support
        proposals, ranked_eq = {}, []
        for j in range(D):
            other = cur.copy()
            other[:, j] = False
            cols = np.arange(j, NL * D, D)
            aj, bj = A[:, cols].copy(), b.copy()
            if other.any():
                q, s, _ = np.linalg.svd(A[:, other.ravel()], full_matrices=False)
                q = q[:, s > s[0] * 1e-12]
                aj -= q @ (q.T @ aj)
                bj -= q @ (q.T @ bj)
            sups = [sup for sup, _ in top_subsets(aj, bj, kmax, keep, width=width)]
            rk = []
            for sup in sups:
                mask = cur.copy()
                mask[:, j] = False
                mask[sup, j] = True
                idx = np.flatnonzero(mask.ravel())
                proposals[tuple(idx)] = mask
                c = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
                rk.append((np.sum((b - A[:, idx] @ c) ** 2) + len(idx) * np.log(nobs), sup))
            ranked_eq.append([s for _, s in sorted(rk, key=lambda v: v[0])[:2]])
        combos = itertools.product(*ranked_eq)
        if 2 ** D > 1024:
            combos = itertools.islice(combos, 1024)
        for comb in combos:
            mask = np.zeros_like(cur)
            for j, sup in enumerate(comb):
                mask[sup, j] = True
            proposals[tuple(np.flatnonzero(mask.ravel()))] = mask
        ranked = []
        for key, mask in proposals.items():
            if key in fits:
                continue
            idx = np.array(key, dtype=int)
            c = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            ranked.append((np.sum((b - A[:, idx] @ c) ** 2) + len(idx) * np.log(nobs), key, mask, c))
        for _, key, mask, c in sorted(ranked, key=lambda v: v[0])[:finalists]:
            C0 = np.zeros(NL * D)
            C0[list(key)] = c
            add(mask, C0.reshape(NL, D), current.u)
        best = min(fits.values(), key=score)
        if best is current:
            break
        current = best
    final = P.fit(current.C, current.support, rho, u_init=current.u, train=P.allobs, max_iter=max_iter,
                  huber=huber)
    if final_rho is not None:
        r_f = np.where(rho <= RHO0 * (1 + 1e-9), final_rho, rho)
        stiff = P.fit(final.C, current.support, r_f, u_init=final.u, train=P.allobs, max_iter=2 * max_iter,
                      huber=huber)
        if np.isfinite(stiff.data_chi2 + stiff.model_chi2):
            final = stiff
    sc = sorted(score(f) for f in fits.values())
    info = {"models_fitted": len(fits), "gap": float(sc[1] - sc[0]) if len(sc) > 1 else None,
            "support_scaled_best": current.support.tolist()}
    if record:
        info["candidates"] = [{"k": f.k, "score": score(f), "xi": lib.from_scaled(f.C, P.scale).tolist()}
                              for f in fits.values()]
    return lib.from_scaled(final.C, P.scale), current, info
