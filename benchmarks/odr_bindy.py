"""Python port of ODR-BINDy (Fung 2025, arXiv:2507.23426; MATLAB code github.com/llfung/ODR-BINDy).

Faithful to FullGreedy/ODR_BINDy_Greedy.m, ODR-DynSys/ODR_BINDy_Regression.m, Jsq_xp.m,
EnsembleLinRegress_Wprior.m and d2Jdp2_GaussNewtonEst.m, including option defaults and control-flow
quirks (minimum 4 initial trials with doubling iteration caps, Failed_run bookkeeping, MaxNoiseIter
returning Inf). One substitution: MATLAB lsqnonlin trust-region-reflective is replaced by a sparse
Levenberg-Marquardt with the same tolerances (FunctionTolerance 5e-8, StepTolerance 1e-12) and
iteration caps; exitflag > 0 means converged within the cap.

Library: the benchmark's degree-3 polynomial library (core.EXPS); coefficients returned as (NLIB, 3).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from eqdisc.core import EXP_ARR

D1_6 = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])


def fd(n, dt, n_dom=6):
    """Interior central-difference operators (FD.m, FullMatFlag=false): I, D1 of shape (n-n_dom, n)."""
    if n_dom != 6:
        raise ValueError("only N_dom=6 (the paper's setting) is ported")
    rows = n - n_dom
    I = sp.diags([np.ones(rows)], [n_dom // 2], shape=(rows, n), format="csr")
    D = sp.diags([np.full(rows, c / dt) for c in D1_6 if True], list(range(n_dom + 1)), shape=(rows, n),
                 format="csr")
    return I, D


def theta(X, deriv=0):
    """Library and its derivatives at X (N, 3): deriv 0 -> (N,M); 1 -> (N,M,3); 2 -> (N,M,3,3)."""
    E = EXP_ARR  # (M, 3)
    if deriv == 0:
        return np.prod(X[:, None, :] ** E, axis=-1)

    def mono(Ex, coefs):
        out = np.prod(X[:, None, :] ** np.maximum(Ex, 0), axis=-1) * coefs
        return np.where((Ex >= 0).all(-1), out, 0.0)

    if deriv == 1:
        out = np.empty((len(X), len(E), 3))
        for b in range(3):
            Eb = E.copy(); Eb[:, b] -= 1
            out[:, :, b] = mono(Eb, E[:, b])
        return out
    out = np.empty((len(X), len(E), 3, 3))
    for a in range(3):
        for b in range(3):
            Eab = E.copy(); Eab[:, a] -= 1; Eab[:, b] -= 1
            c = E[:, a] * (E[:, b] - (a == b))
            out[:, :, a, b] = mono(Eab, c)
    return out


class Problem:
    def __init__(self, y, dt, sigma_x, sigma_y, sigma_p):
        self.y = np.asarray(y, float)
        self.Nx, self.ND = self.y.shape
        self.I, self.D = fd(self.Nx, dt)
        self.Neq = self.I.shape[0]
        self.M = EXP_ARR.shape[0]
        self.sx = np.broadcast_to(np.asarray(sigma_x, float), (self.Nx, self.ND))
        self.sy = float(sigma_y)
        self.sp = float(sigma_p)

    def _split(self, z, mask):
        x = z[:self.Nx * self.ND].reshape(self.ND, self.Nx).T
        p = z[self.Nx * self.ND:]
        ps, k = [], 0
        for d in range(self.ND):
            m = mask[:, d].sum()
            ps.append(p[k:k + m]); k += m
        return x, ps

    def residual(self, z, mask, jac=False):
        """Jsq_xp: out = [eta; delta; p/SigmaP] and its sparse Jacobian."""
        Nx, ND, Neq = self.Nx, self.ND, self.Neq
        x, ps = self._split(z, mask)
        Th = theta(x)
        ITh = self.I @ Th
        dx = self.D @ x
        eta = np.column_stack([(dx[:, d] - ITh[:, mask[:, d]] @ ps[d]) / self.sy for d in range(ND)])
        delta = (x - self.y) / self.sx
        p = z[Nx * ND:]
        out = np.concatenate([eta.T.ravel(), delta.T.ravel(), p / self.sp])
        if not jac:
            return out
        dTh = theta(x, 1)
        blocks = [[None] * ND for _ in range(ND)]
        for d in range(ND):
            df = np.einsum("nmb,m->nb", dTh[:, mask[:, d], :], ps[d])
            for b in range(ND):
                blk = -(self.I @ sp.diags(df[:, b]))
                if b == d:
                    blk = blk + self.D
                blocks[d][b] = blk / self.sy
        detadx = sp.bmat(blocks, format="csr")
        cols = []
        for d in range(ND):
            c = np.zeros((ND * Neq, mask[:, d].sum()))
            c[d * Neq:(d + 1) * Neq] = -ITh[:, mask[:, d]] / self.sy
            cols.append(c)
        detadp = sp.csr_matrix(np.hstack(cols))
        Np = len(p)
        J = sp.bmat([[detadx, detadp],
                     [sp.diags(1 / self.sx.T.ravel()), None],
                     [None, sp.diags(np.full(Np, 1 / self.sp))]], format="csc")
        return out, J

    def _solve(self, A, g):
        """Solve the arrowhead system A s = g by Schur complement on the state block."""
        n = self.Nx * self.ND
        A = A.tocsr()
        Axx, Axp = A[:n, :n].tocsc(), A[:n, n:].toarray()
        App = A[n:, n:].toarray()
        lu = splu(Axx)
        Y = lu.solve(np.column_stack([Axp, g[:n]]))
        S = App - Axp.T @ Y[:, :-1]
        sp_ = np.linalg.solve(S, g[n:] - Axp.T @ Y[:, -1])
        return np.concatenate([Y[:, -1] - Y[:, :-1] @ sp_, sp_])

    def lsq(self, z0, mask, max_iter, ftol=5e-8, xtol=1e-12, gtol=1e-6):
        """Sparse LM standing in for lsqnonlin. Returns z, resnorm, exitflag."""
        z = z0.copy()
        r, J = self.residual(z, mask, jac=True)
        cost = r @ r
        if not np.isfinite(cost):
            return z, np.inf, -1
        lam = 1e-3
        g0 = None
        for _ in range(max_iter):
            g = J.T @ r
            gi = np.abs(g).max()
            g0 = gi if g0 is None else g0
            if gi <= gtol * max(1.0, g0):
                return z, cost, 1
            H = (J.T @ J).tocsc()
            dg = H.diagonal()
            improved = False
            for _ in range(30):
                A = H + sp.diags(lam * np.maximum(dg, 1e-12))
                try:
                    step = -self._solve(A, g)
                except (RuntimeError, np.linalg.LinAlgError):
                    lam *= 10
                    continue
                zn = z + step
                rn = self.residual(zn, mask)
                cn = rn @ rn
                if np.isfinite(cn) and cn < cost:
                    improved = True
                    break
                lam *= 10
                if lam > 1e12:
                    break
            if not improved:
                return z, cost, 1 if cost < np.inf else -1  # no further decrease: stationary
            rel = (cost - cn) / max(cost, 1e-300)
            sn = np.linalg.norm(step)
            z, cost = zn, cn
            r, J = self.residual(z, mask, jac=True)
            lam = max(lam / 10, 1e-12)
            if rel < ftol or sn < xtol * (1 + np.linalg.norm(z)):
                return z, cost, 1
        return z, cost, 0

    def lin_guess(self, x, mask, n_boot, rng):
        """EnsembleLinRegress_Wprior with bootstrap + bragging (median)."""
        ITh = self.I @ theta(x)
        dxall = (self.D @ x) / self.sy ** 2
        out = []
        for d in range(self.ND):
            T = ITh[:, mask[:, d]]
            m = T.shape[1]
            counts = rng.multinomial(self.Neq, np.full(self.Neq, 1 / self.Neq), size=n_boot).astype(float)
            outer = (T[:, :, None] * T[:, None, :]).reshape(self.Neq, -1)
            G = (counts @ outer).reshape(n_boot, m, m) / self.sy ** 2 + np.eye(m) / self.sp ** 2
            rhs = counts @ (T * dxall[:, d:d + 1])
            coef = np.linalg.solve(G, rhs[..., None])[..., 0]
            out.append(np.median(coef, axis=0))
        return np.concatenate(out)

    def evidence_hessian(self, z, mask):
        """d2Jdp2_GaussNewtonEst: Hessian of the profiled objective in p."""
        Nx, ND, Neq = self.Nx, self.ND, self.Neq
        x, ps = self._split(z, mask)
        Th, dTh, ddTh = theta(x), theta(x, 1), theta(x, 2)
        f = np.column_stack([Th[:, mask[:, d]] @ ps[d] for d in range(ND)])
        df = np.stack([np.einsum("nmb,m->nb", dTh[:, mask[:, d], :], ps[d]) for d in range(ND)], 1)  # n,d,b
        ddf = np.stack([np.einsum("nmab,m->nab", ddTh[:, mask[:, d]], ps[d]) for d in range(ND)], 1)
        eta = self.D @ x - self.I @ f
        s = -(self.I.T @ (eta / self.sy ** 2))  # (Nx, ND)
        blocks = [[None] * ND for _ in range(ND)]
        for d in range(ND):
            for b in range(ND):
                blk = -(self.I @ sp.diags(df[:, d, b]))
                if b == d:
                    blk = blk + self.D
                blocks[d][b] = blk / self.sy
        eib = sp.bmat(blocks, format="csr")
        sec = np.einsum("nd,ndba->nba", s, ddf)
        SEC = sp.bmat([[sp.diags(sec[:, b, a]) for a in range(ND)] for b in range(ND)], format="csr")
        LHS = (eib.T @ eib + SEC + sp.diags(1 / self.sx.T.ravel() ** 2)).tocsc()
        Np = sum(len(q) for q in ps)
        A = np.zeros((ND * Nx, Np))
        G = np.zeros((ND * Neq, Np))
        j = 0
        for d in range(ND):
            for m in np.flatnonzero(mask[:, d]):
                for b in range(ND):
                    A[b * Nx:(b + 1) * Nx, j] = s[:, d] * dTh[:, m, b]
                G[d * Neq:(d + 1) * Neq, j] = -(self.I @ Th[:, m]) / self.sy
                j += 1
        RHS = -A - eib.T @ G
        dxdp = splu(LHS).solve(RHS)
        E2 = eib @ dxdp + G
        return E2.T @ E2 + dxdp.T @ (dxdp / self.sx.T.reshape(-1, 1) ** 2) + np.eye(Np) / self.sp ** 2

    def regression(self, mask, max_iter, n_boot, rng, x_ini=None, max_noise_iter=2):
        """ODR_BINDy_Regression (UseGaussNewtonEst=true, LinUseDenoise=true)."""
        x0 = self.y if x_ini is None else x_ini
        z_x = x0.T.ravel()
        H = None
        exitflag, jjj = 1, 0
        Np = int(mask.sum())
        while (H is None or np.linalg.eigvalsh(H).min() < 0) and exitflag > 0 and jjj < max_noise_iter:
            plin = self.lin_guess(x0, mask, n_boot, rng)
            try:
                z, resnorm, exitflag = self.lsq(np.concatenate([z_x, plin]), mask, max_iter)
                if exitflag > 0:
                    H = self.evidence_hessian(z, mask)
                w = np.zeros((self.M, self.ND))
                w.T[mask.T] = z[self.Nx * self.ND:]
                xout = z[:self.Nx * self.ND].reshape(self.ND, self.Nx).T
            except Exception:
                exitflag, w, xout = 0, np.zeros((self.M, self.ND)), np.zeros_like(self.y)
            jjj += 1
        if exitflag <= 0 or jjj >= max_noise_iter or H is None:
            return w, np.inf, xout, exitflag
        try:
            ld = np.sum(np.log(np.diag(np.linalg.cholesky(H / 2 / np.pi))))
        except np.linalg.LinAlgError:
            sgn, l2 = np.linalg.slogdet(H / 2 / np.pi)
            ld = l2 / 2 if sgn > 0 else np.nan
        J = resnorm / 2 + np.log(2 * np.pi) * Np / 2 + Np * np.log(self.sp) + ld
        return w, (J if np.isfinite(J) else np.inf), xout, exitflag


def _removal_decision(mask, J_i, m_idx, M, ND):
    if np.all(np.isinf(J_i)):
        return False, mask, -1
    order = np.argsort(J_i, kind="stable")
    for ii in order:
        nxt = mask.copy()
        nxt.T.flat[m_idx[ii]] = False
        if nxt.any(0).all():
            break
    if np.isinf(J_i[ii]):
        return False, mask, -1
    return True, nxt, ii


def odr_bindy(y, dt, sigma_x, sigma_y=1e-3, sigma_p=100.0, seed=0, min_initial=4, max_initial=16,
              max_failed=2, cont_run_stop=2, max_seconds=None, removal_iters=100):
    """ODR_BINDy_Greedy. Returns (Xi (M,3), info).

    removal_iters: iteration cap of the per-term removal fits (MATLAB: 100). Our LM can need more than
    MATLAB's trust-region solver, so a larger cap is the port-adjusted variant."""
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    P = Problem(y, dt, sigma_x, sigma_y, sigma_p)
    M, ND = P.M, P.ND
    mask = np.ones((M, ND), bool)
    NpAll = M * ND
    it2 = 1000
    J, w, X_ini = np.inf, None, None
    calls, ef = 0, -1
    fits = 0
    while (calls < min_initial or ef <= 0 or J == np.inf) and calls < max_initial:
        w_, J_, X_, ef = P.regression(mask, it2, 1000, rng)
        fits += 1
        calls += 1
        it2 *= 2
        if J_ < J and ef > 0:
            J, w, X_ini = J_, w_, X_
    if calls >= max_initial:
        return w_, {"failed": True, "fits": fits, "seconds": time.perf_counter() - t0}
    hist = []
    j, cont, failed = 1, 0, 0
    while j <= NpAll - ND and cont < cont_run_stop and failed < max_failed:
        if max_seconds and time.perf_counter() - t0 > max_seconds:
            break
        m_idx = np.flatnonzero(mask.T.ravel())
        J_i = np.full(len(m_idx), np.inf)
        res_i = [None] * len(m_idx)
        for ii, i in enumerate(m_idx):
            ml = mask.copy()
            ml.T.flat[i] = False
            if ml.any(0).all():
                w_, J_, X_, _ = P.regression(ml, removal_iters, 100, rng, X_ini)
                fits += 1
                J_i[ii], res_i[ii] = J_, (w_, X_)
        decided = False
        while not decided:
            proceed, mask_, i_min = _removal_decision(mask, J_i, m_idx, M, ND)
            if proceed:
                w_ini, J_ini, X_ini_, ef = P.regression(mask_, it2, 1000, rng, res_i[i_min][1])
                fits += 1
            else:
                ef, J_ini = -1, np.inf
            while (ef <= 0 or J_ini == np.inf) and failed < max_failed:
                w_ini, J_ini, X_ini_, ef = P.regression(mask_, it2, 1000, rng)
                fits += 1
                failed += 1
            if proceed and failed >= max_failed:
                J_i[i_min] = np.inf
                failed = 0
            else:
                decided = True
        if proceed:
            if J_ini < J_i[i_min]:
                hist.append((J_ini, w_ini))
                X_ini = X_ini_
            else:
                hist.append((J_i[i_min], res_i[i_min][0]))
                X_ini = res_i[i_min][1]
            if j > 2:
                cont = cont + 1 if hist[-1][0] >= hist[-2][0] else 0
            mask = mask_
            j += 1
        else:
            X_ini = X_ini_
    info = {"fits": fits, "steps": len(hist), "seconds": time.perf_counter() - t0, "failed": not hist,
            "timed_out": bool(max_seconds and time.perf_counter() - t0 > max_seconds)}
    if not hist:
        return w, info
    k = int(np.argmin([h[0] for h in hist]))
    info["evidence"] = [float(h[0]) for h in hist]
    return hist[k][1], info
