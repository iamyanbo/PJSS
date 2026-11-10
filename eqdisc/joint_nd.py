"""Profiled joint flow in D dimensions (standalone generalisation of joint_discover).

Same objective, solver and selection as joint_discover (config C: rho 0.03, automatic sub-steps,
sigma/phi self-calibration, Huber 2, stiff final refit at rho 0.001, BIC with phi-inflated penalty),
with the state dimension D and the polynomial library (all monomials of degree <= deg in D variables)
as parameters. States are interleaved by time (index k*D + i), so the model residual between k and k+1
couples 2D consecutive unknowns and the state block stays banded with upper bandwidth 2D-1.
Per-equation subset ranking on the profiled quadratic is exhaustive when small, beam search otherwise
(joint_biglib.top_subsets).

Starts come from `weak_starts`: weak-form (compact polynomial test functions) best-subset regression per
equation in the same library, size chosen by BIC on the weak residuals, plus the runner-up sizes.
"""
import itertools

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter

from .core import estimate_sigma
from .joint_biglib import top_subsets


class PolyLib:
    def __init__(self, D, deg):
        self.D, self.deg = D, deg
        E = [e for d in range(deg + 1) for e in itertools.product(range(d + 1), repeat=D) if sum(e) == d]
        self.E = np.array(sorted(set(E), key=lambda e: (sum(e), [-v for v in e])))
        self.M = len(self.E)
        names = []
        for e in self.E:
            t = [f"x{i}" + (f"^{p}" if p > 1 else "") for i, p in enumerate(e) if p]
            names.append("*".join(t) if t else "1")
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}

    def __call__(self, X):
        return np.prod(X[..., None, :] ** self.E, axis=-1)

    def d(self, X):
        out = np.zeros((len(X), self.M, self.D))
        for l in range(self.D):
            e = self.E.copy()
            a = e[:, l].astype(float)
            e[:, l] = np.maximum(e[:, l] - 1, 0)
            out[:, :, l] = a * np.prod(X[:, None, :] ** e, axis=-1)
        return out

    def to_scaled(self, Xi, s):
        return Xi * np.prod(s ** self.E, axis=1)[:, None] / s[None, :]

    def from_scaled(self, Xu, s):
        return Xu / np.prod(s ** self.E, axis=1)[:, None] * s[None, :]

    def coef(self, terms):
        Xi = np.zeros((self.M, self.D))
        for i, eq in enumerate(terms):
            for n, v in eq.items():
                Xi[self.idx[n], i] = v
        return Xi


class Fit:
    def __init__(self, u, C, support, data_chi2, model_chi2, val_chi2, iters, converged):
        self.u, self.C, self.support = u, C, support
        self.data_chi2, self.model_chi2, self.val_chi2 = data_chi2, model_chi2, val_chi2
        self.iters, self.converged = iters, converged

    @property
    def k(self):
        return int(self.support.sum())


def _hw(r, c):
    a = np.abs(r)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))


def _hsum(r, c):
    a = np.abs(r)
    if not np.isfinite(c):
        return float((a ** 2).sum())
    return float(np.where(a <= c, a ** 2, 2 * c * a - c ** 2).sum())


class Solver:
    def __init__(self, lib):
        self.lib = lib
        self.D = lib.D
        self.BW = 2 * lib.D - 1

    def residuals(self, u, C, y, train, dt, sig, rho):
        m = 0.5 * (u[1:] + u[:-1])
        th = self.lib(m)
        rm = (u[1:] - u[:-1] - dt * th @ C) / (rho * sig)
        rd = np.where(train[:, None], (u - y) / sig, 0.0)
        return rd, rm, m, th

    def normal(self, u, C, params, y, train, dt, sig, rho, c=np.inf):
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
        wd = _hw(rd, c)
        ab[BW] += (train[:, None] * wd / sig[None, :] ** 2).ravel()
        gu = np.zeros(n * D)
        gb = np.einsum("kap,ka->kp", B, rm)
        for p in range(2 * D):
            np.add.at(gu, kD + p, gb[:, p])
        gu += (wd * rd / sig).ravel()
        mi, ei = params // D, params % D
        Dm = -dt * th[:, mi] / (rho * sig[ei])[None, :]
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
        cost = _hsum(rd, c) + float((rm ** 2).sum())
        return ab, Auc, Acc, gu, gc, cost

    def schur(self, ab, Auc, Acc, gu, gc, mu):
        ab = ab.copy()
        ab[self.BW] += mu * (1.0 + ab[self.BW])
        acc = Acc + mu * np.diag(1.0 + np.diag(Acc))
        X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
        xg, xc = X[:, 0], X[:, 1:]
        S = acc - Auc.T @ xc
        dc = np.linalg.solve(S + 1e-12 * np.trace(S) / max(len(S), 1) * np.eye(len(S)), -(gc - Auc.T @ xg))
        return -xg - xc @ dc, dc

    def fit(self, y, dt, C0, support, sig, rho, train, u0, max_iter=40, tol=1e-7, valmask=None, c=np.inf):
        params = np.flatnonzero(support.ravel())
        u, C = u0.copy(), np.where(support, C0, 0.0)
        mu, converged, it = 1e-3, False, 0
        ab, Auc, Acc, gu, gc, cost = self.normal(u, C, params, y, train, dt, sig, rho, c)
        for it in range(1, max_iter + 1):
            try:
                du, dc = self.schur(ab, Auc, Acc, gu, gc, mu)
            except (np.linalg.LinAlgError, ValueError):
                mu *= 10
                if mu > 1e8:
                    break
                continue
            un = u + du.reshape(u.shape)
            Cn = C.copy()
            Cn.ravel()[params] += dc
            rd, rm, _, _ = self.residuals(un, Cn, y, train, dt, sig, rho)
            new = _hsum(rd, c) + float((rm ** 2).sum())
            if np.isfinite(new) and new < cost:
                rel = (cost - new) / max(cost, 1e-300)
                u, C = un, Cn
                ab, Auc, Acc, gu, gc, cost = self.normal(u, C, params, y, train, dt, sig, rho, c)
                mu = max(mu / 3, 1e-9)
                if rel < tol:
                    converged = True
                    break
            else:
                mu *= 4
                if mu > 1e8:
                    break
        rd, rm, _, _ = self.residuals(u, C, y, train, dt, sig, rho)
        vm = ~train if valmask is None else valmask
        val = _hsum(((u - y) / sig)[vm], c)
        return Fit(u, C, support.copy(), _hsum(rd, c), float((rm ** 2).sum()), val, it, converged)

    def profiled_quadratic(self, fit, y, dt, sig, rho, train, c=np.inf):
        params = np.arange(self.lib.M * self.D)
        ab, Auc, Acc, gu, gc, _ = self.normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
        X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
        H = Acc - Auc.T @ X[:, 1:]
        g = gc - Auc.T @ X[:, 0]
        H = 0.5 * (H + H.T) + 1e-10 * np.trace(H) / len(H) * np.eye(len(H))
        cstar = fit.C.ravel() - np.linalg.solve(H, g)
        L = np.linalg.cholesky(H)
        return L.T, L.T @ cstar


# Savitzky-Golay window (polyorder 3) of the initial states; final_model.discover(sg_window=...) overrides it
# for the sensitivity sweep (benchmarks/sensitivity_bench.py). 21 is the frozen value.
SG_WINDOW = 21


def _initial_states(y, train):
    u = y.copy()
    idx = np.arange(len(y))
    for i in range(y.shape[1]):
        u[~train, i] = np.interp(idx[~train], idx[train], y[train, i])
    w = min(SG_WINDOW, len(y) - 1 - (len(y) % 2 == 0))
    return savgol_filter(u, w, 3, axis=0)


def auto_substeps(y, dt, target=256, target99=64, cap=12):
    f, f99 = [], []
    for i in range(y.shape[1]):
        s = y[:, i] - y[:, i].mean()
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), dt)
        p[:3] = 0
        f.append(fr[np.argmax(p)])
        floor = np.mean(p[int(0.75 * len(p)):])
        ps = np.where(p > 10 * floor, p - floor, 0.0)
        f99.append(fr[np.searchsorted(np.cumsum(ps) / ps.sum(), 0.99)] if ps.sum() > 0 else f[-1])
    return int(np.clip(max(np.ceil(target * max(f) * dt), np.ceil(target99 * max(f99) * dt), 2), 2, cap))


def _phi(r, kmax=50):
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


def discover(x, dt, starts, lib, *, rho=0.03, rounds=2, kmax=5, keep=2, finalists=10, max_iter=40,
             substeps="auto", substep_cap=12, huber=2.0, selfcal=True, final_rho=0.001, width=60, record=False,
             disc_floor=False):
    """starts: coefficient arrays (lib.M, D) in ORIGINAL units. Returns (Xi, info)."""
    sv = Solver(lib)
    D, NL = lib.D, lib.M
    x = np.asarray(x, float)
    n = len(x)
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    yo = x / scale
    sig = np.maximum(estimate_sigma(yo), 1e-6)
    obs_train = np.ones(n, bool)
    obs_train[2::3] = False
    nobs = n * D
    s_ = auto_substeps(yo, dt, cap=substep_cap) if substeps == "auto" else int(substeps)
    nf = (n - 1) * s_ + 1
    oi = np.arange(n) * s_
    y = np.zeros((nf, D))
    y[oi] = yo
    train = np.zeros(nf, bool)
    train[oi[obs_train]] = True
    valmask = np.zeros(nf, bool)
    valmask[oi[~obs_train]] = True
    dt_f = dt / s_
    u0o = _initial_states(yo, obs_train)
    tf = np.arange(nf) / s_
    u0 = np.column_stack([np.interp(tf, np.arange(n), u0o[:, i]) for i in range(D)])
    allobs = np.zeros(nf, bool)
    allobs[oi] = True
    phi = 1.0

    def score(f):
        return f.data_chi2 + f.model_chi2 + f.val_chi2 + phi * f.k * np.log(nobs)

    fits = {}

    def add(support, C0, u_init):
        key = tuple(np.flatnonzero(support.ravel()))
        if key in fits or not support.any():
            return fits.get(key)
        f = sv.fit(y, dt_f, C0, support, sig, rho, train, u_init, max_iter=max_iter, valmask=valmask, c=huber)
        if np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2):
            fits[key] = f
        return fits.get(key)

    def fit_starts():
        for xi in starts:
            xi = np.asarray(xi, float)
            sup = xi != 0
            if sup.any():
                add(sup, lib.to_scaled(xi, scale), u0)
        if not fits:
            raise RuntimeError("no start model fitted")
        return min(fits.values(), key=score)

    current = fit_starts()
    if selfcal:
        r = yo - current.u[oi]
        sig_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        phi = _phi(r / np.maximum(sig_r, 1e-12))
        if np.any(sig_r > 1.1 * sig):
            sig = np.maximum(sig, sig_r)
            fits.clear()
            current = fit_starts()
    floor_info = None
    if disc_floor:  # v1.2 (see joint_discover.disc_floor_sigma)
        from .joint_discover import disc_floor_sigma
        fl, floor_info = disc_floor_sigma(current.u, current.C, dt_f, rho, phi * np.log(nobs), lib)
        floor_info["active"] = bool(fl is not None and np.any(fl > sig))
        if floor_info["active"]:
            floor_info["factor"] = float(np.max(fl / sig))
            sig = np.maximum(sig, fl)
            fits.clear()
            current = fit_starts()
    for _ in range(rounds):
        A, b = sv.profiled_quadratic(current, y, dt_f, sig, rho, train, huber)
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
        # Cartesian product of each equation's top 2 (2^D); for large D keep the 256 best by the quadratic
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
            # v1.0.1: int dtype so that the empty proposal (key == ()) is scored as the all-zero model
            # instead of raising IndexError (np.array(()) is float); add() never fits empty supports
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
    final = sv.fit(y, dt_f, current.C, current.support, sig, rho, allobs, current.u, max_iter=max_iter,
                   valmask=valmask, c=huber)
    if final_rho is not None:
        stiff = sv.fit(y, dt_f, final.C, current.support, sig, final_rho, allobs, final.u, max_iter=2 * max_iter,
                       valmask=valmask, c=huber)
        if np.isfinite(stiff.data_chi2 + stiff.model_chi2):
            final = stiff
    sc = sorted(score(f) for f in fits.values())
    info = {"models_fitted": len(fits), "sigma": (sig * scale).tolist(), "phi": phi, "substeps": s_,
            "gap": float(sc[1] - sc[0]) if len(sc) > 1 else None}
    if floor_info is not None:
        info["sigma_floor_info"] = floor_info
    if record:
        info["candidates"] = [{"k": f.k, "score": score(f), "xi": lib.from_scaled(f.C, scale).tolist()}
                              for f in fits.values()]
    return lib.from_scaled(final.C, scale), info


# ---------------- weak-form starts ----------------
def weak_system(x, dt, lib, half_width=20, p=4):
    """Weak-form regression problem in scaled states: G (n_test, M), B (n_test, D), scale."""
    x = np.asarray(x, float)
    n, D = x.shape
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
    X = x / scale
    m = min(half_width, (n - 1) // 2)
    n_test = min(max(3 * lib.M, n // max(m // 2, 1)), n - 2 * m)
    centers = np.linspace(m, n - 1 - m, n_test).round().astype(int)
    s = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - s ** 2) ** p
    dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / (m * dt)
    w = np.full(2 * m + 1, dt)
    w[0] = w[-1] = dt / 2
    nrm = (phi * w).sum()
    phi, dphi = phi / nrm, dphi / nrm
    idx = centers[:, None] + np.arange(-m, m + 1)[None, :]
    G = np.einsum("j,kjl->kl", phi * w, lib(X)[idx])
    B = -np.einsum("j,kjd->kd", dphi * w, X[idx])
    return G, B, scale


def weak_starts(x, dt, lib, kmax=5, n_alt=2, width=60):
    """Per-equation weak-form best-subset regression (scaled states), size by BIC on the weak residuals.
    Returns [best, alt1, ...] in original units; each alternative swaps the runner-up size into the
    equation whose BIC choice is least clear-cut."""
    G, B, scale = weak_system(x, dt, lib)
    n_test, D = B.shape
    per_eq = []
    for j in range(D):
        scored = []
        for sup, _ in top_subsets(G, B[:, j], kmax, keep=1, width=width):
            if len(sup) == 0:
                continue
            c = np.linalg.lstsq(G[:, sup], B[:, j], rcond=None)[0]
            rss = float(np.sum((B[:, j] - G[:, sup] @ c) ** 2))
            scored.append((n_test * np.log(max(rss, 1e-300) / n_test) + len(sup) * np.log(n_test), sup, c))
        scored.sort(key=lambda v: v[0])
        per_eq.append(scored)
    best = np.zeros((lib.M, D))
    for j in range(D):
        _, sup, c = per_eq[j][0]
        best[sup, j] = c
    out = [best]
    gaps = [(per_eq[j][1][0] - per_eq[j][0][0], j) for j in range(D) if len(per_eq[j]) > 1]
    for _, j in sorted(gaps)[:n_alt]:
        Xu = best.copy()
        Xu[:, j] = 0
        _, sup, c = per_eq[j][1]
        Xu[sup, j] = c
        out.append(Xu)
    return [lib.from_scaled(Xu, scale) for Xu in out]


def wsindy_stlsq(x, dt, lib, lam, iters=20):
    """WSINDy-style sequentially thresholded least squares on the weak problem (scaled units)."""
    G, B, scale = weak_system(x, dt, lib)
    Xu = np.linalg.lstsq(G, B, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(Xu) < lam
        Xu[small] = 0
        for j in range(B.shape[1]):
            big = ~small[:, j]
            if big.any():
                Xu[big, j] = np.linalg.lstsq(G[:, big], B[:, j], rcond=None)[0]
    return lib.from_scaled(Xu, scale)
