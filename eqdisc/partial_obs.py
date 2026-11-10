"""Profiled joint flow under partial observation (some state components latent).

Same objective, solver and selection as joint_discover / joint_squeeze (config C + degree-weighted prior), with
these changes:

* Data term only for OBSERVED components: residual rows (u_kj - y_kj)/sigma_j exist only for j in `observed`.
  Latent components are constrained by the dynamics residuals alone (their weight uses sigma_lat = mean of the
  observed sigmas, in scaled units).
* Gauge. A latent coordinate is defined only up to an affine change x_L = A xi_L + B x_O + c. Offsets/mixing
  change the support (the sparse search picks among them), but the SCALE of each latent is a continuous symmetry
  of every support, and without fixing it the dynamics residual of a latent equation can be shrunk to zero by
  shrinking the latent. The part of each latent component orthogonal to span{1, observed states} (gauge "proj",
  default) is pinned to unit RMS by a stiff penalty W_g (mean_k e_kj^2 - 1)^2. Its Gauss-Newton term is rank one
  per latent component, handled exactly by Woodbury on top of the banded Cholesky, so a fit stays linear in N.
  (Pinning the raw RMS instead lets the latent shear toward an observed coordinate and soften its own equation.)
* Latent initialisation and restarts. Each restart supplies initial latent trajectories (smoothed derivatives of
  the observed components, phase-randomised surrogates of the observed signals, or user arrays); seeds for the
  coefficients come from weak chi^2 best-subset regression on the pseudo-full trajectory
  [observed, latent init]. The support search (profiled quadratic over ALL coefficients with every state,
  latent included, profiled out; exhaustive top-2 subsets per size and equation; ten finalists; joint refits) then
  runs from each restart, and the best score over all restarts wins. Scores are comparable across restarts (same
  data, same gauge-fixed objective). Restarts (default): derivative proxies, quadratic products of observed
  components, two surrogates.
* Selection score (default "val"): held-out misfit + dynamics misfit + phi * degree-weighted prior BIC; the
  training misfit is left out, because a dense latent model can buy training fit by letting the latent chase
  noise.
* Sparsifying gauge search (regauge): the selected model is mapped exactly through latent changes
  z_L -> s (z_L + A_off z_L + B mono_{<=2}(z_O)) (mixing, shear, offset, and polynomial shifts by observed
  coordinates, which keep a polynomial library closed in cases such as Lorenz z -> z + b x^2); the change that
  minimises the summed relative term contributions is refitted jointly and searched again, and kept if it scores
  better. Without it a gauge-equivalent but denser form is a local optimum of the support search.
* After selection, terms contributing < 0.5% of their equation's RMS right-hand side are pruned and the model
  refitted (dev: spurious extra terms < 0.2%, true terms > 2%).

Optional rho annealing (variational-annealing style homotopy, as in DAHSI) for the start fits: rho_path, e.g.
(1.0, 0.3, 0.1, 0.03), fitted in turn with warm starts.

Truth is never used, except by the explicit diagnostic argument `latent_init` (oracle latent trajectories, for
separating initialisation failures from identifiability failures on dev data).
"""
import itertools

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter

from .core import NLIB, DIM, NAMES, EXP_ARR, estimate_sigma, xi_to_scaled, xi_from_scaled
from .flow_discovery import _top_subsets
from .joint_discover import _hw, _hsum, auto_substeps, _phi, _coeffs_from_states

BW = 2 * DIM - 1
_EMAX = int(EXP_ARR.max())


def _powers(X):
    """(EMAX+1, N, DIM) table of X**p."""
    pw = np.empty((_EMAX + 1,) + X.shape)
    pw[0] = 1.0
    for p in range(1, _EMAX + 1):
        pw[p] = pw[p - 1] * X
    return pw


def library(X):
    """core.library (same values) from a power table: (N, DIM) -> (N, NLIB)."""
    pw = _powers(X)
    out = pw[EXP_ARR[:, 0], :, 0]
    for l in range(1, DIM):
        out = out * pw[EXP_ARR[:, l], :, l]
    return out.T


def dlibrary(X):
    """select_k.dlibrary (same values): d Theta_m / d x_l, (N, NLIB, DIM)."""
    pw = _powers(X)
    out = np.empty((len(X), NLIB, DIM))
    for l in range(DIM):
        e = EXP_ARR.copy()
        a = e[:, l].astype(float)
        e[:, l] = np.maximum(e[:, l] - 1, 0)
        t = a[:, None] * pw[e[:, 0], :, 0]
        for q in range(1, DIM):
            t = t * pw[e[:, q], :, q]
        out[:, :, l] = t.T
    return out
DEGW = 0.2  # complexity prior strength (final_model DEFAULTS["degw"])
PRIOR_W = np.maximum(0.1, 1 + DEGW * (EXP_ARR.sum(1) - 1.5))  # per-term weight, centre 1.5


class POFit:
    def __init__(self, u, C, support, data_chi2, model_chi2, val_chi2, gauge, iters, converged):
        self.u, self.C, self.support = u, C, support
        self.data_chi2, self.model_chi2, self.val_chi2, self.gauge = data_chi2, model_chi2, val_chi2, gauge
        self.iters, self.converged = iters, converged
        self.tag = None

    @property
    def k(self):
        return int(self.support.sum())


# ------------------------------------------------------------------------------------------------
# objective pieces
# ------------------------------------------------------------------------------------------------
def _residuals(u, C, y, dmask, dt, sig, rho):
    m = 0.5 * (u[1:] + u[:-1])
    th = library(m)
    rm = (u[1:] - u[:-1] - dt * th @ C) / (rho * sig)
    rd = np.where(dmask, (u - y) / sig, 0.0)
    return rd, rm, m, th


GAUGE = {"mode": "proj"}  # "proj": unit RMS of the latent's part orthogonal to span{1, observed states}; "rms";
#                          "poly": orthogonal to all library monomials (degree <= 3) of the observed states


def _obs_basis(uo, mode):
    """Orthonormal basis (QR) of span{1, x_O} ("proj") or of every monomial of the observed states up to the
    library degree ("poly"), evaluated along the given trajectory."""
    if mode == "poly":
        no = uo.shape[1]
        exps = sorted({tuple(e) for e in itertools.product(range(_EMAX + 1), repeat=no) if sum(e) <= _EMAX})
        cols = [np.prod(uo ** np.array(e), axis=1) for e in exps]
        # scale columns before QR (powers of unscaled states differ by orders of magnitude)
        M = np.column_stack([c / max(np.sqrt(np.mean(c ** 2)), 1e-300) for c in cols])
    else:
        M = np.column_stack([np.ones(len(uo)), uo])
    Q, _ = np.linalg.qr(M)
    return Q


def _gauge(u, lat, wg):
    """Gauge residuals sqrt(wg)(mean e_j^2 - 1) and their Jacobian columns V (n*DIM, len(lat)).

    mode "proj" (default): e_j = latent j minus its least-squares projection on span{1, observed states}
    (current states). This pins the scale of the part of the latent that is NOT a combination of observed
    coordinates, so shearing the latent toward an observed coordinate (x_L = A xi + B x_O + c with large B) can
    no longer shrink the true latent's share of xi and thereby soften its dynamics constraint. The Jacobian is
    taken with the projection held fixed (exact at the constraint, where the residual is ~0).
    mode "rms": e_j = latent j itself.
    """
    nf = len(u)
    obs = [j for j in range(DIM) if j not in lat]
    if GAUGE["mode"] in ("proj", "poly") and obs:
        Q = _obs_basis(u[:, obs], GAUGE["mode"])
        E = u[:, lat] - Q @ (Q.T @ u[:, lat])
    else:
        E = u[:, lat]
    rg = np.sqrt(wg) * (np.mean(E ** 2, axis=0) - 1.0)
    V = np.zeros((nf * DIM, len(lat)))
    for c, j in enumerate(lat):
        V[j::DIM, c] = np.sqrt(wg) * 2.0 * E[:, c] / nf
    return rg, V


def _tmat(beta, lat):
    """Residual transform T: latent model residuals become r_L - beta r_O (conditional on the observed ones)."""
    T = np.eye(DIM)
    if beta is not None:
        obs = [j for j in range(DIM) if j not in lat]
        T[np.ix_(lat, obs)] = -beta
    return T


def _cost(u, C, y, dmask, dt, sig, rho, c, lat, wg, beta=None):
    rd, rm, _, _ = _residuals(u, C, y, dmask, dt, sig, rho)
    if beta is not None:
        rm = rm @ _tmat(beta, lat).T
    rg, _ = _gauge(u, lat, wg)
    return _hsum(rd[dmask], c) + float((rm ** 2).sum()) + float((rg ** 2).sum())


def _normal(u, C, params, y, dmask, dt, sig, rho, c, lat, wg, beta=None):
    """Gauss-Newton pieces (as joint_discover._normal, per-element data mask, plus gauge rows).

    beta (n_lat, n_obs) or None: with beta the model residuals are r' = T r, (T r)_L = r_L - beta r_O, and the
    parameter vector is [coefficients in `params`, beta.ravel()]. The Gaussian factorisation
    p(r_O) p(r_L | r_O) with a free regression beta makes the objective exactly invariant to shearing a latent
    coordinate by observed ones (x_L -> x_L + b x_O changes r_L by b r_O, which beta absorbs).
    NEGATIVE (dev, not used by default): the model is degenerate -- with Sigma_{L|O} fixed, log det Sigma does not
    depend on beta, and beta -> infinity with r_O = r_L / beta -> 0 absorbs the latent dynamics at no cost (beta ran
    to ~1e3-1e4 and the data misfit grew 40x on Lorenz {x,z}). Kept for the record; shear=False everywhere."""
    n = len(u)
    rd, rm0, m, th = _residuals(u, C, y, dmask, dt, sig, rho)
    J = np.einsum("mi,kml->kil", C, dlibrary(m))
    eye = np.eye(DIM)
    w = 1.0 / (rho * sig)[None, :, None]
    B = np.concatenate([(-eye - 0.5 * dt * J) * w, (eye - 0.5 * dt * J) * w], axis=2)
    T = None
    if beta is not None:
        T = _tmat(beta, lat)
        B = np.einsum("ab,kbp->kap", T, B)
        rm = rm0 @ T.T
    else:
        rm = rm0
    M = np.einsum("kap,kaq->kpq", B, B)
    ab = np.zeros((BW + 1, n * DIM))
    k3 = DIM * np.arange(n - 1)
    for p in range(2 * DIM):
        for q in range(p, 2 * DIM):
            ab[BW + p - q, k3 + q] += M[:, p, q]  # indices unique within one (p, q): same as np.add.at
    wd = _hw(rd, c)
    ab[BW] += (dmask * wd / sig[None, :] ** 2).ravel()
    gu = np.zeros(n * DIM)
    gb = np.einsum("kap,ka->kp", B, rm)
    for p in range(2 * DIM):
        gu[k3 + p] += gb[:, p]
    gu += (dmask * wd * rd / sig).ravel()
    rg, V = _gauge(u, lat, wg)
    gu += V @ rg
    mi, ei = params // DIM, params % DIM
    D = -dt * th[:, mi] / (rho * sig[ei])[None, :]
    P = len(params)
    if T is None:
        Acc = np.zeros((P, P))
        for e in range(DIM):
            s_ = np.flatnonzero(ei == e)
            Acc[np.ix_(s_, s_)] = D[:, s_].T @ D[:, s_]
        Auc = np.zeros((n * DIM, P))
        Bsel = B[:, ei, :] * D[:, :, None]
        for r in range(2 * DIM):
            Auc[k3 + r] += Bsel[:, :, r]
        gc = np.einsum("kp,kp->p", D, rm[:, ei])
    else:
        obs = [j for j in range(DIM) if j not in lat]
        nb = beta.size
        G = np.zeros((n - 1, DIM, P + nb))
        G[:, :, :P] = T[:, ei][None, :, :] * D[:, None, :]  # d(T r)_a / dc_p = T[a, e_p] D_p
        for li, l in enumerate(lat):
            for oj, o in enumerate(obs):
                G[:, l, P + li * len(obs) + oj] = -rm0[:, o]
        Acc = np.einsum("kap,kaq->pq", G, G)
        Auc = np.zeros((n * DIM, P + nb))
        for r in range(2 * DIM):
            Auc[k3 + r] += np.einsum("ka,kap->kp", B[:, :, r], G)
        gc = np.einsum("kap,ka->p", G, rm)
    cost = _hsum(rd[dmask], c) + float((rm ** 2).sum()) + float((rg ** 2).sum())
    return ab, Auc, Acc, gu, gc, cost, V


def _solve_aug(ab, V, R):
    """(A + V V^T)^{-1} R with A banded (Woodbury on one banded Cholesky)."""
    r = R.shape[1]
    Y = solveh_banded(ab, np.column_stack([R, V]), check_finite=False)
    if V.shape[1] == 0:
        return Y
    XR, XV = Y[:, :r], Y[:, r:]
    Mw = np.eye(V.shape[1]) + V.T @ XV
    return XR - XV @ np.linalg.solve(Mw, V.T @ XR)


def _step(ab, Auc, Acc, gu, gc, V, mu):
    ab = ab.copy()
    ab[BW] += mu * (1.0 + ab[BW])
    acc = Acc + mu * np.diag(1.0 + np.diag(Acc))
    X = _solve_aug(ab, V, np.column_stack([gu, Auc]))
    xg, xc = X[:, 0], X[:, 1:]
    S = acc - Auc.T @ xc
    dc = np.linalg.solve(S + 1e-12 * np.trace(S) / max(len(S), 1) * np.eye(len(S)), -(gc - Auc.T @ xg))
    du = -xg - xc @ dc
    return du, dc


def _beta0(u, C, y, dmask, dt, sig, rho, lat):
    """Least-squares regression of the latent model residuals on the observed ones (n_lat, n_obs)."""
    obs = [j for j in range(DIM) if j not in lat]
    _, rm, _, _ = _residuals(u, C, y, dmask, dt, sig, rho)
    return np.linalg.lstsq(rm[:, obs], rm[:, lat], rcond=None)[0].T


def fit_po(y, dt, C0, support, sig, rho, dmask, u0, valmask, lat, *, max_iter=40, tol=1e-7, c=2.0, wg=1e6,
           shear=False):
    """LM on all states + coefficients on `support` (+ the residual regression beta if shear); data rows only
    where dmask (nf, DIM)."""
    params = np.flatnonzero(support.ravel())
    u, C = u0.copy(), np.where(support, C0, 0.0)
    beta = _beta0(u, C, y, dmask, dt, sig, rho, lat) if (shear and 0 < len(lat) < DIM) else None
    mu, converged, it = 1e-3, False, 0
    ab, Auc, Acc, gu, gc, cost, V = _normal(u, C, params, y, dmask, dt, sig, rho, c, lat, wg, beta)
    P = len(params)
    for it in range(1, max_iter + 1):
        try:
            du, dc = _step(ab, Auc, Acc, gu, gc, V, mu)
        except (np.linalg.LinAlgError, ValueError):
            mu *= 10
            if mu > 1e8:
                break
            continue
        un = u + du.reshape(u.shape)
        Cn = C.copy()
        Cn.ravel()[params] += dc[:P]
        bn = None if beta is None else beta + dc[P:].reshape(beta.shape)
        with np.errstate(all="ignore"):
            new = _cost(un, Cn, y, dmask, dt, sig, rho, c, lat, wg, bn)
        if np.isfinite(new) and new < cost:
            rel = (cost - new) / max(cost, 1e-300)
            u, C, beta = un, Cn, bn
            ab, Auc, Acc, gu, gc, cost, V = _normal(u, C, params, y, dmask, dt, sig, rho, c, lat, wg, beta)
            mu = max(mu / 3, 1e-9)
            if rel < tol:
                converged = True
                break
        else:
            mu *= 4
            if mu > 1e8:
                break
    rd, rm, _, _ = _residuals(u, C, y, dmask, dt, sig, rho)
    if beta is not None:
        rm = rm @ _tmat(beta, lat).T
    rg, _ = _gauge(u, lat, wg)
    val = _hsum(((u - y) / sig)[valmask], c)
    f = POFit(u, C, support.copy(), _hsum(rd[dmask], c), float((rm ** 2).sum()), val, float((rg ** 2).sum()),
              it, converged)
    f.beta = beta
    return f


def profiled_quadratic(fit, y, dt, sig, rho, dmask, lat, c=2.0, wg=1e6):
    """(A, b) of the profiled quadratic in ALL NLIB*DIM coefficients, every state (latent too) profiled out
    (and the residual regression beta, if the fit has one)."""
    params = np.arange(NLIB * DIM)
    beta = getattr(fit, "beta", None)
    ab, Auc, Acc, gu, gc, _, V = _normal(fit.u, fit.C, params, y, dmask, dt, sig, rho, c, lat, wg, beta)
    # no state ridge: a ridge eps*mean(diag) is NOT negligible for latent gauge moves (a constant latent shift
    # touches every grid state; at 1% noise eps = 1e-10 cost ~200 and hid every minimal-offset support); only
    # used if the banded Cholesky fails
    try:
        X = _solve_aug(ab, V, np.column_stack([gu, Auc]))
    except (np.linalg.LinAlgError, ValueError):
        ab = ab.copy()
        ab[BW] += 1e-12 * np.mean(ab[BW])
        X = _solve_aug(ab, V, np.column_stack([gu, Auc]))
    H = Acc - Auc.T @ X[:, 1:]
    g = gc - Auc.T @ X[:, 0]
    H = 0.5 * (H + H.T)
    P = NLIB * DIM
    if beta is not None:  # profile beta out as well (second Schur complement)
        Hbb = H[P:, P:] + 1e-10 * np.trace(H[P:, P:]) / max(len(H) - P, 1) * np.eye(len(H) - P)
        Hcb = H[:P, P:]
        K = np.linalg.solve(Hbb, np.column_stack([Hcb.T, g[P:]]))
        H = H[:P, :P] - Hcb @ K[:, :P]
        g = g[:P] - Hcb @ K[:, P]
        H = 0.5 * (H + H.T)
    H = H + 1e-10 * np.trace(H) / len(H) * np.eye(len(H))
    cstar = fit.C.ravel() - np.linalg.solve(H, g)
    L = np.linalg.cholesky(H)
    return L.T, L.T @ cstar


# ------------------------------------------------------------------------------------------------
# latent initialisations (sample grid, scaled units, unit RMS)
# ------------------------------------------------------------------------------------------------
def _unit(v):
    v = v - v.mean()
    return v / max(np.sqrt(np.mean(v ** 2)), 1e-12)


def _smooth_obs(yo, obs_train, w=21):
    u = yo.copy()
    idx = np.arange(len(yo))
    for i in range(yo.shape[1]):
        u[~obs_train, i] = np.interp(idx[~obs_train], idx[obs_train], yo[obs_train, i])
    w = min(w, len(yo) - 1 - (len(yo) % 2 == 0))
    return savgol_filter(u, w, 3, axis=0)


def deriv_proxy(yo_obs, dt, order, comp, w=31):
    """order-th derivative of observed column `comp` (Savitzky-Golay, window w, cubic/quartic)."""
    w = min(w, len(yo_obs) - 1 - (len(yo_obs) % 2 == 0))
    return savgol_filter(yo_obs[:, comp], w, max(3, order + 2), deriv=order, delta=dt)


def surrogate(sig_ref, rng):
    """Phase-randomised surrogate of a (smoothed) reference signal: same spectrum, random phases."""
    F = np.fft.rfft(sig_ref - sig_ref.mean())
    ph = np.exp(2j * np.pi * rng.random(len(F)))
    ph[0] = 1.0
    return np.fft.irfft(F * ph, n=len(sig_ref))


def init_bank(yo_obs, dt, n_lat, kinds, rng, n_random=2):
    """List of (tag, latent init (n, n_lat)) restarts."""
    n, n_obs = yo_obs.shape
    out = []
    if "deriv" in kinds:
        if n_lat == 1:
            for i in range(n_obs):
                out.append((f"d1(o{i})", _unit(deriv_proxy(yo_obs, dt, 1, i))[:, None]))
        else:
            # one observed component typically: successive derivatives (Takens-style derivative coordinates)
            for i in range(n_obs):
                cols = [_unit(deriv_proxy(yo_obs, dt, d + 1, i, w=41)) for d in range(n_lat)]
                out.append((f"d12(o{i})", np.column_stack(cols)))
    if "quad" in kinds:
        # quadratic features of the observed components (a latent often enters as / tracks a product of observed
        # ones, e.g. energy-like coordinates); for two latents, mixed pairs with the derivative coordinates
        sm = savgol_filter(yo_obs, min(21, n - 1 - (n % 2 == 0)), 3, axis=0)
        if n_lat == 1:
            for i in range(n_obs):
                for k in range(i, n_obs):
                    out.append((f"q(o{i}o{k})", _unit(sm[:, i] * sm[:, k])[:, None]))
        else:
            for i in range(n_obs):
                q = _unit(sm[:, i] ** 2)
                d1 = _unit(deriv_proxy(yo_obs, dt, 1, i, w=41))
                d2 = _unit(deriv_proxy(yo_obs, dt, 2, i, w=41))
                out.append((f"d1q(o{i})", np.column_stack([d1, q] + [q] * (n_lat - 2))))
                out.append((f"qd2(o{i})", np.column_stack([q, d2] + [q] * (n_lat - 2))))
    if "random" in kinds:
        for r in range(n_random):
            cols = [_unit(surrogate(yo_obs[:, rng.integers(n_obs)], rng)) for _ in range(n_lat)]
            out.append((f"rand{r}", np.column_stack(cols)))
    return out


# ------------------------------------------------------------------------------------------------
# problem set-up (shared by discover and the DAHSI-style baseline)
# ------------------------------------------------------------------------------------------------
class Problem:
    """Scaled data on the latent grid, masks, noise scales and the selection score for one dataset."""

    def __init__(self, y_obs, dt, observed, *, rho=0.03, huber=2.0, wg=1e6, prior=True, substeps="auto",
                 shear=False, lat_sig=1.0, score_rule="val", gauge="proj"):
        y_obs = np.asarray(y_obs, float)
        self.observed = list(observed)
        self.lat = [j for j in range(DIM) if j not in self.observed]
        self.n = n = len(y_obs)
        self.dt, self.rho, self.huber, self.wg, self.shear = dt, rho, huber, wg, shear
        sc_obs = np.maximum(np.sqrt(np.mean(y_obs ** 2, axis=0)), 1e-10)
        self.yo_obs = yo_obs = y_obs / sc_obs
        self.scale = np.ones(DIM)
        self.scale[self.observed] = sc_obs
        self.sig_obs = np.maximum(estimate_sigma(yo_obs), 1e-6)
        self.lat_sig, self.score_rule = lat_sig, score_rule
        if gauge == "auto":  # dev D: "poly" for one latent, "proj" for two (see GAUGE)
            gauge = "poly" if len(self.lat) == 1 else "proj"
        GAUGE["mode"] = gauge  # module-level switch read by _gauge (one Problem at a time per process)
        self.sig = np.full(DIM, lat_sig * float(np.mean(self.sig_obs)))
        self.sig[self.observed] = self.sig_obs
        self.sig_hf = self.sig.copy()
        self.obs_train = np.ones(n, bool)
        self.obs_train[2::3] = False
        self.nobs = n * len(self.observed)
        self.n_scale = n / max(int((~self.obs_train).sum()), 1)
        self.s = s_ = auto_substeps(yo_obs, dt) if substeps == "auto" else int(substeps)
        self.nf = nf = (n - 1) * s_ + 1
        self.oi = oi = np.arange(n) * s_
        self.dt_f = dt / s_
        self.tf = np.arange(nf) / s_
        ob = np.array(self.observed)[None, :]
        self.y = np.zeros((nf, DIM))
        self.y[oi[:, None], ob] = yo_obs
        self.dmask = np.zeros((nf, DIM), bool)
        self.dmask[oi[self.obs_train][:, None], ob] = True
        self.valmask = np.zeros((nf, DIM), bool)
        self.valmask[oi[~self.obs_train][:, None], ob] = True
        self.allmask = np.zeros((nf, DIM), bool)
        self.allmask[oi[:, None], ob] = True
        self.sm_obs = _smooth_obs(yo_obs, self.obs_train)
        self.w_pen = PRIOR_W if prior else np.ones(NLIB)
        self.phi = 1.0
        self.fits = 0

    def pen(self, sup):
        return float((sup * self.w_pen[:, None]).sum()) * np.log(self.nobs)

    def score(self, f):
        """"sum": train + dynamics + held-out misfit + prior BIC (as joint_discover / final_model).
        "pred": held-out misfit scaled to all observations (n_obs / n_val) replaces train + held-out, so a latent
        path that chases training noise is charged by its held-out error, not rewarded by its training fit.
        "val": as "pred" without the n_obs / n_val factor (that factor also triples the held-out noise, so a
        negligible extra term can win by chance; see the Lorenz xz unit test)."""
        if self.score_rule == "val":
            return f.val_chi2 + f.model_chi2 + self.phi * self.pen(f.support)
        if self.score_rule == "pred":
            return self.n_scale * f.val_chi2 + f.model_chi2 + self.phi * self.pen(f.support)
        return f.data_chi2 + f.model_chi2 + f.val_chi2 + self.phi * self.pen(f.support)

    def to_grid(self, z):
        return np.column_stack([np.interp(self.tf, np.arange(self.n), z[:, i]) for i in range(z.shape[1])])

    def pseudo(self, Lz):
        """Pseudo-full trajectory at the samples: smoothed observed + latent init (scaled units)."""
        Z = np.zeros((self.n, DIM))
        Z[:, self.observed] = self.sm_obs
        Z[:, self.lat] = Lz
        if GAUGE["mode"] in ("proj", "poly"):
            Q = _obs_basis(self.sm_obs, GAUGE["mode"])
            for j in self.lat:
                e = Z[:, j] - Q @ (Q.T @ Z[:, j])
                Z[:, j] = Z[:, j] / max(np.sqrt(np.mean(e ** 2)), 1e-12)
        return Z

    def fit(self, C0, sup, u_init, rho=None, iters=40, mask=None):
        self.fits += 1
        with np.errstate(all="ignore"):
            f = fit_po(self.y, self.dt_f, C0, sup, self.sig, self.rho if rho is None else rho,
                       self.dmask if mask is None else mask, u_init, self.valmask, self.lat, max_iter=iters,
                       c=self.huber, wg=self.wg, shear=self.shear)
        ok = np.isfinite(f.data_chi2 + f.model_chi2 + f.val_chi2) and np.all(np.isfinite(f.C))
        return f if ok else None

    def calibrate(self, best):
        """Self-calibration from a fit: sigma (observed) from MAD of sample residuals if > 1.1x the
        high-frequency estimate, phi from their autocorrelation. Returns True if sigma changed."""
        r = self.yo_obs - best.u[self.oi][:, self.observed]
        sig_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        self.phi = _phi(r / np.maximum(sig_r, 1e-12))
        if np.any(sig_r > 1.1 * self.sig_obs):
            self.sig_obs = np.maximum(self.sig_obs, sig_r)
            self.sig[self.observed] = self.sig_obs
            self.sig[self.lat] = self.lat_sig * float(np.mean(self.sig_obs))
            return True
        return False

    def finish(self, current, max_iter=40, final_rho=0.001):
        """All-sample refit, then stiff-rho refit (coefficients only). Returns (xi original units, u at samples)."""
        final = self.fit(current.C, current.support, current.u, None, max_iter, mask=self.allmask) or current
        if final_rho is not None:
            stiff = self.fit(final.C, current.support, final.u, final_rho, 2 * max_iter, mask=self.allmask)
            if stiff is not None:
                final = stiff
        self.final_fit = final
        return xi_from_scaled(final.C, self.scale), final.u[self.oi] * self.scale


def term_contrib(C, u):
    """Relative contribution of every library term: RMS of c_me theta_m(u) over the trajectory divided by the RMS of
    equation e's right-hand side (invariant to rescaling the coordinates). Inf where the term is absent."""
    Th = library(u)
    out = np.full(C.shape, np.inf)
    for e in range(DIM):
        m = np.flatnonzero(C[:, e])
        if not len(m):
            continue
        rhs = np.sqrt(np.mean((Th @ C[:, e]) ** 2))
        out[m, e] = np.sqrt(np.mean((Th[:, m] * C[m, e]) ** 2, axis=0)) / max(rhs, 1e-300)
    return out


def _regauge_map(u, C, lat, obs, exps, th, Qb=None):
    """Apply the latent change z'_L = s * (z_L + A_off z_L + B mono(z_O)) to states u and to the model's vector
    field; returns (z', f'(z')) with f' the transformed right-hand side at the same points."""
    nl = len(lat)
    f = library(u) @ C
    z, g = u.copy(), f.copy()
    npl = (nl - 1) + len(exps)
    mono = np.column_stack([np.prod(u[:, obs] ** np.array(e), axis=1) for e in exps])
    dmono = np.zeros_like(mono)  # d/dt mono = grad mono . f_O
    for m, e in enumerate(exps):
        for a, o in enumerate(obs):
            if e[a]:
                ee = np.array(e)
                ee[a] -= 1
                dmono[:, m] += e[a] * np.prod(u[:, obs] ** ee, axis=1) * f[:, o]
    for li, j in enumerate(lat):
        p = th[li * npl:(li + 1) * npl]
        others = [k for k in lat if k != j]
        z[:, j] = u[:, j] + (u[:, others] @ p[:nl - 1] if others else 0.0) + mono @ p[nl - 1:]
        g[:, j] = f[:, j] + (f[:, others] @ p[:nl - 1] if others else 0.0) + dmono @ p[nl - 1:]
    if Qb is not None:  # re-pin the scale of each latent's part orthogonal to span{1, observed}
        for j in lat:
            e = z[:, j] - Qb @ (Qb.T @ z[:, j])
            sj = 1.0 / max(np.sqrt(np.mean(e ** 2)), 1e-12)
            z[:, j] *= sj
            g[:, j] *= sj
    return z, g


def regauge(P, cur, deg=2, prune=5e-3, max_fev=3000, polish=0.02):
    """Sparsifying gauge search: look for a latent change z_L -> s (z_L + A_off z_L + B mono_{<=deg}(z_O)) under
    which the fitted model has an exact polynomial form (library closed) that is sparser (smaller sum of relative
    term contributions); refit that form jointly. Returns the refit or None. A gauge-equivalent but denser form
    (latent sheared toward observed coordinates or shifted by a polynomial of them) is otherwise a local optimum
    of the support search, because moving between gauges needs latent states and coefficients to change together."""
    from scipy.optimize import minimize
    lat, obs = P.lat, P.observed
    if not lat or not obs:
        return None
    u, C = cur.u, cur.C
    sub = u[::max(1, len(u) // 600)]
    exps = [e for e in itertools.product(range(deg + 1), repeat=len(obs)) if sum(e) <= deg]
    npar = len(lat) * ((len(lat) - 1) + len(exps))
    Qs = np.linalg.qr(np.column_stack([np.ones(len(sub)), sub[:, obs]]))[0]

    def form(th, pts, Qb):
        z, g = _regauge_map(pts, C, lat, obs, exps, th, Qb)
        Th = library(z)
        Cn = np.linalg.lstsq(Th, g, rcond=None)[0]
        res = np.linalg.norm(Th @ Cn - g) / max(np.linalg.norm(g), 1e-300)
        return z, Cn, Th, g, res

    def obj(th):
        with np.errstate(all="ignore"):
            try:
                z, Cn, Th, g, res = form(th, sub, Qs)
            except (np.linalg.LinAlgError, ValueError):
                return 1e9
        rt = np.sqrt(np.mean(Th ** 2, axis=0))[:, None]
        rg = np.maximum(np.sqrt(np.mean(g ** 2, axis=0)), 1e-300)[None, :]
        v = float((np.abs(Cn) * rt / rg).sum()) + 1e3 * res
        return v if np.isfinite(v) else 1e9

    th0 = np.zeros(npar)
    o0 = obj(th0)
    sol = minimize(obj, th0, method="Powell", options={"maxfev": max_fev, "xtol": 1e-6, "ftol": 1e-9})
    if not (sol.fun < o0 - 0.05):
        return None
    # polish (Powell on an L1 objective stops short): drive the small terms exactly to zero by least squares on
    # their relative contributions (smooth), keeping the transformation exact
    from scipy.optimize import least_squares

    def contrib_rows(th):
        z, Cn, Th, g, res = form(th, sub, Qs)
        rt = np.sqrt(np.mean(Th ** 2, axis=0))[:, None]
        rg = np.maximum(np.sqrt(np.mean(g ** 2, axis=0)), 1e-300)[None, :]
        return np.abs(Cn) * rt / rg, Cn * rt / rg, res

    with np.errstate(all="ignore"):
        try:
            c_abs, _, _ = contrib_rows(sol.x)
            drop = c_abs < polish

            def r(th):
                _, c_sgn, res = contrib_rows(th)
                return np.concatenate([c_sgn[drop], [1e3 * res]])

            th_star = least_squares(r, sol.x, method="lm", max_nfev=2000).x
        except (np.linalg.LinAlgError, ValueError):
            th_star = sol.x
    sol_x = th_star if obj(th_star) <= sol.fun + 1e-9 else sol.x
    Qf = np.linalg.qr(np.column_stack([np.ones(len(u)), u[:, obs]]))[0]
    with np.errstate(all="ignore"):
        z, Cn, _, _, res = form(sol_x, u, Qf)
    if not (res < 1e-4) or not np.all(np.isfinite(z)):
        return None
    sup = term_contrib(np.where(np.abs(Cn) > 0, Cn, 0.0), z) >= prune
    sup &= np.isfinite(term_contrib(Cn, z))
    if not sup.any(0).all():
        return None
    return P.fit(np.where(sup, Cn, 0.0), sup, z)


# ------------------------------------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------------------------------------
def discover(y_obs, dt, observed, *, kinds=("deriv", "quad", "random"), n_random=2, latent_init=None, seed=0,
             rho=0.03, rho_path=None, rounds=4, kmax=5, keep=2, finalists=10, max_iter=40, start_iter=150,
             huber=2.0, prior=True, final_rho=0.001, wg=1e6, substeps="auto", shear=False, lat_sig=1.0,
             score_rule="val", gauge="proj", prune=5e-3, regauge_deg=2, record=False, verbose=False):
    """y_obs (n, len(observed)) noisy observed components; `observed` = indices into the DIM-state.

    Returns (xi (NLIB, DIM): observed components in original units, latent components in the method's gauge
    (unit RMS on the latent grid), info) with info["u"] the estimated states at the samples (same units).
    """
    P = Problem(y_obs, dt, observed, rho=rho, huber=huber, wg=wg, prior=prior, substeps=substeps, shear=shear,
                lat_sig=lat_sig, score_rule=score_rule, gauge=gauge)
    lat, n = P.lat, P.n
    rng = np.random.default_rng(seed)
    score = P.score

    bank = []
    if latent_init is not None:
        li = np.asarray(latent_init, float).reshape(n, len(lat))
        bank.append(("oracle", np.column_stack([_unit(li[:, c]) for c in range(len(lat))])))
    bank += init_bank(P.yo_obs, dt, len(lat), kinds, rng, n_random)

    def start_fits(tag, Lz):
        from . import select_k
        Z = P.pseudo(Lz)
        u0 = P.to_grid(Z)
        seeds = []
        try:
            w = select_k.discover(Z, dt, criterion=["chi2s", "chi2", "bic"])
            seeds += [w["chi2s"], w["chi2"], w["bic"]]
        except Exception:
            pass
        out = {}
        for xi in seeds:
            sup = np.asarray(xi) != 0
            if not sup.any(0).all():
                continue
            key = tuple(np.flatnonzero(sup.ravel()))
            if key in out:
                continue
            C0 = _coeffs_from_states(u0, P.dt_f, sup)
            f = None
            for r in (rho_path or [rho]):
                f = P.fit(C0 if f is None else f.C, sup, u0 if f is None else f.u, r, start_iter)
                if f is None:
                    break
            if f is not None:
                f.tag = tag
                out[key] = f
        return out

    def propose(current, fits):
        A, b = profiled_quadratic(current, P.y, P.dt_f, P.sig, rho, P.dmask, lat, huber, wg)
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
                cc = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
                rk.append((np.sum((b - A[:, idx] @ cc) ** 2) + len(idx) * np.log(P.nobs), sup))
            ranked_eq.append([s for _, s in sorted(rk, key=lambda v: v[0])[:2]])
        for combo in itertools.product(*ranked_eq):
            mask = np.zeros_like(cur)
            for j, sup in enumerate(combo):
                mask[sup, j] = True
            proposals[tuple(np.flatnonzero(mask.ravel()))] = mask
        ranked = []
        for key, mask in proposals.items():
            if key in fits or not mask.any(0).all():
                continue
            idx = np.array(key, dtype=int)
            cc = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            ranked.append((np.sum((b - A[:, idx] @ cc) ** 2) + len(idx) * np.log(P.nobs), key, mask, cc))
        return sorted(ranked, key=lambda v: v[0])[:finalists]

    def search(current, fits, tag):
        for _ in range(rounds):
            try:
                props = propose(current, fits)
            except (np.linalg.LinAlgError, ValueError):
                break
            for _, key, mask, cc in props:
                C0 = np.zeros(NLIB * DIM)
                C0[list(key)] = cc
                f = P.fit(C0.reshape(NLIB, DIM), mask, current.u, rho, max_iter)
                if f is not None:
                    f.tag = tag
                    fits[key] = f
            best = min(fits.values(), key=score)
            if best is current:
                break
            current = best
        return current

    # 1) start fits for every restart; self-calibrate sigma and phi from the best start overall
    per_init = [(tag, Lz, start_fits(tag, Lz)) for tag, Lz in bank]
    allstarts = [f for _, _, fs in per_init for f in fs.values()]
    if not allstarts:
        raise RuntimeError("no start model fitted")
    if P.calibrate(min(allstarts, key=score)):
        per_init = [(tag, Lz, start_fits(tag, Lz)) for tag, Lz, _ in per_init]

    # 2) support search from each restart; pool keeps the best fit of every support over restarts
    pool, trace = {}, []
    for tag, Lz, fits in per_init:
        if not fits:
            continue
        current = search(min(fits.values(), key=score), fits, tag)
        trace.append({"init": tag, "best": float(score(current)), "k": current.k, "fits": len(fits)})
        if verbose:
            print(tag, round(score(current), 2), current.k, len(fits), flush=True)
        for key, f in fits.items():
            if key not in pool or score(f) < score(pool[key]):
                pool[key] = f
    ranked = sorted(pool.values(), key=score)
    current = ranked[0]
    gap = score(ranked[1]) - score(current) if len(ranked) > 1 else np.inf
    regauged = None
    if regauge_deg is not None and regauge_deg >= 0:
        g = regauge(P, current, deg=regauge_deg, prune=prune or 0.0)
        if g is not None:
            g.tag = current.tag + "+regauge"
            g = search(g, {tuple(np.flatnonzero(g.support.ravel())): g}, g.tag)  # support search in the new gauge
            if score(g) < score(current):
                regauged = {"from_k": current.k, "to_k": g.k, "d_score": float(score(g) - score(current))}
                current = g
    pruned = []
    xi, u = P.finish(current, max_iter, final_rho)
    while prune:
        # negligible-term pruning on the final (stiff) fit; dev D/E: spurious extra terms contribute < 0.3% of
        # their equation's RMS right-hand side, true terms > 2%
        fin = P.final_fit
        small = np.argwhere(fin.support & (term_contrib(fin.C, fin.u) < prune))
        if not len(small):
            break
        sup = current.support.copy()
        sup[tuple(small.T)] = False
        if not sup.any(0).all():
            break
        f = P.fit(np.where(sup, current.C, 0.0), sup, current.u, rho, max_iter)
        if f is None:
            break
        pruned += [(NAMES[m], int(e)) for m, e in small]
        f.tag = current.tag
        current = f
        xi, u = P.finish(current, max_iter, final_rho)
    info = {"models_fitted": P.fits, "sigma": (P.sig * P.scale).tolist(), "sigma_hf": (P.sig_hf * P.scale).tolist(),
            "phi": P.phi, "substeps": P.s, "gap": float(gap), "ambiguous": bool(gap < 2.0), "init": current.tag,
            "score": float(score(current)), "pruned": pruned, "regauged": regauged,
            "u": u.tolist(), "trace": trace, "observed": P.observed, "latent": lat}
    if record:
        info["candidates"] = [{"score": float(score(f)), "k": f.k, "init": f.tag, "data": f.data_chi2,
                               "model": f.model_chi2, "val": f.val_chi2, "pen": P.phi * P.pen(f.support),
                               "xi": xi_from_scaled(f.C, P.scale).tolist()} for f in ranked[:20]]
    return xi, info
