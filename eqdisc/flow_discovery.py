"""Sparse equation discovery using profiled short-trajectory sensitivities.

Weak integrals initialise a sparse model. Its flow Jacobian then supplies a
different regression dictionary: the effect of each library term on observed
trajectories, projected off uncertain segment initial states. Sparse proposals
are verified by nonlinear trajectory fits. Non-overlapping held-out segments
rank the resulting models and describe unresolved alternatives.

This is a research prototype for fully observed, uniformly sampled polynomial
ODEs with independent measurement noise. Segment starts are independent nuisance
parameters: this is a segmented output-error objective, not continuity-constrained
multiple shooting. See FLOW_DISCOVERY.md for assumptions and prior art.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.signal import savgol_filter
from numba import njit

from .core import (DIM, NLIB, EXP_ARR, NAMES, library, estimate_sigma,
                  xi_from_scaled, xi_to_scaled)
from .select_k import weak_mats, dlibrary, resid_cov, fit_eq
from .subset import combos


@njit(cache=True)
def _augmented_rhs(state, tangent, xi, exps, terms, equations):
    d, q = tangent.shape
    f = np.zeros(d)
    jac = np.zeros((d, d))
    theta = np.ones(len(exps))
    for m in range(len(exps)):
        for a in range(d):
            theta[m] *= state[a] ** exps[m, a]
        for j in range(d):
            f[j] += theta[m] * xi[m, j]
        for a in range(d):
            if exps[m, a] == 0:
                continue
            deriv = float(exps[m, a])
            for b in range(d):
                deriv *= state[b] ** (exps[m, b] - (1 if a == b else 0))
            for j in range(d):
                jac[j, a] += xi[m, j] * deriv
    ds = jac @ tangent
    for p in range(len(terms)):
        ds[equations[p], p] += theta[terms[p]]
    return f, ds


@njit(cache=True)
def _integrate(initial, xi, exps, terms, equations, dt, length, substeps):
    """RK4 states and analytic coefficient/initial-state sensitivities."""
    nw, d = initial.shape
    p = len(terms)
    states = np.zeros((nw, length, d))
    derivs = np.zeros((nw, length, d, p + d))
    h = dt / substeps
    for w in range(nw):
        x = initial[w].copy()
        tangent = np.zeros((d, p + d))
        for j in range(d):
            tangent[j, p + j] = 1.0
        for t in range(length):
            states[w, t] = x
            derivs[w, t] = tangent
            if t == length - 1:
                break
            for _ in range(substeps):
                k1, s1 = _augmented_rhs(x, tangent, xi, exps, terms, equations)
                k2, s2 = _augmented_rhs(x + h*k1/2, tangent + h*s1/2, xi, exps, terms, equations)
                k3, s3 = _augmented_rhs(x + h*k2/2, tangent + h*s2/2, xi, exps, terms, equations)
                k4, s4 = _augmented_rhs(x + h*k3, tangent + h*s3, xi, exps, terms, equations)
                x += h * (k1 + 2*k2 + 2*k3 + k4) / 6
                tangent += h * (s1 + 2*s2 + 2*s3 + s4) / 6
                if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
                    states[:] = np.nan
                    derivs[:] = np.nan
                    return states, derivs
    return states, derivs


def integrate(initial, xi, dt, length, active=None, substeps=2):
    """Return states and sensitivities, with initial-state columns last.

    ``active`` contains flattened coefficient indices (term * DIM + equation).
    It may include coefficients currently zero, to test adding those terms.
    """
    active = np.flatnonzero(xi.ravel()) if active is None else np.asarray(active, dtype=int)
    return _integrate(np.ascontiguousarray(initial, dtype=float),
                      np.ascontiguousarray(xi, dtype=float), EXP_ARR,
                      active // DIM, active % DIM, float(dt), int(length), int(substeps))


def project_initial_states(residual, parameter_jac, initial_jac):
    """Whiten first; orthogonally eliminate one initial state per segment.

    This is the Schur complement of the joint Gauss--Newton problem, evaluated
    using QR projections rather than forming squared-condition-number matrices.
    """
    projected_r, projected_j = [], []
    for r, a, z in zip(residual, parameter_jac, initial_jac):
        q, _ = np.linalg.qr(z, mode="reduced")
        projected_r.append(r - q @ (q.T @ r))
        projected_j.append(a - q @ (q.T @ a))
    return np.concatenate(projected_r), np.concatenate(projected_j)


@dataclass
class TrajectoryFit:
    xi: np.ndarray
    initial: np.ndarray
    rss: float
    block_rss: np.ndarray
    iterations: int
    converged: bool


def _initial_guess(blocks):
    window = min(7, blocks.shape[1] if blocks.shape[1] % 2 else blocks.shape[1] - 1)
    return savgol_filter(blocks, window, 2, axis=1)[:, 0].copy()


def fit_trajectories(blocks, dt, xi, sigma, support=None, initial=None,
                     max_iter=25, substeps=2, fit_coefficients=True):
    """Fit fixed-support coefficients and free segment starts to raw states.

    Coefficients may be frozen for held-out evaluation. No observed start is
    treated as noiseless. Each observation contributes exactly once.
    """
    blocks = np.asarray(blocks, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if blocks.ndim != 3 or blocks.shape[2] != DIM or blocks.shape[1] < 3:
        raise ValueError("blocks must have shape (segments, length >= 3, DIM)")
    if not np.isfinite(blocks).all() or sigma.shape != (DIM,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("finite observations and positive per-coordinate sigma are required")
    if not np.isfinite(dt) or dt <= 0 or substeps < 1:
        raise ValueError("dt and substeps must be positive")
    xi = np.asarray(xi, dtype=float).copy()
    mask = xi != 0 if support is None else np.asarray(support, dtype=bool)
    if xi.shape != (NLIB, DIM) or mask.shape != xi.shape or not np.isfinite(xi).all():
        raise ValueError("xi and support must have shape (NLIB, DIM), with finite coefficients")
    xi[~mask] = 0
    active = np.flatnonzero(mask.ravel()) if fit_coefficients else np.empty(0, dtype=int)
    p = len(active)
    z = _initial_guess(blocks) if initial is None else np.asarray(initial, dtype=float).copy()
    nblocks, length, d = blocks.shape
    if z.shape != (nblocks, d) or not np.isfinite(z).all() or max_iter < 1:
        raise ValueError("initial must contain one finite state per segment; max_iter must be positive")
    damping = 1e-3
    converged = False
    iteration = 0
    for iteration in range(max_iter):
        pred, deriv = integrate(z, xi, dt, length, active, substeps)
        if not np.isfinite(pred).all() or not np.isfinite(deriv).all():
            break
        r = ((pred - blocks) / sigma).reshape(nblocks, -1)
        a = (deriv[..., :p] / sigma[None, None, :, None]).reshape(nblocks, length*d, p)
        u = (deriv[..., p:] / sigma[None, None, :, None]).reshape(nblocks, length*d, d)
        old_rss = np.sum(r*r)
        rp, ap = project_initial_states(r, a, u)
        if p:
            scales = np.maximum(np.linalg.norm(ap, axis=0), 1e-10)
            an = ap / scales
            delta = np.linalg.lstsq(np.vstack((an, np.sqrt(damping)*np.eye(p))),
                                    np.concatenate((-rp, np.zeros(p))), rcond=None)[0] / scales
        else:
            delta = np.empty(0)
        dz = np.array([np.linalg.lstsq(uj, -(rj + aj @ delta), rcond=None)[0]
                       for uj, rj, aj in zip(u, r, a)])
        accepted = False
        for rate in (1., .5, .25, .125, .0625, .03125):
            trial_xi = xi.copy()
            trial_xi.ravel()[active] += rate * delta
            trial_z = z + rate * dz
            trial, _ = integrate(trial_z, trial_xi, dt, length, np.empty(0, dtype=int), substeps)
            rss = np.sum(((trial - blocks) / sigma)**2)
            if np.isfinite(rss) and rss <= old_rss:
                xi, z = trial_xi, trial_z
                accepted = True
                damping = max(damping / 3, 1e-9)
                converged = old_rss - rss <= 1e-7 * max(old_rss, 1.)
                break
        if not accepted:
            damping *= 10
            if damping > 1e8:
                break
        if converged:
            break
    pred, _ = integrate(z, xi, dt, length, np.empty(0, dtype=int), substeps)
    block_rss = np.sum(((pred - blocks) / sigma)**2, axis=(1, 2))
    if not np.isfinite(block_rss).all():
        block_rss = np.full(nblocks, np.inf)
    return TrajectoryFit(xi, z, float(block_rss.sum()), block_rss, iteration + 1, bool(converged))


def flow_dictionary(blocks, dt, fit, sigma, substeps=2):
    """Linearised flow regression A*c ~= b after profiling segment starts."""
    active = np.arange(NLIB * DIM)
    pred, deriv = integrate(fit.initial, fit.xi, dt, blocks.shape[1], active, substeps)
    nw, length, d = blocks.shape
    r = ((pred - blocks) / sigma).reshape(nw, length*d)
    a = (deriv[..., :len(active)] / sigma[None, None, :, None]).reshape(nw, length*d, -1)
    u = (deriv[..., len(active):] / sigma[None, None, :, None]).reshape(nw, length*d, d)
    rp, ap = project_initial_states(r, a, u)
    return ap, ap @ fit.xi.ravel() - rp


def _fit_seed(blocks, dt, seed, sigma, max_iter, substeps):
    """Try the requested horizon, then continue from a short, stable fit.

    Shrinking the initial vector field changes only the optimiser's starting
    point. The support stays fixed and every coefficient is re-estimated.
    """
    direct = fit_trajectories(blocks, dt, seed, sigma, max_iter=max_iter, substeps=substeps)
    if np.isfinite(direct.rss):
        return direct, False
    length = blocks.shape[1]
    horizons = sorted(set([min(8, length), min(16, length), min(32, length), length]))
    for shrink in (1., .25, .0625, 0.):
        fit = fit_trajectories(blocks[:, :horizons[0]], dt, seed*shrink, sigma,
                               support=seed != 0, max_iter=max_iter, substeps=substeps)
        if not np.isfinite(fit.rss):
            continue
        for horizon in horizons[1:]:
            fit = fit_trajectories(blocks[:, :horizon], dt, fit.xi, sigma, support=seed != 0,
                                   initial=fit.initial, max_iter=max_iter, substeps=substeps)
            if not np.isfinite(fit.rss):
                break
        if np.isfinite(fit.rss):
            return fit, True
    raise RuntimeError("Seed fitting failed even with horizon continuation; shorten segment_length")


def _top_subsets(a, b, kmax, keep=2, max_combinations=250_000):
    """Small-library exhaustive search; explicit size guard, no silent fallback."""
    from math import comb
    if sum(comb(a.shape[1], k) for k in range(kmax + 1)) > max_combinations:
        raise ValueError("Library/search too large for exhaustive flow search; reduce kmax or degree")
    norm = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    an = a / norm
    gram, target = an.T @ an, an.T @ b
    out = [(np.empty(0, dtype=int), float(b @ b))]
    for k in range(1, min(kmax, a.shape[1]) + 1):
        indices = combos(a.shape[1], k)
        g = gram[indices[:, :, None], indices[:, None, :]]
        c = np.linalg.solve(g + 1e-10*np.eye(k), target[indices, None])[..., 0]
        # Compute regularised-fit RSS (not merely b'b - c'g).
        rss = b@b - 2*np.einsum("nk,nk->n", c, target[indices]) + np.einsum("ni,nij,nj->n", c, g, c)
        for i in np.argsort(rss, kind="stable")[:keep]:
            out.append((indices[i].copy(), float(max(rss[i], 0.))))
    return out


def _weak_seed(x, dt, train_mask, sigma, half_width, kmax, keep):
    phi, dphi = weak_mats(len(x), dt, half_width)
    # Select only integrals whose entire nonzero support lies in training data.
    rows = ~np.any(((phi != 0) | (dphi != 0))[:, ~train_mask], axis=1)
    phi, dphi = phi[rows], dphi[rows]
    if len(phi) <= kmax + 2:
        raise ValueError("Too few training weak windows; increase data length or reduce half_width")
    g, b = phi @ library(x), -dphi @ x
    dtheta = dlibrary(x)
    seed = np.zeros((NLIB, DIM))
    pools = []
    for j in range(DIM):
        candidates = _top_subsets(g, b[:, j], kmax, keep)
        pools.append([sup for sup, _ in candidates])
        best_by_size = [next(sup for sup, _ in candidates if len(sup) == k) for k in range(kmax + 1)]
        xis = [fit_eq(g, b[:, j], sup) for sup in best_by_size]
        chi = []
        for xi in xis:
            cov = resid_cov(phi, dphi, dtheta, xi, j, sigma)
            residual = b[:, j] - g @ xi
            whitened = np.linalg.solve(np.linalg.cholesky(cov), residual)
            chi.append(whitened @ whitened)
        dof = len(phi) - np.arange(kmax + 1)
        scale = max(chi[-1] / dof[-1], 1e-12)
        ok = np.asarray(chi)/scale <= dof + 3*np.sqrt(2*dof)
        k = int(np.argmax(ok)) if ok.any() else kmax
        seed[:, j] = xis[k]
    return seed, pools


def _mask_key(mask):
    return tuple(np.flatnonzero(mask.ravel()).tolist())


def _bic(fit, nobs, support):
    # Common initial-state parameter count cancels between same-length models.
    return fit.rss + int(np.sum(support))*np.log(nobs)


@dataclass
class DiscoveryResult:
    coefficients: np.ndarray
    selected_support: np.ndarray
    alternatives: list
    diagnostics: dict

    def equations(self, precision=5):
        out = []
        for j, state in enumerate("xyz"[:DIM]):
            terms = [f"{self.coefficients[m, j]:+.{precision}g} {NAMES[m]}"
                     for m in np.flatnonzero(self.selected_support[:, j])]
            out.append(f"d{state}/dt = " + (" ".join(terms).lstrip("+") if terms else "0"))
        return out


def discover(x, dt, *, segment_length=40, half_width=20, kmax=5, keep=2,
             rounds=2, finalists=8, max_iter=25, substeps=2, ambiguity_z=1.,
             sigma=None, refit=True, flow_proposals=True, extra_models=()):
    """Discover from data alone; no system name, true support, or noise level.

    Every third non-overlapping segment is reserved for validation. Candidate
    proposals and coefficient fits use training data only. Validation profiles
    initial states but freezes coefficients. Selection minimises training plus
    validation squared error and a BIC-style term-count penalty. A paired
    segment-loss heuristic describes alternatives; NOT a confidence set.

    ``sigma``, if given, is observation standard deviation in ORIGINAL units.
    Default estimates are computed within training blocks, avoiding differences
    across missing/held-out intervals. The incomplete trailing block is omitted.

    ``extra_models`` are coefficient arrays (ORIGINAL units) from another method.
    They are refitted on training segments after the search, so they never steer
    it, and are scored alongside; alternatives flag them with ``extra`` = index.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != DIM or not np.isfinite(x).all():
        raise ValueError(f"x must be finite with shape (samples, {DIM})")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if segment_length < 12 or len(x)//segment_length < 9:
        raise ValueError("Need at least nine segments, each of length >= 12")
    if half_width < 2 or 2*half_width >= 2*segment_length or not 1 <= kmax <= NLIB:
        raise ValueError("Invalid weak window or kmax")
    if rounds < 1 or finalists < 1 or keep < 1 or max_iter < 1 or substeps < 1 or ambiguity_z < 0:
        raise ValueError("Search budgets must be positive and ambiguity_z nonnegative")
    nblocks = len(x)//segment_length
    used = nblocks*segment_length
    x = x[:used]
    block_train = np.arange(nblocks) % 3 != 2
    train_mask = np.repeat(block_train, segment_length)
    scale = np.maximum(np.sqrt(np.mean(x[train_mask]**2, axis=0)), 1e-10)
    xn = x/scale
    blocks = xn.reshape(nblocks, segment_length, DIM)
    train, validation = blocks[block_train], blocks[~block_train]
    if sigma is None:
        sig = np.median([estimate_sigma(b) for b in train], axis=0)
        sig = np.maximum(sig, 1e-6)
    else:
        sig = np.asarray(sigma, dtype=float)/scale
        if sig.shape != (DIM,) or not np.isfinite(sig).all() or np.any(sig <= 0):
            raise ValueError("sigma must contain positive finite standard deviations in original units")
    seed, pools = _weak_seed(xn, dt, train_mask, sig, half_width, kmax, keep)
    initial_support = seed != 0
    current, continuation_used = _fit_seed(train, dt, seed, sig, max_iter, substeps)
    fits = {_mask_key(initial_support): (current, initial_support)}
    searched = []
    for round_index in range(rounds):
        a, b = flow_dictionary(train, dt, current, sig, substeps)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("Flow sensitivities diverged; shorten segment_length")
        current_mask = current.xi != 0
        proposals = {_mask_key(current_mask): current_mask.copy()}
        choices = []
        for j in range(DIM):
            # Partial out the OTHER equations' active coefficients before the
            # conditional subset search, so they may readjust after a term swap.
            other = current_mask.copy()
            other[:, j] = False
            indices = np.arange(j, NLIB*DIM, DIM)
            aj, bj = a[:, indices].copy(), b.copy()
            if other.any():
                nuisance = a[:, other.ravel()]
                q, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
                rank = np.sum(singular > singular[0]*max(nuisance.shape)*np.finfo(float).eps)
                q = q[:, :rank]
                aj -= q @ (q.T @ aj)
                bj -= q @ (q.T @ bj)
            flow_choices = ([sup for sup, _ in _top_subsets(aj, bj, kmax, keep)]
                            if flow_proposals else pools[j])
            choices.append(flow_choices)
            for sup in pools[j] + flow_choices:
                mask = current_mask.copy()
                mask[:, j] = False
                mask[sup, j] = True
                proposals[_mask_key(mask)] = mask
        # Also propose simultaneous equation changes: the best linearised
        # per-equation size choices, with a small bounded Cartesian search.
        small_choices = []
        for j, sups in enumerate(choices):
            ranked = []
            for sup in sups:
                mask = current_mask.copy()
                mask[:, j] = False
                mask[sup, j] = True
                idx = np.flatnonzero(mask.ravel())
                c = np.linalg.lstsq(a[:, idx], b, rcond=None)[0]
                loss = np.sum((b - a[:, idx]@c)**2) + len(idx)*np.log(train.size)
                ranked.append((loss, sup))
            small_choices.append([s for _, s in sorted(ranked, key=lambda v: v[0])[:2]])
        for combination in itertools.product(*small_choices):
            mask = np.zeros_like(current_mask)
            for j, sup in enumerate(combination):
                mask[sup, j] = True
            proposals[_mask_key(mask)] = mask
        ranked = []
        for key, mask in proposals.items():
            idx = np.array(key, dtype=int)
            c = np.linalg.lstsq(a[:, idx], b, rcond=None)[0]
            loss = np.sum((b - a[:, idx]@c)**2) + len(idx)*np.log(train.size)
            ranked.append((float(loss), key, c))
        attempted = 0
        for _, key, c in sorted(ranked, key=lambda v: v[0]):
            if key in fits:
                continue
            if attempted >= finalists:
                break
            attempted += 1
            mask = proposals[key]
            candidate = np.zeros_like(seed)
            candidate.ravel()[list(key)] = c
            fit = fit_trajectories(train, dt, candidate, sig, support=mask,
                                   initial=current.initial, max_iter=max_iter, substeps=substeps)
            if np.isfinite(fit.rss):
                fits[key] = (fit, mask)
        current, _ = min(fits.values(), key=lambda pair: _bic(pair[0], train.size, pair[1]))
        searched.append({"round": round_index + 1, "proposals": len(proposals),
                         "nonlinear_attempts": attempted, "best_train_rss": current.rss})
    extra_keys, added = {}, set()
    for index, xi_extra in enumerate(extra_models):
        mask = np.asarray(xi_extra) != 0
        key = _mask_key(mask)
        extra_keys.setdefault(key, index)
        if key in fits or not mask.any():
            continue
        fit = fit_trajectories(train, dt, xi_to_scaled(np.asarray(xi_extra, dtype=float), scale), sig,
                               support=mask, initial=current.initial, max_iter=max_iter, substeps=substeps)
        if np.isfinite(fit.rss):
            fits[key] = (fit, mask)
            added.add(key)
    evaluated = []
    for key, (fit, mask) in fits.items():
        validation_fit = fit_trajectories(validation, dt, fit.xi, sig, fit_coefficients=False,
                                          max_iter=max_iter, substeps=substeps)
        if np.isfinite(validation_fit.rss):
            evaluated.append({"key": key, "fit": fit, "support": mask,
                              "validation": validation_fit,
                              "score": float(fit.rss + validation_fit.rss + len(key)*np.log(blocks.size))})
    if not evaluated:
        raise RuntimeError("No candidate integrated on validation segments; shorten segment_length")
    # Penalise searching among many near-equivalent larger models. This is a
    # BIC-style score evaluated at TRAINING coefficient estimates, not an exact
    # BIC (nor an estimate of a model's posterior probability).
    best = min(evaluated, key=lambda c: c["score"])
    best_losses = best["validation"].block_rss
    for candidate in evaluated:
        differences = candidate["validation"].block_rss - best_losses
        mean = float(differences.mean())
        se = float(differences.std(ddof=1)/np.sqrt(len(differences)))
        candidate["mean_excess"] = mean
        candidate["standard_error"] = se
        candidate["plausible"] = bool(candidate["score"] - best["score"] <= ambiguity_z*se*len(differences) + 1e-9)
    plausible = [c for c in evaluated if c["plausible"]]
    selected = best
    output_fit = selected["fit"]
    refit_accepted = False
    if refit:
        # Combine already-fitted train/validation starts before all-data refit.
        z = np.empty((nblocks, DIM))
        z[block_train] = selected["fit"].initial
        z[~block_train] = selected["validation"].initial
        all_fit = fit_trajectories(blocks, dt, output_fit.xi, sig, support=selected["support"],
                                   initial=z, max_iter=max_iter, substeps=substeps)
        if np.isfinite(all_fit.rss) and all_fit.rss <= output_fit.rss + selected["validation"].rss + 1e-8:
            output_fit = all_fit
            refit_accepted = True
    alternatives = []
    for candidate in sorted(evaluated, key=lambda c: c["score"]):
        alternatives.append({"support": candidate["support"].tolist(),
                             "coefficients": xi_from_scaled(candidate["fit"].xi, scale).tolist(),
                             "terms": len(candidate["key"]),
                             "train_rss": candidate["fit"].rss,
                             "train_converged": candidate["fit"].converged,
                             "validation_rss": candidate["validation"].rss,
                             "validation_converged": candidate["validation"].converged,
                             "selection_score": candidate["score"],
                             "validation_excess_per_segment": candidate["mean_excess"],
                             "paired_standard_error": candidate["standard_error"],
                             "plausible": candidate["plausible"],
                             "extra": extra_keys.get(candidate["key"]),
                             "extra_only": candidate["key"] in added,
                             "selected": candidate is selected})
    a, _ = flow_dictionary(train, dt, selected["fit"], sig, substeps)
    active_a = a[:, selected["support"].ravel()]
    norms = np.maximum(np.linalg.norm(active_a, axis=0), 1e-12)
    singular = np.linalg.svd(active_a/norms, compute_uv=False)
    diagnostics = {"train_segments": int(block_train.sum()), "validation_segments": int((~block_train).sum()),
                   "samples_used": used, "sigma_estimated": (sig*scale).tolist(),
                   "initial_weak_support": initial_support.tolist(), "rounds": searched,
                   "models_evaluated": len(evaluated), "plausible_models": len(plausible),
                   "flow_proposals": flow_proposals, "seed_continuation_used": continuation_used,
                   "selection_rule": "train_rss + validation_rss + terms * log(number_of_used_scalar_observations)",
                   "ambiguity_z": ambiguity_z, "profiled_normalised_singular_values": singular.tolist(),
                   "all_data_refit_accepted": refit_accepted, "final_converged": output_fit.converged,
                   "note": "Paired validation ambiguity is descriptive, not a calibrated confidence set."}
    return DiscoveryResult(xi_from_scaled(output_fit.xi, scale), selected["support"], alternatives, diagnostics)
