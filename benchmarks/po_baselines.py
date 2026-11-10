"""Baselines for partial observation (truth never used).

delay_weak   SINDy on derivative (Takens-style) coordinates: the latent components are replaced by Savitzky-Golay
             derivatives of an observed component (first derivative for one latent; first and second for two),
             then weak-form best-subset SINDy with the self-calibrated chi^2 rule (select_k, "chi2s") is run on
             [observed (noisy), proxies]. One model per choice of observed component; the sparsest is returned
             (ties: first). Under the affine gauge, d/dt of an observed component is a valid latent coordinate
             whenever that derivative is affine in the latent state (e.g. Lorenz dx/dt = s(y - x)).
embed_final  Same coordinates, discovered by the frozen full-observation model (final_model.discover).
dahsi        DAHSI-style (Ribera et al., Chaos 2022): random latent initialisations, coefficients 0, full library;
             variational annealing of the dynamics weight (rho from 1 down to 0.03, geometric) with a hard
             threshold lambda applied after every annealing step; a sweep over lambda; every (init, lambda) model
             is refitted and scored with the same score as partial_obs (train + model + held-out misfit + prior
             BIC), the best one returned. Reimplemented on partial_obs's banded solver (DAHSI uses IPOPT).
"""
import numpy as np
from scipy.signal import savgol_filter

from eqdisc import partial_obs as po
from eqdisc.core import NLIB, DIM, xi_to_scaled


def _proxies(y, dt, obs):
    n_lat = DIM - len(obs)
    out = []
    for i in range(len(obs)):
        if n_lat == 1:
            cols = [po.deriv_proxy(y, dt, 1, i)]
        else:
            cols = [po.deriv_proxy(y, dt, d + 1, i, w=41) for d in range(n_lat)]
        Z = np.zeros((len(y), DIM))
        Z[:, obs] = y
        Z[:, [j for j in range(DIM) if j not in obs]] = np.column_stack(cols)
        out.append((f"o{i}", Z))
    return out


def delay_weak(y, dt, obs):
    from eqdisc import select_k
    best = None
    for tag, Z in _proxies(y, dt, obs):
        xi = select_k.discover(Z, dt, criterion="chi2s")
        k = int((xi != 0).sum())
        if best is None or k < best[0]:
            best = (k, xi, Z, tag)
    return best[1], best[2], {"proxy": best[3]}


def embed_final(y, dt, obs):
    from eqdisc.final_model import discover
    best = None
    for tag, Z in _proxies(y, dt, obs):
        res = discover(Z, dt, confidence=False)
        k = int((res.xi != 0).sum())
        if best is None or k < best[0]:
            best = (k, res.xi, Z, tag)
    return best[1], best[2], {"proxy": best[3]}


def dahsi(y, dt, obs, seed=0, n_init=6, lambdas=(0.2, 0.5, 1.0, 2.0, 4.0), rho_path=(1.0, 0.5, 0.25, 0.12, 0.06, 0.03),
          iters=60, **problem_kw):
    """problem_kw: the same Problem options (gauge, latent sigma, score rule) as the partial_obs arm it is compared
    with, so both are selected by the identical score."""
    P = po.Problem(y, dt, obs, **problem_kw)
    rng = np.random.default_rng(seed)
    lat = P.lat
    full = np.ones((NLIB, DIM), bool)
    models = []
    for r in range(n_init):
        Lz = np.column_stack([po._unit(po.surrogate(P.yo_obs[:, rng.integers(len(obs))], rng)) for _ in lat])
        u0 = P.to_grid(P.pseudo(Lz))
        for lam in lambdas:
            sup, C, u = full.copy(), np.zeros((NLIB, DIM)), u0
            f = None
            for rh in rho_path:
                f = P.fit(C, sup, u, rh, iters)
                if f is None:
                    break
                C, u = f.C, f.u
                keep = np.abs(C) >= lam
                if not keep.any(0).all():  # never empty an equation entirely: keep its largest term
                    for j in np.flatnonzero(~keep.any(0)):
                        keep[np.argmax(np.abs(C[:, j])), j] = True
                sup = sup & keep
                C = np.where(sup, C, 0.0)
            if f is None:
                continue
            g = P.fit(C, sup, u, None, iters)  # selection fit at the working rho, same score as partial_obs
            if g is not None:
                g.tag = f"r{r}_l{lam}"
                models.append(g)
    if not models:
        raise RuntimeError("no DAHSI model fitted")
    P.calibrate(min(models, key=P.score))
    best = min(models, key=P.score)
    xi, u = P.finish(best)
    return xi, u, {"init": best.tag, "fits": P.fits, "n_models": len(models)}
