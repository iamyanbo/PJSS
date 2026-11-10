"""Exhaustive best-subset regression via precomputed Gram matrices."""
import itertools
import numpy as np

_COMBOS = {}


def combos(n, k):
    if (n, k) not in _COMBOS:
        _COMBOS[(n, k)] = np.array(list(itertools.combinations(range(n), k)))
    return _COMBOS[(n, k)]


def best_subsets(A, b, kmax=5):
    """For each k<=kmax return (support idx, rss) of the best k-subset (columns normalised internally)."""
    nrm = np.linalg.norm(A, axis=0) + 1e-12
    An = A / nrm
    G = An.T @ An
    g = An.T @ b
    bb = b @ b
    out = [(np.array([], int), bb)]
    for k in range(1, kmax + 1):
        C = combos(A.shape[1], k)  # (M, k)
        Gs = G[C[:, :, None], C[:, None, :]]  # (M,k,k)
        gs = g[C]  # (M,k)
        try:
            c = np.linalg.solve(Gs + 1e-12 * np.eye(k), gs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(Gs, gs)[0]
        rss = bb - np.einsum("mk,mk->m", c, gs)
        j = rss.argmin()
        out.append((C[j], rss[j]))
    return out


def _rss_batch(G, g, bb, C):
    k = C.shape[1]
    Gs = G[C[:, :, None], C[:, None, :]]
    gs = g[C]
    c = np.linalg.solve(Gs + 1e-12 * np.eye(k), gs[..., None])[..., 0]
    return bb - np.einsum("mk,mk->m", c, gs)


def beam_subsets(A, b, kmax=6, width=40):
    """Beam search approximation to best_subsets: grow supports one term at a time, keep `width` best per size,
    then refine the best of each size by single swaps until no improvement."""
    nrm = np.linalg.norm(A, axis=0) + 1e-12
    An = A / nrm
    G = An.T @ An
    g = An.T @ b
    bb = b @ b
    p = A.shape[1]
    beam = np.zeros((1, 0), int)
    out = [(np.array([], int), bb)]
    for k in range(1, kmax + 1):
        # expand every beam member by every unused term
        cand = np.array([sorted(list(s) + [j]) for s in beam for j in range(p) if j not in s])
        cand = np.unique(cand, axis=0)
        rss = _rss_batch(G, g, bb, cand)
        order = np.argsort(rss)[:width]
        beam = cand[order]
        best, best_rss = beam[0].copy(), rss[order[0]]
        # swap refinement
        improved = True
        while improved:
            improved = False
            sw = np.array([sorted([x for x in best if x != i] + [j]) for i in best for j in range(p) if j not in best])
            r = _rss_batch(G, g, bb, sw)
            if r.min() < best_rss * (1 - 1e-9):
                best, best_rss, improved = sw[r.argmin()], r.min(), True
        out.append((np.array(best), best_rss))
    return out
