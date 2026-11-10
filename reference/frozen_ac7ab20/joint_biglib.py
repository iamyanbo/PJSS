"""Profiled joint flow on large libraries.

joint_discover.discover is reused unchanged; only its per-equation subset ranking on the profiled
quadratic is swapped (via the module attribute) for one that scales: exhaustive while the number of
subsets of size <= kmax is at most `max_exhaustive`, otherwise a beam search of width `width` on the
same normalised Gram system (top `keep` per size, same output format as flow_discovery._top_subsets).

Polynomial libraries of degree > 3 come from core with EQ_DEG set in the environment BEFORE import.
"""
from math import comb

import numpy as np

import joint_discover as jd
from flow_discovery import _top_subsets as _exhaustive


def _rss(gram, target, bb, idx):
    k = idx.shape[1]
    g = gram[idx[:, :, None], idx[:, None, :]]
    t = target[idx]
    c = np.linalg.solve(g + 1e-10 * np.eye(k), t[..., None])[..., 0]
    return bb - 2 * np.einsum("nk,nk->n", c, t) + np.einsum("ni,nij,nj->n", c, g, c)


def beam_subsets(a, b, kmax, keep=2, width=60):
    p = a.shape[1]
    norm = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    an = a / norm
    gram, target, bb = an.T @ an, an.T @ b, float(b @ b)
    out = [(np.empty(0, dtype=int), bb)]
    beam = [()]
    for k in range(1, min(kmax, p) + 1):
        cand = sorted({tuple(sorted(s + (j,))) for s in beam for j in range(p) if j not in s})
        idx = np.array(cand, dtype=int)
        rss = _rss(gram, target, bb, idx)
        order = np.argsort(rss, kind="stable")
        beam = [tuple(idx[i]) for i in order[:width]]
        for i in order[:keep]:
            out.append((idx[i].copy(), float(max(rss[i], 0.0))))
    return out


def top_subsets(a, b, kmax, keep=2, max_exhaustive=250_000, width=60):
    if sum(comb(a.shape[1], k) for k in range(kmax + 1)) <= max_exhaustive:
        return _exhaustive(a, b, kmax, keep)
    return beam_subsets(a, b, kmax, keep, width)


def discover(x, dt, starts, *, width=60, **kw):
    old = jd._top_subsets
    jd._top_subsets = lambda a, b, kmax, keep=2: top_subsets(a, b, kmax, keep, width=width)
    try:
        return jd.discover(x, dt, starts, **kw)
    finally:
        jd._top_subsets = old


# ---------------- non-polynomial ("mixed") library: polynomials up to `deg` plus sin/cos of each state ----
class MixedLibrary:
    """Terms are evaluated in ORIGINAL units and normalised by their RMS on the data, so the scaled-state
    convention of joint_discover / select_k (u = x / rms(x)) is kept: Xi_scaled = Xi * norm / s."""

    def __init__(self, deg=3, trig=True):
        import itertools
        e = [t for d in range(deg + 1) for t in itertools.product(range(d + 1), repeat=3) if sum(t) == d]
        self.exps = np.array(sorted(set(e), key=lambda t: (sum(t), [-v for v in t])))
        poly = ["1" if sum(t) == 0 else "*".join(v + (f"^{p}" if p > 1 else "") for v, p in zip("xyz", t) if p)
                for t in self.exps]
        self.trig = trig
        self.names = poly + ([f"sin({v})" for v in "xyz"] + [f"cos({v})" for v in "xyz"] if trig else [])
        self.M = len(self.names)
        self.npoly = len(poly)

    def raw(self, X):
        P = np.prod(X[:, None, :] ** self.exps, axis=-1)
        return np.hstack([P, np.sin(X), np.cos(X)]) if self.trig else P

    def draw(self, X):
        out = np.zeros((len(X), self.M, 3))
        for l in range(3):
            e = self.exps.copy()
            a = e[:, l].astype(float)
            e[:, l] = np.maximum(e[:, l] - 1, 0)
            out[:, :self.npoly, l] = a * np.prod(X[:, None, :] ** e, axis=-1)
            if self.trig:
                out[:, self.npoly + l, l] = np.cos(X[:, l])
                out[:, self.npoly + 3 + l, l] = -np.sin(X[:, l])
        return out

    def bind(self, x):
        self.scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-10)
        self.norm = np.maximum(np.sqrt(np.mean(self.raw(x) ** 2, axis=0)), 1e-12)
        return self

    def library(self, U):
        return self.raw(U * self.scale) / self.norm

    def dlibrary(self, U):
        return self.draw(U * self.scale) * self.scale[None, None, :] / self.norm[None, :, None]

    def to_scaled(self, Xi, s):
        return Xi * self.norm[:, None] / s[None, :]

    def from_scaled(self, C, s):
        return C * s[None, :] / self.norm[:, None]

    def coef(self, terms):
        Xi = np.zeros((self.M, 3))
        for i, eq in enumerate(terms):
            for n, v in eq.items():
                Xi[self.names.index(n), i] = v
        return Xi

    def rhs(self, Xi):
        return lambda t, x: self.raw(np.asarray(x)[None, :])[0] @ Xi


class use_library:
    """Context manager: point joint_discover and select_k at a bound MixedLibrary."""

    def __init__(self, lib, x):
        self.lib = lib.bind(np.asarray(x, float))

    def __enter__(self):
        import select_k as sk
        L = self.lib
        self.saved = [(m, k, getattr(m, k)) for m, ks in ((jd, ("NLIB", "library", "dlibrary", "xi_to_scaled",
                                                                 "xi_from_scaled")),
                                                            (sk, ("NLIB", "library", "dlibrary", "xi_from_scaled")))
                      for k in ks]
        for m in (jd, sk):
            m.NLIB, m.library, m.dlibrary, m.xi_from_scaled = L.M, L.library, L.dlibrary, L.from_scaled
        jd.xi_to_scaled = L.to_scaled
        return L

    def __exit__(self, *a):
        for m, k, v in self.saved:
            setattr(m, k, v)
