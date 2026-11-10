"""Exact (certified) subset search on the profiled quadratic of joint flow.

Two solvers on a least-squares form (A, b), RSS(S) = min_c |b - A_S c|^2:

  bnb_topk(a, b, kmax, keep)      exact top-`keep` subsets of every size k <= kmax (leaps-and-bounds style
                                  depth-first branch-and-bound; bound = RSS of the node's set plus all
                                  still-free columns, valid because RSS only falls as columns are added).
                                  Same output format as flow_discovery._top_subsets.
  bnb_penalized(A, b, lam)        exact minimiser of RSS(S) + lam |S| over all columns (the BIC surrogate the
                                  proposal step ranks finalists by), with an optional node limit; returns
                                  the best set, its value and the certified lower bound (gap 0 = optimal).

Both work on column-normalised Gram systems with the same 1e-10 ridge as the exhaustive code, and accept
an incumbent list to prune from the start.

`discover_bnb` is joint_discover.discover with the global penalized optimum of each round's quadratic (Gurobi
MIQP, time-limited; certified only when the gap closes) and optionally the `extra` next-best distinct
solutions added to the finalist list; everything else is the
unchanged source, recompiled from joint_discover so it stays in sync.
"""
import heapq
import inspect
import itertools
import textwrap

import numpy as np

RIDGE = 1e-10


class _Gram:
    def __init__(self, a, b):
        norm = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
        an = a / norm
        self.G = an.T @ an
        self.t = an.T @ b
        self.bb = float(b @ b)
        self.p = a.shape[1]
        self.solves = 0

    def rss(self, idx):
        if len(idx) == 0:
            return self.bb
        idx = list(idx)
        g = self.G[np.ix_(idx, idx)]
        t = self.t[idx]
        self.solves += 1
        try:
            L = np.linalg.cholesky(g + RIDGE * np.eye(len(idx)))
            z = np.linalg.solve(L, t)
            c = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(g + RIDGE * np.eye(len(idx)), t, rcond=None)[0]
        return float(max(self.bb - 2 * c @ t + c @ g @ c, 0.0))


def bnb_topk(a, b, kmax, keep=2, node_limit=2_000_000, incumbents=None):
    """Exact top-`keep` subsets per size k = 0..kmax. Returns (list of (idx, rss), certified: bool)."""
    G = _Gram(a, b)
    p = G.p
    kmax = min(kmax, p)
    # order columns by single-column fit (strong columns first -> good incumbents early)
    single = np.array([G.rss([j]) for j in range(p)])
    order = np.argsort(single, kind="stable")
    best = {k: [] for k in range(kmax + 1)}  # max-heaps via negated rss: (-rss, idx)

    def offer(idx, r):
        k = len(idx)
        h = best[k]
        key = tuple(sorted(idx))
        if any(key == s for _, s in h):
            return
        if len(h) < keep:
            heapq.heappush(h, (-r, key))
        elif r < -h[0][0]:
            heapq.heapreplace(h, (-r, key))

    def threshold(k):
        h = best[k]
        return np.inf if len(h) < keep else -h[0][0]

    offer((), G.bb)
    for inc in incumbents or []:
        inc = tuple(int(i) for i in inc)
        if len(inc) <= kmax:
            offer(inc, G.rss(inc))
    nodes = 0
    certified = True
    # depth-first over position in `order`; node = (chosen tuple, next position)
    stack = [((), 0)]
    while stack:
        S, pos = stack.pop()
        nodes += 1
        if nodes > node_limit:
            certified = False
            break
        rest = order[pos:]
        kmin_reach, kmax_reach = len(S), min(kmax, len(S) + len(rest))
        if kmax_reach <= len(S):
            continue
        # bound for every size this node can still reach (sizes > len(S); size len(S) is S itself)
        lb = G.rss(list(S) + list(rest))
        if all(lb >= threshold(k) for k in range(len(S) + 1, kmax_reach + 1)):
            continue
        # branch: include order[pos] (evaluate the new set) / exclude it
        j = int(order[pos])
        T = S + (j,)
        r = G.rss(T)
        offer(T, r)
        stack.append((S, pos + 1))      # exclude
        if len(T) < kmax:
            stack.append((T, pos + 1))  # include (explored first)
    out = [(np.array(sorted(s), dtype=int), r) for k in range(kmax + 1)
           for r, s in sorted((-nr, s) for nr, s in best[k])]
    out[0] = (np.empty(0, dtype=int), G.bb)
    return out, certified, nodes


def bnb_penalized(A, b, lam, node_limit=500_000, incumbents=None, n_best=1):
    """Exact min RSS(S) + lam |S| over all columns of A. Returns (list of (idx, value) for the `n_best`
    best distinct sets found, lower_bound, certified)."""
    G = _Gram(A, b)
    p = G.p
    single = np.array([G.rss([j]) for j in range(p)])
    order = np.argsort(single, kind="stable")
    best = []  # max-heap of (-value, key)

    def offer(S, v):
        key = tuple(sorted(S))
        if any(key == s for _, s in best):
            return
        if len(best) < n_best:
            heapq.heappush(best, (-v, key))
        elif v < -best[0][0]:
            heapq.heapreplace(best, (-v, key))

    def thr():
        return np.inf if len(best) < n_best else -best[0][0]

    offer((), G.bb)
    for inc in incumbents or []:
        inc = tuple(int(i) for i in inc)
        offer(inc, G.rss(inc) + lam * len(inc))
    # best-first search on the bound; node = (bound, chosen tuple, pos)
    full_rss = G.rss(list(range(p)))
    heap = [(full_rss, (), 0)]
    nodes = 0
    open_lb = np.inf
    certified = True
    while heap:
        lb, S, pos = heapq.heappop(heap)
        if lb >= thr():
            continue
        nodes += 1
        if nodes > node_limit:
            certified = False
            open_lb = min(lb, min((h[0] for h in heap), default=np.inf))
            break
        if pos >= p:
            continue
        j = int(order[pos])
        T = S + (j,)
        rT = G.rss(T)
        offer(T, rT + lam * len(T))
        rest = list(order[pos + 1:])
        # include j: bound = RSS(T u rest) + lam |T|
        lb_in = (G.rss(list(T) + rest) if rest else rT) + lam * len(T)
        # exclude j: bound = RSS(S u rest) + lam |S|
        lb_ex = (G.rss(list(S) + rest) if rest else G.rss(S)) + lam * len(S)
        if lb_in < thr():
            heapq.heappush(heap, (lb_in, T, pos + 1))
        if lb_ex < thr():
            heapq.heappush(heap, (lb_ex, S, pos + 1))
    res = sorted(((-nv, s) for nv, s in best))
    best_val = res[0][0]
    lower = best_val if certified else min(open_lb, best_val)
    return [(np.array(s, dtype=int), v) for v, s in res], lower, certified, nodes


def miqp_penalized(A, b, lam, kmax_eq=5, dim=3, n_best=1, time_limit=10.0, incumbents=None):
    """Certified min |b - A c|^2 + lam |supp c| with at most kmax_eq terms per equation (column j belongs
    to equation j % dim, as in the flattened (NLIB, DIM) layout), by Gurobi MIQP with indicator
    constraints (no big-M). The profiled quadratic is square (A = L^T), so RSS-of-superset bounds are
    vacuous and a branch-and-bound on RSS cannot certify; the MIQP's continuous relaxation can.
    Returns (list of (idx, value) best first, lower bound, certified, node count)."""
    import gurobipy as gp
    p = A.shape[1]
    norm = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    An = A / norm
    Q, q, bb = An.T @ An, An.T @ b, float(b @ b)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as m:
            m.Params.Threads = 1
            m.Params.TimeLimit = time_limit
            m.Params.MIPGap = 1e-9
            m.Params.MIPGapAbs = 1e-6
            if n_best > 1:
                m.Params.PoolSearchMode = 2
                m.Params.PoolSolutions = n_best
            c = m.addMVar(p, lb=-gp.GRB.INFINITY, name="c")
            z = m.addMVar(p, vtype=gp.GRB.BINARY, name="z")
            for j in range(p):
                m.addGenConstrIndicator(z[j].item(), False, c[j].item() == 0.0)
            for e in range(dim):
                m.addConstr(z[np.arange(e, p, dim)].sum() <= kmax_eq)
            m.setObjective(c @ Q @ c - 2 * q @ c + bb + lam * z.sum(), gp.GRB.MINIMIZE)
            if incumbents:
                inc = tuple(int(i) for i in incumbents[0])
                start = np.zeros(p)
                if inc:
                    g = Q[np.ix_(inc, inc)] + RIDGE * np.eye(len(inc))
                    start[list(inc)] = np.linalg.solve(g, q[list(inc)])
                c.Start = start
                z.Start = (np.abs(start) > 0).astype(float)
            m.optimize()
            G = _Gram(A, b)
            sols = []
            for s in range(m.SolCount):
                m.Params.SolutionNumber = s
                idx = tuple(int(j) for j in np.flatnonzero(np.asarray(z.Xn) > 0.5))
                if idx not in [t for t, _ in sols]:
                    sols.append((idx, G.rss(idx) + lam * len(idx)))
            sols.sort(key=lambda v: v[1])
            certified = m.Status == gp.GRB.OPTIMAL
            lower = float(m.ObjBound)
            nodes = int(m.NodeCount)
    return [(np.array(s, dtype=int), v) for s, v in sols[:max(n_best, 1)]], lower, certified, nodes


# ---------------- discover variant: inject the certified global optimum into the finalists ----------------
def _make_discover_bnb():
    from eqdisc import joint_discover as jd
    src = textwrap.dedent(inspect.getsource(jd.discover))
    src = src.replace("def discover(x, dt, starts, *,", "def discover_bnb(x, dt, starts, *, bnb_extra=1, bnb_time=5.0, bnb_log=None,")
    anchor = "        for _, key, mask, c in sorted(ranked, key=lambda v: v[0])[:finalists]:\n"
    assert src.count(anchor) == 1
    inject = textwrap.indent(textwrap.dedent("""
        _srt = sorted(ranked, key=lambda v: v[0])
        _top = _srt[:finalists]
        _inc = [k for _, k, _, _ in _srt[:3]] + [tuple(np.flatnonzero(cur.ravel()))]
        _opt, _lb, _cert, _nodes = _miqp(A, b, np.log(nobs), kmax_eq=kmax, n_best=bnb_extra,
                                         time_limit=bnb_time, incumbents=_inc)
        _keys = {k for _, k, _, _ in _top}
        _added = 0
        for _idx, _v in _opt:
            _key = tuple(int(i) for i in _idx)
            if _key in fits or _key in _keys or not len(_key):
                continue
            _mask = np.zeros(NLIB * DIM, bool); _mask[list(_key)] = True
            _cc = np.linalg.lstsq(A[:, list(_key)], b, rcond=None)[0]
            ranked.append((-np.inf, _key, _mask.reshape(NLIB, DIM), _cc))
            _added += 1
        if bnb_log is not None:
            _ok = tuple(int(i) for i in _opt[0][0])
            bnb_log.append({"opt": list(_ok), "opt_val": float(_opt[0][1]), "lower": float(_lb),
                            "certified": bool(_cert), "nodes": int(_nodes),
                            "heur_best": list(_top[0][1]) if _top else [],
                            "heur_best_val": float(_top[0][0]) if _top else None,
                            "opt_in_proposals": _ok in {k for _, k, _, _ in _srt} or _ok in fits,
                            "opt_in_finalists": _ok in _keys or _ok in fits, "added": _added})
        finalists_eff = finalists + _added
    """), "        ")
    src = src.replace(anchor, inject + anchor.replace("[:finalists]", "[:finalists_eff]"))
    ns = dict(vars(jd))
    ns["_bnb_penalized"] = bnb_penalized
    ns["_miqp"] = miqp_penalized
    exec(compile(src, "<discover_bnb>", "exec"), ns)
    return ns["discover_bnb"]


def discover_bnb(*args, **kw):
    global _DISCOVER_BNB
    try:
        f = _DISCOVER_BNB
    except NameError:
        f = _DISCOVER_BNB = _make_discover_bnb()
    return f(*args, **kw)


def capture_quadratics():
    """Context manager: record (A, b, current support, nobs) at every profiled_quadratic call of jd.discover."""
    import contextlib
    from eqdisc import joint_discover as jd

    @contextlib.contextmanager
    def ctx():
        log = []
        orig = jd.profiled_quadratic

        def wrapped(fit, *a, **k):
            A, b = orig(fit, *a, **k)
            log.append({"A": A, "b": b, "cur": fit.support.copy()})
            return A, b
        jd.profiled_quadratic = wrapped
        try:
            yield log
        finally:
            jd.profiled_quadratic = orig
    return ctx()
