"""Precision audit: instrumentation and a numerics-hardened variant of the joint stage (diagnostics only).

Two context managers patch the frozen modules in place (nothing in eqdisc/ changes):

record()    behaviour unchanged; logs, for every call of the dense numerical kernels,
            * profiled quadratic H (joint_discover / joint_colored / joint_nd): eigenvalues of H before the
              1e-10*trace/len ridge -> lam_min / mean diag, the ridge-to-lam_min ratio, number of
              non-positive eigenvalues, and the float64 forward-error bound eps * max diag(A_cc) / lam_min
              (A_cc is the coefficient block before the Schur subtraction, i.e. the scale of the cancellation);
            * subset ranking (flow_discovery._top_subsets / joint_biglib exhaustive + beam / subset.best_subsets):
              b'b, the size of the cancellation eps * b'b relative to the selection penalty scale, and for the
              returned subsets the difference between the normal-equation RSS (with its ridge) and an
              exact-residual QR RSS (no ridge); also whether the per-size winner changes under QR.
hardened()  the same algorithm with backward-stable numerics and (almost) no ridges:
            * every subset RSS from Householder QR of the (normalised) columns, RSS = |b - Q Q'b|^2 (no Gram
              matrix, no b'b - c'g cancellation, no ridge);
            * profiled quadratic from an eigendecomposition of H (eigenvalues clipped at 1e-15 lam_max instead
              of the 1e-10*trace/len ridge); A = Lambda^1/2 V' has the same A'A = H, so every subset RSS is
              unchanged in exact arithmetic.
            In exact arithmetic (up to the dropped ridges) the selected model is the same; only rounding changes.
"""
import contextlib

import numpy as np

EPS = np.finfo(float).eps
LOG = []          # records appended while record() is active
_MODE = ["off"]


def _qr_rss(an, b, idx, chunk=4000):
    """Exact-residual RSS of b on the columns an[:, idx[r]] for each row r of idx (Householder QR)."""
    M, k = idx.shape
    if k == 0:
        return np.full(M, float(b @ b))
    out = np.empty(M)
    for s in range(0, M, chunk):
        I = idx[s:s + chunk]
        As = np.transpose(an[:, I], (1, 0, 2))  # (m', rows, k)
        Q, _ = np.linalg.qr(As)
        proj = np.einsum("Mmk,m->Mk", Q, b)
        r = b[None, :] - np.einsum("Mmk,Mk->Mm", Q, proj)
        out[s:s + chunk] = np.einsum("Mm,Mm->M", r, r)
    return out


def _norm_cols(a, floor):
    norm = np.maximum(np.linalg.norm(a, axis=0), floor) if floor else np.linalg.norm(a, axis=0) + 1e-12
    return a / norm


# ---------------- hardened subset searches (same outputs as the originals) ----------------
def hard_top_subsets(a, b, kmax, keep=2, max_combinations=250_000):
    """flow_discovery._top_subsets / joint_biglib._exhaustive with QR residuals and no ridge."""
    from math import comb
    from eqdisc.subset import combos
    if sum(comb(a.shape[1], k) for k in range(kmax + 1)) > max_combinations:
        raise ValueError("Library/search too large for exhaustive flow search; reduce kmax or degree")
    an = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    out = [(np.empty(0, dtype=int), float(b @ b))]
    for k in range(1, min(kmax, a.shape[1]) + 1):
        idx = combos(a.shape[1], k)
        rss = _qr_rss(an, b, idx)
        for i in np.argsort(rss, kind="stable")[:keep]:
            out.append((idx[i].copy(), float(max(rss[i], 0.))))
    return out


def hard_jb_rss(gram, target, bb, idx):
    """joint_biglib._rss replacement: needs the normalised columns, passed through a module global."""
    an, b = _BEAM_CTX[0]
    return _qr_rss(an, b, idx)


_BEAM_CTX = [None]


def hard_beam_subsets(a, b, kmax, keep=2, width=60):
    from eqdisc import joint_biglib as jb
    _BEAM_CTX[0] = (a / np.maximum(np.linalg.norm(a, axis=0), 1e-12), b)
    try:
        return _ORIG["jb.beam_subsets"](a, b, kmax, keep, width)
    finally:
        _BEAM_CTX[0] = None


def hard_best_subsets(A, b, kmax=5):
    """subset.best_subsets with QR residuals (columns normalised the same way, no ridge)."""
    from eqdisc.subset import combos
    An = A / (np.linalg.norm(A, axis=0) + 1e-12)
    out = [(np.array([], int), b @ b)]
    for k in range(1, kmax + 1):
        C = combos(A.shape[1], k)
        rss = _qr_rss(An, b, C)
        j = rss.argmin()
        out.append((C[j], rss[j]))
    return out


def hard_rss_batch(G, g, bb, C):
    An, b = _SB_CTX[0]
    return _qr_rss(An, b, C)


_SB_CTX = [None]


def hard_sk_beam_subsets(A, b, kmax=6, width=40):
    _SB_CTX[0] = (A / (np.linalg.norm(A, axis=0) + 1e-12), b)
    try:
        return _ORIG["sb.beam_subsets"](A, b, kmax, width)
    finally:
        _SB_CTX[0] = None


# ---------------- profiled quadratic ----------------
def _h_from(ab, Auc, Acc, gu, gc):
    from scipy.linalg import solveh_banded
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    H = Acc - Auc.T @ X[:, 1:]
    g = gc - Auc.T @ X[:, 0]
    return 0.5 * (H + H.T), g


def _h_stats(H, Acc, where):
    w = np.linalg.eigvalsh(H)
    md = float(np.trace(H) / len(H))
    pos = w[w > 0]
    lmin = float(pos.min()) if len(pos) else float("nan")
    LOG.append({"kind": "H", "where": where, "n": len(H), "lam_min": float(w.min()), "lam_max": float(w.max()),
                "mean_diag": md, "cond": float(w.max() / lmin) if len(pos) else float("inf"),
                "ridge_over_lammin": 1e-10 * md / lmin if len(pos) else float("inf"),
                "nonpos": int((w <= 0).sum()),
                "fwd_bound": EPS * float(np.max(np.diag(Acc))) / lmin if len(pos) else float("inf"),
                "cancel": float(np.max(np.diag(Acc)) / max(np.max(np.diag(H)), 1e-300))})


def _hard_factor(H, g, c0):
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 1e-15 * w.max())
    cstar = c0 - V @ ((V.T @ g) / w)
    A = (np.sqrt(w)[:, None] * V.T)
    return A, A @ cstar


def make_pq(module, kind):
    """Replacement profiled_quadratic for joint_discover ("jd"), joint_colored ("jc") or joint_nd.Solver ("jn")."""
    orig = _ORIG[f"{kind}.pq"]

    if kind == "jd":
        def pq(fit, y, dt, sig, rho, train, c=np.inf):
            from eqdisc import joint_discover as jd
            if _MODE[0] == "record":
                params = np.arange(jd.NLIB * jd.DIM)
                ab, Auc, Acc, gu, gc, _ = jd._normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
                H, _ = _h_from(ab, Auc, Acc, gu, gc)
                _h_stats(H, Acc, "jd")
                return orig(fit, y, dt, sig, rho, train, c)
            params = np.arange(jd.NLIB * jd.DIM)
            ab, Auc, Acc, gu, gc, _ = jd._normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
            H, g = _h_from(ab, Auc, Acc, gu, gc)
            return _hard_factor(H, g, fit.C.ravel())
        return pq
    if kind == "jc":
        def pq(fit, obs, used, s, dt, sig, rho, c=np.inf):
            from eqdisc import joint_colored as jc
            params = np.arange(jc.NLIB * jc.DIM)
            ab, Auc, Acc, gu, gc, _ = jc._normal(fit.u, fit.C, params, obs, used, s, dt, sig, rho, c)
            H, g = _h_from(ab, Auc, Acc, gu, gc)
            if _MODE[0] == "record":
                _h_stats(H, Acc, "jc")
                return orig(fit, obs, used, s, dt, sig, rho, c)
            return _hard_factor(H, g, fit.C.ravel())
        return pq

    def pq(self, fit, y, dt, sig, rho, train, c=np.inf):
        params = np.arange(self.lib.M * self.D)
        ab, Auc, Acc, gu, gc, _ = self.normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
        H, g = _h_from(ab, Auc, Acc, gu, gc)
        if _MODE[0] == "record":
            _h_stats(H, Acc, "jn")
            return orig(self, fit, y, dt, sig, rho, train, c)
        return _hard_factor(H, g, fit.C.ravel())
    return pq


def _rec_subsets(orig, where):
    """Wrap a (a, b, kmax, ...) -> [(sup, rss)] subset search: log cancellation and QR-vs-formula RSS."""
    def f(a, b, kmax, *args, **kw):
        out = orig(a, b, kmax, *args, **kw)
        an = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
        bb = float(b @ b)
        diffs, winners_differ = [], 0
        by_size = {}
        for sup, rss in out:
            if len(sup) == 0:
                continue
            q = float(_qr_rss(an, b, np.asarray(sup)[None, :])[0])
            diffs.append(abs(q - rss))
            by_size.setdefault(len(sup), []).append((rss, q))
        for k, lst in by_size.items():
            if len(lst) > 1 and np.argmin([r for r, _ in lst]) != np.argmin([q for _, q in lst]):
                winners_differ += 1
        LOG.append({"kind": "subset", "where": where, "bb": bb, "eps_bb": EPS * bb,
                    "max_rss_diff": max(diffs) if diffs else 0.0,
                    "min_rss": min((r for _, r in out if r > 0), default=0.0),
                    "rank_swaps": winners_differ, "p": a.shape[1]})
        return out
    return f


_ORIG = {}


def _targets():
    from eqdisc import flow_discovery as fd, joint_discover as jd, joint_colored as jc, joint_nd as jn
    from eqdisc import joint_biglib as jb, subset as sb, select_k as sk
    return fd, jd, jc, jn, jb, sb, sk


def _save():
    fd, jd, jc, jn, jb, sb, sk = _targets()
    _ORIG.update({"fd._top": fd._top_subsets, "jd._top": jd._top_subsets, "jc._top": jc._top_subsets,
                  "jb._exhaustive": jb._exhaustive, "jb._rss": jb._rss, "jb.beam_subsets": jb.beam_subsets,
                  "sb.best_subsets": sb.best_subsets, "sb._rss_batch": sb._rss_batch,
                  "sb.beam_subsets": sb.beam_subsets, "sk.best_subsets": sk.best_subsets,
                  "sk.beam_subsets": sk.beam_subsets,
                  "jd.pq": jd.profiled_quadratic, "jc.pq": jc.profiled_quadratic,
                  "jn.pq": jn.Solver.profiled_quadratic})


def _restore():
    fd, jd, jc, jn, jb, sb, sk = _targets()
    fd._top_subsets, jd._top_subsets, jc._top_subsets = _ORIG["fd._top"], _ORIG["jd._top"], _ORIG["jc._top"]
    jb._exhaustive, jb._rss, jb.beam_subsets = _ORIG["jb._exhaustive"], _ORIG["jb._rss"], _ORIG["jb.beam_subsets"]
    sb.best_subsets, sb._rss_batch, sb.beam_subsets = _ORIG["sb.best_subsets"], _ORIG["sb._rss_batch"], \
        _ORIG["sb.beam_subsets"]
    sk.best_subsets, sk.beam_subsets = _ORIG["sk.best_subsets"], _ORIG["sk.beam_subsets"]
    jd.profiled_quadratic, jc.profiled_quadratic = _ORIG["jd.pq"], _ORIG["jc.pq"]
    jn.Solver.profiled_quadratic = _ORIG["jn.pq"]
    _MODE[0] = "off"


@contextlib.contextmanager
def record():
    _save()
    fd, jd, jc, jn, jb, sb, sk = _targets()
    _MODE[0] = "record"
    jd._top_subsets = _rec_subsets(_ORIG["jd._top"], "jd")
    jc._top_subsets = _rec_subsets(_ORIG["jc._top"], "jc")
    jb._exhaustive = _rec_subsets(_ORIG["jb._exhaustive"], "jb_exh")
    jb.beam_subsets = _rec_subsets(_ORIG["jb.beam_subsets"], "jb_beam")
    jd.profiled_quadratic = make_pq(jd, "jd")
    jc.profiled_quadratic = make_pq(jc, "jc")
    jn.Solver.profiled_quadratic = make_pq(jn, "jn")
    try:
        yield LOG
    finally:
        _restore()


@contextlib.contextmanager
def hardened():
    _save()
    fd, jd, jc, jn, jb, sb, sk = _targets()
    _MODE[0] = "hard"
    fd._top_subsets = jd._top_subsets = jc._top_subsets = hard_top_subsets
    jb._exhaustive = hard_top_subsets
    jb._rss = hard_jb_rss
    jb.beam_subsets = hard_beam_subsets
    sb.best_subsets = sk.best_subsets = hard_best_subsets
    sb._rss_batch = hard_rss_batch
    sb.beam_subsets = sk.beam_subsets = hard_sk_beam_subsets
    jd.profiled_quadratic = make_pq(jd, "jd")
    jc.profiled_quadratic = make_pq(jc, "jc")
    jn.Solver.profiled_quadratic = make_pq(jn, "jn")
    try:
        yield
    finally:
        _restore()


def summarise(log):
    """Compact per-run summary of a record() log."""
    H = [r for r in log if r["kind"] == "H"]
    S = [r for r in log if r["kind"] == "subset"]
    out = {"n_H": len(H), "n_subset": len(S)}
    if H:
        out.update(H_cond_max=max(r["cond"] for r in H), H_ridge_over_lammin_max=max(r["ridge_over_lammin"] for r in H),
                   H_fwd_bound_max=max(r["fwd_bound"] for r in H), H_nonpos=sum(r["nonpos"] for r in H),
                   H_cancel_max=max(r["cancel"] for r in H))
    if S:
        out.update(sub_eps_bb_max=max(r["eps_bb"] for r in S), sub_rss_diff_max=max(r["max_rss_diff"] for r in S),
                   sub_rank_swaps=sum(r["rank_swaps"] for r in S))
    return out


# ---------------- (ii) square-root (QR) construction of the profiled quadratic, joint_discover only ----------------
def qr_pq_jd(fit, y, dt, sig, rho, train, c=np.inf):
    """Profiled quadratic of joint_discover WITHOUT forming H = A_cc - A_uc' A_uu^-1 A_uc.

    The states are profiled out row by row: M = J_c - J_u X_c and r~ = r - J_u x_g (X = A_uu^-1 [g_u, A_uc], the
    same banded solve as the original), then A = R and Q'r~ come from one Householder QR of [M | r~]. A'A = H and
    the quadratic is the same in exact arithmetic (no ridge), but the float64 error of H is ~eps*kappa instead of
    ~eps*kappa^2 (kappa = |J_c| / sigma_min(M)). Rows: model residuals (n-1)*D, Huber-weighted data residuals."""
    from scipy.linalg import solveh_banded
    from eqdisc import joint_discover as jd
    D, NL = jd.DIM, jd.NLIB
    params = np.arange(NL * D)
    ab, Auc, Acc, gu, gc, _ = jd._normal(fit.u, fit.C, params, y, train, dt, sig, rho, c)
    X = solveh_banded(ab, np.column_stack([gu, Auc]), check_finite=False)
    xg, Xc = X[:, 0].reshape(-1, D), X[:, 1:].reshape(len(fit.u), D, -1)  # (n, D), (n, D, P)
    u, C = fit.u, fit.C
    rd, rm, m, th = jd._residuals(u, C, y, train, dt, sig, rho)
    J = np.einsum("mi,kml->kil", C, jd.dlibrary(m))
    eye = np.eye(D)
    w = 1.0 / (rho * sig)
    Bm = (-eye - 0.5 * dt * J) * w[None, :, None]
    Bp = (eye - 0.5 * dt * J) * w[None, :, None]
    P = NL * D
    # model rows (k, a)
    Mm = -(np.einsum("kai,kip->kap", Bm, Xc[:-1]) + np.einsum("kai,kip->kap", Bp, Xc[1:]))
    mi, ei = params // D, params % D
    for a in range(D):
        cols = np.flatnonzero(ei == a)
        Mm[:, a, cols] += -dt * th[:, mi[cols]] * w[a]
    rmt = rm - (np.einsum("kai,ki->ka", Bm, xg[:-1]) + np.einsum("kai,ki->ka", Bp, xg[1:]))
    # data rows (k, i), train only, Huber IRLS weight s = sqrt(w_d)
    from eqdisc.joint_discover import _hw
    s = np.sqrt(_hw(rd, c)) * train[:, None]
    Md = -(s / sig[None, :])[:, :, None] * Xc
    rdt = s * rd - (s / sig[None, :]) * xg
    keep = train
    big = np.vstack([np.column_stack([Mm.reshape(-1, P), rmt.reshape(-1)]),
                     np.column_stack([Md[keep].reshape(-1, P), rdt[keep].reshape(-1)])])
    R = np.linalg.qr(big, mode="r")
    A = R[:P, :P]
    qr_ = R[:P, P]
    b = A @ C.ravel() - qr_
    return A, b


@contextlib.contextmanager
def qr_basis():
    """hardened() plus the square-root profiled quadratic for joint_discover (white path)."""
    with hardened():
        from eqdisc import joint_discover as jd
        old = jd.profiled_quadratic
        jd.profiled_quadratic = qr_pq_jd
        try:
            yield
        finally:
            jd.profiled_quadratic = old
