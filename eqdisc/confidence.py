"""Per-term confidence for joint flow from its fitted candidate pool.

Posterior inclusion probability (PIP) of term (library term, equation) j:
    p_j = sum_m w_m [j in S_m],   w_m proportional to exp(-(score_m - min score) / (2 T))
over the run's fitted candidates m (score = data + model + val + phi k log 3n, as in selection). T = 1 is
plain BIC model averaging; T > 1 tempers over-confident weights (T is a design choice, fitted on dev only).

Set-valued output: "sure" terms p >= hi, "possible" terms lo < p < hi, the rest excluded. Reported per run:
  covers     every true term is sure or possible (the truth lies inside the set)
  sure_ok    every sure term is true
  n_possible number of possible terms
Calibration: reliability bins and Brier score over all (run, term) pairs of terms that appear in at least
one candidate (terms in no candidate get p = 0).
"""
import argparse
import json

import numpy as np

NT = 60  # 20 terms x 3 equations (flattened index term*3 + eq)


def pips(cands, T=1.0, nt=NT):
    s = np.array([c["score"] for c in cands])
    w = np.exp(-(s - s.min()) / (2 * T))
    w /= w.sum()
    p = np.zeros(nt)
    for wi, c in zip(w, cands):
        p[c["support"]] += wi
    return p


def fit_isotonic(P, Y):
    """Pool-adjacent-violators fit of P(true | p); returns (knots, values) for a step map."""
    o = np.argsort(P, kind="stable")
    x, y = P[o], Y[o].astype(float)
    blocks = []  # [sum_y, count, max_x]
    for xi, yi in zip(x, y):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, m2 = blocks.pop()
            blocks[-1][0] += s2; blocks[-1][1] += c2; blocks[-1][2] = m2
    knots = np.array([b[2] for b in blocks])
    vals = np.array([b[0] / b[1] for b in blocks])
    return knots, vals


def apply_isotonic(p, cal):
    knots, vals = cal
    return vals[np.minimum(np.searchsorted(knots, p, side="left"), len(vals) - 1)]


def raw_pips(records, T=1.0, nt=NT):
    P, Y = [], []
    for r in records:
        if r.get("cands"):
            y = np.zeros(nt, bool); y[r["truth"]] = True
            P.append(pips(r["cands"], T, nt)); Y.append(y)
    return np.concatenate(P), np.concatenate(Y)


def evaluate(records, T=1.0, lo=0.1, hi=0.9, nt=NT, bins=(0, .05, .2, .4, .6, .8, .95, 1.0001), cal=None):
    P, Y, runs = [], [], []
    for r in records:
        if not r.get("cands"):
            continue
        p = pips(r["cands"], T, nt)
        if cal is not None:
            p = np.where(p > 0, apply_isotonic(p, cal), 0.0)
        y = np.zeros(nt, bool)
        y[r["truth"]] = True
        P.append(p); Y.append(y)
        sure, poss = p >= hi, (p > lo) & (p < hi)
        runs.append({"covers": bool((sure | poss)[y].all()), "sure_ok": bool(y[sure].all()),
                     "n_possible": int(poss.sum()), "exact_point": r["joint"]["exact"],
                     "sure_recall": float(sure[y].mean()), "n_true_absent": int((p[y] == 0).sum()),
                     "n_false_sure": int((sure & ~y).sum()),
                     "system": r["system"], "noise": r["noise"], "n": r["n"],
                     "exact_set": bool((sure | poss)[y].all() and y[sure].all() and poss.sum() == 0)})
    P, Y = np.concatenate(P), np.concatenate(Y)
    brier = float(((P - Y) ** 2).mean())
    rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (P >= a) & (P < b)
        if m.any():
            rel.append((a, b, int(m.sum()), float(P[m].mean()), float(Y[m].mean())))
    return {"brier": brier, "reliability": rel, "runs": runs}


def summary(ev, label=""):
    runs = ev["runs"]
    print(f"{label} Brier {ev['brier']:.4f}")
    print("  bin          count   mean p   frac true")
    for a, b, c, mp, fy in ev["reliability"]:
        print(f"  [{a:.2f},{min(b, 1):.2f})  {c:6d}   {mp:.3f}    {fy:.3f}")
    for name, sel in (("all", lambda r: True), ("point wrong", lambda r: not r["exact_point"]),
                      ("rabinovich", lambda r: r["system"] == "rabinovich"),
                      ("20% noise", lambda r: r["noise"] == 0.2)):
        rs = [r for r in runs if sel(r)]
        if rs:
            print(f"  {name:12} runs {len(rs):4d}  covers {np.mean([r['covers'] for r in rs]):.3f}  "
                  f"sure-all-true {np.mean([r['sure_ok'] for r in rs]):.3f}  "
                  f"possible/run {np.mean([r['n_possible'] for r in rs]):.2f}  "
                  f"sure-recall {np.mean([r['sure_recall'] for r in rs]):.3f}  "
                  f"true-terms-in-no-candidate/run {np.mean([r['n_true_absent'] for r in rs]):.2f}  "
                  f"point exact {np.mean([r['exact_point'] for r in rs]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--T", nargs="+", type=float, default=[1.0])
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--calibrate-from", nargs="+", help="fit isotonic recalibration on these (dev) files")
    args = ap.parse_args()
    recs = [r for f in args.files for r in json.load(open(f))["records"]]
    for T in args.T:
        summary(evaluate(recs, T, args.lo, args.hi), f"T={T} raw")
        if args.calibrate_from:
            dev = [r for f in args.calibrate_from for r in json.load(open(f))["records"]]
            P, Y = raw_pips(dev, T)
            cal = fit_isotonic(P[P > 0], Y[P > 0])
            summary(evaluate(recs, T, args.lo, args.hi, cal=cal), f"T={T} isotonic(dev)")


if __name__ == "__main__":
    main()
