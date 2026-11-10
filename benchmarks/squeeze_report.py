"""Tables for the squeeze track: failure anatomy and variant comparisons against config C (same cached starts)."""
import json
import sys
from collections import Counter

import numpy as np
from scipy.stats import binomtest


def load(f):
    return {r["index"]: r for r in json.load(open(f))["records"]}


def anatomy(f):
    R = list(load(f).values())
    wrong = [r for r in R if r.get("class") not in (None, "ok")]
    print(f"{len(R)} runs, {len(wrong)} wrong: {dict(Counter(r['class'] for r in wrong))}")
    keys = sorted({(r["system"], r["noise"]) for r in wrong})
    systems = sorted({r["system"] for r in wrong})
    print(f"{'system':11}" + "".join(f" {n:>14}" for n in sorted({r['noise'] for r in R})) + "   (A/B/C)")
    for s in systems:
        row = []
        for nz in sorted({r["noise"] for r in R}):
            c = Counter(r["class"] for r in wrong if r["system"] == s and r["noise"] == nz)
            row.append(f"{c['A_search']}/{c['B_select']}/{c['C_local']}")
        print(f"{s:11}" + "".join(f" {v:>14}" for v in row))
    # how far below the winner does the best truth fit get?
    for cls in ("A_search", "C_local"):
        d = [r["diag"]["winner_score"] - min(v for v in (r["diag"]["truth_score_search"],
             r["diag"]["truth_score_own_init"], r["diag"]["truth_score_true_init"]) if v is not None)
             for r in wrong if r["class"] == cls]
        if d:
            print(f"{cls}: truth better by median {np.median(d):.1f} (n={len(d)}); own-init reaches it in "
                  f"{sum(1 for r in wrong if r['class'] == cls and r['diag']['truth_score_own_init'] is not None and r['diag']['truth_score_own_init'] < r['diag']['winner_score'])}")


def compare(base_f, *fs):
    B = load(base_f)
    bx = np.mean([r["final"]["exact"] for r in B.values()])
    print(f"{'variant':28} {'exact':>6} {'W/L vs base':>12} {'p':>8} {'med s':>6} {'fits':>6}")
    print(f"{base_f:28} {bx:6.3f} {'':>12} {'':>8} {np.median([r.get('seconds', 0) for r in B.values()]):6.1f}"
          f" {np.median([r.get('fits', 0) for r in B.values()]):6.0f}")
    for f in fs:
        V = load(f)
        common = [i for i in V if i in B]
        w = sum(V[i]["final"]["exact"] > B[i]["final"]["exact"] for i in common)
        l = sum(V[i]["final"]["exact"] < B[i]["final"]["exact"] for i in common)
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        print(f"{f:28} {np.mean([V[i]['final']['exact'] for i in common]):6.3f} {w:5d}/{l:<5d} {p:9.1e}"
              f" {np.median([V[i].get('seconds', 0) for i in common]):6.1f} {np.median([V[i].get('fits', 0) for i in common]):6.0f}")


def cells(*fs):
    Rs = [load(f) for f in fs]
    keys = sorted({(r["noise"], r["n"]) for r in Rs[0].values()})
    print("cell        " + " ".join(f"{f.split('/')[-1][:14]:>14}" for f in fs))
    for k in keys:
        print(f"{k[0]:<5} {k[1]:<5} " + " ".join(
            f"{np.mean([r['final']['exact'] for r in R.values() if (r['noise'], r['n']) == k]):14.3f}" for R in Rs))
    print("by system (count exact):")
    for s in sorted({r["system"] for r in Rs[0].values()}):
        print(f"  {s:11} " + " ".join(f"{sum(r['final']['exact'] for r in R.values() if r['system'] == s):14.0f}"
                                     for R in Rs))


if __name__ == "__main__":
    cmd, *files = sys.argv[1:]
    {"anatomy": anatomy, "compare": compare, "cells": cells}[cmd](*files)
