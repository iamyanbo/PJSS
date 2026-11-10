"""Report for the enlarged hard-case comparison against DEDi (docs/reports/hard120.md).

  python benchmarks/hard120_report.py [results_hard120]

Inputs: results_hard120/ours_hard120.json (final_bench arms final, C; weak = starts[0]),
results_hard120/dedi_hard120.json (dedi_bench, pre-registered order results_hard120/dedi_order.json),
and for pooling the Lorenz/Chen/Rossler part of the earlier hard-40: results_final/test_confirm.json (final),
results_hybrid/dedi_hard20*.json (DEDi), results_hard120/ours_hard40_repro.json (C, weak, fresh-start final).
Analysis set (pre-registered, results_hard120/PREREGISTRATION.json): the longest prefix of the DEDi order in
which every run finished, truncated to whole (lorenz, chen, rossler) triples.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_hard120")
SYS = ("lorenz", "chen", "rossler")
NAME = {"lorenz": "Lorenz", "chen": "Chen", "rossler": "Rössler"}


def load(f):
    return {r["index"]: r for r in json.loads(Path(f).read_text())["records"]}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0.0), min(c + h, 1.0)


def rate(v):
    k, n = int(sum(v)), len(v)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}; {lo:.0%}–{hi:.0%})" if n else "–"


def sign(a, b):
    w = int(sum(x > y for x, y in zip(a, b)))
    l = int(sum(x < y for x, y in zip(a, b)))
    p = binomtest(w, w + l).pvalue if w + l else 1.0
    return w, l, p


def main():
    ours = load(D / "ours_hard120.json")
    order = json.loads((D / "dedi_order.json").read_text())
    dedi = {}  # session 1 (until the Spark crash at ~21:02) + resumed session (same order, finished runs skipped)
    for f in ("dedi_hard120.json", "dedi_hard120_part2.json"):
        if (D / f).exists():
            dedi |= load(D / f)
    done = 0
    while done < len(order) and order[done] in dedi:
        done += 1
    A = order[: done - done % 3]
    extra = [i for i in dedi if i not in A]
    fin = lambda r: r["arms"]["final"]["exact"]
    C = lambda r: r["arms"]["C"]["exact"]
    wk = lambda r: r["weak"]
    de = lambda r: r["dedi"]["exact"]
    errs = {a: sum("error" in r["arms"][a] for r in ours.values()) for a in ("final", "C")}
    derr = sum("dedi_error" in r for r in dedi.values())

    print(f"DEDi finished {len(dedi)} runs; analysis set = first {len(A)} of the pre-registered order "
          f"({len(A) // 3} per system); finished outside it: {len(extra)}. Errors: ours {errs}, DEDi {derr}.\n")

    print("## Table 1: all 120 new runs, our methods\n")
    print("| System (20%, N = 2000) | n | Weak χ² best-subset | Config C | **Final model** |")
    print("| --- | --- | --- | --- | --- |")
    for s in SYS + (None,):
        ix = [i for i in ours if s is None or ours[i]["system"] == s]
        lab = "**All**" if s is None else NAME[s]
        print(f"| {lab} | {len(ix)} | {rate([wk(ours[i]) for i in ix])} | {rate([C(ours[i]) for i in ix])} | "
              f"{rate([fin(ours[i]) for i in ix])} |")
    w, l, p = sign([fin(ours[i]) for i in ours], [C(ours[i]) for i in ours])
    print(f"\nFinal vs config C on 120: {w}/{l} (p = {p:.2g}).")

    print("\n## Table 2: head to head with DEDi on the pre-registered analysis set\n")
    print("| System | n | Weak χ² | Config C | **Final** | DEDi | Final vs DEDi W/L | sign-test p |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in SYS + (None,):
        ix = [i for i in A if s is None or ours[i]["system"] == s]
        if not ix:
            continue
        w, l, p = sign([fin(ours[i]) for i in ix], [de(dedi[i]) for i in ix])
        lab = "**All (new)**" if s is None else NAME[s]
        print(f"| {lab} | {len(ix)} | {rate([wk(ours[i]) for i in ix])} | {rate([C(ours[i]) for i in ix])} | "
              f"{rate([fin(ours[i]) for i in ix])} | {rate([de(dedi[i]) for i in ix])} | {w}/{l} | {p:.2g} |")

    # earlier hard-40, Lorenz/Chen/Rossler part
    H = json.loads(Path("results_final/hard40_idx.json").read_text())
    T = load("results_final/test_confirm.json")
    OD = {}
    for g in ("dedi_hard20.json", "dedi_hard20_rest.json"):
        OD |= load(Path("results_hybrid") / g)
    R = load(D / "ours_hard40_repro.json")
    H3 = [i for i in H if T[i]["system"] in SYS]
    same = sum(fin(R[i]) == T[i]["arms"]["final"]["exact"] for i in H)
    print(f"\nReproduction of the hard-40 final model with this code and fresh (uncached) starts: "
          f"{sum(fin(R[i]) for i in H):.0f}/40 vs published {sum(T[i]['arms']['final']['exact'] for i in H):.0f}/40; "
          f"{same}/40 runs identical.")

    print("\n## Table 3: pooled (new analysis set + earlier hard-40 Lorenz/Chen/Rössler)\n")
    print("| System | n | Weak χ² | Config C | **Final** | DEDi | Final vs DEDi W/L | sign-test p |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    pool = [("new", i) for i in A] + [("old", i) for i in H3]
    get = lambda src, i, k: ({"fin": fin, "C": C, "wk": wk}[k](ours[i]) if src == "new" else
                             {"fin": lambda r: T[r]["arms"]["final"]["exact"], "C": lambda r: C(R[r]),
                              "wk": lambda r: wk(R[r])}[k](i))
    gd = lambda src, i: de(dedi[i]) if src == "new" else de(OD[i])
    sysof = lambda src, i: ours[i]["system"] if src == "new" else T[i]["system"]
    for s in SYS + (None,):
        ix = [(a, i) for a, i in pool if s is None or sysof(a, i) == s]
        f = [get(a, i, "fin") for a, i in ix]
        d = [gd(a, i) for a, i in ix]
        w, l, p = sign(f, d)
        lab = "**All (pooled)**" if s is None else NAME[s]
        print(f"| {lab} | {len(ix)} | {rate([get(a, i, 'wk') for a, i in ix])} | {rate([get(a, i, 'C') for a, i in ix])} | "
              f"{rate(f)} | {rate(d)} | {w}/{l} | {p:.2g} |")

    if extra:
        f = [fin(ours[i]) for i in extra]
        d = [de(dedi[i]) for i in extra]
        w, l, p = sign(f, d)
        print(f"\nFinished outside the analysis set (not in the headline): {len(extra)} runs, final {sum(f):.0f}, "
              f"DEDi {sum(d):.0f}, W/L {w}/{l}.")

    print("\n## Runtime and coefficients\n")
    tf = [ours[i]["starts_seconds"] + ours[i]["a_seconds"] + ours[i]["arms"]["final"]["seconds"] for i in ours]
    print(f"- Final model, per run (one core, includes the weak starts): median {np.median(tf):.1f} s, "
          f"max {max(tf):.1f} s.")
    if dedi:
        td = [dedi[i]["seconds"] for i in dedi]
        print(f"- DEDi, per run wall time (one worker per pinned core: 12 workers on 12 cores, then 14 on 14 after the restart; XLA threads share those cores): median {np.median(td) / 3600:.2f} h, "
              f"range {min(td) / 3600:.2f}–{max(td) / 3600:.2f} h; ratio of medians ≈ {np.median(td) / np.median(tf):.0f}×.")
        ch = {}
        for i in dedi:
            c = tuple(dedi[i].get("chosen", []))
            ch[c] = ch.get(c, 0) + 1
        print(f"- DEDi chosen (λ, R): {ch}")
        both = [i for i in A if fin(ours[i]) and de(dedi[i])]
        if both:
            ef = [ours[i]["arms"]["final"]["coef_err"] for i in both]
            ed = [dedi[i]["dedi"]["coef_err"] for i in both]
            print(f"- Coefficient error where both are exact ({len(both)} runs): median final {np.median(ef):.2%}, "
                  f"DEDi {np.median(ed):.2%}; DEDi lower in {sum(b < a for a, b in zip(ef, ed))}.")
        fp = lambda r: r.get("fp", 0)
        print(f"- False positives on the analysis set: final {sum(fp(ours[i]['arms']['final']) for i in A):.0f}, "
              f"DEDi {sum(fp(dedi[i]['dedi']) for i in A):.0f}; false negatives: final "
              f"{sum(ours[i]['arms']['final'].get('fn', 0) for i in A):.0f}, DEDi {sum(dedi[i]['dedi'].get('fn', 0) for i in A):.0f}.")


if __name__ == "__main__":
    main()
