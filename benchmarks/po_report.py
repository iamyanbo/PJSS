"""Summaries of partial-observation runs (JSON lines from po_bench.py).

    python benchmarks/po_report.py results_po/dev_A.jsonl [more.jsonl ...] [--ref po] [--md]
Tables: exact (gauge-aware) rate per arm x configuration ("strict": exact with k <= k_true, i.e. a minimal
representative, not a denser gauge-equivalent form), per arm x system x noise, median coefficient error over
exact runs, median valid time (Lyapunov times), median seconds; paired W/L and two-sided sign test vs --ref.
"""
import argparse
import json
from collections import defaultdict
from math import comb

import numpy as np

CFG_ORDER = ["xz", "xy", "yz", "x"]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def strict(r):
    """Exact up to gauge AND no more terms than the reference form (minimal representative)."""
    return bool(r.get("exact")) and r.get("k", 99) <= r.get("k_true", -1)


def load(paths):
    recs = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            recs[(r["i"], r["cfg"], r["arm"])] = r
    return list(recs.values())


def table(recs, ref=None, md=False):
    arms = sorted({r["arm"] for r in recs}, key=lambda a: (a != ref, a))
    by = defaultdict(list)
    for r in recs:
        by[r["arm"]].append(r)
    cfgs = [c for c in CFG_ORDER if any(r["cfg"] == c for r in recs)]
    systems = sorted({r["sys"] for r in recs})
    noises = sorted({r["noise"] for r in recs})
    lines = []
    hdr = ["arm"] + cfgs + ["all", "strict", "errors", "med coef err", "med VT", "med s"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for a in arms:
        rs = by[a]
        row = [a]
        for c in cfgs:
            x = [r for r in rs if r["cfg"] == c]
            row.append(f"{sum(r['exact'] for r in x)}/{len(x)}")
        row.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}")
        row.append(str(sum(strict(r) for r in rs)))
        row.append(str(sum("error" in r for r in rs)))
        ce = [r["coef_err"] for r in rs if r.get("exact")]
        row.append(f"{np.median(ce):.2%}" if ce else "-")
        vt = [r["vt"] for r in rs if "vt" in r]
        row.append(f"{np.median(vt):.2f}" if vt else "-")
        row.append(f"{np.median([r['sec'] for r in rs]):.0f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    hdr = ["arm"] + [f"{s[:3]} {c} {int(n * 100)}%" for s in systems for n in noises for c in ["*"]]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for a in arms:
        row = [a]
        for s in systems:
            for n in noises:
                x = [r for r in by[a] if r["sys"] == s and r["noise"] == n]
                row.append(f"{sum(r['exact'] for r in x)}/{len(x)}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    hdr = ["arm"] + [f"{s[:3]} {c}" for s in systems for c in cfgs]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for a in arms:
        row = [a]
        for s in systems:
            for c in cfgs:
                x = [r for r in by[a] if r["sys"] == s and r["cfg"] == c]
                row.append(f"{sum(r['exact'] for r in x)}/{len(x)}")
        lines.append("| " + " | ".join(row) + " |")
    if ref and ref in by:
        lines.append("")
        lines.append(f"Paired vs {ref} (wins of {ref} / losses of {ref}, sign test p):")
        rmap = {(r["i"], r["cfg"]): r["exact"] for r in by[ref]}
        for a in arms:
            if a == ref:
                continue
            w = l = 0
            for r in by[a]:
                k = (r["i"], r["cfg"])
                if k in rmap:
                    w += rmap[k] and not r["exact"]
                    l += r["exact"] and not rmap[k]
            lines.append(f"- {ref} vs {a}: {w}/{l}, p = {sign_p(w, l):.2g}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default="po")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    print(table(load(a.files), a.ref, a.md))


if __name__ == "__main__":
    main()
