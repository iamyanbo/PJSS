"""Tables for dysts_bench results.

    python dysts_report.py results_dysts/test_c.json [results_dysts/test_c_prior.json ...] [--data data/dysts_test.npz]

Per result file: overall exact-support rate, split new systems / overlap with our SYSTEMS, HELDOUT or nd_systems
(same system, any parameters) / support twins (another system with exactly one of our supports), by noise and N, and per system. With two or more files: paired W/L of
each file against the first on the runs both completed (status ok), with a two-sided sign test.
Runs with status "unsupported" (c_prior for D >= 4) are excluded; errors count as failures.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def load(f):
    J = json.load(open(f))
    return J.get("method", Path(f).stem), {r["index"]: r for r in J["records"] if r.get("status") != "unsupported"}


def group(sys_info, name):
    """overlap = identical / same system other params / same family other D as one of ours;
    twin = different system with exactly the support of one of ours; new = neither"""
    o = sys_info.get(name, {}).get("overlap")
    return "new" if o is None else ("twin" if o == "support_twin" else "overlap")


def rate(rs, key="final"):
    v = [r[key]["exact"] if key == "final" else r[key] for r in rs if (key == "final" or key in r)]
    return (np.mean(v), len(v)) if v else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default=None, help="archive whose sidecar gives overlap flags (default: from args)")
    ap.add_argument("--per-system", action="store_true", default=True)
    a = ap.parse_args()
    runs = [load(f) for f in a.results]
    data = a.data or json.load(open(a.results[0]))["args"]["data"]
    side = json.loads(Path(data).with_suffix(".json").read_text())["systems"]
    for (m, R), f in zip(runs, a.results):
        rs = list(R.values())
        print(f"\n== {f}  ({m}, {len(rs)} runs, errors {sum(r.get('status') == 'error' for r in rs)}, "
              f"median {np.median([r['seconds'] for r in rs]):.1f} s/run)")
        for lab, sub in [("all", rs), ("new systems", [r for r in rs if group(side, r["system"]) == "new"]),
                         ("overlap", [r for r in rs if group(side, r["system"]) == "overlap"]),
                         ("support twin", [r for r in rs if group(side, r["system"]) == "twin"]),
                         ("D = 3", [r for r in rs if r["D"] == 3]), ("D >= 4", [r for r in rs if r["D"] >= 4])]:
            x, n = rate(sub)
            w, nw = rate(sub, "weak")
            print(f"  {lab:12} exact {x:.3f} (n={n})" + (f"   weak {w:.3f} (n={nw})" if nw else ""))
        cell = defaultdict(list)
        for r in rs:
            cell[(r["n"], r["noise"])].append(r)
        print("  by cell: " + "  ".join(f"N{k[0]}/{k[1]}: {rate(v)[0]:.2f}" for k, v in sorted(cell.items())))
        if a.per_system:
            by = defaultdict(list)
            for r in rs:
                by[r["system"]].append(r)
            print(f"  {'system':22} {'D':>2} {'grp':8} {'exact':>6} {'n':>4} {'weak':>6}")
            for s in sorted(by, key=lambda s: (side.get(s, {}).get("D", 0), s)):
                x, n = rate(by[s])
                w, nw = rate(by[s], "weak")
                print(f"  {s:22} {side.get(s, {}).get('D', '?'):>2} {group(side, s):8} {x:6.3f} {n:4d} "
                      + (f"{w:6.3f}" if nw else "     -"))
    if len(runs) > 1:
        m0, B = runs[0]
        print(f"\n== paired vs {m0} ({a.results[0]})")
        for (m, R), f in zip(runs[1:], a.results[1:]):
            common = [i for i in R if i in B and R[i].get("status") == "ok" and B[i].get("status") == "ok"]
            for lab, ids in [("all", common), ("new", [i for i in common if group(side, R[i]["system"]) == "new"])]:
                w = sum(R[i]["final"]["exact"] > B[i]["final"]["exact"] for i in ids)
                l = sum(R[i]["final"]["exact"] < B[i]["final"]["exact"] for i in ids)
                p = binomtest(w, w + l).pvalue if w + l else 1.0
                xb = np.mean([B[i]["final"]["exact"] for i in ids]) if ids else np.nan
                xr = np.mean([R[i]["final"]["exact"] for i in ids]) if ids else np.nan
                print(f"  {m:10} {lab:4} n={len(ids):5}  {m0} {xb:.3f} -> {m} {xr:.3f}   W/L {w}/{l}  p={p:.2g}")


if __name__ == "__main__":
    main()
