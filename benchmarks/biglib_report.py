"""Tables for results_biglib runs: weak (big library) vs joint (big library), with the standard 20-term
joint / weak results on the same instances as reference when a matching results_joint/C_<archive>.json exists."""
import argparse
import json
from pathlib import Path

import numpy as np


def gap(rec):
    c = rec["joint"].get("candidates", [])
    if len(c) < 2:
        return np.inf
    phi = rec["joint"].get("phi", 1.0)
    sc = sorted(x["data"] + x["model"] + x["val"] + phi * x["k"] * np.log(3 * rec["n"]) for x in c)
    return sc[1] - sc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", help="results_joint/C_*.json for 20-term reference on the same indices")
    args = ap.parse_args()
    for f in args.files:
        R = json.loads(Path(f).read_text())["records"]
        ref = {}
        if args.ref:
            ref = {r["index"]: r for r in json.loads(Path(args.ref).read_text())["records"]}
        print(f"== {f}  ({len(R)} runs, library {R[0]['nlib']} terms/eq)")
        hdr = f"{'cell':14} {'n':>4} {'weak':>6} {'joint':>6}"
        if ref:
            hdr += f" {'weak20':>7} {'joint20':>8}"
        print(hdr)

        def row(rs):
            v = [np.mean([r["weak_chi2s"]["exact"] for r in rs]), np.mean([r["joint"]["exact"] for r in rs])]
            s = f"{len(rs):4d} {v[0]:6.3f} {v[1]:6.3f}"
            if ref:
                rr = [ref[r["index"]] for r in rs if r["index"] in ref]
                if rr:
                    s += f" {np.mean([x['weak']['exact'] for x in rr]):7.3f}"
                    s += f" {np.mean([x['joint']['0.03']['final']['exact'] for x in rr]):8.3f}"
            return s

        for cell in sorted({(r["noise"], r["n"]) for r in R}):
            print(f"{cell[0]:<5} {cell[1]:<8}" + row([r for r in R if (r["noise"], r["n"]) == cell]))
        print(f"{'all':14}" + row(R))
        for s in sorted({r["system"] for r in R}):
            rs = [r for r in R if r["system"] == s]
            print(f"   {s:11}" + row(rs))
        jx = np.array([r["joint"]["exact"] for r in R])
        wx = np.array([r["weak_chi2s"]["exact"] for r in R])
        fl = np.array([gap(r) < 2 for r in R])
        fp = np.array([r["joint"].get("fp", 0) for r in R])
        print(f"joint vs weak W/L {(jx > wx).sum()}/{(jx < wx).sum()}; flagged {fl.mean():.2f}, "
              f"acc unflagged {jx[~fl].mean() if (~fl).any() else float('nan'):.3f}; "
              f"mean false-positive terms {fp.mean():.2f}; errors {sum('error' in r['joint'] for r in R)}")
        print(f"seconds: weak median {np.median([r['weak_seconds'] for r in R]):.1f}, joint median "
              f"{np.median([r['joint_seconds'] for r in R]):.1f}, p90 {np.percentile([r['joint_seconds'] for r in R], 90):.1f}\n")


if __name__ == "__main__":
    main()
