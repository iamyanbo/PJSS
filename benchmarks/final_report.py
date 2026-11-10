"""Summary tables for joint-state config C across design and test archives (results_joint/C_*.json)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from joint_report import pick

SETS = [("instances", "dev (design)"), ("instances_colored", "dev coloured (design)"),
        ("instances_student", "dev heavy-tailed (design)"), ("instances_coarse", "dev coarse (design)"),
        ("heldout", "held-out systems (design)"), ("joint_sealed", "sealed (test)"),
        ("hybrid_confirm", "confirmation (test)"), ("heldout2", "held-out systems, new seeds (test)"),
        ("sealed_colored", "coloured, new seeds (test)"), ("sealed_student", "heavy-tailed, new seeds (test)"),
        ("sealed_coarse", "coarse, new seeds (test)")]


def joint(r):
    j = r["joint"]["0.03"]
    return pick(j["candidates"], "bic", r["n"], j.get("phi", 1.0)) if j.get("candidates") else {"exact": 0.0}


def gap(r):
    j = r["joint"]["0.03"]
    n, phi = r["n"], j.get("phi", 1.0)
    sc = sorted(c["data"] + c["model"] + c["val"] + phi * c["k"] * np.log(3 * n) for c in j.get("candidates", []))
    return sc[1] - sc[0] if len(sc) > 1 else np.inf


def main():
    print(f"{'set':38} {'runs':>4} {'weak':>6} {'hybrid':>6} {'joint':>6}  W/L vs hybrid   p      flagged  acc(unflagged)  s/run")
    for key, label in SETS:
        f = Path(f"results_joint/C_{key}.json")
        if not f.exists():
            print(f"{label:38} (missing)")
            continue
        R = json.loads(f.read_text())["records"]
        jx = np.array([joint(r)["exact"] for r in R])
        hx = np.array([r["hyb_bic10"]["exact"] for r in R])
        wx = np.array([r["weak"]["exact"] for r in R])
        w, l = int((jx > hx).sum()), int((jx < hx).sum())
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        fl = np.array([gap(r) < 2 for r in R])
        print(f"{label:38} {len(R):4d} {wx.mean():6.3f} {hx.mean():6.3f} {jx.mean():6.3f}  {w:3d}/{l:<3d}  {p:9.1e}"
              f"   {fl.mean():5.2f}   {jx[~fl].mean():6.3f}      {np.median([r['seconds'] for r in R]):5.1f}")
    for key in ("joint_sealed", "hybrid_confirm", "heldout2"):
        f = Path(f"results_joint/C_{key}.json")
        if not f.exists():
            continue
        R = json.loads(f.read_text())["records"]
        print(f"\n{key} by cell (weak / hybrid / joint):")
        for cell in sorted({(r["noise"], r["n"]) for r in R}):
            rs = [r for r in R if (r["noise"], r["n"]) == cell]
            print(f"  {cell[0]:<5} {cell[1]:<5}" + "".join(f" {np.mean(v):5.2f}" for v in (
                [r["weak"]["exact"] for r in rs], [r["hyb_bic10"]["exact"] for r in rs], [joint(r)["exact"] for r in rs])))
        print("  by system (weak / hybrid / joint, of", len([r for r in R if r["system"] == R[0]["system"]]), "):")
        for s in sorted({r["system"] for r in R}):
            rs = [r for r in R if r["system"] == s]
            print(f"   {s:11} {sum(r['weak']['exact'] for r in rs):3.0f} {sum(r['hyb_bic10']['exact'] for r in rs):3.0f}"
                  f" {sum(joint(r)['exact'] for r in rs):3.0f}")
    f = Path("results_joint/C_hybrid_confirm.json")
    if f.exists():
        R = {r["index"]: r for r in json.loads(f.read_text())["records"]}
        D = [r for g in ("dedi_hard20.json", "dedi_hard20_rest.json")
             for r in json.loads(Path("results_hybrid", g).read_text())["records"]]
        jx = {r["index"]: R[r["index"]]["joint"]["0.03"]["final"] for r in D}
        w = sum(jx[r["index"]]["exact"] > r["dedi"]["exact"] for r in D)
        l = sum(jx[r["index"]]["exact"] < r["dedi"]["exact"] for r in D)
        both = [r for r in D if jx[r["index"]]["exact"] and r["dedi"]["exact"]]
        print(f"\nDEDi hard 40 (20% noise, N=2000): joint {sum(jx[r['index']]['exact'] for r in D):.0f}/40, "
              f"DEDi {sum(r['dedi']['exact'] for r in D):.0f}/40, joint wins {w} loses {l}")
        for s in ("lorenz", "chen", "rossler", "rabinovich"):
            rs = [r for r in D if r["system"] == s]
            print(f"   {s:11} joint {sum(jx[r['index']]['exact'] for r in rs):.0f}  DEDi {sum(r['dedi']['exact'] for r in rs):.0f}")
        if both:
            a = [jx[r["index"]]["coef_err"] for r in both]
            b = [r["dedi"]["coef_err"] for r in both]
            print(f"   both exact {len(both)}: median coefficient error joint {100 * np.median(a):.2f}% vs DEDi "
                  f"{100 * np.median(b):.2f}% (joint lower in {sum(x < y for x, y in zip(a, b))})")
        print(f"   seconds per run: joint median {np.median([R[r['index']]['seconds'] for r in D]):.1f} "
              f"(incl. weak + flow), DEDi median {np.median([r['seconds'] for r in D]):.0f}")


if __name__ == "__main__":
    main()
