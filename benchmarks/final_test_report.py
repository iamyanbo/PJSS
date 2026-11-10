"""Frozen test pass of final_model.py against each archive's previous best (paired by run index).

  python final_test_report.py [results_final]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_final")


def recs(f):
    return {r["index"]: r for r in json.loads(Path(f).read_text())["records"]}


sq = lambda r: r["final"]["exact"]
cj = lambda r: r["joint"]["0.03"]["final"]["exact"]
ar = lambda r: r["variants"]["ar"]["final"]["exact"]
bj = lambda r: r["joint"]["exact"]

# test file, previous-best file + getter + label, optional config C file for reference
TESTS = [
    ("sealed", "test_sealed", ("results_squeeze/test_joint_sealed_deg2.json", sq, "C + prior"),
     ("results_joint/C_joint_sealed.json", cj)),
    ("confirmation", "test_confirm", ("results_squeeze/test_hybrid_confirm_deg2.json", sq, "C + prior"),
     ("results_joint/C_hybrid_confirm.json", cj)),
    ("held-out systems, new seeds", "test_heldout2", ("results_squeeze/test_heldout2_deg2.json", sq, "C + prior"),
     ("results_joint/C_heldout2.json", cj)),
    ("coloured, new seeds", "test_colored", ("results_colored/test_sealed_colored.json", ar, "AR(1)"),
     ("results_joint/C_sealed_colored.json", cj)),
    ("heavy-tailed, new seeds", "test_student", ("results_joint/C_sealed_student.json", cj, "C"), None),
    ("coarse, new seeds", "test_coarse", ("results_joint/C_sealed_coarse.json", cj, "C"), None),
    ("degree 4 (sealed every 3rd)", "test_deg4", ("results_biglib/test_deg4.json", bj, "biglib C"), None),
    ("degree 5 (sealed every 3rd)", "test_deg5", ("results_biglib/test_deg5.json", bj, "biglib C"), None),
    ("deg 3 + trig, polynomial systems", "test_mixedpoly", ("results_biglib/test_mixed_poly.json", bj, "biglib C"),
     None),
    ("deg 3 + trig, Thomas", "test_trig", ("results_biglib/test_mixed_trig.json", bj, "biglib C"), None),
    ("D = 4-10 (nd test)", "test_nd", ("results_nd/test.json", bj, "joint_nd C"), None),
]


def main():
    print("| Archive | Runs | Previous best | Prev. | C | **Final** | W/L vs prev. | p | Flagged | "
          "Unflagged acc. | AR(1) chosen | Median s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = [0, 0]
    rows = {}
    for label, name, (pf, pg, plab), cref in TESTS:
        f = D / f"{name}.json"
        if not f.exists():
            print(f"| {label} | missing | | | | | | | | | | |")
            continue
        T = recs(f)
        P = recs(pf)
        common = [i for i in T if i in P]
        fin = np.array([T[i]["arms"]["final"].get("exact", 0) for i in common])
        prev = np.array([pg(P[i]) for i in common])
        w, l = int((fin > prev).sum()), int((fin < prev).sum())
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        pooled[0] += w
        pooled[1] += l
        c = ""
        if cref:
            C = recs(cref[0])
            c = f"{np.mean([cref[1](C[i]) for i in common if i in C]):.3f}"
        fl = np.array([T[i]["arms"]["final"].get("ambiguous", False) for i in common])
        arc = np.mean([T[i]["arms"]["final"].get("noise_model") == "ar1" for i in common])
        err = sum("error" in T[i]["arms"]["final"] for i in common)
        print(f"| {label} | {len(common)} | {plab} | {prev.mean():.3f} | {c} | **{fin.mean():.3f}** | {w}/{l} | "
              f"{p:.2g} | {fl.mean():.0%} | {fin[~fl].mean():.3f} | {arc:.0%} | "
              f"{np.median([T[i]['arms']['final'].get('seconds', 0) for i in common]):.1f} |"
              + (f" errors {err}" if err else ""))
        rows[name] = (T, P, pg, common)
    w, l = pooled
    print(f"\nPooled over all test archives: {w}/{l} (p = {binomtest(w, w + l).pvalue if w + l else 1:.2g})")
    # hard 40 (subset of the confirmation archive)
    if "test_confirm" in rows:
        T, P, pg, _ = rows["test_confirm"]
        H = json.loads((D / "hard40_idx.json").read_text())
        dedi = {r["index"]: r["dedi"]["exact"] for g in ("dedi_hard20.json", "dedi_hard20_rest.json")
                for r in json.loads(Path("results_hybrid", g).read_text())["records"]}
        fin = {i: T[i]["arms"]["final"]["exact"] for i in H}
        print(f"\nHard 40: final {sum(fin.values()):.0f}/40, previous best (C + prior) {sum(pg(P[i]) for i in H):.0f}/40, "
              f"DEDi {sum(dedi.values()):.0f}/40; vs DEDi {sum(fin[i] > dedi[i] for i in H)}/"
              f"{sum(fin[i] < dedi[i] for i in H)}")
        for s in ("lorenz", "chen", "rossler", "rabinovich"):
            hs = [i for i in H if T[i]["system"] == s]
            print(f"  {s}: final {sum(fin[i] for i in hs):.0f}, prev {sum(pg(P[i]) for i in hs):.0f}, "
                  f"DEDi {sum(dedi[i] for i in hs):.0f}")
    # per-archive detail of changed runs
    print("\nChanged runs (system, noise, n: final vs previous):")
    for name, (T, P, pg, common) in rows.items():
        ch = [(T[i]["system"], T[i]["noise"], T[i]["n"], int(T[i]["arms"]["final"].get("exact", 0)), int(pg(P[i])),
               T[i]["arms"]["final"].get("noise_model")) for i in common
              if T[i]["arms"]["final"].get("exact", 0) != pg(P[i])]
        if ch:
            print(f"  {name}: " + "; ".join(f"{s} {nz} {n}: {a} vs {b} [{m}]" for s, nz, n, a, b, m in ch))
    if "test_nd" in rows:
        T, P, pg, common = rows["test_nd"]
        print("\nnd by system (final / previous):")
        for s in sorted({T[i]["system"] for i in common}):
            cs = [i for i in common if T[i]["system"] == s]
            print(f"  {s}: {np.mean([T[i]['arms']['final']['exact'] for i in cs]):.3f} / "
                  f"{np.mean([pg(P[i]) for i in cs]):.3f}")


if __name__ == "__main__":
    main()
