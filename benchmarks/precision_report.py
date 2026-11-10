"""Tables for the v1.2 pre-registered test pass (results_precision/PREREGISTRATION_v12.md).

    python benchmarks/precision_report.py > results_precision/test_report.md
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

O = Path("results_precision")


def load(p):
    return json.loads(Path(p).read_text())["records"]


def sup(xi):
    return (np.asarray(xi) != 0).tolist()


def wl(R, a="final_v11", b="final_v12"):
    W = sum(r["arms"][b]["exact"] > r["arms"][a]["exact"] for r in R)
    L = sum(r["arms"][a]["exact"] > r["arms"][b]["exact"] for r in R)
    return W, L, (binomtest(W, W + L).pvalue if W + L else 1.0)


def main():
    print("# v1.2 test pass: tables\n")
    R = load(O / "test_clean.json")
    err = [(r["index"], a, v["error"]) for r in R for a, v in r["arms"].items() if "error" in v]
    W, L, p = wl(R)
    e1 = np.mean([r["arms"]["final_v11"]["exact"] for r in R])
    e2 = np.mean([r["arms"]["final_v12"]["exact"] for r in R])
    act = sum(bool((r["arms"]["final_v12"].get("floor") or {}).get("active")) for r in R)
    print(f"## Primary: sealed_clean.npz ({len(R)} runs, eta = 0)\n")
    print(f"v1.1 exact {e1:.3f}, v1.2 exact {e2:.3f}; **W/L {W}/{L}, two-sided sign test p = {p:.2g}**; "
          f"floor active in {act} runs; errors {len(err)} {err[:3]}\n")
    for k in ("fp", "fn"):
        print(f"- mean {k}: v1.1 {np.mean([r['arms']['final_v11'][k] for r in R]):.2f}, "
              f"v1.2 {np.mean([r['arms']['final_v12'][k] for r in R]):.2f}")
    print("\n| Cell | n | v1.1 | v1.2 | W/L |\n| --- | --- | --- | --- | --- |")
    groups = defaultdict(list)
    for r in R:
        groups[f"N = {r['n']}"].append(r)
        groups[r["system"]].append(r)
    for g, rs in groups.items():
        w, l, _ = wl(rs)
        print(f"| {g} | {len(rs)} | {np.mean([r['arms']['final_v11']['exact'] for r in rs]):.3f} | "
              f"{np.mean([r['arms']['final_v12']['exact'] for r in rs]):.3f} | {w}/{l} |")
    losses = [r for r in R if r["arms"]["final_v11"]["exact"] > r["arms"]["final_v12"]["exact"]]
    for r in losses:
        print(f"\nloss: {r['system']} N={r['n']} index {r['index']} v1.2 fp {r['arms']['final_v12']['fp']} "
              f"fn {r['arms']['final_v12']['fn']}")
    rem = [r for r in R if r["arms"]["final_v12"]["exact"] < 1]
    print(f"\nv1.2 non-exact runs: {len(rem)}: " + ", ".join(f"{r['system']}/{r['n']} (fp {r['arms']['final_v12']['fp']}, "
                                                          f"fn {r['arms']['final_v12']['fn']})" for r in rem))

    print("\n## Guards\n")
    print("| Archive | Runs | floor active | coefficients changed | support changed | v1.1 exact | v1.2 exact | W/L | "
          "v1.1 arm = stored v1.1 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    stored = {"guard_white": "results_final_v11/guard_sealed.json", "guard_student": "results_final_v11/guard_student.json",
              "guard_colored": "results_final_v11/test_colored2.json"}
    for g in ("guard_white", "guard_colored", "guard_student", "guard_nd"):
        f = O / f"{g}.json"
        if not f.exists():
            print(f"| {g} | missing | | | | | | | |")
            continue
        R = load(f)
        R = [r for r in R if "error" not in r["arms"]["final_v11"] and "error" not in r["arms"]["final_v12"]]
        act = sum(bool((r["arms"]["final_v12"].get("floor") or {}).get("active")) for r in R)
        ch = sum(r["arms"]["final_v11"]["xi"] != r["arms"]["final_v12"]["xi"] for r in R)
        sc = sum(sup(r["arms"]["final_v11"]["xi"]) != sup(r["arms"]["final_v12"]["xi"]) for r in R)
        w, l, _ = wl(R)
        rep = "-"
        if g in stored and Path(stored[g]).exists():
            S = {r["index"]: r["arms"]["final"]["xi"] for r in load(stored[g])}
            same = sum(S.get(r["index"]) == r["arms"]["final_v11"]["xi"] for r in R)
            rep = f"{same}/{len(R)}"
        print(f"| {g} | {len(R)} | {act} | {ch} | {sc} | {np.mean([r['arms']['final_v11']['exact'] for r in R]):.3f} | "
              f"{np.mean([r['arms']['final_v12']['exact'] for r in R]):.3f} | {w}/{l} | {rep} |")

    print("\n## Secondary (NOT independent): ODEBench sigma = 0, fit IC 0\n")
    o = {}
    for v in ("1.1", "1.2"):
        f = O / f"odebench_s0_v{v}.json"
        if f.exists():
            o[v] = {r["id"]: r for r in load(f)}
    if len(o) == 2:
        ids = sorted(o["1.1"])
        inl = [i for i in ids if "exact" in o["1.1"][i] and o["1.1"][i]["class"] == "in_library"] or \
              [i for i in ids if "exact" in o["1.1"][i]]
        print("| Subset | n | v1.1 | v1.2 |\n| --- | --- | --- | --- |")
        for lab, key, sel in (("exact, in library", "exact", inl), ("reconstruction R2 > 0.9", "r2_rec", ids),
                              ("generalization R2 > 0.9", "r2_gen", ids)):
            f = (lambda x: x) if key == "exact" else (lambda x: float(x > 0.9))
            print(f"| {lab} | {len(sel)} | {np.mean([f(o['1.1'][i].get(key, 0)) for i in sel]):.3f} | "
                  f"{np.mean([f(o['1.2'][i].get(key, 0)) for i in sel]):.3f} |")
        W = sum(o["1.2"][i].get("exact", 0) > o["1.1"][i].get("exact", 0) for i in inl)
        L = sum(o["1.1"][i].get("exact", 0) > o["1.2"][i].get("exact", 0) for i in inl)
        print(f"\nexact W/L (in library) {W}/{L}; errors v1.1 {sum('error' in r for r in o['1.1'].values())}, "
              f"v1.2 {sum('error' in r for r in o['1.2'].values())}; floor active "
              f"{sum(bool((r.get('floor') or {}).get('active')) for r in o['1.2'].values())}/{len(o['1.2'])}")
        for d in sorted({r['dim'] for r in o['1.1'].values()}):
            s = [i for i in inl if o['1.1'][i]['dim'] == d]
            if s:
                print(f"- D = {d}: in-library exact v1.1 {np.mean([o['1.1'][i]['exact'] for i in s]):.3f}, "
                      f"v1.2 {np.mean([o['1.2'][i]['exact'] for i in s]):.3f} (n = {len(s)})")
    pl = O / "pytest.log"
    if pl.exists():
        print("\n## pytest\n\n```\n" + "\n".join(pl.read_text().splitlines()[-4:]) + "\n```")


if __name__ == "__main__":
    main()
