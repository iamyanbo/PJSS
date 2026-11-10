"""Tables for ODEBench with the frozen model (results_odebench/fit_ic*.json -> stdout, markdown).

    python benchmarks/odebench_report.py [--dir results_odebench]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SIG = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


def acc(rs, key, thr=0.9):
    return 100 * np.mean([r.get(key, 0.0) > thr for r in rs]) if rs else float("nan")


def subsets(rs):
    out = [("all", rs), ("D >= 2", [r for r in rs if r["dim"] >= 2]),
           ("in_library", [r for r in rs if r["class"] == "in_library"]),
           ("in_library, D >= 2", [r for r in rs if r["class"] == "in_library" and r["dim"] >= 2]),
           ("poly_high_degree", [r for r in rs if r["class"] == "poly_high_degree"]),
           ("non_polynomial", [r for r in rs if r["class"] == "non_polynomial"])]
    return [(f"{n} ({len({r['id'] for r in sub})} systems)", sub) for n, sub in out if sub]


def r2table(rs, key, title):
    print(f"\n**{title}** (% of runs with R² > 0.9)\n")
    print("| Subset | " + " | ".join(f"σ={s}" for s in SIG) + " | mean over σ |")
    print("| --- " * (len(SIG) + 2) + "|")
    for name, sub in subsets(rs):
        v = [acc([r for r in sub if r["sigma"] == s], key) for s in SIG]
        print(f"| {name} | " + " | ".join(f"{x:.0f}" for x in v) + f" | {np.nanmean(v):.1f} |")


def exact_table(rs, title):
    print(f"\n**{title}** (exact support recovery, in-library systems)\n")
    print("| Subset | " + " | ".join(f"σ={s}" for s in SIG) + " | mean |")
    print("| --- " * (len(SIG) + 2) + "|")
    for name, f in (("in_library (23)", lambda r: True), ("D = 1 (8)", lambda r: r["dim"] == 1),
                    ("D = 2 (6)", lambda r: r["dim"] == 2), ("D = 3 (8)", lambda r: r["dim"] == 3),
                    ("D = 4 (1)", lambda r: r["dim"] == 4)):
        sub = [r for r in rs if r["class"] == "in_library" and f(r)]
        v = [100 * np.mean([r["exact"] for r in sub if r["sigma"] == s]) for s in SIG]
        print(f"| {name} | " + " | ".join(f"{x:.0f}" for x in v) + f" | {np.mean(v):.1f} |")
    ok = [r for r in rs if r["class"] == "in_library" and r.get("exact") == 1.0]
    if ok:
        print(f"\nmedian coefficient error on exact runs: {100 * np.median([r['coef_err'] for r in ok]):.2f} %")


_LIB, _TRAJ = {}, {}


def _setup(root, data_path):
    _LIB.update({int(k): np.asarray(v["E"]) for k, v in json.loads((root / "library_exponents.json").read_text()).items()})
    if Path(data_path).exists():
        for e in json.loads(Path(data_path).read_text()):
            _TRAJ[e["id"]] = [np.asarray(s["y"], float).T for s in e["solutions"][0]]


def exact_posthoc(r, fit_ic, thr=0.01):
    """POST-HOC (not pre-registered): exact support after dropping terms whose RMS contribution on the clean
    fitted trajectory is < thr of the RMS of that equation's discovered right-hand side."""
    if "xi" not in r or r["class"] != "in_library" or r["id"] not in _TRAJ:
        return r.get("exact")
    X = _TRAJ[r["id"]][fit_ic]
    xi = np.asarray(r["xi"])
    phi = np.prod(X[:, None, :] ** _LIB[r["dim"]][None], axis=2)  # (n, M)
    contrib = np.sqrt(np.mean((phi[:, :, None] * xi[None]) ** 2, axis=0))  # (M, D)
    tot = np.sqrt(np.mean((phi @ xi) ** 2, axis=0))
    keep = (xi != 0) & (contrib >= thr * np.maximum(tot, 1e-300)[None, :])
    return float((keep == (np.asarray(r["xi_true"]) != 0)).all())


def posthoc_table(rs, cls_by_id, fit_ic, title):
    print(f"\n**{title}** (POST-HOC, not pre-registered: exact support ignoring terms contributing < 1% of the "
          "equation's RMS)\n")
    print("| Subset | " + " | ".join(f"σ={s}" for s in SIG) + " | mean |")
    print("| --- " * (len(SIG) + 2) + "|")
    for name, f in (("in_library", lambda r: True), ("D = 2", lambda r: r["dim"] == 2),
                    ("D = 3", lambda r: r["dim"] == 3), ("D = 4", lambda r: r["dim"] == 4)):
        sub = [dict(r, xi_true=cls_by_id[r["id"]]["xi_true"]) for r in rs if r["class"] == "in_library" and f(r)]
        if not sub:
            continue
        v = [100 * np.mean([exact_posthoc(r, fit_ic) or 0.0 for r in sub if r["sigma"] == s]) for s in SIG]
        print(f"| {name} ({len({r['id'] for r in sub})}) | " + " | ".join(f"{x:.0f}" for x in v) + f" | {np.mean(v):.1f} |")


def failures(rs):
    err = [r for r in rs if "error" in r]
    by = defaultdict(int)
    for r in err:
        by[(r["dim"], r["error"].split(":")[0])] += 1
    print(f"\nmodel exceptions: {len(err)} of {len(rs)} runs  {dict(by)}")
    nerr = [r for r in rs if "error" not in r]
    print(f"integration failures among fitted runs: reconstruction {sum(r['rec_fail'] for r in nerr)}, "
          f"generalization {sum(r['gen_fail'] for r in nerr)} of {len(nerr)}")
    print(f"median fit time: {np.median([r['seconds'] for r in rs]):.1f} s; by D: " + ", ".join(
        f"D={d}: {np.median([r['seconds'] for r in rs if r['dim'] == d]):.1f} s" for d in (1, 2, 3, 4)))


def per_system(rs):
    print("\n| id | D | class | exact σ=0/.01/.03/.05 | R²rec σ=0 | R²rec σ=.05 | R²gen σ=0 | R²gen σ=.05 | found (σ=0) |")
    print("| --- " * 9 + "|")
    by = defaultdict(dict)
    for r in rs:
        by[r["id"]][r["sigma"]] = r
    for i in sorted(by):
        d = by[i]
        r0 = d[0.0]
        ex = "/".join("-" if "exact" not in d[s] else f"{d[s]['exact']:.0f}" for s in (0.0, 0.01, 0.03, 0.05))
        eq = r0.get("equations", r0.get("error", ""))[:70].replace("\n", "; ").replace("|", "/")
        print(f"| {i} | {r0['dim']} | {r0['class']} | {ex} | {r0['r2_rec']:.2f} | {d[0.05]['r2_rec']:.2f} | "
              f"{r0['r2_gen']:.2f} | {d[0.05]['r2_gen']:.2f} | {eq} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "results_odebench"))
    a = ap.parse_args()
    D = Path(a.dir)
    _setup(D, D.parent / "data" / "odebench" / "strogatz_extended.json")
    cls_by_id = {c["id"]: c for c in json.loads((D / "classes.json").read_text())}
    files = {k: D / f"{k}.json" for k in ("fit_ic0", "fit_ic1", "fit_ic0_emptyfix")}
    data = {k: json.loads(f.read_text())["records"] for k, f in files.items() if f.exists()}
    if "fit_ic0" in data:
        rs = data["fit_ic0"]
        prim = [r for r in rs if r["rep"] == 0]
        print(f"## PRIMARY: fit on IC 0, rep 0 ({len(prim)} runs)")
        r2table(prim, "r2_rec", "Reconstruction")
        r2table(prim, "r2_gen", "Generalization (other IC)")
        exact_table(prim, "Structure")
        posthoc_table(prim, cls_by_id, 0, "Structure, negligible terms ignored")
        failures(prim)
        print("\n## Sensitivity: reconstruction from the clean IC (rep 0)")
        r2table(prim, "r2_rec_clean_y0", "Reconstruction, clean y0")
        print(f"\n## All reps, fit on IC 0 ({len(rs)} runs)")
        r2table(rs, "r2_rec", "Reconstruction")
        r2table(rs, "r2_gen", "Generalization")
        exact_table(rs, "Structure")
        print("\n## Per system (fit IC 0, rep 0)")
        per_system(prim)
    if "fit_ic1" in data:
        rs = data["fit_ic1"]
        print(f"\n## Secondary: fit on IC 1, all reps ({len(rs)} runs)")
        r2table(rs, "r2_rec", "Reconstruction")
        r2table(rs, "r2_gen", "Generalization (IC 0)")
        exact_table(rs, "Structure")
        posthoc_table(rs, cls_by_id, 1, "Structure, negligible terms ignored")
        failures(rs)
    if "fit_ic0_emptyfix" in data:
        rs = data["fit_ic0_emptyfix"]
        print(f"\n## DIAGNOSTIC (not frozen): empty-support crash patched, D = 1, 2, fit IC 0, all reps ({len(rs)} runs)")
        for d in (1, 2):
            sub = [r for r in rs if r["dim"] == d]
            print(f"\nD = {d}: {len(sub)} runs")
            r2table(sub, "r2_rec", f"Reconstruction D = {d}")
            r2table(sub, "r2_gen", f"Generalization D = {d}")
            ins = [r for r in sub if r["class"] == "in_library"]
            for s in SIG:
                x = [r["exact"] for r in ins if r["sigma"] == s]
                print(f"  exact σ={s}: {100 * np.mean(x):.0f}% of {len(x)}")
        failures(rs)


if __name__ == "__main__":
    main()
