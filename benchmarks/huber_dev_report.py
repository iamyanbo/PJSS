"""Tables for the Huber-vs-Gaussian dev diagnostic (results_huber_dev/*.json -> stdout, markdown).

    python benchmarks/huber_dev_report.py
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
R = Path(os.environ.get("HUBER_DIR", ROOT / "results_huber_dev"))
STORED = {"dev_colored": ("dev_colored", "ARp"), "dev_student": ("dev_student", "Cp"),
          "dev_white": ("dev_white", "Cp"), "dev_cmid": ("dev_cmid", "Cp")}


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(prefix):
    recs = {}
    for f in sorted(R.glob(f"{prefix}*.json")):
        if f.name.endswith(".partial.json"):
            continue
        for r in json.loads(f.read_text())["records"]:
            recs[r["index"]] = r
    return recs


def ex(r, arm):
    if arm == "path_gauss":  # Gaussian only where the frozen router chose AR(1)
        v = r["variants"]
        if "ar1_gauss" not in v:
            return None
        return v["ar1_gauss"]["exact"] if v["full"].get("noise_model") == "ar1" else v["full"]["exact"]
    v = r["variants"].get(arm)
    return None if v is None else v["exact"]


def pair(recs, a, b):
    w = l = 0
    for r in recs:
        x, y = ex(r, a), ex(r, b)
        if x is None or y is None:
            continue
        w += x > y
        l += y > x
    return w, l


def fmt_pair(recs, a, b):
    w, l = pair(recs, a, b)
    return f"{w}/{l} (p = {sign_p(w, l):.2g})"


def table(name, recs, arms, comps):
    rs = sorted(recs.values(), key=lambda r: r["index"])
    print(f"\n### {name}: {len(rs)} runs\n")
    print("| Subset | n | " + " | ".join(arms) + " | " + " | ".join(f"{a} vs {b} (W/L)" for a, b in comps) + " |")
    print("| --- " * (2 + len(arms) + len(comps)) + "|")
    groups = [("all", rs)]
    for key, lab in (("noise", "noise"), ("n", "N")):
        vals = sorted({r[key] for r in rs})
        groups += [(f"{lab} = {v}", [r for r in rs if r[key] == v]) for v in vals]
    for g, sub in groups:
        cells = []
        for a in arms:
            xs = [ex(r, a) for r in sub if ex(r, a) is not None]
            cells.append(f"{sum(xs) / len(xs):.3f}" if xs else "-")
        cps = [fmt_pair(sub, a, b) if g == "all" else "{}/{}".format(*pair(sub, a, b)) for a, b in comps]
        print(f"| {g} | {len(sub)} | " + " | ".join(cells) + " | " + " | ".join(cps) + " |")
    nm = defaultdict(int)
    for r in rs:
        nm[r["variants"]["full"].get("noise_model")] += 1
    print(f"\nfrozen router (full): {dict(nm)}")
    errs = sum("error" in v for r in rs for v in r["variants"].values())
    print(f"errors: {errs}")


def repro(setname, recs):
    f, arm = STORED[setname]
    p = ROOT / "results_final" / f"{f}.json"
    if not p.exists():
        return
    stored = {}
    for q in [p] + ([ROOT / "results_final" / "dev_white_odd.json"] if setname == "dev_white" else []):
        for r in json.loads(q.read_text())["records"]:
            if arm in r["arms"]:
                stored[r["index"]] = r["arms"][arm]
    both = [i for i in recs if i in stored]
    same = sum(recs[i]["variants"]["full"]["exact"] == stored[i]["exact"] for i in both)
    ce = sum(abs(recs[i]["variants"]["full"]["coef_err"] - stored[i].get("coef_err", float("nan"))) < 1e-9
             for i in both)
    print(f"reproduction vs results_final/{f}.json arm {arm}: {same}/{len(both)} exact flags equal, "
          f"{ce}/{len(both)} coef_err equal to 1e-9")


def main():
    sets = [("dev_colored", ["full", "gaussian", "ar1", "ar1_gauss", "path_gauss"],
             [("gaussian", "full"), ("ar1_gauss", "ar1"), ("path_gauss", "full")]),
            ("dev_student", ["full", "gaussian"], [("gaussian", "full")]),
            ("dev_white", ["full", "gaussian"], [("gaussian", "full")]),
            ("dev_cmid", ["full", "gaussian"], [("gaussian", "full")])]
    for s, arms, comps in sets:
        prim = {i: r for i, r in load(s + "_o0").items()} if s != "dev_cmid" else {}
        allr = load(s)
        if not allr:
            continue
        if prim:
            table(f"{s} PRIMARY (index = 0 mod 3)", prim, arms, comps)
        if len(allr) > len(prim):
            table(f"{s} all completed runs", allr, arms, comps)
        repro(s, allr)


if __name__ == "__main__":
    main()
