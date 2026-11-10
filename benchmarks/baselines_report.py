"""Tables for docs/reports/baselines.md from results_baselines/*.json, paired with eqdisc.discover's frozen runs.

    python benchmarks/baselines_report.py [--dir results_baselines] [--markdown]

Ours: dysts -> results_dysts/test_final.json (by archive index); hard40 -> results_final/test_confirm.json arm "final".
Baseline rules:
  CV      5-fold regression-row CV pick (truth-free)            AICc   AICc pick (truth-free)
  tuned   one threshold for the whole evaluation set, chosen ON THAT SET with the truth (generous; not truth-free)
  oracle  best threshold per run (upper bound; per-equation k for MIOSR, Pareto front for PySR, beam for ODEFormer)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

NOISES = (0.01, 0.05, 0.1, 0.2)


def ours(setname):
    if setname == "hard40":
        R = json.loads(Path("results_final/test_confirm.json").read_text())["records"]
        return {r["index"]: (r["arms"]["final"]["exact"], r["arms"]["final"]["seconds"]) for r in R}
    R = json.loads(Path("results_dysts/test_final.json").read_text())["records"]
    return {r["index"]: (r["final"]["exact"] if r["status"] == "ok" else 0.0, r["seconds"]) for r in R}


def rules(path):
    """-> {rule: {index: exact}}, meta {index: rec}"""
    d = json.loads(Path(path).read_text())
    recs = d["records"]
    method = d["args"]["method"]
    out = {}
    meta = {r["index"]: r for r in recs}
    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    zero = {r["index"]: 0.0 for r in bad}  # errors count as failures
    if method in ("pysr", "odeformer"):
        out["default"] = {r["index"]: r["best"]["exact"] for r in ok} | zero
        out["oracle"] = {r["index"]: r["oracle"] for r in ok} | zero
    else:
        out["CV"] = {r["index"]: r["cv"]["exact"] for r in ok} | zero
        out["AICc"] = {r["index"]: r["aic"]["exact"] for r in ok} | zero
        if "exact_path" in ok[0]:
            ep = np.array([r["exact_path"] for r in ok])
            g = int(ep.mean(0).argmax())
            out["tuned"] = {r["index"]: float(r["exact_path"][g]) for r in ok} | zero
        out["oracle"] = {r["index"]: r["oracle"] for r in ok} | zero
    return method, out, meta, d


def sign_p(w, l):
    return binomtest(w, w + l, 0.5).pvalue if w + l else 1.0


def row(label, ex, meta, our, side):
    idx = sorted(ex)
    e = np.array([ex[i] for i in idx])
    o = np.array([our[i][0] for i in idx])
    nz = np.array([meta[i]["noise"] for i in idx])
    D = np.array([meta[i]["D"] for i in idx])
    cells = [f"{e[nz == z].mean():.3f}" if (nz == z).any() else "-" for z in NOISES]
    d3 = f"{e[D == 3].mean():.3f}" if (D == 3).any() else "-"
    d4 = f"{e[D >= 4].mean():.3f}" if (D >= 4).any() else "-"
    sec = np.median([meta[i]["seconds"] for i in idx])
    w, l = int(((o == 1) & (e == 0)).sum()), int(((o == 0) & (e == 1)).sum())
    return [label, str(len(idx)), f"{e.mean():.3f}", *cells, d3, d4, f"{sec:.1f}", f"{o.mean():.3f}",
            f"{w}/{l}", f"{sign_p(w, l):.0e}"]


HDR = ["method / rule", "runs", "exact", "1%", "5%", "10%", "20%", "D=3", "D>=4", "median s", "ours same runs",
       "ours W/L", "sign p"]

LABEL = {"fd_stlsq": "SINDy STLSQ (SFD)", "fd_sr3": "SINDy SR3-L0 (SFD)", "fd_esindy": "E-SINDy (SFD)",
         "weak_stlsq": "Weak SINDy STLSQ", "weak_sr3": "Weak SINDy SR3-L0", "weak_esindy": "Weak E-SINDy",
         "weak_miosr": "Weak SINDy MIOSR", "pysr": "PySR", "odeformer": "ODEFormer"}


def table(rows, md):
    if md:
        out = ["| " + " | ".join(HDR) + " |", "|" + "---|" * len(HDR)]
        out += ["| " + " | ".join(r) + " |" for r in rows]
    else:
        w = [max(len(r[k]) for r in rows + [HDR]) for k in range(len(HDR))]
        out = ["  ".join(h.ljust(w[k]) for k, h in enumerate(HDR))]
        out += ["  ".join(c.ljust(w[k]) for k, c in enumerate(r)) for r in rows]
    return "\n".join(out)


# Config used for each rule, fixed in advance from the dev sweep (results_baselines/dev; data/instances.npz every
# 5th run): truth-free rules take the config whose same rule scored best on dev; "tuned"/"oracle" take the
# config with the best dev oracle.  E-SINDy follows its STLSQ counterpart (too costly to sweep).
SELECT = {
    "fd_stlsq": {"CV": "raw_w11", "AICc": "raw_w5", "tuned": "raw_w11", "oracle": "raw_w11"},
    "fd_sr3": {"CV": "w11", "AICc": "w5", "tuned": "w21", "oracle": "w21"},
    "fd_esindy": {"CV": "raw_w11", "AICc": "raw_w11", "tuned": "raw_w11", "oracle": "raw_w11"},
    "weak_stlsq": {"CV": "h10", "AICc": "h10", "tuned": "raw_h20", "oracle": "raw_h20"},
    "weak_sr3": {"CV": "h10", "AICc": "h10", "tuned": "h20", "oracle": "h20"},
    "weak_esindy": {"CV": "h10", "AICc": "h10", "tuned": "raw_h20", "oracle": "raw_h20"},
}
ORDER = ["fd_stlsq", "fd_sr3", "fd_esindy", "weak_stlsq", "weak_sr3", "weak_esindy", "weak_miosr", "pysr", "odeformer"]
SYSTEMS4 = ("lorenz", "chen", "rossler", "rabinovich")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results_baselines")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--all", action="store_true", help="also list every config x rule")
    args = ap.parse_args()
    side = json.loads(Path("data/dysts_test.json").read_text())["systems"]
    files = {}
    for f in sorted(Path(args.dir).glob("*.json")):
        if f.name.endswith(".partial.json"):
            continue
        setname, rest = f.stem.split("_", 1)
        method = next(m for m in sorted(ORDER, key=len, reverse=True) if rest.startswith(m))
        files[(setname, method, rest[len(method):].strip("_") or "default")] = f
    loaded = {k: rules(f) for k, f in files.items()}
    for setname in sorted({k[0] for k in files}):
        our = ours("hard40" if setname == "hard40" else "dysts")
        rows, hard = [], []
        meta0 = None
        for method in ORDER:
            cfgs = [k[2] for k in files if k[0] == setname and k[1] == method]
            if not cfgs:
                continue
            ruleset = list(loaded[(setname, method, cfgs[0])][1])
            for rule in ruleset:
                cfg = SELECT.get(method, {}).get(rule, cfgs[0])
                if cfg not in cfgs:
                    continue
                _, rs, meta, _ = loaded[(setname, method, cfg)]
                meta0 = meta0 or meta
                lab = f"{LABEL[method]} [{cfg}] {rule}"
                rows.append(row(lab, rs[rule], meta, our, side))
                hard.append((lab, {s: int(sum(rs[rule][i] for i in rs[rule] if meta[i]["system"] == s)) for s in SYSTEMS4}))
        ex0 = {i: our[i][0] for i in meta0}
        m2 = {i: dict(meta0[i], seconds=our[i][1]) for i in meta0}
        rows.append(row("eqdisc.discover (final, frozen)", ex0, m2, our, side))
        print(f"\n## {setname}: {len(meta0)} runs. W/L = runs only ours / only the baseline recovers exactly; "
              f"sign test on discordant pairs")
        print(table(rows, args.markdown))
        if setname == "hard40":
            print("\nhard 40 per system (exact / 10):")
            for lab, c in hard:
                print(f"  {lab:48s} " + "  ".join(f"{s} {c[s]:2d}" for s in SYSTEMS4))
            print(f"  {'eqdisc.discover (final)':48s} " + "  ".join(
                f"{s} {int(sum(our[i][0] for i in meta0 if meta0[i]['system'] == s)):2d}" for s in SYSTEMS4))
        if args.all:
            allrows = []
            for (st, method, cfg), (_, rs, meta, _) in loaded.items():
                if st == setname:
                    allrows += [row(f"{LABEL[method]} [{cfg}] {rule}", ex, meta, our, side) for rule, ex in rs.items()]
            print(f"\n### {setname}: every config x rule")
            print(table(allrows, args.markdown))


if __name__ == "__main__":
    main()
