"""Accuracy-vs-runtime tables for the fast baselines on the 80 head-to-head runs.

    python benchmarks/fast_baselines_report.py            # prints markdown, also writes results_fast/report.md

Inputs: results_fast/<method>.json (benchmarks/fast_baselines.py collect) and results_h2h/rows.json (ours v1.2
and DEDi, from benchmarks/h2h_report.py). Numbers only: exact x/n, mean F1, median relative coefficient error
over the runs that method got exact, median seconds per run. No significance tests.
"""
import json
import math
from pathlib import Path

import numpy as np

R = Path("results_fast")
CONDS = ["hard", "white", "coloured", "t3", "short"]


def load(name, field, seconds="seconds"):
    f = R / f"{name}.json"
    if not f.exists():
        return None
    out = {}
    for r in json.loads(f.read_text())["records"]:
        m = r.get(field) or dict(exact=0.0, f1=0.0, coef_err=math.nan)
        out[r["key"]] = dict(cond=r["cond"], exact=int(m["exact"]), f1=float(m["f1"]),
                             coef=float(m.get("coef_err", math.nan)), s=float(r.get(seconds, math.nan)))
    return out


def methods():
    rows = json.loads(Path("results_h2h/rows.json").read_text())
    M = []
    M.append(("**ours v1.2** (h2h timing: core 13, loaded)", {r["key"]: dict(cond=r["cond"], exact=r["ours"], f1=r["ours_f1"], coef=r["ours_coef"],
                                                s=r["ours_s"]) for r in rows}))
    rt = {}
    for c in CONDS:
        f = R / f"ours_retime_{c}.json"
        if f.exists():
            for r in json.loads(f.read_text())["records"]:
                a = r["arms"]["final_v12"]
                rt[f"{c}_{r['index']}"] = dict(cond=c, exact=int(a["exact"]), f1=float(a["f1"]), coef=float(a["coef_err"]),
                                               s=a["seconds"] + r.get("starts_seconds", 0) + r.get("a_seconds", 0))
    if rt:
        M.append(("**ours v1.2, re-timed on cores 0-4**", rt))
    spec = [("WSINDy + MSTLS (authors' MATLAB, ported)", "wsindy", "mstls", "seconds"),
            ("  WSINDy variant: K = all centres", "wsindy_K1", "mstls", "seconds"),
            ("  WSINDy variant: no pre-smoothing", "wsindy_nosmooth", "mstls", "seconds"),
            ("weak STLSQ, CV (pysindy)", "weak_stlsq", "cv", "seconds"),
            ("weak STLSQ, AICc (pysindy)", "weak_stlsq", "aic", "seconds_aic"),
            ("weak E-SINDy, CV (pysindy)", "weak_esindy", "cv", "seconds"),
            ("weak E-SINDy, AICc (pysindy)", "weak_esindy", "aic", "seconds_aic"),
            ("SFD STLSQ, CV (pysindy)", "fd_stlsq", "cv", "seconds"),
            ("SFD STLSQ, AICc (pysindy)", "fd_stlsq", "aic", "seconds_aic"),
            ("SFD E-SINDy, CV (pysindy)", "fd_esindy", "cv", "seconds"),
            ("SFD E-SINDy, AICc (pysindy)", "fd_esindy", "aic", "seconds_aic"),
            ("in-repo weak chi2s (select_k)", "chi2s", "chi2s", "seconds"),
            ("ARGOS lasso, B = 200 bootstraps (R pkg)", "argos_b200", "argos", "seconds"),
            ("ARGOS lasso, B = 2000 = pkg default (R pkg)", "argos_b2000", "argos", "seconds")]
    for label, name, field, sec in spec:
        d = load(name, field, sec)
        if d:
            M.append((label, d))
    for sub, lab in (("single (1e-3, 1)", "DEDi single pair (1e-3, 1)"), ("row lambda=1e-3", "DEDi row lambda=1e-3"),
                     ("full grid", "DEDi full grid")):
        for crit, cl in (("val_error", "val"), ("bic", "BIC")):
            d = {}
            for r in rows:
                full = sub == "full grid"
                d[r["key"]] = dict(cond=r["cond"], exact=r[f"{sub}|{crit}"],
                                   f1=r[f"f1|{crit}"] if full else math.nan,
                                   coef=r[f"coef|{crit}"] if full else math.nan, s=r[f"{sub}|seconds"])
            M.append((f"{lab}, {cl}", d))
    return M


def summ(d, keys):
    v = [d[k] for k in keys if k in d]
    if not v:
        return None
    ex = sum(x["exact"] for x in v)
    f1 = [x["f1"] for x in v]
    co = [x["coef"] for x in v if x["exact"] and np.isfinite(x["coef"])]
    return dict(n=len(v), exact=ex, f1=np.nan if any(np.isnan(f1)) else float(np.mean(f1)),
                coef=float(np.median(co)) if co else np.nan, s=float(np.median([x["s"] for x in v])))


def fmt(x, spec):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def table(M, keys, title):
    lines = [f"\n### {title}\n", "| method | exact | mean F1 | median coef err (exact runs) | median s/run |",
             "| --- | --- | --- | --- | --- |"]
    for label, d in M:
        s = summ(d, keys)
        if s is None:
            continue
        lines.append(f"| {label} | {s['exact']}/{s['n']} | {fmt(s['f1'], '.3f')} | {fmt(s['coef'], '.4f')} | "
                     f"{fmt(s['s'], '.3g')} |")
    return lines


def frontier(M, keys):
    pts = []
    for label, d in M:
        if label.startswith("  ") or "h2h timing" in label:  # sensitivity variants / duplicate ours timing
            continue
        s = summ(d, keys)
        if s and s["n"] == len(keys):
            pts.append((s["s"], s["exact"], label))
    pts.sort(key=lambda p: (p[0], -p[1]))
    best, front = -1, []
    for sec, ex, label in pts:
        if ex > best:
            front.append((sec, ex, label))
            best = ex
    return front


def main():
    rows = json.loads(Path("results_h2h/rows.json").read_text())
    allk = [r["key"] for r in rows]
    M = methods()
    out = table(M, allk, f"Pooled ({len(allk)} runs)")
    for c in CONDS:
        out += table(M, [r["key"] for r in rows if r["cond"] == c], f"{c} (16 runs)")
    sub = [k for k in allk if k in (dict(M).get("ARGOS lasso, B = 2000 = pkg default (R pkg)") or {})]
    if sub:
        out += table(M, sub, f"Subset with ARGOS at its default B = 2000 ({len(sub)} runs, first rounds of order.json)")
    out.append(f"\n### Pooled accuracy-runtime frontier (methods run on all {len(allk)} runs)\n")
    out.append("Methods not dominated by a faster method with at least as many exact runs:\n")
    for sec, ex, label in frontier(M, allk):
        out.append(f"- {label}: {ex}/{len(allk)} exact at median {sec:.3g} s/run")
    chk = R / "wsindy_octave_check.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        out.append(f"\nWSINDy port vs authors' MATLAB code in Octave: identical support on "
                   f"{sum(x['same_support'] for x in c)}/{len(c)} runs, max relative coefficient difference "
                   f"{max(x['max_rel_diff'] for x in c):.1e}.")
    text = "\n".join(out)
    print(text)
    (R / "report.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
