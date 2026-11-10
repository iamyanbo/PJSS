"""Compare joint_colored ('ar') against config C (results_joint/C_<set>.json) on the same instances."""
import argparse
import json

import numpy as np
from scipy.stats import binomtest

from joint_report import pick


def score_list(cands, n, phi):
    return sorted(c["data"] + c["model"] + c["val"] + phi * c["k"] * np.log(3 * n) for c in cands)


def cscore(c, n, phi, rule):
    if rule == "conc":
        d = sum(c["ncomp"] * np.log(max(x, 1e-300) / c["ncomp"]) for x in c["dcomp"])
    else:
        d = c["data"] + c["val"]
    return d + c["model"] + phi * c["k"] * np.log(3 * n)


def sel(v, n, rule="chi2"):
    if not v.get("candidates"):
        return {"exact": 0.0}, np.inf
    phi = v.get("phi", 1.0)
    sc = sorted(v["candidates"], key=lambda c: cscore(c, n, phi, rule))
    g = cscore(sc[1], n, phi, rule) - cscore(sc[0], n, phi, rule) if len(sc) > 1 else np.inf
    return sc[0], g / v.get("lam", 1.0)  # flag when gap < 2 lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ar_file")
    ap.add_argument("c_file")
    ap.add_argument("--variant", default="ar")
    ap.add_argument("--rule", default="chi2")
    args = ap.parse_args()
    R = json.load(open(args.ar_file))["records"]
    C = {r["index"]: r for r in json.load(open(args.c_file))["records"]}
    rows = []
    for r in R:
        c = C[r["index"]]
        a, ga = sel(r["variants"][args.variant], r["n"], args.rule)
        b, gb = sel(c["joint"]["0.03"], r["n"])
        rows.append(dict(noise=r["noise"], n=r["n"], system=r["system"], weak=r["weak"]["exact"],
                         hyb=r["hyb_bic10"]["exact"], C=b["exact"], AR=a["exact"], fa=ga < 2, fc=gb < 2,
                         ar=np.mean(r["variants"][args.variant].get("ar", [np.nan])),
                         secs=r["variants"][args.variant].get("seconds", np.nan), coefA=a.get("coef_err", np.nan),
                         coefC=b.get("coef_err", np.nan)))

    def line(label, rs):
        g = lambda k: np.mean([x[k] for x in rs])
        return f"{label:14} {len(rs):4d} {g('weak'):6.3f} {g('hyb'):6.3f} {g('C'):6.3f} {g('AR'):6.3f}"

    print(f"{'cell':14} {'runs':>4} {'weak':>6} {'hybrid':>6} {'C':>6} {'AR':>6}")
    for cell in sorted({(x["noise"], x["n"]) for x in rows}):
        print(line(f"{cell[0]:<5} {cell[1]}", [x for x in rows if (x["noise"], x["n"]) == cell]))
    print(line("all", rows))
    w = sum(x["AR"] > x["C"] for x in rows)
    l = sum(x["AR"] < x["C"] for x in rows)
    print(f"AR vs C: {w} wins / {l} losses, p = {binomtest(w, w + l).pvalue if w + l else 1:.1e}")
    for k, f in (("C", "fc"), ("AR", "fa")):
        fl = np.array([x[f] for x in rows])
        ex = np.array([x[k] for x in rows])
        wrong = ex == 0
        print(f"{k:2} flagged {fl.mean():.2f}  unflagged acc {ex[~fl].mean():.3f}  "
              f"wrong caught {fl[wrong].mean() if wrong.any() else np.nan:.2f}")
    print("by system (weak / hybrid / C / AR):")
    for s in sorted({x["system"] for x in rows}):
        rs = [x for x in rows if x["system"] == s]
        print(f"   {s:11} " + " ".join(f"{sum(x[k] for x in rs):4.0f}" for k in ("weak", "hyb", "C", "AR")) + f"  /{len(rs)}")
    both = [x for x in rows if x["AR"] and x["C"]]
    print(f"median coef err (both exact, {len(both)}): C {100 * np.median([x['coefC'] for x in both]):.2f}%  "
          f"AR {100 * np.median([x['coefA'] for x in both]):.2f}%")
    print(f"mean estimated a {np.nanmean([x['ar'] for x in rows]):.3f}; median AR seconds "
          f"{np.nanmedian([x['secs'] for x in rows]):.1f}")


if __name__ == "__main__":
    main()
