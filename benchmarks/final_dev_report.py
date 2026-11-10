"""Tables for the final-model dev crossed experiments (results_final/dev_*.json).

  python final_dev_report.py [results_final]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_final")


def load(name):
    f = D / f"{name}.json"
    return json.loads(f.read_text())["records"] if f.exists() else None


def ex(r, arm):
    return r["arms"].get(arm, {}).get("exact", 0.0)


def wl(R, a, b):
    """W/L of arm a against arm b, two-sided sign-test p."""
    w = sum(ex(r, a) > ex(r, b) for r in R)
    l = sum(ex(r, a) < ex(r, b) for r in R)
    return w, l, (binomtest(w, w + l).pvalue if w + l else 1.0)


def table(name, R, arms, pairs):
    print(f"\n### {name} ({len(R)} runs)")
    print("| arm | exact | flagged | unflagged acc | median s | errors |\n|---|---|---|---|---|---|")
    for a in arms:
        v = [r["arms"][a] for r in R if a in r["arms"]]
        fl = np.array([x.get("ambiguous", False) for x in v])
        e = np.array([x.get("exact", 0) for x in v])
        print(f"| {a} | {e.mean():.3f} | {fl.mean():.0%} | {e[~fl].mean() if (~fl).any() else float('nan'):.3f} | "
              f"{np.median([x.get('seconds', 0) for x in v]):.1f} | {sum('error' in x for x in v)} |")
    for a, b in pairs:
        w, l, p = wl(R, a, b)
        sys_w = sorted({r["system"] for r in R if ex(r, a) > ex(r, b)})
        sys_l = sorted({r["system"] for r in R if ex(r, a) < ex(r, b)})
        print(f"- {a} vs {b}: {w}/{l} (p = {p:.2g}); wins {sys_w}; losses {sys_l}")


def ahat(R):
    return np.array([np.mean(r["a_hat"]) for r in R if "a_hat" in r])


def main():
    sets = {}
    for name, arms, pairs in [
        ("dev_white", ["C", "Cp", "AR", "ARp"], [("Cp", "C"), ("AR", "C"), ("ARp", "Cp"), ("ARp", "AR")]),
        ("dev_white_odd", ["C", "Cp", "AR", "ARp"], [("Cp", "C"), ("AR", "C"), ("ARp", "Cp"), ("ARp", "AR")]),
        ("dev_colored", ["C", "Cp", "AR", "ARp", "ARp_e14", "ARp_e12"],
         [("Cp", "C"), ("AR", "C"), ("ARp", "AR"), ("ARp", "Cp"), ("ARp_e14", "AR"), ("ARp_e12", "AR"),
          ("ARp_e14", "ARp"), ("ARp_e12", "ARp")]),
        ("dev_cmid", ["C", "Cp", "AR", "ARp"], [("Cp", "C"), ("AR", "C"), ("ARp", "AR"), ("ARp", "Cp")]),
        ("dev_student", ["Cp", "ARp"], [("ARp", "Cp")]),
        ("dev_coarse", ["Cp", "ARp"], [("ARp", "Cp")]),
        ("dev_heldout3", ["Cp", "ARp"], [("ARp", "Cp")]),
        ("dev_deg4", ["off", "p15", "pmid", "pmean"], [("p15", "off"), ("pmid", "off"), ("pmean", "off"),
                                                       ("p15", "pmid"), ("p15", "pmean")]),
        ("dev_deg5", ["off", "p15", "pmid", "pmean"], [("p15", "off"), ("pmid", "off"), ("pmean", "off"),
                                                       ("p15", "pmid"), ("p15", "pmean")]),
        ("dev_trig", ["off", "p15"], [("p15", "off")]),
        ("dev_mixedpoly", ["off", "p15"], [("p15", "off")]),
        ("dev_nd", ["off", "p15", "p10"], [("p15", "off"), ("p10", "off"), ("p15", "p10")]),
    ]:
        R = load(name)
        if R is None:
            print(f"\n### {name}: missing")
            continue
        sets[name] = R
        if name == "dev_cmid":  # true AR coefficient of each run, from the archive
            rho = np.load("data/dev_colored_mid.npz", allow_pickle=True)["rho"]
            for r in R:
                r["rho"] = float(rho[r["index"]])
        table(name, R, arms, pairs)
        if name == "dev_cmid":
            for rho in (0.3, 0.55):
                Rs = [r for r in R if abs(_rho(r) - rho) < 1e-9]
                if Rs:
                    print(f"  rho {rho}: " + ", ".join(f"{a} {np.mean([ex(r, a) for r in Rs]):.3f}" for a in arms) +
                          "; ARp vs Cp %d/%d (p = %.2g)" % wl(Rs, "ARp", "Cp") +
                          "; AR vs C %d/%d (p = %.2g)" % wl(Rs, "AR", "C"))
        if name == "dev_nd":
            for s in sorted({r["system"] for r in R}):
                Rs = [r for r in R if r["system"] == s]
                print(f"  {s}: " + ", ".join(f"{a} {np.mean([ex(r, a) for r in Rs]):.3f}" for a in arms))
        if name in ("dev_deg5", "dev_deg4"):
            for s in ("rabinovich",):
                Rs = [r for r in R if r["system"] == s]
                print(f"  {s}: " + ", ".join(f"{a} {sum(ex(r, a) for r in Rs):.0f}/{len(Rs)}" for a in arms))
    if "dev_white" in sets and "dev_white_odd" in sets:
        table("dev_white, all 800", sets["dev_white"] + sets["dev_white_odd"], ["C", "Cp", "AR", "ARp"],
              [("Cp", "C"), ("AR", "C"), ("ARp", "Cp"), ("ARp", "AR")])
    if "dev_colored" in sets and "dev_cmid" in sets:
        table("coloured pooled (rho 0.8 + 0.55 + 0.3)", sets["dev_colored"] + sets["dev_cmid"], ["C", "Cp", "AR", "ARp"],
              [("ARp", "AR"), ("ARp", "Cp"), ("AR", "C"), ("Cp", "C")])
    nd2 = load("dev_nd2")
    if "dev_nd" in sets and nd2:
        table("dev_nd2 (second half, extension)", nd2, ["off", "p15"], [("p15", "off")])
        table("dev_nd, all 400", sets["dev_nd"] + nd2, ["off", "p15"], [("p15", "off")])
        for s_ in sorted({r["system"] for r in nd2}):
            Rs = [r for r in sets["dev_nd"] + nd2 if r["system"] == s_]
            print(f"  {s_}: " + ", ".join(f"{a} {np.mean([ex(r, a) for r in Rs]):.3f}" for a in ("off", "p15")))
    # AR estimate distributions and the automatic switch
    print("\n### AR(1) pre-estimate a_hat (mean over components)")
    print("| set | runs | median | 95th pct | max | min |\n|---|---|---|---|---|---|")
    for name in ("dev_white", "dev_white_odd", "dev_student", "dev_coarse", "dev_heldout3", "dev_cmid", "dev_colored"):
        if name in sets:
            if name == "dev_cmid":
                for rho in (0.3, 0.55):
                    a = ahat([r for r in sets[name] if abs(_rho(r) - rho) < 1e-9])
                    print(f"| {name} rho={rho} | {len(a)} | {np.median(a):.3f} | {np.quantile(a, .95):.3f} | "
                          f"{a.max():.3f} | {a.min():.3f} |")
                continue
            a = ahat(sets[name])
            print(f"| {name} | {len(a)} | {np.median(a):.3f} | {np.quantile(a, .95):.3f} | {a.max():.3f} | "
                  f"{a.min():.3f} |")
    print("\n### Switch: auto = ARp if a_hat > thr else Cp (pooled W/L vs Cp everywhere, and vs ARp on coloured)")
    print("| thr | " + " | ".join(n for n in ("dev_white", "dev_white_odd", "dev_student", "dev_coarse", "dev_heldout3", "dev_cmid",
                                                "dev_colored") if n in sets) + " | pooled exact |")
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
        cells, tot = [], []
        for name in ("dev_white", "dev_white_odd", "dev_student", "dev_coarse", "dev_heldout3", "dev_cmid",
                     "dev_colored"):
            if name not in sets:
                continue
            R = [r for r in sets[name] if "a_hat" in r]
            pick = [ex(r, "ARp") if np.mean(r["a_hat"]) > thr else ex(r, "Cp") for r in R]
            base = [ex(r, "Cp") for r in R]
            w = sum(p > b for p, b in zip(pick, base))
            l = sum(p < b for p, b in zip(pick, base))
            cells.append(f"{np.mean(pick):.3f} ({w}/{l} vs Cp; {np.mean([np.mean(r['a_hat']) > thr for r in R]):.0%} AR)")
            tot += pick
        print(f"| {thr} | " + " | ".join(cells) + f" | {np.mean(tot):.3f} |")


def _rho(r):
    return r.get("rho", np.nan)


if __name__ == "__main__":
    main()
