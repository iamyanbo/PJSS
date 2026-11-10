"""Head-to-head report: our model v1.2 vs full-grid DEDi on the pre-registered random sample (results_h2h).

  python benchmarks/h2h_report.py            # prints the tables used in docs/reports/h2h.md

Rules are those of results_h2h/PREREGISTRATION.md: analysis set = longest prefix of complete rounds; DEDi selects
by validation error (NaN as +inf, ties to the first pair in the authors' loop order, lambda outer, R inner); a run
with an errored pair counts as a DEDi failure. Paired 95% CI for d = rate(DEDi) - rate(ours) by Newcombe's
method 10 (Wilson score intervals combined with the phi correlation); sign test on discordant runs.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

H = Path("results_h2h")
LAMBDAS = [10.0**i for i in range(-3, 4)]
RS = [10.0**j for j in range(-4, 1)]
CONDS = ["hard", "white", "coloured", "t3", "short"]
SUBSETS = {"single (1e-3, 1)": [(1e-3, 1.0)], "row lambda=1e-3": [(1e-3, r) for r in RS],
           "full grid": [(l, r) for l in LAMBDAS for r in RS]}


def tag(key, lam, r):
    return f"{key}_l{np.log10(lam):+.0f}_r{np.log10(r):+.0f}"


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def newcombe_paired(n11, n10, n01, n00):
    """95% CI for p1 - p2, p1 = (n11+n10)/n, p2 = (n11+n01)/n (Newcombe 1998, method 10)."""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = (n11 * n00 - n10 * n01) / math.sqrt(den) if den else 0.0
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2 - dl, p1 - p2 + du


def pick(pairs, grid, key):
    """Index into grid of the selected pair (argmin of key, NaN/missing as +inf, first wins ties)."""
    vals = []
    for lr in grid:
        v = pairs[lr].get(key)
        vals.append(math.inf if v is None or not np.isfinite(v) else v)
    return grid[int(np.argmin(vals))]


def main():
    order = json.loads((H / "order.json").read_text())
    ours = {}
    for c in CONDS:
        for r in json.loads((H / f"ours_{c}.json").read_text())["records"]:
            a = r["arms"]["final_v12"]
            secs = a["seconds"] + r.get("starts_seconds", 0) + r.get("a_seconds", 0)
            ours[f"{c}_{r['index']}"] = a | {"total_seconds": secs}
    device = {p.stem: p.suffix[1:] for p in (H / "runs").iterdir()}
    runs = []
    for run in order:
        key = f"{run['cond']}_{run['index']}"
        pairs = {}
        for lam in LAMBDAS:
            for r in RS:
                f = H / "pairs" / f"{tag(key, lam, r)}.json"
                if f.exists():
                    pairs[(lam, r)] = json.loads(f.read_text())
        runs.append((run, key, pairs))
    # analysis set: longest prefix of complete rounds
    k = 0
    while k < len(runs) and len(runs[k][2]) == 35:
        k += 1
    k -= k % len(CONDS)
    runs = runs[:k]
    print(f"analysis set: {k} runs ({k // len(CONDS)} complete rounds)")

    rows = []
    for run, key, pairs in runs:
        errored = any("error" in p for p in pairs.values())
        rec = {"cond": run["cond"], "system": run["system"], "noise": run["noise"], "key": key,
               "device": device.get(key, "?"), "ours": int(ours[key]["exact"]), "ours_f1": ours[key]["f1"],
               "ours_coef": ours[key]["coef_err"], "ours_s": ours[key]["total_seconds"],
               "dedi_s": sum(p["seconds"] for p in pairs.values())}
        for name, grid in SUBSETS.items():
            for crit in ("val_error", "bic"):
                sel = pairs[pick(pairs, grid, crit)]
                ok = 0 if errored or "dedi" not in sel else int(sel["dedi"]["exact"])
                rec[f"{name}|{crit}"] = ok
                if name == "full grid":
                    rec[f"f1|{crit}"] = sel.get("dedi", {}).get("f1", 0.0)
                    rec[f"coef|{crit}"] = sel.get("dedi", {}).get("coef_err", math.nan)
            rec[f"{name}|seconds"] = sum(pairs[lr]["seconds"] for lr in grid)
        rec["oracle"] = int(any(p.get("dedi", {}).get("exact") for p in pairs.values()))
        rows.append(rec)

    def compare(sub, col, label):
        n11 = sum(1 for r in sub if r[col] and r["ours"])
        n10 = sum(1 for r in sub if r[col] and not r["ours"])
        n01 = sum(1 for r in sub if not r[col] and r["ours"])
        n00 = len(sub) - n11 - n10 - n01
        lo, hi = newcombe_paired(n11, n10, n01, n00)
        p = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
        return (f"| {label} | {len(sub)} | {n11 + n01} | {n11 + n10} | {n01}/{n10} | {p:.3g} | "
                f"{(n11 + n10 - n11 - n01) / len(sub):+.3f} [{lo:+.3f}, {hi:+.3f}] |"), (lo, hi, p, n01, n10)

    head = "| subset | n | ours exact | DEDi exact | ours-only / DEDi-only | sign p | d = DEDi - ours [95% CI] |\n| --- | --- | --- | --- | --- | --- | --- |"
    print("\n## Primary: full grid, validation-error selection\n" + head)
    line, (lo, hi, p, w, l) = compare(rows, "full grid|val_error", "**pooled**")
    print(line)
    for c in CONDS:
        print(compare([r for r in rows if r["cond"] == c], "full grid|val_error", c)[0])
    for dv in ("gpu", "cpu"):
        sub = [r for r in rows if r["device"] == dv]
        if sub:
            print(compare(sub, "full grid|val_error", f"device {dv}")[0])
    if hi <= 0.10:
        verdict = "COMPARABLE (upper bound <= 0.10)"
    else:
        verdict = "INCONCLUSIVE"
    if lo > 0 and p < 0.05:
        verdict = "DEDi BETTER"
    if hi < 0 and p < 0.05:
        verdict = "OURS BETTER"
    print(f"\nPre-registered verdict: {verdict}  (d CI [{lo:+.3f}, {hi:+.3f}], sign p = {p:.3g}, {w}/{l})")

    print("\n## Secondary 1: DEDi BIC selection and oracle\n" + head)
    print(compare(rows, "full grid|bic", "pooled, DEDi by BIC")[0])
    for c in CONDS:
        print(compare([r for r in rows if r["cond"] == c], "full grid|bic", f"{c}, BIC")[0])
    print(compare(rows, "oracle", "pooled, DEDi oracle (uses truth)")[0])

    print("\n## Secondary 3: F1 and coefficient error (full grid)")
    print(f"mean F1: ours {np.mean([r['ours_f1'] for r in rows]):.3f}, DEDi val {np.mean([r['f1|val_error'] for r in rows]):.3f}, "
          f"DEDi BIC {np.mean([r['f1|bic'] for r in rows]):.3f}")
    both = [r for r in rows if r["ours"] and r["full grid|val_error"]]
    if both:
        print(f"median rel. coef error on {len(both)} runs both exact (val): ours {np.median([r['ours_coef'] for r in both]):.4f}, "
              f"DEDi {np.median([r['coef|val_error'] for r in both]):.4f}; ours lower on "
              f"{sum(r['ours_coef'] < r['coef|val_error'] for r in both)}/{len(both)}")

    print("\n## Secondary 4-5: cost and accuracy vs compute")
    print("| method | exact (val) | exact (BIC) | median s/run | median ratio to ours |\n| --- | --- | --- | --- | --- |")
    print(f"| ours v1.2 | {sum(r['ours'] for r in rows)}/{len(rows)} | - | {np.median([r['ours_s'] for r in rows]):.1f} | 1 |")
    for name in SUBSETS:
        for dv in ("gpu", "cpu"):
            sub = [r for r in rows if r["device"] == dv]
            if not sub:
                continue
            s = [r[f"{name}|seconds"] for r in sub]
            ratio = [r[f"{name}|seconds"] / r["ours_s"] for r in sub]
            print(f"| DEDi {name} ({dv}, n={len(sub)}) | {sum(r[f'{name}|val_error'] for r in sub)}/{len(sub)} | "
                  f"{sum(r[f'{name}|bic'] for r in sub)}/{len(sub)} | {np.median(s):.0f} | {np.median(ratio):.0f} |")
    for name in SUBSETS:
        print(f"pooled DEDi {name}: val {sum(r[f'{name}|val_error'] for r in rows)}/{len(rows)}, "
              f"BIC {sum(r[f'{name}|bic'] for r in rows)}/{len(rows)}")

    print("\n## Secondary 6: hard condition pooled with the earlier 17 (partly enriched)")
    old = Path("results_dedi_fullgrid/report.txt")
    print("(see docs/reports/dedi_fullgrid.md for the earlier 17: ours 15/17, DEDi val 14/17, BIC 17/17)")
    hard = [r for r in rows if r["cond"] == "hard"]
    print(f"new random hard runs: ours {sum(r['ours'] for r in hard)}/{len(hard)}, DEDi val "
          f"{sum(r['full grid|val_error'] for r in hard)}/{len(hard)}, BIC {sum(r['full grid|bic'] for r in hard)}/{len(hard)}")
    print(f"pooled with earlier 17: ours {sum(r['ours'] for r in hard) + 15}/{len(hard) + 17}, DEDi val "
          f"{sum(r['full grid|val_error'] for r in hard) + 14}/{len(hard) + 17}, BIC {sum(r['full grid|bic'] for r in hard) + 17}/{len(hard) + 17}")

    print("\n## Per-run disagreements (full grid, val / BIC)")
    for r in rows:
        if r["ours"] != r["full grid|val_error"] or r["ours"] != r["full grid|bic"]:
            print(f"{r['key']:16s} {r['system']:11s} {r['noise']:<5} {r['device']}  ours {r['ours']}  DEDi val "
                  f"{r['full grid|val_error']}  BIC {r['full grid|bic']}  oracle {r['oracle']}")
    (H / "rows.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
