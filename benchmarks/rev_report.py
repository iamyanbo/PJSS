"""Report of the revision experiments E2 and E3 (results_rev/PREREGISTRATION.md).

  python benchmarks/rev_report.py

Scoring as in benchmarks/h2h_report.py: analysis set = longest prefix of complete rounds of 8; DEDi selects by
validation error (NaN as +inf, ties to the first pair in the authors' loop order); a run with an errored pair counts as
a DEDi failure; Newcombe paired 95% CI for d = rate(DEDi) - rate(ours); sign test on discordant runs.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, "benchmarks")
from h2h_report import LAMBDAS, RS, SUBSETS, newcombe_paired, pick, tag

R = Path("results_rev")
GRID = [(l, r) for l in LAMBDAS for r in RS]


def load_pairs(folder, key):
    pairs = {}
    for lr in GRID:
        f = folder / f"{tag(key, *lr)}.json"
        if f.exists():
            pairs[lr] = json.loads(f.read_text())
    return pairs


def compare(rows, col):
    n11 = sum(1 for r in rows if r[col] and r["ours"])
    n10 = sum(1 for r in rows if r[col] and not r["ours"])
    n01 = sum(1 for r in rows if not r[col] and r["ours"])
    n00 = len(rows) - n11 - n10 - n01
    lo, hi = newcombe_paired(n11, n10, n01, n00)
    p = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    return {"n": len(rows), "ours": n11 + n01, "dedi": n11 + n10, "ours_only": n01, "dedi_only": n10, "p": p,
            "d": (n10 - n01) / len(rows), "lo": lo, "hi": hi}


def verdict(c):
    if c["lo"] > 0 and c["p"] < 0.05:
        return "DEDi better"
    if c["hi"] < 0 and c["p"] < 0.05:
        return "ours better"
    return "comparable" if c["hi"] <= 0.10 else "inconclusive"


def e2():
    order = json.loads((R / "order_dysts.json").read_text())
    ours = {r["key"]: r for r in json.loads((R / "ours_dysts.json").read_text())}
    runs = []
    for run in order:
        key = f"{run['cond']}_{run['index']}"
        runs.append((run, key, load_pairs(R / "e2" / "pairs", key)))
    k = 0
    while k < len(runs) and len(runs[k][2]) == 35:
        k += 1
    runs = runs[:k - k % 8]
    print(f"E2 analysis set: {len(runs)} runs, {len(runs) // 8} complete rounds, "
          f"{len({r[0]['system'] for r in runs})} systems")
    rows = []
    for run, key, pairs in runs:
        errored = any("error" in p for p in pairs.values())
        rec = {"key": key, "system": run["system"], "noise": run["noise"], "n": run["n"],
               "ours": int(ours[key]["exact"]), "ours_s": ours[key]["seconds"], "errored": int(errored),
               "ambiguous": int(ours[key]["ambiguous"]), "ours_f1": ours[key]["f1"],
               "dedi_s": sum(p["seconds"] for p in pairs.values())}
        for name, grid in SUBSETS.items():
            for crit in ("val_error", "bic"):
                sel = pairs[pick(pairs, grid, crit)]
                rec[f"{name}|{crit}"] = 0 if errored or "dedi" not in sel else int(sel["dedi"]["exact"])
                if name == "full grid":
                    rec[f"f1|{crit}"] = sel.get("dedi", {}).get("f1", 0.0)
        rec["oracle"] = int(any(p.get("dedi", {}).get("exact") for p in pairs.values()))
        rows.append(rec)
    print(f"errored runs: {sum(r['errored'] for r in rows)}; ours ambiguous: {sum(r['ambiguous'] for r in rows)}")
    for label, col in (("val (primary)", "full grid|val_error"), ("BIC", "full grid|bic"), ("oracle", "oracle")):
        c = compare(rows, col)
        print(f"{label}: {c['ours']}/{c['n']} ours, {c['dedi']}/{c['n']} DEDi, ours-only {c['ours_only']}, DEDi-only "
              f"{c['dedi_only']}, p={c['p']:.3g}, d={c['d']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] -> {verdict(c)}")
    for name in SUBSETS:
        print(f"{name}: val {sum(r[name + '|val_error'] for r in rows)}, BIC {sum(r[name + '|bic'] for r in rows)}, "
              f"oracle-free")
    print("by cell (noise, N): ours / DEDi val / DEDi BIC / oracle")
    for e in sorted({r["noise"] for r in rows}):
        for n in sorted({r["n"] for r in rows}):
            s = [r for r in rows if r["noise"] == e and r["n"] == n]
            print(f"  {e:<5} {n:<5} n={len(s)} {sum(r['ours'] for r in s)} {sum(r['full grid|val_error'] for r in s)} "
                  f"{sum(r['full grid|bic'] for r in s)} {sum(r['oracle'] for r in s)}")
    print(f"mean F1: ours {np.mean([r['ours_f1'] for r in rows]):.3f}, DEDi val {np.mean([r['f1|val_error'] for r in rows]):.3f}, "
          f"BIC {np.mean([r['f1|bic'] for r in rows]):.3f}")
    print(f"median s: ours {np.median([r['ours_s'] for r in rows]):.1f}, DEDi GPU process-seconds "
          f"{np.median([r['dedi_s'] for r in rows]):.0f}, median ratio "
          f"{np.median([r['dedi_s'] / r['ours_s'] for r in rows]):.0f}")
    print("disagreements (val/BIC):")
    for r in rows:
        if r["ours"] != r["full grid|val_error"] or r["ours"] != r["full grid|bic"]:
            print(f"  {r['key']:18s} {r['system']:24s} ours {r['ours']} val {r['full grid|val_error']} "
                  f"BIC {r['full grid|bic']} oracle {r['oracle']}")
    (R / "e2_rows.json").write_text(json.dumps(rows, indent=1, default=float))


def e3():
    if not "DONE" in (R / "progress.log").read_text():
        print("E3 not finished")
        return
    order = json.loads((R / "order_timing.json").read_text())
    print("| run | ours s (9 repeats) | DEDi CPU core-s | ratio | DEDi GPU process-s | ratio |")
    out = []
    for run in order:
        key = f"{run['cond']}_{run['index']}"
        ours = [r["seconds"] for c_ in (15, 16, 17) for r in json.loads((R / f"ours_timing_c{c_}.json").read_text())
                if r["key"] == key]
        cpu = load_pairs(R / "e3cpu" / "pairs", key)
        gpu = load_pairs(R / "e3gpu" / "pairs", key)
        c, g = sum(p["seconds"] for p in cpu.values()), sum(p["seconds"] for p in gpu.values())
        m = float(np.median(ours))
        rec = {"key": key, "system": run["system"], "noise": run["noise"], "ours": ours, "cpu": c,
               "cpu_pairs": len(cpu), "gpu": g, "gpu_pairs": len(gpu)}
        out.append(rec)
        print(f"| {key} {run['system']} | {min(ours):.1f}-{max(ours):.1f} | {c:.0f} ({len(cpu)}) | "
              f"{c / m:.0f} | {g:.0f} ({len(gpu)}) | {g / m:.0f} |")
    (R / "e3_rows.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    e2()
    e3()
