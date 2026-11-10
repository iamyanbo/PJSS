"""Cost anatomy and scaling vs DEDi: tables for docs/reports/anatomy.md.

  python benchmarks/anatomy_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

A = Path("results_anatomy")


def load(name):
    f = A / f"{name}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def main():
    ours = {r["key"]: r for r in load("ours_h2h")}
    print("## 1. Cost anatomy: 10 head-to-head runs, DEDi full grid (GPU) vs ours\n")
    print("| run | DEDi joint LM solves | DEDi factorisations | DEDi s | elim. rounds/pair (median) "
          "| ours LM fits | ours linear solves | ours subsets scored | ours s | solves ratio | factor. ratio | time ratio |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    runs = defaultdict(list)
    for r in load("dedi_anatomy"):
        runs[r["key"]].append(r)
    tot = defaultdict(list)
    for key, prs in runs.items():
        o = ours[key]
        lm = sum(p.get("lm_solves", 0) for p in prs)
        fa = sum(p.get("factorisations", 0) for p in prs)
        s = sum(p["seconds"] for p in prs)
        rounds = np.median([p.get("lm_solves", 0) / 2 for p in prs])
        print(f"| {key} ({len(prs)} pairs) | {lm} | {fa} | {s:.0f} | {rounds:.0f} | {o['lm_fits']} | {o['lm_solves']} | "
              f"{o['subsets_scored']} | {o['seconds']:.1f} | {lm / o['lm_fits']:.0f}x | {fa / o['lm_solves']:.0f}x | "
              f"{s / o['seconds']:.0f}x |")
        for k, v in (("lm", lm / o["lm_fits"]), ("fa", fa / o["lm_solves"]), ("t", s / o["seconds"])):
            tot[k].append(v)
    print(f"\nmedians: joint solves {np.median(tot['lm']):.0f}x, factorisations/linear solves {np.median(tot['fa']):.0f}x, "
          f"time {np.median(tot['t']):.0f}x")
    allp = load("dedi_anatomy")
    print(f"DEDi per pair (n={len(allp)}): median {np.median([p['lm_solves'] for p in allp]):.0f} joint LM solves, "
          f"{np.median([p['factorisations'] for p in allp]):.0f} factorisations")
    oh = list(ours.values())
    print(f"ours per run (80 runs): median {np.median([r['lm_fits'] for r in oh]):.0f} LM fits, "
          f"{np.median([r['lm_solves'] for r in oh]):.0f} linear solves, {np.median([r['subsets_scored'] for r in oh]):.0f} "
          f"subsets scored by the quadratic screen")

    print("\n## 2. Scaling: per DEDi pair (lambda = 1e-3 row, GPU) vs one run of ours (one CPU core)\n")
    print("| case | D | terms/eq | ours s | ours exact | DEDi pairs done | DEDi s/pair (median) | DEDi joint solves/pair "
          "| DEDi (1e-3, 1) exact | DEDi any row pair exact | ratio s/pair : ours |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for oname, dname in (("ours_nd", "dedi_nd"), ("ours_poly4", "dedi_poly4")):
        o = {r["key"]: r for r in load(oname)}
        d = defaultdict(list)
        for r in load(dname):
            d[r["key"]].append(r)
        for key, orr in o.items():
            prs = d.get(key, [])
            single = [p for p in prs if p["lam"] == 1e-3 and p["R"] == 1.0]
            sp = np.median([p["seconds"] for p in prs]) if prs else float("nan")
            print(f"| {orr['system']} ({key}) | {orr['D']} | {orr['n_terms']} | {orr['seconds']:.1f} | {int(orr['exact'])} | "
                  f"{len(prs)}/5 | {sp:.0f} | {np.median([p['lm_solves'] for p in prs]) if prs else float('nan'):.0f} | "
                  f"{int(single[0]['exact']) if single else '-'} | {int(any(p['exact'] for p in prs)) if prs else '-'} | "
                  f"{sp / orr['seconds']:.0f}x |")
    print("\nReference, D = 3 with the 20-term library (head-to-head runs): DEDi single pair / ours median "
          f"{np.median([sum(p['seconds'] for p in runs[k] if p['lam'] == 1e-3 and p['R'] == 1.0) / ours[k]['seconds'] for k in runs]):.0f}x")


if __name__ == "__main__":
    main()
