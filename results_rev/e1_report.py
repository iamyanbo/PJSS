"""E1 of results_rev/PREREGISTRATION.md: no_schur and refit_only against the stored ablation."""
import json
from pathlib import Path

from scipy.stats import binomtest

R = Path("results_rev")
A = Path("results_ablation")
SETS = ["sealed", "colored", "hard40", "dysts"]


def recs(p):
    return {r["index"]: r for r in json.loads(Path(p).read_text())["records"]}


def p2(w, l):
    return binomtest(w, w + l, 0.5).pvalue if w + l else float("nan")


pool = {}
for s in SETS:
    old, new, chk = recs(A / f"{s}.json"), recs(R / f"abl_{s}.json"), recs(R / f"abl_check_{s}.json")
    assert set(old) == set(new), s
    full = {i: old[i]["variants"]["full"]["exact"] for i in old}
    same = sum(chk[i]["variants"]["full"]["exact"] == full[i] for i in chk)
    err = sum("error" in r["variants"][v] for r in new.values() for v in r["variants"])
    print(f"{s}: n={len(old)} full={sum(full.values())} check {same}/{len(chk)} identical, errors {err}")
    arms = {"no_schur": {i: new[i]["variants"]["no_schur"]["exact"] for i in old},
            "refit_only": {i: new[i]["variants"]["refit_only"]["exact"] for i in old},
            "no_proposals": {i: old[i]["variants"]["no_proposals"]["exact"] for i in old},
            "single_start": {i: old[i]["variants"]["single_start"]["exact"] for i in old}}
    for a, v in arms.items():
        w = sum(full[i] > v[i] for i in old)
        l = sum(full[i] < v[i] for i in old)
        print(f"  {a}: exact {int(sum(v.values()))}/{len(old)} rate {sum(v.values())/len(old):.3f} full W/L {w}/{l} p={p2(w, l):.3g}")
        x = pool.setdefault(a, [0, 0, 0, 0])
        x[0] += int(sum(v.values())); x[1] += len(old); x[2] += w; x[3] += l
for a, x in pool.items():
    print(f"POOLED {a}: {x[0]}/{x[1]} full W/L {x[2]}/{x[3]} p={p2(x[2], x[3]):.3g}")
