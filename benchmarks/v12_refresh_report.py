"""v1.2 refresh: compare v1.2 (current default) with the stored v1.0 results on dysts and hard-120.

  python benchmarks/v12_refresh_report.py
"""
import json
from pathlib import Path

from scipy.stats import binomtest


def recs(path):
    d = json.loads(Path(path).read_text())
    r = d["records"] if isinstance(d, dict) else d
    return {x["index"]: x for x in r}


def exact(x, arm=None):
    if arm:
        return int(x["arms"][arm]["exact"])
    return int(x["final"]["exact"])  # dysts_bench record, method "final"


def main():
    new, old = recs("results_v12refresh/dysts_v12.json"), recs("results_dysts/test_final.json")
    common = sorted(set(new) & set(old))
    n = [exact(new[i]) for i in common]
    o = [exact(old[i]) for i in common]
    w = sum(a and not b for a, b in zip(n, o))
    l = sum(b and not a for a, b in zip(n, o))
    p = binomtest(w, w + l).pvalue if w + l else 1.0
    print(f"dysts: {len(common)} runs; v1.2 {sum(n) / len(common):.3f}, stored v1.0 {sum(o) / len(common):.3f}; "
          f"v1.2-only/v1.0-only {w}/{l} (sign p = {p:.3g})")
    flips = [(new[i].get("system"), new[i].get("noise"), new[i].get("n"), exact(old[i]), exact(new[i]))
             for i in common if exact(new[i]) != exact(old[i])]
    for f in flips:
        print("  flip", f)
    hn, ho = recs("results_v12refresh/hard120_v12.json"), recs("results_hard120/ours_hard120.json")
    hc = sorted(set(hn) & set(ho))
    ch = sum(exact(hn[i], "final_v12") != exact(ho[i], "final") for i in hc)
    print(f"hard-120: v1.2 {sum(exact(hn[i], 'final_v12') for i in hc)}/{len(hc)}, stored v1.0 "
          f"{sum(exact(ho[i], 'final') for i in hc)}/{len(hc)}, outcome changes {ch}")


if __name__ == "__main__":
    main()
