"""Tables for the component ablation (benchmarks/ablation.py outputs in results_ablation/).

    python benchmarks/ablation_report.py [--dir results_ablation] > results_ablation/report.md

Per evaluation set and variant: exact-support rate; paired comparison with the full final model on the same
runs ("full+ / var+" = runs only the full model / only the variant gets exactly right) with a two-sided sign
test; median relative coefficient error ||xi - xi*|| / ||xi*|| on the runs both get exact; median runtime of
the joint stage (seeds excluded, they are shared). The ladder compares each rung with the previous one.
Also checks that "full" reproduces the frozen test pass (results_final/test_*.json, arm "final").
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SETS = [("sealed", "Sealed (white)"), ("colored", "Coloured AR(1)"), ("hard40", "Hard 40"),
        ("dysts", "dysts D=3")]
LOO = [("full", "**Full final model**"),
       ("weak", "Weak best-subset only (no joint states)"),
       ("hybrid", "Hybrid (weak/flow, no joint states)"),
       ("no_proposals", "− profiled-quadratic proposals (refit starts only)"),
       ("greedy", "Greedy (stepwise) instead of best-subset proposals"),
       ("single_start", "− multiple starts (weak start only)"),
       ("no_selfcal", "− σ/φ self-calibration (HF σ, φ = 1)"),
       ("no_phi", "− φ autocorrelation inflation"),
       ("gaussian", "− Huber (Gaussian loss)"),
       ("s2", "− auto sub-steps (s = 2)"),
       ("no_prior", "− complexity prior (degw = 0)"),
       ("white", "− AR(1) auto-switch (noise = white)"),
       ("oracle", "Oracle noise level (not deployable)")]
LADDER = [("weak", "Weak best-subset (χ²s)"),
          ("L1_refit", "+ joint states (refit weak start; s=2, HF σ, Gaussian)"),
          ("L2_proposals", "+ profiled-quadratic best-subset proposals"),
          ("L3_starts", "+ multiple starts (hybrid, flow)"),
          ("L4_selfcal", "+ σ/φ self-calibration"),
          ("L5_substeps", "+ auto sub-steps"),
          ("L6_huber", "+ Huber loss (= config C)"),
          ("white", "+ complexity prior (= C + prior)"),
          ("full", "+ AR(1) auto-switch (= final)")]
FROZEN = {"sealed": "test_sealed.json", "colored": "test_colored.json", "hard40": "test_confirm.json"}


def get(r, v):
    if v in ("weak", "hybrid"):
        return r[v]
    return r["variants"].get(v)


def sign(w, l):
    return binomtest(w, w + l).pvalue if w + l else 1.0


def fmt_p(p):
    return "1" if p >= 0.995 else (f"{p:.2f}" if p >= 0.01 else f"{p:.0e}".replace("e-0", "e-"))


def pair(R, a, b):
    """runs where only a / only b is exact, over runs having both."""
    A = [(get(r, a), get(r, b)) for r in R if get(r, a) is not None and get(r, b) is not None]
    wa = sum(x["exact"] > y["exact"] for x, y in A)
    wb = sum(x["exact"] < y["exact"] for x, y in A)
    return wa, wb, len(A)


def coef(R, v):
    e = [get(r, v)["coef_err"] for r in R if get(r, v) is not None and get(r, v).get("exact")
         and get(r, "full")["exact"] and "coef_err" in get(r, v)]
    return np.median(e) if e else np.nan


def secs(R, v):
    if v in ("weak", "hybrid"):
        return np.nan
    s = [get(r, v)["seconds"] for r in R if get(r, v) is not None]
    return np.median(s) if s else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results_ablation")
    a = ap.parse_args()
    D = Path(a.dir)
    data = {}
    for s, _ in SETS:
        f = D / f"{s}.json"
        if f.exists():
            data[s] = json.loads(f.read_text())["records"]
    sets = [(s, lab) for s, lab in SETS if s in data]

    print("## Integrity\n")
    for s, lab in sets:
        R = data[s]
        errs = sum("error" in v for r in R for v in r["variants"].values())
        line = f"- {lab}: {len(R)} runs; variant errors {errs}"
        if all(r["starts_cached"] for r in R):
            line += f"; weak re-run equals cached seed {sum(r['weak_matches_cache'] for r in R)}/{len(R)}"
        else:
            line += "; seeds computed here (final_model.default_starts)"
        if s in FROZEN and (Path("results_final") / FROZEN[s]).exists():
            T = {r["index"]: r for r in json.loads((Path("results_final") / FROZEN[s]).read_text())["records"]}
            same = [r for r in R if r["index"] in T]
            eq = sum(r["variants"]["full"]["exact"] == T[r["index"]]["arms"]["final"]["exact"] and
                     np.isclose(r["variants"]["full"]["coef_err"], T[r["index"]]["arms"]["final"]["coef_err"],
                                rtol=1e-9, atol=1e-12) for r in same)
            line += f"; full = frozen test pass on {eq}/{len(same)} runs (exact flag and coefficient error)"
        if s == "dysts" and Path("results_dysts/test_final.json").exists():
            T = {r["index"]: r for r in json.loads(Path("results_dysts/test_final.json").read_text())["records"]}
            eq = sum(r["variants"]["full"]["exact"] == T[r["index"]]["final"]["exact"] and
                     np.isclose(r["variants"]["full"]["coef_err"], T[r["index"]]["final"]["coef_err"], rtol=1e-6)
                     for r in R if r["index"] in T)
            all3 = np.mean([T[i]["final"]["exact"] for i in T if T[i]["D"] == 3])
            line += (f"; full = frozen dysts run (results_dysts/test_final.json) on {eq}/{len(R)} runs; "
                     f"this subsample's full rate vs all 4,880 D=3 runs: "
                     f"{np.mean([r['variants']['full']['exact'] for r in R]):.3f} vs {all3:.3f}")
        print(line)
    print()

    hdr = "| Variant | " + " | ".join(lab for _, lab in sets) + " | Pooled full+/var+ (p) |"
    print("## Leave-one-out from the full model\n")
    print("Cells: exact rate · full+/var+ (sign-test p) · median coef. error on jointly exact runs · median s/run "
          "(joint stage).\n")
    print(hdr)
    print("|" + " --- |" * (len(sets) + 2))
    for v, lab in LOO:
        cells, W, L = [], 0, 0
        for s, _ in sets:
            R = data[s]
            have = [r for r in R if get(r, v) is not None]
            if not have:
                cells.append("–")
                continue
            ex = np.mean([get(r, v)["exact"] for r in have])
            if v == "full":
                cells.append(f"**{ex:.3f}** · – · {coef(R, v) * 100:.2f}% · {secs(R, v):.1f}s")
                continue
            wf, wv, _ = pair(R, "full", v)
            if v != "oracle":
                W, L = W + wf, L + wv
            t = secs(R, v)
            cells.append(f"{ex:.3f} · {wf}/{wv} ({fmt_p(sign(wf, wv))}) · {coef(R, v) * 100:.2f}%"
                         + ("" if np.isnan(t) else f" · {t:.1f}s"))
        pooled = "–" if v in ("full", "oracle") else f"{W}/{L} ({fmt_p(sign(W, L))})"
        print(f"| {lab} | " + " | ".join(cells) + f" | {pooled} |")

    print("\n## Cumulative ladder (weak best-subset → final)\n")
    print("Cells: exact rate · wins/losses vs the previous rung (sign-test p).\n")
    print("| Rung | " + " | ".join(lab for _, lab in sets) + " |")
    print("|" + " --- |" * (len(sets) + 1))
    prev = None
    for v, lab in LADDER:
        cells = []
        for s, _ in sets:
            R = data[s]
            ex = np.mean([get(r, v)["exact"] for r in R])
            if prev is None:
                cells.append(f"{ex:.3f}")
            else:
                w, l, _ = pair(R, v, prev)
                cells.append(f"{ex:.3f} · {w}/{l} ({fmt_p(sign(w, l))})")
        print(f"| {lab} | " + " | ".join(cells) + " |")
        prev = v

    print("\n## Detail\n")
    for s, lab in sets:
        R = data[s]
        cached = all(r["starts_cached"] for r in R)
        st = [r.get("starts_seconds") for r in R if r.get("starts_seconds") is not None]
        print(f"- {lab}: median weak-selector time {np.median([r['weak_seconds'] for r in R]):.2f}s"
              + ("" if cached else f", median all-seeds time {np.median(st):.1f}s")
              + f"; AR(1) chosen by full model in {np.mean([r['variants']['full'].get('noise_model') == 'ar1' for r in R]):.0%}"
              + f"; median auto sub-steps {np.median([r['variants']['full'].get('substeps') or 0 for r in R]):.0f}")
        for key in ("noise", "n"):
            vals = sorted({r[key] for r in R})
            row = []
            for x in vals:
                Rx = [r for r in R if r[key] == x]
                row.append(f"{key}={x}: n={len(Rx)} full {np.mean([r['variants']['full']['exact'] for r in Rx]):.2f}"
                           f" / weak {np.mean([r['weak']['exact'] for r in Rx]):.2f}")
            print("  - " + "; ".join(row))


if __name__ == "__main__":
    main()
