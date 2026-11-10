"""Full-grid DEDi vs our final model vs reduced-grid DEDi on a pre-registered subset of hard-120.

  python benchmarks/dedi_fullgrid_report.py --make-order   # writes results_dedi_fullgrid/order.json (pre-registration)
  python benchmarks/dedi_fullgrid_report.py [--pairs results_dedi_fullgrid/main/pairs]

Pre-registration: results_dedi_fullgrid/PREREGISTRATION.json. Per-pair results come from
benchmarks/dedi_fullgrid.py. Analysis set: the longest prefix of order.json in which all 35 pairs of every run
finished. Selection (primary): argmin validation error over the 35 pairs, NaN as +inf, ties to the first pair
in the authors' loop order (lambda outer, R inner); a run with any errored pair counts as a DEDi failure.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

H = Path("results_hard120")
F = Path("results_dedi_fullgrid")
SYS = ("lorenz", "chen", "rossler")
NAME = {"lorenz": "Lorenz", "chen": "Chen", "rossler": "Rössler"}
LAMBDAS = [10.0**i for i in range(-3, 4)]
RS = [10.0**j for j in range(-4, 1)]
SMALL = {(1e-2, 1e-4), (1e-2, 1e-2), (1.0, 1e-4), (1.0, 1e-2)}
SEED = 20261026


def load(f):
    return {r["index"]: r for r in json.loads(Path(f).read_text())["records"]}


def base():
    """Our final model and reduced-grid DEDi on the hard-120 analysis set (as in hard120_report.py)."""
    ours = load(H / "ours_hard120.json")
    dedi = load(H / "dedi_hard120.json") | load(H / "dedi_hard120_part2.json")
    order = json.loads((H / "dedi_order.json").read_text())
    k = 0
    while k < len(order) and order[k] in dedi:
        k += 1
    A = order[: k - k % 3]
    fin = {i: int(ours[i]["arms"]["final"]["exact"]) for i in A}
    red = {i: int(dedi[i]["dedi"]["exact"]) for i in A}
    return ours, dedi, A, fin, red


def stratum(f, d):
    return {(1, 0): "A", (0, 1): "B", (1, 1): "C", (0, 0): "D"}[(f, d)]


def make_order():
    _, _, A, fin, red = base()
    s = {i: stratum(fin[i], red[i]) for i in A}
    rng = np.random.default_rng(SEED)
    t1 = [int(i) for i in rng.permutation(sorted(i for i in A if s[i] in "ABD"))]
    t2 = [int(i) for i in rng.permutation(sorted(i for i in A if s[i] == "C"))]
    order = []
    while t1 or t2:  # blocks of three disagreement/both-fail runs followed by one both-exact run
        order += [t1.pop(0) for _ in range(min(3, len(t1)))]
        if t2:
            order.append(t2.pop(0))
    out = F / "order.json"
    out.write_text(json.dumps(order))
    print(f"wrote {out}: {len(order)} runs; strata {Counter(s.values())}; first 12: "
          f"{[(i, s[i]) for i in order[:12]]}")


def select(pairs, key="val_error"):
    """pairs: list of 35 per-pair records in loop order. Returns the chosen record (None if any pair errored)."""
    if any("error" in p for p in pairs):
        return None
    v = [p[key] if p[key] == p[key] else math.inf for p in pairs]
    return pairs[int(np.argmin(v))]  # argmin returns the first minimum = authors' stable sort


def sign(a, b):
    w = sum(x > y for x, y in zip(a, b))
    l = sum(x < y for x, y in zip(a, b))
    return w, l, (binomtest(w, w + l).pvalue if w + l else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-order", action="store_true")
    ap.add_argument("--pairs", default=str(F / "main" / "pairs"))
    args = ap.parse_args()
    if args.make_order:
        return make_order()
    ours, dedi, A, fin, red = base()
    strat = {i: stratum(fin[i], red[i]) for i in A}
    order = json.loads((F / "order.json").read_text())
    P = Path(args.pairs)

    def run(i):
        fs = [P / f"{i:04d}_l{np.log10(l):+.0f}_r{np.log10(r):+.0f}.json" for l in LAMBDAS for r in RS]
        return [json.loads(f.read_text()) for f in fs] if all(f.exists() for f in fs) else None

    runs = {}
    for i in order:
        g = run(i)
        if g is None:
            break
        runs[i] = g
    S = list(runs)
    n_pairs = len(list(P.glob("*.json")))
    print(f"Analysis set: first {len(S)} of {len(order)} runs in the pre-registered order "
          f"({n_pairs} pair results on disk). Strata in set: {dict(Counter(strat[i] for i in S))}.")
    if not S:
        return
    err = {i: sum("error" in p for p in runs[i]) for i in S}
    print(f"Runs with an errored pair: {sum(1 for v in err.values() if v)} ({sum(err.values())} pairs).")

    chosen = {i: select(runs[i]) for i in S}
    full = {i: int(chosen[i] is not None and chosen[i]["dedi"]["exact"]) for i in S}
    # check: the 4 reduced-grid pairs, selected the same way, should reproduce the stored reduced-grid result
    sm = {i: select([p for p in runs[i] if (p["lam"], p["R"]) in SMALL]) for i in S}
    same = sum(sm[i] is not None and [sm[i]["lam"], sm[i]["R"]] == dedi[i]["chosen"]
               and sm[i]["dedi"]["exact"] == dedi[i]["dedi"]["exact"] for i in S)
    print(f"Reproduction check: reduced-grid choice and exactness reproduced on {same}/{len(S)} runs.\n")

    def row(lab, ix, d):
        f = [fin[i] for i in ix]
        r = [red[i] for i in ix]
        u = [d[i] for i in ix]
        w1, l1, p1 = sign(f, u)
        w0, l0, p0 = sign(f, r)
        wr, lr, pr = sign(u, r)
        print(f"| {lab} | {len(ix)} | {sum(f)} | {sum(r)} | {sum(u)} | {w1}/{l1} | {p1:.2g} | {w0}/{l0} | {p0:.2g} "
              f"| {wr}/{lr} | {pr:.2g} |")

    hdr = ("| Subset | n | Final | DEDi reduced | DEDi full | Final vs full W/L | p | Final vs reduced W/L | p "
           "| Full vs reduced W/L | p |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    def table(title, d):
        print(f"## {title}\n\n{hdr}")
        for s in SYS:
            ix = [i for i in S if ours[i]["system"] == s]
            if ix:
                row(NAME[s], ix, d)
        for st, lab in (("A", "stratum A (final only)"), ("B", "stratum B (reduced DEDi only)"),
                        ("C", "stratum C (both exact)"), ("D", "stratum D (neither)")):
            ix = [i for i in S if strat[i] == st]
            if ix:
                row(lab, ix, d)
        row("**All**", S, d)
        print()

    table("Primary: full grid, selection by validation error (FindBestModel default)", full)

    # projection to the 114-run analysis set: within each stratum, the sampled outcome rates
    N = Counter(strat.values())
    W = L = 0.0
    for st in "ABCD":
        ix = [i for i in S if strat[i] == st]
        if ix:
            W += N[st] * np.mean([fin[i] > full[i] for i in ix])
            L += N[st] * np.mean([fin[i] < full[i] for i in ix])
    print(f"Stratum-weighted projection to all {len(A)} runs (sampled strata only: "
          f"{ {st: f'{sum(strat[i] == st for i in S)}/{N[st]}' for st in 'ABCD'} }): "
          f"final vs full-grid DEDi about {W:.1f} wins / {L:.1f} losses (reduced grid: 26/5).")
    # post hoc (not pre-registered): an unsampled stratum keeps its reduced-grid outcome instead of counting 0
    miss = [st for st in "ABCD" if not any(strat[i] == st for i in S)]
    W2 = W + sum(N[st] for st in miss if st == "A")
    L2 = L + sum(N[st] for st in miss if st == "B")
    print(f"Post hoc variant, unsampled strata {miss} kept at their reduced-grid outcome: "
          f"about {W2:.1f} wins / {L2:.1f} losses.\n")

    # post hoc: the 4 reduced-grid pairs recomputed in this run, selected the same way (same numerics as full)
    red2 = {i: int(sm[i] is not None and sm[i]["dedi"]["exact"]) for i in S}
    a = sign([full[i] for i in S], [red2[i] for i in S])
    b = sign([fin[i] for i in S], [red2[i] for i in S])
    print(f"Post hoc: reduced grid recomputed in this run: {sum(red2.values())}/{len(S)} exact (stored reduced-grid "
          f"run: {sum(red[i] for i in S)}/{len(S)}). Full vs recomputed reduced W/L {a[0]}/{a[1]} (p = {a[2]:.2g}); "
          f"final vs recomputed reduced {b[0]}/{b[1]} (p = {b[2]:.2g}).\n")
    for i in S:
        if sm[i] is None or [sm[i]["lam"], sm[i]["R"]] != dedi[i]["chosen"] or red2[i] != red[i]:
            print(f"  not reproduced: run {i} ({ours[i]['system']}): stored choice {dedi[i]['chosen']} exact {red[i]}; "
                  f"recomputed choice {[sm[i]['lam'], sm[i]['R']]} exact {red2[i]}")
    print()

    # secondary (pre-registered): BIC on the validation points; oracle (any of 35 pairs exact)
    bic = {i: int((c := select(runs[i], "bic")) is not None and c["dedi"]["exact"]) for i in S}
    table("Secondary: full grid, selection by validation BIC", bic)
    orc = {i: int(any(p.get("dedi", {}).get("exact") for p in runs[i])) for i in S}
    table("Secondary: oracle DEDi (exact if any of the 35 pairs is exact; uses the truth, not a method)", orc)
    errx = {i: int((c := select([p for p in runs[i] if "error" not in p])) is not None and c["dedi"]["exact"])
            for i in S}
    if errx != full:
        table("Sensitivity: errored pairs dropped instead of failing the run", errx)

    print("## Chosen hyperparameters (full grid, validation error)\n")
    cnt = Counter((c["lam"], c["R"]) for c in chosen.values() if c)
    print(", ".join(f"λ={l:g}, R={r:g}: {k}" for (l, r), k in cnt.most_common()))
    lam_c = Counter(c["lam"] for c in chosen.values() if c)
    print("λ marginal: " + ", ".join(f"{l:g}: {lam_c[l]}" for l in LAMBDAS))
    fp_f = sum(c["dedi"]["fp"] for c in chosen.values() if c)
    fn_f = sum(c["dedi"]["fn"] for c in chosen.values() if c)
    fp_r = sum(dedi[i]["dedi"]["fp"] for i in S)
    fn_r = sum(dedi[i]["dedi"]["fn"] for i in S)
    fp_o = sum(ours[i]["arms"]["final"]["fp"] for i in S)
    fn_o = sum(ours[i]["arms"]["final"]["fn"] for i in S)
    print(f"\nFalse positives / negatives on the set: final {fp_o}/{fn_o}, DEDi reduced {fp_r}/{fn_r}, "
          f"DEDi full {fp_f}/{fn_f}.")
    both = [i for i in S if fin[i] and full[i]]
    if both:
        ce_o = np.median([ours[i]["arms"]["final"]["coef_err"] for i in both])
        ce_d = np.median([chosen[i]["dedi"]["coef_err"] for i in both])
        print(f"Median relative coefficient error where both exact (n = {len(both)}): final {ce_o:.2%}, "
              f"full-grid DEDi {ce_d:.2%}.")
    exact_by = Counter((p["lam"], p["R"]) for i in S for p in runs[i] if p.get("dedi", {}).get("exact"))
    print("\nExact pairs per (λ, R) over the set (how often each grid point alone would recover the truth):")
    print("| λ \\ R | " + " | ".join(f"{r:g}" for r in RS) + " |\n| --- |" + " --- |" * len(RS))
    for l in LAMBDAS:
        print(f"| {l:g} | " + " | ".join(str(exact_by[(l, r)]) for r in RS) + " |")
    sec = [p["seconds"] for i in S for p in runs[i]]
    per_run = [sum(p["seconds"] for p in runs[i]) / 3600 for i in S]
    print(f"\nCost: median {np.median(sec) / 60:.1f} min per pair (max {max(sec) / 3600:.2f} h); "
          f"median {np.median(per_run):.1f} core-hours per run (range {min(per_run):.1f}–{max(per_run):.1f}).")
    by_lam = {l: np.median([p["seconds"] for i in S for p in runs[i] if p["lam"] == l]) / 60 for l in LAMBDAS}
    print("Median minutes per pair by λ: " + ", ".join(f"{l:g}: {v:.0f}" for l, v in by_lam.items()))


if __name__ == "__main__":
    main()
