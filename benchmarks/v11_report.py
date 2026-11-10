"""Tables for the v1.1 pre-registered test (results_final_v11/PREREGISTRATION.md) -> stdout, markdown.

    python benchmarks/v11_report.py [--dir results_final_v11]

Primary: data/sealed_colored2.npz (fresh, 800 runs), arms "final" (v1.1) and "final_v101" (v1.0.1) of
final_bench.py in the same records (shared seeds and AR(1) pre-estimate). Metric: exact support recovery.
W/L = runs only v1.1 / only v1.0.1 gets exactly right; two-sided exact sign test (binomial, p = 0.5).
Guards: v1.1 runs compared run for run with the v1.0.1 re-runs in results_v101/ (xi, gap, exact, noise model).
Repro: v1.1 / v1.0.1 on a dev subset vs the stored results_huber_dev ar1_gauss / full records.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]


def _recs(p):
    return {r["index"]: r for r in json.loads(Path(p).read_text())["records"]}


def sign_p(w, l):
    return 1.0 if w + l == 0 else float(binomtest(w, w + l, 0.5).pvalue)


def wl(pairs):
    w = sum(1 for a, b in pairs if a == 1 and b == 0)
    l = sum(1 for a, b in pairs if a == 0 and b == 1)
    return w, l


def fmt_p(p):
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def primary(path):
    R = _recs(path)
    rows = []
    for i, r in sorted(R.items()):
        a, b = r["arms"]["final"], r["arms"]["final_v101"]
        rows.append(dict(i=i, sys=r["system"], eta=r["noise"], n=r["n"], new=a.get("exact", 0.0),
                         old=b.get("exact", 0.0), nm_new=a.get("noise_model"), nm_old=b.get("noise_model"),
                         err=("error" in a) or ("error" in b), ce_new=a.get("coef_err"), ce_old=b.get("coef_err"),
                         f1_new=a.get("f1", 0.0), f1_old=b.get("f1", 0.0),
                         same=a.get("xi") == b.get("xi"), s_new=a.get("seconds", 0), s_old=b.get("seconds", 0)))
    print(f"## Primary: `{Path(path).name}` ({len(rows)} runs)\n")
    w, l = wl([(r["new"], r["old"]) for r in rows])
    p = sign_p(w, l)
    print("| Arm | Exact | Mean F1 |\n| --- | --- | --- |")
    print(f"| v1.0.1 (Huber in AR(1) likelihood) | {np.mean([r['old'] for r in rows]):.3f} | "
          f"{np.mean([r['f1_old'] for r in rows]):.3f} |")
    print(f"| v1.1 (Gaussian in AR(1) likelihood) | {np.mean([r['new'] for r in rows]):.3f} | "
          f"{np.mean([r['f1_new'] for r in rows]):.3f} |")
    print(f"\n**v1.1 vs v1.0.1: W/L = {w}/{l}, two-sided sign test p = {fmt_p(p)}**\n")
    nm = defaultdict(int)
    for r in rows:
        nm[(r["nm_old"], r["nm_new"])] += 1
    print(f"Routing (v1.0.1, v1.1): {dict(nm)}; exceptions: {sum(r['err'] for r in rows)}; "
          f"identical xi: {sum(r['same'] for r in rows)}/{len(rows)}; "
          f"identical xi among white-routed: {sum(r['same'] for r in rows if r['nm_old'] == 'white')}/"
          f"{sum(1 for r in rows if r['nm_old'] == 'white')}\n")
    both = [(r["ce_old"], r["ce_new"]) for r in rows if r["old"] == 1 and r["new"] == 1]
    if both:
        print(f"Median coefficient error on runs both get exact ({len(both)}): v1.0.1 "
              f"{100 * np.median([a for a, _ in both]):.2f}%, v1.1 {100 * np.median([b for _, b in both]):.2f}%; "
              f"median seconds per run (joint stage): v1.0.1 {np.median([r['s_old'] for r in rows]):.1f}, "
              f"v1.1 {np.median([r['s_new'] for r in rows]):.1f}\n")
    print("| Cell | n | v1.0.1 | v1.1 | W/L | p |\n| --- | --- | --- | --- | --- | --- |")
    for lab, key in (("η", "eta"), ("N", "n")):
        for v in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == v]
            w_, l_ = wl([(r["new"], r["old"]) for r in sub])
            vv = f"{100 * v:g}%" if key == "eta" else v
            print(f"| {lab} = {vv} | {len(sub)} | {np.mean([r['old'] for r in sub]):.3f} | "
                  f"{np.mean([r['new'] for r in sub]):.3f} | {w_}/{l_} | {fmt_p(sign_p(w_, l_))} |")
    print("\n| System | n | v1.0.1 | v1.1 | W/L |\n| --- | --- | --- | --- | --- |")
    for s in sorted({r["sys"] for r in rows}):
        sub = [r for r in rows if r["sys"] == s]
        w_, l_ = wl([(r["new"], r["old"]) for r in sub])
        print(f"| {s} | {len(sub)} | {np.mean([r['old'] for r in sub]):.3f} | {np.mean([r['new'] for r in sub]):.3f}"
              f" | {w_}/{l_} |")
    return w, l, p


def guard(new_path, old_path, label):
    N, O = _recs(new_path), _recs(old_path)
    idx = sorted(N)
    changed, pairs, ar = [], [], 0
    for i in idx:
        a, b = N[i]["arms"]["final"], O[i]["arms"]["final"]
        ar += b.get("noise_model") == "ar1"
        same = (a.get("xi") == b.get("xi") and a.get("gap") == b.get("gap")
                and a.get("noise_model") == b.get("noise_model") and a.get("exact") == b.get("exact"))
        if not same:
            changed.append(i)
        pairs.append((a.get("exact", 0.0), b.get("exact", 0.0)))
    w, l = wl(pairs)
    print(f"| {label} | {len(idx)} | {ar} | {len(changed)} | {np.mean([b for _, b in pairs]):.3f} | "
          f"{np.mean([a for a, _ in pairs]):.3f} | {w}/{l} |" + (f" changed: {changed[:20]}" if changed else ""))
    return len(idx), ar, len(changed), w, l


def repro(path, hub_files):
    R = _recs(path)
    H = {}
    for f in hub_files:
        if Path(f).exists():
            H.update(_recs(f))
    ok_new = ok_old = n = 0
    for i, r in R.items():
        if i not in H:
            continue
        n += 1
        v = H[i]["variants"]
        a, b = r["arms"]["final"], r["arms"]["final_v101"]
        ok_new += a["coef_err"] == v["ar1_gauss"]["coef_err"] and a["gap"] == v["ar1_gauss"]["gap"]
        ok_old += b["coef_err"] == v["full"]["coef_err"] and b["gap"] == v["full"]["gap"]
    print(f"Repro on dev coloured ({n} runs): v1.1 = stored ar1_gauss in {ok_new}/{n}; "
          f"v1.0.1 = stored full in {ok_old}/{n} (coef_err and gap, exact equality)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "results_final_v11"))
    D = Path(ap.parse_args().dir)
    V = REPO / "results_v101"
    if (D / "repro_dev_colored.json").exists():
        repro(D / "repro_dev_colored.json", [REPO / f"results_huber_dev/dev_colored_o{k}.json" for k in range(3)])
    if (D / "test_colored2.json").exists():
        primary(D / "test_colored2.json")
    print("\n## Guards (v1.1 vs the v1.0.1 re-run of the same archive)\n")
    print("| Archive | Runs | Routed to AR(1) by v1.0.1 | Answers changed | v1.0.1 exact | v1.1 exact | W/L |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, old, label in (("guard_sealed", "test_sealed", "white, `joint_sealed.npz` (primary guard)"),
                             ("guard_student", "test_student", "t₃, `sealed_student.npz` (primary guard)"),
                             ("guard_coarse_ar1", "test_coarse", "coarse, AR(1)-routed runs only (secondary)"),
                             ("guard_heldout2_ar1", "test_heldout2", "held-out 2, AR(1)-routed runs only (secondary)"),
                             ("support_colored", "test_colored", "old coloured test `sealed_colored.npz` "
                                                                 "(seen before; non-independent)")):
        if (D / f"{name}.json").exists() and (V / f"{old}.json").exists():
            guard(D / f"{name}.json", V / f"{old}.json", label)


if __name__ == "__main__":
    main()
