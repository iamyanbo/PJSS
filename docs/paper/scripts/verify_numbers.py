"""Recompute headline numbers of the paper directly from the result JSON files.
Single process, read-only. Run from the repository root:
    python docs/paper/scripts/verify_numbers.py
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parents[3]

# This archive is excluded from Git. When it is available, audit the exact
# development cell cited in the manuscript rather than an aggregate rate.
if (R / 'data/instances.npz').exists():
    import numpy as np
    with np.load(R / 'data/instances.npz', allow_pickle=True) as archive:
        cell = ((archive['sys'] == 'lorenz') & (archive['noise'] == 0.05)
                & (archive['n'] == 2000))
    print('== Lorenz development diagnostic (5% noise, N=2000; saved results)')
    for method in ('fd9', 'weak20', 'eweak20', 'ps_weak', 'bs_chi2s'):
        path = R / 'results' / f'{method}.npy'
        if not path.exists():
            print(f'  {method}: missing {path}')
            continue
        exact = np.load(path)[cell, :, 0].max(axis=1)
        print(f'  {method}: {int(exact.sum())}/{int(cell.sum())} exact (best stored threshold per run)')
else:
    print('== Lorenz diagnostic skipped: data/instances.npz unavailable')


def load(p):
    return json.loads((R / p).read_text())["records"]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


print("== results_final test archives (arm final, C if present)")
for name in ["sealed", "confirm", "heldout2", "colored", "student", "coarse", "deg4", "deg5", "mixedpoly", "trig", "nd"]:
    recs = load(f"results_final/test_{name}.json")
    arms = recs[0]["arms"].keys()
    out = {a: rate([r["arms"][a]["exact"] for r in recs]) for a in arms}
    flag = rate([r["arms"]["final"].get("ambiguous", False) for r in recs])
    unfl = [r["arms"]["final"]["exact"] for r in recs if not r["arms"]["final"].get("ambiguous", False)]
    weak = [r["weak"] for r in recs if isinstance(r.get("weak"), (int, float))]
    print(f"  {name:10s} n={len(recs):4d} " + " ".join(f"{a}={v:.3f}" for a, v in out.items()),
          f"flag={flag:.2f} unflagged_acc={rate(unfl):.3f}", f"weak={rate(weak):.3f}" if weak else "")

print("== sealed by noise (final)")
recs = load("results_final/test_sealed.json")
by = defaultdict(list)
for r in recs:
    by[(r["noise"], r["n"])].append(r["arms"]["final"]["exact"])
print("  ", {k: round(rate(v), 3) for k, v in sorted(by.items())})

print("== hard 40: final vs DEDi vs ODR")
idx = json.loads((R / "results_final/hard40_idx.json").read_text())
conf = {r["index"]: r for r in load("results_final/test_confirm.json")}
dedi = {r["index"]: r for f in ["results_hybrid/dedi_hard20.json", "results_hybrid/dedi_hard20_rest.json"] for r in load(f)}
odr = {r["index"]: r for r in load("results_odr/hard40.json")}
per = defaultdict(lambda: [0, 0, 0, 0])
w = l = 0
for i in idx:
    s = conf[i]["system"]
    f = conf[i]["arms"]["final"]["exact"]
    d = dedi[i]["dedi"]["exact"]
    o = odr[i]["odr"]["0.005"]["exact"] if isinstance(odr[i]["odr"]["0.005"], dict) and "exact" in odr[i]["odr"]["0.005"] else float("nan")
    per[s][0] += f
    per[s][1] += d
    per[s][2] += o
    per[s][3] += conf[i]["weak"] if isinstance(conf[i].get("weak"), (int, float)) else 0
    w += f > d
    l += d > f
print("  per system [final, dedi, odr5e-3, weak]:", dict(per))
print(f"  final vs DEDi W/L {w}/{l} p={sign_p(w, l):.4f}")
ds = sorted(r["seconds"] for r in dedi.values())
print("  DEDi median s:", ds[len(ds) // 2])

print("== dysts")
for m in ["weak", "c", "final"]:
    recs = load(f"results_dysts/test_{m}.json")
    key = "final"
    ex = [r[key]["exact"] if isinstance(r.get(key), dict) else float("nan") for r in recs]
    print(f"  {m}: n={len(ex)} exact={rate(ex):.3f}")

print("== nd test (joint = C; final in results_final/test_nd.json)")
recs = load("results_nd/test.json")
print("  joint", round(rate([r["joint"]["exact"] for r in recs]), 3))

print("== hard120 (ours only; DEDi pending)")
recs = load("results_hard120/ours_hard120.json")
by = defaultdict(lambda: [0, 0, 0, 0])
for r in recs:
    by[r["system"]][0] += r["arms"]["final"]["exact"]
    by[r["system"]][1] += r["arms"]["C"]["exact"]
    by[r["system"]][2] += r["weak"]
    by[r["system"]][3] += 1
print("  [final, C, weak, n]", dict(by))

print("== final vs config C, paired (results_joint/C_*.json vs results_final/test_*.json)")
pairs = [("sealed", "C_joint_sealed"), ("confirm", "C_hybrid_confirm"), ("heldout2", "C_heldout2"),
         ("colored", "C_sealed_colored"), ("student", "C_sealed_student"), ("coarse", "C_sealed_coarse")]
for fin, cf in pairs:
    F = {r["index"]: r for r in load(f"results_final/test_{fin}.json")}
    C = {r["index"]: r for r in load(f"results_joint/{cf}.json")}
    w = l = 0
    cs, hs, ws = [], [], []
    for i, rf in F.items():
        rc = C[i]
        j = rc["joint"]
        jj = j[list(j.keys())[0]] if "final" not in j else j
        ce = jj["final"]["exact"] if "final" in jj else jj["exact"]
        fe = rf["arms"]["final"]["exact"]
        w += fe > ce
        l += ce > fe
        cs.append(ce)
        hs.append(rc["hyb_bic10"]["exact"])
        ws.append(rc["weak"]["exact"])
    print(f"  {fin:9s} n={len(F)} weak={rate(ws):.3f} hybrid={rate(hs):.3f} C={rate(cs):.3f} final W/L vs C {w}/{l} p={sign_p(w, l):.2g}")

print("== head-to-head, 80 random runs (results_h2h/rows.json; ours v1.2 vs full-grid DEDi)")
rows = json.loads((R / "results_h2h/rows.json").read_text())
for label, keep in [("all", lambda r: True), ("all but coloured", lambda r: r["cond"] != "coloured")] + \
        [(c, (lambda c: lambda r: r["cond"] == c)(c)) for c in ["hard", "white", "coloured", "t3", "short"]]:
    sub = [r for r in rows if keep(r)]
    out = [f"  {label:16s} n={len(sub):2d} ours={sum(r['ours'] for r in sub)}"]
    for rule in ["full grid|val_error", "full grid|bic", "oracle"]:
        d = [r[rule] for r in sub]
        w = sum(o > x for o, x in zip((r["ours"] for r in sub), d))
        l = sum(x > o for o, x in zip((r["ours"] for r in sub), d))
        out.append(f"{rule.split('|')[-1]}={sum(d)} W/L {w}/{l} p={sign_p(w, l):.2g}")
    print("  ".join(out))
nocpu = [r for r in rows if r["device"] == "gpu"]
w = sum(r["ours"] > r["full grid|val_error"] for r in nocpu)
l = sum(r["full grid|val_error"] > r["ours"] for r in nocpu)
print(f"  GPU runs only: n={len(nocpu)} ours={sum(r['ours'] for r in nocpu)} val={sum(r['full grid|val_error'] for r in nocpu)} W/L {w}/{l} p={sign_p(w, l):.2g}")
for dev in ["gpu", "cpu"]:
    rat = sorted(r["full grid|seconds"] / r["ours_s"] for r in rows if r["device"] == dev)
    print(f"  median full-grid/ours time ratio on {dev}: {rat[len(rat) // 2]:.0f} (n={len(rat)})")

print("== fast baseline: WSINDy + MSTLS on the 80 runs (results_fast/wsindy.json)")
recs = json.loads((R / "results_fast/wsindy.json").read_text())["records"]
print(f"  exact {sum(int(r['mstls']['exact']) for r in recs)}/{len(recs)}")

print("== sensitivity sweep (results_sensitivity/arms/*.json; 300 held-out dysts runs)")
arms = {p.stem: json.loads(p.read_text())["records"] for p in sorted((R / "results_sensitivity/arms").glob("*.json"))}
base = {k: v["metrics"]["exact"] for k, v in arms.pop("default").items()}
print(f"  default exact {rate(list(base.values())):.3f} (n={len(base)})")
worst = 0.0
for name, recs in arms.items():
    ex = {k: v["metrics"]["exact"] for k, v in recs.items()}
    diff = 100 * (rate(list(ex.values())) - rate(list(base.values())))
    worst = max(worst, abs(diff))
    w = sum(ex[k] > base[k] for k in base)
    l = sum(base[k] > ex[k] for k in base)
    print(f"  {name:15s} diff {diff:+.1f} pts W/L {w}/{l}")
print(f"  largest |diff| {worst:.1f} pts")

print("== timing: same-core runtime vs N (results_timing; Lorenz 10%, DEDi setting (1e-3, 1))")
sc = json.loads((R / "results_timing/scaling_ours_poly3.json").read_text())["records"]
dd = {json.loads(l)["index"]: json.loads(l)["seconds"] for l in (R / "results_timing/scaling_dedi.jsonl").read_text().splitlines()}
for i in sorted(dd):
    n = sc[i]["n"]
    ours = [r["arms"]["final_v12"]["seconds"] + r["starts_seconds"] + r["a_seconds"] for r in sc if r["n"] == n]
    print(f"  N={n:5d} ours {sum(ours) / len(ours):5.1f} s  DEDi {dd[i]:7.1f} s  ratio {dd[i] / (sum(ours) / len(ours)):.0f}")

print("== revision experiments (results_rev; run benchmarks/rev_report.py first for e2_rows.json, e3_rows.json)")
e2 = json.loads((R / "results_rev/e2_rows.json").read_text())
for col in ("full grid|val_error", "full grid|bic", "oracle"):
    w = sum(r["ours"] > r[col] for r in e2)
    l = sum(r["ours"] < r[col] for r in e2)
    print(f"  E2 {col:20s} ours {sum(r['ours'] for r in e2)}/{len(e2)} DEDi {sum(r[col] for r in e2)} W/L {w}/{l} p={sign_p(w, l):.2g}")
for r in json.loads((R / "results_rev/e3_rows.json").read_text()):
    m = sorted(r["ours"])[len(r["ours"]) // 2]
    print(f"  E3 {r['key']:10s} ours {m:.1f} s  CPU {r['cpu']:.0f} ({r['cpu'] / m:.0f}x)  GPU {r['gpu']:.0f} ({r['gpu'] / m:.0f}x)")
