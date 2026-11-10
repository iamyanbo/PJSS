"""Compare dedi_gpu_bench.py JSONL records with the stored full-grid CPU pairs (results_dedi_fullgrid/main/pairs).

usage (from the repo root): python benchmarks/dedi_gpu_compare.py results_dedi_gpu/gpu1.jsonl [...]
"""
import json, sys, glob, numpy as np
ref = {}
for f in glob.glob("results_dedi_fullgrid/main/pairs/*.json"):
    r = json.load(open(f)); ref[(r["index"], r["lam"], r["R"])] = r
rows = [json.loads(l) for fn in sys.argv[1:] for l in open(fn)]
for r in rows:
    s = ref.get((r["index"], r["lam"], r["R"]))
    if s is None or "xi" not in s: print("no ref", r["index"], r["lam"], r["R"]); continue
    a, b = np.array(r["xi"]), np.array(s["xi"])
    same_sup = np.array_equal(a != 0, b != 0)
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    print(f'{r["tag"]:>8} {r["device"]} idx={r["index"]:3d} lam={r["lam"]:g} R={r["R"]:g} t={r["seconds"]:7.1f}s (ref {s["seconds"]:6.0f}s w{s["worker"]}) shim={r["shim_seconds"]:6.1f} '
          f'support_same={same_sup} nz={r["nonzero"]}/{s["nonzero"]} max_rel_coef_diff={rel:.2e} '
          f'val={r["val_error"]:.10e}/{s["val_error"]:.10e} bitident={a.tolist()==b.tolist() and r["val_error"]==s["val_error"]} exact={r["dedi"]["exact"]}/{s["dedi"]["exact"]}')
