"""Compare re-run records (v1.0.1) with the stored frozen-v1.0 records, run for run.

    python benchmarks/v101_compare.py final  results_final/test_sealed.json results_v101/test_sealed.json
    python benchmarks/v101_compare.py dysts  results_dysts/test_final.json   results_v101/dysts_test_final.json
    python benchmarks/v101_compare.py odebench results_odebench/fit_ic0.json results_odebench_v101/fit_ic0.json
    python benchmarks/v101_compare.py all    (every pair listed in PAIRS below that exists; markdown summary)

Every stored output field is compared with exact equality (floats as stored in JSON, i.e. repr round trip;
NaN == NaN); wall-clock fields are ignored. A run "matches" only if every field matches. Runs that raised an
exception in the old file are listed separately (they are expected to differ: that is the bug fix).
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IGNORE = {"seconds", "total_seconds", "starts_seconds", "a_seconds", "xi"}  # xi absent in old final_bench files

# (kind, old, new, arm)
PAIRS = [
    ("final", "results_final/test_sealed.json", "results_v101/test_sealed.json", "final"),
    ("final", "results_final/test_confirm.json", "results_v101/test_confirm.json", "final"),
    ("final", "results_final/test_colored.json", "results_v101/test_colored.json", "final"),
    ("final", "results_final/test_student.json", "results_v101/test_student.json", "final"),
    ("final", "results_final/test_coarse.json", "results_v101/test_coarse.json", "final"),
    ("final", "results_final/test_heldout2.json", "results_v101/test_heldout2.json", "final"),
    ("final", "results_final/test_trig.json", "results_v101/test_trig.json", "final"),
    ("final", "results_final/test_mixedpoly.json", "results_v101/test_mixedpoly.json", "final"),
    ("final", "results_final/test_deg4.json", "results_v101/test_deg4.json", "final"),
    ("final", "results_final/test_deg5.json", "results_v101/test_deg5.json", "final"),
    ("final", "results_final/test_nd.json", "results_v101/test_nd.json", "final"),
    ("final", "results_hard120/ours_hard120.json", "results_v101/hard120_final.json", "final"),
    ("dysts", "results_dysts/test_final.json", "results_v101/dysts_test_final.json", None),
    ("odebench", "results_odebench/fit_ic0.json", "results_odebench_v101/fit_ic0.json", None),
    ("odebench", "results_odebench/fit_ic1.json", "results_odebench_v101/fit_ic1.json", None),
    ("odebench_fix", "results_odebench/fit_ic0_emptyfix.json", "results_odebench_v101/fit_ic0.json", None),
]


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _diff_fields(a, b, ignore=IGNORE):
    keys = {k for k in set(a) | set(b) if k.removeprefix("arm.") not in ignore}
    return sorted(k for k in keys if not _eq(a.get(k), b.get(k)))


def _load(p):
    return json.loads((REPO / p).read_text())["records"]


def compare(kind, old, new, arm=None):
    A, B = _load(old), _load(new)
    if kind == "final":
        key = lambda r: r["index"]
        flat = lambda r: {**{k: v for k, v in r.items() if k != "arms"}, **{f"arm.{k}": v for k, v in
                                                                             r["arms"][arm].items()}}
        old_err = lambda r: "error" in r["arms"][arm]
    elif kind == "dysts":
        key, flat, old_err = (lambda r: r["index"]), (lambda r: r), (lambda r: r.get("status") == "error")
    else:  # odebench records: one per (id, sigma, rep)
        key = lambda r: (r["id"], r["sigma"], r["rep"])
        flat, old_err = (lambda r: r), (lambda r: bool(r.get("error")))
    nb = {key(r): r for r in B}
    out = {"old": old, "new": new, "n_old": len(A), "n_new": len(B), "compared": 0, "identical": 0,
           "differ": [], "old_errors": 0, "old_errors_now_ok": 0, "new_errors": 0, "missing": 0}
    ign = IGNORE - ({"xi"} if kind in ("dysts", "odebench", "odebench_fix") else set())
    for r in A:
        k = key(r)
        if k not in nb:
            out["missing"] += 1
            continue
        s = nb[k]
        if old_err(r):
            out["old_errors"] += 1
            if not s.get("error") and s.get("status", "ok") == "ok" and "error" not in s.get("arms", {}).get(arm, {}):
                out["old_errors_now_ok"] += 1
            continue
        out["compared"] += 1
        d = _diff_fields(flat(r), flat(s), ign)
        if d:
            out["differ"].append((k, d))
        else:
            out["identical"] += 1
    out["new_errors"] = sum(1 for s in B if (s.get("error") or s.get("status") == "error"
                                              or (arm and "error" in s["arms"][arm])))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        kind, old, new = sys.argv[1:4]
        print(json.dumps(compare(kind, old, new, "final" if kind == "final" else None), indent=1, default=str))
        return
    print("| Old (frozen v1.0) | New (v1.0.1) | Runs compared | Identical | Differ | Old exceptions (now ok) | New exceptions |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    tot = [0, 0, 0]
    for kind, old, new, arm in PAIRS:
        if not (REPO / old).exists() or not (REPO / new).exists():
            continue
        o = compare(kind, old, new, arm)
        if kind != "odebench_fix":
            tot[0] += o["compared"]; tot[1] += o["identical"]; tot[2] += len(o["differ"])
        print(f"| `{old}` | `{new}` | {o['compared']} | {o['identical']} | {len(o['differ'])} | "
              f"{o['old_errors']} ({o['old_errors_now_ok']}) | {o['new_errors']} |"
              + (f" missing {o['missing']}" if o["missing"] else ""))
        for k, d in o["differ"][:10]:
            print(f"|   differs: {k} {d} | | | | | | |")
    print(f"\nTotal (excluding the diagnostic emptyfix pair): {tot[0]} runs compared, {tot[1]} identical, "
          f"{tot[2]} differ")


if __name__ == "__main__":
    main()
