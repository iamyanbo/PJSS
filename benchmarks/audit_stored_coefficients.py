"""Audit saved dysts coefficients and baseline metric records without rerunning methods.

python benchmarks/audit_stored_coefficients.py --output results_submission_audit/stored_coefficients.json

Inputs are read-only. The output is created exclusively and is never overwritten.
Truth is the archived generator sidecar, not an independent rederivation of each ODE.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PJSS_FILES = [
    "results_dysts/test_final.json", "results_dysts/test_c.json",
    "results_dysts/test_weak.json", "results_v12refresh/dysts_v12.json",
]
PAIRED = [
    ("dysts7_fd_stlsq_raw_w11", "cv"), ("dysts7_fd_sr3_w11", "cv"),
    ("dysts7_weak_stlsq_h10", "cv"), ("dysts7_weak_sr3_h10", "cv"),
    ("dysts7_weak_miosr_h10", "cv"), ("dysts7_weak_miosr_h10", "oracle"),
    ("dysts21_fd_esindy_raw_w11", "cv"), ("dysts21_weak_esindy_h10", "cv"),
    ("dysts49_pysr", "best"), ("dysts21_odeformer", "best"),
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def audit_file(path, side, meta, full_archive):
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    indices = [r["index"] for r in records]
    stride = 1 if full_archive else int(path.stem.split("_")[0][5:])
    expected = set(range(0, len(meta["sys"]), stride))
    issues, metric_mismatches = [], []
    totals, checked, statuses = Counter(), Counter(), Counter()
    out_of_library = 0
    checked_exact = {}
    for record in records:
        i = record["index"]
        statuses[record.get("status", "absent")] += 1
        if not isinstance(i, int) or not 0 <= i < len(meta["sys"]):
            issues.append({"index": i, "issue": "invalid index"})
            continue
        for field, key in (("system", "sys"), ("noise", "noise"), ("n", "n")):
            if record[field] != meta[key][i]:
                issues.append({"index": i, "issue": "metadata", "field": field})
        truth = np.asarray(side[record["system"]]["xi"], dtype=float)
        if record["D"] != truth.shape[1]:
            issues.append({"index": i, "issue": "dimension mismatch"})
        if record.get("status") != "ok":
            continue
        pairs = [("xi", "final")] if full_archive else [
            ("aic_xi", "aic"), ("cv_xi", "cv"), ("best_xi", "best")]
        for coefficient_key, metric_key in pairs:
            if coefficient_key not in record:
                continue
            x = np.asarray(record[coefficient_key], dtype=float)
            checked["coefficient_arrays"] += 1
            if x.shape != truth.shape or not np.isfinite(x).all():
                issues.append({"index": i, "issue": "invalid coefficients", "arm": metric_key})
                continue
            support, target = x != 0, truth != 0
            tp = int((support & target).sum())
            fp = int((support & ~target).sum())
            fn = int((~support & target).sum())
            exact = int(np.array_equal(support, target))
            stored = record[metric_key]
            # The matrix holds only in-library polynomial terms. An out-of-library
            # expression is non-exact even when its retained polynomial part matches.
            if stored.get("out_of_library", False):
                out_of_library += 1
                exact = 0
            computed = {
                "exact": exact, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "fp": fp, "fn": fn,
                "coef_err": float(np.linalg.norm(x - truth) / np.linalg.norm(truth)),
            }
            for field, value in computed.items():
                if field in stored:
                    checked["metric_values"] += 1
                    if not np.isclose(value, stored[field], rtol=1e-10, atol=1e-12):
                        metric_mismatches.append({"index": i, "arm": metric_key,
                                                  "field": field, "computed": value,
                                                  "stored": stored[field]})
            totals[metric_key] += exact
            if full_archive:
                checked_exact[i] = exact
        if "exact_path" in record:
            checked["threshold_paths"] += 1
            path_exact = np.asarray(record["exact_path"])
            eq_exact = np.asarray(record["eq_exact_path"])
            if not np.array_equal(path_exact, eq_exact.all(axis=1)):
                issues.append({"index": i, "issue": "path vs equation flags"})
            if record["oracle"] != path_exact.max():
                issues.append({"index": i, "issue": "oracle vs path"})
            for rule in ("cv", "aic"):
                if record[rule]["exact"] != path_exact[record[rule + "_idx"]]:
                    issues.append({"index": i, "issue": "selected vs path", "arm": rule})
        if "eq_exact_k" in record:
            checked["miosr_paths"] += 1
            eq = record["eq_exact_k"]
            any_k = int(all(any(row) for row in eq))
            true_sizes = (truth != 0).sum(axis=0)
            true_k = int(all(0 < k <= len(eq[j]) and eq[j][k - 1]
                             for j, k in enumerate(true_sizes)))
            totals["miosr_true_k"] += true_k
            if any_k != record["oracle"] or true_k != record["oracle"]:
                issues.append({"index": i, "issue": "MIOSR oracle vs true k"})
    result = {
        "file": path.relative_to(ROOT).as_posix(), "records": len(records),
        "duplicate_indices": sorted(k for k, v in Counter(indices).items() if v != 1),
        "missing_indices": sorted(expected - set(indices)),
        "unexpected_indices": sorted(set(indices) - expected), "statuses": dict(statuses),
        "checked": dict(checked), "exact_counts": dict(totals),
        "out_of_library_selected_arrays": out_of_library,
        "issues": issues, "metric_mismatches": metric_mismatches,
    }
    if full_archive:
        result["groups"] = {}
        for label, condition in [
            ("all", lambda r: True),
            ("new", lambda r: side[r["system"]].get("overlap") is None),
            ("D3", lambda r: r["D"] == 3), ("D4plus", lambda r: r["D"] > 3),
        ]:
            group = [r for r in records if condition(r)]
            result["groups"][label] = {
                "runs": len(group), "exact_from_coefficients": sum(checked_exact.get(r["index"], 0) for r in group),
                "systems": len(set(r["system"] for r in group)),
            }
        result["runs_per_system_distribution"] = dict(Counter(Counter(r["system"] for r in records).values()))
        result["cells"] = {f"noise={noise},n={n}": count for (noise, n), count in
                           sorted(Counter((r["noise"], r["n"]) for r in records).items())}
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    inputs = [ROOT / "data/dysts_test.json", ROOT / "data/dysts_test.npz"]
    files = [ROOT / name for name in PJSS_FILES] + sorted((ROOT / "results_baselines").glob("dysts*.json"))
    inputs += files
    side = json.loads(inputs[0].read_text(encoding="utf-8"))["systems"]
    with np.load(inputs[1], allow_pickle=False) as archive:
        meta = {key: archive[key] for key in ("sys", "noise", "n", "dt")}
    results, by_file = [], {}
    for path in files:
        result, records = audit_file(path, side, meta, path.relative_to(ROOT).as_posix() in PJSS_FILES)
        results.append(result)
        by_file[path.stem] = {r["index"]: r for r in records}
    ours = by_file["test_final"]
    paired = []
    for filename, rule in PAIRED:
        records = list(by_file[filename].values())
        baseline = [r[rule] if rule == "oracle" else r[rule]["exact"] for r in records]
        own = [ours[r["index"]]["final"]["exact"] for r in records]
        paired.append({"file": filename, "rule": rule, "runs": len(records),
                       "ours_exact": int(sum(own)), "baseline_exact": int(sum(baseline)),
                       "ours_only": sum(a > b for a, b in zip(own, baseline)),
                       "baseline_only": sum(a < b for a, b in zip(own, baseline))})
    fresh = by_file["dysts_v12"]
    changed = {"v12_only_exact": [], "original_only_exact": []}
    for i in ours.keys() & fresh.keys():
        a, b = fresh[i]["final"]["exact"], ours[i]["final"]["exact"]
        if a > b: changed["v12_only_exact"].append(i)
        if b > a: changed["original_only_exact"].append(i)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script": digest(Path(__file__)),
        "inputs": {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs},
        "files": results, "paired_headlines": paired, "v12_vs_original": changed,
        "coefficient_arrays_checked": sum(r["checked"].get("coefficient_arrays", 0) for r in results),
        "metric_mismatches": sum(len(r["metric_mismatches"]) for r in results),
        "issues": sum(len(r["issues"]) + len(r["duplicate_indices"]) + len(r["missing_indices"]) + len(r["unexpected_indices"]) for r in results),
        "evidence_boundaries": [
            "No numerical method was rerun. This audits saved coefficients, metadata and derived scores.",
            "Truth is the archived dysts generator sidecar; its ODE derivation and original package behavior are not independently rederived here.",
            "Threshold-path and oracle coefficients were not saved; their flags are checked for internal consistency only.",
            "For symbolic regressors, the saved in-library coefficient matrix is checked; the saved out_of_library classification is trusted, not independently parsed from the expressions.",
            "The new-system grouping uses archived overlap labels, not an independent audit of literature or system-family independence.",
            "Input hashes identify files inspected now; they do not establish historical freeze dates or preregistration provenance.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("coefficient_arrays_checked", "metric_mismatches", "issues")}))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
