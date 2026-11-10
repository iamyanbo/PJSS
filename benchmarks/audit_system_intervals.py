"""Reproduce system-cluster percentile intervals from saved paper results.

Example (from the repository root):
    python benchmarks/audit_system_intervals.py --output results_submission_audit/system_intervals.json

This is a post-hoc reporting audit; it does not rerun discovery or recover the
undocumented seed of the historical bootstrap. Each analysis resets an independent
NumPy Generator to the same explicit seed. Systems are sorted lexicographically,
sampled uniformly with replacement, and retained with all their recorded runs.
The statistic is the pooled run rate (or paired rate difference), not the equally
weighted mean of system rates. Both E2 comparisons consequently use identical
cluster draws. The output path must be new; an existing file is never overwritten.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "dysts_results": "results_dysts/test_final.json",
    "dysts_metadata": "data/dysts_test.json",
    "e2_results": "results_rev/e2_rows.json",
}


def read_inputs():
    documents, provenance = {}, {}
    for name, relative in INPUTS.items():
        raw = (ROOT / relative).read_bytes()
        documents[name] = json.loads(raw)
        provenance[name] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    return documents, provenance


def binary(value, label):
    if value not in (0, 1):
        raise ValueError(f"Expected binary exact-recovery value at {label}: {value!r}")
    return int(value)


def cluster_interval(rows, value, *, seed, replicates, fields, description):
    """Bootstrap sum(value)/number of runs, keeping each system as a cluster."""
    labels = sorted({row["system"] for row in rows})
    if not labels:
        raise ValueError("Cannot bootstrap an empty set of systems")
    cluster_id = {label: i for i, label in enumerate(labels)}
    numerator = np.zeros(len(labels), dtype=np.int64)
    denominator = np.zeros(len(labels), dtype=np.int64)
    for row in rows:
        i = cluster_id[row["system"]]
        numerator[i] += value(row)
        denominator[i] += 1
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    # Chunking limits peak memory; the flattened draw sequence is identical to
    # rng.integers(..., size=(replicates, len(labels))) in one call.
    for start in range(0, replicates, 2000):
        stop = min(start + 2000, replicates)
        indices = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        estimates[start:stop] = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "description": description,
        "fields": fields,
        "runs": len(rows),
        "systems": len(labels),
        "cluster_size_frequencies": {
            str(size): count for size, count in sorted(Counter(denominator.tolist()).items())
        },
        "numerator": int(numerator.sum()),
        "denominator": int(denominator.sum()),
        "estimate": float(numerator.sum() / denominator.sum()),
        "ci95": [float(low), float(high)],
        "ci95_rounded_3dp": [round(float(low), 3), round(float(high), 3)],
        "clusters": [
            {"system": label, "numerator": int(numerator[i]), "runs": int(denominator[i])}
            for i, label in enumerate(labels)
        ],
    }


def audit(seed, replicates):
    documents, provenance = read_inputs()
    dysts = documents["dysts_results"]["records"]
    metadata = documents["dysts_metadata"]["systems"]
    e2 = documents["e2_results"]
    if len({row["index"] for row in dysts}) != len(dysts):
        raise ValueError("Duplicate dysts result indices")
    if any(row.get("status") != "ok" for row in dysts):
        raise ValueError("Unexpected non-ok dysts row; choose an explicit failure policy before bootstrapping")
    if any(row["system"] not in metadata for row in dysts):
        raise ValueError("Missing dysts system metadata")
    if len({row["key"] for row in e2}) != len(e2):
        raise ValueError("Duplicate E2 result keys")
    new = [row for row in dysts if metadata[row["system"]].get("overlap") is None]
    for row in e2:
        for key in ("ours", "full grid|val_error", "full grid|bic"):
            binary(row[key], f"{row['key']}.{key}")

    def exact(row):
        return binary(row["final"]["exact"], f"dysts index {row['index']}.final.exact")

    intervals = {}
    for name, rows in (("dysts_all", dysts), ("dysts_new", new)):
        intervals[name] = cluster_interval(
            rows, exact, seed=seed, replicates=replicates,
            fields={"system": "records[*].system", "exact": "records[*].final.exact"},
            description="PJSS exact-support rate" + ("; metadata overlap is null/absent" if name == "dysts_new" else ""),
        )
    for name, column in (("e2_validation", "full grid|val_error"), ("e2_bic", "full grid|bic")):
        intervals[name] = cluster_interval(
            e2, lambda row, col=column: int(row[col]) - int(row["ours"]),
            seed=seed, replicates=replicates,
            fields={"system": "[*].system", "dedi": column, "pjss": "ours"},
            description="Paired exact-support rate difference, DEDi minus PJSS",
        )
        intervals[name]["pjss_exact"] = sum(int(row["ours"]) for row in e2)
        intervals[name]["dedi_exact"] = sum(int(row[column]) for row in e2)

    return {
        "audit": "post-hoc system-cluster bootstrap of stored paper results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "benchmarks/audit_system_intervals.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs": provenance,
        "procedure": {
            "replicates": replicates,
            "seed": seed,
            "rng": "numpy.random.default_rng (PCG64)",
            "seed_convention": "Reset an independent Generator to the same seed for every analysis; E2 rules share identical cluster draws.",
            "cluster_order": "Lexicographically sorted system labels",
            "resampling": "Sample G systems uniformly with replacement from the G observed systems; include all runs of each sampled system, including multiplicities.",
            "statistic": "Sum of exact-recovery indicators, or paired DEDi-minus-PJSS indicators, divided by total resampled run count.",
            "estimand_note": "Pooled run rate, not mean of per-system rates; total resampled run count can vary for E2.",
            "interval": "Two-sided 95% percentile interval; numpy.quantile at [0.025,0.975], method='linear'.",
            "historical_note": "Historical bootstrap seeds and executable procedures were not located; these are newly specified reproducible intervals, not exact replay of historical endpoints.",
        },
        "intervals": intervals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--replicates", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output path; refuses overwrite")
    args = parser.parse_args()
    if args.seed < 0 or args.replicates < 1:
        parser.error("seed must be nonnegative and replicates must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    result = audit(args.seed, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent writer after the check.
    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    for name, item in result["intervals"].items():
        print(f"{name}: n={item['runs']}, G={item['systems']}, estimate={item['estimate']:.9f}, CI={item['ci95']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
