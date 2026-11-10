"""Refactored `eqdisc.discover(..., version="1.0.1")` must return bit-identical answers to the frozen v1.0
`final_model.discover` (v1.0.1 only adds the empty-support crash fix, which no archive run here triggers;
v1.1 changes the loss on AR(1)-routed runs by design and is not tested here).

The frozen flat modules are extracted from commit ac7ab20 (`git show`), checked against
results_final/frozen.md5 (up to CRLF/LF line endings), and imported from a temporary directory. Each sampled archive run goes through both
versions in the same spawned worker (one BLAS thread); coefficients are compared with np.array_equal, and the
gap, ambiguity flag, per-term confidences and chosen noise model must match too.

    python -m pytest tests/test_equivalence.py -s          # 42 runs, 3 workers (~10-20 min on 12 cores)
    EQDISC_EQUIV_WORKERS=2 python -m pytest tests -s

Needs the archives in data/ (gitignored; regenerate with the generators in benchmarks/, checksums in
data/MANIFEST.sha256) and git. On a machine without git, export the frozen files first
(`git archive ac7ab20 <FROZEN_FILES> results_final/frozen.md5 | tar -x -C DIR`) and set EQDISC_FROZEN_SRC=DIR;
the md5 check below still applies.
"""
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "ac7ab20"
FROZEN_FILES = ["final_model.py", "core.py", "subset.py", "select_k.py", "flow_discovery.py", "joint_discover.py",
                "joint_colored.py", "joint_biglib.py", "joint_nd.py", "joint_squeeze.py", "confidence.py",
                "results_final/confidence_isotonic.json"]
WORKERS = int(os.environ.get("EQDISC_EQUIV_WORKERS", 3))

# (archive, library, how many runs, EQ_DEG, filter); runs are drawn with a fixed seed
PLAN = [
    ("joint_sealed.npz", "poly3", 10, 3, None),      # sealed test archive, white noise
    ("sealed_colored.npz", "poly3", 8, 3, None),     # AR(1) rho 0.8 -> exercises the noise="auto" switch
    ("sealed_student.npz", "poly3", 3, 3, None),     # heavy-tailed t3
    ("sealed_coarse.npz", "poly3", 3, 3, None),      # every 4th sample
    ("trig_test.npz", "trig3", 4, 3, None),          # Thomas (sin/cos terms)
    ("joint_sealed.npz", "trig3", 2, 3, None),       # polynomial systems with the trig library
    ("nd_test.npz", "auto", 5, 3, "small_nd"),       # D = 4-6 (joint_nd); D = 10 skipped for time
    ("joint_sealed.npz", "poly4", 2, 4, None),       # degree-4 library (EQ_DEG=4)
    ("joint_sealed.npz", "poly5", 5, 5, None),       # degree-5 library (EQ_DEG=5)
]


def _sample(archive, k, filt, seed):
    with np.load(REPO / "data" / archive, allow_pickle=True) as d:
        ok = [i for i in range(len(d["X"]))
              if filt != "small_nd" or (np.asarray(d["X"][i]).shape[1] <= 6 and int(d["n"][i]) == 500)]
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(ok, size=k, replace=False))


def _git_show(path):
    src = os.environ.get("EQDISC_FROZEN_SRC")
    if src:
        return (Path(src) / path).read_bytes()
    return subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{path}"], cwd=REPO, check=True,
                          capture_output=True).stdout


@pytest.fixture(scope="session")
def frozen_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("frozen_ac7ab20")
    md5 = {}
    for line in _git_show("results_final/frozen.md5").decode().splitlines():
        h, name = line.split()
        md5[name] = h
    for f in FROZEN_FILES:
        blob = _git_show(f)
        # frozen.md5 was computed on a copy where some files had CRLF line endings; the snapshot
        # commit stores them with LF, so accept the hash of the file as committed or with LF -> CRLF
        ok = (hashlib.md5(blob).hexdigest(), hashlib.md5(blob.replace(b"\n", b"\r\n")).hexdigest())
        assert md5[f] in ok, f"{f} at {FROZEN_COMMIT} does not match frozen.md5"
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_bytes(blob)
    return d


def test_calibration_map_unchanged():
    """The package ships a copy of the frozen isotonic map; it must be byte-identical to the original."""
    assert (REPO / "eqdisc" / "confidence_isotonic.json").read_bytes() == \
        (REPO / "results_final" / "confidence_isotonic.json").read_bytes()


def test_equivalence_on_archive_runs(frozen_dir):
    from _equiv_worker import run_one
    missing = sorted({a for a, *_ in PLAN if not (REPO / "data" / a).exists()})
    if missing:
        pytest.skip(f"archives not available: {missing}")
    by_deg = {}
    for seed, (archive, lib, k, deg, filt) in enumerate(PLAN):
        for i in _sample(archive, k, filt, 1000 + seed):
            by_deg.setdefault(deg, []).append((str(frozen_dir), str(REPO / "data" / archive), i, lib))
    results = []
    saved = {k: os.environ.get(k) for k in ("EQ_DEG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                            "PYTHONWARNINGS")}
    try:
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          PYTHONWARNINGS="ignore")
        for deg, jobs in sorted(by_deg.items()):
            os.environ["EQ_DEG"] = str(deg)  # inherited by the spawned workers before `core` is imported
            with ProcessPoolExecutor(min(WORKERS, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
                for r in ex.map(run_one, jobs):
                    results.append(r)
                    print(f"[{len(results)}] {Path(r['archive']).name}#{r['index']} {r['library']} {r['system']} "
                          f"eta={r['noise']} n={r['n']} D={r['D']} terms={r['n_terms']} noise={r['noise_model'][0]} "
                          f"equal={r['xi_equal']} maxdiff={r['max_abs_diff']:g}", flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    keys = ("xi_equal", "names_equal", "gap_equal", "ambiguous_equal", "confidence_equal")
    bad = [r for r in results if not all(r[k] for k in keys) or r["noise_model"][0] != r["noise_model"][1]]
    print(f"\nequivalence: {len(results)} runs, {len(bad)} differing, "
          f"max |xi_frozen - xi_new| = {max(r['max_abs_diff'] for r in results):g}")
    assert len(results) == sum(k for _, _, k, _, _ in PLAN)
    assert not bad, json.dumps(bad, indent=1, default=str)
