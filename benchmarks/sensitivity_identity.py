"""Bit-identity check for the sensitivity-sweep plumbing (run BEFORE the sweep).

Making the inner constants overridable (final_model.discover huber_white= / sg_window=, rho passed on to the AR(1)
pre-estimate, joint_discover / joint_nd SG_WINDOW) must not change the default answer. On a fixed set of runs this
compares, with np.array_equal on xi (and equality of noise model, gap, ambiguity flag):
  ref       the committed code (git archive <ref> eqdisc), default call
  new       the working-tree code, default call
  explicit  the working-tree code with every swept constant passed explicitly at its frozen value
            (degw .2, centre 1.5, ar_threshold .6, pen_exp 1/3, huber_white 2, rho .03, kmax 5, finalists 10,
            sg_window 21)
Each (version, run) is one subprocess, one BLAS thread.

    python benchmarks/sensitivity_identity.py --ref HEAD --output results_sensitivity/identity_check.json
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# (archive, index): 6 dysts new-system runs (4 D = 3, 2 D = 4 -> joint_nd), n = 500, plus 2 sealed coloured runs
# (AR(1) rho .8 -> exercises the noise="auto" AR(1) route), plus 2 dysts n = 2000 runs. Fixed by --seed.
EXPLICIT = dict(degw=0.2, centre=1.5, ar_threshold=0.6, pen_exp=1 / 3, huber_white=2.0, rho=0.03, kmax=5,
                finalists=10, sg_window=21)

WORKER = r'''
import json, os, sys
import numpy as np
from threadpoolctl import threadpool_limits
arch, i, mode, explicit = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
with np.load(arch, allow_pickle=True) as d:
    x, dt = np.asarray(d["X"][i], float), float(d["dt"][i])
from eqdisc import final_model as fm
kw = {}
if mode == "explicit":
    kw = dict(explicit)
    if x.shape[1] != 3:   # joint_nd path: AR(1) options do not exist there; centre/degw/huber/rho/kmax/... do
        kw.pop("ar_threshold"); kw.pop("pen_exp")
with threadpool_limits(limits=1):
    r = fm.discover(x, dt, **kw)
print(json.dumps({"xi": np.asarray(r.xi).tolist(), "gap": r.gap, "ambiguous": r.ambiguous,
                  "noise_model": r.info.get("noise_model"), "file": fm.__file__}))
'''


def pick(seed):
    rng = np.random.default_rng(seed)
    side = json.loads((REPO / "data/dysts_test.json").read_text())["systems"]
    with np.load(REPO / "data/dysts_test.npz", allow_pickle=True) as d:
        X, sy, nn = d["X"], d["sys"], d["n"]  # load once (npz re-reads on every d[...] access)
    new = [i for i in range(len(X)) if side[str(sy[i])]["overlap"] in (None, "None")]
    d3s = [i for i in new if X[i].shape[1] == 3 and int(nn[i]) == 500]
    d4s = [i for i in new if X[i].shape[1] == 4 and int(nn[i]) == 500]
    l3 = [i for i in new if X[i].shape[1] == 3 and int(nn[i]) == 2000]
    with np.load(REPO / "data/sealed_colored.npz", allow_pickle=True) as d:
        noise = d["noise"]
    col = [i for i in range(len(noise)) if float(noise[i]) >= 0.1]
    runs = [("data/dysts_test.npz", int(i)) for i in rng.choice(d3s, 4, replace=False)]
    runs += [("data/dysts_test.npz", int(i)) for i in rng.choice(d4s, 2, replace=False)]
    runs += [("data/dysts_test.npz", int(i)) for i in rng.choice(l3, 2, replace=False)]
    runs += [("data/sealed_colored.npz", int(i)) for i in rng.choice(col, 2, replace=False)]
    return runs


def run(src, arch, i, mode):
    env = dict(os.environ, PYTHONPATH=str(src), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", WORKER, str(REPO / arch), str(i), mode, json.dumps(EXPLICIT)],
                         env=env, cwd=src, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="eqdisc_ref_"))
    tar = tmp / "ref.tar"
    subprocess.run(["git", "archive", "-o", str(tar), a.ref, "eqdisc"], cwd=REPO, check=True)
    with tarfile.open(tar) as t:
        t.extractall(tmp / "src", filter="data")
    recs, ok = [], True
    for arch, i in pick(a.seed):
        r = {m: run(tmp / "src" if m == "ref" else REPO, arch, i, "explicit" if m == "explicit" else "default")
             for m in ("ref", "new", "explicit")}
        assert "eqdisc_ref_" in r["ref"]["file"] and "eqdisc_ref_" not in r["new"]["file"], (r["ref"]["file"], r["new"]["file"])
        same = {m: bool(np.array_equal(np.array(r["ref"]["xi"]), np.array(r[m]["xi"]))
                        and r["ref"]["gap"] == r[m]["gap"] and r["ref"]["ambiguous"] == r[m]["ambiguous"]
                        and r["ref"]["noise_model"] == r[m]["noise_model"]) for m in ("new", "explicit")}
        ok &= all(same.values())
        recs.append({"archive": arch, "index": i, "noise_model": r["ref"]["noise_model"], "identical": same,
                     "support_size": int(np.count_nonzero(r["ref"]["xi"]))})
        print(recs[-1], flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps({"ref": a.ref, "explicit": EXPLICIT, "all_identical": bool(ok),
                                          "runs": recs}, indent=1))
    print("ALL IDENTICAL" if ok else "MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
