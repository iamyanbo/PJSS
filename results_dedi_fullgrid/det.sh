#!/bin/bash
# determinism check: index 0, pair (1e-2,1e-2), twice on a slow core (10, 11) and twice on fast cores (15, 16)
cd ~/eqdisc_v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=~/eqdisc_v3
cat > /tmp/dfg_det.py <<'PY'
import sys, os, json, time, numpy as np
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
sys.path[:0] = [os.path.expanduser("~/eqdisc_dedi/shim"), os.path.expanduser("~/eqdisc_dedi/ext/DEDi")]
sys.path.insert(0, "benchmarks")
from dedi_fullgrid import term
from DEDi import ode_find
from eqdisc.core import SYSTEMS, EXP_ARR, metrics
import contextlib, io
d = np.load("data/hard120.npz", allow_pickle=True); i = 0
x = np.asarray(d["X"][i], float); dt = float(d["dt"][i])
vm = np.ones_like(x, dtype=bool); vm[2::3] = False
with contextlib.redirect_stdout(io.StringIO()):
    m = ode_find(x, [term(e) for e in EXP_ARR], [0.0, dt * (len(x) - 1)], validation_mask=vm)
    m.FindBestModel([1e-2], [1e-2])
xi = np.array(m.best_model()[0], float)
print(json.dumps({"tag": sys.argv[1], "val": float(m.tested_models[0][3]), "m": metrics(xi, SYSTEMS["lorenz"]), "xi_sum": float(np.abs(xi).sum())}))
PY
for c in 10 11 15 16; do taskset -c $c ~/eqdisc_dedi/.venv/bin/python /tmp/dfg_det.py core$c >> results_dedi_fullgrid/determinism.log 2>/dev/null & done
wait; echo DONE >> results_dedi_fullgrid/determinism.log
