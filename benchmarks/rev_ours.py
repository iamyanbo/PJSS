"""Our method (eqdisc.discover defaults) on an order file; one process, for pinning with taskset.

  python benchmarks/rev_ours.py --order results_rev/order_dysts.json --output results_rev/ours_dysts.json
                                [--truth-json data/dysts_test.json] [--repeats 1]
"""
import argparse
import json
import os
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

from eqdisc import discover
from eqdisc.core import SYSTEMS, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--truth-json", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    truth = dict(SYSTEMS)
    if a.truth_json:
        side = json.loads(Path(a.truth_json).read_text())["systems"]
        truth = {k: np.asarray(v["xi"], float) for k, v in side.items()}
    out = []
    for run in json.loads(Path(a.order).read_text()):
        d = np.load(run["data"], allow_pickle=True)
        i = int(run["index"])
        x, dt, system = np.asarray(d["X"][i], float), float(d["dt"][i]), str(d["sys"][i])
        for rep in range(a.repeats):
            t = time.perf_counter()
            res = discover(x, dt)
            rec = dict(run, key=f"{run['cond']}_{i}", rep=rep, seconds=time.perf_counter() - t)
            rec.update(metrics(res.xi, truth[system]), ambiguous=bool(res.ambiguous))
            out.append(rec)
            print(rec["key"], rep, rec["exact"], round(rec["seconds"], 1), flush=True)
    Path(a.output).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
