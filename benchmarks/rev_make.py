"""Samples for the revision experiments (results_rev/PREREGISTRATION.md).

  python benchmarks/rev_make.py

order_dysts.json: 64 runs of the dysts test archive, 3-D systems new to us only, 8 per (noise, N) cell, order
round-robin over the 8 cells so that any prefix of complete rounds is balanced.
order_timing.json: the first three "white" runs of results_h2h/order.json.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20261030
NOISES = (0.01, 0.05, 0.1, 0.2)
NS = (500, 2000)
PER = 8
DATA = "data/dysts_test.npz"


def main():
    rng = np.random.default_rng(SEED)
    d = np.load(DATA, allow_pickle=True)
    sys_, n_, noise_ = d["sys"], d["n"], d["noise"]
    side = json.loads(Path("data/dysts_test.json").read_text())["systems"]
    new3 = sorted(k for k, v in side.items() if v.get("overlap") is None and int(v["D"]) == 3)
    slots = [new3[k % len(new3)] for k in rng.permutation(PER * len(NOISES) * len(NS))]
    cells, used, picks = [(e, n) for e in NOISES for n in NS], set(), []
    for k, system in enumerate(slots):
        eta, n = cells[k % len(cells)]
        ok = [i for i in range(len(sys_)) if str(sys_[i]) == system and int(n_[i]) == n
              and np.isclose(float(noise_[i]), eta) and i not in used]
        i = int(rng.choice(ok))
        used.add(i)
        picks.append({"cond": f"e{int(eta * 100)}n{n}", "data": DATA, "index": i, "system": system,
                      "noise": eta, "n": n})
    out = Path("results_rev")
    out.mkdir(exist_ok=True)
    (out / "order_dysts.json").write_text(json.dumps(picks, indent=1))
    h2h = json.loads(Path("results_h2h/order.json").read_text())
    (out / "order_timing.json").write_text(json.dumps([r for r in h2h if r["cond"] == "white"][:3], indent=1))
    print(len(new3), "eligible systems,", len(picks), "runs,", len({p["system"] for p in picks}), "distinct")


if __name__ == "__main__":
    main()
