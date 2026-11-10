"""Build the pre-registered head-to-head sample (ours v1.2 vs full-grid DEDi): results_h2h/order.json.

Five conditions x 16 runs, drawn at random (seed 20261027), each condition stratified over its systems and
split evenly over two noise levels. Unlike results_dedi_fullgrid/order.json this sample is NOT enriched for
disagreement runs. The order is round-robin over conditions (one run per condition per round), so any prefix of
complete rounds is balanced. See results_h2h/PREREGISTRATION.md.

  python benchmarks/h2h_make.py
"""
import json
from pathlib import Path

import numpy as np

SEED = 20261027
PER = 16
# name, archive, n, two noise levels
CONDS = [
    ("hard", "data/hard120.npz", 2000, (0.2, 0.2)),
    ("white", "data/joint_sealed.npz", 2000, (0.1, 0.2)),
    ("coloured", "data/sealed_colored2.npz", 2000, (0.1, 0.2)),
    ("t3", "data/sealed_student.npz", 2000, (0.1, 0.2)),
    ("short", "data/joint_sealed.npz", 500, (0.05, 0.1)),
]


def main():
    rng = np.random.default_rng(SEED)
    done = set(json.loads(Path("results_dedi_fullgrid/order.json").read_text())[:17])  # already analysed
    picks = {}
    for cond, path, n, levels in CONDS:
        d = np.load(path, allow_pickle=True)
        systems = sorted(set(map(str, d["sys"])))
        slots = [systems[k % len(systems)] for k in rng.permutation(PER)]  # balanced over systems
        noises = [levels[k % 2] for k in range(PER)]
        chosen = []
        for system, noise in zip(slots, noises):
            ok = [i for i in range(len(d["X"])) if str(d["sys"][i]) == system and int(d["n"][i]) == n
                  and np.isclose(float(d["noise"][i]), noise) and i not in chosen
                  and not (cond == "hard" and i in done)]
            chosen.append(int(rng.choice(ok)))
        picks[cond] = [{"cond": cond, "data": path, "index": i, "system": str(d["sys"][i]),
                        "noise": float(d["noise"][i]), "n": int(d["n"][i])} for i in chosen]
    order = [picks[c][k] for k in range(PER) for c, *_ in CONDS]
    out = Path("results_h2h")
    out.mkdir(exist_ok=True)
    (out / "order.json").write_text(json.dumps(order, indent=1))
    for c, *_ in CONDS:
        (out / f"idx_{c}.json").write_text(json.dumps([p["index"] for p in picks[c]]))
    print(len(order), "runs")
    for c, *_ in CONDS:
        print(c, sorted((p["system"], p["noise"]) for p in picks[c]))


if __name__ == "__main__":
    main()
