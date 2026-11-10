"""Huber vs Gaussian loss on DEV archives only (diagnostic follow-up of docs/reports/ablation.md, item 6).

    python benchmarks/huber_dev.py --set dev_colored --every 3 --offset 0 --workers 5 \
        --variants full gaussian ar1 ar1_gauss --output results_huber_dev/dev_colored_o0.json

Thin wrapper around benchmarks/ablation.py: the same run_variant / _job machinery (so "full" is the frozen
final model and "gaussian" is huber=inf in the joint fits AND the AR(1) pre-estimate), with the dev archives
registered as extra sets and two forced-noise-model arms added:

  full        frozen final model, noise="auto"
  gaussian    noise="auto", huber=inf
  ar1         noise="ar1" (AR(1) likelihood forced), Huber c = 2 (frozen default)
  ar1_gauss   noise="ar1", huber=inf
  white_gauss noise="white", huber=inf  (only for completeness on white / t3)

Nothing in eqdisc/ is edited; truth is used only for scoring. Archives cycle the noise level with index mod 4,
so only strides coprime with 4 are allowed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation  # noqa: E402

ablation.VARIANTS.update({
    "ar1": {"noise": "ar1"},
    "ar1_gauss": {"noise": "ar1", "huber": np.inf},
    "white_gauss": {"noise": "white", "huber": np.inf},
})
ablation.SETS.update({
    # dev archives (design data; never test)
    "dev_colored": dict(data="data/instances_colored.npz", starts=None, white=False),      # AR(1) rho = 0.8
    "dev_student": dict(data="data/instances_student.npz", starts=None, white=True),       # Student-t, 3 dof
    "dev_white": dict(data="data/instances.npz", starts="results_squeeze/starts_dev.pkl", white=True),
    "dev_cmid": dict(data="data/dev_colored_mid.npz", starts=None, white=False),           # rho = 0.3 / 0.55
})

if __name__ == "__main__":
    for i, a in enumerate(sys.argv):
        if a == "--every" and int(sys.argv[i + 1]) % 2 == 0:
            raise SystemExit("--every must be coprime with 4 (noise level cycles with index mod 4)")
    ablation.main()
