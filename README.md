# PJSS: Profiled Joint Subset Search

Sparse equation discovery from a single noisy trajectory. PJSS jointly estimates denoised states and sparse ODE coefficients, screens alternative supports using a profiled quadratic approximation, and nonlinearly refits the best proposals. The Python package is named `eqdisc` (version 1.2.0).

This is a public research-code snapshot, with substantial benchmark evidence, tests, manuscript sources and recorded audits. It is not a claim of peer-reviewed publication. The original local research workspace and its history are preserved separately.

## Start here

- [Full manuscript source](docs/paper/main.tex)
- [Detailed research README](RESEARCH_README.md): method, versions, commands, experiments and evidence boundaries
- [Recorded submission audits](results_submission_audit/)
- [Third-party notices and redistribution scope](THIRD_PARTY.md)

## Recorded evidence

The manuscript reports 89.0% exact support recovery across 6,160 simulated trajectories from 77 in-library chaotic systems, including 85.3% on 49 systems outside the development benchmarks. In a paired comparison on 64 runs from 35 new systems, PJSS recovers 53 versus 41 for DEDi with BIC selection. These are stored research results, not new experiments run for this release, and not a universal superiority claim.

The recorded audit checks 44,424 coefficient arrays against saved metrics and finds all 5,250 raw DEDi fits complete and consistent. All 43 available manifest archives matched their stored SHA-256 hashes at that audit. Unit, smoke, baseline-reproducibility and historical-equivalence tests are included. Main full-observation archives exist in the original research workspace.

### What this public snapshot includes

Current source, tests, benchmark runners, manuscript sources and figures, research reports, saved result JSON files and audit records are included. Large raw trajectory archives, downloaded third-party datasets, scratch files, environment directories and original Git history are not included. This repository therefore supports inspection and stored-result verification, but is not yet a complete downloadable data release. Checksums do not regenerate archives with unrecorded original seeds.

A narrow historical issue affects some weak-baseline tables, not the main PJSS/DEDi comparison: random weak domains were originally constructed before seeding. The runner is corrected, and the completed 14-run order-independence diagnostic passes. Historical tables remain historical, rather than being relabeled as fresh corrected runs. See [the audit details](RESEARCH_README.md#submission-audit).

Four secondary-study original archives are missing locally, and the exact original benchmark software environment has not been recovered. Positive results concern uniformly sampled, fully observed, in-library simulated dynamics; measured-data and model-misspecification tests include negative results. These boundaries are documented in the manuscripts and do not imply that the project lacks testing.

## Install and try

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_flow_discovery.py -q
```

```python
import numpy as np
from eqdisc import discover
from eqdisc.core import SYSTEMS, X0, DT, simulate

clean = simulate(SYSTEMS["lorenz"], X0["lorenz"], DT["lorenz"], 2000, burn=1000)
y = clean + 0.05 * clean.std(axis=0) * np.random.default_rng(0).standard_normal(clean.shape)
result = discover(y, DT["lorenz"])
print(result.equations())
```

The example generates a synthetic trajectory locally. Broader benchmark and archive-dependent tests need the corresponding data and optional dependencies described in the research README. The optional WSINDy port is excluded pending redistribution permission; see THIRD_PARTY.md.

## Reproduction notes

`python docs/paper/scripts/verify_numbers.py` reads saved results, rather than rerunning the algorithms. Historical-equivalence tests normally read commit ac7ab20. Because this is a clean public snapshot, its required source files are included under `reference/frozen_ac7ab20`; set `EQDISC_FROZEN_SRC` to that directory and provide the required archives. Do not confuse this reference snapshot or a current dependency freeze with the original benchmark environment.

## AI development disclosure

Generative AI tools were used extensively in research and experiment design, code development, and manuscript drafting and revision. The manuscript records that the author assumes responsibility for all content. AI assistance is disclosed as part of the development process, not as a substitute for reproducible evidence or independent review.

## Publication and reuse

This release makes the research inspectable without claiming that every historical experiment is independently reproducible from this download alone. No new open-source license has been assigned to original PJSS materials. See THIRD_PARTY.md for the included third-party MIT notice and the intentionally omitted optional port.
