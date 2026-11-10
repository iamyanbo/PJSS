# Profiled Flow Discovery

This prototype searches for polynomial governing equations using the effect of
each term on a short trajectory. It is implemented in `eqdisc/flow_discovery.py` and
extends the existing weak-form work without changing the previous methods.

The research hypothesis is specific: **a term that is difficult to distinguish
in weak integral regression may become distinguishable through its propagated
effect on the other state variables.** Uncertain initial states must be removed
from that comparison, otherwise the algorithm can mistake an initial-state error
for evidence about an equation.

## Algorithm

Assume uniformly sampled, fully observed states

$$y_i=x(t_i)+\epsilon_i,\qquad \dot x=\Theta(x)c,$$

with independent, additive measurement noise. The current implementation uses
the repository's three-dimensional polynomial library, degree three by default.
It does not handle unobserved states, inputs, irregular sampling, PDEs or process
noise. The library must contain the desired equation.

1. Divide the data into disjoint 40-sample segments. Reserve every third segment
   for validation. Estimate coordinate scales and noise levels from training
   segments only. An incomplete final segment is omitted and reported.
2. Initialise with the existing idea of weak best-subset regression and a
   self-scaled chi-square size selector. Only weak windows entirely contained in
   training data are used. Retain two candidates per equation and model size.
3. Fit the seed's coefficients and an independent initial state for each
   training segment. Use a damped Gauss–Newton solver, analytic sensitivities,
   and RK4 integration. If the seed diverges, try short-horizon continuation
   with rescaled initial coefficient guesses, keeping its support fixed.
4. Integrate the sensitivities of **every** library coefficient, including
   inactive terms. Remove initial-state directions to form a new regression
   dictionary in observation space.
5. Search subsets in that dictionary, allowing other equations' active
   coefficients to readjust. Propose equation replacements and a bounded set
   of simultaneous replacements. Nonlinearly fit the eight best unvisited
   proposals. Repeat twice by default.
6. Evaluate the resulting candidate coefficients on validation segments,
   fitting only their initial states. Select using the explicit complexity
   score below; return alternative models and a descriptive ambiguity measure.
7. Refit the chosen support on all used observations. Selection diagnostics
   and alternative-model scores remain from the preceding split.

This is segmented output-error fitting. Independent starts deliberately relax
continuity across segments; it is **not** equivalent to a full-trajectory
likelihood or continuity-constrained multiple shooting. Validation fits are
checks of equation consistency with new segments, not forecasts from known
initial states: their starts are estimated from the entire validation segment.

## The distinctive regression dictionary

Let $F(c,z)$ be the stacked states predicted on one segment. At the current fit,
compute $S=\partial F/\partial c$ and $U=\partial F/\partial z$ by integrating

$$\dot S_{mj}=J_f S_{mj}+e_j\theta_m(x),\qquad
  \dot U=J_f U,\qquad S(0)=0,\quad U(0)=I.$$

Let $W$ divide observations by their estimated noise standard deviations. If
$Q$ spans the columns of $WU$, define $P=I-QQ^T$. Linearising the observations
and eliminating the unknown initial-state correction gives

$$A=PWS,\qquad b=PW[y-F(c_0,z_0)]+A c_0,\qquad Ac\approx b.$$

Stack these projected systems over segments. A column of $A$ represents a
term's effect over time and across all measured coordinates, after removing the
part explainable by changing the starting state. This is the proposed term
selection representation. The local information matrix is $A^TA$; the reported
normalised singular values describe conditioning of the selected support only.
They are not a global identifiability certificate.

For a fixed support, QR projection is the Schur complement of joint
Gauss–Newton fitting of coefficients and initial states. Projection is applied
without explicitly constructing a dense $P$. Exhaustive conditional searches
are bounded at 250,000 subsets; the complete search over all coupled equations
is approximate. It can miss the true support, even when that support would fit.

## Selection and ambiguity

For each candidate, the score is

$$D(S)=R_{\rm train}(S)+R_{\rm val}(S)+|S|\log M,$$

where $R$ is noise-standardised squared observation error, $M$ is the number of
used scalar observations, and coefficients are fit on training data. The common
number of segment-start parameters cancels. This is **BIC-style**, not exact BIC:
coefficients have not been maximised on the combined data, noise is estimated,
and supports were searched adaptively.

The selected candidate minimises $D$. Alternatives are marked `plausible` if
their excess score is no larger than `ambiguity_z` times the paired standard
error of the summed validation-block losses, relative to the selected model.
This heuristic does not account for every source of fitting or selection
uncertainty, and is not a confidence set or posterior probability. Finding one
plausible candidate does not prove that the law is uniquely identified. Setting
`ambiguity_z` changes these labels, not the selected equation.

Convergence flags are exposed for every nonlinear fit. Finite fits that exhaust
their iteration budget remain candidates, so a poor optimisation result may
affect selection. If no usable seed or validation model can be integrated, the
method raises an explicit error; the benchmark counts that run as a failure.

## Reproduction

Dependencies were verified with Python 3.13.13, NumPy 2.4.4, SciPy 1.17.1 and
Numba 0.65.0. Numba compiles the integrator on its first invocation.

```powershell
python -m pip install -r requirements-flow.txt

# Headerless CSV, one row per sample, columns x,y,z, in original units:
python archive/discover_flow.py observations.csv --dt 0.01 --output discovered.json

# Included example: existing Lorenz instance 43, 20% measurement noise.
python archive/discover_flow.py data/flow_example.npy --dt 0.01 --output archive/results_flow/example_lorenz.json

# Existing difficult cases; all ten seeds in each chosen cell.
python archive/flow_bench.py --output archive/results_flow/hard_cases.json

# Same optimiser and selection, but no term proposals from flow sensitivities.
python archive/flow_bench.py --noise .2 --weak-proposals-only --output archive/results_flow/weak_proposals_ablation.json

# Fresh noise and initial-condition draws after freezing the algorithm.
python benchmarks/flow_generate.py --seed 20261023
python archive/flow_bench.py --data data/flow_fresh.npz --output archive/results_flow/fresh_validation.json

python -m pytest -q tests/test_flow_discovery.py
```

The Python interface returns coefficients in original physical units:

```python
from flow_discovery import discover

result = discover(X, dt=0.01)
print(*result.equations(), sep="\n")
Xi = result.coefficients
alternatives = result.alternatives
```

If supplied, `sigma` must be per-coordinate measurement standard deviations in
**original units**. The default noise estimator assumes temporally independent
noise. Coloured and heavy-tailed noise have not been validated for this method.
Changing `segment_length`, `substeps`, `kmax` or the candidate budget can change
the result. The method is not tuning-free. At least nine segments are required.
Longer or stiffer segments may need more integration substeps, a shorter horizon,
or a different integrator.

The library is controlled by the existing `EQ_DEG` environment variable.
Higher-degree libraries can exceed the explicit exhaustive-search budget;
there is no hidden claim of scalability beyond that limit.

## Development benchmark

All comparisons below use exact recovery of **all terms in all three equations**,
not approximate coefficient matching. The baseline is the existing
`select_k.discover(..., criterion="chi2s")`, using all observations. The new
method receives no ground-truth coefficients, term counts or noise labels.
Its validation data participates in model selection, so validation scores are
not an untouched test metric. Ground-truth support is used only afterward to
score the synthetic experiment.

The existing difficult cases were used during development. An initial version
overselected tiny terms and failed to integrate some seeds; this motivated the
explicit complexity penalty, reuse of the established 20-sample weak half-width,
and seed continuation. Initial results are retained separately in
`archive/results_flow/hard_cases_initial.json`. The following are final development
results, not independent confirmation.

Exact successes out of ten, 2,000 samples per run:

| System | Noise | Existing weak selector | Profiled flow |
|---|---:|---:|---:|
| Lorenz | 5% | 9/10 | 8/10 |
| Lorenz | 20% | 0/10 | 7/10 |
| Chen | 5% | 10/10 | 9/10 |
| Chen | 20% | 0/10 | 9/10 |
| Rössler | 5% | 9/10 | 4/10 |
| Rössler | 20% | 0/10 | 2/10 |
| Rabinovich–Fabrikant | 5% | 0/10 | 0/10 |
| Rabinovich–Fabrikant | 20% | 0/10 | 0/10 |

Three of the final 80 flow runs failed numerically (all Rössler) and are included
in the denominators. At 20% noise, median coefficient error fell from 9.60% to
1.76% for Lorenz and from 13.26% to 1.82% for Chen. The remaining systems do not
justify a broad superiority claim. For Rabinovich–Fabrikant the true support was
never in the evaluated pool; this experiment cannot establish a data limit.

For the included Lorenz example, the identified equation is

$$\dot x=-9.7906x+9.8934y,$$
$$\dot y=27.896x-1.0272y-0.99751xz,$$
$$\dot z=-2.6221z+0.98264xy.$$

The complete per-run supports, coefficients, failures, convergence flags and
candidate scores are retained in the JSON files, rather than only averages.

## Fresh-data check and mechanism ablation

After the above development changes, the algorithm was frozen. The independent
generator seed `20261023` produced new attractor starting points and Gaussian
noise draws, with ten draws per system and 2,000 observations per draw. No
algorithm settings changed after viewing these results. These remain synthetic
replications on the same four systems, not tests of generalisation to unseen
physical systems or real measurements.

| System | Noise | Existing weak selector | Profiled flow |
|---|---:|---:|---:|
| Lorenz | 5% | 10/10 | 9/10 |
| Lorenz | 20% | 0/10 | 5/10 |
| Chen | 5% | 10/10 | 9/10 |
| Chen | 20% | 0/10 | 10/10 |
| Rössler | 5% | 9/10 | 5/10 |
| Rössler | 20% | 0/10 | 2/10 |
| Rabinovich–Fabrikant | 5% | 0/10 | 0/10 |
| Rabinovich–Fabrikant | 20% | 0/10 | 0/10 |

Seven of these 80 flow runs failed numerically, all Rössler, and count as
failures in the table. Across all four systems, exact recovery was 17/40 versus
0/40 at 20% noise, but 23/40 versus 29/40 at 5% noise. This supports a useful
high-noise research direction, not replacing the weak selector everywhere.
The fresh-data median coefficient errors at 20% noise were 4.51% versus 10.08%
for Lorenz and 1.58% versus 13.66% for Chen (flow versus weak).

A matched ablation on the original 40 high-noise cases retained the same flow
fitter, initial-state projection, screening, selection score, and budgets, but
limited term proposals to the weak-form candidate pool:

| System | Weak proposals + flow fitting | Full flow proposals |
|---|---:|---:|
| Lorenz | 0/10 | 7/10 |
| Chen | 1/10 | 9/10 |
| Rössler | 0/10 | 2/10 |
| Rabinovich–Fabrikant | 0/10 | 0/10 |

This is evidence that proposing new supports using the propagated
sensitivities matters within this implementation. It is not a comparison to
an independently implemented published shooting method, and the finite
proposal budget remains a confounder to examine in a broader study.

Runtime for the final 80-case development batch was 87.4 seconds with four
worker processes; the fresh batch took 83.1 seconds on this machine, including
the existing baseline evaluations. These are local wall-clock measurements,
not hardware-independent speed claims. Doubling RK4 substeps from two to four
for the included noisy Lorenz example preserved its support and changed its
coefficient vector by only $4.12\times10^{-7}$ in relative norm.

## Novelty and related work

This is a proposed research implementation, **not a verified literature-first
claim**. Sparse equation recovery, weak integrals, trajectory fitting,
Gauss–Newton sensitivities, nuisance projection, information criteria and
validation are established ideas.

- [WSINDy](https://arxiv.org/abs/2005.04339) supplies the weak-form starting point.
- [Mangan et al.](https://arxiv.org/abs/1701.01773) already combine sparse candidate
  generation with information criteria. Shortlisting and comparing trajectories
  alone is not the proposed novelty.
- [Meissner and Glasner, revised 2026](https://arxiv.org/html/2406.06707v2)
  jointly infer states and equation parameters, with sparse pruning and model
  comparison. Joint fitting alone is not new.
- [JSINDy](https://arxiv.org/html/2511.18555v1) alternates state/parameter fitting
  with sparsification on an estimated trajectory. In contrast, this prototype's
  proposals use the propagated coefficient sensitivities with initial-state
  directions removed, followed by nonlinear verification.
- [Gallo et al.](https://arxiv.org/html/2607.18490v1) study attractor geometry and
  identifiability. The conditioning discussion here does not establish a new
  identifiability theorem or a recovery guarantee.

The narrow contribution to investigate is **support replacement using a
nuisance-projected flow dictionary**, rather than running a sparse selector
again on instantaneous or weak library values. The matched weak-proposals
ablation tests whether that extra mechanism is useful in this implementation.
A publishable novelty claim would require a broader review of output-error
structure selection, sparse nonlinear least squares, and variable projection,
plus comparisons to independently implemented modern baselines.

## Numerical verification

The 13 tests check coefficient and initial-state sensitivities against finite
differences, integration against an independent adaptive solver, projection
against a full joint least-squares solve, parameter recovery from independently
simulated clean trajectories, frozen validation coefficients, exclusion of
validation observations from weak initialisation, zero dynamics, divergent
models, invalid observations, and an end-to-end noisy recovery with strict JSON
serialization. These establish numerical consistency, not statistical coverage
or universal recovery.
