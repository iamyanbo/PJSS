PJSS CVIS extended abstract: four content pages plus cited references

This submission draft follows the presentation of the supplied UAV reference:
an architecture figure beneath the title, a column-width abstract, numbered
Introduction / Method / Experimental setup / Results and discussion / Conclusion
sections, table captions above the data and a two-column bibliography. It contains
four content pages and one references-only page (12 cited sources).

The prose follows the reference's explanatory flow as well as its layout:
task and motivation first, then each component's purpose, then its operation.
The paper defines the selected terms (support), explains profiling as allowing
the trajectory to adjust, and links each comparison to the question it tests.
The joint-fitting objective and profiled-score definition remain in the paper.
REPRODUCE.txt retains the full matrix derivation, calibration formulas, solver
constants, search settings and baseline grids in a numerical recipe section.

Compile from this folder:
  pdflatex -interaction=nonstopmode -halt-on-error pjss-extended-abstract.tex
  pdflatex -interaction=nonstopmode -halt-on-error pjss-extended-abstract.tex
Or run python docs/cvis2026/build.py from the repository root.

The current CVIS Overleaf template still needs verification before submission.
CVIS-requirements.txt distinguishes official rules from editorial structure.
opening-abstract.txt is the current opening paragraph for form entry.

The opening explains the cost bottleneck and the fit / screen / refit procedure
before presenting equations. A first-page architecture diagram shows the input,
initial joint fit, cheap profiled screening, shortlisted joint refits and final
sparse model, with the feedback loop and two-round limit. Blue and orange
distinguish cheap screening from nonlinear fitting. A recorded noisy Lorenz trace
and its actual discovered equations illustrate the input and output. The script
reads cached coefficients and rounds them for display. Editable SVG and vector
PDF versions are included, alongside the figure-generation script.
Runtime is a main result: 14.4 s vs 5,930 s under
the reported execution setups, with a 411-fold ratio of medians. The reproduction
guide also preserves the distinct 343-fold median paired ratio on the 73
GPU-allocated DEDi runs. Operation counts
show 21 PJSS joint fits (median over 80 runs) and a 49-fold median paired fit-count
ratio in the ten-case full-grid DEDi study. These statistics are verified from
raw timing and instrumentation records, which are included in the source ZIP.

The methodology explains fitting a trajectory and equation, screening alternative
equations and selecting the final model. The paper retains the core objective,
search budget and selection criterion. The companion guide supplies the detailed
state elimination, calibration, AR(1) path, seeds and baseline grids.

Comparisons include STLSQ, SR3, E-SINDy, PySR, ODEFormer, WSINDy, ARGOS and DEDi,
with PJSS paired on the same inputs. DEDi uses the full 35-setting grid in both
the 80-run core and 64-run new-system studies. Mixed findings are retained.
Ablations separate support search from state profiling. Historical weak-domain
PySINDy baseline rows are excluded because their feature seeds were not fixed.

Five figures show the architecture, sparse recovery, all 77 system rates, three
held-out forecasts and nine attractors (including two support failures). The sparse
recovery matrix places all 20 candidate terms in columns and the nine fitted
equations in rows, retaining absent terms. Fresh examples and historical benchmark
records remain separate. No benchmark fitting was repeated
for this editorial revision; all displayed comparison counts were re-summed.

Page four pairs the plots with written analysis: exact support versus forecast
error, the seven completed exact-support reconstructions and two escaping
failures, limits of visual agreement and priorities for further evaluation.
The forecast panels retain all three examples and error curves; a five-plus-four
attractor layout retains all nine systems while allowing more discussion.

REPRODUCE.txt is the detailed guide, including exact table regeneration, fresh-fit
commands, record indices, package information, archive checksums and limits.
The source ZIP contains solver code, runners, historical records, figure caches
and run lists. The data ZIP supplies the original observation archives separately.
The historical external DEDi checkout, banded/Schur solver shim and complete
historical environment lock must still be archived for exact refitting.

To regenerate the deliverables from the repository root:
  python docs/cvis2026/make_comparisons.py
  python docs/cvis2026/make_illustrations.py
  python docs/cvis2026/make_gallery.py
  python docs/cvis2026/make_architecture.py
  python docs/cvis2026/build.py
The figure scripts reuse saved illustration fits. See the guide before refitting.
