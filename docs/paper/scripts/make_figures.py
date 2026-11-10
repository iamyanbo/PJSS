"""Paper figures from existing result files (no benchmark is re-run).
Single process, light CPU. Run from the repository root:
    python docs/paper/scripts/make_figures.py
Writes docs/paper/figures/*.pdf
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[3]
OUT = R / "docs/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

# reference categorical palette (slots 1-3) + neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "pdf.fonttype": 42,
})


def load(p):
    return json.loads((R / p).read_text())["records"]


def rate(v):
    return sum(v) / len(v)


# ---- Figure 1: dysts exact recovery by noise level, N = 500 / 2000 ----
def fig_dysts_noise():
    methods = [("final", "Final model", BLUE, "o"), ("c", "Config C (no prior, no AR(1))", AQUA, "s"),
               ("weak", r"Weak best-subset ($\chi^2_s$)", ORANGE, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.0), sharey=True)
    noises = [0.01, 0.05, 0.1, 0.2]
    for ax, n in zip(axes, [500, 2000]):
        for m, lab, col, mk in methods:
            recs = load(f"results_dysts/test_{m}.json")
            by = defaultdict(list)
            for r in recs:
                if r["n"] == n:
                    by[r["noise"]].append(r["final"]["exact"])
            ax.plot([100 * s for s in noises], [rate(by[s]) for s in noises], marker=mk, color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel("noise (% of state std)")
        ax.set_title(f"N = {n}", color=INK)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("exact support recovery")
    axes[1].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dysts_noise.pdf")
    plt.close(fig)


# ---- Figure 2: dysts per-system exact rate (final model), sorted ----
def fig_dysts_systems():
    recs = load("results_dysts/test_final.json")
    weak = {(r["index"]): r["final"]["exact"] for r in load("results_dysts/test_weak.json")}
    by, byw, dim = defaultdict(list), defaultdict(list), {}
    for r in recs:
        by[r["system"]].append(r["final"]["exact"])
        byw[r["system"]].append(weak[r["index"]])
        dim[r["system"]] = r["D"]
    names = sorted(by, key=lambda s: (rate(by[s]), s))
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    xs = range(len(names))
    ax.vlines(list(xs), [rate(byw[s]) for s in names], [rate(by[s]) for s in names], color=GRID, lw=1.2, zorder=1)
    ax.scatter(list(xs), [rate(byw[s]) for s in names], s=10, color=ORANGE, label="weak best-subset", zorder=2)
    for d3, mk, lab in [(True, "o", "final model, D = 3"), (False, "D", "final model, D = 4-6")]:
        sel = [i for i, s in enumerate(names) if (dim[s] == 3) == d3]
        ax.scatter(sel, [rate(by[names[i]]) for i in sel], s=14, marker=mk, color=BLUE if d3 else AQUA,
                   edgecolor="white", linewidth=0.5, label=lab, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=90, fontsize=4.6)
    ax.set_ylabel("exact recovery (80 runs)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-1, len(names))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "dysts_systems.pdf")
    plt.close(fig)


# ---- Figure 3: cumulative ablation ladder ----
def fig_ladder():
    rungs = [("weak", "weak\nbest-subset"), ("L1_refit", "+ joint\nstates"), ("L2_proposals", "+ profiled\nproposals"),
             ("L3_starts", "+ multiple\nstarts"), ("L4_selfcal", "+ " r"$\sigma/\varphi$" "\nself-cal."),
             ("L5_substeps", "+ auto\nsub-steps"), ("L6_huber", "+ Huber\n(= C)"), ("white", "+ complexity\nprior"),
             ("full", "+ AR(1)\nswitch")]
    sets = [("sealed", "sealed (160)", BLUE, "o"), ("colored", "coloured AR(1) (89)", ORANGE, "s"),
            ("hard40", "hard 40", AQUA, "^"), ("dysts", "dysts D = 3 (287)", "#6250d6", "D")]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for f, lab, col, mk in sets:
        recs = load(f"results_ablation/{f}.json")
        ys = []
        for key, _ in rungs:
            ys.append(rate([r["weak"]["exact"] if key == "weak" else r["variants"][key]["exact"] for r in recs]))
        ax.plot(range(len(rungs)), ys, marker=mk, color=col, label=lab)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([l for _, l in rungs], fontsize=6.3)
    ax.set_ylabel("exact support recovery")
    ax.set_ylim(-0.03, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_ladder.pdf")
    plt.close(fig)


# ---- Figure 4: Rabinovich-Fabrikant exact recovery vs N (prospective archive) ----
def fig_rf():
    recs = load("results_forecast/rf_needed_C.json")
    by = defaultdict(list)
    for r in recs:
        by[(r["noise"], r["n"])].append(r["joint"]["0.03"]["final"]["exact"])
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for noise, col, mk in [(0.01, AQUA, "^"), (0.05, BLUE, "o"), (0.1, ORANGE, "s")]:
        ns = sorted(n for (s, n) in by if s == noise)
        if ns:
            ax.plot(ns, [rate(by[(noise, n)]) for n in ns], marker=mk, color=col, label=f"{int(100 * noise)}% noise")
    ax.axvline(2000, color=MUTED, lw=0.8, ls=":")
    ax.text(2080, 0.93, "benchmark N = 2000", color=MUTED, fontsize=6.5, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("exact recovery (10 runs)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "rf_data.pdf")
    plt.close(fig)


# ---- Figure: ours against SINDy baselines by noise level, paired dysts7 runs ----
def fig_baselines_noise():
    ours = {r["index"]: (r["final"]["exact"] if r["status"] == "ok" else 0.0) for r in load("results_dysts/test_final.json")}
    stlsq = load("results_baselines/dysts7_fd_stlsq_raw_w11.json")
    miosr = load("results_baselines/dysts7_weak_miosr_h10.json")
    get = lambda r, k: (r[k]["exact"] if isinstance(r.get(k), dict) else r.get(k, 0.0)) if r["status"] == "ok" else 0.0
    series = [("Ours (truth-free)", BLUE, "o", {r["index"]: ours[r["index"]] for r in stlsq}),
              ("MIOSR, true number of terms", AQUA, "s", {r["index"]: get(r, "oracle") for r in miosr}),
              ("STLSQ, best threshold per run", ORANGE, "^", {r["index"]: get(r, "oracle") for r in stlsq}),
              ("STLSQ, CV (truth-free)", MUTED, "v", {r["index"]: get(r, "cv") for r in stlsq})]
    meta = {r["index"]: r for r in stlsq}
    noises = [0.01, 0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.25), sharey=True)
    for ax, n in zip(axes, [500, 2000]):
        for lab, col, mk, v in series:
            ys = [rate([e for i, e in v.items() if meta[i]["n"] == n and meta[i]["noise"] == s]) for s in noises]
            ax.plot([100 * s for s in noises], ys, marker=mk, color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel("noise (% of state std)")
        ax.set_title(f"N = {n}", color=INK)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("exact support recovery")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(OUT / "baselines_noise.pdf")
    plt.close(fig)
    print({lab: round(rate(list(v.values())), 3) for lab, _, _, v in series}, len(stlsq))


# ---- Figure: accuracy against cost on the 80 head-to-head runs ----
def fig_cost():
    # docs/reports/fast_baselines.md: exact of 80, median wall seconds per run (DEDi rows from results_h2h/rows.json)
    pts = [("WSINDy", 0.27, 24, ORANGE, "^", (5, -9)), ("Weak selector", 0.48, 36, ORANGE, "^", (5, 3)),
           ("STLSQ", 3.1, 17, ORANGE, "^", (0, 6)), ("E-SINDy", 60, 17, ORANGE, "^", (-16, -11)),
           ("ARGOS", 126, 17, ORANGE, "^", (4, 5)), ("Ours", 14.4, 66, BLUE, "o", (-6, 6)),
           ("DEDi, 1 setting", 120, 44, AQUA, "s", (5, -3)), ("DEDi, 5 settings (BIC)", 569, 56, AQUA, "s", (-80, -2)),
           ("DEDi, 35 settings (val)", 5930, 55, AQUA, "s", (-72, -10)),
           ("DEDi, 35 settings (BIC)", 5930, 61, AQUA, "s", (-76, 6))]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for lab, x, y, col, mk, off in pts:
        ax.scatter(x, y, color=col, marker=mk, s=26, zorder=3, linewidths=0.4, edgecolors="white")
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=INK)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=l) for c, m, l in
               [(BLUE, "o", "Ours"), (AQUA, "s", "DEDi"), (ORANGE, "^", "Fast baselines")]]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 12000)
    ax.set_ylim(0, 80)
    ax.set_xlabel("median seconds per run (log scale)")
    ax.set_ylabel("exact recovery (of 80 runs)")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_cost.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_baselines_noise()
    fig_cost()
    fig_dysts_noise()
    fig_dysts_systems()
    fig_ladder()
    fig_rf()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
