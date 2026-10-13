"""Step 07 — paper figures (PNG + PDF in outputs/figures/, _dev suffix with --dev).

Fig 1  learning curves: mean character accuracy and share of usable participants vs k
Fig 2  calibration budgets per cohort: B80 (accuracy) and AUC budget, bootstrap 95% CI
Fig 3  RQ3 moderators: standardised meta-regression coefficients (± 95% CI)
Fig 4  RQ4 transportability: observed budget vs leave-one-cohort-out typical budget\nFig 4b transport error vs within-cohort reference and local pilot (needs step 09); used as Fig 4 in the paper
Usage
  python scripts/07_make_figures.py [--dev]
"""
from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, output_name, project_path  # noqa: E402

# Palette: reference categorical slots 1–2, neutral ink/grid (dataviz skill palette.md)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52524e", "#a3a29b", "#e5e4df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "lines.linewidth": 2,
    "legend.frameon": False,
})
KTICKS = [1, 2, 4, 8, 16, 32]


PAPER = False  # set by --paper: white background, no headline titles (captions carry them)


def headline(obj, text, **kw):
    """Figure/axes headline; omitted in paper mode."""
    if PAPER:
        return
    if hasattr(obj, "suptitle"):
        obj.suptitle(text, **kw)
    else:
        obj.set_title(text, **kw)


def save(fig, out: Path, stem: str, dev: bool):
    stem = stem + ("_paper" if PAPER else "")
    for ext in ("png", "pdf"):
        fig.savefig(out / output_name(f"{stem}.{ext}", dev), dpi=300 if PAPER else 200,
                    bbox_inches="tight")
    plt.close(fig)


def log_k_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(KTICKS, [str(k) for k in KTICKS])
    ax.minorticks_off()


def fig_learning_curves(T, out, dev, cfg):
    lc = pd.read_csv(T / output_name("learning_curves.csv", dev))
    b80 = pd.read_csv(T / output_name("b80.csv", dev)).set_index("study")
    val = pd.read_csv(T / output_name("decoding_validity.csv", dev)).set_index("study")
    studies = list(b80.sort_values(["censored", "b80_chars"]).index)
    n = len(studies)
    ncol = 5 if n > 9 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.95 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, studies):
        g = lc[(lc.study == s) & (lc.data_coverage >= cfg["budget"]["coverage"])].sort_values("k")
        ok = bool(val.loc[s, "valid_decoding"]) if s in val.index else True
        c1, c2 = (BLUE, ORANGE) if ok else (MUTED, MUTED)
        ax.axhline(cfg["budget"]["usable_accuracy"], color=INK2, lw=0.8, ls=":")
        ax.axhline(cfg["budget"]["coverage"], color=INK2, lw=0.8, ls="--")
        ax.plot(g.k, g.mean_acc, color=c1, marker="o", ms=3.5, label="Mean character accuracy")
        ax.plot(g.k, g.share_usable, color=c2, lw=1.4, ls=(0, (4, 2)) if not ok else "-",
                label=f"Share of participants ≥ {cfg['budget']['usable_accuracy']:.0%}")
        r = b80.loc[s]
        if not r.censored:
            ax.axvline(r.b80_chars, color=INK, lw=1)
            ax.text(r.b80_chars, 0.04, f" B80={r.b80_chars:.0f}", fontsize=7.5, color=INK)
        else:
            ax.text(0.97, 0.04, f"B80 > {r.max_evaluable_k:.0f}", fontsize=7.5, color=INK2,
                    ha="right", transform=ax.transAxes)
        title = f"{s}  (n={int(r.n_participants)})" + ("" if ok else "  · invalid")
        ax.set_title(title, loc="left", color=INK if ok else INK2)
        log_k_axis(ax)
        ax.set_ylim(0, 1.02)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("Proportion")
    for ax in axes[max(0, n - ncol):n]:
        ax.set_xlabel("Calibration characters (k)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    headline(fig, f"Calibration learning curves per cohort (dotted: {cfg['budget']['usable_accuracy']:.0%} accuracy; "
             f"dashed: {cfg['budget']['coverage']:.0%} coverage)",
             y=1.07, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig1_learning_curves", dev)


def fig_budgets(T, out, dev, cfg):
    acc = pd.read_csv(T / output_name("b80.csv", dev)).set_index("study")
    auc = pd.read_csv(T / output_name("b80_auc.csv", dev)).set_index("study")
    val = pd.read_csv(T / output_name("decoding_validity.csv", dev)).set_index("study")
    order = auc.assign(key=np.where(auc.censored, 1e3 + auc.max_evaluable_k, auc.b80_chars)) \
               .sort_values("key", ascending=False).index
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(order) + 1.2))
    for i, s in enumerate(order):
        for df, color, dy, lab in ((acc, BLUE, 0.17, "B80 (character accuracy)"),
                                   (auc, ORANGE, -0.17, f"AUC budget (AUC ≥ {cfg['budget']['auc_threshold']:.2f})")):
            r = df.loc[s]
            invalid = df is acc and s in val.index and not val.loc[s, "valid_decoding"]
            col = MUTED if invalid else color
            if r.censored:
                ax.plot(r.max_evaluable_k, i + dy, marker=">", ms=7, color=col, ls="none",
                        mfc=SURFACE, mew=1.6, label=None)
            else:
                lo, hi = r.ci_low, (r.ci_high if not r.ci_high_censored else r.max_evaluable_k)
                ax.plot([lo, hi], [i + dy] * 2, color=col, lw=1.6, solid_capstyle="round")
                ax.plot(r.b80_chars, i + dy, "o", ms=6.5, color=col,
                        mfc=SURFACE if invalid else col, mew=1.6,
                        label=lab if (i == len(order) - 1 or s == order[-1]) else None)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    log_k_axis(ax)
    ax.set_xlabel(f"Calibration characters needed for {cfg['budget']['coverage']:.0%} of participants (log scale)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker="o", label="B80 (character accuracy)"),
               Line2D([], [], color=ORANGE, marker="o", label=f"AUC budget (AUC ≥ {cfg['budget']['auc_threshold']:.2f})"),
               Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, label="B80, decoding not valid"),
               Line2D([], [], color=INK2, marker=">", ls="none", mfc=SURFACE, label="Not reached (> max k)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    reached = auc[~auc.censored].b80_chars
    ratio = reached.max() / reached.min() if len(reached) >= 2 else float("nan")
    headline(ax, f"Calibration budget varies about {ratio:.0f}-fold across cohorts", loc="left", pad=34)
    fig.tight_layout()
    save(fig, out, "fig2_budgets", dev)


def fig_moderators(T, out, dev):
    panels = [("acc_valid", "Learning-curve slope (log-odds per log k)\nvalid-decoding cohorts", BLUE),
              ("auc_all", "Learning-curve slope (AUC per log k)\nall cohorts", ORANGE)]
    panels = [p for p in panels if (T / output_name(f"moderators_{p[0]}.csv", dev)).exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    labels = {"seconds_per_calib_char": "Seconds per calibration character",
              "test_flashes_per_char": "Test flashes per character", "grid_size": "Grid size",
              "isi_ms": "Inter-stimulus interval", "share_rc": "Row-column share",
              "lit_per_flash": "Characters lit per flash", "als_cohort": "ALS cohort",
              "train_reject_rate": "Artefact rejection rate"}
    for ax, (tag, title, color) in zip(axes, panels):
        m = pd.read_csv(T / output_name(f"moderators_{tag}.csv", dev))
        m = m[m.descriptor.isin(labels)].reset_index(drop=True)
        y = np.arange(len(m))
        ax.axvline(0, color=INK2, lw=0.9)
        for i, r in m.iterrows():
            if not np.isfinite(r.beta):
                continue
            sig = r.p_holm < 0.05
            ax.plot([r.beta - 1.96 * r.se, r.beta + 1.96 * r.se], [i, i], color=color, lw=1.6)
            ax.plot(r.beta, i, "o", ms=6, color=color, mfc=color if sig else SURFACE, mew=1.6)
            ptxt = f"p={r.p:.3f}" if r.p < 0.01 else f"p={r.p:.2f}"
            ax.text(1.02, i, f"{ptxt} (Holm {r.p_holm:.2f})", transform=ax.get_yaxis_transform(),
                    fontsize=7, color=INK2, va="center")
        ax.set_yticks(y, [labels[d] for d in m.descriptor])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("Standardised coefficient (per SD)")
        ax.grid(axis="y", visible=False)
    sig = []
    for tag, _, _ in panels:
        m = pd.read_csv(T / output_name(f"moderators_{tag}.csv", dev))
        sig += [labels.get(d, d) for d in m.loc[m.p_holm < 0.05, "descriptor"]]
    head = (f"{' and '.join(sorted(set(sig)))} explains slope differences after Holm correction"
            if sig else "No protocol descriptor explains slope differences after Holm correction")
    headline(fig, head + " (filled = significant)", fontsize=9.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig3_moderators", dev)


def fig_loco(T, out, dev, cfg):
    tag = "auc_all"
    path = T / output_name(f"loco_{tag}.csv", dev)
    if not path.exists():
        return
    lo = pd.read_csv(path)
    lo = lo.assign(key=np.where(lo.observed_censored, 1e3 + lo.observed_max_k, lo.observed_b80)) \
           .sort_values("key").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(lo) + 1.2))
    for i, r in lo.iterrows():
        obs = r.observed_max_k if r.observed_censored else r.observed_b80
        ax.plot([r.pred_median, obs], [i, i], color=MUTED, lw=1.4)
        ax.plot(r.pred_median, i, "|", ms=11, mew=2, color=INK2)
        ax.plot(obs, i, ">" if r.observed_censored else "o", ms=6.5, color=ORANGE,
                mfc=SURFACE if r.observed_censored else ORANGE, mew=1.6)
        err = f"{r.err_median:.0f}" + ("+" if r.err_median_type == "lower_bound" else "")
        ax.text(max(obs, r.pred_median) * 1.12, i, err, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(lo)), lo.study)
    log_k_axis(ax)
    ax.set_xlabel("AUC budget (calibration characters, log scale)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="o", ls="none", label="Observed budget"),
                       Line2D([], [], color=ORANGE, marker=">", ls="none", mfc=SURFACE, label="Observed: not reached"),
                       Line2D([], [], color=INK2, marker="|", ls="none", ms=10, mew=2,
                              label="Predicted from other cohorts (median)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    e = cfg["hypotheses"]["h4_min_error_chars"]
    share = (lo.err_median >= e).mean()
    headline(ax, f"A budget borrowed from other cohorts misses by ≥ {e} characters in {share:.0%} of cohorts",
             loc="left", pad=44)
    fig.tight_layout()
    save(fig, out, "fig4_transportability", dev)


def fig_transport_error(T, out, dev, cfg):
    """Fig 4: how far each rule's budget is from a cohort's own budget (factor, log2 scale).

    grey  = within-cohort split-half (sampling noise within one protocol)
    orange = borrowed (median of the other cohorts' budgets); open triangle = lower bound (censored)
    blue  = local pilot of n participants from the same cohort
    """
    npil = 5
    labels = {"acc": f"B80 (character accuracy, valid-decoding cohorts)",
              "auc": f"AUC budget (AUC $\\geq$ {cfg['budget']['auc_threshold']}, all cohorts)"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0), height_ratios=[9, 14], sharex=True)
    for ax, metric, obsf in zip(axes, ("acc", "auc"), ("b80.csv", "b80_auc.csv")):
        ref = pd.read_csv(T / f"transport_reference_{metric}.csv")
        pil = pd.read_csv(T / f"pilot_{metric}.csv")
        obs = pd.read_csv(T / obsf).set_index("study")
        cohorts = sorted(ref.study.unique() if metric == "acc" else obs.index)
        if metric == "acc":
            cohorts = sorted(set(pd.read_csv(T / "pilot_acc.csv").study))
        sh = ref[ref.rule == "split_half"].groupby("study").log2_err.median()
        bo = ref[ref.rule == "borrowed_median"].set_index("study").log2_err
        pi = pil[(pil.rule == "pilot") & (pil.n_pilot == npil)].groupby("study").log2_err.median()
        reached = obs[~obs.censored].b80_chars
        lb = {}
        for s in cohorts:
            if bool(obs.loc[s, "censored"]):
                pred = reached.drop(index=s, errors="ignore").median()
                lb[s] = max(0.0, np.log2(obs.loc[s, "max_evaluable_k"] / pred))
        order = sorted(cohorts, key=lambda s: -(bo.get(s, np.nan) if s in bo else lb.get(s, 0)))
        y = np.arange(len(order))
        for i, s in enumerate(order):
            ax.plot([0, 3.2], [i, i], color=GRID, lw=0.6, zorder=0)
            if s in sh and np.isfinite(sh[s]):
                ax.plot(sh[s], i, "o", color=MUTED, mfc="none", mew=1.4, ms=8, zorder=5)
            if s in bo and np.isfinite(bo[s]):
                ax.plot(bo[s], i, "o", color=ORANGE, ms=6, zorder=3)
            elif s in lb:
                ax.plot(lb[s], i, ">", color=ORANGE, mfc="white", ms=6, zorder=3)
            if s in pi and np.isfinite(pi[s]):
                ax.plot(pi[s], i, "D", color=BLUE, ms=4.5, zorder=4)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(labels[metric], loc="left", fontsize=8, fontweight="bold")
    axes[-1].set_xticks([0, 0.5, 1, 2, 3], ["1$\\times$", "1.4$\\times$", "2$\\times$", "4$\\times$", "8$\\times$"])
    axes[-1].set_xlim(-0.1, 3.2)
    axes[-1].set_xlabel("Error of predicted budget (factor, log scale)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MUTED, marker="o", mfc="none", mew=1.4, ms=8, ls="", label="Within cohort (split-half)"),
         Line2D([], [], color=BLUE, marker="D", ls="", ms=4.5, label=f"Local pilot (n = {npil})"),
         Line2D([], [], color=ORANGE, marker="o", ls="", label="Borrowed (other cohorts)"),
         Line2D([], [], color=ORANGE, marker=">", mfc="white", ls="", label="Borrowed, lower bound")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    headline(fig, "Borrowed budgets miss by 2-8x; a 5-person local pilot is close to within-cohort noise")
    fig.tight_layout()
    save(fig, out, "fig4_transport_error", dev)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--paper", action="store_true", help="print versions for the manuscript")
    args = ap.parse_args()
    global PAPER, SURFACE
    PAPER = args.paper
    if PAPER:
        SURFACE = "#ffffff"
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                             "savefig.facecolor": SURFACE})
    cfg = load_config()
    T = project_path(cfg["paths"]["tables"])
    out = project_path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_learning_curves, fig_budgets, fig_loco):
        f(T, out, args.dev, cfg)
    if (T / "transport_reference_acc.csv").exists():
        fig_transport_error(T, out, args.dev, cfg)
    fig_moderators(T, out, args.dev)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
