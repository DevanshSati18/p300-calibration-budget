"""Step 07 — paper figures (PNG + PDF in outputs/figures/, _dev suffix with --dev).

Fig 1  learning curves: mean character accuracy and share of usable participants vs k
Fig 2  calibration budgets per cohort: B80 (accuracy) and AUC budget, bootstrap 95% CI
Fig 3  RQ3 moderators: standardised meta-regression coefficients (± 95% CI)
Fig 4  RQ4 transportability: observed budget vs leave-one-cohort-out typical budget
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


def save(fig, out: Path, stem: str, dev: bool):
    for ext in ("png", "pdf"):
        fig.savefig(out / output_name(f"{stem}.{ext}", dev), dpi=200, bbox_inches="tight")
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
                label="Share of participants ≥ 70%")
        r = b80.loc[s]
        if not r.censored:
            ax.axvline(r.b80_chars, color=INK, lw=1)
            ax.text(r.b80_chars, 0.04, f" B80={r.b80_chars:.0f}", fontsize=7.5, color=INK)
        else:
            ax.text(0.97, 0.04, f"B80 > {r.max_evaluable_k:.0f}", fontsize=7.5, color=INK2,
                    ha="right", transform=ax.transAxes)
        title = f"{s}  (n={int(r.n_participants)})" + ("" if ok else "  · decoding not valid")
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
    fig.suptitle("Calibration learning curves per cohort (dotted: 70% accuracy; dashed: 80% coverage)",
                 y=1.07, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig1_learning_curves", dev)


def fig_budgets(T, out, dev):
    acc = pd.read_csv(T / output_name("b80.csv", dev)).set_index("study")
    auc = pd.read_csv(T / output_name("b80_auc.csv", dev)).set_index("study")
    val = pd.read_csv(T / output_name("decoding_validity.csv", dev)).set_index("study")
    order = auc.assign(key=np.where(auc.censored, 1e3 + auc.max_evaluable_k, auc.b80_chars)) \
               .sort_values("key", ascending=False).index
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(order) + 1.2))
    for i, s in enumerate(order):
        for df, color, dy, lab in ((acc, BLUE, 0.17, "B80 (character accuracy)"),
                                   (auc, ORANGE, -0.17, "AUC budget (AUC ≥ 0.75)")):
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
    ax.set_xlabel("Calibration characters needed for 80% of participants (log scale)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker="o", label="B80 (character accuracy)"),
               Line2D([], [], color=ORANGE, marker="o", label="AUC budget (AUC ≥ 0.75)"),
               Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, label="B80, decoding not valid"),
               Line2D([], [], color=INK2, marker=">", ls="none", mfc=SURFACE, label="Not reached (> max k)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    ax.set_title("Calibration budget varies about 15-fold across cohorts", loc="left", pad=34)
    fig.tight_layout()
    save(fig, out, "fig2_budgets", dev)


def fig_moderators(T, out, dev):
    panels = [("acc_valid", "Learning-curve slope (log-odds per log k)\nvalid-decoding cohorts", BLUE),
              ("auc_all", "Learning-curve slope (AUC per log k)\nall cohorts", ORANGE)]
    panels = [p for p in panels if (T / output_name(f"moderators_{p[0]}.csv", dev)).exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.2), sharey=True)
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
            ax.text(1.02, i, f"p={r.p:.2f} (Holm {r.p_holm:.2f})", transform=ax.get_yaxis_transform(),
                    fontsize=7, color=INK2, va="center")
        ax.set_yticks(y, [labels[d] for d in m.descriptor])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("Standardised coefficient (per SD of descriptor)")
        ax.grid(axis="y", visible=False)
    sig = []
    for tag, _, _ in panels:
        m = pd.read_csv(T / output_name(f"moderators_{tag}.csv", dev))
        sig += [labels.get(d, d) for d in m.loc[m.p_holm < 0.05, "descriptor"]]
    head = (f"{' and '.join(sorted(set(sig)))} explains slope differences after Holm correction"
            if sig else "No protocol descriptor explains slope differences after Holm correction")
    fig.suptitle(head + " (filled = significant)", fontsize=9.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig3_moderators", dev)


def fig_loco(T, out, dev):
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
    share = (lo.err_median >= 2).mean()
    ax.set_title(f"A budget borrowed from other cohorts misses by ≥ 2 characters in {share:.0%} of cohorts",
                 loc="left", pad=44)
    fig.tight_layout()
    save(fig, out, "fig4_transportability", dev)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    T = project_path(cfg["paths"]["tables"])
    out = project_path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_learning_curves,):
        f(T, out, args.dev, cfg)
    for f in (fig_budgets, fig_moderators, fig_loco):
        f(T, out, args.dev)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
