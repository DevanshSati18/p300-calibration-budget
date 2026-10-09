"""Step 05 — learning curves and the calibration budget B80 per cohort.

Inputs
  outputs/tables/sweep_results.csv   (or sweep_results_dev.csv with --dev)
Outputs
  outputs/tables/participant_curves[_<metric>].csv   participant × k accuracy
  outputs/tables/learning_curves[_<metric>].csv      cohort × k summary
  outputs/tables/b80[_<metric>].csv                  B80 per cohort + bootstrap CI
Usage
  python scripts/05_compute_budget.py [--metric acc|acc_r_secondary] [--dev]
"""
from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.budget import cohort_curves, compute_b80, participant_curves  # noqa: E402
from p300budget.config import load_config, output_name, project_path  # noqa: E402


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="acc", choices=["acc", "acc_r_secondary", "auc"],
                    help="acc = as-run character accuracy; acc_r_secondary = fixed R; auc = flash-level AUC")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="override the usable threshold")
    args = ap.parse_args()

    cfg = load_config()
    tables = project_path(cfg["paths"]["tables"])
    b = cfg["budget"]
    if args.metric == "auc":
        thr = args.threshold if args.threshold is not None else b["auc_threshold"]
    else:
        thr = args.threshold if args.threshold is not None else b["usable_accuracy"]
    suffix = "" if args.metric == "acc" else f"_{args.metric}"
    if args.threshold is not None:
        suffix += f"_t{int(round(thr * 100))}"

    sweep = pd.read_csv(tables / output_name("sweep_results.csv", args.dev))
    pc = participant_curves(sweep, "acc" if args.metric == "auc" else args.metric)
    if args.metric == "auc":
        pc["acc"] = pc["auc"]  # the budget is computed on whichever column is named 'acc'
    cc = cohort_curves(pc, thr)
    b80 = compute_b80(pc, thr, b["coverage"],
                      cfg["stats"]["bootstrap_reps"], cfg["stats"]["seed"])
    for df in (pc, cc, b80):
        df["metric"] = args.metric
        df["config_hash"] = cfg["_hash"]

    pc.to_csv(tables / output_name(f"participant_curves{suffix}.csv", args.dev), index=False)
    cc.to_csv(tables / output_name(f"learning_curves{suffix}.csv", args.dev), index=False)
    b80.to_csv(tables / output_name(f"b80{suffix}.csv", args.dev), index=False)

    print(f"Metric: {args.metric}   usable >= {thr}   coverage {b['coverage']}\n")
    show = b80.copy()
    show["B80"] = show.apply(lambda r: f">{r.max_evaluable_k:.0f} (not reached)" if r.censored
                             else f"{r.b80_chars:.0f}", axis=1)
    show["95% CI"] = show.apply(
        lambda r: f"[{r.ci_low:.0f}, {'>max' if r.ci_high_censored else f'{r.ci_high:.0f}'}]"
        if pd.notna(r.ci_low) and r.ci_low != float('inf') else "n/a", axis=1)
    print(show[["study", "n_participants", "B80", "95% CI", "b80_seconds",
                "share_usable_full", "share_boot_censored"]].round(2).to_string(index=False))
    reached = b80[~b80.censored]
    if len(reached) >= 2:
        print(f"\nB80 range: {reached.b80_chars.min():.0f}–{reached.b80_chars.max():.0f} characters; "
              f"max/min ratio = {reached.b80_chars.max() / reached.b80_chars.min():.1f} "
              f"(H2 predicts >= 2)")
    if args.dev:
        print("\nDEV MODE: few participants per cohort, so B80 here only checks the code runs.")


if __name__ == "__main__":
    main()
