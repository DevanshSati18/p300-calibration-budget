"""Step 06b — robustness of the calibration budget (pre-specified in PROPOSAL.md §8).

Variants, each for B80 (character accuracy, valid-decoding subset) and the AUC budget
(all cohorts):
  base          as in the primary analysis
  first_session each participant's first session only
  low_artefact  records with <= 20% rejected calibration epochs only
  thr_low/high  accuracy threshold 0.60 / 0.80 (B80) or AUC 0.70 / 0.80 (AUC budget)
Outputs
  outputs/tables/robustness_budgets.csv   variant × metric × cohort
  outputs/tables/robustness_summary.csv   variant × metric: reached, range, ratio
Usage
  python scripts/06b_robustness.py [--dev]
"""
from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.budget import compute_b80, participant_curves  # noqa: E402
from p300budget.config import load_config, output_name, project_path  # noqa: E402
from p300budget.stats_models import decoding_validity  # noqa: E402

MAX_REJECT = 0.20


def curves(sweep: pd.DataFrame, metric: str) -> pd.DataFrame:
    pc = participant_curves(sweep, "acc")
    if metric == "auc":
        pc["acc"] = pc["auc"]
    return pc


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    T = project_path(cfg["paths"]["tables"])
    b, st = cfg["budget"], cfg["stats"]
    sweep = pd.read_csv(T / output_name("sweep_results.csv", args.dev))
    valid = decoding_validity(sweep, b["decoding_tolerance"])
    valid_set = set(valid.loc[valid.valid_decoding, "study"])

    # Record-level filters
    s = sweep[sweep.control == "none"]
    first = (s.groupby("participant").session.min().rename("first_session"))
    s_first = s.join(first, on="participant")
    s_first = s_first[s_first.session == s_first.first_session].drop(columns="first_session")
    full = s[(s.k == s.n_train_chars_total) & (s.status == "ok")]
    rej = full.n_rejected_in_budget / (full.n_train_epochs + full.n_rejected_in_budget)
    clean = set(full.loc[rej <= MAX_REJECT, "record"])
    s_clean = s[s.record.isin(clean)]

    thr = {"acc": b["usable_accuracy"], "auc": b["auc_threshold"]}
    alt = {"acc": (0.60, 0.80), "auc": tuple(b["auc_threshold_sensitivity"])}
    variants = [("base", s, None), ("first_session", s_first, None), ("low_artefact", s_clean, None),
                ("thr_low", s, 0), ("thr_high", s, 1)]

    rows = []
    for metric in ("acc", "auc"):
        for name, data, alt_i in variants:
            t = thr[metric] if alt_i is None else alt[metric][alt_i]
            pc = curves(data, metric)
            if metric == "acc":
                pc = pc[pc.study.isin(valid_set)]
            res = compute_b80(pc, t, b["coverage"], st["bootstrap_reps"] // 4, st["seed"])
            res["variant"], res["metric"], res["threshold"] = name, metric, t
            rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    out["config_hash"] = cfg["_hash"]
    out.to_csv(T / output_name("robustness_budgets.csv", args.dev), index=False)

    def summ(g):
        r = g[~g.censored]
        return pd.Series(dict(
            n_cohorts=len(g), n_reached=len(r),
            min=r.b80_chars.min() if len(r) else np.nan,
            max=r.b80_chars.max() if len(r) else np.nan,
            ratio=r.b80_chars.max() / r.b80_chars.min() if len(r) >= 2 else np.nan,
            n_participants=int(g.n_participants.sum())))
    summary = out.groupby(["metric", "variant", "threshold"], sort=False).apply(summ).reset_index()
    summary.to_csv(T / output_name("robustness_summary.csv", args.dev), index=False)
    print(summary.round(2).to_string(index=False))

    wide = out.assign(budget=out.apply(lambda r: f">{r.max_evaluable_k:.0f}" if r.censored
                                       else f"{r.b80_chars:.0f}", axis=1))
    print("\nBudget per cohort and variant (characters)")
    print(wide.pivot_table(index="study", columns=["metric", "variant"], values="budget",
                           aggfunc="first").to_string())


if __name__ == "__main__":
    main()
