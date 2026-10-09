"""Step 06 — statistics for RQ1–RQ4.

--metric acc  (B80, character accuracy): analysed on the valid-decoding subset by
              default (cohorts whose offline full-calibration accuracy is within
              budget.decoding_tolerance of online accuracy).
--metric auc  (AUC budget, co-primary): analysed on all cohorts.

Outputs (outputs/tables/, tag = <metric>_<subset>, _dev suffix with --dev)
  cohort_fits_<tag>.csv, heterogeneity_<tag>.json, moderators_<tag>.csv,
  b80_moderators_<tag>.csv, loco_<tag>.csv, decoding_validity.csv
Usage
  python scripts/06_run_stats.py [--metric acc|auc] [--subset valid|all] [--dev]
"""
from __future__ import annotations

import argparse
import warnings
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, output_name, project_path  # noqa: E402
import p300budget.stats_models as sm_mod  # noqa: E402
from p300budget.stats_models import (cohort_descriptors, decoding_validity,  # noqa: E402
                                     fit_all_cohorts, holm, loco_b80, meta_regression,
                                     random_effects)

DESCRIPTORS = ["seconds_per_calib_char", "test_flashes_per_char", "grid_size", "isi_ms",
               "share_rc", "lit_per_flash", "als_cohort", "train_reject_rate"]


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="acc", choices=["acc", "auc"])
    ap.add_argument("--subset", default=None, choices=["valid", "all"],
                    help="default: valid for acc, all for auc")
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    subset = args.subset or ("valid" if args.metric == "acc" else "all")
    tag = f"{args.metric}_{subset}"

    cfg = load_config()
    sm_mod.K_CENTER = float(cfg["stats"]["k_center"])
    h4 = cfg["hypotheses"]
    T = project_path(cfg["paths"]["tables"])
    b = cfg["budget"]
    name = lambda n: T / output_name(n, args.dev)  # noqa: E731
    suffix = "" if args.metric == "acc" else "_auc"
    thr = b["usable_accuracy"] if args.metric == "acc" else b["auc_threshold"]

    sweep = pd.read_csv(name("sweep_results.csv"))
    pc = pd.read_csv(name(f"participant_curves{suffix}.csv"))
    b80 = pd.read_csv(name(f"b80{suffix}.csv"))
    records = pd.read_csv(T / "inventory_records.csv")
    ext_path = T / "extraction_log.csv"
    extraction = pd.read_csv(ext_path if ext_path.exists() else name("extraction_log.csv"))

    validity = decoding_validity(sweep, b["decoding_tolerance"])
    validity.to_csv(name("decoding_validity.csv"), index=False)
    if subset == "valid":
        keep = set(validity.loc[validity.valid_decoding, "study"])
        dropped = sorted(set(pc.study) - keep)
        pc, b80 = pc[pc.study.isin(keep)], b80[b80.study.isin(keep)]
        print(f"Valid-decoding subset (|offline - online| <= {b['decoding_tolerance']}): "
              f"{len(keep)} cohorts; dropped {dropped}\n")

    # Stage 1 ---------------------------------------------------------------
    fits = fit_all_cohorts(pc, "binomial" if args.metric == "acc" else "linear")
    fits.to_csv(name(f"cohort_fits_{tag}.csv"), index=False)
    ok = fits[fits.status == "ok"]
    unit = "log-odds correct" if args.metric == "acc" else "AUC"
    print(f"Stage 1 — {unit} ~ log k per cohort (intercept at k = {sm_mod.K_CENTER:g})")
    print(ok[["study", "n_participants", "intercept", "intercept_se", "slope", "slope_se"]]
          .round(3).to_string(index=False))

    # Stage 2: heterogeneity (RQ1/RQ2) --------------------------------------
    het = {"metric": args.metric, "subset": subset, "threshold": thr,
           "slope": random_effects(ok.slope, ok.slope_se),
           "intercept": random_effects(ok.intercept, ok.intercept_se)}
    reached = b80[~b80.censored]
    het["budget"] = dict(
        n_cohorts=int(len(b80)), n_reached=int(len(reached)),
        min=float(reached.b80_chars.min()) if len(reached) else None,
        max=float(reached.b80_chars.max()) if len(reached) else None,
        ratio_max_min=float(reached.b80_chars.max() / reached.b80_chars.min()) if len(reached) >= 2 else None,
        sd_log=float(np.log(reached.b80_chars).std(ddof=1)) if len(reached) >= 2 else None,
        h2_supported=bool(len(reached) >= 2 and reached.b80_chars.max() / reached.b80_chars.min() >= 2),
    )
    het["config_hash"] = cfg["_hash"]
    with open(name(f"heterogeneity_{tag}.json"), "w") as fh:
        json.dump(het, fh, indent=2, default=float)
    s, i, hb = het["slope"], het["intercept"], het["budget"]
    print(f"\nRandom effects — slope: pooled {s['pooled']:.3f}, tau {s['tau']:.3f}, I2 {s['I2']:.2f}"
          + (f", PI [{s['pi_low']:.2f}, {s['pi_high']:.2f}]" if "pi_low" in s else ""))
    print(f"Random effects — intercept: pooled {i['pooled']:.3f}, tau {i['tau']:.3f}, I2 {i['I2']:.2f}")
    print(f"Budget: {hb['n_reached']}/{hb['n_cohorts']} reached; range {hb['min']}–{hb['max']}; "
          f"ratio {hb['ratio_max_min']}; H2 supported: {hb['h2_supported']}")

    # RQ3: moderators -------------------------------------------------------
    desc = cohort_descriptors(records, extraction, cfg["data"].get("als_studies", []), sweep)
    m = ok.merge(desc, on="study", how="left")
    mods = pd.DataFrame([{"descriptor": d, **meta_regression(m.slope, m.slope_se,
                                                              het["slope"]["tau2"], m[d])}
                         for d in DESCRIPTORS])
    mods["p_holm"] = holm(mods.p)
    mods.to_csv(name(f"moderators_{tag}.csv"), index=False)
    print("\nRQ3 — meta-regression of cohort slope on each descriptor (standardised, Holm)")
    print(mods[["descriptor", "k", "beta", "se", "p", "p_holm", "share_explained"]]
          .round(3).to_string(index=False))

    bm = b80.merge(desc, on="study", how="left")
    bm = bm[~bm.censored]
    rows = []
    for d in DESCRIPTORS:
        x = bm[d].to_numpy(float)
        ok_x = len(bm) >= 4 and np.nanstd(x) > 0
        rho, p = stats.spearmanr(np.log(bm.b80_chars), x, nan_policy="omit") if ok_x else (np.nan, np.nan)
        rows.append(dict(descriptor=d, n=len(bm), spearman_rho=rho, p=p))
    bmod = pd.DataFrame(rows)
    bmod["p_holm"] = holm(bmod.p)
    bmod.to_csv(name(f"b80_moderators_{tag}.csv"), index=False)
    print("\nRQ3 — Spearman correlation of log budget with each descriptor (reached cohorts)")
    print(bmod.round(3).to_string(index=False))

    # RQ4: leave-one-cohort-out ----------------------------------------------
    loco = loco_b80(pc, b80, thr, b["coverage"], h4["h4_min_error_chars"])
    loco.to_csv(name(f"loco_{tag}.csv"), index=False)
    print("\nRQ4 — leave-one-cohort-out budget prediction (characters)")
    print(loco[["study", "observed_b80", "observed_censored", "pred_pooled", "err_pooled",
                "pred_median", "err_median", "err_median_type"]].round(1).to_string(index=False))
    for which in ("pooled", "median"):
        defined = loco[loco[f"err_{which}_type"] != "undefined"]
        label = "primary" if which == "pooled" else "secondary"
        if len(defined):
            share = (defined[f"err_{which}"] >= h4["h4_min_error_chars"]).mean()
            print(f"{which:6} ({label}): n={len(defined)}, median error "
                  f"{defined[f'err_{which}'].median():.1f}; share >= {h4['h4_min_error_chars']} chars = {share:.2f} "
                  f"(H4 predicts >= {h4['h4_min_share']:.2f}) -> supported: {share >= h4['h4_min_share']}")
        else:
            print(f"{which:6} ({label}): undefined for every cohort (pooled budget never reached)")
    if args.dev:
        print("\nDEV MODE: few cohorts/participants; NaNs are expected.")


if __name__ == "__main__":
    main()
