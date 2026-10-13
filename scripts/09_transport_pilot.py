"""Step 09 — transport error against a within-cohort reference, and a local-pilot rule.

Added after the internal audit (13 Oct 2026; post-hoc, PROPOSAL.md change log).
For each budget definition (B80 on valid-decoding cohorts; AUC budget on all cohorts):

  reference  split-half: budget from one random half of a cohort's participants vs the other half.
             This is the error expected from sampling participants alone, within one protocol.
  borrowed   leave-one-cohort-out median budget of the other cohorts (as in step 06).
  pilot      budget estimated from n randomly chosen participants of the SAME cohort
             (n in transport_extension.pilot_sizes), applied to that cohort's remaining participants.

Errors are |log2(predicted / observed)|, since the budget grid is roughly log-spaced
(a factor-of-2 miss = 1). Each rule is also scored by whether, at the budget it recommends,
>= 80% of the evaluation participants reach the usable threshold ("target met"),
and by the calibration it asks for.

Also: bootstrap CI of the max/min budget ratio; meta-regression of cohort intercepts
(accuracy level at k = 4) on the protocol descriptors.

Outputs (outputs/tables/): transport_reference_<metric>.csv, pilot_<metric>.csv,
  intercept_moderators_<metric>.csv, transport_summary.json
Usage: python scripts/09_transport_pilot.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.budget import _b80_from_matrix  # noqa: E402
from p300budget.config import load_config, project_path  # noqa: E402
import p300budget.stats_models as sm  # noqa: E402
from p300budget.stats_models import (cohort_descriptors, decoding_validity, holm,  # noqa: E402
                                     meta_regression, random_effects)

DESCRIPTORS = ["seconds_per_calib_char", "test_flashes_per_char", "grid_size", "isi_ms",
               "share_rc", "lit_per_flash", "als_cohort", "train_reject_rate"]


def matrix(g: pd.DataFrame):
    m = g.pivot(index="participant", columns="k", values="acc")
    return m.to_numpy(float), m.columns.to_numpy(float)


def share_usable_at(acc, ks, k, thr):
    """Share of participants (with data) usable at the smallest grid level >= k (or the largest level)."""
    j = np.flatnonzero(ks >= k - 1e-9)
    j = j[0] if j.size else len(ks) - 1
    col = acc[:, j]
    col = col[~np.isnan(col)]
    return (float(np.mean(col >= thr)) if col.size else np.nan), float(ks[j])


def log2err(pred, obs):
    return float(abs(np.log2(pred / obs)))


def analyse(pc, thr, cov, ext, rng, obs_tab):
    cohorts = sorted(pc.study.unique())
    obs = obs_tab.set_index("study").loc[cohorts]
    reached = obs[~obs.censored]
    mats = {s: matrix(pc[pc.study == s]) for s in cohorts}
    ref_rows, pil_rows = [], []
    for s in cohorts:
        acc, ks = mats[s]
        n = acc.shape[0]
        # within-cohort split-half reference
        for _ in range(ext["split_half_reps"]):
            p = rng.permutation(n); h = n // 2
            b1, c1, _ = _b80_from_matrix(acc[p[:h]], ks, thr, cov)
            b2, c2, _ = _b80_from_matrix(acc[p[h:2 * h]], ks, thr, cov)
            if not c1 and not c2:
                ref_rows.append(dict(study=s, rule="split_half", log2_err=log2err(b1, b2), abs_err=abs(b1 - b2)))
        # borrowed (leave-one-cohort-out median), scored on the whole cohort
        borrowed = float(reached.drop(index=s, errors="ignore").b80_chars.median())
        o = obs.loc[s]
        if not o.censored:
            ref_rows.append(dict(study=s, rule="borrowed_median", log2_err=log2err(borrowed, o.b80_chars),
                                 abs_err=abs(borrowed - o.b80_chars)))
        # pilot vs borrowed, scored on the same held-out participants
        for npil in ext["pilot_sizes"]:
            if n - npil < 3:
                continue
            for _ in range(ext["pilot_reps"]):
                p = rng.permutation(n)
                pil, ev = acc[p[:npil]], acc[p[npil:]]
                bp, cp, maxp = _b80_from_matrix(pil, ks, thr, cov)
                k_pilot = maxp if cp else bp          # not reached in pilot -> calibrate fully
                be, ce, maxe = _b80_from_matrix(ev, ks, thr, cov)
                for rule, k in (("pilot", k_pilot), ("borrowed_median", borrowed)):
                    if not np.isfinite(k):
                        continue
                    share, k_used = share_usable_at(ev, ks, k, thr)
                    err = log2err(k_used, be) if (not ce and not (rule == "pilot" and cp)) else np.nan
                    pil_rows.append(dict(study=s, n_pilot=npil, rule=rule, k_recommended=k_used,
                                         share_usable_eval=share, target_met=share >= cov,
                                         log2_err=err, eval_censored=ce))
    return pd.DataFrame(ref_rows), pd.DataFrame(pil_rows)


def ratio_ci(pc, thr, cov, reps, rng):
    boots = []
    for s, g in pc.groupby("study"):
        acc, ks = matrix(g)
        v = np.empty(reps)
        for i in range(reps):
            b, c, _ = _b80_from_matrix(acc[rng.integers(0, len(acc), len(acc))], ks, thr, cov)
            v[i] = np.nan if c else b
        boots.append(v)
    M = np.vstack(boots)
    r = np.nanmax(M, 0) / np.nanmin(M, 0)
    return dict(median=float(np.nanmedian(r)), ci_low=float(np.nanpercentile(r, 2.5)),
                ci_high=float(np.nanpercentile(r, 97.5)), p_ge5=float(np.nanmean(r >= 5)),
                p_ge10=float(np.nanmean(r >= 10)))


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    ext, b = cfg["transport_extension"], cfg["budget"]
    sm.K_CENTER = float(cfg["stats"]["k_center"])
    T = project_path(cfg["paths"]["tables"])
    rng = np.random.default_rng(ext["seed"])
    sweep = pd.read_csv(T / "sweep_results.csv", low_memory=False)
    valid = decoding_validity(sweep, b["decoding_tolerance"])
    vset = set(valid.loc[valid.valid_decoding, "study"])
    desc = cohort_descriptors(pd.read_csv(T / "inventory_records.csv"), pd.read_csv(T / "extraction_log.csv"),
                              cfg["data"].get("als_studies", []), sweep)
    summary = {"config_hash": cfg["_hash"]}
    specs = (("acc", "valid", pd.read_csv(T / "participant_curves.csv"), b["usable_accuracy"], "b80.csv"),
             ("auc", "all", pd.read_csv(T / "participant_curves_auc.csv"), b["auc_threshold"], "b80_auc.csv"))
    for metric, subset, pc, thr, obsf in specs:
        if subset == "valid":
            pc = pc[pc.study.isin(vset)]
        ref, pil = analyse(pc, thr, b["coverage"], ext, rng, pd.read_csv(T / obsf))
        ref.to_csv(T / f"transport_reference_{metric}.csv", index=False)
        pil.to_csv(T / f"pilot_{metric}.csv", index=False)
        # per-cohort medians first, then median across cohorts (each cohort counts once)
        def cohort_median(df, col):
            return float(df.groupby("study")[col].median().median())
        s = {"split_half_median_log2": cohort_median(ref[ref.rule == "split_half"], "log2_err"),
             "split_half_share_abs_ge2": float((ref[ref.rule == "split_half"].abs_err >= 2).mean()),
             "borrowed_median_log2": float(ref[ref.rule == "borrowed_median"].log2_err.median()),
             "n_borrowed_exact": int((ref.rule == "borrowed_median").sum())}
        for npil, g in pil.groupby("n_pilot"):
            for rule, h in g.groupby("rule"):
                s[f"n{npil}_{rule}"] = dict(
                    target_met=round(float(h.groupby("study").target_met.mean().mean()), 3),
                    median_log2=round(cohort_median(h.dropna(subset=["log2_err"]), "log2_err"), 3),
                    median_k=float(h.groupby("study").k_recommended.median().median()))
        s["ratio"] = ratio_ci(pc, thr, b["coverage"], ext["ratio_bootstrap_reps"], rng)
        # intercept meta-regression (accuracy / AUC level at k = k_center)
        fits = pd.read_csv(T / f"cohort_fits_{metric}_{subset}.csv")
        fits = fits[fits.status == "ok"].merge(desc, on="study", how="left")
        tau2 = random_effects(fits.intercept, fits.intercept_se)["tau2"]
        im = pd.DataFrame([{"descriptor": d, **meta_regression(fits.intercept, fits.intercept_se, tau2, fits[d])}
                           for d in DESCRIPTORS])
        im["p_holm"] = holm(im.p)
        im.to_csv(T / f"intercept_moderators_{metric}.csv", index=False)
        s["intercept_min_p_holm"] = float(im.p_holm.min())
        summary[metric] = s
    (T / "transport_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
