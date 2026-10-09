"""Learning curves and the calibration budget B80.

Definitions (PROPOSAL.md §6):
- Participant accuracy at k = pooled correct / decoded over that participant's
  records (sessions × conditions) that support budget k.
- Usable = participant accuracy >= usable_accuracy (0.70).
- A budget k is evaluable for a cohort when at least `coverage` (80%) of the
  cohort's participants have data at k.
- B80 = smallest evaluable k at which >= coverage of participants are usable.
  Right-censored ("not reached") if no evaluable k qualifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = {
    "acc": ("n_correct", "n_decoded"),
    "acc_r_secondary": ("n_correct_r_secondary", "n_decoded_r_secondary"),
}


def participant_curves(sweep: pd.DataFrame, metric: str = "acc") -> pd.DataFrame:
    """Participant × k accuracy, pooled over records. Insufficient-data rows count as 0 correct."""
    num, den = METRICS[metric]
    s = sweep[(sweep.control == "none") & sweep.status.isin(["ok", "insufficient_data"])].copy()
    # A record that could not be trained at k decodes nothing correctly; give it its test count.
    s.loc[s.status == "insufficient_data", den] = np.nan
    fill = s.groupby("record")[den].transform("max")
    s[den] = s[den].fillna(fill)
    s[num] = s[num].fillna(0)
    g = s.groupby(["study", "participant", "k"]).agg(
        n_correct=(num, "sum"), n_decoded=(den, "sum"),
        n_records=("record", "nunique"), calib_seconds=("calib_seconds", "median"),
        auc=("auc", "mean"),
    ).reset_index()
    g["acc"] = g.n_correct / g.n_decoded
    return g


def cohort_curves(pc: pd.DataFrame, usable: float) -> pd.DataFrame:
    """Cohort × k learning curve summary."""
    n_part = pc.groupby("study").participant.nunique().rename("n_participants_cohort")
    g = pc.assign(usable=pc.acc >= usable).groupby(["study", "k"]).agg(
        n_participants=("participant", "nunique"),
        mean_acc=("acc", "mean"), sd_acc=("acc", "std"), median_acc=("acc", "median"),
        share_usable=("usable", "mean"), calib_seconds=("calib_seconds", "median"),
        mean_auc=("auc", "mean"),
    ).reset_index().join(n_part, on="study")
    g["data_coverage"] = g.n_participants / g.n_participants_cohort
    return g


def _b80_from_matrix(acc: np.ndarray, ks: np.ndarray, usable: float, coverage: float):
    """acc: (n_participants, n_k) with NaN where a participant has no data at k.

    Returns (B80 or nan, censored flag, largest evaluable k).
    """
    has = ~np.isnan(acc)
    data_cov = has.mean(axis=0)
    with np.errstate(invalid="ignore"):
        share = np.nanmean(np.where(has, acc >= usable, np.nan), axis=0)
    evaluable = data_cov >= coverage
    if not evaluable.any():
        return np.nan, True, np.nan
    max_eval = ks[evaluable].max()
    hit = evaluable & (share >= coverage)
    if hit.any():
        return float(ks[np.flatnonzero(hit)[0]]), False, float(max_eval)
    return np.nan, True, float(max_eval)


def compute_b80(pc: pd.DataFrame, usable: float, coverage: float,
                n_boot: int, seed: int) -> pd.DataFrame:
    """B80 per cohort with a participant-cluster bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for study, g in pc.groupby("study"):
        mat = g.pivot(index="participant", columns="k", values="acc")
        ks = mat.columns.to_numpy(dtype=float)
        acc = mat.to_numpy(dtype=float)
        b80, cens, max_eval = _b80_from_matrix(acc, ks, usable, coverage)
        secs = g.groupby("k").calib_seconds.median()
        boot = np.empty(n_boot)
        boot_cens = 0
        n = acc.shape[0]
        for i in range(n_boot):
            bb, bc, bmax = _b80_from_matrix(acc[rng.integers(0, n, n)], ks, usable, coverage)
            if bc:
                boot_cens += 1
                boot[i] = np.inf  # not reached: above every evaluable budget
            else:
                boot[i] = bb
        if np.isfinite(boot).any():
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:  # every resample censored
            lo, hi = np.nan, np.inf
        rows.append(dict(
            study=study, n_participants=n,
            b80_chars=b80, censored=cens, max_evaluable_k=max_eval,
            b80_seconds=float(secs.get(b80, np.nan)) if not cens else np.nan,
            ci_low=lo, ci_high=hi if np.isfinite(hi) else np.nan,
            ci_high_censored=not np.isfinite(hi),
            share_boot_censored=boot_cens / n_boot,
            share_usable_full=float((g.loc[g.groupby("participant").k.idxmax(), "acc"] >= usable).mean()),
        ))
    return pd.DataFrame(rows)
