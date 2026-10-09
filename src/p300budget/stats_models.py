"""Cohort-level statistics: learning-curve fits, heterogeneity, moderators, transportability.

Two-stage approach (pre-specified fallback in PROPOSAL.md §6, used as primary):
  Stage 1: per cohort, binomial GLM  correct/decoded ~ log(k), participant-clustered SEs.
  Stage 2: DerSimonian–Laird random-effects meta-analysis of cohort intercepts and slopes;
           meta-regression of slopes on protocol descriptors (Holm-corrected).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .budget import _b80_from_matrix

K_CENTER = 4.0  # intercept = log-odds of a correct character at k = 4


# ---------------------------------------------------------------- stage 1
def fit_cohort_curve(pc: pd.DataFrame, kind: str = "binomial") -> dict:
    """Learning curve of one cohort on log(k), participant-clustered SEs.

    kind='binomial': correct/decoded ~ log k (logistic; character accuracy)
    kind='linear'  : AUC ~ log k (OLS; flash-level AUC)
    """
    d = pc[pc.n_decoded > 0] if kind == "binomial" else pc[pc.acc.notna()]
    x = np.log(d.k.to_numpy(float)) - np.log(K_CENTER)
    X = sm.add_constant(x)
    groups = pd.factorize(d.participant)[0]
    if kind == "binomial":
        endog = np.column_stack([d.n_correct, d.n_decoded - d.n_correct])
        model = sm.GLM(endog, X, family=sm.families.Binomial())
    else:
        model = sm.OLS(d.acc.to_numpy(float), X)
    clustered = len(np.unique(groups)) >= 2
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}) if clustered else model.fit()
    return dict(intercept=res.params[0], intercept_se=res.bse[0],
                slope=res.params[1], slope_se=res.bse[1],
                n_participants=int(len(np.unique(groups))), n_rows=int(len(d)))


def fit_all_cohorts(pc: pd.DataFrame, kind: str = "binomial") -> pd.DataFrame:
    rows = []
    for study, g in pc.groupby("study"):
        try:
            rows.append({"study": study, **fit_cohort_curve(g, kind), "status": "ok"})
        except Exception as exc:
            rows.append({"study": study, "status": f"failed: {str(exc)[:80]}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 2
def random_effects(est: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian–Laird pooling with Hartung–Knapp CI and a prediction interval."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    k = len(est)
    w = 1 / se**2
    fixed = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - fixed) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    wr = 1 / (se**2 + tau2)
    pooled = float(np.sum(wr * est) / np.sum(wr))
    out = dict(k=k, pooled=pooled, tau=float(np.sqrt(tau2)), tau2=float(tau2), Q=q,
               I2=float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0)
    if k >= 2:
        # Hartung–Knapp variance and t-based CI
        var_hk = np.sum(wr * (est - pooled) ** 2) / ((k - 1) * np.sum(wr))
        t = stats.t.ppf(0.975, k - 1)
        out.update(ci_low=pooled - t * np.sqrt(var_hk), ci_high=pooled + t * np.sqrt(var_hk),
                   p_Q=float(stats.chi2.sf(q, k - 1)))
    if k >= 3:
        t2 = stats.t.ppf(0.975, k - 2)
        half = t2 * np.sqrt(tau2 + 1 / np.sum(wr))
        out.update(pi_low=pooled - half, pi_high=pooled + half)
    return out


def meta_regression(est, se, tau2_null, x) -> dict:
    """Weighted regression of cohort estimates on one standardised moderator."""
    est, se, x = (np.asarray(v, float) for v in (est, se, x))
    ok = np.isfinite(est) & np.isfinite(se) & np.isfinite(x)
    est, se, x = est[ok], se[ok], x[ok]
    k = len(est)
    if k < 4 or np.std(x) == 0:
        return dict(k=k, beta=np.nan, se=np.nan, p=np.nan, tau2_resid=np.nan, share_explained=np.nan)
    z = (x - x.mean()) / x.std(ddof=1)
    X = sm.add_constant(z)
    # Method-of-moments residual tau^2
    w0 = 1 / se**2
    fit0 = sm.WLS(est, X, weights=w0).fit()
    q_res = float(np.sum(w0 * fit0.resid**2))
    P = np.diag(w0) - (w0[:, None] * X) @ np.linalg.inv(X.T @ (w0[:, None] * X)) @ (X.T * w0)
    tau2_res = max(0.0, (q_res - (k - 2)) / np.trace(P))
    w = 1 / (se**2 + tau2_res)
    fit = sm.WLS(est, X, weights=w).fit()
    # Knapp–Hartung: t with k-2 df, scale by residual variance
    s2 = max(1.0, float(np.sum(w * fit.resid**2) / (k - 2)))
    cov = np.linalg.inv(X.T @ (w[:, None] * X)) * s2
    b, bse = fit.params[1], np.sqrt(cov[1, 1])
    p = float(2 * stats.t.sf(abs(b / bse), k - 2))
    share = (1 - tau2_res / tau2_null) if tau2_null > 0 else np.nan
    return dict(k=k, beta=float(b), se=float(bse), p=p, tau2_resid=float(tau2_res),
                share_explained=float(max(0.0, share)) if np.isfinite(share) else np.nan)


def holm(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = np.isfinite(p).sum()
    adj = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return pd.Series(adj, index=pvals.index)


# ---------------------------------------------------------------- descriptors
def cohort_descriptors(records: pd.DataFrame, extraction: pd.DataFrame,
                       als_studies: list[str], sweep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Protocol descriptors per cohort, from the inventory and extraction log."""
    r = records[records.record_eligible]
    d = r.groupby("study").agg(
        grid_size=("n_grid", "median"),
        isi_ms=("isi_ms", "median"),
        test_flashes_per_char=("test_flashes_per_char", "median"),
        lit_per_flash=("lit_per_flash", "median"),
        share_rc=("condition", lambda s: float(np.mean(s.str.upper() == "RC"))),
    )
    e = extraction[extraction.status.isin(["ok", "cached"])]
    for col, out, how in (("median_train_char_s", "seconds_per_calib_char", "median"),
                          ("train_reject_rate", "train_reject_rate", "mean")):
        d[out] = e.groupby("study")[col].agg(how) if col in e.columns else np.nan
    if sweep is not None:  # more reliable than the extraction log
        s = sweep[(sweep.control == "none") & (sweep.status == "ok") & (sweep.k > 0)]
        d["seconds_per_calib_char"] = (s.calib_seconds / s.k).groupby(s.study).median()
        full = s[s.k == s.n_train_chars_total]
        rej = full.n_rejected_in_budget / (full.n_train_epochs + full.n_rejected_in_budget)
        d["train_reject_rate"] = rej.groupby(full.study).mean()
    d["als_cohort"] = d.index.isin(als_studies).astype(float)
    return d.reset_index()


# ---------------------------------------------------------------- RQ4
def _loco_error(pred, pred_cens, obs_b80, obs_cens, obs_max):
    """Absolute error in characters, honest about censoring."""
    if not pred_cens and not obs_cens:
        return abs(pred - obs_b80), "exact"
    if not pred_cens and obs_cens:
        return max(0.0, obs_max - pred), "lower_bound"  # true B80 is above obs_max
    return np.nan, "undefined"


def loco_b80(pc: pd.DataFrame, b80: pd.DataFrame, usable: float, coverage: float) -> pd.DataFrame:
    """Leave one cohort out. Two predictors of the withheld cohort's B80:
      pooled  (primary)   — B80 of all other cohorts' participants pooled together
      median  (secondary) — median B80 of the other cohorts that reached B80
    """
    obs_all = b80.set_index("study")
    rows = []
    for study in sorted(pc.study.unique()):
        rest = pc[pc.study != study]
        mat = rest.assign(pid=rest.study + ":" + rest.participant).pivot(
            index="pid", columns="k", values="acc")
        pred, cens, _ = _b80_from_matrix(mat.to_numpy(float), mat.columns.to_numpy(float),
                                         usable, coverage)
        others = obs_all.drop(index=study)
        others = others[~others.censored]
        pred_med = float(others.b80_chars.median()) if len(others) else np.nan
        obs = obs_all.loc[study]
        e1, t1 = _loco_error(pred, cens, obs.b80_chars, bool(obs.censored), obs.max_evaluable_k)
        e2, t2 = _loco_error(pred_med, not np.isfinite(pred_med), obs.b80_chars,
                             bool(obs.censored), obs.max_evaluable_k)
        rows.append(dict(study=study, observed_b80=obs.b80_chars,
                         observed_censored=bool(obs.censored), observed_max_k=obs.max_evaluable_k,
                         pred_pooled=pred, pred_pooled_censored=cens,
                         err_pooled=e1, err_pooled_type=t1,
                         pred_median=pred_med, err_median=e2, err_median_type=t2))
    out = pd.DataFrame(rows)
    out["err_pooled_ge_2"] = out.err_pooled >= 2
    out["err_median_ge_2"] = out.err_median >= 2
    return out


def decoding_validity(sweep: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Per cohort: offline accuracy at full calibration vs original online accuracy."""
    s = sweep[(sweep.control == "none") & (sweep.status == "ok")]
    full = s[s.k == s.n_train_chars_total]
    v = full.groupby("study").agg(offline_acc=("acc", "mean"), online_acc=("online_acc", "mean"))
    v["gap"] = v.offline_acc - v.online_acc
    v["valid_decoding"] = v.gap.abs() <= tolerance
    return v.reset_index()
