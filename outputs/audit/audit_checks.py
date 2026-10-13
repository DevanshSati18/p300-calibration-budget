"""Audit checks (13 Oct 2026): sensitivity analyses requested by the independent audit.
Reads only saved outputs (and a few cached epoch files for the covariance check).
Usage: python outputs/audit/audit_checks.py <part>   part in {slopes, budgets, cov}
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import p300budget.stats_models as sm
from p300budget.stats_models import fit_cohort_curve, random_effects, meta_regression, holm, cohort_descriptors, decoding_validity
from p300budget.budget import _b80_from_matrix
warnings.filterwarnings("ignore")
T = ROOT / "outputs/tables"; OUT = ROOT / "outputs/audit"
rng = np.random.default_rng(20261013)
sweep = pd.read_csv(T / "sweep_results.csv", low_memory=False)
pc_acc = pd.read_csv(T / "participant_curves.csv"); pc_auc = pd.read_csv(T / "participant_curves_auc.csv")
valid = decoding_validity(sweep, 0.10); VALID = set(valid[valid.valid_decoding].study)
pcv = pc_acc[pc_acc.study.isin(VALID)]
res = {}

def re_summary(df):
    r = random_effects(df.slope.values, df.se.values)
    return {k: round(float(r[k]), 4) for k in ("pooled", "tau", "I2", "ci_low", "ci_high", "pi_low", "pi_high") if k in r}

def slopes(pc, kind, kmax=None, boot=0):
    rows = []
    for s, g in pc.groupby("study"):
        if kmax: g = g[g.k <= kmax]
        f = fit_cohort_curve(g, kind)
        row = dict(study=s, slope=f["slope"], se=f["slope_se"], n_clusters=f["n_participants"], kmax=g.k.max())
        if boot:
            parts = g.participant.unique(); bs = []
            for _ in range(boot):
                pick = rng.choice(parts, len(parts))
                gb = pd.concat([g[g.participant == p].assign(participant=f"{p}_{i}") for i, p in enumerate(pick)])
                try: bs.append(fit_cohort_curve(gb, kind)["slope"])
                except Exception: pass
            row["se_boot"] = float(np.std(bs, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

part = sys.argv[1]
if part == "slopes":
    for name, pc, kind in (("acc_valid", pcv, "binomial"), ("auc_all", pc_auc, "linear")):
        d = slopes(pc, kind, boot=300)
        d["se_ratio_boot_over_cr"] = d.se_boot / d.se
        res[name] = dict(cohorts=d.round(4).to_dict("records"), re_cluster_se=re_summary(d),
                         re_bootstrap_se=re_summary(d.assign(se=d.se_boot)),
                         spearman_slope_vs_kmax=[round(x, 3) for x in stats.spearmanr(d.slope, d.kmax)])
        c = slopes(pc, kind, kmax=18)
        res[name]["re_common_k_le_18"] = re_summary(c)
        res[name]["slopes_common_k_le_18"] = c[["study", "slope", "se"]].round(4).to_dict("records")
    # artefact moderator influence (AUC slope, all cohorts) + Holm over 16 tests
    recs = pd.read_csv(T / "inventory_records.csv"); ext = pd.read_csv(T / "extraction_log.csv")
    desc = cohort_descriptors(recs, ext, ["B", "F", "L", "N"], sweep)
    d = slopes(pc_auc, "linear").merge(desc, on="study")
    tau2 = random_effects(d.slope, d.se)["tau2"]
    full = meta_regression(d.slope, d.se, tau2, d.train_reject_rate)
    loo = []
    for s in d.study:
        e = d[d.study != s]; t2 = random_effects(e.slope, e.se)["tau2"]
        m = meta_regression(e.slope, e.se, t2, e.train_reject_rate); loo.append(dict(drop=s, beta=round(m["beta"], 4), p=round(m["p"], 4)))
    e = d[~d.study.isin(["I", "J", "K", "M"])]; t2 = random_effects(e.slope, e.se)["tau2"]
    noijkm = meta_regression(e.slope, e.se, t2, e.train_reject_rate)
    res["artefact_auc"] = dict(full={k: round(v, 4) for k, v in full.items()}, loo=loo,
        excl_IJKM={k: round(v, 4) for k, v in noijkm.items()},
        spearman_all=[round(x, 4) for x in stats.spearmanr(d.slope, d.train_reject_rate)],
        reject_rate=d.set_index("study").train_reject_rate.round(3).to_dict())
    ps = pd.concat([pd.read_csv(T / "moderators_acc_valid.csv").p, pd.read_csv(T / "moderators_auc_all.csv").p], ignore_index=True)
    res["holm_16"] = dict(min_p_holm=round(float(holm(ps).min()), 4), n_sig=int((holm(ps) < .05).sum()))

if part == "budgets":
    def boot_b80(pc, thr, B=2000):
        out = {}
        for s, g in pc.groupby("study"):
            mat = g.pivot(index="participant", columns="k", values="acc"); ks = mat.columns.to_numpy(float); a = mat.to_numpy(float)
            v = np.empty(B)
            for i in range(B):
                b, c, _ = _b80_from_matrix(a[rng.integers(0, len(a), len(a))], ks, thr, 0.8); v[i] = np.inf if c else b
            out[s] = v
        return out
    def split_half(pc, thr, R=300):
        errs = []
        for s, g in pc.groupby("study"):
            mat = g.pivot(index="participant", columns="k", values="acc"); ks = mat.columns.to_numpy(float); a = mat.to_numpy(float)
            for _ in range(R):
                p = rng.permutation(len(a)); h = len(a) // 2
                b1, c1, _ = _b80_from_matrix(a[p[:h]], ks, thr, 0.8); b2, c2, _ = _b80_from_matrix(a[p[h:2*h]], ks, thr, 0.8)
                if not c1 and not c2: errs.append(dict(study=s, abs=abs(b1 - b2), log2=abs(np.log2(b1 / b2))))
        e = pd.DataFrame(errs)
        return dict(median_abs=float(e["abs"].median()), median_log2=round(float(e.log2.median()), 3),
                    share_ge2=round(float((e["abs"] >= 2).mean()), 3), n=len(e))
    for name, pc, thr, obsf in (("acc_valid", pcv, 0.70, "b80.csv"), ("auc_all", pc_auc, 0.75, "b80_auc.csv")):
        bt = boot_b80(pc, thr); M = np.vstack(list(bt.values()))
        fin = np.where(np.isfinite(M), M, np.nan)
        ratio = np.nanmax(fin, 0) / np.nanmin(fin, 0)
        obs = pd.read_csv(T / obsf); obs = obs[obs.study.isin(pc.study.unique())].set_index("study")
        reached = obs[~obs.censored]
        # LOCO baselines, errors in characters and log2
        rows = []
        for s in obs.index:
            others = reached.drop(index=s, errors="ignore").b80_chars
            o = obs.loc[s]
            preds = {"median": others.median(), "max_conservative": others.max(), "p75": others.quantile(.75)}
            for kname, pv in preds.items():
                if not o.censored: rows.append(dict(study=s, pred=kname, err=abs(pv - o.b80_chars), log2=abs(np.log2(pv / o.b80_chars)), under=pv < o.b80_chars, type="exact"))
                else: rows.append(dict(study=s, pred=kname, err=max(0, o.max_evaluable_k - pv), log2=np.nan, under=True, type="lower_bound"))
        L = pd.DataFrame(rows)
        loco = L.groupby("pred").agg(median_err=("err", "median"), median_log2_exact=("log2", "median"), share_under=("under", "mean"), share_err_ge2=("err", lambda x: (x >= 2).mean())).round(3)
        res[name] = dict(ratio_boot=dict(median=round(float(np.median(ratio)), 2), ci=[round(float(x), 2) for x in np.percentile(ratio, [2.5, 97.5])],
                                         p_ge2=round(float((ratio >= 2).mean()), 3), p_ge5=round(float((ratio >= 5).mean()), 3), p_ge10=round(float((ratio >= 10).mean()), 3)),
                         ratio_drop_one={s: round(float(reached.drop(index=s).b80_chars.max() / reached.drop(index=s).b80_chars.min()), 2) for s in reached.index},
                         split_half_within_cohort=split_half(pc, thr), loco_baselines=loco.reset_index().to_dict("records"),
                         loco_detail=L[L.pred == "median"].round(3).to_dict("records"))

if part == "cov":
    from sklearn.covariance import ledoit_wolf_shrinkage
    from sklearn.preprocessing import StandardScaler
    rows = []
    for coh in ["A", "B", "K", "Q", "S1", "J"]:
        files = sorted((ROOT / "data/processed/epochs" / coh).glob("*.npz"))[:2]
        for f in files:
            d = np.load(f)
            for k in (1, 4, int(d["t_train_char_s"].shape[0])):
                m = (d["c_train"] < k) & ~d["r_train"].astype(bool)
                X = StandardScaler().fit_transform(d["X_train"][m]); y = d["y_train"][m].astype(int)
                if y.sum() < 2 or (1 - y).sum() < 2: rows.append(dict(record=f.stem, k=k, n=int(m.sum()), n_target=int(y.sum()), note="too few")); continue
                pri = np.bincount(y) / len(y); S = 0; lams = []; mindiag = np.inf
                for c in (0, 1):
                    Z = X[y == c]; sd = Z.std(0); mindiag = min(mindiag, float((sd**2).min()))
                    lam = float(ledoit_wolf_shrinkage(Z / sd)); lams.append(lam)
                    Sc = np.cov(Z, rowvar=False, bias=True); S = S + pri[c] * ((1 - lam) * Sc + lam * np.diag(np.diag(Sc)))
                ev = np.linalg.eigvalsh(S)
                rows.append(dict(record=f.stem, k=k, n=int(m.sum()), n_target=int(y.sum()), d=X.shape[1], lam0=round(lams[0], 3), lam1=round(lams[1], 3),
                                 min_feature_var=float(f"{mindiag:.3g}"), min_eig=float(f"{ev.min():.3g}"), cond=float(f"{ev.max()/ev.min():.3g}")))
    res["cov"] = rows
(OUT / f"audit_{part}.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float)[:6000])
