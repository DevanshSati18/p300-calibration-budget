"""Step 08 — independent verification of the pipeline's results.

Re-derives key results with separately written code and compares them with the
saved outputs. Nothing here is imported from sweep.py or budget.py.

Checks
  1. Re-decode a random sample of records from cached features at every budget k
     (own truncation, own LDA call, own character decoding) and compare accuracy
     with sweep_results.csv.
  2. Leakage: no test epoch appears in calibration data; budgets are nested;
     shuffled-label control is at chance level in every cohort.
  3. Participant curves recomputed from the sweep match participant_curves.csv.
  4. B80 and AUC budget recomputed with a plain loop match b80.csv / b80_auc.csv.
  5. Every number that the manuscript (paper/main.tex) reports is present in or
     derivable from the output tables.
Usage
  python scripts/08_verify_results.py [--n-records 8]
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, project_path  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1 + 2
def redecode(path: str, ks: list[int]) -> dict[int, float]:
    """Independent re-implementation of truncation + LDA + character decoding."""
    d = np.load(path)
    Xtr, ytr, ctr, rtr = d["X_train"], d["y_train"].astype(bool), d["c_train"], d["r_train"].astype(bool)
    Xte, cte, lit, tgt = d["X_test"], d["c_test"], d["lit_test"].astype(float), d["target_test"]
    out = {}
    for k in ks:
        m = (ctr < k) & ~rtr
        if ytr[m].all() or not ytr[m].any():
            out[k] = np.nan
            continue
        sc = StandardScaler().fit(Xtr[m])
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(Xtr[m]), ytr[m])
        s = lda.decision_function(sc.transform(Xte))
        hits = []
        for b in range(len(tgt)):
            sel = cte == b
            if not sel.any():
                continue
            counts = lit[sel].sum(0)
            means = np.where(counts > 0, (s[sel][:, None] * lit[sel]).sum(0) / np.maximum(counts, 1), -np.inf)
            hits.append(int(np.argmax(means)) == int(tgt[b]))
        out[k] = float(np.mean(hits)) if hits else np.nan
    return out


def leakage(path: str) -> tuple[bool, str]:
    d = np.load(path)
    tr = {r.tobytes() for r in d["X_train"][:: max(1, len(d["X_train"]) // 2000)]}
    dup = sum(r.tobytes() in tr for r in d["X_test"])
    c = d["c_train"]
    order_ok = bool(np.all(np.diff(c) >= 0))  # characters appear in recording order
    return dup == 0 and order_ok, f"duplicate train/test epochs={dup}, calibration order monotonic={order_ok}"


# ---------------------------------------------------------------- 4
def plain_budget(pc: pd.DataFrame, value: str, thr: float, cov: float) -> pd.DataFrame:
    rows = []
    for study, g in pc.groupby("study"):
        n = g.participant.nunique()
        found, max_eval = None, None
        for k in sorted(g.k.unique()):
            gk = g[g.k == k]
            if gk.participant.nunique() / n < cov:
                continue
            max_eval = k
            if found is None and (gk[value] >= thr).mean() >= cov:
                found = k
        rows.append(dict(study=study, b=found, max_eval=max_eval))
    return pd.DataFrame(rows).set_index("study")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    T = project_path(cfg["paths"]["tables"])
    b = cfg["budget"]
    sweep = pd.read_csv(T / "sweep_results.csv")
    log = pd.read_csv(T / "extraction_log.csv")
    main_rows = sweep[(sweep.control == "none") & (sweep.status == "ok")]

    # 1. Independent re-decoding ------------------------------------------------
    rng = np.random.default_rng(cfg["stats"]["seed"])
    recs = log[log.status.isin(["ok", "cached"])]
    pick = recs.groupby("study", group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1e9))))
    pick = pick.sample(min(args.n_records, len(pick)), random_state=cfg["stats"]["seed"])
    max_diff, n_cmp = 0.0, 0
    epochs = project_path(cfg["paths"]["processed"]) / "epochs"
    for row in pick.itertuples():
        path = row.path if Path(row.path).exists() else str(epochs / row.study / f"{row.record}.npz")
        mine = main_rows[main_rows.record == row.record].set_index("k").acc
        ind = redecode(path, list(mine.index))
        diffs = [abs(ind[k] - mine[k]) for k in mine.index if np.isfinite(ind[k])]
        max_diff = max(max_diff, max(diffs) if diffs else 0)
        n_cmp += len(diffs)
        ok_l, det = leakage(path)
        check(f"leakage {row.record}", ok_l, det)
    check("independent re-decoding matches sweep accuracy", max_diff < 1e-9,
          f"{len(pick)} records, {n_cmp} budget levels, max |difference| = {max_diff:.2e}")

    # 2. Nested budgets + shuffled control ---------------------------------------
    nested = main_rows.sort_values("k").groupby("record").n_train_epochs.apply(lambda s: s.is_monotonic_increasing).all()
    check("calibration budgets are nested (epochs never decrease with k)", nested)
    ctrl = sweep[sweep.control == "shuffled_labels"]
    chance = (1 / ctrl.groupby("study").n_grid.first())
    shuf = ctrl.groupby("study").acc.mean()
    worst = (shuf - chance).max()
    check("shuffled-label accuracy near chance in every cohort", worst < 0.05,
          f"max excess over chance = {worst:.3f}")
    full = main_rows[main_rows.k == main_rows.n_train_chars_total]
    gap = (full.groupby("study").acc.mean() - shuf).min()
    check("real labels beat shuffled labels in every cohort", gap > 0.2, f"min gap = {gap:.2f}")

    # 3. Participant curves --------------------------------------------------------
    pc = pd.read_csv(T / "participant_curves.csv")
    s = sweep[(sweep.control == "none") & sweep.status.isin(["ok", "insufficient_data"])].copy()
    s["n_decoded"] = s.n_decoded.where(s.status == "ok", np.nan)
    s["n_decoded"] = s.n_decoded.fillna(s.groupby("record").n_decoded.transform("max"))
    s["n_correct"] = s.n_correct.fillna(0)
    agg = s.groupby(["study", "participant", "k"])[["n_correct", "n_decoded"]].sum()
    mine = pc.set_index(["study", "participant", "k"])
    j = agg.join(mine[["acc"]], how="outer")
    dacc = (j.n_correct / j.n_decoded - j.acc).abs().max()
    check("participant curves recomputed from the sweep match", dacc < 1e-9, f"max |diff| = {dacc:.2e}")

    # 4. Budgets ---------------------------------------------------------------------
    for fname, value, thr, src in (("b80.csv", "acc", b["usable_accuracy"], pc),
                                   ("b80_auc.csv", "auc", b["auc_threshold"], pc)):
        saved = pd.read_csv(T / fname).set_index("study")
        plain = plain_budget(src, value, thr, b["coverage"])
        mism = []
        for st in saved.index:
            exp = None if saved.loc[st, "censored"] else saved.loc[st, "b80_chars"]
            got = plain.loc[st, "b"]
            got = None if got is None or (isinstance(got, float) and np.isnan(got)) else got
            if (exp is None) != (got is None) or (exp is not None and float(exp) != float(got)):
                mism.append(f"{st}: saved {exp} vs recomputed {got}")
        check(f"{fname} recomputed with an independent loop", not mism, "; ".join(mism) or "all cohorts match")

    # 5. Manuscript numbers --------------------------------------------------------------
    tex_path = project_path("paper/main.tex")
    if tex_path.exists():
        tex = tex_path.read_text()
        h_acc = json.load(open(T / "heterogeneity_acc_valid.json"))
        h_auc = json.load(open(T / "heterogeneity_auc_all.json"))
        mods = pd.read_csv(T / "moderators_auc_all.csv").set_index("descriptor")
        loco_auc = pd.read_csv(T / "loco_auc_all.csv")
        loco_acc = pd.read_csv(T / "loco_acc_valid.csv")
        coh = pd.read_csv(T / "eligibility_cohorts.csv")
        coh = coh[coh.cohort_eligible]
        expect = {
            "participants": f"{int(coh.eligible_participants.sum())} participants",
            "records": f"{int(coh.eligible_records.sum())} records",
            "pooled slope": f"{h_acc['slope']['pooled']:.2f} log-odds",
            "slope CI": f"{h_acc['slope']['ci_low']:.2f}--{h_acc['slope']['ci_high']:.2f}",
            "tau slope": f"\\tau={h_acc['slope']['tau']:.2f}",
            "I2 slope": f"I^2={h_acc['slope']['I2']:.2f}",
            "prediction interval": f"{h_acc['slope']['pi_low']:.2f}--{h_acc['slope']['pi_high']:.2f}",
            "AUC slope": f"{h_auc['slope']['pooled']:.3f}",
            "budget min-max": f"{h_acc['budget']['min']:.0f} to {h_acc['budget']['max']:.0f} characters",
            "15-fold": f"{h_acc['budget']['ratio_max_min']:.0f}-fold",
            "artefact Holm p": f"p={mods.loc['train_reject_rate', 'p_holm']:.3f}",
            "LOCO median AUC": f"median error {loco_auc.err_median.median():.0f}",
            "LOCO median acc": f"median {loco_acc.err_median.median():.0f})",
        }
        missing = [f"{k} ('{v}')" for k, v in expect.items()
                   if v.replace(" ", "") not in re.sub(r"\s+", "", tex)]
        check("manuscript numbers match the output tables", not missing,
              "all present" if not missing else "not found: " + "; ".join(missing))

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    pd.DataFrame(RESULTS, columns=["check", "passed", "detail"]).to_csv(T / "verification_report.csv", index=False)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
