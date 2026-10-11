"""Step 04 — run the calibration-budget sweep on every cached record.

For each record and each budget k (config budget.grid, capped at the record's
calibration length, plus its full length) trains shrinkage LDA on the first k
calibration characters and decodes all test characters:
  primary   — as-run test length (decoding.repetitions)
  secondary — fixed R (decoding.secondary_repetitions)
plus one shuffled-label control at full budget per record.

Outputs
  data/interim/sweep/<study>/<record>.csv   per-record results (resume cache)
  outputs/tables/sweep_results.csv          all records × budgets
Usage
  python scripts/04_run_sweep.py [--jobs 4] [--studies A B] [--overwrite] [--dev]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, output_name, project_path, select_dev  # noqa: E402
from p300budget.sweep import load_record, run_record  # noqa: E402


def sweep_one(row: dict, cfg: dict, cache_dir: Path, overwrite: bool) -> pd.DataFrame:
    out = cache_dir / row["study"] / f"{row['record']}.csv"
    if out.exists() and not overwrite:
        cached = pd.read_csv(out)
        if not (cached.status == "failed").any():  # never reuse a failed result
            return cached
    meta = {k: row[k] for k in ("study", "participant", "session", "condition", "record")}
    try:
        path = Path(row["path"])
        if not path.exists():  # logs store absolute paths; rebuild for clones / moved repos
            path = project_path(cfg["paths"]["processed"]) / "epochs" / row["study"] / f"{row['record']}.npz"
        d = load_record(path)
        rows = run_record(d, cfg["budget"]["grid"], cfg["decoding"]["repetitions"],
                          cfg["decoding"]["secondary_repetitions"],
                          shuffle_seed=cfg["stats"]["seed"])
        df = pd.DataFrame([{**meta, **r} for r in rows])
        df["n_grid"] = int(d["n_grid"])
        df["n_train_chars_total"] = int(d["t_train_char_s"].shape[0])
        df["online_acc"] = float((d["selected_test"] == d["target_test"]).mean())
    except Exception as exc:
        df = pd.DataFrame([{**meta, "status": "failed", "error": str(exc)[:300]}])
    df["config_hash"] = cfg["_hash"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dev", action="store_true", help="quick run on a small subset (config: dev)")
    args = ap.parse_args()

    cfg = load_config()
    tables = project_path(cfg["paths"]["tables"])
    cache_dir = project_path(cfg["paths"]["interim"]) / "sweep"

    log_path = tables / "extraction_log.csv"
    if not log_path.exists() and args.dev:
        log_path = tables / "extraction_log_dev.csv"
    log = pd.read_csv(log_path)
    todo = log[log.status.isin(["ok", "cached"])]
    if args.studies:
        todo = todo[todo.study.isin(args.studies)]
    if args.dev:
        todo = select_dev(todo, cfg)
        print("DEV MODE: " + ", ".join(todo.record))
    print(f"Sweeping {len(todo)} records from {todo.study.nunique()} cohorts")

    parts = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(sweep_one)(row, cfg, cache_dir, args.overwrite) for row in todo.to_dict("records")
    )
    res = pd.concat(parts, ignore_index=True)
    name = "sweep_results.csv" if not args.studies else f"sweep_results_{'_'.join(args.studies)}.csv"
    out = tables / output_name(name, args.dev)
    res.to_csv(out, index=False)
    print(f"Saved {out}")

    failed = res[res.status == "failed"]
    if len(failed):
        print(f"\nFailed records: {failed.record.nunique()}")
        print(failed[["record", "error"]].drop_duplicates().head(10).to_string(index=False))

    if "control" not in res or not (res.status == "ok").any():
        print("No successful records to summarise.")
        return
    main_rows = res[(res.control == "none") & (res.status == "ok")]
    full = main_rows[main_rows.k == main_rows.n_train_chars_total]
    ctrl = res[res.control == "shuffled_labels"]
    summ = full.groupby("study").agg(records=("record", "nunique"),
                                     offline_acc_full=("acc", "mean"),
                                     online_acc=("online_acc", "mean"),
                                     auc_full=("auc", "mean"))
    for k in (1, 4, 8):
        summ[f"acc_k{k}"] = main_rows[main_rows.k == k].groupby("study").acc.mean()
    summ["shuffled_acc"] = ctrl.groupby("study").acc.mean()
    summ["chance"] = 1 / full.groupby("study").n_grid.first()
    print("\nSanity check: offline at full calibration should be close to online;")
    print("shuffled-label accuracy should be near chance.\n")
    print(summ.round(3).to_string())
    insuff = (res.status == "insufficient_data").sum()
    print(f"\nBudget rows with too little clean calibration data to train: {insuff}")


if __name__ == "__main__":
    main()
