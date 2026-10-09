"""Step 03 — extract and cache features for every eligible record.

For each eligible participant × session × condition, reads its Train and Test
EDFs (in run order), filters, epochs, baselines and decimates the 16 shared
channels, and saves one .npz per record. Later steps never touch EDFs again.

Saved arrays per record (data/processed/epochs/<study>/<record>.npz)
  X_train  (n, 832) float32   calibration features, recording order
  y_train  (n,) bool          target flash
  c_train  (n,) int           calibration character index (0 = first character)
  r_train  (n,) bool          epoch exceeds the artefact threshold
  X_test, y_test, c_test, r_test   same for Test
  lit_test (n_test, n_grid) bool   characters lit by each test flash
  target_test   (n_test_chars,) intended character per test character
  selected_test (n_test_chars,) online selection per test character
  t_train_char_s (n_train_chars,) seconds each calibration character took

Outputs
  data/processed/epochs/...           one .npz per record
  outputs/tables/extraction_log.csv   one row per record
Usage
  python scripts/03_extract_epochs.py [--jobs 4] [--studies A B] [--overwrite] [--dev]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, output_name, project_path, select_dev  # noqa: E402
from p300budget.events import extract_events  # noqa: E402
from p300budget.io_edf import read_edf  # noqa: E402
from p300budget.preprocess import preprocess_file  # noqa: E402


def load_phase(paths: list[str], cfg: dict) -> dict:
    """Preprocess and concatenate one phase's files, keeping recording order."""
    pp, chans = cfg["preprocess"], cfg["data"]["eeg_channels"]
    X, y, c, r, lit, tgt, sel, tchar = [], [], [], [], [], [], [], []
    offset, n_grid = 0, None
    for p in paths:
        rec = read_edf(p, chans)
        ev = extract_events(rec)
        feats, keep, rej = preprocess_file(rec.eeg, ev.onsets, rec.sfreq, pp)
        # Only characters with a known intended target are usable.
        known = ev.target_char >= 0
        cid = ev.char_id[keep]
        ok = known[cid]
        # Renumber known characters consecutively across files.
        new_id = np.cumsum(known) - 1
        X.append(feats[ok])
        y.append(ev.is_target[keep][ok])
        c.append(new_id[cid[ok]] + offset)
        r.append(rej[ok])
        lit.append(ev.lit[keep][ok])
        tgt.append(ev.target_char[known])
        sel.append(ev.selected_char[known])
        # Duration of each character: first to last flash onset, plus one ISI.
        on = ev.onsets[keep][ok]
        for b in np.unique(cid[ok]):
            ob = on[cid[ok] == b]
            isi = np.median(np.diff(ob)) if len(ob) > 1 else 0
            tchar.append((ob[-1] - ob[0] + isi) / rec.sfreq)
        offset += int(known.sum())
        n_grid = ev.n_grid if n_grid is None else n_grid
        if ev.n_grid != n_grid:
            raise ValueError(f"grid size changes within phase: {p}")
    return dict(
        X=np.vstack(X).astype(np.float32), y=np.concatenate(y), c=np.concatenate(c),
        r=np.concatenate(r), lit=np.vstack(lit), target=np.concatenate(tgt),
        selected=np.concatenate(sel), tchar=np.array(tchar), n_grid=n_grid,
    )


def stats_from_npz(path: Path) -> dict:
    """Summary statistics read back from a cached record (used for the log)."""
    d = np.load(path)
    return {"n_grid": int(d["n_grid"]),
            "n_train_chars": int(d["t_train_char_s"].shape[0]),
            "n_test_chars": int(d["target_test"].shape[0]),
            "n_train_flashes": int(d["y_train"].shape[0]),
            "n_test_flashes": int(d["y_test"].shape[0]),
            "train_reject_rate": float(d["r_train"].mean()),
            "test_reject_rate": float(d["r_test"].mean()),
            "median_train_char_s": float(np.median(d["t_train_char_s"])),
            "online_acc": float(np.mean(d["selected_test"] == d["target_test"]))}


def process_record(row: dict, files: pd.DataFrame, cfg: dict, out_dir: Path, overwrite: bool) -> dict:
    rid = f"{row['participant']}_{row['session']}_{row['condition']}"
    out = out_dir / row["study"] / f"{rid}.npz"
    log = dict(study=row["study"], participant=row["participant"], session=row["session"],
               condition=row["condition"], record=rid, path=str(out))
    if out.exists() and not overwrite:
        return {**log, "status": "cached", "error": "", **stats_from_npz(out)}
    try:
        f = files[(files.participant == row["participant"]) & (files.session == row["session"])
                  & (files.condition == row["condition"])].sort_values("run")
        tr = load_phase(f[f.phase == "Train"].path.tolist(), cfg)
        te = load_phase(f[f.phase == "Test"].path.tolist(), cfg)
        if tr["n_grid"] != te["n_grid"]:
            raise ValueError("grid size differs between Train and Test")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 X_train=tr["X"], y_train=tr["y"], c_train=tr["c"], r_train=tr["r"],
                 t_train_char_s=tr["tchar"],
                 X_test=te["X"], y_test=te["y"], c_test=te["c"], r_test=te["r"],
                 lit_test=te["lit"], target_test=te["target"], selected_test=te["selected"],
                 n_grid=tr["n_grid"], config_hash=cfg["_hash"])
        return {**log, "status": "ok", "error": "",
                "n_grid": tr["n_grid"],
                "n_train_chars": len(tr["target"]), "n_test_chars": len(te["target"]),
                "n_train_flashes": len(tr["y"]), "n_test_flashes": len(te["y"]),
                "train_reject_rate": float(tr["r"].mean()), "test_reject_rate": float(te["r"].mean()),
                "median_train_char_s": float(np.median(tr["tchar"])),
                "online_acc": float(np.mean(te["selected"] == te["target"]))}
    except Exception as exc:
        return {**log, "status": "failed", "error": str(exc)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--studies", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dev", action="store_true", help="quick run on a small subset (config: dev)")
    args = ap.parse_args()

    cfg = load_config()
    tables = project_path(cfg["paths"]["tables"])
    out_dir = project_path(cfg["paths"]["processed"]) / "epochs"

    recs = pd.read_csv(tables / "inventory_records.csv")
    coh = pd.read_csv(tables / "eligibility_cohorts.csv")
    files = pd.read_csv(tables / "inventory_files.csv")
    eligible = set(coh.loc[coh.cohort_eligible, "study"])
    todo = recs[recs.record_eligible & recs.study.isin(eligible)]
    if args.studies:
        todo = todo[todo.study.isin(args.studies)]
    if args.dev:
        todo = select_dev(todo.assign(record=todo.participant + "_" + todo.session + "_" + todo.condition), cfg)
        print("DEV MODE: " + ", ".join(todo.record))
    print(f"Extracting {len(todo)} records from {todo.study.nunique()} cohorts")

    logs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process_record)(row, files, cfg, out_dir, args.overwrite)
        for row in todo.to_dict("records")
    )
    log = pd.DataFrame(logs)
    log["config_hash"] = cfg["_hash"]
    log_path = tables / output_name("extraction_log.csv", args.dev)
    if log_path.exists() and args.studies and not args.dev:  # partial run: merge with earlier log
        old = pd.read_csv(log_path)
        log = pd.concat([old[~old.record.isin(log.record)], log], ignore_index=True)
    log.to_csv(log_path, index=False)

    print(log.status.value_counts().to_string())
    done = log[log.status.isin(["ok", "cached"])]
    if len(done):
        print(done.groupby("study").agg(records=("record", "size"),
                                        train_chars=("n_train_chars", "median"),
                                        test_chars=("n_test_chars", "median"),
                                        reject=("train_reject_rate", "mean"),
                                        online_acc=("online_acc", "mean")).round(3).to_string())
    failed = log[log.status == "failed"]
    if len(failed):
        print("\nFailed records:")
        print(failed[["record", "error"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
