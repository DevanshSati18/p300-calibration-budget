"""Step 02 — build the session × condition inventory and apply eligibility.

Reads only marker and character channels (no EEG). For every record
(participant, session, condition) it counts calibration characters, test
characters, the repetitions each test block supports (R_max), grid size,
flash timing, and label consistency.

Outputs
  outputs/tables/inventory_files.csv    one row per EDF file
  outputs/tables/inventory_records.csv  one row per participant-session-condition
  outputs/tables/eligibility_cohorts.csv one row per study
Usage
  python scripts/02_build_inventory.py [--jobs 4] [--studies A B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, project_path  # noqa: E402
from p300budget.events import check_consistency, extract_events, flashes_per_character  # noqa: E402
from p300budget.io_edf import parse_filename, read_edf  # noqa: E402


def file_row(path: Path) -> dict:
    info = parse_filename(path)
    row = dict(path=str(path), file=path.name, study=info.study, participant=info.participant,
               session=info.session, condition=info.condition, phase=info.phase, run=info.run)
    try:
        rec = read_edf(path, load_eeg=False)
        ev = extract_events(rec)
        fpc = flashes_per_character(ev)
        isi = np.diff(ev.onsets)
        blocks = len(ev.target_char)
        row.update(
            ok=True, error="",
            sfreq=rec.sfreq,
            n_grid=ev.n_grid,
            n_flashes=len(ev.onsets),
            n_chars=blocks,
            n_chars_known_target=int((ev.target_char >= 0).sum()),
            r_min=int(fpc.min()) if blocks else 0,
            r_max=int(fpc.max()) if blocks else 0,
            lit_per_flash=float(ev.lit.sum(1).mean()) if len(ev.onsets) else np.nan,
            target_rate=float(ev.is_target.mean()) if len(ev.onsets) else np.nan,
            isi_ms=float(np.median(isi) / rec.sfreq * 1000) if len(isi) else np.nan,
            flashes_per_char=float(len(ev.onsets) / blocks) if blocks else np.nan,
            consistency=check_consistency(ev),
            online_acc=float(np.mean(ev.selected_char[ev.target_char >= 0]
                                     == ev.target_char[ev.target_char >= 0]))
            if (ev.target_char >= 0).any() else np.nan,
            fake_feedback=bool((rec.markers["FakeFeedback"] > 0).any()),
        )
    except Exception as exc:
        row.update(ok=False, error=str(exc)[:200])
    return row


def build_records(files: pd.DataFrame) -> pd.DataFrame:
    keys = ["study", "participant", "session", "condition"]
    ok = files[files["ok"]]
    rows = []
    for key, g in ok.groupby(keys):
        tr, te = g[g.phase == "Train"], g[g.phase == "Test"]
        rows.append(dict(
            zip(keys, key),
            has_train=len(tr) > 0, has_test=len(te) > 0,
            n_train_files=len(tr), n_test_files=len(te),
            n_train_chars=int(tr.n_chars_known_target.sum()),
            n_test_chars=int(te.n_chars_known_target.sum()),
            n_grid=int(g.n_grid.mode().iat[0]),
            train_r_min=int(tr.r_min.min()) if len(tr) else 0,
            test_r_min=int(te.r_min.min()) if len(te) else 0,
            isi_ms=float(g.isi_ms.median()),
            lit_per_flash=float(g.lit_per_flash.mean()),
            train_flashes_per_char=float(tr.flashes_per_char.mean()) if len(tr) else np.nan,
            test_flashes_per_char=float(te.flashes_per_char.mean()) if len(te) else np.nan,
            consistency=float(g.consistency.min()),
            online_acc=float(np.average(te.online_acc, weights=te.n_chars)) if len(te) else np.nan,
            fake_feedback_any=bool(te.fake_feedback.any()) if len(te) else False,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--studies", nargs="*", default=None, help="limit to these studies, e.g. A B")
    args = ap.parse_args()

    cfg = load_config()
    root = project_path(cfg["paths"]["data_root"])
    out = project_path(cfg["paths"]["tables"])
    out.mkdir(parents=True, exist_ok=True)
    el = cfg["eligibility"]
    grid = cfg["budget"]["grid"]

    files = [f for f in sorted(root.rglob("*.edf")) if parse_filename(f)]
    if args.studies:
        files = [f for f in files if parse_filename(f).study in args.studies]
    print(f"Reading markers from {len(files)} files")
    fdf = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=5)(delayed(file_row)(f) for f in files))
    fdf["config_hash"] = cfg["_hash"]
    fdf.to_csv(out / "inventory_files.csv", index=False)

    rec = build_records(fdf)
    # Record-level eligibility: own Train and Test, labels consistent, no forced feedback.
    rec["k_levels"] = rec.n_train_chars.apply(lambda n: sum(k <= n for k in grid))
    rec["record_eligible"] = (
        rec.has_train & rec.has_test & (rec.n_test_chars > 0)
        & (rec.consistency >= el["min_label_consistency"]) & ~rec.fake_feedback_any
        & (rec.k_levels >= el["min_k_levels"])
    )
    rec["config_hash"] = cfg["_hash"]
    rec.to_csv(out / "inventory_records.csv", index=False)

    e = rec[rec.record_eligible]
    coh = rec.groupby("study").agg(
        records=("participant", "size"),
        participants=("participant", "nunique"),
    ).join(e.groupby("study").agg(
        eligible_records=("participant", "size"),
        eligible_participants=("participant", "nunique"),
        conditions=("condition", lambda s: ",".join(sorted(s.unique()))),
        grid=("n_grid", lambda s: ",".join(map(str, sorted(s.unique())))),
        median_train_chars=("n_train_chars", "median"),
        median_test_chars=("n_test_chars", "median"),
        min_test_r=("test_r_min", "min"),
        isi_ms=("isi_ms", "median"),
        online_acc=("online_acc", "mean"),
    )).reset_index()
    coh["excluded_upfront"] = coh.study.isin(cfg["data"]["excluded_studies"])
    coh["cohort_eligible"] = (~coh.excluded_upfront
                              & (coh.eligible_participants.fillna(0) >= el["min_participants"]))
    coh.to_csv(out / "eligibility_cohorts.csv", index=False)

    print(coh.to_string(index=False))
    elig = coh[coh.cohort_eligible]
    print(f"\nEligible cohorts: {len(elig)}  -> {', '.join(elig.study)}")
    if len(elig):
        r = int(elig.min_test_r.min())
        print(f"Largest R supported by every eligible cohort's Test data: {r}")
        print("Write this into config/frozen.yaml -> decoding.repetitions")
    print(f"Failed files: {(~fdf.ok).sum()}")


if __name__ == "__main__":
    main()
