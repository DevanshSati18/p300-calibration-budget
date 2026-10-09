"""Step 01 — audit the archive: list every EDF, parse names, check headers.

Outputs
  outputs/tables/archive_files.csv   one row per EDF file
  outputs/tables/archive_summary.csv one row per study
Usage
  python scripts/01_audit_archive.py [--jobs 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, project_path  # noqa: E402
from p300budget.io_edf import MARKER_CHANNELS, parse_filename  # noqa: E402


def header_row(path: Path, eeg_channels: list[str]) -> dict:
    info = parse_filename(path)
    row = {"path": str(path), "file": path.name, "size_mb": path.stat().st_size / 1e6}
    if info is None:
        return {**row, "parsed": False}
    row.update(parsed=True, study=info.study, participant=info.participant,
               session=info.session, condition=info.condition, phase=info.phase, run=info.run)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
        names = set(raw.ch_names)
        row.update(
            sfreq=round(float(raw.info["sfreq"]), 3),
            duration_s=round(raw.n_times / raw.info["sfreq"], 2),
            n_channels=len(raw.ch_names),
            n_eeg=sum(c.startswith("EEG_") for c in raw.ch_names),
            has_montage16=all(c in names for c in eeg_channels),
            has_markers=all(c in names for c in MARKER_CHANNELS),
            read_ok=True, error="",
        )
    except Exception as exc:  # keep going; record the failure
        row.update(read_ok=False, error=str(exc)[:200])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    root = project_path(cfg["paths"]["data_root"])
    out = project_path(cfg["paths"]["tables"])
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("*.edf"))
    print(f"Found {len(files)} EDF files under {root}")
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(header_row)(f, cfg["data"]["eeg_channels"]) for f in files
    )
    df = pd.DataFrame(rows)
    df["config_hash"] = cfg["_hash"]
    df.to_csv(out / "archive_files.csv", index=False)

    ok = df[df["parsed"] & df["read_ok"].fillna(False)]
    summary = ok.groupby("study").agg(
        participants=("participant", "nunique"),
        sessions=("session", lambda s: s.groupby(ok.loc[s.index, "participant"]).nunique().sum()),
        conditions=("condition", lambda s: ",".join(sorted(s.unique()))),
        train_files=("phase", lambda s: (s == "Train").sum()),
        test_files=("phase", lambda s: (s == "Test").sum()),
        sfreq=("sfreq", lambda s: ",".join(map(str, sorted(s.unique())))),
        all_montage16=("has_montage16", "all"),
        all_markers=("has_markers", "all"),
        total_mb=("size_mb", "sum"),
    ).reset_index()
    summary["excluded_upfront"] = summary["study"].isin(cfg["data"]["excluded_studies"])
    summary.to_csv(out / "archive_summary.csv", index=False)

    print(summary.to_string(index=False))
    bad = df[~df["parsed"] | ~df["read_ok"].fillna(False)]
    print(f"\nUnparsed or unreadable files: {len(bad)}")
    if len(bad):
        print(bad[["file", "error"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
