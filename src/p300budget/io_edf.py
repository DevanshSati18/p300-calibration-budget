"""Read bigP3BCI EDF+ files and parse their file names.

File layout: StudyX/<participant>/<session>/<Train|Test>/<condition>/
             <participant>_<session>_<condition>_<Train|Test><run>.edf
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

FNAME_RE = re.compile(
    r"^(?P<participant>(?P<study>[A-Z]\d?)_\d+)_(?P<session>SE\d+)_"
    r"(?P<condition>.+?)_(?P<phase>Train|Test)(?P<run>\d+)\.edf$"
)

MARKER_CHANNELS = [
    "StimulusType", "SelectedTarget", "SelectedRow", "SelectedColumn",
    "PhaseInSequence", "StimulusBegin", "StimulusCode", "CurrentTarget",
    "FakeFeedback", "DisplayResults",
]


@dataclass
class FileInfo:
    path: Path
    study: str
    participant: str
    session: str
    condition: str
    phase: str
    run: int


def parse_filename(path: Path | str) -> FileInfo | None:
    path = Path(path)
    m = FNAME_RE.match(path.name)
    if m is None:
        return None
    return FileInfo(
        path=path,
        study=m["study"],
        participant=m["participant"],
        session=m["session"],
        condition=m["condition"],
        phase=m["phase"],
        run=int(m["run"]),
    )


def is_character_channel(name: str) -> bool:
    """Character flag channels look like 'A_1_1', 'Sp_5_3', '?_6_1' (label_row_col)."""
    if name.startswith("EEG_") or name in MARKER_CHANNELS:
        return False
    parts = name.rsplit("_", 2)
    return len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()


@dataclass
class EDFRecord:
    info: FileInfo
    sfreq: float
    eeg: np.ndarray            # (n_eeg, n_samples) in microvolts
    eeg_names: list[str]
    markers: dict[str, np.ndarray]   # marker channel -> (n_samples,) int
    char_flags: np.ndarray     # (n_chars, n_samples) bool: character lit
    char_labels: list[str]     # label part of each character channel


def read_edf(path: Path | str, eeg_channels: list[str] | None = None,
             load_eeg: bool = True) -> EDFRecord:
    """Read one EDF. EEG is returned in microvolts; markers as integers."""
    path = Path(path)
    info = parse_filename(path)
    if info is None:
        raise ValueError(f"Unrecognised file name: {path.name}")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    names = raw.ch_names
    char_names = [c for c in names if is_character_channel(c)]
    missing = [c for c in MARKER_CHANNELS if c not in names]
    if missing:
        raise ValueError(f"{path.name}: missing marker channels {missing}")

    if eeg_channels is None:
        eeg_channels = [c for c in names if c.startswith("EEG_")]
    absent = [c for c in eeg_channels if c not in names]
    if absent:
        raise ValueError(f"{path.name}: missing EEG channels {absent}")

    picks = (eeg_channels if load_eeg else []) + MARKER_CHANNELS + char_names
    data = raw.get_data(picks=picks)
    n_eeg = len(eeg_channels) if load_eeg else 0
    eeg = data[:n_eeg] * 1e6 if load_eeg else np.empty((0, data.shape[1]))
    mk = np.rint(data[n_eeg:n_eeg + len(MARKER_CHANNELS)]).astype(np.int64)
    flags = data[n_eeg + len(MARKER_CHANNELS):] > 0.5
    return EDFRecord(
        info=info,
        sfreq=float(raw.info["sfreq"]),
        eeg=eeg,
        eeg_names=list(eeg_channels) if load_eeg else [],
        markers={c: mk[i] for i, c in enumerate(MARKER_CHANNELS)},
        char_flags=flags,
        char_labels=[c.rsplit("_", 2)[0] for c in char_names],
    )
