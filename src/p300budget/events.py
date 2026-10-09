"""Turn marker channels into flashes and characters (selections).

Conventions verified on Studies A and B (9 Oct 2026):
- A flash starts at each rising edge of StimulusBegin.
- PhaseInSequence == 2 marks the flashing period of one character.
- CurrentTarget is a 1-based index into the character channels.
- StimulusType == 1 exactly when the CurrentTarget character is lit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_edf import EDFRecord

FLASH_PHASE = 2


@dataclass
class FileEvents:
    onsets: np.ndarray        # (n_flash,) sample index of each flash onset
    is_target: np.ndarray     # (n_flash,) bool, from StimulusType
    char_id: np.ndarray       # (n_flash,) int, 0-based character within the file
    lit: np.ndarray           # (n_flash, n_chars) bool, characters lit by the flash
    target_char: np.ndarray   # (n_char_in_file,) 0-based intended character, -1 if unknown
    selected_char: np.ndarray  # (n_char_in_file,) 0-based online selection, -1 if none
    n_grid: int


def rising_edges(x: np.ndarray) -> np.ndarray:
    on = (np.asarray(x) > 0).astype(np.int8)
    return np.flatnonzero(np.diff(on, prepend=0) == 1)


def extract_events(rec: EDFRecord) -> FileEvents:
    mk = rec.markers
    onsets = rising_edges(mk["StimulusBegin"])
    phase = mk["PhaseInSequence"]

    # Character blocks: maximal runs of PhaseInSequence == FLASH_PHASE.
    in_flash = (phase == FLASH_PHASE).astype(np.int8)
    starts = np.flatnonzero(np.diff(in_flash, prepend=0) == 1)
    ends = np.flatnonzero(np.diff(in_flash, append=0) == -1) + 1  # exclusive

    # Assign each flash to the block containing its onset.
    char_id = np.searchsorted(starts, onsets, side="right") - 1
    valid = (char_id >= 0) & (onsets < ends[np.clip(char_id, 0, len(ends) - 1)])
    onsets, char_id = onsets[valid], char_id[valid]

    n_blocks = len(starts)
    target_char = np.full(n_blocks, -1, dtype=np.int64)
    selected_char = np.full(n_blocks, -1, dtype=np.int64)
    ct = mk["CurrentTarget"]
    sel = mk["SelectedTarget"]
    for b in range(n_blocks):
        seg = ct[starts[b]:ends[b]]
        seg = seg[seg > 0]
        if seg.size:
            target_char[b] = int(np.bincount(seg).argmax()) - 1
        # Online selection appears after the flashing block, before the next one.
        nxt = starts[b + 1] if b + 1 < n_blocks else len(sel)
        s = sel[ends[b]:nxt]
        s = s[s > 0]
        if s.size:
            selected_char[b] = int(np.bincount(s).argmax()) - 1

    lit = rec.char_flags[:, onsets].T
    return FileEvents(
        onsets=onsets,
        is_target=mk["StimulusType"][onsets] > 0,
        char_id=char_id,
        lit=lit,
        target_char=target_char,
        selected_char=selected_char,
        n_grid=rec.char_flags.shape[0],
    )


def flashes_per_character(ev: FileEvents) -> np.ndarray:
    """For each character block, the minimum number of times any grid character was lit.

    This is the largest R (repetitions) that block can support.
    """
    out = []
    for b in range(len(ev.target_char)):
        m = ev.char_id == b
        out.append(int(ev.lit[m].sum(0).min()) if m.any() else 0)
    return np.array(out, dtype=np.int64)


def check_consistency(ev: FileEvents) -> float:
    """Share of flashes where StimulusType agrees with 'intended character lit'."""
    tc = ev.target_char[ev.char_id]
    known = tc >= 0
    if not known.any():
        return float("nan")
    lit_t = ev.lit[np.flatnonzero(known), tc[known]]
    return float(np.mean(lit_t == ev.is_target[known]))
