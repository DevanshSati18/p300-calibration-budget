"""Calibration-budget sweep for one record.

For each budget k, train the decoder on the first k calibration characters
(recording order, artefact epochs removed) and decode every test character.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .decoder import decode_character, fit_decoder


def load_record(path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def budget_levels(grid: list[int], n_train_chars: int) -> list[int]:
    """Grid values the record supports, plus the record's full calibration."""
    ks = [k for k in grid if k <= n_train_chars]
    if n_train_chars not in ks:
        ks.append(n_train_chars)
    return ks


def resolve_repetitions(value) -> int | None:
    """Config value -> decode_character argument ('as_run' = all flashes)."""
    return None if value in (None, "as_run") else int(value)


def decode_all(scores: np.ndarray, d: dict, repetitions: int | None):
    """Decode every test character. Returns (correct flags, n characters skipped)."""
    c, lit, target = d["c_test"], d["lit_test"], d["target_test"]
    correct, skipped = [], 0
    for b in range(len(target)):
        m = c == b
        if not m.any():
            skipped += 1
            continue
        try:
            pred = decode_character(scores[m], lit[m], repetitions)
        except ValueError:  # block never reaches the requested repetitions
            skipped += 1
            continue
        correct.append(pred == target[b])
    return np.array(correct, dtype=bool), skipped


def run_record(d: dict, grid: list[int], rep_primary, rep_secondary,
               shuffle_seed: int | None = None) -> list[dict]:
    """One row per budget k (plus a shuffled-label control row at full budget)."""
    n_train_chars = int(d["t_train_char_s"].shape[0])
    rp, rs = resolve_repetitions(rep_primary), resolve_repetitions(rep_secondary)
    rows = []
    levels = budget_levels(grid, n_train_chars)
    for k in levels:
        rows.append(_one_budget(d, k, rp, rs, shuffle=False))
    if shuffle_seed is not None:
        row = _one_budget(d, n_train_chars, rp, rs, shuffle=True, seed=shuffle_seed)
        row["control"] = "shuffled_labels"
        rows.append(row)
    return rows


def _one_budget(d: dict, k: int, rp, rs, shuffle: bool, seed: int = 0) -> dict:
    mask = (d["c_train"] < k) & ~d["r_train"].astype(bool)
    y = d["y_train"][mask].astype(bool)
    row = dict(k=k, control="none",
               calib_seconds=float(d["t_train_char_s"][:k].sum()),
               n_train_epochs=int(mask.sum()), n_train_targets=int(y.sum()),
               n_rejected_in_budget=int(((d["c_train"] < k) & d["r_train"].astype(bool)).sum()))
    if y.sum() < 1 or y.sum() == len(y):
        return {**row, "status": "insufficient_data", "acc": np.nan, "n_correct": 0,
                "n_decoded": 0, "acc_r_secondary": np.nan, "n_correct_r_secondary": 0,
                "n_decoded_r_secondary": 0, "n_skipped_r_secondary": 0, "auc": np.nan}
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    clf = fit_decoder(d["X_train"][mask], y)
    scores = clf.decision_function(d["X_test"])
    ok_p, _ = decode_all(scores, d, rp)
    ok_s, skip_s = decode_all(scores, d, rs)
    yt = d["y_test"].astype(bool)
    auc = roc_auc_score(yt, scores) if 0 < yt.sum() < len(yt) else np.nan
    return {**row, "status": "ok",
            "acc": float(ok_p.mean()) if len(ok_p) else np.nan,
            "n_correct": int(ok_p.sum()), "n_decoded": int(len(ok_p)),
            "acc_r_secondary": float(ok_s.mean()) if len(ok_s) else np.nan,
            "n_correct_r_secondary": int(ok_s.sum()), "n_decoded_r_secondary": int(len(ok_s)),
            "n_skipped_r_secondary": int(skip_s), "auc": float(auc)}
