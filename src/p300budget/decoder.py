"""Shrinkage LDA flash classifier and offline character decoding."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_decoder():
    """Standardise on calibration data, then Ledoit-Wolf shrinkage LDA."""
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def fit_decoder(X: np.ndarray, y: np.ndarray):
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Calibration data must contain both target and non-target flashes.")
    clf = make_decoder()
    clf.fit(X, y.astype(int))
    return clf


def decode_character(scores: np.ndarray, lit: np.ndarray, repetitions: int | None) -> int:
    """Pick the character with the highest mean score over the flashes that lit it.

    scores: (n_flash,) classifier scores for one character block, in flash order.
    lit:    (n_flash, n_grid) bool.
    repetitions: stop once every grid character has been lit this many times
                 (None = use all flashes).
    """
    if repetitions is not None:
        counts = np.cumsum(lit, axis=0).min(axis=1)
        reached = np.flatnonzero(counts >= repetitions)
        if reached.size == 0:
            raise ValueError("Block has fewer flashes per character than requested repetitions.")
        n_use = reached[0] + 1
        scores, lit = scores[:n_use], lit[:n_use]
    total = scores @ lit.astype(float)
    n = lit.sum(axis=0)
    mean = np.divide(total, n, out=np.full(lit.shape[1], -np.inf), where=n > 0)
    return int(np.argmax(mean))
