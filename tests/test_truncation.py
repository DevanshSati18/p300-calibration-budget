"""Calibration truncation must take the first k characters in recording order only."""
import numpy as np


def first_k_mask(char_index: np.ndarray, k: int) -> np.ndarray:
    """Reference rule used by the sweep: keep flashes from characters 0..k-1."""
    return char_index < k


def test_first_k_is_chronological():
    char_index = np.repeat(np.arange(10), 12)
    m = first_k_mask(char_index, 3)
    assert m.sum() == 36
    assert m[:36].all() and not m[36:].any()


def test_masks_are_nested():
    char_index = np.repeat(np.arange(20), 5)
    prev = np.zeros_like(char_index, dtype=bool)
    for k in [1, 2, 4, 8, 16]:
        cur = first_k_mask(char_index, k)
        assert (prev <= cur).all()  # every smaller budget is a subset of a larger one
        prev = cur
