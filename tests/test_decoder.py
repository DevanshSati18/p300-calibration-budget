"""Character decoding picks the character whose flashes score highest."""
import numpy as np
import pytest

from p300budget.decoder import decode_character, fit_decoder


def test_decode_picks_highest_mean():
    lit = np.eye(4, dtype=bool).repeat(3, axis=0)  # each char lit 3 times, in order
    scores = np.array([0, 0, 0, 1, 1, 1, 5, 5, 5, 0, 0, 0], dtype=float)
    assert decode_character(scores, lit, None) == 2


def test_repetition_truncation():
    lit = np.tile(np.eye(3, dtype=bool), (4, 1))  # 4 sequences of 3 flashes
    scores = np.zeros(12)
    scores[0] = 10.0        # char 0 wins in sequence 1
    scores[4:12:3] = 4.0    # char 1 wins over later sequences
    assert decode_character(scores, lit, 1) == 0
    assert decode_character(scores, lit, None) == 1
    with pytest.raises(ValueError):
        decode_character(scores, lit, 5)


def test_lda_learns_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    y = np.zeros(400, dtype=bool)
    y[:60] = True
    X[y, :5] += 1.5
    clf = fit_decoder(X, y)
    assert (clf.decision_function(X[y]).mean() > clf.decision_function(X[~y]).mean())
