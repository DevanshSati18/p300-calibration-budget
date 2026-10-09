"""Epoching, baseline and feature shapes."""
import numpy as np

from p300budget.preprocess import baseline_correct, epoch, make_features


def test_epoch_shapes_and_edges():
    eeg = np.random.default_rng(0).normal(size=(16, 2000))
    onsets = np.array([10, 500, 1990])  # first and last run past the edges
    ep, keep = epoch(eeg, onsets, 256.0, -0.2, 0.8)
    assert keep.tolist() == [False, True, False]
    assert ep.shape == (1, 16, 256)


def test_baseline_and_features():
    ep = np.ones((2, 16, 256)) * 7.0
    ep = baseline_correct(ep, 256.0, -0.2, (-0.2, 0.0))
    assert np.allclose(ep, 0.0)
    feats = make_features(ep, 256.0, -0.2, 0.0, 4)
    assert feats.shape == (2, 16 * 52)
