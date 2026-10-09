"""Filter continuous EEG, cut flash-locked epochs, baseline, decimate, flag artefacts."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(eeg: np.ndarray, sfreq: float, band: tuple[float, float], order: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass on (n_ch, n_samples)."""
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1)


def epoch(eeg: np.ndarray, onsets: np.ndarray, sfreq: float, tmin: float, tmax: float):
    """Cut epochs (n_epochs, n_ch, n_times). Returns epochs and a mask of onsets kept
    (epochs running past the recording edges are dropped)."""
    s0 = int(round(tmin * sfreq))
    s1 = int(round(tmax * sfreq))
    keep = (onsets + s0 >= 0) & (onsets + s1 <= eeg.shape[1])
    idx = onsets[keep][:, None] + np.arange(s0, s1)[None, :]
    ep = eeg[:, idx].transpose(1, 0, 2)
    return ep, keep


def baseline_correct(ep: np.ndarray, sfreq: float, tmin: float, base: tuple[float, float]) -> np.ndarray:
    b0 = int(round((base[0] - tmin) * sfreq))
    b1 = int(round((base[1] - tmin) * sfreq))
    return ep - ep[:, :, b0:b1].mean(axis=2, keepdims=True)


def make_features(ep: np.ndarray, sfreq: float, tmin: float, feat_tmin: float, decim: int) -> np.ndarray:
    """Keep the post-stimulus window, decimate, flatten to (n_epochs, n_ch * n_times)."""
    start = int(round((feat_tmin - tmin) * sfreq))
    x = ep[:, :, start::decim]
    return x.reshape(x.shape[0], -1).astype(np.float32)


def preprocess_file(eeg: np.ndarray, onsets: np.ndarray, sfreq: float, pp: dict):
    """Full chain for one file. Returns (features, kept_mask, rejected_mask_over_kept)."""
    filt = bandpass(eeg, sfreq, tuple(pp["bandpass_hz"]), pp["filter_order"])
    ep, keep = epoch(filt, onsets, sfreq, pp["epoch_tmin_s"], pp["epoch_tmax_s"])
    ep = baseline_correct(ep, sfreq, pp["epoch_tmin_s"], tuple(pp["baseline_s"]))
    rejected = np.abs(ep).max(axis=(1, 2)) > pp["reject_uv"]
    feats = make_features(ep, sfreq, pp["epoch_tmin_s"], pp["feature_tmin_s"], pp["decim"])
    return feats, keep, rejected
