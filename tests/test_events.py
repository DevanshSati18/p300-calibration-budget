"""Event parsing on a synthetic two-character recording."""
import numpy as np

from p300budget.events import extract_events, flashes_per_character, check_consistency
from p300budget.io_edf import MARKER_CHANNELS, EDFRecord, FileInfo, is_character_channel


def synthetic_record(n_grid=4, reps=3, isi=10, targets=(1, 3)):
    """Two character blocks; each flash lights one character (row/col free)."""
    seq = []
    for t in targets:
        seq.append(("pause", None, t))
        for _ in range(reps):
            for c in range(n_grid):
                seq.append(("flash", c, t))
        seq.append(("result", None, t))
    n = len(seq) * isi + 50
    mk = {c: np.zeros(n, dtype=np.int64) for c in MARKER_CHANNELS}
    flags = np.zeros((n_grid, n), dtype=bool)
    pos = 10
    for kind, c, t in seq:
        sl = slice(pos, pos + isi)
        mk["CurrentTarget"][sl] = t + 1
        if kind == "pause":
            mk["PhaseInSequence"][sl] = 1
        elif kind == "flash":
            mk["PhaseInSequence"][sl] = 2
            mk["StimulusBegin"][pos:pos + 3] = 1
            flags[c, pos:pos + 3] = True
            mk["StimulusType"][pos:pos + 3] = int(c == t)
        else:
            mk["PhaseInSequence"][sl] = 3
            mk["SelectedTarget"][sl] = t + 1
        pos += isi
    info = FileInfo(path=None, study="Z", participant="Z_01", session="SE001",
                    condition="CB", phase="Train", run=1)
    return EDFRecord(info=info, sfreq=256.0, eeg=np.zeros((0, n)), eeg_names=[],
                     markers=mk, char_flags=flags, char_labels=list("ABCD")[:n_grid])


def test_blocks_and_targets():
    ev = extract_events(synthetic_record())
    assert len(ev.onsets) == 2 * 3 * 4
    assert ev.target_char.tolist() == [1, 3]
    assert ev.selected_char.tolist() == [1, 3]
    assert set(np.unique(ev.char_id)) == {0, 1}


def test_repetitions_and_consistency():
    ev = extract_events(synthetic_record(reps=5))
    assert flashes_per_character(ev).tolist() == [5, 5]
    assert check_consistency(ev) == 1.0
    assert ev.is_target.sum() == 2 * 5


def test_character_channel_names():
    assert is_character_channel("A_1_1")
    assert is_character_channel("Sp_5_3")
    assert is_character_channel("?_6_1")
    assert not is_character_channel("EEG_Pz")
    assert not is_character_channel("StimulusType")
