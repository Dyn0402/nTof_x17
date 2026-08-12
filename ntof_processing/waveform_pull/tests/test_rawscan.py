"""Unit tests for the raw scan and the window arithmetic.

A synthetic raw file is built bank by bank, so the expected answer is known
exactly rather than inferred from a real file where it would not be.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ntof_processing.waveform_pull import config as C          # noqa: E402
from ntof_processing.waveform_pull.rawscan import (            # noqa: E402
    ScanStats, merge_intervals, overlaps, scan_file)
from ntof_processing.waveform_pull.windows import build, reconcile  # noqa: E402


# --------------------------------------------------------------- synthetic raw
def bank(tag: bytes, payload: bytes) -> bytes:
    assert len(payload) % 4 == 0
    return struct.pack('<4sIII', tag, 1, 0, len(payload) // 4) + payload


def eveh(bunch: int, event: int = 0) -> bytes:
    w = [0, bunch, 224572, event, 0, 0, 0, 0, 0, 0]
    return bank(b'EVEH', struct.pack('<10I', *w))


def acqc(det: str, chan: int, blocks) -> bytes:
    """blocks: [(start, samples ndarray)]"""
    p = det.encode('ascii')[:4].ljust(4, b' ') + struct.pack('<II', chan, 0)
    for start, s in blocks:
        s = np.asarray(s, dtype=C.SAMPLE_DTYPE)
        p += struct.pack('<QQ', start, len(s)) + s.tobytes()
    if len(p) % 4:
        p += b'\0' * (4 - len(p) % 4)
    return bank(b'ACQC', p)


def write_raw(path, events):
    """events: [(bunch, [(det, chan, [(start, samples)])])]"""
    with open(path, 'wb') as f:
        f.write(bank(b'RCTR', struct.pack('<8I', 224572, 0, 0, 0, 0, 0, 0, 0)))
        for i, (bunch, chans) in enumerate(events):
            f.write(eveh(bunch, i))
            for det, chan, blocks in chans:
                f.write(acqc(det, chan, blocks))
    return path


# ------------------------------------------------------------------ intervals
def test_merge_intervals_coalesces_and_sorts():
    # 0-60 and 50-120 touch, 100-200 joins them -> one interval; 500-600 stays
    iv = merge_intervals([100, 0, 50, 500], [200, 60, 120, 600])
    assert iv.tolist() == [[0, 200], [500, 600]]


def test_merge_intervals_empty():
    assert merge_intervals([], []).shape == (0, 2)


def test_overlaps_edges():
    iv = merge_intervals([100], [200])
    blo = np.array([0, 99, 100, 150, 199, 200, 300])
    bhi = blo + 1
    # [lo, hi) semantics: a block touching 100..199 overlaps, 200 does not
    assert overlaps(iv, blo, bhi).tolist() == \
        [False, False, True, True, True, False, False]


def test_overlaps_block_straddling_interval():
    iv = merge_intervals([1000], [1010])
    # one long block that starts well before and ends well after
    assert overlaps(iv, np.array([0]), np.array([5000])).tolist() == [True]


# ----------------------------------------------------------------- the scanner
def test_scan_keeps_only_overlapping_blocks(tmp_path):
    s = np.arange(50, dtype=np.int16)
    p = write_raw(tmp_path / 'r.raw', [
        (7, [('WALA', 1, [(0, np.zeros(30, np.int16)),        # flash block
                          (1000, s), (5000, s), (9000, s)])]),
    ])
    # tof space: block at start=1000 covers [741, 791); 5000 -> [4741, 4791)
    win = {7: {'WALA': {1: merge_intervals([4700], [4800])}}}
    got = list(scan_file(p, win))
    assert len(got) == 1
    bunch, det, chan, tof0, samples = got[0]
    assert (bunch, det, chan, tof0) == (7, 'WALA', 1, 5000 - C.PRE_SAMPLES)
    assert samples.tolist() == s.tolist()


def test_scan_skips_unwanted_bunch_and_detector(tmp_path):
    s = np.arange(10, dtype=np.int16)
    p = write_raw(tmp_path / 'r.raw', [
        (7, [('WALA', 1, [(1000, s)]), ('PSSA', 2, [(1000, s)])]),
        (8, [('WALA', 1, [(1000, s)])]),
    ])
    win = {7: {'PSSA': {2: merge_intervals([0], [10_000])}}}
    st = ScanStats()
    got = list(scan_file(p, win, stats=st))
    assert [(g[0], g[1], g[2]) for g in got] == [(7, 'PSSA', 2)]
    assert st.bunches == {7, 8}          # both seen, one wanted
    assert st.banks_read == 1            # only the PSSA payload was read


def test_scan_flash_block_opt_in(tmp_path):
    p = write_raw(tmp_path / 'r.raw', [
        (7, [('LIQA', 1, [(0, np.arange(100, dtype=np.int16))])])])
    win = {7: {'LIQA': {1: merge_intervals([0], [100])}}}
    assert len(list(scan_file(p, win))) == 0
    got = list(scan_file(p, win, keep_flash=True))
    assert len(got) == 1 and got[0][3] == 0      # tof0 == 0, no pre-samples


def test_scan_truncated_tail_is_clean(tmp_path):
    s = np.arange(40, dtype=np.int16)
    p = write_raw(tmp_path / 'r.raw', [(7, [('WALA', 1, [(1000, s)])])])
    data = p.read_bytes()
    (tmp_path / 'cut.raw').write_bytes(data[:len(data) - 30])
    win = {7: {'WALA': {1: merge_intervals([0], [10_000])}}}
    got = list(scan_file(tmp_path / 'cut.raw', win))
    assert got == [] or len(got[0][4]) < 40      # no exception, no bad data


def test_scan_rejects_garbage(tmp_path):
    (tmp_path / 'bad.raw').write_bytes(b'XXXX' + b'\0' * 32)
    with pytest.raises(ValueError, match='bad bank'):
        list(scan_file(tmp_path / 'bad.raw', {}))


# ------------------------------------------------------------------- windows
def _events(bunches, preds, flash=None):
    return {'bunch': np.array(bunches, np.int64),
            't_pred_ns': np.array(preds, float),
            'is_flash': np.array(flash if flash is not None
                                 else [0] * len(bunches), np.int8),
            'eventId': np.arange(len(bunches), dtype=np.uint64),
            'matched': np.ones(len(bunches), np.int8)}


def test_build_centres_on_pred_plus_tflash():
    tf = {(C.DET_CODE['WALA'], 1, 5): 1000.0}
    win = build(_events([5], [20_000.0]), tf, window_ns=100,
                with_control=False, dets=('WALA',), extra_dets=())
    assert win[5]['WALA'][1].tolist() == [[20_900, 21_101]]


def test_build_adds_control_window():
    tf = {(C.DET_CODE['WALA'], 1, 5): 0.0}
    win = build(_events([5], [1_000.0]), tf, window_ns=10,
                control_shift_ns=100_000, dets=('WALA',), extra_dets=())
    assert win[5]['WALA'][1].tolist() == [[990, 1011], [100_990, 101_011]]


def test_build_drops_flash_triggers():
    tf = {(C.DET_CODE['WALA'], 1, 5): 0.0}
    win = build(_events([5, 5], [1_000.0, 2_000.0], flash=[1, 0]), tf,
                window_ns=10, with_control=False, dets=('WALA',), extra_dets=())
    assert win[5]['WALA'][1].tolist() == [[1_990, 2_011]]


def test_build_pkup_gets_union_of_all_detectors():
    tf = {(C.DET_CODE['WALA'], 1, 5): 0.0, (C.DET_CODE['LIQA'], 1, 5): 5_000.0}
    win = build(_events([5], [1_000.0]), tf, window_ns=10, with_control=False,
                dets=('WALA', 'LIQA'), extra_dets=('PKUP',))
    assert win[5]['PKUP'][1].tolist() == [[990, 1011], [5_990, 6_011]]


def test_build_skips_detector_without_tflash():
    win = build(_events([5], [1_000.0]), {(C.DET_CODE['WALA'], 1, 5): 0.0},
                window_ns=10, with_control=False, dets=C.SCINT_DETS,
                extra_dets=())
    assert set(win[5]) == {'WALA'}
    assert set(win[5]['WALA']) == {1}          # only the channel that had one


# -------------------------------------------------------------------- tflash
def test_tflash_from_slim_uses_control_only_channels():
    """Regression: a (det, bunch) whose only hits are CONTROL hits must still
    get a flash time. Excluding control hits lost LIQC in bunches 398/399 of
    the reference pair entirely -- no window, no waveforms, and the only sign
    was four uncovered hits in the closure check."""
    from ntof_processing.waveform_pull.windows import tflash_from_slim
    events = {'eventId': np.array([0], np.uint64), 'bunch': np.array([398]),
              't_pred_ns': np.array([1_000.0]), 'is_flash': np.array([0]),
              'matched': np.array([1])}
    hits = {'eventId': np.array([0], np.uint64), 'det': np.array([10]),
            'detn': np.array([1]),
            'tof': np.array([1_000.0 + 100_000.0 + 11_600.0 + 5.0]),
            'dt_ns': np.array([5.0]), 'is_control': np.array([1])}
    tf = tflash_from_slim(hits, events, control_shift_ns=100_000.0)
    assert (10, 1, 398) in tf
    assert tf[(10, 1, 398)][0] == pytest.approx(11_600.0)


def test_fill_gaps_uses_the_nearest_bunch_not_the_median():
    """The run median inherits every outlier (LIQA/LIQB carry ~120 ns of them,
    PSS the known ~350 ns mis-tags); the neighbour inherits only the step."""
    from ntof_processing.waveform_pull.windows import fill_gaps
    table = {(0, 1, 1): (100.0, 0), (0, 1, 2): (101.0, 0), (0, 1, 50): (900.0, 0)}
    out, rep = fill_gaps(table, [1, 2, 3, 50], dets=('WALA',))
    assert out[(0, 1, 3)] == (101.0, 2)       # neighbour is bunch 2, not median
    assert rep['n_filled_from_neighbour_bunch'] == 1
    # WALA has 8 channels; 2-8 never appeared, so they are filled as dead
    assert rep['n_filled_dead_channel'] == 7 * 4


def test_build_reports_skipped_det_bunch():
    rep = {}
    build(_events([5], [1_000.0]), {(C.DET_CODE['WALA'], 1, 5): 0.0},
          window_ns=10, with_control=False, dets=('WALA', 'LIQA'),
          extra_dets=(), report=rep)
    # WALA channels 2-8 have no tflash, and LIQA's single channel none either
    assert rep['skipped_det_bunch'] == {'WALA': 7, 'LIQA': 1}
    assert rep['n_skipped_det_bunch'] == 8


def test_reconcile_raises_on_disagreement():
    with pytest.raises(ValueError, match='tflash disagrees'):
        reconcile({(0, 1, 5): 100.0}, {(0, 1, 5): (450.0, 12, 0.1)})


def test_reconcile_prefers_processed_and_fills_gaps():
    table, rep = reconcile({(0, 1, 5): 100.0}, {(0, 1, 5): (100.4, 9, 0.2),
                                                (1, 1, 5): (7.0, 3, 0.0)})
    assert table == {(0, 1, 5): (100.0, 0), (1, 1, 5): (7.0, 1)}
    assert rep['n_cross_checked'] == 1 and rep['n_slim_only'] == 1
