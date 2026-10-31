"""SPS cache: a bad (oscillating / dead) channel never seeds a window (rerun/sps_audit, 2026-10-30).

Y ch 372 (158.34 mm) oscillates by +-400-900 ADC in some sub-runs.  The brightest-strip seed used to pick it, so 43-46 %
of the run71 RAW Y windows sat on the oscillation instead of the track.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'rerun', 'model_audit'))
import sps_build_cache as sbc                                   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW                      # noqa: E402


def _zs_event(track_mm=(185.0, 230.0), osc=(372,), osc_amp=1500.0, ns=32):
    """ZS-style samples (256 + ADC): a 5-strip track per view and a big swing on the oscillating channels."""
    t = np.arange(ns)
    ch, sm, am = [], [], []
    for view, x0 in zip(('x', 'y'), track_mm):
        chs = np.flatnonzero(VIEW == view)
        for c in chs[np.abs(POSITION_MM[chs] - x0) < 2.0]:
            a = 800 * np.exp(-0.5 * ((t - 12) / 3) ** 2) * np.exp(-abs(POSITION_MM[c] - x0))
            keep = a > 20
            ch += [c] * keep.sum(); sm += list(t[keep]); am += list(256 + a[keep])
    for c in osc:
        ch += [c] * ns; sm += list(t); am += list(256 + osc_amp * np.sin(t / 2.0))
    return np.array(ch), np.array(sm), np.array(am, np.float32)


def test_bad_channels_in_seed_mask():
    assert sbc.SEED_BAD[372] and sbc.SEED_BAD[510]
    assert abs(POSITION_MM[372] - 158.34) < 0.01 and VIEW[372] == 'y'


def test_oscillating_channel_does_not_seed():
    ch, sm, am = _zs_event()
    rec = sbc.event_record(ch, sm, am, raw=False, nsmp=32)
    assert rec is not None
    assert rec['seed_ch_y'] != 372
    assert abs(POSITION_MM[rec['seed_ch_y']] - 230.0) < 1.0           # the track, not 158.34
    assert abs(rec['ref_mesh_y'] - 230.0) < 1.0
    assert abs(POSITION_MM[rec['seed_ch_x']] - 185.0) < 1.0
    # the pre-fix behaviour: with no seed mask the window lands on the oscillation (here it then holds < 4 strips and
    # the event is lost; on RAW, where every strip is read out, it is kept and fitted on noise)
    old = sbc.event_record(ch, sm, am, raw=False, nsmp=32, seed_bad=np.zeros(512, bool))
    assert old is None or old['seed_ch_y'] == 372


def test_bad_channel_kept_in_window_for_censoring():
    # a track next to ch 372: the window still carries 372 (the fit censors it), the ref ignores it
    ch, sm, am = _zs_event(track_mm=(185.0, 162.0))
    rec = sbc.event_record(ch, sm, am, raw=False, nsmp=32)
    assert 372 in rec['y']['ch']
    assert not np.asarray(rec['y']['W'])[list(rec['y']['ch']).index(372)].any()   # blanked: no start on the oscillation
    assert abs(POSITION_MM[rec['seed_ch_y']] - 162.0) < 1.0
    assert abs(rec['ref_mesh_y'] - 162.0) < 1.0


def test_no_signal_but_oscillation_is_rejected():
    ch, sm, am = _zs_event(track_mm=(185.0, -999.0))                    # no Y track at all
    assert sbc.event_record(ch, sm, am, raw=False, nsmp=32) is None


def _seed_counts(n=3000, hot=None, hot_frac=0.25, band=None):
    """Seed counts of a beam spread over Y 160-290 mm and X live bands, plus one oscillating channel."""
    rng = np.random.default_rng(1); cnt = np.zeros(512)
    for view, lo, hi in (('x', 150, 161) if band else ('x', 100, 240), ('y', 160, 290)):
        chs = np.flatnonzero((VIEW == view) & (POSITION_MM >= lo) & (POSITION_MM <= hi))
        np.add.at(cnt, rng.choice(chs, n), 1)
    if hot is not None:
        cnt[hot] += hot_frac * n
    return cnt


def test_hot_channel_found_per_plateau():
    hot = sbc.hot_channels(_seed_counts(hot=373), 3000)
    assert list(np.flatnonzero(hot)) == [373]                        # run55 d700: the neighbour of 372 oscillates


def test_beam_and_narrow_live_band_not_hot():
    assert not sbc.hot_channels(_seed_counts(), 3000).any()
    assert not sbc.hot_channels(_seed_counts(band=True), 3000).any()  # all X seeds inside one 12 mm live band
