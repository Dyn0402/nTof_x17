#!/usr/bin/env python3
"""The discharge-aware veto (wft.seed.SPARK_RESCUE, rerun/effbudget/single/DISCHARGE_VETO_2026-11-02.md).

    .venv/bin/python -m pytest wft/tests/test_spark_rescue.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import seed as ws  # noqa: E402
from wft import reco as wr  # noqa: E402

FEU_X, FEU_Y = 1, 2
POS = {FEU_X: np.arange(512) * ws.PITCH_MM, FEU_Y: np.arange(512) * ws.PITCH_MM}


def _hits(eid, xs, ys, sig=50.0):
    rows = [(eid, FEU_X, c, 10 * sig, sig) for c in xs] + [(eid, FEU_Y, c, 10 * sig, sig) for c in ys]
    return pd.DataFrame(rows, columns=['eventId', 'feu', 'channel', 'amplitude', 'significance'])


def test_muon_plus_other_keeps_the_footprint_cluster_only():
    """A clean 6-strip muon plus a 60-strip deposit far away: the event is over the veto, the rescue seeds the muon."""
    H = _hits(1, list(range(100, 106)) + list(range(300, 360)), range(200, 206))
    assert ws.seeds_from_hits(H, POS, FEU_X, FEU_Y)[1]['spark']          # the veto still fires on the count
    r = ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y, n_hits=72)
    assert not r['spark'] and r['n_hits'] == 72
    assert len(r['x']) == 1 and set(r['x'][0].channels) == set(range(100, 106))
    assert r['x'][0].n_dropped == 0                                      # the wide deposit is not a competing track
    assert r['spark_rescue']['n_wide_strips_x'] == 60
    assert r['spark_rescue']['n_wide_x'] == 1 and r['spark_rescue']['n_wide_y'] == 0
    assert len(r['y']) == 1


def test_nothing_clean_keeps_the_hard_veto():
    H = _hits(2, range(50, 150), range(60, 160))                         # one wide region per view
    r = ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y, n_hits=200)
    assert r['spark'] and r['x'] == [] and r['y'] == []


def test_cluster_cap_is_the_module_constant():
    H = _hits(3, range(100, 145), range(200, 206))                       # 45 strips in x
    old = ws.SPARK_CLUSTER_MAX
    try:
        ws.SPARK_CLUSTER_MAX = 40
        assert ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y)['x'] == []
        ws.SPARK_CLUSTER_MAX = 50
        assert len(ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y)['x']) == 1
    finally:
        ws.SPARK_CLUSTER_MAX = old


def test_floor_is_reapplied_without_the_discharge():
    """A ringing chip's huge hits set the plane's 10 % floor; once they are dropped a fainter muon passes it."""
    H = _hits(4, range(130, 136), range(200, 206), sig=20.0)
    big = _hits(4, range(0, 64), [], sig=5000.0)                         # chip 0 of x, 250x brighter
    H = pd.concat([H, big])
    assert ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y)['x'] == []        # under the floor, and chip 0 is too wide
    r = ws.spark_rescue_seeds(H, POS, FEU_X, FEU_Y, ringing={'x': np.array([0])})
    assert len(r['x']) == 1 and set(r['x'][0].channels) == set(range(130, 136))
    assert r['spark_rescue']['ringing_x'] == [0]


def test_ringing_chips_sees_a_coherent_swing_not_a_track():
    rng = np.random.default_rng(1)
    ns = 32
    ped = np.full(512, 300.0)
    noise = np.full(512, 10.0)
    raw = ped[:, None] + rng.normal(0, 10, (512, ns))
    raw[70:90, 10:14] += 2000.0          # a steep track: 20 strips of chip 1, a few samples each
    raw[256:320, 8:20] += 800.0          # chip 4 swings as a whole (discharge)
    assert list(ws.ringing_chips(raw, ped, noise, sig=20.0)) == [4]
    assert len(ws.ringing_chips(raw, ped, noise, sig=0.0)) == 0          # 0 = no chip mask


def _row(**kw):
    r = dict(event_id=1, x_ok=True, y_ok=True, x_quality_ok=True, y_quality_ok=True, x_isochronous=False,
             y_isochronous=False, x_t0=300.0, y_t0=300.0, x_q_u50=400.0, y_q_u50=400.0, x_ftst=3, y_ftst=3,
             x_q_sum=1000.0, y_q_sum=1000.0, x_n_sat_strips=0, y_n_sat_strips=0, x_n_flagged_strips=0,
             y_n_flagged_strips=0, x_n_band_strips=0, y_n_band_strips=0)
    r.update(kw)
    return r


def test_rescue_gate():
    pairing = {'lq_class': {'clean': {'median': 0.0, 'rsig': 0.06, 'n': 100}}}
    df = pd.DataFrame([_row(), _row(x_t0=700.0), _row(x_q_sum=3000.0), _row(y_isochronous=True),
                       _row(x_quality_ok=False), _row(y_ok=False), _row(x_ftst=4, x_t0=290.0)])
    g = wr.spark_rescue_gate(df, pairing, 200.0)
    assert g['rg_ok'].tolist() == [True, False, False, False, False, False, True]
    assert abs(g['rg_tc'].iloc[6]) < 1e-9                                # ftst phase: 10 ns per tick
    # no charge calibration in the bundle: the balance is not judged, the rest still is
    g = wr.spark_rescue_gate(df, {}, 200.0)
    assert g['rg_ok'].tolist() == [True, False, True, False, False, False, True]


def test_censor_ringing_chip_rows_and_rails():
    from types import SimpleNamespace
    ns = 8
    ch = np.arange(60, 70)                                               # straddles chips 0 and 1
    win = SimpleNamespace(ch=ch.astype(np.int16), W=np.zeros((len(ch), ns), np.float32))
    raw = np.full((512, ns), 300, dtype=np.int16)
    raw[66, 3] = 4095                                                    # high rail
    raw[67, 5] = 0                                                       # low rail
    m = wr.spark_censor(win, np.array([0]), raw)
    assert m[:4].all()                                                   # channels 60-63: chip 0, every sample
    assert m[6, 3] and m[7, 5] and m[4:].sum() == 2                      # chip 1: only the two railed samples
    old = ws.SPARK_RAIL_CENSOR
    try:
        ws.SPARK_RAIL_CENSOR = False
        assert wr.spark_censor(win, np.array([0]), raw)[4:].sum() == 0
        assert wr.spark_censor(win, None, raw) is None
    finally:
        ws.SPARK_RAIL_CENSOR = old


def test_rescue_is_off_by_default():
    """Off (DISCHARGE_VETO_2026-11-02.md: rescued tracks are far less pure than ordinary ones); the switch is a study."""
    assert ws.SPARK_RESCUE is False
