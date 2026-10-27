#!/usr/bin/env python3
"""Position-dependent angle mapping (``CalibrationBundle.angle_map``, ws_widthtilt 2026-10-27): absent = the
constant mapping bit for bit; present = tan = (w - w0(p)) / (s(p) kw v) at the mid-gap position; and the map
survives save -> load.

Self-contained (synthetic bundle, like test_share_modes).

    ../../.venv/bin/python wft/tests/test_angle_map.py
"""
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft.calib import CalibrationBundle              # noqa: E402
from wft import reco as wr                           # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402


def test_absent_is_constant():
    cal = synth_bundle()
    cal.w0, cal.kw = {'x': -0.2}, {'x': 1.02}
    w, p0 = 0.0072, 150.0
    assert wr.tan_of(cal, 'x', w, p0) == (w * 1e3 + 0.2) / (1.02 * cal.v_drift)


def test_map_applied_at_mid_gap():
    cal = synth_bundle()
    cal.w0, cal.kw = {'x': -0.2}, {'x': 1.0}
    cal.angle_map = {'x': dict(p=[50.0, 350.0], w0=[-0.6, 0.6], s=[1.0, 1.1])}
    w, p0 = 0.0072, 150.0
    t1 = (w * 1e3 + 0.2) / cal.v_drift
    p = p0 + 15.0 * t1
    f = (p - 50.0) / 300.0
    expect = (w * 1e3 - (-0.6 + 1.2 * f)) / ((1.0 + 0.1 * f) * cal.v_drift)
    assert abs(wr.tan_of(cal, 'x', w, p0) - expect) < 1e-12
    # the other plane has no map: constant mapping
    assert wr.tan_of(cal, 'y', w, p0) == w * 1e3 / cal.v_drift


def test_round_trip():
    cal = synth_bundle()
    cal.angle_map = {'y': dict(p=[10.0, 200.0, 390.0], w0=[-0.5, 0.0, 0.4], s=[0.95, 1.0, 1.05])}
    with tempfile.TemporaryDirectory() as d:
        cal.save(d)
        back = CalibrationBundle.load(d)
    assert back.angle_map == cal.angle_map


if __name__ == '__main__':
    test_absent_is_constant(); test_map_applied_at_mid_gap(); test_round_trip()
    print('ok')
