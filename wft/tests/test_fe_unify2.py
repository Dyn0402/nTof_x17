#!/usr/bin/env python3
"""Model audit follow-up FE (2026-10-27): noise-band censoring, the column-time x-y gate, per-class charge pairing.

    ../../.venv/bin/python -m pytest wft/tests/test_fe_unify2.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft import reco as wr                           # noqa: E402
from wft.reco import PlaneFit                        # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

HYPER = dict(c1=0.1, c2=0.05, kY=2.0, tau_s=150.0, sigma_s=50.0, sigma_p0=0.25, Dp=0.015)


@pytest.fixture()
def cal():
    c = synth_bundle()
    c.hyper = dict(HYPER)
    wm.use_calibration(c)
    wm.set_share_mode('delay')
    wm.set_nsamp(32)
    wm.set_depth_bins(18)
    yield c


def _plane(w=0.006, seed=1, noise=3.0):
    ch = np.arange(100, 140)
    pos = ch * wm.PITCH
    q = np.full(wm.K, 2000.0)
    q[13:] = 0.0
    W = wm.model_waveforms('x', pos, 120 * wm.PITCH, w, 300.0, q, HYPER)
    W = W + np.random.default_rng(seed).normal(0, noise, W.shape)
    return dict(W=W, pos=pos, noise=np.full(len(ch), noise), ch=ch)


# ------------------------------------------------------------------ bands
def test_band_rows_flags_offsets_not_tracks(cal):
    P = _plane()
    assert not wm.band_rows(P['W'], P['noise']).any()          # a track never looks like a band
    W = P['W'].copy()
    W[3] += 400.0                                               # positive whole-window offset
    W[30] -= 200.0                                              # negative band (makes no hit, still caught)
    W[10, -2:] += 900.0                                         # late only: a pulse tail, not a band
    b = wm.band_rows(W, P['noise'])
    assert b[3] and b[30] and not b[10] and b.sum() == 2


def test_band_strip_is_censored_and_track_recovered(cal):
    P = _plane(seed=3)
    clean = wr.fit_plane(P, 'x', cal)
    W = P['W'].copy()
    i = int(np.argmax(W.max(axis=1)))                           # the band lands on the track's brightest strip
    W[i - 1:i + 2] += 1500.0
    banded = wr.fit_plane(dict(P, W=W), 'x', cal)
    assert banded.n_band_strips == 3 and clean.n_band_strips == 0
    assert banded.dof < clean.dof
    assert abs(banded.p0 - clean.p0) < 0.5 and abs(banded.w - clean.w) < 1.5e-3
    old = wm.BAND_NSIG
    try:
        wm.BAND_NSIG = 0.0                                      # the study override: bands fitted as data
        raw = wr.fit_plane(dict(P, W=W), 'x', cal)
    finally:
        wm.BAND_NSIG = old
    assert raw.chi2 / raw.dof > 10 * banded.chi2 / banded.dof


# ------------------------------------------------------------ x-y gate
def _fit(t0, u50=400.0, q_sum=2e4, nsat=0, nflag=0, nband=0, dchi2=500.0):
    f = PlaneFit(p0=100.0, w=0.005, t0=t0, tan_theta=0.14, theta_deg=8.0, chi2=500.0, dof=400, p0_err=0.4,
                 w_err=1e-3, tan_err=0.03, t0_err=15.0, q_sum=q_sum, q_u50=u50, q_u90=600.0, q_uend=700.0,
                 n_strips=8, n_seed=6, n_dropped=0, slope_reliable=True, quality_ok=True, n_candidates=1,
                 n_flagged_strips=nflag, n_band_strips=nband, n_sat_strips=nsat)
    f._plausible, f._dchi2 = True, dchi2
    return f


class Cal:
    dt_xy = {}


def test_column_time_is_phase_corrected():
    # same latency, x FEU two ftst classes later: t0x is 20 ns earlier in its own window
    fx, fy = _fit(280.0, u50=400.0), _fit(300.0, u50=400.0)
    assert wr.col_dt(fx, fy, ftst_diff=2) == pytest.approx(0.0)
    # a t0 slid by one depth bin moves u50 the other way: the column time does not move
    assert wr.col_dt(_fit(340.0, u50=350.0), _fit(300.0, u50=390.0), 0) == pytest.approx(0.0)


def test_gate_uses_bundle_tolerance():
    old = wr.TC_TOL_NS
    try:
        wr.TC_TOL_NS = wr.tc_tol({'tc_tol': 90.0})
        assert wr.TC_TOL_NS == 90.0
        assert wr.select_tracks({'x': [_fit(300.0)], 'y': [_fit(380.0)]}, 0, Cal())[0][2]
        assert not wr.select_tracks({'x': [_fit(300.0)], 'y': [_fit(400.0)]}, 0, Cal())[0][2]
        assert wr.tc_tol({}) == wr.TC_TOL_DEFAULT_NS
    finally:
        wr.TC_TOL_NS = old


# ------------------------------------------------------- pairing classes
def test_pairing_class_and_widths():
    assert wr.pairing_class(_fit(0), _fit(0)) == 'clean'
    assert wr.pairing_class(_fit(0, nflag=1), _fit(0)) == 'cens'
    assert wr.pairing_class(_fit(0, nband=2), _fit(0, nsat=1)) == 'sat'
    p = dict(features=['lq'], median={'lq': 0.0}, rsig={'lq': 0.1},
             lq_class={'clean': dict(median=0.0, rsig=0.05), 'sat': dict(median=-0.2, rsig=0.2)})
    assert wr.lq_balance(p, 'clean') == (0.0, 0.05)
    assert wr.lq_balance(p, 'sat') == (-0.2, 0.2)
    assert wr.lq_balance(p, 'cens') == (0.0, 0.1)               # class not measured -> the chamber's one width
    # a saturated pair is judged on its own (wider, shifted) balance
    fx, fy = _fit(0, q_sum=1.0e4, nsat=1), _fit(0, q_sum=1.22e4)
    c_sat = wr.xy_pair_cost(fx, fy, p, 0.0)
    fx.n_sat_strips = 0
    c_clean = wr.xy_pair_cost(fx, fy, p, 0.0)
    assert c_sat < 0.01 and c_clean > 10


def test_tc_feature_accepted():
    from wft.calib import check_xy_pairing
    check_xy_pairing(dict(features=['lq', 'tc'], median={'lq': 0.0, 'tc': 0.0}, rsig={'lq': 0.05, 'tc': 40.0}))


# ------------------------------------------------------------ 20 ns grid
def test_depth_subdiv_grid(cal, tmp_path):
    from wft.calib import CalibrationBundle
    P = _plane(w=0.010, seed=5)
    a = wr.fit_plane(P, 'x', cal, t0_prior=(300.0, 0.05))      # the 20 ns grid needs a pinned t0 (free: t0/p0 slide)
    cal.depth_subdiv = 3
    cal.save(str(tmp_path / 'b'))
    back = CalibrationBundle.load(str(tmp_path / 'b'))
    assert back.depth_subdiv == 3
    wm.use_calibration(back)
    wm.set_share_mode('delay'); wm.set_nsamp(32); wm.set_depth_bins(18)
    try:
        assert wm.K == 54 and wm.DT == pytest.approx(20.0)
        b = wr.fit_plane(P, 'x', back, t0_prior=(300.0, 0.05))
        assert len(b._q) == 54
        assert abs(b.w - a.w) < 1.5e-3 and abs(b.p0 - a.p0) < 0.5
    finally:
        cal.depth_subdiv = 1
        wm.use_calibration(cal); wm.set_depth_bins(18)
