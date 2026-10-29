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


# ------------------------------------------------------------ v map slot
def test_v_map_slot(cal, tmp_path):
    from wft.calib import CalibrationBundle
    cal.v_map = {'t': {'clock': 'dream_ts_10ns', 't_h': [0.0, 1.0, 2.0], 'v_rel': [1.0, 1.1, 1.2]},
                 'xy': {'e': [0, 100, 200, 300, 400], 'v_rel': [[0.9, 0.9, 0.9, 0.9], [1.0, 1.0, 1.0, 1.0],
                                                           [1.1, 1.1, 1.1, 1.1], [float('nan')] * 4]}}
    cal.save(str(tmp_path / 'b'))
    back = CalibrationBundle.load(str(tmp_path / 'b'))
    assert back.v_map['xy']['e'] == cal.v_map['xy']['e']
    assert wr.v_scale(back) == 1.0
    assert wr.v_rel_xy(back, 150.0, 200.0) == pytest.approx(1.0)              # a cell centre
    assert wr.v_rel_xy(back, 200.0, 50.0) == pytest.approx(1.05)              # bilinear between centres
    assert wr.v_rel_xy(back, -50.0, 50.0) == pytest.approx(0.9)               # clamped
    assert wr.v_rel_xy(back, 350.0, 50.0) == pytest.approx(1.0)               # empty cells -> the median
    assert wr.v_rel_t(back, 1.5) == pytest.approx(1.15)
    # the v map replaces the bow's linear scale term in the angle conversion
    back.bow = {'A_x': dict(cu=0.1), 's_x': dict(su=0.5, sw=0.0)}
    t_map = wr.tan_bow(back, 'x', 0.01, -1.5, -0.5)                           # S = v_rel(50, 150) = 0.9
    back.v_map = {}
    t_lin = wr.tan_bow(back, 'x', 0.01, -1.5, -0.5)                           # S = 1 + 0.5 * (-1.5) = 0.25
    assert t_map == pytest.approx(t_lin * 0.25 / 0.9, rel=1e-9)
    cal.v_map = {}


# ------------------------------------------------------------ the bow block (FB spec)
BOW = {'A_x': dict(cu=0.2, cw=-0.02, auu=2.0, aww=2.0, auw=0.08), 's_x': dict(su=-0.02, sw=0.013),
       'A_y': dict(cu=-0.01, cw=0.3, auu=2.0, aww=2.0, auw=-0.02), 's_y': dict(su=-0.005, sw=0.015),
       'h': dict(gu=0.03, gw=-0.3, kuu=2.0, kww=2.0, kuw=-0.16), 'source_row': 'synthetic dish'}


def test_bow_dish_reproduces_tan(cal, tmp_path):
    from types import SimpleNamespace
    from wft.calib import CalibrationBundle
    cal.bow, cal.w0, cal.kw, cal.v_drift = BOW, {'x': 0.3, 'y': -0.2}, {'x': 1.02, 'y': 0.97}, 36.0
    cal.save(str(tmp_path / 'b'))
    back = CalibrationBundle.load(str(tmp_path / 'b'))
    assert back.bow == BOW
    rng = np.random.default_rng(3)
    for _ in range(50):
        tx, ty = rng.uniform(-0.5, 0.5, 2)
        fx = SimpleNamespace(p0=rng.uniform(20, 380), w=0.0, tan_theta=0.0, theta_deg=0.0, slope_reliable=False)
        fy = SimpleNamespace(p0=rng.uniform(20, 380), w=0.0, tan_theta=0.0, theta_deg=0.0, slope_reliable=False)
        for _it in range(60):                           # w consistent with U, W read from its own constant-mapping tan
            U, W = wr.bow_UW(back, fx, fy)
            for p, f, t in (('x', fx, tx), ('y', fy, ty)):
                A, s = BOW[f'A_{p}'], BOW[f's_{p}']
                f.w = (back.w0[p] + wr._bow_poly(A, U, W)
                       + back.kw[p] * back.v_drift * t * (1 + s['su'] * U + s['sw'] * W)) / 1e3
        wr.apply_bow(back, fx, fy)
        assert fx.tan_theta == pytest.approx(tx, abs=1e-9) and fy.tan_theta == pytest.approx(ty, abs=1e-9)
        assert np.isfinite(fx._bow_h)
    cal.bow = {}


def test_bow_tan_form_transfers_across_v(cal):
    """Tan-form block (a_p): the same geometric tilt reproduces tan at any v; the um/ns form (A_p) does not."""
    from types import SimpleNamespace
    a = {'x': dict(cu=0.006, cw=-0.001, auu=0.05, aww=0.05, auw=0.002), 'y': dict(cu=0.0, cw=0.008, auu=0.05, aww=0.05, auw=0.0)}
    base = dict(BOW, a_x=a['x'], a_y=a['y'])
    cal.w0, cal.kw = {'x': 0.3, 'y': -0.2}, {'x': 1.02, 'y': 0.97}
    for v in (36.0, 18.0):                               # the same chamber at two drift fields
        cal.v_drift, cal.bow = v, base
        tx, ty = 0.21, -0.13
        fx = SimpleNamespace(p0=95.0, w=0.0, tan_theta=0.0, theta_deg=0.0, slope_reliable=False)
        fy = SimpleNamespace(p0=310.0, w=0.0, tan_theta=0.0, theta_deg=0.0, slope_reliable=False)
        for _it in range(60):
            U, W = wr.bow_UW(cal, fx, fy)
            for p, f, t in (('x', fx, tx), ('y', fy, ty)):
                s = BOW[f's_{p}']
                f.w = (cal.w0[p] + cal.kw[p] * v * (1 + s['su'] * U + s['sw'] * W) * (t + wr._bow_poly(a[p], U, W))) / 1e3
        wr.apply_bow(cal, fx, fy)
        assert fx.tan_theta == pytest.approx(tx, abs=1e-9) and fy.tan_theta == pytest.approx(ty, abs=1e-9)
    cal.bow = {}
