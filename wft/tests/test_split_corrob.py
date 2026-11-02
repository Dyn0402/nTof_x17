#!/usr/bin/env python3
"""
Unit tests for the charge-corroborated two-track split (xy_pairing['split_corrob'] = 'charge';
rerun/effbudget/single/MERGED_PROJECTION_2026-11-02.md).

Two particles that overlap in one projection give one candidate there and two in the other view.  The 'count' rule
corroborates the split only if the other view's two are plausible and accepts it only if both children are; the
'charge' rule asks the other view's two to be time-coincident with the merged candidate and to carry its charge, and
accepts a split with one plausible child whose two children's charges match the two other-view candidates.

    ../../.venv/bin/python -m pytest wft/tests/test_split_corrob.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm          # noqa: E402
from wft import reco as wr           # noqa: E402
from wft.calib import check_xy_pairing   # noqa: E402
from wft.tests.test_two_track import CAL, NSAMP, make_window   # noqa: E402

PAIRING = dict(features=['lq', 'u50', 'u90', 'tc'],
               median=dict(lq=0.0, u50=0.0, u90=0.0, tc=0.0),
               rsig=dict(lq=0.06, u50=40.0, u90=40.0, tc=40.0), tc_tol=200.0,
               lq_class=dict(clean=dict(median=0.0, rsig=0.06, n=1000)))


def setup_function(_fn):
    wm.use_calibration(CAL)
    wm.set_nsamp(NSAMP)


def cand(p0, q, t0=0.0, u50=350.0, plausible=True):
    f = wr.PlaneFit(p0=p0, w=0.004, t0=t0, tan_theta=0.1, theta_deg=6.0, chi2=500.0, dof=400, p0_err=0.4,
                    w_err=1e-3, tan_err=0.03, t0_err=15.0, q_sum=q, q_u50=u50, q_u90=600.0, q_uend=700.0,
                    n_strips=12, n_seed=8, n_dropped=0, slope_reliable=True, quality_ok=True)
    f._plausible = plausible
    return f


# ------------------------------------------------------------------ the switch
def test_mode_switch():
    assert wr.split_corrob_mode(None) == 'count'
    assert wr.split_corrob_mode(PAIRING) == 'count', 'absent = the pass-1 / n_TOF behaviour'
    assert wr.split_corrob_mode(dict(PAIRING, split_corrob='charge')) == 'charge'
    assert wr.split_corrob_mode(dict(split_corrob='charge')) == 'count', 'no pairing calibration = count'
    check_xy_pairing(dict(PAIRING, split_corrob='charge'))
    with pytest.raises(ValueError):
        check_xy_pairing(dict(PAIRING, split_corrob='plausible'))
    try:
        wr.SPLIT_CORROB_MODE = 'charge'
        assert wr.split_corrob_mode(PAIRING) == 'charge'
    finally:
        wr.SPLIT_CORROB_MODE = None


# ------------------------------------------------------------- the trigger
def test_charge_corroboration_ignores_plausibility():
    """The other view's two candidates may be implausible (a short, steep bench secondary): the count rule refuses
    to corroborate, the charge rule does."""
    f = cand(100.0, 2.0e4)
    two = [cand(100.0, 1.0e4), cand(130.0, 1.0e4, plausible=False)]
    assert not wr._cross_plane_mismatch([f], two)
    c = wr._charge_corroboration(f, 'x', [f], two, PAIRING)
    assert c is not None and abs(c['z_sum']) < 1e-9
    assert c['z_alone'] > 10, 'the merged candidate is twice either one'


def test_charge_corroboration_conditions():
    f = cand(100.0, 2.0e4)
    two = [cand(100.0, 1.0e4), cand(130.0, 1.0e4)]
    assert wr._charge_corroboration(f, 'x', [f, cand(140.0, 1e4)], two, PAIRING) is None, 'needs a count mismatch'
    late = [cand(100.0, 1.0e4), cand(130.0, 1.0e4, t0=900.0)]
    assert wr._charge_corroboration(f, 'x', [f], late, PAIRING) is None, 'the other two must be coincident'
    off = cand(100.0, 2.0e4, t0=600.0)
    assert wr._charge_corroboration(off, 'x', [off], two, PAIRING) is None, 'and coincident with this candidate'
    bright = cand(100.0, 2.0e4 * np.exp(5 * 0.06))
    assert wr._charge_corroboration(bright, 'x', [bright], two, PAIRING) is None, 'more charge than both: no'
    faint = cand(100.0, 1.0e4)
    assert wr._charge_corroboration(faint, 'x', [faint], two, PAIRING) is not None, \
        'less than the sum is allowed: a one-track fit of a merged window explains only part of it'
    # orientation: in the y view the balance is ln(qx/qy) with x = the other view
    fy = cand(100.0, 2.0e4 * np.exp(-0.2))
    c = wr._charge_corroboration(fy, 'y', [fy], two, PAIRING)
    assert c is not None and c['z_sum'] < 0, 'y candidate fainter than the x sum reads negative'
    # a short secondary (soft electron near the cathode) has a column time far from the muon's: same trigger t0 is
    # what counts here; the column time is checked per particle after the split
    short = [cand(100.0, 1.0e4), cand(130.0, 1.0e4, u50=750.0)]
    assert wr._charge_corroboration(f, 'x', [f], short, PAIRING) is not None


def test_child_column_time_reported():
    corr = dict(a=cand(0, 1.0e4), b=cand(0, 1.0e4, u50=900.0), ref=(0.0, 0.06))
    _sc, _z1, _z2, tc = wr._child_charge_match((cand(0, 1.0e4), cand(0, 1.0e4)), corr, 'x')
    assert tc > wr.TC_TOL_NS, 'a child at the muon time matched to the late candidate is not the same particle'
    _sc, _z1, _z2, tc = wr._child_charge_match((cand(0, 1.0e4), cand(0, 1.0e4, u50=900.0)), corr, 'x')
    assert tc < 1e-9


def test_child_charge_match_takes_the_best_assignment():
    corr = dict(a=cand(0, 1.0e4), b=cand(0, 3.0e4), ref=(0.0, 0.06))
    sc, _z1, _z2, tc = wr._child_charge_match((cand(0, 3.0e4), cand(0, 1.0e4)), corr, 'x')
    assert sc < 1e-9 and tc < 1e-9
    sc, _z1, _z2, _tc = wr._child_charge_match((cand(0, 2.0e4), cand(0, 2.0e4)), corr, 'x')
    assert sc > wr.CHARGE_CORROB_CHILD_NSIG, '40/60 of the total does not match 25/75'


# ------------------------------------------------------- end to end in resolve_two_tracks
def _merged_event():
    """x: two tracks 22 mm apart in one window (one candidate); y: two candidates carrying the children's charge."""
    P = make_window([(150.0, 0.004, -40.0, 1400.0), (172.0, -0.003, -40.0, 1400.0)], seed=6)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win, f._rescued = True, 1000.0, 0, False
    r = wr.fit_plane_two(P, 'x', CAL, f, f_thresh=-np.inf)
    ca, cb = r['children']
    ys = [cand(150.0, ca.q_sum, t0=f.t0, u50=f.q_u50), cand(172.0, cb.q_sum, t0=f.t0, u50=f.q_u50)]
    return P, f, ys, r


def _resolve(P, f, ys, pairing):
    f._split_replaced = False
    return wr.resolve_two_tracks({'x': [f], 'y': ys}, {'x': [P], 'y': [None, None]}, CAL,
                                 selected={'x': {0}, 'y': set()}, pairing=pairing)


def test_one_implausible_child_count_refuses_charge_accepts():
    P, f, ys, r0 = _merged_event()
    tans = sorted(abs(c.tan_theta) for c in r0['children'])
    assert r0['fstat'] > wr.TWO_TRACK_F_CORROB and r0['distinguishable'] and r0['column_shared'], r0
    old = wr.TAN_MAX
    try:
        wr.TAN_MAX = 0.5 * (tans[0] + tans[1])            # the steeper child (the "secondary") is implausible
        _o, sp, rep = _resolve(P, f, ys, PAIRING)
        assert len(sp) == 1 and not sp[0]['accepted'] and not sp[0]['both_plausible'], sp
        assert rep['x'] == []
        out, sp, rep = _resolve(P, f, ys, dict(PAIRING, split_corrob='charge'))
        assert len(sp) == 1 and sp[0]['accepted'] and sp[0]['accepted_charge'] and sp[0]['corrob_charge'], sp
        assert not sp[0]['accepted_count']
        assert rep['x'] == [f] and len(out['x']) == 2
        # the y charges do not match the children: refused
        bad = [cand(150.0, 3 * ys[0].q_sum, t0=f.t0, u50=f.q_u50),
               cand(170.0, 0.3 * ys[1].q_sum, t0=f.t0, u50=f.q_u50)]
        _o, sp, rep = _resolve(P, f, bad, dict(PAIRING, split_corrob='charge'))
        assert not any(s['accepted'] for s in sp) and rep['x'] == [], sp
        # neither child plausible: refused (one must be the muon)
        wr.TAN_MAX = 0.5 * tans[0]
        _o, sp, rep = _resolve(P, f, ys, dict(PAIRING, split_corrob='charge'))
        assert not any(s['accepted'] for s in sp), sp
    finally:
        wr.TAN_MAX = old


def test_charge_mode_never_refuses_what_count_accepts():
    """Both children plausible and the count rule fires: accepted in both modes."""
    P, f, ys, _r0 = _merged_event()
    _o, sp_c, _ = _resolve(P, f, ys, PAIRING)
    _o, sp_q, _ = _resolve(P, f, ys, dict(PAIRING, split_corrob='charge'))
    assert sp_c[0]['accepted'] and sp_q[0]['accepted'] and sp_q[0]['accepted_count']


def test_single_track_with_one_candidate_per_view_is_untouched():
    """1 x 1 candidates: no count mismatch, so the charge rule cannot fire; same answer as 'count'."""
    P = make_window([(150.0, 0.004, -40.0, 1400.0)], seed=7)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win, f._rescued = True, 1000.0, 0, False
    y = [cand(150.0, f.q_sum, t0=f.t0, u50=f.q_u50)]
    out, sp, rep = _resolve(P, f, y, dict(PAIRING, split_corrob='charge'))
    assert out['x'] == [f] and rep['x'] == []
    assert all(not s['corrob_charge'] and not s['accepted'] for s in sp)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
