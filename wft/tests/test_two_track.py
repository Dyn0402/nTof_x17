#!/usr/bin/env python3
"""
Unit tests for the joint two-track fit (wft.model.chi2_plane_two /
fit_plane_two_raw, wft.reco.fit_plane_two / resolve_two_tracks).

These pin the LOGIC, on planes built by the forward model itself with a
synthetic calibration: recovery, label order, crossings, the degeneracy guard,
the switch-off contract and the candidate bookkeeping. How well it does on real
charge is the overlay bench's question, not this file's
(sept26_prelim_analysis/two_track_synth.py then intra_bench.py).

    ../../.venv/bin/python -m pytest wft/tests/test_two_track.py -q
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm          # noqa: E402
from wft import reco as wr           # noqa: E402
from wft.calib import CalibrationBundle   # noqa: E402

PITCH = 0.78
NSAMP = 20
NOISE = 12.0


def _bundle():
    """A small, self-consistent bundle: a one-pole-ish impulse response on a
    10 ns grid, unit gain, a kernel that obeys c2 < c1."""
    grid = np.arange(0.0, 1400.0, 10.0)
    tmpl = (grid / 120.0) ** 2 * np.exp(-grid / 120.0)
    tmpl = tmpl / tmpl.max()
    return CalibrationBundle(
        hyper=dict(c1=0.06, c2=0.0, c2_over_c1=0.6, kY=1.4, tau_s=170.0,
                   sigma_s=10.0, sigma_p0=0.42, Dp=0.014),
        v_drift=42.6, grid=grid,
        tmpl={'x': tmpl, 'y': tmpl},
        gain={'x': np.ones(512), 'y': np.ones(512)},
        dt_xy={0: -18.8}, pitch_mm=PITCH, sample_ns=60.0, n_depth_bins=18,
        sat_adc=3700.0, share_mode='delay', detector='test', run_key='test',
        conditions=dict(run='test'), provenance=dict(note='unit test'))


CAL = _bundle()
wm.use_calibration(CAL)
wm.set_nsamp(NSAMP)


def setup_function(_fn):
    """wft.model's calibration is process-global, so another test module that
    installs its own bundle changes K, NSAMP and the caches under this one.
    Re-install ours before every test rather than relying on import order."""
    wm.use_calibration(CAL)
    wm.set_nsamp(NSAMP)


def make_window(tracks, seed=0, pad=3, noise=NOISE):
    """A plane window holding ``tracks`` = [(p0, w, t0, q_total)], cut the way
    the production seeder+``extract_window`` would cut it."""
    rng = np.random.default_rng(seed)
    allpos = np.arange(512) * PITCH
    W = np.zeros((512, wm.NSAMP))
    for p0, w, t0, qt in tracks:
        q = np.zeros(wm.K)
        q[:13] = 1.0
        q *= qt / q.sum()
        W += (wm.build_matrix('x', allpos, p0, w, t0, wm.HYPER) @ q
              ).reshape(512, wm.NSAMP)
    live = np.flatnonzero(W.max(axis=1) / noise > 5.0)
    lo, hi = max(0, live.min() - pad), min(511, live.max() + pad)
    ch = np.arange(lo, hi + 1)
    return dict(W=W[ch] + rng.normal(0.0, noise, (len(ch), wm.NSAMP)),
                pos=allpos[ch], noise=np.full(len(ch), noise), ch=ch)


def _fit_two(tracks, seed=0, t0_mode=None):
    P = make_window(tracks, seed=seed)
    f = wr.fit_plane(P, 'x', CAL)
    assert f is not None
    return P, f, wr.fit_plane_two(P, 'x', CAL, f, f_thresh=-np.inf,
                                  t0_mode=t0_mode)


# --------------------------------------------------------------- the model
def test_two_track_model_contains_one():
    """chi2 of two tracks can never exceed chi2 of one at the same theta_a:
    setting qb = 0 reproduces the one-track model exactly. That is what makes
    dchi2 a model-selection statistic rather than a fit artefact."""
    P = make_window([(160.0, 0.004, -40.0, 1400.0)], seed=3)
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    pa = (160.0, 0.004, -40.0)
    c1, _q = wm.chi2_plane('x', W, noise, pos, sat, *pa, wm.HYPER, snap_t0=False)
    for pb in ((166.0, -0.003, -40.0), (155.0, 0.010, 100.0)):
        c2, qa, qb = wm.chi2_plane_two('x', W, noise, pos, sat, pa, pb,
                                       wm.HYPER, snap_t0=False)
        assert c2 <= c1 + 1e-6, f'{c2} > {c1} for pb={pb}'
        assert len(qa) == wm.K and len(qb) == wm.K
        assert (qa >= 0).all() and (qb >= 0).all()


def test_separation_measures():
    """Two tracks that CROSS mid-column are still separated: the measure is a
    charge-weighted r.m.s. over the drift column, not a distance at one depth."""
    u_mid = 0.5 * wm.K * wm.DT
    a = (100.0, +0.01, 0.0)
    b = (100.0 + 0.02 * u_mid, -0.01, 0.0)      # crosses a exactly at u_mid
    assert abs((a[0] + a[1] * u_mid) - (b[0] + b[1] * u_mid)) < 1e-9
    assert wm.two_track_separation(a, b) > 3.0
    assert wm.two_track_distinguishability(a, b) > 1.0
    # identical tracks are degenerate, and a time offset alone rescues them
    assert wm.two_track_distinguishability(a, a) < 1.0
    assert wm.two_track_distinguishability(a, (a[0], a[1], a[2] + 200.0)) > 1.0


# ------------------------------------------------------------------ the fit
def test_recovers_two_separated_tracks():
    tracks = [(150.0, 0.004, -40.0, 1400.0), (170.0, -0.003, -40.0, 1400.0)]
    _P, _f, r = _fit_two(tracks, seed=1)
    assert r is not None
    ca, cb = r['children']
    assert abs(ca.p0 - 150.0) < 1.5, ca.p0
    assert abs(cb.p0 - 170.0) < 1.5, cb.p0
    assert r['fstat'] > 50.0
    assert r['guards_ok'], r


def test_label_order_and_crossing():
    """Child a is the track at the smaller position in the middle of the drift
    column, and two tracks that cross in this plane are still found."""
    tracks = [(150.0, +0.012, -40.0, 1400.0), (162.0, -0.012, -40.0, 1400.0)]
    _P, _f, r = _fit_two(tracks, seed=2)
    assert r is not None
    ca, cb = r['children']
    u_mid = 0.5 * wm.K * wm.DT
    assert ca.p0 + ca.w * u_mid <= cb.p0 + cb.w * u_mid
    got = sorted([ca.p0, cb.p0])
    # MATCH_MM, the bench's tolerance. Crossing tracks are the worst case for
    # p0 AT THE MESH: each has ~13 mm of transverse travel over the column and
    # they are only 12 mm apart, so the mesh intercept is the least constrained
    # thing about them.
    assert abs(got[0] - 150.0) < 3.0 and abs(got[1] - 162.0) < 3.0, got


def test_parallel_co_located_pair_is_degenerate():
    """Two PARALLEL tracks on the same strips, whatever their time offset, are
    not separable — and not because the fitter is weak.

    The charge profile q_k is free, so a track arriving 260 ns later is the same
    data as one track whose column runs 260 ns longer, up to w x 260 ns of
    transverse slide — under a strip pitch at any slope we fit. The one-track
    fit of such a pair already reaches chi2/dof = 1 (measured here), so there is
    no chi2 left for a second track to claim. Nothing in the reconstruction can
    recover this case; it has to be accounted for as inefficiency."""
    P = make_window([(150.0, 0.003, -60.0, 1400.0),
                     (150.5, 0.003, 200.0, 1400.0)], seed=4)
    f = wr.fit_plane(P, 'x', CAL)
    assert f.chi2 / f.dof < 1.3, f.chi2 / f.dof
    r = wr.fit_plane_two(P, 'x', CAL, f, f_thresh=-np.inf, t0_mode='tied')
    assert r['fstat'] < wr.TWO_TRACK_F, r['fstat']


def test_free_t0_mode_reaches_a_time_offset_pair():
    """The free-t0 mode finds a pair 260 ns apart with opposite slopes.

    Note what this does NOT show: that free t0 is needed. On this same pair the
    TIED fit scores higher (418 against 298) — the charge profile absorbs a good
    deal of a time offset on its own. Nothing measured so far gives `free` an
    advantage over `tied`, which is why `tied` is the default and `free` is a
    study option, not a fallback."""
    tracks = [(150.0, 0.008, -60.0, 1400.0), (150.5, -0.008, 200.0, 1400.0)]
    _P, _f, r = _fit_two(tracks, seed=4, t0_mode='free')
    assert r is not None and r['guards_ok'], r
    assert r['fstat'] > 100.0, r['fstat']


# ------------------------------------------------------- the contract in reco
class _Cal:
    dt_xy = {0: -18.8}
    hyper = CAL.hyper
    w0, kw, v_drift = CAL.w0, CAL.kw, CAL.v_drift


def test_switch_off_is_identical():
    """With WFT_TWO_TRACK_FIT off, resolve_two_tracks returns its input."""
    assert wr.TWO_TRACK is False, 'the default must be off'
    P = make_window([(150.0, 0.004, -40.0, 1400.0),
                     (162.0, -0.003, -40.0, 1400.0)], seed=5)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win = True, 1000.0, 0
    fits = {'x': [f], 'y': []}
    out, splits, replaced = wr.resolve_two_tracks(fits, {'x': [P], 'y': []}, CAL)
    assert out is fits and splits == [] and replaced == {'x': [], 'y': []}


def test_accepted_split_replaces_its_parent():
    P = make_window([(150.0, 0.004, -40.0, 1400.0),
                     (172.0, -0.003, -40.0, 1400.0)], seed=6)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win, f._rescued = True, 1000.0, 0, False
    fits = {'x': [f], 'y': []}
    try:
        wr.TWO_TRACK = True
        out, splits, replaced = wr.resolve_two_tracks(
            fits, {'x': [P], 'y': []}, CAL, selected={'x': {0}, 'y': set()})
    finally:
        wr.TWO_TRACK = False
    assert len(splits) == 1 and splits[0]['accepted'], splits
    assert len(out['x']) == 2, 'the parent should be replaced by two children'
    assert f not in out['x'] and replaced['x'] == [f]
    assert all(getattr(c, '_split_child', False) for c in out['x'])
    assert all(c.n_candidates == 2 for c in out['x'])
    rows = wr.candidate_rows(1, out, [(0, 0, True)], None, replaced)
    xr = [r for r in rows if r['plane'] == 'x']
    assert sum(r['split_child'] for r in xr) == 2
    assert sum(r['split_replaced'] for r in xr) == 1
    assert [r['rank'] for r in xr] == [0, 1, -1], [r['rank'] for r in xr]


def test_only_selected_candidates_are_reconsidered():
    """A candidate the selector did not choose is not re-fitted. That is where
    the cost goes: 75 % of real candidates are extra clusters in busy events."""
    P = make_window([(150.0, 0.004, -40.0, 1400.0),
                     (172.0, -0.003, -40.0, 1400.0)], seed=6)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win, f._rescued = True, 1000.0, 0, False
    fits = {'x': [f], 'y': []}
    try:
        wr.TWO_TRACK = True
        out, splits, replaced = wr.resolve_two_tracks(
            fits, {'x': [P], 'y': []}, CAL, selected={'x': set(), 'y': set()})
    finally:
        wr.TWO_TRACK = False
    assert out['x'] == [f] and splits == [] and replaced['x'] == []


def test_cross_plane_corroboration_needs_a_count_mismatch():
    """The discount fires only when the OTHER plane resolves more tracks than
    this one — not merely when it resolves two. Both-planes-resolve-two is the
    easy case, and discounting there splits correct candidates."""
    def cand(t0, p0):
        f = wr.PlaneFit(p0=p0, w=0.004, t0=t0, tan_theta=0.1, theta_deg=6.0,
                        chi2=500.0, dof=400, p0_err=0.4, w_err=1e-3, tan_err=0.03,
                        t0_err=15.0, q_sum=2e4, q_u50=350.0, q_u90=600.0,
                        q_uend=700.0, n_strips=12, n_seed=8, n_dropped=0,
                        slope_reliable=True, quality_ok=True)
        f._plausible = True
        return f

    one, two = [cand(0.0, 100.0)], [cand(0.0, 100.0), cand(20.0, 130.0)]
    assert wr._cross_plane_mismatch(one, two), 'one against two is the merged case'
    assert not wr._cross_plane_mismatch(two, two), 'two against two is the easy case'
    assert not wr._cross_plane_mismatch(one, one)
    far = [cand(0.0, 100.0), cand(900.0, 130.0)]
    assert not wr._cross_plane_mismatch(one, far), 'the other two must be coincident'


def test_refused_split_leaves_the_candidate_alone():
    P = make_window([(150.0, 0.004, -40.0, 1400.0)], seed=7)
    f = wr.fit_plane(P, 'x', CAL)
    f._plausible, f._dchi2, f._win, f._rescued = True, 1000.0, 0, False
    fits = {'x': [f], 'y': []}
    try:
        wr.TWO_TRACK = True
        out, splits, replaced = wr.resolve_two_tracks(
            fits, {'x': [P], 'y': []}, CAL, selected={'x': {0}, 'y': set()})
    finally:
        wr.TWO_TRACK = False
    assert out['x'] == [f], 'a refused split must leave the parent in place'
    assert replaced['x'] == []
    assert all(not s['accepted'] for s in splits)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
