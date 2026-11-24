"""bow['phi'] with outside = 'block' (interior only, bench re-finalization step 4): outside the map's region a view keeps the
plain bow block with the pass-1 constants, bit for bit; per-view domains; scalar and vectorised paths agree; round trip."""
import copy
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import reco                                    # noqa: E402
from wft.calib import CalibrationBundle                 # noqa: E402
from wft.tests.test_share_modes import synth_bundle     # noqa: E402
from wft.tests.test_bow_phi import _blk, _cal           # noqa: E402

PLAIN = {'a_x': {'cu': 0.05, 'auw': 0.01}, 'A_y': {'cw': 30.0}, 's_x': {'su': 0.1}, 'h': {'gu': 0.3, 'kuu': 0.1}}
FB = {'w0': {'x': 0.4, 'y': -0.3}, 'kw': {'x': 0.97, 'y': 1.03}}


def _interior_cal(domain_x=None):
    blk = dict(_blk(), outside='block', fallback=copy.deepcopy(FB))
    if domain_x:
        blk['domain_x'] = domain_x
    cal = _cal(dict(PLAIN, phi=blk))                     # w0 = (0.1, -0.2), kw = (1.01, 0.98): the constants WITH the map
    ref = SimpleNamespace(bow=dict(PLAIN), v_map={}, w0=FB['w0'], kw=FB['kw'], v_drift=37.0)    # the pass-1 bundle
    return cal, ref


def _clamp_twin(cal):
    c = copy.deepcopy(cal)
    c.bow['phi'].pop('outside')
    c.bow['phi'].pop('fallback')
    return c


def _events(n=400, seed=5):
    rng = np.random.default_rng(seed)
    wx, wy = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    px, py = rng.uniform(-20, 420, n), rng.uniform(-20, 420, n)
    px[::40] = np.nan
    return wx, px, wy, py


def test_outside_is_pass1_bit_for_bit_inside_is_the_map():
    cal, ref = _interior_cal()
    args = _events()
    v, r, c = reco.bow_pairs(cal, *args), reco.bow_pairs(ref, *args), reco.bow_pairs(_clamp_twin(cal), *args)
    for p in 'xy':
        use = v[f'phi_{p}']
        assert use.sum() > 20 and (~use).sum() > 20
        for k in ('tan', 'a', 'h'):
            assert np.array_equal(v[f'{k}_{p}'][~use], r[f'{k}_{p}'][~use], equal_nan=True), (k, p)
        # inside, the same as the clamp bundle (which evaluates inside points the same way)
        for k in ('tan', 'a', 'h'):
            assert np.array_equal(v[f'{k}_{p}'][use], c[f'{k}_{p}'][use]), (k, p)
    # the plain bow block would have given another angle with the bundle's own constants (so the fallback matters)
    plain_own = reco.bow_pairs(SimpleNamespace(bow=dict(PLAIN), v_map={}, w0=cal.w0, kw=cal.kw, v_drift=37.0), *args)
    out = ~v['phi_x'] & np.isfinite(args[1])
    assert np.abs(plain_own['tan_x'][out] - r['tan_x'][out]).max() > 1e-3


def test_scalar_path_matches_pairs_and_pass1():
    cal, ref = _interior_cal()
    args = _events(150, 6)
    v = reco.bow_pairs(cal, *args)

    def fit(i, a, b):
        return SimpleNamespace(p0=args[a][i], w=args[b][i], tan_theta=0, theta_deg=0, slope_reliable=True)
    for i in range(150):
        fx, fy, gx, gy = fit(i, 1, 0), fit(i, 3, 2), fit(i, 1, 0), fit(i, 3, 2)
        reco.apply_bow(cal, fx, fy)
        reco.apply_bow(ref, gx, gy)
        for f, g, p in ((fx, gx, 'x'), (fy, gy, 'y')):
            t = v[f'tan_{p}'][i]
            assert (np.isnan(t) and np.isnan(f.tan_theta)) or abs(f.tan_theta - t) < 1e-12
            assert abs(f._bow_h - v[f'h_{p}'][i]) < 1e-12
            if not v[f'phi_{p}'][i]:
                assert f.tan_theta == g.tan_theta and f.slope_reliable == g.slope_reliable and f._bow_h == g._bow_h


def test_tan_of_outside_is_pass1():
    cal, ref = _interior_cal()
    assert reco.tan_of(cal, 'x', 0.004, 5.0) == reco.tan_of(ref, 'x', 0.004, 5.0)        # off the map's region
    assert reco.tan_of(cal, 'x', 0.004, np.nan) == reco.tan_of(ref, 'x', 0.004, np.nan)
    assert reco.tan_of(cal, 'x', 0.004, 190.0) != reco.tan_of(ref, 'x', 0.004, 190.0)    # inside: the map


def test_per_view_domain_and_roundtrip():
    cal, ref = _interior_cal(domain_x={'U': [0.0, 1.69], 'W': _blk()['W_domain']})
    args = _events(600, 7)
    v, r = reco.bow_pairs(cal, *args), reco.bow_pairs(ref, *args)
    wx, px = args[0], args[1]
    U = ((px + 15 * (wx * 1e3 - 0.1) / (1.01 * 37.0) - 200) / 100)
    with np.errstate(invalid='ignore'):
        assert not np.any(v['phi_x'] & (U < 0))          # the X view: only x >= 200 mm
        assert np.any(v['phi_y'] & (U < 0))              # the Y view keeps the whole domain
    out = ~v['phi_x']
    assert np.array_equal(v['tan_x'][out], r['tan_x'][out], equal_nan=True)
    with tempfile.TemporaryDirectory() as d:
        c = synth_bundle()
        c.bow = copy.deepcopy(cal.bow)
        c.save(d)
        back = CalibrationBundle.load(d)
        assert json.dumps(back.bow, sort_keys=True) == json.dumps(c.bow, sort_keys=True)
        bad = copy.deepcopy(c)
        bad.bow['phi'].pop('fallback')
        with pytest.raises(ValueError):
            bad.save(os.path.join(d, 'bad'))
        bad = copy.deepcopy(c)
        bad.bow['phi']['outside'] = 'nope'
        with pytest.raises(ValueError):
            bad.save(os.path.join(d, 'bad2'))


def test_clamp_default_uses_map_for_every_finite_pair():
    c = _cal({'phi': _blk()})
    args = _events(200, 8)
    v = reco.bow_pairs(c, *args)
    assert v['phi_x'][np.isfinite(args[1])].all()
    assert not reco.bow_pairs(_cal({'a_x': {'cu': 0.05}}), *args)['phi_x'].any()
