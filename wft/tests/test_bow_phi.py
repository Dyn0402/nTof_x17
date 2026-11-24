"""bow['phi'] (wft/bow_phi.py): the per-mounting spline angle map.  Reproduces rerun/paper1_gap/audit a2_final.evaluate,
clamps at the domain, round-trips through the bundle, and leaves bundles without it bit for bit as they were."""
import copy
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import reco, bow_phi                          # noqa: E402
from wft.calib import CalibrationBundle                # noqa: E402
from wft.tests.test_share_modes import synth_bundle    # noqa: E402

A2 = os.path.join(REPO, 'rerun/paper1_gap/audit')
PROTO = os.path.join(A2, 'results/a2_final_map_det3.json')
GATE = '/media/ucla/work/claude-jobs/e1a7a26e/gap/pass1_all'


def _a2_evaluate():
    # a2_final.py imports a2_lib (paper1_gap work tree); take just its standalone evaluator from the source
    src = open(os.path.join(A2, 'a2_final.py')).read()
    src = src[src.index('def evaluate'):src.index("if __name__")]
    ns = {'np': np, 'json': json}
    exec(src, ns)
    return ns['evaluate']


def _cal(bow, v_map=None):
    return SimpleNamespace(bow=bow, v_map=v_map or {}, w0={'x': 0.1, 'y': -0.2}, kw={'x': 1.01, 'y': 0.98}, v_drift=37.0)


def _blk():
    return bow_phi.from_a2_map(json.load(open(PROTO)))


def test_reproduces_a2_evaluate():
    M = json.load(open(PROTO)); blk = _blk(); ev = _a2_evaluate()
    rng = np.random.default_rng(1)
    U = rng.uniform(-1.9, 1.9, 4000); W = rng.uniform(-1.7, 1.7, 4000)    # includes points outside (clamped by both)
    ax, ay, sx, sy = ev(M, U, W)
    Ax, Ay, Sx, Sy = bow_phi.evaluate(blk, U, W)
    vx, vy = M['v_used']['x'], M['v_used']['y']
    dev = max(np.abs(Ax * vx - ax).max(), np.abs(Ay * vy - ay).max(),
              np.abs((1 + Sx) * vx - sx).max(), np.abs((1 + Sy) * vy - sy).max())
    print('max deviation vs a2_final.evaluate [um/ns]:', dev)
    assert dev < 1e-12


def _uw(wx, px, wy, py):
    return ((px + 15 * (wx * 1e3 - 0.1) / (1.01 * 37.0) - 200) / 100, (py + 15 * (wy * 1e3 + 0.2) / (0.98 * 37.0) - 200) / 100)


def test_conversion_formula_scalar_and_pairs():
    cal = _cal({'phi': _blk(), 'h': {'gu': 0.3}})
    rng = np.random.default_rng(2)
    wx, wy = rng.normal(0, 0.01, 60), rng.normal(0, 0.01, 60)
    px, py = rng.uniform(40, 360, 60), rng.uniform(40, 360, 60)
    v = reco.bow_pairs(cal, wx, px, wy, py)
    for i in range(60):
        fx = SimpleNamespace(p0=px[i], w=wx[i], tan_theta=0, theta_deg=0, slope_reliable=True)
        fy = SimpleNamespace(p0=py[i], w=wy[i], tan_theta=0, theta_deg=0, slope_reliable=True)
        reco.apply_bow(cal, fx, fy)
        assert abs(fx.tan_theta - v['tan_x'][i]) < 1e-12 and abs(fy.tan_theta - v['tan_y'][i]) < 1e-12
        assert abs(fx._bow_h - v['h_x'][i]) < 1e-12
        U, W = (c[i] for c in _uw(wx, px, wy, py))
        ax, ay, sx, sy = (a[0] for a in bow_phi.evaluate(cal.bow['phi'], U, W))
        assert abs(fx.tan_theta - ((wx[i] * 1e3 - 0.1) / (1.01 * 37.0 * (1 + sx)) - ax)) < 1e-12
        assert abs(fy.tan_theta - ((wy[i] * 1e3 + 0.2) / (0.98 * 37.0 * (1 + sy)) - ay)) < 1e-12
    assert np.allclose(v['a_x'], bow_phi.evaluate(cal.bow['phi'], *_uw(wx, px, wy, py))[0])


def test_replaces_tilt_and_scale():
    plain = {'a_x': {'cu': 0.05}, 's_x': {'su': 0.1}, 'h': {'gu': 0.3}}
    c1 = _cal(dict(plain, phi=_blk())); c2 = _cal({'phi': _blk(), 'h': {'gu': 0.3}})
    assert reco.tan_bow(c1, 'x', 0.004, 0.5, -0.3) == reco.tan_bow(c2, 'x', 0.004, 0.5, -0.3)
    c3 = _cal({'phi': _blk()}, v_map={'xy': {'e': [0, 200, 400], 'v_rel': [[1.2, 1.2], [1.2, 1.2]]}})
    assert reco.tan_bow(c3, 'x', 0.004, 0.5, -0.3) == reco.tan_bow(c2, 'x', 0.004, 0.5, -0.3)


def test_clamped_to_domain():
    blk = _blk(); (ulo, uhi), (wlo, whi) = blk['U_domain'], blk['W_domain']
    a = bow_phi.evaluate(blk, [uhi + 0.4, ulo - 0.4], [whi + 0.3, wlo - 0.3])
    b = bow_phi.evaluate(blk, [uhi, ulo], [whi, wlo])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    cal = _cal({'phi': blk})
    assert reco.tan_bow(cal, 'x', 0.004, uhi + 0.4, 0.0) == reco.tan_bow(cal, 'x', 0.004, uhi, 0.0)


def test_nan_falls_back_to_plain_block():
    plain = {'a_x': {'cu': 0.05}, 'a_y': {'cw': 0.02}, 's_x': {'su': 0.1}}
    cal = _cal(dict(plain, phi=_blk())); ref = _cal(plain)
    assert reco.tan_of(cal, 'x', 0.004, np.nan) == reco.tan_of(ref, 'x', 0.004, np.nan)
    assert reco.tan_of(cal, 'x', 0.004, 150.0) != reco.tan_of(ref, 'x', 0.004, 150.0)
    fx = SimpleNamespace(p0=np.nan, w=0.004, tan_theta=0, theta_deg=0, slope_reliable=True)
    fy = SimpleNamespace(p0=150.0, w=0.002, tan_theta=0, theta_deg=0, slope_reliable=True)
    gx = copy.copy(fx); gy = copy.copy(fy)
    reco.apply_bow(cal, fx, fy); reco.apply_bow(ref, gx, gy)
    assert fx.tan_theta == gx.tan_theta and fy.tan_theta == gy.tan_theta
    args = ([0.004, 0.003], [np.nan, 150.0], [0.002, 0.001], [150.0, 160.0])
    v, w = reco.bow_pairs(cal, *args), reco.bow_pairs(ref, *args)
    assert v['tan_x'][0] == w['tan_x'][0] and v['tan_y'][0] == w['tan_y'][0]
    assert v['tan_x'][1] != w['tan_x'][1]            # the finite pair does use the map


def test_bundle_roundtrip():
    cal = synth_bundle()
    cal.bow = {'h': {'gu': 0.2}, 'phi': _blk()}
    with tempfile.TemporaryDirectory() as d:
        cal.save(d)
        back = CalibrationBundle.load(d)
        assert json.dumps(back.bow, sort_keys=True) == json.dumps(cal.bow, sort_keys=True)
        a = bow_phi.evaluate(cal.bow['phi'], [0.3, -1.0], [0.7, 0.2]); b = bow_phi.evaluate(back.bow['phi'], [0.3, -1.0], [0.7, 0.2])
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        bad = copy.deepcopy(cal); bad.bow['phi']['version'] = 99
        with pytest.raises(ValueError):
            bad.save(os.path.join(d, 'bad'))


# ----------------------------------------------------------------------------- no-phi path unchanged
def _old_bow_pairs(cal, wx, p0x, wy, p0y):
    """bow_pairs verbatim as before 2026-11-24 (no phi)"""
    wx, p0x, wy, p0y = (np.asarray(v, float) for v in (wx, p0x, wy, p0y))
    b = cal.bow or {}
    kv = {p: cal.kw.get(p, 1.0) * cal.v_drift for p in 'xy'}
    tc = {p: (w * 1e3 - cal.w0.get(p, 0.0)) / kv[p] for p, w in (('x', wx), ('y', wy))}
    U = ((p0x + reco.BOW_MID_MM * tc['x']) - 200.0) / 100.0
    W = ((p0y + reco.BOW_MID_MM * tc['y']) - 200.0) / 100.0
    with np.errstate(invalid='ignore'):
        U = np.where(np.isfinite(U) & (np.abs(U) <= reco.BOW_UW_MAX), U, 0.0)
        W = np.where(np.isfinite(W) & (np.abs(W) <= reco.BOW_UW_MAX), W, 0.0)
    out = {}
    for p, w in (('x', wx), ('y', wy)):
        A, sb = b.get(f'A_{p}') or {}, b.get(f's_{p}') or {}
        S = 1.0 + sb.get('su', 0.0) * U + sb.get('sw', 0.0) * W
        a = b.get(f'a_{p}')
        if a:
            av = reco._bow_poly(a, U, W) * np.ones_like(U)
            out[f'tan_{p}'] = (w * 1e3 - cal.w0.get(p, 0.0)) / (kv[p] * S) - av
        else:
            av = np.zeros_like(U)
            out[f'tan_{p}'] = (w * 1e3 - cal.w0.get(p, 0.0) - reco._bow_poly(A, U, W)) / (kv[p] * S)
        out[f'a_{p}'] = av
        h = b.get(f'h_{p}') or b.get('h')
        out[f'h_{p}'] = (reco._bow_poly(h, U, W, ('gu', 'gw'), ('kuu', 'kww', 'kuw')) * np.ones_like(U)) if h else np.zeros_like(U)
    return out


def test_no_phi_synthetic_identical():
    cal = _cal({'a_x': {'cu': 0.01, 'auw': 0.003}, 'A_y': {'cw': 30.0}, 'h_x': {'gu': 0.5}, 's_y': {'su': 0.02}})
    rng = np.random.default_rng(3)
    args = (rng.normal(0, 0.01, 500), rng.uniform(-50, 450, 500), rng.normal(0, 0.01, 500), rng.uniform(-50, 450, 500))
    args[1][::50] = np.nan
    new, old = reco.bow_pairs(cal, *args), _old_bow_pairs(cal, *args)
    for k in old:
        assert np.array_equal(new[k], old[k], equal_nan=True), k


@pytest.mark.skipif(not glob.glob(f'{GATE}/*/*/*/bow_gate_table.parquet'), reason='pass-1 gate tables not on this machine')
def test_no_phi_real_bundles_identical():
    import pandas as pd
    n = 0
    for d in sorted(glob.glob(f'{GATE}/*/*/*/')):
        b = json.load(open(d + 'calib_bundle/bundle.json'))
        if b.get('v_map', {}).get('xy'):
            continue
        cal = SimpleNamespace(bow=b.get('bow') or {}, v_map=b.get('v_map') or {}, w0=b['w0'], kw=b['kw'], v_drift=b['v_drift'])
        t = pd.read_parquet(d + 'bow_gate_table.parquet', columns=['x_w', 'x_p0', 'y_w', 'y_p0'])
        new, old = reco.bow_pairs(cal, t.x_w, t.x_p0, t.y_w, t.y_p0), _old_bow_pairs(cal, t.x_w, t.x_p0, t.y_w, t.y_p0)
        for k in old:
            assert np.array_equal(new[k], old[k], equal_nan=True), (d, k)
        n += len(t)
    print('rows compared:', n)
    assert n > 10000
