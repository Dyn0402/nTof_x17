"""The n_TOF spline angle map (rerun/ntof/stage3/spline_map.py, fitted by rerun/ntof/improve/spline_map_fit.py).

Synthetic checks of the evaluator against the fitting code, of the clamps and of the round trip; then map_view on one
small real table (skipped where the views are not staged): a cell map must reproduce stage3_r2_vtmap exactly, and a
spline map with zero fields must leave the angles unchanged.

    .venv/bin/python -m pytest sept26_prelim_analysis/tests/test_spline_map.py -q
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rerun.ntof.stage3 import spline_map as SM                    # noqa: E402
from rerun.ntof.improve import spline_map_lib as SL                # noqa: E402

COMMITTED = REPO / 'rerun/ntof/stage3/angle_map/angle_map_r3.json'
WIDE = {'a_x': [-1e9, 1e9], 'a_y': [-1e9, 1e9], 's_x': [-1e9, 1e9], 's_y': [-1e9, 1e9]}


def _random_map(seed=0, tilt='pot', scale='sep', amp_t=2.0, amp_s=0.05):
    """A map file of one arm per chamber from random spline coefficients, via the fitting code's own layout."""
    rng = np.random.default_rng(seed)
    maps, models = {}, {}
    for arm in ('A', 'C', 'D'):
        sp = SL.Spline(tilt=tilt, scale=scale)
        sp.beta = np.concatenate([rng.normal(0, amp_t, sp.ntc), rng.normal(0, amp_s, sp.nsc)])
        t, s = sp.coef_arrays()
        maps[arm] = dict(arm=arm, g=list(SM.G[arm]),
                         tilt_basis=dict(lo=SL.LO, hi=SL.HI, h=sp.ht, nb=sp.nt),
                         scale_basis=dict(lo=SL.LO, hi=SL.HI, h=sp.hs, nb=sp.ns),
                         tilt=t, scale=s, clamp=dict(WIDE))
        models[arm] = sp
    return dict(meta=dict(model='spline'), maps=maps), models


@pytest.mark.parametrize('tilt,scale', [('pot', 'sep'), ('pot', 'common'), ('free', 'sep')])
def test_evaluator_matches_fitting_code(tilt, scale):
    M, models = _random_map(1, tilt, scale)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-200, 200, 5000), rng.uniform(-200, 200, 5000)
    for arm, sp in models.items():
        ref = sp.fields(x, y)
        got = SM.fields(M, arm, x, y)
        for k, v in zip(('a_x', 'a_y', 's_x', 's_y'), got):
            assert np.allclose(v, ref[k], rtol=0, atol=1e-12 * max(1.0, np.abs(ref[k]).max())), (arm, k)


def test_potential_tilt_is_a_gradient():
    """a_x, a_y of a potential map are the derivatives of phi (finite differences of the deriv-0 evaluation)."""
    M, _ = _random_map(3)
    m = M['maps']['A']
    bt = m['tilt_basis']
    x, y, h = np.array([-123.4, 0.3, 77.7]), np.array([55.5, -1.2, 150.0]), 1e-4
    phi = lambda xx, yy: SM.tensor_eval(m['tilt']['phi'], xx, yy, bt['lo'], bt['hi'], bt['h'])  # noqa: E731
    ax, ay, _, _ = SM.fields(M, 'A', x, y)
    assert np.allclose(ax, (phi(x + h, y) - phi(x - h, y)) / (2 * h), atol=1e-6)
    assert np.allclose(ay, (phi(x, y + h) - phi(x, y - h)) / (2 * h), atol=1e-6)


def test_positions_clamped_outside_domain():
    M, _ = _random_map(4)
    far = np.array([-1e4, -250.0, 210.0, 3e5])
    for arm in ('A', 'C', 'D'):
        out = SM.fields(M, arm, far, far[::-1])
        edge = SM.fields(M, arm, np.clip(far, -200, 200), np.clip(far[::-1], -200, 200))
        for o, e in zip(out, edge):
            assert np.all(np.isfinite(o))
            assert np.array_equal(o, e)


def test_fields_clamped_outside_training_support():
    """A field that blows up in a corner without lines is held to its range over the training lines."""
    M, _ = _random_map(5)
    m = M['maps']['C']
    phi = np.asarray(m['tilt']['phi'])
    phi[:4, :4] += 500.0 * np.arange(16).reshape(4, 4)        # a wild corner at x, y ~ -200
    m['tilt']['phi'] = phi.tolist()
    rng = np.random.default_rng(6)
    Dc = pd.DataFrame(dict(x=rng.uniform(-100, 100, 4000), y=rng.uniform(-100, 100, 4000)))
    m['clamp'] = SL.clamp_block(m, Dc)
    raw = SM.raw_fields(m, np.array([-199.0]), np.array([-199.0]))
    got = SM.fields(M, 'C', [-199.0], [-199.0])
    assert abs(raw[0][0]) > 10 * max(abs(v) for v in m['clamp']['a_x'])     # the corner is wild ...
    for k, v in zip(('a_x', 'a_y', 's_x', 's_y'), got):                      # ... and never reaches a track
        assert m['clamp'][k][0] - 1e-15 <= v[0] <= m['clamp'][k][1] + 1e-15


def test_nan_positions_unchanged():
    M, _ = _random_map(7)
    x = np.array([np.nan, 10.0, 20.0, np.nan])
    y = np.array([5.0, np.nan, 30.0, np.nan])
    tx, ty = np.array([0.1, -0.2, 0.3, 0.05]), np.array([-0.3, 0.2, 0.1, 0.0])
    for arm in ('A', 'C', 'D'):
        cx, cy = SM.correct_tan(M, arm, x, y, tx, ty)
        bad = ~(np.isfinite(x) & np.isfinite(y))
        assert np.array_equal(cx[bad], tx[bad]) and np.array_equal(cy[bad], ty[bad])
        assert not np.array_equal(cx[~bad], tx[~bad])
    bx, by = SM.correct_tan(M, 'B', x, y, tx, ty)                 # no map for B
    assert np.array_equal(bx, tx) and np.array_equal(by, ty)


@pytest.mark.parametrize('source', ['random', 'committed'])
def test_round_trip_and_global_frame(source):
    if source == 'committed':
        if not COMMITTED.exists():
            pytest.skip('angle_map_r3.json not built')
        M = SM.load(COMMITTED)
    else:
        M, _ = _random_map(8, amp_t=0.5, amp_s=0.02)
    rng = np.random.default_rng(9)
    n = 20000
    x, y = rng.uniform(-230, 230, n), rng.uniform(-230, 230, n)
    tx, ty = rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n)
    for arm in ('A', 'C', 'D'):
        mx, my = SM.uncorrect_tan(M, arm, x, y, tx, ty)
        cx, cy = SM.correct_tan(M, arm, x, y, mx, my)
        assert np.allclose(cx, tx, atol=1e-12) and np.allclose(cy, ty, atol=1e-12)
        gx, gy = SM.G[arm]
        sx, sy = SM.correct_global(M, arm, x, y, gx * mx, gy * my)       # the global frame gives the same answer
        assert np.allclose(sx, gx * cx, atol=1e-14) and np.allclose(sy, gy * cy, atol=1e-14)


def test_committed_map_contract():
    if not COMMITTED.exists():
        pytest.skip('angle_map_r3.json not built')
    M = SM.load(COMMITTED)
    assert SM.is_spline(M)
    assert M['meta']['version'] == 'r3'
    for arm in ('A', 'C', 'D'):
        m = M['maps'][arm]
        assert tuple(m['g']) == SM.G[arm]
        assert m['tilt']['kind'] == 'potential' and m['scale']['kind'] == 'per_view'
        assert np.asarray(m['tilt']['phi']).shape == (m['tilt_basis']['nb'],) * 2
        for k in ('a_x', 'a_y', 's_x', 's_y'):
            lo, hi = m['clamp'][k]
            assert lo < hi and abs(lo) < 0.5 and abs(hi) < 0.5
    assert M['maps']['D']['fit']['pointing']['n_cells'] > 0
    assert M['maps']['A']['fit']['pointing'] is None


# ------------------------------------------------------------------------------------------------ map_view on data
P = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1'
TABLE = 'tracks_run_131_cosbounce_cos_0000.parquet'
CELL_MAP = REPO / 'rerun/ntof/stage3/angle_map/angle_map_r2.json'


def _need_data():
    if not ((P / 'stage3_r2_vt' / TABLE).exists() and (P / 'stage3_r2_vtmap' / TABLE).exists()):
        pytest.skip('stage3_r2_vt / stage3_r2_vtmap not staged')


def _frames_equal(a, b):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
            assert np.array_equal(x.to_numpy(float), y.to_numpy(float), equal_nan=True), c
        else:
            assert x.astype(object).where(x.notna(), None).tolist() == y.astype(object).where(y.notna(), None) \
                .tolist(), c


def test_cell_map_through_map_view_unchanged(tmp_path):
    _need_data()
    from rerun.ntof.stage3 import map_view as MV
    M = json.load(open(CELL_MAP))
    assert MV.map_model(M) == 'cell'
    md5 = hashlib.md5(open(CELL_MAP, 'rb').read()).hexdigest()
    MV.one(P / 'stage3_r2_vt' / TABLE, tmp_path, M, CELL_MAP, md5, True)
    _frames_equal(pd.read_parquet(tmp_path / TABLE), pd.read_parquet(P / 'stage3_r2_vtmap' / TABLE))
    meta = json.load(open(tmp_path / TABLE.replace('.parquet', '.meta.json')))['map']
    assert meta['map_md5'] == md5 and meta['map_model'] == 'cell' and meta['map_file'] == str(CELL_MAP)


def test_zero_spline_map_is_identity(tmp_path):
    _need_data()
    from rerun.ntof.stage3 import map_view as MV
    M, _ = _random_map(10)
    for m in M['maps'].values():
        m['tilt']['phi'] = np.zeros_like(np.asarray(m['tilt']['phi'])).tolist()
        for k in ('s_x', 's_y'):
            m['scale'][k] = np.zeros_like(np.asarray(m['scale'][k])).tolist()
    MV.one(P / 'stage3_r2_vt' / TABLE, tmp_path, M, 'zero.json', 'x', True)
    t = pd.read_parquet(tmp_path / TABLE)
    for c in ('tanx', 'tany', 'd_x', 'd_y', 'd_z', 'gated'):
        assert np.array_equal(t[c].to_numpy(float), t[c + '_nomap'].to_numpy(float), equal_nan=True), c
    assert json.load(open(tmp_path / TABLE.replace('.parquet', '.meta.json')))['map']['map_model'] == 'spline'
