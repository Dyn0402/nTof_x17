"""The y orientation of stage 3 (../../COORDINATES.md, HANDOFF_Y_FLIP_FIX.md).

Synthetic, data-free checks of the convention, the loader flag and the guard.
The data check (cosmic-muon timing on run_149) is the last test and skips where
its table is not staged.

    .venv/bin/python -m pytest sept26_prelim_analysis/tests/test_y_convention.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ntof_tracking.reco import geometry as G                       # noqa: E402
from sept26_prelim_analysis import build_tracks as BT              # noqa: E402
from sept26_prelim_analysis import y_guard                         # noqa: E402

H = BT.STRIP_MAP_HALF
D_PERP = 234.6


def _tr(arm):
    c = G.W_HAT[arm] * D_PERP - G.U_HAT[arm] * G.PINWHEEL[arm]
    ang = {'A': 0.0, 'C': 180.0, 'B': -90.0, 'D': 90.0}[arm]
    return G.DetTransform(name=f'mx17_{arm}', arm=arm, center=c, R=G._rot_y(ang))


def _raw(n=400, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame(dict(x_p0=r.uniform(40, 360, n), y_p0=r.uniform(40, 360, n),
                             x_tan_theta=r.uniform(-.5, .5, n),
                             y_tan_theta=r.uniform(-.5, .5, n)))


@pytest.mark.parametrize('arm', 'AC')
def test_transform_axes(arm):
    """Local v is global +y for every arm (the y view's hardware direction)."""
    tr = _tr(arm)
    np.testing.assert_allclose(tr.R @ [0, 1, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(tr.R @ [1, 0, 0], G.U_HAT[arm], atol=1e-12)


@pytest.mark.parametrize('arm', 'AC')
def test_y_is_strip_up_and_views_share_tan_convention(arm):
    """Hardware: y strip coordinate = H + global y offset (channel 0 at the
    bottom).  Reco: both views obey strip(w) = p0 + tan_raw * k * w, so x (the
    externally measured view) and y are on one convention."""
    k, w = 1.3, 17.0
    df = BT.local_and_global(_raw(), _tr(arm), k=k)
    tr = _tr(arm)
    P0 = df[['p0_x', 'p0_y', 'p0_z']].to_numpy()
    D = df[['d_x', 'd_y', 'd_z']].to_numpy()
    # strip y at the strip plane = H + global y - centre y
    np.testing.assert_allclose(H + P0[:, 1] - tr.center[1], df.y_p0, atol=1e-9)
    # walk to depth w along the line and read both strip coordinates back
    loc0 = (P0 - tr.center) @ tr.R
    dl = D @ tr.R
    s = w / -dl[:, 2]
    loc = loc0 + s[:, None] * dl
    np.testing.assert_allclose(-loc[:, 2], w, atol=1e-9)
    sx = H - loc[:, 0]                       # IN_PLANE_SIGN = -1 on x
    sy = H + loc[:, 1]                       # the y strip coordinate points up
    np.testing.assert_allclose(sx, df.x_p0 + df.x_tan_theta * k * w, atol=1e-9)
    np.testing.assert_allclose(sy, df.y_p0 + df.y_tan_theta * k * w, atol=1e-9)


def test_fix_is_an_exact_y_mirror_of_the_0907_convention():
    raw, tr, k = _raw(), _tr('A'), 1.2
    new = BT.local_and_global(raw, tr, k=k)
    xl = -(raw.x_p0 - H).to_numpy()
    yl = -(raw.y_p0 - H).to_numpy()                       # 2026-09-07..10-13
    tx, ty = raw.x_tan_theta.to_numpy() * k, raw.y_tan_theta.to_numpy() * k
    P0 = tr.local_to_global(xl, yl, 0 * xl)
    D = tr.local_to_global(xl - tx * 30, yl - ty * 30, 30 + 0 * xl) - P0
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    np.testing.assert_allclose(new.p0_y, -P0[:, 1], atol=1e-9)
    np.testing.assert_allclose(new.d_y, -D[:, 1], atol=1e-12)
    np.testing.assert_allclose(new.tany, -ty, atol=1e-12)
    for c, i in (('x', 0), ('z', 2)):
        np.testing.assert_allclose(new[f'p0_{c}'], P0[:, i], atol=1e-9)
        np.testing.assert_allclose(new[f'd_{c}'], D[:, i], atol=1e-12)


def _late_tracks(src_y, n=600, seed=1):
    """Stage-3-like rows for lines from a source at height src_y on the axis."""
    r = np.random.default_rng(seed)
    rows = []
    for arm in 'AC':
        tr = _tr(arm)
        P0 = tr.local_to_global(r.uniform(-150, 150, n), r.uniform(-150, 150, n), np.zeros(n))
        S = np.c_[r.normal(0, 3, n), src_y + r.normal(0, 8, n), r.normal(0, 3, n)]
        D = P0 - S
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        rows.append(pd.DataFrame(dict(
            arm=arm, p0_x=P0[:, 0], p0_y=P0[:, 1], p0_z=P0[:, 2],
            d_x=D[:, 0], d_y=D[:, 1], d_z=D[:, 2], tanx=0.1, tany=0.1, gated=True,
            angle_calibrated=True, t_since_flash_ns=30e6)))
    return pd.concat(rows, ignore_index=True)


def test_guard_passes_nose_and_refuses_mirror():
    ok = y_guard.late_y_crossing(_late_tracks(-30.0))
    assert ok['verdict'] == 'pass', ok
    assert all(abs(ok[a]['y_cross_mm'] + 30) < 3 for a in 'AC'), ok
    bad = y_guard.late_y_crossing(_late_tracks(+33.0))
    assert bad['verdict'] == 'FAIL', bad
    with pytest.raises(RuntimeError):
        y_guard.enforce(bad, 'synthetic mirror')
    few = y_guard.late_y_crossing(_late_tracks(-30.0, n=10))
    assert few['verdict'] == 'insufficient'


def test_stored_y_sign(tmp_path):
    def side(name, **kw):
        (tmp_path / f'tracks_run_1_{name}.meta.json').write_text(json.dumps(kw))
    side('a', built='2026-10-14T00:00:00+00:00', y_convention='up')
    assert BT.stored_y_sign(tmp_path) == 1.0
    side('b', built='2026-09-20T00:00:00+00:00')
    with pytest.raises(ValueError):                      # mixed
        BT.stored_y_sign(tmp_path)
    assert BT.stored_y_sign(tmp_path / 'tracks_run_1_b.parquet') == -1.0
    side('c', built='2026-08-30T00:00:00+00:00')
    with pytest.raises(ValueError):                      # pre-09-07: not a mirror
        BT.stored_y_sign(tmp_path / 'tracks_run_1_c.parquet')
    df = pd.DataFrame(dict(p0_y=[1.0], d_y=[.5], tany=[.2], p0_x=[3.0]))
    up = BT.y_up(df, tmp_path / 'tracks_run_1_b.parquet')
    assert up.p0_y[0] == -1 and up.d_y[0] == -.5 and up.tany[0] == -.2 and up.p0_x[0] == 3


AC149 = REPO / 'ntof_cosmics/results/y_orientation/ac149.parquet'


@pytest.mark.skipif(not AC149.is_file(), reason='run_149 cosmic table not staged here')
def test_cosmic_muons_hit_the_higher_arm_first():
    """Primary guard.  Rebuild the table first:
    python ntof_cosmics/y_orientation_check.py cosmic"""
    sys.path.insert(0, str(REPO / 'ntof_cosmics'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('yoc', REPO / 'ntof_cosmics/y_orientation_check.py')
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['yoc', 'none']
    spec.loader.exec_module(m)
    meta = sorted((REPO / 'ntof_cosmics/results/tracking/k_run_147').glob('tracks_*.meta.json'))
    if not meta or json.loads(meta[0].read_text()).get('y_convention') != 'up':
        pytest.skip('run_149 tracks not yet rebuilt on the fixed convention')
    v = m.cosmic_verdict(AC149)
    assert v['ok'], v


def test_regen_compare_accepts_only_a_pure_mirror():
    from sept26_prelim_analysis.regen_stage3 import compare
    old = pd.DataFrame(dict(tag='t', event_id=[1, 2, 3], arm='A', track_id=0,
                            p0_x=[1., 2, 3], p0_y=[4., -5, 6], d_y=[.1, -.2, .3],
                            tany=[.1, .2, -.3], gated=True, pred_ls=[True, False, True]))
    new = old.copy()
    for c in ('p0_y', 'd_y', 'tany'):
        new[c] = -old[c]
    new['pred_ls'] = [True, True, True]
    r = compare(old, new.iloc[::-1])
    assert r['ok'] and r['changed']['pred_ls'] == 1, r
    bad = new.copy()
    bad['p0_x'] = bad.p0_x + 1
    r = compare(old, bad)
    assert not r['ok'] and r['y_ok'] and 'p0_x' in r['drift']
    assert not compare(old, old)['y_ok']


def _fake_dir(tmp_path, n=2):
    for i in range(n):
        old = pd.DataFrame(dict(tag='t', event_id=[1, 2], arm='A', track_id=0, p0_x=[1., 2],
                                p0_y=[3., -4], d_y=[.1, .2], tany=[.1, -.2], gated=True,
                                k_source='run_145'))
        old.to_parquet(tmp_path / f'tracks_run_1_s{i}.parquet')
        (tmp_path / f'tracks_run_1_s{i}.meta.json').write_text(json.dumps(dict(
            run='run_1', subrun=f's{i}', reco_dir='x', built='2026-09-20T00:00:00')))


def _fake_build(bad_subrun=None):
    def build(run, subrun, reco, **kw):
        t = pd.read_parquet(f'{REGEN_DIR}/tracks_{run}_{subrun}.parquet').drop(columns='k_source')
        for c in ('p0_y', 'd_y', 'tany'):
            t[c] = -t[c]
        if subrun == bad_subrun:
            t['p0_y'] = t.p0_y + 1
        return t, dict(y_convention='up', y_guard=dict(verdict='insufficient'))
    return build


def test_regen_is_all_or_nothing(tmp_path, monkeypatch):
    global REGEN_DIR
    REGEN_DIR = tmp_path
    from sept26_prelim_analysis import regen_stage3 as RS
    _fake_dir(tmp_path)
    monkeypatch.setattr(BT, 'build', _fake_build(bad_subrun='s1'))
    monkeypatch.setattr(sys, 'argv', ['x', str(tmp_path), '--jobs', '1'])
    assert RS.main() == 1
    assert 'y_convention' not in json.loads((tmp_path / 'tracks_run_1_s0.meta.json').read_text())
    monkeypatch.setattr(BT, 'build', _fake_build())
    assert RS.main() == 0
    assert BT.stored_y_sign(tmp_path) == 1.0
    new = pd.read_parquet(tmp_path / 'tracks_run_1_s0.parquet')
    assert list(new.p0_y) == [-3., 4.] and (new.k_source == 'run_145').all()
    assert (tmp_path / 'pre_yfix' / 'tracks_run_1_s0.parquet').exists()
    assert not (tmp_path / RS.STAGING).exists()
