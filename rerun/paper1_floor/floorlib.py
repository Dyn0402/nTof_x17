"""floorlib.py -- shared helpers of the §8 floor study (README.md): the frames, FD's s68 convention, the tilt x-validation."""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FRAMES = os.environ.get('FLOOR_FRAMES') or os.path.join(HERE, 'frames')
CALIB = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
HEADLINE = {'det2': 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
            'det3': 'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3'}
CELL_MM = 40.0
BUDGET = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', 'results', 'budget_pass1.json')


def load(key):
    return pd.read_parquet(os.path.join(FRAMES, f'{key}.parquet'))


def s68(a):
    """FD's convention: 68th percentile of |d - median| over |d| <= 5 deg."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    if len(a) < 50:
        return np.nan
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def boot_s68(a, n=200, seed=1):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    if len(a) < 50:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [s68(rng.choice(a, len(a))) for _ in range(n)]
    return s68(a), float(np.std(b))


def q(a, b):
    """quadrature difference sqrt(a^2 - b^2), 0 when b >= a."""
    return float(np.sqrt(max(a * a - b * b, 0.0)))


def xval(d, keys, half, min_n=8):
    """cross-validated (even/odd event id) cell-median correction (fd_pass1.xval)."""
    df = pd.DataFrame({'d': d, 'h': half, **{f'k{i}': k for i, k in enumerate(keys)}})
    kc = [f'k{i}' for i in range(len(keys))]
    corr = np.zeros(len(d))
    for h in (0, 1):
        tr = df[(df.h != h) & (df.d.abs() <= 5)]
        med = tr.groupby(kc).d.agg(['median', 'size'])
        med = med[med['size'] >= min_n]['median']
        te = df[df.h == h]
        idx = pd.MultiIndex.from_frame(te[kc]) if len(kc) > 1 else pd.Index(te[kc[0]])
        corr[te.index] = med.reindex(idx).fillna(0.0).to_numpy()
    return d - corr


def sample(F, p, arm='d_vt', ig=True):
    """one view's events with a finite residual; adds ds (slope removed) and dt (slope + 2D 40 mm tilt removed)."""
    g = F[(F.view == p) & np.isfinite(F[arm]) & np.isfinite(F.p0) & np.isfinite(F.o_p0)]
    if ig:
        g = g[g.ig]
    g = g.reset_index(drop=True).copy()
    d, t = g[arm].to_numpy(), g.tan_ref.to_numpy()
    k = np.abs(d) < 5
    g['ds'] = d - np.polyval(np.polyfit(t[k], d[k], 1), t)
    half = (g.event_id % 2).to_numpy()
    g['dt'] = xval(g.ds.to_numpy(), (np.floor(g.p0 / CELL_MM).astype(int), np.floor(g.o_p0 / CELL_MM).astype(int)), half)
    return g


def budget():
    return json.load(open(BUDGET))
