"""Audit task D: shared helpers (rows of all five chambers, selection, cell maps, arms).

Imports the audited modules (gapmap, track_test, cellobs) and reuses their helpers; nothing is written into the repo.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_gap'))
sys.path.insert(0, REPO)
import gapmap as GM  # noqa: E402
import track_test as TT  # noqa: E402
import cellobs as CO  # noqa: E402
from common import mx17_active_area as AA  # noqa: E402

WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/d'
os.makedirs(OUT, exist_ok=True)
ALL = f'{WORK}/pass1_all'

# key: (det, run dir, sub-run, label).  Voltages from each run_config.json / hv_monitor channel map
# (0:6 bottom drift, 0:7 top drift, 3:3 bottom resist, 3:4 top resist); 6-25 is not on disk (ambiguous in the notes).
ROWS = {
    'det2:long':   ('mx17_2', 'mx17_det2_det3_overnight_6-22-26', 'long_run', '6-22 long'),
    'det2:longer': ('mx17_2', 'mx17_det2_det3_overnight_6-22-26', 'longer_run', '6-22 longer'),
    'det2:short':  ('mx17_2', 'mx17_det2_det3_overnight_6-22-26', 'short_run', '6-22 short'),
    'det3:d625':   ('mx17_3', 'mx17_det3_day_6-25-26', 'long_run', '6-25 day'),
    'det3:sat':    ('mx17_3', 'mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V', '6-27 sat'),
    'det3:p2':     ('mx17_3', 'mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check', '6-28 p2'),
    'det4:623':    ('mx17_4', 'mx17_det3_det4_overnight_6-23-26', 'long_run', '6-23 long'),
    'det4:624':    ('mx17_4', 'mx17_det4_day_6-24-26', 'long_run', '6-24 day'),
    'det6:long':   ('mx17_6', 'mx17_det6_det7_overnight_6-26-26', 'long_run', '6-26 long'),
    'det6:longer': ('mx17_6', 'mx17_det6_det7_overnight_6-26-26', 'longer_run', '6-26 longer'),
    'det6:short':  ('mx17_6', 'mx17_det6_det7_overnight_6-26-26', 'short_run', '6-26 short'),
    'det7:long':   ('mx17_7', 'mx17_det6_det7_overnight_6-26-26', 'long_run', '6-26 long'),
    'det7:longer': ('mx17_7', 'mx17_det6_det7_overnight_6-26-26', 'longer_run', '6-26 longer'),
    'det7:short':  ('mx17_7', 'mx17_det6_det7_overnight_6-26-26', 'short_run', '6-26 short'),
}
VOLT = {  # (mesh/resist V, drift V)
    'det2:long': (495, 1000), 'det2:longer': (495, 1000), 'det2:short': (495, 1000),
    'det3:d625': (None, 500), 'det3:sat': (490, 1000), 'det3:p2': (490, 1000),
    'det4:623': (495, 600), 'det4:624': (495, 600),
    'det6:long': (495, 700), 'det6:longer': (495, 700), 'det6:short': (495, 700),
    'det7:long': (495, 700), 'det7:longer': (495, 700), 'det7:short': (495, 700),
}
CHAMBERS = {'det2': ['det2:long', 'det2:longer', 'det2:short'],
            'det3': ['det3:d625', 'det3:sat', 'det3:p2'],
            'det4': ['det4:623', 'det4:624'],
            'det6': ['det6:long', 'det6:longer', 'det6:short'],
            'det7': ['det7:long', 'det7:longer', 'det7:short']}
PUBLIC = {'det2': 'B', 'det3': 'A', 'det4': 'E', 'det6': 'C', 'det7': 'D'}
UL, WL = GM.__dict__.get('UL', (-1.7, 1.7)), (-1.51, 1.51)
UL = (-1.7, 1.7)
MARGIN = 30.0


def path(key, name):
    det, run, sub, _ = ROWS[key]
    return f'{ALL}/{run}/{sub}/{det}/{name}'


def active(key):
    det = ROWS[key][0]
    a = AA.TRUE_ACTIVE_BY_DET[det]
    return (a['x'][0] + MARGIN, a['x'][1] - MARGIN), (a['y'][0] + MARGIN, a['y'][1] - MARGIN)


def load(key, xhalf=False):
    """GM.load_table's selection with the chamber's own interior (30 mm margin).  xhalf: only xm >= 200 mm."""
    d = pd.read_parquet(path(key, 'bow_gate_table.parquet'))
    d = d[(d.n_tracks == 1) & d.x_ok & d.y_ok & d.x_quality_ok & d.y_quality_ok & ~d.spark
          & (d.Chi2X < 1) & (d.Chi2Y < 1) & (d.NClusX >= 4) & (d.NClusY >= 4)].copy()
    d['xm'] = d.ref_mesh_x + GM.MID * d.tan_ref_x
    d['ym'] = d.ref_mesh_y + GM.MID * d.tan_ref_y
    ix, iy = active(key)
    d = d[(d.xm > ix[0]) & (d.xm < ix[1]) & (d.ym > iy[0]) & (d.ym < iy[1])]
    if xhalf:
        d = d[d.xm >= 200.0]
    d = d.reset_index()
    d['Urec'] = (d.x_p0 + 14 * d.x_tan_theta - 200) / 100
    d['Wrec'] = (d.y_p0 + 14 * d.y_tan_theta - 200) / 100
    d['Uref'] = (d.xm - 200) / 100
    d['Wref'] = (d.ym - 200) / 100
    return d


def align(key):
    return json.load(open(path(key, 'alignment/alignment.json')))


def bundle(key):
    return json.load(open(path(key, 'calib_bundle/bundle.json')))


def view(d, p, idx='rec'):
    """arrays for view p within |tan_ref| <= 0.6: U, W, w (um/ns), tan_ref, pass-1 tan"""
    U = d['Urec'].values if idx == 'rec' else d['Uref'].values
    W = d['Wrec'].values if idx == 'rec' else d['Wref'].values
    w = d[f'{p}_w'].values * 1e3
    tr = d[f'tan_ref_{p}'].values
    tp = d[f'{p}_tan_theta'].values
    s = np.abs(tr) <= GM.TAN_LADDER
    return U[s], W[s], w[s], tr[s], tp[s]


def cells(U, W, w, tr, n, ul=UL, wl=WL, minn=40):
    """track_grid.cells with a settable range; cells with fewer than `minn` events are NaN (CO.line returns NaN < 40)."""
    eu, ew = np.linspace(*ul, n + 1), np.linspace(*wl, n + 1)
    i, j = np.searchsorted(eu, U) - 1, np.searchsorted(ew, W) - 1
    out = []
    for a in range(n):
        for b in range(n):
            m = (i == a) & (j == b)
            if m.sum() < minn:
                w0 = v = np.nan
                nk = 0
            else:
                w0, v, _, _, nk = CO.line(tr[m], w[m])
            out.append(dict(U=0.5 * (eu[a] + eu[a + 1]), W=0.5 * (ew[b] + ew[b + 1]), w0=w0, v=v, n=int(m.sum())))
    vm = np.nanmean([c['v'] for c in out]); wm = np.nanmean([c['w0'] for c in out])
    for c in out:
        c['vrel'] = c['v'] / vm; c['dw0'] = c['w0'] - wm
    return out


def arms(train, test, n, ul=UL, wl=WL):
    """angle (tan) arrays of the test view for const / w0+v / w0 using a map from `train` (track_grid procedure).
    train, test: tuples (U, W, w, tr, tp)."""
    U, W, w, tr, tp = test
    Uo, Wo, wo, tro, _ = train
    res = {}
    c = TT.robust_const(tr, w, 0.0, 1.0)
    res['const'] = (w - c[0]) / c[1]
    if n:
        C = cells(Uo, Wo, wo, tro, n, ul, wl)
        if np.sum(np.isfinite([c['v'] for c in C])) < 4:
            return res
        vr = TT.bilinear(C, 'vrel', U, W); dw = TT.bilinear(C, 'dw0', U, W)
        for arm, (dd, vv) in {'w0+v': (dw, vr), 'w0': (dw, np.ones_like(vr))}.items():
            c = TT.robust_const(tr, w, dd, vv)
            res[arm] = (w - c[0] - dd) / (c[1] * vv)
    return res


def s68(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    q = np.percentile(a, [16, 84]); return float((q[1] - q[0]) / 2)


def ddeg(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))


def boot_idx(n, nb, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, (nb, n))
