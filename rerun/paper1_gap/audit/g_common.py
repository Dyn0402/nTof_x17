"""g_common.py -- shared helpers of audit test G (bench fine angle map vs the n_TOF in-situ angle map).

Frames (derivation in RESULT.md, section 1):
  n_TOF local position   x_local = -(x_p0 - 199.29)   (build_tracks.IN_PLANE_SIGN = -1, line 59/263)
                         y_local = +(y_p0 - 199.29)   (IN_PLANE_SIGN_Y = +1, run145_target_imaging.py:97)
  n_TOF map residual     r = (global d_p/d_z of the chamber's track) - (joined A-C slope), cells indexed by (x_local,y_local)
  global slope from the fit angle (build_tracks.local_and_global + geometry.DetTransform, checked on the tables):
       x view, A and C : d_x/d_z = + x_tan_theta
       y view, A       : d_y/d_z = - y_tan_theta          (TAN_SIGN_Y = -1 and A looks along -z)
       y view, C       : d_y/d_z = + y_tan_theta          (TAN_SIGN_Y = -1 and C looks along +z)
  bench raw residual     e = x_tan_theta - tan_ref_x ; tan convention d(strip coordinate)/dz_up (COORDINATES.md 4b).
  => a_ntof(x_local,y_local) = SGN * a_bench(strip -> local), s_ntof = s_bench; SGN = +1 (x, A and C), -1 (y, A), +1 (y, C).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = '/home/dylan/PycharmProjects/nTof_x17'
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/g'
sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/rerun/paper1_gap')
HALF = 199.29
EDGES = np.arange(-200.0, 201.0, 50.0)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
NC = len(CENTRES)
MID = 14.0

ROWS = {
    'A': ['mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
          'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'],
    'C': ['mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
          'mx17_det6_det7_overnight_6-26-26/longer_run/mx17_6'],
}
DET = {'A': 'mx17_3', 'C': 'mx17_6'}
SGN = {('A', 'x'): 1.0, ('A', 'y'): -1.0, ('C', 'x'): 1.0, ('C', 'y'): 1.0}
TAN_MAX = 0.45      # the n_TOF clean-line cut |j| < 0.45
RES_MAX = 0.15      # |r| < 0.15
N_MIN_NTOF = 80


def load_bench(arm, rel, margin=30.0):
    """GM.load_table's selection with the chamber's own active area; margin = interior margin in mm (30 = standard)."""
    from common import mx17_active_area as AA
    box = AA.TRUE_ACTIVE_BY_DET[DET[arm]]
    d = pd.read_parquet(f'{WORK}/pass1_all/{rel}/bow_gate_table.parquet')
    d = d[(d.n_tracks == 1) & d.x_ok & d.y_ok & d.x_quality_ok & d.y_quality_ok & ~d.spark
          & (d.Chi2X < 1) & (d.Chi2Y < 1) & (d.NClusX >= 4) & (d.NClusY >= 4)].copy()
    d['xm'] = d.ref_mesh_x + MID * d.tan_ref_x
    d['ym'] = d.ref_mesh_y + MID * d.tan_ref_y
    d = d[(d.xm > box['x'][0] + margin) & (d.xm < box['x'][1] - margin)
          & (d.ym > box['y'][0] + margin) & (d.ym < box['y'][1] - margin)]
    d['x_local'] = -(d.x_p0 - HALF)
    d['y_local'] = +(d.y_p0 - HALF)
    return d


def robust_line(j, r, n_iter=3):
    """angle_map.robust_line, verbatim."""
    keep = np.ones(len(j), bool)
    a = s = 0.0
    for _ in range(n_iter):
        if keep.sum() < 20:
            break
        s, a = np.polyfit(j[keep], r[keep], 1)
        res = r - (a + s * j)
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * sig
    return float(a), float(s), int(keep.sum())


def cell_index(x, y):
    ix = np.clip(np.digitize(x, EDGES) - 1, 0, NC - 1)
    iy = np.clip(np.digitize(y, EDGES) - 1, 0, NC - 1)
    return ix, iy


def fit_cells(x, y, j, r, n_min, nboot=0, seed=1):
    """per-cell robust a, s (and bootstrap errors); arrays [ix, iy] in (x_local, y_local)."""
    ix, iy = cell_index(x, y)
    A = np.full((NC, NC), np.nan); S = A.copy(); N = np.zeros((NC, NC), int)
    eA = A.copy(); eS = A.copy()
    rng = np.random.default_rng(seed)
    for i in range(NC):
        for k in range(NC):
            m = (ix == i) & (iy == k)
            N[i, k] = m.sum()
            if m.sum() >= n_min:
                jj, rr = j[m], r[m]
                A[i, k], S[i, k], _ = robust_line(jj, rr)
                if nboot:
                    b = []
                    for _ in range(nboot):
                        q = rng.integers(0, len(jj), len(jj))
                        b.append(robust_line(jj[q], rr[q])[:2])
                    b = np.array(b)
                    eA[i, k], eS[i, k] = b[:, 0].std(), b[:, 1].std()
    return dict(a=A, s=S, n=N, ea=eA, es=eS)


def bench_maps(arm, margin=30.0, n_min=60, nboot=40):
    """returns {'rows': [map_row1, map_row2], 'all': map of the pooled rows, 'tab': pooled table}."""
    tabs = [load_bench(arm, rel, margin) for rel in ROWS[arm]]
    out = {'rows': [], 'n_events': [len(t) for t in tabs]}
    allt = pd.concat([t.assign(_row=i) for i, t in enumerate(tabs)])
    # time halves: parity of the acquisition-file index inside each row (same run, interleaved in time)
    par = np.zeros(len(allt), int)
    pos = 0
    for i, t in enumerate(tabs):
        tags = sorted(t.acq_tag.unique())
        mp = {tg: k % 2 for k, tg in enumerate(tags)}
        par[pos:pos + len(t)] = t.acq_tag.map(mp).to_numpy()
        pos += len(t)
    out['n_tags'] = [t.acq_tag.nunique() for t in tabs]
    for lab, t in [('r1', tabs[0]), ('r2', tabs[1]), ('all', allt), ('h0', allt[par == 0]), ('h1', allt[par == 1])]:
        res = {}
        for p in 'xy':
            j = t[f'tan_ref_{p}'].to_numpy(float)
            r = (t[f'{p}_tan_theta'] - t[f'tan_ref_{p}']).to_numpy(float)
            ok = (np.abs(j) < TAN_MAX) & (np.abs(r) < RES_MAX) & np.isfinite(r)
            res[p] = fit_cells(t.x_local.to_numpy()[ok], t.y_local.to_numpy()[ok], j[ok], r[ok], n_min,
                               nboot=nboot if lab == 'all' else 0)
            res[p]['a0'], res[p]['s0'], _ = robust_line(j[ok], r[ok])
        out[lab] = res
    return out


def ntof_map(arm, view, which='r2'):
    f = {'r2': f'{REPO}/rerun/ntof/stage3/angle_map/angle_map_r2.json',
         'r1': f'{REPO}/rerun/ntof/stage3/angle_map/angle_map.json'}[which]
    m = json.load(open(f))['maps'][f'{arm}_{view}']
    return {k: np.asarray(m[k], float) for k in ('a', 's', 'n')} | {'a0': m['a0'], 's0': m['s0']}


def pearson(u, v):
    ok = np.isfinite(u) & np.isfinite(v)
    return float(np.corrcoef(u[ok], v[ok])[0, 1]) if ok.sum() > 3 else np.nan


def boot_corr(u, v, nb=2000, seed=5):
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = u[ok], v[ok]
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(nb):
        q = rng.integers(0, len(u), len(u))
        if np.std(u[q]) > 0 and np.std(v[q]) > 0:
            b.append(np.corrcoef(u[q], v[q])[0, 1])
    return float(np.std(b))


def ols(x, y):
    """slope, intercept, slope error (y against x)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    p, cov = np.polyfit(x, y, 1, cov=True)
    return float(p[0]), float(p[1]), float(np.sqrt(cov[0, 0]))


def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))
