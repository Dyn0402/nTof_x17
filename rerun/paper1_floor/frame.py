#!/usr/bin/env python3
"""frame.py --out-dir D -- one per-event analysis frame per golden row for the §8 floor study (README.md).

Table level only, READ-ONLY on Analysis_pass1.  Per row and view, every M3-matched event of the full-coverage angles
(angles/per_event.npz, FD's 'full' sample) with:
  the angle residual   d_pass, d_vt [deg] = atan(tan_fit) - atan(tan_ref); the vt arm is res_rows.vt_arm (v -> v r(t),
                       w0/kw re-folded), identical to pass on a row with no v(t) map
  the sample flags     ig (03_angles' interior_good sample, paper1_eff/angles/<rel>/per_event.npz)
  the fit              every column of events.parquet for the view (chi2, dof, q_sum, q_u*, n_strips, n_sat/band/
                       flagged/dropped/candidates, w, t0, ftst, tan_err, ...) plus the other view's tan/ok/p0
  the M3 ray           Chi2X, Chi2Y, NClusX, NClusY, rayN of the recipe-selected track (m3_tracking_root_v2: chi2 < 1 and
                       NClus = 4 in both coordinates, exactly one such track), joined on evn = event_id
  the effbudget ray    mult, hw, cls, mesh_x/y, tan_x/y, dist_x/y (paper1_eff/budget/<rel>/rays_<det>.parquet)
Run on lxplus with RES_CODE=<code archive with rerun/paper1_res> (lock venv).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
import uproot

CODE = os.environ['RES_CODE']
sys.path[:0] = [os.path.join(CODE, 'rerun', 'paper1_res')]
import res_rows as rr  # noqa: E402
# two-track re-pass: FRAME_PASS_ROOT / FRAME_EFF_ROOT override rr.P1 / rr.EFF (defaults: Analysis_pass1, paper1_eff)
rr.P1 = os.environ.get('FRAME_PASS_ROOT', rr.P1)
rr.EFF = os.environ.get('FRAME_EFF_ROOT', rr.EFF)

ROWS = ('c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'o22_long_det2',
        'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'sat_det3',
        'g_det4', 'g_det6_long', 'g_det7_long')
JUNE = '/eos/experiment/ntof/data/x17/cosmic_bench/june_tests'
M3_CHI2, M3_NCLUS = 1.0, 4                       # qa_config.M3_CHI2_CUT / M3_MIN_NCLUS


def m3_tracks(r):
    """the recipe-selected M3 track per event (one good track), keyed on evn."""
    rows = []
    for f in sorted(glob.glob(os.path.join(JUNE, r['run'], r['subrun'], 'm3_tracking_root_v2', '*_rays.root'))):
        with uproot.open(f) as F:
            a = F['T'].arrays(['evn', 'rayN', 'Chi2X', 'Chi2Y', 'NClusX', 'NClusY'], library='ak')
        for evn, cx, cy, nx, ny in zip(a.evn.to_numpy(), a.Chi2X.to_list(), a.Chi2Y.to_list(), a.NClusX.to_list(),
                                       a.NClusY.to_list()):
            good = [i for i in range(len(cx)) if cx[i] < M3_CHI2 and cy[i] < M3_CHI2 and nx[i] >= M3_NCLUS
                    and ny[i] >= M3_NCLUS]
            if len(good) == 1:
                i = good[0]
                rows.append((int(evn), cx[i], cy[i], nx[i], ny[i], len(cx)))
    m = pd.DataFrame(rows, columns=['event_id', 'm3_chi2x', 'm3_chi2y', 'm3_nclusx', 'm3_nclusy', 'm3_ntracks'])
    return m[~m.event_id.duplicated(keep=False)].set_index('event_id')


def do_row(r, by, vtrows, out_dir):
    d = rr.SHORT[r['det']]
    wdir = os.path.join(rr.P1, rr.rel(r), 'wft')
    m, cal = rr.load_calib(wdir)
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    df = df[~df.event_id.duplicated(keep=False)].set_index('event_id')
    vrel = None
    if r['key'] in vtrows:
        vrel = pd.read_parquet(vtrows[r['key']]['table']).drop_duplicates('event_id').set_index('event_id')['v_rel']
    V = rr.vt_arm(wdir, df, cal, vrel)
    Z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    ANG = np.load(os.path.join(rr.EFF, 'angles', rr.rel(r), 'per_event.npz'))
    rays = pd.read_parquet(os.path.join(rr.EFF, 'budget', rr.rel(r), f'rays_{d}.parquet'))
    rays = rays.drop_duplicates('event_id', keep=False).set_index('event_id')
    M3 = m3_tracks(r)
    keep_r = ['mult', 'hw', 'cls', 'mesh_x', 'mesh_y', 'tan_x', 'tan_y', 'dist_x', 'dist_y']
    frames = []
    for p, o in (('x', 'y'), ('y', 'x')):
        e = Z[f'{p}_eid'].astype(int)
        k = ~pd.Index(e).duplicated(keep=False) & np.isin(e, df.index.to_numpy())
        e, tref = e[k], Z[f'{p}_tan_ref'][k]
        f = pd.DataFrame(index=pd.Index(e, name='event_id'))
        f['view'] = p
        f['tan_ref'] = tref
        t_pass = df[f'{p}_tan_theta'].reindex(e).to_numpy()
        t_vt = V[p]['tan'].reindex(e).to_numpy()
        t_vt = np.where(np.isfinite(t_vt), t_vt, t_pass)
        f['d_pass'] = np.degrees(np.arctan(t_pass) - np.arctan(tref))
        f['d_vt'] = np.degrees(np.arctan(t_vt) - np.arctan(tref))
        f['ig'] = np.isin(e, ANG[f'{p}_eid'].astype(int))
        for c in [c for c in df.columns if c.startswith(p + '_')]:
            f[c[2:]] = df[c].reindex(e).to_numpy()
        for c in ('tan_theta', 'ok', 'p0', 'q_sum', 'chi2', 'dof'):
            f['o_' + c] = df[f'{o}_{c}'].reindex(e).to_numpy()
        for c in ('n_hits', 'spark', 'n_tracks', 'bow_h', 'timestamp', 'n_splits'):
            f[c] = df[c].reindex(e).to_numpy()
        for c in keep_r:
            f['r_' + c] = rays[c].reindex(e).to_numpy() if c in rays else np.nan
        for c in M3.columns:
            f[c] = M3[c].reindex(e).to_numpy()
        frames.append(f.reset_index())
    F = pd.concat(frames, ignore_index=True)
    F['key'], F['det'] = r['key'], d
    path = os.path.join(out_dir, f"{r['key']}.parquet")
    F.to_parquet(path)
    print(r['key'], len(F), 'm3 joined', float(np.isfinite(F.m3_chi2x).mean()), 'ig', int(F.ig.sum()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    reg = list(csv.DictReader(open(os.path.join(CODE, 'rerun', 'pass_registry.csv'))))
    by = {r['key']: r for r in reg}
    vtrows = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'vt_all', 'results',
                                                                    'vt_corrected_rows.csv')))}
    for k in (a.only.split(',') if a.only else ROWS):
        do_row(by[k], by, vtrows, a.out_dir)


if __name__ == '__main__':
    main()
