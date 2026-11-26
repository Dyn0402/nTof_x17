#!/usr/bin/env python3
"""build.py [tag ...] -- one table per row: the pass-1 effbudget rays joined with the M3 track that made each ray.

Inputs (fetch.py): the row's production M3 rays and its alignment; the effbudget per-ray table (rows.RAYS).
For every ray of the effbudget table (the recipe: chi2 < 1, NClus = 4, one good ray per event) we add what the M3
fit knows about itself: Chi2X, Chi2Y (sum of squared residuals of the 4-point line fit, mm^2, 2 degrees of freedom
each), the M3 slopes and positions, and the chi2 projected into each view of the chamber (the view x of a chamber
rotated by theta to M3 sees c^2 Chi2X + s^2 Chi2Y, as the pointing variance does).

A second table lists every single-track M3 ray with NClus = 4 at any chi2 (the relaxed set), for the reconstruction
of the chi2 > 1 tracks that the pass never offered to the fit.

Writes $WORK/tables/<tag>.parquet and $WORK/tables/<tag>.relaxed.parquet.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'cosmic_bench_analysis')]
import awkward as ak  # noqa: E402
from M3RefTracking import M3RefTracking  # noqa: E402
from wft.seed import SPARK_VETO_HITS  # noqa: E402
from common.mx17_fiducial import good_x  # noqa: E402
from rows import ROWS, INPUTS, RAYS, WORK, SHORT  # noqa: E402

OUT = os.path.join(WORK, 'tables')


def m3_table(m3dir, chi2_cut, al):
    """One row per kept event: the single recipe track, evaluated at the alignment z of each view."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        R = M3RefTracking(m3dir, chi2_cut=chi2_cut, min_nclus=4)
    d = R.ray_data
    g = {k: ak.to_numpy(d[k]).astype(float) for k in ('X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Z_Up', 'Z_Down',
                                                       'Chi2X', 'Chi2Y')}
    dz = g['Z_Down'] - g['Z_Up']
    sx = (g['X_Down'] - g['X_Up']) / dz
    sy = (g['Y_Down'] - g['Y_Up']) / dz
    T = pd.DataFrame(dict(event_id=ak.to_numpy(d['evn']).astype(int), chi2_X=g['Chi2X'], chi2_Y=g['Chi2Y'],
                          m3_sx=sx, m3_sy=sy))
    for v, z in (('x', al['z_x']), ('y', al['z_y'])):
        t = (z - g['Z_Up']) / dz
        T[f'm3_X_at_{v}'] = g['X_Up'] + t * (g['X_Down'] - g['X_Up'])
        T[f'm3_Y_at_{v}'] = g['Y_Up'] + t * (g['Y_Down'] - g['Y_Up'])
    return T.drop_duplicates('event_id', keep=False)


def add_view_chi2(T, al):
    th = np.deg2rad(al['theta_deg']); c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    T['chi2_vx'] = c2 * T.chi2_X + s2 * T.chi2_Y        # view x of the chamber
    T['chi2_vy'] = s2 * T.chi2_X + c2 * T.chi2_Y
    T['tan_m3'] = np.hypot(T.m3_sx, T.m3_sy)
    return T


def build(r):
    rel = f"{r['run']}/{r['subrun']}"
    al = json.load(open(f"{INPUTS}/{rel}/{r['det']}/alignment.json"))
    m3dir = f'{INPUTS}/{rel}/m3_tracking_root_v2/'
    rays = pd.read_parquet(f"{RAYS}/{rel}/{r['det']}/rays_{SHORT[r['det']]}.parquet")
    T = add_view_chi2(m3_table(m3dir, 1.0, al), al)
    D = rays.merge(T, on='event_id', how='left', validate='one_to_one')
    # closure: the ray's M3 position must be the one recomputed here
    d = np.abs(D.ref_x - al['ref_x_sign'] * D.m3_X_at_x)
    if not (np.nanmax(d) < 1e-6 and D.chi2_X.notna().all()):
        raise RuntimeError(f"{r['tag']}: M3 join does not close (max |d| {np.nanmax(d):.3g}, "
                           f"missing {D.chi2_X.isna().sum()})")
    D['good_x'] = good_x(r['det'], D.lo_x.to_numpy())
    D['interior_good'] = (D.region == 'interior') & D.good_x
    D['spark_veto'] = D.mult > SPARK_VETO_HITS
    D['tag'] = r['tag']; D['det'] = r['det']; D['z_x'] = al['z_x']; D['z_y'] = al['z_y']
    D['theta_deg'] = al['theta_deg']
    os.makedirs(OUT, exist_ok=True)
    D.to_parquet(f"{OUT}/{r['tag']}.parquet")
    Rx = add_view_chi2(m3_table(m3dir, 1e12, al), al)
    Rx['recipe'] = Rx.event_id.isin(T.event_id)
    Rx.to_parquet(f"{OUT}/{r['tag']}.relaxed.parquet")
    ig = D.interior_good & D.x_ok & D.y_ok & ~D.spark_veto
    print(f"{r['tag']:8s} rays {len(D):6d}  interior_good+fit {ig.sum():6d}  relaxed single NClus4 {len(Rx):6d} "
          f"(chi2>1: {(~Rx.recipe).sum()})  theta {al['theta_deg']:.2f}  z {al['z_x']:.0f}", flush=True)


if __name__ == '__main__':
    want = set(sys.argv[1:])
    for r in ROWS:
        if not want or r['tag'] in want:
            build(r)
