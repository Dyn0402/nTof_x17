#!/usr/bin/env python3
"""edge_angles.py --rays-root R --angles-root A [--out results/edge_angles.json] [--figs figures]
-- the edge zone seen by the M3 reference (Paper I §5, D4, on pass-1 waveform products; the old 32_edge_fringe_field.py
used hits angles).  Per golden row, against the distance of the M3 ray at mid-gap to each side of the active-area box:

  * detection efficiency (the budget's 'detected': both views fitted, |r| <= 5 mm, <= 2 competing strips), the turn-on;
  * the angle bias: median of tan_fit - tan_ref per view (03_angles per_event.npz, the pass's own, no fiducial cut),
    and its s68.  A drift-field fringe bends the drift lines near a side, so it shows as a tan bias of opposite sign on
    the two opposite sides of the view it acts along;
  * the position bias: median of the budget's res_x / res_y.
Read-only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402
from edge_study import GOLD, SHORT  # noqa: E402

D_EDGES = np.array([-10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 200])
SIDES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) >= 15 else np.nan


def load(det, rays_root, ang_root):
    row = GOLD[det]
    R = pd.read_parquet(glob.glob(os.path.join(rays_root, row, det, f'rays_{SHORT[det]}.parquet'))[0])
    R = R[~R.not_measured].copy()
    z = np.load(glob.glob(os.path.join(ang_root, row, det, 'wft', 'angles', 'per_event.npz'))[0])
    for p in 'xy':
        A = pd.DataFrame({'event_id': z[f'{p}_eid'], f'dt_{p}': z[f'{p}_tan_fit'] - z[f'{p}_tan_ref']})
        R = R.merge(A.drop_duplicates('event_id'), on='event_id', how='left')
    b = TRUE_ACTIVE_BY_DET[det]
    px, py = R.mesh_x + 15 * R.tan_x, R.mesh_y + 15 * R.tan_y
    dd = np.stack([px - b['x'][0], b['x'][1] - px, py - b['y'][0], b['y'][1] - py])
    R['d'] = dd.min(axis=0); R['side'] = np.argmin(dd, axis=0)
    R['det'] = R.cat_xy_5 == 'detected'
    return R


def prof(R, side, col, fn):
    out = []
    for i in range(len(D_EDGES) - 1):
        s = (R.side == side) & (R.d >= D_EDGES[i]) & (R.d < D_EDGES[i + 1])
        v = R.loc[s, col]
        out.append(dict(d_lo=float(D_EDGES[i]), d_hi=float(D_EDGES[i + 1]), n=int(s.sum()),
                        m=float(fn(v)) if s.sum() >= 15 else float('nan')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rays-root', required=True); ap.add_argument('--angles-root', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'edge_angles.json'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figures'))
    a = ap.parse_args()
    res = {}
    for det in GOLD:
        try:
            R = load(det, a.rays_root, a.angles_root)
        except IndexError:
            print(det, 'missing'); continue
        r = {}
        for k, s in enumerate(SIDES):
            r[s] = dict(eff=prof(R, k, 'det', np.mean),
                        dtx=prof(R, k, 'dt_x', np.nanmedian), dty=prof(R, k, 'dt_y', np.nanmedian),
                        s68x=prof(R, k, 'dt_x', s68), s68y=prof(R, k, 'dt_y', s68),
                        resx=prof(R, k, 'res_x', np.nanmedian), resy=prof(R, k, 'res_y', np.nanmedian))
        inn = R.d >= 30
        r['interior'] = dict(eff=float(R.det[inn].mean()), dtx=float(np.nanmedian(R.dt_x[inn])), dty=float(np.nanmedian(R.dt_y[inn])),
                             s68x=s68(R.dt_x[inn]), s68y=s68(R.dt_y[inn]))
        res[SHORT[det]] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    figures(res, a.figs)


def figures(res, outdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    dets = list(res)
    rows = (('eff', 'efficiency', 100.0), ('dt', 'median tan_fit - tan_ref', 1.0), ('s68', 's68 of tan_fit - tan_ref', 1.0))
    fig, axs = plt.subplots(len(rows), len(dets), figsize=(3.6 * len(dets), 9), squeeze=False)
    for k, d in enumerate(dets):
        for i, (key, lab, sc) in enumerate(rows):
            ax = axs[i, k]
            for s in SIDES:
                view = 'x' if s.startswith('x') else 'y'
                P = res[d][s][key if key == 'eff' else key + view]
                c = [(p['d_lo'] + p['d_hi']) / 2 for p in P]
                ax.plot(c, [sc * p['m'] for p in P], '.-', ms=3, lw=0.8, label=f"{s.replace('_', ' ')}" + ('' if key == 'eff' else f' ({view} view)'))
            ax.axvline(30, color='grey', ls=':', lw=0.8); ax.axvline(0, color='k', lw=0.5)
            ax.set_xscale('symlog', linthresh=10); ax.grid(alpha=0.3); ax.set_ylabel(lab)
            ax.set_xlabel('M3 ray at mid-gap: distance to the active-area side [mm]')
            if i == 0:
                ax.set_title(d, fontsize=10)
            if k == 0:
                ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'edge_turnon_angles.png'), dpi=110)
    fig.savefig(os.path.join(outdir, 'edge_turnon_angles.pdf')); plt.close(fig)


if __name__ == '__main__':
    main()
