#!/usr/bin/env python3
"""capsule_point.py -- the beam-side truth for n_TOF reconstruction choices: do the tracks point back to the capsule?

    capsule_point.py --reco R.parquet --arm A --run-config run_config.json [--label L] [--out summary.json]

A correctly fitted electron from the 3He capsule passes the capsule axis (vertical, +y = beam, at x ~ -9.5 mm,
z ~ -3.7 mm: the focus/imaging fits, CLAUDE.md "Coordinates") within the gas radius (10 mm) plus resolution;
a mis-fitted angle misses it by a lot.  So the share of tracks whose line passes within R of the axis, and the shape
of that distance, measure angle PURITY on beam data, where there is no reference line.  Background (wall
electrons, through-goers, flash tail) adds a flat floor that is the same for every choice compared on the same events,
so differences between arms are what count, not the absolute fraction.

Geometry as ac_line.py (build_tracks.local_and_global, k = 1: the bundle's v is the working v).  The angle scale is the
bundle's, so the DCA scale is only as good as v -- compare arms on the same bundle v.  Positions are v-independent.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

AXIS_X, AXIS_Z = -9.5, -3.7          # mm; capsule axis (focus / imaging fits, 10-13/14)
R_CUTS = (10.0, 15.0, 25.0)
TAN_BINS = (0, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0)


def axis_dca(df):
    """Distance [mm] of each track's global line to the vertical capsule axis, and the y of closest approach."""
    P = df[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    D = df[['d_x', 'd_y', 'd_z']].to_numpy(float)
    # in the xz plane the axis is a point; the line's xz projection passes it at distance |cross| / |d_xz|
    dx, dz = P[:, 0] - AXIS_X, P[:, 2] - AXIS_Z
    ux, uz = D[:, 0], D[:, 2]
    n = np.hypot(ux, uz)
    with np.errstate(invalid='ignore', divide='ignore'):
        dca = np.abs(dx * uz - dz * ux) / n
        t = -(dx * ux + dz * uz) / n ** 2
    y_at = P[:, 1] + t * D[:, 1]
    return dca, y_at


def score(reco, arm, run_config, label=''):
    from ac_line import tracks
    from ntof_tracking.reco import geometry as G
    trs = G.detector_transforms(json.load(open(run_config)))
    T = tracks(reco, arm, trs[f'mx17_{arm}'])
    T = T[T.gated].copy()
    T['dca'], T['y_at'] = axis_dca(T)
    tmax = np.maximum(T.tanx.abs(), T.tany.abs())
    out = dict(label=label, arm=arm, n_tracks=int(len(T)), n_events=int(T.event_id.nunique()),
               dca_med=float(np.nanmedian(T.dca)))
    for r in R_CUTS:
        out[f'n_point_{int(r)}'] = int((T.dca < r).sum())
        out[f'frac_point_{int(r)}'] = float((T.dca < r).mean())
    out['bins'] = [dict(tan_lo=lo, tan_hi=hi, n=int(m.sum()),
                        frac_point_15=float((T.dca[m] < 15).mean()) if m.any() else np.nan,
                        n_point_15=int((T.dca[m] < 15).sum()))
                   for lo, hi in zip(TAN_BINS[:-1], TAN_BINS[1:]) for m in [(tmax >= lo) & (tmax < hi)]]
    return out, T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reco', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--run-config', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    s, _ = score(a.reco, a.arm, a.run_config, a.label)
    print(json.dumps({k: v for k, v in s.items() if k != 'bins'}))
    print(pd.DataFrame(s['bins']).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    if a.out:
        json.dump(s, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
