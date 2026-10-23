#!/usr/bin/env python3
"""xasym_pointing.py -- the X-asymmetry correction as a +- variation on chamber A's capsule pointing
(XASYM_CONTINGENCY.md, n_TOF item 2).

Chamber A is det3's June-15 board, the one with a measured X-asymmetry map (wft/xasym_maps.json).  On the
same stage-3 tracks, the pointing-band crossing of arm A is fitted with production x and with
``wft.xasym.correct_x(x_p0, y_p0, 'det3 June-15', scale)``; the paired shift of the crossing is the
systematic.  The n_TOF session scale is not measured (it needs the raw-ADC estimator on the beam waveforms),
so the bench's det3 range is used: 0.44, 1.0, 1.57.  The crossing is linear in the correction, so the
shift at any other scale is proportional.

The crossing (source_imaging.crossing) uses the scintillator-confirmed gated tracks (wall_A & plastic_A), a
30-130 mm lever window from the perpendicular foot, the robust line, and a bootstrap error.  It is in A's own
u (x_local); A's u maps onto global X with |du/dX| = 1, so the shift in mm is the capsule-X shift from A.
Chamber C carries no map: its +-0.1 mm flat systematic moves the A+C mean by +-0.05 mm and the half
difference by the same.

    xasym_pointing.py [tracks_*.parquet ...]   -> <out>/xasym_pointing.json (default: run_145 on this disk)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from wft import xasym                                              # noqa: E402
from ntof_tracking import run145_target_imaging as TI              # noqa: E402
from sept26_prelim_analysis import source_imaging as SI            # noqa: E402
from sept26_prelim_analysis.build_tracks import y_up               # noqa: E402

BUILD = 'det3 June-15'
SCALES = (0.44, 1.0, 1.57)
DEFAULT = '/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_run_145_stat090_*.parquet'
OUT = '/media/dylan/data/x17/sept26_prelim/xasym_pointing.json'


def main():
    files = sys.argv[1:] or sorted(glob.glob(DEFAULT))
    T = pd.concat([y_up(pd.read_parquet(f), f) for f in files], ignore_index=True)
    a = T[(T.arm == 'A') & T.gated & T.wall_A.astype(bool) & T.plastic_A.astype(bool)].copy()
    win = dict(foot=TI.PINWHEEL['A'], lo=30.0, hi=130.0)
    tx = a.tanx.to_numpy(float)
    base = SI.crossing(a.x_local.to_numpy(float), tx, win)
    dx = xasym.dx_mm(a.x_p0, a.y_p0, BUILD) * -xasym.maps()['K']      # correction added to x_p0 at scale 1
    out = dict(files=[os.path.basename(f) for f in files], build=BUILD, n=base['n'], baseline=base,
               corr_um=dict(mean=float(dx.mean() * 1e3), rms=float(dx.std() * 1e3),
                            max_abs=float(np.abs(dx).max() * 1e3)), scales={})
    print(f'arm A: {base["n"]} confirmed band tracks, crossing u0 = {base["x0"]:+.3f} +- {base["err"]:.3f} mm')
    print(f'  correction on these tracks (scale 1): mean {dx.mean()*1e3:+.1f} um, rms {dx.std()*1e3:.1f} um, '
          f'max {np.abs(dx).max()*1e3:.1f} um')
    for s in SCALES:
        xc = xasym.correct_x(a.x_p0, a.y_p0, BUILD, scale=s)
        # the table's x_local is the same mirror of x_p0; shift it by the correction rather than rebuild it
        xl = a.x_local.to_numpy(float) + TI.IN_PLANE_SIGN * (xc - a.x_p0.to_numpy(float))
        c = SI.crossing(xl, tx, win)
        # the board-mean part is alignment: remove it (re-align), as the plan says
        xl_r = xl - np.mean(xl - a.x_local.to_numpy(float))
        cr = SI.crossing(xl_r, tx, win)
        out['scales'][f'{s:g}'] = dict(shift_mm=c['x0'] - base['x0'], shift_realigned_mm=cr['x0'] - base['x0'],
                                       slope_ratio=c['slope'] / base['slope'])
        print(f'  scale {s:4.2f}: crossing shift {c["x0"]-base["x0"]:+.4f} mm, after removing the board mean '
              f'{cr["x0"]-base["x0"]:+.4f} mm, slope x{c["slope"]/base["slope"]:.5f}')
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print('->', OUT)


if __name__ == '__main__':
    main()
