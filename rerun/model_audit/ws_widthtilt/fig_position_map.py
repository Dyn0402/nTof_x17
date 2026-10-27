#!/usr/bin/env python3
"""fig_position_map.py — where in the chamber, at what angle: median angle-scale r = (w − w0)/(v·kw·tan_ref) in cells of
(track position at mid-gap along the view's axis, tan_ref), one panel per chamber/view.  The telescope acceptance
fills a diagonal band on chambers far from the telescope's pivot (det6: position and angle are tied), so a scale that
depends on position looks like one that depends on angle unless the two are separated.
Writes ws_widthtilt/fig_position_map.png."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT

DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def main():
    fig, axs = plt.subplots(2, 5, figsize=(20, 7.5), sharey=True)
    tb = np.array([-0.45, -0.30, -0.20, -0.12, -0.06, 0.06, 0.12, 0.20, 0.30, 0.45])
    pb = np.linspace(0, 400, 11)
    for i, det in enumerate(DETS):
        b, T = load(det)
        for k, p in enumerate('xy'):
            df = T[p]; pm = df.p0 + 15.0 * df.t
            r = df.r / b['kw'][p]
            M = np.full((len(tb) - 1, len(pb) - 1), np.nan)
            for a in range(len(tb) - 1):
                if tb[a] == -0.06:
                    continue
                for c in range(len(pb) - 1):
                    m = (df.t >= tb[a]) & (df.t < tb[a + 1]) & (pm >= pb[c]) & (pm < pb[c + 1])
                    if m.sum() >= 15:
                        M[a, c] = np.median(r[m])
            ax = axs[k, i]
            im = ax.pcolormesh(pb, tb, M, cmap='RdBu_r', norm=TwoSlopeNorm(1.0, 0.8, 1.2), shading='flat')
            ax.set_title(f'{det} {p.upper()}'); ax.set_xlabel(f'{p} at mid-gap [mm]')
            if i == 0:
                ax.set_ylabel('tan θ (M3)')
    fig.colorbar(im, ax=axs, shrink=0.8, label='measured / true slope (calibrated scale = 1)')
    fig.suptitle('Angle scale by position and angle (cells with ≥ 15 tracks; head-on band blank)', fontsize=13)
    o = f'{ROOT}/ws_widthtilt/fig_position_map.png'; fig.savefig(o, dpi=110, bbox_inches='tight'); print('wrote', o)


if __name__ == '__main__':
    main()
