#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_extended.py — the run_58 ev1054 road-extension figure: original vs
road-extended per-plane fits, against the physically allowed drift window.

Shows, per Det-A plane: all clean hits, the ORIGINAL RANSAC fit (thick,
translucent), the ROAD-EXTENDED fit (dashed) and the hits it newly picked up
(squares), over a grey band marking [t0, t0+T_max] — the only times at which
gap ionisation can arrive. Lines whose span exceeds that band cannot be a
single track.

Usage:  .venv/bin/python ntof_july_analysis/detA_doubletrack/fig_extended.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import extend as X, dtrack_lib as D, scan as SC  # noqa: E402
from ntof_tracking.reco import io, noise  # noqa: E402

RUN, SUBRUN, EVID = 'run_58', 'sngPS_dr300_r580_036', 1054
V, T0 = 26.0, 180.0          # Garfield pure Ar/iso 90/10 @ E=100 V/cm
TMAX = 30000.0 / V
COL = ['crimson', 'royalblue']


def main():
    hits = SC.load_detA_hits(RUN, SUBRUN)
    g = noise.flag_noise(hits[hits.eventId == EVID])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    for ax, pl in zip(axes, ('x', 'y')):
        gp = g[(g.plane == pl) & g.clean]
        gn = g[(g.plane == pl) & ~g.clean]
        pos = gp.pos_mm.to_numpy(float)
        tim = gp.time.to_numpy(float)
        amp = gp.amplitude.to_numpy(float)
        orig = D.plane_lines(gp)
        ext = [X.road_extend(l, pos, tim, amp) for l in orig]
        ax.axvspan(T0, T0 + TMAX, color='0.85', alpha=.5, zorder=0,
                   label=f'physical drift window\n(t0={T0:.0f}, '
                         f'T_max={TMAX:.0f} ns @ {V:.0f} um/ns)')
        ax.axvline(T0 + TMAX, color='k', ls=':', lw=1.5, zorder=1)
        ax.scatter(gn.time, gn.pos_mm, s=6, c='0.85', marker='.', zorder=1)
        ax.scatter(tim, pos, s=20, c='0.35', zorder=2, label='clean hits')
        for i, (a, b) in enumerate(zip(orig, ext)):
            c = COL[i % 2]
            new = np.setdiff1d(b['idx'], a['idx'])
            ax.scatter(tim[a['idx']], pos[a['idx']], s=40, facecolors='none',
                       edgecolors=c, lw=1.8, zorder=4)
            if len(new):
                ax.scatter(tim[new], pos[new], s=70, marker='s',
                           facecolors='none', edgecolors=c, lw=1.8, zorder=4,
                           label=f'{pl}{i} newly picked up')
            ta = np.array([a['t0_ns'], a['t1_ns']])
            tb = np.array([b['t0_ns'], b['t1_ns']])
            ax.plot(ta, a['slope_mm_ns'] * ta + a['intercept_mm'],
                    c=c, lw=3, alpha=.45, zorder=3)
            ax.plot(tb, b['slope_mm_ns'] * tb + b['intercept_mm'],
                    c=c, lw=1.6, ls='--', zorder=3)
            ax.text(b['t1_ns'] + 30,
                    b['slope_mm_ns'] * b['t1_ns'] + b['intercept_mm'],
                    f"{pl}{i}: {a['tspan_ns']:.0f}->{b['tspan_ns']:.0f}ns\n"
                    f"({b['tspan_ns']/TMAX:.2f}xT_max)",
                    fontsize=7, color=c, va='center')
        ax.set_title(f'Det A plane {pl}')
        ax.set_xlabel('drift time [ns]')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='upper right')
    axes[0].set_ylabel('strip position [mm, centred]')
    fig.suptitle(
        'run_58 ev1054 @ drift 300 V — thick=original fit, dashed=road-extended,'
        ' squares=picked-up hits.\nGrey band = physically allowed drift window; '
        'only y0 (1199 ns) is a genuine full-gap crossing.', fontsize=11)
    fig.tight_layout()
    out = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'detA_doubletrack',
                       'extend', 'ev1054_extended.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print('->', out)


if __name__ == '__main__':
    main()
