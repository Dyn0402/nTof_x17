#!/usr/bin/env python3
"""make_edge_figure.py [--det det3] [--out papers/paper1/figures/edge_turnon.pdf] -- the Paper I §5 edge figure from
edge_angles.py and edge_study.py results: efficiency turn-on, the angle bias of the view the side is perpendicular to, and
the column duration, against the distance of the track to each side of the active area.  Pass 1, waveform products."""
import argparse
import json
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
LAB = {'x_lo': r'$x = 0$ side', 'x_hi': r'$x = 398.6$ mm side', 'y_lo': r'low-$y$ side (passivated)',
       'y_hi': r'high-$y$ side (passivated)'}
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402
# Sides are not chambers: x sides solid (X view style), y sides dashed; low side dark, high side light.
COL = {'x_lo': ps.INK, 'x_hi': ps.ACCENT, 'y_lo': ps.INK, 'y_hi': ps.ACCENT}
LS = {'x_lo': '-', 'x_hi': '-', 'y_lo': '--', 'y_hi': '--'}
MK = {'x_lo': 'o', 'x_hi': 'o', 'y_lo': 's', 'y_hi': 's'}
FILL = {'x_lo': True, 'x_hi': True, 'y_lo': False, 'y_hi': False}


def mid(P):
    return np.array([(p['d_lo'] + p['d_hi']) / 2 for p in P]), np.array([p['m'] for p in P], float)


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='det3')
    ap.add_argument('--out', default=os.path.join(REPO, 'papers', 'paper1', 'figures', 'edge_turnon.pdf'))
    ap.add_argument('--dmax', type=float, default=60.0)
    a = ap.parse_args()
    ps.use()
    A = json.load(open(os.path.join(HERE, 'results', 'edge_angles.json')))[a.det]
    S = json.load(open(os.path.join(HERE, 'results', 'edge_study.json')))[a.det]
    fig, axs = ps.fig_for('full', height=2.45, nrows=1, ncols=3, sharex=True)
    rows = []
    for s in LAB:
        v = s[0]
        st = dict(color=COL[s], ls=LS[s], marker=MK[s], ms=3.2, lw=1.0, mfc=COL[s] if FILL[s] else 'white', mec=COL[s])
        # (a) efficiency, binomial error from the bin's ray count
        E = A[s]['eff']; c, m = mid(E); n = np.array([p['n'] for p in E], float); k = (c >= 0) & (c <= a.dmax)
        em = np.sqrt(np.clip(m * (1 - m), 0, None) / np.maximum(n, 1))
        axs[0].errorbar(c[k], 100 * m[k], yerr=100 * em[k], label=LAB[s], **st)
        rows += [dict(panel='a_eff_pct', side=s, d_mm=x, value=100 * y, err=100 * ee) for x, y, ee in zip(c, m, em)]
        # (b) median tan bias of the perpendicular view; error 1.2533 s68 / sqrt(n detected) (median of a Gaussian)
        D = A[s]['dt' + v]; c, m = mid(D); k = (c >= 0) & (c <= a.dmax)
        s68 = np.array([p['m'] for p in A[s]['s68' + v]], float)
        ed = 1.2533 * s68 / np.sqrt(np.maximum(n * np.nan_to_num(np.array([p['m'] for p in E], float)), 1))
        axs[1].errorbar(c[k], m[k], yerr=ed[k], **st)
        rows += [dict(panel=f'b_dtan_{v}', side=s, d_mm=x, value=y, err=ee) for x, y, ee in zip(c, m, ed)]
        P = S['prof_T_side'][s]
        c = np.array([(p['d_lo'] + p['d_hi']) / 2 for p in P]); m = np.array([p['m'] for p in P], float)
        e = np.array([p['e'] for p in P], float); k = (c >= 0) & (c <= a.dmax)
        axs[2].errorbar(c[k], m[k] - S['T_interior'], yerr=e[k], **st)
        rows += [dict(panel='c_dT_ns', side=s, d_mm=x, value=y - S['T_interior'], err=ee) for x, y, ee in zip(c, m, e)]
    axs[0].set_ylabel('detection efficiency [%]'); axs[0].set_ylim(0, 102)
    axs[1].set_ylabel(r'$\tan\theta_{\mathrm{fit}} - \tan\theta_{\mathrm{M3}}$')
    axs[2].set_ylabel(r'$T - T_{\mathrm{interior}}$ [ns]')
    for ax in axs[1:]:
        ax.axhline(0, color=ps.MUTED, lw=0.6, zorder=0)
    for ax in axs:
        ax.axvspan(0, 30, color=ps.GRID, alpha=0.8, lw=0, zorder=0)
        ax.set_xlim(0, a.dmax); ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel('distance to the side [mm]')
    ps.panel_labels(axs)
    h, l = axs[0].get_legend_handles_labels()
    h = [hh[0] if isinstance(hh, tuple) else hh for hh in h]
    ps.legend_above(fig, h, l, ncol=4, handlelength=2.4)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300)                                # the paper copy (pdf only)
    ps.save(fig, os.path.join(HERE, 'figures', f'edge_turnon_{a.det}'), data=pd.DataFrame(rows))   # pdf + png + csv
    print('wrote', a.out)


if __name__ == '__main__':
    main()
