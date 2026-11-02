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
LAB = {'x_lo': 'x = 0 side', 'x_hi': 'x = 398.6 side', 'y_lo': 'y low side (passivated)', 'y_hi': 'y high side (passivated)'}
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
    fig, axs = plt.subplots(1, 3, figsize=(ps.FULL[0], 2.5))
    rows = []
    for s in LAB:
        v = s[0]
        st = dict(color=COL[s], ls=LS[s], marker=MK[s], ms=3.5, lw=1.1, mfc=COL[s] if FILL[s] else 'white', mec=COL[s])
        c, m = mid(A[s]['eff']); k = (c >= 0) & (c <= a.dmax)
        axs[0].plot(c[k], 100 * m[k], label=LAB[s], **st)
        rows += [dict(panel='a_eff_pct', side=s, d_mm=x, value=100 * y) for x, y in zip(c, m)]
        c, m = mid(A[s]['dt' + v]); k = (c >= 0) & (c <= a.dmax)
        axs[1].plot(c[k], m[k], **st)
        rows += [dict(panel=f'b_dtan_{v}', side=s, d_mm=x, value=y) for x, y in zip(c, m)]
        P = S['prof_T_side'][s]
        c = np.array([(p['d_lo'] + p['d_hi']) / 2 for p in P]); m = np.array([p['m'] for p in P], float)
        e = np.array([p['e'] for p in P], float); k = (c >= 0) & (c <= a.dmax)
        axs[2].errorbar(c[k], m[k] - S['T_interior'], yerr=e[k], **st)
        rows += [dict(panel='c_dT_ns', side=s, d_mm=x, value=y - S['T_interior'], err=ee) for x, y, ee in zip(c, m, e)]
    axs[0].set_ylabel('detection efficiency [%]'); axs[0].set_ylim(0, 100)
    axs[1].set_ylabel(r'median $\tan\theta_{fit}-\tan\theta_{M3}$'); axs[1].axhline(0, color='k', lw=0.5)
    axs[2].set_ylabel(r'$T - T_{interior}$ [ns]'); axs[2].axhline(0, color='k', lw=0.5)
    for ax, t in zip(axs, ('(a) turn-on', '(b) angle bias (perp. view)', '(c) column duration')):
        ax.axvspan(0, 30, color=ps.GRID, alpha=0.8, lw=0, zorder=0)
        ax.set_xlim(0, a.dmax); ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_title(t)
    axs[1].set_xlabel('distance of the M3 ray to the side of the active area [mm]')
    axs[0].text(15, 4, 'edge zone\n(< 30 mm)', ha='center', va='bottom', fontsize=7, color=ps.MUTED)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.4)
    fig.tight_layout(w_pad=1.0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)                                         # the paper copy (pdf only)
    ps.save(fig, os.path.join(HERE, 'figures', f'edge_turnon_{a.det}'), data=pd.DataFrame(rows))   # pdf + png + csv
    print('wrote', a.out)


if __name__ == '__main__':
    main()
