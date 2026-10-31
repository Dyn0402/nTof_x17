#!/usr/bin/env python3
"""make_edge_figure.py [--det det3] [--out papers/paper1/figures/edge_turnon.pdf] -- the Paper I §5 edge figure from
edge_angles.py and edge_study.py results: efficiency turn-on, the angle bias of the view the side is perpendicular to, and
the column duration, against the distance of the track to each side of the active area.  Pass 1, waveform products."""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
LAB = {'x_lo': 'x = 0 side', 'x_hi': 'x = 398.6 side', 'y_lo': 'y low side (passivated)', 'y_hi': 'y high side (passivated)'}
COL = {'x_lo': '#1f77b4', 'x_hi': '#2ca02c', 'y_lo': '#ff7f0e', 'y_hi': '#d62728'}


def mid(P):
    return np.array([(p['d_lo'] + p['d_hi']) / 2 for p in P]), np.array([p['m'] for p in P], float)


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='det3')
    ap.add_argument('--out', default=os.path.join(REPO, 'papers', 'paper1', 'figures', 'edge_turnon.pdf'))
    a = ap.parse_args()
    A = json.load(open(os.path.join(HERE, 'results', 'edge_angles.json')))[a.det]
    S = json.load(open(os.path.join(HERE, 'results', 'edge_study.json')))[a.det]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for s in LAB:
        v = s[0]
        c, m = mid(A[s]['eff']); axs[0].plot(c, 100 * m, '.-', color=COL[s], label=LAB[s])
        c, m = mid(A[s]['dt' + v]); axs[1].plot(c, m, '.-', color=COL[s], label=f'{LAB[s]}: {v} view')
        P = S['prof_T_side'][s]
        c = np.array([(p['d_lo'] + p['d_hi']) / 2 for p in P]); m = np.array([p['m'] for p in P], float)
        e = np.array([p['e'] for p in P], float)
        axs[2].errorbar(c, m - S['T_interior'], yerr=e, fmt='.-', color=COL[s], ms=4, lw=0.8)
    axs[0].set_ylabel('detection efficiency [%]'); axs[0].set_ylim(0, 100)
    axs[1].set_ylabel(r'median $\tan\theta_{\rm fit}-\tan\theta_{\rm M3}$'); axs[1].axhline(0, color='k', lw=0.5)
    axs[2].set_ylabel(r'column duration $T - T_{\rm interior}$ [ns]'); axs[2].axhline(0, color='k', lw=0.5)
    axs[2].set_ylim(-500, 1250)
    for ax, t in zip(axs, ('(a) turn-on (M3 ray at mid-gap)', '(b) angle bias of the perpendicular view',
                           '(c) column duration (every trigger)')):
        ax.axvline(30, color='grey', ls=':', lw=0.9); ax.axvline(0, color='k', lw=0.4)
        ax.set_xscale('symlog', linthresh=10); ax.set_xlim(-10, 200); ax.grid(alpha=0.3)
        ax.set_xlabel('distance to the side of the active area [mm]'); ax.set_title(t, fontsize=9)
    axs[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(f'{a.det}, golden row (pass 1): the edge zone', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out); fig.savefig(os.path.join(HERE, 'figures', f'edge_turnon_{a.det}.png'), dpi=120)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
