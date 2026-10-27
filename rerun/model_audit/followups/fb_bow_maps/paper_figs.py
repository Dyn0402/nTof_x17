#!/usr/bin/env python3
"""paper_figs.py -- the two figures of the detector-paper bow section (papers/paper1/sections/08b_bow.tex):
figures/bow_profiles.pdf  mesh height across the chamber from positions, X and Y strips, five chambers
figures/bow_closure.pdf   measured drift-tilt gradient against the one the position-route curvature predicts
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import RES, FIG, DETS

COL = {'det2': '#1f5fa8', 'det3': '#2f8a5b', 'det4': '#c63d4f', 'det6': '#7a52b8', 'det7': '#c99318'}


def J(n):
    return json.load(open(os.path.join(RES, n)))


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, p in zip(axs, 'xy'):
        for d in DETS:
            pr = J(f'profiles_{d}.json')[p]
            pos, h, he = np.array(pr['pos']), np.array(pr['h']), np.array(pr['h_err'])
            c = np.polyval(pr['h_par'], 0.0)
            ok = np.abs(h - c) < 4
            ax.errorbar(pos[ok], (h - c)[ok], yerr=he[ok], fmt='o-', ms=3, lw=1.2, capsize=1.5, color=COL[d], label=d)
        ax.axhline(0, color='grey', lw=0.6, ls=':')
        ax.set_xlabel(f'track {p} at mid-gap [mm]'); ax.set_title(f'{p.upper()} strips', fontsize=10)
    axs[0].set_ylabel(r'mesh height $h - h_{\rm centre}$ [mm]')
    axs[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'bow_profiles.{ext}'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    xs = np.linspace(0, 10, 10)
    ax.plot(xs, xs, 'k--', lw=1, label=r'$r_f = 1$ (cathode follows)')
    ax.plot(xs, 0.5 * xs, 'k:', lw=1, label=r'$r_f = 1/2$ (flat cathode)')
    for d in DETS:
        P = J(f'profiles_{d}.json')
        for p, mk in (('x', 'o'), ('y', 'D')):
            pred = P[p]['b_med'] * P[p]['h_curv'] * 1e3          # um/ns per m
            meas = P[p]['w0_slope'] * 1e3
            ax.plot(pred, meas, mk, color=COL[d], ms=6, label=(d + (' (circle X, diamond Y)' if d == 'det2' else '')) if p == 'x' else None)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_xlabel(r'$v\,h^{\prime\prime}$ from positions [µm/ns per m]')
    ax.set_ylabel(r'measured $dw_0/dp$ [µm/ns per m]')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'bow_closure.{ext}'), dpi=200)
    print('wrote', FIG)


if __name__ == '__main__':
    main()
