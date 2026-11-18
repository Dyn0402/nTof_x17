"""Poster-size redraws of two Paper I figures, from their own outputs (no data access needed).

Paper figures are drawn for a 180 mm column with 7-9 pt text; pasted on an A0 poster they shrink or stay tiny.
These are drawn at their printed size on the poster with 20-28 pt text, as SVG (vector in the PDF).

    assets/fig_air.svg     why 1 deg is enough: scattering in the air before the gas vs lepton energy
                           (papers/paper1_v2/figs/out/pointing.json, block 'air_only')
    assets/fig_angres.svg  angular resolution s68 vs reference angle, chamber A, X and Y
                           (papers/paper1_v2/figs/out/angle_resolution_v2.csv, panel c; .json for the totals)

    ../.venv/bin/python make_poster_figs.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'assets')
P1 = os.path.join(REPO, 'papers/paper1_v2/figs/out')

INK, MUTED, GRID = '#16202e', '#5b6675', '#e3e7ed'
NAVY, ACCENT, SKY = '#0b3d91', '#d9631e', '#4a90d9'
MM = 1 / 25.4

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 22, 'axes.labelsize': 24, 'xtick.labelsize': 21,
    'ytick.labelsize': 21, 'axes.edgecolor': '#8a94a3', 'axes.linewidth': 1.2, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 6, 'ytick.major.size': 6,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0, 'figure.facecolor': 'none',
    'axes.facecolor': 'none', 'savefig.transparent': True,
})


def fig_air():
    a = json.load(open(os.path.join(P1, 'pointing.json')))['air_only']
    T = np.array([4, 8, 12, 16.])
    air = np.array([a[f'{int(t)}MeV']['theta_air_deg'] for t in T])
    meas = np.array([a[f'{int(t)}MeV']['local_deg'] for t in T])
    # where the air alone equals the chamber's 1 deg (log-log interpolation between the tabulated energies)
    lt, la = np.log(T), np.log(air)
    t1 = float(np.exp(np.interp(0.0, la[::-1], lt[::-1])))
    fig, ax = plt.subplots(figsize=(330 * MM, 175 * MM), layout='constrained')
    ax.axhspan(0, 1.0, color=NAVY, alpha=0.07, lw=0)
    ax.axhline(1.0, color=NAVY, lw=3, ls=(0, (6, 3)))
    ax.plot(T, air, color=ACCENT, lw=4, marker='o', ms=12, mec='white', mew=2)
    ax.plot(T, meas, color=INK, lw=2.5, marker='s', ms=10, mec='white', mew=2, alpha=0.75)
    ax.text(4.25, 0.80, 'the chamber alone: 1°', color=NAVY, fontsize=23, fontweight='bold', va='top')
    ax.text(5.0, 3.35, 'air in front of the gas', color=ACCENT, fontsize=23, fontweight='bold')
    ax.text(10.3, 2.05, 'what is measured\n(air ⊕ chamber)', color=INK, fontsize=21, alpha=0.85)
    ax.axvline(t1, color=MUTED, lw=1.5, ls=':')
    ax.annotate(f'≈ {t1:.0f} MeV', (t1, 3.4), xytext=(t1 + 0.4, 3.4), fontsize=21, color=MUTED, va='center')
    ax.set_xlim(3.5, 16.5); ax.set_ylim(0, 3.8)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_xlabel('lepton kinetic energy  [MeV]')
    ax.set_ylabel('error on the direction  [°]')
    fig.savefig(os.path.join(OUT, 'fig_air.svg'))
    plt.close(fig)
    return t1


def fig_angres():
    df = pd.read_csv(os.path.join(P1, 'angle_resolution_v2.csv'))
    tot = json.load(open(os.path.join(P1, 'angle_resolution_v2.json')))['A']
    c = df[(df.panel == 'c') & (df.chamber == 'A')]
    fig, ax = plt.subplots(figsize=(360 * MM, 165 * MM), layout='constrained')
    for v, col, mk, dx in (('X', NAVY, 'o', -0.25), ('Y', SKY, 's', 0.25)):
        d = c[c.view == v]
        ax.errorbar(d.x + dx, d.s68, yerr=d.s68_err, fmt=mk, ms=12, color=col, mec='white', mew=1.8,
                    elinewidth=2.2, capsize=0, label=f'{v} view')
        s = tot[v.lower()]['s68']
        ax.text(21.5, s + (0.09 if v == 'X' else -0.13), f'{v}  {s:.2f}°', color=col, fontsize=23,
                fontweight='bold', va='center')
    ax.set_xlim(-21, 27); ax.set_ylim(0, 2.0)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel('reference track angle  [°]')
    ax.set_ylabel('angular resolution  s$_{68}$  [°]')
    ax.legend(loc='upper left', frameon=False, ncol=2, fontsize=22, handletextpad=0.3, columnspacing=1.2)
    fig.savefig(os.path.join(OUT, 'fig_angres.svg'))
    plt.close(fig)
    return {v: tot[v]['s68'] for v in ('x', 'y')}, int(tot['x']['n'])


if __name__ == '__main__':
    print('air = 1 deg at', round(fig_air(), 1), 'MeV')
    print('s68 (chamber A), n tracks:', fig_angres())
