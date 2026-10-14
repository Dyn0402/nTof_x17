#!/usr/bin/env python3
"""
make_lnl_plots.py -- the two plots of the Italy-talk LNL slide, on linear axes.

    python ntof_italy_26/make_lnl_plots.py          # -> ntof_italy_26/figures/lnl_{rate_bars,significance}.png

1. X17 per day, produced and with both legs triggered: n_TOF Plan A (MeV neutrons,
   2026 stack) against LNL with the n_TOF stack and with the big plastics (1 µA).
2. Expected significance at the ATOMKI ratio against days of beam at LNL.

Sources (no number typed in, except the n_TOF pair-trigger acceptance):
  * figures/lnl_numbers.json  <- x17_facility_search lnl/out/talk/numbers.json
    (Geant4 reach, FEASIBILITY_SIM §14.5; produced/day at R = 5.8e-6, 1 µA; pair_tag).
  * mpgd26/data/x17_rate_3He.txt  -- the n_TOF X17/day table behind the Athens "Plan A"
    slides (EAR2 flux, nominal 500 atm cell, IPC/capture 2.1e-3, X17/IPC 2.5e-2);
    Plan A = 0.1-10 MeV, apportioned as in mpgd26/make_x17_rate.py.
  * NTOF_PAIR_TAG = 2.3 %: both X17 legs triggered with the as-built stack for the
    n_TOF X17 kinematics (lnl-x17-feasibility note, "vs n_TOF" slide).
The two facilities' X17 normalisations are each "at the claimed strength" for their
nucleus (⁴He at n_TOF, ⁸Be at LNL); they are not the same coupling.
Z(t) = 3·sqrt(t / days_3σ), as lnl/talk/lnl_summary_figs.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIG = HERE / 'figures'
sys.path.insert(0, str(REPO / 'mpgd26'))

NTOF_PAIR_TAG = 0.023
INK, MUTE, GRID = '#23373B', '#5D7176', '#E3E7E8'
SLATE, COPPER, PURPLE = '#4F7C8A', '#CF8A3F', '#8A3F8F'
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 11, 'axes.edgecolor': MUTE,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True})


def ntof_plan_a():
    import make_x17_rate as R
    n = R.numbers() if hasattr(R, 'numbers') else None
    if n is None:
        raise SystemExit('mpgd26/make_x17_rate.py has no numbers()')
    return float(n['mev'])


def rate_bars(N, mev):
    C = N['cases']
    groups = [('n_TOF Plan A\nMeV n, 2026 stack', mev, mev * NTOF_PAIR_TAG, SLATE),
              ('LNL\nn_TOF stack', C['as built|1']['produced_per_day'],
               C['as built|1']['produced_per_day'] * N['pair_tag']['as built'], COPPER),
              ('LNL\nbig plastics', C['big plastics|1']['produced_per_day'],
               C['big plastics|1']['produced_per_day'] * N['pair_tag']['big plastics'], PURPLE)]
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=300)
    w = 0.38
    for i, (lab, prod, trig, c) in enumerate(groups):
        ax.bar(i - w / 2, prod, w, color=c, alpha=0.30, edgecolor=c, linewidth=1.2)
        ax.bar(i + w / 2, trig, w, color=c, edgecolor=c, linewidth=1.2)
        ax.text(i - w / 2, prod + 15, f'{prod:.0f}', ha='center', va='bottom', fontsize=10.5, color=INK)
        ax.text(i + w / 2, trig + 15, f'{trig:.0f}' if trig >= 1.5 else f'{trig:.1f}', ha='center', va='bottom',
                fontsize=12, color=INK, weight='bold')
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=10.5)
    ax.set_ylim(0, 1100)
    ax.set_ylabel('X17 per day')
    ax.grid(axis='x', visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTE, alpha=0.3, edgecolor=MUTE, label='produced'),
                       Patch(facecolor=MUTE, edgecolor=MUTE, label='both legs triggered')],
              loc='upper left', bbox_to_anchor=(0.0, 0.72), frameon=False, fontsize=10)
    ax.set_title('X17 per day: n_TOF vs LNL (1 µA)', loc='left', fontsize=12.5, color=INK, weight='bold')
    fig.tight_layout()
    out = FIG / 'lnl_rate_bars.png'
    fig.savefig(out)
    print('wrote', out, [(g[0].replace('\n', ' '), round(g[1], 1), round(g[2], 2)) for g in groups])


def significance(N):
    C = N['cases']
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=300)
    t = np.linspace(0, 20, 801)
    for hw, c, lab in (('as built', COPPER, 'n_TOF stack'), ('big plastics', PURPLE, 'big plastics')):
        for I, ls, lw in ((1, '-', 2.8), (4, '--', 1.6)):
            d3 = C[f'{hw}|{I}']['days_3sigma']
            ax.plot(t, 3 * np.sqrt(t / d3), ls, color=c, lw=lw, label=f'{lab}, {I} µA')
        d3 = C[f'{hw}|1']['days_3sigma']
        ax.plot([d3], [3], 'o', color=c, ms=7, zorder=5)
        ax.annotate(f'{d3:.0f} d' if d3 >= 2 else f'{d3:.1f} d', (d3, 3), xytext=(6, -16), textcoords='offset points',
                    color=c, fontsize=12, weight='bold')
    for z in (3, 5):
        ax.axhline(z, color=MUTE, ls=':', lw=1)
        ax.text(7.0, z + 0.1, f'{z}σ', ha='center', va='bottom', color=MUTE, fontsize=10.5)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21, 5))
    ax.set_ylim(0, 10)
    ax.set_xlabel('days of beam')
    ax.set_ylabel('expected significance at the ATOMKI ratio')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.93), frameon=True, framealpha=0.95, edgecolor='none',
              fontsize=9.5)
    ax.set_title('Time to the ATOMKI sensitivity at LNL', loc='left', fontsize=12.5, color=INK, weight='bold')
    fig.tight_layout()
    out = FIG / 'lnl_significance.png'
    fig.savefig(out)
    print('wrote', out)


def main():
    N = json.loads((FIG / 'lnl_numbers.json').read_text(encoding='utf8'))
    mev = ntof_plan_a()
    rate_bars(N, mev)
    significance(N)


if __name__ == '__main__':
    main()
