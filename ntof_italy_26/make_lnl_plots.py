#!/usr/bin/env python3
"""
make_lnl_plots.py -- the two plots of the Italy-talk LNL slide, on linear axes.

    python ntof_italy_26/make_lnl_plots.py          # -> ntof_italy_26/figures/lnl_{rate_bars,significance}.png

1. X17 per day, produced and with both legs accepted: n_TOF Plan A (MeV neutrons,
   big plastics) against LNL with the n_TOF stack and with the big plastics (1 µA).
2. Expected significance at the ATOMKI ratio against days of beam at LNL.

Sources (no number typed in, except the n_TOF Micromegas pair acceptance):
  * figures/lnl_numbers.json  <- x17_facility_search lnl/out/talk/numbers.json
    (Geant4 reach, FEASIBILITY_SIM §14.5; produced/day at R = 5.8e-6, 1 µA; pair_tag).
  * mpgd26/data/x17_rate_3He.txt  -- the n_TOF X17/day table behind the Athens "Plan A"
    slides (EAR2 flux, nominal 500 atm cell, IPC/capture 2.1e-3, X17/IPC 2.5e-2).
    Plan A here = the file's own 0.2-2 MeV band (its second, commented-out table),
    which arrives 1.0-3.2 µs after the γ flash over 19.5 m: nothing before 1 µs is
    taken as ever recordable, so the 2-10 MeV neutrons are cut.  Interpolating the
    decade bins instead under-reads this band (18 vs 24.9/day: the spectrum peaks
    inside it), which is why the tabulated band is used.  Extending down to 0.1 MeV
    (4.5 µs) would add roughly 3-4/day, from the decade bins.
  * NTOF_MM_ACC = 27.6 %: both X17 legs inside the Micromegas active area (two arms),
    x17_facility_search/trigger_scint sim/accept/summary.json, X17 T0 "MM only".
    No trigger factor: at n_TOF the read-out is started by the γ flash, so every
    bunch is recorded and the plastics are not needed to trigger.  Caveat: that sim
    uses the at-rest (cold-neutron) X17 kinematics; at 0.2-2 MeV the X17 edge moves
    only 110 -> 99 deg.
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

NTOF_MM_ACC = 0.2756
INK, MUTE, GRID = '#23373B', '#5D7176', '#E3E7E8'
SLATE, COPPER, PURPLE = '#4F7C8A', '#CF8A3F', '#8A3F8F'
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 11, 'axes.edgecolor': MUTE,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True})


def ntof_plan_a(lo_ev=2.0e5, hi_ev=2.0e6):
    """X17/day produced in the rate file's tabulated 0.2-2 MeV band (its last column).

    That band sits in the file's second table, commented out with '#' (which is why
    make_x17_rate.load() skips it), so it is found here by its edges.
    """
    for line in (REPO / 'mpgd26' / 'data' / 'x17_rate_3He.txt').read_text(encoding='utf8').splitlines():
        f = line.lstrip('#').split()
        try:
            elo, ehi = float(f[0]), float(f[1])
        except (IndexError, ValueError):
            continue
        if abs(elo / lo_ev - 1) < 1e-3 and abs(ehi / hi_ev - 1) < 1e-3:
            return float(f[-1])
    raise SystemExit(f'no {lo_ev:g}-{hi_ev:g} eV row in x17_rate_3He.txt')


def rate_bars(N, mev):
    C = N['cases']
    groups = [('n_TOF Plan A\n0.2–2 MeV n,\nbig plastics', mev, mev * NTOF_MM_ACC, SLATE),
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
                       Patch(facecolor=MUTE, edgecolor=MUTE, label='both legs accepted\n(n_TOF: in the MMs,\nLNL: triggered)')],
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
