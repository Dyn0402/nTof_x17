#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_alignment.py -- QA figures for the DREAM <-> n_TOF per-event alignment.

Four panels, telling the story in order:
  1. dt vs t_since_flash BEFORE any correction -- the diagonal band that shows the
     smear is a clock RATE difference, not jitter.
  2. the corrected residual over +-3 us -- main peak plus the +330 ns satellite.
  3. the main peak zoomed, with the accidental level it sits on.
  4. the same peak per arm -- all four agree, so it is not one sector's cabling.

Usage: python plot_alignment.py [run] [subrun] [ntof_run] [n_bunches]
Figures land in <analysis>/ntof_dream_merge/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.beam_july_paths import ANALYSIS_DIR                      # noqa: E402
from ntof_dream_merge.bunch_join import dream_event_to_bunch         # noqa: E402
from ntof_dream_merge.intra_burst_align import align, ARMS, CORE_NS, SAT_NS  # noqa: E402

# Categorical hues in fixed order, one per arm -- validated for CVD separation
# (worst adjacent pair dE 8.1 deutan / 23.8 normal). Arms are always drawn A,B,C,D
# in this order so a colour means the same arm in every figure.
ARM_COLOR = {'A': '#3b82f6', 'B': '#f59e0b', 'C': '#10b981', 'D': '#ef4444'}
INK, MUTED, GRID = '#1f2328', '#5b6169', '#d8dbe0'


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def make_figure(r, title, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.patch.set_facecolor('white')

    # --- 1. the uncorrected drift -------------------------------------------
    ax = axes[0, 0]
    m = np.abs(r['DT']) < 20_000
    ax.hist2d(r['ET'][m] / 1e6, r['DT'][m] / 1000, bins=(60, 80),
              range=((0, 76), (-8, 12)), cmap='Blues', norm='log')
    tt = np.linspace(0, 76, 2)
    ax.plot(tt, r['t0'] / 1000 + r['k'] * tt * 1e3, color=INK, lw=2, ls='--')
    ax.annotate(f'fit: dt = {r["t0"]:.0f} ns + {r["k"]*1e6:.1f} ppm x t',
                xy=(0.04, 0.93), xycoords='axes fraction', fontsize=8.5,
                color=INK, va='top')
    ax.set_xlabel('DREAM t since flash [ms]')
    ax.set_ylabel('dt = t(n_TOF) - t(DREAM)  [us]')
    ax.set_title('1. before correction: the lag grows across the burst',
                 fontsize=9.5, color=INK, loc='left')
    _style(ax)

    # --- 2. corrected, wide --------------------------------------------------
    ax = axes[0, 1]
    ax.hist(r['resid'], bins=300, range=(-3000, 3000), color='#3b82f6', lw=0)
    ax.axvspan(-CORE_NS, CORE_NS, color=INK, alpha=0.07)
    ax.axvspan(*SAT_NS, color='#ef4444', alpha=0.09)
    ax.annotate('main', xy=(0, ax.get_ylim()[1] * 0.92), fontsize=8.5,
                color=INK, ha='center')
    ax.annotate('+330 ns\nsatellite', xy=(np.mean(SAT_NS), ax.get_ylim()[1] * 0.66),
                fontsize=8.5, color='#b91c1c', ha='left')
    ax.set_xlabel('corrected residual [ns]')
    ax.set_ylabel('pairs / 20 ns')
    ax.set_title('2. after removing the rate mismatch', fontsize=9.5,
                 color=INK, loc='left')
    _style(ax)

    # --- 3. main peak zoom ---------------------------------------------------
    ax = axes[1, 0]
    h, e = np.histogram(r['resid'], bins=160, range=(-400, 400))
    c = 0.5 * (e[1:] + e[:-1])
    ax.step(c, h, where='mid', color='#3b82f6', lw=2)
    ax.axhline(r['ped_per_ns'] * 5, color=MUTED, lw=1.5, ls=':')
    n, acc, exc = r['main']
    ax.annotate(f'sigma {r["sigma_ns"]:.0f} ns\n{exc:.0f} real / {acc:.0f} accidental'
                f'  ({exc/n:.0%} pure)',
                xy=(0.62, 0.9), xycoords='axes fraction', fontsize=8.5,
                color=INK, va='top')
    ax.annotate('accidentals', xy=(-380, r['ped_per_ns'] * 5), fontsize=8,
                color=MUTED, va='bottom')
    ax.set_xlabel('corrected residual [ns]')
    ax.set_ylabel('pairs / 5 ns')
    ax.set_title('3. the main peak', fontsize=9.5, color=INK, loc='left')
    _style(ax)

    # --- 4. per arm ----------------------------------------------------------
    ax = axes[1, 1]
    for arm in ARMS:
        sel = r['arm'] == arm
        h, e = np.histogram(r['resid'][sel], bins=80, range=(-400, 400))
        c = 0.5 * (e[1:] + e[:-1])
        ax.step(c, h, where='mid', color=ARM_COLOR[arm], lw=1.8, label=f'arm {arm}')
        ax.annotate(arm, xy=(c[h.argmax()], h.max()), fontsize=9, weight='bold',
                    color=ARM_COLOR[arm], ha='center', va='bottom')
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper right')
    ax.set_xlabel('corrected residual [ns]')
    ax.set_ylabel('pairs / 10 ns')
    ax.set_title('4. all four arms agree', fontsize=9.5, color=INK, loc='left')
    _style(ax)

    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white')
    print(f'wrote {out}')


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    r = align(nt, ev, bunches)
    make_figure(r, f'DREAM {run}/{sub}  <->  n_TOF {nt}   '
                   f'({r["n_bunches"]} bunches, {r["n_events"]:,} events)',
                ANALYSIS_DIR / 'ntof_dream_merge' / 'figures' /
                f'intra_burst_align_{run}_{sub}_{nt}.png')
