#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tight_figures.py -- figures for `tight_coincidence.py`.

Three panels, in the order the argument runs:

  1. `tight_delta_t`   WHY the cut is where it is: the arm1-arm2 time
     difference, with the accepted band drawn on it.
  2. `tight_spectrum`  WHAT survives: opening angle of the tight pairs against
     the loose-tagged sample and the event-mixed null, both normalised to the
     tight count so the comparison is of shape.
  3. `tight_window_scan` WHETHER IT MATTERS: surviving pairs and their median
     opening angle across the (per-arm, mutual) grid, so a reader can see
     whether the chosen point sits on a cliff or a plateau.

Every panel is labelled with its own n, because on run_145 alone n is ~20 and
no shape statement is supportable at that size. The campaign pass is what makes
this figure worth reading; until then it is a method demonstration.

    python -m sept26_prelim_analysis.make_tight_figures --run run_145
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths          # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.tight_coincidence import (  # noqa: E402
    ARM_CORE_NS, MUTUAL_NS, BACK_TO_BACK_DEG)

X17_MIN = 109.0


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def fig_delta_t(m, out):
    """arm1 - arm2, with the accepted band. The accidental floor is the point."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.55, 3.31),
                           constrained_layout=True)
    bins = np.arange(-200, 201, 12.5)
    ax.hist(m.delta_t.clip(-199, 199), bins=bins, color=fs.LINE,
            edgecolor=fs.MUTED, lw=1.0, label=f'two-arm tagged  (n={len(m)})')
    ax.axvspan(-MUTUAL_NS, MUTUAL_NS, color=fs.ACCENT, alpha=0.16, lw=0,
               label=f'accepted  |$\\Delta t$| $\\leq$ {MUTUAL_NS:.0f} ns')
    ax.axvline(0.0, color=fs.INK, lw=1.0, ls=':')
    ax.set_xlabel('$t_{arm1} - t_{arm2}$   [ns]')
    ax.set_ylabel('pairs per 12.5 ns')
    ax.set_xlim(-200, 200)
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.62, loc='upper left')
    fs.title(ax, 'One arm is the trigger; the other is often not',
             f'median |$\\Delta t$| = {m.delta_t.abs().median():.0f} ns   '
             f'against a ~5 ns single-arm reference')
    fs.preliminary(ax, loc='upper right')
    fs.save(fig, out / 'tight_delta_t',
            data=m[['subrun', 'eventId', 'arm1', 'arm2', 't1', 't2',
                    'delta_t', 'open_deg', 'topo', 'tight']])


def fig_spectrum(hist, m, out):
    """Opening angle: tight against the loose sample and the mixed null."""
    plt = _plt()
    mid = 0.5 * (hist.lo + hist.hi).to_numpy()
    n_t = int(hist.n_tight.sum())
    fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.62, 3.74),
                           constrained_layout=True)
    ax.axvspan(110, 140, color=fs.BAND_SIGNAL, alpha=0.10, lw=0,
               label='X17 signal region')
    ax.step(mid, hist.exp_mixed, where='mid', color=fs.MUTED, lw=2.0,
            label=f'event-mixed, scaled  (n={int(hist.n_mixed.sum())})')
    ax.step(mid, hist.exp_tagged, where='mid', color=fs.COPPER, lw=2.0,
            ls='--', label=f'all two-arm tagged, scaled  (n={len(m)})')
    n_p = int(hist.n_tight_pair.sum())
    # The two series differ ONLY above BACK_TO_BACK_DEG, and that difference is
    # the point of the panel: a single particle through both opposite chambers
    # passes the timing cut perfectly, because it is one particle.
    ax.axvspan(BACK_TO_BACK_DEG, 180, color=fs.BAND_DEAD, alpha=0.13, lw=0,
               label='one particle, both chambers')
    ax.errorbar(mid, hist.n_tight,
                yerr=np.sqrt(np.clip(hist.n_tight, 1, None)),
                fmt='o', ms=7, mfc='none', color=fs.MUTED, lw=1.4, capsize=3,
                zorder=5, label=f'tight, all  (n={n_t})')
    ax.errorbar(mid, hist.n_tight_pair,
                yerr=np.sqrt(np.clip(hist.n_tight_pair, 1, None)),
                fmt='o', ms=7, color=fs.ACCENT, lw=1.8, capsize=3, zorder=6,
                label=f'tight, back-to-back removed  (n={n_p})')
    ax.axvline(X17_MIN, color=fs.INK, lw=1.2, ls=':')
    ax.set_xlabel('opening angle  [deg]')
    ax.set_ylabel('pairs per bin')
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.57, loc='upper left')
    fs.title(ax, 'Timing coincidence enriches a single-particle background',
             f'{n_t - n_p} of {n_t} survivors are one track through two chambers')
    fs.preliminary(ax, loc='lower right')
    fs.save(fig, out / 'tight_spectrum', data=hist)


def fig_window_scan(scan, out):
    """Does the answer depend on exactly where the cut is put?"""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDE[0] * 0.78, 3.17),
                             constrained_layout=True)
    arms = sorted(scan.arm_ns.unique())
    cmap = plt.get_cmap('viridis')
    for i, a in enumerate(arms):
        g = scan[scan.arm_ns == a].sort_values('mutual_ns')
        c = cmap(i / max(len(arms) - 1, 1))
        axes[0].plot(g.mutual_ns, g.n, 'o-', color=c, lw=1.8, ms=5,
                     label=f'{a:.0f} ns')
        axes[1].plot(g.mutual_ns, g.median_open, 'o-', color=c, lw=1.8, ms=5)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('mutual window  |$\\Delta t$|  [ns]')
        ax.axvline(MUTUAL_NS, color=fs.INK, lw=1.0, ls=':')
    axes[0].set_ylabel('surviving pairs')
    axes[1].set_ylabel('median opening angle  [deg]')
    axes[1].axhline(X17_MIN, color=fs.ACCENT, lw=1.2, ls='--')
    axes[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.52,
                   title='per-arm window', title_fontsize=fs.BASE_PT * 0.52)
    fig.suptitle('The cut is a plateau, not a cliff -- '
                 'but n falls fast on the tight side',
                 fontsize=fs.BASE_PT * 0.98)
    fs.save(fig, out / 'tight_window_scan', data=scan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    src = paths.out('tight_coincidence')
    fig_dir = paths.figures('tight_coincidence')

    m = pd.read_parquet(src / f'pairs_tight_{a.run}.parquet')
    hist = pd.read_csv(src / f'angle_hist_{a.run}.csv')
    scan = pd.read_csv(src / f'window_scan_{a.run}.csv')

    fig_delta_t(m, fig_dir)
    fig_spectrum(hist, m, fig_dir)
    fig_window_scan(scan, fig_dir)
    print(f'  -> {fig_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
