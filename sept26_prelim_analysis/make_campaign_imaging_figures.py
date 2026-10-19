#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_campaign_imaging_figures.py -- figures for `campaign_imaging.py`.

Four figures, in the order the question is asked:

  1. `img_per_run`   IS THE CROSSING STABLE -- each chamber's capsule crossing
     against run number, with the campaign band behind it.  This is the whole
     QA in one panel: the band's width is the answer.
  2. `img_axis`      THE SOURCE, AND THE ALIGNMENT -- the A/C mean (the capsule)
     and half their difference (the relative in-plane alignment), per run.
     Two quantities that only an opposing pair of chambers can separate.
  3. `img_vs_k`      AGAINST THE ANGLE SCALE -- the crossing is scale-free, so
     this should be flat.  Where it is not, something moved that changed both.
  4. `img_y`         THE k-DEPENDENT HALF -- the y offset against the polycone
     forward model, per run.  Read beside figure 1: a run normal in the
     crossing and abnormal here has an angle-scale fault, not a geometry one.

    python -m sept26_prelim_analysis.make_campaign_imaging_figures
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sept26_prelim_analysis import figstyle as F  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    ARMS, K_BLOCK, read_k)

ARM_AXIS = {'A': 'X', 'C': 'X', 'B': 'Z', 'D': 'Z'}


def _block_span(ax, R):
    """Shade the 128-147 `k` excursion so every panel reads against it."""
    lo, hi = K_BLOCK
    ax.axvspan(lo - 0.5, hi + 0.5, color='#999999', alpha=0.13, zorder=0,
               lw=0)


def fig_per_run(R: pd.DataFrame, od: Path):
    """Crossing vs run, one panel per arm, campaign band behind."""
    post = R[R.condition == 'post_access_27jul']
    fig, axes = plt.subplots(2, 2, figsize=(F.FULL[0], F.FULL[1] * 1.25),
                             sharex=True)
    for ax, arm in zip(axes.ravel(), ARMS):
        g = post[post.arm == arm].sort_values('num')
        pre = R[(R.condition == 'pre_access_27jul') & (R.arm == arm)]
        _block_span(ax, R)
        if len(g):
            med, sd = g.mm.median(), g.mm.std(ddof=1)
            ax.axhspan(med - sd, med + sd, color=F.DET_COLOR[arm], alpha=0.13,
                       lw=0, zorder=1)
            ax.axhline(med, color=F.DET_COLOR[arm], lw=1.2, ls='--', zorder=2)
            ax.errorbar(g.num, g.mm, yerr=g.err_stat, fmt='o', ms=4.5,
                        lw=1.1, color=F.DET_COLOR[arm], zorder=3,
                        label=f'median {med:+.2f}, sd {sd:.2f} mm')
        if len(pre):
            ax.errorbar(pre.num, pre.mm, yerr=pre.err_stat, fmt='s', ms=4.5,
                        mfc='none', lw=1.1, color='#666666', zorder=3,
                        label='pre-access (different detector)')
        # Robust limits.  One chamber-B run has a bootstrap error of ~30 mm on
        # 200 tracks and autoscaling to it flattens every other point in the
        # panel into a line -- the figure would then hide exactly the spread it
        # exists to show.  The point is still drawn; only the view is clipped.
        v = pd.concat([g.mm, pre.mm]).dropna()
        if len(v) > 2:
            c, s = v.median(), max(v.std(ddof=1), 0.05)
            ax.set_ylim(c - 4.5 * s, c + 4.5 * s)
        ax.set_title(f'chamber {arm}  →  global {ARM_AXIS[arm]}',
                     color=F.DET_COLOR[arm], fontsize=F.BASE_PT * 1.05)
        ax.set_ylabel(f'crossing, {ARM_AXIS[arm]} (mm)')
        ax.legend(loc='best', fontsize=F.BASE_PT * 0.98, framealpha=0.9)
        F.strip(ax)
    for ax in axes[1]:
        ax.set_xlabel('run number')
    fig.suptitle('The capsule crossing is scale-free, and it does not move',
                 fontweight='bold')
    F.note(fig, 'shaded band: the campaign median ± 1 sd of the runs; '
                'grey column: runs 128–147, where every arm’s k rose '
                'together. Error bars are the bootstrap on the band fit.')
    fig.tight_layout()
    return F.save(fig, od / 'img_per_run', R)


def fig_axis(A: pd.DataFrame, od: Path):
    """The source in X, and the A-C alignment, per run."""
    post = A[A.condition == 'post_access_27jul'].sort_values('num')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(F.FULL[0], F.FULL[1] * 1.1),
                                   sharex=True)
    for ax, col, lab, col_c in (
            (ax1, 'x_source_mm', 'capsule X  =  (A + C) / 2', '#0072B2'),
            (ax2, 'x_align_half_diff_mm', 'alignment  =  (A − C) / 2',
             '#CC79A7')):
        _block_span(ax, A)
        v = post[col].dropna()
        med, sd = v.median(), v.std(ddof=1)
        ax.axhspan(med - sd, med + sd, color=col_c, alpha=0.13, lw=0)
        ax.axhline(med, color=col_c, lw=1.2, ls='--')
        ax.errorbar(post.num, post[col], yerr=post.x_err_mm, fmt='o', ms=4.5,
                    lw=1.1, color=col_c)
        ax.set_ylabel('mm')
        ax.set_title(f'{lab}   —   {med:+.2f} ± {sd:.2f} mm',
                     color=col_c, fontsize=F.BASE_PT * 1.03)
        F.strip(ax)
    ax2.set_xlabel('run number')
    fig.suptitle('One opposing pair gives the source AND the alignment',
                 fontweight='bold')
    F.note(fig, 'The mean of two chambers that face each other is the capsule; '
                'half their difference is their relative in-plane offset. A '
                'single chamber can produce neither on its own.')
    fig.tight_layout()
    return F.save(fig, od / 'img_axis', post)


def fig_vs_k(R: pd.DataFrame, K: pd.DataFrame, od: Path):
    """Crossing against the per-run angle scale.  Algebra says flat."""
    from scipy import stats
    m = R[R.condition == 'post_access_27jul'].merge(K, on=['run', 'arm'])
    arms = [a for a in ARMS if (m.arm == a).sum() >= 6]
    fig, axes = plt.subplots(1, len(arms), figsize=(F.FULL[0], F.FULL[1] * .8),
                             squeeze=False)
    for ax, arm in zip(axes[0], arms):
        g = m[m.arm == arm]
        rho, p = stats.spearmanr(g.mm, g.k)
        inb = g.k_block.astype(bool)
        ax.scatter(g.k[~inb], g.mm[~inb], s=34, color=F.DET_COLOR[arm],
                   label='outside 128–147')
        ax.scatter(g.k[inb], g.mm[inb], s=48, facecolor='none',
                   edgecolor=F.DET_COLOR[arm], lw=1.8,
                   label='inside 128–147')
        ax.set_title(f'{arm}   ρ = {rho:+.2f}, p = {p:.3f}',
                     color=F.DET_COLOR[arm], fontsize=F.BASE_PT * 1.02)
        ax.set_xlabel('per-run k')
        ax.set_ylabel(f'crossing, {ARM_AXIS[arm]} (mm)')
        ax.legend(loc='best', fontsize=F.BASE_PT * 0.93, framealpha=0.9)
        F.strip(ax)
    fig.suptitle('A scale-free observable, against the scale',
                 fontweight='bold')
    F.note(fig, 'k cancels out of the crossing algebraically, so a correlation '
                'here is not scale dependence — it is something that moved '
                'and changed both.')
    fig.tight_layout()
    return F.save(fig, od / 'img_vs_k', m)


def fig_y(Y: pd.DataFrame, od: Path):
    """The y offset against the polycone model -- the k-dependent half."""
    Y = Y.copy()
    Y['num'] = Y.run.str.extract(r'(\d+)').astype(int)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(F.FULL[0], F.FULL[1] * 1.1),
                                   sharex=True)
    for arm in ARMS:
        g = Y[(Y.arm == arm) & Y.offset_mm.notna()].sort_values('num')
        if not len(g):
            continue
        ax1.plot(g.num, g.offset_mm, 'o-', ms=4, lw=1.0,
                 color=F.DET_COLOR[arm], label=f'{arm}')
        ax2.plot(g.num, g.width_ratio, 'o-', ms=4, lw=1.0,
                 color=F.DET_COLOR[arm])
    for ax in (ax1, ax2):
        _block_span(ax, Y)
        F.strip(ax)
    ax1.axhline(0.0, color='#333333', lw=1.0, ls=':')
    ax2.axhline(1.0, color='#333333', lw=1.0, ls=':')
    ax1.set_ylabel('median observed − model (mm)')
    ax2.set_ylabel('IQR observed / model')
    ax2.set_xlabel('run number')
    ax1.legend(ncol=4, loc='best', fontsize=F.BASE_PT * 0.98, framealpha=0.9)
    ax1.set_title('y offset against the He-3 polycone forward model',
                  fontsize=F.BASE_PT * 1.02)
    ax2.set_title('and the width ratio — 1.0 would mean the model is right',
                  fontsize=F.BASE_PT * 1.02)
    fig.suptitle('The one imaging axis that DOES move with the angle scale',
                 fontweight='bold')
    F.note(fig, 'target_y_mm is built from the calibrated direction, so unlike '
                'the crossing it carries k. The dotted lines are what a correct '
                'model and a perfect reconstruction would give.')
    fig.tight_layout()
    return F.save(fig, od / 'img_y', Y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None,
                    help='default <out>/imaging_campaign')
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('imaging_campaign')
    od = d / 'figures'
    od.mkdir(parents=True, exist_ok=True)
    F.use()

    R = pd.read_csv(paths.require(d / 'per_run.csv', 'per-run crossings'))
    A = pd.read_csv(paths.require(d / 'axis_per_run.csv', 'per-run axes'))
    fig_per_run(R, od)
    fig_axis(A, od)
    kp = d / 'versus_k.csv'
    if kp.exists():
        fig_vs_k(R, read_k(), od)
    yp = d / 'y_per_run.csv'
    if yp.exists():
        fig_y(pd.read_csv(yp), od)
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
