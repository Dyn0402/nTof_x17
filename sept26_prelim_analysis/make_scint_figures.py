#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_scint_figures.py -- the three figures for the scintillator page.

  scint_roles       what each element can localise, and how coarsely.  The
                    figure that says "filter, not measurement" in one look.
  wall_along_bar    log(A_1/A_2) against the Micromegas track's y at the wall,
                    one panel per chamber, with the robust fit and the implied
                    attenuation length.  Chamber D's slope runs the other way.
  wall_selfcheck    the two estimators against each other -- no Micromegas at
                    all, so chamber B is in it too.

    python -m sept26_prelim_analysis.make_scint_figures --run run_145
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

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def _scaled(plt, f=0.74):
    return plt.rc_context({'font.size': fs.BASE_PT * f,
                           'axes.labelsize': fs.BASE_PT * f,
                           'axes.titlesize': fs.BASE_PT * f * 1.05,
                           'xtick.labelsize': fs.BASE_PT * f * 0.86,
                           'ytick.labelsize': fs.BASE_PT * f * 0.86})


# ------------------------------------------------------------------- roles
def fig_roles(aud: pd.DataFrame, cal: pd.DataFrame, out):
    """How coarsely each element localises a particle, in each direction.

    The bar length IS the position uncertainty, so a long bar is a bad
    detector.  That is the whole argument of the page in one axis.
    """
    plt = _plt()
    # (label, across the wall [u], along the bar [v], note)
    lr = cal.dropna(subset=['lr_resid_mm'])
    along_meas = float(lr.lr_resid_mm.min()) if len(lr) else np.nan
    rows = [
        ('wall segment\n(4 groups of 4 bars)', 100.0, 500.0,
         'what we use today: which group fired'),
        ('plastic bar\n(2 bars)', 200.0, 300.0,
         'what we use today: which bar fired'),
        ('liquid cell\n(1 channel)', 451.0, 450.0,
         'no internal structure at all'),
        ('wall, BOTH ENDS\n(this work)', 100.0, along_meas,
         'log amplitude ratio along the bar'),
    ]
    with _scaled(plt, 0.78):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.72, 3.53),
                               constrained_layout=True)
        y = np.arange(len(rows))[::-1]
        h = 0.34
        for i, (lab, u, v, note) in zip(y, rows):
            new = 'this work' in lab
            for off, val, col, nm in ((+h / 2, u, fs.DET_COLOR['A'], 'across'),
                                      (-h / 2, v, fs.ACCENT, 'along the bar')):
                if not np.isfinite(val):
                    continue
                ax.barh(i + off, val, height=h, color=col,
                        alpha=0.95 if new else 0.42,
                        edgecolor=col, lw=1.6 if new else 0, zorder=3)
                ax.annotate(f'{val:.0f} mm', (val, i + off),
                            textcoords='offset points', xytext=(7, 0),
                            va='center', fontsize=fs.BASE_PT * 0.64,
                            color=col, fontweight='bold' if new else 'normal')
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel('position uncertainty of one fired element  [mm]')
        ax.set_xlim(0, 600)
        ax.set_title('What each element can localise — and the one we do '
                     'not read yet')
        # The per-row notes live in the caption, not in the plot: at this
        # aspect they collide with the value labels, and the message of the
        # figure is the bar lengths.
        h1 = ax.barh([-9], [0], color=fs.DET_COLOR['A'], alpha=0.6,
                     label='across the wall  (u)')
        h2 = ax.barh([-9], [0], color=fs.ACCENT, alpha=0.6,
                     label='along the bar  (v = beam axis, y)')
        ax.legend(frameon=False, loc='lower right',
                  fontsize=fs.BASE_PT * 0.74)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        fs.preliminary(ax, loc='upper right')
        fs.save(fig, out / 'scint_roles',
                data=pd.DataFrame(rows, columns=['element', 'across_mm',
                                                 'along_mm', 'note']))


# --------------------------------------------------------------- along bar
def fig_along_bar(m: pd.DataFrame, cal: pd.DataFrame, out):
    """The calibration itself: log ratio vs the track's y at the wall."""
    plt = _plt()
    arms = [a for a in ARMS if a in set(m.arm)]
    with _scaled(plt, 0.7):
        fig, axes = plt.subplots(1, len(arms), figsize=(fs.WIDE[0], 3.31),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        edges = np.arange(-250, 251, 50.0)
        rows = []
        for ax, arm in zip(axes, arms):
            g = m[m.arm == arm]
            c = cal[cal.arm == arm].iloc[0]
            ax.hexbin(g.y_wall, g.log_ratio_adj, gridsize=34,
                      extent=(-260, 260, -1.6, 1.6), mincnt=1,
                      cmap='Blues', linewidths=0, zorder=1)
            mid, med, err = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = g[(g.y_wall >= lo) & (g.y_wall < hi)]
                if len(s) < 25:
                    continue
                mid.append(0.5 * (lo + hi))
                med.append(float(s.log_ratio_adj.median()))
                err.append(float(1.4826 * np.median(np.abs(
                    s.log_ratio_adj - s.log_ratio_adj.median()))
                    / np.sqrt(len(s))))
                rows.append(dict(arm=arm, y_mid=mid[-1], log_ratio=med[-1],
                                 err=err[-1], n=len(s)))
            ax.errorbar(mid, med, yerr=err, fmt='o', ms=6,
                        color=fs.DET_COLOR[arm], lw=1.8, capsize=3, zorder=4)
            xx = np.array([-250, 250.])
            ax.plot(xx, c.lr_slope * xx, '-',
                    color=fs.DET_COLOR[arm], lw=2.2, zorder=5)
            ax.axhline(0, color=fs.MUTED, lw=0.9, ls=':', zorder=2)
            ax.axvline(0, color=fs.MUTED, lw=0.9, ls=':', zorder=2)
            flip = '  ← the other way' if c.lr_slope > 0 else ''
            ax.set_title(f'chamber {arm}{flip}',
                         color=fs.DET_COLOR[arm], fontweight='600')
            ax.annotate(f'$\\lambda$ = {abs(c.lambda_mm):.0f} mm\n'
                        f'r = {c.lr_corr:+.2f}\n'
                        f'$\\sigma_y$ < {c.lr_resid_mm:.0f} mm',
                        (0.03, 0.03), xycoords='axes fraction',
                        va='bottom', fontsize=fs.BASE_PT * 0.64,
                        color=fs.INK, linespacing=1.5)
            ax.set_xlabel('track y at the wall  [mm]')
            ax.set_xlim(-260, 260)
            ax.set_ylim(-1.6, 1.6)
        axes[0].set_ylabel('log( A$_1$ / A$_2$ ),  group offset removed')
        fig.suptitle('The wall does measure position along its bars '
                     '— and chamber D reads it backwards',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(axes[-1], loc='upper right')
        fs.save(fig, out / 'wall_along_bar', data=pd.DataFrame(rows))


# --------------------------------------------------------------- self check
def fig_selfcheck(pairs: pd.DataFrame, sc: pd.DataFrame, out):
    """dt against log ratio -- two different physics, one coordinate, no MM."""
    plt = _plt()
    g0 = pairs[pairs.physical]
    arms = [a for a in ARMS if a in set(g0.arm)]
    with _scaled(plt, 0.7):
        fig, axes = plt.subplots(1, len(arms), figsize=(fs.WIDE[0], 3.17),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        rows = []
        for ax, arm in zip(axes, arms):
            g = g0[g0.arm == arm].copy()
            # Per-group centring, because each bar group is read through its
            # own cables: pooled, the four groups make four parallel bands and
            # the correlation is an underestimate of what one group carries.
            g['dt_c'] = g.dt_ns_ends - g.groupby('grp').dt_ns_ends.transform('median')
            g['lr_c'] = g.log_ratio - g.groupby('grp').log_ratio.transform('median')
            r = float(np.corrcoef(g.dt_c, g.lr_c)[0, 1])
            med = 0.0
            ax.hexbin(g.dt_c - med, g.lr_c, gridsize=32,
                      extent=(-8, 8, -1.6, 1.6), mincnt=1, cmap='Purples',
                      linewidths=0)
            ax.axhline(0, color=fs.MUTED, lw=0.9, ls=':')
            ax.axvline(0, color=fs.MUTED, lw=0.9, ls=':')
            tag = '  (no tracks)' if arm == 'B' else ''
            ax.set_title(f'chamber {arm}{tag}', color=fs.DET_COLOR[arm],
                         fontweight='600')
            ax.annotate(f'r = {r:+.2f}\nn = {len(g):,}',
                        (0.03, 0.03), xycoords='axes fraction', va='bottom',
                        fontsize=fs.BASE_PT * 0.67, color=fs.INK,
                        linespacing=1.5)
            ax.set_xlabel('$\\Delta t$, group-centred  [ns]')
            rows.append(dict(arm=arm, n=len(g), corr=r,
                             dt_median=float(g.dt_ns_ends.median())))
        axes[0].set_ylabel('log( A$_1$ / A$_2$ ),  group-centred')
        fig.suptitle('Delay and attenuation agree with each other in all four '
                     'chambers — including B, which has no tracks',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(axes[-1], loc='upper right')
        fs.save(fig, out / 'wall_selfcheck', data=pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    sd = paths.out('scint')
    od = paths.out('scint', 'figures')

    aud = pd.read_csv(paths.require(sd / f'audit_{a.run}.csv', 'the audit'))
    cal = pd.read_csv(paths.require(sd / f'wall_calibration_{a.run}.csv',
                                    'the wall calibration'))
    sc = pd.read_csv(paths.require(sd / f'self_consistency_{a.run}.csv',
                                   'the self-consistency table'))
    m = pd.read_parquet(paths.require(sd / f'wall_matched_{a.run}.parquet',
                                      'the matched sample'))

    from sept26_prelim_analysis import scintillators as S
    slim = S.read_slim(a.run, ['stat090_0000', 'stat090_0001', 'stat090_0002'])
    pairs = S.wall_pairs(slim)

    fig_roles(aud, cal, od)
    fig_along_bar(m, cal, od)
    fig_selfcheck(pairs, sc, od)
    print(f'wrote 3 figures (+ CSVs) to {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
