#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_campaign_angle_figures.py -- figures for `campaign_angle.py`.

Four figures, in the order the reader needs them:

  1. `ang_topology`  WHAT THE GEOMETRY ALLOWS -- the three topologies' raw
     spectra side by side, with the X17 threshold drawn. Read this first: the
     opposing sample cannot produce an angle below ~90 deg and the intra sample
     cannot produce one above ~110, so "fraction above 109 deg" is a statement
     about the chambers before it is a statement about physics.
  2. `ang_b2b`       THE BACKGROUND INSIDE THE SIGNAL REGION -- the opposing
     spectrum with and without the back-to-back cut. One particle through both
     chambers reads as a perfectly coincident 180 deg pair.
  3. `ang_cuts`      WHAT THE TIMING CUT DOES -- the same topology at each
     stage of selection, shape-normalised, against the event-mixed null.
  4. `ang_models`    AGAINST THE PREDICTION -- the tight coincident sample
     against the thermal Born expectation folded through the acceptance.

    python -m sept26_prelim_analysis.make_campaign_angle_figures
"""
from __future__ import annotations

import argparse
import json
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
from sept26_prelim_analysis.campaign_angle import (  # noqa: E402
    BINS, TOPOLOGIES, X17_MIN_DEG, BACK_TO_BACK_DEG)

TOPO_COLOR = {'intra': '#0072B2', 'perpendicular': '#E69F00',
              'opposing': '#009E73'}
SEL_STYLE = {'all_no_b2b': ('-', 1.8, 'every pair'),
             'tagged': ('--', 1.5, 'both arms scintillator-tagged'),
             'tight_pair': ('-', 2.6, 'tight coincidence'),
             'mixed': (':', 1.6, 'event-mixed (accidentals)')}


def _series(S, topo, sel):
    g = S[(S.topology == topo) & (S.selection == sel)].sort_values('theta')
    return (g.theta.to_numpy(), g.n.to_numpy(float)) if len(g) else (None, None)


def _x17_band(ax):
    ax.axvspan(X17_MIN_DEG, 180.0, color='#CC79A7', alpha=0.10, lw=0, zorder=0)
    ax.axvline(X17_MIN_DEG, color='#CC79A7', lw=1.3, ls='--', zorder=1)


def fig_topology(S: pd.DataFrame, od: Path):
    """The raw spectra, one panel per topology.  Counts, not shapes."""
    fig, axes = plt.subplots(1, 3, figsize=(F.FULL[0], F.FULL[1] * 0.85),
                             squeeze=False)
    for ax, topo in zip(axes[0], TOPOLOGIES):
        x, y = _series(S, topo, 'all_no_b2b')
        _x17_band(ax)
        if x is not None:
            ax.step(x, y, where='mid', lw=2.2, color=TOPO_COLOR[topo])
            ax.fill_between(x, 0, y, step='mid', color=TOPO_COLOR[topo],
                            alpha=0.20)
            frac = y[x > X17_MIN_DEG].sum() / max(y.sum(), 1)
            ax.set_title(topo, color=TOPO_COLOR[topo])
            # The counts go INSIDE the axes: three panel titles of this length
            # overlap each other at slide width and the figure becomes
            # unreadable exactly where the numbers are.
            ax.annotate(f'{int(y.sum()):,} pairs\n{100 * frac:.1f} % above 109°',
                        (0.04, 0.94), xycoords='axes fraction', va='top',
                        fontsize=F.BASE_PT * 0.98, color=TOPO_COLOR[topo],
                        fontweight='bold')
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 45))
        ax.set_xlabel('opening angle (deg)')
        F.strip(ax)
    axes[0][0].set_ylabel('pairs per 15°')
    fig.suptitle('The chambers, not the physics, set where a pair can appear',
                 fontweight='bold')
    F.note(fig, 'Every real two-track trigger pair, back-to-back removed. '
                'The shaded region above 109° is where an X17 pair must land; '
                'the intra sample cannot reach it and the opposing sample '
                'cannot leave it.')
    fig.tight_layout()
    return F.save(fig, od / 'ang_topology', S[S.selection == 'all_no_b2b'])


def fig_b2b(S: pd.DataFrame, P: pd.DataFrame, K: pd.DataFrame, od: Path):
    """The opposing spectrum before and after the single-particle cut."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(F.FULL[0], F.FULL[1] * .85))
    real = P[(~P.mixed) & (P.topo == 'opposing')]
    edges = np.arange(90.0, 180.01, 2.0)
    h, _ = np.histogram(real.open_deg, bins=edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    _x17_band(ax1)
    ax1.step(mid, h, where='mid', lw=1.6, color=TOPO_COLOR['opposing'])
    ax1.axvline(BACK_TO_BACK_DEG, color='#D55E00', lw=1.6)
    ax1.annotate(f'{BACK_TO_BACK_DEG:.0f}°', (BACK_TO_BACK_DEG, ax1.get_ylim()[1]),
                 xytext=(-6, -14), textcoords='offset points', ha='right',
                 color='#D55E00', fontweight='bold')
    ax1.set_xlim(90, 180)
    ax1.set_xlabel('opening angle (deg)')
    ax1.set_ylabel('pairs per 2°')
    ax1.set_title('the spike is one particle, not a pair',
                  fontsize=F.BASE_PT * 1.02)
    F.strip(ax1)

    x, ya = _series(S, 'opposing', 'all')
    _, yb = _series(S, 'opposing', 'all_no_b2b')
    xt, yt = _series(S, 'opposing', 'tight')
    _, ytp = _series(S, 'opposing', 'tight_pair')
    _x17_band(ax2)
    ax2.step(x, ya / ya.sum(), where='mid', lw=1.5, ls='--', color='#666666',
             label='all opposing')
    ax2.step(x, yb / yb.sum(), where='mid', lw=2.0,
             color=TOPO_COLOR['opposing'], label='back-to-back removed')
    if yt is not None:
        ax2.step(xt, yt / yt.sum(), where='mid', lw=1.5, ls=':',
                 color='#D55E00', label='tight (b2b still in)')
        ax2.step(xt, ytp / ytp.sum(), where='mid', lw=2.6, color='#CC79A7',
                 label='tight_pair')
    ax2.set_xlim(90, 180)
    ax2.set_xlabel('opening angle (deg)')
    ax2.set_ylabel('fraction of the sample')
    ax2.legend(loc='upper left', fontsize=F.BASE_PT * 0.98, framealpha=0.9)
    ax2.set_title('and the timing cut ENRICHES it',
                  fontsize=F.BASE_PT * 1.02)
    F.strip(ax2)
    fig.suptitle('The back-to-back background sits inside the signal region',
                 fontweight='bold')
    opp = K[K.topology == 'opposing']
    n_b2b = int(opp.n_b2b.iloc[0]) if len(opp) else 0
    F.note(fig, f'{n_b2b:,} of the opposing pairs are above '
                f'{BACK_TO_BACK_DEG:.0f}°. A single charged particle that '
                f'crosses the target and punches through both chambers is '
                f'perfectly time-coincident, because it is one particle.')
    fig.tight_layout()
    return F.save(fig, od / 'ang_b2b',
                  {'fine': pd.DataFrame(dict(theta=mid, n=h)),
                   'binned': S[S.topology == 'opposing']})


def fig_cuts(S: pd.DataFrame, od: Path):
    """Each selection's SHAPE, per topology, against the mixed null."""
    topos = [t for t in TOPOLOGIES
             if len(S[(S.topology == t) & (S.selection == 'tight_pair')])]
    if not topos:
        topos = list(TOPOLOGIES)
    fig, axes = plt.subplots(1, len(topos),
                             figsize=(F.FULL[0], F.FULL[1] * 0.85),
                             squeeze=False)
    for ax, topo in zip(axes[0], topos):
        _x17_band(ax)
        for sel, (ls, lw, lab) in SEL_STYLE.items():
            x, y = _series(S, topo, sel)
            if x is None or y.sum() == 0:
                continue
            col = ('#666666' if sel == 'mixed' else
                   TOPO_COLOR[topo] if sel == 'tight_pair' else None)
            ax.step(x, y / y.sum(), where='mid', ls=ls, lw=lw, color=col,
                    label=f'{lab}  (n = {int(y.sum()):,})')
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 45))
        ax.set_xlabel('opening angle (deg)')
        ax.set_title(topo, color=TOPO_COLOR[topo])
        ax.legend(loc='upper left', fontsize=F.BASE_PT * 0.89, framealpha=0.9)
        F.strip(ax)
    axes[0][0].set_ylabel('fraction of the sample')
    fig.suptitle('What the timing cut removes, and what it leaves',
                 fontweight='bold')
    F.note(fig, 'Shape-normalised, because the samples differ by a factor 60 '
                'in size. The intra topology has only one arm and therefore no '
                'arm-to-arm time difference, so the tight cut cannot be formed '
                'for it at all.')
    fig.tight_layout()
    return F.save(fig, od / 'ang_cuts', S)


def fig_models(S: pd.DataFrame, E: pd.DataFrame, C: pd.DataFrame, od: Path):
    """The coincident sample against the folded expectation."""
    topos = [t for t in ('opposing', 'perpendicular')
             if len(S[(S.topology == t) & (S.selection == 'tight_pair')])]
    fig, axes = plt.subplots(1, max(len(topos), 1),
                             figsize=(F.FULL[0], F.FULL[1] * 0.85),
                             squeeze=False)
    for ax, topo in zip(axes[0], topos):
        x, y = _series(S, topo, 'tight_pair')
        _x17_band(ax)
        n = y.sum()
        ax.errorbar(x, y, yerr=np.sqrt(np.clip(y, 1, None)), fmt='o', ms=5,
                    lw=1.4, color='#111111', zorder=5,
                    label=f'tight coincident  (n = {int(n):,})')
        cc = C[(C.topology == topo) & (C.selection == 'tight_pair')]
        for name, g in E[E.topology == topo].groupby('model'):
            p = g.sort_values('theta').frac.to_numpy()
            if not np.isfinite(p).any() or p.sum() <= 0:
                continue
            row = cc[cc.model == name]
            lab = name + (f'  χ²/dof {row.chi2dof.iloc[0]:.0f}'
                          if len(row) else '')
            ax.step(x, p * n, where='mid', lw=1.6, label=lab)
        mx = S[(S.topology == topo) & (S.selection == 'mixed')] \
            .sort_values('theta').n.to_numpy(float)
        if mx.sum():
            row = cc[cc.model.str.startswith('event-mixed')]
            lab = 'event-mixed' + (f'  χ²/dof {row.chi2dof.iloc[0]:.0f}'
                                   if len(row) else '')
            ax.step(x, mx / mx.sum() * n, where='mid', lw=2.0, ls=':',
                    color='#666666', label=lab)
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 45))
        ax.set_xlabel('opening angle (deg)')
        ax.set_title(topo, color=TOPO_COLOR[topo])
        ax.legend(loc='upper left', fontsize=F.BASE_PT * 0.86, framealpha=0.9)
        F.strip(ax)
    axes[0][0].set_ylabel('pairs per 15°')
    fig.suptitle('The coincident sample against the thermal Born prediction',
                 fontweight='bold')
    F.note(fig, 'Every model is normalised to the observed count — no rate is '
                'claimed. The acceptance folded in is run_145’s, borrowed: '
                'there is no campaign acceptance, and that is the leading '
                'systematic on every curve here.')
    fig.tight_layout()
    return F.save(fig, od / 'ang_models', {'obs': S, 'models': E})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None, help='default <out>/angle_campaign')
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('angle_campaign')
    od = d / 'figures'
    od.mkdir(parents=True, exist_ok=True)
    F.use()

    S = pd.read_csv(paths.require(d / 'spectra.csv', 'the spectra'))
    E = pd.read_csv(paths.require(d / 'expectation.csv', 'the expectation'))
    C = pd.read_csv(paths.require(d / 'compare.csv', 'the comparison'))
    K = pd.read_csv(paths.require(d / 'census.csv', 'the census'))
    P = pd.read_parquet(paths.require(d / 'pairs.parquet', 'the pairs'))

    fig_topology(S, od)
    fig_b2b(S, P, K, od)
    fig_cuts(S, od)
    fig_models(S, E, C, od)
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
