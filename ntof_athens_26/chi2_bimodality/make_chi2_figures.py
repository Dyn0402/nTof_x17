#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chi2_figures.py -- the five figures of the chi2 bimodality investigation.

Drawing only.  Every number comes from ``<out>/chi2_bimodality/``, which
`chi2_shape.py` writes; nothing is recomputed here.

  1. ``chi2_decomposition``  **the answer.**  One panel per chamber: the
     chi2/dof distribution the QA plot shows, with the same tracks split by
     track length underneath it.  The two bumps are the two length populations,
     on every chamber -- and A is the only one where they separate.
  2. ``chi2_vs_length``      the universal rise and the per-chamber offset, with
     the noise floor drawn.  This is what sets the two modes apart.
  3. ``chi2_vs_charge``      at fixed length, the U in pulse amplitude: at the
     floor in a middle band, up at both ends.
  4. ``chi2_by_run``         the offset is a chamber constant -- 36 runs, both
     access conditions, ordering never moves.
  5. ``pair_link``           back to the published figure: the pair-level
     `chi2dof_worst` curve, split by whether chamber A is a leg.

    python ntof_athens_26/chi2_bimodality/make_chi2_figures.py
    python ntof_athens_26/chi2_bimodality/make_chi2_figures.py --only chi2_vs_length
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402
from mpgd26 import plotstyle as ps                                # noqa: E402

OUT = HERE / 'figures'
SRC = paths.spell('out', 'chi2_bimodality')

ARMS = ('A', 'B', 'C', 'D')
DASH = '—'

#: What each chamber is, in one clause, for the panel decks.
WHAT = {
    'A': 'reaches the noise floor',
    'B': 'no field cage — no ladder to fit',
    'C': 'a bundle generation behind',
    'D': 'no channel mask, prior off',
}

#: The two length classes the decomposition draws as the two modes.  The middle
#: classes are still in the grey 'all' curve; they are the crossover and drawing
#: every one of them would bury the point.
SHORT, LONG = '11-13', ['34-59', '60+']

FLOOR = 1.5


def _load(name: str) -> pd.DataFrame:
    f = SRC / f'{name}.csv'
    if not f.exists():
        raise SystemExit(f'missing {f} -- run chi2_shape.py first')
    return pd.read_csv(f)


def _summary() -> dict:
    return json.loads((SRC / 'summary.json').read_text())


def _logx(ax, lo=-0.6, hi=3.4):
    """A log10 axis labelled in chi2/dof, not in the log.

    Decade ticks only, with unlabelled minors: labelling the half-decades wrote
    '3.16228' under every panel and the axis became the loudest thing on the
    figure.
    """
    ax.set_xlim(lo, hi)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['1', '10', '100', '1000'])
    ax.set_xticks([np.log10(v) for v in
                   (0.3, 0.5, 3, 5, 30, 50, 300, 500)], minor=True)


def _floor_line(ax, label=True):
    ax.axvline(np.log10(1.0), color=ps.MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    if label:
        ax.text(np.log10(1.0), ax.get_ylim()[1], ' noise floor', rotation=90,
                va='top', ha='left', fontsize=8.5, color=ps.MUTED)


# --------------------------------------------------------------------------- #
def fig_decomposition(save=True):
    """The two bumps ARE short tracks and long tracks -- on every chamber.

    Top row: the distribution as the QA figure shows it, with the two length
    populations stacked underneath, so the reader sees how much of the sample
    each mode is.  Bottom row: the same two populations each normalised to
    ITSELF, so their MODES can be compared -- which is the actual claim, and it
    is invisible in the top row because the short-track population is a tenth
    of the long one.
    """
    h = _load('chi2_hist')
    S = _summary()
    arms = [a for a in ARMS if a in set(h.arm)]
    fig, axes = plt.subplots(2, len(arms), figsize=(3.7 * len(arms), 7.4),
                             sharex=True)
    for k, arm in enumerate(arms):
        g = h[h.arm == arm]
        col = ps.DET_COLOR[arm]
        st = S['per_arm'].get(arm, {})
        short = g[g['slice'] == SHORT].sort_values('log_chi2dof')
        lg = (g[g['slice'].isin(LONG)]
              .groupby('log_chi2dof', as_index=False)[['frac', 'n']].sum()
              .sort_values('log_chi2dof'))
        lg_own = lg.assign(frac_own=lg.n / max(lg.n.sum(), 1))

        # ---- top: as published, with the components in place
        ax = axes[0, k]
        allc = g[g['slice'] == 'all'].sort_values('log_chi2dof')
        ax.fill_between(allc.log_chi2dof, 0, allc.frac, color=ps.LINE,
                        zorder=1, label='all tracks')
        ax.fill_between(short.log_chi2dof, 0, short.frac, color=col,
                        alpha=0.85, lw=0, zorder=3,
                        label=f'short, {SHORT} strips')
        ax.plot(lg.log_chi2dof, lg.frac, color=ps.INK, lw=1.6, ls='--',
                zorder=2, label='long, 34+ strips')
        ax.set_ylim(0, max(allc.frac.max(), 1e-3) * 1.30)
        ps.title(ax, f'chamber {arm}', WHAT[arm])
        ps.strip(ax, left=(k == 0))
        if k == 0:
            ax.set_ylabel('fraction of all tracks')
            ax.legend(frameon=False, fontsize=8.5, loc='upper left')
        else:
            ax.set_yticks([])

        # ---- bottom: each population normalised to itself
        ax = axes[1, k]
        ax.fill_between(short.log_chi2dof, 0, short.frac_own, color=col,
                        alpha=0.85, lw=0, zorder=3)
        ax.plot(lg_own.log_chi2dof, lg_own.frac_own, color=ps.INK, lw=1.6,
                ls='--', zorder=2)
        top = max(short.frac_own.max(), lg_own.frac_own.max(), 1e-3) * 1.35
        ax.set_ylim(0, top)
        _logx(ax)
        _floor_line(ax, label=(k == 0))
        sep = st.get('separation')
        if sep:
            ms, ml = st.get('median_short') or 0, st.get('median_long') or 0
            ax.annotate('', xy=(np.log10(ml), top * 0.80),
                        xytext=(np.log10(ms), top * 0.80),
                        arrowprops=dict(arrowstyle='<->', color=ps.MUTED, lw=1.1))
            ax.text(np.log10(np.sqrt(ms * ml)), top * 0.84, f'{sep:.0f}×',
                    ha='center', va='bottom', fontsize=10.5, color=ps.INK,
                    fontweight='bold')
            ax.text(0.97, 0.97, f'short {ms:>5.1f}\nlong  {ml:>5.0f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=ps.MUTED, family='monospace')
        ps.strip(ax, left=(k == 0))
        if k == 0:
            ax.set_ylabel('each population,\nnormalised to itself')
        else:
            ax.set_yticks([])
    for ax in axes[0]:
        _logx(ax)
    fig.suptitle('The two bumps are short tracks and long tracks',
                 x=0.007, ha='left', fontsize=15, fontweight='bold',
                 color=ps.INK)
    fig.supxlabel('χ²/dof of the x-view waveform fit', fontsize=11,
                  color=ps.INK)
    ps.note(fig, f'{S["n_tracks"]:,} tracks, the legs of the published pair '
            'sample (gated, angle-calibrated, DCA < 30 mm, in a trigger that '
            'made two).  χ²/dof is the mean squared residual PER '
            'WAVEFORM SAMPLE in units of that strip’s own measured noise '
            '(dof = 20 × n_strips), so 1.0 is the noise floor on every '
            'chamber and the three panels are directly comparable.  Both '
            'populations exist on all three; only on A does the short-track '
            'mode reach the floor, and that is what opens the gap.  Chamber B '
            'carries no angle calibration and so is in no pairing and not on '
            'this figure.', y=-0.015)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    if save:
        _save(fig, 'chi2_decomposition', h)
    return fig


def fig_vs_length(save=True):
    """chi2/dof climbs with track length on every chamber, from a different floor."""
    t = _load('chi2_vs_len')
    order = list(pd.unique(t.len_class))
    xi = {lab: i for i, lab in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhspan(0, FLOOR, color=ps.LINE, alpha=0.55, zorder=0)
    ax.text(len(order) - 0.5, FLOOR * 0.97, 'at the noise floor', ha='right',
            va='top', fontsize=9.5, color=ps.MUTED)
    for arm in ARMS:
        g = t[t.arm == arm].copy()
        if not len(g):
            continue
        g['x'] = g.len_class.map(xi)
        g = g.sort_values('x')
        c = ps.DET_COLOR[arm]
        ax.fill_between(g.x, g.p25, g.p75, color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(g.x, g['median'], color=c, lw=2.1, marker=ps.DET_MARKER[arm],
                ms=6, zorder=3)
        ps.end_label(ax, g.x.iloc[-1], g['median'].iloc[-1], f'  {arm}', c)
    ax.set_yscale('log')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.3, len(order) - 0.3)
    ax.set_xlabel('strips in the fitted window  (track length)')
    ax.set_ylabel('χ²/dof   median, with the quartile band')
    ps.title(ax, 'Every chamber degrades with track length',
             'the offset between them is what decides whether the two modes separate')
    ps.strip(ax)
    # spelled from the table, not typed: this note quotes four numbers and a
    # re-run that moved them would otherwise leave the caption quietly wrong
    ends = {a: (t[t.arm == a].sort_values('len_class', key=lambda s: s.map(xi)))
            for a in ARMS if (t.arm == a).any()}
    say = '  '.join(
        f'{a} runs {g["median"].iloc[0]:.1f} → {g["median"].iloc[-1]:.0f} '
        f'({g["median"].iloc[-1] / g["median"].iloc[0]:.0f}×).'
        for a, g in ends.items())
    ps.note(fig, f'Legs of the published pair sample, x view.  {say}  The rise '
            'is universal, so the bimodality is universal; the floor each '
            'chamber starts from is not — and a chamber whose short tracks '
            'already sit near its long ones can never show two bumps.', y=-0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        _save(fig, 'chi2_vs_length', t)
    return fig


def fig_vs_charge(save=True):
    """At fixed length the driver is pulse amplitude.

    In the SELECTED sample the dependence is monotone: chi2/dof climbs from the
    floor at the quiet end to 3-20 at the loud end.  The low-amplitude upturn
    that shows in the unselected stage-3 population -- fits that explained
    almost nothing, so the whole pulse is residual -- is mostly cut by the
    gated / angle-calibrated / DCA selection, and survives here only on C and D
    in the 18-23 strip panel.  The figure says the monotone thing, because that
    is the thing this sample supports.
    """
    g = _load('chi2_grid')
    keep = [c for c in ('11-13', '14-17', '18-23') if c in set(g.len_class)]
    fig, axes = plt.subplots(1, len(keep), figsize=(13.5, 4.4), sharey=True)
    for ax, lab in zip(np.atleast_1d(axes), keep):
        for arm in ARMS:
            s = g[(g.arm == arm) & (g.len_class == lab)].sort_values('q_bin')
            if len(s) < 3:
                continue
            ax.plot(s.q_bin, s['median'], color=ps.DET_COLOR[arm], lw=2.0,
                    marker=ps.DET_MARKER[arm], ms=6)
            ps.end_label(ax, s.q_bin.iloc[-1], s['median'].iloc[-1], f'  {arm}',
                         ps.DET_COLOR[arm])
        ax.axhspan(0, FLOOR, color=ps.LINE, alpha=0.55, zorder=0)
        ax.set_yscale('log')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['quietest', '', 'middle', '', 'loudest'])
        ax.set_xlim(-0.3, 4.6)
        ps.title(ax, f'{lab} strips')
        ps.strip(ax, left=(lab == keep[0]))
        if lab == keep[0]:
            ax.set_ylabel('χ²/dof   median')
    fig.suptitle('At fixed track length, the loud tracks are the bad fits',
                 x=0.007, ha='left', fontsize=15, fontweight='bold', color=ps.INK)
    fig.supxlabel('charge per strip, in that chamber’s own quintiles',
                  fontsize=11, color=ps.INK)
    ps.note(fig, 'Quintiles are per chamber, because the gains are not '
            'cross-calibrated — a bin means "this chamber’s quietest '
            'fifth", never a shared charge.  This is the mechanism behind the '
            'length trend: the model’s residual is a fixed FRACTION of the '
            'pulse, so on a quiet track it is buried in the noise and '
            'χ²/dof sits on the floor, while on a loud one it '
            'outgrows the noise.  A’s low-χ² tracks carry ~2.5× '
            'less charge than its high-χ² ones at identical n_strips.  '
            'The upturn at the quiet end of C and D in the third panel is the '
            'other failure — a fit that explained nothing — which this '
            'selection mostly, but not entirely, removes.', y=-0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    if save:
        _save(fig, 'chi2_vs_charge', g)
    return fig


def fig_by_run(save=True):
    """The per-chamber offset is a constant of the chamber, not of the beam."""
    t = _load('chi2_by_run')
    runs = sorted(pd.unique(t.run), key=lambda r: int(str(r).split('_')[-1]))
    xi = {r: i for i, r in enumerate(runs)}
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    pre = t[t.condition == 'pre_access_27jul']
    if len(pre):
        edge = max(xi[r] for r in pd.unique(pre.run)) + 0.5
        ax.axvspan(-0.5, edge, color=ps.LINE, alpha=0.45, zorder=0)
        ax.text(edge - 0.2, ax.get_ylim()[1], ' pre-access  ', ha='right',
                va='top', fontsize=9, color=ps.MUTED)
    last = {}
    for arm in ARMS:
        g = t[t.arm == arm].copy()
        if not len(g):
            continue
        g['x'] = g.run.map(xi)
        g = g.sort_values('x')
        ax.plot(g.x, g['median'], color=ps.DET_COLOR[arm], lw=1.6,
                marker=ps.DET_MARKER[arm], ms=4.5)
        last[arm] = float(g['median'].iloc[-1])
    # C and D end within a few percent of each other, so a label placed at each
    # curve's own y lands on top of the other one
    for rank, (arm, y) in enumerate(sorted(last.items(), key=lambda kv: -kv[1])):
        ax.plot([len(runs) - 1, len(runs) - 0.4], [y, y], color=ps.LINE, lw=0.8,
                zorder=0)
        ps.end_label(ax, len(runs) - 0.3, y * (1.10 if rank == 0 else
                                               0.91 if rank == 1 else 1.0),
                     f' {arm}', ps.DET_COLOR[arm])
    ax.set_yscale('log')
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([str(r).replace('run_', '') for r in runs], fontsize=8,
                       rotation=90)
    ax.set_xlim(-0.6, len(runs) + 1.2)
    ax.set_xlabel('run')
    ax.set_ylabel('χ²/dof  median, short tracks (11–13 strips)')
    gap = (np.median([v for a, v in last.items() if a != 'A']) / last['A']
           if 'A' in last and len(last) > 1 else float('nan'))
    ps.title(ax, 'A sits below the others in every single run',
             f'short tracks, {len(runs)} runs, both access conditions '
             f'— A is a factor {gap:.1f} under C and D throughout')
    ps.strip(ax)
    ps.note(fig, 'Short tracks only, so the length mixture cannot move between '
            'runs.  Nothing here tracks the beam, the period or the 27 July '
            'access — A holds its separation run by run, which rules out a '
            'run-dependent cause and points at what is fixed per chamber: the '
            'calibration bundle.  C and D interleave with each other, so this '
            'figure separates A from the pair and does NOT rank C against D.',
            y=-0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        _save(fig, 'chi2_by_run', t)
    return fig


def fig_pair_link(save=True):
    """Most of the published bump is a normalisation artefact.  Some of it is not.

    Left: the intra pairs exactly as `make_pair_qa_figures` draws them --
    ``n/tot/dx`` with dx the LINEAR width of a geometric bin, so the height is
    the per-bin fraction divided by chi2/dof.  Right: the same counts as a
    per-bin fraction, which is what the log axis invites the reader to read.
    The low bump loses about 5x of its height and stops being a mode.
    """
    f = SRC / 'pair_hist.csv'
    if not f.exists():
        print('  .. no pair_hist.csv; skipping pair_link')
        return None
    t = pd.read_csv(f)
    intra = t[t.topology == 'intra'] if 'intra' in set(t.topology) else t
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    for ax, col, lab in ((axes[0], 'dens_linear',
                          'density per unit χ²/dof   (as published)'),
                         (axes[1], 'frac', 'fraction of pairs per bin')):
        for pair, g in intra.groupby('pair'):
            g = g.sort_values('log_chi2dof')
            a = pair.split('-')[0]
            c = ps.DET_COLOR.get(a, ps.MUTED)
            ax.plot(g.log_chi2dof, g[col], color=c, lw=2.0)
            j = int(np.argmax(g[col].values))
            ax.annotate(f' {pair}', xy=(g.log_chi2dof.values[j],
                                        g[col].values[j]), color=c,
                        fontsize=10, fontweight='bold')
        _logx(ax, lo=0.0, hi=3.4)
        ps.strip(ax)
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    ps.title(axes[0], 'as the QA set drew it until 2026-09-12',
             'n / total / Δχ² — a geometric bin’s linear width')
    ps.title(axes[1], 'as it draws it now',
             'the low bump is a 5 % shoulder, not a second mode')
    for x, y in ((np.log10(2.0), 0),):
        axes[0].axvline(x, color=ps.COPPER, lw=1.0, ls=(0, (3, 2)))
        axes[1].axvline(x, color=ps.COPPER, lw=1.0, ls=(0, (3, 2)))
    fig.suptitle('Most of the published low bump was the normalisation',
                 x=0.007, ha='left', fontsize=15, fontweight='bold',
                 color=ps.INK)
    fig.supxlabel('worst χ²/dof of the four track fits in the pair',
                  fontsize=11, color=ps.INK)
    ps.note(fig, '`make_pair_qa_figures._hist` returned n/tot/np.diff(edges) on '
            'GEOMETRIC edges, so each bin was divided by a linear width that '
            'grows with χ²/dof and the left of the axis was lifted '
            'about 5× relative to the peak.  Only 5.3 % of A–A pairs '
            'are below χ²/dof = 5 (C–C 1.5 %, D–D 0.8 %), so '
            'the A-specific excess is real and worth the investigation — '
            'but it is a shoulder, and the figure rendered it as a co-equal '
            'mode.  All five log-scaled panels of that set carried the same '
            'lift.  FIXED 2026-09-12: `_hist` now returns the per-bin fraction, '
            'which is what its y-axis label always claimed; the right-hand panel '
            'is what the set draws today.', y=-0.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    if save:
        _save(fig, 'pair_link', t)
    return fig


# --------------------------------------------------------------------------- #
def _save(fig, name: str, table: pd.DataFrame | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    if table is not None:
        table.to_csv(OUT / f'{name}.csv', index=False)
    print(f'  -> figures/{name}.png (+.pdf, +.csv)')


def fig_axis_and_binning(save=True):
    """The low peak at 0.1 resolution, its context, and the log view beside it.

    The first version of this figure binned the linear axis at a width of 1,
    which put 36 % of chamber A in a single bin -- that bin WAS the peak, and
    the panel showed a spike with no shape.  At 0.1 the shape is there, and it
    says something the coarse view could not: A's peak sits ON chi2/dof = 1,
    not merely near it.

    Left and middle are the same curve at two resolutions and share their y
    units, because both are a DENSITY per unit chi2 -- which on a linear axis
    is simply the correct normalisation, and is exactly the thing that is wrong
    on a log axis (``pair_link``).  Right is the log view for the decades the
    linear axis cannot reach, per-bin fraction, correctly normalised.
    """
    t = _load('chi2_axis')
    m = _load('modality').set_index('arm')
    fp = _load('floor_peak').set_index('arm')
    arms = [a for a in ARMS if a in set(t.arm)]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1))

    # ---------------------------------------------------------- 1: the zoom
    ax = axes[0]
    g0 = t[t.view == 'lin_fine']
    for arm in arms:
        g = g0[g0.arm == arm].sort_values('chi2dof')
        ax.plot(g.chi2dof, g.dens, color=ps.DET_COLOR[arm], lw=2.0)
    ax.axvline(1.0, color=ps.MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    top = g0.dens.max() * 1.20
    ax.set_ylim(0, top)
    ax.set_xlim(0, 10)
    ax.text(1.08, top * 0.985, 'the noise floor,\nχ²/dof = 1',
            fontsize=9.5, color=ps.MUTED, va='top')
    # C's and D's peaks are 0.3 apart at nearly the same height, so an inline
    # label on each curve lands on the other one.  Only A is unambiguous
    # inline; the rest go in a keyed block that also carries the peak position,
    # which is the number this panel exists to show.
    ranked = sorted((a for a in arms if a in fp.index),
                    key=lambda a: -fp.loc[a].peak_dens)
    if ranked:
        r = fp.loc[ranked[0]]
        ax.annotate(ranked[0], xy=(r.peak_at + 0.3, r.peak_dens * 0.97),
                    color=ps.DET_COLOR[ranked[0]], fontsize=12,
                    fontweight='bold', va='top', ha='left')
    for i, arm in enumerate(ranked):
        r = fp.loc[arm]
        ax.text(0.97, 0.80 - 0.085 * i, f'{arm}   peak {r.peak_at:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10.5,
                color=ps.DET_COLOR[arm], fontweight='bold',
                family='monospace')
    ps.title(ax, 'linear, 0.1 wide', 'the low peak, resolved')
    ax.set_ylabel('fraction of tracks per unit χ²/dof')
    ps.strip(ax)

    # ------------------------------------------------------- 2: the context
    ax = axes[1]
    g0 = t[t.view == 'lin_full']
    for arm in arms:
        g = g0[g0.arm == arm].sort_values('chi2dof')
        ax.plot(g.chi2dof, g.dens, color=ps.DET_COLOR[arm], lw=1.8)
    ax.axvspan(0, 10, color=ps.LINE, alpha=0.55, zorder=0)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, top)
    ax.text(10.6, top * 0.90, 'the panel on the left', fontsize=9.5,
            color=ps.MUTED)
    off = g0.groupby('arm').frac_offaxis.first()
    ax.text(0.97, 0.62, 'beyond 60:\n' + '\n'.join(
        f'{a}  {off[a]:4.0%}' for a in arms if a in off),
        transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
        color=ps.MUTED, family='monospace')
    ps.title(ax, 'linear, same units', 'where the rest of the sample is')
    ax.set_xlabel('χ²/dof')
    ps.strip(ax)

    # ----------------------------------------------------------- 3: the log
    ax = axes[2]
    g0 = t[t.view == 'log']
    for arm in arms:
        g = g0[g0.arm == arm].sort_values('chi2dof')
        ax.plot(np.log10(g.chi2dof), g.frac, color=ps.DET_COLOR[arm], lw=1.9)
        j = int(np.argmax(g.frac.values))
        ax.annotate(f' {arm}', xy=(np.log10(g.chi2dof.values[j]),
                                   g.frac.values[j]),
                    color=ps.DET_COLOR[arm], fontsize=11.5, fontweight='bold',
                    va='bottom')
    _logx(ax)
    _floor_line(ax, label=False)
    ax.set_ylim(bottom=0)
    if 'A' in m.index:
        r = m.loc['A']
        ax.annotate(f'A’s dip,\n{r.low_mode / r.dip:.1f}× down',
                    xy=(np.log10(r.dip_at), r.dip),
                    xytext=(np.log10(r.dip_at) + 0.15, r.dip * 0.45),
                    fontsize=9.5, color=ps.COPPER, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-|>', color=ps.COPPER, lw=1.2,
                                    connectionstyle='arc3,rad=0.2'))
    ps.title(ax, 'log, fraction per bin',
             'the decades the linear axis cannot reach')
    ax.set_ylabel('fraction of tracks per bin')
    ps.strip(ax)

    fig.suptitle('The low peak sits on the noise floor — and on A it holds '
                 'a third of the sample', x=0.007, ha='left', fontsize=15,
                 fontweight='bold', color=ps.INK)
    pk = '   '.join(
        f'{a} peak {fp.loc[a].peak_at:.2f} (FWHM {fp.loc[a].fwhm_lo:.1f}–'
        f'{fp.loc[a].fwhm_hi:.1f}), {fp.loc[a].frac_1_to_2:.0%} in 1–2'
        for a in arms if a in fp.index)
    ps.note(fig, f'Single tracks, x view, the same numbers in all three panels.  '
            f'{pk}.  All three chambers DO have a peak at the noise floor — '
            'the difference is how much of the sample is in it and how tight it '
            'is: A’s is 3.6× taller than C’s and half the width.  '
            'On the linear axis the high mode never appears, because it is '
            'spread over decades; that is not a disagreement with the log '
            'panel, it is the two axes answering different questions.  Left and '
            'middle are a density per unit χ², which is the correct '
            'normalisation on a linear axis and the wrong one on a log axis.',
            y=-0.02)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    if save:
        _save(fig, 'axis_and_binning', t)
    return fig


FIGS = {
    'axis_and_binning': fig_axis_and_binning,
    'chi2_decomposition': fig_decomposition,
    'chi2_vs_length': fig_vs_length,
    'chi2_vs_charge': fig_vs_charge,
    'chi2_by_run': fig_by_run,
    'pair_link': fig_pair_link,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', choices=sorted(FIGS), help='one figure')
    a = ap.parse_args()
    ps.use()
    for name, fn in FIGS.items():
        if a.only and name != a.only:
            continue
        fn()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
