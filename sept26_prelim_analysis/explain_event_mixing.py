#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_event_mixing.py -- the figures for the event-mixing explainer page.

Every panel is measured from a product already on disk; nothing here is drawn
by hand and nothing is a cartoon (the two schematics live as inline SVG in the
HTML, where they belong).  Deliberately does NOT import `figstyle`: this page
is a fresh start on presentation, so its look is defined here and nowhere else.

    python explain_event_mixing.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = Path('/media/dylan/data/x17/sept26_prelim/event_mixing')
FIG = OUT / 'figures'
SRC = Path('/media/dylan/data/x17/sept26_prelim')
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = '#1e2530', '#6b7684', '#dfe3e8'
REAL, MIXED = '#1b3a6b', '#c86a1e'
ARMC = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}

plt.rcParams.update({
    'figure.figsize': (6.6, 4.2), 'figure.dpi': 160,
    'savefig.dpi': 160, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'font.size': 10.0, 'font.family': 'DejaVu Sans',
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'axes.titlesize': 10.5,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 8, 'axes.labelpad': 5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'legend.fontsize': 9.0,
    'text.color': INK,
})


def note(fig, s):
    fig.text(0.0, -0.035, s, ha='left', va='top', fontsize=7.6, color=MUTED)


def save(fig, name, data=None):
    p = FIG / f'{name}.png'
    fig.savefig(p)
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / f'{name}.csv', index=False)
    print(f'  {p.name}')
    return p


# --------------------------------------------------------------------------- #
RESULTS: dict = {}


def f1_construction():
    """Pairs per arm combination: real and mixed, by construction identical."""
    P = pd.read_parquet(SRC / 'angle_campaign' / 'pairs.parquet')
    g = (P.assign(combo=P.arm1 + '-' + P.arm2)
           .groupby(['combo', 'mixed']).size().unstack(fill_value=0))
    g.columns = ['real', 'mixed']
    g = g.sort_values('real', ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.real, 0.4, color=REAL, label='real pairs (one trigger)')
    ax.bar(x + 0.2, g['mixed'], 0.4, color=MIXED,
           label='event-mixed pairs (two triggers)')
    for i, (r, m) in enumerate(zip(g.real, g['mixed'])):
        ax.text(i, max(r, m) * 1.04, f'{r:,}', ha='center', va='bottom',
                fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_xlabel('chamber combination')
    ax.set_ylabel('pairs')
    ax.set_ylim(0, g.values.max() * 1.22)
    ax.set_title('The mixed sample is built pair for pair with the data',
                 loc='left')
    ax.legend(loc='upper right', ncol=1)
    note(fig, 'sept26_prelim/angle_campaign/pairs.parquet — campaign pass, '
              'all runs pooled. Counts agree exactly in every combination '
              'because _pairs_mixed draws one mixed pair per real pair.')
    RESULTS['n_real_pairs'] = int(g.real.sum())
    RESULTS['combos'] = {k: int(v) for k, v in g.real.items()}
    RESULTS['identical'] = bool((g.real == g['mixed']).all())
    return save(fig, 'f1_construction', g.reset_index())


def f2_spectra():
    """Opening angle: observed against the mixed null, per topology."""
    S = pd.read_csv(SRC / 'angle_campaign' / 'spectra.csv')
    topos = ['intra', 'perpendicular', 'opposing']
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=False)
    rows = []
    for ax, topo in zip(axes, topos):
        r = S[(S.topology == topo) & (S.selection == 'all_no_b2b')].sort_values('theta')
        m = S[(S.topology == topo) & (S.selection == 'mixed')].sort_values('theta')
        nr, nm = r.n.sum(), m.n.sum()
        fr, fm = r.n / nr, m.n * (nr / nm) / nr
        w = r.hi - r.lo
        ax.bar(r.theta, fr, width=w * 0.92, color=REAL, alpha=0.85,
               label='observed')
        ax.step(np.r_[m.lo.values, m.hi.values[-1]], np.r_[fm.values, fm.values[-1]],
                where='post', color=MIXED, lw=2.0, label='event-mixed null')
        ax.axvline(109, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel('opening angle  [deg]')
        ax.set_title(f'{topo}   (n = {nr:,})', loc='left')
        rows.append(dict(topology=topo, n_obs=int(nr), n_mixed=int(nm)))
    axes[0].set_ylabel('fraction of pairs / bin')
    axes[0].legend(loc='upper right')
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axes[2].text(107, axes[2].get_ylim()[1] * 0.45, '109 deg, X17 minimum  ',
                 fontsize=8, color=MUTED, va='center', ha='right', rotation=90)
    fig.suptitle('The observed spectrum and its accidental null are the same '
                 'distribution', x=0.0, ha='left', fontweight='bold', y=1.03)
    note(fig, 'sept26_prelim/angle_campaign/spectra.csv — mixed scaled to the '
              'observed total (it is a shape, not a rate). '
              'Back-to-back pairs removed from the observed opposing sample.')
    return save(fig, 'f2_spectra', pd.DataFrame(rows))


def f3_chi2():
    """chi2/dof of every candidate shape, mixed included, per topology."""
    C = pd.read_csv(SRC / 'angle_campaign' / 'compare.csv')
    C = C[C.selection == 'all_no_b2b']
    topos = ['intra', 'perpendicular', 'opposing']
    models = ['IPC M1 only', 'IPC E0 only', 'IPC thermal (M1+E0)',
              'X17 (17 MeV boson)', 'event-mixed (accidental shape)']
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    w = 0.16
    for k, mdl in enumerate(models):
        vals, xs = [], []
        for i, t in enumerate(topos):
            g = C[(C.topology == t) & (C.model == mdl)]
            if len(g):
                vals.append(float(g.chi2dof.iloc[0]))
                xs.append(i + (k - 2) * w)
        colour = MIXED if 'mixed' in mdl else ('#7d8896' if 'X17' in mdl else REAL)
        alpha = 1.0 if 'mixed' in mdl else (0.45 + 0.18 * k)
        ax.bar(xs, vals, w, color=colour, alpha=min(alpha, 1.0), label=mdl)
    ax.set_yscale('log')
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels(topos)
    ax.set_ylabel(r'$\chi^2$ / dof   (shape only, log scale)')
    # the headline factor is read off the table, never typed in
    lo, hi = [], []
    for t in topos:
        g = C[C.topology == t]
        mx = float(g[g.model.str.startswith('event-mixed')].chi2dof.iloc[0])
        bm = float(g[~g.model.str.startswith('event-mixed')].chi2dof.min())
        lo.append(bm / mx)
    ax.set_title(f'The accidental shape fits the data {min(lo):.0f}–{max(lo):.0f}× '
                 'better than any pair spectrum', loc='left')
    ax.set_ylim(top=ax.get_ylim()[1] * 14)
    ax.legend(loc='upper left', ncol=2, fontsize=8.4)
    note(fig, 'sept26_prelim/angle_campaign/compare.csv, selection all_no_b2b. '
              'Every shape is normalised to the observed total, so this is a '
              'shape test with no free parameter.')
    best = (C.sort_values('chi2dof').groupby('topology').first()
             .reset_index()[['topology', 'model', 'chi2dof']])
    RESULTS['best_shape'] = best.to_dict('records')
    return save(fig, 'f3_chi2', C[['topology', 'model', 'chi2', 'dof', 'chi2dof']])


def f4_tight():
    """What mixing is FOR: the timing-selected subsample against the same null."""
    S = pd.read_csv(SRC / 'angle_campaign' / 'spectra.csv')
    C = pd.read_csv(SRC / 'angle_campaign' / 'compare.csv')
    topos = ['perpendicular', 'opposing']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, topo in zip(axes, topos):
        a = S[(S.topology == topo) & (S.selection == 'all_no_b2b')].sort_values('theta')
        t = S[(S.topology == topo) & (S.selection == 'tight_pair')].sort_values('theta')
        m = S[(S.topology == topo) & (S.selection == 'mixed')].sort_values('theta')
        na, nt, nm = a.n.sum(), t.n.sum(), m.n.sum()
        ax.step(np.r_[m.lo.values, 180], np.r_[m.n.values, m.n.values[-1]] / nm,
                where='post', color=MIXED, lw=2.0, label='event-mixed null')
        ax.step(np.r_[a.lo.values, 180], np.r_[a.n.values, a.n.values[-1]] / na,
                where='post', color=MUTED, lw=1.3, ls=(0, (4, 2)),
                label=f'all pairs (n = {na:,})')
        e = np.sqrt(t.n.values) / nt
        ax.errorbar(t.theta, t.n / nt, yerr=e, fmt='o', ms=4.2, lw=1.2,
                    color=REAL, label=f'prompt-coincident (n = {nt:,})')
        cc = C[(C.selection == 'tight_pair') & (C.topology == topo) &
               (C.model == 'event-mixed (accidental shape)')]
        if len(cc):
            ax.text(0.03, 0.62, r'$\chi^2$/dof vs null = '
                    f'{float(cc.chi2dof.iloc[0]):.1f}', transform=ax.transAxes,
                    va='top', fontsize=8.8, color=INK, fontweight='bold')
        ax.axvline(109, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel('opening angle  [deg]')
        ax.set_title(topo, loc='left')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    axes[1].axvspan(170, 180, color='#f0dede', zorder=0)
    axes[0].set_ylabel('fraction of pairs / bin')
    axes[0].legend(loc='upper left', fontsize=8.0)
    fig.suptitle('Cut to genuinely coincident pairs and the data starts to '
                 'leave the null', x=0.0, ha='left', fontweight='bold', y=1.03)
    note(fig, 'sept26_prelim/angle_campaign/{spectra,compare}.csv — tight_pair '
              'is |t_tag| <= 30 ns per arm AND |t1 - t2| <= 20 ns '
              '(tight_coincidence.py).')
    RESULTS['tight'] = C[(C.selection == 'tight_pair') &
                         (C.model == 'event-mixed (accidental shape)')][
        ['topology', 'n_obs', 'chi2dof']].to_dict('records')
    return save(fig, 'f4_tight')


def f5_vertex():
    """The vertex null, in the two places it has been run."""
    r = pd.read_parquet(SRC / 'imaging' / 'vertices_run_145.parquet')
    m = pd.read_parquet(SRC / 'imaging' / 'vertices_mixed_run_145.parquet')
    VS = pd.read_csv(SRC / 'imaging' / 'vertex_summary_run_145.csv')
    VX = pd.read_csv(SRC / 'det_a_intra' / 'vertex_excess.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))
    b = np.logspace(0, 3.6, 34)
    ax1.hist(r.v_r.clip(1, 4000), bins=b, color=REAL, alpha=0.8,
             label=f'real (n = {len(r):,})')
    ax1.hist(m.v_r.clip(1, 4000), bins=b, histtype='step', color=MIXED, lw=2.0,
             label=f'mixed (n = {len(m):,})')
    ax1.axvline(20, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.30)
    ax1.annotate('capsule bore\nr < 20 mm', xy=(20, ax1.get_ylim()[1] * 0.52),
                 xytext=(1.6, ax1.get_ylim()[1] * 0.60), fontsize=8,
                 color=MUTED, ha='left', va='center',
                 arrowprops=dict(arrowstyle='->', color='#b9c0c9', lw=0.9))
    ax1.set_xscale('log')
    ax1.set_xlabel('vertex distance from the beam axis  [mm]')
    ax1.set_ylabel('pairs / bin')
    ax1.set_title('Inter-chamber pairs, run_145: null not beaten', loc='left')
    ax1.legend(loc='upper right', fontsize=8.4)

    lab, ratio, err = [], [], []
    for _, row in VS.iterrows():
        if np.isfinite(row.get('lift', np.nan)):
            lab.append(f"run_145 {row.topology}\n(n = {int(row.n_real)})")
            ratio.append(float(row.lift))
            err.append(float(row.err) / float(row.frac_mixed))
    for _, row in VX.iterrows():
        lab.append(f"det A intra, {row.selection}\n(n = {int(row.n_real):,})")
        ratio.append(float(row.ratio))
        err.append(float(row.err))
    y = np.arange(len(lab))[::-1]
    cols = [MIXED if v - e < 1.0 else '#1e7a4d' for v, e in zip(ratio, err)]
    ax2.errorbar(ratio, y, xerr=err, fmt='o', ms=5, lw=1.4,
                 ecolor=MUTED, mfc='none', mec='none')
    ax2.scatter(ratio, y, s=44, c=cols, zorder=3)
    ax2.axvline(1.0, color=INK, lw=1.1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(lab, fontsize=8.2)
    ax2.set_xlabel('vertex rate,  real / mixed')
    ax2.set_xlim(0, max(np.array(ratio) + np.array(err)) * 1.2)
    ax2.grid(axis='y', visible=False)
    ax2.set_title('Lift over the null', loc='left')
    fig.subplots_adjust(wspace=0.62)
    note(fig, 'sept26_prelim/imaging/vertices_*_run_145.parquet and '
              'det_a_intra/vertex_excess.csv — a "vertex" is r < 20 mm and '
              'line-to-line approach < 30 mm in both.')
    RESULTS['vertex'] = dict(run145=VS.to_dict('records'),
                             det_a=VX.to_dict('records'))
    return save(fig, 'f5_vertex')


def f6_timing():
    """Why mixing is illegitimate on the trigger-referenced timing variable."""
    b = SRC / 'accidental_timing'
    d = pd.read_parquet(b / 'two_arm_pairs_run_145.parquet')
    s = pd.read_parquet(b / 'single_arm_hits_run_145.parquet')
    rng = np.random.default_rng(0)

    def mix(pool, n=400_000):
        ev = (pool.subrun + ':' + pool.eventId.astype(str)).to_numpy()
        t = pool.dt_ns.to_numpy()
        i = rng.integers(0, len(pool), n)
        j = rng.integers(0, len(pool), n)
        ok = ev[i] != ev[j]
        return t[i][ok] - t[j][ok]

    s = s.assign(a=s.dt_ns.abs())
    tagged = s.loc[s.groupby(['subrun', 'eventId', 'arm']).a.idxmin()]
    mix_all, mix_tag = mix(s), mix(tagged)
    real = d.delta_t.to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bins = np.arange(-500, 501, 20)
    for v, c, ls, lab in (
            (real, REAL, '-', f'real two-arm pairs (n = {len(real)})'),
            (mix_all, MIXED, '-', 'mixed, drawn from every hit'),
            (mix_tag, '#8a3f8f', (0, (4, 2)), 'mixed, drawn from tagging hits')):
        h, _ = np.histogram(v, bins=bins)
        ax.step(bins[:-1], h / h.sum(), where='post', color=c, lw=1.9, ls=ls,
                label=lab)
    ax.axvspan(-20, 20, color=GRID, alpha=0.7, zorder=0)
    ax.set_xlabel(r'$t_1 - t_2$  between the two arms  [ns]')
    ax.set_ylabel('fraction / 20 ns')
    ax.set_title('On a trigger-referenced clock, the null has no fixed answer',
                 loc='left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(loc='upper right', fontsize=8.4)
    rows = []
    for v, lab in ((real, 'real'), (mix_all, 'mixed / all hits'),
                   (mix_tag, 'mixed / tagging hits')):
        f = float((np.abs(v) < 20).mean())
        rows.append(dict(sample=lab, n=len(v), frac_within_20ns=round(f, 4),
                         median_abs_ns=round(float(np.median(np.abs(v))), 1)))
        print(f'    {lab:24s} |dt|<20 ns: {100*f:5.1f} %')
    ax.text(0.02, 0.95,
            '\n'.join(f'{r["sample"]}: {100*r["frac_within_20ns"]:.0f} % '
                      'within 20 ns' for r in rows),
            transform=ax.transAxes, va='top', fontsize=8.4, color=INK)
    note(fig, 'sept26_prelim/accidental_timing/*_run_145.parquet — dt_ns is '
              'measured from each event\'s OWN trigger, so the mixed answer is '
              'set by which hits enter the pool, not by physics.')
    RESULTS['timing'] = rows
    return save(fig, 'f6_timing', pd.DataFrame(rows))


def main():
    print('figures ->', FIG)
    f1_construction()
    f2_spectra()
    f3_chi2()
    f4_tight()
    f5_vertex()
    f6_timing()
    (OUT / 'numbers.json').write_text(json.dumps(RESULTS, indent=2, default=str))
    print('numbers ->', OUT / 'numbers.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
