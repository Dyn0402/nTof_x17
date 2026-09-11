#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_accidental_timing_figures.py -- the four figures for the accidental-
timing study (HANDOFF_ACCIDENTAL_TIMING.md, followed through 2026-09-08).

  single_arm_classes   single-active-arm events: the wall's and the plastic's
                       own dt_ns, split by whether the OTHER element also fired
                       in the peak core -- does the coincidence requirement
                       actually buy purity?
  window_scan          peak/pedestal purity vs accept-window half-width, two
                       centres, with the production window and the
                       recommended one marked.
  two_arm_delta_t      THE test: arm1-arm2 scintillator time for the real
                       inter-chamber MM pairs, with the fitted prompt +
                       accidental components overlaid.
  f_by_topology        the fitted true-coincidence fraction, opposing vs
                       perpendicular vs all.

    python -m sept26_prelim_analysis.make_accidental_timing_figures --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import accidental_timing as AT  # noqa: E402


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


BOTH_COLOR = fs.ACCENT
SOLO_COLOR = fs.COPPER
FAM_NAME = {'WAL': 'wall', 'PSS': 'plastic'}


# ------------------------------------------------------------- single-arm
def fig_single_arm_classes(hits: pd.DataFrame, out):
    """Does requiring the coincidence buy a cleaner sample than one element?"""
    plt = _plt()
    edges = np.arange(-300, 301, 10.0)
    rows = []
    with _scaled(plt, 0.72):
        fig, axes = plt.subplots(1, 2, figsize=(fs.WIDE[0], 3.46),
                                 sharey=True, constrained_layout=True)
        for ax, fam, solo_cls in zip(axes, ('WAL', 'PSS'),
                                     ('wall_only', 'plastic_only')):
            g = hits[hits.family == fam]
            both = g[g.cls == 'both'].dt_ns.to_numpy()
            solo = g[g.cls == solo_cls].dt_ns.to_numpy()
            hb, _ = np.histogram(both, bins=edges)
            hs, _ = np.histogram(solo, bins=edges)
            mid = 0.5 * (edges[:-1] + edges[1:])
            ax.stairs(np.clip(hb, 1, None), edges, color=BOTH_COLOR, lw=2.2,
                      fill=True, alpha=0.35, baseline=1)
            ax.stairs(np.clip(hb, 1, None), edges, color=BOTH_COLOR, lw=2.2)
            ax.stairs(np.clip(hs, 1, None), edges, color=SOLO_COLOR, lw=2.0,
                      baseline=1)
            ax.axvspan(*AT.CORE_WINDOW, color=fs.MUTED, alpha=0.08, zorder=0)
            ax.set_yscale('log')
            ax.set_title(f'{FAM_NAME[fam]}', color=fs.INK, fontweight='600')
            ax.set_xlabel('dt$_{ns}$  [ns]  (relative to the DREAM trigger)')
            ax.set_xlim(-300, 300)
            for lo, hi, cls, n in ((edges[0], edges[-1], 'both', hb.sum()),
                                   (edges[0], edges[-1], solo_cls, hs.sum())):
                rows.append(dict(family=fam, cls=cls, n_shown=int(n)))
        axes[0].set_ylabel('hits / 10 ns')
        h1 = axes[0].plot([], [], color=BOTH_COLOR, lw=2.2,
                          label='both fire (coincidence)')[0]
        h2 = axes[0].plot([], [], color=SOLO_COLOR, lw=2.0,
                          label='this element only')[0]
        axes[0].legend(handles=[h1, h2], frameon=False, loc='upper left',
                       fontsize=fs.BASE_PT * 0.72)
        fig.suptitle('Single-arm events: a coincidence is a much sharper '
                     'timing sample than either element alone',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(axes[-1], loc='upper right')
        fs.note(fig, 'shaded band: peak core '
               f'({AT.CORE_WINDOW[0]:.0f}, {AT.CORE_WINDOW[1]:.0f}) ns used to '
               'classify "fired". y-axis clipped at 1 for the log scale; bars '
               'touching the floor are zero-count bins.')
        fs.save(fig, out / 'single_arm_classes', data=pd.DataFrame(rows))


# ------------------------------------------------------------- window scan
def fig_window_scan(scan: pd.DataFrame, rec: dict, out):
    plt = _plt()
    with _scaled(plt, 0.76):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.75, 3.46),
                               constrained_layout=True)
        for c, col, lab in ((-20, fs.MUTED, 'centre −20 ns (production)'),
                            (0, fs.ACCENT, 'centre 0 ns (peak core)')):
            g = scan[scan.center == c].sort_values('half_width')
            ax.plot(g.half_width, 100 * g.purity, 'o-', color=col, lw=2.2,
                    ms=6, label=lab)
        prod = scan[(scan.center == -20) & (scan.half_width == 80)].iloc[0]
        ax.plot([prod.half_width], [100 * prod.purity], 'X', ms=16,
               color=fs.BAND_DEAD, zorder=5, mec='white', mew=1.2)
        ax.annotate('production\n(−100, +60)', (prod.half_width, 100 * prod.purity),
                   textcoords='offset points', xytext=(10, -28),
                   fontsize=fs.BASE_PT * 0.68, color=fs.BAND_DEAD, ha='left')
        ax.plot([rec['half_width']], [100 * rec['purity']], '*', ms=20,
               color=fs.ACCENT, zorder=5, mec='white', mew=1.0)
        ax.annotate(f"recommended\n({rec['lo']:.0f}, {rec['hi']:.0f})",
                   (rec['half_width'], 100 * rec['purity']),
                   textcoords='offset points', xytext=(10, 10),
                   fontsize=fs.BASE_PT * 0.68, color=fs.ACCENT, ha='left')
        ax.set_xlabel('window half-width  [ns]')
        ax.set_ylabel('purity  [%]  (1 − pedestal / total)')
        ax.set_ylim(85, 100)
        ax.legend(frameon=False, loc='lower left', fontsize=fs.BASE_PT * 0.72)
        ax.set_title('The production window sits off-centre and wide: '
                     '88% purity where a centred one gets 95%+',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(ax, loc='upper right')
        fs.save(fig, out / 'window_scan', data=scan)


# ------------------------------------------------------------- two-arm dt
def fig_two_arm_delta_t(m: pd.DataFrame, fits: pd.DataFrame,
                        templates: dict, out):
    plt = _plt()
    edges = np.arange(-160, 161, 20.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    obs, _ = np.histogram(m.delta_t, bins=edges)
    f_row = fits[fits.topo == 'all'].iloc[0]
    ph, _ = np.histogram(templates['prompt'], bins=edges, density=True)
    ah, _ = np.histogram(templates['acc'], bins=edges, density=True)
    ph, ah = ph / ph.sum(), ah / ah.sum()
    pred_p = f_row.f_hat * ph * obs.sum()
    pred_a = (1 - f_row.f_hat) * ah * obs.sum()
    with _scaled(plt, 0.76):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.78, 3.74),
                               constrained_layout=True)
        ax.bar(mid, obs, width=18, color=fs.INK, alpha=0.85, zorder=3,
              label=f'observed, N={int(obs.sum())} pairs')
        ax.plot(mid, pred_p + pred_a, '-', color=fs.ACCENT, lw=2.4, zorder=5,
               label=f'fit: f = {f_row.f_hat:.2f} '
                     f'(+{f_row.f_hi - f_row.f_hat:.2f}/'
                     f'−{f_row.f_hat - f_row.f_lo:.2f}) prompt + '
                     f'{1 - f_row.f_hat:.2f} accidental')
        ax.plot(mid, pred_p, '--', color=fs.ACCENT, lw=1.8, alpha=0.8,
               label='prompt component alone')
        ax.plot(mid, pred_a, '--', color=fs.MUTED, lw=1.8, alpha=0.8,
               label='accidental component alone')
        ax.axvline(0, color=fs.MUTED, lw=1.0, ls=':', zorder=1)
        ax.set_xlabel(r'$\Delta t$ = t(arm 1) $-$ t(arm 2)  [ns]')
        ax.set_ylabel('pairs / 20 ns')
        ax.set_xlim(-160, 160)
        ax.legend(frameon=False, loc='upper right', fontsize=fs.BASE_PT * 0.67)
        ax.set_title('Real inter-chamber MM pairs: a spike at zero on top of '
                     'an accidental floor',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(ax, loc='upper left')
        fs.note(fig, 'prompt template: bootstrap difference of two draws from '
               'the single-arm reference. accidental template: one draw from '
               'the reference, one from is_control, both restricted to the '
               'accept window used to define a tag.')
        data = pd.DataFrame(dict(mid=mid, observed=obs, fit_prompt=pred_p,
                                 fit_accidental=pred_a))
        fs.save(fig, out / 'two_arm_delta_t', data=data)


# ------------------------------------------------------------- f by topology
def fig_f_by_topology(fits: pd.DataFrame, out):
    plt = _plt()
    g = fits[fits.f_hat.notna()].copy()
    order = ['opposing', 'perpendicular', 'all']
    g['order'] = g.topo.map({t: i for i, t in enumerate(order)})
    g = g.sort_values('order')
    with _scaled(plt, 0.78):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.55, 3.31),
                               constrained_layout=True)
        y = np.arange(len(g))
        colors = [fs.ACCENT if t == 'opposing' else
                 (fs.DET_COLOR['D'] if t == 'perpendicular' else fs.MUTED)
                 for t in g.topo]
        for i, (yi, r, c) in enumerate(zip(y, g.itertuples(), colors)):
            ax.plot([r.f_lo, r.f_hi], [yi, yi], '-', color=c, lw=3, zorder=2)
            ax.plot([r.f_hat], [yi], 'o', color=c, ms=11, zorder=3,
                    mec='white', mew=1.2)
            ax.annotate(f'{100 * r.f_hat:.0f}%  (n={int(r.n)})',
                       (r.f_hi, yi), textcoords='offset points',
                       xytext=(10, 0), va='center',
                       fontsize=fs.BASE_PT * 0.74, color=c)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace('opposing', 'opposing (A–C, signal region)')
                            .replace('perpendicular', 'perpendicular')
                            .replace('all', 'all inter-chamber')
                            for t in g.topo])
        ax.set_xlabel('fitted true-coincidence fraction  f')
        ax.set_xlim(0, 0.85)
        ax.set_ylim(-0.7, len(g) - 1 + 1.3)
        ax.axvline(0, color=fs.INK, lw=0.8)
        ax.set_title('The opposing (signal) topology carries more real '
                     'coincidence than perpendicular',
                     fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(ax, loc='upper right')
        fs.save(fig, out / 'f_by_topology', data=g.drop(columns='order'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    sd = paths.out('accidental_timing')
    od = paths.out('accidental_timing', 'figures')

    hits = pd.read_parquet(paths.require(sd / f'single_arm_hits_{a.run}.parquet',
                                         'the single-arm hit classes'))
    scan = pd.read_csv(paths.require(sd / f'window_scan_{a.run}.csv',
                                     'the window scan'))
    m = pd.read_parquet(paths.require(sd / f'two_arm_pairs_{a.run}.parquet',
                                      'the two-arm pairs'))
    fits = pd.read_csv(paths.require(sd / f'fit_by_topology_{a.run}.csv',
                                     'the fit'))
    meta = json.load(open(paths.require(sd / f'accidental_timing_{a.run}.meta.json',
                                        'the meta')))
    npz = np.load(sd / f'templates_{a.run}.npz')
    templates = dict(prompt=npz['prompt'], acc=npz['acc'])

    fig_single_arm_classes(hits, od)
    fig_window_scan(scan, meta['recommended_window'], od)
    fig_two_arm_delta_t(m, fits, templates, od)
    fig_f_by_topology(fits, od)
    print(f'wrote 4 figures (+ CSVs) to {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
