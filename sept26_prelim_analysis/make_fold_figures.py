#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fold_figures.py -- figures for the per-run acceptance and the capsule fold.

Reads what `campaign_efficiency.py`, `campaign_acceptance.py` and
`campaign_fold.py` wrote; computes nothing new.  Every figure ships its numbers
beside it as a CSV, as PLAN sec 7 requires.

    python -m sept26_prelim_analysis.make_fold_figures
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

from sept26_prelim_analysis import figstyle, paths  # noqa: E402
from sept26_prelim_analysis.campaign_angle import (  # noqa: E402
    TOPOLOGIES, X17_MIN_DEG)

TOPO_COLOR = {'intra': '#0072B2', 'perpendicular': '#E69F00',
              'opposing': '#009E73'}
ARM_COLOR = {'A': '#0072B2', 'B': '#999999', 'C': '#009E73', 'D': '#D55E00'}
VAR_COLOR = {'flat': '#999999', 'u_map': '#0072B2', 'incidence': '#D55E00'}
MODEL_COLOR = {
    'Al capsule (after wall)': '#CC79A7',
    'Al capsule (birth)': '#E7A2C4',
    '3He gas M1+E0': '#0072B2',
    '3He gas M1+E0 (after wall)': '#56B4E9',
    'X17 (17 MeV boson)': '#D55E00',
    'event-mixed (accidental shape)': '#666666',
}


# --------------------------------------------------------------------------- #
# the efficiency, run by run
# --------------------------------------------------------------------------- #
def fig_efficiency(ed: Path, fd: Path):
    """Headline efficiency against run number, per arm, with run_145 marked."""
    import matplotlib.pyplot as plt
    H = pd.read_csv(ed / 'headline_per_run.csv')
    H['num'] = H.run.str.split('_').str[1].astype(int)
    fig, ax = figstyle.figure(figstyle.WIDE)
    for arm in ('A', 'C', 'D', 'B'):
        g = H[H.arm == arm].sort_values('num')
        if g.empty:
            continue
        ax.plot(g.num, 100 * g.efficiency, 'o-', ms=3.5, lw=1.0,
                color=ARM_COLOR[arm], label=f'{arm}'
                + (' (hits)' if arm == 'B' else ''))
    ax.axvspan(128, 147, color='#000000', alpha=0.05, lw=0)
    ax.text(137.5, ax.get_ylim()[1], 'k excursion 128–147', ha='center',
            va='top', fontsize=figstyle.BASE_PT * 0.98, color='#666666')
    ax.axvline(145, color='#D55E00', lw=0.8, ls=':')
    ax.set_xlabel('run number')
    ax.set_ylabel('efficiency [%]')
    figstyle.fig_title(fig, 'The scintillator-tagged efficiency barely moves',
                   'P(gated track | wall AND plastic in the same arm), '
                   'accidental-corrected, per run')
    ax.legend(ncol=4, loc='lower left')
    return figstyle.save(fig, fd / 'eff_per_run',
                         H[['run', 'num', 'arm', 'basis', 'efficiency', 'err',
                            'p0', 'n_tagged', 'condition', 'k_block']])


def fig_incidence(ed: Path, fd: Path):
    """The head-on dip: tracking rate against the wall group's incidence."""
    import matplotlib.pyplot as plt
    I = pd.read_csv(ed / 'incidence_per_run.csv')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    ax = axes[0]
    for arm in ('A', 'C', 'D', 'B'):
        g = I[I.arm == arm]
        if g.empty:
            continue
        m = g.groupby('tan_expected').p_tracked_rel.agg(['median', 'std'])
        ax.errorbar(m.index, m['median'], yerr=m['std'].fillna(0), fmt='o-',
                    color=ARM_COLOR[arm], label=arm, capsize=3)
    ax.axvspan(-0.08, 0.08, color='#D55E00', alpha=0.10, lw=0)
    ax.text(0.0, 1.02, 'head-on', ha='center',
            fontsize=figstyle.BASE_PT * 0.98, color='#D55E00')
    ax.set_xlabel('expected tan(incidence) at the wall group')
    ax.set_ylabel('tracking rate / arm best')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.14 * (hi - lo), hi)   # room for the legend below the data
    ax.legend(ncol=4, loc='lower left', fontsize=figstyle.BASE_PT * 0.98)

    D = pd.read_csv(ed / 'head_on_dip_per_run.csv')
    D['num'] = D.run.str.split('_').str[1].astype(int)
    ax = axes[1]
    for arm in ('A', 'C', 'D', 'B'):
        g = D[D.arm == arm].sort_values('num')
        if g.empty:
            continue
        ax.plot(g.num, g.track_ratio, 'o-', ms=6, lw=1.4,
                color=ARM_COLOR[arm], label=arm)
    ax.axhline(1.0, color='#333333', lw=1.2)
    ax.set_xlabel('run number')
    ax.set_ylabel('head-on / neighbours')
    figstyle.fig_title(fig, 'Head-on tracks are reconstructed less often, in '
                            'every run',
                   'the abscissa is the scintillators, so it never touches the '
                   'Micromegas')
    return figstyle.save(fig, fd / 'eff_incidence',
                         {'curve': I, 'dip': D})


# --------------------------------------------------------------------------- #
# the acceptance
# --------------------------------------------------------------------------- #
def fig_acceptance_runs(ad: Path, fd: Path, variant: str = 'incidence'):
    """Every run's A(theta) per topology, and the pair-weighted mean."""
    import matplotlib.pyplot as plt
    C = pd.read_csv(ad / 'acceptance_per_run.csv')
    P = pd.read_csv(ad / 'acceptance_pooled.csv')
    C = C[(C.variant == variant) & (C.vertex == 'gas')]
    P = P[P.variant == variant]
    fig, axes = figstyle.figure(figstyle.FULL, nrows=3, sharex=True)
    for ax, topo in zip(axes, TOPOLOGIES):
        g = C[C.group == topo]
        for run, h in g.groupby('run'):
            h = h.sort_values('theta')
            ax.plot(h.theta, 1e4 * h.acc, lw=0.8, color='#BBBBBB', alpha=0.7)
        p = P[P.group == topo].sort_values('theta')
        ax.plot(p.theta, 1e4 * p.acc, lw=3.0, color=TOPO_COLOR[topo])
        ax.axvline(X17_MIN_DEG, color='#333333', lw=1.2, ls='--')
        ax.set_title(topo, loc='left', color=TOPO_COLOR[topo],
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        ax.set_ylabel(r'A [$10^{-4}$]')
    axes[-1].set_xlabel('opening angle [deg]')
    figstyle.fig_title(fig, 'The acceptance is the same in every run',
                   f'grey: one run each; colour: pair-weighted campaign mean '
                   f'({variant})')
    return figstyle.save(fig, fd / 'acc_runs', {'per_run': C, 'pooled': P})


def fig_variants(ad: Path, fd: Path):
    """The three efficiency variants, and how much the shape moves."""
    import matplotlib.pyplot as plt
    P = pd.read_csv(ad / 'acceptance_pooled.csv')
    fig, axes = figstyle.figure(figstyle.FULL, nrows=3, sharex=True)
    for ax, topo in zip(axes, TOPOLOGIES):
        ref = P[(P.group == topo) & (P.variant == 'u_map')].sort_values('theta')
        for v in ('flat', 'u_map', 'incidence'):
            g = P[(P.group == topo) & (P.variant == v)].sort_values('theta')
            if g.empty or g.acc.sum() <= 0:
                continue
            ax.plot(g.theta, g.acc / g.acc.sum(), lw=2.4,
                    color=VAR_COLOR[v], label=v)
        ax.axvline(X17_MIN_DEG, color='#333333', lw=1.2, ls='--')
        ax.set_title(topo, loc='left', color=TOPO_COLOR[topo],
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        ax.set_ylabel('shape')
    axes[-1].set_xlabel('opening angle [deg]')
    axes[0].legend(loc='upper right', ncol=3,
                   fontsize=figstyle.BASE_PT * 0.98)
    figstyle.fig_title(fig, 'Applying the efficiency in incidence rather than '
                            'position tilts the acceptance',
                   'one measured efficiency, three ways into the toy; never a '
                   'product of two of them')
    return figstyle.save(fig, fd / 'acc_variants', P)


# --------------------------------------------------------------------------- #
# the fold
# --------------------------------------------------------------------------- #
def fig_shapes(fdir: Path, fd: Path):
    """The birth spectra, before any acceptance -- what the physics says."""
    import matplotlib.pyplot as plt
    R = pd.read_csv(fdir / 'shape_cache.csv')
    fig, ax = figstyle.figure(figstyle.WIDE)
    pairs = [('capsule_birth', 'Al capsule (birth)', ':'),
             ('capsule_after_wall', 'Al capsule (after wall)', '-'),
             ('he3_birth', '3He gas M1+E0', ':'),
             ('he3_after_wall', '3He gas M1+E0 (after wall)', '-')]
    for col, name, ls in pairs:
        if col not in R.columns:
            continue
        ax.plot(R.theta_mid, R[col], ls=ls,
                color=MODEL_COLOR.get(name, '#333333'), label=name)
    ax.axvline(X17_MIN_DEG, color='#333333', lw=1.2, ls='--')
    ax.text(X17_MIN_DEG + 3, ax.get_ylim()[1] * 0.4, 'X17 threshold',
            fontsize=figstyle.BASE_PT * 0.98, color='#333333')
    ax.set_yscale('log')
    ax.set_xlabel('opening angle [deg]')
    ax.set_ylabel(r'd$N$/d$\theta$ [normalised]')
    ax.legend(fontsize=figstyle.BASE_PT * 0.98)
    figstyle.fig_title(fig, 'The capsule pair is wider than the gas pair, before '
                       'any apparatus',
                   'and the wall it is born in widens it further; both are '
                   'Born multipole calculations')
    return figstyle.save(fig, fd / 'fold_shapes', R)


def fig_folded(fdir: Path, fd: Path, selection: str = 'tight_pair'):
    """The measured spectrum against each folded model, per topology."""
    import matplotlib.pyplot as plt
    F = pd.read_csv(fdir / 'folded.csv')
    S = pd.read_csv(paths.out('angle_campaign') / 'spectra.csv')
    show = ['Al capsule (after wall)', '3He gas M1+E0', 'X17 (17 MeV boson)']
    fig, axes = figstyle.figure(figstyle.FULL, nrows=3, sharex=True)
    obs_out = []
    for ax, topo in zip(axes, TOPOLOGIES):
        used = selection
        o = S[(S.topology == topo) & (S.selection == selection)] \
            .sort_values('theta')
        if o.empty:
            # intra has no timing cut and cannot have one: one arm, no t1 - t2
            used = 'all_no_b2b'
            o = S[(S.topology == topo) & (S.selection == used)] \
                .sort_values('theta')
        n = o.n.to_numpy(float)
        if n.sum() <= 0:
            continue
        ax.errorbar(o.theta, n / n.sum(), yerr=np.sqrt(np.clip(n, 1, None))
                    / n.sum(), fmt='o', color='#333333', capsize=3, zorder=5)
        obs_out.append(o.assign(frac=n / n.sum(), selection_used=used))
        mx = S[(S.topology == topo) & (S.selection == 'mixed')] \
            .sort_values('theta').n.to_numpy(float)
        if mx.sum() > 0:
            ax.step(o.theta, mx / mx.sum(), where='mid', lw=2.0,
                    color=MODEL_COLOR['event-mixed (accidental shape)'],
                    label='event-mixed')
        for m in show:
            g = F[(F.topology == topo) & (F.model == m)].sort_values('theta')
            if g.empty or g.frac.sum() <= 0:
                continue
            ax.step(g.theta, g.frac, where='mid', lw=2.2,
                    color=MODEL_COLOR.get(m, '#333333'), label=m)
        ax.axvline(X17_MIN_DEG, color='#333333', lw=1.2, ls='--')
        ax.set_title(f'{topo}   n = {int(n.sum()):,}   ({used})', loc='left',
                     color=TOPO_COLOR[topo],
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        ax.set_ylabel('fraction')
        hi = max([float(o.frac.max() if 'frac' in o else (n / n.sum()).max())]
                 + [float(F[(F.topology == topo) & (F.model == m)].frac.max())
                    for m in show
                    if len(F[(F.topology == topo) & (F.model == m)])]
                 + ([float((mx / mx.sum()).max())] if mx.sum() > 0 else []))
        ax.set_ylim(0, hi * 1.15)
    axes[-1].set_xlabel('opening angle [deg]')
    # one legend for the whole figure, built from proxies: a per-axes legend
    # would only list the models that happen to be non-zero in that topology
    from matplotlib.lines import Line2D
    proxies = [Line2D([], [], color='#333333', marker='o', ls='none',
                      label='data')]
    proxies += [Line2D([], [], color=MODEL_COLOR['event-mixed (accidental '
                                                 'shape)'], lw=2.0,
                       label='event-mixed')]
    proxies += [Line2D([], [], color=MODEL_COLOR.get(m, '#333333'), lw=2.2,
                       label=m) for m in show]
    top = figstyle.fig_title(
        fig, 'The folded capsule continuum against the coincident sample',
        f'selection {selection}; every model normalised to the observed count')
    # placed AFTER the title so it can be anchored under the space the title
    # reserved, instead of on top of it
    fig.legend(handles=proxies, loc='upper right', ncol=2, frameon=False,
               fontsize=figstyle.BASE_PT * 0.89,
               bbox_to_anchor=(0.99, top - 0.004))
    return figstyle.save(fig, fd / 'fold_models',
                         {'folded': F, 'obs': (pd.concat(obs_out)
                                               if obs_out else F.head(0))})


def fig_two_comp(fdir: Path, fd: Path):
    """The accidental share the angles want, against the one timing measured."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    T = pd.read_csv(fdir / 'two_component.csv')
    from sept26_prelim_analysis.campaign_fold import F_TIMING
    T = T[T.topology.isin(F_TIMING)]
    if T.empty:
        raise ValueError('no rows with an independent timing measurement')
    fig, ax = figstyle.figure(figstyle.WIDE)
    # One row per (selection, topology); the pair model is the colour, so the
    # row label stays one line and the timing band is drawn once per row
    # instead of once per model.
    rows = list(dict.fromkeys(zip(T.selection, T.topology)))
    models = list(dict.fromkeys(T.pair_model))
    off = np.linspace(-0.16, 0.16, max(len(models), 1))
    for y, (sel, topo) in enumerate(rows):
        t = F_TIMING[topo]
        ax.errorbar(t[0], y, xerr=[[t[0] - t[1]], [t[2] - t[0]]], fmt='s',
                    capsize=5, color='#666666', zorder=2)
        ax.axhspan(y - 0.42, y + 0.42, color='#000000',
                   alpha=0.03 if y % 2 else 0.0, lw=0)
        for k, m in enumerate(models):
            g = T[(T.selection == sel) & (T.topology == topo)
                  & (T.pair_model == m)]
            if g.empty:
                continue
            r = g.iloc[0]
            ax.errorbar(r.f_acc, y + off[k],
                        xerr=[[r.f_acc - r.f_lo], [r.f_hi - r.f_acc]],
                        fmt='o', capsize=4, zorder=3,
                        color=MODEL_COLOR.get(m, '#333333'))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{topo}, {sel}' for sel, topo in rows],
                       fontsize=figstyle.BASE_PT * 0.98)
    # an empty band above the top row, so the legend never lands on a point
    ax.set_ylim(-0.6, len(rows) + 0.35)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel('accidental share of the sample')
    handles = [Line2D([], [], color='#666666', marker='s', ls='none',
                      label='measured, arm-to-arm timing')]
    handles += [Line2D([], [], color=MODEL_COLOR.get(m, '#333333'),
                       marker='o', ls='none', label=f'fitted with {m}')
                for m in models]
    ax.legend(handles=handles, fontsize=figstyle.BASE_PT * 0.86,
              loc='upper left')
    figstyle.fig_title(fig, 'The angles want more accidentals than the timing '
                            'found',
                       'circles: fitted from the opening-angle shape; '
                       'squares: measured from the arm-to-arm scintillator '
                       'timing')
    return figstyle.save(fig, fd / 'fold_two_comp', T)


def fig_corrected(fdir: Path, fd: Path, selection: str = 'all_no_b2b'):
    """The data divided by the acceptance, against the two birth spectra."""
    import matplotlib.pyplot as plt
    U = pd.read_csv(fdir / 'corrected.csv')
    R = pd.read_csv(fdir / 'shape_cache.csv')
    U = U[(U.selection == selection) & U.theta.notna() & U.live]
    fig, ax = figstyle.figure(figstyle.WIDE)
    for topo in TOPOLOGIES:
        g = U[U.topology == topo].sort_values('theta')
        if g.empty or g.corrected.sum() <= 0:
            continue
        w = 15.0
        y = g.corrected / (g.corrected.sum() * w)
        ax.errorbar(g.theta, y, yerr=g.err / (g.corrected.sum() * w),
                    fmt='o', capsize=3, color=TOPO_COLOR[topo], label=topo)
    for col, name in (('capsule_after_wall', 'Al capsule (after wall)'),
                      ('he3_after_wall', '3He gas M1+E0 (after wall)')):
        if col in R.columns:
            ax.plot(R.theta_mid, R[col], lw=2.4,
                    color=MODEL_COLOR.get(name, '#333333'), label=name)
    ax.axvline(X17_MIN_DEG, color='#333333', lw=1.2, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('opening angle [deg]')
    ax.set_ylabel(r'd$N$/d$\theta$ [normalised]')
    ax.legend(fontsize=figstyle.BASE_PT * 0.89, ncol=2)
    figstyle.fig_title(fig, 'Divided by the acceptance, each topology recovers a '
                       'different spectrum',
                   'which is what an acceptance that is wrong, or a sample '
                   'that is not pairs, looks like')
    return figstyle.save(fig, fd / 'fold_corrected',
                         {'corrected': U, 'birth': R})


def fig_distortion(fdir: Path, fd: Path):
    """Birth median against folded median, per model and topology."""
    import matplotlib.pyplot as plt
    D = pd.read_csv(fdir / 'distortion.csv')
    fig, ax = figstyle.figure(figstyle.WIDE)
    marks = {'intra': 'o', 'perpendicular': 's', 'opposing': '^', 'all': 'D'}
    for r in D.itertuples():
        if r.topology not in marks:
            continue
        ax.plot(r.birth_median_deg, r.folded_median_deg, marks[r.topology],
                ms=11, color=MODEL_COLOR.get(r.model, '#333333'),
                mfc='none' if r.topology != 'all' else None, mew=2.0)
    lim = [0, 180]
    ax.plot(lim, lim, lw=1.2, color='#333333', ls='--')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 180)
    ax.set_xlabel('median at birth [deg]')
    ax.set_ylabel('median after the acceptance [deg]')
    for m, c in MODEL_COLOR.items():
        if m in set(D.model):
            ax.plot([], [], 'o', color=c, label=m, ms=8)
    for t, mk in marks.items():
        ax.plot([], [], mk, color='#333333', mfc='none', label=t, ms=8)
    ax.legend(ncol=2, fontsize=figstyle.BASE_PT * 0.86, loc='upper left')
    figstyle.fig_title(fig, 'The apparatus, not the physics, sets the observed '
                       'median',
                   'every model is dragged to its topology’s own band; the '
                   'dashed line is no distortion')
    return figstyle.save(fig, fd / 'fold_distortion', D)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--eff', default=None)
    ap.add_argument('--acceptance', default=None)
    ap.add_argument('--fold', default=None)
    ap.add_argument('--variant', default='incidence')
    a = ap.parse_args()
    ed = Path(a.eff) if a.eff else paths.out('efficiency_campaign')
    ad = Path(a.acceptance) if a.acceptance else paths.out(
        'acceptance_campaign')
    fdir = Path(a.fold) if a.fold else paths.out('fold_campaign')
    fd = paths.figures('fold_campaign')
    figstyle.use()

    made = []
    for fn, args in ((fig_efficiency, (ed, fd)),
                     (fig_incidence, (ed, fd)),
                     (fig_acceptance_runs, (ad, fd, a.variant)),
                     (fig_variants, (ad, fd)),
                     (fig_shapes, (fdir, fd)),
                     (fig_folded, (fdir, fd)),
                     (fig_two_comp, (fdir, fd)),
                     (fig_corrected, (fdir, fd)),
                     (fig_distortion, (fdir, fd))):
        try:
            made.append(fn(*args))
        except FileNotFoundError as e:
            print(f'  skipped {fn.__name__}: {e}')
    print(f'\n{len(made)} figure(s) -> {fd}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
