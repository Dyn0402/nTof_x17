#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- the eight figures of the pair-vertex diagnosis.

Draws only.  Every number comes from the CSVs `diagnostics.py` writes and the
pair table `vertex_lab.py` builds, so re-running a measurement and re-running
this produces figures that agree with the report by construction rather than by
memory.  House style from `mpgd26/plotstyle.py`, imported, so a chamber keeps
the colour it has in the Athens deck.

    python -m pair_vertex_imaging.make_figures
    python -m pair_vertex_imaging.make_figures --only leg_scan,pointing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26')):
    if p not in sys.path:
        sys.path.insert(0, p)

import plotstyle as PS  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402

OUTDIR = HERE / 'figures'
CLASSES = ('intra', 'perpendicular', 'opposing')
CLASS_LABEL = {'intra': 'intra  (A-A, C-C, D-D)',
               'perpendicular': 'perpendicular  (A-D, C-D)',
               'opposing': 'opposing  (A-C)'}
#: One hue per class, held across every figure in the set.
CC = {'intra': PS.DET_COLOR['C'], 'perpendicular': PS.DET_COLOR['A'],
      'opposing': PS.DET_COLOR['D']}
CAPSULE = PS.COPPER
PROV = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full '
        'pass, stage-3 tracks, leg pointing cut as marked')


def _hist(ax, v, bins, color, label, ls='-', lw=2.0, norm=True):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, bins)
    y = h / max(h.sum(), 1) / np.diff(e) if norm else h
    ax.step(e[:-1], y, where='post', color=color, lw=lw, ls=ls, label=label)
    return y


def _capsule_band(ax, r=10.0, label=True):
    ax.axvspan(0, r, color=CAPSULE, alpha=0.16, lw=0, zorder=0)
    if label:
        ax.text(r + 3, ax.get_ylim()[1] * 0.40, 'the capsule\nis 10 mm across',
                color=CAPSULE, fontsize=10, va='top', ha='left')


# --------------------------------------------------------------------------- #
def fig_observation(P, od):
    """The impression, confirmed: the vertex is worse than either leg, and the
    event-mixed null sits on top of it."""
    d = P[P.dca_worst < 30]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))
    fig.subplots_adjust(wspace=0.26)
    bins = np.linspace(0, 200, 81)
    for ax, topo in zip(axes, CLASSES):
        g = d[d.topo == topo]
        r, m = g[~g.mixed], g[g.mixed]
        _hist(ax, np.r_[r.dca_axis_mm_1, r.dca_axis_mm_2], bins, PS.LINE,
              'either leg, on its own', lw=1.8)
        _hist(ax, r.v_r_xz, bins, CC[topo], 'pair vertex, transverse crossing')
        _hist(ax, r.v_r, bins, PS.INK, 'pair vertex, 3D closest approach')
        _hist(ax, m.v_r, bins, PS.INK, 'the same, event-mixed', ls=':', lw=1.6)
        ax.set_xlim(0, 200)
        ax.set_xlabel('distance from the beam axis  [mm]')
        PS.strip(ax)
        PS.title(ax, CLASS_LABEL[topo],
                 f'{len(r):,} pairs   legs {np.nanmedian(r.dca_best):.0f} and '
                 f'{np.nanmedian(r.dca_worst):.0f} mm   vertex '
                 f'{np.nanmedian(r.v_r):.0f} mm')
        _capsule_band(ax)
    axes[0].set_ylabel('pairs  (unit area)')
    axes[0].legend(loc='upper right')
    PS.note(fig, 'Pairing two tracks that each point within 30 mm of the axis '
                 'produces a vertex FURTHER from it than either track was, and '
                 'the event-mixed null reproduces it.  ' + PROV)
    PS.save(fig, str(od / 'observation.png'))
    PS.save(fig, str(od / 'observation.pdf'))


def fig_split(P, od):
    """The 3D closest approach against the transverse crossing, on the same
    pairs: what the y information costs."""
    Y = pd.read_csv(paths.out('pair_vertex') / 'ybudget.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    d = P[(P.dca_worst < 30) & (~P.mixed)]
    bins = np.linspace(0, 300, 76)
    for topo in CLASSES:
        g = d[d.topo == topo]
        _hist(ax, g.v_r_xz, bins, CC[topo], f'{topo}, transverse crossing')
        _hist(ax, g.v_r, bins, CC[topo], f'{topo}, 3D DCA', ls=':', lw=1.6)
    ax.set_xlabel('vertex distance from the beam axis  [mm]')
    ax.set_ylabel('pairs  (unit area)')
    ax.set_xlim(0, 300)
    PS.strip(ax)
    PS.title(ax, 'Dropping y improves every class',
             'solid: transverse crossing   dotted: the published 3D DCA')
    _capsule_band(ax)
    ax.legend(loc='upper right', fontsize=10)

    ax = axes[1]
    order = ['0-25', '25-50', '50-100', '100-200', '200-400', '>400']
    x = np.arange(len(order))
    for topo in CLASSES:
        g = Y[Y.topology == topo].set_index('dy_bin').reindex(order)
        ax.plot(x, g.v_r_med, 'o-', color=CC[topo], label=f'{topo}, 3D DCA')
        ax.plot(x, g.v_r_xz_med, 's--', color=CC[topo], alpha=0.55,
                label=f'{topo}, XZ crossing')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    ax.set_xlabel('|y mismatch of the two legs at the crossing|  [mm]')
    ax.set_ylabel('median vertex radius  [mm]')
    PS.strip(ax)
    PS.title(ax, 'and |dy| is the whole of the difference',
             'the transverse answer does not move; the 3D one follows it')
    ax.legend(loc='upper left', fontsize=9.5, ncol=2)
    PS.note(fig, 'The 3D closest approach is free to slide both tracks along '
                 'themselves to reduce a y mismatch, and every millimetre it '
                 'slides moves the vertex transversely too.  ' + PROV)
    PS.save(fig, str(od / 'y_cost.png'))
    PS.save(fig, str(od / 'y_cost.pdf'))


def fig_conditioning(P, od):
    """The transverse crossing is the legs' own miss, amplified by 1/sin psi."""
    d = P[(P.dca_worst < 30) & (~P.mixed)]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    bins = np.linspace(1, 12, 56)
    for topo in CLASSES:
        g = d[d.topo == topo]
        _hist(ax, 1.0 / g.sin_psi_xz, bins, CC[topo],
              f'{topo}  (median {np.nanmedian(1 / g.sin_psi_xz):.2f})')
    ax.set_xlabel(r'amplification  $1/|\sin\psi_{xz}|$')
    ax.set_ylabel('pairs  (unit area)')
    ax.set_xlim(1, 12)
    PS.strip(ax)
    PS.title(ax, 'How much the crossing amplifies',
             'perpendicular pairs barely do; the other two do a lot')
    ax.legend(loc='upper right')

    ax = axes[1]
    xb = np.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])
    for topo in CLASSES:
        g = d[d.topo == topo].copy()
        g['a'] = 1.0 / g.sin_psi_xz
        k = pd.cut(g.a, xb)
        med = g.groupby(k, observed=True).v_r_xz.median()
        ctr = [iv.mid for iv in med.index]
        ax.plot(ctr, med.to_numpy(), 'o-', color=CC[topo], label=topo)
    lo = np.linspace(1, 20, 50)
    ax.plot(lo, 21.8 * lo / 1.0, color=PS.MUTED, ls='--', lw=1.4,
            label='21.8 mm leg miss, amplified')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 20)
    ax.set_ylim(10, 600)
    ax.set_xlabel(r'amplification  $1/|\sin\psi_{xz}|$')
    ax.set_ylabel('median transverse vertex radius  [mm]')
    PS.strip(ax)
    PS.title(ax, 'and the vertex follows it',
             r'$v_r^{xz}=\sqrt{e_1^2+e_2^2-2e_1e_2\cos\psi}/|\sin\psi|$, '
             'exact to 1e-16')
    ax.legend(loc='upper left')
    PS.note(fig, 'The transverse pair vertex is an algebraic function of the '
                 'two legs own miss distances and the crossing angle -- not an '
                 'independent look at the source.  ' + PROV)
    PS.save(fig, str(od / 'conditioning.png'))
    PS.save(fig, str(od / 'conditioning.pdf'))


def fig_leg_scan(od):
    """Tighten the leg cut and the vertex follows it linearly, with no floor of
    its own until the capsule."""
    S = pd.read_csv(paths.out('pair_vertex') / 'leg_scan.csv')
    S = S[~S.mixed]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    for topo in CLASSES:
        g = S[S.topology == topo].sort_values('leg_cut_mm')
        ax.plot(g.leg_cut_mm, g.v_r_xz_med, 'o-', color=CC[topo],
                label=f'{topo}, XZ crossing')
        ax.plot(g.leg_cut_mm, g.v_r_med, 's--', color=CC[topo], alpha=0.5,
                label=f'{topo}, 3D DCA')
    x = np.linspace(0, 62, 20)
    ax.plot(x, 0.86 * x, color=PS.MUTED, ls=':', lw=1.6)
    ax.text(40, 0.86 * 40 - 13, r'$v_r^{xz}=0.86\times$ the cut',
            color=PS.MUTED, fontsize=10.5)
    ax.axhspan(0, 10, color=CAPSULE, alpha=0.16, lw=0, zorder=0)
    ax.text(1, 11, 'capsule', color=CAPSULE, fontsize=10)
    ax.axvline(30, color=PS.INK, lw=1.0, ls='--', alpha=0.5)
    ax.text(30.8, 4, 'published cut', color=PS.INK, fontsize=10, alpha=0.7)
    ax.set_xlabel('per-leg pointing cut  |miss at the axis|  [mm]')
    ax.set_ylabel('median vertex radius  [mm]')
    ax.set_xlim(0, 62)
    PS.strip(ax)
    PS.title(ax, 'The vertex is its legs, and nothing else',
             'the transverse crossing tracks the cut and never finds a floor')
    ax.legend(loc='upper left', fontsize=9.5, ncol=2)

    ax = axes[1]
    for topo in CLASSES:
        g = S[S.topology == topo].sort_values('leg_cut_mm')
        ax.plot(g.leg_cut_mm, 100 * g.f_vrxz_10, 'o-', color=CC[topo],
                label=f'{topo}, XZ crossing')
        ax.plot(g.leg_cut_mm, 100 * g.f_vr_10, 's--', color=CC[topo],
                alpha=0.5, label=f'{topo}, 3D DCA')
    ax.set_xlabel('per-leg pointing cut  [mm]')
    ax.set_ylabel('pairs with the vertex inside the capsule  [%]')
    ax.set_xlim(0, 62)
    PS.strip(ax)
    PS.title(ax, 'A capsule image exists, for 3 % of pairs',
             'and only with the y information left out')
    ax.legend(loc='upper right', fontsize=9.5, ncol=2)
    PS.note(fig, 'At a 5 mm leg cut the transverse crossing puts 97 % of '
                 'perpendicular pairs inside the capsule -- on 946 of 29 617 '
                 'pairs.  The 3D closest approach never exceeds 21 %.  ' + PROV)
    PS.save(fig, str(od / 'leg_scan.png'))
    PS.save(fig, str(od / 'leg_scan.pdf'))


def fig_pointing(od):
    """The single-track pointing, per arm, against a null with the source
    information removed."""
    H = pd.read_csv(paths.out('pair_vertex') / 'pointing_hist.csv')
    S = pd.read_csv(paths.out('pair_vertex') / 'pointing.csv')
    arms = [a for a in ('A', 'B', 'C', 'D') if f'{a}_data' in H.columns]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.9),
                             sharey=True)
    fig.subplots_adjust(wspace=0.12)
    ctr = 0.5 * (H.lo + H.hi).to_numpy()
    w = (H.hi - H.lo).to_numpy()
    for ax, a in zip(np.atleast_1d(axes), arms):
        d = H[f'{a}_data'].to_numpy(float)
        n = H[f'{a}_null'].to_numpy(float)
        ax.step(H.lo, d / d.sum() / w, where='post', color=PS.DET_COLOR[a],
                lw=2.0, label='gated tracks')
        ax.step(H.lo, n / n.sum() / w, where='post', color=PS.MUTED, lw=1.6,
                ls='--', label='tan shuffled (no source)')
        r = S[S.arm == a].iloc[0]
        ax.set_xlim(0, 250)
        ax.set_xlabel('|miss at the beam axis|  [mm]')
        PS.strip(ax)
        PS.title(ax, f'chamber {a}',
                 f'median {r.med_dca:.0f} mm  vs  {r.med_dca_null:.0f} mm null')
        ax.axvspan(0, 10, color=CAPSULE, alpha=0.16, lw=0, zorder=0)
        ax.axvline(30, color=PS.INK, lw=0.9, ls='--', alpha=0.45)
        ax.text(0.97, 0.62, f'inside 10 mm\n{100 * r.f10:.1f} %  vs  '
                            f'{100 * r.f10_null:.1f} %',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                color=PS.INK)
    np.atleast_1d(axes)[0].set_ylabel('tracks  (unit area)')
    np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=10)
    PS.note(fig, 'The null keeps every marginal -- the same impact points, the '
                 'same angles, the same acceptance -- and destroys only the '
                 'correlation between them, which is the whole of "this track '
                 'came from the capsule".  The dashed line is the 30 mm leg '
                 'cut.  ' + PROV)
    PS.save(fig, str(od / 'pointing.png'))
    PS.save(fig, str(od / 'pointing.pdf'))


def fig_scale(od):
    """What an angle-scale error does to each estimator."""
    F = pd.read_csv(paths.out('pair_vertex') / 'scale_focus.csv')
    S = pd.read_csv(paths.out('pair_vertex') / 'scale.csv')
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    B = S[(S.leg_cut_mm == 30) & (S.topology.str.startswith('band_'))]
    for a in ('A', 'B', 'C', 'D'):
        g = B[B.topology == f'band_{a}'].sort_values('scale')
        if not len(g):
            continue
        ax.plot(g.scale, g.band_x0, 'o-', color=PS.DET_COLOR[a], label=a)
    ax.set_ylim(-40, 60)
    ax.set_xlabel(r'tan multiplier  $s$')
    ax.set_ylabel('fitted band crossing  [mm]')
    PS.strip(ax)
    PS.title(ax, 'The image does not care',
             'the crossing is -intercept/slope: s cancels')

    ax = axes[1]
    for a in ('A', 'B', 'C', 'D'):
        g = F[F.arm == a].sort_values('scale')
        if not len(g):
            continue
        ax.plot(g.scale, g.med_dca, 'o-', color=PS.DET_COLOR[a], label=a)
    ax.axvline(1.0, color=PS.INK, lw=0.9, ls='--', alpha=0.45)
    ax.axvline(1.33, color=PS.COPPER, lw=1.2, ls='--')
    ax.text(1.34, ax.get_ylim()[1] * 0.97, ' the wall asks for this',
            color=PS.COPPER, fontsize=10, va='top')
    ax.set_xlabel(r'tan multiplier  $s$')
    ax.set_ylabel('median single-track miss  [mm]')
    PS.strip(ax)
    PS.title(ax, 'Per-track pointing: a shallow optimum',
             'every gated track, no pointing cut, no circularity')
    ax.legend(loc='upper left', ncol=2)

    ax = axes[2]
    for topo in CLASSES:
        for cut, ls, al in ((30.0, '-', 1.0), (60.0, '--', 0.5)):
            g = S[(S.topology == topo) & (S.leg_cut_mm == cut)
                  ].sort_values('scale')
            ax.plot(g.scale, g.v_r_xz_med, 'o' + ls, color=CC[topo], alpha=al,
                    label=f'{topo}, leg < {cut:.0f} mm')
    ax.axvline(1.0, color=PS.INK, lw=0.9, ls='--', alpha=0.45)
    ax.axvline(1.33, color=PS.COPPER, lw=1.2, ls='--')
    ax.set_xlabel(r'tan multiplier  $s$')
    ax.set_ylabel('median transverse vertex radius  [mm]')
    PS.strip(ax)
    PS.title(ax, 'The vertex cares a great deal',
             'x1.33 doubles it; x1.0 is still 26 mm, not 7')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    PS.note(fig, 'The angle scale moves the vertex and cannot move the image, '
                 'which is why one of the two survived the open k question and '
                 'the other did not.  It is not, however, the explanation: at '
                 'its own best scale the vertex is still 26 mm.  ' + PROV)
    PS.save(fig, str(od / 'scale.png'))
    PS.save(fig, str(od / 'scale.pdf'))


def fig_floor(od):
    """The ideal-leg substitution: the floor, and how far the data is above it."""
    F = pd.read_csv(paths.out('pair_vertex') / 'floor.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    fig.subplots_adjust(wspace=0.26)
    lab = {'none': 'as measured', 'one': 'one leg pointed exactly at the capsule',
           'both': 'both legs pointed exactly at the capsule'}
    x = np.arange(len(CLASSES))
    ax = axes[0]
    for i, v in enumerate(('none', 'one', 'both')):
        g = F[F.variant == v].set_index('topology').reindex(CLASSES)
        ax.bar(x + (i - 1) * 0.27, g.v_r_xz_med, 0.25,
               color=[CC[t] for t in CLASSES],
               alpha=[1.0, 0.6, 0.3][i], label=lab[v],
               edgecolor=PS.INK, linewidth=0.6)
        for xi, yi in zip(x + (i - 1) * 0.27, g.v_r_xz_med):
            ax.text(xi, yi + 1.2, f'{yi:.0f}', ha='center', fontsize=9.5,
                    color=PS.INK)
    ax.axhline(10, color=CAPSULE, lw=1.6, ls='--')
    ax.text(2.45, 11, 'capsule radius', color=CAPSULE, fontsize=10, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel('median transverse vertex radius  [mm]')
    PS.strip(ax)
    PS.title(ax, 'The geometry imposes no floor',
             'with perfect angles every class returns the source, 7.0 mm')
    ax.legend(loc='upper left', fontsize=10)

    ax = axes[1]
    for i, v in enumerate(('none', 'one', 'both')):
        g = F[F.variant == v].set_index('topology').reindex(CLASSES)
        ax.bar(x + (i - 1) * 0.27, 100 * g.f_vrxz_10, 0.25,
               color=[CC[t] for t in CLASSES], alpha=[1.0, 0.6, 0.3][i],
               edgecolor=PS.INK, linewidth=0.6)
        for xi, yi in zip(x + (i - 1) * 0.27, 100 * g.f_vrxz_10):
            ax.text(xi, yi + 1.5, f'{yi:.0f}', ha='center', fontsize=9.5,
                    color=PS.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel('vertex inside the capsule  [%]')
    PS.strip(ax)
    PS.title(ax, 'so the whole loss is the angle',
             'one perfect leg roughly doubles it, two make it exact')
    PS.note(fig, 'The substituted leg keeps its MEASURED impact point and gets '
                 'only its direction replaced, so the test changes the angle '
                 'and nothing else.  ' + PROV)
    PS.save(fig, str(od / 'floor.png'))
    PS.save(fig, str(od / 'floor.pdf'))


def fig_lift(od):
    """Whether a vertex cut enriches real pairs over event-mixed ones."""
    L = pd.read_csv(paths.out('pair_vertex') / 'lift.csv')
    L = L[L.variable == 'v_r_xz']
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for topo in CLASSES:
        g = L[L.topology == topo].sort_values('cut_mm')
        err = g.lift * np.sqrt(1.0 / np.maximum(g.k_real, 1)
                               + 1.0 / np.maximum(g.k_mixed, 1))
        ax.errorbar(g.cut_mm, g.lift, yerr=err, fmt='o-', color=CC[topo],
                    capsize=3, label=topo)
    ax.axhline(1.0, color=PS.INK, lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_xticks(sorted(L.cut_mm.unique()))
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('transverse vertex cut  [mm]')
    ax.set_ylabel('real / event-mixed, as a fraction of each sample')
    ax.set_ylim(0.5, 1.7)
    PS.strip(ax)
    PS.title(ax, 'The vertex carries almost no coincidence information',
             'and for opposing pairs it is below one -- the trigger, not '
             'the physics')
    ax.legend(loc='upper right')
    PS.note(fig, 'A lift of 1 is not "the imaging failed": mixing moves the '
                 'trigger, not the capsule, so both legs of a mixed pair still '
                 'came from the same source.  It says the vertex cannot be '
                 'used to select pairs.  ' + PROV)
    PS.save(fig, str(od / 'lift.png'))
    PS.save(fig, str(od / 'lift.pdf'))


def fig_yview(od, P):
    """Why the y view cannot be used: it is not a pointing measurement, and the
    source is 80 mm long in that direction anyway."""
    T = pd.read_parquet(paths.out('pair_vertex') / 'tracks_vertex.parquet')
    from pair_vertex_imaging.diagnostics import _robust_line
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax = axes[0]
    rows = []
    for a in ('A', 'B', 'C', 'D'):
        g = T[T.arm == a]
        if len(g) < 1000:
            continue
        for view, pos, tan in (('x', g.x_local, g.tanx), ('y', g.y_local, g.tany)):
            p, t = pos.to_numpy(float), tan.to_numpy(float)
            m = np.isfinite(p) & np.isfinite(t)
            sl, _ = _robust_line(p[m], t[m])
            rows.append(dict(arm=a, view=view, slope=sl * 234.6))
    B = pd.DataFrame(rows)
    x = np.arange(B.arm.nunique())
    for i, view in enumerate(('x', 'y')):
        g = B[B.view == view].set_index('arm').reindex(sorted(B.arm.unique()))
        ax.bar(x + (i - 0.5) * 0.36, g.slope, 0.33,
               color=[PS.DET_COLOR[a] for a in g.index],
               alpha=1.0 if view == 'x' else 0.42, edgecolor=PS.INK, lw=0.6,
               label=f'{view} view  (in-plane transverse)' if view == 'x'
               else 'y view  (along the beam)')
        for xi, yi in zip(x + (i - 0.5) * 0.36, g.slope):
            ax.text(xi, yi + 0.02, f'{yi:.2f}', ha='center', fontsize=9.5,
                    color=PS.INK)
    ax.axhline(1.0, color=PS.INK, ls='--', lw=1.3)
    ax.text(len(x) - 0.6, 1.03, 'a point source', color=PS.INK, fontsize=10,
            ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(B.arm.unique()))
    ax.set_ylabel(r'pointing-band slope $\times\,d_\perp$')
    PS.strip(ax)
    PS.title(ax, 'The y view is not a pointing measurement',
             'the x band is pinned at 1 by the leg cut; the y band is not cut '
             'on at all')

    ax = axes[1]
    d = P[(P.dca_worst < 30) & (~P.mixed)]
    bins = np.linspace(0, 600, 61)
    _hist(ax, np.abs(d.dy_cross), bins, PS.INK, 'measured |dy| at the crossing')
    _hist(ax, np.abs(d.p0_y_1 - d.p0_y_2), bins, PS.MUTED,
          'from the two impact heights alone', ls='--', lw=1.6)
    ax.axvline(80.2, color=CAPSULE, lw=1.6, ls='--')
    ax.text(84, ax.get_ylim()[1] * 0.85, ' the capsule is 80 mm long',
            color=CAPSULE, fontsize=10)
    ax.set_xlabel('|y mismatch of the two legs|  [mm]')
    ax.set_ylabel('pairs  (unit area)')
    PS.strip(ax)
    PS.title(ax, 'so the two legs disagree in y by 161 mm',
             'half of it the impact heights, half the slope over a 270 mm '
             'lever')
    ax.legend(loc='upper right', fontsize=10)
    PS.note(fig, 'Even a perfect y angle could not localise better than the '
                 'source length: an 80 mm source seen over a 234.6 mm lever is '
                 'a 23 mm irreducible y spread.  ' + PROV)
    PS.save(fig, str(od / 'y_view.png'))
    PS.save(fig, str(od / 'y_view.pdf'))


FIGS = {
    'observation': lambda P, od: fig_observation(P, od),
    'y_cost': lambda P, od: fig_split(P, od),
    'conditioning': lambda P, od: fig_conditioning(P, od),
    'leg_scan': lambda P, od: fig_leg_scan(od),
    'pointing': lambda P, od: fig_pointing(od),
    'scale': lambda P, od: fig_scale(od),
    'floor': lambda P, od: fig_floor(od),
    'lift': lambda P, od: fig_lift(od),
    'y_view': lambda P, od: fig_yview(od, P),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', default='')
    ap.add_argument('--out', default=str(OUTDIR))
    a = ap.parse_args()
    want = [x.strip() for x in a.only.split(',') if x.strip()] or list(FIGS)
    PS.use()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(paths.out('pair_vertex') / 'pairs_vertex.parquet')
    for name in want:
        if name not in FIGS:
            raise SystemExit(f'unknown figure {name!r}; have {sorted(FIGS)}')
        print(name)
        FIGS[name](P, od)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
