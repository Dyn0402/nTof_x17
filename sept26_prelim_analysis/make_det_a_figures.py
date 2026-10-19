#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_det_a_figures.py -- figures for the detector-A intra-chamber analysis.

Reads what `det_a_intra.py` wrote; computes nothing new.  Every figure ships
its numbers beside it as a CSV, as PLAN sec 7 requires.

    python -m sept26_prelim_analysis.make_det_a_figures
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
from sept26_prelim_analysis.det_a_intra import (  # noqa: E402
    OFFTIME_NS, PROMPT_NS)

SEL_COLOR = {'all': '#333333', 'slope': '#0072B2', 'prompt': '#D55E00',
             'slope+prompt': '#CC79A7', 'slope+offtime': '#009E73',
             'pointing': '#E69F00', 'mixed': '#999999',
             'slope+mixed': '#BBBBBB'}
MODEL_COLOR = {'event-mixed (no pair physics)': '#666666',
               'Al capsule (after wall)': '#CC79A7',
               'Al capsule (birth)': '#E7A2C4',
               '3He gas M1+E0': '#0072B2',
               '3He gas M1+E0 (after wall)': '#56B4E9',
               '3He gas E0 only': '#009E73',
               '3He gas M1 only': '#7FCDBB',
               'X17 (17 MeV boson)': '#D55E00'}


def _step(ax, x, y, **kw):
    ax.step(x, y, where='mid', **kw)


# --------------------------------------------------------------------------- #
# the spectrum and the fold
# --------------------------------------------------------------------------- #
def fig_spectrum(d: Path, fd: Path):
    """The measured spectrum per selection, shape-normalised."""
    import matplotlib.pyplot as plt
    S = pd.read_csv(d / 'spectra.csv')
    fig, ax = figstyle.figure(figstyle.WIDE)
    for sel in ('all', 'mixed', 'slope', 'slope+prompt', 'slope+offtime'):
        g = S[S.selection == sel].sort_values('theta')
        if g.empty or g.n.sum() < 20:
            continue
        _step(ax, g.theta, g.frac, lw=2.4, color=SEL_COLOR.get(sel, '#333333'),
              label=f'{sel} (n = {int(g.n.sum()):,})')
    ax.set_xlabel('opening angle [deg]')
    ax.set_ylabel('fraction per 10 deg bin')
    ax.legend(fontsize=figstyle.BASE_PT * 0.93)
    figstyle.fig_title(fig, 'Requiring a measurable slope on both legs moves the '
                       'intra-A spectrum by 20 degrees',
                   'and the in-chamber prompt and off-time samples are the '
                   'same shape as each other')
    return figstyle.save(fig, fd / 'a_spectrum', S)


def fig_fold(d: Path, fd: Path):
    """Measured against folded, for the two selections that matter."""
    import matplotlib.pyplot as plt
    S = pd.read_csv(d / 'spectra.csv')
    F = pd.read_csv(d / 'folded.csv')
    show = ['Al capsule (after wall)', '3He gas M1+E0', '3He gas E0 only']
    MIXED_OF = {'all': 'mixed', 'slope': 'slope+mixed'}
    panels = [('all', 'acc_sep'), ('slope', 'acc_sep_slope')]
    fig, axes = figstyle.figure(figstyle.FULL, nrows=2, sharex=True)
    for ax, (sel, acc) in zip(axes, panels):
        o = S[S.selection == sel].sort_values('theta')
        n = o.n.to_numpy(float)
        if n.sum() <= 0:
            continue
        ax.errorbar(o.theta, o.frac, yerr=o.err, fmt='o', color='#333333',
                    capsize=3, zorder=5)
        mx = S[S.selection == MIXED_OF[sel]].sort_values('theta')
        if len(mx) and mx.n.sum() > 0:
            _step(ax, mx.theta, mx.n / mx.n.sum(), lw=2.4,
                  color=MODEL_COLOR['event-mixed (no pair physics)'],
                  label='event-mixed (no pair physics)')
        for m in show:
            g = F[(F.acceptance == acc) & (F.model == m)].sort_values('theta')
            if g.empty:
                continue
            _step(ax, g.theta, g.frac, lw=2.2,
                  color=MODEL_COLOR.get(m, '#333333'), label=m)
        ax.set_title(f'{sel}   n = {int(n.sum()):,}   ({acc})', loc='left',
                     color=SEL_COLOR.get(sel, '#333333'),
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        ax.set_ylabel('fraction')
        # explicit headroom: `margins` will not lift a top that a step curve
        # lands exactly on, and the tallest model bin did land on it
        hi = max([o.frac.max()] + [
            F[(F.acceptance == acc) & (F.model == m)].frac.max()
            for m in show if len(F[(F.acceptance == acc) & (F.model == m)])])
        ax.set_ylim(0, float(hi) * 1.15)
    axes[-1].set_xlabel('opening angle [deg]')
    h, la = axes[0].get_legend_handles_labels()
    top = figstyle.fig_title(
        fig, 'The capsule continuum, folded, against the data and against a '
             'null with no pair physics in it',
        'points are the data; every curve is normalised to the observed count')
    # placed AFTER the title so it can be anchored under the space the title
    # reserved, instead of on top of it
    fig.legend(h, la, loc='upper right', ncol=2, frameon=False,
               fontsize=figstyle.BASE_PT * 0.89,
               bbox_to_anchor=(0.99, top - 0.004))
    return figstyle.save(fig, fd / 'a_fold', {'obs': S, 'folded': F})


# --------------------------------------------------------------------------- #
# the in-chamber clock
# --------------------------------------------------------------------------- #
def fig_dt0(d: Path, fd: Path):
    """dt0, split by slope and by scintillator tag, shape-normalised."""
    import matplotlib.pyplot as plt
    H = pd.read_csv(d / 'dt0_histogram.csv')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    DT_COLOR = {'all': '#333333', 'slope': '#0072B2', 'no slope': '#D55E00',
                'tagged': '#009E73', 'untagged': '#999999'}
    for ax, group in zip(axes, (('all', 'slope', 'no slope'),
                                ('tagged', 'untagged'))):
        for name in group:
            g = H[H['sample'] == name].sort_values('dt0')
            if g.empty or g.n.sum() < 20:
                continue
            _step(ax, g.dt0, g.n / g.n.sum(), lw=2.2,
                  color=DT_COLOR.get(name, '#333333'),
                  label=f'{name} (n = {int(g.n.sum()):,})')
        ax.axvspan(-PROMPT_NS, PROMPT_NS, color='#D55E00', alpha=0.10, lw=0)
        for s in (-1, 1):
            ax.axvspan(s * OFFTIME_NS[0], s * OFFTIME_NS[1],
                       color='#0072B2', alpha=0.07, lw=0)
        ax.set_xlabel(r'$\Delta t_0$ between the two legs [ns]')
        ax.set_xlim(-600, 600)
        ax.legend(fontsize=figstyle.BASE_PT * 0.86)
    axes[0].set_ylabel('fraction per bin')
    figstyle.fig_title(fig, 'The peak is in the tracks without a measurable '
                            'slope, and the scintillators do not see it',
                   'orange band: the prompt window; blue bands: the off-time '
                   'control')
    return figstyle.save(fig, fd / 'a_dt0', H)


def fig_slope_profile(d: Path, fd: Path):
    """Core-over-wing and opening angle against the pair's smallest slope."""
    import matplotlib.pyplot as plt
    from wft import reco as WR
    S = pd.read_csv(d / 'slope_profile.csv')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    ax = axes[0]
    ax.errorbar(S.tan_mid, S.core_over_wing,
                yerr=S.core_over_wing * np.sqrt(1 / np.clip(S.n, 1, None)),
                fmt='o-', color='#D55E00', capsize=3)
    ax.axhline(1.0, color='#333333', lw=1.2, ls='--')
    ax.axvline(WR.TAN_MIN_SLOPE, color='#0072B2', lw=1.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.10 * (hi - lo))
    ax.text(WR.TAN_MIN_SLOPE * 0.92, ax.get_ylim()[1], 'TAN_MIN_SLOPE',
            ha='right', va='top', fontsize=figstyle.BASE_PT * 0.93,
            color='#0072B2')
    ax.set_xscale('log')
    ax.set_xlabel(r'smallest |tan| in the pair')
    ax.set_ylabel('prompt / off-time')
    ax = axes[1]
    ax.errorbar(S.tan_mid, S.median_open_deg, fmt='s-', color='#0072B2',
                capsize=3)
    ax.axvline(WR.TAN_MIN_SLOPE, color='#0072B2', lw=1.6)
    ax.set_xscale('log')
    ax.set_xlabel(r'smallest |tan| in the pair')
    ax.set_ylabel('median opening angle [deg]')
    figstyle.fig_title(fig, 'The prompt peak disappears exactly where the '
                            'reconstruction regains a timing slope',
                   'above the threshold the ratio is consistent with one, so '
                   'there is no peak left to select on')
    return figstyle.save(fig, fd / 'a_slope_profile', S)


def fig_dt_sep(d: Path, fd: Path):
    """The 2D correlation: dt0 against in-plane separation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    G = pd.read_csv(d / 'dt_sep_grid.csv')
    Pr = pd.read_csv(d / 'dt_sep_profile.csv')
    fig, axes = figstyle.figure(figstyle.FULL, ncols=2, nrows=1)
    for ax, sel in zip(axes, ('all', 'slope')):
        g = G[G.selection == sel]
        if g.empty or g.n.sum() == 0:
            continue
        piv = g.pivot(index='sep_mm', columns='dt0_ns', values='n')
        im = ax.pcolormesh(piv.columns.to_numpy(), piv.index.to_numpy(),
                           np.clip(piv.to_numpy(), 0.5, None),
                           norm=LogNorm(), cmap='magma_r', shading='auto')
        ax.axvline(-PROMPT_NS, color='#0072B2', lw=1.2, ls='--')
        ax.axvline(PROMPT_NS, color='#0072B2', lw=1.2, ls='--')
        ax.axhline(40.0, color='#009E73', lw=1.6)
        ax.set_xlabel(r'$\Delta t_0$ [ns]')
        ax.set_title(f'{sel}   n = {int(g.n.sum()):,}', loc='left',
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        fig.colorbar(im, ax=ax, label='pairs', fraction=0.046, pad=0.03)
    axes[0].set_ylabel('in-plane separation [mm]')
    figstyle.fig_title(fig, 'Time against position, the correlation a real '
                            'pair would have and this sample does not',
                   'green line: the double-track resolution edge at 40 mm; '
                   'blue: the prompt window')
    return figstyle.save(fig, fd / 'a_dt_sep', {'grid': G, 'profile': Pr})


def fig_dudv(d: Path, fd: Path):
    """Where the two legs land relative to each other, real against mixed."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    P = pd.read_parquet(d / 'pairs.parquet')
    edges = np.arange(-420, 421, 20.0)
    # three SQUARE panels in a row: a slide-height canvas leaves the row short
    # and the space above it empty, so use the shorter one
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=3)
    out = {}
    for ax, (name, m) in zip(axes, (('real pairs', ~P.mixed.to_numpy()),
                                    ('event-mixed', P.mixed.to_numpy()))):
        g = P[m]
        # symmetrise: the pair is unordered, so (du, dv) and (-du, -dv) are the
        # same object and plotting one of them would invent an asymmetry
        du = np.r_[g.du_mm.to_numpy(), -g.du_mm.to_numpy()]
        dv = np.r_[g.dv_mm.to_numpy(), -g.dv_mm.to_numpy()]
        Hh, xe, ye = np.histogram2d(du, dv, bins=[edges, edges])
        im = ax.pcolormesh(xe, ye, np.clip(Hh.T, 0.5, None), norm=LogNorm(),
                           cmap='magma_r', shading='auto')
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(40 * np.cos(th), 40 * np.sin(th), color='#009E73', lw=2.0)
        ax.set_xlabel(r'$\Delta u$ [mm]')
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nn = {int(m.sum()):,}', loc='left',
                     fontsize=figstyle.BASE_PT * 0.98, pad=6)
        fig.colorbar(im, ax=ax, label='pairs', fraction=0.046, pad=0.03)
        out[name.split()[0]] = pd.DataFrame(
            dict(du=np.repeat(0.5 * (xe[:-1] + xe[1:]), len(ye) - 1),
                 dv=np.tile(0.5 * (ye[:-1] + ye[1:]), len(xe) - 1),
                 n=Hh.ravel()))
    # the third panel is the evidence the efficiency is built on: real over
    # event-mixed, which turns the two occupancy maps into one efficiency map
    ae = np.arange(0.0, 401.0, 25.0)

    def h2(g):
        z, _, _ = np.histogram2d(g.du_mm.abs(), g.dv_mm.abs(), bins=[ae, ae])
        return z
    hr, hm = h2(P[~P.mixed]), h2(P[P.mixed])
    k = hr.sum() / hm.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = np.where(hm > 0, hr / (k * hm), np.nan)
    ax = axes[2]
    im = ax.pcolormesh(ae, ae, np.clip(rat.T, 0, 2), cmap='RdBu_r', vmin=0,
                       vmax=2, shading='auto')
    fig.colorbar(im, ax=ax, label='real / mixed', fraction=0.046,
                 pad=0.03)
    ax.set_xlabel(r'$|\Delta u|$ [mm]')
    ax.set_ylabel(r'$|\Delta v|$ [mm]')
    ax.set_aspect('equal')
    ax.set_title('ratio\na cross, not a disc', loc='left',
                 fontsize=figstyle.BASE_PT * 0.98, pad=6)
    am = 0.5 * (ae[:-1] + ae[1:])
    out['ratio'] = pd.DataFrame(dict(
        du=np.repeat(am, len(am)), dv=np.tile(am, len(am)),
        n_real=hr.ravel(), n_mixed=hm.ravel(), ratio=rat.ravel()))
    axes[0].set_ylabel(r'$\Delta v$ [mm]')
    figstyle.fig_title(fig, 'The double-track loss is a CROSS, not a disc: a '
                            'pair that shares one view is lost whatever the '
                            'other view does',
                       'so an efficiency in the radial separation alone passes '
                       'pairs the reconstruction never finds')
    return figstyle.save(fig, fd / 'a_dudv', out)


# --------------------------------------------------------------------------- #
# the n_TOF timing
# --------------------------------------------------------------------------- #
def fig_ntof(d: Path, fd: Path):
    """Arm-A wall and plastic dt_ns, on pair triggers, all triggers, control."""
    import matplotlib.pyplot as plt
    N = pd.read_csv(d / 'ntof_timing.csv')
    fams = list(dict.fromkeys(N.family))
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=len(fams))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        for name, col in (('pair triggers', '#D55E00'),
                          ('all triggers', '#0072B2'),
                          ('control', '#999999')):
            g = N[(N.family == fam) & (N['sample'] == name)].sort_values('dt_ns')
            if g.empty or g.n.sum() < 20:
                continue
            _step(ax, g.dt_ns, g.frac, lw=2.2, color=col,
                  label=f'{name} (n = {int(g.n.sum()):,})')
        ax.set_xlabel(r'$\Delta t$ to the DREAM trigger [ns]')
        ax.set_title(fam, loc='left', fontsize=figstyle.BASE_PT * 0.98, pad=6)
        ax.legend(fontsize=figstyle.BASE_PT * 0.8)
    axes[0].set_ylabel('fraction per 10 ns bin')
    figstyle.fig_title(fig, 'The arm-A scintillators are prompt on the pair '
                            'triggers, as they are on every trigger',
                   'grey: the n_TOF random-coincidence control, flat by '
                   'construction')
    return figstyle.save(fig, fd / 'a_ntof', N)


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def fig_effmap(d: Path, fd: Path):
    """The 2D efficiency map, campaign mean, and its run-to-run scatter."""
    import matplotlib.pyplot as plt
    M = pd.read_csv(d / 'eff_map_2d.csv')
    S = pd.read_csv(d / 'eff_map_stability.csv')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    m = M.groupby(['u_mid', 'v_mid']).eff_single.mean().reset_index()
    piv = m.pivot(index='v_mid', columns='u_mid', values='eff_single')
    im = axes[0].pcolormesh(piv.columns.to_numpy(), piv.index.to_numpy(),
                            piv.to_numpy(), cmap='viridis', shading='auto')
    fig.colorbar(im, ax=axes[0], label='P(1 track | tagged, seeded)',
                 fraction=0.046, pad=0.03)
    axes[0].set_xlabel('u [mm]')
    axes[0].set_ylabel('v [mm]')
    axes[0].set_aspect('equal')
    if not S.empty:
        p2 = S.pivot(index='v_mid', columns='u_mid', values='shape_sd')
        im2 = axes[1].pcolormesh(p2.columns.to_numpy(), p2.index.to_numpy(),
                                 p2.to_numpy(), cmap='magma_r',
                                 shading='auto')
        fig.colorbar(im2, ax=axes[1], label='run-to-run sd of the shape',
                     fraction=0.046, pad=0.03)
        axes[1].set_xlabel('u [mm]')
        axes[1].set_aspect('equal')
    figstyle.fig_title(fig, 'Detector A, two-dimensional and stable across '
                            'the campaign',
                   'u and v are offsets from the plane centre, the frame the '
                   'acceptance toy uses')
    return figstyle.save(fig, fd / 'a_effmap', {'map': M, 'stability': S})


def fig_twotrack(d: Path, fd: Path):
    """The measured two-track resolution, and what it does to the acceptance."""
    import matplotlib.pyplot as plt
    E = pd.read_csv(d / 'two_track_efficiency.csv')
    X = (pd.read_csv(d / 'two_track_efficiency_axis.csv')
         if (d / 'two_track_efficiency_axis.csv').exists() else pd.DataFrame())
    A = pd.read_csv(d / 'acceptance_pooled.csv')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    ax = axes[0]
    for axis, col in (('u', '#0072B2'), ('v', '#D55E00')):
        g = X[(X.axis == axis) & np.isfinite(X.eff)] if len(X) else X
        if not len(g):
            continue
        ax.errorbar(g.sep_mid, g.eff, yerr=g.err, fmt='o-', color=col,
                    capsize=3, label=f'in {axis}, other view > 100 mm')
    g = E[np.isfinite(E.eff)]
    ax.plot(g.sep_mid, g.eff, lw=1.6, ls='--', color='#999999',
            label='radial (superseded)')
    ax.axhline(1.0, color='#333333', lw=1.2, ls=':')
    ax.set_xlim(0, 260)
    ax.set_xlabel('separation in that view [mm]')
    ax.set_ylabel('P(both tracks found)')
    ax.legend(fontsize=figstyle.BASE_PT * 0.86, loc='lower right')
    ax = axes[1]
    for col, lab, c in (('acc', 'legs only', '#999999'),
                        ('acc_slope', '+ slope required', '#0072B2'),
                        ('acc_sep', '+ two-track resolution', '#009E73'),
                        ('acc_sep_slope', '+ both', '#D55E00')):
        if col not in A.columns:
            continue
        ax.plot(A.theta, 1e4 * A[col], lw=2.4, color=c, label=lab)
    ax.set_xlabel('opening angle [deg]')
    ax.set_ylabel(r'acceptance [$10^{-4}$]')
    ax.legend(fontsize=figstyle.BASE_PT * 0.86)
    figstyle.fig_title(fig, 'Each view loses a pair on its own, and that '
                            'reshapes the acceptance',
                       'measured per view with the other held above 100 mm; '
                       'the radial curve it replaces is dashed')
    return figstyle.save(fig, fd / 'a_twotrack',
                         {'axis': X, 'radial': E, 'acc': A})


def fig_vertex(d: Path, fd: Path):
    """How close the two lines come, and where -- real against event-mixed."""
    import matplotlib.pyplot as plt
    P = pd.read_parquet(d / 'pairs.parquet')
    fig, axes = figstyle.figure(figstyle.WIDE, ncols=2)
    out = {}
    for ax, (col, lab, hi) in zip(axes, (('dca_pair_mm',
                                          'closest approach of the two lines '
                                          '[mm]', 400.0),
                                         ('v_r_mm',
                                          'radius of that point from the beam '
                                          'axis [mm]', 600.0))):
        bins = np.linspace(0, hi, 61)
        rows = []
        for name, m, c in (('real', ~P.mixed.to_numpy(), '#0072B2'),
                           ('event-mixed', P.mixed.to_numpy(), '#999999')):
            v = P[col].to_numpy(float)[m]
            h, e = np.histogram(v, bins=bins)
            tot = h.sum()
            if tot == 0:
                continue
            _step(ax, 0.5 * (e[:-1] + e[1:]), h / tot, lw=2.4, color=c,
                  label=f'{name} (n = {tot:,})')
            rows.append(pd.DataFrame(dict(sample=name,
                                          x=0.5 * (e[:-1] + e[1:]), n=h,
                                          frac=h / tot)))
        ax.set_xlabel(lab)
        ax.legend(fontsize=figstyle.BASE_PT * 0.93)
        out[col] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    axes[0].set_ylabel('fraction per bin')
    figstyle.fig_title(fig, 'The two legs do not converge on the target, and '
                            'the mixed sample converges better',
                   'which is the double-track resolution talking, not the '
                   'physics')
    return figstyle.save(fig, fd / 'a_vertex', out)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('det_a_intra')
    paths.require(d / 'det_a_intra.meta.json', 'the det_a_intra products')
    fd = paths.figures('det_a_intra')
    figstyle.use()
    made = []
    for fn in (fig_spectrum, fig_fold, fig_dt0, fig_slope_profile,
               fig_dt_sep, fig_dudv, fig_ntof, fig_effmap, fig_twotrack,
               fig_vertex):
        try:
            made.append(fn(d, fd))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f'  skipped {fn.__name__}: {e}')
    print(f'\n{len(made)} figure(s) -> {fd}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
