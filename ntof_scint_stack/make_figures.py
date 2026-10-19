#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_scint_stack_figures.py -- the figures for the scintillator-stack report.

Reads only what `scint_stack_ana` wrote under ``<out>/scint_stack/ana``; every
PNG is written beside the CSV behind it (`figstyle.save`).

    python -m ntof_scint_stack.make_figures
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402
from ntof_scint_stack.ana import (  # noqa: E402
    LS_HALF_U, LS_HALF_V, MM_HALF_U, MM_HALF_V, PLAS_HALF_V, WALL_EDGES,
    WALL_HALF_V)

ARMS = ('A', 'B', 'C', 'D')


def ana() -> Path:
    return paths.spell('scint') / 'ana'


def figdir() -> Path:
    d = paths.spell('scint') / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d


def rd(name):
    return pd.read_parquet(ana() / f'{name}.parquet')


def geo():
    G = pd.read_csv(paths.spell('scint') / 'geometry.csv')
    return G.drop_duplicates('arm').set_index('arm')


def outline(ax, layer, g, arm):
    kw = dict(fill=False, lw=0.8, ec=fs.MUTED, zorder=5)
    if layer == 'wall':
        for lo, hi in zip(WALL_EDGES[:-1], WALL_EDGES[1:]):
            ax.add_patch(Rectangle((lo, -WALL_HALF_V), hi - lo, 2 * WALL_HALF_V,
                                   **kw))
    elif layer == 'plas':
        for n in (1, 2):
            c = g.loc[arm, f'plas_u_{n}']
            ax.add_patch(Rectangle((c - 100, -PLAS_HALF_V), 200,
                                   2 * PLAS_HALF_V, **kw))
    elif layer == 'liq':
        ax.add_patch(Rectangle((-LS_HALF_U, -LS_HALF_V), 2 * LS_HALF_U,
                               2 * LS_HALF_V, **kw))
        # where the two plastic bars sit, seen from the liquid's own centre
        for n in (1, 2):
            c = g.loc[arm, f'plas_u_{n}'] - g.loc[arm, 'u_ls']
            ax.add_patch(Rectangle((c - 100, -PLAS_HALF_V), 200,
                                   2 * PLAS_HALF_V, fill=False, lw=0.6,
                                   ec=fs.COPPER, ls='--', zorder=5))
    elif layer == 'mm':
        ax.add_patch(Rectangle((g.loc[arm, 'u_mm'] - MM_HALF_U, -MM_HALF_V),
                               2 * MM_HALF_U, 2 * MM_HALF_V, **kw))


def mesh(ax, M, val, bx, x0, y0, vmin, vmax, cmap='viridis', xcol='ix',
         ycol='iy'):
    """Draw a cell table (integer cell indices) as an image."""
    M = M[np.isfinite(M[val])]
    if not len(M):
        return None
    ix0, ix1 = M[xcol].min(), M[xcol].max()
    iy0, iy1 = M[ycol].min(), M[ycol].max()
    Z = np.full((iy1 - iy0 + 1, ix1 - ix0 + 1), np.nan)
    Z[M[ycol] - iy0, M[xcol] - ix0] = M[val]
    xe = x0 + bx * np.arange(ix0, ix1 + 2)
    ye = y0 + bx * np.arange(iy0, iy1 + 2)
    return ax.pcolormesh(xe, ye, Z, vmin=vmin, vmax=vmax, cmap=cmap,
                         shading='flat')


# --------------------------------------------------------------------------- #
def fig_stack():
    """Side view of one arm: the four layers and their lever arms."""
    g = geo().loc['A']
    fig, ax = fs.figure(figsize=(6.8, 2.6))
    layers = [('strip plane', 0.0, 1.0, fs.ACCENT),
              ('SiPM wall\n3 mm, 4 groups x 2 ends', g.L_wall, 3.0, '#0072B2'),
              ('plastic\n20 mm PVT, 2 bars', g.L_plas, 20.0, '#009E73'),
              ('liquid\n21 mm, 1 cell', g.L_ls, 21.2, '#CC79A7')]
    for name, w, th, c in layers:
        ax.add_patch(Rectangle((w - th / 2, -1), max(th, 2), 2, color=c,
                               alpha=0.75, lw=0))
        ax.text(w, 1.25, name, ha='center', va='bottom', fontsize=9,
                color=fs.INK)
        if w:
            ax.annotate(f'{w:.0f} mm', xy=(w, -1.25), ha='center', va='top',
                        fontsize=9, color=fs.MUTED)
    ax.annotate('', xy=(-40, 0), xytext=(-110, 0),
                arrowprops=dict(arrowstyle='->', color=fs.MUTED))
    ax.text(-112, 0.25, 'from the target', ha='left', fontsize=9,
            color=fs.MUTED)
    ax.set_xlim(-120, 300)
    ax.set_ylim(-2.2, 2.6)
    ax.set_yticks([])
    ax.set_xlabel('depth past the chamber strip plane, mm')
    ax.grid(False)
    fs.title(ax, 'Every track is walked back through three layers')
    fs.save(fig, figdir() / 'stack', data=pd.DataFrame(
        [dict(layer=n.split('\n')[0], lever_mm=w, thickness_mm=t)
         for n, w, t, _ in layers]))


def fig_t0():
    T = rd('t0_profile')
    meta = json.loads((ana() / 'meta.json').read_text())
    fig, axs = fs.figure(figsize=fs.FULL, nrows=2, ncols=2, sharex=True,
                         sharey=True)
    for ax, arm in zip(axs.flat, ARMS):
        st = fs.det_style(arm)
        for samp, ls, lab in (('all', '-', 'every trigger'),
                              ('unbiased', ':', 'another arm triggered, > 10 ms')):
            d = T[(T.arm == arm) & (T['sample'] == samp) & (T.n >= 100)]
            ax.plot(0.5 * (d.lo + d.hi), d.net, ls=ls, color=st['color'],
                    label=lab)
        lo, hi = meta['t0_window'][arm]
        ax.axvspan(lo, hi, color=fs.GRID, zorder=0)
        ax.text(0.02, 0.92, f'chamber {arm}: in time = [{lo:.0f}, {hi:.0f}) ns, '
                f'{100 * meta["frac_intime"][arm]:.0f} % of tracks',
                transform=ax.transAxes, fontsize=9, color=fs.INK)
        ax.set_xlim(-800, 1000)
        ax.set_ylim(-0.02, 0.85)
    axs[0, 0].legend(loc='center right', fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel('track t0 from its own drift times, ns')
    for ax in axs[:, 0]:
        ax.set_ylabel('confirmed by wall or plastic\n(net of accidentals)')
    fs.preliminary(axs[0, 1])
    fs.fig_title(fig, 'A third of the tracks crossed the chamber at another '
                 'time than the trigger',
                 'The chamber integrates its whole drift window; nothing in '
                 'the prompt scintillator window can confirm those tracks.')
    fs.save(fig, figdir() / 't0_profile', data=T)


def fig_confirm_map():
    M = rd('mask')
    g = geo()
    fig, axs = fs.figure(figsize=(9.6, 8.6), nrows=2, ncols=2)
    for ax, arm in zip(axs.flat, ARMS):
        m = M[M.arm == arm].copy()
        m['net_rel'] = m.net / m.arm_median
        m.loc[m.n < 30, 'net_rel'] = np.nan
        im = mesh(ax, m, 'net_rel', 20.0, -260, -260, 0, 1.3, cmap='viridis',
                  xcol='iu', ycol='iv')
        bad = m[m.masked & (m.v.abs() <= MM_HALF_V)]
        for _, r in bad.iterrows():
            ax.add_patch(Rectangle((r.u - 10, r.v - 10), 20, 20, fill=False,
                                   ec=fs.BAND_DEAD, lw=0.8, zorder=6))
        outline(ax, 'mm', g, arm)
        kept = 1 - (m.masked & (m.n >= 30)).sum() / max((m.n >= 30).sum(), 1)
        ax.set_title(f'chamber {arm} — {100 * (1 - kept):.0f} % of judged '
                     f'cells masked', fontsize=10, loc='left')
        ax.set_aspect('equal')
        ax.set_xlim(-250, 230)
        ax.set_ylim(-250, 250)
        ax.grid(False)
    for ax in axs[1]:
        ax.set_xlabel('u at the strip plane, mm')
    for ax in axs[:, 0]:
        ax.set_ylabel('v at the strip plane, mm')
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.02)
    cb.set_label('confirmation rate / arm median (in-time tracks)')
    fs.preliminary(axs[0, 1])
    fig.suptitle('Where on each chamber the tracks are not real: cells '
                 'outlined red are masked', x=0.01, ha='left', fontsize=11.5,
                 fontweight='bold')
    fs.save(fig, figdir() / 'confirm_map', data=M)


def fig_quality():
    Q = rd('confirm_quality')
    vars_ = [('t0', 'track t0, ns'), ('chi2dof_max', 'max chi2/dof'),
             ('pointing_y', 'tan_y x sign(v): + points from the target'),
             ('abs_tan_y', '|raw slope|, y plane')]
    fig, axs = fs.figure(figsize=fs.FULL, nrows=2, ncols=2, sharey=True)
    for ax, (var, lab) in zip(axs.flat, vars_):
        for arm in ARMS:
            d = Q[(Q.arm == arm) & (Q['var'] == var) & (Q['sample'] == 'all')]
            if not len(d):
                continue
            x = np.arange(len(d))
            ax.errorbar(x, d.net, yerr=d.err, capsize=0, ms=4,
                        **fs.det_style(arm))
            ax.set_xticks(x)
            ax.set_xticklabels([_lab(lo, hi) for lo, hi in zip(d.lo, d.hi)],
                               rotation=40, ha='right', fontsize=8)
        ax.set_xlabel(lab)
        ax.set_ylim(-0.02, 0.95)
    axs[0, 0].legend(fontsize=8.5, ncol=2)
    for ax in axs[:, 0]:
        ax.set_ylabel('confirmed (net)')
    fs.preliminary(axs[0, 1])
    fs.fig_title(fig, 'What an unconfirmed track looks like: off-time, a bad '
                 'fit, or a slope that does not point at the target',
                 'Single tracks reaching the wall and the plastic away from '
                 'their edges, every trigger; wall OR plastic, net of the '
                 'pre-trigger window.')
    fs.save(fig, figdir() / 'quality', data=Q)


def _lab(lo, hi):
    def f(v):
        if abs(v) >= 1e8:
            return '∞' if v > 0 else '−∞'
        return f'{v:g}'
    return f'[{f(lo)},{f(hi)})'


def fig_scales():
    """Edge width at the wall for each extrapolation, and the free fit's
    lambda against the campaign k.  The 2 Oct free raw-tan scale is drawn
    beside them for comparison."""
    P = rd('pointing')
    P = P[(P.fit_pass == 2) & (P.layer == 'wall') & (P.axis == 'u')]
    S = rd('scales')
    S = S[(S.fit_pass == 2) & (S.layer == 'wall') & (S.axis == 'u')]
    fig, ax = fs.figure(figsize=fs.FIG)
    vs = (('old', 'free raw-tan scale (2 Oct)', '#c99318'),
          ('k_only', 'bare imaging k', '#7d8796'),
          ('capsule', 'k, shrunk toward the capsule', '#c8601a'),
          ('used', 'k, calibrated on the wall (used)', '#1f5fa8'))
    rows = []
    for i, arm in enumerate(ARMS):
        for j, (v, lab, c) in enumerate(vs):
            if v == 'old':
                sg = float(S[S.arm == arm].sigma.iloc[0])
            else:
                sg = float(P[(P.arm == arm) & (P.variant == v)].sigma.iloc[0])
            ax.bar(i - 0.3 + 0.2 * j, sg, 0.18, color=c,
                   label=lab if i == 0 else None)
            rows.append(dict(arm=arm, variant=v, sigma=sg))
    for i, arm in enumerate(ARMS):
        r = P[(P.arm == arm) & (P.variant == 'used')].iloc[0]
        ax.text(i, 1.0, f'α {r.alpha:+.2f}\nλ {r.lam:.2f}', ha='center',
                va='bottom', fontsize=8, color='white')
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f'chamber {a}' for a in ARMS])
    ax.set_ylabel('wall edge width, mm (smaller = sharper)')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_ylim(0, 30)
    fs.preliminary(ax)
    fs.title(ax, 'How sharply each extrapolation finds the wall groups',
             'u_wall = u + L (α a_capsule + λ k tan_raw) − δ; single in-time '
             'tracks, good cells')
    fs.save(fig, figdir() / 'scales', data=pd.DataFrame(rows))


def fig_edges():
    E = rd('edge_profile')
    g = geo()
    fig, axs = fs.figure(figsize=(9.6, 7.0), nrows=3, ncols=4, sharey='row')
    cols = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for j, arm in enumerate(ARMS):
        ax = axs[0, j]
        d = E[(E.arm == arm) & (E.layer == 'wall')]
        for gg in range(4):
            x = d[d.what == f'g{gg}']
            ax.plot(x.x, x.p, color=cols[gg], lw=1.3, label=f'group {gg}')
        for b in WALL_EDGES:
            ax.axvline(b, color=fs.MUTED, lw=0.6, ls=':')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=10)
        ax.set_xlim(-260, 220)
        ax = axs[1, j]
        d = E[(E.arm == arm) & (E.layer == 'plas')]
        ax.plot(d.x, d.p, color=fs.INK, lw=1.3)
        gap = 0.5 * ((g.loc[arm, 'plas_u_1'] + 100) + (g.loc[arm, 'plas_u_2'] - 100))
        ax.axvline(gap, color=fs.MUTED, lw=0.6, ls=':')
        ax.set_xlim(-260, 220)
        ax = axs[2, j]
        for axis, ls in (('u', '-'), ('v', '--')):
            d = E[(E.arm == arm) & (E.layer == 'liq') & (E.axis == axis)]
            c0 = g.loc[arm, 'u_ls'] if axis == 'u' else g.loc[arm, 'v_ls']
            ax.plot(d.x - c0, d.p, ls=ls, color=fs.DET_COLOR[arm],
                    label=f'along {axis}')
        ax.axvline(-LS_HALF_U, color=fs.MUTED, lw=0.6, ls=':')
        ax.axvline(LS_HALF_U, color=fs.MUTED, lw=0.6, ls=':')
        gapl = gap - g.loc[arm, 'u_ls']
        ax.axvline(gapl, color=fs.COPPER, lw=0.8, ls='-.')
        ax.set_xlim(-300, 300)
        ax.set_xlabel('predicted position, mm')
    axs[0, 0].set_ylabel('share of lit group')
    axs[1, 0].set_ylabel('share that is bar 2 (R)')
    axs[2, 0].set_ylabel('liquid lit | wall+plastic')
    axs[0, 0].legend(fontsize=7.5, loc='center left')
    axs[2, 0].legend(fontsize=7.5)
    fs.preliminary(axs[0, 3])
    fs.fig_title(fig, 'With the fitted scale the wall groups and the plastic '
                 'gap switch where the survey puts them',
                 'Rows: wall groups (single group lit), plastic bar 2 share '
                 '(single bar lit), liquid response (u solid, v dashed; '
                 'dash-dot = plastic L/R gap). Single in-time tracks, > 10 ms.')
    fs.save(fig, figdir() / 'edges', data=E)


def fig_eff_summary():
    E = rd('eff')
    rows = [('wall', 'wany', 'wall | plastic'), ('plas', 'pm', 'plastic | wall'),
            ('liq', 'lf', 'liquid | wall+plastic')]
    fig, axs = fs.figure(figsize=fs.WIDE, ncols=3)
    for ax, (lay, pr, lab) in zip(axs, rows):
        for i, arm in enumerate(ARMS):
            st = fs.det_style(arm)
            for samp, dx, fill in (('unbiased', -0.13, st['color']),
                                   ('all_late', 0.13, 'none')):
                r = E[(E.arm == arm) & (E.layer == lay) & (E.probe == pr)
                      & (E['sample'] == samp)]
                if not len(r):
                    continue
                ax.errorbar(i + dx, r.eff.iloc[0], yerr=r.err.iloc[0],
                            marker=st['marker'], color=st['color'], mfc=fill,
                            mec=st['color'], ms=7, mew=1.2, ls='none')
        ax.set_xticks(range(4))
        ax.set_xticklabels(ARMS)
        ax.set_title(lab, loc='left', fontsize=10)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel('probability (net of accidentals)')
    axs[0].text(0.02, 0.04, 'filled: another arm triggered\nopen: every late '
                'trigger (self-triggered\nwall and plastic fire by design)',
                transform=axs[0].transAxes, fontsize=8, color=fs.MUTED)
    fs.preliminary(axs[2])
    fs.fig_title(fig, 'Tag and probe, layer by layer: the trigger hides most '
                 'of the inefficiency', 'Single in-time tracks in good chamber '
                 'cells, > 10 ms after the flash.')
    fs.save(fig, figdir() / 'eff_summary', data=E)


def fig_eff_maps(layer, samp, vmax=1.0, fname=None):
    M = rd('eff_map')
    M = M[(M.layer == layer) & (M['sample'] == samp)]
    g = geo()
    b = {'wall': 25.0, 'plas': 25.0, 'liq': 50.0}[layer] * (2 if samp == 'unbiased' else 1)
    x0, y0 = {'wall': (-250, -275), 'plas': (-250, -175), 'liq': (-250, -250)}[layer]
    figsize = (9.6, 4.4) if layer != 'plas' else (9.6, 3.4)
    fig, axs = fs.figure(figsize=figsize, ncols=4, sharey=True)
    im = None
    for ax, arm in zip(axs, ARMS):
        m = M[M.arm == arm]
        r = mesh(ax, m, 'eff', b, x0, y0, 0, vmax)
        im = r or im
        outline(ax, layer, g, arm)
        ax.set_title(f'chamber {arm}', loc='left', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('u on the layer, mm')
        lim = {'wall': ((-255, 205), (-280, 280)),
               'plas': ((-240, 210), (-180, 180)),
               'liq': ((-260, 260), (-260, 260))}[layer]
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    axs[0].set_ylabel('v on the layer, mm')
    if im is not None:
        cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
        cb.set_label('efficiency (net)')
    fs.preliminary(axs[-1])
    name = {'wall': 'Wall', 'plas': 'Plastic', 'liq': 'Liquid'}[layer]
    what = {'unbiased': 'another arm triggered, > 10 ms',
            'all_late': 'every trigger > 10 ms (trigger-biased high)'}[samp]
    fig.suptitle(f'{name} efficiency map — {what}', x=0.01, ha='left',
                 fontsize=11.5, fontweight='bold')
    fs.save(fig, figdir() / (fname or f'eff_map_{layer}_{samp}'), data=M)


def fig_liquid():
    E = rd('eff')
    L = rd('liq_vs_plas')
    fig, axs = fs.figure(figsize=fs.WIDE, ncols=2)
    ax = axs[0]
    for i, arm in enumerate(ARMS):
        st = fs.det_style(arm)
        for bar, dx, fill in ((1, -0.13, 'none'), (2, 0.13, st['color'])):
            r = E[(E.arm == arm) & (E.probe == f'lf_behind_bar{bar}')
                  & (E['sample'] == 'all_late')]
            ax.errorbar(i + dx, 100 * r.eff.iloc[0], yerr=100 * r.err.iloc[0],
                        marker=st['marker'], color=st['color'], mfc=fill,
                        mec=st['color'], ms=7, mew=1.2, ls='none')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'LIQ {a}' for a in ARMS])
    ax.set_ylabel('liquid lit | wall + plastic, %')
    ax.text(0.03, 0.9, 'open: behind plastic bar 1 (L)\nfilled: behind bar 2 (R)',
            transform=ax.transAxes, fontsize=8.5, color=fs.MUTED, va='top')
    ax.set_title('Behind the L bar vs behind the R bar', loc='left',
                 fontsize=10)
    ax = axs[1]
    for arm in ARMS:
        d = L[(L.arm == arm) & (L['sample'] == 'all_late')]
        x = 0.5 * (d.e_lo + np.minimum(d.e_hi, 14000)) / 1000
        ax.errorbar(x, 100 * d.eff, yerr=100 * d.err, **fs.det_style(arm))
    ax.set_xlabel('energy left in the plastic, MeVee (path-corrected)')
    ax.set_ylabel('liquid lit, %')
    ax.set_title('…and mostly when the plastic saw a through-going deposit',
                 loc='left', fontsize=10)
    ax.legend(fontsize=8.5)
    fs.preliminary(ax)
    fs.fig_title(fig, 'Liquids A and D respond mostly near their +u edge; '
                 'liquid C answers only to deposits above ~8 MeVee',
                 'Single in-time tracks confirmed by the wall (both ends) and '
                 'the plastic, > 10 ms; net of the pre-trigger window.')
    fs.save(fig, figdir() / 'liquid', data={'halves': E[E.layer == 'liq'],
                                            'vs_plastic': L})


def fig_gain_maps(layer, samp, unit_label, vrel=(0.7, 1.3), fname=None):
    M = rd('gain_map')
    q = {'wall': 'wall_gm', 'plas': 'plas_kevee'}[layer]
    M = M[(M.layer == layer) & (M.quantity == q) & (M['sample'] == samp)].copy()
    g = geo()
    b = 25.0 * (2 if samp == 'unbiased' else 1)
    x0, y0 = {'wall': (-250, -275), 'plas': (-250, -175)}[layer]
    figsize = (9.6, 4.4) if layer == 'wall' else (9.6, 3.4)
    fig, axs = fs.figure(figsize=figsize, ncols=4, sharey=True)
    im = None
    for ax, arm in zip(axs, ARMS):
        m = M[M.arm == arm].copy()
        med = np.nanmedian(m.med)
        m['rel'] = m.med / med
        r = mesh(ax, m, 'rel', b, x0, y0, *vrel, cmap='RdBu_r')
        im = r or im
        outline(ax, layer, g, arm)
        mtxt = f'{med:.0f}' if unit_label == 'keVee' else f'{med:.1f}'
        ax.set_title(f'{arm}: median {mtxt} {unit_label}', loc='left',
                     fontsize=9)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('u on the layer, mm')
        lim = {'wall': ((-255, 205), (-280, 280)),
               'plas': ((-240, 210), (-180, 180))}[layer]
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    axs[0].set_ylabel('v on the layer, mm')
    if im is not None:
        cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
        cb.set_label('response / arm median')
    fs.preliminary(axs[-1])
    name = {'wall': 'Wall MIP response, sqrt(top x bottom) x cos',
            'plas': 'Plastic response, keVee x cos'}[layer]
    what = {'unbiased': 'another arm triggered, > 10 ms',
            'all_late': 'every trigger > 10 ms'}[samp]
    fig.suptitle(f'{name} — {what}', x=0.01, ha='left', fontsize=11.5,
                 fontweight='bold')
    fs.save(fig, figdir() / (fname or f'gain_map_{layer}_{samp}'), data=M)


def fig_gain_summary():
    G = rd('gain')
    fig, axs = fs.figure(figsize=fs.WIDE, ncols=2)
    ax = axs[0]
    for i, arm in enumerate(ARMS):
        st = fs.det_style(arm)
        d = G[(G.arm == arm) & (G.layer == 'wall') & (G['sample'] == 'all_late')]
        ax.plot(i + np.array([-0.24, -0.08, 0.08, 0.24]), d['median'],
                marker=st['marker'], color=st['color'], ls='none', ms=7)
        for k, (_, r) in enumerate(d.iterrows()):
            ax.text(i + [-0.24, -0.08, 0.08, 0.24][k], r['median'] + 0.6,
                    f'g{k}', ha='center', fontsize=7, color=fs.MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'WAL{a}' for a in ARMS])
    ax.set_ylabel('MIP response, mV')
    ax.set_ylim(0, 42)
    ax.set_title('Wall, per read-out group', loc='left', fontsize=10)
    ax = axs[1]
    for i, arm in enumerate(ARMS):
        st = fs.det_style(arm)
        for samp, dx, fill in (('unbiased', -0.15, st['color']),
                               ('all_late', 0.0, 'none')):
            d = G[(G.arm == arm) & (G.layer == 'plas') & (G['sample'] == samp)
                  & G.channel.isin(['bar1', 'bar2'])]
            ax.plot(i + dx + np.array([-0.05, 0.05]), d['median'] / 1000,
                    marker=st['marker'], color=st['color'], mfc=fill,
                    mec=st['color'], ls='none', ms=7, mew=1.2)
        d = G[(G.arm == arm) & (G.layer == 'plas') & (G['sample'] == 'all_late')
              & G.channel.isin(['bar1_through', 'bar2_through'])]
        ax.plot(i + 0.17 + np.array([-0.04, 0.04]), d['median'] / 1000,
                marker='*', color=st['color'], ls='none', ms=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'PSS{a}' for a in ARMS])
    ax.set_ylabel('median deposit, MeVee')
    ax.set_ylim(0, 5)
    ax.set_title('Plastic, bars 1 and 2', loc='left', fontsize=10)
    ax.text(0.02, 0.97, 'filled: another arm triggered\nopen: every late '
            'trigger\nstar: liquid also lit (through-going)',
            transform=ax.transAxes, fontsize=8, color=fs.MUTED, va='top')
    fs.preliminary(ax)
    fs.fig_title(fig, 'WALA is 30 % low and WALD group 3 is 35 % low; the '
                 'trigger doubles the plastic median',
                 'Single in-time tracks, > 10 ms. A through-going particle '
                 'leaves 3.0-3.4 MeVee in 20 mm of PVT on every arm.')
    fs.save(fig, figdir() / 'gain_summary', data=G)


def fig_attenuation():
    A = rd('wall_atten')
    C = rd('wallpos_cal')
    fig, axs = fs.figure(figsize=fs.FULL, nrows=2, ncols=2, sharex=True)
    cols = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for ax, arm in zip(axs.flat, ARMS):
        d = A[A.arm == arm]
        for gg in range(4):
            for end, ls in ((1, '-'), (2, '--')):
                x = d[(d.grp == gg) & (d.end == end)]
                ax.plot(x.v, x['median'], ls=ls, color=cols[gg], lw=1.2,
                        label=f'g{gg}' if end == 1 else None)
        c = C[C.arm == arm]
        ax.set_title(f'WAL{arm}: attenuation {c.atten_mm.min():.0f}-'
                     f'{c.atten_mm.max():.0f} mm', loc='left', fontsize=10)
        ax.set_ylim(0, None)
    axs[0, 0].legend(fontsize=8, ncol=4, loc='lower center')
    for ax in axs[1]:
        ax.set_xlabel('predicted v along the bar, mm')
    for ax in axs[:, 0]:
        ax.set_ylabel('median amplitude x cos, mV')
    fs.preliminary(axs[0, 1])
    fs.fig_title(fig, 'Each end of each wall group, against where the track '
                 'crossed the bar', 'solid: end 1 (odd detn), dashed: end 2 '
                 '(even detn). On D the two ends are the other way round.')
    fs.save(fig, figdir() / 'attenuation', data=A)


def fig_wallpos():
    S = rd('wallpos_sample')
    R = rd('wallpos_res')
    fig, axs = fs.figure(figsize=fs.FULL, nrows=2, ncols=4)
    for j, arm in enumerate(ARMS):
        d = S[S.arm == arm]
        ax = axs[0, j]
        ax.hist2d(d.v_pred, d.v_lr, bins=[np.arange(-200, 201, 10),
                                          np.arange(-300, 301, 10)],
                  cmap='Greys', cmin=1)
        ax.plot([-200, 200], [-200, 200], color=fs.TRACK, lw=0.9)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-300, 300)
        ax.set_title(f'WAL{arm}', loc='left', fontsize=10)
        ax.set_xlabel('v from the MM track, mm')
        ax.grid(False)
        ax = axs[1, j]
        for est, c, lab in (('lr', fs.DET_COLOR[arm], 'amplitude ratio'),
                            ('dt', fs.MUTED, 'time difference')):
            r = d[f'v_{est}'] - d.v_pred
            ax.hist(r, bins=np.arange(-400, 401, 10), histtype='step',
                    color=c, lw=1.3, density=True, label=lab)
        s = R[(R.arm == arm) & (R.est == 'lr') & (R.by == 'all')].sigma.iloc[0]
        ax.text(0.97, 0.92, f'sigma = {s:.0f} mm', transform=ax.transAxes,
                fontsize=9, color=fs.INK, ha='right')
        ax.set_xlabel('wall v minus MM v, mm')
        ax.set_yticks([])
    axs[0, 0].set_ylabel('v from ln(top/bottom), mm')
    axs[1, 0].legend(fontsize=7.5, loc='upper left')
    fs.preliminary(axs[0, 3])
    fs.fig_title(fig, 'The wall measures v along its bars to ~5 cm from the '
                 'amplitude ratio; the time difference adds nothing',
                 'Calibrated per group on even events, shown on odd ones. The '
                 'width includes the MM prediction error, so it is an upper '
                 'limit.')
    fs.save(fig, figdir() / 'wallpos', data={'sample': S, 'res': R})


def fig_both_ends():
    B = rd('both_ends')
    BV = rd('both_ends_v')
    S = rd('wallpos_sample')
    C = rd('wallpos_cal')
    fig, axs = fs.figure(figsize=(9.6, 3.6), ncols=3)
    ax = axs[0]
    for i, arm in enumerate(ARMS):
        st = fs.det_style(arm)
        r = B[(B.arm == arm) & (B['sample'] == 'all_late')].iloc[0]
        ru = B[(B.arm == arm) & (B['sample'] == 'unbiased')].iloc[0]
        ax.plot(i - 0.1, 100 * (1 - r.keep_real_both), marker=st['marker'],
                color=st['color'], ms=7, ls='none', mfc='none', mew=1.2)
        ax.plot(i - 0.1, 100 * (1 - ru.keep_real_both), marker=st['marker'],
                color=st['color'], ms=7, ls='none')
        ax.plot(i + 0.1, 100 * (1 - r.keep_acc_both), marker='x',
                color=st['color'], ms=8, ls='none', mew=1.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel('removed by demanding both ends, %')
    ax.set_title('Real hits lost vs accidentals removed', loc='left',
                 fontsize=10)
    ax.set_ylim(-3, 65)
    ax.text(0.25, 0.97, 'circle: real hits lost (filled unbiased,\n'
            'open every late trigger)\ncross: accidentals removed',
            transform=ax.transAxes, fontsize=7.8, color=fs.MUTED, va='top')
    ax = axs[1]
    for arm in ARMS:
        d = BV[BV.arm == arm]
        ax.plot(d.v, 100 * (d.only1 + d.only2) / d['any'], **fs.det_style(arm))
    ax.set_xlabel('predicted v along the bar, mm')
    ax.set_ylabel('one-ended real hits, % of lit')
    ax.set_title('Where the one-ended hits are', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2]
    S = S.merge(C[['arm', 'grp', 'dt_slope', 'dt_icpt']], on=['arm', 'grp'])
    S['r'] = (S.dt - (S.dt_icpt + S.dt_slope * S.v_pred)).abs()
    w = np.array([2, 3, 5, 7, 10, 15, 20, 30, 40, 60])
    rows = []
    for arm in ARMS:
        r = S[S.arm == arm].r.to_numpy()
        k = [(r < x).mean() for x in w]
        ax.plot(w, 100 * (1 - np.array(k)), **fs.det_style(arm))
        rows += [dict(arm=arm, window_ns=x, lost=1 - y) for x, y in zip(w, k)]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('top-bottom coincidence half-window, ns')
    ax.set_ylabel('real hits lost, %')
    ax.set_title('…and what a time coincidence costs', loc='left', fontsize=10)
    fs.preliminary(ax)
    fs.fig_title(fig, 'Demanding a signal at both ends of the wall costs < 1 % '
                 'of real hits (2 % on A) and removes 15-55 % of accidentals')
    fs.save(fig, figdir() / 'both_ends', data={'summary': B, 'profile': BV,
                                               'window': pd.DataFrame(rows)})


def fig_stability():
    R = rd('by_run')
    fig, axs = fs.figure(figsize=(9.6, 5.2), nrows=3, sharex=True)
    for ax, col, lab in zip(axs, ('wall_gm_mV', 'plas_kevee', 'liq_amp_mV'),
                            ('wall MIP', 'plastic', 'liquid')):
        for arm in ARMS:
            d = R[R.arm == arm].sort_values('rn')
            y = d[col] / np.nanmedian(d[col])
            ax.plot(d.rn, y, **fs.det_style(arm), lw=1.0)
        ax.axhline(1, color=fs.MUTED, lw=0.7)
        ax.set_ylabel(f'{lab}\n/ arm median')
        ax.set_ylim(0.9, 1.1)
        ax.axvspan(128, 147, color=fs.GRID, zorder=0)
    axs[0].legend(fontsize=8, ncol=4, loc='lower left')
    axs[-1].set_xlabel('run (grey: the 3-5 Aug k-block)')
    fs.preliminary(axs[0])
    fs.fig_title(fig, 'The wall and the plastic hold their gain to a few per cent '
                 'across the campaign; the liquid scatters within its statistics',
                 'Median response per run, single in-time tracks > 10 ms; the '
                 'liquid median is in mV over tracks that lit it.')
    fs.save(fig, figdir() / 'stability', data=R)


def main() -> int:
    fs.use()
    fig_stack()
    fig_t0()
    fig_confirm_map()
    fig_quality()
    fig_scales()
    fig_edges()
    fig_eff_summary()
    for lay in ('wall', 'plas', 'liq'):
        for samp in ('unbiased', 'all_late'):
            fig_eff_maps(lay, samp, vmax=1.0 if lay != 'liq' else 0.2)
    fig_liquid()
    fig_gain_maps('wall', 'all_late', 'mV')
    fig_gain_maps('plas', 'unbiased', 'keVee')
    fig_gain_maps('plas', 'all_late', 'keVee')
    fig_gain_summary()
    fig_attenuation()
    fig_wallpos()
    fig_both_ends()
    fig_stability()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
