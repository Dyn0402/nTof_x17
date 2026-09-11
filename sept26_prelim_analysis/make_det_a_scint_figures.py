#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_det_a_scint_figures.py -- figures for the arm-A scintillator confirmation.

Reads what `det_a_scint.py` wrote; computes nothing new.  Every figure ships its
numbers beside it as a CSV, as PLAN sec 7 requires.

THE COLOUR SCHEME IS NOT A TASTE.  The four wall groups and the two plastic bars
are IDENTITIES, not magnitudes, so they get a categorical palette assigned in
fixed spatial order and never cycled.  The order is
``#0072B2, #E69F00, #009E73, #8a3f8f`` -- Okabe-Ito plus this repo's own mx17
purple -- validated to CVD dE 11.4 on the worst adjacent pair and 24.2 in normal
vision against the house surface.  A pink fourth slot (the obvious choice, and
the first one tried) fails: it sits at dE 7.6 from the green next to it under
deuteranopia, which on a map whose whole point is telling two adjacent regions
apart is the one failure that matters.  Every categorical figure also carries a
direct label, so identity is never colour alone.

Match FRACTION is a magnitude and gets a single-hue sequential ramp instead.

    python -m sept26_prelim_analysis.make_det_a_scint_figures
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
from sept26_prelim_analysis.det_a_scint import (  # noqa: E402
    N_WALL_GROUP, PLASTIC_HALF_U, PLASTIC_HALF_V, SIPM_BAR_HALF_U,
    SIPM_HALF_V, WINDOWS)

#: The four wall read-out groups, in ascending u.  Validated; see the module
#: docstring before changing any one of them.
GROUP_COLOR = ['#0072B2', '#E69F00', '#009E73', '#8a3f8f']
#: The two plastic bars, detn 1 (-u) and detn 2 (+u).
PLASTIC_COLOR = {1: '#56B4E9', 2: '#D55E00'}
#: Magnitude, so one hue light -> dark.
SEQ = 'Blues'
#: Selections, in the order every figure reads them.
SEL_ORDER = ('all', 'fiducial', 'single', 'slope', 'pointing')
SEL_COLOR = {'all': '#6a7583', 'fiducial': '#8a3f8f', 'single': '#0072B2',
             'slope': '#E69F00', 'pointing': '#009E73'}


def _meta(d: Path) -> dict:
    return json.loads((d / 'det_a_scint.meta.json').read_text())


def _grid(M: pd.DataFrame, plane: str, col: str):
    """(u edges, v edges, 2D array) for one map column, NaN where absent."""
    g = M[M.plane == plane]
    us = np.sort(g.u.unique())
    vs = np.sort(g.v.unique())
    Z = np.full((len(vs), len(us)), np.nan)
    ui = {u: i for i, u in enumerate(us)}
    vi = {v: i for i, v in enumerate(vs)}
    for u, v, z in zip(g.u, g.v, g[col]):
        Z[vi[v], ui[u]] = z
    du = (us[1] - us[0]) if len(us) > 1 else 20.0
    dv = (vs[1] - vs[0]) if len(vs) > 1 else 20.0
    return (np.r_[us - du / 2, us[-1] + du / 2],
            np.r_[vs - dv / 2, vs[-1] + dv / 2], Z)


def _square(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)


# --------------------------------------------------------------------------- #
# the maps the question asked for
# --------------------------------------------------------------------------- #
#: Canvas for the square geometry maps.  NOT ``figstyle.WIDE``: these panels
#: carry ``aspect='equal'`` because they are maps of real millimetres, and on a
#: 13.3 x 5.0 in canvas an equal-aspect panel shrinks to a fraction of its cell
#: while ``tight_layout`` still lays the labels out around the CELL -- which is
#: what threw the y-label across the headline and the legend across the x-label
#: on the first pass.  Sized so a 400 x 400 mm panel is close to square in its
#: own cell, and the labels land where the axes are.
MAP2 = (8.6, 4.75)
MAP3 = (9.6, 3.9)


def _band_labels(ax, M, plane, col, colors, names, y_frac=0.94):
    """Direct-label each colour band on the map itself.

    The palette validation permits an adjacent CVD pair only with a secondary
    encoding, and on a categorical map a legend box is a poor one: the reader
    has to carry four swatches across the figure. The label sits in the band.
    """
    g = M[(M.plane == plane) & (M[col] >= 0)]
    lo, hi = ax.get_ylim()
    y = lo + y_frac * (hi - lo)
    for i, name in enumerate(names):
        u = g.u[g[col] == i]
        if not len(u):
            continue
        ax.text(float(u.median()), y, name, ha='center', va='top',
                color='white', fontsize=figstyle.BASE_PT * 0.89,
                fontweight='bold',
                path_effects=_stroke(colors[i]))


def _stroke(color):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=3.2, foreground=color)]


def fig_mm_channel(d: Path, fd: Path):
    """The chamber surface, coloured by WHICH scintillator confirmed it."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    M = pd.read_csv(d / 'maps.csv')
    meta = _meta(d)
    fig, axes = plt.subplots(1, 2, figsize=MAP2)
    for a in axes:
        figstyle.strip(a)

    for ax, col, colors, names, lay in (
            (axes[0], 'grp_mode', GROUP_COLOR,
             [f'group {g}' for g in range(N_WALL_GROUP)], 'SiPM wall'),
            (axes[1], 'plas_mode', [PLASTIC_COLOR[1], PLASTIC_COLOR[2]],
             ['bar L', 'bar R'], 'plastic')):
        ue, ve, Z = _grid(M, 'mm', col)
        G = M[M.plane == 'mm'].copy()
        if col == 'plas_mode':
            Z = np.where(Z > 0, Z - 1, np.nan)
            G[col] = np.where(G[col] > 0, G[col] - 1, -1)
        Z = np.where(Z >= 0, Z, np.nan)
        cm = ListedColormap(colors)
        nb = BoundaryNorm(np.arange(-0.5, len(colors)), cm.N)
        ax.pcolormesh(ue, ve, Z, cmap=cm, norm=nb, shading='flat')
        ax.set_xlabel('u on the strip plane [mm]')
        _square(ax)
        ax.set_title(lay, loc='left', fontsize=figstyle.BASE_PT * 1.02,
                     color=figstyle.INK)
        _band_labels(ax, G, 'mm', col, colors, names)
    axes[0].set_ylabel('v on the strip plane [mm]')
    figstyle.fig_title(
        fig, 'The chamber surface maps onto the scintillator channel behind '
             'it, and the map is the geometry',
        f'the channel that confirmed the most tracks in each 20 mm cell; '
        f'{meta["n_tracks"]:,} gated arm-A tracks, {len(meta["runs"])} runs')
    return figstyle.save(fig, fd / 'a_mm_channel', M[M.plane == 'mm'])


def fig_mm_fraction(d: Path, fd: Path):
    """The same surface, coloured by HOW OFTEN it is confirmed."""
    import matplotlib.pyplot as plt
    M = pd.read_csv(d / 'maps.csv')
    fig, axes = plt.subplots(1, 3, figsize=MAP3)
    for a in axes:
        figstyle.strip(a)
    panels = (('frac_match_w', 'SiPM wall', axes[0]),
              ('frac_match_p', 'plastic', axes[1]),
              ('grp_purity', 'wall group purity', axes[2]))
    for col, name, ax in panels:
        ue, ve, Z = _grid(M, 'mm', col)
        im = ax.pcolormesh(ue, ve, Z, cmap=SEQ, vmin=0, vmax=1,
                           shading='flat')
        ax.set_xlabel('u [mm]')
        _square(ax)
        ax.set_title(name, loc='left', fontsize=figstyle.BASE_PT * 1.02,
                     color=figstyle.INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=figstyle.BASE_PT * 0.86)
    axes[0].set_ylabel('v [mm]')
    figstyle.fig_title(
        fig, 'Confirmation is flat over the middle of the chamber and thins in '
             'pale stripes exactly on the channel boundaries',
        'left and centre: fraction of predictable tracks that point at a '
        'channel that fired -- the vertical pale lines sit on the wall group '
        'edges and on the plastic L/R gap, where a track is as often confirmed '
        'by the neighbour. right: how single-valued the wall group is per cell')
    return figstyle.save(fig, fd / 'b_mm_fraction', M[M.plane == 'mm'])


def _draw_wall(ax, meta):
    from matplotlib.patches import Rectangle
    gu = {int(k): v for k, v in meta['group_u'].items()}
    for g in range(N_WALL_GROUP):
        lo, _c, hi = gu[g]
        ax.add_patch(Rectangle((lo, -SIPM_HALF_V), hi - lo, 2 * SIPM_HALF_V,
                               fill=False, ec=GROUP_COLOR[g], lw=2.2,
                               zorder=5))
        ax.text(0.5 * (lo + hi), SIPM_HALF_V - 16, str(g), ha='center',
                va='top', color='white', fontsize=figstyle.BASE_PT * 0.98,
                fontweight='bold', zorder=6,
                path_effects=_stroke(GROUP_COLOR[g]))
        for b in range(4):
            u = lo + SIPM_BAR_HALF_U * (2 * b + 1)
            ax.plot([u + SIPM_BAR_HALF_U] * 2, [-SIPM_HALF_V, SIPM_HALF_V],
                    color=GROUP_COLOR[g], lw=0.6, alpha=0.45, zorder=4)


def _draw_plastic(ax, meta):
    from matplotlib.patches import Rectangle
    for n, u in ((1, meta['plas_u']['1']), (2, meta['plas_u']['2'])):
        ax.add_patch(Rectangle((u - PLASTIC_HALF_U, -PLASTIC_HALF_V),
                               2 * PLASTIC_HALF_U, 2 * PLASTIC_HALF_V,
                               fill=False, ec=PLASTIC_COLOR[n], lw=2.2,
                               zorder=5))
        ax.text(u, PLASTIC_HALF_V - 16, f'{"LR"[n - 1]} (detn {n})',
                ha='center', va='top', color='white',
                fontsize=figstyle.BASE_PT * 0.98, fontweight='bold', zorder=6,
                path_effects=_stroke(PLASTIC_COLOR[n]))


def fig_layer_projection(d: Path, fd: Path, plane: str, name: str, tag: str):
    """Where the tracks land ON one layer, against that layer's own segments."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    M = pd.read_csv(d / 'maps.csv')
    meta = _meta(d)
    fig, axes = plt.subplots(1, 2, figsize=MAP2)
    for a in axes:
        figstyle.strip(a)

    ue, ve, Z = _grid(M, plane, 'n')
    im = axes[0].pcolormesh(ue, ve, Z, cmap=SEQ, shading='flat')
    cb = fig.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.ax.tick_params(labelsize=figstyle.BASE_PT * 0.77)
    axes[0].set_title('tracks arriving', loc='left',
                      fontsize=figstyle.BASE_PT * 1.02, color=figstyle.INK)

    col = 'grp_mode' if plane == 'wall' else 'plas_mode'
    colors = (GROUP_COLOR if plane == 'wall'
              else [PLASTIC_COLOR[1], PLASTIC_COLOR[2]])
    ue2, ve2, Z2 = _grid(M, plane, col)
    if col == 'plas_mode':
        Z2 = np.where(Z2 > 0, Z2 - 1, np.nan)
    Z2 = np.where(Z2 >= 0, Z2, np.nan)
    cm = ListedColormap(colors)
    axes[1].pcolormesh(ue2, ve2, Z2, cmap=cm,
                       norm=BoundaryNorm(np.arange(-0.5, len(colors)), cm.N),
                       shading='flat')
    axes[1].set_title('channel that confirmed them', loc='left',
                      fontsize=figstyle.BASE_PT * 1.02, color=figstyle.INK)

    for ax in axes:
        (_draw_wall if plane == 'wall' else _draw_plastic)(ax, meta)
        ax.set_xlabel(f'u on the {name} [mm]')
        _square(ax)
    axes[0].set_ylabel(f'v on the {name} [mm]')
    lever = meta['lever_wall'] if plane == 'wall' else meta['lever_plas']
    figstyle.fig_title(
        fig, f'Projected onto the {name}, the confirmed channel follows the '
             f'segmentation drawn on top of it',
        f'every gated arm-A track extrapolated {lever:.0f} mm past the strip '
        f'plane; outlines are the surveyed active volumes from the DAQ config')
    return figstyle.save(fig, fd / tag, M[M.plane == plane])


# --------------------------------------------------------------------------- #
# the rates, the tolerance, the angle scale
# --------------------------------------------------------------------------- #
def fig_rates(d: Path, fd: Path):
    """Confirmation rate per selection, against its own accidental floor."""
    import matplotlib.pyplot as plt
    R = pd.read_csv(d / 'rates.csv')
    fig, axes = plt.subplots(1, 2, figsize=figstyle.WIDE)
    for a in axes:
        figstyle.strip(a)
    x = np.arange(len(SEL_ORDER))
    for ax, lay, name in ((axes[0], 'wall', 'SiPM wall'),
                          (axes[1], 'plas', 'plastic')):
        sig = R[(R.layer == lay) & (R.window == 'prod')].set_index('selection')
        ctl = R[(R.layer == lay) & (R.window == 'ctrl')].set_index('selection')
        off = R[(R.layer == lay) & (R.window == 'off')].set_index('selection')
        y = [sig.frac_match_pred.get(s, np.nan) for s in SEL_ORDER]
        ax.bar(x, y, width=0.62, color=[SEL_COLOR[s] for s in SEL_ORDER],
               zorder=3)
        for i, s in enumerate(SEL_ORDER):
            ax.text(i, y[i] + 0.02, f'{100 * y[i]:.0f}%', ha='center',
                    va='bottom', fontsize=figstyle.BASE_PT * 0.98,
                    color=figstyle.INK, fontweight='bold')
        ax.plot(x, [ctl.frac_match_pred.get(s, np.nan) for s in SEL_ORDER],
                'o--', color=figstyle.MUTED, lw=2, ms=9, zorder=4,
                label='accidental floor (is_control)')
        ax.plot(x, [off.frac_match_pred.get(s, np.nan) for s in SEL_ORDER],
                '^:', color=figstyle.COPPER, lw=2, ms=9, zorder=4,
                label='accidental floor (pre-trigger)')
        ax.set_xticks(x)
        ax.set_xticklabels(SEL_ORDER)
        ax.set_ylim(0, 1.0)
        ax.set_title(name, loc='left', fontsize=figstyle.BASE_PT * 1.02,
                     color=figstyle.INK)
        ax.legend(fontsize=figstyle.BASE_PT * 0.86, loc='upper left')
    axes[0].set_ylabel('confirmed / predictable')
    figstyle.fig_title(
        fig, 'Both layers confirm the tracks they should, far above two '
             'independently measured accidental floors',
        'the two floors are the same test in the slim’s own is_control '
        'sample and in a pre-trigger window of identical width, and they agree')
    return figstyle.save(fig, fd / 'e_rates', R)


def fig_edge(d: Path, fd: Path):
    """P(this group fired) against the predicted crossing -- the tolerance."""
    import matplotlib.pyplot as plt
    E = pd.read_csv(d / 'edge_profile.csv')
    W = pd.read_csv(d / 'edge_width.csv')
    meta = _meta(d)
    fig, ax = figstyle.figure((13.333, 6.0))
    for g in range(N_WALL_GROUP):
        p = E[E.group == g].sort_values('u')
        if not len(p):
            continue
        ax.errorbar(p.u, p.p, yerr=p.p_err, fmt='o-', ms=6, lw=2,
                    color=GROUP_COLOR[g], label=f'group {g}')
    gu = {int(k): v for k, v in meta['group_u'].items()}
    for g in range(N_WALL_GROUP):
        for e in (gu[g][0], gu[g][2]):
            ax.axvline(e, color=figstyle.MUTED, lw=1.4, ls=':', alpha=0.8,
                       zorder=1)
    s = meta.get('edge_sigma_mm', np.nan)
    ax.set_xlabel('predicted crossing u on the wall [mm]')
    ax.set_ylabel('P(this group fired)')
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=figstyle.BASE_PT * 0.96, ncol=4, loc='upper center')
    figstyle.fig_title(
        fig, f'The fired group switches over {s:.0f} mm, not the 1.8 mm the '
             f'fit errors claim',
        'single-track, single-group events only; grey lines are the surveyed '
        'group boundaries. This width, not the formal error, is the position '
        'tolerance the match test should use')
    return figstyle.save(fig, fd / 'f_edge_profile', {'': E, 'width': W})


def fig_angle_scale(d: Path, fd: Path):
    """The boundary shift against the track slope, at two lever arms."""
    import matplotlib.pyplot as plt
    V = pd.read_csv(d / 'edge_vs_tan.csv')
    meta = _meta(d)
    AS = meta.get('angle_scale') or {}
    AP = meta.get('angle_scale_plastic') or {}
    fig, ax = figstyle.figure((13.333, 6.0))
    for lay, name, color, fit in (
            ('wall', f'SiPM wall (lever {meta["lever_wall"]:.0f} mm)',
             '#0072B2', AS),
            ('plas', f'plastic (lever {meta["lever_plas"]:.0f} mm)',
             '#D55E00', AP)):
        p = V[(V.layer == lay) & V.u_fit.notna() & (V.u_fit_err < 30)]
        if not len(p):
            continue
        ax.errorbar(p.tan_mean, p.shift_mm, yerr=p.u_fit_err, fmt='o', ms=10,
                    lw=0, elinewidth=2, color=color, label=name, zorder=3)
        if np.isfinite(fit.get('slope_mm', np.nan)):
            xs = np.linspace(p.tan_mean.min(), p.tan_mean.max(), 20)
            ax.plot(xs, fit['slope_mm'] * xs + fit['intercept_mm'], '-',
                    color=color, lw=2.4, alpha=0.85, zorder=2)
    ax.axhline(0, color=figstyle.MUTED, lw=1.4, ls='--', zorder=1)
    ax.set_xlabel('mean in-plane slope tan of the tracks in the bin')
    ax.set_ylabel('boundary shift [mm]')
    ax.legend(fontsize=figstyle.BASE_PT * 0.96, loc='upper left')
    e = 100 * AS.get('eps', np.nan)
    ee = 100 * AS.get('eps_err', np.nan)
    figstyle.fig_title(
        fig, f'A surveyed boundary appears to move with the track’s own '
             f'slope, which only an angle-scale error can do',
        f'the slope of each line is eps x lever arm; the wall gives eps = '
        f'{e:+.0f} +- {ee:.0f} %. A survey or mapping error would shift every '
        f'bin alike and change only the intercept')
    return figstyle.save(fig, fd / 'g_angle_scale', V)


def fig_confusion(d: Path, fd: Path):
    """Predicted wall group against the group that actually fired.

    On a QUARTER canvas the three-line headline overwhelmed the 4x4 panel and
    the tight crop collapsed it to nothing; a 4x4 matrix needs a square-ish
    canvas of its own, not a slide strip.
    """
    import matplotlib.pyplot as plt
    C = pd.read_csv(d / 'confusion.csv').set_index('predicted')
    Z = C[[c for c in C.columns if c.isdigit()]].to_numpy(float)
    Zn = Z / Z.sum(1, keepdims=True)
    fig, ax = figstyle.figure((8.6, 6.2))
    im = ax.imshow(Zn, cmap=SEQ, vmin=0, vmax=1, origin='lower')
    for i in range(Zn.shape[0]):
        for j in range(Zn.shape[1]):
            ax.text(j, i, f'{100 * Zn[i, j]:.0f}', ha='center', va='center',
                    fontsize=figstyle.BASE_PT * 1.02,
                    color='white' if Zn[i, j] > 0.5 else figstyle.INK,
                    fontweight='bold')
    ax.set_xticks(range(Zn.shape[1]))
    ax.set_yticks(range(Zn.shape[0]))
    ax.set_xlabel('group that fired')
    ax.set_ylabel('group the track points at')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=figstyle.BASE_PT * 0.86)
    figstyle.fig_title(
        fig, 'The diagonal is the read-out order, measured not inherited',
        'row-normalised percentages, single-group events; a descending '
        'bar-to-detn map would put this population on the anti-diagonal')
    return figstyle.save(fig, fd / 'h_confusion', C.reset_index())


def fig_per_run(d: Path, fd: Path):
    """Is the confirmation rate stable over the campaign?"""
    import matplotlib.pyplot as plt
    H = pd.read_csv(d / 'headline.csv')
    H['n'] = H.run.str.split('_').str[1].astype(int)
    H = H.sort_values('n')
    fig, ax = figstyle.figure(figstyle.WIDE)
    for col, name, color, mk in (
            ('frac_match_wall', 'SiPM wall', '#0072B2', 'o'),
            ('frac_match_plas', 'plastic', '#D55E00', 's'),
            ('frac_ctrl_wall', 'wall accidental floor', figstyle.MUTED, '^')):
        ax.plot(H.n, H[col], mk + '-', color=color, lw=2, ms=8, label=name)
    blk = H[H.k_block]
    if len(blk):
        ax.axvspan(blk.n.min() - 0.5, blk.n.max() + 0.5,
                   color=figstyle.BAND_CONTROL, alpha=0.12, zorder=0)
        ax.text(0.5 * (blk.n.min() + blk.n.max()), 0.03,
                'the 128-147 k block', ha='center', va='bottom',
                color=figstyle.MUTED, fontsize=figstyle.BASE_PT * 0.93)
    ax.set_xlabel('run number')
    ax.set_ylabel('confirmed / predictable')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=figstyle.BASE_PT * 0.96, loc='upper left', ncol=3)
    figstyle.fig_title(
        fig, 'The confirmation rate over the campaign, run by run',
        'every gated arm-A track in each run; the shaded band is the 48-hour '
        'block in which the fitted angle scale k rises on all three arms')
    return figstyle.save(fig, fd / 'i_per_run', H)


def fig_residual(d: Path, fd: Path):
    """The continuous residual, against the error the fit claims."""
    import matplotlib.pyplot as plt
    R = pd.read_parquet(d / 'residuals.parquet')
    meta = _meta(d)
    fig, axes = plt.subplots(1, 2, figsize=figstyle.WIDE)
    for a in axes:
        figstyle.strip(a)
    b = np.arange(-250, 251, 10.0)
    tab = {}
    for ax, col, name in ((axes[0], 'res_u_wall', 'SiPM wall'),
                          (axes[1], 'res_u_plas', 'plastic')):
        x = R[col].dropna()
        h, _ = np.histogram(x, bins=b)
        ax.step(0.5 * (b[:-1] + b[1:]), h / max(h.sum(), 1), where='mid',
                lw=2.4, color='#0072B2')
        ax.set_xlabel(f'residual on the {name} [mm]')
        ax.set_title(f'{name}: median |res| '
                     f'{np.nanmedian(np.abs(x)):.0f} mm', loc='left',
                     fontsize=figstyle.BASE_PT * 1, color=figstyle.INK)
        tab[name.split()[-1]] = pd.DataFrame(
            {'u': 0.5 * (b[:-1] + b[1:]), 'n': h})
    axes[0].set_ylabel('fraction per 10 mm')
    sw = meta['sig_u_wall_median']
    figstyle.fig_title(
        fig, 'The residual is set by the read-out granularity, not by the '
             'track fit',
        f'single-channel events. The formal extrapolated fit error is '
        f'{sw:.1f} mm at the wall; the channels are 100 and 200 mm wide, and '
        f'that is what the width of these distributions is')
    return figstyle.save(fig, fd / 'j_residual', tab)


def fig_both(d: Path, fd: Path):
    """Two layers at different depths, agreeing or not, per selection."""
    import matplotlib.pyplot as plt
    B = pd.read_csv(d / 'both_layers.csv').set_index('selection')
    fig, ax = figstyle.figure(figstyle.WIDE)
    x = np.arange(len(SEL_ORDER))
    parts = (('both', 'both layers confirm', '#0072B2'),
             ('wall_only', 'wall only', '#E69F00'),
             ('plas_only', 'plastic only', '#009E73'),
             ('neither', 'neither', '#c9ced6'))
    bot = np.zeros(len(SEL_ORDER))
    for col, name, color in parts:
        y = np.array([B[col].get(s, np.nan) for s in SEL_ORDER])
        ax.bar(x, y, 0.62, bottom=bot, color=color, label=name, zorder=3,
               edgecolor=figstyle.SURFACE, linewidth=2)
        for i in range(len(x)):
            if y[i] > 0.07:
                ax.text(i, bot[i] + y[i] / 2, f'{100 * y[i]:.0f}%',
                        ha='center', va='center', color=figstyle.INK,
                        fontsize=figstyle.BASE_PT * 0.98, fontweight='bold')
        bot += np.nan_to_num(y)
    ax.set_xticks(x)
    ax.set_xticklabels(SEL_ORDER)
    ax.set_ylabel('fraction of tracks')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=figstyle.BASE_PT * 0.93, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.10))
    figstyle.fig_title(
        fig, 'Two independent layers at different depths confirm the same '
             'track more often than not',
        'and on the beam-pointing tracks they agree on three quarters of them; '
        'the same partition in the control window puts "both" below 1 %')
    return figstyle.save(fig, fd / 'd_both_layers', B.reset_index())


def fig_rail(d: Path, fd: Path):
    """The v rail: a fifth of the track table outside the chamber, unconfirmed."""
    import matplotlib.pyplot as plt
    from sept26_prelim_analysis.det_a_scint import FIDUCIAL_V
    V = pd.read_csv(d / 'v_profile.csv')
    R = pd.read_csv(d / 'rail_census.csv')
    fig, axes = plt.subplots(2, 1, figsize=(figstyle.FULL[0], 4.75), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    for a in axes:
        figstyle.strip(a)
    axes[0].fill_between(V.v, V.n, step='mid', color='#0072B2', alpha=0.25)
    axes[0].step(V.v, V.n, where='mid', lw=2.4, color='#0072B2')
    axes[0].set_ylabel('tracks per 10 mm')
    axes[1].step(V.v, V.frac_match, where='mid', lw=2.4, color='#0072B2',
                 label='confirmed by the wall')
    axes[1].step(V.v, V.frac_ctrl, where='mid', lw=2.0, ls='--',
                 color=figstyle.MUTED, label='accidental floor')
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel('confirmed / predictable')
    axes[1].set_xlabel('v of the reconstructed impact point [mm]')
    axes[1].legend(fontsize=figstyle.BASE_PT * 0.93, loc='lower center')
    for ax in axes:
        for s_ in (-FIDUCIAL_V, FIDUCIAL_V):
            ax.axvline(s_, color=figstyle.BAND_DEAD, lw=1.8, ls=':')
        ax.axvspan(V.v.min(), -FIDUCIAL_V, color=figstyle.BAND_DEAD,
                   alpha=0.07, zorder=0)
        ax.axvspan(FIDUCIAL_V, V.v.max(), color=figstyle.BAND_DEAD,
                   alpha=0.07, zorder=0)
    rr = R.set_index('sample')
    axes[0].annotate('the chamber is 340 mm tall;\nshaded is outside it',
                     xy=(-198, V.n.max() * 0.72), color=figstyle.BAND_DEAD,
                     fontsize=figstyle.BASE_PT * 0.98, fontweight='bold',
                     ha='left')
    figstyle.fig_title(
        fig, 'A fifth of the arm-A track table lands outside the chamber in v, '
             'and the scintillators do not confirm it',
        f'the fitted y position rails just past the active area; '
        f'{100 * rr.frac_of_all.get("in the v rail", np.nan):.0f} % of all '
        f'tracks sit in one 20 mm window there and are confirmed at '
        f'{100 * rr.frac_match_wall.get("in the v rail", np.nan):.0f} % '
        f'against {100 * rr.frac_match_wall.get("inside the active area", np.nan):.0f} % inside')
    return figstyle.save(fig, fd / 'k_v_rail', {'': V, 'census': R})


FIGURES = (
    ('a_mm_channel', fig_mm_channel),
    ('b_mm_fraction', fig_mm_fraction),
    ('c_wall_plane', lambda d, f: fig_layer_projection(
        d, f, 'wall', 'SiPM wall', 'c_wall_plane')),
    ('c2_plastic_plane', lambda d, f: fig_layer_projection(
        d, f, 'plas', 'plastic layer', 'c2_plastic_plane')),
    ('d_both_layers', fig_both),
    ('e_rates', fig_rates),
    ('f_edge_profile', fig_edge),
    ('g_angle_scale', fig_angle_scale),
    ('h_confusion', fig_confusion),
    ('i_per_run', fig_per_run),
    ('j_residual', fig_residual),
    ('k_v_rail', fig_rail),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=str(paths.out('det_a_scint')))
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    d = Path(paths.require(a.dir, 'the det_a_scint products'))
    fd = paths.figures('det_a_scint')
    figstyle.use()
    want = {x for x in a.only.split(',') if x}
    bad = 0
    for name, fn in FIGURES:
        if want and name not in want:
            continue
        print(f'{name}:')
        try:
            fn(d, fd)
        except Exception as e:
            bad += 1
            print(f'  FAILED -- {type(e).__name__}: {e}')
    print(f'\n-> {fd}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
