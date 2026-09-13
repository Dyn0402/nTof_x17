#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_insitu_figures.py -- the in-situ performance figures for the Athens deck.

Five figures, meant to be shown in this order.  The first two are the ones
asked for; the third is what makes the second believable; the fourth is the
declared test; the fifth is chamber B.

  1. ``hitmap_triggered``   the trigger-biased maps -- A, C, D, on
     scintillator-matched tracks, with the wall groups and the plastic bars
     drawn on top so the reader sees the trigger's own footprint printed on the
     chamber rather than having to take it on trust.
  2. ``hitmap_unbiased``    the bystander maps -- **all four chambers**, read
     out on another arm's trigger.  Chamber B is on this figure and on no other,
     because this is the one map that needs no angle.
  3. ``hitmap_profiles``    the two samples projected on u and on v, per
     chamber, normalised to their own means.  This is the proof: the plastic
     bar-length cliff and the gap shadow at u ~ +7 mm are in the triggered
     projection and absent from the bystander one.
  4. ``shadow_test``        the prediction, declared in `insitu_maps` before it
     was run and answered here: the gap shadow is the TRIGGER's, so it must
     vanish in the bystander sample except where dead readout channels sit
     underneath it.  A loses its dip; C and D keep theirs.
  5. ``detector_b``         why B carries no angle, and why that is a field-cage
     fault and not a dead detector -- its map, its cluster shape against the
     other three, and the ring-chain current that identified it.

Everything comes from ``<out>/athens_insitu/``, which `insitu_maps.py` writes.
No number is recomputed here; this module is drawing only.

    python ntof_athens_26/make_insitu_figures.py
    python ntof_athens_26/make_insitu_figures.py --only hitmap_unbiased

On the Windows box point the tree at the drive letter first::

    X17_ROOT=D:/x17 python ntof_athens_26/make_insitu_figures.py
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
from matplotlib.colors import (LinearSegmentedColormap,          # noqa: E402
                               LogNorm)
from matplotlib.patches import Rectangle                          # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402
from mpgd26 import plotstyle as ps                                # noqa: E402
from ntof_athens_26 import insitu_maps as IM                      # noqa: E402

OUT = HERE / 'figures'
SRC = paths.spell('out', 'athens_insitu')

DASH = '\u2014'
RSQ = '\u2019'
APPROX = '\u2248'

#: The active area, mm from the plane centre (the strip map is 398.58 square).
ACTIVE = 199.29
#: Wall read-out group boundaries in u, and the plastic bars, from the DAQ
#: survey (`det_a_scint.layer_geometry`, identical in all 36 runs).  Projected
#: back onto the CHAMBER through the lever arm, which is what makes them
#: comparable with a cluster position: the wall sits 97.4 mm behind the strip
#: plane at a 234.6 mm throw from the target, so a feature at u on the wall
#: shadows u / LEVER on the chamber.
LEVER_WALL = (234.6 + 97.4) / 234.6
LEVER_PLAS = (234.6 + 190.6) / 234.6
WALL_EDGES_U = (-225.0, -125.0, -25.0, 75.0, 175.0)
#: Plastic bar centres and half-width in u, and the half-length in v.
PLAS_U = (-118.07, 85.37)
PLAS_HALF_U = 100.0
PLAS_HALF_V = 150.0

#: The shared relative colour scale, in units of the panel's own median.
REL_LO, REL_HI = 0.1, 10.0

#: A profile bin needs this many live cells before it is drawn: near the
#: fiducial edge and across D's dead bands a row can have almost nothing
#: left to read, and dividing by it turns Poisson noise into a spike.
MIN_LIVE_CELLS = 8


def chamber_cmap():
    """Occupancy ramp -- dark where nothing lands, warm where it does.

    Deliberately NOT the deck's efficiency ramp: that one runs bad-to-good and
    an occupancy map has no good end.  This one is sequential in lightness so
    it reads as a quantity, and it is smooth enough that the eye finds the dead
    stripes rather than the colour steps.
    """
    return LinearSegmentedColormap.from_list(
        'occ', ['#0d1117', '#1d2a44', '#2f4a6d', '#4a7089', '#7c9a93',
                '#b8b487', '#e0c07a', '#f5e6b8'])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(src: Path) -> dict:
    """Every table `insitu_maps` wrote, or a clear word about what is missing."""
    need = ('maps.parquet', 'profiles.csv', 'census.csv', 'shadow.csv',
            'shape.csv', 'masked_cells.csv', 'mask_census.csv',
            'trigger_census.csv', 'insitu_maps.meta.json')
    missing = [n for n in need if not (src / n).exists()]
    if missing:
        raise FileNotFoundError(
            f'{src} is missing {", ".join(missing)}\n'
            f'  run:  python ntof_athens_26/insitu_maps.py --jobs 6')
    T = dict(
        maps=pd.read_parquet(src / 'maps.parquet'),
        profiles=pd.read_csv(src / 'profiles.csv'),
        census=pd.read_csv(src / 'census.csv'),
        shadow=pd.read_csv(src / 'shadow.csv'),
        shape=pd.read_csv(src / 'shape.csv'),
        hot=pd.read_csv(src / 'masked_cells.csv'),
        mask_census=pd.read_csv(src / 'mask_census.csv'),
        trigger=pd.read_csv(src / 'trigger_census.csv'),
        meta=json.loads((src / 'insitu_maps.meta.json').read_text()))
    T['masks'] = {a: hot_grid(T['hot'], a) | IM.fiducial_mask()
              for a in T['maps'].arm.unique()}
    return T


def grid(maps: pd.DataFrame, arm: str, sample: str) -> np.ndarray:
    """The long table back into a (u, v) image, zero-filled."""
    d = maps[(maps.arm == arm) & (maps['sample'] == sample)]
    H = np.zeros((len(IM.CENTRES), len(IM.CENTRES)))
    if not len(d):
        return H
    H[np.searchsorted(IM.CENTRES, d.u.to_numpy()),
      np.searchsorted(IM.CENTRES, d.v.to_numpy())] = d.n.to_numpy()
    return H


def hot_grid(hot: pd.DataFrame, arm: str) -> np.ndarray:
    """The mask `insitu_maps.hot_mask` derived, as a boolean image."""
    M = np.zeros((len(IM.CENTRES), len(IM.CENTRES)), dtype=bool)
    d = hot[hot.arm == arm] if len(hot) else hot
    if len(d):
        M[np.searchsorted(IM.CENTRES, d.u.to_numpy()),
          np.searchsorted(IM.CENTRES, d.v.to_numpy())] = True
    return M


# --------------------------------------------------------------------------- #
# shared drawing
# --------------------------------------------------------------------------- #
def draw_map(ax, H: np.ndarray, arm: str, *, trigger_overlay: bool,
             mask: np.ndarray | None = None):
    """One chamber surface, in its own local (u, v), mm.

    TWO CHOICES, both forced by the data rather than by taste.

    **Relative to the chamber's own MEAN over live cells, on a log scale.**
    The four chambers differ by two orders of magnitude in absolute occupancy
    and the question here is about STRUCTURE, so an absolute scale would be
    four incomparable pictures.  The mean, not the median, and that is not
    arbitrary: a quarter of chamber D's x plane is dead, so its median live
    cell IS a dead cell (27 counts against a mean of 1 973) and normalising to
    it would report the working three quarters as uniformly hot.  The mean is
    the "if it were flat" reference `hit_maps.py` already uses for this view.
    Log, because the dynamic range demands it: the ratio of the 99th percentile
    cell to the median is ~5 in chambers A, B and C and **207 in chamber D**,
    whose occupancy is genuinely bimodal.  One shared scale, 0.1x to 10x, so a
    colour means the same thing in all four panels and D's condition is visible
    rather than hidden by a per-panel rescale.

    **Masked cells are flat grey, not zero.**  A hot channel, or a cell outside
    the fiducial where the plane fit rails, is a place the chamber cannot be
    read; painting it the colour of "nothing landed here" would be a different
    and false claim.
    """
    e = IM.EDGES
    G = H.astype(float).copy()
    if mask is not None:
        G[mask] = np.nan
    live = G[np.isfinite(G)]
    ref = live.mean() if len(live) else 1.0
    R = G / max(ref, 1e-9)

    cmap = chamber_cmap().copy()
    cmap.set_bad('#4a4f57')
    im = ax.imshow(np.ma.masked_invalid(R).T, origin='lower',
                   extent=[e[0], e[-1], e[0], e[-1]], cmap=cmap,
                   norm=LogNorm(vmin=REL_LO, vmax=REL_HI),
                   interpolation='nearest', aspect='equal')

    ax.add_patch(Rectangle((-ACTIVE, -ACTIVE), 2 * ACTIVE, 2 * ACTIVE,
                           fill=False, ec='#ffffff', lw=0.9, alpha=0.35))
    if trigger_overlay:
        for u in WALL_EDGES_U:
            ax.axvline(u / LEVER_WALL, color='#ff4f36', lw=0.9, ls=(0, (4, 3)),
                       alpha=0.75)
        for c in PLAS_U:
            for sgn in (-1, 1):
                ax.axvline((c + sgn * PLAS_HALF_U) / LEVER_PLAS,
                           color='#5ad0ff', lw=1.0, alpha=0.8)
        for sgn in (-1, 1):
            ax.axhline(sgn * PLAS_HALF_V / LEVER_PLAS, color='#5ad0ff', lw=1.0,
                       alpha=0.8)

    ax.set_xlim(e[0], e[-1])
    ax.set_ylim(e[0], e[-1])
    ax.set_xticks([-150, 0, 150])
    ax.set_yticks([-150, 0, 150])
    ax.tick_params(colors=ps.MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f'chamber {arm}', loc='left', color=ps.DET_COLOR[arm],
                 fontsize=13, fontweight='bold', pad=6)
    return im


def shared_colorbar(fig, im, axes) -> None:
    """One bar for the whole row -- the scale is shared, so the bar is too."""
    cb = fig.colorbar(im, ax=list(axes), fraction=0.020, pad=0.015,
                      shrink=0.63)
    cb.ax.tick_params(labelsize=8.5, colors=ps.MUTED)
    cb.outline.set_visible(False)
    cb.set_label('occupancy / this chamber' + RSQ + 's mean live cell',
                 color=ps.MUTED, fontsize=9.5)


def n_of(census: pd.DataFrame, arm: str, sample: str) -> int:
    d = census[(census.arm == arm) & (census['sample'] == sample)]
    return int(d.n_tracks.iloc[0]) if len(d) else 0


def thousands(n) -> str:
    return f'{int(n):,}'.replace(',', '\u2009')


def headline(fig, title: str, sub: str) -> None:
    """Title and deck, placed so they cannot collide at any figure height.

    Both are `fig.text` in figure coordinates above the axes, and both are laid
    out from the TOP downwards.  A `suptitle` is vertically centred on its
    anchor, so pairing one with a `fig.text` puts the deck through the title's
    descenders on any figure whose aspect changes -- which is exactly what the
    first version of this module did.
    """
    fig.text(0.006, 1.105, title, ha='left', va='top', fontsize=15,
             color=ps.INK, fontweight='bold')
    fig.text(0.006, 1.048, sub, ha='left', va='top', fontsize=10.5,
             color=ps.MUTED)


def preliminary(fig, x=0.995, y=1.105) -> None:
    """The badge PLAN.md sec 8 attaches to every reconstructed quantity."""
    fig.text(x, y, 'PRELIMINARY', ha='right', va='top', fontsize=10.5,
             fontweight='bold', color='#b04a3a', alpha=0.85,
             bbox=dict(boxstyle='round,pad=0.30', facecolor='#fdf1ef',
                       edgecolor='#e3bdb6', lw=0.8))


# --------------------------------------------------------------------------- #
# 1. the trigger-biased maps
# --------------------------------------------------------------------------- #
def fig_hitmap_triggered(T: dict):
    arms = ('A', 'C', 'D')
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0))
    for ax, arm in zip(axes, arms):
        im = draw_map(ax, grid(T['maps'], arm, 'triggered'), arm,
                      trigger_overlay=True, mask=T['masks'].get(arm))
        ax.set_xlabel('u  [mm]', color=ps.MUTED, fontsize=10)
        ax.text(0.02, 0.02,
                f'{thousands(n_of(T["census"], arm, "triggered"))} tracks',
                transform=ax.transAxes, color='#f5e6b8', fontsize=10,
                va='bottom')
    axes[0].set_ylabel('v  [mm]', color=ps.MUTED, fontsize=10)
    shared_colorbar(fig, im, axes)

    headline(fig,
             f'Where a matched track lands {DASH} and where the trigger '
             f'cannot look',
             'tracks whose extrapolation hits the wall group AND the plastic '
             f'bar that actually fired, in the chamber{RSQ}s own local frame')
    ps.note(fig,
            f'Red dashes: the four wall read-out groups.  Blue: the two '
            f'plastic bars{RSQ} edges in u and their 300 mm length in v '
            f'{DASH} both projected back onto the chamber through the lever '
            f'arm, which is what makes them comparable with a cluster '
            f'position.  The bars stop at |v| = {PLAS_HALF_V / LEVER_PLAS:.0f} '
            f'mm on the chamber and the gap between them shadows '
            f'u {APPROX} +7 mm; both are plainly in the data.  Colour is '
            f'occupancy relative to each chamber{RSQ}s own mean live cell on a '
            f'shared log scale, so a hue means the same thing in all three '
            f'panels.  Grey: hot channels, and the fiducial edge beyond which '
            f'the plane fit rails.  White square: the 398.6 mm active area.',
            y=0.02)
    preliminary(fig)
    fig.subplots_adjust(top=0.92, bottom=0.15, wspace=0.14)
    return fig, T['maps'][T['maps']['sample'] == 'triggered']


# --------------------------------------------------------------------------- #
# 2. the unbiased maps
# --------------------------------------------------------------------------- #
def fig_hitmap_unbiased(T: dict):
    arms = ('A', 'B', 'C', 'D')
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.9))
    for ax, arm in zip(axes, arms):
        im = draw_map(ax, grid(T['maps'], arm, 'bystander'), arm,
                      trigger_overlay=False, mask=T['masks'].get(arm))
        ax.set_xlabel('u  [mm]', color=ps.MUTED, fontsize=10)
        ax.text(0.02, 0.02,
                f'{thousands(n_of(T["census"], arm, "bystander"))} tracks',
                transform=ax.transAxes, color='#f5e6b8', fontsize=10,
                va='bottom')
    axes[0].set_ylabel('v  [mm]', color=ps.MUTED, fontsize=10)
    shared_colorbar(fig, im, axes)

    headline(fig,
             f'The same chambers, read out on somebody else{RSQ}s trigger',
             f'events in which THIS chamber{RSQ}s scintillators were silent '
             f'and another arm made the trigger {DASH} no cut from its own '
             f'acceptance, and no angle needed anywhere')
    ps.note(fig,
            f'The bar-length cliff and the gap shadow are gone: the whole '
            f'surface is lit, out to the active edge.  Chamber B is on this '
            f'figure and on no other {DASH} a bystander is defined by which '
            f'arm{RSQ}s scintillators fired, not by where a track points, so '
            f'it needs none of the drift-field geometry B cannot supply.  '
            f'Occupancy, not efficiency: the illumination is the beam{RSQ}s '
            f'own and is not flat, and nothing here confirms an individual '
            f'track.  Grey: hot channels, and the fiducial edge beyond which '
            f'the plane fit rails.  Chamber D is the exception and it is a '
            f'real result, not a display artefact: a quarter of its x plane is '
            f'dead and its hot channels carry a third of what is left, so its '
            f'unbiased map is the one that does not work.',
            y=0.02)
    preliminary(fig)
    fig.subplots_adjust(top=0.92, bottom=0.16, wspace=0.14)
    return fig, T['maps'][T['maps']['sample'] == 'bystander']


# --------------------------------------------------------------------------- #
# 3. the projections -- the proof
# --------------------------------------------------------------------------- #
def fig_hitmap_profiles(T: dict):
    """Both samples projected, each normalised to its own mean over the fiducial.

    THREE chambers, not four, and the omission is the point rather than an
    oversight: the comparison on this figure is trigger-matched AGAINST
    bystander, and chamber B has no trigger-matched sample at all -- it carries
    no angle, so no track of B's can be said to point at the channel that
    fired.  B appears on the unbiased map and on its own figure, where it has
    something to say.
    """
    arms = ('A', 'C', 'D')
    P = T['profiles']
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.2), sharey='row')
    rows = []
    for j, arm in enumerate(arms):
        for i, axis in enumerate(('u', 'v')):
            ax = axes[i, j]
            for sample, colour, lab in (
                    ('triggered', ps.TRACK, 'trigger-matched'),
                    ('bystander', ps.DET_COLOR[arm], 'bystander')):
                d = P[(P.arm == arm) & (P['sample'] == sample)
                      & (P.axis == axis)].sort_values('coord')
                if not len(d) or d.n.sum() == 0:
                    continue
                x = d.coord.to_numpy()
                y = d.n.to_numpy(float)
                live = d.n_live_cells.to_numpy(float)
                # per LIVE cell, so a masked channel reads as absent rather
                # than as a hole, and a bin with almost nothing left to read
                # is dropped rather than amplified into a spike.
                keep = live >= MIN_LIVE_CELLS
                r = np.full(len(x), np.nan)
                r[keep] = y[keep] / live[keep]
                m = np.nanmean(r)
                if not np.isfinite(m) or m <= 0:
                    continue
                ax.step(x, r / m, where='mid', color=colour, lw=1.6,
                        alpha=0.95 if sample == 'bystander' else 0.8, label=lab)
                rows.append(pd.DataFrame(dict(arm=arm, sample=sample,
                                              axis=axis, coord=x, rel=r / m)))
            if axis == 'u':
                ax.axvspan(IM.SHADOW_U[0], IM.SHADOW_U[1], color='#5ad0ff',
                           alpha=0.18, lw=0, zorder=0)
                for a, b in IM.FLANK_U:
                    ax.axvspan(a, b, color=ps.LINE, alpha=0.45, lw=0, zorder=0)
            else:
                for sgn in (-1, 1):
                    ax.axvline(sgn * PLAS_HALF_V / LEVER_PLAS, color='#5ad0ff',
                               lw=1.2, alpha=0.85, zorder=0)
            ax.axhline(1.0, color=ps.LINE, lw=0.8, zorder=0)
            ax.set_xlim(-IM.FIDUCIAL, IM.FIDUCIAL)
            ax.set_ylim(0, 2.5)
            ps.strip(ax)
            ax.tick_params(colors=ps.MUTED, labelsize=9)
            ax.set_xlabel(f'{axis}  [mm]', color=ps.MUTED, fontsize=10)
            if i == 0:
                ax.set_title(f'chamber {arm}', loc='left',
                             color=ps.DET_COLOR[arm], fontsize=12.5,
                             fontweight='bold', pad=6)
            if j == 0:
                ax.set_ylabel(f'along {axis}\nrelative occupancy',
                              color=ps.MUTED, fontsize=10)
    axes[0, 0].legend(fontsize=9.5, loc='lower left', labelcolor=ps.MUTED,
                      frameon=True, framealpha=0.9, facecolor=ps.SURFACE,
                      edgecolor='none')

    headline(fig, f'The trigger{RSQ}s footprint, and its absence',
             'occupancy per live cell, each curve normalised to its own mean, '
             'so the comparison is of shape and not of rate')
    ps.note(fig,
            f'Top: along u, with the plastic-gap shadow window shaded blue and '
            f'the two flank windows grey {DASH} the three windows the shadow '
            f'test reads.  Bottom: along v, with the bar-length limit marked.  '
            f'In the matched sample (red) both features are deep; in the '
            f'bystander sample they are gone in A, survive in C where ~10 dead '
            f'channels sit at the same u, and are unreadable in D, a quarter '
            f'of whose x plane is dead.  Chamber B is absent because it has no '
            f'trigger-matched sample to compare against {DASH} which is the '
            f'whole reason the bystander construction exists.  Bins with fewer '
            f'than {MIN_LIVE_CELLS} live cells are dropped rather than '
            f'amplified.',
            y=0.02)
    preliminary(fig)
    fig.subplots_adjust(top=0.92, bottom=0.16, hspace=0.34, wspace=0.16)
    return fig, pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 4. the declared test
# --------------------------------------------------------------------------- #
def fig_shadow_test(T: dict):
    S = T['shadow']
    arms = ('A', 'B', 'C', 'D')
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    w = 0.34
    for k, sample in enumerate(('triggered', 'bystander')):
        xs, ys, es = [], [], []
        for i, arm in enumerate(arms):
            d = S[(S.arm == arm) & (S['sample'] == sample)]
            if not len(d) or not np.isfinite(d.depth.iloc[0]):
                continue
            xs.append(i + (k - 0.5) * w)
            ys.append(float(d.depth.iloc[0]))
            es.append(float(d.depth_err.iloc[0]))
        ax.bar(xs, ys, width=w, yerr=es, capsize=3,
               color=ps.TRACK if sample == 'triggered' else '#3d6b8f',
               alpha=0.9, label=sample, error_kw=dict(lw=1.0, ecolor=ps.MUTED))
    ax.axhline(0.0, color=ps.INK, lw=1.0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f'chamber {a}' for a in arms], fontsize=11.5)
    ax.set_ylabel('depth of the plastic-gap dip\n(1 = fully blind, 0 = no dip)',
                  color=ps.MUTED, fontsize=10.5)
    ax.set_ylim(-0.6, 1.05)
    ps.strip(ax)
    ax.tick_params(colors=ps.MUTED, labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc='upper left', labelcolor=ps.MUTED)

    def depth(arm, sample):
        d = S[(S.arm == arm) & (S['sample'] == sample)]
        return float(d.depth.iloc[0]) if len(d) else np.nan

    ax.annotate('no dead channels:\nthe dip is gone',
                xy=(0.17, depth('A', 'bystander')), xytext=(0.45, -0.50),
                fontsize=10, color='#2e8b57',
                arrowprops=dict(arrowstyle='->', color='#2e8b57', lw=1.2))
    ax.annotate('dead readout channels under\nthe shadow: it stays',
                xy=(2.17, depth('C', 'bystander')), xytext=(2.34, 0.90),
                fontsize=10, color='#b04a3a', ha='left',
                arrowprops=dict(arrowstyle='->', color='#b04a3a', lw=1.2))

    headline(fig, 'The test, declared before it was run',
             f'if the dip at u {APPROX} +7 mm is the trigger{RSQ}s, it must '
             f'vanish when another arm does the triggering {DASH} except where '
             f'the chamber is genuinely blind')
    ps.note(fig,
            f'Depth = 1 \u2212 (occupancy per live cell in '
            f'{IM.SHADOW_U[0]:.0f}\u2026{IM.SHADOW_U[1]:.0f} mm) / (the same in '
            f'the two flanks), read only at |v| < {IM.SHADOW_V:.0f} mm so the '
            f'bar-length cut cannot leak in, and with hot channels masked in '
            f'both samples alike.  Chamber A has no dead runs; C has ~10 dead '
            f'channels and D ~130, both under this window (STATUS.md, '
            f'2026-09-08).  A{RSQ}s dip inverts because the beam itself '
            f'illuminates the centre more {DASH} which is what the trigger had '
            f'been carving a hole in.  Chamber B has no trigger-matched bar: no '
            f'angle, no match.',
            y=-0.02)
    preliminary(fig)
    fig.subplots_adjust(top=0.93, bottom=0.17)
    return fig, S


# --------------------------------------------------------------------------- #
# 5. chamber B
# --------------------------------------------------------------------------- #
#: The HV monitor, run_145, as `STATUS.md` records it after Dylan's 2026-09-08
#: correction.  A, C and D ground their degrader rings through three ~1.3 GOhm
#: resistors and it is that chain which draws the 0.18 uA; B has no chain, so
#: zero current is what it SHOULD read and the monitor says nothing about
#: whether B's cathode is at voltage.  It should be.
HV_ROWS = (('A', 0.180, 0.088), ('B', 0.000, 2.136),
           ('C', 0.180, 0.013), ('D', 0.180, 0.800))


def fig_detector_b(T: dict):
    SH = T['shape'].set_index('arm')
    arms = ('A', 'B', 'C', 'D')
    fig = plt.figure(figsize=(15.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.35], wspace=0.38)

    # (a) B works as a position detector
    ax = fig.add_subplot(gs[0, 0])
    draw_map(ax, grid(T['maps'], 'B', 'bystander'), 'B', trigger_overlay=False,
             mask=T['masks'].get('B'))
    ax.set_xlabel('u  [mm]', color=ps.MUTED, fontsize=10)
    ax.set_ylabel('v  [mm]', color=ps.MUTED, fontsize=10)
    ax.set_title('B sees the whole surface', loc='left',
                 color=ps.DET_COLOR['B'], fontsize=12.5, fontweight='bold',
                 pad=6)
    ax.text(0.02, 0.02,
            f'{thousands(n_of(T["census"], "B", "bystander"))} tracks',
            transform=ax.transAxes, color='#f5e6b8', fontsize=10, va='bottom')

    # (b) the fringing-field signature: wide and dilute
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(arms))
    wid = [float(SH.loc[a, 'width_x_vs_A']) for a in arms]
    den = [float(SH.loc[a, 'q_per_strip_x_vs_A']) for a in arms]
    ax.bar(x - 0.19, wid, width=0.36, color=[ps.DET_COLOR[a] for a in arms],
           alpha=0.92, label='cluster width (x)')
    ax.bar(x + 0.19, den, width=0.36, color=[ps.DET_COLOR[a] for a in arms],
           alpha=0.42, hatch='///', edgecolor='white', lw=0,
           label='charge per strip')
    ax.axhline(1.0, color=ps.INK, lw=1.0)
    for i in range(len(arms)):
        ax.text(i - 0.19, wid[i] + 0.05, f'{wid[i]:.2f}', ha='center',
                fontsize=9.5, color=ps.INK)
        ax.text(i + 0.19, den[i] + 0.05, f'{den[i]:.2f}', ha='center',
                fontsize=9.5, color=ps.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=11.5)
    ax.set_ylabel('relative to chamber A', color=ps.MUTED, fontsize=10.5)
    ax.set_ylim(0, 2.35)
    ps.strip(ax)
    ax.tick_params(colors=ps.MUTED, labelsize=10)
    ax.legend(fontsize=9.5, loc='upper left', labelcolor=ps.MUTED,
              frameon=True, framealpha=0.92, facecolor=ps.SURFACE,
              edgecolor='none')
    ax.set_title('wide AND dilute', loc='left', color=ps.INK, fontsize=12.5,
                 fontweight='bold', pad=6)

    # (c) the hardware, in the HV monitor
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('the cause, in the HV monitor', loc='left', color=ps.INK,
                 fontsize=12.5, fontweight='bold', pad=6)
    y = 0.93
    for lab, xx, ha in (('arm', 0.0, 'left'), ('drift I', 0.32, 'right'),
                        ('resistive I', 0.66, 'right'),
                        ('ring chain', 0.72, 'left')):
        ax.text(xx, y, lab, fontsize=9.5, color=ps.MUTED, fontweight='bold',
                ha=ha)
    for arm, di, ri in HV_ROWS:
        y -= 0.105
        bad = arm == 'B'
        col = '#b04a3a' if bad else ps.INK
        wgt = 'bold' if bad else 'normal'
        ax.text(0.0, y, arm, fontsize=11.5, color=ps.DET_COLOR[arm],
                fontweight='bold' if bad else 'normal')
        ax.text(0.32, y, f'{di:.3f} \u00b5A', fontsize=10.5, ha='right',
                color=col, fontweight=wgt)
        ax.text(0.66, y, f'{ri:.3f} \u00b5A', fontsize=10.5, ha='right',
                color=col, fontweight=wgt)
        ax.text(0.72, y, 'yes' if not bad else 'NONE', fontsize=10.5,
                color=col, fontweight=wgt)
    ax.text(0.0, y - 0.10,
            '700 V / 0.180 \u00b5A = 3.89 G\u03a9 = 3 \u00d7 1.30 G\u03a9 '
            '\u2014 the degrader\ndivider, and nothing else.  B draws zero '
            'because B has no\ndivider to draw it.  Without the rings the '
            'drift field fringes,\nso there is no clean time\u2194depth ladder '
            'and no angle \u2014 with\nthe cathode powered exactly as it '
            'should be.',
            fontsize=9.8, color=ps.INK, va='top', linespacing=1.55)
    ax.text(0.0, y - 0.42,
            'The amplification stage is untouched, so B still measures\n'
            'position and time. That is what the map on the left is,\n'
            'and it is where B belongs.',
            fontsize=9.8, color='#2e8b57', va='top', linespacing=1.55)

    headline(fig, 'Chamber B is a field-cage fault, not a dead detector',
             'it has no field-shaping ring chain, so it carries no angle '
             f'{DASH} and everything else about it works')
    ps.note(fig,
            f'Cluster shape: median over all gated clusters, the basis '
            f'`chamber_b.py` established on run_145 and this reproduces across '
            f'the campaign.  A fringing field spreads the same charge over more '
            f'strips, so the prediction is a wide, DILUTE cluster and not a '
            f'weak one {DASH} and that is exactly what separates B: widest of '
            f'the four, and clear of the other three on density.  The map is '
            f'the '
            f'bystander sample, which is the only one B can enter.  '
            f'B{RSQ}s resistive channel drawing 2.136 \u00b5A against '
            f'A{RSQ}s 0.088 is a separate anomaly and is still unexplained.',
            y=0.0)
    preliminary(fig)
    fig.subplots_adjust(top=0.90, bottom=0.17)
    return fig, T['shape']


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def save(fig, name: str, data: pd.DataFrame | None) -> None:
    """PNG for the slide, PDF for live text, CSV so the figure can be checked."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}')
        print(f'  -> {OUT / f"{name}.{ext}"}')
    plt.close(fig)
    if data is not None:
        data.to_csv(OUT / f'{name}.csv', index=False)
        print(f'  -> {OUT / f"{name}.csv"}')


FIGURES = {
    'hitmap_triggered': (fig_hitmap_triggered,
                         'The trigger-biased maps. Matched tracks only, with '
                         'the wall groups and plastic bars projected onto the '
                         'chamber.'),
    'hitmap_unbiased': (fig_hitmap_unbiased,
                        'The bystander maps \u2014 all four chambers, read out '
                        'on another arm\u2019s trigger. The unbiased view.'),
    'hitmap_profiles': (fig_hitmap_profiles,
                        'Both samples projected on u and v. The proof that the '
                        'trigger footprint is gone.'),
    'shadow_test': (fig_shadow_test,
                    'The declared test: the gap shadow vanishes where the '
                    'chamber is healthy and survives where it is not.'),
    'detector_b': (fig_detector_b,
                   'Chamber B \u2014 the missing ring chain, the wide dilute '
                   'clusters it predicts, and the map B can still make.'),
}

INDEX_CSS = """
body{margin:0;padding:32px 40px;background:#fbfcfe;color:#1b2430;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
h1{font-size:22px;margin:0 0 4px}p.lede{color:#6a7583;margin:0 0 28px;max-width:62em}
figure{margin:0 0 40px}figure img{width:100%;max-width:1280px;display:block;
 border:1px solid #e4e8ee;border-radius:6px;background:#fff}
figcaption{color:#6a7583;font-size:13px;margin-top:8px;max-width:62em}
code{background:#eef1f5;padding:1px 5px;border-radius:3px;font-size:12.5px}
"""


def write_index(built: list) -> None:
    rows = '\n'.join(
        f'<figure><img src="{n}.png" alt="{n}">'
        f'<figcaption><code>{n}.png</code> &middot; <code>{n}.pdf</code> '
        f'&middot; <code>{n}.csv</code><br>{cap}</figcaption></figure>'
        for n, cap in built)
    (OUT / 'insitu_index.html').write_text(
        '<!doctype html><meta charset="utf-8">'
        '<title>Athens deck \u2014 in-situ performance</title>'
        f'<style>{INDEX_CSS}</style>'
        '<h1>In-situ efficiency and performance</h1>'
        '<p class="lede">Built by <code>make_insitu_figures.py</code> from '
        '<code>insitu_maps.py</code>. Preview only \u2014 the deck uses the '
        'PDFs.</p>' + rows, encoding='utf-8')
    print(f'  -> {OUT / "insitu_index.html"}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(SRC), help='where insitu_maps wrote')
    ap.add_argument('--only', choices=sorted(FIGURES), nargs='*')
    a = ap.parse_args()

    ps.use()
    T = load(Path(a.src))
    print(f'{T["meta"]["n_tracks"]:,} tracks, {len(T["meta"]["runs"])} runs\n')

    built = []
    for name in (a.only or list(FIGURES)):
        fn, cap = FIGURES[name]
        print(name)
        fig, data = fn(T)
        save(fig, name, data)
        built.append((name, cap))
    write_index(built)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
