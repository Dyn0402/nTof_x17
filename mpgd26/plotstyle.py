#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotstyle.py -- the deck's house style for DATA plots.

`style.py` covers the 3-D renders.  This covers the matplotlib figures, so the
charts read as part of the same set: same ink and muted greys as the slide CSS,
same type scale, recessive axes, no chartjunk.

Colour rule (from the dataviz procedure, in order): the form is picked first,
colour is assigned by the job it does, and the categorical palette is VALIDATED
rather than eyeballed.  The four-detector palette below is the Okabe-Ito subset;
run through `dataviz/scripts/validate_palette.js` against the deck's #fbfcfe
surface it returns ALL CHECKS PASS, with two warnings that the plots discharge
explicitly:

  * CVD separation 7.6 dE (deutan) sits in the 6-8 floor band, legal only WITH
    secondary encoding -> every detector series carries its own marker shape AND
    a direct end-label, so identity never rests on hue alone.
  * #CC79A7 contrast 2.98:1 against the surface is just under 3:1 -> it is only
    ever used with a visible label attached, never as a bare fill.

The deck is deliberately light-only (the slide CSS defines one palette and no
`prefers-color-scheme` block), so there is no dark variant here either.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Palette -- mirrors the slide CSS custom properties
# --------------------------------------------------------------------------- #
INK = '#1b2430'
MUTED = '#6a7583'
LINE = '#d4d9e0'
SURFACE = '#fbfcfe'
ACCENT = '#8a3f8f'          # mx17 purple, the deck accent
COPPER = '#d18a44'          # caution / annotation accent
TRACK = '#ff4f36'           # one sharp highlight, used sparingly

# Categorical, fixed order, never cycled. det -> (colour, marker)
DET_COLOR = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}
DET_MARKER = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}

# Status / annotation fills, reserved and never reused as a series colour.
BAND_SIGNAL = '#8a3f8f'     # the thermal arrival window
BAND_DEAD = '#b04a3a'       # the blind region


def efficiency_cmap():
    """Bad -> good ramp for efficiency maps, in the deck's own inks.

    Sequential colour maps built for *quantities* (viridis, plasma) are the
    wrong tool here: efficiency has a good end and a bad end, and the reader is
    looking for departures from uniformity, not for a value.  This runs from
    the same red the loss budget reserves for genuine blindness, through the
    copper it uses for "our position, not their signal", to the green it uses
    for the answer -- so a bin's colour on the map means what the same colour
    means on the bar chart beside it.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'efficiency', ['#b04a3a', '#c9743f', '#d9a86a', '#cdc08d', '#8fb87a',
                       '#4f9c62', '#2e8b57'])


def use() -> None:
    """Apply the house rcParams.  Call once before plotting."""
    mpl.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12.5,
        'axes.labelcolor': INK,
        'axes.edgecolor': LINE,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': LINE,
        'grid.linewidth': 0.7,
        'grid.alpha': 0.7,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': 11,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'text.color': INK,
        'figure.dpi': 160,
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.18,
    })


def strip(ax, left=True, bottom=True) -> None:
    """Recessive frame: keep the two axes that carry a scale, drop the rest."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def title(ax, headline: str, sub: str | None = None) -> None:
    """Left-aligned headline + optional grey deck, matching the slide type."""
    ax.set_title(headline, loc='left', color=INK, pad=18 if sub else 10)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=11, color=MUTED)


def end_label(ax, x, y, text, color, dx=0.0, dy=0.0, **kw) -> None:
    """Direct series label -- the secondary encoding the palette check requires."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=11.5, fontweight='bold', va='center',
                ha=kw.pop('ha', 'left'), **kw)


def note(fig, text: str, y=-0.02) -> None:
    """Provenance line under the plot -- every figure says where it came from."""
    fig.text(0.0, y, text, ha='left', va='top', fontsize=9.5, color=MUTED,
             wrap=True)


def save(fig, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f'  -> {path}')
