#!/usr/bin/env python3
"""evd_style.py -- the look of every Paper I event display (strip position x drift time), on top of papers/paper_style.py.

Used by papers/paper1/figs/make_principle.py, papers/paper1/figs/make_two_particle_displays.py and the A1 loss displays
(rerun/paper1_figs/a1_displays.py), so the principle figure, the section 5 displays and the appendix read as one set:

Superseded for the paper's displays by evd.py (make_displays.py); kept for make_principle.py and the A1 contact sheets,
with evd.py's colours since 2026-11-03 (Dylan: zero white, a saturated not dark maximum, the reference solid black):

* amplitude: white -> saturated red (evd.STYLES['reds']), cells below 4 % of the view's 99.5 % quantile left white;
* M3 reference track: black, solid (a reference is black, papers/paper_style.py);
* the forward-model fit: blue, solid;
* a second fitted track (two-particle displays): green, solid;
* both lines carry a thin white halo so they read on the dark cells;
* clipped samples (>= 3550 ADC): amber outline; samples on the bottom rail after a clip: sky outline.
"""
import os
import sys

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import paper_style as ps  # noqa: E402

PITCH = 398.58 / 512           # readout strip pitch [mm]
TS = 60.0                      # sample period [ns]
S_SAT = 3550.0                 # clip level [ADC]
CMAP = LinearSegmentedColormap.from_list('w2red', ['#fde0d5', '#fcae91', '#fb6a4a', '#e8251f', '#d10f0f'])
M3_KW = dict(color='black', lw=1.1, ls='-')
FIT_KW = dict(color='#0a6fd6', lw=1.2, ls=(0, (4, 2)))
FIT2_KW = dict(color='#11a36b', lw=1.2, ls=(0, (4, 2)))
CLIP_C = ps.DET_COLOR['det6']
BOT_C = ps.DET_COLOR['det2']
HALO = [pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()]


def cells(ax, W, P, cen, zoom, vmax=None, frac=0.04, mark=False):
    """Draw the strip x sample cells of one view (W: nstrip x nsamp, P: strip positions [mm]) within |P - cen| < zoom.
    Returns the PatchCollection (for a colour bar) and the vmax used.  ``mark``: outline clipped / bottom-rail cells."""
    P = np.asarray(P, float)
    ok = np.isfinite(P)
    o = np.argsort(np.where(ok, P, np.inf))[:ok.sum()]
    P, W = P[o], np.asarray(W, float)[o]
    sel = np.abs(P - cen) < zoom
    if vmax is None:
        vmax = max(float(np.percentile(W[sel], 99.5)) if sel.any() else 100.0, 60.0)
    rects, vals, clip, bot = [], [], [], []
    for i in np.flatnonzero(sel):
        for k in range(W.shape[1]):
            val = float(W[i, k])
            r = (P[i] - PITCH / 2, k * TS)
            if mark and val >= S_SAT:
                clip.append(Rectangle(r, PITCH, TS))
            elif mark and val <= -200 and W[i].max() >= S_SAT:
                bot.append(Rectangle(r, PITCH, TS))
            if val < frac * vmax:
                continue
            rects.append(Rectangle(r, PITCH, TS))
            vals.append(min(val / vmax, 1.0))
    pc = PatchCollection(rects, cmap=CMAP, linewidths=0)
    pc.set_array(np.array(vals))
    pc.set_clim(0, 1)
    ax.add_collection(pc)
    for lst, col in ((clip, CLIP_C), (bot, BOT_C)):
        if lst:
            ax.add_collection(PatchCollection(lst, facecolor='none', edgecolor=col, linewidths=0.8))
    ax.set_facecolor('white')
    ax.grid(False)
    return pc, vmax


def track(ax, p0, w, t0, v_mm_ns, kw, label=None, gap=30.0):
    """A straight track in the display: position p0 at t0 (the mesh) moving w [mm/ns] over the gap's drift time."""
    ts = np.array([t0, t0 + gap / v_mm_ns])
    return ax.plot(p0 + w * (ts - t0), ts, label=label, path_effects=HALO, **kw)[0]


def m3_track(ax, mesh, tan, t0, v_mm_ns, label='M3 reference', gap=30.0):
    """The M3 track extrapolated through the gap: mesh crossing at t0, tan = dx/dz."""
    return track(ax, mesh, tan * v_mm_ns, t0, v_mm_ns, M3_KW, label=label, gap=gap)


def frame(ax, cen, zoom, view, tmax=1920, xlabel=True):
    ax.set_xlim(cen - zoom, cen + zoom)
    ax.set_ylim(tmax, 0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)
    if xlabel:
        ax.set_xlabel(f'{view} strip position [mm]')
