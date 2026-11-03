#!/usr/bin/env python3
"""evd.py -- the Paper I event display (2026-11-03): one look for every display in the paper.

Dylan's feedback on the 11-02 displays: fit lines dominated the data (thinner, dashed); the reference track must be
more obvious; plot ADC (or one normalisation for both views) rather than each view normalised to itself; try several
colour schemes; no gap between the X and Y views.  This module is the one place those choices live; the producers
(make_displays.py, make_principle.py) only pick events.

A display is two views side by side, sharing the drift-time axis, with no gap between them:

* x: strip position in that view [mm]; y: drift time [ns], **increasing upward** (the cathode is on top, as in the
  chamber), right axis: distance from the mesh z = v (t - t0) [mm];
* cells: one rectangle per strip x 60 ns sample, colour = ADC on **one scale for both views** (``vmax`` = the 99.5 %
  quantile of both views' samples inside the windows, a colour bar in ADC), cells below ``frac`` x vmax left blank;
* the drift gap: thin grey lines at the mesh (t0) and at the cathode (t0 + g / v), so a track visibly starts at the mesh
  and ends at the cathode;
* the reference ray (the M3 telescope, called "reference" in the paper): see ``REF_MODES``;
* the pass fit: a thin dashed line (first track ACCENT, a second track amber, a candidate it did not keep dotted).

Styles (``STYLES``) only change colours; ``set_style(name)`` picks one, ``DEFAULT`` is the paper's.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, PowerNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import paper_style as ps  # noqa: E402

PITCH = 398.58 / 512           # readout strip pitch [mm]
TS = 60.0                      # sample period [ns]
S_SAT = 3550.0                 # clip level [ADC]
GAP = 28.19                    # chamber A drift gap [mm] (sections/06_gas.tex)
SAT_C = '#E69F00'              # outline of samples on the ADC rail


def _trunc(name, lo=0.0, hi=1.0, n=256):
    import matplotlib
    c = matplotlib.colormaps[name](np.linspace(lo, hi, n))
    return ListedColormap(c, name=f'{name}_{lo}_{hi}')


def _from(colors, name):
    return LinearSegmentedColormap.from_list(name, colors)


NAVY = _from(['#e8eef8', '#9ab3dc', '#3d6bb5', ps.DET_COLOR['det3'], '#041a45'], 'navy')
#: name -> colours.  bg: axes face; cmap; ref / fit / fit2: line colours; gapc: mesh/cathode line colour; ink: text on
#: the axes (the dark style needs light text inside the frame).
STYLES = {
    'navy': dict(cmap=NAVY, bg='white', ref='black', fit=ps.ACCENT, fit2=ps.DET_COLOR['det6'], gapc='#9aa3ad',
                 label='white to navy (the 11-02 scheme), ADC'),
    'magma': dict(cmap=_trunc('magma_r', 0.0, 0.92), bg='white', ref='black', fit='#1f77b4', fit2='#17becf',
                  gapc='#9aa3ad', label='magma (white-yellow-red-black)'),
    'grey': dict(cmap=_trunc('Greys', 0.12, 1.0), bg='white', ref=ps.DET_COLOR['det3'], fit=ps.ACCENT,
                 fit2=ps.DET_COLOR['det6'], gapc='#b8bec6', label='grey data, coloured lines'),
    'viridis': dict(cmap=_trunc('viridis_r', 0.0, 1.0), bg='white', ref='black', fit=ps.ACCENT,
                    fit2='#e7298a', gapc='#9aa3ad', label='viridis (yellow to purple)'),
    'dark': dict(cmap=_trunc('inferno', 0.12, 1.0), bg='#0b0b12', ref='white', fit='#5ad1ff', fit2='#a6f28c',
                 gapc='#59606b', label='dark background, inferno'),
}
DEFAULT = 'navy'
S = dict(STYLES[DEFAULT])
#: how the reference ray is drawn: 'dashed' (a dashed line with a halo), 'band' (a translucent band plus a thin line),
#: 'ticks' (a thin dashed line and arrowheads outside the gap, where there is no charge to hide)
REF_MODES = ('dashed', 'band', 'ticks')
OPT = dict(ref_mode='dashed', gamma=1.0, frac=0.03, abut=True, equal=True)
TLIM = (150.0, 1550.0)


def set_style(name=None, **opt):
    S.clear()
    S.update(STYLES[name or DEFAULT])
    OPT.update(opt)


@dataclass
class Event:
    """Everything a display needs.  W/P per view: ADC (nstrip x nsamp) and strip positions [mm]; ref per view:
    (mesh position, tan = d pos / d z) or None; t0 per view [ns]; fits: dicts view, p0, w [mm/ns], t0, role
    ('first' | 'second' | 'candidate')."""
    eid: int
    W: dict
    P: dict
    v_mm: float                      # drift velocity [mm/ns]
    t0: dict
    ref: dict = field(default_factory=dict)
    fits: list = field(default_factory=list)
    title: str = ''
    split: float = 30.0              # window segments: split where nothing lies within this many mm


# --------------------------------------------------------------------------- drawing primitives
def _sorted(W, P):
    P = np.asarray(P, float)
    ok = np.isfinite(P)
    o = np.argsort(np.where(ok, P, np.inf))[:ok.sum()]
    return np.asarray(W, float)[o], P[o]


def window(ev, view, pad=6.0, min_half=10.0, split=30.0):
    """The view's window as a list of segments (centre, half-width) [mm]: the reference ray, every fit and the strips
    with charge, split where nothing lies within ``split`` mm (two particles far apart get one zoomed segment each,
    drawn as a broken axis)."""
    W, P = _sorted(ev.W[view], ev.P[view])
    pts = []                              # (lo, hi) extents
    r = ev.ref.get(view)
    if r is not None and np.isfinite(r[0]):
        pts.append(sorted([r[0], r[0] + GAP * r[1]]))
    for f in ev.fits:
        if f['view'] == view:
            pts.append(sorted([f['p0'], f['p0'] + f['w'] * GAP / ev.v_mm]))
    amp = W.max(1)
    lit = amp > 0.2 * np.nanmax(amp)
    for x in P[lit]:
        pts.append([x, x])
    pts.sort()
    segs = []
    for lo, hi in pts:
        if segs and lo - segs[-1][1] < split:
            segs[-1][1] = max(segs[-1][1], hi)
        else:
            segs.append([lo, hi])
    return [(0.5 * (lo + hi), max(0.5 * (hi - lo) + pad, min_half)) for lo, hi in segs]


def cells(ax, W, P, cen, half, vmax, frac=None):
    W, P = _sorted(W, P)
    sel = np.abs(P - cen) < half + PITCH
    frac = OPT['frac'] if frac is None else frac
    rects, vals, clip = [], [], []
    for i in np.flatnonzero(sel):
        for k in range(W.shape[1]):
            val = W[i, k]
            if val >= S_SAT:
                clip.append(Rectangle((P[i] - PITCH / 2, k * TS), PITCH, TS))
            if val < frac * vmax:
                continue
            rects.append(Rectangle((P[i] - PITCH / 2, k * TS), PITCH, TS))
            vals.append(val)
    norm = PowerNorm(OPT['gamma'], 0, vmax) if OPT['gamma'] != 1 else Normalize(0, vmax)
    pc = PatchCollection(rects, cmap=S['cmap'], norm=norm, linewidths=0, rasterized=True)
    pc.set_array(np.array(vals))
    ax.add_collection(pc)
    if clip and OPT.get('mark_sat', False):     # samples on the ADC rail (censored in the fit): outlined
        ax.add_collection(PatchCollection(clip, facecolor='none', edgecolor=SAT_C, linewidths=0.5, zorder=3))
    ax.set_facecolor(S['bg'])
    ax.grid(False)
    return pc


def _line(ax, p0, w, t0, v, ts, **kw):
    ts = np.asarray(ts, float)
    return ax.plot(p0 + w * (ts - t0), ts, **kw)[0]


def reference(ax, mesh, tan, t0, v, label='reference track'):
    """The reference ray through the gap (mesh at t0, cathode at t0 + g / v), drawn per OPT['ref_mode']."""
    w = tan * v
    t1 = t0 + GAP / v
    col = S['ref']
    halo = [pe.Stroke(linewidth=3.2, foreground=S['bg']), pe.Normal()]
    mode = OPT['ref_mode']
    if mode == 'band':
        ts = np.array([t0, t1])
        xs = mesh + w * (ts - t0)
        hw = 1.6   # mm, a visual width (about two strips), not an uncertainty
        ax.fill_betweenx(ts, xs - hw, xs + hw, color=col, alpha=0.13, lw=0, zorder=4)
        return _line(ax, mesh, w, t0, v, [t0, t1], color=col, lw=0.8, ls=(0, (5, 3)), zorder=5, label=label)
    if mode == 'ticks':
        h = _line(ax, mesh, w, t0, v, [t0, t1], color=col, lw=0.8, ls=(0, (5, 3)), zorder=5, label=label)
        ylo, yhi = sorted(ax.get_ylim())
        for ta, tb in ((t1 + 0.05 * (yhi - ylo), t1 + 0.16 * (yhi - ylo)), (t0 - 0.05 * (yhi - ylo), t0 - 0.16 * (yhi - ylo))):
            xa, xb = mesh + w * (ta - t0), mesh + w * (tb - t0)
            ax.annotate('', xy=(xa, ta), xytext=(xb, tb), zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4, mutation_scale=9, shrinkA=0, shrinkB=0))
        return h
    return _line(ax, mesh, w, t0, v, [t0, t1], color=col, lw=1.5, ls=(0, (4, 2.2)), zorder=5, path_effects=halo,
                 label=label)


FIT_LABEL = {'first': 'fit', 'second': 'fit, second track', 'candidate': 'fitted candidate, not kept'}


def fit(ax, f, v, label=None):
    col = S['fit'] if f['role'] == 'first' else S['fit2']
    ls = (0, (1.2, 1.6)) if f['role'] == 'candidate' else (0, (6, 2.5))
    t1 = f['t0'] + GAP / v
    return _line(ax, f['p0'], f['w'], f['t0'], v, [f['t0'], t1], color=col, lw=1.0, ls=ls, zorder=6,
                 label=label or FIT_LABEL[f['role']])


def gap_lines(ax, t0, v, label=False):
    for t, name, va in ((t0, 'mesh', 'top'), (t0 + GAP / v, 'cathode', 'bottom')):
        ax.axhline(t, color=S['gapc'], lw=0.6, ls='-', zorder=1)
        if label:
            ax.text(0.985, t + (-12 if va == 'top' else 12), name, transform=ax.get_yaxis_transform(), ha='right',
                    va=va, fontsize=6.5, color=S['gapc'], zorder=7)


# --------------------------------------------------------------------------- the display
def shared_vmax(ev, wins, q=99.5):
    vals = []
    for v, segs in wins.items():
        W, P = _sorted(ev.W[v], ev.P[v])
        for cen, half in segs:
            sel = np.abs(P - cen) < half
            if sel.any():
                vals.append(W[sel].ravel())
    allv = np.concatenate(vals) if vals else np.array([100.0])
    return float(min(max(np.percentile(allv[allv > 0], q) if (allv > 0).any() else 100.0, 150.0), S_SAT))


def equalise(ev, wins, panels_in, tlim=None, gaps_in=0.0):
    """Widen the segments so that 1 mm across the strips and 1 mm of drift (v x 1 ns) print the same length on panels
    of ``panels_in`` = (width, height) inches: the display is then a true picture of the track in the strip-depth
    plane.  Events wider than that keep their windows (the depth is then stretched; second return value False)."""
    tlim = tlim or TLIM
    span = (tlim[1] - tlim[0]) * ev.v_mm
    target = span * (panels_in[0] - gaps_in) / panels_in[1]
    segs = [(v, c, h) for v in 'xy' for c, h in wins[v]]
    need = sum(2 * h for _, c, h in segs)
    if need >= target:
        return wins, False
    extra = (target - need) / (2 * len(segs))
    return {v: [(c, h + extra) for c, h in wins[v]] for v in 'xy'}, True


def _break_marks(ax, side, color='#555555'):
    """Diagonal '//' marks on the left or right frame edge of a broken axis."""
    d = 0.012
    x = 0.0 if side == 'left' else 1.0
    kw = dict(transform=ax.transAxes, color=color, clip_on=False, lw=0.8, zorder=9)
    for y in (0.0, 1.0):
        ax.plot([x - d, x + d], [y - 2.5 * d, y + 2.5 * d], **kw)


def draw(fig, gs, ev, wins=None, vmax=None, tlim=None, ylabel=True, zaxis=True, colorbar=True, legend=True,
         view_labels=True, cbar_ax=None, xlabel=True):
    """The X and Y views of ``ev`` side by side in the SubplotSpec ``gs``, sharing the time axis.  A view with two
    particles far apart is a broken axis (one segment each, '//' between).  Every segment has the same mm scale.
    Returns dict(axes, views (view per axis), pc, vmax, wins, handles)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from matplotlib.ticker import FixedLocator, MaxNLocator
    tlim = tlim or TLIM
    wins = wins or {v: window(ev, v, split=ev.split) for v in 'xy'}
    vmax = vmax or shared_vmax(ev, wins)
    segs = [(v, c, h) for v in 'xy' for c, h in wins[v]]
    BRK = 0.06                                  # inches between segments of one view
    if OPT['equal']:
        bb = gs.get_position(fig)
        W_in, H_in = fig.get_size_inches()
        nb = len(segs) - 2
        wins, _ = equalise(ev, wins, (bb.width * W_in, bb.height * H_in), tlim, gaps_in=max(nb, 0) * BRK)
        segs = [(v, c, h) for v in 'xy' for c, h in wins[v]]
    # columns: segment, (break spacer), segment, ... ; views abut (or a small gap when OPT['abut'] is False)
    bb = gs.get_position(fig)
    mm_per_in = sum(2 * h for _, _, h in segs) / max(bb.width * fig.get_size_inches()[0] - (len(segs) - 2) * BRK, 0.5)
    cols, kinds = [], []
    for i, (v, c, h) in enumerate(segs):
        if i:
            if segs[i - 1][0] == v:
                cols.append(BRK * mm_per_in); kinds.append('brk')
            elif not OPT['abut']:
                cols.append(2 * BRK * mm_per_in); kinds.append('gap')
        cols.append(2 * h); kinds.append(i)
    sub = GridSpecFromSubplotSpec(1, len(cols), subplot_spec=gs, width_ratios=cols, wspace=0.0)
    axs, views = [], []
    t0m = np.nanmean([ev.t0[v] for v in 'xy'])
    pc = None
    for j, k in enumerate(kinds):
        if not isinstance(k, int):
            continue
        v, cen, half = segs[k]
        ax = fig.add_subplot(sub[0, j], sharey=axs[0] if axs else None)
        first_of_view = k == 0 or segs[k - 1][0] != v
        last_of_view = k == len(segs) - 1 or segs[k + 1][0] != v
        ax.set_xlim(cen - half, cen + half)
        ax.set_ylim(*tlim)
        pc = cells(ax, ev.W[v], ev.P[v], cen, half, vmax)
        gap_lines(ax, ev.t0[v], ev.v_mm, label=(v == 'y' and last_of_view))
        r = ev.ref.get(v)
        if r is not None and np.isfinite(r[0]):
            reference(ax, r[0], r[1], ev.t0[v], ev.v_mm)
        for f in sorted((f for f in ev.fits if f['view'] == v), key=lambda f: ['candidate', 'second', 'first'].index(f['role'])):
            fit(ax, f, ev.v_mm)
        for sname, sp in ax.spines.items():
            sp.set_visible(True)
            sp.set_color('#555555')
        if not first_of_view:
            ax.spines['left'].set_visible(False)
            _break_marks(ax, 'left')
        if not last_of_view:
            ax.spines['right'].set_visible(False)
            _break_marks(ax, 'right')
        if view_labels and first_of_view:
            ink = 'white' if S['bg'] != 'white' else ps.INK
            ax.text(0.04, 0.97, f'{v.upper()} view',
                    transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color=ink, fontweight='bold', zorder=8)
        ax.tick_params(direction='in', top=True, length=2.5)
        if axs:
            ax.tick_params(labelleft=False)
        lo, hi = cen - half, cen + half
        w_in = ax.get_position().width * fig.get_size_inches()[0]
        tk = MaxNLocator(max(2, min(5, int(w_in / 0.32))), steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        m = 0.1 * (hi - lo)
        ax.xaxis.set_major_locator(FixedLocator([t for t in tk if lo + m <= t <= hi - m]))
        axs.append(ax); views.append(v)
    if xlabel:   # one label per view, centred under its segments (gridspec positions are final here)
        for v in 'xy':
            a_v = [a for a, vv in zip(axs, views) if vv == v]
            p0, p1 = a_v[0].get_position(), a_v[-1].get_position()
            a_v[0].set_xlabel(f'${v}$ [mm]')
            if len(a_v) > 1:      # label offset below the axis: tick labels + pad, in axes fraction
                dy = 0.21 / (p0.height * fig.get_size_inches()[1])
                a_v[0].xaxis.set_label_coords((0.5 * (p0.x0 + p1.x1) - p0.x0) / p0.width, -dy)
    if ylabel:
        axs[0].set_ylabel('drift time [ns]')
    if zaxis:
        sec = axs[-1].secondary_yaxis('right', functions=(lambda t: (t - t0m) * ev.v_mm, lambda z: z / ev.v_mm + t0m))
        sec.set_ylabel('distance from mesh [mm]')
        sec.tick_params(direction='in', length=2.5)
    out = dict(axes=axs, views=views, pc=pc, vmax=vmax, wins=wins)
    if colorbar:
        cb = fig.colorbar(pc, cax=cbar_ax, ax=None if cbar_ax is not None else axs, pad=0.1 if zaxis else 0.02,
                          fraction=0.04, aspect=25)
        cb.set_label('amplitude [ADC]', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        out['cb'] = cb
    hs = {}
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    out['handles'] = hs
    return out


def legend(fig, handles, ax=None, **kw):
    order = ['reference track', 'fit', 'fit, second track', 'fitted candidate, not kept']
    hs = {k: handles[k] for k in order if k in handles}
    opts = dict(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(hs), frameon=False, fontsize=7.5,
                handlelength=2.6)
    opts.update(kw)
    tgt = ax if ax is not None else fig
    return tgt.legend(list(hs.values()), list(hs.keys()), **opts)
