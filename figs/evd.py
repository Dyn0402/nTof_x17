#!/usr/bin/env python3
"""evd.py -- the Paper I event display: one look for every display in the paper.

History: 11-02 feedback (fits thinner and dashed, reference obvious, one amplitude scale for both views, views touching);
11-03 second pass (Dylan): keep the mesh and cathode lines; the reference a SOLID BLACK line running past them; fits
thin, dashed, in a colour distinct from both the data and the reference; zero amplitude WHITE; a saturated (not dark)
maximum; a linear amplitude scale, either normalised to the ADC full scale (4096) or to the largest amplitude in the
event; one display (X and Y views together) per row; broken axes only where a figure asks for them.  11-03 third pass
(Dylan's pick): jet with zero white, normalised to 4096; the dashed fits, like the reference, run across the whole time
axis (past the mesh and cathode).  This module is the
one place those choices live; the producers (make_displays.py, make_tpc_principle.py) only pick events and layouts.

A display is two views side by side, sharing the drift-time axis, with no gap between them:

* x: strip position in that view [mm]; y: drift time [ns], **increasing upward** (the cathode is on top, as in the
  chamber), right axis: distance from the mesh z = v (t - t0) [mm];
* cells: one rectangle per strip x 60 ns sample, colour = amplitude on **one linear scale for both views**
  (``OPT['norm']``: 'full' = ADC / 4096, 'event' = ADC / the event's largest sample in either view, 'adc' = raw ADC up
  to the 99.5 % quantile); samples below ``OPT['thr_adc']`` ADC are not drawn (white);
* the drift gap: thin grey lines at the mesh (t0) and at the cathode (t0 + g / v);
* the reference ray (the M3 telescope, "reference" in the paper): solid black, across the whole time axis;
* the pass fit: a thin dashed line (first track ``S['fit']``, a second track ``S['fit2']``, a candidate it did not
  keep dotted).

Styles (``STYLES``) only change colours; ``set_style(name, **opt)`` picks one, ``DEFAULT`` is the paper's.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
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


FULL_SCALE = 4096.0             # ADC full scale (12 bit)


def _jet():
    """jet without its dark ends (0.10 .. 0.88: blue .. pure red); zero is never drawn, so it stays white."""
    return _trunc('jet', 0.10, 0.88)


#: name -> colours.  cmap (its 0 is the threshold, its 1 the maximum; below-threshold cells are not drawn, so zero is
#: white in every style); ref / fit / fit2: line colours; gapc: mesh/cathode line colour.
STYLES = {
    'reds': dict(cmap=_from(['#fde0d5', '#fcae91', '#fb6a4a', '#e8251f', '#d10f0f'], 'w2red'), bg='white', ref='black',
                 fit='#0a6fd6', fit2='#11a36b', gapc='#8f98a3', label='white to red'),
    'greens': dict(cmap=_from(['#dcf2dd', '#a8e0a9', '#55bf63', '#16a03c', '#008c2e'], 'w2green'), bg='white',
                   ref='black', fit='#d1127a', fit2='#7a3fd1', gapc='#8f98a3', label='white to green'),
    'fire': dict(cmap=_from(['#fff2b0', '#ffe266', '#ffad2e', '#f45a1a', '#d81414'], 'w2fire'), bg='white', ref='black',
                 fit='#0a6fd6', fit2='#7a3fd1', gapc='#8f98a3', label='white to yellow to red'),
    'jet': dict(cmap=_jet(), bg='white', ref='black', fit='#e600e6', fit2='#7a4a1e', gapc='#8f98a3',
                label='jet (zero white)'),
}
DEFAULT = 'jet'                     # Dylan 11-03: jet, normalised to 4096 (OPT norm='full')
S = dict(STYLES[DEFAULT])
#: amplitude normalisations: 'full' = ADC / 4096 (every event on the same scale), 'event' = ADC / the event's largest
#: sample (X and Y together), 'adc' = raw ADC (the 99.5 % quantile of the windows as the top of the scale)
NORMS = ('full', 'event', 'adc')
NORM_LABEL = {'full': 'amplitude / full scale (4096 ADC)', 'event': 'amplitude / event maximum', 'adc': 'amplitude [ADC]'}
OPT = dict(norm='full', thr_adc=60.0, frac=0.03, gamma=1.0, abut=True, equal=True, stretch=1.0, mark_sat=False)
TLIM = (150.0, 1550.0)


_OPT0 = dict(OPT)


def set_style(name=None, **opt):
    """Pick a colour style (``DEFAULT`` if None) and reset OPT to its defaults plus ``opt``."""
    S.clear()
    S.update(STYLES[name or DEFAULT])
    OPT.clear()
    OPT.update(_OPT0)
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


def window(ev, view, pad=8.0, min_half=10.0, split=30.0, pad_frac=0.08, tlim=None):
    """The view's window as a list of segments (centre, half-width) [mm]: the reference ray, every fit and the strips
    with charge, split where nothing lies within ``split`` mm (two particles far apart get one zoomed segment each,
    drawn as a broken axis).  Margin past the outermost charge or line: ``pad`` mm plus ``pad_frac`` of the segment's
    extent, so nothing sits on the frame even in a wide window; the reference counts over the whole time axis (it is
    drawn across it)."""
    W, P = _sorted(ev.W[view], ev.P[view])
    pts = []                              # (lo, hi) extents
    r = ev.ref.get(view)
    if r is not None and np.isfinite(r[0]):
        tl = tlim or TLIM
        t0 = ev.t0[view]
        pts.append(sorted([r[0] + r[1] * ev.v_mm * (tl[0] - t0), r[0] + r[1] * ev.v_mm * (tl[1] - t0)]))
    for f in ev.fits:
        if f['view'] == view:
            pts.append(sorted([f['p0'], f['p0'] + f['w'] * GAP / ev.v_mm]))
    amp = W.max(1)
    lit = amp > 0.1 * np.nanmax(amp)
    for x in P[lit]:
        pts.append([x, x])
    pts.sort()
    segs = []
    for lo, hi in pts:
        if segs and lo - segs[-1][1] < split:
            segs[-1][1] = max(segs[-1][1], hi)
        else:
            segs.append([lo, hi])
    return [(0.5 * (lo + hi), max(0.5 * (hi - lo) * (1 + 2 * pad_frac) + pad, min_half)) for lo, hi in segs]


def cells(ax, W, P, cen, half, vmax, frac=None, scale=1.0, dt=0.0):
    """The strip x sample cells of one view within |P - cen| < half.  Colour = W / scale on a linear scale
    0 .. vmax / scale (``scale`` = 1: ADC, as make_tpc_principle.py calls it).  Samples below OPT['thr_adc'] ADC (or,
    if that is None, below ``frac`` x vmax) are not drawn, so zero is the white background."""
    W, P = _sorted(W, P)
    sel = np.abs(P - cen) < half + PITCH
    frac = OPT['frac'] if frac is None else frac
    thr = OPT['thr_adc'] if OPT.get('thr_adc') is not None else frac * vmax
    rects, vals, clip = [], [], []
    for i in np.flatnonzero(sel):
        for k in range(W.shape[1]):
            val = W[i, k]
            if val >= S_SAT:
                clip.append(Rectangle((P[i] - PITCH / 2, k * TS + dt), PITCH, TS))
            if val < thr:
                continue
            rects.append(Rectangle((P[i] - PITCH / 2, k * TS + dt), PITCH, TS))
            vals.append(val / scale)
    top = vmax / scale
    norm = PowerNorm(OPT['gamma'], 0, top) if OPT['gamma'] != 1 else Normalize(0, top)
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
    """The reference ray (mesh crossing at t0, tan = d pos / d z): a solid black line across the whole time axis of
    ``ax`` (set its ylim first), so it runs past the mesh and cathode lines."""
    w = tan * v
    ts = np.array(sorted(ax.get_ylim()))
    return _line(ax, mesh, w, t0, v, ts, color=S['ref'], lw=1.0, ls='-', zorder=5, label=label,
                 solid_capstyle='butt')


FIT_LABEL = {'first': 'fit', 'second': 'fit, second track', 'candidate': 'fitted candidate, not kept'}


def fit(ax, f, v, label=None, dt=0.0):
    col = S['fit'] if f['role'] == 'first' else S['fit2']
    ls = (0, (1.2, 1.6)) if f['role'] == 'candidate' else (0, (4, 2))
    ts = np.array(sorted(ax.get_ylim()))               # 11-03 Dylan: like the reference, across the whole time axis
    lw = 1.4 if f['role'] == 'candidate' else 1.0      # dotted: a little heavier, so it reads on the reference
    return _line(ax, f['p0'], f['w'], f['t0'] + dt, v, ts, color=col, lw=lw, ls=ls, zorder=6,
                 label=label or FIT_LABEL[f['role']], dash_capstyle='butt')


def gap_lines(ax, t0, v, label=False):
    for t, name, va in ((t0, 'mesh', 'top'), (t0 + GAP / v, 'cathode', 'bottom')):
        ax.axhline(t, color=S['gapc'], lw=0.6, ls='-', zorder=1)
        if label:
            ax.text(0.965, t + (-12 if va == 'top' else 12), name, transform=ax.get_yaxis_transform(), ha='right',
                    va=va, fontsize=6.5, color=S['gapc'], zorder=7,
                    bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none', alpha=0.75))


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


def event_max(ev):
    """The largest sample of the event, X and Y views together (the 'event' normalisation)."""
    return float(max(np.nanmax(np.asarray(ev.W[v], float)[np.isfinite(np.asarray(ev.P[v], float))]) for v in 'xy'))


def amp_scale(ev, wins, norm=None):
    """(vmax, scale) [ADC] of the colour scale for OPT['norm']: colours run over 0 .. vmax / scale."""
    norm = norm or OPT['norm']
    if norm == 'full':
        return FULL_SCALE, FULL_SCALE
    if norm == 'event':
        m = event_max(ev)
        return m, m
    return shared_vmax(ev, wins), 1.0


def colorbar(fig, pc, cax=None, axs=None, norm=None, orientation='vertical', **kw):
    """The amplitude colour bar, labelled for OPT['norm'] (0.0 .. 1.0 in steps of 0.2 for the normalised scales)."""
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    norm = norm or OPT['norm']
    cb = fig.colorbar(pc, cax=cax, ax=None if cax is not None else axs, orientation=orientation, **kw)
    cb.set_label(NORM_LABEL[norm], fontsize=7.5)
    if norm != 'adc':
        cb.locator = MultipleLocator(0.2)
        cb.formatter = FormatStrFormatter('%.1f')
        cb.update_ticks()
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    return cb


def equalise(ev, wins, panels_in, tlim=None, gaps_in=0.0):
    """Widen the segments so that 1 mm across the strips and 1 mm of drift (v x 1 ns) print the same length on panels
    of ``panels_in`` = (width, height) inches: the display is then a true picture of the track in the strip-depth
    plane.  Events wider than that keep their windows (the depth is then stretched; second return value False)."""
    tlim = tlim or TLIM
    span = (tlim[1] - tlim[0]) * ev.v_mm
    target = span * (panels_in[0] - gaps_in) / panels_in[1] / OPT.get('stretch', 1.0)
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
    wins = wins or {v: window(ev, v, split=ev.split, tlim=tlim) for v in 'xy'}
    vmax_n, scale = amp_scale(ev, wins)
    vmax = vmax or vmax_n
    if OPT['norm'] == 'adc':
        scale = 1.0
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
        # 11-03 (Dylan): the mesh and cathode lines must sit at the same height in X and Y.  The two views have their own
        # fitted t0 (they differ by tens of ns: separate FEUs / planes), so each view is drawn in time since ITS mesh
        # crossing, shifted onto the common t0m (the mean, which the right-hand distance axis already used).
        dt = t0m - ev.t0[v]
        pc = cells(ax, ev.W[v], ev.P[v], cen, half, vmax, scale=scale, dt=dt)
        gap_lines(ax, t0m, ev.v_mm, label=(v == 'y' and last_of_view))
        r = ev.ref.get(v)
        if r is not None and np.isfinite(r[0]):
            reference(ax, r[0], r[1], t0m, ev.v_mm)
        for f in sorted((f for f in ev.fits if f['view'] == v), key=lambda f: ['candidate', 'second', 'first'].index(f['role'])):
            fit(ax, f, ev.v_mm, dt=dt)
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
                    transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color=ink, fontweight='bold', zorder=8,
                    bbox=dict(boxstyle='square,pad=0.12', fc=S['bg'], ec='none', alpha=0.8))
        ax.tick_params(direction='in', top=True, length=2.5)
        if axs:      # only the first segment carries time ticks: none at a break or at the X|Y junction
            ax.tick_params(labelleft=False, left=False, right=False)
        lo, hi = cen - half, cen + half
        w_in = ax.get_position().width * fig.get_size_inches()[0]
        tk = MaxNLocator(max(2, min(6, int(w_in / 0.34))), steps=[1, 2, 5, 10]).tick_values(lo, hi)
        mm_in = (hi - lo) / max(w_in, 1e-3)            # mm per inch on this segment
        # keep tick labels away from the frame edges: 0.2 in from a break or the X|Y junction (labels of the
        # neighbouring segment sit there), 0.1 in from an outer edge
        m_lo = (0.2 if axs else 0.1) * mm_in
        m_hi = (0.2 if k < len(segs) - 1 else 0.1) * mm_in
        ax.xaxis.set_major_locator(FixedLocator([t for t in tk if lo + m_lo <= t <= hi - m_hi]))
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
    else:
        sec = None
    out = dict(sec=sec, axes=axs, views=views, pc=pc, vmax=vmax, scale=scale, wins=wins)
    if colorbar:
        kw = {} if cbar_ax is not None else dict(pad=0.1 if zaxis else 0.02, fraction=0.04, aspect=25)
        out['cb'] = globals()['colorbar'](fig, pc, cax=cbar_ax, axs=axs, **kw)
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
