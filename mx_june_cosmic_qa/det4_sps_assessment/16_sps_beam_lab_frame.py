#!/usr/bin/env python3
"""
16_sps_beam_lab_frame.py — the H4 beam and det4, in the frame of the beam table.

Takes the beam spot reconstructed on the P2 BASKET board by `15_sps_beam_board.py`
and rotates it into the lab, using the P2 basket's stated mounting:

  * the fan sits with its wide (outer-radius) side DOWN and the apex / inner
    radius pointing straight UP, centred — i.e. the fan bisector is vertical;
  * there is a 130 mm gap from the lowest point of the P2 active area to the
    mechanical table top, which is the origin of height here.

det4 is then placed next to it in the orientation of `board_map_g_det4_rot90ccw.png`
— rotated 90 deg CCW so the live bands run HORIZONTALLY (detector-local X is
vertical, detector-local Y is horizontal and increases to the left) — resting on
the bare edge of its PCB, not the active area:

  local X = -20.32 mm (the PCB edge) sits on the table  ->  height = local X + 20.32

and the drawing carries the 38 mm live band (local X 177-215) and the readout
square you get from the four highlighted cables X4+X5 and Y4+Y5 (local
149.76-248.82 mm on both axes, 99.06 mm square, 256 channels).

Products (written next to this script):
  sps_beam_lab_frame.png
  sps_beam_lab_frame.json

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/16_sps_beam_lab_frame.py
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.collections import PolyCollection            # noqa: E402
from matplotlib.patches import Ellipse, Rectangle            # noqa: E402
import matplotlib.colors as mcolors                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_b = import_module('15_sps_beam_board')

TABLE_GAP = 130.0          # table top -> lowest point of the P2 active area

# det4, detector-local mm (see 14_board_map.py)
D4_PCB = (-20.32, 449.68)          # PCB outline, both axes
D4_ACTIVE = (0.0, 399.36)          # metallised square
D4_BAND = (177.0, 215.0)           # the live band, local X
D4_SQUARE = (149.76, 248.82)       # X4+X5 / Y4+Y5, both axes
D4_CABLES = 'X4+X5, Y4+Y5'
# connector k spans 49.92*(k-1) .. +49.14 mm in the coordinate it measures
D4_CONN = [(k, 49.92 * (k - 1), 49.92 * (k - 1) + 49.14) for k in range(1, 9)]
D4_HILITE = (4, 5)
# Both mezzanine banks sit in the WIDE (50.32 mm) margins, on the +X and +Y
# edges: DFS3498A_det.gbr puts all 32 mezzanine mounting holes at local
# 425.18-440.18 mm on one axis (16 on each edge). The narrow 20.32 mm margins on
# the -X/-Y edges are bare, and one of those is what det4 rests on.
D4_MEZZ = (425.18, 440.18)         # mezzanine hole band, across-edge coordinate
D4_BANK = (423.0, 447.0)           # where the cards are drawn: over the holes,
                                   # outboard against the PCB edge at 449.68
SHIM_MM = 30.0                     # height gained under det4, 2026-07-31

INK, MUTED, BEAM, DET4 = _b.INK, _b.MUTED, _b.BEAM, _b.DET4
TABLE = '#6b5b4a'


def to_lab(p, apex, bisector_deg, h_apex, mirror=-1):
    """Pad frame -> lab. The fan bisector is rotated onto straight-down, the
    apex sits at height h_apex, and the fan is centred on x = 0.

    `mirror` sets which side the scene is viewed from. The P2 pad frame and the
    det4 Gerber frame are each drawn from their own board's side, so their
    handedness is not tied together by anything in the data; the convention here
    is the beam's-eye view (looking downstream, beam into the page) with det4's X
    connector bank on the right, which fixes mirror = -1 for P2."""
    th = math.radians(-90.0 - bisector_deg)
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(p, float) - apex
    x = d[..., 0] * c - d[..., 1] * s
    y = d[..., 0] * s + d[..., 1] * c
    return np.stack([mirror * x, y + h_apex], axis=-1)


def _overlap(lo, hi, a, b):
    """Fraction of the interval [lo, hi] that lies inside [a, b]."""
    return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None) / (hi - lo)


def slab_fraction(ext, w, a, b, axis=1):
    """Weight fraction inside a slab, with each pad spread over its true extent.

    Treating a pad as a point quantises any band scan at the 12 mm pad pitch;
    spreading its counts uniformly across the pad removes that and is the right
    model anyway — a pad's hits are distributed over its area, not at its centre.
    """
    lo, hi = ext[axis]
    return float((w * _overlap(lo, hi, a, b)).sum() / w.sum())


def box_fraction(ext, w, ya, yb, xa, xb):
    """Same, for a rectangle (separable in x and y within a pad)."""
    f = _overlap(ext[1][0], ext[1][1], ya, yb) * \
        _overlap(ext[0][0], ext[0][1], xa, xb)
    return float((w * f).sum() / w.sum())


def smeared_profile(ext, w, axis=1, bin_mm=2.0):
    """Illumination profile with each pad spread over its extent."""
    lo, hi = ext[axis]
    edges = np.arange(math.floor(lo.min() / bin_mm) * bin_mm,
                      hi.max() + bin_mm, bin_mm)
    a, b = edges[:-1], edges[1:]
    h = (w[:, None] * _overlap(lo[:, None], hi[:, None],
                               a[None, :], b[None, :])).sum(axis=0)
    return 0.5 * (a + b), h


def slab_edges(centres, h, frac=0.5):
    """Hard edges of a flat-topped profile: where it crosses `frac` of the
    plateau (the median of the bins above half maximum)."""
    plateau = np.median(h[h > 0.5 * h.max()])
    over = h > frac * plateau
    i, j = int(np.argmax(over)), len(h) - 1 - int(np.argmax(over[::-1]))
    lo = np.interp(frac * plateau, [h[i - 1], h[i]], [centres[i - 1], centres[i]]) \
        if i else centres[i]
    hi = np.interp(frac * plateau, [h[j + 1], h[j]], [centres[j + 1], centres[j]]) \
        if j < len(h) - 1 else centres[j]
    return float(lo), float(hi)


def span1090(centres, h):
    """10-90 % width. Stabler than FWHM on a flat-topped, non-Gaussian profile."""
    c = np.cumsum(h) / h.sum()
    return float(np.interp(0.9, c, centres) - np.interp(0.1, c, centres))


def d4_to_lab(local_x, local_y, x0, shim):
    """det4 detector-local -> lab: bands horizontal, PCB edge on the table.

    Local X is vertical (increasing up) and rests on the bare -X edge, so
    height = local X + 20.32 + shim. Local Y increases to the LEFT, which is what
    a 90 deg counter-clockwise rotation of the bench view does: with the X cards
    on the bottom and the Y cards on the right, X1 is on the left and Y1 at the
    bottom (the physical convention), so rotating CCW puts the X bank on the
    right with X1 at the bottom and the Y bank on top with **Y1 on the right**.
    This agrees with `board_map_g_det4_rot90ccw.png`; it is not its mirror.

    The board margin and the X bank are drawn mirrored about the active-area
    centre (`MIRROR_Y`), exactly as `14_board_map.py` does, so that the wide
    50.32 mm connector margin and the X cards land together on the right.
    """
    return (x0 - (np.asarray(local_y, float) - np.mean(D4_SQUARE)),
            np.asarray(local_x, float) - D4_PCB[0] + shim)


def mirror_y(v):
    """Mirror a detector-local Y about the active-area centre (drawing only)."""
    return 2.0 * np.mean(D4_SQUARE) - np.asarray(v, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='eff_map_P2_MID_eff_nominal_*.csv')
    ap.add_argument('--gap', type=float, default=TABLE_GAP,
                    help='table top to the lowest point of the P2 active area')
    ap.add_argument('--shim', type=float, default=SHIM_MM,
                    help='height packed under det4 [mm]')
    ap.add_argument('--p2-mirror', type=int, choices=(1, -1), default=-1,
                    help='handedness of the P2 pad frame in the beam view')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()

    m, polys = _b.load_board()
    apex = _b.fan_apex(m)
    (r_lo, r_hi), (phi_lo, phi_hi) = _b.active_extent(polys, apex)
    bis = 0.5 * (phi_lo + phi_hi)
    h_apex = args.gap + r_hi

    ill, files = _b.load_illumination(args.maps)
    keep = ill.n_tag.values > 0
    pad = np.stack([ill.pad_cx.values[keep], ill.pad_cy.values[keep]], axis=1)
    w = ill.n_tag.values[keep].astype(float)
    lab = to_lab(pad, apex, bis, h_apex, args.p2_mirror)
    lx, ly = lab[:, 0], lab[:, 1]

    (bx, by), C, s_maj, s_min, ang = _b.moments(lx, ly, w)
    sig_v = math.sqrt(np.average((ly - by) ** 2, weights=w))
    sig_h = math.sqrt(np.average((lx - bx) ** 2, weights=w))

    # per-pad lab extents, so bands can be scanned without pad-pitch quantisation
    rows = m.set_index('channel_id').index.get_indexer(
        ill.channel_id.values[keep])
    corners = to_lab(polys[rows].reshape(-1, 2), apex, bis, h_apex,
                     args.p2_mirror).reshape(-1, 4, 2)
    ext = ((corners[:, :, 0].min(axis=1), corners[:, :, 0].max(axis=1)),
           (corners[:, :, 1].min(axis=1), corners[:, :, 1].max(axis=1)))
    cu_v, h_v = smeared_profile(ext, w, axis=1, bin_mm=6.0)
    cu_h, h_h = smeared_profile(ext, w, axis=0, bin_mm=6.0)
    fw_v, fw_h = _b.fwhm(cu_v, h_v), _b.fwhm(cu_h, h_h)
    q_v, q_h = span1090(cu_v, h_v), span1090(cu_h, h_h)
    slab = slab_edges(cu_v, h_v)          # the trigger acceptance, hard-edged
    cu_v, cu_h = cu_v - by, cu_h - bx          # relative to the beam centre

    # --- det4, resting on the table, readout square centred on the beam ------
    x0 = bx
    band_lo, band_hi = (np.array(D4_BAND) - D4_PCB[0] + args.shim)
    sq_lo, sq_hi = (np.array(D4_SQUARE) - D4_PCB[0] + args.shim)
    sq_half = 0.5 * (D4_SQUARE[1] - D4_SQUARE[0])

    def frac_band(shift=0.0):
        return slab_fraction(ext, w, band_lo + shift, band_hi + shift)

    def frac_square(shift=0.0):
        return box_fraction(ext, w, sq_lo + shift, sq_hi + shift,
                            x0 - sq_half, x0 + sq_half)

    shifts = np.arange(-60, 121, 1.0)
    f_band = np.array([frac_band(s) for s in shifts])
    f_sq = np.array([frac_square(s) for s in shifts])
    best_shift = float(shifts[int(np.argmax(f_band))])

    # the same 38 mm band, turned vertical instead (bands along the beam)
    bw = D4_BAND[1] - D4_BAND[0]
    xs = np.arange(bx - 60, bx + 60, 1.0)
    f_vert = max(slab_fraction(ext, w, c - bw / 2, c + bw / 2, axis=0)
                 for c in xs)

    # ----------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.55,
                          wspace=0.17, left=0.06, right=0.965,
                          top=0.895, bottom=0.075)
    axl = fig.add_subplot(gs[:, 0])
    axv = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[1, 1])
    axf = fig.add_subplot(gs[2, 1])

    # ---- elevation view ----------------------------------------------------
    plab = to_lab(polys.reshape(-1, 2), apex, bis, h_apex,
                  args.p2_mirror).reshape(polys.shape)
    instrumented = m.channel_id.between(384, 895).values
    counts = np.zeros(len(m))
    counts[m.set_index('channel_id').index.get_indexer(ill.channel_id.values)] \
        = ill.n_tag.values
    axl.add_collection(PolyCollection(plab[~instrumented], facecolors='#f2f2f2',
                                      edgecolors='#dcdcdc', linewidths=0.3,
                                      zorder=2))
    hot = counts[instrumented]
    pc = PolyCollection(plab[instrumented], array=hot, cmap='Blues',
                        norm=mcolors.LogNorm(vmin=max(hot[hot > 0].min(), 1),
                                             vmax=hot.max()),
                        edgecolors='#c8d4e0', linewidths=0.25, zorder=3)
    axl.add_collection(pc)
    cb = fig.colorbar(pc, ax=axl, pad=0.012, fraction=0.043)
    cb.set_label('tagged tracks per pad', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # table top
    xlim = (-360, 360)
    axl.axhspan(-70, 0, color=TABLE, alpha=0.22, lw=0, zorder=1)
    axl.axhline(0, color=TABLE, lw=2.2, zorder=6)
    axl.text(xlim[0] + 8, -34, 'mechanical table top', fontsize=9, color=TABLE,
             va='center', zorder=7)

    if args.shim > 0:
        axl.add_patch(Rectangle((x0 - 219.61, 0.0), 470.0, args.shim,
                                facecolor='#e8e2d8', edgecolor='#a99b86',
                                lw=1.0, hatch='///', zorder=5))
        axl.text(x0 - 214.0, args.shim / 2, f'{args.shim:.0f} mm riser',
                 fontsize=8, color='#7a6a55', va='center', ha='left', zorder=8)

    # det4
    for lo_hi, kw in (((D4_PCB[0], D4_PCB[1]),
                       dict(ec='#555555', lw=1.6, ls='-', label='det4 PCB')),
                      ((D4_ACTIVE[0], D4_ACTIVE[1]),
                       dict(ec='#999999', lw=1.1, ls='--',
                            label='det4 active area'))):
        (xa, ya), (xb, yb) = (d4_to_lab(lo_hi[0], mirror_y(lo_hi[0]), x0, args.shim),
                              d4_to_lab(lo_hi[1], mirror_y(lo_hi[1]), x0, args.shim))
        axl.add_patch(Rectangle((min(xa, xb), min(ya, yb)), abs(xb - xa),
                                abs(yb - ya), fill=False, zorder=7, **kw))
    ax_lo, ax_hi = sorted(d4_to_lab(0.0, np.array(D4_ACTIVE), x0, args.shim)[0])
    axl.add_patch(Rectangle((ax_lo, band_lo), ax_hi - ax_lo, band_hi - band_lo,
                            facecolor=DET4, alpha=0.30, lw=0, zorder=6,
                            label='det4 live band, 38 mm'))
    axl.add_patch(Rectangle((x0 - sq_half, sq_lo), 2 * sq_half, sq_hi - sq_lo,
                            fill=False, ec=DET4, lw=2.0, zorder=8,
                            label=f'readout square, {D4_CABLES}'))

    # det4 connector banks, both in the wide +X / +Y margins (Gerber):
    # X bank on the right edge (it measures the vertical coordinate, so its
    # connectors stack vertically), Y bank along the top edge.
    bank_x = np.sort(d4_to_lab(0.0, mirror_y(np.array(D4_BANK)),
                               x0, args.shim)[0])
    bank_y = np.array(D4_BANK) - D4_PCB[0] + args.shim
    for k, lo, hi in D4_CONN:
        on = k in D4_HILITE
        ec, fc = ((DET4, DET4) if on else ('#7f8c99', '#dfe5ea'))
        kw = dict(facecolor=fc, edgecolor=ec, alpha=0.85 if on else 0.7,
                  lw=1.4 if on else 0.7, zorder=7)
        tkw = dict(fontsize=6.5, ha='center', va='center',
                   color=DET4 if on else '#55606b', zorder=8)
        y0_, y1_ = lo - D4_PCB[0] + args.shim, hi - D4_PCB[0] + args.shim
        axl.add_patch(Rectangle((bank_x[0], y0_), bank_x[1] - bank_x[0],
                                y1_ - y0_, **kw))
        axl.text(bank_x.mean(), 0.5 * (y0_ + y1_), f'X{k}', rotation=90, **tkw)
        xa, xb = sorted(d4_to_lab(0.0, np.array([lo, hi]), x0, args.shim)[0])
        axl.add_patch(Rectangle((xa, bank_y[0]), xb - xa,
                                bank_y[1] - bank_y[0], **kw))
        axl.text(0.5 * (xa + xb), bank_y.mean(), f'Y{k}', **tkw)
    # beam
    for n in (1, 2):
        axl.add_patch(Ellipse((bx, by), 2 * n * s_maj, 2 * n * s_min, angle=ang,
                              fill=False, ec=BEAM, lw=2.0 if n == 1 else 1.4,
                              zorder=9))
    axl.plot([bx], [by], '+', color=BEAM, ms=13, mew=2.2, zorder=10)
    for e in slab:
        axl.axhline(e, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=5)
    axl.text(xlim[0] + 8, slab[1] + 8, f'trigger slab {slab[0]:.0f}-{slab[1]:.0f} mm',
             fontsize=8.5, color=INK, va='bottom', zorder=7)
    axl.axhline(by, color=BEAM, lw=1.0, ls=(0, (6, 4)), zorder=5)
    axl.text(xlim[0] + 8, by + 7, f'beam axis, {by:.0f} mm above the table',
             fontsize=9, color=BEAM, ha='left', va='bottom', zorder=10)

    # dimensions
    axl.annotate('', xy=(-300, 0), xytext=(-300, args.gap),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=1.1), zorder=8)
    axl.text(-296, args.gap / 2, f'{args.gap:.0f} mm', fontsize=8.5, color=INK,
             va='center', ha='left', zorder=8)
    band_c = 0.5 * (band_lo + band_hi)
    dz = by - band_c
    if abs(dz) >= 12:
        xd = x0 - sq_half - 30
        axl.annotate('', xy=(xd, band_c), xytext=(xd, by),
                     arrowprops=dict(arrowstyle='<->', color=DET4, lw=1.3),
                     zorder=9)
        axl.text(xd - 5, 0.5 * (by + band_c),
                 f'beam axis is\n{dz:.0f} mm above\nthe band centre',
                 fontsize=9, color=DET4, va='center', ha='right', zorder=9)
    else:
        axl.text(xlim[0] + 8, band_lo - 8,
                 f'band centre {band_c:.0f} mm — beam {abs(dz):.0f} mm '
                 f'{"above" if dz > 0 else "below"} it',
                 fontsize=9, color=DET4, va='top', ha='left', zorder=9)

    axl.set_xlim(*xlim)
    axl.set_ylim(-70, 830)
    axl.set_aspect('equal')
    axl.set_xlabel('horizontal [mm]   (0 = P2 fan centre-line)   —   '
                   'view from upstream, beam into the page')
    axl.set_ylabel('height above the mechanical table top [mm]')
    axl.set_title('Beam\'s-eye view: P2 as mounted (apex up, bisector vertical) '
                  f'and det4 on its {args.shim:.0f} mm riser',
                  fontsize=10.5, loc='left')
    axl.tick_params(labelsize=8.5)
    for s in ('top', 'right'):
        axl.spines[s].set_visible(False)
    h_, l_ = axl.get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color=BEAM, lw=2.0),
           plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (5, 3)))]
    h_ += [Rectangle((0, 0), 1, 1, facecolor='#dfe5ea', edgecolor='#7f8c99',
                     lw=0.7)]
    l_ += ['triggered flux, 1$\\sigma$ / 2$\\sigma$', 'trigger slab edges',
           'connector banks (X4/X5, Y4/Y5 filled)']
    axl.legend(h_, l_, fontsize=8.5, loc='upper right', framealpha=0.95,
               edgecolor='#d8d8d8')

    # ---- vertical profile --------------------------------------------------
    axv.plot(h_v / h_v.max(), cu_v + by, color=BEAM, lw=1.6)
    axv.fill_betweenx(cu_v + by, 0, h_v / h_v.max(), color=BEAM, alpha=0.12)
    for e in slab:
        axv.axhline(e, color=INK, lw=1.0, ls=(0, (5, 3)))
    axv.text(0.03, slab[1] + 6, f'trigger acceptance slab {slab[0]:.0f}-'
             f'{slab[1]:.0f} mm ({slab[1] - slab[0]:.0f} mm tall)',
             fontsize=8.5, color=INK, va='bottom')
    axv.axhspan(band_lo, band_hi, color=DET4, alpha=0.28, lw=0)
    axv.axhspan(sq_lo, sq_hi, color=DET4, alpha=0.10, lw=0)
    axv.axhline(by, color=BEAM, lw=0.9, ls=(0, (6, 4)))
    axv.set_ylim(110, 400)
    axv.set_xlim(0, 1.15)
    axv.set_xlabel('illumination (peak = 1)', fontsize=9)
    axv.set_ylabel('height above table [mm]', fontsize=8.5)
    axv.text(0.97, 0.05, 'vertical extent is the TRIGGER,\nnot the beam',
             transform=axv.transAxes, ha='right', va='bottom', fontsize=8.5)
    axv.set_title('Vertical: a hard-edged trigger slab, with det4\'s band (dark) '
                  'and square (pale)', fontsize=10.5, loc='left')

    # ---- horizontal profile ------------------------------------------------
    axh.step(cu_h + bx, h_h / h_h.max(), where='mid', color=BEAM, lw=1.6)
    axh.fill_between(cu_h + bx, 0, h_h / h_h.max(), step='mid', color=BEAM,
                     alpha=0.12)
    axh.axvspan(x0 - sq_half, x0 + sq_half, color=DET4, alpha=0.14, lw=0)
    axh.set_xlim(-180, 180)
    axh.set_ylim(0, 1.15)
    axh.set_xlabel('horizontal position [mm]  (upstream view)', fontsize=9)
    axh.set_ylabel('illumination\n(peak = 1)', fontsize=8.5)
    axh.text(0.985, 0.90, f'$\\sigma_h$ = {sig_h:.1f} mm   '
             f'10-90 % span = {q_h:.0f} mm',
             transform=axh.transAxes, ha='right', va='top', fontsize=8.5)
    axh.set_title('Horizontal: this one IS the beam '
                  f'(readout square {2 * sq_half:.0f} mm wide)',
                  fontsize=10.5, loc='left')

    # ---- fraction vs shim --------------------------------------------------
    axf.plot(shifts, 100 * f_band, '-', color=DET4, lw=2.0,
             label='on the 38 mm live band')
    axf.plot(shifts, 100 * f_sq, '--', color=MUTED, lw=1.6,
             label=f'in the {2 * sq_half:.0f} mm readout square')
    axf.axvline(0, color=INK, lw=0.9, ls=':')
    axf.plot([0], [100 * frac_band()], 'o', color=BEAM, ms=8, zorder=5)
    axf.annotate(f'as mounted (+{args.shim:.0f} mm): {100 * frac_band():.0f}%',
                 (0, 100 * frac_band()), textcoords='offset points',
                 xytext=(-8, -20), fontsize=9, color=BEAM, ha='right')
    axf.plot([best_shift], [100 * f_band.max()], '*', color=DET4, ms=15, zorder=5)
    axf.annotate(f'{best_shift:+.0f} mm more: {100 * f_band.max():.0f}%',
                 (best_shift, 100 * f_band.max()), textcoords='offset points',
                 xytext=(8, 6), fontsize=9, color=DET4)
    axf.set_xlabel(f'further shim, on top of the {args.shim:.0f} mm already '
                   'under det4 [mm]')
    axf.set_ylabel('% of triggers', fontsize=8.5)
    axf.set_ylim(0, 100)
    axf.set_title('What det4 collects vs how high it is packed up',
                  fontsize=10.5, loc='left')
    axf.legend(fontsize=8.5, loc='upper right', frameon=False)
    axf.grid(alpha=0.18, lw=0.6)

    for ax in (axv, axh, axf):
        ax.tick_params(labelsize=8.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.suptitle('SPS H4 beam and det4 in the table frame  —  P2 apex up, '
                 f'{args.gap:.0f} mm active-area clearance, det4 bands '
                 f'horizontal on a {args.shim:.0f} mm riser',
                 fontsize=12, x=0.06, ha='left')
    png = os.path.join(args.out, 'sps_beam_lab_frame.png')
    fig.savefig(png, dpi=170)
    print('wrote', png)

    out = {
        'assumptions': {
            'fan_bisector': 'vertical, apex up, fan centred on x = 0',
            'table_gap_mm': args.gap,
            'view': 'from upstream, looking downstream (beam into the page); '
                    'det4 X connector bank on the right',
            'p2_mirror': args.p2_mirror,
            'det4': 'bands horizontal; the bare -X PCB edge (local X = -20.32) '
                    'rests on the table; local Y increases to the right so the '
                    '+Y mezzanine edge carries the X bank on the right; readout '
                    'square centred on the beam',
            'det4_margins_mm': {'bare_edges_-X_-Y': 20.32,
                                'mezzanine_edges_+X_+Y': 50.32},
            'det4_shim_mm': args.shim,
        },
        'p2': {
            'fan_apex_height_mm': h_apex,
            'active_radius_mm': [r_lo, r_hi],
            'active_phi_deg': [phi_lo, phi_hi],
            'bisector_deg': bis,
        },
        'beam_in_lab': {
            'height_above_table_mm': float(by),
            'horizontal_offset_from_centreline_mm': float(bx),
            'trigger_slab_height_mm': [slab[0], slab[1]],
            'trigger_slab_thickness_mm': slab[1] - slab[0],
            'sigma_vertical_mm_IS_THE_SLAB_NOT_THE_BEAM': float(sig_v),
            'sigma_horizontal_mm': float(sig_h),
            'fwhm_vertical_mm': float(fw_v),
            'fwhm_horizontal_mm': float(fw_h),
            'span_10_90_vertical_mm': float(q_v),
            'span_10_90_horizontal_mm': float(q_h),
            'ellipse_major_mm': float(s_maj), 'ellipse_minor_mm': float(s_min),
            'ellipse_major_axis_deg_from_horizontal': float(ang),
        },
        'det4': {
            'band_height_mm': [float(band_lo), float(band_hi)],
            'readout_square_height_mm': [float(sq_lo), float(sq_hi)],
            'readout_square_width_mm': float(2 * sq_half),
            'cables': D4_CABLES,
            'shim_mm': args.shim,
            'fraction_on_band_as_mounted': float(frac_band()),
            'fraction_in_square_as_mounted': float(frac_square()),
            'best_extra_shim_mm': best_shift,
            'best_total_shim_mm': args.shim + best_shift,
            'fraction_on_band_best': float(f_band.max()),
            'fraction_in_square_best': float(f_sq.max()),
            'fraction_if_band_were_vertical': float(f_vert),
        },
    }
    js = os.path.join(args.out, 'sps_beam_lab_frame.json')
    with open(js, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', js)

    print(f'\nP2 apex sits {h_apex:.0f} mm above the table')
    print(f'beam: {by:.1f} mm above the table, {bx:+.1f} mm off the centre-line')
    print(f'      trigger slab {slab[0]:.0f}-{slab[1]:.0f} mm '
          f'({slab[1] - slab[0]:.0f} mm tall, hard-edged)')
    print(f'      sigma_v {sig_v:.1f} (= the slab) / sigma_h {sig_h:.1f} mm, '
          f'10-90%% span {q_v:.0f} x {q_h:.0f} mm, major axis '
          f'{ang:.1f} deg from horizontal')
    print(f'det4 on a {args.shim:.0f} mm riser: band spans '
          f'{band_lo:.0f}-{band_hi:.0f} mm, catches {100 * frac_band():.1f}% '
          f'(square {100 * frac_square():.1f}%)')
    print(f'{best_shift:+.0f} mm more (total {args.shim + best_shift:.0f}) -> '
          f'band {100 * f_band.max():.1f}%, square {100 * f_sq.max():.1f}%')
    print(f'the same 38 mm band turned VERTICAL would catch '
          f'{100 * f_vert:.1f}%')


if __name__ == '__main__':
    main()
