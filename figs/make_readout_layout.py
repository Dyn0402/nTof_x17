#!/usr/bin/env python3
"""make_readout_layout.py -- Paper I, fig.readout_layout: the readout board, top view (Dylan 2026-11-03).

    .venv/bin/python papers/paper1/figs/make_readout_layout.py

Geometry is the production gerbers (DFS3498A, read by extract_readout_gerber.py into data/readout_gerber_extract.npz,
committed) plus the measured efficient area (common/mx17_active_area.py, Y passivation 18.0 / 18.7 mm).
Frame: detector-local mm, origin at the centre of pad (0,0) = gerber + 199.29 mm in both axes, the same frame as
the strip maps and the efficiency map.  ASSUMPTION: local x, y = gerber x, y (no mirror); the gerber carries
the connectors on +x and +y, and the figure shows them there.
(a) the board; (b) five-by-five pad window: the checkerboard of vias; (c) one +y connector block and its fan-out.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Circle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / 'papers'), str(REPO)]
import paper_style as ps                                            # noqa: E402
from common.mx17_active_area import (PASSIVATION_Y_LOW_MM, PASSIVATION_Y_HIGH_MM, STRIP_MAX_MM)  # noqa: E402

OUT = HERE / 'out' / 'readout_layout'
E = np.load(HERE / 'data' / 'readout_gerber_extract.npz')
C0 = 199.29                         # gerber -> local shift [mm]; pad (0,0) centre
PITCH, PAD = 0.78, 0.68
YC, XC = '#009E73', ps.ACCENT       # Y-strip (L5) and X-strip (L6) colours
BOARD_FILL, PADFILL = '#f3f4f6', '#d5dbe3'
FS = ps.BASE_PT * 0.85              # annotation size (7.65 pt)
ps.use()
plt.rcParams['savefig.pad_inches'] = 0.01
plt.rcParams['axes.grid'] = False

W, H = 6.1, 3.3
fig = plt.figure(figsize=(W, H))


def at(x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


# ---------------------------------------------------------------------------------------------------------- (a)
BIG = 3.10
axa = at(0.12, 0.10, BIG, BIG)
axa.add_patch(Rectangle((-220 + C0, -220 + C0), 470, 470, fc=BOARD_FILL, ec=ps.INK, lw=0.8, zorder=1))
axa.add_patch(Rectangle((-206 + C0, -206 + C0), 412, 412, fc='none', ec=ps.MUTED, lw=0.6, ls=(0, (4, 2)), zorder=2))
mx0, mx1 = -0.39, 2 * C0 + 0.39                                   # metallised pad edges (local)
axa.add_patch(Rectangle((mx0, mx0), mx1 - mx0, mx1 - mx0, fc=PADFILL, ec='none', zorder=3))
for p in np.linspace(40, 360, 9):                                 # a few strips per view, for the direction
    axa.plot([p, p], [0, 2 * C0], color=YC, lw=0.35, alpha=0.8, zorder=4)
    axa.plot([0, 2 * C0], [p, p], color=XC, lw=0.35, alpha=0.8, zorder=4)
lo, hi = PASSIVATION_Y_LOW_MM, STRIP_MAX_MM - PASSIVATION_Y_HIGH_MM
for ya, yb in ((mx0, lo), (hi, mx1)):
    axa.add_patch(Rectangle((mx0, ya), mx1 - mx0, yb - ya, fc='white', ec=ps.INK, hatch='//////', lw=0, alpha=0.9,
                            zorder=5))
axa.add_patch(Rectangle((mx0, lo), mx1 - mx0, hi - lo, fc='none', ec=ps.INK, lw=1.0, zorder=6))
axa.add_patch(Rectangle((mx0, mx0), mx1 - mx0, mx1 - mx0, fc='none', ec=ps.INK, lw=0.5, zorder=6))
for key, col in (('fan_top', YC), ('fan_right', XC)):
    axa.add_collection(LineCollection([[(a + C0, b + C0), (c + C0, d + C0)] for a, b, c, d in E[key]], colors=col,
                                      linewidths=0.12, zorder=7))
conn = E['conn']
blocks = []
for sel in ((conn[:, 1] > 215) & (conn[:, 0] < 0), (conn[:, 1] > 215) & (conn[:, 0] > 0),
            (conn[:, 0] > 215) & (conn[:, 1] > 0), (conn[:, 0] > 215) & (conn[:, 1] < 0)):
    p = conn[sel] + C0
    blocks.append((p[:, 0].min() - 1.5, p[:, 1].min() - 1.5, np.ptp(p[:, 0]) + 3, np.ptp(p[:, 1]) + 3))
for b in blocks:
    axa.add_patch(Rectangle(b[:2], b[2], b[3], fc='white', ec=ps.INK, lw=0.5, zorder=8))
axa.set_xlim(-78, 490); axa.set_ylim(-52, 516); axa.set_aspect('equal'); axa.axis('off')
INKA = dict(arrowstyle='<->', lw=0.7, color=ps.INK, shrinkA=0, shrinkB=0, mutation_scale=6)
txt = dict(fontsize=FS, color=ps.INK, ha='center', va='center')
# dimension column on the left (local y) and below (x)
xd = -34
for (ya, yb, t, rot) in ((mx0, lo, '18.0', 0), (lo, hi, '359.9 mm', 90), (hi, mx1, '18.7', 0)):
    axa.annotate('', (xd, ya), (xd, yb), arrowprops=INKA)
    if rot:
        axa.text(xd - 9, (ya + yb) / 2, t, rotation=90, **txt)
    else:
        axa.text(xd - 6, (ya + yb) / 2, t, ha='right', va='center', fontsize=FS, color=ps.INK)
axa.annotate('', (mx0, -34), (mx1, -34), arrowprops=INKA)
axa.text(C0, -44, '399.4 mm', **txt)
axa.text(C0, 262, '512 $\\times$ 512 pads, 0.78 mm pitch\n(0.68 mm pads, L4)', zorder=9,
         bbox=dict(fc=PADFILL, ec='none', pad=1.5), fontsize=FS * 1.05, color=ps.INK, ha='center', va='center')
for ya, yb in ((mx0, lo), (hi, mx1)):
    axa.text(C0 - 70, (ya + yb) / 2, 'passivated', zorder=9, bbox=dict(fc='white', ec='none', pad=0.5),
             fontsize=FS * 0.9, color=ps.INK, ha='center', va='center')
axa.text(C0, 120, 'x sides: not passivated', zorder=9, bbox=dict(fc=PADFILL, ec='none', pad=1.0), **txt)
for xe, xt in ((6, C0 - 118), (2 * C0 - 6, C0 + 118)):
    axa.annotate('', (xe, 120), (xt, 120), arrowprops=dict(arrowstyle='->', lw=0.6, color=ps.INK, shrinkA=0,
                                                              shrinkB=0, mutation_scale=6), zorder=9)
axa.text(C0, 470, 'Y-strip connectors (+y)', **txt)
axa.text(473, C0, 'X-strip connectors (+x)', rotation=90, **txt)
axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, ha='left', va='top', fontsize=ps.BASE_PT, fontweight='bold',
         color=ps.INK)

# ---------------------------------------------------------------------------------------------------------- (b)
axb = at(3.88, 1.58, 1.50, 1.50)
d = 4.7
pc = E['patch_pad'] + C0
# windows: gerber (-1..3.7) -> choose the 5 x 5 cells around gerber (0..4) of the cached window
cells = [(x, y) for x, y in E['patch_pad']]
xs = sorted({round(x, 2) for x, _ in cells}); ys = sorted({round(y, 2) for _, y in cells})
xlo, xhi, ylo, yhi = xs[0] - 0.45, xs[-1] + 0.45, ys[0] - 0.45, ys[-1] + 0.45
sy, sx = E['patch_seg_y'], E['patch_seg_x']
dy_, dx_ = E['patch_dot_y'], E['patch_dot_x']


def stub_end(dots, segs):
    # dot (x, y) is the via of a view when a strip stub ends on it
    ends = np.vstack([segs[:, :2], segs[:, 2:]])
    return {(round(x, 2), round(y, 2)) for x, y in dots if np.min(np.hypot(ends[:, 0] - x, ends[:, 1] - y)) < 0.06}


isY = stub_end(dy_, sy); isX = stub_end(dx_, sx)
for x, y in cells:
    axb.add_patch(Rectangle((x - PAD / 2, y - PAD / 2), PAD, PAD, fc=PADFILL, ec=ps.MUTED, lw=0.4, zorder=2))
for (segs, col, z) in ((sx, XC, 3), (sy, YC, 4)):
    lines = []
    for a, b, c, dd in segs:
        lines.append([(np.clip(a, xlo, xhi), np.clip(b, ylo, yhi)), (np.clip(c, xlo, xhi), np.clip(dd, ylo, yhi))])
    axb.add_collection(LineCollection(lines, colors=col, linewidths=1.6, zorder=z, capstyle='butt'))
for x, y in cells:
    k = (round(x, 2), round(y, 2))
    col = YC if k in isY else XC if k in isX else 'k'
    axb.add_patch(Circle((x, y), 0.25, fc=col, ec='white', lw=0.5, zorder=6))
axb.set_xlim(xlo, xhi); axb.set_ylim(ylo, yhi); axb.set_aspect('equal'); axb.axis('off')
# dimensions
x_a, y_a = xs[1], ys[0]
axb.annotate('', (xs[0] - PAD / 2, ylo + 0.12), (xs[0] + PAD / 2, ylo + 0.12),
             arrowprops=dict(arrowstyle='<->', lw=0.6, color=ps.INK, shrinkA=0, shrinkB=0), zorder=8)
axb.text(xs[0], ylo - 0.05, '0.68 mm', fontsize=FS * 0.9, ha='left', va='top', color=ps.INK)
axb.annotate('', (xs[1], yhi - 0.12), (xs[2], yhi - 0.12),
             arrowprops=dict(arrowstyle='<->', lw=0.6, color=ps.INK, shrinkA=0, shrinkB=0), zorder=8)
axb.text((xs[1] + xs[2]) / 2, yhi + 0.03, '0.78 mm', fontsize=FS * 0.9, ha='center', va='bottom', color=ps.INK)
axb.text(-0.0, 1.0, '(b)', transform=axb.transAxes, ha='right', va='top', fontsize=ps.BASE_PT, fontweight='bold',
         color=ps.INK)


# marker for window (b) on (a): a ring at pad (255, 255); window (c) outline
axa.add_patch(Circle((C0 + 2.3, C0 + 2.3), 7, fc='none', ec='k', lw=0.8, zorder=10))
CX0, CX1, CY0, CY1 = -207.0 + C0, 6.0 + C0, 195.0 + C0, 252.0 + C0
axa.add_patch(Rectangle((CX0, CY0), CX1 - CX0, CY1 - CY0, fc='none', ec='k', lw=0.8, zorder=10))

# ---------------------------------------------------------------------------------------------------------- legend
axl = at(5.42, 1.58, 0.68, 1.50); axl.axis('off'); axl.set_xlim(0, 0.68); axl.set_ylim(0, 1.50)
for y, c, t in ((1.15, YC, 'via to\nY strip\n(L5)'), (0.62, XC, 'via to\nX strip\n(L6)')):
    axl.add_patch(Circle((0.07, y), 0.055, fc=c, ec='white', lw=0.5))
    axl.text(0.17, y, t, fontsize=FS * 0.95, color=ps.INK, ha='left', va='center', linespacing=1.05)
axl.add_patch(Rectangle((0.02, 0.10), 0.11, 0.11, fc=PADFILL, ec=ps.MUTED, lw=0.4))
axl.text(0.17, 0.155, 'pad (L4)', fontsize=FS * 0.95, color=ps.INK, ha='left', va='center')

# ---------------------------------------------------------------------------------------------------------- (c)
axc = at(3.78, 0.62, 2.30, 0.60)
axc.add_patch(Rectangle((CX0, CY0), CX1 - CX0, 199.68 + C0 - CY0, fc=PADFILL, ec='none', zorder=1))
ft = E['fan_top']
axc.add_collection(LineCollection([[(a + C0, b + C0), (c + C0, d + C0)] for a, b, c, d in ft], colors=YC,
                                  linewidths=0.25, zorder=3))
vy = E['via_y']; vy = vy[(vy[:, 0] > CX0 - C0) & (vy[:, 0] < CX1 - C0)] + C0
axc.add_collection(PatchCollection([Rectangle((x - 0.3, y - 0.5), 0.6, 1.0) for x, y in vy], fc='k', ec='none',
                                   zorder=4))
cp = E['conn']; cp = cp[(cp[:, 1] > 215) & (cp[:, 0] < 6)] + C0
axc.add_collection(PatchCollection([Circle((x, y), 0.45) for x, y in cp], fc=ps.INK, ec='none', zorder=4))
axc.set_xlim(CX0, CX1); axc.set_ylim(CY0, CY1); axc.set_aspect('equal'); axc.axis('off')
axc.text(-0.0, 1.02, '(c)', transform=axc.transAxes, ha='left', va='bottom', fontsize=ps.BASE_PT, fontweight='bold',
         color=ps.INK)
axc.annotate('', (CX1 - 30, CY0 - 5), (CX1 - 10, CY0 - 5), arrowprops=INKA, annotation_clip=False)
axc.text(CX1 - 35, CY0 - 5, '20 mm', fontsize=FS * 0.9, color=ps.INK, ha='right', va='center')
fig.text(3.78 / W, 0.47 / H,
         'green: traces of the even strips\nblack bars: vias of the odd strips (to L7)\ndots: connector contact pads',
         fontsize=FS * 0.95, color=ps.INK, ha='left', va='top', linespacing=1.15)

fig.savefig(OUT.with_suffix('.png'), dpi=200)
fig.canvas.draw(); _b = fig.get_tightbbox(fig.canvas.get_renderer()); print('content bbox in', round(_b.x0, 2), round(_b.x1, 2), 'w', round(_b.width, 2), 'h', round(_b.height, 2))
ps.save(fig, OUT)
print('ok', OUT)
