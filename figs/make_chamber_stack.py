#!/usr/bin/env python3
"""make_chamber_stack.py -- Paper I, fig.chamber_stack: 2D cross-section of an MX17 chamber (Dylan 2026-11-03).

    .venv/bin/python papers/paper1/figs/make_chamber_stack.py

All numbers from docs/detector_spec/spec.py (STACK, ELEMENTS; status ok / cad / asm / open there).  Two scales:
(a) the whole chamber, z to scale (0.0515 in/mm), a cut through the centre in x; (b) the amplification gap and readout
board, z at 1.55 in/mm (30x (a)) and x at 0.72 in/mm (z stretched 2.2x against x), three pads across.  Values that spec.py
marks assumed ('asm') or unknown ('open') carry a dagger.  Drawn in inches (one axes, data = inches).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Ellipse, FancyArrowPatch
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / 'papers')]
import paper_style as ps                                           # noqa: E402

OUT = HERE / 'out' / 'chamber_stack'
ps.use()
plt.rcParams['savefig.pad_inches'] = 0.01
FS = ps.BASE_PT * 0.85
INK, MUTED = ps.INK, ps.MUTED
YC, XC = '#009E73', ps.ACCENT
CU, KAP, FR4, GAS, ROH, AL, MYL, ESL = '#b87333', '#f0d28a', '#dde6dc', '#eef5fa', '#f2ebd8', '#c3c8cf', '#8fc8e6', '#444444'
D = '†'                                                       # dagger: assumed / unconfirmed in spec.py
LW = 0.6

W, H = 6.1, 3.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
rows = []                                                          # for the csv: what was drawn


def R(x0, y0, x1, y1, fc, ec=INK, lw=LW, z=2, **kw):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def T(x, y, s, ha='left', va='center', fs=FS, c=INK, **kw):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, zorder=12, **kw)


def line(xs, ys, c=INK, lw=LW, ls='-', z=3):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='butt')


def arrow(p0, p1, c=INK, lw=0.7, style='<->', ms=6, z=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0,
                                 zorder=z))


# ====================================================================================================== (a)
KA = 0.0515                              # in / mm
Y0 = 3.00                                # top of the Mylar window
ya = lambda mm: Y0 - KA * mm
XI0, XI1 = 1.42, 1.98                    # inner gas width (not to scale)
XW = 0.08                                # side walls
T(0.0, 3.17, '(a)', va='top', fs=ps.BASE_PT, fontweight='bold')
T(2.15, 3.17, '(b)', va='top', fs=ps.BASE_PT, fontweight='bold')
mm = dict(win=(0, 0.06), flange=(0.06, 5.06), cat=(5.06, 5.118), drift=(5.118, 35.118), amp=(35.118, 35.268),
          board=(35.268, 36.968), roh=(36.968, 41.968), myl=(41.968, 41.993), plate=(41.993, 49.993))
R(XI0, ya(mm['flange'][1]), XI1, ya(mm['flange'][0]), GAS, ec='none')
R(XI0, ya(mm['drift'][1]), XI1, ya(mm['drift'][0]), GAS, ec='none')
# window (Mylar), cathode (kapton + Cu), mesh
R(XI0 - XW, ya(0.5), XI1 + XW, ya(0), MYL, lw=0.4)
R(XI0, ya(0.25), XI1, ya(0.06), MYL, ec=MYL, lw=0.1)
R(XI0, ya(mm['cat'][1]) - 0.003, XI1, ya(mm['cat'][0]) + 0.003, KAP, lw=0.4)
line([XI0, XI1], [ya(mm['cat'][0]) + 0.004] * 2, c=CU, lw=1.4)
# walls: Al window flange (5 mm), Al gas frame (30 mm)
for xa, xb in ((XI0 - XW, XI0), (XI1, XI1 + XW)):
    R(xa, ya(mm['flange'][1]), xb, ya(mm['flange'][0] + 0.5), AL)
    R(xa, ya(mm['drift'][1]), xb, ya(mm['drift'][0]), AL)
    R(xa, ya(mm['plate'][1]), xb, ya(mm['roh'][0]), AL)
# mesh + amplification gap + board (to scale: sub-pixel lines) and the zoom box
line([XI0, XI1], [ya(mm['drift'][1])] * 2, c=INK, lw=1.0)
R(XI0, ya(mm['board'][1]), XI1, ya(mm['board'][0]), FR4, lw=0.5)
R(XI0, ya(mm['roh'][1]), XI1, ya(mm['roh'][0]), ROH, lw=0.5)
R(XI0, ya(mm['myl'][1]) - 0.003, XI1, ya(mm['myl'][0]) + 0.003, MYL, lw=0.4)
# horizontal break marks (chamber is 440 mm wide, drawn 0.6 in)
for xb in (XI0 - XW - 0.0, XI1 + XW):
    pass
LX = 1.15
lab = [('Mylar window 60 \u00b5m', 3.04, 0.03, XI0 - XW), ('gas gap 5.0 mm', ya(2.6), 2.6, XI0 - XW),
       ('cathode: kapton\n50 \u00b5m + Cu' + D, ya(5.09) - 0.07, 5.09, XI0 - XW), ('drift gas', ya(14), 14, XI0),
       ('mesh (ground)', 1.40, 35.12, XI0), ('board 1.70 mm', 1.19, 36.1, XI0),
       ('Rohacell 5 mm', 0.99, 39.5, XI0), ('back Mylar 25 \u00b5m' + D, 0.80, 41.98, XI0),
       ('Al plate 8 mm,\n402 mm aperture', 0.58, 46.0, XI0 - XW)]
for s_, y, zz, xr in lab:
    T(LX, y, s_, ha='right', linespacing=1.0)
    line([LX + 0.02, xr - 0.02], [y, ya(zz)], c=MUTED, lw=0.4, z=1)
# drift gap dimension
arrow((1.29, ya(5.118)), (1.29, ya(35.118)))
T(1.245, ya(21), '30.000 mm', ha='center', va='center', rotation=90, fs=FS)
# muon track, ionisation, electron drift
xt = lambda y: 1.50 + 0.08 * (Y0 + 0.05 - y)
ax.plot([xt(Y0 + 0.05), xt(ya(48))], [Y0 + 0.05, ya(48)], color=ps.ACCENT, lw=1.2, zorder=6)
ax.plot([xt(ya(5.2)), xt(ya(35))], [ya(5.2), ya(35)], color=ps.ACCENT, lw=1.6, zorder=6)
T(xt(Y0 + 0.05) + 0.05, Y0 + 0.08, 'muon', c=ps.ACCENT, fontstyle='italic', ha='left', va='center')
for zz in (8, 12.5, 17, 21.5, 26, 30):
    x, y = xt(ya(zz)), ya(zz)
    ax.plot(x, y, 'o', ms=2.0, color=ps.ACCENT, mew=0, zorder=7)
    arrow((x, y - 0.03), (x, ya(34.4)), c=ps.DET_COLOR['det2'], lw=0.5, style='-|>', ms=4)
T(xt(ya(28)) + 0.07, ya(27), 'e$^-$', ha='left', fs=FS, c=ps.DET_COLOR['det2'])
# E_drift: arrow up (cathode is the negative electrode, mesh at ground)
arrow((XI1 - 0.07, ya(32)), (XI1 - 0.07, ya(8)), c=INK, lw=1.0, style='-|>', ms=7)
T(XI1 - 0.14, ya(20), '$E_\\mathrm{drift}$', ha='center', rotation=90, fs=FS)
# voltages
T(XI1 + XW + 0.03, ya(5.09), '$-HV_\\mathrm{drift}$', ha='left', fs=FS, c=INK)
# zoom box + funnel (to panel b)
BZ = (XI0 - 0.03, ya(36.2) - 0.05, XI1 + 0.03, ya(34.6) + 0.04)
R(BZ[0], BZ[1], BZ[2], BZ[3], 'none', ec=INK, lw=0.8, z=8)

# ====================================================================================================== (b)
KV, KH = 1.55, 0.66                       # in / mm  (vertical z, horizontal x)
XB0, XB1 = 2.92, 4.45                     # drawing
YM = 2.42                                 # mesh centre plane
yb = lambda zmm: YM - KV * zmm
xc = (XB0 + XB1) / 2
PITCH, PADW = 0.78, 0.68
padx = [xc + (i - 1) * PITCH * KH for i in (0, 1, 2)]               # three pads
padx = [xc + (i - 1) * PITCH * KH for i in range(3)]
zl = {}                                                            # layer z-extents (mm below the mesh centre)
z = 0.0095
zl['gap'] = (z, z + 0.150); z += 0.150
zl['esl'] = (z, z + 0.010); z += 0.010
zl['ins'] = (z, z + 0.0705); z += 0.0705
zl['L4'] = (z, z + 0.026); z += 0.026
for k in ('f1', 'L5', 'f2', 'L6', 'f3', 'L7', 'f4'):
    t = 0.026 if k.startswith('L') else 0.2644
    zl[k] = (z, z + t); z += t
ZEND = z
# gas above the mesh and in the gap
R(XB0, yb(zl['gap'][0]), XB1, yb(-0.35), GAS, ec='none', z=0)
R(XB0, yb(zl['gap'][1]), XB1, yb(zl['gap'][0]), '#e3eef7', ec='none', z=0)
# FR4 and insulation
for k in ('f1', 'f2', 'f3', 'f4'):
    R(XB0, yb(zl[k][1]), XB1, yb(zl[k][0]), FR4, lw=0.4)
R(XB0, yb(zl['ins'][1]), XB1, yb(zl['ins'][0]), KAP, lw=0.4)
# resistive strips (550 um, 800 um pitch), above the pads, offset: three strips
rp = 0.800 * KH
rx0 = xc - 0.5 * rp - rp
for i in range(-1, 5):
    xa = rx0 + i * rp + 0.5 * rp - 0.55 * KH / 2
    xa0, xa1 = max(xa, XB0), min(xa + 0.55 * KH, XB1)
    if xa1 <= xa0:
        continue
    R(xa0, yb(zl['esl'][1]), xa1, yb(zl['esl'][0]), ESL, lw=0.3, z=4)
# mesh: woven wires, 19 um on a 67 um pitch (schematic)
nw = int((XB1 - XB0) / (0.067 * KH))
for i in range(nw):
    xw = XB0 + (i + 0.5) * (XB1 - XB0) / nw
    ax.add_patch(Ellipse((xw, yb(0)), 0.019 * KH, 0.019 * KV, fc='#777777', ec='none', zorder=5))
# pads (L4), strips L5 (Y buses + lands), L6 (X, behind the cut), L7
for i, x in enumerate(padx):
    R(x - PADW * KH / 2, yb(zl['L4'][1]), x + PADW * KH / 2, yb(zl['L4'][0]), CU, lw=0.3, z=4)
line([XB0, XB1], [yb(sum(zl['L7']) / 2)] * 2, c=CU, lw=0.8, ls=(0, (3, 2)), z=3)
line([XB0, XB1], [yb(sum(zl['L6']) / 2)] * 2, c=XC, lw=0.6, ls=(0, (3, 2)), z=3)
bx = [(padx[i] + padx[i + 1]) / 2 for i in range(2)] + [padx[0] - PITCH * KH / 2, padx[2] + PITCH * KH / 2]
for x in sorted(bx):
    R(x - 0.049 * KH, yb(zl['L5'][1]), x + 0.049 * KH, yb(zl['L5'][0]), YC, lw=0.2, z=4)
R(XB0, yb(zl['f4'][1]), XB1, yb(zl['f4'][0]), FR4, lw=0.4, z=1)
for i, x in enumerate(padx):
    view = 'Y' if i % 2 == 0 else 'X'
    key, col = ('L5', YC) if view == 'Y' else ('L6', XC)
    R(x - 0.0015, yb(zl[key][0]), x + 0.0015, yb(zl['L4'][0]), col, ec=col, lw=0.8, z=5)         # via barrel
    R(x - 0.25 * KH, yb(zl[key][1]), x + 0.25 * KH, yb(zl[key][0]), col, lw=0.2, z=5)           # land (0.5 mm)
    if view == 'Y':        # stub to the bus on its left
        xs = x - PITCH * KH / 2
        R(xs, yb(zl['L5'][1]), x - 0.25 * KH, yb(zl['L5'][0]), YC, ec=YC, lw=0.3, z=5)
# frame
for yy in (yb(-0.35), yb(ZEND)):
    pass
line([XB0, XB0], [yb(-0.35), yb(ZEND)], c=INK, lw=0.5)
line([XB1, XB1], [yb(-0.35), yb(ZEND)], c=INK, lw=0.5)
line([XB0, XB1], [yb(ZEND)] * 2, c=INK, lw=0.5)
for k in ('esl', 'L4', 'L5', 'L6', 'L7'):
    pass
# track electrons arriving, avalanche under a mesh hole above the middle strip
xa_ = rx0 + 1 * rp + 0.5 * rp
for dx, a in ((-0.55, 0.35), (-0.18, 0.5), (0.18, 0.5), (0.55, 0.35)):
    pass
for xe in (xa_ - 0.17, xa_ - 0.03, xa_ + 0.11):
    arrow((xe, yb(-0.33)), (xe + 0.0, yb(-0.02)), c=ps.DET_COLOR['det2'], lw=0.6, style='-|>', ms=4)
yt, yb2 = yb(zl['gap'][0]), yb(zl['gap'][1])
cone = Polygon([(xa_ - 0.04, yt), (xa_ + 0.04, yt), (xa_ + 0.27, yb2), (xa_ - 0.27, yb2)], closed=True, fc=ps.ACCENT,
               ec='none', alpha=0.30, zorder=3)
ax.add_patch(cone)
rng = np.random.default_rng(3)
for _ in range(34):
    f = rng.uniform(0.05, 1.0)
    ax.plot(xa_ + rng.normal(0, 0.03 + 0.07 * f), yt + (yb2 - yt) * f, '.', ms=2.0, color=ps.ACCENT, mew=0, zorder=6)
# E_amp arrow + voltage levels
xe_ = XB0 + 0.07
arrow((xe_, yb2 + 0.01), (xe_, yt - 0.01), c=INK, lw=1.0, style='-|>', ms=6)
T(XB0 - 0.04, (yt + yb2) / 2, '$E_\\mathrm{amp}$', ha='right', fs=FS)
T(XB0 - 0.04, yb(-0.05) + 0.05, '0 V', ha='right', fs=FS)
T(XB0 - 0.04, yb2 - 0.07, '$+HV_\\mathrm{resist}$', ha='right', fs=FS)
# dimension: amplification gap
xd = XB1 + 0.10
arrow((xd, yt), (xd, yb2), lw=0.6)
# horizontal dimensions on the pads (above the gas, with extension lines)
ydim = yb(-0.28)
for xa, xb, s, off in ((padx[0] - PADW * KH / 2, padx[0] + PADW * KH / 2, '0.68 mm', 0.0),):
    pass
yt2 = yb(-0.33) + 0.10
ax.plot([padx[0], padx[0]], [yb(zl['L4'][0]) + 0.0, yt2], color=MUTED, lw=0.3, ls=':', zorder=1)
ax.plot([padx[1], padx[1]], [yb(zl['L4'][0]) + 0.0, yt2], color=MUTED, lw=0.3, ls=':', zorder=1)
arrow((padx[0], yt2 - 0.02), (padx[1], yt2 - 0.02), lw=0.6)
T((padx[0] + padx[1]) / 2, yt2 + 0.07, 'pitch 0.78 mm', ha='center', fs=FS * 0.95)
# labels (right)
RX = XB1 + 0.20
ent = [
    (yb(0.0), 2.82, 'mesh' + D + ': woven steel,\n19 \u00b5m wire, 67 \u00b5m pitch'),
    ((yt + yb2) / 2, 2.56, 'amplification gap 150 \u00b5m\n$E_\\mathrm{amp}$ ~ 33 kV/cm (490 V)'),
    (yb(sum(zl['esl']) / 2), 2.30, 'resistive strips' + D + ': 550 \u00b5m,\n0.80 mm pitch, 10 \u00b5m'),
    (yb(sum(zl['ins']) / 2), 2.04, 'kapton' + D + ' 50 \u00b5m + glue\n(70.5 \u00b5m in all)'),
    (yb(sum(zl['L4']) / 2), 1.78, 'L4 pads: 0.68 mm,\n0.78 mm pitch, Cu' + D),
    (yb(sum(zl['L5']) / 2), None, 'L5: 512 Y strips'),
    (yb(sum(zl['L6']) / 2), None, 'L6: 512 X strips'),
    (yb(sum(zl['L7']) / 2), None, 'L7 fan-out'),
    (yb(sum(zl['f4']) / 2), None, 'FR4' + D + ', 264 \u00b5m each'),
]
for y, yl, s_ in ent:
    yl = y if yl is None else yl
    T(RX, yl, s_, linespacing=1.0, fs=FS * 0.93)
    line([XB1 + 0.02, RX - 0.03], [y, yl], c=MUTED, lw=0.4, z=1)
# funnel (a) -> (b)
for ya_, yb_ in ((BZ[3], yb(-0.35)), (BZ[1], yb(ZEND))):
    ax.plot([BZ[2], XB0 - 0.30], [ya_, yb_], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=0)
T(0.03, 0.18, D + ' assumed or unconfirmed (detector spec)\n(b): $z$ at ' + f'{KV / KA:.0f}\u00d7 the scale of (a),\n$z$ stretched {KV / KH:.1f}\u00d7 against $x$',
  fs=FS * 0.9, c=MUTED, linespacing=1.1)
rows = pd.DataFrame([dict(layer=k, z_lo_mm=v[0], z_hi_mm=v[1]) for k, v in zl.items()])
fig.canvas.draw()
_bb = fig.get_tightbbox(fig.canvas.get_renderer())
fig.savefig(OUT.with_suffix('.tmp.png'), dpi=200)
print('y', round(_bb.y0, 2), round(_bb.y1, 2), 'content bbox in', round(_bb.x0, 2), round(_bb.x1, 2), 'width', round(_bb.width, 2), 'height', round(_bb.height, 2))
ps.save(fig, OUT, data=rows)
print('ok', OUT, 'board bottom z', ZEND)
