#!/usr/bin/env python3
"""make_chamber_stack.py -- Paper I v2, fig.chamber_stack: 2D cross-section of an MX17 chamber (Dylan 2026-11-03).

    .venv/bin/python papers/paper1_v2/figs/make_chamber_stack.py            # (a) + (b)
    .venv/bin/python papers/paper1_v2/figs/make_chamber_stack.py --only-a   # (a) alone -> out/chamber_stack_a (Fig. 3)

v2 copy of papers/paper1/figs/make_chamber_stack.py (v2 feedback note, Fig. 3, 2026-11-12): (a) E_drift arrow clear of
the -HV label, Al plate drawn light (it is to scale in z, 8 mm, and not the point), zoom box top above the electron
arrowheads; (b) the vias as the gerbers have them: every pad has one 0.25 mm plated via drilled L4 -> L7
(DFS3498A_pla2-5.gbr, "Plated,4,7"), with a copper land in every cell on L5 and L6 (0.50 mm) and L7 (0.65 mm); a land
joins its strip only on its own layer.  The cut runs through a row of pad centres: the Y stubs run along the cut (to
the bus on their +x side), the X stubs run along y, so the X strip of this row lies 0.39 mm behind the cut (drawn light).

Layers carry short names only; every dimension is in the .tex caption (numbers from docs/detector_spec/spec.py, STACK /
ELEMENTS; status ok / cad / asm / open there; daggers = asm/open, in the caption).  Two scales:
(a) the chamber, z to scale (0.0515 in/mm), a cut through the centre in x, cut off (faded) on the right;
(b) the amplification gap and readout board, z at KV in/mm (30x (a)) and x at KH in/mm (z stretched KV/KH against x).
In (b) every width, pitch and thickness is to scale in its own direction (pad 0.68/0.78, resistive strip 550/800 um,
Y bus 98 um, via land 0.5 mm, layer thicknesses); the via barrel (0.15 mm) and the woven mesh are schematic, the
X strip is drawn solid.  Drawn in inches (one axes, data = inches).
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

ONLY_A = '--only-a' in sys.argv     # panel (a) alone, for the v2 Fig. 3 with the 3D panel (2026-11-14)
OUT = HERE / 'out' / ('chamber_stack_a' if ONLY_A else 'chamber_stack')
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
XI0, XI1 = 1.13, 2.62                    # inner gas; the chamber continues to the right (faded out from XF0)
XF0 = 2.28
XW = 0.08                                # left wall
T(0.0, 3.17, '(a)', va='top', fs=ps.BASE_PT, fontweight='bold')
if not ONLY_A:
    T(2.86, 3.17, '(b)', va='top', fs=ps.BASE_PT, fontweight='bold')
mm = dict(win=(0, 0.06), flange=(0.06, 5.06), cat=(5.06, 5.118), drift=(5.118, 35.118), amp=(35.118, 35.268),
          board=(35.268, 36.968), roh=(36.968, 41.968), myl=(41.968, 41.993), plate=(41.993, 49.993))
R(XI0, ya(mm['flange'][1]), XI1, ya(mm['flange'][0]), GAS, ec='none')
R(XI0, ya(mm['drift'][1]), XI1, ya(mm['drift'][0]), GAS, ec='none')
R(XI0 - XW, ya(0.5), XI1, ya(0), MYL, lw=0.4)
R(XI0, ya(0.25), XI1, ya(0.06), MYL, ec=MYL, lw=0.1)
R(XI0, ya(mm['cat'][1]) - 0.003, XI1, ya(mm['cat'][0]) + 0.003, KAP, lw=0.4)
line([XI0, XI1], [ya(mm['cat'][0]) + 0.004] * 2, c=CU, lw=1.4)
# left side only, widths in proportion to the frame (XW = its 15 mm wall; spec.py, CAD): Al window flange to its
# 400.2 mm aperture (19.9 mm), Al gas frame (15 mm), Al support plate (8 mm thick) to its 402 mm aperture (19 mm):
# no Al behind the active area.  Board, Rohacell and back Mylar are 470 mm plates offset +15 mm, so on this side they
# end at -220 mm, flush with the frame's outer face (Dylan 2026-11-04).
XO = XI0 - XW                                                      # outer face of the frame
wmm = lambda w: XO + XW * w / 15.0
R(XO, ya(mm['flange'][1]), wmm(19.9), ya(mm['flange'][0] + 0.5), AL)
R(XO, ya(mm['drift'][1]), XI0, ya(mm['drift'][0]), AL)
R(XO, ya(mm['plate'][1]), wmm(19.0), ya(mm['plate'][0]), '#e6e9ed', ec=MUTED, lw=0.35)   # light: not the point
line([XI0, XI1], [ya(mm['drift'][1])] * 2, c=INK, lw=1.0)
R(XO, ya(mm['board'][1]), XI1, ya(mm['board'][0]), FR4, lw=0.5)
R(XO, ya(mm['roh'][1]), XI1, ya(mm['roh'][0]), ROH, lw=0.5)
R(XO, ya(mm['myl'][1]) - 0.003, XI1, ya(mm['myl'][0]) + 0.003, MYL, lw=0.4)
LX = 1.00
XIN = XI0 + 0.12                                                  # leader tips: inside the layer, past the Al
lab = [('Mylar window', 3.05, 0.0, XIN + 0.06), ('gas gap', ya(2.6), 2.6, XIN),
       ('cathode', ya(5.09) - 0.07, 5.09, XIN), ('drift gas', ya(14), 14, XIN),
       ('readout board', 1.24, 36.1, XIN), ('Rohacell', 0.98, 39.5, XIN),
       ('back Mylar', 0.75, 41.98, XIN), ('Al plate', 0.52, 46.0, wmm(9.5))]
for s_, y, zz, xr in lab:
    T(LX, y, s_, ha='right', linespacing=1.0)
    line([LX + 0.02, xr], [y, ya(zz)], c=MUTED, lw=0.45, z=11)
    ax.plot(xr, ya(zz), 'o', ms=1.6, color=MUTED, mew=0, zorder=11)
# muon track, ionisation, electron drift
xt = lambda y: 1.50 + 0.10 * (Y0 + 0.05 - y)
ax.plot([xt(Y0 + 0.05), xt(ya(48))], [Y0 + 0.05, ya(48)], color=ps.ACCENT, lw=1.2, zorder=6)
ax.plot([xt(ya(5.2)), xt(ya(35))], [ya(5.2), ya(35)], color=ps.ACCENT, lw=1.6, zorder=6)
T(xt(Y0 + 0.05) + 0.05, Y0 + 0.08, 'muon', c=ps.ACCENT, fontstyle='italic', ha='left', va='center')
for zz in (8, 12.5, 17, 21.5, 26, 30):
    x, y = xt(ya(zz)), ya(zz)
    ax.plot(x, y, 'o', ms=2.0, color=ps.ACCENT, mew=0, zorder=7)
    arrow((x, y - 0.03), (x, ya(34.4)), c=ps.DET_COLOR['det2'], lw=0.5, style='-|>', ms=4)
T(xt(ya(28)) + 0.07, ya(27), 'e$^-$', ha='left', fs=FS, c=ps.DET_COLOR['det2'])
arrow((XF0 - 0.12, ya(32)), (XF0 - 0.12, ya(10.5)), c=INK, lw=1.0, style='-|>', ms=7)
T(XF0 - 0.20, ya(21), '$E_\\mathrm{drift}$', ha='center', rotation=90, fs=FS)
T(XF0 - 0.02, ya(5.09) - 0.13, '$-HV_\\mathrm{drift}$', ha='right', fs=FS, c=INK)
# cut-off: fade to white on the right (the chamber continues)
fade = np.zeros((2, 64, 4)); fade[..., :3] = 1.0; fade[..., 3] = np.linspace(0, 1, 64) ** 0.8
ax.imshow(fade, extent=[XF0, XI1 + 0.01, ya(50.2), 3.07], aspect='auto', zorder=10, interpolation='bilinear')
ax.set_xlim(0, W); ax.set_ylim(0, H)
BZ = (XI0 + 0.02, ya(36.2) - 0.05, XF0 - 0.02, ya(32.6))           # top crosses the electron lines above their heads
if ONLY_A:                                                         # no zoom box: nothing beside it to zoom into
    WA = XI1 + 0.04                                                # the axes spans the figure: crop it to (a)
    fig.set_size_inches(WA, H); ax.set_xlim(0, WA)
    fig.canvas.draw()
    ps.save(fig, OUT)
    print('ok', OUT)
    sys.exit(0)
R(BZ[0], BZ[1], BZ[2], BZ[3], 'none', ec=MUTED, lw=0.6, z=11, ls=(0, (3, 2)))

# ====================================================================================================== (b)
KV, KH = 1.55, 0.70                       # in / mm  (vertical z, horizontal x)
XB0, XB1 = 3.28, 5.19                     # drawing
YM = 2.42                                 # mesh centre plane
yb = lambda zmm: YM - KV * zmm
xc = (XB0 + XB1) / 2
PITCH, PADW = 0.78, 0.68
padx = [xc + (i - 1) * PITCH * KH for i in range(3)]               # three pads: Y, X, Y
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
R(XB0, yb(zl['gap'][0]), XB1, yb(-0.35), GAS, ec='none', z=0)
R(XB0, yb(zl['gap'][1]), XB1, yb(zl['gap'][0]), '#e3eef7', ec='none', z=0)
# board body: one FR4 block (L3 guard ring is outside the active area, L7 routing copper is not in the cut), kapton on top
R(XB0, yb(ZEND), XB1, yb(zl['ins'][1]), FR4, ec='none', z=1)
R(XB0, yb(zl['ins'][1]), XB1, yb(zl['ins'][0]), KAP, lw=0.4)
# resistive strips: 550 um wide on an 800 um pitch (their phase against the pads is not known: drawn arbitrarily)
rp = 0.800 * KH
for i in range(-2, 8):
    xa = XB0 + (i + 0.35) * rp
    xa0, xa1 = max(xa, XB0), min(xa + 0.55 * KH, XB1)
    if xa1 > xa0:
        R(xa0, yb(zl['esl'][1]), xa1, yb(zl['esl'][0]), ESL, lw=0.3, z=4)
# mesh: woven wires, 19 um on a 67 um pitch (schematic)
nw = int((XB1 - XB0) / (0.067 * KH))
for i in range(nw):
    xw = XB0 + (i + 0.5) * (XB1 - XB0) / nw
    ax.add_patch(Ellipse((xw, yb(0)), 0.019 * KH, 0.019 * KV, fc='#777777', ec='none', zorder=5))
# L4 pads
for x in padx:
    R(x - PADW * KH / 2, yb(zl['L4'][1]), x + PADW * KH / 2, yb(zl['L4'][0]), CU, ec='none', z=4)
# vias and strip copper, as in the gerbers (see the docstring).  Cut through a row of pad centres: pads Y, X, Y.
VIA, LAND, LAND7 = 0.25, 0.50, 0.65                                # drill, L5/L6 land, L7 land (mm)
lay = lambda k: (yb(zl[k][1]), yb(zl[k][0]))
view = ('Y', 'X', 'Y')
# L6: the X strip of this row, 0.39 mm behind the cut (its stubs run along y), drawn light
XLIGHT = tuple(1 - 0.40 * (1 - c) for c in matplotlib.colors.to_rgb(XC))
R(XB0, lay('L6')[0], XB1, lay('L6')[1], XLIGHT, ec='none', z=3)
# L5: the Y buses (98 um) in every gap between pad columns, cut across
for i in range(4):
    xbus = xc + (i - 1.5) * PITCH * KH
    R(xbus - 0.049 * KH, lay('L5')[0], xbus + 0.049 * KH, lay('L5')[1], YC, ec='none', z=4)
xa_ = padx[1] - 0.10
for i, x in enumerate(padx):
    col = YC if view[i] == 'Y' else XC
    # barrel L4 -> L7: plated wall (drawn 30 um) round a hole whose fill is not in the spec (drawn as a light tint)
    tint = tuple(1 - 0.30 * (1 - c) for c in matplotlib.colors.to_rgb(col))
    R(x - VIA / 2 * KH, lay('L7')[0], x + VIA / 2 * KH, yb(zl['L4'][0]), tint, ec='none', z=5)
    for sx_ in (-1, 1):
        R(x + sx_ * VIA / 2 * KH - (sx_ > 0) * 0.030 * KH, lay('L7')[0], x + sx_ * VIA / 2 * KH + (sx_ < 0) * 0.030 * KH,
          yb(zl['L4'][0]), col, ec='none', z=6)
    for k, w in (('L5', LAND), ('L6', LAND), ('L7', LAND7)):                                       # a land per layer
        R(x - w / 2 * KH, lay(k)[0], x + w / 2 * KH, lay(k)[1], col, ec='none', z=7)
    if view[i] == 'Y':                                                                             # stub to its bus
        R(x, lay('L5')[0], x + PITCH / 2 * KH, lay('L5')[1], YC, ec='none', z=7)
# frame
line([XB0, XB0], [yb(-0.35), yb(ZEND)], c=INK, lw=0.5)
line([XB1, XB1], [yb(-0.35), yb(ZEND)], c=INK, lw=0.5)
line([XB0, XB1], [yb(ZEND)] * 2, c=INK, lw=0.5)
# electrons arriving, avalanche under a mesh hole
for xe in (xa_ - 0.17, xa_ - 0.03, xa_ + 0.11):
    arrow((xe, yb(-0.33)), (xe, yb(-0.02)), c=ps.DET_COLOR['det2'], lw=0.6, style='-|>', ms=4)
yt, yb2 = yb(zl['gap'][0]), yb(zl['gap'][1])
ax.add_patch(Polygon([(xa_ - 0.04, yt), (xa_ + 0.04, yt), (xa_ + 0.27, yb2), (xa_ - 0.27, yb2)], closed=True,
                     fc=ps.ACCENT, ec='none', alpha=0.30, zorder=3))
rng = np.random.default_rng(3)
for _ in range(34):
    f = rng.uniform(0.05, 1.0)
    ax.plot(xa_ + rng.normal(0, 0.03 + 0.07 * f), yt + (yb2 - yt) * f, '.', ms=2.0, color=ps.ACCENT, mew=0, zorder=6)
xe_ = XB0 + 0.07
arrow((xe_, yb2 + 0.01), (xe_, yt - 0.01), c=INK, lw=1.0, style='-|>', ms=6)
T(xe_ + 0.06, (yt + yb2) / 2, '$E_\\mathrm{amp}$', ha='left', fs=FS)
# labels (right): names only; dimensions are in the caption
RX = XB1 + 0.14
ent = [
    (yb(0.0), 2.80, 'mesh (0 V)'),
    ((yt + yb2) / 2, 2.57, 'amplification gap'),
    (yb(sum(zl['esl']) / 2), 2.34, 'resistive strips (+HV)'),
    (yb(sum(zl['ins']) / 2), 2.11, 'kapton'),
    (yb(sum(zl['L4']) / 2), 1.88, 'pads'),
    (yb(sum(zl['L5']) / 2), None, 'Y strips'),
    (yb(sum(zl['L6']) / 2), None, 'X strip (behind the cut)'),
    (yb(sum(zl['L7']) / 2), None, 'via lands (L7)'),
    (yb(zl['f4'][0] + 0.5 * (zl['f4'][1] - zl['f4'][0])), None, 'FR4'),
]
for y, yl, s_ in ent:
    yl = y if yl is None else yl
    T(RX, yl, s_, linespacing=1.0, fs=FS)
    line([XB1 + 0.02, RX - 0.03], [y, yl], c=MUTED, lw=0.4, z=1)
for ya_, yb_ in ((BZ[3], yb(-0.35)), (BZ[1], yb(ZEND))):
    ax.plot([BZ[2], XB0], [ya_, yb_], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=0)
rows = pd.DataFrame([dict(layer=k, z_lo_mm=v[0], z_hi_mm=v[1]) for k, v in zl.items()])
fig.canvas.draw()
ps.save(fig, OUT, data=rows)
print('ok', OUT, 'board bottom z', ZEND, 'KV/KH', KV / KH, 'KV/KA', KV / KA)
