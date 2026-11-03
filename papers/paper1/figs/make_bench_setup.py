#!/usr/bin/env python3
"""make_bench_setup.py -- Paper I, fig.bench_setup: the Saclay cosmic bench, side view in the physical orientation: z vertical, muons going down (Dylan 2026-11-03).

    .venv/bin/python papers/paper1/figs/make_bench_setup.py

Rework of mx_june_cosmic_qa/engineer_package/make_bench_diagram.py, same numbers.  Geometry from the run configs
(run_config.json, detectors[].det_center_coords z, mm):
  M3 reference planes  z = 24, 144 (below), 1185, 1302 (above)       [m3_bot_bot, m3_bot_top, m3_top_bot, m3_top_top]
  test slots           z = 232 (bottom), 702 (top)                   [P2_1 / mx17_3 in mx17_det3_p2_det1_overnight_6-27-26]
  chamber A (det3)     top slot, z = 702 (both golden rows, p2_overnight and saturday_scan)
  scintillators        60 x 60 cm, one each side of the stack; their z was never recorded: drawn just outside the
                       stack at schematic positions, no z label.
z is drawn compressed to ZC = 0.35 of the transverse scale (a scale bar shows both); slab thicknesses are exaggerated.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / 'papers')]
import paper_style as ps                                           # noqa: E402

OUT = HERE / 'out' / 'bench_setup'
ps.use()
plt.rcParams['savefig.pad_inches'] = 0.01
plt.rcParams['axes.grid'] = False
FS = ps.BASE_PT * 0.9

M3_Z = [24, 144, 1185, 1302]
SLOT_Z = {'bottom': 232, 'top': 702}
SCINT_Z = {'bottom': -95, 'top': 1405}                              # schematic, not measured
M3_W, MX_W, SC_W = 500, 400, 600
ZC = 0.35                                                           # vertical compression of z
M3_T, MX_T, SC_T = 10, 14, 12                                        # drawn thicknesses, plot units (not to scale)
M3_C, SC_C, OTHER_C, A_C = '#cfd8e3', '#f0d58c', '#b9c4d2', ps.DET_COLOR['det3']
LW = 0.7

W, H = 6.1, 2.55
fig = plt.figure(figsize=(W, H))
AX_W, AX_H = 6.0, 2.40
XL, XR, YB, YT = -640, 1000, -95, 560                                # x: transverse mm; y: ZC * z [mm]
ax = fig.add_axes([0.05 / W, 0.05 / H, AX_W / W, AX_H / H])
ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.axis('off')
Y = lambda z: ZC * z


def slab(z, w, t, fc, lw=LW, zo=3):
    ax.add_patch(Rectangle((-w / 2, Y(z) - t / 2), w, t, fc=fc, ec=ps.INK, lw=lw, zorder=zo))


for z in SCINT_Z.values():
    slab(z, SC_W, SC_T, SC_C)
for z in M3_Z:
    slab(z, M3_W, M3_T, M3_C)
slab(SLOT_Z['bottom'], MX_W, MX_T, A_C, lw=1.0)
slab(SLOT_Z['top'], MX_W, MX_T, A_C, lw=1.0)

# muon, going down
x0, z0, x1, z1 = 150, 1510, -135, -150
ax.annotate('', (x1, Y(z1)), (x0, Y(z0)), arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.3, mutation_scale=9,
                                                          shrinkA=0, shrinkB=0), zorder=6)
xat = lambda z: x0 + (x1 - x0) * (z - z0) / (z1 - z0)
for z in M3_Z + [SLOT_Z['bottom'], SLOT_Z['top']]:
    ax.plot(xat(z), Y(z), 'o', ms=2.6, color=ps.ACCENT, mec='white', mew=0.4, zorder=7)
ax.text(xat(950) - 45, Y(950), 'cosmic\nmuon', color=ps.ACCENT, fontsize=FS, fontstyle='italic', ha='right',
        va='center', zorder=7, linespacing=1.05)

# right column: labels
XT = 335
ax.text(XT, Y(SCINT_Z['top']), 'trigger scintillator', ha='left', va='center', fontsize=FS,
        color=ps.INK)
ax.text(XT, Y(SCINT_Z['bottom']), 'trigger scintillator', ha='left', va='center', fontsize=FS, color=ps.INK)
for (za, zb), lab in (((M3_Z[2], M3_Z[3]), 'M3 reference planes\n(2 + 2)'),
                      ((M3_Z[0], M3_Z[1]), 'M3 reference planes')):
    xb = 275
    ax.plot([xb - 12, xb, xb, xb - 12], [Y(za), Y(za), Y(zb), Y(zb)], color=ps.INK, lw=LW)
    t = ax.text(XT, Y((za + zb) / 2), lab, ha='left', va='center', fontsize=FS, color=ps.INK, linespacing=1.05)
    t.set_gid(ps.KEEP_GID)
ax.text(XT - 90, Y(SLOT_Z['top']), 'MX17 (top slot)', color=A_C, fontweight='bold', fontsize=FS, ha='left', va='center')
ax.text(XT - 90, Y(SLOT_Z['bottom']), 'MX17 (bottom slot)', color=A_C, fontweight='bold', fontsize=FS, ha='left', va='center')

# left column: z ticks and lever arm
XTK = -285
for z in M3_Z + [SLOT_Z['bottom'], SLOT_Z['top']]:
    col = A_C if z in SLOT_Z.values() else ps.INK
    ax.plot([XTK + 10, XTK + 40], [Y(z)] * 2, color=col, lw=LW)
    ax.text(XTK, Y(z), f'{z}', ha='right', va='center', fontsize=FS * 0.95, color=col,
            fontweight='bold' if z in SLOT_Z.values() else 'normal')
ax.text(XTK, Y(1500), '$z$ [mm]', ha='right', va='center', fontsize=FS, color=ps.INK)

# small scale bars (200 mm), tucked into the empty space right of the muon, above the top-slot label
xs0, ys0, L = 760, 285, 200
ax.plot([xs0, xs0 + L], [ys0, ys0], color=ps.INK, lw=1.0)
ax.plot([xs0, xs0], [ys0, ys0 + ZC * L], color=ps.INK, lw=1.0)
ax.text(xs0 + L / 2, ys0 - 10, '200 mm', ha='center', va='top', fontsize=FS * 0.9, color=ps.INK)
ax.text(xs0 + 12, ys0 + ZC * L / 2, '200 mm in $z$', ha='left', va='center', fontsize=FS * 0.9, color=ps.INK)

geom = pd.DataFrame(
    [dict(element='scintillator', z_mm=z, width_mm=SC_W, z_measured=False) for z in SCINT_Z.values()]
    + [dict(element='M3', z_mm=z, width_mm=M3_W, z_measured=True) for z in M3_Z]
    + [dict(element=f'MX17 {k} slot', z_mm=z, width_mm=MX_W, z_measured=True) for k, z in SLOT_Z.items()])
fig.savefig(OUT.with_suffix('.tmp.png'), dpi=200)
fig.canvas.draw(); _b = fig.get_tightbbox(fig.canvas.get_renderer()); print('content bbox in', round(_b.x0, 2), round(_b.x1, 2), 'w', round(_b.width, 2), 'h', round(_b.height, 2))
ps.save(fig, OUT, data=geom)
print('ok', OUT)
