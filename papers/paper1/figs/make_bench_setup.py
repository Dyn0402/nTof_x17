#!/usr/bin/env python3
"""make_bench_setup.py -- Paper I, fig.bench_setup: the Saclay cosmic bench, side view with z horizontal (Dylan 2026-11-03).

    .venv/bin/python papers/paper1/figs/make_bench_setup.py

Rework of mx_june_cosmic_qa/engineer_package/make_bench_diagram.py, same numbers.  Geometry from the run configs
(run_config.json, detectors[].det_center_coords z, mm):
  M3 reference planes  z = 24, 144 (below), 1185, 1302 (above)       [m3_bot_bot, m3_bot_top, m3_top_bot, m3_top_top]
  test slots           z = 232 (bottom), 702 (top)                   [P2_1 / mx17_3 in mx17_det3_p2_det1_overnight_6-27-26]
  chamber A (det3)     top slot, z = 702 (both golden rows, p2_overnight and saturday_scan)
  scintillators        60 x 60 cm, one each side of the stack; their z was never recorded: drawn just outside the
                       stack at schematic positions, no z label.
z positions are to scale; the transverse direction is drawn at 0.70 of the z scale and slab thicknesses are exaggerated.
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
M3_T, MX_T, SC_T = 16, 26, 22                                       # drawn thicknesses (not to scale)
M3_C, SC_C, OTHER_C, A_C = '#cfd8e3', '#f0d58c', '#b9c4d2', ps.DET_COLOR['det3']
LW = 0.7

W, H = 6.1, 2.40
fig = plt.figure(figsize=(W, H))
AX_W, AX_H = 5.95, 2.05
ZL, ZR, YB, YT = -230, 1520, -455, 405                               # mm
ax = fig.add_axes([0.08 / W, 0.12 / H, AX_W / W, AX_H / H])
ax.set_xlim(ZL, ZR); ax.set_ylim(YB, YT); ax.axis('off')
kz, ky = AX_W / (ZR - ZL), AX_H / (YT - YB)                          # in per mm
print('transverse/z scale', ky / kz)


def slab(z, w, t, fc, lw=LW, zo=3):
    ax.add_patch(Rectangle((z - t / 2, -w / 2), t, w, fc=fc, ec=ps.INK, lw=lw, zorder=zo))


for z in SCINT_Z.values():
    slab(z, SC_W, SC_T, SC_C)
for z in M3_Z:
    slab(z, M3_W, M3_T, M3_C)
slab(SLOT_Z['bottom'], MX_W, MX_T, OTHER_C)
slab(SLOT_Z['top'], MX_W, MX_T, A_C, lw=1.0)

# muon
z0, y0, z1, y1 = 1405, 120, -60, -125
ax.annotate('', (z1, y1), (z0, y0), arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.3, mutation_scale=9,
                                                    shrinkA=0, shrinkB=0), zorder=6)
yat = lambda z: y0 + (y1 - y0) * (z - z0) / (z1 - z0)
for z in M3_Z + [SLOT_Z['bottom'], SLOT_Z['top']]:
    ax.plot(z, yat(z), 'o', ms=2.6, color=ps.ACCENT, mec='white', mew=0.4, zorder=7)
ax.text(470, yat(470) + 62, 'cosmic muon', color=ps.ACCENT, fontsize=FS, fontstyle='italic', ha='center',
        va='bottom', zorder=7)

# trigger scintillators: rotated labels outside the stack
ax.text(SCINT_Z['bottom'] - 40, 0, 'trigger scintillator', rotation=90, ha='right', va='center', fontsize=FS,
        color=ps.INK)
ax.text(SCINT_Z['top'] + 40, 0, 'trigger scintillator', rotation=270, ha='left', va='center', fontsize=FS,
        color=ps.INK)

# M3 bracket: (2 + 2) reference planes
yb_ = 345
for za, zb in ((M3_Z[0], M3_Z[1]), (M3_Z[2], M3_Z[3])):
    ax.plot([za, za, zb, zb], [yb_ - 28, yb_, yb_, yb_ - 28], color=ps.INK, lw=LW)
ax.plot([M3_Z[1], M3_Z[2]], [yb_, yb_], color=ps.INK, lw=LW)
t = ax.text((M3_Z[1] + M3_Z[2]) / 2, yb_, 'M3 reference planes (2 + 2)', ha='center', va='center', fontsize=FS,
            color=ps.INK, bbox=dict(fc='white', ec='none', pad=1.5))
t.set_gid(ps.KEEP_GID)
# chamber labels
ax.text(SLOT_Z['top'], 225, 'chamber A (top slot)', color=A_C, fontweight='bold', fontsize=FS, ha='center', va='bottom')
ax.text(SLOT_Z['bottom'] + 30, 225, 'other MX17\nchamber', color=ps.INK, fontsize=FS, ha='center', va='bottom', linespacing=1.05)

# lever arm and z ticks
yd = -372
ax.annotate('', (M3_Z[0], yd), (M3_Z[3], yd), arrowprops=dict(arrowstyle='<->', lw=LW, color=ps.INK, shrinkA=0,
                                                              shrinkB=0, mutation_scale=7))
ax.text((M3_Z[0] + M3_Z[3]) / 2, yd, 'reference lever arm 1278 mm', ha='center', va='center', fontsize=FS,
        color=ps.INK, bbox=dict(fc='white', ec='none', pad=1.5))
yt = -415
for z in M3_Z + [SLOT_Z['bottom'], SLOT_Z['top']]:
    col = A_C if z == SLOT_Z['top'] else ps.INK
    ax.plot([z, z], [yt + 22, yt + 40], color=col, lw=LW)
    ax.text(z, yt + 10, f'{z}', ha='center', va='top', fontsize=FS * 0.95, color=col,
            fontweight='bold' if z == SLOT_Z['top'] else 'normal')
ax.text(ZR - 5, yt + 10, '$z$ [mm]', ha='right', va='top', fontsize=FS, color=ps.INK)

geom = pd.DataFrame(
    [dict(element='scintillator', z_mm=z, width_mm=SC_W, z_measured=False) for z in SCINT_Z.values()]
    + [dict(element='M3', z_mm=z, width_mm=M3_W, z_measured=True) for z in M3_Z]
    + [dict(element=f'MX17 {k} slot', z_mm=z, width_mm=MX_W, z_measured=True) for k, z in SLOT_Z.items()])
fig.savefig(OUT.with_suffix('.tmp.png'), dpi=200)
fig.canvas.draw(); _b = fig.get_tightbbox(fig.canvas.get_renderer()); print('content bbox in', round(_b.x0, 2), round(_b.x1, 2), 'w', round(_b.width, 2), 'h', round(_b.height, 2))
ps.save(fig, OUT, data=geom)
print('ok', OUT)
