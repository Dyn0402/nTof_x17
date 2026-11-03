#!/usr/bin/env python3
"""make_sps_setup.py -- Paper I, fig.sps_setup: the SPS H4 setup as a 2D schematic in the style of the bench figure
(Dylan 2026-11-03: "more schematic, like the bench view; we don't need the P2 detectors").

    .venv/bin/python papers/paper1/figs/make_sps_setup.py

Plan view (looking down), beam along z, drawn to scale in both directions (slab thicknesses exaggerated).
Geometry: mpgd26/geometry.py SPS_STATIONS = run_59/run_config.json det_center_coords z and det_orientation
(sps_beam_test_26/det4_sps_assessment/SPS_BEAM_GEOMETRY_2026-07-31.md sect. 1):
  EIC uRWELL front z = 0, back z = 1370 mm (127.4 mm active, URW_ACTIVE_MM)
  chamber E (mx17_4) z = 1155 mm: run-config NOMINAL, flagged PLACEHOLDER in the config itself (not surveyed);
                    yawed 25.64 deg about the vertical (det_orientation.y), a real mounting angle.
  transverse x = y = 0 for all stations: nominal, never surveyed.
The P2 BASKET sectors (z = 320, 630, 940) are left out on purpose.
ASSUMPTIONS, not checked against the run config: the sign of the yaw (drawn so the chamber normal points away from the
beam axis towards +x), and the sides of the drift window (upstream) and readout PCB (downstream), i.e. drift along the beam direction.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Arc
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / 'papers')]
import paper_style as ps                                           # noqa: E402

OUT = HERE / 'out' / 'sps_setup'
ps.use()
plt.rcParams['savefig.pad_inches'] = 0.01
plt.rcParams['axes.grid'] = False
FS = ps.BASE_PT * 0.9
LW = 0.7

URW_Z = {'front': 0.0, 'back': 1370.0}
E_Z, E_YAW = 1155.0, 25.64
URW_W, URW_T = 127.4, 14                                            # active width; drawn thickness
E_W, E_T = 400, 18
URW_C, E_C = '#cfd8e3', ps.DET_COLOR['det4']

W, H = 6.1, 1.85
fig = plt.figure(figsize=(W, H))
AX_W, AX_H = 6.0, 1.75
ZL, ZR, XB, XT_ = -400, 1580, -370, 215
ax = fig.add_axes([0.05 / W, 0.05 / H, AX_W / W, AX_H / H])
ax.set_xlim(ZL, ZR); ax.set_ylim(XB, XT_); ax.axis('off')
print('scales in/mm  z', AX_W / (ZR - ZL), ' x', AX_H / (XT_ - XB))
ax.set_aspect('equal', adjustable='datalim')

for z in URW_Z.values():
    ax.add_patch(Rectangle((z - URW_T / 2, -URW_W / 2), URW_T, URW_W, fc=URW_C, ec=ps.INK, lw=LW, zorder=3))

th = np.radians(E_YAW)
n = np.array([np.cos(th), np.sin(th)])                              # chamber normal in (z, x)
t = np.array([-np.sin(th), np.cos(th)])                             # in-plane direction
c = np.array([E_Z, 0.0])
ax.add_patch(Polygon([c + a * E_W / 2 * t + b * E_T / 2 * n for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                     fc=E_C, ec=ps.INK, lw=1.0, zorder=4))

# beam
ax.annotate('', (ZR - 40, 0), (ZL + 20, 0), arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.3,
                                                             mutation_scale=9, shrinkA=0, shrinkB=0), zorder=6)
for z in list(URW_Z.values()) + [E_Z]:
    ax.plot(z, 0, 'o', ms=2.6, color=ps.ACCENT, mec='white', mew=0.4, zorder=7)
ax.text(ZL + 20, 28, 'SPS H4 beam', color=ps.ACCENT, fontsize=FS, fontstyle='italic', ha='left', va='bottom')

# angle to the beam
ax.plot([E_Z, E_Z + 175], [0, 0], color=ps.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=2)   # beam reference for the angle
ax.add_patch(Arc((E_Z, 0), 120, 120, theta1=0, theta2=E_YAW, color=ps.INK, lw=LW, zorder=8))
ax.text(E_Z + 55, -26, '25.6°', fontsize=FS * 0.95, color=ps.INK, ha='left', va='center')

# the two faces of chamber E: drift window upstream (front), readout PCB downstream (back).  ASSUMED sense (cathode
# upstream, drift along the beam), as before.  Horizontal arrows at transverse x = XA, ending on the faces.
XA = 90
zf = lambda sgn: E_Z + (sgn * E_T / 2 - np.sin(th) * XA) / np.cos(th)
arr = dict(arrowstyle='-|>', color=ps.INK, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0)
ax.annotate('', (zf(-1) - 3, XA), (zf(-1) - 130, XA), arrowprops=arr, zorder=8)
ax.text(zf(-1) - 138, XA, 'drift window', fontsize=FS, color=ps.INK, ha='right', va='center')
ax.annotate('', (zf(+1) + 3, XA), (zf(+1) + 130, XA), arrowprops=arr, zorder=8)
ax.text(zf(+1) + 138, XA, 'readout PCB', fontsize=FS, color=ps.INK, ha='left', va='center')

# labels
ax.text(URW_Z['front'], -URW_W / 2 - 22, 'uRWELL reference\n(front)', fontsize=FS, color=ps.INK, ha='center',
        va='top', linespacing=1.05)
ax.text(1255, -URW_W / 2 - 22, 'uRWELL reference\n(back)', fontsize=FS, color=ps.INK, ha='left', va='top',
        linespacing=1.05)
ax.text(E_Z - 95, -E_W / 2 * np.cos(th) - 5, 'chamber E', fontsize=FS, color=ps.INK, fontweight='bold',
        ha='center', va='top')

# z axis
yz = -330
ax.plot([0, URW_Z['back']], [yz, yz], color=ps.INK, lw=LW)
for z, lab in ((0, '0'), (E_Z, '1155'), (URW_Z['back'], '1370')):
    ax.plot([z, z], [yz, yz + 14], color=ps.INK, lw=LW)
    ax.text(z, yz - 12, lab, fontsize=FS * 0.95, color=ps.INK, ha='center', va='top')
ax.text(URW_Z['back'] + 40, yz + 4, '$z$ [mm]', fontsize=FS, color=ps.INK, ha='left', va='bottom')
ax.text(ZL, yz + 4, 'plan view, to scale', fontsize=FS * 0.95, color=ps.MUTED, ha='left', va='bottom',
        linespacing=1.1)

geom = pd.DataFrame([dict(element='uRWELL front', z_mm=0.0, yaw_deg=0.0, z_measured=True),
                     dict(element='uRWELL back', z_mm=1370.0, yaw_deg=0.0, z_measured=True),
                     dict(element='chamber E', z_mm=E_Z, yaw_deg=E_YAW, z_measured=False)])
fig.canvas.draw(); _b = fig.get_tightbbox(fig.canvas.get_renderer()); print('content bbox in', round(_b.x0, 2), round(_b.x1, 2), 'w', round(_b.width, 2), 'h', round(_b.height, 2))
ps.save(fig, OUT, data=geom)
print('ok', OUT)
