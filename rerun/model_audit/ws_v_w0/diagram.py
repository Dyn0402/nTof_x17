#!/usr/bin/env python3
"""diagram.py -- schematic of why a chamber tilted against the telescope reads a perpendicular track as inclined,
and why the same tilt shows in the position residuals (figures/w0_tilt_diagram.png).  Tilt exaggerated ~20x."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
a = np.radians(10.0)                      # drawn tilt (the real ones are 0.1-0.9 deg)
c, s = np.cos(a), np.sin(a)
L, G = 4.0, 1.2                           # chamber length, drift gap (drawing units)


def xy(u, z):                             # chamber frame (u along the mesh, z along its normal) -> drawing
    return np.array([u * c - z * s, u * s + z * c])


fig, ax = plt.subplots(figsize=(11, 5.6))
for z, lab, col, lw in ((0, 'mesh (strips below)', '#4C72B0', 3), (G, 'cathode', '0.45', 2)):
    p0, p1 = xy(-L / 2, z), xy(L / 2, z)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=lw)
    ax.text(p0[0] - 0.1, p0[1], lab, ha='right', va='center', fontsize=10, color=col)
# vertical muon at drawing x = X0
X0 = 0.5
ax.plot([X0, X0], [-0.9, 2.4], color='#C44E52', lw=2.4)
ax.text(X0 + 0.07, 2.35, 'muon, vertical for M3 (tan_ref = 0)', color='#C44E52', fontsize=10, va='top')
# drift paths along the normal from points of the track inside the gap
for zc in np.linspace(0.15, G - 0.1, 5):
    # point of the vertical line with chamber-frame height zc:  -u s + zc c ... solve x = u c - zc s = X0
    u = (X0 + zc * s) / c
    ps, pm = xy(u, zc), xy(u, 0)
    ax.annotate('', xy=pm, xytext=ps, arrowprops=dict(arrowstyle='->', color='0.3', lw=1.1))
    ax.plot(*pm, 'o', color='#DD8452', ms=6)
pm0, pm1 = xy((X0 + 0.15 * s) / c, 0), xy((X0 + (G - 0.1) * s) / c, 0)
ax.text(2.6, -0.95, 'electrons drift along the chamber normal (grey):\n'
        'the landing point (orange) moves along the strips\nwith depth, so a perpendicular track reads\n'
        'w = w0 ≠ 0, i.e. an angle offset α = w0 / v', fontsize=10, va='top')
# the normal vs vertical
pn = xy(-1.5, 0); pn2 = xy(-1.5, 1.45)
ax.annotate('', xy=pn2, xytext=pn, arrowprops=dict(arrowstyle='->', color='#55A868', lw=1.8))
ax.plot([pn[0], pn[0]], [pn[1], pn[1] + 1.5], ':', color='#55A868', lw=1.5)
ax.text(pn[0] - 0.15, pn[1] + 1.55, 'tilt α', color='#55A868', fontsize=11, ha='center')
# mesh height difference across the chamber
pa, pb = xy(-L / 2, 0), xy(L / 2, 0)
xr = pb[0] + 0.35
ax.plot([pa[0], xr], [pa[1], pa[1]], ':', color='0.6', lw=1)
ax.annotate('', xy=(xr, pb[1]), xytext=(xr, pa[1]), arrowprops=dict(arrowstyle='<->', color='0.2'))
ax.text(xr + 0.1, 0.5 * (pa[1] + pb[1]), 'mesh height differs across the chamber,\nso where an inclined track crosses\n'
        'the mesh shifts by tan·Δz:\nposition residual ∝ tan × position\n(an independent measurement of α)',
        fontsize=9.5, va='center')
ax.set_xlim(-4.3, 6.6); ax.set_ylim(-1.9, 2.6)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('What w0 is: the chamber sits tilted against the telescope (tilt drawn ×20)', fontsize=12)
fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, 'w0_tilt_diagram.png'), dpi=140)
