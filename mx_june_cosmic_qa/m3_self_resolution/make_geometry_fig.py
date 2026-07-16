#!/usr/bin/env python3
"""Geometry / method schematic for the M3 self-resolution study."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

Z = [1302, 1185, 144, 24]
fig, ax = plt.subplots(figsize=(7.2, 8.4))

# a slightly kinked cosmic track (bottom -> top), kink in the gap
zt = np.array([0, 24, 144, 702, 1185, 1302, 1340])
xt = np.array([-15, -12, -6, 12, 30, 35, 37]) * 1.0
# recompute a smooth-ish path: two straight segments with a small kink near DUT
def track_x(z):
    if z < 663:
        return -14 + (z - 24) * (6 / 120)     # bottom-doublet slope
    else:
        return track_x(662.9) + (z - 663) * (10 / 639)
zz = np.linspace(-30, 1350, 200)
xx = np.array([track_x(z) for z in zz])
ax.plot(xx, zz, color='crimson', lw=1.6, zorder=5, label='cosmic muon (with MS kink)')

# stations (each: X plane + Y plane, drawn as one bar)
for i, z in enumerate(Z):
    ax.add_patch(plt.Rectangle((-60, z - 6), 120, 12, color='steelblue', alpha=0.85, zorder=3))
    ax.text(70, z, f'Station L{i}\n z = {z} mm  (X+Y)', va='center', fontsize=9)

# doublet brackets
ax.annotate('', xy=(-80, 1302), xytext=(-80, 1185),
            arrowprops=dict(arrowstyle='<->', color='navy'))
ax.text(-92, 1243, 'top doublet\n$\\Delta z$=117 mm', rotation=90, va='center',
        ha='right', fontsize=8, color='navy')
ax.annotate('', xy=(-80, 144), xytext=(-80, 24),
            arrowprops=dict(arrowstyle='<->', color='navy'))
ax.text(-92, 84, 'bottom doublet\n$\\Delta z$=120 mm', rotation=90, va='center',
        ha='right', fontsize=8, color='navy')

# DUT slots
for z, lab in [(232, 'DUT slot z=232'), (702, 'DUT slot z=702')]:
    ax.add_patch(plt.Rectangle((-45, z - 4), 90, 8, color='green', alpha=0.30, zorder=2))
    ax.plot([-45, 45], [z, z], color='green', ls='--', lw=1.1)
    ax.text(70, z, lab, va='center', fontsize=8.5, color='green')

# lever-arm annotation
ax.annotate('', xy=(150, 1185), xytext=(150, 144),
            arrowprops=dict(arrowstyle='<->', color='gray'))
ax.text(158, 664, '~1041 mm lever arm\n(interpolation region)', rotation=90,
        va='center', fontsize=8, color='gray')

ax.set_xlim(-160, 260)
ax.set_ylim(-60, 1380)
ax.set_ylabel('z  [mm]')
ax.set_xlabel('measured coordinate  [mm]')
ax.set_title('MX17 "M3" reference telescope\n4 stations $\\times$ (X,Y) = two doublets straddling the DUT')
ax.legend(loc='upper left', fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figs/geometry.png', dpi=150, bbox_inches='tight')
print('wrote figs/geometry.png')
