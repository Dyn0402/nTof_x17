#!/usr/bin/env python3
"""fig_concepts.py OUTDIR -- the F16 explainer diagram: what 'column duration', 'ladder' and 'span route' mean."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

out = sys.argv[1] if len(sys.argv) > 1 else '.'
INK, MUTED, ACC, ACC2, WARN = '#1f2933', '#7b8794', '#1f6feb', '#d97706', '#b42318'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED})

fig = plt.figure(figsize=(15, 5.6))
# ---- panel 1: the chamber ----------------------------------------------------------------------------------
ax = fig.add_axes([0.02, 0.08, 0.36, 0.84]); ax.set_xlim(-2, 20); ax.set_ylim(-6, 34); ax.axis('off')
ax.add_patch(Rectangle((0, 0), 17, 30, fc='#f5f7fa', ec=MUTED, lw=1))
ax.plot([0, 17], [30, 30], color=INK, lw=3); ax.text(17.3, 30, 'cathode', va='center', color=INK)
ax.plot([0, 17], [0, 0], color=INK, lw=3, ls=(0, (2, 1))); ax.text(17.3, 0, 'mesh +\nstrips', va='center', color=INK)
th = np.radians(25.64); x0 = 2.0
ax.plot([x0 + 30 * np.tan(th) + 1.2, x0 - 1.2 * np.tan(th) * 0 - 0.6], [32.5, -2.5], color=ACC, lw=2.5)
ax.text(x0 + 30 * np.tan(th) + 1.5, 32.6, 'beam particle\n(25.64° mount)', color=ACC, fontsize=11)
for z in np.arange(2, 30, 4):
    xa = x0 + z * np.tan(th) + (32.5 - 30) * 0
    xa = x0 + (z + 2.5) * (np.tan(th) * 30 + 1.2 + 0.6) / 35.0 - 0.6 + 0.0
    ax.add_patch(FancyArrowPatch((xa, z), (xa, 0.4), arrowstyle='-|>', mutation_scale=9, color=ACC2, lw=1))
ax.annotate('', xy=(-0.8, 0), xytext=(-0.8, 30), arrowprops=dict(arrowstyle='<->', color=INK))
ax.text(-1.6, 15, 'gap\n28.8–30 mm', ha='right', va='center', rotation=90, color=INK)
ax.annotate('', xy=(x0 - 0.6 + 0.2, -3.6), xytext=(x0 + 30 * np.tan(th) + 0.6, -3.6), arrowprops=dict(arrowstyle='<->', color=INK))
ax.text(x0 + 7, -5.6, 'lateral extent  L = gap × tan θ', ha='center', color=INK)
ax.text(8.5, 21, 'electrons drift\ndown at speed v', color=ACC2, fontsize=11)
ax.set_title('one track, side view', color=INK, loc='left')

# ---- panel 2: what the strips see in time -----------------------------------------------------------------
ax = fig.add_axes([0.43, 0.16, 0.25, 0.68])
t = np.linspace(0, 3.2, 400); T = 2.0
box = ((t > 0.6) & (t < 0.6 + T)).astype(float)
ax.fill_between(t, 0, box, color=ACC2, alpha=0.25, lw=0)
ax.plot(t, box, color=ACC2, lw=2)
ax.annotate('', xy=(0.6, 1.12), xytext=(0.6 + T, 1.12), arrowprops=dict(arrowstyle='<->', color=INK))
ax.text(0.6 + T / 2, 1.18, 'column duration  T = gap / v', ha='center', color=INK)
ax.text(0.62, 0.45, 'first electrons\n(next to mesh)', fontsize=10, color=MUTED)
ax.text(2.0, 0.45, 'last electrons\n(from cathode)', fontsize=10, color=MUTED)
ax.set_ylim(0, 1.4); ax.set_xlabel('time after the particle [µs]'); ax.set_yticks([])
ax.set_title('charge arriving at the strips', color=INK, loc='left')

# ---- panel 3: the ladder ------------------------------------------------------------------------------------
ax = fig.add_axes([0.73, 0.16, 0.25, 0.68])
tt = np.linspace(0, 0.5, 50)
ax.plot(tt, 13.5 * tt, color=INK, lw=2, label="column: v = 28.8 mm / T = 13.5")
ax.plot(tt, 12.6 * tt, color=WARN, lw=2, ls="--", label="fitted slope at 25.64° = 12.6")
ax.scatter([0.48], [12.6 * 0.48], color=WARN, zorder=5)
ax.set_xlabel('tan θ  (known mount angle)'); ax.set_ylabel('fitted lateral speed w [µm/ns]')
ax.legend(frameon=False, fontsize=10, loc='upper left')
ax.text(0.30, 1.0, 'slope of this line = the\n"ladder" velocity', fontsize=10, color=MUTED)
ax.set_title('the "ladder": w against tan θ', color=INK, loc='left')
for a in fig.axes[1:]:
    a.spines[['top', 'right']].set_visible(False)
fig.savefig(os.path.join(out, 'f16_concepts.png'), dpi=150)
print(os.path.join(out, 'f16_concepts.png'))
