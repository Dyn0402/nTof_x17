"""
18_geometry_diagram.py — Schematic of the source -> wall -> plastic imaging test,
in both transverse (u) and beam-axis (v) projections.  Uses the real geometry
from mx17_geom.py (arm C numbers).

Usage: python 18_geometry_diagram.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import mx17_geom as G

BASE = Path(__file__).parent
OUT = BASE / 'figures' / '17_imaging'
OUT.mkdir(parents=True, exist_ok=True)

ST = 'C'
Rw = G.R_wall(ST)
Rp = G.R_plastic(ST)
s = Rw / Rp
GU = G.group_u_centers()                 # [-17.5,-7.5,2.5,12.5]
PU = G.plastic_u_centers(ST)             # [-11.9, 8.4]

C_SRC = '#111827'
C_WALL = '#c2410c'
C_PLAS_1 = '#2563eb'      # -u bar
C_PLAS_2 = '#dc2626'      # +u bar
C_RAY = '#6b7280'
C_ILL1 = '#93c5fd'
C_ILL2 = '#fca5a5'


def ray(ax, x1, y1, x2, y2, **kw):
    ax.plot([x1, x2], [y1, y2], **kw)


# ============================== figure ========================================
fig, (axU, axV) = plt.subplots(1, 2, figsize=(15, 6.4))

# ---------- Panel A: transverse (w vs u) — U-imaging --------------------------
ax = axU
# source
ax.plot(0, 0, 'o', color=C_SRC, ms=11, zorder=6)
ax.annotate('He-3 source\n(origin)', (0, 0), (2.5, -6.5), fontsize=10,
            arrowprops=dict(arrowstyle='->', color=C_SRC))

# illuminated wall regions FIRST (behind), one per plastic bar
for pu, cill in ((PU[0], C_ILL1), (PU[1], C_ILL2)):
    w0, w1 = (pu - 10) * s, (pu + 10) * s
    ax.add_patch(Rectangle((Rw - 0.9, w0), 1.8, w1 - w0, facecolor=cill, alpha=0.9,
                           edgecolor='none', zorder=3))
# wall plane at Rw with 4 groups (each 10 cm), instrumented span u in [-20,20]
for g in range(4):
    lo, hi = GU[g] - 5, GU[g] + 5
    ax.add_patch(Rectangle((Rw - 0.9, lo), 1.8, hi - lo, facecolor='none',
                           edgecolor=C_WALL, lw=1.6, zorder=5))
    ax.text(Rw - 2.4, GU[g], f'g{g+1}', color=C_WALL, fontsize=9, va='center', ha='right')
ax.text(28, 27, 'SiPM wall\n(4 groups)', color=C_WALL, ha='center', fontsize=10)

# plastic plane at Rp, two bars (each 20 cm), gap between
for pu, col in ((PU[0], C_PLAS_1), (PU[1], C_PLAS_2)):
    ax.add_patch(Rectangle((Rp - 0.7, pu - 10), 1.4, 20, facecolor=col, alpha=0.30,
                           edgecolor=col, lw=1.8, zorder=4))
ax.text(Rp + 3.5, 14, 'plastic\nbar 2 (+u)', color=C_PLAS_2, ha='center', fontsize=9)
ax.text(Rp + 3.5, -18, 'plastic\nbar 1 (−u)', color=C_PLAS_1, ha='center', fontsize=9)

# limiting rays: source through each plastic bar's two edges -> wall, coloured by bar
for pu, cray in ((PU[0], C_PLAS_1), (PU[1], C_PLAS_2)):
    for edge in (pu - 10, pu + 10):
        ray(ax, 0, 0, Rp, edge, color=cray, lw=1.0, ls='-', alpha=0.7, zorder=2)
# crossover line
u_cross = PU.mean() * s
ax.plot([Rw - 0.9, Rw + 0.9], [u_cross, u_cross], color='#111827', lw=2.5, zorder=6)
ax.annotate(f'bar-1↔bar-2 boundary\nprojects to wall u={u_cross:.1f} cm\n'
            '(data crossover −1 cm ✓)', (Rw, u_cross), (5, 16), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#111827'))

ax.set_xlim(-3, Rp + 8)
ax.set_ylim(-27, 30)
ax.set_xlabel('w = radial distance from beam axis [cm]')
ax.set_ylabel('u = transverse (bar-across) [cm]')
ax.set_title('U-imaging: each plastic bar projects onto the wall\n'
             '(beam +Y = out of page)')
ax.set_aspect('equal')
ax.grid(alpha=0.15)

# ---------- Panel B: side view (w vs v) — V-imaging + incidence angle ----------
ax = axV
ax.plot(0, 0, 'o', color=C_SRC, ms=11, zorder=6)
ax.text(1.5, 2.2, 'source', fontsize=10, color=C_SRC)

# wall bar (50 cm along v) with top/bottom SiPMs at the ends
ax.add_patch(Rectangle((Rw - 0.6, -25), 1.2, 50, facecolor='none', edgecolor=C_WALL,
                       lw=1.8, zorder=5))
for vend, tag in ((25, 'top SiPM'), (-25, 'bottom SiPM')):
    ax.plot(Rw, vend, 's', color=C_WALL, ms=8, zorder=6)
    ax.text(Rw + 1.5, vend, tag, color=C_WALL, fontsize=9, va='center')
ax.text(Rw - 3.5, 0, 'wall bar\n50 cm (v)', color=C_WALL, ha='center', fontsize=9,
        rotation=90, va='center')

# plastic (30 cm along v)
ax.add_patch(Rectangle((Rp - 0.7, -15), 1.4, 30, facecolor='#9ca3af', alpha=0.35,
                       edgecolor='#374151', lw=1.6, zorder=4))
ax.text(Rp + 2.5, 0, 'plastic\n30 cm (v)', color='#374151', ha='center', fontsize=9,
        rotation=90, va='center')

# limiting rays through plastic v-edges -> projected wall region
vproj = 15 * s
for ve in (-15, 15):
    ray(ax, 0, 0, Rp, ve, color=C_RAY, lw=1.0, zorder=2)
ax.add_patch(Rectangle((Rw - 0.9, -vproj), 1.8, 2 * vproj, facecolor='#fde68a',
                       alpha=0.85, edgecolor='none', zorder=3))
ax.annotate(f'plastic height (30 cm) projects\nto wall |v| < {vproj:.1f} cm\n'
            '(⊕ ~10 cm resolution)', (Rw - 0.9, -vproj + 3), (-2, -25), fontsize=9,
            color='#a16207', ha='left',
            arrowprops=dict(arrowstyle='->', color='#a16207'))

# incidence-angle track to an off-centre wall point, with normal
vhit = 12.5
ax.add_patch(FancyArrowPatch((0, 0), (Rw, vhit), arrowstyle='-|>', mutation_scale=13,
                             color='#16a34a', lw=2.0, zorder=7))
ax.plot([Rw, Rw + 7], [vhit, vhit], color='#16a34a', ls='--', lw=1)   # wall normal (w)
th = np.degrees(np.arctan2(vhit, Rw))
ax.annotate(f'source track, θ={th:.0f}°\npath = 3 mm/cosθ\n'
            f'MIP deposit ∝ 1/cosθ = {1/np.cos(np.radians(th)):.2f}',
            (Rw, vhit), (11, 20), fontsize=9, color='#166534',
            arrowprops=dict(arrowstyle='->', color='#166534'))

ax.set_xlim(-3, Rp + 9)
ax.set_ylim(-27, 30)
ax.set_xlabel('w = radial distance from beam axis [cm]')
ax.set_ylabel('v = beam-axis position (+Y) [cm]')
ax.set_title('V-imaging + incidence angle: plastic height limits wall |v|,\n'
             'off-axis tracks arrive steeper → larger MIP deposit')
ax.set_aspect('equal')
ax.grid(alpha=0.15)

fig.suptitle(f'MX17 arm {ST}: source → wall → plastic geometry  '
             f'(R_wall={Rw:.0f} cm, R_plastic={Rp:.0f} cm, projection ×{s:.2f})',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / 'geometry_diagram.png', dpi=150)
print(f'-> {OUT / "geometry_diagram.png"}')
