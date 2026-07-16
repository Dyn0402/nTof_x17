"""
11_concept_diagrams.py — Top-down concept sketches of one MX17 arm (back-first
stack: MM -> SiPM wall -> back plastic -> LS-1 -> LS-2), illustrating:

  1. what the current wall-plastic coincidence selects: a track through the MM
     (cluster dots) and one 4-bar wall group, stopping in a plastic bar;
  2. what the LIQ readout adds: the same but traversing the plastic into LS-1
     -> proves passage THROUGH the plastic (true plastic MIP selection).

Schematic only (thicknesses exaggerated for visibility).

Usage: python 11_concept_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).parent / 'figures' / '11_diagrams'
OUT.mkdir(parents=True, exist_ok=True)

# depths along the arm axis [cm] (schematic, spacing exaggerated)
TGT = 6.0
MM_FRONT, MM_BACK = 18.0, 22.0
SW_C, SW_T = 28.0, 1.4            # wall centre, drawn thickness (real 0.3)
BS_C, BS_T = 35.0, 2.2            # back plastic
LS1_C, LS2_C, LS_T = 41.5, 46.5, 2.2

MM_W, SW_W, LS_W = 38.0, 48.0, 45.0
BAR_W = SW_W / 20                 # 20 vertical bars seen end-on
BS_BAR, BS_GAP = 20.0, 0.3

C_MM = '#dbeafe'
C_BAR_READ = '#e5e7eb'
C_BAR_UNREAD = '#9ca3af'
C_GROUP = '#14b8a6'
C_PLASTIC = '#fb923c'
C_LS = '#d8b4fe'
C_LS_HIT = '#7c3aed'
C_TRACK = '#c2410c'

TRACK_SLOPE = -9.0 / (SW_C - TGT)   # aims at wall group 2


def u_at(depth):
    return TRACK_SLOPE * (depth - TGT)


def draw_arm(ax, liq_hit):
    # target + beam
    ax.add_patch(Circle((TGT, 0), 1.0, color='#374151', zorder=5))
    ax.text(TGT, -3.6, 'He-3 target\n(beam $\\perp$ page)', fontsize=9,
            ha='center', va='top', color='#374151')

    # MM volume + clusters
    ax.add_patch(Rectangle((MM_FRONT, -MM_W / 2), MM_BACK - MM_FRONT, MM_W,
                           fc=C_MM, ec='#1d4ed8', lw=1))
    ax.text((MM_FRONT + MM_BACK) / 2, MM_W / 2 + 1.4, 'Micromegas', fontsize=10,
            color='#1d4ed8', ha='center')
    for d in (18.6, 19.6, 20.6, 21.6):
        ax.add_patch(Circle((d, u_at(d)), 0.42, color='#1d4ed8', zorder=6))
    ax.annotate('clusters', (20.1, u_at(20.1) - 0.6), (16.5, -14), fontsize=9,
                color='#1d4ed8',
                arrowprops=dict(arrowstyle='->', color='#1d4ed8'))

    # SiPM wall: 20 bars end-on; 2 unread each edge; group 2 highlighted
    for b in range(20):
        u0 = -SW_W / 2 + b * BAR_W
        unread = b < 2 or b >= 18
        grp = (b - 2) // 4 if not unread else None
        hit = grp == 1
        ax.add_patch(Rectangle((SW_C - SW_T / 2, u0), SW_T, BAR_W * 0.92,
                               fc=C_GROUP if hit else
                               (C_BAR_UNREAD if unread else C_BAR_READ),
                               ec='#4b5563', lw=0.4))
    ax.text(SW_C, SW_W / 2 + 1.4, 'SiPM wall\n20 bars, 16 read\n(top+bottom SiPMs)',
            fontsize=10, color='#0f766e', ha='center')
    ax.annotate('hit group\n(4 bars)', (SW_C, u_at(SW_C) + 2.0), (SW_C - 3.5, -21.5),
                fontsize=9, color='#0f766e', ha='center',
                arrowprops=dict(arrowstyle='->', color='#0f766e'))

    # back plastic: two bars side by side in u (hit bar on the track side)
    for u0, hit in ((-BS_BAR - BS_GAP / 2, True), (BS_GAP / 2, False)):
        ax.add_patch(Rectangle((BS_C - BS_T / 2, u0), BS_T, BS_BAR,
                               fc=C_PLASTIC if hit else '#fde8d7',
                               ec='#c2410c', lw=1))
    ax.text(BS_C - 1.5, BS_BAR + BS_GAP / 2 + 1.4, 'back plastic\n2 bars (L/R)',
            fontsize=10, color='#c2410c', ha='center')

    # liquid scintillators
    for d, hit in ((LS1_C, liq_hit), (LS2_C, False)):
        ax.add_patch(Rectangle((d - LS_T / 2, -LS_W / 2), LS_T, LS_W,
                               fc=C_LS_HIT if hit else C_LS,
                               alpha=0.55 if hit else 0.35, ec='#7c3aed', lw=1))
    ax.text((LS1_C + LS2_C) / 2 + 4.0, LS_W / 2 + 1.4,
            'liquid scint.\nLS-1, LS-2\n(NOT read out)', fontsize=10,
            color='#7c3aed', ha='center')

    # track
    d_end = LS1_C if liq_hit else (BS_C + 0.55)
    ax.add_patch(FancyArrowPatch((TGT, 0), (d_end, u_at(d_end)),
                                 arrowstyle='-', color=C_TRACK, lw=2.2, zorder=4))
    ax.add_patch(Circle((d_end, u_at(d_end)), 0.55, color=C_TRACK, zorder=6))
    stop_txt = 'stops in LS-1' if liq_hit else 'stops in plastic'
    ax.annotate(stop_txt, (d_end, u_at(d_end) - 0.7), (d_end + 1.5, -25),
                fontsize=11, fontweight='bold', color=C_TRACK, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_TRACK))
    ax.text(11.0, u_at(11.0) + 1.5, 'e$^\\pm$ / $\\mu$', fontsize=11,
            color=C_TRACK, rotation=-15)

    ax.set_xlim(1, 53)
    ax.set_ylim(-28.5, 31.5)
    ax.set_aspect('equal')
    ax.axis('off')


fig, ax = plt.subplots(figsize=(9.5, 6.2))
draw_arm(ax, liq_hit=False)
ax.set_title('Current selection: wall $\\times$ plastic coincidence\n'
             '(track needs only to REACH the plastic — MIP selection for the '
             'wall, hit selection for the plastic)', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'concept_wall_plastic.png', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
draw_arm(ax, liq_hit=True)
ax.set_title('With LIQ readout (want today): wall $\\times$ plastic $\\times$ LS-1\n'
             '(track must pass THROUGH the plastic — true MIP selection for the '
             'plastic)', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'concept_wall_plastic_liq.png', dpi=170)
plt.close(fig)
print(OUT)
