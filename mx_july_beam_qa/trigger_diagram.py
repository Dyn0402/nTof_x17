"""trigger_diagram.py — in-plot detector schematic for the trigger-rate figures.

draw(ax, mode, wall=None):
  mode 'coinc'   — SiPM (thin) + plastic (thick) both hit; arrow through both.
  mode 'sipm'    — only the SiPM wall coloured (hit); no arrow.
  mode 'plastic' — only the plastic coloured (hit); no arrow.
  mode 'sum'     — top-down view of all 4 walls around the beam, summed
                   (A + B + C + D); each arm is its own SiPM.AND.plastic, made
                   explicit that this is a SUM of singles, not a wall-wall coincidence.
"""
import numpy as np
from matplotlib.patches import Rectangle, Circle

HIT = '#ff9900'
OFF = 'white'
EDGE = '#333'
ARROW = '#333'

# side-view box geometry (inset coords)
_SX, _SW = 0.30, 0.08          # SiPM x, width (thin)
_PX, _PW = 0.52, 0.21          # plastic x, width (thick)
_Y0, _H = 0.17, 0.52


def _side_inset(ax):
    ins = ax.inset_axes([0.605, 0.33, 0.33, 0.45])
    ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.axis('off')
    return ins


def draw(ax, mode, wall=None):
    if mode == 'sum':
        return _draw_sum(ax)
    ins = _side_inset(ax)
    ins.text(0.50, 0.92, wall, ha='center', va='center', fontsize=27,
             fontweight='bold', color='#222')
    sipm_on = mode in ('coinc', 'sipm')
    plas_on = mode in ('coinc', 'plastic')
    ins.add_patch(Rectangle((_SX, _Y0), _SW, _H, facecolor=HIT if sipm_on else OFF,
                            edgecolor=EDGE, lw=1.6))
    ins.add_patch(Rectangle((_PX, _Y0), _PW, _H, facecolor=HIT if plas_on else OFF,
                            edgecolor=EDGE, lw=1.6))
    ins.text(_SX + _SW / 2, 0.06, 'SiPM', ha='center', fontsize=8.5)
    ins.text(_PX + _PW / 2, 0.06, 'plastic', ha='center', fontsize=8.5)
    if mode == 'coinc':                     # particle passes through both
        ins.annotate('', xy=(0.86, 0.43), xytext=(0.10, 0.43),
                     arrowprops=dict(arrowstyle='-|>', lw=2.0, color=ARROW))
    return ins


def _arm(ins, cx, cy, dx, dy, letter):
    """One top-down arm: thin SiPM bar (inner) + thick plastic bar (outer),
    perpendicular to the radial direction (dx,dy), both hit; letter at the tip."""
    half = 0.085                            # bar half-length (transverse)
    ts, tp = 0.028, 0.052                   # SiPM / plastic bar thickness (radial)
    r_s, r_p = 0.115, 0.175                 # inner-edge radii
    for r, th in ((r_s, ts), (r_p, tp)):
        if dx:                              # left/right arm -> vertical bars
            x = cx + dx * r - (0 if dx > 0 else th)
            ins.add_patch(Rectangle((x, cy - half), th, 2 * half,
                                    facecolor=HIT, edgecolor=EDGE, lw=1.3))
        else:                               # up/down arm -> horizontal bars
            y = cy + dy * r - (0 if dy > 0 else th)
            ins.add_patch(Rectangle((cx - half, y), 2 * half, th,
                                    facecolor=HIT, edgecolor=EDGE, lw=1.3))
    ins.text(cx + dx * 0.335, cy + dy * 0.335, letter, ha='center', va='center',
             fontsize=13, fontweight='bold', color='#222', zorder=8)


# distinct colours for the 4 tracks = 4 independent events
_TRACK = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
_TILT = [13, -15, 10, -12]      # deg off radial, so the tracks are visibly different


def _draw_sum(ax):
    ins = ax.inset_axes([0.55, 0.22, 0.44, 0.66])
    ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.axis('off')
    cx, cy = 0.5, 0.60
    arms = [(0, 1, 'A'), (1, 0, 'B'), (0, -1, 'C'), (-1, 0, 'D')]
    for dx, dy, L in arms:
        _arm(ins, cx, cy, dx, dy, L)
    # 4 tracks (4 different events) from the beam axis through each wall, tilted
    for (dx, dy, _), col, tl in zip(arms, _TRACK, _TILT):
        ang = np.arctan2(dy, dx) + np.radians(tl)
        x0, y0 = cx + 0.05 * np.cos(ang), cy + 0.05 * np.sin(ang)
        x1, y1 = cx + 0.295 * np.cos(ang), cy + 0.295 * np.sin(ang)
        ins.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='-|>', color=col, lw=1.9, alpha=0.95),
                     zorder=6)
    # beam OUT of page (dotted circle) at centre, drawn over the track tails
    ins.add_patch(Circle((cx, cy), 0.033, facecolor='white', edgecolor=EDGE, lw=1.3, zorder=7))
    ins.add_patch(Circle((cx, cy), 0.009, facecolor=EDGE, edgecolor=EDGE, zorder=8))
    # '+' in the diagonal gaps -> we ADD the arms
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ins.text(cx + sx * 0.235, cy + sy * 0.235, '+', ha='center', va='center',
                 fontsize=15, color='#999', fontweight='bold')
    # explicit caption: sum of independent singles, NOT a wall-wall coincidence
    ins.text(0.5, 0.115, r'total $= A + B + C + D$', ha='center', fontsize=10)
    ins.text(0.5, 0.035, 'each arm = SiPM $\\wedge$ plastic  (4 independent events,\n'
             'summed — NOT a wall–wall coincidence)', ha='center', va='center',
             fontsize=7.3, color='#555')
    return ins
