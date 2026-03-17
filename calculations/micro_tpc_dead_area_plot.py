#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 17 4:31 PM 2026
Created in PyCharm
Created as nTof_x17/micro_tpc_dead_area_plot.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


def main():
    # ── Configuration ─────────────────────────────────────────────────────────────
    DETECTOR_SIZE_MM = 400  # side length of square detector (mm)
    TARGET_DISTANCE_MM = 200  # distance from target to detector face (mm)
    ANGLES_DEG = [6, 12, 45]  # minimum resolvable angles (degrees)

    # Colors for each angle (in order)
    COLORS = ['#E24B4A', '#EF9F27', '#378ADD', '#7F77DD', '#1D9E75']
    # ─────────────────────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(12, 6.5))
    fig.patch.set_facecolor('#F8F7F2')

    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.07, right=0.97,
                          top=0.95, bottom=0.16)
    ax_side = fig.add_subplot(gs[0])
    ax_front = fig.add_subplot(gs[1], aspect='equal')

    half = DETECTOR_SIZE_MM / 2

    # ── Side view ─────────────────────────────────────────────────────────────────
    ax_side.set_facecolor('#F8F7F2')
    ax_side.set_xlim(-20, TARGET_DISTANCE_MM * 1.18)
    ax_side.set_ylim(-half * 1.15, half * 1.15)

    # Grid
    for y in np.arange(-half, half + 1, 50):
        ax_side.axhline(y, color='#DDDBD0', lw=0.5, zorder=0)
    ax_side.axhline(0, color='#BFBDB0', lw=0.8, ls='--', zorder=1)

    # Side-view label positions: place each at a different depth fraction
    # to avoid stacking when angles are close together
    side_label_depth_frac = [0.9, 0.72, 0.38]  # fraction of TARGET_DISTANCE_MM

    # Angle cones
    for i, angle in enumerate(ANGLES_DEG):
        color = COLORS[i % len(COLORS)]
        r = dead_radius(angle, TARGET_DISTANCE_MM)
        r_clipped = min(r, half)
        verts = np.array([[0, 0], [TARGET_DISTANCE_MM, r_clipped],
                          [TARGET_DISTANCE_MM, -r_clipped]])
        tri = plt.Polygon(verts, closed=True,
                          facecolor=color, alpha=0.18, zorder=2)
        ax_side.add_patch(tri)
        ax_side.plot([0, TARGET_DISTANCE_MM], [0, r_clipped],
                     color=color, lw=1.8, zorder=3)
        ax_side.plot([0, TARGET_DISTANCE_MM], [0, -r_clipped],
                     color=color, lw=1.8, zorder=3)
        # Angle label: stagger depth positions to avoid overlap
        frac = side_label_depth_frac[i] if i < len(side_label_depth_frac) else 0.42
        label_d = TARGET_DISTANCE_MM * frac
        label_r = label_d * np.tan(np.radians(angle))
        ax_side.text(label_d - 5, min(label_r, half * 0.95) + 5,
                     f'{angle}°', color=color, fontsize=10, fontweight='bold',
                     va='bottom', zorder=5)

    # Detector line
    ax_side.plot([TARGET_DISTANCE_MM, TARGET_DISTANCE_MM], [-half, half],
                 color='#444441', lw=3, solid_capstyle='round', zorder=4)
    ax_side.text(TARGET_DISTANCE_MM + 4, 0, 'detector', color='#888780',
                 fontsize=9, va='center', rotation=90, ha='left')

    # Target dot
    ax_side.plot(0, 0, 'o', color='#2C2C2A', ms=7, zorder=6)
    ax_side.text(-8, -22, 'target', color='#888780', fontsize=9)

    # Distance annotation
    ax_side.annotate('', xy=(TARGET_DISTANCE_MM, -half * 1.08),
                     xytext=(0, -half * 1.08),
                     arrowprops=dict(arrowstyle='<->', color='#888780', lw=1))
    ax_side.text(TARGET_DISTANCE_MM / 2, -half * 1.08 - 10,
                 f'{TARGET_DISTANCE_MM} mm', color='#888780',
                 fontsize=9, ha='center', va='top')

    ax_side.set_xlabel('Depth from target (mm)', fontsize=10, color='#444441')
    ax_side.set_ylabel('Transverse position (mm)', fontsize=10, color='#444441')
    ax_side.set_title('Side view — cone geometry', fontsize=11,
                      color='#2C2C2A', pad=8)
    ax_side.tick_params(colors='#888780', labelsize=9)
    for spine in ax_side.spines.values():
        spine.set_edgecolor('#DDDBD0')

    # ── Front view ────────────────────────────────────────────────────────────────
    ax_front.set_facecolor('#F8F7F2')
    ax_front.set_xlim(-half * 1.15, half * 1.15)
    ax_front.set_ylim(-half * 1.15, half * 1.15)

    # Detector square
    det_rect = Rectangle((-half, -half), DETECTOR_SIZE_MM, DETECTOR_SIZE_MM,
                         linewidth=1.2, edgecolor='#378ADD',
                         facecolor='#EBF3FC', alpha=0.35, zorder=1,
                         linestyle='--')
    ax_front.add_patch(det_rect)

    # Grid circles
    for r_mm in [50, 100, 150, 200]:
        circle = Circle((0, 0), r_mm, fill=False,
                        edgecolor='#DDDBD0', lw=0.5, zorder=0)
        ax_front.add_patch(circle)
        ax_front.text(r_mm * 0.707 + 2, r_mm * 0.707 + 2,
                      f'{r_mm}', color='#BFBDB0', fontsize=8)

    # Crosshairs
    ax_front.axhline(0, color='#DDDBD0', lw=0.5, ls='--', zorder=0)
    ax_front.axvline(0, color='#DDDBD0', lw=0.5, ls='--', zorder=0)

    # Label directions (angle in degrees from +x axis) and offset fractions,
    # one per entry in ANGLES_DEG (indexed smallest→largest).
    # Spread across different directions so they don't pile up.
    label_directions_deg = [0, 90, 135]  # where to point the label from center
    label_r_fracs        = [3.0, 1.5, 1.0]  # how far along the radius to anchor

    # Dead zones — largest first so smaller ones render on top
    for i, angle in enumerate(reversed(ANGLES_DEG)):
        orig_i = len(ANGLES_DEG) - 1 - i
        color = COLORS[orig_i % len(COLORS)]
        r = dead_radius(angle, TARGET_DISTANCE_MM)
        circle = Circle((0, 0), r, facecolor=color, alpha=0.25,
                        edgecolor=color, lw=1.8, zorder=2 + i)
        ax_front.add_patch(circle)
        pct = dead_area_fraction(angle, TARGET_DISTANCE_MM, DETECTOR_SIZE_MM)

        dir_deg = label_directions_deg[orig_i % len(label_directions_deg)]
        r_frac  = label_r_fracs[orig_i % len(label_r_fracs)]
        dir_rad = np.radians(dir_deg)
        lx = r * r_frac * np.cos(dir_rad)
        ly = r * r_frac * np.sin(dir_rad)

        # Anchor text so it stays inside the circle regardless of direction
        # ha = 'left' if np.cos(dir_rad) >= 0 else 'right'
        # va = 'bottom' if np.sin(dir_rad) >= 0 else 'top'
        ha = 'center'
        va = 'center'

        ax_front.text(lx, ly,
                      f'r = {r:.1f} mm\n({pct * 100:.1f}% dead)',
                      color=color, fontsize=8.5, fontweight='bold',
                      ha=ha, va=va, zorder=6)

    # Center dot
    ax_front.plot(0, 0, '+', color='#2C2C2A', ms=9, mew=1.5, zorder=7)

    ax_front.set_xlabel('x (mm)', fontsize=10, color='#444441')
    ax_front.set_ylabel('y (mm)', fontsize=10, color='#444441')
    ax_front.set_title('Detector face — dead zones', fontsize=11,
                       color='#2C2C2A', pad=8)
    ax_front.tick_params(colors='#888780', labelsize=9)
    for spine in ax_front.spines.values():
        spine.set_edgecolor('#DDDBD0')

    # ── Legend ────────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], lw=2,
               label=f'{a}°  (r = {dead_radius(a, TARGET_DISTANCE_MM):.1f} mm)')
        for i, a in enumerate(ANGLES_DEG)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(ANGLES_DEG),
               frameon=True, fontsize=9.5, title='Minimum resolvable angle',
               title_fontsize=9, bbox_to_anchor=(0.5, 0.01),
               facecolor='#F8F7F2', edgecolor='#DDDBD0')

    # fig.suptitle(
    #     f'Angular dead zones  ·  {DETECTOR_SIZE_MM}×{DETECTOR_SIZE_MM} mm detector  ·  '
    #     f'target at {TARGET_DISTANCE_MM} mm',
    #     fontsize=12, color='#2C2C2A', y=0.97
    # )

    # plt.savefig('detector_dead_zones.png', dpi=150, bbox_inches='tight',
    #             facecolor=fig.get_facecolor())
    plt.show()
    print('donzo')


def dead_radius(angle_deg, distance_mm):
    """Projected radius of the dead zone on the detector face."""
    return distance_mm * np.tan(np.radians(angle_deg))


def dead_area_fraction(angle_deg, distance_mm, detector_size_mm):
    """Fraction of detector area that is within the dead zone."""
    r = dead_radius(angle_deg, distance_mm)
    half = detector_size_mm / 2
    # Numerical integration to handle circle clipped by square detector
    n = 2000
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    inside_circle = (xx**2 + yy**2) <= r**2
    dead_area = inside_circle.sum() * dx * dy
    total_area = detector_size_mm**2
    return dead_area / total_area


if __name__ == '__main__':
    main()
