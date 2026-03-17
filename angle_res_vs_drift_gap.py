#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 01 9:47 AM 2026
Created in PyCharm
Created as nTof_x17/angle_res_vs_drift_gap.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def angle_res(drift_gap, pitch, n_strips):
    """Calculates angular resolution in degrees."""
    return np.rad2deg(np.arctan((pitch * n_strips) / drift_gap))


def add_resolution_point(ax, drift_gap, pitch, n_strips, color='gray'):
    """Draws horizontal and vertical lines to 'pick out' a point on the curve."""
    res = angle_res(drift_gap, pitch, n_strips)

    # Draw vertical line to x-axis
    ax.vlines(x=drift_gap, ymin=0, ymax=res, colors=color, linestyles='--', alpha=0.5)
    # Draw horizontal line to y-axis
    # ax.hlines(y=res, xmin=0, xmax=drift_gap, colors=color, linestyles='--', alpha=0.5)

    # Add a point and a label for clarity
    ax.plot(drift_gap, res, 'o', color=color)
    ax.text(drift_gap + 0.5, res + 0.5, f'{res:.2f}°', color=color, fontsize=9)


def main():
    # Configuration
    pitch_1, pitch_2 = 0.8, 1.6  # mm
    n_strips = 4
    drift_range = np.linspace(1, 40, 1000)

    # Specific points of interest (Drift Gaps to highlight)
    targets = [3, 6, 11, 15, 30]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Curves
    res_1 = angle_res(drift_range, pitch_1, n_strips)
    res_2 = angle_res(drift_range, pitch_2, n_strips)

    line1, = ax.plot(drift_range, res_1, label=f'pitch={pitch_1}mm, n={n_strips}')
    line2, = ax.plot(drift_range, res_2, label=f'pitch={pitch_2}mm, n={n_strips}')

    # Add Horizontal/Vertical pickers for each target drift gap
    for gap in targets:
        add_resolution_point(ax, gap, pitch_1, n_strips, color=line1.get_color())
        add_resolution_point(ax, gap, pitch_2, n_strips, color=line2.get_color())

    # Formatting
    ax.set_xlabel('Drift Gap (mm)')
    ax.set_ylabel('Minimum Angle That Can Be Resolved (deg)')
    ax.set_title('Angular Resolution vs. Drift Gap')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
