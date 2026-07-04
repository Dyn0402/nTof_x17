#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schematic diagrams for the det3 weekend report (no TikZ on this machine).

Geometry (7-03 revision): mechanical drift gap 30 mm; only the ~19.4 mm of
drift column closest to the mesh produces recorded signal — electrons from
higher up are lost to attachment in the contaminated (humid) gas.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(HERE, exist_ok=True)

GAP = 30.0          # mm, mechanical drift gap (cathode to mesh)
ZVIS = 19.4         # mm, visible drift column (signal above threshold)
PITCH = 0.78        # mm
V = 28.1            # um/ns


def draw_detector(ax, x0=0.0, x1=26.0, show_zvis=True):
    """Cathode on top, mesh + strips at the bottom."""
    ax.add_patch(Rectangle((x0, GAP), x1 - x0, 1.4, fc='dimgray', ec='k'))
    ax.text(x1 - 0.3, GAP + 2.0, 'drift cathode  (−HV$_\\mathrm{drift}$)',
            ha='right', fontsize=10)
    if show_zvis:
        ax.axhspan(ZVIS, GAP, xmin=0.02, xmax=0.98, color='crimson', alpha=0.07)
        ax.axhline(ZVIS, color='crimson', lw=1.0, ls='--', alpha=0.7)
        ax.text(x0 + 0.4, ZVIS + 0.7, f'$z_\\mathrm{{vis}} \\approx {ZVIS:g}$ mm '
                '(threshold: signal from above is lost to attachment)',
                fontsize=8.5, color='crimson')
    # mesh
    ax.plot([x0, x1], [0, 0], color='k', lw=1.2)
    for xm in np.arange(x0, x1, 0.55):
        ax.plot(xm, 0, marker='o', ms=1.6, color='k')
    ax.text(x1 - 0.3, 0.55, 'micromesh (gnd)', ha='right', fontsize=10)
    # amplification + strips
    for xs in np.arange(x0 + 0.2, x1, PITCH):
        ax.add_patch(Rectangle((xs, -2.0), PITCH * 0.72, 0.95, fc='goldenrod', ec='k', lw=0.4))
    ax.text(x1 - 0.3, -3.6, 'resistive strips (+HV$_\\mathrm{resist}$) / readout strips',
            ha='right', fontsize=10)
    ax.axhspan(-1.05, 0, color='orange', alpha=0.18)


def fig_principle():
    fig, (ax, axt) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={'width_ratios': [1.65, 1]})
    draw_detector(ax)
    # inclined muon track
    th = np.radians(30)
    zt = np.array([GAP + 3.2, -3.2])
    xc = 10.0
    xt = xc + (zt - GAP / 2) * np.tan(th)
    ax.plot(xt, zt, color='crimson', lw=2.2)
    ax.annotate('$\\mu$', (xt[0] + 0.4, zt[0]), color='crimson', fontsize=14)
    # ionisation electrons + drift arrows; above ZVIS the charge is attached
    zs = np.linspace(1.5, GAP - 1.2, 10)
    for z in zs:
        x = xc + (z - GAP / 2) * np.tan(th)
        alive = z < ZVIS
        ax.plot(x, z, 'o', color='royalblue', ms=5,
                mfc='royalblue' if alive else 'none', alpha=1.0 if alive else 0.6)
        if alive:
            ax.add_patch(FancyArrowPatch((x, z), (x, 0.4), arrowstyle='-|>',
                                         mutation_scale=11, color='royalblue',
                                         lw=1.1, alpha=0.75))
        else:
            ax.add_patch(FancyArrowPatch((x, z), (x, z - 3.2), arrowstyle='-|>',
                                         mutation_scale=9, color='royalblue',
                                         lw=0.9, ls=':', alpha=0.5))
            ax.plot(x, z - 4.2, marker='x', ms=6, color='royalblue', alpha=0.6)
    ax.annotate('electrons drift at $v_d$\n(time to mesh $= z/v_d$)',
                (xc + 8.6, 6.5), fontsize=10, color='royalblue')
    ax.annotate('deep charge captured\n(O$_2$/H$_2$O attachment)',
                (xc + 8.2, ZVIS + 4.5), fontsize=9, color='royalblue', alpha=0.8)
    ax.annotate('track angle $\\theta$', (xc - 8.2, GAP * 0.38), fontsize=11, color='crimson')
    ax.plot([xc, xc], [0, GAP], color='gray', lw=0.8, ls=':')
    ax.set_xlim(0, 26); ax.set_ylim(-4.6, GAP + 4.6)
    ax.set_aspect('auto'); ax.axis('off')
    ax.set_title('micro-TPC principle: 30 mm drift gap, strip readout')

    # right: time vs strip position (only the visible column is recorded)
    zs_vis = zs[zs < ZVIS]
    zs_lost = zs[zs >= ZVIS]
    x_vis = xc + (zs_vis - GAP / 2) * np.tan(th)
    axt.plot(x_vis, zs_vis * 1000.0 / V, 'o', color='royalblue', ms=6,
             label='recorded strips')
    axt.plot(xc + (zs_lost - GAP / 2) * np.tan(th), zs_lost * 1000.0 / V, 'x',
             color='royalblue', ms=6, alpha=0.5, label='below threshold')
    p = np.polyfit(x_vis, zs_vis * 1000.0 / V, 1)
    xx = np.linspace(x_vis.min() - 1, x_vis.max() + 1, 10)
    axt.plot(xx, np.polyval(p, xx), '-', color='crimson', lw=1.6)
    axt.set_xlabel('strip position $x$ [mm]')
    axt.set_ylabel('strip signal time $t$ [ns]')
    axt.set_title('per-strip fit:  $t = x/(v_d\\tan\\theta) + t_0$\n'
                  'slope$^{-1}$ = $\\mathrm{d}x/\\mathrm{d}t = v_d\\tan\\theta$')
    axt.legend(fontsize=9, loc='upper left')
    axt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'diagram_principle.png'), dpi=180)
    plt.close(fig)


def fig_offset_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, th_deg, title in [(axes[0], 0, 'vertical track ($\\theta_{ref}=0$)'),
                              (axes[1], 25, 'inclined track ($\\theta_{ref}=25^\\circ$)')]:
        draw_detector(ax, 0, 26)
        th = np.radians(th_deg)
        xc = 11.0
        zt = np.array([GAP + 3.2, -3.2])
        xt = xc + (zt - GAP / 2) * np.tan(th)
        ax.plot(xt, zt, color='crimson', lw=2.2)
        zs = np.linspace(0.8, GAP - 0.8, 12)
        for z in zs:
            x = xc + (z - GAP / 2) * np.tan(th)
            alive = z < ZVIS
            ax.plot(x, z, 'o', color='royalblue', ms=4,
                    mfc='royalblue' if alive else 'none', alpha=1.0 if alive else 0.5)
            if alive:
                ax.add_patch(FancyArrowPatch((x, z), (x, 0.4), arrowstyle='-|>',
                                             mutation_scale=9, color='royalblue',
                                             lw=0.9, alpha=0.6))
        # charge footprint on the strips: only the VISIBLE column contributes
        geo = ZVIS * np.tan(th)
        x_mid = xc + (ZVIS / 2 - GAP / 2) * np.tan(th)
        x_lo = x_mid - geo / 2 - 1.1   # w/2 ≈ 1.1 mm each side
        x_hi = x_mid + geo / 2 + 1.1
        ax.add_patch(Rectangle((x_lo, -2.55), x_hi - x_lo, 0.55,
                               fc='mediumseagreen', ec='k', lw=0.6, alpha=0.85))
        ax.annotate('cluster width $= z_{vis}\\tan\\theta$ (geometric) $+\\ w$ (spreading)'
                    if th_deg else
                    'cluster width $= w \\approx 2$ mm (spreading only!)',
                    (xc, -4.5), ha='center', fontsize=10.5, color='darkgreen')
        # time span marker over the visible column only
        ax.add_patch(FancyArrowPatch((24.2, 0.3), (24.2, ZVIS - 0.2), arrowstyle='<|-|>',
                                     mutation_scale=12, color='purple', lw=1.4))
        ax.text(24.9, ZVIS / 2, 'time span\n$= z_{vis}/v_d$\n(ALWAYS,\nany $\\theta$)',
                fontsize=9.5, color='purple', va='center')
        ax.set_xlim(0, 29); ax.set_ylim(-5.6, GAP + 4.4)
        ax.axis('off'); ax.set_title(title, fontsize=12)
    fig.suptitle('why $|\\theta_{det}| > |\\theta_{ref}|$:  fitted slope '
                 '$\\mathrm{d}x/\\mathrm{d}t \\approx$ (cluster width)/(time span) '
                 '$= v_d(\\tan\\theta \\pm w/z_{vis})$', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, 'diagram_offset.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    fig_principle()
    fig_offset_mechanism()
    print('diagrams written to', HERE)
