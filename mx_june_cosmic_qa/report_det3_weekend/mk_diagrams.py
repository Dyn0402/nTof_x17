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
ZVIS = 23.5         # mm, recorded drift column (attachment + threshold)
PITCH = 0.78        # mm
V = 34.0            # um/ns (geometry estimator)


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


def _pulse(t, t0, amp, tau=70.0):
    x = np.maximum(t - t0, 0.0) / tau
    return amp * (x ** 2) * np.exp(2.0 - x) / 4.0


def fig_sharing_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2),
                             gridspec_kw={'width_ratios': [1, 1.2, 1.1]})

    # --- (a) charge sharing between strips ---
    ax = axes[0]
    for k, x in enumerate(np.arange(5) * 1.0):
        ax.add_patch(Rectangle((x, 0), 0.72, 0.28, fc='goldenrod', ec='k'))
        ax.text(x + 0.36, -0.18, f'$j{k-2:+d}$' if k != 2 else '$j$',
                ha='center', fontsize=11)
    ax.annotate('', xy=(2.36, 0.4), xytext=(2.36, 2.1),
                arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2.5))
    ax.text(2.5, 1.3, 'avalanche\ncharge', color='royalblue', fontsize=10)
    for dx, c, lab in [(-1, 'crimson', '$c_1\\approx0.5$'),
                       (1, 'crimson', '$c_1$'),
                       (-2, 'darkorange', '$c_2\\approx0.05$–$0.15$'),
                       (2, 'darkorange', '$c_2$')]:
        ax.annotate('', xy=(2.36 + dx, 0.45), xytext=(2.36, 0.75),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6,
                                    connectionstyle=f'arc3,rad={0.35*np.sign(dx)}'))
        if dx in (-1, -2):
            ax.text(2.36 + dx - 0.15, 0.95 + 0.35 * (abs(dx) - 1), lab,
                    color=c, fontsize=9.5, ha='right')
    ax.text(2.36, 2.45, '(a) half the charge appears on the\nneighbouring strips '
            '(+ one-sample delay)', ha='center', fontsize=10.5)
    ax.set_xlim(-0.7, 5.4); ax.set_ylim(-0.6, 2.9); ax.axis('off')

    # --- (b) waveform superposition on a weak deep-end strip ---
    ax = axes[1]
    t = np.linspace(0, 1200, 500)
    direct = _pulse(t, 700, 300)              # weak direct charge, late (deep)
    shared = _pulse(t, 480 + 60, 0.5 * 900)   # copy of earlier neighbour
    ax.plot(t, _pulse(t, 480, 900), ':', color='gray', lw=1.5,
            label='neighbour $j-1$ (earlier, big)')
    ax.plot(t, shared, '--', color='crimson', lw=1.8,
            label='shared onto $j$: $c_1\\times$(neighbour, delayed)')
    ax.plot(t, direct, '--', color='royalblue', lw=1.8,
            label='direct charge of $j$ (deep → weak, late)')
    ax.plot(t, direct + shared, '-', color='k', lw=2.4,
            label='measured waveform of $j$ (sum)')
    tsum = direct + shared
    icfd = int(np.argmax(tsum >= 0.5 * tsum.max()))
    ax.axvline(t[icfd], color='k', lw=1, ls='-.')
    ax.annotate('measured start:\npulled to the\nneighbour’s time',
                (t[icfd] - 30, 430), ha='right', fontsize=9.5)
    ax.axvline(700, color='royalblue', lw=1, ls='-.')
    ax.annotate('true arrival\nof $j$’s charge', (715, 520), fontsize=9.5,
                color='royalblue')
    ax.set_xlabel('time [ns]'); ax.set_ylabel('ADC')
    ax.set_title('(b) why the deep-end strip fires early', fontsize=11)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.25)

    # --- (c) the distorted ladder ---
    ax = axes[2]
    xs = np.arange(10) * PITCH
    t_true = 150 + xs * 1000.0 / (V * 0.4)          # v=34, tan=0.4
    t_meas = t_true.copy()
    t_meas[0] += 130          # mesh-end skirt late
    t_meas[1] += 40
    t_meas[-1] -= 170         # deep-end early (shared-dominated)
    t_meas[-2] -= 90
    t_meas[-3] -= 30
    ax.plot(xs, t_true, 'o--', color='gray', mfc='none', ms=8,
            label='true arrival times (slope $=1/(v_d\\tan\\theta)$)')
    ax.plot(xs, t_meas, 'o', color='crimson', ms=7,
            label='measured times (S-shaped)')
    p = np.polyfit(xs, t_meas, 1)
    ax.plot(xs, np.polyval(p, xs), '-', color='crimson', lw=1.6,
            label='straight-line fit → steeper ns/mm\n→ velocity too LOW')
    ax.annotate('skirt late', (xs[0] + 0.15, t_meas[0] + 10), fontsize=9.5)
    ax.annotate('deep end early\n(shared-dominated)',
                (xs[-4], t_meas[-1] - 10), fontsize=9.5)
    ax.set_xlabel('strip position [mm]'); ax.set_ylabel('strip time [ns]')
    ax.set_title('(c) the time–position ladder distortion', fontsize=11)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.25)

    fig.suptitle('Charge sharing on resistive strips: mechanism of the '
                 'time-fit velocity bias', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, 'diagram_sharing.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    fig_principle()
    fig_offset_mechanism()
    fig_sharing_mechanism()
    print('diagrams written to', HERE)
