#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures_note.py -- the explanatory figures for the published write-up.

Three things the analysis figures do not show on their own:
  1. what the number actually changed, drawn on the chamber face;
  2. why the charge-balance requirement is the measurement, not a detail;
  3. that two independent methods land on the same two millimetres.

    .venv/bin/python -m ntof_active_area.figures_note
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import mx17_active_area as JUNE
from .clusters import BENCH_ALIAS, CHAMBERS, N_STRIPS, PITCH_MM, STRIP_MAX_MM
from .figures_mm import _clean
from .mm_edges import OUT, FIG, span_profile

# validated categorical palette (dataviz reference instance, light mode)
BEAM, JUNE_C, THIRD = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

SIM_U, SIM_V = 380.0, 340.0          # the old estimate, mm
NEW_U, NEW_V = 399.36, 359.9         # measured


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def fig_area_diagram():
    """The chamber face: what was assumed, what is there, and where it is lost."""
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    c = STRIP_MAX_MM / 2

    # metallised strip region
    ax.add_patch(Rectangle((-PITCH_MM / 2, -PITCH_MM / 2), NEW_U, NEW_U,
                           fill=False, ec=MUTED, lw=1.2, ls=(0, (5, 4))))
    ax.text(NEW_U / 2, 412, '512 strips × 0.78 mm  =  399.4 mm of metal, both axes',
            ha='center', va='bottom', color=MUTED, fontsize=9)

    # passivated bands
    lo, hi = c - NEW_V / 2, c + NEW_V / 2
    for y0, y1 in ((-PITCH_MM / 2, lo), (hi, STRIP_MAX_MM + PITCH_MM / 2)):
        ax.add_patch(Rectangle((-PITCH_MM / 2, y0), NEW_U, y1 - y0,
                               facecolor=MUTED, alpha=0.16, ec='none', hatch='///'))
    ax.text(NEW_U / 2, lo / 2, 'passivated  ~19 mm', ha='center', va='center',
            color=MUTED, fontsize=10, style='italic')
    ax.text(NEW_U / 2, (hi + STRIP_MAX_MM) / 2, 'passivated  ~19 mm',
            ha='center', va='center', color=MUTED, fontsize=10, style='italic')

    # the old estimate
    ax.add_patch(Rectangle((c - SIM_U / 2, c - SIM_V / 2), SIM_U, SIM_V,
                           fill=False, ec=JUNE_C, lw=2.0, ls=(0, (6, 3))))
    ax.annotate('simulation until now\n38 × 34 cm — an estimate',
                xy=(c - SIM_U / 2, c - SIM_V / 2), xytext=(28, 96),
                color=JUNE_C, fontsize=10.5, fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='-', color=JUNE_C, lw=1.2))

    # the measurement
    ax.add_patch(Rectangle((-PITCH_MM / 2, lo), NEW_U, NEW_V,
                           fill=False, ec=BEAM, lw=2.6))
    ax.annotate('measured\n39.9 × 36.0 cm',
                xy=(NEW_U + PITCH_MM / 2, 250), xytext=(412, 250),
                color=BEAM, fontsize=11.5, fontweight='bold', ha='left',
                va='center',
                arrowprops=dict(arrowstyle='-', color=BEAM, lw=1.4))

    ax.annotate('', xy=(-46, 300), xytext=(-46, 100),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    ax.text(-56, 200, 'neutron beam', rotation=90, va='center', ha='center',
            color=INK, fontsize=10)

    ax.set_xlim(-70, 545)
    ax.set_ylim(-30, 436)
    ax.set_aspect('equal')
    ax.set_xlabel('u — tangential  [mm]', color=INK)
    ax.set_ylabel('v — along the beam  [mm]', color=INK)
    ax.set_title('One chamber face, seen from the target\n'
                 'The 4 cm that goes missing goes missing on the beam axis',
                 color=INK, fontsize=12.5, pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'area_diagram.png', dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_why_balance(chamber: str = 'B'):
    """One chamber's y plane, two ways of counting, on a common base.

    Both series are normalised to their OWN interior level (100-300 mm), so they
    share one axis and the comparison is of shape. The shapes disagree: outside
    the chamber the raw count goes UP, because the outer channels are noisy,
    while charge-balanced tracks go to zero.
    """
    data = np.load(OUT / 'profiles.npz')
    sel = _clean(data[f'pairs_{chamber}'])
    s = np.arange(N_STRIPS) * PITCH_MM
    raw = data[f'occ_{chamber}y'].astype(float)
    paired = span_profile(sel, 'v')
    interior = slice(128, 385)                      # 100-300 mm
    raw_n = raw / np.median(raw[interior])
    pair_n = paired / np.median(paired[interior])
    ratio = np.median(raw[int(380 / PITCH_MM) + 1:]) / np.median(raw[interior])

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    m = (s >= 300) & (s <= STRIP_MAX_MM)
    ax.step(s[m], raw_n[m], where='mid', color=JUNE_C, lw=1.9,
            label='every cluster on the plane')
    ax.step(s[m], pair_n[m], where='mid', color=BEAM, lw=2.2,
            label='clusters with a charge-balanced partner on the other plane')
    ax.axhline(1.0, color=GRID, lw=1.4, zorder=1)
    ax.text(301, 1.25, 'the chamber\u2019s own interior level', color=MUTED,
            fontsize=9, va='bottom')
    ax.axvline(379.1, color=INK, lw=1.3, ls=(0, (4, 3)))
    ax.text(377.5, ax.get_ylim()[1] * 0.97, 'active area ends here', rotation=90,
            ha='right', va='top', color=INK, fontsize=9.5)

    ax.annotate(f'outside the chamber the raw count\nis {ratio:.1f}\u00d7 the '
                f'interior \u2014 pure noise',
                xy=(392, raw_n[int(392 / PITCH_MM)]), xytext=(316, 3.4),
                color=JUNE_C, fontsize=10.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=JUNE_C, lw=1.3))
    ax.annotate('real tracks stop dead', xy=(382.5, 0.02), xytext=(322, 1.75),
                color=BEAM, fontsize=10.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=BEAM, lw=1.3,
                                connectionstyle='arc3,rad=-0.15'))

    ax.set_xlabel('v \u2014 along the beam  [mm]', color=INK)
    ax.set_ylabel('counts \u00f7 this series\u2019 own interior level', color=INK)
    ax.set_title(f'Chamber {chamber}, high edge of the beam axis \u2014 why the '
                 f'balance cut IS the measurement', color=INK, fontsize=12.5, pad=10)
    ax.legend(loc='upper left', frameon=False, fontsize=9.5, labelcolor=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'why_balance.png', dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_two_methods():
    """Seven measurements of the same two edges, by two methods that share
    nothing -- not the reference, not the beam, not the definition of "edge"."""
    mm = json.loads((OUT / 'results_mm.json').read_text())
    rows = []
    for ch in CHAMBERS:
        v = mm['chambers'][ch]['planes']['v']
        beam = ((v['live_lo_mm'], v['live_hi_mm'])
                if v['lo_determined'] and v['hi_determined'] else None)
        rows.append((ch, beam, JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]['y']))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, idx, title, ref in ((axes[0], 0, 'low edge', 19.2),
                                (axes[1], 1, 'high edge', 379.1)):
        for i, (ch, beam, june) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([june[idx]], [y], 'o', ms=10, color=JUNE_C, zorder=3,
                    mec='white', mew=1.5)
            if beam is not None:
                ax.plot([beam[idx]], [y], 'o', ms=10, color=BEAM, zorder=4,
                        mec='white', mew=1.5)
                ax.plot([beam[idx], june[idx]], [y, y], color=MUTED, lw=1.2,
                        zorder=2, alpha=0.6)
            else:
                ax.text(0.02, y, 'beam: not measurable', va='center',
                        ha='left', fontsize=8.5, color=MUTED, style='italic',
                        transform=ax.get_yaxis_transform())
        ax.axvline(ref, color=GRID, lw=1.4, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f'{ch}  ({BENCH_ALIAS[ch]})'
                            for ch, _, _ in rows][::-1])
        ax.set_xlim(ref - 2.5, ref + 2.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel('detector-local v  [mm]', color=INK)
        ax.set_title(title, color=INK, fontsize=11)
        _style(ax)
        ax.grid(axis='y', visible=False)
    h = [plt.Line2D([], [], marker='o', ms=9, ls='none', color=BEAM,
                    label='n_TOF beam, no external reference'),
         plt.Line2D([], [], marker='o', ms=9, ls='none', color=JUNE_C,
                    label='June cosmic bench, M3 telescope')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False,
               fontsize=9.5, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Two methods that share nothing, landing within 1–2 mm',
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / 'two_methods.png', dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_area_diagram()
    fig_why_balance()
    fig_two_methods()
    print('note figures ->', FIG)


if __name__ == '__main__':
    main()
