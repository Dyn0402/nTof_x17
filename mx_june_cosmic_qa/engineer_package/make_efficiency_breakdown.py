#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_breakdown.py

Rebuild the detector-A (mx17_3) efficiency-breakdown bar for the engineer
package in PLAIN LANGUAGE. The stock analysis figure (09_efficiency_breakdown.py)
labels the bars with the internal category names (reco_far, hit_no_reco,
no_hit), which mean nothing to a construction audience and hide the headline
message: the chamber is essentially NEVER blind, and the loss off the 88.8 %
efficiency is a spark coincidence plus an edge/near-miss position tail, not a
failure to detect.

This reads the numbers straight from the on-disk `efficiency_breakdown.txt`
(so they can never drift from the analysis) and re-renders with plain labels,
a detected-vs-blind grouping, and an explanatory annotation.

Categories (from 09_efficiency_breakdown.py), % of active-area crossing muons:
  reco_near   hit reconstructed within 5 mm of the telescope track  -> EFFICIENCY
  reco_far    valid X+Y point formed, but > 5 mm off the track       -> detected, position missed
  spark       full-detector discharge (> 50 strips) during the crossing
  hit_no_reco strips fired but no valid point could be formed
  no_hit      chamber produced no signal at all                      -> genuine blindness

Output: figures/21-det3A-efficiency-breakdown.{png,pdf}
Usage:  ../.venv/bin/python make_efficiency_breakdown.py
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')

# The headline detector-A run: the 52k-ray weekend run the 88.8 % figure is quoted on.
BREAKDOWN_TXT = os.path.expanduser(
    '~/x17/cosmic_bench/Analysis/mx17_det3_p2_det1_overnight_6-27-26/'
    'long_run_p2_det1_sanity_check/mx17_3/efficiency/efficiency_breakdown.txt')

ACCENT = '#2E598C'


def parse_breakdown(path):
    """Return dict cat->(count, pct) plus n_rays, has_any, reco_all."""
    with open(path) as f:
        txt = f.read()
    cats = {}
    for k in ('reco_near', 'reco_far', 'spark', 'hit_no_reco', 'no_hit'):
        m = re.search(rf'{k}\s*:\s*([\d,]+)\s*\(\s*([\d.]+)%\)', txt)
        if not m:
            raise RuntimeError(f'could not parse {k} from {path}')
        cats[k] = (int(m.group(1).replace(',', '')), float(m.group(2)))
    n = int(re.search(r'active-area clean M3 rays:\s*([\d,]+)', txt)
            .group(1).replace(',', ''))
    has_any = float(re.search(r'has_any=([\d.]+)%', txt).group(1))
    reco_all = float(re.search(r'reco-at-all=([\d.]+)%', txt).group(1))
    return cats, n, has_any, reco_all


def main():
    cats, n, has_any, reco_all = parse_breakdown(BREAKDOWN_TXT)

    # plain-language rows, top (best) -> bottom, matched to analysis categories
    rows = [
        ('reco_near', 'Reconstructed within 5 mm\n(the efficiency)', '#2e8b3d'),
        ('reco_far', 'Detected, but point >5 mm\noff the telescope track', '#f0a028'),
        ('spark', 'Sparked during this muon\n(self-quenching, no dead time)', '#7d3ea0'),
        ('hit_no_reco', 'Fired, no valid point formed', '#9a9a9a'),
        ('no_hit', 'Silent — no signal at all\n(genuine blindness)', '#cc2a2a'),
    ]

    fig, ax = plt.subplots(figsize=(11.6, 4.7))
    ypos = list(range(len(rows)))[::-1]  # first row at top
    for y, (key, label, color) in zip(ypos, rows):
        cnt, pct = cats[key]
        ax.barh(y, pct, color=color, height=0.62,
                edgecolor='white', linewidth=0.6, zorder=3)
        # value label: inside the bar (white) when the bar is long, else just past it
        vlabel = f'{pct:.1f}%   ({cnt:,})'
        if pct > 60:
            ax.text(pct - 1.2, y, vlabel, va='center', ha='right',
                    fontsize=11, fontweight='bold', color='white', zorder=4)
        else:
            ax.text(pct + 1.0, y, vlabel, va='center', ha='left',
                    fontsize=11, fontweight='bold', color='#222')
        ax.text(-1.2, y, label, va='center', ha='right', fontsize=10.3, color='#222')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel('% of muons the telescope sent through the active area', fontsize=10.5)
    ax.set_title('Detector A (mx17_3): where do the crossing muons go?  '
                 f'({n:,} muons, 5 mm match)',
                 fontsize=12.5, fontweight='bold', pad=12)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#dddddd', lw=0.7, zorder=0)
    ax.margins(y=0)

    # headline annotation box: detected vs blind
    blind = cats['no_hit'][1] + cats['hit_no_reco'][1]
    note = (f'The chamber produces a signal for {has_any:.1f}% of muons and '
            f'reconstructs a track point for {reco_all:.1f}%.\n'
            f'Genuine blindness (no signal at all) is only {cats["no_hit"][1]:.1f}%.  '
            f'The two biggest "losses" off the 88.8% are NOT the chamber\n'
            f'failing to see the muon: a {cats["spark"][1]:.1f}% spark coincidence '
            f'(self-quenching, zero dead time after), and a {cats["reco_far"][1]:.1f}% '
            f'edge / near-miss\nposition tail — almost all within 5–10 mm, so at a '
            f'10 mm match the efficiency recovers to ~95%.')
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha='center', va='top',
            fontsize=9.4, color='#333',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f7fb',
                      edgecolor=ACCENT, linewidth=1.1))

    fig.subplots_adjust(left=0.26, right=0.97, top=0.88, bottom=0.30)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'21-det3A-efficiency-breakdown.{ext}'),
                    dpi=170 if ext == 'png' else None)
    print(f'wrote 21-det3A-efficiency-breakdown.png/.pdf  '
          f'(eff {cats["reco_near"][1]:.1f}%, reco_far {cats["reco_far"][1]:.1f}%, '
          f'spark {cats["spark"][1]:.1f}%, blind {blind:.1f}%)')


if __name__ == '__main__':
    main()
