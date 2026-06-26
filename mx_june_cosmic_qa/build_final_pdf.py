#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_final_pdf.py

Compile the June cosmic-bench QA into a single PDF with ONE PAGE PER DETECTOR.
Each page shows the headline metrics plus the key plots already produced by the
QA pipeline (08 + 12 + 03 + plot_amplitude_vs_strip), with the sliding-window
within-5mm efficiency map as the centrepiece.

Usage:
    python build_final_pdf.py [key1 key2 ...] [--out=PATH]

With no keys, defaults to the grand-compilation set g_det2 g_det3 g_det4 g_det6
g_det7 (skipping any whose Analysis tree is absent, e.g. det6/det7 not yet
decoded). Reads everything from each key's OUT_BASE under .../Analysis/.
"""
import os
import re
import sys
import json
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from qa_config import get_config, setup_paths
setup_paths()

DEFAULT_KEYS = ['g_det2', 'g_det3', 'g_det4',
                'g_det6', 'g_det6_longer', 'g_det6_long',
                'g_det7', 'g_det7_longer', 'g_det7_long']
OUT_DEFAULT = '/home/dylan/x17/cosmic_bench/Analysis/june_grand_qa.pdf'


def _find(base, *cands):
    """First existing path among OUT_BASE-relative candidates, else None."""
    for c in cands:
        p = os.path.join(base, c)
        if os.path.isfile(p):
            return p
    return None


def parse_breakdown(path):
    """Pull the headline numbers out of efficiency_breakdown.txt."""
    m = {}
    if not path or not os.path.isfile(path):
        return m
    txt = open(path).read()
    for key, pat in [
        ('rays', r'clean M3 rays:\s*([\d]+)'),
        ('reco_near_pct', r'reco_near\s*:\s*\d+\s*\(\s*([\d.]+)%'),
        ('hit_no_reco_pct', r'hit_no_reco\s*:\s*\d+\s*\(\s*([\d.]+)%'),
        ('no_hit_pct', r'no_hit\s*:\s*\d+\s*\(\s*([\d.]+)%'),
        ('has_any_pct', r'has_any=([\d.]+)%'),
        ('within_pct', r'within5mm=([\d.]+)%'),
        ('reco_at_all_pct', r'reco-at-all=([\d.]+)%'),
        ('core_sigma_mm', r'core sigma\([^)]*\)=([\d.]+)\s*mm'),
        ('median_r_mm', r'median \|r\|=([\d.]+)\s*mm'),
    ]:
        mm = re.search(pat, txt)
        if mm:
            m[key] = mm.group(1)
    return m


def place(ax, img_path, label):
    ax.axis('off')
    if img_path and os.path.isfile(img_path):
        ax.imshow(mpimg.imread(img_path), interpolation='antialiased', resample=True)
    else:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='whitesmoke', ec='grey', ls='--'))
        ax.text(0.5, 0.5, f'(missing)\n{label}', ha='center', va='center',
                fontsize=8, color='grey', transform=ax.transAxes)
    ax.set_title(label, fontsize=8, pad=2)


def detector_page(pdf, key):
    cfg = get_config(key)
    base = cfg.OUT_BASE
    if not os.path.isdir(base):
        print(f'  [skip] {key}: no Analysis tree at {base}')
        return False

    bd = parse_breakdown(_find(base, 'efficiency/efficiency_breakdown.txt'))
    align = {}
    ap = _find(base, 'alignment_tpc_veto50/alignment.json', 'alignment_tpc/alignment.json')
    if ap:
        align = json.load(open(ap))
    sld = {}
    sp = _find(base, 'efficiency/efficiency_map_sliding.json')
    if sp:
        sld = json.load(open(sp))

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.05, 1.5, 1.2, 1.2, 1.2],
                  hspace=0.33, wspace=0.06, left=0.04, right=0.96, top=0.96, bottom=0.03)

    detnum = cfg.DET_NAME.split('_')[-1]
    eff = bd.get('within_pct', '—')
    sigma = bd.get('core_sigma_mm', '—')

    # ---- header: title + headline performance stats + small reference box ----
    hax = fig.add_subplot(gs[0, :]); hax.axis('off')
    hax.set_xlim(0, 1); hax.set_ylim(0, 1)

    hax.text(0.0, 0.97, f'Detector {detnum}', fontsize=26, fontweight='bold', va='top')

    # efficiency colour cue
    try:
        ev = float(eff); ecol = '#1a7f37' if ev >= 50 else '#b26a00' if ev >= 25 else '#b3261e'
    except (TypeError, ValueError):
        ecol = 'black'

    def stat_card(x, value, unit, label, color='black'):
        sep = '' if unit == '%' else ' '
        val = f'{value}{sep}{unit}' if unit else f'{value}'
        hax.text(x, 0.44, val, fontsize=22, fontweight='bold', color=color, va='center', ha='left')
        hax.text(x, 0.08, label, fontsize=9, color='dimgrey', va='center', ha='left')

    stat_card(0.00, f'{eff}', '%', 'Efficiency (≤5 mm)', ecol)
    stat_card(0.26, f'{sigma}', 'mm', 'Resolution (core σ)')
    stat_card(0.50, f"{bd.get('has_any_pct','—')}", '%', 'Fired any strip')
    stat_card(0.71, f"{bd.get('reco_at_all_pct','—')}", '%', 'Reconstructed')

    # reference details — small boxed text, top-right (for reference, not headline)
    ref = '\n'.join([
        f"{cfg.RUN}",
        f"subrun: {cfg.SUB_RUN}",
        f"FEU X/Y: {cfg.MX17_FEUS[0]}/{cfg.MX17_FEUS[1]}    z: {cfg.DET_PLANE_Z:.0f} mm",
        f"align: θ={align.get('theta_deg','—')}°  z={align.get('z_x','—')} mm",
        f"clean M3 rays: {bd.get('rays', sld.get('n_rays','—'))}",
        f"median |r|: {bd.get('median_r_mm','—')} mm",
        f"loss: hit-no-reco {bd.get('hit_no_reco_pct','—')}%  silent {bd.get('no_hit_pct','—')}%",
    ])
    fig.text(0.965, 0.992, ref, fontsize=6.4, va='top', ha='right', family='monospace',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f4', ec='#bbbbbb', lw=0.6))

    # ---- headline: sliding-window efficiency map ----
    place(fig.add_subplot(gs[1, :]),
          _find(base, 'efficiency/efficiency_map_sliding.png'),
          'Sliding-window efficiency map  (reco within 5 mm | has_any | rays/kernel)')

    # ---- supporting plots ----
    place(fig.add_subplot(gs[2, 0]),
          _find(base, 'efficiency/map_within_5mm.png'),
          'Binned efficiency map (within 5 mm)')
    place(fig.add_subplot(gs[2, 1]),
          _find(base, 'efficiency/efficiency_breakdown.png'),
          'Efficiency breakdown')

    place(fig.add_subplot(gs[3, 0]),
          _find(base, 'alignment_tpc_veto50/resolution_map_sliding_r50mm.png',
                'alignment_tpc/resolution_map_sliding_r50mm.png'),
          'Sliding-window spatial resolution map')
    place(fig.add_subplot(gs[3, 1]),
          _find(base, 'alignment_tpc_veto50/radial_residuals.png',
                'alignment_tpc_veto50/residuals.png'),
          'Alignment residuals')

    place(fig.add_subplot(gs[4, 0]),
          _find(base, 'raw_detector_qa/amplitude_vs_strip.png'),
          'Pulse height vs strip')
    place(fig.add_subplot(gs[4, 1]),
          _find(base, 'efficiency/scatter_within_5mm.png'),
          'Hit/miss scatter (within 5 mm)')

    fig.text(0.96, 0.005, datetime.date.today().isoformat(), ha='right', fontsize=6, color='grey')
    pdf.savefig(fig, dpi=300); plt.close(fig)
    print(f'  [page] {key}: eff={eff}% sigma={sigma}mm')
    return True


def select_keys(keys):
    """Keep, per physical detector (DET_NAME), only the key with the most analysed
    rays. Lets the caller pass every candidate subrun key; the best one wins the
    page. Keys with no breakdown yet (not analysed) count as 0 rays and are dropped
    if a better sibling exists; a detector with only un-analysed keys is skipped."""
    best = {}  # det_name -> (n_rays, sort_idx, key)
    order = []
    for k in keys:
        try:
            cfg = get_config(k)
        except KeyError:
            print(f'  [skip] unknown key {k}'); continue
        det = cfg.DET_NAME
        if det not in order:
            order.append(det)
        bd = parse_breakdown(_find(cfg.OUT_BASE, 'efficiency/efficiency_breakdown.txt'))
        nrays = int(bd.get('rays', 0)) if bd.get('rays') else 0
        analysed = os.path.isdir(cfg.OUT_BASE) and nrays >= 0 and bool(bd)
        cur = best.get(det)
        cand = (nrays, 1 if analysed else 0, k)
        if cur is None or cand[:2] > cur[:2]:
            best[det] = cand
    # detector number for stable ordering
    def detnum(d):
        try:
            return int(d.split('_')[-1])
        except ValueError:
            return 999
    return [best[d][2] for d in sorted(order, key=detnum)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    keys = select_keys(args if args else DEFAULT_KEYS)
    out = next((a.split('=')[1] for a in sys.argv if a.startswith('--out=')), OUT_DEFAULT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f'PDF keys (best per detector): {keys}')

    n = 0
    with PdfPages(out) as pdf:
        for k in keys:
            try:
                if detector_page(pdf, k):
                    n += 1
            except Exception as e:
                print(f'  [error] {k}: {e}')
        # Cover/contents if nothing rendered would leave an invalid PDF
        if n == 0:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.5, 'No detector pages available yet.', ha='center')
            pdf.savefig(fig); plt.close(fig)
    print(f'Wrote {n} detector page(s) -> {out}')


if __name__ == '__main__':
    main()
