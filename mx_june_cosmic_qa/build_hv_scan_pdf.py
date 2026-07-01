#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hv_scan_pdf.py

Compile the resist-HV scans into one styled PDF (matching june_grand_qa.pdf):
  - a summary page overlaying efficiency-vs-HV and resolution-vs-HV for every
    scan, and
  - one page per detector with headline stat cards + efficiency-vs-HV and
    resolution-vs-HV, overlaying ALL scans of that detector (a detector may have
    been scanned in more than one run / HV range).

Reads each scan's `efficiency_vs_hv.csv` (written by 10_hv_scan_efficiency.py)
from Analysis/<RUN>/hv_scan/<DET_NAME>/ and REPLOTS from the CSV (vector, no blur).

Usage:
  python build_hv_scan_pdf.py [key1 key2 ...] [--out=PATH]

Default keys = the June scans grouped by detector:
  det2,det3  -> 6-22 (resist stepped together)
  det6,det7  -> 6-26 hv_scan (400-500 V, both same V) + 6-26 overnight (higher V)
The 6-23 scan is excluded (degraded M3 reference).
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from qa_config import get_config, setup_paths
setup_paths()

DEFAULT_KEYS = ['g_det2', 'g_det3',
                'g_det6_hv', 'g_det6_long',
                'g_det7_hv', 'g_det7_long']
OUT_DEFAULT = '/home/dylan/x17/cosmic_bench/Analysis/june_hv_scans.pdf'
DET_COLOR = {'2': '#1f77b4', '3': '#2ca02c', '4': '#9467bd', '6': '#d62728', '7': '#ff7f0e'}
SCAN_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']  # per-scan, on a detector page


def run_tag(run):
    if 'hv_scan' in run:
        return 'HV-scan run'
    if 'overnight' in run:
        return 'overnight run'
    return run.split('_')[-1]


def hv_dir(cfg):
    root = os.path.dirname(cfg.BASE_PATH.rstrip('/'))   # .../cosmic_bench
    return os.path.join(root, 'Analysis', cfg.RUN, 'hv_scan', cfg.DET_NAME)


def load_scan(key):
    cfg = get_config(key)
    csv = os.path.join(hv_dir(cfg), 'efficiency_vs_hv.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv).sort_values('hv').reset_index(drop=True)
    if df.empty:
        return None
    drift = ''
    m = re.search(r'drift_(\d+)V', str(df['subrun'].iloc[0]))
    if m:
        drift = f'{m.group(1)} V'
    return dict(key=key, cfg=cfg, df=df, detnum=cfg.DET_NAME.split('_')[-1],
                drift=drift, tag=run_tag(cfg.RUN),
                hv_lo=float(df['hv'].min()), hv_hi=float(df['hv'].max()))


def summary_page(pdf, scans):
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                  hspace=0.22, left=0.10, right=0.95, top=0.87, bottom=0.07)
    fig.suptitle('June resist-HV scans — efficiency & resolution',
                 fontsize=17, fontweight='bold', x=0.10, ha='left', y=0.955)
    fig.text(0.10, 0.925,
             'Efficiency = reco within 5 mm of the M3 track, in a fixed per-detector active box.\n'
             '6-22: det2+det3 resist stepped together (drift 1000 V).   '
             '6-26: det6/det7 stepped together (drift 700 V).',
             fontsize=8, color='#444444', va='top')

    markers = {}
    ax = fig.add_subplot(gs[0])
    for s in scans:
        c = DET_COLOR.get(s['detnum'], 'black')
        mk = 'o' if markers.get(s['detnum']) is None else 's'
        markers[s['detnum']] = mk
        d = s['df']
        ax.errorbar(d['hv'], d['eff_reco'], yerr=d['eff_reco_err'], fmt=f'{mk}-',
                    color=c, capsize=3, lw=1.8, ms=6,
                    label=f"Detector {s['detnum']} · {s['tag']} (drift {s['drift']})")
    ax.set_xlabel('Resist HV [V]'); ax.set_ylabel('Efficiency (reco ≤5 mm)')
    ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('Efficiency vs resist HV', fontsize=11)

    markers = {}
    ax2 = fig.add_subplot(gs[1])
    for s in scans:
        c = DET_COLOR.get(s['detnum'], 'black')
        mk = 'o' if markers.get(s['detnum']) is None else 's'
        markers[s['detnum']] = mk
        d = s['df']
        if 'sigma_x_mm' in d:
            ax2.plot(d['hv'], d['sigma_x_mm'], f'{mk}-', color=c, lw=1.7, ms=5,
                     label=f"Det {s['detnum']} · {s['tag']} σ_x")
    ax2.set_xlabel('Resist HV [V]'); ax2.set_ylabel('Core resolution σ_x [mm]')
    ax2.set_ylim(0, None); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=7, ncol=2)
    ax2.set_title('Spatial resolution vs resist HV', fontsize=11)

    pdf.savefig(fig, dpi=200); plt.close(fig)


def detector_page(pdf, detnum, det_scans):
    # primary scan = the one whose peak efficiency is highest (for the stat cards)
    def peak(s):
        v = s['df']['eff_reco']
        return float(v.max()) if v.notna().any() else -1
    prim = max(det_scans, key=peak)
    dp = prim['df']
    ipk = dp['eff_reco'].idxmax() if dp['eff_reco'].notna().any() else None
    peak_eff = f"{dp.loc[ipk, 'eff_reco']*100:.1f}" if ipk is not None else '—'
    peak_hv = f"{dp.loc[ipk, 'hv']:.0f}" if ipk is not None else '—'
    sig_pk = (f"{dp.loc[ipk, 'sigma_x_mm']:.2f}"
              if ipk is not None and 'sigma_x_mm' in dp and np.isfinite(dp.loc[ipk, 'sigma_x_mm'])
              else '—')
    dcol = DET_COLOR.get(detnum, 'black')

    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.7, 1.25, 1.25],
                  hspace=0.30, left=0.11, right=0.93, top=0.95, bottom=0.06)

    hax = fig.add_subplot(gs[0]); hax.axis('off'); hax.set_xlim(0, 1); hax.set_ylim(0, 1)
    hax.text(0.0, 0.95, f'Detector {detnum}', fontsize=24, fontweight='bold', va='top')
    hax.text(0.0, 0.62, 'resist HV scan', fontsize=12, color='dimgrey', va='top')

    def card(x, value, unit, label):
        sep = '' if unit == '%' else ' '
        hax.text(x, 0.34, f'{value}{sep}{unit}' if unit else f'{value}',
                 fontsize=20, fontweight='bold', va='center', color=dcol)
        hax.text(x, 0.02, label, fontsize=8.5, color='dimgrey', va='center')
    card(0.00, peak_eff, '%', 'Peak efficiency')
    card(0.27, peak_hv, 'V', 'at resist HV')
    card(0.50, sig_pk, 'mm', 'σ at peak')
    full_lo = min(s['hv_lo'] for s in det_scans)
    full_hi = max(s['hv_hi'] for s in det_scans)
    card(0.73, f'{int(full_lo)}–{int(full_hi)}', 'V', 'HV range')

    cfg = prim['cfg']
    ref = '\n'.join(
        [f"FEU X/Y: {cfg.MX17_FEUS[0]}/{cfg.MX17_FEUS[1]}   z: {cfg.DET_PLANE_Z:.0f} mm"]
        + [f"{s['tag']}: {int(s['hv_lo'])}–{int(s['hv_hi'])} V (drift {s['drift']})" for s in det_scans])
    fig.text(0.93, 0.985, ref, fontsize=6.6, va='top', ha='right', family='monospace',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f4', ec='#bbbbbb', lw=0.6))

    # efficiency vs HV (overlay all scans of this detector); spark rate on twin axis
    axe = fig.add_subplot(gs[1])
    axe_s = axe.twinx()
    have_spark = False
    for i, s in enumerate(det_scans):
        c = SCAN_COLORS[i % len(SCAN_COLORS)]
        d = s['df']
        axe.errorbar(d['hv'], d['eff_reco'], yerr=d['eff_reco_err'], fmt='o-',
                     color=c, capsize=4, lw=2, ms=7,
                     label=f"{s['tag']} (reco ≤5 mm)")
        axe.plot(d['hv'], d['eff_anyhit'], 's--', color=c, ms=5, alpha=0.5,
                 label=f"{s['tag']} (any hit)")
        if 'spark_frac' in d and d['spark_frac'].notna().any():
            axe_s.plot(d['hv'], d['spark_frac'] * 100, 'x:', color=c, ms=7, lw=1.3,
                       label=f"{s['tag']} (spark %)")
            have_spark = True
    axe.set_xlabel('Resist HV [V]'); axe.set_ylabel('Efficiency (fixed active box)')
    axe.set_ylim(0, 1.02); axe.grid(True, alpha=0.3)
    axe.set_title('Efficiency (left) + spark rate (right) vs resist HV', fontsize=11)
    if have_spark:
        axe_s.set_ylabel('Spark fraction [%]', color='crimson')
        axe_s.tick_params(axis='y', labelcolor='crimson')
        axe_s.set_ylim(0, None)
        h1, l1 = axe.get_legend_handles_labels(); h2, l2 = axe_s.get_legend_handles_labels()
        axe.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    else:
        axe_s.set_yticks([]); axe.legend(fontsize=8)

    # resolution vs HV
    axr = fig.add_subplot(gs[2])
    for i, s in enumerate(det_scans):
        c = SCAN_COLORS[i % len(SCAN_COLORS)]
        d = s['df']
        if 'sigma_x_mm' in d:
            axr.plot(d['hv'], d['sigma_x_mm'], 'o-', color=c, lw=2, ms=6, label=f"{s['tag']} σ_x")
            axr.plot(d['hv'], d['sigma_y_mm'], 's--', color=c, lw=1.6, ms=5, alpha=0.6,
                     label=f"{s['tag']} σ_y")
    axr.set_xlabel('Resist HV [V]'); axr.set_ylabel('Core spatial resolution σ [mm]')
    axr.set_ylim(0, None); axr.grid(True, alpha=0.3); axr.legend(fontsize=8)
    axr.set_title('Spatial resolution vs resist HV', fontsize=11)

    pdf.savefig(fig, dpi=200); plt.close(fig)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    keys = args if args else DEFAULT_KEYS
    out = next((a.split('=')[1] for a in sys.argv if a.startswith('--out=')), OUT_DEFAULT)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    scans = []
    for k in keys:
        s = load_scan(k)
        if s is None:
            print(f'  [skip] {k}: no efficiency_vs_hv.csv')
        else:
            scans.append(s)
            print(f'  [ok]   {k}: detector {s["detnum"]}, {len(s["df"])} pts, '
                  f'{int(s["hv_lo"])}-{int(s["hv_hi"])} V')
    if not scans:
        print('No HV scans found.'); return

    # group by detector number, ordered numerically; scans within a det by HV range
    groups = {}
    for s in scans:
        groups.setdefault(s['detnum'], []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: s['hv_lo'])
    order = sorted(groups, key=lambda d: int(d) if d.isdigit() else 999)

    with PdfPages(out) as pdf:
        summary_page(pdf, scans)
        for detnum in order:
            detector_page(pdf, detnum, groups[detnum])
    print(f'Wrote {1 + len(order)} pages -> {out}')


if __name__ == '__main__':
    main()
