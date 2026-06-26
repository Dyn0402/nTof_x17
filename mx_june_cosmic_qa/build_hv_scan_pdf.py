#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hv_scan_pdf.py

Compile the resist-HV scans into one styled PDF (matching june_grand_qa.pdf):
  - a summary page overlaying efficiency-vs-HV and resolution-vs-HV for every
    detector scanned, and
  - one page per detector with headline stat cards + the efficiency-vs-HV and
    resolution-vs-HV curves.

Reads each detector's `efficiency_vs_hv.csv` (written by 10_hv_scan_efficiency.py)
from Analysis/<RUN>/hv_scan/<DET_NAME>/ and REPLOTS from the CSV, so the figures
are native/vector (no blur).

Usage:
  python build_hv_scan_pdf.py [key1 key2 ...] [--out=PATH]

Default keys = the viable June scans: g_det2 g_det3 (6-22, both resist stepped
together) and g_det6_long g_det7_long (6-26, det6/det7 stepped at different V).
The 6-23 scan is intentionally excluded (degraded M3 reference -> no reliable
track reference for efficiency).
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

DEFAULT_KEYS = ['g_det2', 'g_det3', 'g_det6_long', 'g_det7_long']
OUT_DEFAULT = '/home/dylan/x17/cosmic_bench/Analysis/june_hv_scans.pdf'
COLORS = {'2': '#1f77b4', '3': '#2ca02c', '4': '#9467bd', '6': '#d62728', '7': '#ff7f0e'}


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
    detnum = cfg.DET_NAME.split('_')[-1]
    drift = ''
    m = re.search(r'drift_(\d+)V', str(df['subrun'].iloc[0]))
    if m:
        drift = f'{m.group(1)} V'
    return dict(key=key, cfg=cfg, df=df, detnum=detnum, drift=drift,
                color=COLORS.get(detnum, 'black'))


def summary_page(pdf, scans):
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                  hspace=0.22, left=0.10, right=0.95, top=0.87, bottom=0.07)
    fig.suptitle('June resist-HV scans — efficiency & resolution',
                 fontsize=17, fontweight='bold', x=0.10, ha='left', y=0.955)
    fig.text(0.10, 0.925,
             'Efficiency = reco within 5 mm of the M3 track, in a fixed per-detector active box.\n'
             '6-22: det2+det3 resist stepped together (drift 1000 V).   '
             '6-26: det6/det7 stepped at different V (drift 700 V).',
             fontsize=8, color='#444444', va='top')

    ax = fig.add_subplot(gs[0])
    for s in scans:
        d = s['df']
        date = s['cfg'].RUN.split('_')[-1]
        ax.errorbar(d['hv'], d['eff_reco'], yerr=d['eff_reco_err'], fmt='o-',
                    color=s['color'], capsize=3, lw=1.8, ms=6,
                    label=f"Detector {s['detnum']}  ({date}, drift {s['drift']})")
    ax.set_xlabel('Resist HV [V]'); ax.set_ylabel('Efficiency (reco ≤5 mm)')
    ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('Efficiency vs resist HV', fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    for s in scans:
        d = s['df']
        if 'sigma_x_mm' in d:
            ax2.plot(d['hv'], d['sigma_x_mm'], 'o-', color=s['color'], lw=1.8, ms=6,
                     label=f"Detector {s['detnum']} σ_x")
            ax2.plot(d['hv'], d['sigma_y_mm'], 's--', color=s['color'], lw=1.4, ms=5,
                     alpha=0.6, label=f"Detector {s['detnum']} σ_y")
    ax2.set_xlabel('Resist HV [V]'); ax2.set_ylabel('Core resolution σ [mm]')
    ax2.set_ylim(0, None); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=7, ncol=2)
    ax2.set_title('Spatial resolution vs resist HV', fontsize=11)

    pdf.savefig(fig, dpi=200); plt.close(fig)


def detector_page(pdf, s):
    cfg, d = s['cfg'], s['df']
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.7, 1.25, 1.25],
                  hspace=0.30, left=0.11, right=0.93, top=0.95, bottom=0.06)

    # headline stats
    valid = d[d['eff_reco'].notna()]
    ipk = valid['eff_reco'].idxmax() if not valid.empty else None
    peak_eff = f"{d.loc[ipk, 'eff_reco']*100:.1f}" if ipk is not None else '—'
    peak_hv = f"{d.loc[ipk, 'hv']:.0f}" if ipk is not None else '—'
    sig_pk = (f"{d.loc[ipk, 'sigma_x_mm']:.2f}"
              if ipk is not None and 'sigma_x_mm' in d and np.isfinite(d.loc[ipk, 'sigma_x_mm'])
              else '—')

    hax = fig.add_subplot(gs[0]); hax.axis('off'); hax.set_xlim(0, 1); hax.set_ylim(0, 1)
    hax.text(0.0, 0.95, f"Detector {s['detnum']}", fontsize=24, fontweight='bold', va='top')
    hax.text(0.0, 0.62, 'resist HV scan', fontsize=12, color='dimgrey', va='top')

    def card(x, value, unit, label):
        sep = '' if unit == '%' else ' '
        hax.text(x, 0.34, f'{value}{sep}{unit}' if unit else f'{value}',
                 fontsize=20, fontweight='bold', va='center', color=s['color'])
        hax.text(x, 0.02, label, fontsize=8.5, color='dimgrey', va='center')
    card(0.00, peak_eff, '%', 'Peak efficiency')
    card(0.27, peak_hv, 'V', 'at resist HV')
    card(0.50, sig_pk, 'mm', 'σ at peak')
    card(0.73, f'{len(d)}', '', 'HV points')

    hv_lo, hv_hi = d['hv'].min(), d['hv'].max()
    ref = '\n'.join([
        f"{cfg.RUN}", f"FEU X/Y: {cfg.MX17_FEUS[0]}/{cfg.MX17_FEUS[1]}",
        f"z: {cfg.DET_PLANE_Z:.0f} mm   drift: {s['drift']}",
        f"HV range: {hv_lo:.0f}–{hv_hi:.0f} V",
    ])
    fig.text(0.93, 0.985, ref, fontsize=6.6, va='top', ha='right', family='monospace',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f4', ec='#bbbbbb', lw=0.6))

    # efficiency vs HV
    axe = fig.add_subplot(gs[1])
    axe.errorbar(d['hv'], d['eff_reco'], yerr=d['eff_reco_err'], fmt='o-',
                 color=s['color'], capsize=4, lw=2, ms=7, label='reco within 5 mm')
    axe.plot(d['hv'], d['eff_anyhit'], 's--', color='dimgrey', ms=6, alpha=0.8,
             label='any hit on detector')
    axe.set_xlabel('Resist HV [V]'); axe.set_ylabel('Efficiency (fixed active box)')
    axe.set_ylim(0, 1.02); axe.grid(True, alpha=0.3); axe.legend()
    axe.set_title('Efficiency vs resist HV', fontsize=11)

    # resolution vs HV
    axr = fig.add_subplot(gs[2])
    if 'sigma_x_mm' in d:
        axr.plot(d['hv'], d['sigma_x_mm'], 'o-', color=s['color'], lw=2, ms=7, label='σ_x')
        axr.plot(d['hv'], d['sigma_y_mm'], 's--', color='darkorange', lw=2, ms=6, label='σ_y')
    axr.set_xlabel('Resist HV [V]'); axr.set_ylabel('Core spatial resolution σ [mm]')
    axr.set_ylim(0, None); axr.grid(True, alpha=0.3); axr.legend()
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
            print(f'  [ok]   {k}: {len(s["df"])} HV points')
    if not scans:
        print('No HV scans found.'); return

    with PdfPages(out) as pdf:
        summary_page(pdf, scans)
        for s in scans:
            detector_page(pdf, s)
    print(f'Wrote {1 + len(scans)} pages -> {out}')


if __name__ == '__main__':
    main()
