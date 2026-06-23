#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_pedestal_qa.py

Pedestal-metrics QA for a single mx17 detector. The strip-reconstruction hit finder
keeps a pulse only if peakAmplitude >= thresholdSigma * pedestalRMS_channel, with
thresholdSigma = 5.0 and the pedestal RMS computed NON-robustly (plain variance, no
spark/saturation rejection) -- see mm_strip_reconstruction WaveformAnalyzer. A few
discharge samples in the pedestal run therefore inflate a channel's RMS many-fold and
silently raise its hit threshold into the hundreds-to-thousands of ADC, killing real
micro-TPC signal on that strip for the whole subrun.

This script reads the per-channel pedestal mean / RMS that the processing ACTUALLY
applied (the `pedestals` TTree written inside each per-FEU hits_root file) and plots,
for each of the detector's two FEUs:
  * pedestal mean vs channel / strip position (baseline)
  * pedestal RMS  vs channel / strip position, with the per-FEU median + 95th pct
  * nominal hit threshold (5 * RMS) per channel vs the proposed FLAT threshold
    (3 * median RMS), so the suppressed strips are obvious
  * histogram of RMS (with median, 95th pct, and the implied flat sigma)

Usage:  python 11_pedestal_qa.py o22_det3
Products: <OUT_BASE>/pedestal_qa/  (per-FEU pngs + pedestal_summary.txt)

It also prints the per-FEU median RMS -> the flat sigma to feed the flat-threshold
reprocessing (WFA_FLAT_SIGMA = median RMS, WFA_THRESHOLD_SIGMA = 3).
"""

import os
import glob
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig

THRESHOLD_SIGMA = 5.0      # nominal thresholdSigma in WaveformAnalyzer
FLAT_NSIGMA     = 3.0      # proposed flat threshold = FLAT_NSIGMA * median(RMS)
CH_PER_CONN     = 64       # DREAM channels per connector (8 connectors / FEU)


def hits_root_dir():
    return f'{CFG.BASE_PATH}{CFG.RUN}/{CFG.SUB_RUN}/hits_root/'


def load_pedestals(feu):
    """Return DataFrame(channel, mean, rms) for one FEU (first hits_root file -- the
    pedestal is per-subrun per-FEU, identical across file-indices)."""
    pat = os.path.join(hits_root_dir(), f'*_{feu:02d}_hits.root')
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f'no hits_root for FEU {feu} in {hits_root_dir()}')
    t = uproot.open(files[0])
    if 'pedestals' not in [k.split(';')[0] for k in t.keys()]:
        raise KeyError(f'no pedestals tree in {files[0]}')
    ped = t['pedestals'].arrays(['channel', 'mean', 'rms'], library='np')
    df = pd.DataFrame(ped).sort_values('channel').reset_index(drop=True)
    return df, os.path.basename(files[0])


def map_positions(df, feu, det):
    """Attach strip position (mm) per channel via det.map_hit; mark connected strips.
    Unconnected DREAM channels -> pos_mm NaN, connected=False."""
    coord = 0 if feu == CFG.MX17_FEU_X else 1   # X-FEU uses x_position, Y-FEU uses y_position
    pos, conn = [], []
    for ch in df['channel'].to_numpy():
        p = det.map_hit(feu, int(ch))
        if p is not None:
            pos.append(p[coord]); conn.append(True)
        else:
            pos.append(np.nan); conn.append(False)
    df = df.copy()
    df['pos_mm'] = pos
    df['connected'] = conn
    return df


def plot_feu(df, feu, axis_label, out_dir, summary):
    conn = df['connected'].values
    rms_c = df['rms'].values[conn]                  # connected strips only
    med = float(np.median(rms_c))                   # flat sigma = median over real strips
    p95 = float(np.percentile(rms_c, 95))
    flat_thr = FLAT_NSIGMA * med
    nominal_thr = THRESHOLD_SIGMA * df['rms'].values
    # connected strips whose nominal threshold exceeds the flat one (= suppressed signal)
    n_suppressed = int(np.sum((nominal_thr > flat_thr) & conn))
    n_conn = int(conn.sum())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ch = df['channel'].values
    cc, uc = conn, ~conn

    def conngrid(ax):
        for c in range(CH_PER_CONN, 512, CH_PER_CONN):
            ax.axvline(c, color='grey', lw=0.5, alpha=0.4)
        ax.grid(alpha=0.3)

    # (0,0) pedestal mean vs channel
    ax = axes[0, 0]
    ax.plot(ch[cc], df['mean'].values[cc], '.', ms=3, color='#264653', label='strip')
    ax.plot(ch[uc], df['mean'].values[uc], '.', ms=3, color='#cccccc', label='unconnected')
    ax.set_xlabel('DREAM channel'); ax.set_ylabel('pedestal mean (ADC)')
    ax.set_title(f'FEU {feu} ({axis_label})  pedestal baseline'); conngrid(ax)
    ax.legend(fontsize=8)

    # (0,1) pedestal RMS vs channel
    ax = axes[0, 1]
    ax.plot(ch[cc], df['rms'].values[cc], '.', ms=3, color='#2a9d8f', label='strip')
    ax.plot(ch[uc], df['rms'].values[uc], '.', ms=3, color='#cccccc', label='unconnected')
    ax.axhline(med, color='green', lw=1.2, label=f'strip median={med:.1f}')
    ax.axhline(p95, color='orange', lw=1.0, ls='--', label=f'strip 95th={p95:.1f}')
    ax.set_xlabel('DREAM channel'); ax.set_ylabel('pedestal RMS (ADC)')
    ax.set_title('pedestal RMS (sets the hit threshold)'); conngrid(ax)
    ax.legend(fontsize=8)

    # (1,0) threshold: nominal 5*RMS vs flat 3*median
    ax = axes[1, 0]
    ax.plot(ch[cc], nominal_thr[cc], '.', ms=3, color='#e76f51',
            label=f'nominal {THRESHOLD_SIGMA:.0f}σ × RMS_ch (strips)')
    ax.axhline(flat_thr, color='blue', lw=1.4,
               label=f'flat {FLAT_NSIGMA:.0f}σ × median = {flat_thr:.0f} ADC')
    ax.set_xlabel('DREAM channel'); ax.set_ylabel('hit threshold (ADC)')
    ax.set_title(f'hit threshold per channel  ({n_suppressed}/{n_conn} strips nominal>flat)')
    conngrid(ax); ax.legend(fontsize=8); ax.set_yscale('log')

    # (1,1) RMS histogram (connected strips)
    ax = axes[1, 1]
    ax.hist(rms_c, bins=80, color='#2a9d8f', alpha=0.8)
    ax.axvline(med, color='green', lw=1.4, label=f'median={med:.1f}  (flat σ)')
    ax.axvline(p95, color='orange', lw=1.0, ls='--', label=f'95th={p95:.1f}')
    ax.set_xlabel('pedestal RMS (ADC), connected strips'); ax.set_ylabel('channels')
    ax.set_title('pedestal RMS distribution'); ax.set_yscale('log')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{CFG.DET_NAME}  {CFG.RUN}/{CFG.SUB_RUN}  —  FEU {feu} ({axis_label}) pedestals',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, f'pedestal_feu{feu}_{axis_label}.png')
    fig.savefig(out, dpi=130); plt.close(fig)

    line = (f'FEU {feu:2d} ({axis_label}): strip median RMS={med:6.2f}  95th={p95:6.2f}  '
            f'max={rms_c.max():7.2f}  flat_thr(3×med)={flat_thr:6.1f} ADC  '
            f'strips nominal-thr>flat: {n_suppressed}/{n_conn}')
    summary.append(line)
    print('  ' + line)
    return med


def main():
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    out_dir = CFG.out_dir('pedestal_qa')
    summary = [f'Pedestal QA — {CFG.DET_NAME}  {CFG.RUN}/{CFG.SUB_RUN}',
               f'hit threshold = {THRESHOLD_SIGMA:.0f} × per-channel pedestal RMS (non-robust)',
               f'proposed flat threshold = {FLAT_NSIGMA:.0f} × per-FEU median RMS', '']
    feu_meds = {}
    for feu, axis in ((CFG.MX17_FEU_X, 'X'), (CFG.MX17_FEU_Y, 'Y')):
        df, src = load_pedestals(feu)
        df = map_positions(df, feu, det)
        print(f'FEU {feu} ({axis}) from {src}: {len(df)} channels')
        feu_meds[feu] = plot_feu(df, feu, axis, out_dir, summary)

    summary.append('')
    summary.append('Reprocess flat-threshold with (per FEU):')
    for feu, m in feu_meds.items():
        summary.append(f'  WFA_THRESHOLD_SIGMA={FLAT_NSIGMA:.0f}  WFA_FLAT_SIGMA={m:.2f}   (FEU {feu})')
    open(os.path.join(out_dir, 'pedestal_summary.txt'), 'w').write('\n'.join(summary) + '\n')
    print(f'\nPedestal QA written to: {out_dir}')


if __name__ == '__main__':
    main()
