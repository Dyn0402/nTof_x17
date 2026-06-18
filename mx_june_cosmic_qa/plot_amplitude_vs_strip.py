#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_amplitude_vs_strip.py

2D histogram of hit pulse height (max_sample, raw ADC) vs strip position, one
panel per plane (X / Y).  Reveals per-strip gain variation, dead strips, hot
strips and the bulk pulse-height band.  Uses max_sample (not the `amplitude`
integral).

Usage:  python plot_amplitude_vs_strip.py <run_key> [--channel] [--ymax=N] [--field=F]
  --channel : x-axis = raw FEU channel index instead of strip position [mm]
  --ymax=N  : clip the pulse-height axis at N (default: 99.9th percentile)
  --field=F : pulse-height branch (default local_max = baseline-subtracted peak
              ADC, 0..4095).  Note `max_sample` is a near-constant small quantity
              here (not the pulse height); `amplitude` is the integral, not peak.

Writes amplitude_vs_strip.png into the run's raw_detector_qa output dir.
"""
import os, sys
import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig

USE_CHANNEL = '--channel' in sys.argv
YMAX = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--ymax=')), None)
AMP = next((a.split('=')[1] for a in sys.argv if a.startswith('--field=')), 'local_max')


def main():
    out_dir = CFG.out_dir('raw_detector_qa')
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    files = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                   if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in files], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    df = cm._map_strip_positions(df, det)
    print(f'Loaded {len(df):,} hits ({CFG.DET_NAME}, FEUs {CFG.MX17_FEUS})')

    amp = df[AMP].to_numpy()
    ymax = YMAX if YMAX is not None else float(np.nanpercentile(amp, 99.9))
    ymin = min(0.0, float(np.nanpercentile(amp, 0.1)))

    planes = [('X', CFG.MX17_FEU_X, 'x_position_mm', 'tab:red'),
              ('Y', CFG.MX17_FEU_Y, 'y_position_mm', 'tab:blue')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (lbl, feu, poscol, _c) in zip(axes, planes):
        sub = df[df['feu'] == feu]
        if USE_CHANNEL:
            x = sub['channel'].to_numpy(); xlabel = f'{lbl} FEU {feu} channel'; xr = [0, 512]
        else:
            x = sub[poscol].to_numpy(); xlabel = f'{lbl} strip position [mm]'; xr = [0, 400]
        y = sub[AMP].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        nx = 128 if USE_CHANNEL else 200
        h = ax.hist2d(x[m], y[m], bins=[nx, 200], range=[xr, [ymin, ymax]],
                      norm=LogNorm(), cmap='viridis')
        plt.colorbar(h[3], ax=ax, label='hits')
        ax.set_xlabel(xlabel); ax.set_ylabel(f'{AMP} [ADC]')
        ax.set_title(f'{CFG.DET_NAME} {lbl} plane (FEU {feu}) — {len(y[m]):,} hits')
    fig.suptitle(f'{AMP} vs strip — {CFG.DET_NAME}  {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout()
    out = os.path.join(out_dir, 'amplitude_vs_strip.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Written: {out}')


if __name__ == '__main__':
    main()
