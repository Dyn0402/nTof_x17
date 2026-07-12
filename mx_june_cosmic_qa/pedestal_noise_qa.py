#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pedestal_noise_qa.py

Per-channel pedestal mean and noise sigma from a DREAM pedestal-threshold run,
computed BOTH raw and after common-noise subtraction (CNS).

The mm_processor (WaveformAnalyzer::computePedestals) computes the per-channel
mean/RMS straight from the raw pedestal-file amplitudes -- NO common-noise
subtraction -- and uses thresholdSigma * RMS as the per-channel hit threshold.
This script reproduces that raw RMS and additionally applies the SAME CNS the
processor would apply to data (median across 64-channel blocks, per sample, per
event; WaveformAnalyzer::applyCommonNoiseSubtraction) so we can see the true
uncorrelated noise floor -- the right basis for a flat threshold.

Usage:
    python pedestal_noise_qa.py <decoded_root_dir> [out_dir]
Reads every *_pedthr_*_NN.root (decoded 'nt' tree) in the dir, one per FEU.
"""
import os
import re
import sys
import glob

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLOCK = 64        # DREAM common-noise block size
NCHAN = 512       # channels per FEU


def feu_of(path):
    m = re.search(r'_(\d{3})_(\d{2})[._]', os.path.basename(path))
    return int(m.group(2)) if m else -1


def analyse_file(path):
    """Return per-channel (mean_raw, sig_raw, mean_cns, sig_cns, count) arrays."""
    with uproot.open(path) as f:
        nt = f['nt']
        ch_j = nt['channel'].array(library='np')
        sm_j = nt['sample'].array(library='np')
        amp_j = nt['amplitude'].array(library='np')

    s_sum   = np.zeros(NCHAN); s_sq   = np.zeros(NCHAN); s_n   = np.zeros(NCHAN)
    c_sum   = np.zeros(NCHAN); c_sq   = np.zeros(NCHAN)
    nsamp_seen = 0

    for ch, sm, amp in zip(ch_j, sm_j, amp_j):
        ch = np.asarray(ch, dtype=np.int64)
        sm = np.asarray(sm, dtype=np.int64)
        amp = np.asarray(amp, dtype=np.float64)
        nsamp = int(sm.max()) + 1 if sm.size else 0
        if nsamp == 0:
            continue
        nsamp_seen = max(nsamp_seen, nsamp)

        # dense [NCHAN, nsamp], NaN where absent
        dense = np.full((NCHAN, nsamp), np.nan)
        dense[ch, sm] = amp

        # raw accumulation
        valid = ~np.isnan(dense)
        d0 = np.where(valid, dense, 0.0)
        s_sum += d0.sum(axis=1)
        s_sq  += (d0 * d0).sum(axis=1)
        s_n   += valid.sum(axis=1)

        # common-noise subtraction: median across the 64-ch block per sample
        blk = dense.reshape(NCHAN // BLOCK, BLOCK, nsamp)
        med = np.nanmedian(blk, axis=1)                 # [nblock, nsamp]
        cns = (blk - med[:, None, :]).reshape(NCHAN, nsamp)
        c0 = np.where(valid, cns, 0.0)
        c_sum += c0.sum(axis=1)
        c_sq  += (c0 * c0).sum(axis=1)

    n = np.where(s_n > 0, s_n, np.nan)
    mean_raw = s_sum / n
    sig_raw  = np.sqrt(np.maximum(s_sq / n - mean_raw**2, 0.0))
    mean_cns = c_sum / n
    sig_cns  = np.sqrt(np.maximum(c_sq / n - mean_cns**2, 0.0))
    return mean_raw, sig_raw, mean_cns, sig_cns, s_n


def main():
    decoded_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else decoded_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(decoded_dir, '*_pedthr_*.root')),
                   key=feu_of)
    if not files:
        print(f'No *_pedthr_*.root in {decoded_dir}'); sys.exit(1)

    results = {}
    for p in files:
        feu = feu_of(p)
        print(f'[FEU {feu:02d}] {os.path.basename(p)}')
        results[feu] = analyse_file(p)

    # ---- summary table ----
    print('\n=== pedestal noise summary (median over live channels) ===')
    print(f'{"FEU":>4} {"#live":>6} {"mean_raw":>9} {"sig_raw":>8} {"sig_CNS":>8} {"CNS/raw":>8}')
    summary = {}
    for feu, (mr, sr, mc, sc, n) in results.items():
        live = n > 0
        msr = np.nanmedian(sr[live]); msc = np.nanmedian(sc[live])
        mmr = np.nanmedian(mr[live])
        summary[feu] = (msr, msc)
        print(f'{feu:>4} {int(live.sum()):>6} {mmr:>9.1f} {msr:>8.2f} {msc:>8.2f} {msc/msr:>8.2f}')

    # ---- plot: rows = FEU, col0 = mean(raw), col1 = sigma raw vs CNS ----
    nf = len(results)
    fig, axes = plt.subplots(nf, 2, figsize=(13, 2.6 * nf), squeeze=False)
    x = np.arange(NCHAN)
    for r, (feu, (mr, sr, mc, sc, n)) in enumerate(results.items()):
        live = n > 0
        a0, a1 = axes[r][0], axes[r][1]
        a0.plot(x[live], mr[live], '.', ms=2, color='navy')
        a0.set_ylabel(f'FEU {feu:02d}\nmean [ADC]')
        a0.grid(alpha=.3)
        a1.plot(x[live], sr[live], '.', ms=2, color='crimson', label='raw')
        a1.plot(x[live], sc[live], '.', ms=2, color='green', label='CNS-subtracted')
        a1.axhline(np.nanmedian(sc[live]), color='green', ls='--', lw=1,
                   label=f'median CNS={np.nanmedian(sc[live]):.2f}')
        a1.set_ylabel('sigma [ADC]')
        a1.set_ylim(bottom=0)
        a1.grid(alpha=.3)
        if r == 0:
            a0.set_title('Pedestal mean vs channel (raw)')
            a1.set_title('Pedestal noise sigma vs channel')
            a1.legend(fontsize=7, loc='upper right')
    axes[-1][0].set_xlabel('channel'); axes[-1][1].set_xlabel('channel')
    fig.suptitle('Pedestal QA  —  pedestals_06-22-26_19-14-37  (raw vs common-noise-subtracted)', y=1.0)
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'pedestal_noise_raw_vs_cns.png')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f'\nWrote {out_png}')


if __name__ == '__main__':
    main()
