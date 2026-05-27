#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zero_suppress_analysis.py

Checks the zero-suppression efficiency for run_70 by comparing the number of
waveform samples actually stored per event against the theoretical maximum
(n_channels × n_samples_per_waveform from the DREAM DAQ config).

For each sub-run:
  - Counts stored samples per event for each FEU
  - Plots a distribution histogram with the expected (no ZS) line marked
After all sub-runs:
  - Plots mean stored samples and actual/expected ratio vs sub-run

Usage:
    python zero_suppress_analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN = 'run_70'
RUN_DIR  = Path(f'/media/dylan/data/x17/may_beam/runs/{RUN}')
CFG_PATH = RUN_DIR / 'run_config.json'
OUT_DIR  = Path(__file__).parent / 'output' / RUN / 'zero_suppress_analysis'

FEUS = [1, 2, 3]
FEU_LABELS = {1: 'FEU 1 (mx17_3 X)', 2: 'FEU 2 (mx17_3 Y)', 3: 'FEU 3 (mx17_4)'}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_run_config():
    with open(CFG_PATH) as fh:
        cfg = json.load(fh)
    dream = cfg['dream_daq_info']
    n_samples  = dream['n_samples_per_waveform']
    zero_supp  = dream.get('zero_suppress', False)
    samp_beyond = dream.get('samples_beyond_threshold', None)
    subruns    = [s['sub_run_name'] for s in cfg['sub_runs']]
    return n_samples, zero_supp, samp_beyond, subruns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def decoded_files(subrun: str, feu: int):
    return sorted((RUN_DIR / subrun / 'decoded_root').glob(f'*_{feu:02d}.root'))


def detect_n_channels(subrun: str, feu: int, n_probe: int = 20) -> int:
    """Scan the first n_probe events across files to find the max channel index."""
    max_ch = 0
    seen = 0
    for fpath in decoded_files(subrun, feu):
        if seen >= n_probe:
            break
        with uproot.open(fpath) as f:
            if 'nt' not in f:
                continue
            t = f['nt']
            remaining = n_probe - seen
            chs = t['channel'].array(library='np', entry_stop=remaining)
            for c in chs:
                if len(c):
                    max_ch = max(max_ch, int(c.max()))
            seen += len(chs)
    return max_ch + 1  # channels are 0-indexed


def load_sample_counts(subrun: str, feu: int) -> np.ndarray:
    """Return array of stored-sample counts, one entry per event."""
    counts = []
    for fpath in decoded_files(subrun, feu):
        with uproot.open(fpath) as f:
            if 'nt' not in f:
                continue
            t = f['nt']
            samples = t['sample'].array(library='np')
            counts.extend(int(len(s)) for s in samples)
    return np.array(counts, dtype=np.int64)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.name}')


# ---------------------------------------------------------------------------
# Per-subrun distribution
# ---------------------------------------------------------------------------

def plot_subrun_distribution(subrun: str, data: dict, n_expected: dict):
    n = len(FEUS)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, feu in zip(axes, FEUS):
        counts   = data[feu]
        expected = n_expected[feu]

        if len(counts) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(FEU_LABELS[feu])
            continue

        lo = min(counts.min(), int(expected * 0.80))
        hi = max(counts.max(), int(expected * 1.02))
        bins = np.linspace(lo, hi, 120)

        ax.hist(counts, bins=bins, color='steelblue', edgecolor='none')
        ax.axvline(expected, color='red', lw=1.5, ls='--',
                   label=f'Expected (no ZS) = {expected:,}')
        ax.axvline(counts.mean(), color='orange', lw=1.8, ls='-',
                   label=f'Mean = {counts.mean():.0f}')

        ratio = counts.mean() / expected
        ax.set_xlabel('Samples per event')
        ax.set_ylabel('Events')
        ax.set_title(f'{FEU_LABELS[feu]}\nmean/expected = {ratio:.4f}  '
                     f'({len(counts):,} events)')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(
        f'Zero-suppression — stored samples per event\n{RUN} / {subrun}',
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, f'dist_{subrun}.png')


# ---------------------------------------------------------------------------
# Average vs sub-run
# ---------------------------------------------------------------------------

def plot_avg_vs_subrun(subruns, all_means, all_stds, all_expected):
    fig, (ax_abs, ax_ratio) = plt.subplots(
        2, 1, figsize=(max(10, len(subruns) * 1.1), 8), sharex=True
    )

    x      = np.arange(len(subruns))
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for feu, color in zip(FEUS, colors):
        means    = np.array([all_means[sr][feu] for sr in subruns])
        stds     = np.array([all_stds[sr][feu]  for sr in subruns])
        expected = all_expected[subruns[0]][feu]
        ratios   = means / expected

        ax_abs.errorbar(x, means, yerr=stds, fmt='o-', color=color,
                        ms=6, capsize=3, label=FEU_LABELS[feu])
        ax_ratio.plot(x, ratios, 'o-', color=color, ms=6,
                      label=FEU_LABELS[feu])

    # dashed expected lines
    for feu, color in zip(FEUS, colors):
        exp = all_expected[subruns[0]][feu]
        ax_abs.axhline(exp, color=color, ls=':', lw=1.0, alpha=0.6,
                       label=f'Expected {FEU_LABELS[feu]} = {exp:,}')

    ax_abs.set_ylabel('Mean stored samples per event')
    ax_abs.set_title(
        f'{RUN}: stored waveform samples per event vs sub-run\n'
        f'(zero-suppressed; error bars = 1σ across events)'
    )
    ax_abs.legend(fontsize=8, ncol=2, loc='lower right')
    ax_abs.grid(True, alpha=0.3)

    ax_ratio.axhline(1.0, color='red', ls='--', lw=1.0, alpha=0.7,
                     label='No suppression (ratio = 1)')
    ax_ratio.set_ylabel('Mean actual / expected')
    ax_ratio.set_xlabel('Sub-run')
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(subruns, rotation=30, ha='right', fontsize=8)
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, 'avg_samples_vs_subrun.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_samples, zero_supp, samp_beyond, subruns = load_run_config()

    print(f'Run:                       {RUN}')
    print(f'zero_suppress:             {zero_supp}')
    print(f'n_samples_per_waveform:    {n_samples}')
    print(f'samples_beyond_threshold:  {samp_beyond}')
    print(f'Sub-runs ({len(subruns)}):             {subruns}')
    print()

    # Detect n_channels per FEU from first subrun
    print('Detecting n_channels per FEU from first subrun ...')
    n_ch = {}
    for feu in FEUS:
        n_ch[feu] = detect_n_channels(subruns[0], feu)
        print(f'  FEU {feu}: {n_ch[feu]} channels  →  '
              f'expected {n_ch[feu] * n_samples:,} samples/event (no ZS)')
    print()

    all_means    = {}
    all_stds     = {}
    all_expected = {}

    for subrun in subruns:
        print(f'Processing {subrun} ...')
        data     = {}
        expected = {}

        for feu in FEUS:
            counts        = load_sample_counts(subrun, feu)
            exp           = n_ch[feu] * n_samples
            data[feu]     = counts
            expected[feu] = exp

            if len(counts):
                ratio = counts.mean() / exp
                print(f'  FEU {feu}: {len(counts):6,} events  '
                      f'mean={counts.mean():8.0f} ± {counts.std():.0f}  '
                      f'ratio={ratio:.4f}')
            else:
                print(f'  FEU {feu}: no data')

        plot_subrun_distribution(subrun, data, expected)

        all_means[subrun]    = {feu: float(data[feu].mean()) if len(data[feu]) else 0.0
                                for feu in FEUS}
        all_stds[subrun]     = {feu: float(data[feu].std())  if len(data[feu]) else 0.0
                                for feu in FEUS}
        all_expected[subrun] = expected
        print()

    print('Plotting average vs sub-run ...')
    plot_avg_vs_subrun(subruns, all_means, all_stds, all_expected)

    print(f'\nDone. Plots saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
