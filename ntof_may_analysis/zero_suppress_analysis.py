#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zero_suppress_analysis.py

Checks the zero-suppression efficiency and event-type structure for a run:
  - Distribution of stored samples per event per FEU vs subrun
  - Suppression fraction vs sample-time-bin per FEU per subrun
  - 2D histogram of hit count vs amplitude sum to separate event types
  - Average stored samples and ratio vs subrun (summary)

Usage:
    python zero_suppress_analysis.py
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN = 'run_70'
# RUN = 'run_36'
RUN_DIR  = Path(f'/media/dylan/data/x17/may_beam/runs/{RUN}')
CFG_PATH = RUN_DIR / 'run_config.json'
OUT_DIR  = Path(__file__).parent / 'output' / RUN / 'zero_suppress_analysis'

FEUS = [1, 2, 3]
FEU_LABELS = {1: 'FEU 1 (mx17_3 X)', 2: 'FEU 2 (mx17_3 Y)', 3: 'FEU 3 (mx17_4)'}

# Minimum ADC amplitude to count a sample as a "hit" for the event-type 2D plot.
# Baseline is ~260 ADC; p99 of noise is ~315; real signals appear above ~400.
HIT_AMP_THRESHOLD = 400

# Cut line for the 2D hit plot (amp_sum = slope * n_hits + intercept).
#   None          → no cut line drawn
#   'interactive' → click two points on the displayed figure to define the line
#                   (slope/intercept printed to stdout so you can hardcode next run)
#   dict          → e.g. {'slope': 50.0, 'intercept': 0.0}
CUT_CONFIG = None


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
        sample_counts_by_feu = {}
        sample_times_by_feu  = {}
        amps_by_feu          = {}
        expected             = {}

        for feu in FEUS:
            sample_times_list, amps_list = load_event_data_full(subrun, feu)
            counts = np.array([len(st) for st in sample_times_list], dtype=np.int64)
            exp    = n_ch[feu] * n_samples

            sample_counts_by_feu[feu] = counts
            sample_times_by_feu[feu]  = sample_times_list
            amps_by_feu[feu]          = amps_list
            expected[feu]             = exp

            if len(counts):
                ratio = counts.mean() / exp
                print(f'  FEU {feu}: {len(counts):6,} events  '
                      f'mean={counts.mean():8.0f} ± {counts.std():.0f}  '
                      f'ratio={ratio:.4f}')
            else:
                print(f'  FEU {feu}: no data')

        plot_subrun_distribution(subrun, sample_counts_by_feu, expected)
        plot_suppression_profile(subrun, sample_times_by_feu, n_samples, n_ch)
        plot_hit_2d(subrun, amps_by_feu, CUT_CONFIG)

        all_means[subrun]    = {feu: float(sample_counts_by_feu[feu].mean())
                                if len(sample_counts_by_feu[feu]) else 0.0
                                for feu in FEUS}
        all_stds[subrun]     = {feu: float(sample_counts_by_feu[feu].std())
                                if len(sample_counts_by_feu[feu]) else 0.0
                                for feu in FEUS}
        all_expected[subrun] = expected
        print()

    print('Plotting average vs sub-run ...')
    plot_avg_vs_subrun(subruns, all_means, all_stds, all_expected)
    plt.show()

    print(f'\nDone. Plots saved to {OUT_DIR}')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_run_config():
    with open(CFG_PATH) as fh:
        cfg = json.load(fh)
    dream = cfg['dream_daq_info']
    n_samples   = dream['n_samples_per_waveform']
    zero_supp   = dream.get('zero_suppress', False)
    samp_beyond = dream.get('samples_beyond_threshold', None)
    subruns     = [s['sub_run_name'] for s in cfg['sub_runs']]
    return n_samples, zero_supp, samp_beyond, subruns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def decoded_files(subrun: str, feu: int):
    return sorted((RUN_DIR / subrun / 'decoded_root').glob(f'*_{feu:02d}.root'))


def detect_n_channels(subrun: str, feu: int, n_probe: int = 20) -> int:
    """Scan the first n_probe events to find the max channel index."""
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
    return max_ch + 1


def load_event_data_full(subrun: str, feu: int) -> tuple:
    """Return (sample_times_list, amps_list) as lists of per-event arrays.

    sample_times_list[i]: time-bin indices (0..n_samples-1) stored in event i
    amps_list[i]:         ADC amplitudes of those stored samples
    """
    sample_times_list = []
    amps_list         = []
    for fpath in decoded_files(subrun, feu):
        with uproot.open(fpath) as f:
            if 'nt' not in f:
                continue
            t = f['nt']
            sample_times_list.extend(t['sample'].array(library='np'))
            amps_list.extend(t['amplitude'].array(library='np'))
    return sample_times_list, amps_list


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved {path.name}')


# ---------------------------------------------------------------------------
# Per-subrun stored-sample distribution
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

        lo   = min(counts.min(), int(expected * 0.80))
        hi   = max(counts.max(), int(expected * 1.02))
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
# Suppression fraction vs sample-time-bin
# ---------------------------------------------------------------------------

def plot_suppression_profile(subrun: str, sample_times_by_feu: dict,
                             n_samples: int, n_ch: dict):
    """Fraction of (event × channel) slots suppressed at each time bin."""
    n = len(FEUS)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes  = axes[0]
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for ax, feu, color in zip(axes, FEUS, colors):
        times_list = sample_times_by_feu[feu]
        n_events   = len(times_list)

        if n_events == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(FEU_LABELS[feu])
            continue

        stored_counts = np.zeros(n_samples, dtype=np.int64)
        for times in times_list:
            if len(times):
                stored_counts += np.bincount(
                    times.astype(np.int64).clip(0, n_samples - 1),
                    minlength=n_samples,
                )[:n_samples]

        total_possible   = n_events * n_ch[feu]
        suppression_frac = 1.0 - stored_counts / total_possible

        ax.plot(np.arange(n_samples), suppression_frac, color=color, lw=1.0)
        ax.set_xlabel('Sample time bin')
        ax.set_ylabel('Suppression fraction')
        ax.set_title(f'{FEU_LABELS[feu]}\n({n_events:,} events)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_samples - 1)
        ax.text(0.97, 0.97, f'mean = {suppression_frac.mean():.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    fig.suptitle(
        f'Sample suppression fraction vs time bin\n{RUN} / {subrun}',
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, f'suppression_profile_{subrun}.png')


# ---------------------------------------------------------------------------
# 2D event-type plot: hit count vs amplitude sum
# ---------------------------------------------------------------------------

def plot_hit_2d(subrun: str, amps_by_feu: dict, cut_config):
    """2D histogram of above-threshold hit count vs amplitude sum, per FEU + combined."""
    n_panels = len(FEUS) + 1  # three FEUs + one combined panel
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    metrics = {}  # feu → (n_hits, amp_sum) arrays

    for ax, feu in zip(axes[:len(FEUS)], FEUS):
        amps_list = amps_by_feu[feu]

        if not amps_list:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(FEU_LABELS[feu])
            continue

        n_hits   = np.array([int((a > HIT_AMP_THRESHOLD).sum()) for a in amps_list])
        above    = [a[a > HIT_AMP_THRESHOLD] for a in amps_list]
        amp_avg  = np.array([float(a.mean()) if len(a) else 0.0 for a in above])
        metrics[feu] = (n_hits, amp_avg)

        _draw_hit_2d_panel(ax, n_hits, amp_avg,
                           title=f'{FEU_LABELS[feu]}\n({len(n_hits):,} events)')

    # Combined panel: sum metrics across FEUs (assumes identical event ordering)
    ax_combined = axes[len(FEUS)]
    feus_with_data = [f for f in FEUS if f in metrics]
    if len(feus_with_data) > 1:
        n_events = min(len(metrics[f][0]) for f in feus_with_data)
        combined_hits = sum(metrics[f][0][:n_events] for f in feus_with_data)
        combined_avg  = sum(metrics[f][1][:n_events] for f in feus_with_data) / len(feus_with_data)
        _draw_hit_2d_panel(ax_combined, combined_hits, combined_avg,
                           title=f'Combined (all FEUs)\n({n_events:,} events)')
        metrics['combined'] = (combined_hits, combined_avg)
    else:
        ax_combined.set_visible(False)

    fig.suptitle(
        f'Event-type separation: hit count vs mean amplitude\n'
        f'{RUN} / {subrun}  (threshold = {HIT_AMP_THRESHOLD} ADC)',
        fontsize=11,
    )
    fig.tight_layout()

    # Interactive cut: block here to let user click before saving
    if cut_config == 'interactive':
        plt.draw()
        plt.pause(0.1)
        print(f'\n  [hit_2d {subrun}] Click TWO points to define the cut line '
              f'(amp_sum = slope × n_hits + intercept).')
        pts = plt.ginput(n=2, timeout=0, show_clicks=True)
        if len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            if x2 != x1:
                slope     = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                print(f"  → CUT_CONFIG = {{'slope': {slope:.6g}, 'intercept': {intercept:.6g}}}")
                cut_config = {'slope': slope, 'intercept': intercept}
            else:
                print('  → Points have same x; skipping cut line.')
                cut_config = None
        else:
            print('  → Fewer than 2 points; skipping cut line.')
            cut_config = None

    if isinstance(cut_config, dict):
        for ax in axes[:len(FEUS) + 1]:
            if not ax.get_visible():
                continue
            _overlay_cut_line(ax, cut_config)

    _save(fig, f'hit_2d_{subrun}.png')


def _draw_hit_2d_panel(ax, n_hits, amp_sum, title: str):
    if n_hits.max() == 0 and amp_sum.max() == 0:
        ax.text(0.5, 0.5, f'No hits above\nthreshold {HIT_AMP_THRESHOLD}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    h, xedges, yedges, img = ax.hist2d(
        n_hits, amp_sum,
        bins=200,
        norm=mcolors.LogNorm(),
        cmap='viridis',
    )
    plt.colorbar(img, ax=ax, label='Events')
    ax.set_xlabel(f'Hit count  (amp > {HIT_AMP_THRESHOLD})')
    ax.set_ylabel('Mean amplitude  (above threshold)')
    ax.set_title(title)


def _overlay_cut_line(ax, cut_config: dict):
    slope     = cut_config['slope']
    intercept = cut_config['intercept']
    xl = np.array(ax.get_xlim())
    ax.plot(xl, slope * xl + intercept, 'r-', lw=2,
            label=f'Cut: y = {slope:.3g}·x + {intercept:.3g}')
    ax.legend(fontsize=8)


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


if __name__ == '__main__':
    main()
