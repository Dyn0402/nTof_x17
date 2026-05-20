#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic QA plots for run_70 / resist_final_790V beam data.

Reads all combined_hits ROOT files from the subrun and produces:
  - Amplitude distribution (log y, threshold line)
  - Hits vs channel (per FEU)
  - Hits per event per FEU — zoomed 0–HITS_ZOOM + full range (log y)
    Threshold version includes events with 0 hits (spike at 0)
  - Event rate vs time
  - Hit time distribution (full + zoomed) with time window lines
  - Hits above threshold vs time (scatter)
  - Track candidate filtering + plots for filtered events

Orientation mapping (from run_config):
  FEU 1          → X strips  (mx17_3)
  FEU 2          → Y strips  (mx17_3)
  FEU 3 ch 0-255 → X strips  (mx17_4)
  FEU 3 ch 256+  → Y strips  (mx17_4)

Track candidates: events with ≥1 qualifying hit in each orientation,
where qualifying = amp > AMP_THRESHOLD AND time in [TIME_WIN_MIN, TIME_WIN_MAX].

Usage:
    python beam_qa.py
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import uproot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBRUN_DIR   = Path('/media/dylan/data/x17/may_beam/runs/run_70/resist_final_790V')
COMBINED_DIR = SUBRUN_DIR / 'combined_hits_root'
OUT_DIR      = Path(__file__).parent / 'output' / 'run_70_resist_final_790V'

DECODED_DIR   = SUBRUN_DIR / 'decoded_root'

TITLE         = 'run_70 / resist_final_790V'
AMP_THRESHOLD = 200    # ADC — hit amplitude threshold
TIME_WIN_MIN  = 300    # ns — lower edge of beam time window
TIME_WIN_MAX  = 1000   # ns — upper edge of beam time window
HITS_ZOOM     = 30     # upper x-limit for zoomed hits/event panels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df          = load_hits()
    n_total     = count_total_events()
    n_combined  = df['eventId'].nunique()
    df          = assign_orientation(df)

    track_ids   = filter_track_candidates(df)
    n_track     = len(track_ids)
    print(f'\nEvent counts:')
    print(f'  Total triggers (decoded root): {n_total:,}')
    print(f'  Events in combined_hits:       {n_combined:,}  ({100*n_combined/n_total:.1f}% of total)')
    print(f'  Track candidates:              {n_track:,}  ({100*n_track/n_total:.1f}% of total)')

    print('\nPlotting amplitude distribution ...')
    plot_amplitude_distribution(df)

    print('Plotting hits vs channel ...')
    plot_hits_vs_channel(df)
    plot_hits_vs_channel_threshold(df)

    print('Plotting hits per event ...')
    plot_hits_per_event(df)

    print('Plotting event rate ...')
    plot_event_rate(df)

    print('Plotting hit time distribution ...')
    plot_hit_time_dist(df)

    print('Plotting hits above threshold vs time ...')
    plot_hits_above_threshold_vs_time(df)

    print('Plotting track candidate diagnostics ...')
    df_track = df[df['eventId'].isin(track_ids)]
    plot_hits_per_event(df_track, suffix='_track', extra_title='track candidates')
    plot_hit_time_dist(df_track, suffix='_track', extra_title='track candidates')
    plot_track_candidate_summary(n_total, n_combined, n_track)

    print(f'\nDone. Plots saved to {OUT_DIR}')
    plt.show()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_file_num(name: str):
    m = re.match(r'.*_(\d{3})_feu-combined', name)
    return int(m.group(1)) if m else None


def load_hits() -> pd.DataFrame:
    files = sorted(
        f for f in COMBINED_DIR.iterdir()
        if f.suffix == '.root' and '_datrun_' in f.name and 'feu-combined' in f.name
    )
    if not files:
        raise FileNotFoundError(f'No combined_hits files in {COMBINED_DIR}')
    print(f'Loading {len(files)} combined_hits files ...')
    df = uproot.concatenate([f'{f}:hits' for f in files], library='pd')
    print(f'  {len(df):,} hits  |  {df["eventId"].nunique():,} events  |  FEUs: {sorted(df["feu"].unique())}')
    return df


def count_total_events() -> int:
    """Count total triggered events from decoded root (FEU 01 files cover every trigger)."""
    feu1_files = sorted(f for f in DECODED_DIR.iterdir() if f.name.endswith('_01.root'))
    if not feu1_files:
        print('[warn] No decoded root FEU-01 files found; total event count unavailable')
        return 0
    total = 0
    for fpath in feu1_files:
        with uproot.open(fpath) as uf:
            if 'nt' in uf:
                total += len(uf['nt']['eventId'].array(library='np'))
    return total


def assign_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'orientation' column: 'x' or 'y' based on FEU and channel."""
    conditions = [
        df['feu'] == 1,
        df['feu'] == 2,
        (df['feu'] == 3) & (df['channel'] < 256),
        (df['feu'] == 3) & (df['channel'] >= 256),
    ]
    df = df.copy()
    df['orientation'] = np.select(conditions, ['x', 'y', 'x', 'y'], default='unknown')
    return df


def filter_track_candidates(df: pd.DataFrame) -> np.ndarray:
    """
    Return array of eventIds that have ≥1 qualifying hit in EACH orientation.
    Qualifying: amp > AMP_THRESHOLD AND TIME_WIN_MIN ≤ time ≤ TIME_WIN_MAX.
    """
    qual = df[
        (df['amplitude'] > AMP_THRESHOLD) &
        (df['time'] >= TIME_WIN_MIN) &
        (df['time'] <= TIME_WIN_MAX)
    ]
    has_x = set(qual.loc[qual['orientation'] == 'x', 'eventId'].unique())
    has_y = set(qual.loc[qual['orientation'] == 'y', 'eventId'].unique())
    return np.array(sorted(has_x & has_y))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.name}')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_amplitude_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df['amplitude'].values, bins=300, color='steelblue', edgecolor='none', log=True)
    ax.axvline(AMP_THRESHOLD, color='red', lw=1.5, ls='--',
               label=f'Threshold = {AMP_THRESHOLD}')
    ax.set_xlabel('Amplitude [ADC]')
    ax.set_ylabel('Hits')
    ax.set_title(f'Amplitude distribution — {TITLE}')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'amplitude_distribution.png')


def plot_hits_vs_channel(df: pd.DataFrame):
    feus = sorted(df['feu'].unique())
    n    = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, feu in zip(axes, feus):
        ch   = df.loc[df['feu'] == feu, 'channel'].values
        lo, hi = int(ch.min()), int(ch.max())
        ax.hist(ch, bins=hi - lo + 1, range=(lo - 0.5, hi + 0.5),
                color='steelblue', edgecolor='none')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Hits')
        ax.set_title(f'FEU {feu}')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Strip occupancy (all hits) — {TITLE}', fontsize=10)
    fig.tight_layout()
    _save(fig, 'hits_vs_channel.png')


def plot_hits_vs_channel_threshold(df: pd.DataFrame):
    """Same as above but only hits above amplitude threshold."""
    df_thr = df[df['amplitude'] >= AMP_THRESHOLD]
    feus   = sorted(df['feu'].unique())
    n      = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, feu in zip(axes, feus):
        ch = df_thr.loc[df_thr['feu'] == feu, 'channel'].values
        if len(ch) == 0:
            ax.set_title(f'FEU {feu} (no hits above threshold)')
            continue
        lo, hi = int(ch.min()), int(ch.max())
        ax.hist(ch, bins=hi - lo + 1, range=(lo - 0.5, hi + 0.5),
                color='steelblue', edgecolor='none')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Hits')
        ax.set_title(f'FEU {feu}')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Strip occupancy (amp ≥ {AMP_THRESHOLD}) — {TITLE}', fontsize=10)
    fig.tight_layout()
    _save(fig, 'hits_vs_channel_threshold.png')


def plot_hits_per_event(df: pd.DataFrame, suffix: str = '', extra_title: str = ''):
    """
    Per-FEU hits/event.  Two figures:
      1. All hits (no threshold)
      2. Threshold hits — includes events with 0 hits (spike at zero)

    Each figure: FEUs as columns, rows = (zoomed 0–HITS_ZOOM, full log-y).
    suffix      : appended to output filename (e.g. '_track')
    extra_title : appended to suptitle (e.g. 'track candidates')
    """
    feus          = sorted(df['feu'].unique())
    n             = len(feus)
    all_event_ids = df['eventId'].unique()
    title_suffix  = f' [{extra_title}]' if extra_title else ''

    for label, df_src, include_zeros in [
        ('all hits',               df,                                False),
        (f'amp > {AMP_THRESHOLD}', df[df['amplitude'] > AMP_THRESHOLD], True),
    ]:
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), squeeze=False)

        for col, feu in enumerate(feus):
            df_feu = df_src[df_src['feu'] == feu]

            counts = df_feu.groupby('eventId').size()
            if include_zeros:
                counts = counts.reindex(all_event_ids, fill_value=0)

            if counts.empty:
                for row in range(2):
                    axes[row, col].set_visible(False)
                continue

            hi = int(counts.max())

            # --- zoomed panel (row 0) ---
            ax_z = axes[0, col]
            bins_z = np.arange(-0.5, HITS_ZOOM + 1.5, 1)
            ax_z.hist(counts.clip(upper=HITS_ZOOM), bins=bins_z,
                      color='steelblue', edgecolor='none')
            ax_z.set_xlabel('Hits per event')
            ax_z.set_ylabel('Events')
            ax_z.set_title(f'FEU {feu} — 0–{HITS_ZOOM} (linear)')
            ax_z.set_xlim(-0.5, HITS_ZOOM + 0.5)
            ax_z.grid(True, axis='y', alpha=0.3)

            # --- full range panel (row 1) ---
            ax_f = axes[1, col]
            ax_f.hist(counts, bins=min(hi + 1, 500), range=(-0.5, hi + 0.5),
                      color='steelblue', edgecolor='none', log=True)
            ax_f.set_xlabel('Hits per event')
            ax_f.set_ylabel('Events')
            ax_f.set_title(f'FEU {feu} — full range (log y)')
            ax_f.grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'Hits per event ({label}){title_suffix} — {TITLE}', fontsize=10)
        fig.tight_layout()
        base = 'hits_per_event_all' if 'all' in label else 'hits_per_event_threshold'
        _save(fig, f'{base}{suffix}.png')


def plot_event_rate(df: pd.DataFrame):
    if 'trigger_timestamp_ns' not in df.columns:
        return
    ts    = df.groupby('eventId')['trigger_timestamp_ns'].first().values
    t_sec = (ts - ts.min()) / 1e9
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(t_sec, bins=100, color='steelblue', edgecolor='none')
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel('Events / bin')
    ax.set_title(f'Event rate — {TITLE}')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'event_rate.png')


def plot_hit_time_dist(df: pd.DataFrame, suffix: str = '', extra_title: str = ''):
    times = df['time'].values
    p_lo  = np.percentile(times, 1)
    p_hi  = np.percentile(times, 99)
    pad   = max((p_hi - p_lo) * 0.05, 1.0)
    title_suffix = f' [{extra_title}]' if extra_title else ''

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4))

    ax_full.hist(times, bins=200, color='steelblue', edgecolor='none', log=True)
    ax_full.axvspan(p_lo, p_hi, alpha=0.12, color='green',
                    label=f'1–99th pct  [{p_lo:.0f}, {p_hi:.0f}] ns')
    ax_full.axvline(TIME_WIN_MIN, color='red',    lw=1.5, ls='--', label=f'{TIME_WIN_MIN} ns')
    ax_full.axvline(TIME_WIN_MAX, color='orange', lw=1.5, ls='--', label=f'{TIME_WIN_MAX} ns')
    ax_full.set_xlabel('Hit time [ns]')
    ax_full.set_ylabel('Hits')
    ax_full.set_title('Full range (log y)')
    ax_full.legend(fontsize=8)
    ax_full.grid(True, axis='y', alpha=0.3)

    mask = (times >= p_lo - pad) & (times <= p_hi + pad)
    ax_zoom.hist(times[mask], bins=200, color='steelblue', edgecolor='none')
    ax_zoom.axvline(TIME_WIN_MIN, color='red',    lw=1.5, ls='--', label=f'{TIME_WIN_MIN} ns')
    ax_zoom.axvline(TIME_WIN_MAX, color='orange', lw=1.5, ls='--', label=f'{TIME_WIN_MAX} ns')
    ax_zoom.set_xlabel('Hit time [ns]')
    ax_zoom.set_ylabel('Hits')
    ax_zoom.set_title(f'Zoomed: 1–99th pct  [{p_lo:.0f}, {p_hi:.0f}] ns')
    ax_zoom.legend(fontsize=8)
    ax_zoom.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Hit time distribution{title_suffix} — {TITLE}', fontsize=10)
    fig.tight_layout()
    _save(fig, f'hit_time_dist{suffix}.png')


def plot_hits_above_threshold_vs_time(df: pd.DataFrame):
    if 'trigger_timestamp_ns' not in df.columns:
        return
    ts    = df.groupby('eventId')['trigger_timestamp_ns'].first()
    t_sec = (ts - ts.min()) / 1e9

    counts = (df[df['amplitude'] >= AMP_THRESHOLD]
              .groupby('eventId').size()
              .reindex(ts.index, fill_value=0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t_sec.values, counts.values, s=2, alpha=0.3,
               color='steelblue', linewidths=0)
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel(f'Hits ≥ {AMP_THRESHOLD} ADC per event')
    ax.set_title(f'Hits above threshold vs time — {TITLE}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'hits_above_threshold_vs_time.png')


def plot_track_candidate_summary(n_total: int, n_combined: int, n_track: int):
    """Bar chart showing event counts at each selection stage."""
    labels  = ['Total triggers', 'Any hit\n(combined)', 'Track candidates\n(X+Y, time window)']
    counts  = [n_total, n_combined, n_track]
    colors  = ['#4c78a8', '#72b7b2', '#e45756']
    pcts    = [100.0, 100 * n_combined / n_total if n_total else 0,
               100 * n_track / n_total if n_total else 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor='none')
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f'{bar.get_height():,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Events')
    ax.set_title(f'Event selection summary — {TITLE}')
    ax.set_ylim(0, n_total * 1.18)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'event_selection_summary.png')


if __name__ == '__main__':
    main()