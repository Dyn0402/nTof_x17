#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-the-fly QA plots for nTof beam data.

Produces per-detector QA plots from combined_hits ROOT files and writes them
to <base_out_dir>/analysis/<run_name>/<subrun_name>/<det_name>/, which is
the directory tree the flask Online QA tab serves.

Usage:
    python detector_qa.py \\
        --subrun_dir  /path/to/run_1/max_hv_1 \\
        --run_config  /path/to/run_1/run_config.json \\
        [--mode all|first|per_file] \\
        [--file_num N]

Modes:
    all      — accumulate all combined_hits files in the subrun (default)
    first    — use only file_num == 0 (fast for long runs)
    per_file — use only the single file_num given by --file_num

Detectors processed: included mx17 detectors only (scintillators skipped).
Position plots (hits_vs_position, scatter, amplitude_map) are produced only
for detectors whose mx_cards field contains 'M1'.
"""

import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d
from pathlib import Path
import uproot

_HERE          = Path(__file__).parent
_NTOF_X17_ROOT = _HERE.parent
sys.path.insert(0, str(_NTOF_X17_ROOT))

from common.Mx17StripMap import Mx17StripMap, Detector as Mx17Detector

MAP_CSV           = _NTOF_X17_ROOT / 'mx17_m1_map.csv'
COMBINED_INNER    = 'combined_hits_root'
STRIP_PITCH_MM    = 0.78
AMP_MAP_BINS      = 25
HITS_AMP_THRESHOLD  = 200   # ADC — used for hits-above-threshold scatter and hits/event plots
HITS_PER_EVENT_ZOOM = 50    # upper x-limit for the zoomed hits/event panels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='nTof on-the-fly detector QA')
    parser.add_argument('--subrun_dir',  required=True, help='Path to the subrun directory')
    parser.add_argument('--run_config',  required=True, help='Path to run_config.json')
    parser.add_argument('--mode',        default='all', choices=['all', 'first', 'per_file'])
    parser.add_argument('--file_num',    type=int,      default=None,
                        help='File number to process (per_file mode only)')
    args = parser.parse_args()

    run_qa(Path(args.subrun_dir), Path(args.run_config), args.mode, args.file_num)


# ---------------------------------------------------------------------------
# Core QA runner
# ---------------------------------------------------------------------------

def run_qa(subrun_dir: Path, run_config_path: Path, mode: str = 'all', file_num: int = None):
    with open(run_config_path) as f:
        run_cfg = json.load(f)

    run_name    = run_cfg['run_name']
    subrun_name = subrun_dir.name
    base_out    = Path(run_cfg['base_out_dir'])

    combined_dir = subrun_dir / COMBINED_INNER
    if not combined_dir.exists():
        print(f'[qa] No {COMBINED_INNER} in {subrun_dir}, skipping')
        return

    included = set(run_cfg.get('included_detectors') or [])

    for det_cfg in run_cfg.get('detectors', []):
        name = det_cfg['name']
        if included and name not in included:
            continue
        if 'dream_feus' not in det_cfg:
            continue  # Skip scintillators and non-strip detectors

        feu_ids     = {v[0] for v in det_cfg['dream_feus'].values()}
        is_mx17     = 'mx17' in det_cfg.get('det_type', '')
        has_mapping = is_mx17 and 'M1' in det_cfg.get('mx_cards', '')

        df = _load_hits(combined_dir, feu_ids, mode, file_num)
        if df is None or df.empty:
            print(f'[qa] {name} — no hits found, skipping')
            continue

        n_events = df['eventId'].nunique() if 'eventId' in df.columns else '?'
        print(f'[qa] {name} — {len(df):,} hits  ({n_events:,} events)')

        if has_mapping:
            if not MAP_CSV.exists():
                print(f'[qa] {name} — map CSV not found at {MAP_CSV}, skipping position plots')
                has_mapping = False
            else:
                try:
                    strip_map = Mx17StripMap(str(MAP_CSV))
                    det       = Mx17Detector(name=name, det_cfg=det_cfg, strip_map=strip_map)
                    df        = _map_strip_positions(df, det)
                except Exception as e:
                    print(f'[qa] {name} — strip mapping failed: {e}')
                    has_mapping = False

        out_dir = base_out / 'analysis' / run_name / subrun_name / name
        out_dir.mkdir(parents=True, exist_ok=True)

        title = f'{run_name} / {subrun_name} / {name}'

        # --- Plots for all detectors ---
        _plot_hits_vs_channel(df, title, out_dir)
        _plot_event_rate(df, title, out_dir)
        _plot_amplitude_vs_time(df, title, out_dir)
        _plot_hits_above_threshold_vs_time(df, title, out_dir)
        _plot_hit_time_dist(df, title, out_dir)
        _plot_amplitude_distribution(df, title, out_dir)
        _plot_hits_per_event(df, title, out_dir)
        _plot_time_vs_channel(df, title, out_dir)

        # --- Position plots for M1 detectors only ---
        if has_mapping:
            _plot_hits_vs_position(df, title, out_dir)
            pos_df = _get_earliest_positions(df)
            if not pos_df.empty:
                _plot_position_scatter(pos_df, title, out_dir)
                _plot_amplitude_map(pos_df['x_mm'].values, pos_df['y_mm'].values,
                                    pos_df['amplitude'].values, title, out_dir)

        plt.close('all')
        print(f'[qa] {name} — saved to {out_dir}')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_file_num(filename: str):
    m = re.match(r'.*_(\d{3})_feu-combined', filename)
    return int(m.group(1)) if m else None


def _load_hits(combined_dir: Path, feu_ids: set, mode: str,
               file_num: int = None) -> pd.DataFrame:
    all_files = sorted(
        f for f in combined_dir.iterdir()
        if f.suffix == '.root' and '_datrun_' in f.name and 'feu-combined' in f.name
    )
    if not all_files:
        return None

    if mode == 'first':
        all_files = [f for f in all_files if _extract_file_num(f.name) == 0]
    elif mode == 'per_file' and file_num is not None:
        all_files = [f for f in all_files if _extract_file_num(f.name) == file_num]

    if not all_files:
        return None

    try:
        df = uproot.concatenate([f'{f}:hits' for f in all_files], library='pd')
    except Exception as e:
        print(f'[qa] Failed to load hits: {e}')
        return None

    return df[df['feu'].isin(feu_ids)].copy()


def _map_strip_positions(df: pd.DataFrame, det) -> pd.DataFrame:
    xs, ys = [], []
    for feu, ch in zip(df['feu'].to_numpy(), df['channel'].to_numpy()):
        pos = det.map_hit(int(feu), int(ch))
        xs.append(pos[0] if pos is not None else None)
        ys.append(pos[1] if pos is not None else None)
    df = df.copy()
    df['x_position_mm'] = xs
    df['y_position_mm'] = ys
    return df


def _get_earliest_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest-arrival X + Y strip per event → reconstructed (x_mm, y_mm)."""
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]
    if df_x.empty or df_y.empty:
        return pd.DataFrame()

    idx_x = df_x.groupby('eventId')['time'].idxmin()
    idx_y = df_y.groupby('eventId')['time'].idxmin()

    x_df = (df_x.loc[idx_x, ['eventId', 'x_position_mm', 'amplitude']]
            .set_index('eventId')
            .rename(columns={'x_position_mm': 'x_mm', 'amplitude': 'amp_x'}))
    y_df = (df_y.loc[idx_y, ['eventId', 'y_position_mm', 'amplitude']]
            .set_index('eventId')
            .rename(columns={'y_position_mm': 'y_mm', 'amplitude': 'amp_y'}))

    pos = x_df.join(y_df, how='inner').dropna(subset=['x_mm', 'y_mm'])
    pos['amplitude'] = 0.5 * (pos['amp_x'] + pos['amp_y'])
    return pos[['x_mm', 'y_mm', 'amplitude']].reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_hits_vs_channel(df: pd.DataFrame, title: str, out_dir: Path):
    feus = sorted(df['feu'].unique())
    n = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, feu in zip(axes, feus):
        ch = df.loc[df['feu'] == feu, 'channel'].values
        lo, hi = int(ch.min()), int(ch.max())
        ax.hist(ch, bins=hi - lo + 1, range=(lo - 0.5, hi + 0.5),
                color='steelblue', edgecolor='none')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Hits')
        ax.set_title(f'FEU {feu}')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Strip occupancy — {title}', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, 'hits_vs_channel.png')


def _plot_event_rate(df: pd.DataFrame, title: str, out_dir: Path):
    if 'trigger_timestamp_ns' not in df.columns:
        return
    ts = df.groupby('eventId')['trigger_timestamp_ns'].first().values
    t_sec = (ts - ts.min()) / 1e9
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(t_sec, bins=100, color='steelblue', edgecolor='none')
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel('Events / bin')
    ax.set_title(f'Event rate — {title}')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'event_rate.png')


def _plot_amplitude_vs_time(df: pd.DataFrame, title: str, out_dir: Path, n_bins: int = 50):
    if 'trigger_timestamp_ns' not in df.columns:
        return
    ev = df.groupby('eventId').agg(ts=('trigger_timestamp_ns', 'first'),
                                    amp=('amplitude', 'mean'))
    t_sec = (ev['ts'] - ev['ts'].min()) / 1e9
    mean_amp, edges, _ = binned_statistic(t_sec, ev['amp'], statistic='mean',  bins=n_bins)
    std_amp,  _,    _ = binned_statistic(t_sec, ev['amp'], statistic='std',   bins=n_bins)
    count,    _,    _ = binned_statistic(t_sec, ev['amp'], statistic='count',  bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    err = std_amp / np.sqrt(np.maximum(count, 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(centres, mean_amp, yerr=err,
                fmt='o-', color='steelblue', capsize=3, ms=4, lw=1.5)
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel('Mean amplitude [ADC]')
    ax.set_title(f'Amplitude vs time — {title}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'amplitude_vs_time.png')


def _plot_hits_vs_position(df: pd.DataFrame, title: str, out_dir: Path):
    df_x = df['x_position_mm'].dropna()
    df_y = df['y_position_mm'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, xlabel in [(axes[0], df_x, 'X [mm]'), (axes[1], df_y, 'Y [mm]')]:
        if data.empty:
            ax.set_visible(False)
            continue
        lo, hi = data.min(), data.max()
        n_bins = max(1, int(round((hi - lo) / STRIP_PITCH_MM)) + 1)
        ax.hist(data, bins=n_bins, color='steelblue', edgecolor='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Hits')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Hits vs position — {title}', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, 'hits_vs_position.png')


def _plot_position_scatter(pos_df: pd.DataFrame, title: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pos_df['x_mm'], pos_df['y_mm'],
               s=5, alpha=0.4, color='steelblue', linewidths=0)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_aspect('equal')
    ax.set_title(f'Hit positions ({len(pos_df):,} events)\n{title}')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir, 'hit_position_scatter.png')


def _plot_amplitude_map(x: np.ndarray, y: np.ndarray, amp: np.ndarray,
                        title: str, out_dir: Path):
    mean_amp, xedges, yedges, _ = binned_statistic_2d(
        x, y, amp, statistic='mean', bins=AMP_MAP_BINS)
    count, _, _, _ = binned_statistic_2d(x, y, amp, statistic='count', bins=AMP_MAP_BINS)
    mean_amp[count < 2] = np.nan

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='lightgrey')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_amp.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)
    plt.colorbar(im, ax=ax, label='Mean amplitude [ADC]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'Amplitude map — {title}')
    fig.tight_layout()
    _save(fig, out_dir, 'amplitude_map.png')


def _plot_hits_above_threshold_vs_time(df: pd.DataFrame, title: str, out_dir: Path):
    """Scatter: hits above HITS_AMP_THRESHOLD per event vs time since run start."""
    if 'trigger_timestamp_ns' not in df.columns:
        return
    ts = df.groupby('eventId')['trigger_timestamp_ns'].first()
    t_sec = (ts - ts.min()) / 1e9

    counts = (df[df['amplitude'] >= HITS_AMP_THRESHOLD]
              .groupby('eventId').size()
              .reindex(ts.index, fill_value=0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t_sec.values, counts.values, s=2, alpha=0.3,
               color='steelblue', linewidths=0)
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel(f'Hits ≥ {HITS_AMP_THRESHOLD} ADC per event')
    ax.set_title(f'Hits above threshold vs time — {title}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'hits_above_threshold_vs_time.png')


def _plot_hit_time_dist(df: pd.DataFrame, title: str, out_dir: Path):
    """Hit time histogram: full range (log y) + auto-zoomed to 1st–99th percentile."""
    times = df['time'].values
    p_lo  = np.percentile(times, 1)
    p_hi  = np.percentile(times, 99)
    pad   = max((p_hi - p_lo) * 0.05, 1.0)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4))

    ax_full.hist(times, bins=200, color='steelblue', edgecolor='none', log=True)
    ax_full.axvspan(p_lo, p_hi, alpha=0.12, color='green',
                    label=f'1–99th pct  [{p_lo:.0f}, {p_hi:.0f}] ns')
    ax_full.set_xlabel('Hit time [ns]')
    ax_full.set_ylabel('Hits')
    ax_full.set_title('Full range (log y)')
    ax_full.legend(fontsize=8)
    ax_full.grid(True, axis='y', alpha=0.3)

    zoom_mask = (times >= p_lo - pad) & (times <= p_hi + pad)
    ax_zoom.hist(times[zoom_mask], bins=200, color='steelblue', edgecolor='none')
    ax_zoom.set_xlabel('Hit time [ns]')
    ax_zoom.set_ylabel('Hits')
    ax_zoom.set_title(f'Zoomed: 1–99th pct  [{p_lo:.0f}, {p_hi:.0f}] ns')
    ax_zoom.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Hit time distribution — {title}', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, 'hit_time_dist.png')


def _plot_amplitude_distribution(df: pd.DataFrame, title: str, out_dir: Path):
    """Amplitude histogram (log y) with threshold line."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df['amplitude'].values, bins=200, color='steelblue',
            edgecolor='none', log=True)
    ax.axvline(HITS_AMP_THRESHOLD, color='red', lw=1.5, ls='--',
               label=f'Threshold = {HITS_AMP_THRESHOLD}')
    ax.set_xlabel('Amplitude [ADC]')
    ax.set_ylabel('Hits')
    ax.set_title(f'Amplitude distribution — {title}')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'amplitude_distribution.png')


def _plot_hits_per_event(df: pd.DataFrame, title: str, out_dir: Path):
    """
    Hits per event: 2×2 grid.
    Rows: all hits | hits above threshold.
    Columns: zoomed to HITS_PER_EVENT_ZOOM (linear y) | full range (log y).
    """
    counts_all = df.groupby('eventId').size()
    counts_thr = (df[df['amplitude'] >= HITS_AMP_THRESHOLD]
                  .groupby('eventId').size())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    rows = [
        (counts_all, 'All hits'),
        (counts_thr, f'Amp ≥ {HITS_AMP_THRESHOLD}'),
    ]

    for row_idx, (counts, row_label) in enumerate(rows):
        if counts.empty:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        hi        = int(counts.max())
        bins_full = np.arange(0.5, hi + 1.5, 1)
        bins_zoom = np.arange(0.5, HITS_PER_EVENT_ZOOM + 1.5, 1)

        # Zoomed panel
        ax_z = axes[row_idx, 0]
        ax_z.hist(counts.clip(upper=HITS_PER_EVENT_ZOOM), bins=bins_zoom,
                  color='steelblue', edgecolor='none')
        ax_z.set_xlabel('Hits per event')
        ax_z.set_ylabel('Events')
        ax_z.set_title(f'{row_label}  —  0–{HITS_PER_EVENT_ZOOM} (linear)')
        ax_z.grid(True, axis='y', alpha=0.3)

        # Full range panel
        ax_f = axes[row_idx, 1]
        ax_f.hist(counts, bins=min(hi, 300), color='steelblue',
                  edgecolor='none', log=True)
        ax_f.set_xlabel('Hits per event')
        ax_f.set_ylabel('Events')
        ax_f.set_title(f'{row_label}  —  full range (log y)')
        ax_f.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Hits per event — {title}', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, 'hits_per_event.png')


def _plot_time_vs_channel(df: pd.DataFrame, title: str, out_dir: Path):
    """2D histogram of hit time vs channel per FEU (time auto-zoomed to 1–99th pct)."""
    times = df['time'].values
    t_lo  = np.percentile(times, 1)
    t_hi  = np.percentile(times, 99)

    feus = sorted(df['feu'].unique())
    n    = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')

    for ax, feu in zip(axes, feus):
        df_feu = df[(df['feu'] == feu) & (df['time'] >= t_lo) & (df['time'] <= t_hi)]
        if df_feu.empty:
            ax.set_title(f'FEU {feu}  (no data)')
            continue
        h, xedges, yedges = np.histogram2d(
            df_feu['channel'].values, df_feu['time'].values, bins=(128, 100))
        im = ax.imshow(np.where(h > 0, h, np.nan).T, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)
        plt.colorbar(im, ax=ax, label='Hits')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Hit time [ns]')
        ax.set_title(f'FEU {feu}')

    fig.suptitle(f'Hit time vs channel — {title}', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, 'time_vs_channel.png')


if __name__ == '__main__':
    main()
