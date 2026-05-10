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

from common.Mx17StripMap import RunConfig

MAP_CSV        = _NTOF_X17_ROOT / 'mx17_m1_map.csv'
COMBINED_INNER = 'combined_hits_root'
STRIP_PITCH_MM = 0.78
AMP_MAP_BINS   = 25


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
        has_mapping = 'M1' in det_cfg.get('mx_cards', '')

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
                    rc  = RunConfig(str(run_config_path), str(MAP_CSV))
                    det = rc.get_detector(name)
                    df  = _map_strip_positions(df, det)
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


if __name__ == '__main__':
    main()
