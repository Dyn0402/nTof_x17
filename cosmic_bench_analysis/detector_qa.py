#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_qa.py

Quick QA plots for a single detector run/subrun:
  1. Hits vs channel number (strip occupancy, one subplot per FEU)
  2. Hits vs strip position in X and Y [mm]
  3. Event rate vs time (trigger_timestamp_ns)
  4. Reconstructed hit positions (earliest-arrival X + Y) — scatter
  5. Average hit amplitude vs time
  6. Average amplitude 2D map — earliest-arrival X+Y pair per event
  7. Average amplitude 2D map — all X+Y pairs matched by nearest time
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from cosmic_micro_tpc_analysis import _map_strip_positions
from common.Mx17StripMap import RunConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = '/media/dylan/data/x17/cosmic_bench/det_4/'
RUN       = 'mx17_det4_ArIso_HV_Scan_5-7-26'
SUB_RUN   = 'resist_530V_drift_900V'
MX17_FEUS = [3, 4]

RUN_CONFIG_PATH = f'{BASE_PATH}{RUN}/run_config.json'
MAP_CSV_PATH    = f'{_ROOT}/mx17_m1_map.csv'

STRIP_PITCH_MM = 0.78   # nominal strip pitch for position histogram bins
AMP_MAP_BINS   = 25     # bins per axis for 2D amplitude maps

OUT_DIR = f'{BASE_PATH}Analysis/QA/{RUN}/{SUB_RUN}/'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Optional[str], name: str, dpi: int = 150) -> None:
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hits(
    base_path: str,
    run: str,
    sub_run: str,
    mx17_feus: list,
    run_config_path: str,
    map_csv_path: str,
) -> pd.DataFrame:
    """Load hits from ROOT files and attach strip positions."""
    hits_dir = f'{base_path}{run}/{sub_run}/combined_hits_root/'
    hit_files = sorted(
        f for f in os.listdir(hits_dir)
        if f.endswith('.root') and '_datrun_' in f
    )
    if not hit_files:
        raise FileNotFoundError(f'No hit files found in {hits_dir}')

    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(mx17_feus)].copy()

    rc  = RunConfig(run_config_path, map_csv_path)
    det = rc.get_detector('mx17_1')
    df  = _map_strip_positions(df, det)

    print(f'Loaded {len(df):,} hits over {df["eventId"].nunique():,} events')
    return df


# ---------------------------------------------------------------------------
# Reconstructed hit positions
# ---------------------------------------------------------------------------

def get_hit_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event take the earliest-arrival strip in X and Y.
    Returns: event_id, x_mm, y_mm, amplitude (mean of the two earliest hits).
    Only events where both axes fired are included.
    """
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

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
    return pos[['x_mm', 'y_mm', 'amplitude']].reset_index().rename(columns={'eventId': 'event_id'})


def get_hit_positions_time_paired(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each event, match every X-hit to the nearest Y-hit in time.
    This gives one (x_mm, y_mm) point per X-hit (rather than just one per
    event), sampling the full length of each track.

    Returns: event_id, x_mm, y_mm, amplitude (mean of paired hit amplitudes),
             dt_ns (|t_x - t_y| of the matched pair).
    """
    records = []
    for event_id, grp in df.groupby('eventId'):
        df_x = grp[grp['x_position_mm'].notna()][['x_position_mm', 'time', 'amplitude']].reset_index(drop=True)
        df_y = grp[grp['y_position_mm'].notna()][['y_position_mm', 'time', 'amplitude']].reset_index(drop=True)
        if df_x.empty or df_y.empty:
            continue

        t_y  = df_y['time'].values
        y_mm = df_y['y_position_mm'].values
        a_y  = df_y['amplitude'].values

        for _, row in df_x.iterrows():
            j = int(np.argmin(np.abs(t_y - row['time'])))
            records.append({
                'event_id':  event_id,
                'x_mm':      row['x_position_mm'],
                'y_mm':      y_mm[j],
                'amplitude': 0.5 * (row['amplitude'] + a_y[j]),
                'dt_ns':     abs(row['time'] - t_y[j]),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Shared 2D amplitude map renderer
# ---------------------------------------------------------------------------

def _plot_amplitude_map(
    x: np.ndarray,
    y: np.ndarray,
    amp: np.ndarray,
    title: str,
    n_bins: int = AMP_MAP_BINS,
    out_dir: Optional[str] = None,
    fig_name: str = 'amplitude_map.png',
) -> None:
    """Bin (x, y) positions and show mean amplitude per bin as a 2D heatmap."""
    mean_amp, xedges, yedges, _ = binned_statistic_2d(
        x, y, amp, statistic='mean', bins=n_bins,
    )
    count, _, _, _ = binned_statistic_2d(x, y, amp, statistic='count', bins=n_bins)
    mean_amp[count < 2] = np.nan  # mask bins with too few entries

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='lightgrey')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        mean_amp.T, origin='lower', aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
    )
    plt.colorbar(im, ax=ax, label='Mean amplitude [ADC]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_dir, fig_name)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hits_vs_channel(df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """Hits per FEU channel (0–511), one subplot per active FEU."""
    feus = sorted(df['feu'].unique())
    n = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False, squeeze=False)
    axes = axes[0]

    for ax, feu in zip(axes, feus):
        channels = df.loc[df['feu'] == feu, 'channel'].values
        lo, hi = int(channels.min()), int(channels.max())
        ax.hist(channels, bins=hi - lo + 1, range=(lo - 0.5, hi + 0.5),
                color='steelblue', edgecolor='none')
        ax.set_xlabel('Channel number')
        ax.set_ylabel('Hits')
        ax.set_title(f'FEU {feu}')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Strip occupancy — {RUN} / {SUB_RUN}', fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'hits_vs_channel.png')


def plot_hits_vs_position(df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """Hits vs physical strip position in X and Y [mm]."""
    df_x = df[df['x_position_mm'].notna()]['x_position_mm']
    df_y = df[df['y_position_mm'].notna()]['y_position_mm']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, xlabel, title in [
        (axes[0], df_x, 'X position [mm]', 'Hits vs X position'),
        (axes[1], df_y, 'Y position [mm]', 'Hits vs Y position'),
    ]:
        if data.empty:
            ax.set_visible(False)
            continue
        lo, hi = data.min(), data.max()
        n_bins = max(1, int(round((hi - lo) / STRIP_PITCH_MM)) + 1)
        ax.hist(data, bins=n_bins, color='steelblue', edgecolor='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Hits')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{RUN} / {SUB_RUN}', fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'hits_vs_position.png')


def plot_hits_vs_time(df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """Event rate vs time, derived from trigger_timestamp_ns."""
    if 'trigger_timestamp_ns' not in df.columns:
        print('trigger_timestamp_ns not in data — skipping time plot')
        return

    ts = df.groupby('eventId')['trigger_timestamp_ns'].first().values
    t0 = ts.min()
    t_sec = (ts - t0) / 1e9

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(t_sec, bins=100, color='steelblue', edgecolor='none')
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel('Events per bin')
    ax.set_title(f'Event rate vs time — {RUN} / {SUB_RUN}')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'hits_vs_time.png')


def plot_hit_position_scatter(pos_df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """Semi-transparent scatter of reconstructed (earliest-arrival) hit positions."""
    if pos_df.empty:
        print('No reconstructed positions to scatter-plot.')
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        pos_df['x_mm'], pos_df['y_mm'],
        s=5, alpha=0.5, color='steelblue', linewidths=0,
    )
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_aspect('equal')
    ax.set_title(
        f'Reconstructed hit positions (earliest arrival)\n'
        f'{RUN} / {SUB_RUN}   —   {len(pos_df):,} events'
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'hit_position_scatter.png')


def plot_amplitude_vs_time(df: pd.DataFrame, n_bins: int = 50, out_dir: Optional[str] = None) -> None:
    """
    Mean hit amplitude per time bin over the run.
    Computes mean amplitude per event first, then bins those by trigger time.
    """
    if 'trigger_timestamp_ns' not in df.columns:
        print('trigger_timestamp_ns not in data — skipping amplitude-vs-time plot')
        return

    event_stats = df.groupby('eventId').agg(
        timestamp=('trigger_timestamp_ns', 'first'),
        mean_amp=('amplitude', 'mean'),
    )
    t0    = event_stats['timestamp'].min()
    t_sec = (event_stats['timestamp'] - t0) / 1e9
    amps  = event_stats['mean_amp'].values

    mean_amp, edges, _ = binned_statistic(t_sec, amps, statistic='mean',    bins=n_bins)
    std_amp,  _,    _ = binned_statistic(t_sec, amps, statistic='std',     bins=n_bins)
    count,    _,    _ = binned_statistic(t_sec, amps, statistic='count',   bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    err = std_amp / np.sqrt(np.maximum(count, 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(centres, mean_amp, yerr=err,
                fmt='o-', color='steelblue', capsize=3, ms=4, lw=1.5)
    ax.set_xlabel('Time since run start [s]')
    ax.set_ylabel('Mean hit amplitude [ADC]')
    ax.set_title(f'Average amplitude vs time — {RUN} / {SUB_RUN}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'amplitude_vs_time.png')


def plot_amplitude_map_earliest(pos_df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """2D amplitude map using the earliest-arrival X+Y pair per event."""
    if pos_df.empty:
        return
    _plot_amplitude_map(
        pos_df['x_mm'].values,
        pos_df['y_mm'].values,
        pos_df['amplitude'].values,
        title=f'Mean amplitude — earliest hit pair\n{RUN} / {SUB_RUN}',
        out_dir=out_dir,
        fig_name='amplitude_map_earliest.png',
    )


def plot_amplitude_map_time_paired(paired_df: pd.DataFrame, out_dir: Optional[str] = None) -> None:
    """2D amplitude map using all within-event X+Y pairs matched by nearest time."""
    if paired_df.empty:
        return
    _plot_amplitude_map(
        paired_df['x_mm'].values,
        paired_df['y_mm'].values,
        paired_df['amplitude'].values,
        title=f'Mean amplitude — time-paired hits\n{RUN} / {SUB_RUN}',
        n_bins=AMP_MAP_BINS,
        out_dir=out_dir,
        fig_name='amplitude_map_time_paired.png',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_hits(BASE_PATH, RUN, SUB_RUN, MX17_FEUS,
                   RUN_CONFIG_PATH, MAP_CSV_PATH)

    plot_hits_vs_channel(df, out_dir=OUT_DIR)
    plot_hits_vs_position(df, out_dir=OUT_DIR)
    plot_hits_vs_time(df, out_dir=OUT_DIR)
    plot_amplitude_vs_time(df, out_dir=OUT_DIR)

    pos_df = get_hit_positions(df)
    print(f'Reconstructed {len(pos_df):,} events with both X and Y hits')
    plot_hit_position_scatter(pos_df, out_dir=OUT_DIR)
    plot_amplitude_map_earliest(pos_df, out_dir=OUT_DIR)

    print('Building time-paired hit positions ...')
    paired_df = get_hit_positions_time_paired(df)
    print(f'  {len(paired_df):,} time-paired (x, y) points from {paired_df["event_id"].nunique():,} events')
    plot_amplitude_map_time_paired(paired_df, out_dir=OUT_DIR)

    plt.show()


if __name__ == '__main__':
    main()