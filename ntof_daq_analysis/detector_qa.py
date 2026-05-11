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
from common.DreamConfig import find_dream_config

MAP_CSV              = _NTOF_X17_ROOT / 'mx17_m1_map.csv'
COMBINED_INNER       = 'combined_hits_root'
DECODED_ROOT_DIR     = 'decoded_root'
STRIP_PITCH_MM       = 0.78
AMP_MAP_BINS         = 25
HITS_AMP_THRESHOLD   = 200   # ADC — used for hits-above-threshold scatter and hits/event plots
HITS_PER_EVENT_ZOOM  = 50    # upper x-limit for the zoomed hits/event panels
WF_NS_PER_SAMPLE     = 20.0  # default waveform sample period [ns]; overridden by DreamConfig
WF_N_FIRST           = 10    # number of first events to plot
WF_N_RANDOM          = 20    # number of additional random events to plot


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

        # --- Neutron beam: waveform + hits per event ---
        if run_cfg.get('beam_type') == 'neutrons':
            _plot_neutron_waveforms(subrun_dir, df, feu_ids, title, out_dir)

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


# ---------------------------------------------------------------------------
# Neutron beam — waveform + hits per event
# ---------------------------------------------------------------------------

def _load_wf_for_events(decoded_dir: Path, feu_ids, event_ids) -> dict:
    """
    Load raw waveforms from decoded_root for the given event IDs.

    Returns {feu_id: {event_id: (samples_arr, channels_arr, amplitudes_arr)}}.
    """
    target    = set(int(e) for e in event_ids)
    result    = {feu: {} for feu in feu_ids}
    all_files = sorted(decoded_dir.iterdir(), key=lambda p: p.name)

    for feu in feu_ids:
        feu_str   = f'_{feu:02d}.'
        feu_files = [f for f in all_files if f.suffix == '.root' and feu_str in f.name]
        for fpath in feu_files:
            try:
                with uproot.open(fpath) as uf:
                    if 'nt' not in uf:
                        continue
                    tree        = uf['nt']
                    evt_ids_arr = tree['eventId'].array(library='np')
                    samples     = tree['sample'].array(library='np')
                    channels    = tree['channel'].array(library='np')
                    amps        = tree['amplitude'].array(library='np')
            except Exception as e:
                print(f'[qa/wf] Error reading {fpath.name}: {e}')
                continue
            for i, eid in enumerate(evt_ids_arr):
                if int(eid) in target:
                    result[feu][int(eid)] = (
                        np.asarray(samples[i]),
                        np.asarray(channels[i]),
                        np.asarray(amps[i]),
                    )
    return result


def _plot_waveform_hits_event(waveforms: dict, df_all: pd.DataFrame, evt_id: int,
                               feu_ids, title: str, out_dir: Path,
                               ns_per_sample: float) -> None:
    """
    Two rows per FEU (waveform + hits scatter) for a single event, shared time axis.
    Saves to out_dir/waveform_event_<evt_id>.png.
    """
    present = [feu for feu in sorted(feu_ids) if evt_id in waveforms.get(feu, {})]
    if not present:
        return

    n_feus = len(present)
    fig, axes = plt.subplots(n_feus * 2, 1, figsize=(12, 3.5 * n_feus),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]

    fig.suptitle(f'{title}\nEvent {evt_id} — waveforms + hits', fontsize=10)

    df_evt = df_all[df_all['eventId'] == evt_id]

    for i, feu in enumerate(present):
        ax_wf = axes[i * 2]
        ax_ht = axes[i * 2 + 1]

        # Waveforms — channels coloured by channel number
        samples, channels, amplitudes = waveforms[feu][evt_id]
        t_us      = samples.astype(float) * ns_per_sample / 1000
        unique_ch = np.unique(channels)
        cmap      = plt.get_cmap('coolwarm')
        norm      = plt.Normalize(vmin=unique_ch.min(), vmax=unique_ch.max())
        for ch in unique_ch:
            mask = channels == ch
            ax_wf.plot(t_us[mask], amplitudes[mask], lw=0.7, color=cmap(norm(ch)), alpha=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_wf, label='Channel')
        ax_wf.axhline(HITS_AMP_THRESHOLD, color='red', lw=1, ls='--',
                      label=f'thr={HITS_AMP_THRESHOLD}', zorder=0)
        ax_wf.set_ylabel('Amplitude [ADC]')
        ax_wf.set_title(f'FEU {feu}  ({len(unique_ch)} channels fired)', fontsize=9)
        ax_wf.legend(fontsize=7)
        ax_wf.grid(True, alpha=0.2)

        # Hits scatter — time vs channel, colour = amplitude
        df_feu = df_evt[df_evt['feu'] == feu]
        if not df_feu.empty:
            t_hit_us = df_feu['time'].values / 3 / 1000  # 1/3 ns units → μs
            sc = ax_ht.scatter(t_hit_us, df_feu['channel'].values,
                               c=df_feu['amplitude'].values, cmap='plasma',
                               s=20, linewidths=0)
            plt.colorbar(sc, ax=ax_ht, label='Amplitude [ADC]')
        ax_ht.set_ylabel('Channel')
        ax_ht.set_title(f'FEU {feu} hits', fontsize=9)
        ax_ht.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time [μs]')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    _save(fig, out_dir, f'waveform_event_{evt_id:06d}.png')


def _plot_neutron_waveforms(subrun_dir: Path, df: pd.DataFrame, feu_ids,
                             title: str, out_dir: Path) -> None:
    """
    Plot waveform + hits for the first WF_N_FIRST events and WF_N_RANDOM random
    events from the subrun.  Figures are saved to out_dir/waveforms/.
    """
    decoded_dir = subrun_dir / DECODED_ROOT_DIR
    if not decoded_dir.exists():
        print(f'[qa/wf] No {DECODED_ROOT_DIR}/ in {subrun_dir}, skipping waveform plots')
        return

    if not feu_ids:
        return

    dream_cfg     = find_dream_config(subrun_dir)
    ns_per_sample = (dream_cfg.ns_per_sample
                     if dream_cfg and dream_cfg.ns_per_sample is not None
                     else WF_NS_PER_SAMPLE)

    all_event_ids = sorted(df['eventId'].unique())
    n_total       = len(all_event_ids)
    first_n       = min(WF_N_FIRST, n_total)
    first_events  = all_event_ids[:first_n]

    remaining     = all_event_ids[first_n:]
    n_random      = min(WF_N_RANDOM, len(remaining))
    rng           = np.random.default_rng(42)
    random_events = (sorted(rng.choice(remaining, size=n_random, replace=False).tolist())
                     if n_random > 0 else [])

    selected = list(first_events) + random_events
    if not selected:
        return

    print(f'[qa/wf] Loading waveforms for {len(selected)} events '
          f'({first_n} first + {len(random_events)} random)')
    waveforms = _load_wf_for_events(decoded_dir, sorted(feu_ids), selected)

    wf_dir = out_dir / 'waveforms'
    wf_dir.mkdir(exist_ok=True)

    for evt_id in selected:
        _plot_waveform_hits_event(waveforms, df, evt_id, feu_ids,
                                   title, wf_dir, ns_per_sample)
        plt.close('all')

    print(f'[qa/wf] Saved {len(selected)} waveform figures → {wf_dir}')


if __name__ == '__main__':
    main()
