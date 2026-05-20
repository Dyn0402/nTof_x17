#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_display.py

Quick-look event display for a single nTof subrun.

Summary figure (6 panels):
  - Hits/event zoomed to ZOOM_MAX_HITS (linear y)
  - Hits/event full range (log y)
  - Channel occupancy per FEU
  - Amplitude distribution
  - Hit-time distribution
  - 2D reconstructed position scatter (earliest X+Y strip pair per event)

Per-event figures (N_EVENTS events):
  - Left:  X strips  — strip position [mm] on x-axis, hit time on y-axis
  - Right: Y strips  — strip position [mm] on x-axis, hit time on y-axis
  - Marker colour encodes amplitude

Configure BASE_PATH / RUN / SUBRUN / MX17_DETECTORS below, then run.
Set EVENT_IDS to specific event IDs, or None to auto-select track-like events.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.Mx17StripMap import Detector, Mx17StripMap
from common.DreamConfig import find_dream_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# BASE_PATH = '/home/dylan/x17/may_beam/runs/'
# RUN       = 'run_6'
# SUBRUN    = 'hv_scan_0'

# BASE_PATH = '/home/dylan/x17/cosmic_bench/det_4/'
# RUN       = 'mx17_det4_ArIso_HV_Scan_5-7-26'
# SUBRUN    = 'resist_510V_drift_900V'

BASE_PATH = '/home/dylan/x17/cosmic_bench/det_3/'
RUN       = 'mx17_det3_long_run_5-6-26'
SUBRUN    = 'long_run'

MX17_DETECTORS = ['mx17_1']
MAP_CSV_PATH   = f'{_ROOT}/mx17_m1_map.csv'

FIG_OUT_DIR: Optional[str] = None  # set to a path to save figures, or None to just show

# Hit selection
AMP_THRESHOLD  = 200   # ignore hits below this amplitude
MAX_HITS_TRACK = 200   # exclusive upper bound for "track-like" event

# Hits/event distribution
ZOOM_MAX_HITS = 50     # upper x-limit of the zoomed panel

# Event display
N_EVENTS       = 5     # number of events to display
MIN_HITS_TRACK = 3     # exclusive lower bound — events must have MORE than this many hits above threshold
EVENT_IDS: Optional[List[int]] = None  # e.g. [42, 100, 7] to override auto-selection

# Waveforms
NS_PER_SAMPLE = 60     # nanoseconds per waveform sample (from DREAM ADC clock)


# ---------------------------------------------------------------------------
# Config / detector setup
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(os.path.join(BASE_PATH, RUN, 'run_config.json')) as f:
        return json.load(f)


def get_detectors(cfg: dict) -> Dict[str, Tuple[List[int], Detector]]:
    """Return {det_name: (feu_ids, Detector)} for each name in MX17_DETECTORS."""
    strip_map = Mx17StripMap(MAP_CSV_PATH)
    result = {}
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        det = Detector(name=name, det_cfg=det_cfg, strip_map=strip_map)
        result[name] = (sorted(det.feu_map.keys()), det)
    return {n: result[n] for n in MX17_DETECTORS if n in result}


# ---------------------------------------------------------------------------
# Data loading & position mapping
# ---------------------------------------------------------------------------

def load_hits(feu_ids: List[int], det: Detector) -> Optional[pd.DataFrame]:
    """Load hits, filter to FEU IDs, and attach strip positions."""
    hits_dir = os.path.join(BASE_PATH, RUN, SUBRUN, 'combined_hits_root')
    if not os.path.isdir(hits_dir):
        print(f'[SKIP] No combined_hits_root in {hits_dir}')
        return None
    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    if not hit_files:
        print(f'[SKIP] No hit files found in {hits_dir}')
        return None
    file_sources = [f'{hits_dir}/{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(feu_ids)].copy()
    df = _map_strip_positions(df, det)
    print(f'Loaded {len(df):,} hits over {df["eventId"].nunique():,} events  '
          f'(FEUs {feu_ids})')
    return df


def load_dream_config():
    """Parse the DREAM .cfg in raw_daq_data/ for the current subrun."""
    subrun_dir = os.path.join(BASE_PATH, RUN, SUBRUN)
    return find_dream_config(subrun_dir)


def _map_strip_positions(df: pd.DataFrame, det: Detector) -> pd.DataFrame:
    xs, ys = [], []
    for feu, ch in zip(df['feu'].to_numpy(), df['channel'].to_numpy()):
        pos = det.map_hit(int(feu), int(ch))
        xs.append(pos[0] if pos is not None else None)
        ys.append(pos[1] if pos is not None else None)
    df = df.copy()
    df['x_position_mm'] = xs
    df['y_position_mm'] = ys
    return df


def get_earliest_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Earliest-arrival X strip + earliest-arrival Y strip per event.
    X strips localise the Y coordinate (y_position_mm); Y strips localise X.
    Returns a DataFrame with event_id, x_mm, y_mm, amplitude.
    """
    df_thr = df[df['amplitude'] >= AMP_THRESHOLD] if AMP_THRESHOLD > 0 else df
    df_x = df_thr[df_thr['y_position_mm'].notna()]   # X strips → y_mm
    df_y = df_thr[df_thr['x_position_mm'].notna()]   # Y strips → x_mm
    if df_x.empty or df_y.empty:
        return pd.DataFrame(columns=['event_id', 'x_mm', 'y_mm', 'amplitude'])

    idx_x = df_x.groupby('eventId')['time'].idxmin()
    idx_y = df_y.groupby('eventId')['time'].idxmin()

    x_df = (df_x.loc[idx_x, ['eventId', 'y_position_mm', 'amplitude']]
            .set_index('eventId')
            .rename(columns={'y_position_mm': 'y_mm', 'amplitude': 'amp_x'}))
    y_df = (df_y.loc[idx_y, ['eventId', 'x_position_mm', 'amplitude']]
            .set_index('eventId')
            .rename(columns={'x_position_mm': 'x_mm', 'amplitude': 'amp_y'}))

    pos = x_df.join(y_df, how='inner').dropna(subset=['x_mm', 'y_mm'])
    pos['amplitude'] = 0.5 * (pos['amp_x'] + pos['amp_y'])
    return (pos[['x_mm', 'y_mm', 'amplitude']]
            .reset_index()
            .rename(columns={'eventId': 'event_id'}))


# ---------------------------------------------------------------------------
# Waveform loading
# ---------------------------------------------------------------------------

def load_waveforms(event_ids: List[int], feu_ids: List[int]) -> Dict[int, Dict[int, tuple]]:
    """
    Load raw waveforms from decoded_root for a set of event IDs.

    Returns:
        {feu_num: {event_id: (samples_arr, channels_arr, amplitudes_arr)}}
    where the arrays are 1-D numpy arrays of the same length (one entry per hit
    in that event × FEU).
    """
    decoded_dir = os.path.join(BASE_PATH, RUN, SUBRUN, 'decoded_root')
    if not os.path.isdir(decoded_dir):
        print('[WARN] No decoded_root directory — waveforms unavailable')
        return {}

    target = set(event_ids)
    result: Dict[int, Dict[int, tuple]] = {feu: {} for feu in feu_ids}

    all_files = sorted(f for f in os.listdir(decoded_dir) if f.endswith('.root'))

    for feu in feu_ids:
        feu_str  = f'_{feu:02d}.'
        feu_files = [f for f in all_files if feu_str in f]
        if not feu_files:
            print(f'[WARN] No decoded files found for FEU {feu}')
            continue

        for fname in feu_files:
            with uproot.open(os.path.join(decoded_dir, fname)) as f:
                if 'nt' not in f:
                    continue
                tree    = f['nt']
                evt_ids = tree['eventId'].array(library='np')
                samples = tree['sample'].array(library='np')
                channels= tree['channel'].array(library='np')
                amps    = tree['amplitude'].array(library='np')

            for i, eid in enumerate(evt_ids):
                if eid in target:
                    result[feu][int(eid)] = (
                        np.asarray(samples[i]),
                        np.asarray(channels[i]),
                        np.asarray(amps[i]),
                    )

    return result


# ---------------------------------------------------------------------------
# Hit counting helpers
# ---------------------------------------------------------------------------

def hits_per_event(df: pd.DataFrame) -> pd.Series:
    df_thr = df[df['amplitude'] >= AMP_THRESHOLD] if AMP_THRESHOLD > 0 else df
    return df_thr.groupby('eventId').size()


def is_track(counts: pd.Series) -> pd.Series:
    return (counts > MIN_HITS_TRACK) & (counts < MAX_HITS_TRACK)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_fig(fig, name: str) -> None:
    if FIG_OUT_DIR is None:
        return
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_OUT_DIR, name), dpi=150, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_summary(df: pd.DataFrame, det_name: str) -> None:
    """Six-panel summary figure."""
    counts  = hits_per_event(df)
    n_track = int(is_track(counts).sum())
    n_total = len(counts)
    feus    = sorted(df['feu'].unique())
    title   = f'{RUN} / {SUBRUN} / {det_name}'
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, fontsize=12)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    max_count = int(counts.max())
    bins_full = np.arange(0.5, max_count + 1.5, 1)
    bins_zoom = np.arange(0.5, ZOOM_MAX_HITS + 1.5, 1)
    window_label = (f'Track window ({MIN_HITS_TRACK+1}–{MAX_HITS_TRACK-1})  '
                    f'→  {n_track}/{n_total}  ({100*n_track/n_total:.1f}%)')

    # ---- Hits/event zoomed ----
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(counts.clip(upper=ZOOM_MAX_HITS), bins=bins_zoom,
             color='steelblue', edgecolor='none')
    ax0.axvspan(MIN_HITS_TRACK + 0.5, min(MAX_HITS_TRACK, ZOOM_MAX_HITS) - 0.5,
                alpha=0.18, color='green', label=window_label)
    ax0.axvline(MIN_HITS_TRACK + 0.5, color='green', lw=1.2, ls='--')
    if MAX_HITS_TRACK <= ZOOM_MAX_HITS:
        ax0.axvline(MAX_HITS_TRACK - 0.5, color='green', lw=1.2, ls='--')
    ax0.set_xlabel('Hits per event')
    ax0.set_ylabel('Events')
    ax0.set_title(f'Hits/event  (0 – {ZOOM_MAX_HITS}, linear)')
    ax0.legend(fontsize=7)
    ax0.grid(True, axis='y', alpha=0.3)

    # ---- Hits/event full range (log) ----
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(counts, bins=bins_full, color='steelblue', edgecolor='none', log=True)
    ax1.axvspan(MIN_HITS_TRACK + 0.5, MAX_HITS_TRACK - 0.5,
                alpha=0.18, color='green')
    ax1.axvline(MIN_HITS_TRACK + 0.5, color='green', lw=1.2, ls='--')
    ax1.axvline(MAX_HITS_TRACK - 0.5, color='green', lw=1.2, ls='--')
    ax1.set_xlabel('Hits per event')
    ax1.set_ylabel('Events')
    ax1.set_title('Hits/event  (full range, log y)')
    ax1.grid(True, axis='y', alpha=0.3)

    # ---- Channel occupancy ----
    ax2 = fig.add_subplot(gs[1, 0])
    for feu, color in zip(feus, colors):
        ch = df.loc[df['feu'] == feu, 'channel'].values
        lo, hi = int(ch.min()), int(ch.max())
        edges = np.arange(lo - 0.5, hi + 1.5, 1)
        ax2.stairs(*np.histogram(ch, bins=edges), color=color, label=f'FEU {feu}')
    ax2.set_xlabel('Channel number')
    ax2.set_ylabel('Hits')
    ax2.set_title('Strip occupancy')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    # ---- Amplitude distribution ----
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(df['amplitude'].values, bins=100, color='steelblue',
             edgecolor='none', log=True)
    if AMP_THRESHOLD > 0:
        ax3.axvline(AMP_THRESHOLD, color='red', lw=1.2, ls='--',
                    label=f'Threshold = {AMP_THRESHOLD}')
        ax3.legend(fontsize=8)
    ax3.set_xlabel('Amplitude [ADC]')
    ax3.set_ylabel('Hits')
    ax3.set_title('Amplitude distribution')
    ax3.grid(True, axis='y', alpha=0.3)

    # ---- Hit-time distribution ----
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df['time'].values, bins=100, color='steelblue', edgecolor='none')
    ax4.set_xlabel('Hit time [ns]')
    ax4.set_ylabel('Hits')
    ax4.set_title('Hit-time distribution')
    ax4.grid(True, axis='y', alpha=0.3)

    # ---- 2D position scatter ----
    ax5 = fig.add_subplot(gs[2, 1])
    pos_df = get_earliest_positions(df)
    if pos_df.empty:
        ax5.text(0.5, 0.5, 'No reconstructed positions',
                 ha='center', va='center', transform=ax5.transAxes)
    else:
        sc = ax5.scatter(pos_df['x_mm'], pos_df['y_mm'],
                         c=pos_df['amplitude'], cmap='viridis',
                         s=4, alpha=0.5, linewidths=0)
        plt.colorbar(sc, ax=ax5, label='Amplitude [ADC]')
        ax5.set_aspect('equal')
        ax5.set_xlabel('X [mm]')
        ax5.set_ylabel('Y [mm]')
    ax5.set_title(f'Reconstructed positions  ({len(pos_df):,} events)')
    ax5.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, f'summary_{det_name}.png')


# ---------------------------------------------------------------------------
# Per-event display
# ---------------------------------------------------------------------------

def select_events(df: pd.DataFrame) -> List[int]:
    if EVENT_IDS is not None:
        return list(EVENT_IDS)
    counts   = hits_per_event(df)
    track_ids = counts[is_track(counts)].index.tolist()
    if not track_ids:
        print('[WARN] No track-like events found; using events with any hits')
        track_ids = counts.index.tolist()
    if len(track_ids) <= N_EVENTS:
        return track_ids
    indices = np.linspace(0, len(track_ids) - 1, N_EVENTS, dtype=int)
    return [track_ids[i] for i in indices]


def plot_event(df: pd.DataFrame, event_id: int, det_name: str) -> None:
    """
    Left panel:  X strips (y_position_mm on x-axis, time on y-axis)
    Right panel: Y strips (x_position_mm on x-axis, time on y-axis)
    Colour encodes amplitude.  Falls back to channel number if positions unavailable.
    """
    ev = df[df['eventId'] == event_id].copy()
    if ev.empty:
        print(f'  Event {event_id}: no hits')
        return

    n_hits     = len(ev)
    n_hits_thr = len(ev[ev['amplitude'] >= AMP_THRESHOLD]) if AMP_THRESHOLD > 0 else n_hits

    # Split by strip axis
    x_strips = ev[ev['y_position_mm'].notna()].copy()  # X strips → y_mm
    y_strips = ev[ev['x_position_mm'].notna()].copy()  # Y strips → x_mm

    # Fallback to channel number if position mapping returned nothing
    x_pos_col  = 'y_position_mm' if not x_strips.empty else None
    y_pos_col  = 'x_position_mm' if not y_strips.empty else None
    x_fallback = x_strips.empty
    y_fallback = y_strips.empty
    if x_fallback:
        x_strips = ev[ev['feu'] == sorted(ev['feu'].unique())[0]].copy()
        x_pos_col = 'channel'
    if y_fallback and len(ev['feu'].unique()) > 1:
        y_strips = ev[ev['feu'] == sorted(ev['feu'].unique())[1]].copy()
        y_pos_col = 'channel'
    elif y_fallback:
        y_strips = pd.DataFrame()
        y_pos_col = None

    amp_all = ev['amplitude'].values
    vmin, vmax = amp_all.min(), amp_all.max()
    cmap = plt.get_cmap('plasma')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        f'{det_name}  —  event {event_id}  '
        f'({n_hits} hits total,  {n_hits_thr} above threshold = {AMP_THRESHOLD})',
        fontsize=11,
    )

    panel_data = [
        (axes[0], x_strips, x_pos_col, 'X strips',
         'Y position [mm]' if not x_fallback else 'Channel'),
        (axes[1], y_strips, y_pos_col, 'Y strips',
         'X position [mm]' if not y_fallback else 'Channel'),
    ]

    for ax, hits, pos_col, label, xlabel in panel_data:
        if hits.empty or pos_col is None:
            ax.text(0.5, 0.5, f'No {label} hits', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        pos = hits[pos_col].values
        t   = hits['time'].values
        amp = hits['amplitude'].values
        sc  = ax.scatter(pos, t, c=amp, cmap=cmap, vmin=vmin, vmax=vmax,
                         s=40, edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label='Amplitude [ADC]')
        if AMP_THRESHOLD > 0:
            # mark below-threshold hits differently
            low = hits[hits['amplitude'] < AMP_THRESHOLD]
            if not low.empty:
                ax.scatter(low[pos_col].values, low['time'].values,
                           marker='x', c='grey', s=20, label='below threshold', zorder=0)
                ax.legend(fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Time [ns]')
        ax.set_title(label)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, f'event_{event_id}_{det_name}.png')


def plot_event_waveforms(
    waveforms: Dict[int, Dict[int, tuple]],
    event_id: int,
    feu_ids: List[int],
    det_name: str,
    ns_per_sample: float = NS_PER_SAMPLE,
) -> None:
    """
    Raw waveform plot for one event, one subplot per FEU (stacked, shared x-axis).
    Each channel in the event is drawn as a separate line, coloured by channel number.
    X-axis: time in ns  (sample index × ns_per_sample).
    """
    # Filter to FEUs that actually have data for this event
    present = [feu for feu in feu_ids if event_id in waveforms.get(feu, {})]
    if not present:
        print(f'  Event {event_id}: no waveform data found')
        return

    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    fig.suptitle(
        f'{det_name}  —  event {event_id}  —  raw waveforms',
        fontsize=11,
    )

    for ax, feu in zip(axes, present):
        samples, channels, amplitudes = waveforms[feu][event_id]
        t_ns = samples.astype(float) * ns_per_sample

        unique_ch = np.unique(channels)
        cmap = plt.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=unique_ch.min(), vmax=unique_ch.max())

        for ch in unique_ch:
            mask = channels == ch
            ax.plot(t_ns[mask], amplitudes[mask],
                    lw=0.8, color=cmap(norm(ch)), alpha=0.8)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Channel')

        if AMP_THRESHOLD > 0:
            ax.axhline(AMP_THRESHOLD, color='red', lw=1, ls='--',
                       label=f'threshold={AMP_THRESHOLD}', zorder=0)
            ax.legend(fontsize=7)

        ax.set_ylabel('Amplitude [ADC]')
        ax.set_title(f'FEU {feu}  ({len(unique_ch)} channels fired)')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time [ns]')
    fig.tight_layout()
    _save_fig(fig, f'waveforms_{event_id}_{det_name}.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg      = load_config()
    det_info = get_detectors(cfg)

    if not det_info:
        print(f'No detectors from {MX17_DETECTORS} found in config')
        return

    # Resolve sample period from Dream config, fall back to NS_PER_SAMPLE
    dream_cfg = load_dream_config()
    if dream_cfg is not None:
        ns_per_sample = dream_cfg.ns_per_sample or NS_PER_SAMPLE
        print(f'Dream config: {dream_cfg}')
    else:
        ns_per_sample = NS_PER_SAMPLE
        print(f'No Dream config found — using NS_PER_SAMPLE={NS_PER_SAMPLE} ns')

    for det_name, (feu_ids, det) in det_info.items():
        print(f'\n=== {det_name}  (FEUs {feu_ids}) ===')
        df = load_hits(feu_ids, det)
        if df is None or df.empty:
            continue

        plot_summary(df, det_name)

        event_ids = select_events(df)
        print(f'Displaying {len(event_ids)} events: {event_ids}')

        print('Loading waveforms ...')
        waveforms = load_waveforms(event_ids, feu_ids)

        for eid in event_ids:
            plot_event(df, eid, det_name)
            plot_event_waveforms(waveforms, eid, feu_ids, det_name,
                                 ns_per_sample=ns_per_sample)

    plt.show()


if __name__ == '__main__':
    main()
