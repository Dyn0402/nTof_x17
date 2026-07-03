#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/beam_hv_scan.py

Combined beam HV-scan analysis for runs that contain both gamma-flash and
thermal-track events within a single HV-scan subrun structure.

Each triggered event is classified as:
  * gamma-flash  — >GAMMA_STRIP_FRACTION of all MX17 strips fire within
                   FLASH_DETECT_WINDOW_NS of t = 0.
  * thermal      — everything else.

Then for each resist-HV step:
  * Gamma-flash  → mean hits/event in each FLASH_TIME_BIN (ns) vs. HV.
  * Thermal      → track-finding efficiency vs. HV (hits in thermal time
                   window inside (THERMAL_MIN_HITS, THERMAL_MAX_HITS)).

Output PNGs are written to FIG_OUT_DIR for web viewing.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.Mx17StripMap import Detector, Mx17StripMap

# ---------------------------------------------------------------------------
# Configuration — tune these for each new run / measurement campaign
# ---------------------------------------------------------------------------
BASE_PATH = '/mnt/data/x17/beam_may/runs/'

# List of runs whose subruns form one combined HV scan.
RUNS = ['run_54']

MX17_DETECTORS = ['mx17_3', 'mx17_4']
MAP_CSV_PATH = f'{_ROOT}/mx17_m1_map.csv'

RUN_LABEL = RUNS[0] if len(RUNS) == 1 else f'{RUNS[0]}+{RUNS[-1]}'
FIG_OUT_DIR = f'{BASE_PATH}Analysis/Beam_HV_Scan/{RUN_LABEL}/'

# Amplitude threshold applied to all hit selection (ADC counts).
AMP_THRESHOLD = 400

# ---------------------------------------------------------------------------
# Gamma-flash classification
# ---------------------------------------------------------------------------
# An event is a gamma flash when the fraction of all MX17 strips that fire
# within FLASH_DETECT_WINDOW_NS of t=0 is at least GAMMA_STRIP_FRACTION.
# In practice the observable maximum fraction is ~0.65 (not all connected strips
# can fire in the flash window), so the threshold is set below that value.
# Inspect the event-classification diagnostic plot to tune this for new runs.
FLASH_DETECT_WINDOW_NS = 2000   # ns  — must be ≤ first FLASH_TIME_BIN upper edge
GAMMA_STRIP_FRACTION   = 0.40   # fraction threshold (data shows clear gap ~0.05–0.45)

# Time bins (ns) for hits-per-event vs HV (gamma-flash events).
# Aim: 2 bins.  Add or remove entries here to change granularity.
FLASH_TIME_BINS: List[List[int]] = [
    [0,    2000],   # direct flash / prompt ionisation
    [2000, 10000],  # post-flash / drift hits
]

# ---------------------------------------------------------------------------
# Thermal-track selection
# ---------------------------------------------------------------------------
# Only hits in [THERMAL_T0_NS, THERMAL_T1_NS] above AMP_THRESHOLD are counted
# when deciding whether a thermal event contains a track.
THERMAL_T0_NS    =    0   # ns
THERMAL_T1_NS    = 3000   # ns  — upper edge of the thermal track time window
THERMAL_MIN_HITS =    2   # inclusive lower bound: event needs ≥ this many hits
THERMAL_MAX_HITS =  200   # exclusive upper bound: event needs < this many hits

# Diagnostic subrun: (run, subrun_name) or None → first subrun found.
DIAGNOSTIC_SUBRUN: Optional[Tuple[str, str]] = None

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(base_path: str, run: str) -> dict:
    with open(os.path.join(base_path, run, 'run_config.json')) as f:
        return json.load(f)


def build_detector_feu_map(cfg: dict) -> Dict[str, List[int]]:
    """Return {det_name: sorted list of FEU IDs} for each requested detector."""
    strip_map = Mx17StripMap(MAP_CSV_PATH)
    result: Dict[str, List[int]] = {}
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        det = Detector(name=name, det_cfg=det_cfg, strip_map=strip_map)
        result[name] = sorted(det.feu_map.keys())
    return {name: result[name] for name in MX17_DETECTORS if name in result}


def build_hv_maps(cfgs: Dict[str, dict]) -> Dict[str, Dict[Tuple[str, str], int]]:
    """Build {det_name: {(run, subrun_name): resist_hv}} from all run configs."""
    first_cfg = next(iter(cfgs.values()))
    resist_channels: Dict[str, Tuple[int, int]] = {}
    for det_cfg in first_cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        resist = det_cfg.get('hv_channels', {}).get('resist')
        if resist is not None:
            resist_channels[name] = (int(resist[0]), int(resist[1]))

    hv_maps: Dict[str, Dict[Tuple[str, str], int]] = {n: {} for n in MX17_DETECTORS}
    for run, cfg in cfgs.items():
        for sub in cfg.get('sub_runs', []):
            subrun_name = sub['sub_run_name']
            hvs = sub.get('hvs', {})
            for det_name, (card, ch) in resist_channels.items():
                hv = hvs.get(str(card), {}).get(str(ch))
                if hv is not None:
                    hv_maps[det_name][(run, subrun_name)] = int(hv)
    return hv_maps


def find_subruns(base_path: str, cfgs: Dict[str, dict]) -> List[Tuple[str, str]]:
    """Return (run, subrun_name) pairs that exist on disk, in config order."""
    result = []
    for run, cfg in cfgs.items():
        run_dir = os.path.join(base_path, run)
        on_disk = {name for name in os.listdir(run_dir)
                   if os.path.isdir(os.path.join(run_dir, name))}
        for sub in cfg.get('sub_runs', []):
            name = sub['sub_run_name']
            if name in on_disk:
                result.append((run, name))
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hits(run: str, subrun: str, feu_ids: List[int]) -> Optional[pd.DataFrame]:
    """Load combined hits for one subrun, filtered to the given FEU IDs."""
    hits_dir = os.path.join(BASE_PATH, run, subrun, 'combined_hits_root')
    if not os.path.isdir(hits_dir):
        print(f'  [SKIP] {run}/{subrun}: no combined_hits_root directory')
        return None
    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    if not hit_files:
        print(f'  [SKIP] {run}/{subrun}: no hit files found')
        return None
    sources = [f'{hits_dir}/{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(sources, library='pd')
    df = df[df['feu'].isin(feu_ids)].copy()
    print(f'  Loaded {len(df):,} hits over {df["eventId"].nunique():,} events '
          f'(FEUs {feu_ids})')
    return df


def get_total_events(run: str, subrun: str) -> Optional[int]:
    """Return total triggered events from the maximum eventId in decoded_root."""
    decoded_dir = os.path.join(BASE_PATH, run, subrun, 'decoded_root')
    if not os.path.isdir(decoded_dir):
        return None
    root_files = sorted(f for f in os.listdir(decoded_dir) if f.endswith('.root'))
    if not root_files:
        return None
    max_id = 0
    for fname in root_files:
        with uproot.open(os.path.join(decoded_dir, fname)) as f:
            if 'nt' not in f:
                continue
            ids = f['nt']['eventId'].array(library='np')
            if len(ids) > 0:
                max_id = max(max_id, int(ids.max()))
    return max_id if max_id > 0 else None


def compute_total_strips(
    base_path: str,
    run: str,
    subrun: str,
    all_feus: List[int],
    det_feus: Dict[str, List[int]],
) -> Tuple[int, Dict[str, int]]:
    """
    Count unique (feu, channel) pairs per detector and combined, from one
    subrun's data.  Used as the denominator for gamma-flash fraction.
    Returns (combined_total, {det_name: n_strips}).
    """
    hits_dir = os.path.join(base_path, run, subrun, 'combined_hits_root')
    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    sources = [f'{hits_dir}/{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(sources, library='pd')
    df = df[df['feu'].isin(all_feus)]
    combined = df[['feu', 'channel']].drop_duplicates().shape[0]
    per_det = {
        name: df[df['feu'].isin(feus)][['feu', 'channel']].drop_duplicates().shape[0]
        for name, feus in det_feus.items()
    }
    return combined, per_det


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

def classify_events(
    df: pd.DataFrame,
    total_strips: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Classify each event as gamma-flash or thermal.

    Returns (flash_event_ids, thermal_event_ids, fractions_series).
    fractions_series is indexed by eventId, values are early-strip fractions.
    """
    df_amp   = df[df['amplitude'] >= AMP_THRESHOLD]
    df_early = df_amp[df_amp['time'].between(0, FLASH_DETECT_WINDOW_NS)]

    # Unique (feu, channel) pairs firing in the flash window per event.
    early_counts = (df_early[['eventId', 'feu', 'channel']]
                    .drop_duplicates()
                    .groupby('eventId')
                    .size()
                    .rename('n_early'))

    all_ids   = df['eventId'].unique()
    fractions = early_counts.reindex(all_ids, fill_value=0) / total_strips

    flash_ids   = fractions[fractions >= GAMMA_STRIP_FRACTION].index.values
    thermal_ids = fractions[fractions <  GAMMA_STRIP_FRACTION].index.values
    return flash_ids, thermal_ids, fractions


# ---------------------------------------------------------------------------
# Per-subrun metrics
# ---------------------------------------------------------------------------

def flash_hits_per_event(
    df: pd.DataFrame,
    flash_ids: np.ndarray,
    feu_ids: List[int],
) -> List[float]:
    """Mean hits/event in each FLASH_TIME_BIN for one detector, flash events only."""
    n = len(flash_ids)
    if n == 0:
        return [np.nan] * len(FLASH_TIME_BINS)
    df_f = df[df['eventId'].isin(flash_ids) &
              df['feu'].isin(feu_ids) &
              (df['amplitude'] >= AMP_THRESHOLD)]
    return [
        len(df_f[df_f['time'].between(t0, t1)]) / n
        for t0, t1 in FLASH_TIME_BINS
    ]


def thermal_efficiency(
    df: pd.DataFrame,
    thermal_ids: np.ndarray,
    feu_ids: List[int],
    n_total_thermal: int,
) -> Tuple[int, float, float, float]:
    """
    Compute track-finding efficiency for thermal events in one detector.
    Returns (n_tracks, efficiency, efficiency_error, mean_amplitude).
    """
    df_t = df[df['eventId'].isin(thermal_ids) &
              df['feu'].isin(feu_ids) &
              (df['amplitude'] >= AMP_THRESHOLD) &
              df['time'].between(THERMAL_T0_NS, THERMAL_T1_NS)]

    counts   = df_t.groupby('eventId').size().reindex(thermal_ids, fill_value=0)
    mean_amp = float(df_t['amplitude'].mean()) if not df_t.empty else np.nan
    n_tracks = int(((counts >= THERMAL_MIN_HITS) & (counts < THERMAL_MAX_HITS)).sum())
    n_denom  = n_total_thermal if n_total_thermal > 0 else 1
    eff  = n_tracks / n_denom
    err  = float(np.sqrt(eff * (1 - eff) / n_denom))
    return n_tracks, eff, err, mean_amp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_OUT_DIR, name), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_flash_hits_vs_hv(flash_results: Dict[str, dict], gas: str = '') -> None:
    """
    One subplot per FLASH_TIME_BIN showing mean hits/flash-event vs resist HV.
    Each detector is a separate coloured line.
    """
    n_bins = len(FLASH_TIME_BINS)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(n_bins, 1, figsize=(9, 3.5 * n_bins), sharex=True)
    if n_bins == 1:
        axs = [axs]

    for bi, (t0, t1) in enumerate(FLASH_TIME_BINS):
        ax = axs[bi]
        for (det_name, res), color in zip(flash_results.items(), colors):
            hv_arr  = np.array(res['hv'])
            hpe_arr = np.array(res['hits_per_event'][bi])
            valid   = ~np.isnan(hpe_arr)
            if not valid.any():
                continue
            order = np.argsort(hv_arr[valid])
            ax.plot(hv_arr[valid][order], hpe_arr[valid][order],
                    marker='o', color=color, label=det_name, ms=5, lw=1.5)

        ax.axhline(0, color='gray', lw=0.8, zorder=0)
        ax.set_ylabel('Hits / flash event')
        t_label = f'{t0 / 1000:.1f}–{t1 / 1000:.1f} μs'
        ax.text(0.02, 0.96, t_label, transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                          alpha=1.0, edgecolor='black'))
        ax.grid(True, alpha=0.3)
        if bi == 0:
            ax.legend(loc='upper right')

    axs[-1].set_xlabel('Resist HV [V]')
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(
        f'Gamma-flash: hits/event vs HV  —  {RUN_LABEL}{gas_s}\n'
        f'flash fraction ≥ {GAMMA_STRIP_FRACTION:.0%},  '
        f'detect window {FLASH_DETECT_WINDOW_NS/1000:.1f} μs,  '
        f'amp ≥ {AMP_THRESHOLD} ADC',
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, 'flash_hits_vs_hv.png')


def plot_thermal_efficiency_vs_hv(therm_results: Dict[str, dict], gas: str = '') -> None:
    """Track-finding efficiency vs resist HV, one curve per detector."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(9, 5))

    for (det_name, res), color in zip(therm_results.items(), colors):
        hv_arr  = np.array(res['hv'])
        eff_arr = np.array(res['eff'])
        err_arr = np.array(res['err'])
        valid   = ~np.isnan(eff_arr)
        if not valid.any():
            continue
        order = np.argsort(hv_arr[valid])
        ax.plot(hv_arr[valid][order], eff_arr[valid][order],
                '-', color=color, lw=1.5, zorder=1)
        ax.errorbar(hv_arr[valid], eff_arr[valid], yerr=err_arr[valid],
                    fmt='o', color=color, capsize=4, ms=6, elinewidth=1.5,
                    lw=0, zorder=2, label=det_name)

    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Track efficiency  (tracks / thermal events)')
    gas_s = f'  —  {gas}' if gas else ''
    ax.set_title(
        f'Thermal-track efficiency vs HV  —  {RUN_LABEL}{gas_s}\n'
        f'hits ∈ [{THERMAL_MIN_HITS}, {THERMAL_MAX_HITS}),  '
        f't ∈ [{THERMAL_T0_NS / 1000:.1f}, {THERMAL_T1_NS / 1000:.1f}] μs,  '
        f'amp ≥ {AMP_THRESHOLD} ADC'
    )
    all_eff = [e for res in therm_results.values()
               for e in res['eff'] if not np.isnan(e)]
    ymax = max(all_eff) * 1.25 if all_eff and max(all_eff) > 0 else 1.05
    ax.set_ylim(0, min(1.05, ymax))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'thermal_efficiency_vs_hv.png')


def plot_amplitude_vs_hv(therm_results: Dict[str, dict], gas: str = '') -> None:
    """Mean thermal-hit amplitude vs resist HV, one curve per detector."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(9, 5))

    for (det_name, res), color in zip(therm_results.items(), colors):
        hv_arr  = np.array(res['hv'])
        amp_arr = np.array(res['mean_amp'])
        valid   = ~np.isnan(amp_arr)
        if not valid.any():
            continue
        order = np.argsort(hv_arr[valid])
        ax.plot(hv_arr[valid][order], amp_arr[valid][order],
                marker='o', color=color, label=det_name)

    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Mean thermal hit amplitude [ADC]')
    gas_s = f'  —  {gas}' if gas else ''
    ax.set_title(
        f'Mean thermal hit amplitude vs HV  —  {RUN_LABEL}{gas_s}\n'
        f't ∈ [{THERMAL_T0_NS / 1000:.1f}, {THERMAL_T1_NS / 1000:.1f}] μs,  '
        f'amp ≥ {AMP_THRESHOLD} ADC'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'thermal_amplitude_vs_hv.png')


def plot_event_classification_diag(
    fractions: pd.Series,
    run: str,
    subrun: str,
    hv_per_det: Dict[str, Optional[int]],
) -> None:
    """Histogram of early-strip fractions showing the flash/thermal split."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fractions.values, bins=50, range=(0, 1),
            color='steelblue', edgecolor='none', log=True)
    ax.axvline(GAMMA_STRIP_FRACTION, color='red', lw=1.5, ls='--',
               label=f'Flash threshold = {GAMMA_STRIP_FRACTION:.0%}')

    n_flash = int((fractions >= GAMMA_STRIP_FRACTION).sum())
    n_therm = int((fractions <  GAMMA_STRIP_FRACTION).sum())
    ax.text(0.98, 0.95,
            f'Flash: {n_flash}  |  Thermal: {n_therm}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    ax.set_xlabel(f'Fraction of strips firing in first '
                  f'{FLASH_DETECT_WINDOW_NS / 1000:.1f} μs')
    ax.set_ylabel('Events')
    hv_str = ', '.join(f'{d}={v} V' for d, v in hv_per_det.items() if v)
    ax.set_title(f'Event classification  —  {run}/{subrun}  ({hv_str})')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, f'diag_event_classification_{run}_{subrun}.png')


def plot_hits_per_event_diag(
    df: pd.DataFrame,
    flash_ids: np.ndarray,
    thermal_ids: np.ndarray,
    det_feus: Dict[str, List[int]],
    run: str,
    subrun: str,
    hv_per_det: Dict[str, Optional[int]],
) -> None:
    """
    Hits-per-event distributions (flash vs thermal) for each detector,
    with the thermal track window shaded.
    """
    n = len(det_feus)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]
    df_amp = df[df['amplitude'] >= AMP_THRESHOLD]

    for ax, (det_name, feus) in zip(axes, det_feus.items()):
        df_d   = df_amp[df_amp['feu'].isin(feus)]
        df_t_w = df_d[df_d['time'].between(THERMAL_T0_NS, THERMAL_T1_NS)]

        counts_flash = df_d[df_d['eventId'].isin(flash_ids)].groupby('eventId').size()
        counts_therm = df_t_w[df_t_w['eventId'].isin(thermal_ids)].groupby('eventId').size()

        if counts_flash.empty and counts_therm.empty:
            ax.set_visible(False)
            continue

        max_count = max(
            int(counts_flash.max()) if not counts_flash.empty else 0,
            int(counts_therm.max()) if not counts_therm.empty else 0,
        )
        bins = np.arange(0.5, max_count + 1.5, 1)

        if not counts_flash.empty:
            ax.hist(counts_flash, bins=bins, alpha=0.6, color='tomato',
                    edgecolor='none', log=True, label='gamma flash')
        if not counts_therm.empty:
            ax.hist(counts_therm, bins=bins, alpha=0.6, color='steelblue',
                    edgecolor='none', log=True,
                    label=f'thermal (t ≤ {THERMAL_T1_NS/1000:.1f} μs)')

        ax.axvspan(THERMAL_MIN_HITS - 0.5, THERMAL_MAX_HITS - 0.5,
                   alpha=0.15, color='green',
                   label=f'Track window [{THERMAL_MIN_HITS}–{THERMAL_MAX_HITS-1}]')
        ax.axvline(THERMAL_MIN_HITS - 0.5, color='green', lw=1.2, ls='--')
        ax.axvline(THERMAL_MAX_HITS - 0.5, color='green', lw=1.2, ls='--')

        hv_label = f'{hv_per_det.get(det_name)} V' if hv_per_det.get(det_name) else '?'
        ax.set_title(f'{det_name}  —  HV = {hv_label},  amp ≥ {AMP_THRESHOLD}')
        ax.set_xlabel('Hits per event')
        ax.set_ylabel('Events')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Hits-per-event: flash vs thermal  —  {run}/{subrun}', fontsize=11)
    fig.tight_layout()
    _save(fig, f'diag_hits_per_event_{run}_{subrun}.png')


def plot_hit_time_distribution(
    df: pd.DataFrame,
    flash_ids: np.ndarray,
    thermal_ids: np.ndarray,
    det_feus: Dict[str, List[int]],
    run: str,
    subrun: str,
    hv_per_det: Dict[str, Optional[int]],
) -> None:
    """
    Hits/event vs time for flash and thermal events, one subplot per detector.
    Vertical dashed lines mark FLASH_DETECT_WINDOW_NS and THERMAL_T1_NS.
    """
    time_edges = np.arange(-500, 8001, 100)   # 100 ns bins, −0.5 µs to 8 µs
    bin_mids   = (time_edges[:-1] + time_edges[1:]) / 2

    n_det = len(det_feus)
    fig, axes = plt.subplots(1, n_det, figsize=(8 * n_det, 5), squeeze=False)
    axes = axes[0]

    df_amp = df[df['amplitude'] >= AMP_THRESHOLD]

    for ax, (det_name, feus) in zip(axes, det_feus.items()):
        df_d = df_amp[df_amp['feu'].isin(feus)]

        n_flash = len(flash_ids)
        n_therm = len(thermal_ids)

        if n_flash > 0:
            flash_hist, _ = np.histogram(
                df_d[df_d['eventId'].isin(flash_ids)]['time'].values,
                bins=time_edges)
            ax.step(bin_mids / 1000, flash_hist / n_flash, where='mid',
                    color='tomato', lw=1.5, label=f'Gamma flash  (n={n_flash})')

        if n_therm > 0:
            therm_hist, _ = np.histogram(
                df_d[df_d['eventId'].isin(thermal_ids)]['time'].values,
                bins=time_edges)
            ax.step(bin_mids / 1000, therm_hist / n_therm, where='mid',
                    color='steelblue', lw=1.5, label=f'Thermal  (n={n_therm})')

        ax.axvline(FLASH_DETECT_WINDOW_NS / 1000, color='tomato', ls='--', lw=1.2,
                   label=f'Flash window  {FLASH_DETECT_WINDOW_NS / 1000:.1f} µs')
        ax.axvline(THERMAL_T1_NS / 1000, color='steelblue', ls='--', lw=1.2,
                   label=f'Thermal window  {THERMAL_T1_NS / 1000:.1f} µs')

        ax.set_yscale('log')
        ax.set_xlabel('Hit time [µs]')
        ax.set_ylabel('Hits / event')
        hv_label = f'{hv_per_det.get(det_name)} V' if hv_per_det.get(det_name) else '?'
        ax.set_title(f'{det_name}  —  HV = {hv_label},  amp ≥ {AMP_THRESHOLD} ADC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Hit time distribution  —  {run}/{subrun}', fontsize=11)
    fig.tight_layout()
    _save(fig, f'diag_time_distribution_{run}_{subrun}.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    missing_runs: List[str] = []
    cfgs: Dict[str, dict] = {}
    for run in RUNS:
        try:
            cfgs[run] = load_config(BASE_PATH, run)
        except (FileNotFoundError, OSError):
            missing_runs.append(run)
            print(f'[WARN] Run not found, skipping: {run}')

    if not cfgs:
        print(f'ERROR: None of the requested runs exist: {RUNS}')
        return

    first_cfg = next(iter(cfgs.values()))
    gas      = first_cfg.get('gas', '')
    det_feus = build_detector_feu_map(first_cfg)
    hv_maps  = build_hv_maps(cfgs)
    subruns  = find_subruns(BASE_PATH, cfgs)

    if not det_feus:
        print(f'No detectors found in config for {MX17_DETECTORS}')
        return
    if not subruns:
        print(f'No subruns found on disk in {BASE_PATH} for runs {RUNS}')
        return

    all_feus = sorted({feu for feus in det_feus.values() for feu in feus})

    print('Detector → FEU mapping:')
    for det_name, feus in det_feus.items():
        print(f'  {det_name}: FEUs {feus}')
    print(f'\nFound {len(subruns)} subruns across {len(RUNS)} run(s)')

    # ---- Pre-compute total strips (denominator for flash fraction) ----------
    total_strips_all: Optional[int] = None
    total_strips_det: Dict[str, int] = {}
    for run, subrun in subruns:
        hits_dir = os.path.join(BASE_PATH, run, subrun, 'combined_hits_root')
        if os.path.isdir(hits_dir):
            total_strips_all, total_strips_det = compute_total_strips(
                BASE_PATH, run, subrun, all_feus, det_feus)
            print(f'\nTotal strips (from {run}/{subrun}):')
            for det_name, n in total_strips_det.items():
                print(f'  {det_name}: {n}')
            print(f'  Combined: {total_strips_all}')
            break

    if not total_strips_all:
        print('ERROR: Could not compute total strip count — no data found.')
        return

    # ---- Diagnostic plots (one subrun) -------------------------------------
    diag = DIAGNOSTIC_SUBRUN if DIAGNOSTIC_SUBRUN in subruns else subruns[0]
    diag_run, diag_subrun = diag
    print(f'\n{"=" * 60}')
    print(f'Diagnostic: {diag_run}/{diag_subrun}')
    print(f'{"=" * 60}')
    diag_df = load_hits(diag_run, diag_subrun, all_feus)
    if diag_df is not None:
        flash_ids, thermal_ids, fractions = classify_events(diag_df, total_strips_all)
        hv_per_det = {det_name: hv_maps[det_name].get(diag) for det_name in det_feus}
        print(f'  Flash: {len(flash_ids)}, Thermal (in data): {len(thermal_ids)}')
        plot_event_classification_diag(fractions, diag_run, diag_subrun, hv_per_det)
        plot_hit_time_distribution(diag_df, flash_ids, thermal_ids, det_feus,
                                    diag_run, diag_subrun, hv_per_det)
        plot_hits_per_event_diag(diag_df, flash_ids, thermal_ids, det_feus,
                                  diag_run, diag_subrun, hv_per_det)

    # ---- Result containers -------------------------------------------------
    flash_results: Dict[str, dict] = {
        name: {
            'hv': [],
            'hits_per_event': [[] for _ in FLASH_TIME_BINS],
            'n_flash': [],
        }
        for name in det_feus
    }
    therm_results: Dict[str, dict] = {
        name: {
            'hv': [], 'eff': [], 'err': [],
            'n_tracks': [], 'n_thermal': [], 'mean_amp': [],
        }
        for name in det_feus
    }

    # ---- Full HV scan loop -------------------------------------------------
    for run, subrun in subruns:
        print(f'\n{"=" * 60}')
        print(f'Subrun: {run}/{subrun}')
        print(f'{"=" * 60}')

        df_all = load_hits(run, subrun, all_feus)
        if df_all is None:
            continue

        n_total = get_total_events(run, subrun)
        if n_total is None:
            print('  [WARN] No decoded ROOT files; falling back to hit-event count')
            n_total = df_all['eventId'].nunique()

        flash_ids, thermal_ids, _ = classify_events(df_all, total_strips_all)
        n_thermal = n_total - len(flash_ids)   # includes zero-hit thermals
        print(f'  Flash: {len(flash_ids)},  Thermal: {n_thermal}  '
              f'(of {n_total} total triggered events)')

        hv_per_det = {det_name: hv_maps[det_name].get((run, subrun))
                      for det_name in det_feus}

        for det_name, feus in det_feus.items():
            hv = hv_per_det[det_name]
            if hv is None:
                print(f'  [SKIP] {det_name}: no HV found for {run}/{subrun}')
                continue

            # --- Gamma-flash analysis ---
            hpe = flash_hits_per_event(df_all, flash_ids, feus)
            flash_results[det_name]['hv'].append(hv)
            flash_results[det_name]['n_flash'].append(len(flash_ids))
            for bi in range(len(FLASH_TIME_BINS)):
                flash_results[det_name]['hits_per_event'][bi].append(hpe[bi])
            bin_str = '  '.join(f'bin{bi}={v:.2f}' for bi, v in enumerate(hpe))
            print(f'  {det_name} flash:  HV={hv} V  n_flash={len(flash_ids)}  '
                  f'{bin_str}')

            # --- Thermal-track analysis ---
            n_tracks, eff, err, mean_amp = thermal_efficiency(
                df_all, thermal_ids, feus, n_thermal)
            therm_results[det_name]['hv'].append(hv)
            therm_results[det_name]['eff'].append(eff)
            therm_results[det_name]['err'].append(err)
            therm_results[det_name]['n_tracks'].append(n_tracks)
            therm_results[det_name]['n_thermal'].append(n_thermal)
            therm_results[det_name]['mean_amp'].append(mean_amp)
            print(f'  {det_name} therm:  HV={hv} V  n_thermal={n_thermal}  '
                  f'n_tracks={n_tracks}  eff={eff:.4f}±{err:.4f}  '
                  f'mean_amp={mean_amp:.1f}')

    # ---- Summary -----------------------------------------------------------
    print(f'\n{"=" * 60}')
    for det_name in det_feus:
        res = therm_results[det_name]
        if not res['hv']:
            continue
        print(f'\n{det_name}:')
        print(f'  {"HV [V]":>8}  {"Efficiency":>12}  {"± Err":>8}  '
              f'{"Tracks":>8}  {"Thermals":>10}')
        for hv, eff, err, nt, ne in zip(res['hv'], res['eff'], res['err'],
                                         res['n_tracks'], res['n_thermal']):
            print(f'  {str(hv):>8}  {eff:>12.4f}  {err:>8.4f}  '
                  f'{nt:>8}  {ne:>10}')

    # ---- Save plots --------------------------------------------------------
    plot_flash_hits_vs_hv(flash_results, gas=gas)
    plot_thermal_efficiency_vs_hv(therm_results, gas=gas)
    plot_amplitude_vs_hv(therm_results, gas=gas)
    print(f'\nFigures saved to: {FIG_OUT_DIR}')

    if missing_runs:
        print('\n' + '=' * 60)
        print('WARNING: The following runs were not found:')
        for r in missing_runs:
            print(f'  {r}')
        print('=' * 60)

    plt.show()


if __name__ == '__main__':
    main()
