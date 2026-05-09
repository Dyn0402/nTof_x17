#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hv_scan_efficiency_no_ref.py

HV-scan efficiency analysis without M3 reference tracks.

For each subrun the script:
  1. Loads MX17 detector hits from combined_hits_root.
  2. Counts hits per event above AMP_THRESHOLD.
  3. Labels an event as a "track" if hit count falls in (MIN_HITS_TRACK, MAX_HITS_TRACK).
  4. Computes efficiency = n_track_events / n_events_with_hits.
  5. Plots efficiency vs. resist HV.

A diagnostic hits-per-event distribution and strip occupancy plot are produced
for one subrun (DIAGNOSTIC_SUBRUN) to help tune MIN_HITS_TRACK / MAX_HITS_TRACK.
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = '/mnt/data/x17/beam_may/runs/'
RUN       = 'run_1'
MX17_FEUS = [3]

FIG_OUT_DIR = f'{BASE_PATH}Analysis/HV_Scan_NoRef/{RUN}/'
CSV_OUT_DIR = f'{BASE_PATH}Analysis/HV_Scan_NoRef/'

# Subrun used for the diagnostic plots.
# Set to None to use the highest-HV subrun (first in sorted order).
DIAGNOSTIC_SUBRUN: Optional[str] = None

# Hit selection
AMP_THRESHOLD  = 0     # ADC counts; hits below this value are ignored
MIN_HITS_TRACK = 2     # exclusive lower bound — event must have MORE than this many hits
MAX_HITS_TRACK = 50    # exclusive upper bound — event must have FEWER than this many hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Optional[str], name: str, dpi: int = 150) -> None:
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=dpi, bbox_inches='tight')


def _resist_hv_channel(cfg: dict) -> Optional[Tuple[int, int]]:
    """Return (card, channel) for the resist HV of the active mx17 detector."""
    included = set(cfg.get('included_detectors', []))
    for det in cfg.get('detectors', []):
        if included and det['name'] not in included:
            continue
        if det.get('det_type') != 'mx17':
            continue
        if 'resist' in det.get('hv_channels', {}):
            card, ch = det['hv_channels']['resist']
            return int(card), int(ch)
    return None


def _build_hv_map(cfg: dict) -> Dict[str, int]:
    """Map subrun name → resist HV voltage from run_config sub_runs list."""
    resist = _resist_hv_channel(cfg)
    if resist is None:
        return {}
    card, ch = resist
    hv_map: Dict[str, int] = {}
    for sub in cfg.get('sub_runs', []):
        name = sub['sub_run_name']
        hv = sub.get('hvs', {}).get(str(card), {}).get(str(ch))
        if hv is not None:
            hv_map[name] = int(hv)
    return hv_map


def find_subruns(base_path: str, run: str) -> List[Tuple[str, int]]:
    """
    Read run_config.json and return (subrun_name, resist_hv_volts) pairs for
    all subruns that have an on-disk directory, sorted by HV descending.
    """
    run_dir = os.path.join(base_path, run)
    with open(os.path.join(run_dir, 'run_config.json')) as f:
        cfg = json.load(f)

    hv_map = _build_hv_map(cfg)
    pairs = []
    for name in sorted(os.listdir(run_dir)):
        if not os.path.isdir(os.path.join(run_dir, name)):
            continue
        if name in hv_map:
            pairs.append((name, hv_map[name]))
    return sorted(pairs, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hits(subrun: str) -> Optional[pd.DataFrame]:
    """Load hits for one subrun and filter to MX17 FEUs."""
    hits_dir = f'{BASE_PATH}{RUN}/{subrun}/combined_hits_root/'
    if not os.path.isdir(hits_dir):
        print(f'  [SKIP] {subrun}: no combined_hits_root directory')
        return None
    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    if not hit_files:
        print(f'  [SKIP] {subrun}: no hit files found')
        return None
    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(MX17_FEUS)].copy()
    print(f'  Loaded {len(df):,} hits over {df["eventId"].nunique():,} events')
    return df


# ---------------------------------------------------------------------------
# Hit counting and track definition
# ---------------------------------------------------------------------------

def hits_per_event(df: pd.DataFrame) -> pd.Series:
    """
    Count hits per event above AMP_THRESHOLD.
    Returns a Series indexed by eventId. Events with zero qualifying hits are
    absent (they don't appear in the ROOT data after FEU filtering).
    """
    if AMP_THRESHOLD > 0:
        df = df[df['amplitude'] >= AMP_THRESHOLD]
    return df.groupby('eventId').size()


def is_track(counts: pd.Series) -> pd.Series:
    """Boolean mask: True when MIN_HITS_TRACK < count < MAX_HITS_TRACK."""
    return (counts > MIN_HITS_TRACK) & (counts < MAX_HITS_TRACK)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hits_per_event_dist(
    counts: pd.Series,
    subrun: str,
    hv: int,
    out_dir: Optional[str] = None,
) -> None:
    """Histogram of hits per event with the track window shaded."""
    max_count = int(counts.max())
    bins = np.arange(0.5, max_count + 1.5, 1)

    n_tracks = int(is_track(counts).sum())
    n_total  = len(counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=bins, color='steelblue', edgecolor='none', log=True)
    ax.axvspan(MIN_HITS_TRACK + 0.5, MAX_HITS_TRACK - 0.5,
               alpha=0.18, color='green',
               label=f'Track window ({MIN_HITS_TRACK+1}–{MAX_HITS_TRACK-1} hits)  '
                     f'→  {n_tracks}/{n_total} events  ({100*n_tracks/n_total:.1f}%)')
    ax.axvline(MIN_HITS_TRACK + 0.5, color='green', lw=1.2, ls='--')
    ax.axvline(MAX_HITS_TRACK - 0.5, color='green', lw=1.2, ls='--')
    ax.set_xlabel('Hits per event')
    ax.set_ylabel('Events')
    ax.set_title(
        f'Hits per event distribution  —  {subrun}\n'
        f'HV = {hv} V,  amp ≥ {AMP_THRESHOLD}'
    )
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, f'hits_per_event_{hv}V.png')


def plot_strip_occupancy(
    df: pd.DataFrame,
    subrun: str,
    hv: int,
    out_dir: Optional[str] = None,
) -> None:
    """Strip occupancy: hits per channel number, one subplot per FEU."""
    feus = sorted(df['feu'].unique())
    n = len(feus)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
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

    fig.suptitle(
        f'Strip occupancy  —  {subrun}  (HV = {hv} V)',
        fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, f'strip_occupancy_{hv}V.png')


def plot_efficiency_vs_hv(
    hv_values: List[int],
    efficiencies: List[float],
    eff_errors: List[float],
    out_dir: Optional[str] = None,
) -> None:
    """Track-finding efficiency vs. resist HV with binomial error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(hv_values, efficiencies, yerr=eff_errors,
                fmt='o-', color='steelblue', capsize=5, lw=2, ms=8)
    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Track efficiency  (tracks / triggered events)')
    ax.set_title(
        f'Track efficiency vs. HV  —  {RUN}\n'
        f'hit window: ({MIN_HITS_TRACK}, {MAX_HITS_TRACK}),  '
        f'amp ≥ {AMP_THRESHOLD}'
    )
    ymax = max(efficiencies) * 1.25 if efficiencies and max(efficiencies) > 0 else 1.05
    ax.set_ylim(0, min(1.05, ymax))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'efficiency_vs_hv.png')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_summary_csv(
    hv_values: List[int],
    efficiencies: List[float],
    eff_errors: List[float],
    n_tracks_list: List[int],
    n_events_list: List[int],
    out_dir: str,
) -> None:
    df = pd.DataFrame({
        'hv_v':       hv_values,
        'efficiency': efficiencies,
        'eff_err':    eff_errors,
        'n_tracks':   n_tracks_list,
        'n_events':   n_events_list,
    })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'efficiency_vs_hv.csv')
    df.to_csv(path, index=False)
    print(f'\nSummary saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    subruns = find_subruns(BASE_PATH, RUN)
    if not subruns:
        print(f'No HV-scan subruns found in {BASE_PATH}{RUN}/')
        return
    print(f'Found {len(subruns)} subruns:')
    for name, hv in subruns:
        print(f'  {name}  ({hv} V)')

    # ---- Diagnostic: hits-per-event distribution and occupancy for one subrun ----
    diag_name, diag_hv = next(
        ((n, hv) for n, hv in subruns if n == DIAGNOSTIC_SUBRUN),
        subruns[0],
    ) if DIAGNOSTIC_SUBRUN else subruns[0]
    print(f'\n{"="*60}')
    print(f'Diagnostic subrun: {diag_name}  (HV = {diag_hv} V)')
    print(f'{"="*60}')
    diag_df = load_hits(diag_name)
    if diag_df is not None:
        diag_counts = hits_per_event(diag_df)
        plot_hits_per_event_dist(diag_counts, diag_name, diag_hv, out_dir=FIG_OUT_DIR)
        plot_strip_occupancy(diag_df, diag_name, diag_hv, out_dir=FIG_OUT_DIR)

    # ---- Full HV scan ----
    hv_values, efficiencies, eff_errors = [], [], []
    n_tracks_list, n_events_list = [], []

    for subrun, hv in subruns:
        print(f'\n{"="*60}')
        print(f'Subrun: {subrun}  (HV = {hv} V)')
        print(f'{"="*60}')

        df = load_hits(subrun)
        if df is None:
            continue

        counts   = hits_per_event(df)
        n_events = len(counts)
        n_tracks = int(is_track(counts).sum())
        eff = n_tracks / n_events if n_events > 0 else np.nan
        err = float(np.sqrt(eff * (1 - eff) / n_events)) if n_events > 0 else np.nan

        print(f'  n_events={n_events:,}  n_tracks={n_tracks:,}  '
              f'efficiency={eff:.4f} ± {err:.4f}')

        hv_values.append(hv)
        efficiencies.append(eff)
        eff_errors.append(err)
        n_tracks_list.append(n_tracks)
        n_events_list.append(n_events)

    if not hv_values:
        print('\nNo valid subruns processed.')
        return

    # ---- Summary table ----
    print(f'\n{"HV [V]":>8}  {"Efficiency":>12}  {"± Err":>8}  '
          f'{"Tracks":>8}  {"Events":>8}')
    for hv, eff, err, nt, ne in zip(hv_values, efficiencies, eff_errors,
                                    n_tracks_list, n_events_list):
        print(f'{hv:>8}  {eff:>12.4f}  {err:>8.4f}  {nt:>8}  {ne:>8}')

    plot_efficiency_vs_hv(hv_values, efficiencies, eff_errors, out_dir=FIG_OUT_DIR)
    save_summary_csv(hv_values, efficiencies, eff_errors,
                     n_tracks_list, n_events_list, CSV_OUT_DIR)

    plt.show()


if __name__ == '__main__':
    main()
