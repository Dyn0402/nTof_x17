#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hv_scan_efficiency_no_ref.py

HV-scan efficiency analysis without M3 reference tracks.

For each subrun the script:
  1. Loads hits from combined_hits_root for the FEUs belonging to each detector.
  2. Counts hits per event above AMP_THRESHOLD, per detector.
  3. Labels an event as a "track" if hit count falls in (MIN_HITS_TRACK, MAX_HITS_TRACK).
  4. Computes efficiency = n_track_events / n_total_events, per detector.
  5. Plots efficiency and mean amplitude vs. resist HV, one curve per detector.

Multiple runs can be combined by listing them in RUNS — their subruns are pooled
and treated as a single scan (no distinction between runs in the output).

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

from common.Mx17StripMap import Detector, Mx17StripMap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = '/mnt/data/x17/beam_may/runs/'

# List of runs to combine into a single scan.  Set to a single entry for one run.
# RUNS = ['run_3', 'run_4']
RUNS = ['run_5', 'run_6', 'run_7']

# Label used for output directory and plot titles.
RUN_LABEL = RUNS[0] if len(RUNS) == 1 else f'{RUNS[0]}+{RUNS[-1]}'

# Detector names to analyse — FEU IDs are read from run_config.json automatically.
MX17_DETECTORS = ['mx17_3', 'mx17_4']

MAP_CSV_PATH = f'{_ROOT}/mx17_m1_map.csv'

FIG_OUT_DIR = f'{BASE_PATH}Analysis/HV_Scan_NoRef/{RUN_LABEL}/'
CSV_OUT_DIR = f'{BASE_PATH}Analysis/HV_Scan_NoRef/'

# Subrun used for the diagnostic plots, as a (run, subrun_name) tuple.
# Set to None to use the first subrun found.
DIAGNOSTIC_SUBRUN: Optional[Tuple[str, str]] = None

# Hit selection
AMP_THRESHOLD  = 200   # ADC counts; hits below this value are ignored
MIN_HITS_TRACK = 1     # exclusive lower bound — event must have MORE than this many hits
MAX_HITS_TRACK = 200   # exclusive upper bound — event must have FEWER than this many hits


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(base_path: str, run: str) -> dict:
    with open(os.path.join(base_path, run, 'run_config.json')) as f:
        return json.load(f)


def build_detector_feu_map(cfg: dict) -> Dict[str, List[int]]:
    """
    Use Detector from Mx17StripMap to look up the FEU IDs for each name in
    MX17_DETECTORS.  Returns {det_name: sorted list of FEU IDs}.
    Detectors not found in the config are silently skipped.
    """
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
    """
    Build {det_name: {(run, subrun_name): resist_hv}} by reading hv_channels.resist
    from each detector config and looking it up in every subrun's hvs dict, across
    all runs.  cfgs is {run_name: cfg_dict}.
    """
    # Use first run's config to find resist channels (assumed same hardware across runs)
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
    """
    Return (run, subrun_name) pairs that exist on disk, in config order across
    all runs.
    """
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
    """Load hits for one subrun and filter to the given FEU IDs."""
    hits_dir = os.path.join(BASE_PATH, run, subrun, 'combined_hits_root')
    if not os.path.isdir(hits_dir):
        print(f'  [SKIP] {run}/{subrun}: no combined_hits_root directory')
        return None
    hit_files = sorted(f for f in os.listdir(hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    if not hit_files:
        print(f'  [SKIP] {run}/{subrun}: no hit files found')
        return None
    file_sources = [f'{hits_dir}/{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(feu_ids)].copy()
    print(f'  Loaded {len(df):,} hits over {df["eventId"].nunique():,} events '
          f'(FEUs {feu_ids})')
    return df


def get_total_events(run: str, subrun: str) -> Optional[int]:
    """
    Return the total number of triggered events by finding the maximum eventId
    across all decoded ROOT files (tree 'nt') for this subrun.
    Returns None if no decoded files are found.
    """
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
            evt_ids = f['nt']['eventId'].array(library='np')
            if len(evt_ids) > 0:
                max_id = max(max_id, int(evt_ids.max()))
    return max_id


# ---------------------------------------------------------------------------
# Hit counting and track definition
# ---------------------------------------------------------------------------

def hits_per_event(df: pd.DataFrame) -> pd.Series:
    """
    Count hits per event above AMP_THRESHOLD.
    Returns a Series indexed by eventId.
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
    counts_per_det: Dict[str, pd.Series],
    run: str,
    subrun: str,
    hv_per_det: Dict[str, Optional[int]],
    out_dir: Optional[str] = None,
) -> None:
    """Histogram of hits per event for each detector, with the track window shaded."""
    n = len(counts_per_det)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (det_name, counts) in zip(axes, counts_per_det.items()):
        if counts.empty:
            ax.set_visible(False)
            continue
        max_count = int(counts.max())
        bins = np.arange(0.5, max_count + 1.5, 1)
        n_tracks = int(is_track(counts).sum())
        n_total  = len(counts)
        hv_label = f'{hv_per_det.get(det_name)} V' if hv_per_det.get(det_name) else '?'

        ax.hist(counts, bins=bins, color='steelblue', edgecolor='none', log=True)
        ax.axvspan(MIN_HITS_TRACK + 0.5, MAX_HITS_TRACK - 0.5,
                   alpha=0.18, color='green',
                   label=f'Track window ({MIN_HITS_TRACK+1}–{MAX_HITS_TRACK-1})  '
                         f'→  {n_tracks}/{n_total}  ({100*n_tracks/n_total:.1f}%)')
        ax.axvline(MIN_HITS_TRACK + 0.5, color='green', lw=1.2, ls='--')
        ax.axvline(MAX_HITS_TRACK - 0.5, color='green', lw=1.2, ls='--')
        ax.set_xlabel('Hits per event')
        ax.set_ylabel('Events')
        ax.set_title(f'{det_name}  —  HV = {hv_label},  amp ≥ {AMP_THRESHOLD}')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Hits per event distribution  —  {run}/{subrun}', fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, f'hits_per_event_{run}_{subrun}.png')


def plot_strip_occupancy(
    df_per_det: Dict[str, pd.DataFrame],
    run: str,
    subrun: str,
    out_dir: Optional[str] = None,
) -> None:
    """Strip occupancy: hits per channel number, one subplot per FEU across all detectors."""
    all_feus = sorted({feu for df in df_per_det.values() for feu in df['feu'].unique()})
    n = len(all_feus)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    feu_to_det = {feu: det_name for det_name, df in df_per_det.items()
                  for feu in df['feu'].unique()}
    df_all = pd.concat(df_per_det.values(), ignore_index=True)
    for ax, feu in zip(axes, all_feus):
        channels = df_all.loc[df_all['feu'] == feu, 'channel'].values
        lo, hi = int(channels.min()), int(channels.max())
        ax.hist(channels, bins=hi - lo + 1, range=(lo - 0.5, hi + 0.5),
                color='steelblue', edgecolor='none')
        ax.set_xlabel('Channel number')
        ax.set_ylabel('Hits')
        ax.set_title(f'FEU {feu}  ({feu_to_det.get(feu, "?")})')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Strip occupancy  —  {run}/{subrun}', fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, f'strip_occupancy_{run}_{subrun}.png')


_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']


def _run_marker_map() -> Dict[str, str]:
    return {run: _MARKERS[i % len(_MARKERS)] for i, run in enumerate(RUNS)}


def _build_legend(ax, det_results: Dict[str, dict], colors: list) -> None:
    """Two-section legend: detector colours on the left, run markers on the right."""
    from matplotlib.lines import Line2D
    run_marker = _run_marker_map()
    active_runs = sorted({r for res in det_results.values() for r in res['runs']},
                         key=RUNS.index)

    det_handles = [Line2D([0], [0], color=c, lw=2, label=name)
                   for (name, _), c in zip(det_results.items(), colors)
                   if det_results[name]['hv_values']]
    run_handles = [Line2D([0], [0], marker=run_marker[r], color='k',
                          lw=0, ms=8, label=r)
                   for r in active_runs]
    ax.legend(handles=det_handles + run_handles)


def plot_efficiency_vs_hv(
    det_results: Dict[str, dict],
    gas: str = '',
    out_dir: Optional[str] = None,
) -> None:
    """Track-finding efficiency vs. resist HV. Color = detector, marker = run."""
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    run_marker = _run_marker_map()
    fig, ax    = plt.subplots(figsize=(9, 5))

    for (det_name, res), color in zip(det_results.items(), colors):
        hv_arr  = np.array(res['hv_values'])
        eff_arr = np.array(res['efficiencies'])
        err_arr = np.array(res['eff_errors'])
        run_arr = np.array(res['runs'])
        if len(hv_arr) == 0:
            continue
        # connecting line across all points
        order = np.argsort(hv_arr)
        ax.plot(hv_arr[order], eff_arr[order], '-', color=color, lw=1.5, zorder=1)
        # markers per run
        for run in RUNS:
            mask = run_arr == run
            if not mask.any():
                continue
            ax.errorbar(hv_arr[mask], eff_arr[mask], yerr=err_arr[mask],
                        fmt=run_marker[run], color=color,
                        capsize=4, ms=8, elinewidth=1.5, lw=0, zorder=2)

    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Track efficiency  (tracks / triggered events)')
    gas_str = f'  —  {gas}' if gas else ''
    ax.set_title(
        f'Track efficiency vs. HV  —  {RUN_LABEL}{gas_str}\n'
        f'hit window: ({MIN_HITS_TRACK}, {MAX_HITS_TRACK}),  amp ≥ {AMP_THRESHOLD}'
    )
    all_effs = [e for res in det_results.values() for e in res['efficiencies']]
    ymax = max(all_effs) * 1.25 if all_effs and max(all_effs) > 0 else 1.05
    ax.set_ylim(0, min(1.05, ymax))
    _build_legend(ax, det_results, colors)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'efficiency_vs_hv.png')


def plot_amplitude_vs_hv(
    det_results: Dict[str, dict],
    gas: str = '',
    out_dir: Optional[str] = None,
) -> None:
    """Mean hit amplitude vs. resist HV. Color = detector, marker = run."""
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    run_marker = _run_marker_map()
    fig, ax    = plt.subplots(figsize=(9, 5))

    for (det_name, res), color in zip(det_results.items(), colors):
        hv_arr  = np.array(res['hv_values'])
        amp_arr = np.array(res['mean_amp'])
        run_arr = np.array(res['runs'])
        if len(hv_arr) == 0:
            continue
        order = np.argsort(hv_arr)
        ax.plot(hv_arr[order], amp_arr[order], '-', color=color, lw=1.5, zorder=1)
        for run in RUNS:
            mask = run_arr == run
            if not mask.any():
                continue
            ax.plot(hv_arr[mask], amp_arr[mask],
                    run_marker[run], color=color, ms=8, zorder=2)

    ax.set_xlabel('Resist HV [V]')
    ax.set_ylabel('Mean hit amplitude [ADC]')
    gas_str = f'  —  {gas}' if gas else ''
    ax.set_title(
        f'Mean hit amplitude vs. HV  —  {RUN_LABEL}{gas_str}\n'
        f'amp ≥ {AMP_THRESHOLD}'
    )
    _build_legend(ax, det_results, colors)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'amplitude_vs_hv.png')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_summary_csv(det_results: Dict[str, dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for det_name, res in det_results.items():
        df = pd.DataFrame({
            'hv_v':       res['hv_values'],
            'efficiency': res['efficiencies'],
            'eff_err':    res['eff_errors'],
            'n_tracks':   res['n_tracks'],
            'n_events':   res['n_events'],
            'mean_amp':   res['mean_amp'],
        })
        path = os.path.join(out_dir, f'efficiency_vs_hv_{RUN_LABEL}_{det_name}.csv')
        df.to_csv(path, index=False)
        print(f'  Saved → {path}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Optional[str], name: str, dpi: int = 150) -> None:
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load all configs
    cfgs = {run: load_config(BASE_PATH, run) for run in RUNS}

    # Use first run's config for gas label and detector FEU mapping
    first_cfg = next(iter(cfgs.values()))
    gas      = first_cfg.get('gas', '')
    det_feus = build_detector_feu_map(first_cfg)
    hv_maps  = build_hv_maps(cfgs)
    subruns  = find_subruns(BASE_PATH, cfgs)

    if not det_feus:
        print(f'No detectors found in config for {MX17_DETECTORS}')
        return
    print('Detector → FEU mapping:')
    for det_name, feus in det_feus.items():
        print(f'  {det_name}: FEUs {feus}')

    if not subruns:
        print(f'No subruns found on disk in {BASE_PATH} for runs {RUNS}')
        return
    print(f'\nFound {len(subruns)} subruns across {len(RUNS)} run(s)')

    all_feus = sorted({feu for feus in det_feus.values() for feu in feus})

    # ---- Diagnostic ----
    diag = DIAGNOSTIC_SUBRUN if DIAGNOSTIC_SUBRUN in subruns else subruns[0]
    diag_run, diag_subrun = diag
    print(f'\n{"="*60}')
    print(f'Diagnostic: {diag_run}/{diag_subrun}')
    print(f'{"="*60}')
    diag_df = load_hits(diag_run, diag_subrun, all_feus)
    if diag_df is not None:
        counts_per_det = {
            det_name: hits_per_event(diag_df[diag_df['feu'].isin(feus)])
            for det_name, feus in det_feus.items()
        }
        hv_per_det = {det_name: hv_maps[det_name].get(diag)
                      for det_name in det_feus}
        plot_hits_per_event_dist(counts_per_det, diag_run, diag_subrun,
                                 hv_per_det, out_dir=FIG_OUT_DIR)
        df_per_det = {det_name: diag_df[diag_df['feu'].isin(feus)]
                      for det_name, feus in det_feus.items()}
        plot_strip_occupancy(df_per_det, diag_run, diag_subrun, out_dir=FIG_OUT_DIR)

    # ---- Full HV scan ----
    det_results = {
        name: {'hv_values': [], 'efficiencies': [], 'eff_errors': [],
               'n_tracks': [], 'n_events': [], 'mean_amp': [], 'runs': []}
        for name in det_feus
    }

    for run, subrun in subruns:
        print(f'\n{"="*60}')
        print(f'Subrun: {run}/{subrun}')
        print(f'{"="*60}')

        df_all = load_hits(run, subrun, all_feus)
        if df_all is None:
            continue

        n_total = get_total_events(run, subrun)
        if n_total is None:
            print('  [WARN] No decoded ROOT files found; falling back to hit-event count')

        for det_name, feus in det_feus.items():
            hv = hv_maps[det_name].get((run, subrun))
            df  = df_all[df_all['feu'].isin(feus)]
            counts   = hits_per_event(df)
            df_amp   = df[df['amplitude'] >= AMP_THRESHOLD] if AMP_THRESHOLD > 0 else df
            mean_amp = float(df_amp['amplitude'].mean()) if not df_amp.empty else np.nan
            n_events = n_total if n_total is not None else len(counts)
            n_tracks = int(is_track(counts).sum())
            eff = n_tracks / n_events if n_events > 0 else np.nan
            err = float(np.sqrt(eff * (1 - eff) / n_events)) if n_events > 0 else np.nan
            print(f'  {det_name}  HV={hv} V  n_total={n_events:,}  '
                  f'n_tracks={n_tracks:,}  eff={eff:.4f} ± {err:.4f}  '
                  f'mean_amp={mean_amp:.1f}')
            det_results[det_name]['hv_values'].append(hv)
            det_results[det_name]['efficiencies'].append(eff)
            det_results[det_name]['eff_errors'].append(err)
            det_results[det_name]['n_tracks'].append(n_tracks)
            det_results[det_name]['n_events'].append(n_events)
            det_results[det_name]['mean_amp'].append(mean_amp)
            det_results[det_name]['runs'].append(run)

    # ---- Summary ----
    for det_name, res in det_results.items():
        if not res['hv_values']:
            continue
        print(f'\n{det_name}:')
        print(f'  {"HV [V]":>8}  {"Efficiency":>12}  {"± Err":>8}  '
              f'{"Tracks":>8}  {"Events":>8}')
        for hv, eff, err, nt, ne in zip(res['hv_values'], res['efficiencies'],
                                        res['eff_errors'], res['n_tracks'],
                                        res['n_events']):
            print(f'  {str(hv):>8}  {eff:>12.4f}  {err:>8.4f}  {nt:>8}  {ne:>8}')

    plot_efficiency_vs_hv(det_results, gas=gas, out_dir=FIG_OUT_DIR)
    plot_amplitude_vs_hv(det_results, gas=gas, out_dir=FIG_OUT_DIR)
    print('\nSaving CSVs:')
    save_summary_csv(det_results, CSV_OUT_DIR)

    plt.show()


if __name__ == '__main__':
    main()
