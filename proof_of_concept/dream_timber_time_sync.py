#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on March 18 4:27 PM 2026
Created in PyCharm
Created as nTof_x17/dream_timber_time_sync.py

@author: Dylan Neff, dylan
"""

import os
import pandas as pd
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from plot_beam_hits import get_run_time, get_run_start, load_subrun


def main():
    runs_path = '/media/dylan/data/x17/feb_beam/runs/'
    beam_data_dir = '/media/dylan/data/x17/feb_beam/ntof_bunch_intensities/'
    # run = 'run_88'
    # run = 'run_17'
    # run = 'run_18'
    run = 'run_19'
    feus = [4, 5]
    offset_range = np.array([0, 15])  # Seconds
    min_intensity = 0.1  # Minimum intensity in beam data for matching

    # Discover all subruns
    run_dir = os.path.join(runs_path, run)
    sub_runs = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))])
    print(f'Found {len(sub_runs)} subruns in {run}: {sub_runs}')

    # Filter out sub_runs without run_start metadata
    valid_sub_runs = []
    for sub in sub_runs:
        try:
            get_run_start(base_path=runs_path, run=run, sub_run=sub)
            valid_sub_runs.append(sub)
        except Exception as e:
            print(f'  Skipping {sub} (missing run_start): {e}')
    sub_runs = valid_sub_runs
    print(f'{len(sub_runs)} subruns with valid run_start metadata: {sub_runs}')

    # Load beam data once — covers the whole run
    print('\nLoading beam intensity data...')
    run_starts = [get_run_start(base_path=runs_path, run=run, sub_run=sub) for sub in sub_runs]

    # Get bracketing files for each subrun, flatten, and deduplicate by filepath
    all_bracketing = [f for run_start in run_starts
                      for f in get_bracketing_csv_files(beam_data_dir, run_start)]
    seen = set()
    bracketing_files = []
    for item in all_bracketing:
        if item[1] not in seen:
            seen.add(item[1])
            bracketing_files.append(item)
    bracketing_files.sort(key=lambda x: x[0])

    beam_df = load_beam_csvs(bracketing_files)
    print(f'Loaded {len(beam_df)} beam rows spanning '
          f'{beam_df["time_s"].min():.2f} – {beam_df["time_s"].max():.2f} s\n')

    # Iterate over subruns
    results = []
    for sub_run in sub_runs:
        print(f'--- {sub_run} ---')
        result = get_subrun_offset(
            runs_path=runs_path,
            run=run,
            sub_run=sub_run,
            beam_df=beam_df,
            feus=feus,
            coarse_range=tuple(offset_range * 1e6),
            min_intensity=min_intensity,
            plot=False,
        )
        if result is not None:
            results.append({'sub_run': sub_run, **result})
            print(f"  offset={result['offset']/1e6:.4f} s  "
                  f"mean_residual={result['mean_residual']:.1f} µs  "
                  f"matched={result['n_matched']}/{result['n_total']}")
        else:
            print(f'  Skipped (no triggers or error).')

    # Summary plot: offset per subrun
    if results:
        plot_offsets(results, run)

    print('\ndonzo')


def get_subrun_offset(
        runs_path: str,
        run: str,
        sub_run: str,
        beam_df: pd.DataFrame,
        feus: list,
        coarse_range: tuple = (-30_000_000, 30_000_000),
        coarse_steps: int = 2000,
        max_match_distance: float = 600_000,
        min_intensity: float = 0,
        plot: bool = False,
) -> dict | None:
    """
    Compute the DREAM→beam timestamp offset for a single subrun.

    Loads the subrun, extracts trigger timestamps, filters beam data to the
    subrun window, runs find_timestamp_offset, and optionally plots diagnostics.

    Parameters
    ----------
    runs_path         : root path containing run directories
    run               : run name (e.g. 'run_17')
    sub_run           : subrun name (e.g. 'resist_hv_460V')
    beam_df           : pre-loaded beam DataFrame (from load_beam_csvs)
    feus              : list of FEU IDs to load
    coarse_range      : (min, max) offset search range in µs
    coarse_steps      : number of points in coarse grid
    max_match_distance: match threshold in µs (half the ~1.2 s n_TOF cycle)
    min_intensity     : Minimum intensity in beam data to include in matching
    plot              : if True, show diagnostic plots for this subrun

    Returns
    -------
    dict from find_timestamp_offset, or None if the subrun cannot be processed.
    """
    try:
        run_start = get_run_start(base_path=runs_path, run=run, sub_run=sub_run)
        run_time  = get_run_time(base_path=runs_path, run=run, sub_run=sub_run)
    except Exception as e:
        print(f'  Could not read run metadata: {e}')
        return None

    try:
        df, det = load_subrun(runs_path, run, sub_run, feus=feus, map_csv_path='../mx17_m4_map.csv')
    except Exception as e:
        print(f'  Could not load subrun: {e}')
        return None

    if df.empty or 'trigger_timestamp_ns' not in df.columns:
        print(f'  No trigger data found.')
        return None

    print(df.columns)
    input('Enter any key to continue...')
    # Trigger timestamps in µs (absolute unix time)
    trigger_timestamps = (df['trigger_timestamp_ns'].unique() + run_start * 1e9) / 1e3

    # Hits-per-event series for optional plotting
    trigger_timestamps_df = df['trigger_timestamp_ns'].value_counts().sort_index()
    trigger_timestamps_df.index = trigger_timestamps_df.index / 1e9 + run_start

    # Restrict beam to subrun window (+/- coarse range margin) and significant pulses only
    min_trig = trigger_timestamps_df.index.min()
    max_trig = trigger_timestamps_df.index.max()
    beam_window = beam_df[
        (beam_df['time_s'] > min_trig - coarse_range[0] / 1e6) &
        (beam_df['time_s'] < max_trig + coarse_range[1] / 1e6) &
        (beam_df['intensity'] > min_intensity)
    ]
    beam_timestamps = beam_window['timestamp_us'].values

    if len(beam_timestamps) == 0:
        print(f'  No beam pulses found in subrun window.')
        return None

    if plot:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(trigger_timestamps_df.index, trigger_timestamps_df.values,
                marker='.', lw=0.6, label='Dream', color='blue')
        ax2.plot(beam_window['time_s'], beam_window['intensity'],
                 marker='.', lw=0.6, label='PS', color='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Dream Hits in Readout')
        ax2.set_ylabel('PS Bunch Intensity')
        ax.set_title(f'{run} / {sub_run} — raw (unaligned)')
        ax_ylim = ax.get_ylim()
        ax.set_ylim(top=ax_ylim[1] + (ax_ylim[1] - ax_ylim[0]) * 0.1)
        ax2_ylim = ax2.get_ylim()
        ax2.set_ylim(top=ax2_ylim[1] + ax2_ylim[0] * 0.1)
        fig.tight_layout()

    result = find_timestamp_offset(
        ts_A=trigger_timestamps,
        ts_B=beam_timestamps,
        coarse_range=coarse_range,
        coarse_steps=coarse_steps,
        max_match_distance=max_match_distance,
    )

    if plot:
        residuals = result["best_residuals"]
        sorted_tsa = result['best_shifted_tsa'] / 1e6
        match_mask = residuals > max_match_distance

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(trigger_timestamps_df.index + result["offset"] / 1e6, trigger_timestamps_df.values,
                marker='.', lw=0.6, label='Dream', color='blue')
        ax.scatter(sorted_tsa[match_mask], trigger_timestamps_df.values[match_mask], marker='x', color='red',
                s=20, zorder=20, label='Missed')
        ax2.plot(beam_window['time_s'], beam_window['intensity'],
                 marker='.', lw=0.6, label='PS', color='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Dream Hits in Readout')
        ax2.set_ylabel('PS Bunch Intensity')
        ax.set_title(f'{run} / {sub_run} — aligned')
        ax_ylim = ax.get_ylim()
        ax.set_ylim(top=ax_ylim[1] + (ax_ylim[1] - ax_ylim[0]) * 0.15)
        ax2_ylim = ax2.get_ylim()
        ax2.set_ylim(top=ax2_ylim[1] + ax2_ylim[0] * 0.15)
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(result['coarse_offsets'] / 1e6, result['coarse_costs'], marker='.', lw=0.6)
        ax.set_xlabel('Offset (s)')
        ax.set_ylabel('Mean residual (µs)')
        ax.set_title(f'{run} / {sub_run} — coarse offset search')
        ax.axvline(result['offset'] / 1e6, color='r', linestyle='--',
                   label=f"Best: {result['offset'] / 1e6:.3f} s")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return result


def plot_offsets(results: list[dict], run: str):
    """
    Plot best-fit offset (and match fraction) for each subrun.

    Parameters
    ----------
    results : list of dicts, each containing 'sub_run' and find_timestamp_offset output
    run     : run name, used in the plot title
    """
    sub_runs      = [r['sub_run'] for r in results]
    offsets_s     = [r['offset'] / 1e6 for r in results]
    match_frac    = [r['n_matched'] / r['n_total'] if r['n_total'] > 0 else 0 for r in results]

    x = np.arange(len(sub_runs))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x, offsets_s, marker='o')
    axes[0].set_ylabel('Best offset (s)')
    axes[0].set_title(f'{run} — DREAM→beam time offset per subrun')

    axes[1].plot(x, match_frac, color='steelblue', marker='o')
    axes[1].set_ylabel('Match fraction')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sub_runs, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.show()


def get_bracketing_csv_files(beam_data_dir, run_start_unix):
    """
    Find the three CSV files bracketing run_start_unix (in seconds).
    File timestamps are in microseconds. Files are daily (86400s apart).
    Returns sorted list of (timestamp_us, filepath) for the day before,
    day of, and day after the run start.
    """
    run_start_us = run_start_unix * 1e6  # convert to microseconds

    csv_files = []
    for fname in os.listdir(beam_data_dir):
        if fname.startswith('ntof_bunch_intensities_') and fname.endswith('.csv'):
            try:
                ts_us = int(fname.replace('ntof_bunch_intensities_', '').replace('.csv', ''))
                csv_files.append((ts_us, os.path.join(beam_data_dir, fname)))
            except ValueError:
                continue
    csv_files.sort(key=lambda x: x[0])

    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {beam_data_dir}')

    day_of_idx = None
    for i, (ts_us, _) in enumerate(csv_files):
        if ts_us <= run_start_us:
            day_of_idx = i
        else:
            break

    if day_of_idx is None:
        raise ValueError(f'Run start {run_start_unix:.2f}s is before all available CSV files.')

    indices = [day_of_idx - 1, day_of_idx, day_of_idx + 1]
    bracketing = []
    for idx in indices:
        if 0 <= idx < len(csv_files):
            bracketing.append(csv_files[idx])
        else:
            print(f'Warning: bracketing index {idx} out of range, skipping.')

    return bracketing


def load_beam_csvs(csv_file_list):
    """
    Load and concatenate a list of (timestamp_us, filepath) beam intensity CSVs.
    Skips the first header line (VARIABLE: ...) and reads Timestamp + Value columns.
    Returns a DataFrame sorted by timestamp with unix time in seconds as 'time_s'.
    """
    dfs = []
    for ts_us, fpath in csv_file_list:
        df = pd.read_csv(fpath, skiprows=1)
        df.columns = ['timestamp_us', 'intensity']
        df['time_s'] = df['timestamp_us'] / 1e6
        dfs.append(df)
        print(f'  Loaded: {os.path.basename(fpath)} ({len(df)} rows)')

    combined = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    return combined


def find_timestamp_offset(
        ts_A: np.ndarray,
        ts_B: np.ndarray,
        coarse_range: tuple,
        coarse_steps: int = 1000,
        fine_window: float = None,
        max_match_distance: float = None,
) -> dict:
    """
    Find the constant offset such that ts_A + offset aligns with ts_B.

    ts_B is assumed to be the complete reference set; ts_A may have missing
    or extra entries. Uses a coarse grid search followed by scipy scalar
    minimization, with mean residual as the cost function.

    Parameters
    ----------
    ts_A : array of timestamps to shift (e.g. your trigger timestamps, µs)
    ts_B : array of reference timestamps (e.g. beam bunch timestamps, µs)
    coarse_range : (min_offset, max_offset) to search, same units as timestamps
    coarse_steps : number of points in coarse grid
    fine_window : search window around coarse best for fine minimization;
                  defaults to 10x the coarse step size
    max_match_distance : points in ts_A with nearest-neighbor residual above
                         this are counted as unmatched. If None, uses
                         3x the median residual at the best offset.

    Returns
    -------
    dict with keys:
        offset          – best-fit offset to add to ts_A
        mean_residual   – mean residual of matched points
        n_matched       – number of ts_A points matched within max_match_distance
        n_total         – total number of ts_A points
        coarse_offsets  – offsets evaluated in coarse search
        coarse_costs    – mean residuals at each coarse offset
    """
    ts_A = np.sort(np.asarray(ts_A, dtype=np.float64))
    ts_B = np.sort(np.asarray(ts_B, dtype=np.float64))

    def nearest_residuals(offset):
        shifted = ts_A + offset
        idx = np.searchsorted(ts_B, shifted, side='left')
        idx = np.clip(idx, 1, len(ts_B) - 1)
        diff_left  = np.abs(shifted - ts_B[idx - 1])
        diff_right = np.abs(shifted - ts_B[idx])
        return np.minimum(diff_left, diff_right)

    def cost(offset):
        return np.mean(nearest_residuals(offset))

    # --- Coarse grid search ---
    coarse_offsets = np.linspace(coarse_range[0], coarse_range[1], coarse_steps)
    coarse_costs   = np.array([cost(o) for o in coarse_offsets])

    best_coarse = coarse_offsets[np.argmin(coarse_costs)]
    step = (coarse_range[1] - coarse_range[0]) / coarse_steps
    fw   = fine_window if fine_window is not None else step * 10

    # --- Fine minimization ---
    result = minimize_scalar(
        cost,
        bounds=(best_coarse - fw, best_coarse + fw),
        method='bounded',
        options={'xatol': 1.0},  # 1 µs precision
    )
    best_offset = result.x

    # --- Match stats at best offset ---
    residuals = nearest_residuals(best_offset)
    threshold = max_match_distance if max_match_distance is not None \
                else 3 * np.median(residuals)
    mask = residuals < threshold

    return {
        "offset":           best_offset,
        "mean_residual":    np.mean(residuals[mask]),
        "n_matched":        int(mask.sum()),
        "n_total":          len(ts_A),
        "coarse_offsets":   coarse_offsets,
        "coarse_costs":     coarse_costs,
        "best_residuals":   residuals,
        "best_shifted_tsa": ts_A + best_offset,
    }


if __name__ == '__main__':
    main()
