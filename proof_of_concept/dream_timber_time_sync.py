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
import matplotlib.cm as cm
from scipy.optimize import minimize_scalar
from plot_beam_hits import get_run_start, get_run_time, load_subrun, get_run_json


def main():
    runs_path = '/media/dylan/data/x17/feb_beam/runs/'
    beam_data_dir = '/media/dylan/data/x17/feb_beam/ntof_bunch_intensities/'
    feus = [4, 5]
    offset_range = np.array([0, 5])  # Seconds
    min_intensity = 0.1  # Minimum intensity in beam data for matching

    process_all_runs(runs_path, beam_data_dir, feus, offset_range, min_intensity)

    # run = 'run_30'
    # plot_outliers_range = (1.5, 2.5)
    # plot_single_run(runs_path, beam_data_dir, feus, offset_range, min_intensity, run, plot_outliers_range)

    print('\ndonzo')


def plot_single_run(runs_path, beam_data_dir, feus, offset_range, min_intensity, run, plot_outliers_range=None):
    """
    Plot a single run
    """
    process_run(
        runs_path=runs_path,
        beam_data_dir=beam_data_dir,
        run=run,
        feus=feus,
        offset_range=offset_range,
        min_intensity=min_intensity,
        plot=False,
        plot_outliers_range=plot_outliers_range,
        write_csv=False,
    )
    plt.show()



def process_all_runs(runs_path, beam_data_dir, feus, offset_range, min_intensity):
    """
    Do matching between DREAM and PS timestamps for all runs.
    """
    # Discover all runs
    all_runs = sorted([
        d for d in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, d)) and d.startswith('run_') and not d.endswith('_testing')
    ], key=lambda r: int(r.split('_')[1]))

    good_runs = []
    for run in all_runs:
        if run == 'run_52' or run == 'run_54':
            continue
        json = get_run_json(runs_path, run)
        print(run)
        dream_config_file = os.path.basename(json['dream_daq_info']['daq_config_template_path'])
        print(dream_config_file)
        if dream_config_file.startswith('Tcm_Mx17_') and dream_config_file != 'Tcm_Mx17_SiPM_trig.cfg':
            good_runs.append(run)

    all_runs = good_runs

    print(f'Found {len(all_runs)} runs: {all_runs}')

    # Process each run, accumulating results tagged by run name
    all_results = []  # list of dicts with keys: run, sub_run, run_start, offset, ...
    for run in all_runs:
        print(f'\n{"=" * 60}')
        print(f'Processing {run}')
        print(f'{"=" * 60}')
        run_results = process_run(
            runs_path=runs_path,
            beam_data_dir=beam_data_dir,
            run=run,
            feus=feus,
            offset_range=offset_range,
            min_intensity=min_intensity,
            plot=False,
            write_csv=True,
        )
        all_results.extend(run_results)

    if all_results:
        plot_offsets(all_results)


def process_run(
        runs_path: str,
        beam_data_dir: str,
        run: str,
        feus: list,
        offset_range: np.ndarray,
        min_intensity: float = 0.1,
        plot: bool = False,
        write_csv: bool = False,
        plot_outliers_range: tuple = None,
) -> list[dict]:
    """
    Process all subruns in a single run: load beam data once, iterate subruns.

    Returns a list of result dicts (one per successfully processed subrun),
    each augmented with 'run', 'sub_run', and 'run_start' keys.
    """
    run_dir = os.path.join(runs_path, run)
    sub_runs = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))])
    print(f'  Found {len(sub_runs)} subruns: {sub_runs}')

    # Filter out sub_runs without run_start metadata, collecting run_starts in parallel
    valid_sub_runs, run_starts, run_durations = [], [], []
    for sub in sub_runs:
        try:
            rs = get_run_start(base_path=runs_path, run=run, sub_run=sub)
            rd = get_run_time(base_path=runs_path, run=run, sub_run=sub)
            valid_sub_runs.append(sub)
            run_starts.append(rs)
            run_durations.append(rd)
        except Exception as e:
            print(f'  Skipping {sub} (missing run_start): {e}')
    sub_runs = valid_sub_runs
    if not sub_runs:
        print(f'  No valid subruns found, skipping run.')
        return []
    print(f'  {len(sub_runs)} subruns with valid run_start metadata')

    # Load beam data covering all subruns in this run
    all_bracketing = [f for rs, rd in zip(run_starts, run_durations)
                      for f in get_bracketing_csv_files(beam_data_dir, rs, rd)]
    seen = set()
    bracketing_files = []
    for item in all_bracketing:
        if item[1] not in seen:
            seen.add(item[1])
            bracketing_files.append(item)
    bracketing_files.sort(key=lambda x: x[0])

    print(f'\n  Loading beam intensity data...')
    beam_df = load_beam_csvs(bracketing_files)
    print(f'  Loaded {len(beam_df)} beam rows spanning '
          f'{beam_df["time_s"].min():.2f} - {beam_df["time_s"].max():.2f} s\n')

    results = []
    for sub_run, run_start, run_duration in zip(sub_runs, run_starts, run_durations):
        print(f'  --- {sub_run} ---')

        result = get_subrun_offset(
            runs_path=runs_path,
            run=run,
            sub_run=sub_run,
            beam_df=beam_df,
            feus=feus,
            coarse_range=tuple(offset_range * 1e6),
            min_intensity=min_intensity,
            plot=plot,
            plot_outliers_range=plot_outliers_range,
            write_csv=write_csv,
        )
        if result is not None:
            results.append({'run': run, 'sub_run': sub_run, 'run_start': run_start, 'run_duration': run_duration,
                            **result})
            print(f"    offset={result['offset']/1e6:.4f} s  "
                  f"mean_residual={result['mean_residual']:.1f} us  "
                  f"matched={result['n_matched']}/{result['n_total']}")
        else:
            print(f'    Skipped (no triggers or error).')

    return results


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
        plot_outliers_range: tuple = None,
        write_csv: bool = False,
) -> dict | None:
    """
    Compute the DREAM->beam timestamp offset for a single subrun.

    Loads the subrun, extracts trigger timestamps, filters beam data to the
    subrun window, runs find_timestamp_offset, and optionally plots diagnostics.

    Parameters
    ----------
    runs_path           : root path containing run directories
    run                 : run name (e.g. 'run_17')
    sub_run             : subrun name (e.g. 'resist_hv_460V')
    beam_df             : pre-loaded beam DataFrame (from load_beam_csvs)
    feus                : list of FEU IDs to load
    coarse_range        : (min, max) offset search range in us
    coarse_steps        : number of points in coarse grid
    max_match_distance  : match threshold in us (half the ~1.2 s n_TOF cycle)
    min_intensity       : Minimum intensity in beam data to include in matching
    plot                : if True, show diagnostic plots for this subrun
    plot_outliers_range : if not None, range of best offest outside of which to plot.
    write_csv           : if True, write per-event beam intensity to
                          {runs_path}/{run}/{sub_run}/beam_intensity.csv

    Returns
    -------
    dict from find_timestamp_offset, or None if the subrun cannot be processed.
    """
    try:
        run_start = get_run_start(base_path=runs_path, run=run, sub_run=sub_run)
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

    # Sort df on timestamp
    df.sort_values('trigger_timestamp_ns', inplace=True)

    # One row per unique event: eventId and its raw timestamp in ns
    event_map = (
        df[['eventId', 'trigger_timestamp_ns']]
        .drop_duplicates(subset='eventId')
        .sort_values('trigger_timestamp_ns')
        .reset_index(drop=True)
    )

    # Trigger timestamps in us (absolute unix time), aligned to event_map order
    trigger_timestamps = (event_map['trigger_timestamp_ns'].values + run_start * 1e9) / 1e3

    # Hits-per-event series for optional plotting (indexed by absolute unix time in s)
    trigger_timestamps_df = df['trigger_timestamp_ns'].value_counts().sort_index()
    trigger_timestamps_df.index = trigger_timestamps_df.index / 1e9 + run_start

    # Restrict beam to subrun window (+/- coarse range margin) and significant pulses only
    min_trig = trigger_timestamps_df.index.min()
    max_trig = trigger_timestamps_df.index.max()
    beam_window = beam_df[
        (beam_df['time_s'] > min_trig - coarse_range[0] / 1e6) &
        (beam_df['time_s'] < max_trig + coarse_range[1] / 1e6) &
        (beam_df['intensity'] > min_intensity)
    ].reset_index(drop=True)
    beam_timestamps = beam_window['timestamp_us'].values

    if len(beam_timestamps) == 0:
        print(f'  No beam pulses found in subrun window.')
        return None

    result = find_timestamp_offset(
        ts_A=trigger_timestamps,
        ts_B=beam_timestamps,
        coarse_range=coarse_range,
        coarse_steps=coarse_steps,
        max_match_distance=max_match_distance,
    )

    # --- Match each trigger to its nearest beam pulse at best offset ---
    shifted = trigger_timestamps + result['offset']  # us, same order as event_map
    idx = np.searchsorted(beam_timestamps, shifted, side='left')
    idx = np.clip(idx, 1, len(beam_timestamps) - 1)
    residuals = np.minimum(
        np.abs(shifted - beam_timestamps[idx - 1]),
        np.abs(shifted - beam_timestamps[idx]),
    )
    # Pick the closer of the two neighbors
    nearest_idx = np.where(
        np.abs(shifted - beam_timestamps[idx - 1]) <= np.abs(shifted - beam_timestamps[idx]),
        idx - 1, idx,
    )
    threshold    = max_match_distance if max_match_distance is not None else 3 * np.median(residuals)
    matched_mask = residuals < threshold

    if plot_outliers_range is not None:
        if result["offset"] / 1e6 < plot_outliers_range[0] or result["offset"] / 1e6 > plot_outliers_range[1]:
            plot = True

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
        ax.set_title(f'{run} / {sub_run} - raw (unaligned)')
        ax_ylim = ax.get_ylim()
        ax.set_ylim(top=ax_ylim[1] + (ax_ylim[1] - ax_ylim[0]) * 0.1)
        ax2_ylim = ax2.get_ylim()
        ax2.set_ylim(top=ax2_ylim[1] + ax2_ylim[0] * 0.1)
        fig.tight_layout()
        aligned_times_s = trigger_timestamps_df.index + result['offset'] / 1e6
        hits_per_event  = trigger_timestamps_df.values

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(aligned_times_s[matched_mask], hits_per_event[matched_mask],
                marker='.', lw=0.6, label='Dream (matched)', color='blue')
        if (~matched_mask).any():
            ax.scatter(aligned_times_s[~matched_mask], hits_per_event[~matched_mask],
                       marker='x', s=80, color='red', zorder=5,
                       label=f'Dream (unmatched, n={(~matched_mask).sum()})')
        ax2.plot(beam_window['time_s'], beam_window['intensity'],
                 marker='.', lw=0.6, label='PS', color='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel('Time (s, aligned)')
        ax.set_ylabel('Dream Hits in Readout')
        ax2.set_ylabel('PS Bunch Intensity')
        ax.set_title(f'{run} / {sub_run} - aligned  (offset={result["offset"]/1e6:.4f} s)')
        ax_ylim = ax.get_ylim()
        ax.set_ylim(top=ax_ylim[1] + (ax_ylim[1] - ax_ylim[0]) * 0.15)
        ax2_ylim = ax2.get_ylim()
        ax2.set_ylim(top=ax2_ylim[1] + ax2_ylim[0] * 0.15)
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(result['coarse_offsets'] / 1e6, result['coarse_costs'], marker='.', lw=0.6)
        ax.set_xlabel('Offset (s)')
        ax.set_ylabel('Mean residual (us)')
        ax.set_title(f'{run} / {sub_run} - coarse offset search')
        ax.axvline(result['offset'] / 1e6, color='r', linestyle='--',
                   label=f"Best: {result['offset'] / 1e6:.3f} s")
        ax.legend()
        plt.tight_layout()

    if write_csv:
        out_df = pd.DataFrame({
            'eventId':              event_map['eventId'].values,
            'trigger_timestamp_ns': event_map['trigger_timestamp_ns'].values,
            'trigger_timestamp_us': trigger_timestamps,
            'shifted_timestamp_us': shifted,
            'beam_timestamp_us':    np.where(matched_mask, beam_timestamps[nearest_idx], np.nan),
            'beam_intensity':       np.where(matched_mask, beam_window['intensity'].values[nearest_idx], np.nan),
            'matched':              matched_mask,
        })
        out_dir = os.path.join(runs_path, run, sub_run)
        out_path = os.path.join(out_dir, 'beam_intensity.csv')
        out_df.to_csv(out_path, index=False)
        print(f'  Wrote beam intensity CSV: {out_path}  ({len(out_df)} events, '
              f'{matched_mask.sum()} matched)')

    return result


def plot_offsets(results: list[dict]):
    """
    Plot best-fit offset and match fraction for all subruns across all runs.

    X-axis is the subrun start timestamp (unix seconds). Points are coloured
    by run — one colour per run, no individual legend entry per run. Run
    boundaries are marked with vertical dashed lines, and a compact legend
    spans the full width above the top axis.

    Parameters
    ----------
    results : list of dicts from process_run, each containing 'run',
              'sub_run', 'run_start', 'offset', 'n_matched', 'n_total'
    """
    unique_runs = sorted(set(r['run'] for r in results),
                         key=lambda x: int(x.split('_')[1]))
    cmap = cm.get_cmap('tab20', len(unique_runs))
    run_color = {run: cmap(i) for i, run in enumerate(unique_runs)}

    offsets_s  = np.array([r['offset'] / 1e6 for r in results])
    match_frac = np.array([r['n_matched'] / r['n_total'] if r['n_total'] > 0 else 0
                           for r in results])
    colors     = [run_color[r['run']] for r in results]
    timestamps = [r['run_start'] for r in results]

    dt_labels = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%m-%d %H:%M')
                 for t in timestamps]

    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, values, ylabel in zip(
        axes,
        [offsets_s, match_frac],
        ['Best offset (s)', 'Match fraction'],
    ):
        ax.scatter(x, values, c=colors, s=30, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    axes[1].set_ylim(0, 1.05)

    # X-axis: thinned datetime labels so they don't overlap
    n = len(results)
    step = max(1, n // 40)
    tick_positions = list(range(0, n, step))
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([dt_labels[i] for i in tick_positions],
                             rotation=45, ha='right', fontsize=7)
    axes[1].set_xlabel('Subrun start time (UTC)')

    # Vertical dashed lines at run boundaries
    for i in range(1, len(results)):
        if results[i]['run'] != results[i - 1]['run']:
            for ax in axes:
                ax.axvline(i - 0.5, color='gray', lw=0.8, linestyle='--', alpha=0.5)

    # Legend spanning full width above the top axis
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=run_color[r],
                   markersize=6, label=r)
        for r in unique_runs
    ]
    legend = axes[0].legend(
        handles=handles,
        title='Run',
        fontsize=6,
        title_fontsize=7,
        ncol=max(1, len(unique_runs) // 4),
        loc='lower left',
        bbox_to_anchor=(0, 1.02, 1, 0),
        mode='expand',
        borderaxespad=0,
        framealpha=0.7,
    )

    # Title above the legend box, not competing with it
    axes[0].set_title('DREAM->beam time offset across all runs',
                      pad=legend.get_window_extent().height)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()


def get_bracketing_csv_files(beam_data_dir, run_start_unix, run_duration):
    """
    Find all daily CSV files covering the run, plus one extra day on each side.
    File timestamps are in microseconds. Files are daily (86400s apart).
    Returns sorted list of (timestamp_us, filepath).
    """
    run_start_us = run_start_unix * 1e6
    run_end_us   = (run_start_unix + run_duration) * 1e6

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

    timestamps_us = np.array([ts for ts, _ in csv_files])

    # Index of last file starting at or before run start, and first at or after run end
    start_idx = np.searchsorted(timestamps_us, run_start_us, side='right') - 1
    end_idx   = np.searchsorted(timestamps_us, run_end_us,   side='left')

    if start_idx < 0:
        raise ValueError(f'Run start {run_start_unix:.2f}s is before all available CSV files.')

    # Expand by one day on each side and clamp to valid range
    first = max(start_idx - 1, 0)
    last  = min(end_idx   + 1, len(csv_files) - 1)

    return csv_files[first:last + 1]


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
    ts_A : array of timestamps to shift (e.g. your trigger timestamps, us)
    ts_B : array of reference timestamps (e.g. beam bunch timestamps, us)
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
        offset          - best-fit offset to add to ts_A
        mean_residual   - mean residual of matched points
        n_matched       - number of ts_A points matched within max_match_distance
        n_total         - total number of ts_A points
        coarse_offsets  - offsets evaluated in coarse search
        coarse_costs    - mean residuals at each coarse offset
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
        options={'xatol': 1.0},  # 1 us precision
    )
    best_offset = result.x

    # --- Match stats at best offset ---
    residuals = nearest_residuals(best_offset)
    threshold = max_match_distance if max_match_distance is not None \
                else 3 * np.median(residuals)
    mask = residuals < threshold

    return {
        "offset":         best_offset,
        "mean_residual":  np.mean(residuals[mask]),
        "n_matched":      int(mask.sum()),
        "n_total":        len(ts_A),
        "coarse_offsets": coarse_offsets,
        "coarse_costs":   coarse_costs,
    }


if __name__ == '__main__':
    main()
