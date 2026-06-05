#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 28 2026
Created in PyCharm
Created as nTof_x17/ntof_may_analysis/dream_timber_time_sync_flash.py

@author: Dylan Neff, dylan

Adapted from proof_of_concept/dream_timber_time_sync.py for may data where each
beam pulse produces multiple DREAM triggers (gamma flash + subsequent thermal events).

Strategy: identify gamma flash events using high hit-count + early timing criteria,
then use only those timestamps for DREAM<->timber synchronisation — same matching
logic as the original but restricted to one-per-pulse flash events.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize_scalar
import uproot

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from plot_beam_hits import get_run_start, get_run_time, get_run_json

# ── Gamma flash identification parameters ─────────────────────────────────────
# An event is classified as a gamma flash if at least one FEU has more than
# GAMMA_FLASH_MAX_HITS hits satisfying amp > GAMMA_FLASH_AMP AND time < GAMMA_FLASH_TIME_NS.
GAMMA_FLASH_MAX_HITS  = 200    # hits per FEU per event
GAMMA_FLASH_AMP       = 500    # ADC — minimum amplitude
GAMMA_FLASH_TIME_NS   = 1000.0  # ns  — only hits before this time count


def main():
    runs_path    = '/home/dylan/x17/may_beam/runs/'
    beam_data_dir = '/home/dylan/x17/may_beam/ntof_bunch_intensities/'
    offset_range  = np.array([0, 10])  # seconds
    min_intensity = 0.1  # minimum PS bunch intensity to include in matching

    # run = 'run_70'
    run = 'run_52'
    plot_outliers_range = None
    plot_single_run(runs_path, beam_data_dir, offset_range, min_intensity, run, plot_outliers_range)

    print('\ndonzo')


def plot_single_run(runs_path, beam_data_dir, offset_range, min_intensity, run,
                   plot_outliers_range=None):
    process_run(
        runs_path=runs_path,
        beam_data_dir=beam_data_dir,
        run=run,
        offset_range=offset_range,
        min_intensity=min_intensity,
        plot=True,
        plot_outliers_range=plot_outliers_range,
        write_csv=False,
    )
    plt.show()


def process_all_runs(runs_path, beam_data_dir, offset_range, min_intensity,
                     min_events_for_fit=10):
    all_runs = sorted([
        d for d in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, d)) and d.startswith('run_') and not d.endswith('_testing')
    ], key=lambda r: int(r.split('_')[1]))

    good_runs = []
    for run in all_runs:
        if run in ('run_52', 'run_54'):
            continue
        cfg = get_run_json(runs_path, run)
        print(run)
        dream_config_file = os.path.basename(cfg['dream_daq_info']['daq_config_template_path'])
        print(dream_config_file)
        if dream_config_file.startswith('Tcm_Mx17_') and dream_config_file != 'Tcm_Mx17_SiPM_trig.cfg':
            good_runs.append(run)

    all_runs = good_runs
    print(f'Found {len(all_runs)} runs: {all_runs}')

    all_results = []
    for run in all_runs:
        print(f'\n{"=" * 60}')
        print(f'Processing {run}')
        print(f'{"=" * 60}')
        run_results = process_run(
            runs_path=runs_path,
            beam_data_dir=beam_data_dir,
            run=run,
            offset_range=offset_range,
            min_intensity=min_intensity,
            plot=False,
            write_csv=True,
            min_events_for_fit=min_events_for_fit,
        )
        all_results.extend(run_results)

    if all_results:
        all_results.sort(key=lambda r: r['run_start'])
        plot_offsets(all_results)
        plot_offsets_timeline(all_results)
        plt.show()


def process_run(
        runs_path: str,
        beam_data_dir: str,
        run: str,
        offset_range: np.ndarray,
        min_intensity: float = 0.1,
        plot: bool = False,
        write_csv: bool = False,
        plot_outliers_range: tuple = None,
        min_events_for_fit: int = 5,
) -> list[dict]:
    """
    Process all subruns in a single run: load beam data once, iterate subruns.

    For subruns with fewer flash events than min_events_for_fit, the offset is
    interpolated from the mean of the two nearest fitted subruns.
    """
    feus = get_feus_from_run_config(runs_path, run)
    print(f'  FEUs from config: {feus}')

    run_dir  = os.path.join(runs_path, run)
    sub_runs = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))])
    print(f'  Found {len(sub_runs)} subruns: {sub_runs}')

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

    all_bracketing = [f for rs, rd in zip(run_starts, run_durations)
                      for f in get_bracketing_csv_files(beam_data_dir, rs, rd)]
    seen, bracketing_files = set(), []
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
            min_events_for_fit=min_events_for_fit,
            plot=plot,
            plot_outliers_range=plot_outliers_range,
            write_csv=False,
        )
        if result is not None:
            results.append({'run': run, 'sub_run': sub_run, 'run_start': run_start,
                            'run_duration': run_duration, 'offset_interpolated': False, **result})
            print(f"    offset={result['offset']/1e6:.4f} s  "
                  f"mean_residual={result['mean_residual']:.1f} us  "
                  f"matched={result['n_matched']}/{result['n_total']}"
                  + ("  [too few events — will interpolate]"
                     if result.get('too_few_events') else ""))
        else:
            print(f'    Skipped (no flash triggers or error).')

    # Interpolate offsets for sparse subruns
    fitted_indices = [i for i, r in enumerate(results) if not r.get('too_few_events')]
    fitted_starts  = np.array([results[i]['run_start'] for i in fitted_indices])
    fitted_offsets = np.array([results[i]['offset']    for i in fitted_indices])

    for i, r in enumerate(results):
        if not r.get('too_few_events'):
            continue
        if len(fitted_indices) == 0:
            print(f"  Warning: no fitted subruns to interpolate {r['sub_run']}, skipping.")
            continue
        t      = r['run_start']
        diffs  = np.abs(fitted_starts - t)
        order  = np.argsort(diffs)
        neighbours    = order[:2]
        interp_offset = float(np.mean(fitted_offsets[neighbours]))
        results[i]['offset'] = interp_offset
        results[i]['offset_interpolated'] = True
        print(f"  Interpolated offset for {r['sub_run']}: {interp_offset/1e6:.4f} s "
              f"(mean of {len(neighbours)} neighbour(s))")

    if write_csv:
        for r in results:
            get_subrun_offset(
                runs_path=runs_path,
                run=run,
                sub_run=r['sub_run'],
                beam_df=beam_df,
                feus=feus,
                coarse_range=tuple(offset_range * 1e6),
                min_intensity=min_intensity,
                min_events_for_fit=min_events_for_fit,
                plot=False,
                write_csv=True,
                forced_offset=r['offset'],
            )

    return results


def identify_flash_events(df: pd.DataFrame) -> np.ndarray:
    """
    Return eventIds classified as gamma flash events.

    A flash event has > GAMMA_FLASH_MAX_HITS hits with amplitude > GAMMA_FLASH_AMP
    and time < GAMMA_FLASH_TIME_NS in at least one FEU.
    """
    gf = df[(df['amplitude'] > GAMMA_FLASH_AMP) & (df['time'] < GAMMA_FLASH_TIME_NS)]
    if gf.empty:
        return np.array([], dtype=np.int64)
    hits_per_feu = gf.groupby(['eventId', 'feu']).size().unstack(fill_value=0)
    flash_mask   = (hits_per_feu > GAMMA_FLASH_MAX_HITS).any(axis=1)
    return hits_per_feu.index[flash_mask].to_numpy(dtype=np.int64)


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
        min_events_for_fit: int = 5,
        plot: bool = False,
        plot_outliers_range: tuple = None,
        write_csv: bool = False,
        forced_offset: float = None,
) -> dict | None:
    """
    Compute the DREAM->beam timestamp offset for a single subrun using only
    gamma flash events as the reference timestamps.

    Loads the subrun, identifies flash events, extracts their trigger timestamps,
    filters beam data to the subrun window, and runs find_timestamp_offset.

    Parameters are the same as the original dream_timber_time_sync.py, with
    flash identification applied automatically from module-level parameters.
    """
    try:
        run_start = get_run_start(base_path=runs_path, run=run, sub_run=sub_run)
    except Exception as e:
        print(f'  Could not read run metadata: {e}')
        return None

    try:
        df = _load_hits_df(runs_path, run, sub_run, feus)
    except Exception as e:
        print(f'  Could not load subrun: {e}')
        return None

    if df is None or df.empty or 'trigger_timestamp_ns' not in df.columns:
        print(f'  No trigger data found.')
        return None

    df.sort_values('trigger_timestamp_ns', inplace=True)

    # All events (for event-type diagnostic)
    all_event_map = (
        df[['eventId', 'trigger_timestamp_ns']]
        .drop_duplicates(subset='eventId')
        .sort_values('trigger_timestamp_ns')
        .reset_index(drop=True)
    )
    n_total_events = len(all_event_map)

    # Identify gamma flash events and restrict matching to them
    flash_ids  = identify_flash_events(df)
    n_flash    = len(flash_ids)
    n_non_flash = n_total_events - n_flash
    print(f'    Events: {n_total_events} total  |  {n_flash} flash  |  {n_non_flash} non-flash')

    if n_flash == 0:
        print(f'    No gamma flash events found — cannot match.')
        return None

    event_map = all_event_map[all_event_map['eventId'].isin(flash_ids)].reset_index(drop=True)

    # Trigger timestamps in us (absolute unix time), flash events only
    trigger_timestamps = (event_map['trigger_timestamp_ns'].values + run_start * 1e9) / 1e3

    # Flash event hit counts indexed by absolute unix time (for plotting)
    flash_df    = df[df['eventId'].isin(flash_ids)]
    flash_ts_df = flash_df['trigger_timestamp_ns'].value_counts().sort_index()
    flash_ts_df.index = flash_ts_df.index / 1e9 + run_start

    # Restrict beam to subrun window (+/- coarse range margin) and significant pulses only
    min_trig   = flash_ts_df.index.min()
    max_trig   = flash_ts_df.index.max()
    beam_window = beam_df[
        (beam_df['time_s'] > min_trig - coarse_range[0] / 1e6) &
        (beam_df['time_s'] < max_trig + coarse_range[1] / 1e6) &
        (beam_df['intensity'] > min_intensity)
    ].reset_index(drop=True)
    beam_timestamps = beam_window['timestamp_us'].values

    if len(beam_timestamps) == 0:
        print(f'  No beam pulses found in subrun window.')
        return None

    too_few = n_flash < min_events_for_fit

    if forced_offset is not None:
        result = {
            'offset':         forced_offset,
            'mean_residual':  np.nan,
            'n_matched':      0,
            'n_total':        n_flash,
            'coarse_offsets': np.array([]),
            'coarse_costs':   np.array([]),
            'too_few_events': False,
        }
    elif too_few:
        print(f'    Only {n_flash} flash events (< {min_events_for_fit}), skipping minimization.')
        return {
            'offset':         np.nan,
            'mean_residual':  np.nan,
            'n_matched':      0,
            'n_total':        n_flash,
            'coarse_offsets': np.array([]),
            'coarse_costs':   np.array([]),
            'too_few_events': True,
        }
    else:
        result = find_timestamp_offset(
            ts_A=trigger_timestamps,
            ts_B=beam_timestamps,
            coarse_range=coarse_range,
            coarse_steps=coarse_steps,
            max_match_distance=max_match_distance,
        )
        result['too_few_events'] = False

    # Match each flash trigger to its nearest beam pulse at best offset
    shifted     = trigger_timestamps + result['offset']
    idx         = np.searchsorted(beam_timestamps, shifted, side='left')
    idx         = np.clip(idx, 1, len(beam_timestamps) - 1)
    residuals   = np.minimum(
        np.abs(shifted - beam_timestamps[idx - 1]),
        np.abs(shifted - beam_timestamps[idx]),
    )
    nearest_idx = np.where(
        np.abs(shifted - beam_timestamps[idx - 1]) <= np.abs(shifted - beam_timestamps[idx]),
        idx - 1, idx,
    )
    threshold    = max_match_distance if max_match_distance is not None else 3 * np.median(residuals)
    matched_mask = residuals < threshold

    if plot_outliers_range is not None:
        if result['offset'] / 1e6 < plot_outliers_range[0] or result['offset'] / 1e6 > plot_outliers_range[1]:
            plot = True

    if plot:
        # ── Diagnostic: hits-per-event histogram to verify flash/non-flash separation ──
        hits_per_event = df.groupby('eventId').size()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(hits_per_event.values, bins=100, log=True, color='steelblue', edgecolor='none')
        ax.axvline(GAMMA_FLASH_MAX_HITS, color='red', ls='--',
                   label=f'Flash threshold ({GAMMA_FLASH_MAX_HITS} hits)')
        ax.set_xlabel('Hits per event')
        ax.set_ylabel('Events (log scale)')
        ax.set_title(f'{run} / {sub_run} — hits-per-event distribution\n'
                     f'(amp>{GAMMA_FLASH_AMP}, t<{GAMMA_FLASH_TIME_NS:.0f} ns flash criteria)')
        ax.legend()
        fig.tight_layout()

        # ── Raw (unaligned): flash event rate vs beam pulses ──
        all_ts_df = df['trigger_timestamp_ns'].value_counts().sort_index()
        all_ts_df.index = all_ts_df.index / 1e9 + run_start

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(all_ts_df.index, all_ts_df.values, ls='none', marker='.', lw=0.6,
                label='All DREAM events', color='lightblue', alpha=0.5)
        ax.plot(flash_ts_df.index, flash_ts_df.values, ls='none', marker='.', lw=0.6,
                label=f'Flash events ({n_flash})', color='blue')
        ax2.plot(beam_window['time_s'], beam_window['intensity'], ls='none', marker='.', lw=0.6,
                 label='PS', color='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DREAM Hits in Readout')
        ax2.set_ylabel('PS Bunch Intensity')
        ax.set_title(f'{run} / {sub_run} — raw (unaligned), flash events highlighted')
        fig.tight_layout()

        # ── Aligned: flash events vs beam pulses ──
        aligned_times_s = flash_ts_df.index + result['offset'] / 1e6
        hits_per_flash  = flash_ts_df.values

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(aligned_times_s, hits_per_flash, ls='none', marker='.', lw=0.6,
                label='Flash (matched)', color='blue')
        if (~matched_mask).any():
            ax.scatter(aligned_times_s[~matched_mask], hits_per_flash[~matched_mask],
                       marker='x', s=80, color='red', zorder=5,
                       label=f'Flash (unmatched, n={(~matched_mask).sum()})')
        ax2.plot(beam_window['time_s'], beam_window['intensity'], ls='none', marker='.', lw=0.6,
                 label='PS', color='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel('Time (s, aligned)')
        ax.set_ylabel('DREAM Hits in Readout')
        ax2.set_ylabel('PS Bunch Intensity')
        ax.set_title(f'{run} / {sub_run} — aligned  (offset={result["offset"]/1e6:.4f} s)')
        ax_ylim  = ax.get_ylim()
        ax.set_ylim(top=ax_ylim[1] + (ax_ylim[1] - ax_ylim[0]) * 0.15)
        ax2_ylim = ax2.get_ylim()
        ax2.set_ylim(top=ax2_ylim[1] + (ax2_ylim[1] - ax2_ylim[0]) * 0.1)
        fig.tight_layout()

        # ── Coarse offset search ──
        fig, ax = plt.subplots()
        ax.plot(result['coarse_offsets'] / 1e6, result['coarse_costs'], marker='.', lw=0.6)
        ax.set_xlabel('Offset (s)')
        ax.set_ylabel('Mean residual (us)')
        ax.set_title(f'{run} / {sub_run} — coarse offset search (flash events)')
        ax.axvline(result['offset'] / 1e6, color='r', linestyle='--',
                   label=f"Best: {result['offset'] / 1e6:.3f} s")
        ax.legend()
        fig.tight_layout()

    if write_csv:
        out_df = pd.DataFrame({
            'eventId':              event_map['eventId'].values,
            'trigger_timestamp_ns': event_map['trigger_timestamp_ns'].values,
            'trigger_timestamp_us': trigger_timestamps,
            'shifted_timestamp_us': shifted,
            'beam_timestamp_us':    np.where(matched_mask, beam_timestamps[nearest_idx], np.nan),
            'beam_intensity':       np.where(matched_mask, beam_window['intensity'].values[nearest_idx], np.nan),
            'matched':              matched_mask,
            'is_flash':             True,
        })
        out_dir  = os.path.join(runs_path, run, sub_run)
        out_path = os.path.join(out_dir, 'beam_intensity_flash.csv')
        out_df.to_csv(out_path, index=False)
        print(f'  Wrote beam intensity CSV: {out_path}  ({len(out_df)} events, '
              f'{matched_mask.sum()} matched)')

    return result


# ── Summary plots ──────────────────────────────────────────────────────────────

def _make_run_legend(fig, axes, unique_runs, run_color):
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=run_color[r],
                   markersize=6, label=r)
        for r in unique_runs
    ]
    axes[0].legend(
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
    axes[0].set_title('DREAM->beam time offset across all runs (flash-based sync)', pad=80)


def plot_offsets(results: list[dict]):
    unique_runs = sorted(set(r['run'] for r in results), key=lambda x: int(x.split('_')[1]))
    cmap        = cm.get_cmap('tab20', len(unique_runs))
    run_color   = {run: cmap(i) for i, run in enumerate(unique_runs)}

    offsets_s  = np.array([r['offset'] / 1e6 for r in results])
    match_frac = np.array([r['n_matched'] / r['n_total'] if r['n_total'] > 0 else 0
                           for r in results])
    colors     = [run_color[r['run']] for r in results]
    dt_labels  = [datetime.fromtimestamp(r['run_start'], tz=timezone.utc).strftime('%m-%d %H:%M')
                  for r in results]
    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, values, ylabel in zip(axes, [offsets_s, match_frac],
                                   ['Best offset (s)', 'Match fraction']):
        ax.scatter(x, values, c=colors, s=30, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    axes[1].set_ylim(0, 1.05)
    n    = len(results)
    step = max(1, n // 40)
    tick_positions = list(range(0, n, step))
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([dt_labels[i] for i in tick_positions], rotation=45, ha='right', fontsize=7)
    axes[1].set_xlabel('Subrun start time (UTC)')
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    for i in range(1, len(results)):
        if results[i]['run'] != results[i - 1]['run']:
            for ax in axes:
                ax.axvline(i - 0.5, color='gray', lw=0.8, linestyle='--', alpha=0.5)

    _make_run_legend(fig, axes, unique_runs, run_color)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, top=0.84, left=0.052, right=0.995)


def plot_offsets_timeline(results: list[dict]):
    import matplotlib.dates as mdates

    unique_runs = sorted(set(r['run'] for r in results), key=lambda x: int(x.split('_')[1]))
    cmap        = cm.get_cmap('tab20', len(unique_runs))
    run_color   = {run: cmap(i) for i, run in enumerate(unique_runs)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    def to_dt(unix_s):
        return datetime.fromtimestamp(unix_s, tz=timezone.utc)

    for r in results:
        t_start    = to_dt(r['run_start'])
        t_end      = to_dt(r['run_start'] + r['run_duration'])
        offset_s   = r['offset'] / 1e6
        match_frac = r['n_matched'] / r['n_total'] if r['n_total'] > 0 else 0
        color      = run_color[r['run']]
        for ax, value in zip(axes, [offset_s, match_frac]):
            ax.hlines(value, t_start, t_end, colors=color, linewidths=5, alpha=1.0)

    for ax, ylabel in zip(axes, ['Best offset (s)', 'Match fraction']):
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=timezone.utc))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    axes[1].set_xlabel('Time (UTC)')

    _make_run_legend(fig, axes, unique_runs, run_color)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, top=0.84, left=0.052, right=0.995)


# ── Data loading helpers ───────────────────────────────────────────────────────

def get_feus_from_run_config(runs_path, run):
    cfg      = get_run_json(runs_path, run)
    included = set(cfg.get('included_detectors') or [])
    feu_ids  = set()
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if included and name not in included:
            continue
        if 'dream_feus' not in det_cfg:
            continue
        for v in det_cfg['dream_feus'].values():
            feu_ids.add(v[0])
    return sorted(feu_ids)


def _load_hits_df(runs_path, run, sub_run, feus):
    hits_dir  = os.path.join(runs_path, run, sub_run, 'combined_hits_root')
    hit_files = [f for f in os.listdir(hits_dir) if f.endswith('.root') and '_datrun_' in f]
    if not hit_files:
        return pd.DataFrame()
    file_sources = [os.path.join(hits_dir, hf) + ':hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    if not df.empty:
        df = df[df['feu'].isin(feus)]
    return df


def get_bracketing_csv_files(beam_data_dir, run_start_unix, run_duration):
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
    start_idx = np.searchsorted(timestamps_us, run_start_us, side='right') - 1
    end_idx   = np.searchsorted(timestamps_us, run_end_us,   side='left')

    if start_idx < 0:
        raise ValueError(f'Run start {run_start_unix:.2f}s is before all available CSV files.')

    first = max(start_idx - 1, 0)
    last  = min(end_idx   + 1, len(csv_files) - 1)
    return csv_files[first:last + 1]


def load_beam_csvs(csv_file_list):
    dfs = []
    for ts_us, fpath in csv_file_list:
        df = pd.read_csv(fpath, skiprows=1)
        df.columns = ['timestamp_us', 'intensity']
        df['time_s'] = df['timestamp_us'] / 1e6
        dfs.append(df)
        print(f'  Loaded: {os.path.basename(fpath)} ({len(df)} rows)')
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)


# ── Offset finding ─────────────────────────────────────────────────────────────

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

    ts_B is the complete reference set (beam); ts_A may have missing entries.
    Uses coarse grid search followed by scipy scalar minimization.
    """
    ts_A = np.sort(np.asarray(ts_A, dtype=np.float64))
    ts_B = np.sort(np.asarray(ts_B, dtype=np.float64))

    def nearest_residuals(offset):
        shifted = ts_A + offset
        idx     = np.searchsorted(ts_B, shifted, side='left')
        idx     = np.clip(idx, 1, len(ts_B) - 1)
        return np.minimum(np.abs(shifted - ts_B[idx - 1]), np.abs(shifted - ts_B[idx]))

    def cost(offset):
        return np.mean(nearest_residuals(offset))

    coarse_offsets = np.linspace(coarse_range[0], coarse_range[1], coarse_steps)
    coarse_costs   = np.array([cost(o) for o in coarse_offsets])

    best_coarse = coarse_offsets[np.argmin(coarse_costs)]
    step = (coarse_range[1] - coarse_range[0]) / coarse_steps
    fw   = fine_window if fine_window is not None else step * 10

    result = minimize_scalar(
        cost,
        bounds=(best_coarse - fw, best_coarse + fw),
        method='bounded',
        options={'xatol': 1.0},
    )
    best_offset = result.x

    residuals = nearest_residuals(best_offset)
    threshold = max_match_distance if max_match_distance is not None \
                else 3 * np.median(residuals)
    mask = residuals < threshold

    return {
        'offset':         best_offset,
        'mean_residual':  np.mean(residuals[mask]),
        'n_matched':      int(mask.sum()),
        'n_total':        len(ts_A),
        'coarse_offsets': coarse_offsets,
        'coarse_costs':   coarse_costs,
    }


if __name__ == '__main__':
    main()
