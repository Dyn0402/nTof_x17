#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 25 2026
Created in PyCharm
Created as nTof_x17/beam_track_finding.py

Track-finding algorithm for strip micromegas detector (X and Y projections).

Algorithm overview (applied to each projection independently):
  1. Find dead strips from all-event hit-rate analysis.
  2. Load single-event hits and apply amplitude threshold.
  3. Remove isolated hits: a hit is isolated if it has no neighbour within
     ISO_POS_MM in position AND ISO_TIME_NS in time simultaneously.
  4. Walk from late time (sparse) to early time (dense):
     - Seed on the latest unassigned hit.
     - Grow the track by repeatedly finding the next hit that lies within
       ROAD_WIDTH_MM of the extrapolated track line.
     - Misses at known dead-strip positions do NOT count against MAX_MISSED.
     - Stop when MAX_MISSED non-dead-strip consecutive misses accumulate.
  5. Accept tracks with >= MIN_TRACK_HITS hits.

@author: Dylan Neff, dylan
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from tqdm import tqdm

from plot_beam_hits import load_subrun, add_xy_pos


# ─── Event / run selection ────────────────────────────────────────────────────
BASE_PATH = '/media/dylan/data/x17/feb_beam/runs/'
RUN = 'run_34'
SUBRUN = 'resist_425V_drift_600V'
FEU_NUMS = {4: 'y', 5: 'x'}
EVENT = 14
MIN_HIT_AMP = 400

# ─── Physics parameters ───────────────────────────────────────────────────────
DRIFT_VELOCITY_MM_US = 22.0   # mm/μs  (20 μm/ns = 20 mm/μs)
DRIFT_GAP_MM = 30.0           # total drift length in mm

# ─── Run mode ────────────────────────────────────────────────────────────────
RUN_MODE = 'all'          # 'single' → visualise one event; 'all' → metrics over full subrun
FREE_THREADS = 1          # CPU cores to leave free; workers = os.cpu_count() - FREE_THREADS

# ─── Dead-strip finder ────────────────────────────────────────────────────────
DEAD_STRIP_AMP_THRESHOLD = 500          # amplitude cut for dead-strip analysis
DEAD_STRIP_RATE_FRACTION = 0.50         # fraction of median below which a strip is dead

# ─── Isolation filter ─────────────────────────────────────────────────────────
ISO_POS_MM = 10.0             # mm: neighbourhood radius in position
ISO_TIME_NS = 500.0           # ns: neighbourhood radius in time

# ─── Track-finding parameters ────────────────────────────────────────────────
ROAD_WIDTH_MM = 2.0           # mm:  max distance of a hit from the track line
ROAD_WIDTH_SEED_MM = 5.0      # mm:  road width when slope is not yet known (first step)
MAX_TIME_GAP_US = 0.3         # μs:  max time gap between consecutive track hits
MIN_TRACK_HITS = 2            # minimum hits to accept a track
MAX_MISSED = 1                # max consecutive time-step attempts with no hit before stopping
MAX_STRIP_GAP = 2             # max consecutive *live* strips skipped between accepted hits


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def raw_time_to_us(raw_time):
    """Convert raw hit time (ROOT file units) to microseconds."""
    return raw_time / 3.0 / 1000.0


def get_strip_positions(det):
    """
    Return sorted arrays of all possible strip positions (mm) for each projection,
    derived directly from the detector strip map.

    X-projection positions come from 'y'-axis strips (strips running along Y
    measure the X coordinate).  Y-projection positions come from 'x'-axis strips.

    Returns (x_strip_positions_mm, y_strip_positions_mm).
    """
    x_pos, y_pos = set(), set()
    for (axis, _conn, _ch), (xp, yp) in det.strip_map.map.items():
        if axis == 'y' and xp is not None:
            x_pos.add(float(xp))
        elif axis == 'x' and yp is not None:
            y_pos.add(float(yp))
    return np.array(sorted(x_pos)), np.array(sorted(y_pos))


def get_strip_pitch(all_x_strips, all_y_strips):
    """Median spacing between adjacent strips for each projection."""
    def _pitch(arr):
        return float(np.median(np.diff(arr))) if len(arr) >= 2 else np.nan
    return _pitch(all_x_strips), _pitch(all_y_strips)


# ─────────────────────────────────────────────────────────────────────────────
# Dead-strip finder
# ─────────────────────────────────────────────────────────────────────────────

def find_dead_strips(base_path, run, subrun, feu_nums, det,
                     all_x_strips, all_y_strips,
                     amp_threshold=DEAD_STRIP_AMP_THRESHOLD,
                     dead_fraction=DEAD_STRIP_RATE_FRACTION):
    """
    Identify dead strips from the per-strip hit rate across all events.

    Loads all hits in the subrun with amplitude >= amp_threshold, counts hits
    per strip position, normalises by the number of events, and flags strips
    whose rate is below dead_fraction * median(active strip rates).

    Returns
    -------
    dead_x, dead_y : arrays of dead strip positions (mm) for each projection
    x_rates, y_rates : pd.Series  (index = strip position, value = hits/event)
    thresh_x, thresh_y : float thresholds used
    n_events : int
    """
    print(f'Loading all events for dead-strip analysis ({run}/{subrun}) …')
    df, _ = load_subrun(base_path, run, subrun, list(feu_nums.keys()))
    df = df[df['amplitude'] >= amp_threshold].copy()
    n_events = int(df['eventId'].nunique())
    print(f'  {len(df)} hits over {n_events} events (amp >= {amp_threshold})')

    df = add_xy_pos(df, det)
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

    # Hit count per strip position (exact float key from the map)
    x_counts = df_x['x_position_mm'].value_counts()
    y_counts = df_y['y_position_mm'].value_counts()

    x_rates = pd.Series(
        {p: x_counts.get(p, 0) / n_events for p in all_x_strips},
        index=all_x_strips, name='hits_per_event'
    )
    y_rates = pd.Series(
        {p: y_counts.get(p, 0) / n_events for p in all_y_strips},
        index=all_y_strips, name='hits_per_event'
    )

    # Threshold: fraction of median over strips that fired at all
    def _threshold(rates):
        active = rates[rates > 0]
        if len(active) == 0:
            return 0.0
        return float(np.median(active)) * dead_fraction

    thresh_x = _threshold(x_rates)
    thresh_y = _threshold(y_rates)

    dead_x = x_rates.index[x_rates < thresh_x].to_numpy()
    dead_y = y_rates.index[y_rates < thresh_y].to_numpy()

    print(f'  Dead strips: X={len(dead_x)}  Y={len(dead_y)}  '
          f'(threshold: X={thresh_x:.3f}  Y={thresh_y:.3f} hits/event)')
    return dead_x, dead_y, x_rates, y_rates, thresh_x, thresh_y, n_events


def plot_dead_strip_figure(x_rates, y_rates, dead_x, dead_y,
                           thresh_x, thresh_y, n_events, pitch_x, pitch_y, run, subrun):
    """
    Two-panel figure showing hits/event vs strip position for X and Y,
    with dead strips highlighted and threshold lines drawn.
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f'Dead-strip analysis  —  {run}/{subrun}\n'
        f'{n_events} events  |  amp ≥ {DEAD_STRIP_AMP_THRESHOLD}  |  '
        f'dead threshold = {DEAD_STRIP_RATE_FRACTION:.0%} of median',
        fontsize=11
    )

    for ax, rates, dead, thresh, label, pitch in [
        (axs[0], x_rates, dead_x, thresh_x, 'X', pitch_x),
        (axs[1], y_rates, dead_y, thresh_y, 'Y', pitch_y),
    ]:
        positions = rates.index.to_numpy()
        values = rates.to_numpy()
        dead_set = set(dead)

        colors = ['tomato' if p in dead_set else 'steelblue' for p in positions]
        ax.bar(positions, values, width=pitch,
               color=colors, align='center', zorder=2)

        median_active = thresh / DEAD_STRIP_RATE_FRACTION
        ax.axhline(median_active, color='green', ls='--', lw=1.2,
                   label=f'Median active: {median_active:.3f}')
        ax.axhline(thresh, color='orange', ls='--', lw=1.2,
                   label=f'Dead threshold: {thresh:.3f}')

        dead_patch = mpatches.Patch(color='tomato', label=f'Dead strips ({len(dead)})')
        live_patch = mpatches.Patch(color='steelblue', label=f'Active strips ({(~np.isin(positions, dead)).sum()})')
        ax.legend(handles=[live_patch, dead_patch,
                            mpatches.Patch(color='green', ls='--', fill=False, label=f'Median: {median_active:.3f}'),
                            mpatches.Patch(color='orange', ls='--', fill=False, label=f'Threshold: {thresh:.3f}')],
                  fontsize=8)
        ax.set_xlabel(f'{label} strip position (mm)')
        ax.set_ylabel('Hits / event')
        ax.set_title(f'{label} projection — {len(dead)} dead strip(s)')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Isolation filter
# ─────────────────────────────────────────────────────────────────────────────

def remove_isolated_hits(pos_mm, time_us, iso_pos_mm, iso_time_us):
    """
    Return boolean mask (True = keep) for hits that have at least one
    neighbour within iso_pos_mm in position AND iso_time_us in time.
    """
    n = len(pos_mm)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        dp = np.abs(pos_mm - pos_mm[i])
        dt = np.abs(time_us - time_us[i])
        neighbours = (dp < iso_pos_mm) & (dt < iso_time_us)
        neighbours[i] = False
        if np.any(neighbours):
            keep[i] = True
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Track finding
# ─────────────────────────────────────────────────────────────────────────────

def _fit_slope(time_us, pos_mm):
    """Linear fit; returns (slope [mm/μs], intercept [mm])."""
    if len(time_us) < 2:
        return None, None
    _RankWarning = getattr(np.exceptions, 'RankWarning', np.RankWarning)
    with warnings.catch_warnings():
        warnings.simplefilter('error', _RankWarning)
        try:
            coeffs = np.polyfit(time_us, pos_mm, 1)
        except _RankWarning:
            print(f'WARNING: polyfit poorly conditioned — time_us={time_us}  pos_mm={pos_mm}')
            return None, None
    return float(coeffs[0]), float(coeffs[1])


def _count_dead_strips_between(p1, p2, dead_strip_positions, strip_pitch):
    """
    Count dead strips whose centres lie strictly between p1 and p2
    (with half-pitch tolerance at the edges so the endpoint strips are excluded).
    """
    if dead_strip_positions is None or len(dead_strip_positions) == 0:
        return 0
    lo, hi = min(p1, p2), max(p1, p2)
    hw = strip_pitch / 2.0
    return int(np.sum((dead_strip_positions > lo + hw) & (dead_strip_positions < hi - hw)))


def _near_dead_strip(pos_mm_val, dead_strip_positions, strip_pitch):
    """True if pos_mm_val is within 1.5 strip pitches of any dead strip."""
    if dead_strip_positions is None or len(dead_strip_positions) == 0:
        return False
    return bool(np.any(np.abs(dead_strip_positions - pos_mm_val) < 1.5 * strip_pitch))


def _find_next_hit(pos_mm, time_us, assigned, track_indices,
                   road_width_mm, road_width_seed_mm,
                   max_time_gap_us, max_missed):
    """
    Find the best next hit to extend the current track toward earlier times.

    With >= 2 track hits, uses the linear-fit extrapolation and road_width_mm.
    With only 1 hit (seed), uses road_width_seed_mm and no slope constraint.
    Searches within [t_head - max_time_gap_us*(max_missed+1), t_head).
    Returns the index of the chosen hit, or None.
    """
    t_head = time_us[track_indices[-1]]
    slope, intercept = _fit_slope(time_us[track_indices], pos_mm[track_indices])

    if slope is None:
        road = road_width_seed_mm
        p_ref = pos_mm[track_indices[-1]]
    else:
        road = road_width_mm

    t_min = t_head - max_time_gap_us * (max_missed + 1)
    mask = (~assigned) & (time_us < t_head) & (time_us >= t_min)
    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return None

    cand_t = time_us[cand_idx]
    cand_p = pos_mm[cand_idx]

    if slope is not None:
        p_expected = slope * cand_t + intercept
    else:
        p_expected = np.full(len(cand_idx), p_ref)

    within_road = np.abs(cand_p - p_expected) < road
    if not np.any(within_road):
        return None

    valid = cand_idx[within_road]
    return int(valid[np.argmax(time_us[valid])])  # closest in time


def find_tracks_1d(pos_mm, time_us,
                   road_width_mm=ROAD_WIDTH_MM,
                   road_width_seed_mm=ROAD_WIDTH_SEED_MM,
                   max_time_gap_us=MAX_TIME_GAP_US,
                   min_hits=MIN_TRACK_HITS,
                   max_missed=MAX_MISSED,
                   max_strip_gap=MAX_STRIP_GAP,
                   dead_strip_positions=None,
                   strip_pitch=0.8):
    """
    Find straight-line tracks in a 1-D position-vs-drift-time projection.

    Walks from late time (sparse) to early time (dense).

    Two independent gap constraints:
      max_missed     : max consecutive time-step attempts returning no hit at
                       all (controls when to give up searching).
      max_strip_gap  : max consecutive *live* strips that may be skipped
                       between two accepted hits.  Dead strips in the gap are
                       subtracted from the count so they don't penalise the
                       connection.  This is the primary quality constraint that
                       prevents joining hits across many missing strips.

    Parameters
    ----------
    pos_mm, time_us       : hit positions (mm) and drift times (μs)
    road_width_mm         : max distance from track line for a valid hit
    road_width_seed_mm    : road width used before the slope is established
    max_time_gap_us       : max time step between consecutive track hits
    min_hits              : minimum hits to accept a track
    max_missed            : max consecutive time-step attempts with no hit
    max_strip_gap         : max consecutive live strips skipped between hits
    dead_strip_positions  : array of dead strip positions (mm), or None
    strip_pitch           : strip pitch (mm)

    Returns
    -------
    tracks     : list of np.ndarray of hit indices (one per track)
    noise_mask : boolean array, True for hits not on any track
    """
    n = len(pos_mm)
    if n == 0:
        return [], np.zeros(0, dtype=bool)

    assigned = np.zeros(n, dtype=bool)
    tracks = []

    order = np.argsort(time_us)[::-1]   # late → early

    for seed_i in order:
        if assigned[seed_i]:
            continue

        track = [seed_i]
        assigned[seed_i] = True
        real_missed = 0          # misses NOT attributable to dead strips
        total_extra_missed = 0   # dead-strip misses (unlimited)

        while True:
            effective_missed = real_missed + total_extra_missed
            nxt = _find_next_hit(pos_mm, time_us, assigned, track,
                                  road_width_mm, road_width_seed_mm,
                                  max_time_gap_us, effective_missed)

            if nxt is not None:
                # ── Strip-gap check ───────────────────────────────────────────
                # Count how many *live* strips were skipped between the current
                # head and the candidate hit.  Dead strips in the gap are free.
                p_head = pos_mm[track[-1]]
                p_new = pos_mm[nxt]
                n_strips_between = max(0.0, abs(p_new - p_head) / strip_pitch - 1.0)
                n_dead_between = _count_dead_strips_between(
                    p_head, p_new, dead_strip_positions, strip_pitch)
                live_strip_gap = max(0, round(n_strips_between) - n_dead_between)

                if live_strip_gap <= max_strip_gap:
                    track.append(nxt)
                    assigned[nxt] = True
                    real_missed = 0
                    total_extra_missed = 0
                else:
                    # Hit found but too many live strips skipped — treat as a
                    # real miss so the track eventually stops.  The hit itself
                    # stays unassigned so another seed can claim it.
                    real_missed += 1
                    if real_missed > max_missed:
                        break
            else:
                # Determine whether this miss is at a dead strip
                slope, intercept = _fit_slope(time_us[track], pos_mm[track])
                is_dead_miss = False
                if slope is not None and dead_strip_positions is not None:
                    t_head = time_us[track[-1]]
                    t_step = t_head - max_time_gap_us * (effective_missed + 1)
                    p_step = slope * t_step + intercept
                    is_dead_miss = _near_dead_strip(p_step, dead_strip_positions, strip_pitch)

                if is_dead_miss:
                    total_extra_missed += 1
                    # Safety cap: don't loop forever over consecutive dead strips
                    if total_extra_missed > len(dead_strip_positions) + max_missed:
                        break
                else:
                    real_missed += 1
                    if real_missed > max_missed:
                        break

        if len(track) >= min_hits:
            tracks.append(np.array(track))
        else:
            for idx in track:
                assigned[idx] = False

    return tracks, ~assigned


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_TRACK_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _track_color(i):
    return _TRACK_COLORS[i % len(_TRACK_COLORS)]


def plot_overview_figure(pos_x, time_x, keep_x,
                         pos_y, time_y, keep_y,
                         run, subrun, event):
    """2×2 overview: raw hits and after-isolation for both projections."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='row')
    fig.suptitle(f'Hit overview  —  Event {event}  |  {run}/{subrun}', fontsize=12)

    for row, (proj, pos_raw, time_raw, keep) in enumerate([
        ('X', pos_x, time_x, keep_x),
        ('Y', pos_y, time_y, keep_y),
    ]):
        # Raw
        axs[row, 0].scatter(time_raw, pos_raw, s=8, c='steelblue', alpha=0.8)
        axs[row, 0].set_title(f'{proj}: Raw hits  (n={len(pos_raw)})')

        # After isolation
        axs[row, 1].scatter(time_raw[~keep], pos_raw[~keep],
                            s=15, c='tomato', marker='x', zorder=3,
                            label=f'Removed ({(~keep).sum()})')
        axs[row, 1].scatter(time_raw[keep], pos_raw[keep],
                            s=8, c='steelblue', alpha=0.8, zorder=2,
                            label=f'Kept ({keep.sum()})')
        axs[row, 1].set_title(f'{proj}: After isolation filter')
        axs[row, 1].legend(fontsize=8)

        for col in range(2):
            axs[row, col].set_xlabel('Drift time (μs)')
            axs[row, col].set_ylabel(f'{proj} position (mm)')

    plt.tight_layout()
    return fig


def plot_track_figure(pos_filt, time_filt, tracks, noise_mask,
                      dead_strip_positions, strip_pitch, all_strips, proj_label,
                      run, subrun, event):
    """
    Large standalone figure for one projection showing track-finding result.

    Dead strip positions are drawn as horizontal bands.
    Fit lines are extended slightly beyond the hit range.
    No legend (clean display).
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(
        f'{proj_label} projection — Track finding  |  Event {event}  |  {run}/{subrun}',
        fontsize=12
    )

    # Strip grid lines — one per strip position, very faint
    ax.set_yticks(all_strips, minor=True)
    ax.grid(True, which='minor', axis='y', color='grey', linewidth=0.4, alpha=0.3)
    ax.tick_params(which='minor', length=0)

    # Unassigned / noise hits
    ax.scatter(time_filt[noise_mask], pos_filt[noise_mask],
               s=12, c='lightgray', zorder=2)

    # Tracks
    for i, idx in enumerate(tracks):
        col = _track_color(i)
        t_tr = time_filt[idx]
        p_tr = pos_filt[idx]
        ax.scatter(t_tr, p_tr, s=30, color=col, zorder=4)

        if len(idx) >= 2:
            slope, intercept = _fit_slope(t_tr, p_tr)
            dt = (t_tr.max() - t_tr.min()) * 0.1   # 10% extension each side
            t_fit = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 200)
            ax.plot(t_fit, slope * t_fit + intercept,
                    color=col, lw=1.5, ls='--', zorder=3,
                    label=f'Track {i+1}: {len(idx)} hits')

    # Dead strip bands
    if dead_strip_positions is not None and len(dead_strip_positions) > 0:
        hw = strip_pitch / 2.0
        for dp in dead_strip_positions:
            ax.axhspan(dp - hw, dp + hw, color='salmon', alpha=0.25, zorder=1)

    ax.set_xlabel('Drift time (μs)', fontsize=11)
    ax.set_ylabel(f'{proj_label} position (mm)', fontsize=11)

    # Secondary y-axis: strip number (0-indexed from the first strip in the map)
    strip_origin = all_strips[0]
    ax_strip = ax.twinx()
    p_lim = ax.get_ylim()
    ax_strip.set_ylim((p_lim[0] - strip_origin) / strip_pitch,
                      (p_lim[1] - strip_origin) / strip_pitch)
    ax_strip.set_ylabel(f'{proj_label} strip number', fontsize=11)

    # Parameter box (bottom-right)
    param_text = (
        f'Amp ≥ {MIN_HIT_AMP}  |  '
        f'Iso: ±{ISO_POS_MM} mm, ±{ISO_TIME_NS} ns  |  '
        f'Road: {ROAD_WIDTH_MM} mm  |  '
        f'Max gap: {MAX_TIME_GAP_US} μs  |  '
        f'Max missed: {MAX_MISSED}  |  '
        f'Min hits: {MIN_TRACK_HITS}'
    )
    ax.annotate(param_text, xy=(0.01, 0.01), xycoords='axes fraction',
                fontsize=8, va='bottom',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def print_track_summary(tracks, pos_mm, time_us, label):
    print(f'\n── {label} tracks ────────────────────────────────────────')
    if not tracks:
        print('  No tracks found.')
        return
    for i, idx in enumerate(tracks):
        t_tr, p_tr = time_us[idx], pos_mm[idx]
        slope, _ = _fit_slope(t_tr, p_tr)
        angle_deg = (np.degrees(np.arctan(slope / DRIFT_VELOCITY_MM_US))
                     if slope is not None else float('nan'))
        print(f'  Track {i+1}: {len(idx):3d} hits | '
              f't=[{t_tr.min():.3f}, {t_tr.max():.3f}] μs | '
              f'pos=[{p_tr.min():.1f}, {p_tr.max():.1f}] mm | '
              f'slope={slope:.2f} mm/μs | angle≈{angle_deg:.1f}°')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _process_event(args):
    """
    Worker function: process a single event and return a list of track-record dicts.
    Must be a module-level function so it is picklable by ProcessPoolExecutor.
    """
    evt_id, pos_x, time_x, pos_y, time_y, dead_x, dead_y, pitch_x, pitch_y = args
    iso_time_us = ISO_TIME_NS / 1000.0
    records = []

    for proj, pos, time, dead, pitch in [
        ('x', pos_x, time_x, dead_x, pitch_x),
        ('y', pos_y, time_y, dead_y, pitch_y),
    ]:
        if len(pos) < MIN_TRACK_HITS:
            continue
        keep = remove_isolated_hits(pos, time, ISO_POS_MM, iso_time_us)
        pos_f, time_f = pos[keep], time[keep]
        if len(pos_f) < MIN_TRACK_HITS:
            continue
        tracks, _ = find_tracks_1d(pos_f, time_f,
                                    dead_strip_positions=dead, strip_pitch=pitch)
        for idx in tracks:
            t_tr = time_f[idx]
            p_tr = pos_f[idx]
            slope, _ = _fit_slope(t_tr, p_tr)
            angle = (np.degrees(np.arctan(slope / DRIFT_VELOCITY_MM_US))
                     if slope is not None else np.nan)
            records.append({
                'event_id':   evt_id,
                'projection': proj,
                'n_hits':     len(idx),
                'slope':      slope,
                'angle_deg':  angle,
                'time_min':   t_tr.min(),
                'time_max':   t_tr.max(),
                'time_span':  t_tr.max() - t_tr.min(),
                'pos_min':    p_tr.min(),
                'pos_max':    p_tr.max(),
                'pos_span':   p_tr.max() - p_tr.min(),
            })
    return records


def collect_all_tracks(df_pos, dead_x, dead_y, pitch_x, pitch_y):
    """
    Run isolation filter + track finding on every event in df_pos in parallel.

    df_pos must already have x_position_mm / y_position_mm columns (from
    add_xy_pos) and be amplitude pre-filtered.

    Returns a DataFrame with one row per found track.
    """
    n_workers = max(1, os.cpu_count() - FREE_THREADS)
    print(f'  Using {n_workers} worker processes ({os.cpu_count()} cores, '
          f'{FREE_THREADS} kept free)')

    # Pre-extract per-event arrays once so workers receive compact numpy data
    args_list = []
    for evt_id, df_evt in df_pos.groupby('eventId'):
        dx = df_evt[df_evt['x_position_mm'].notna()]
        dy = df_evt[df_evt['y_position_mm'].notna()]
        args_list.append((
            evt_id,
            dx['x_position_mm'].to_numpy(), raw_time_to_us(dx['time'].to_numpy()),
            dy['y_position_mm'].to_numpy(), raw_time_to_us(dy['time'].to_numpy()),
            dead_x, dead_y, pitch_x, pitch_y,
        ))

    records = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for event_records in tqdm(executor.map(_process_event, args_list),
                                  total=len(args_list), desc='Events', unit='evt'):
            records.extend(event_records)

    return pd.DataFrame(records)


METRIC_WINDOWS = [
    (0,    1,    '0–1 μs'),
    (1,    2,    '1–2 μs'),
    (2,    3,    '2–3 μs'),
    (3,    None, '3+ μs'),
    (0,    None, 'All'),
]
_WINDOW_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']


def _window_filter(df, lo, hi):
    m = df['time_min'] >= lo
    if hi is not None:
        m &= df['time_min'] < hi
    return df[m]


def plot_track_metrics(df_tracks, run, subrun):
    """
    Combined overview: all start-time windows overlaid on each metric panel.
    X and Y projections are merged within each window.
    """
    WINDOWS = METRIC_WINDOWS
        (0,    None, 'All'),
    ]
    COLORS = _WINDOW_COLORS

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Track metrics by start-time window  —  {run}/{subrun}', fontsize=12)

    def _hist(ax, col, xlabel, bins=30, integer=False):
        # Build global bins from all data so windows are directly comparable
        all_vals = df_tracks[col].dropna()
        if len(all_vals) == 0:
            return
        if integer:
            lo_v, hi_v = int(all_vals.min()), int(all_vals.max())
            bins = np.arange(lo_v - 0.5, hi_v + 1.5, 1)
        else:
            bins = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

        for (lo, hi, label), color in zip(WINDOWS, COLORS):
            vals = _window_filter(df_tracks, lo, hi)[col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, histtype='step', lw=1.5,
                    color=color, label=f'{label} (n={len(vals)})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Tracks')
        ax.legend(fontsize=7)

    _hist(axs[0, 0], 'n_hits',    'Hits per track',        integer=True)
    _hist(axs[0, 1], 'angle_deg', 'Track angle (°)',       bins=40)
    _hist(axs[0, 2], 'time_span', 'Time span (μs)',        bins=30)
    _hist(axs[1, 0], 'pos_span',  'Position span (mm)',    bins=30)
    _hist(axs[1, 1], 'time_min',  'Track start time (μs)', bins=30)

    # Tracks per event
    ax = axs[1, 2]
    all_n = df_tracks.groupby('event_id').size()
    max_n = all_n.max() if len(all_n) else 1
    bins_n = np.arange(0.5, max_n + 1.5, 1)
    for (lo, hi, label), color in zip(WINDOWS, COLORS):
        sub = _window_filter(df_tracks, lo, hi)
        n_per_evt = sub.groupby('event_id').size()
        if len(n_per_evt) == 0:
            continue
        ax.hist(n_per_evt, bins=bins_n, histtype='step', lw=1.5,
                color=color, label=f'{label} (n={len(n_per_evt)} evts)')
    ax.set_xlabel('Tracks per event')
    ax.set_ylabel('Events')
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_track_metrics_window(df_tracks, lo, hi, label, run, subrun):
    """
    Metric histograms for a single start-time window with X and Y overlaid.
    Uses the same global bin edges as the combined figure for comparability.
    """
    df_win = _window_filter(df_tracks, lo, hi)
    df_x = df_win[df_win['projection'] == 'x']
    df_y = df_win[df_win['projection'] == 'y']

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Track metrics  —  {label}  —  {run}/{subrun}\n'
        f'X: {len(df_x)} tracks   Y: {len(df_y)} tracks',
        fontsize=12
    )

    kw = dict(alpha=0.6, histtype='stepfilled')

    def _hist(ax, col, xlabel, bins=30, integer=False):
        all_vals = df_tracks[col].dropna()   # global range for consistent bins
        if len(all_vals) == 0:
            return
        if integer:
            lo_v, hi_v = int(all_vals.min()), int(all_vals.max())
            bins = np.arange(lo_v - 0.5, hi_v + 1.5, 1)
        else:
            bins = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
        for df_p, proj_label, color in [(df_x, 'X', 'steelblue'), (df_y, 'Y', 'darkorange')]:
            vals = df_p[col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, color=color, label=f'{proj_label} (n={len(vals)})', **kw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Tracks')
        ax.legend(fontsize=8)

    _hist(axs[0, 0], 'n_hits',    'Hits per track',        integer=True)
    _hist(axs[0, 1], 'angle_deg', 'Track angle (°)',       bins=40)
    _hist(axs[0, 2], 'time_span', 'Time span (μs)',        bins=30)
    _hist(axs[1, 0], 'pos_span',  'Position span (mm)',    bins=30)
    _hist(axs[1, 1], 'time_min',  'Track start time (μs)', bins=30)

    ax = axs[1, 2]
    all_n = df_tracks.groupby('event_id').size()
    max_n = all_n.max() if len(all_n) else 1
    bins_n = np.arange(0.5, max_n + 1.5, 1)
    for df_p, proj_label, color in [(df_x, 'X', 'steelblue'), (df_y, 'Y', 'darkorange')]:
        n_per_evt = df_p.groupby('event_id').size()
        if len(n_per_evt) == 0:
            continue
        ax.hist(n_per_evt, bins=bins_n, color=color,
                label=f'{proj_label} (n={len(n_per_evt)} evts)', **kw)
    ax.set_xlabel('Tracks per event')
    ax.set_ylabel('Events')
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def main():
    iso_time_us = ISO_TIME_NS / 1000.0

    # ── Detector / strip geometry ─────────────────────────────────────────────
    print(f'Loading subrun {RUN}/{SUBRUN} …')
    df_all, det = load_subrun(BASE_PATH, RUN, SUBRUN, list(FEU_NUMS.keys()))

    all_x_strips, all_y_strips = get_strip_positions(det)
    pitch_x, pitch_y = get_strip_pitch(all_x_strips, all_y_strips)
    print(f'Strip pitch:  X = {pitch_x:.4f} mm  |  Y = {pitch_y:.4f} mm')
    print(f'Strip count:  X = {len(all_x_strips)}  |  Y = {len(all_y_strips)}')

    # ── Dead-strip analysis ────────────────────────────────────────────────────
    dead_x, dead_y, x_rates, y_rates, thresh_x, thresh_y, n_events = find_dead_strips(
        BASE_PATH, RUN, SUBRUN, FEU_NUMS, det,
        all_x_strips, all_y_strips
    )

    if RUN_MODE == 'single':
        # ── Single-event visualisation ────────────────────────────────────────
        plot_dead_strip_figure(x_rates, y_rates, dead_x, dead_y,
                               thresh_x, thresh_y, n_events, pitch_x, pitch_y, RUN, SUBRUN)

        print(f'\nProcessing event {EVENT} …')
        df = df_all[(df_all['eventId'] == EVENT) & (df_all['amplitude'] >= MIN_HIT_AMP)].copy()
        df = add_xy_pos(df, det)

        df_x = df[df['x_position_mm'].notna()].copy().reset_index(drop=True)
        df_y = df[df['y_position_mm'].notna()].copy().reset_index(drop=True)
        df_x['time_us'] = raw_time_to_us(df_x['time'])
        df_y['time_us'] = raw_time_to_us(df_y['time'])

        pos_x  = df_x['x_position_mm'].to_numpy()
        time_x = df_x['time_us'].to_numpy()
        pos_y  = df_y['y_position_mm'].to_numpy()
        time_y = df_y['time_us'].to_numpy()
        print(f'Raw hits: X={len(pos_x)}  Y={len(pos_y)}')

        keep_x = remove_isolated_hits(pos_x, time_x, ISO_POS_MM, iso_time_us)
        keep_y = remove_isolated_hits(pos_y, time_y, ISO_POS_MM, iso_time_us)
        print(f'After isolation: X={keep_x.sum()} kept ({(~keep_x).sum()} removed)  '
              f'Y={keep_y.sum()} kept ({(~keep_y).sum()} removed)')

        pos_xf, time_xf = pos_x[keep_x], time_x[keep_x]
        pos_yf, time_yf = pos_y[keep_y], time_y[keep_y]

        tracks_x, noise_x = find_tracks_1d(pos_xf, time_xf,
                                            dead_strip_positions=dead_x, strip_pitch=pitch_x)
        tracks_y, noise_y = find_tracks_1d(pos_yf, time_yf,
                                            dead_strip_positions=dead_y, strip_pitch=pitch_y)

        print_track_summary(tracks_x, pos_xf, time_xf, 'X')
        print_track_summary(tracks_y, pos_yf, time_yf, 'Y')

        plot_overview_figure(pos_x, time_x, keep_x,
                             pos_y, time_y, keep_y,
                             RUN, SUBRUN, EVENT)
        plot_track_figure(pos_xf, time_xf, tracks_x, noise_x,
                          dead_x, pitch_x, all_x_strips, 'X', RUN, SUBRUN, EVENT)
        plot_track_figure(pos_yf, time_yf, tracks_y, noise_y,
                          dead_y, pitch_y, all_y_strips, 'Y', RUN, SUBRUN, EVENT)

    else:
        # ── All-events metrics ────────────────────────────────────────────────
        print(f'\nRunning over all events in {RUN}/{SUBRUN} …')
        df_pos = df_all[df_all['amplitude'] >= MIN_HIT_AMP].copy()
        print(f'Mapping {len(df_pos)} hits to strip positions (this may take a moment) …')
        df_pos = add_xy_pos(df_pos, det)

        df_tracks = collect_all_tracks(df_pos, dead_x, dead_y, pitch_x, pitch_y)
        print(f'\nTotal tracks found: {len(df_tracks)}  '
              f'(X={len(df_tracks[df_tracks["projection"]=="x"])}  '
              f'Y={len(df_tracks[df_tracks["projection"]=="y"])})')

        plot_track_metrics(df_tracks, RUN, SUBRUN)
        for lo, hi, label in METRIC_WINDOWS:
            plot_track_metrics_window(df_tracks, lo, hi, label, RUN, SUBRUN)

    plt.show()
    print('\ndonzo')


if __name__ == '__main__':
    main()
