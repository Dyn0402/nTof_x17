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
# RUN = 'run_34'
# SUBRUN = 'resist_425V_drift_600V'
# RUN = 'run_131'
# SUBRUN = 'resist_500V_drift_1000V'
# RUN = 'run_64'
# SUBRUN = 'resist_650V_drift_600V'
RUN = 'run_80'
SUBRUN = 'resist_640V_drift_1000V'
EVENT = 17
MIN_HIT_AMP = 200
FEU_NUMS = {4: 'y', 5: 'x'}

# ─── Physics parameters ───────────────────────────────────────────────────────
DRIFT_VELOCITY_MM_US = 22.0   # mm/μs  (20 μm/ns = 20 mm/μs)
DRIFT_GAP_MM = 30.0           # total drift length in mm
FULL_DRIFT_GAP_TIME = DRIFT_GAP_MM / DRIFT_VELOCITY_MM_US  # μs for electrons to drift across the full gap

# ─── Run mode ────────────────────────────────────────────────────────────────
RUN_MODE = 'single'          # 'single' → visualise one event; 'all' → metrics over full subrun
FREE_THREADS = 1          # CPU cores to leave free; workers = os.cpu_count() - FREE_THREADS

# ─── Dead-strip finder ────────────────────────────────────────────────────────
DEAD_STRIP_AMP_THRESHOLD = 500          # amplitude cut for dead-strip analysis
DEAD_STRIP_RATE_FRACTION = 0.50         # fraction of median below which a strip is dead

# ─── Isolation filter ─────────────────────────────────────────────────────────
ISO_POS_MM = 10.0             # mm: neighbourhood radius in position
ISO_TIME_NS = 500.0           # ns: neighbourhood radius in time

# ─── Hit quality exclusion (amp+ToT box excluded from tracking) ──────────────
EXCLUDE_AMP_MAX = 600         # } hits with amp <= this AND ToT <= EXCLUDE_TOT_MAX
EXCLUDE_TOT_MAX = 2500        # } are excluded from tracking (shown grey on track plot)

# ─── Track-finding parameters ────────────────────────────────────────────────
ROAD_WIDTH_MM = 2.0           # mm:  max distance of a hit from the track line
ROAD_WIDTH_SEED_MM = 5.0      # mm:  road width when slope is not yet known (first step)
MAX_TIME_GAP_US = 0.1         # μs:  max time gap between consecutive track hits
MIN_TRACK_HITS = 2            # minimum hits to accept a track
MAX_MISSED = 1                # max consecutive time-step attempts with no hit before stopping
MAX_STRIP_GAP = 2             # max consecutive *live* strips skipped between accepted hits


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

        amp_x  = df_x['amplitude'].to_numpy()
        tot_x  = df_x['time_over_threshold'].to_numpy()
        lmax_x = df_x['local_max'].to_numpy()
        intg_x = df_x['integral'].to_numpy()

        amp_y  = df_y['amplitude'].to_numpy()
        tot_y  = df_y['time_over_threshold'].to_numpy()
        lmax_y = df_y['local_max'].to_numpy()
        intg_y = df_y['integral'].to_numpy()

        print(f'Raw hits: X={len(pos_x)}  Y={len(pos_y)}')

        plot_hit_segments_figure(
            pos_x, time_x, amp_x, tot_x,
            pos_y, time_y, amp_y, tot_y,
            RUN, SUBRUN, EVENT)

        keep_x = remove_isolated_hits(pos_x, time_x, ISO_POS_MM, iso_time_us)
        keep_y = remove_isolated_hits(pos_y, time_y, ISO_POS_MM, iso_time_us)
        print(f'After isolation: X={keep_x.sum()} kept ({(~keep_x).sum()} removed)  '
              f'Y={keep_y.sum()} kept ({(~keep_y).sum()} removed)')

        pos_xf, time_xf = pos_x[keep_x], time_x[keep_x]
        pos_yf, time_yf = pos_y[keep_y], time_y[keep_y]
        amp_xf, tot_xf = amp_x[keep_x], tot_x[keep_x]
        amp_yf, tot_yf = amp_y[keep_y], tot_y[keep_y]

        # ── Amp+ToT exclusion ─────────────────────────────────────────────
        excl_x = (amp_xf <= EXCLUDE_AMP_MAX) & (tot_xf <= EXCLUDE_TOT_MAX)
        excl_y = (amp_yf <= EXCLUDE_AMP_MAX) & (tot_yf <= EXCLUDE_TOT_MAX)
        print(f'Excluded (amp≤{EXCLUDE_AMP_MAX} & ToT≤{EXCLUDE_TOT_MAX}): '
              f'X={excl_x.sum()}  Y={excl_y.sum()}')

        tracks_x, noise_x = find_tracks_1d(pos_xf, time_xf,
                                            dead_strip_positions=dead_x, strip_pitch=pitch_x,
                                            excl_mask=excl_x)
        tracks_y, noise_y = find_tracks_1d(pos_yf, time_yf,
                                            dead_strip_positions=dead_y, strip_pitch=pitch_y,
                                            excl_mask=excl_y)

        print_track_summary(tracks_x, pos_xf, time_xf, 'X (pass 1)')
        print_track_summary(tracks_y, pos_yf, time_yf, 'Y (pass 1)')

        # ── Pass-2: extend existing tracks + new seeds using all hits ─────────
        tracks_x_ext, tracks_x2 = find_tracks_1d_pass2(
            tracks_x, pos_xf, time_xf,
            dead_strip_positions=dead_x, strip_pitch=pitch_x, excl_mask=excl_x)
        tracks_y_ext, tracks_y2 = find_tracks_1d_pass2(
            tracks_y, pos_yf, time_yf,
            dead_strip_positions=dead_y, strip_pitch=pitch_y, excl_mask=excl_y)

        # Recompute noise masks: hits on any pass-2 track (extended or new)
        on_p2_x = np.zeros(len(pos_xf), dtype=bool)
        for t in tracks_x_ext + tracks_x2:
            on_p2_x[t] = True
        on_p2_y = np.zeros(len(pos_yf), dtype=bool)
        for t in tracks_y_ext + tracks_y2:
            on_p2_y[t] = True
        noise_x2 = ~on_p2_x
        noise_y2 = ~on_p2_y

        print_track_summary(tracks_x_ext, pos_xf, time_xf, 'X (pass 2 extended)')
        if tracks_x2:
            print_track_summary(tracks_x2, pos_xf, time_xf, 'X (pass 2 new)')
        print_track_summary(tracks_y_ext, pos_yf, time_yf, 'Y (pass 2 extended)')
        if tracks_y2:
            print_track_summary(tracks_y2, pos_yf, time_yf, 'Y (pass 2 new)')

        plot_overview_figure(pos_x, time_x, keep_x,
                             pos_y, time_y, keep_y,
                             RUN, SUBRUN, EVENT)
        for vals_x, vals_y, var_label, cmap in [
            (amp_x,  amp_y,  'Amplitude (ADC)',          'viridis'),
            (tot_x,  tot_y,  'Time over threshold (ns)', 'plasma'),
            (lmax_x, lmax_y, 'Local max (ADC)',          'inferno'),
            (intg_x, intg_y, 'Integral (ADC·samples)',   'cividis'),
        ]:
            plot_hit_colorcoded_figure(
                pos_x, time_x, vals_x,
                pos_y, time_y, vals_y,
                var_label, cmap, RUN, SUBRUN, EVENT)
        plot_amp_vs_tot_figure(amp_x, tot_x, amp_y, tot_y, RUN, SUBRUN, EVENT)
        plot_amp_vs_tot_figure(amp_x, tot_x, amp_y, tot_y, RUN, SUBRUN, EVENT,
                               amp_max=600, tot_max=3000)
        _explorer = launch_region_explorer(
            pos_x, time_x, amp_x, tot_x,
            pos_y, time_y, amp_y, tot_y,
            RUN, SUBRUN, EVENT, amp_max=600, tot_max=3000)
        plot_track_figure(pos_xf, time_xf, tracks_x_ext, noise_x2,
                          dead_x, pitch_x, all_x_strips, 'X', RUN, SUBRUN, EVENT,
                          excl_mask=excl_x, tracks_p2=tracks_x2)
        plot_track_figure(pos_yf, time_yf, tracks_y_ext, noise_y2,
                          dead_y, pitch_y, all_y_strips, 'Y', RUN, SUBRUN, EVENT,
                          excl_mask=excl_y, tracks_p2=tracks_y2)

        # ── X–Y pairing ───────────────────────────────────────────────────────
        all_tracks_x = tracks_x_ext + tracks_x2
        all_tracks_y = tracks_y_ext + tracks_y2

        objects_x = _build_objects(all_tracks_x, pos_xf, time_xf,
                                   pos_x, time_x, keep_x, MAX_TIME_GAP_US)
        objects_y = _build_objects(all_tracks_y, pos_yf, time_yf,
                                   pos_y, time_y, keep_y, MAX_TIME_GAP_US)
        pairs, unmatched_x, unmatched_y = pair_xy_objects(objects_x, objects_y)
        print_xy_pairs(pairs, unmatched_x, unmatched_y)
        plot_xy_pairs_figure(pairs, unmatched_x, unmatched_y, RUN, SUBRUN, EVENT)

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
    _set_window_title(fig, f'Dead strips — {run}/{subrun}')
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
    _RankWarning = getattr(np.exceptions if hasattr(np, 'exceptions') else np, 'RankWarning', Warning)
    with warnings.catch_warnings():
        warnings.simplefilter('error', _RankWarning)
        try:
            coeffs = np.polyfit(time_us, pos_mm, 1)
        except _RankWarning:
            # print(f'WARNING: polyfit poorly conditioned — time_us={time_us}  pos_mm={pos_mm}')
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


def _find_next_hit(pos_mm, time_us, track_indices,
                   road_width_mm, road_width_seed_mm,
                   max_time_gap_us, max_missed,
                   assigned=None, forward=False):
    """
    Find the best next hit to extend the current track.

    forward=False (default): extend toward earlier times from track[-1].
    forward=True           : extend toward later  times from track[0].

    With >= 2 track hits, uses the linear-fit extrapolation and road_width_mm.
    With only 1 hit (seed), uses road_width_seed_mm and no slope constraint.
    assigned : optional boolean array; True entries are excluded from candidacy.
    Returns the index of the chosen hit, or None.
    """
    slope, intercept = _fit_slope(time_us[track_indices], pos_mm[track_indices])

    if forward:
        t_head = time_us[track_indices[0]]   # latest hit is track[0]
        t_bound = t_head + max_time_gap_us * (max_missed + 1)
        mask = (time_us > t_head) & (time_us <= t_bound)
    else:
        t_head = time_us[track_indices[-1]]  # earliest hit is track[-1]
        t_bound = t_head - max_time_gap_us * (max_missed + 1)
        mask = (time_us < t_head) & (time_us >= t_bound)

    if slope is None:
        road = road_width_seed_mm
        p_ref = pos_mm[track_indices[0] if forward else track_indices[-1]]
    else:
        road = road_width_mm

    if assigned is not None:
        mask &= ~assigned
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
    if forward:
        return int(valid[np.argmin(time_us[valid])])  # closest forward in time
    else:
        return int(valid[np.argmax(time_us[valid])])  # closest backward in time


def _grow_track(track, pos_mm, time_us,
                road_width_mm, road_width_seed_mm,
                max_time_gap_us, max_missed, max_strip_gap,
                dead_strip_positions, strip_pitch,
                assigned=None, forward=False):
    """
    Grow a track list (latest→earliest index order) in one direction.

    forward=False (default): extend toward earlier times, appending to track[-1].
    forward=True           : extend toward later  times, prepending to track[0].

    assigned : optional boolean array; when given those hits are not candidates
               (pass None in pass-2 to allow sharing of already-assigned hits).
    Modifies *track* in place and returns it.
    """
    real_missed = 0
    total_extra_missed = 0
    while True:
        effective_missed = real_missed + total_extra_missed
        nxt = _find_next_hit(pos_mm, time_us, track,
                              road_width_mm, road_width_seed_mm,
                              max_time_gap_us, effective_missed,
                              assigned=assigned, forward=forward)
        if nxt is not None:
            p_head = pos_mm[track[0] if forward else track[-1]]
            p_new  = pos_mm[nxt]
            n_strips_between = max(0.0, abs(p_new - p_head) / strip_pitch - 1.0)
            n_dead_between = _count_dead_strips_between(
                p_head, p_new, dead_strip_positions, strip_pitch)
            live_strip_gap = max(0, round(n_strips_between) - n_dead_between)
            if live_strip_gap <= max_strip_gap:
                if forward:
                    track.insert(0, nxt)   # prepend: track stays latest→earliest
                else:
                    track.append(nxt)
                real_missed = 0
                total_extra_missed = 0
            else:
                real_missed += 1
                if real_missed > max_missed:
                    break
        else:
            slope, intercept = _fit_slope(time_us[track], pos_mm[track])
            is_dead_miss = False
            if slope is not None and dead_strip_positions is not None:
                t_head = time_us[track[0] if forward else track[-1]]
                sign = +1 if forward else -1
                t_step = t_head + sign * max_time_gap_us * (effective_missed + 1)
                p_step = slope * t_step + intercept
                is_dead_miss = _near_dead_strip(p_step, dead_strip_positions, strip_pitch)
            if is_dead_miss:
                total_extra_missed += 1
                if total_extra_missed > len(dead_strip_positions) + max_missed:
                    break
            else:
                real_missed += 1
                if real_missed > max_missed:
                    break
    return track


def find_tracks_1d(pos_mm, time_us,
                   road_width_mm=ROAD_WIDTH_MM,
                   road_width_seed_mm=ROAD_WIDTH_SEED_MM,
                   max_time_gap_us=MAX_TIME_GAP_US,
                   min_hits=MIN_TRACK_HITS,
                   max_missed=MAX_MISSED,
                   max_strip_gap=MAX_STRIP_GAP,
                   dead_strip_positions=None,
                   strip_pitch=0.8,
                   excl_mask=None):
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

    excl_mask : optional boolean array (same length as pos_mm).  True marks a
                hit as excluded from seeding; excluded hits may still join an
                existing track but cannot start one.  Each hit is assigned to
                at most one accepted track (exclusive ownership).

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
    excl_mask             : boolean array of hits excluded from seeding

    Returns
    -------
    tracks     : list of np.ndarray of hit indices (one per track)
    noise_mask : boolean array, True for hits not on any accepted track
    """
    n = len(pos_mm)
    if n == 0:
        return [], np.zeros(0, dtype=bool)

    seeded   = np.zeros(n, dtype=bool)   # prevents retrying the same seed
    assigned = np.zeros(n, dtype=bool)   # exclusive: locked to an accepted track
    tracks   = []

    kw = dict(road_width_mm=road_width_mm,
              road_width_seed_mm=road_width_seed_mm,
              max_time_gap_us=max_time_gap_us,
              max_missed=max_missed,
              max_strip_gap=max_strip_gap,
              dead_strip_positions=dead_strip_positions,
              strip_pitch=strip_pitch)

    order = np.argsort(time_us)[::-1]   # late → early

    for seed_i in order:
        if seeded[seed_i] or assigned[seed_i]:
            continue
        if excl_mask is not None and excl_mask[seed_i]:
            continue  # excluded hits cannot seed a track

        seeded[seed_i] = True
        track = [seed_i]
        _grow_track(track, pos_mm, time_us, assigned=assigned, **kw)

        if len(track) >= min_hits:
            tracks.append(np.array(track))
            assigned[track] = True

    return tracks, ~assigned


def find_tracks_1d_pass2(tracks_p1, pos_mm, time_us,
                          road_width_mm=ROAD_WIDTH_MM,
                          road_width_seed_mm=ROAD_WIDTH_SEED_MM,
                          max_time_gap_us=MAX_TIME_GAP_US,
                          min_hits=MIN_TRACK_HITS,
                          max_missed=MAX_MISSED,
                          max_strip_gap=MAX_STRIP_GAP,
                          dead_strip_positions=None,
                          strip_pitch=0.8,
                          excl_mask=None):
    """
    Second-pass tracking: all hits (including already-assigned) are candidates.

    1. Each pass-1 track is extended backward using the full hit pool.
    2. Unassigned hits (not on any pass-1 track) seed new tracks, again using
       the full hit pool so they can bridge through pass-1 hits.

    excl_mask is still respected for seeding in step 2.

    Parameters
    ----------
    tracks_p1 : list of np.ndarray returned by find_tracks_1d
    (remaining parameters identical to find_tracks_1d)

    Returns
    -------
    extended   : list of np.ndarray, same length as tracks_p1; each track may
                 be longer than its pass-1 counterpart (or identical if no
                 extension was found)
    new_tracks : list of np.ndarray of additional tracks seeded from
                 previously-unassigned hits
    """
    n = len(pos_mm)
    if n == 0:
        return [np.array(t) for t in tracks_p1], []

    assigned_p1 = np.zeros(n, dtype=bool)
    for t in tracks_p1:
        assigned_p1[t] = True

    kw = dict(road_width_mm=road_width_mm,
              road_width_seed_mm=road_width_seed_mm,
              max_time_gap_us=max_time_gap_us,
              max_missed=max_missed,
              max_strip_gap=max_strip_gap,
              dead_strip_positions=dead_strip_positions,
              strip_pitch=strip_pitch)

    # ── 1. Extend existing tracks (no assignment constraint, both directions) ──
    extended = []
    for track_arr in tracks_p1:
        # Sort latest→earliest so backward growth starts from track[-1]
        track = sorted(list(track_arr), key=lambda i: -time_us[i])
        _grow_track(track, pos_mm, time_us, assigned=None, forward=False, **kw)  # earlier
        _grow_track(track, pos_mm, time_us, assigned=None, forward=True,  **kw)  # later
        # Remove duplicates (should not occur, but guard)
        seen, dedup = set(), []
        for idx in track:
            if idx not in seen:
                seen.add(idx)
                dedup.append(idx)
        extended.append(np.array(dedup))

    # ── 2. New tracks from unassigned seeds ───────────────────────────────────
    seeded = assigned_p1.copy()   # skip hits already on a pass-1 track as seeds
    new_tracks = []
    order = np.argsort(time_us)[::-1]   # late → early

    for seed_i in order:
        if seeded[seed_i]:
            continue
        if excl_mask is not None and excl_mask[seed_i]:
            continue
        seeded[seed_i] = True
        track = [seed_i]
        _grow_track(track, pos_mm, time_us, assigned=None, forward=False, **kw)  # earlier
        _grow_track(track, pos_mm, time_us, assigned=None, forward=True,  **kw)  # later
        if len(track) >= min_hits:
            new_tracks.append(np.array(track))

    # ── 3. Remove tracks whose hits are a strict subset of another track ──────
    all_tracks = extended + new_tracks
    sets = [set(t.tolist()) for t in all_tracks]
    keep = []
    for i, s_i in enumerate(sets):
        contained = any(s_i < s_j for j, s_j in enumerate(sets) if j != i)
        if not contained:
            keep.append(i)

    n_ext = len(extended)
    extended   = [all_tracks[i] for i in keep if i < n_ext]
    new_tracks = [all_tracks[i] for i in keep if i >= n_ext]

    return extended, new_tracks


# ─────────────────────────────────────────────────────────────────────────────
# X–Y pairing
# ─────────────────────────────────────────────────────────────────────────────

def _build_objects(tracks, pos_filt, time_filt,
                   pos_raw, time_raw, keep_mask,
                   max_time_gap_us):
    """
    Build a list of pairable 'objects' for one projection.

    Priority order:
      1. Each track from pass-2 output         → type 'track'
      2. Filtered hits not on any track         → type 'hit', source 'filtered'
      3. Raw hits removed by isolation filter   → type 'hit', source 'raw'

    Each object dict contains:
      type   : 'track' or 'hit'
      source : 'track', 'filtered', or 'raw'
      pos    : np.ndarray of mm positions
      time   : np.ndarray of μs times
      t_min  : float  (min hit time)
      t_max  : float  (max hit time)
      t_lo   : float  (t_min − max_time_gap_us, for overlap scoring)
      t_hi   : float  (t_max + max_time_gap_us)
    """
    tol = max_time_gap_us
    objects = []

    # Mark which filtered indices are already on a track
    on_track = np.zeros(len(pos_filt), dtype=bool)
    for t in tracks:
        if len(t) == 0:
            continue
        on_track[t] = True
        p = pos_filt[t]
        ti = time_filt[t]
        objects.append(dict(type='track', source='track',
                            pos=p, time=ti,
                            t_min=float(ti.min()), t_max=float(ti.max()),
                            t_lo=float(ti.min()) - tol,
                            t_hi=float(ti.max()) + tol))

    # Untracked filtered hits
    for i in range(len(pos_filt)):
        if on_track[i]:
            continue
        ti = float(time_filt[i])
        pi = float(pos_filt[i])
        objects.append(dict(type='hit', source='filtered',
                            pos=np.array([pi]), time=np.array([ti]),
                            t_min=ti, t_max=ti,
                            t_lo=ti - tol, t_hi=ti + tol))

    # Raw hits removed by isolation (keep_mask[i] == False)
    for i in range(len(pos_raw)):
        if keep_mask[i]:
            continue
        ti = float(time_raw[i])
        pi = float(pos_raw[i])
        objects.append(dict(type='hit', source='raw',
                            pos=np.array([pi]), time=np.array([ti]),
                            t_min=ti, t_max=ti,
                            t_lo=ti - tol, t_hi=ti + tol))

    return objects


def pair_xy_objects(objects_x, objects_y, min_score=0.25):
    """
    Greedy time-overlap matching of X and Y objects.

    Score = IoU = overlap / (span_x + span_y - overlap), computed on the
    tolerance-expanded intervals.  IoU rewards longer absolute matches:
    two long tracks overlapping well score near 1.0, while a short track
    fully contained inside a long one scores short_span / long_span.
    This ensures the greedy pass prefers pairing long tracks with long
    tracks over matching a long track with a shorter coincident one.
    Only pairs with score >= min_score are accepted.

    Returns
    -------
    pairs       : list of (obj_x, obj_y, score), best-first
    unmatched_x : list of unmatched X objects
    unmatched_y : list of unmatched Y objects
    """
    candidates = []
    for i, ox in enumerate(objects_x):
        for j, oy in enumerate(objects_y):
            overlap = max(0.0, min(ox['t_hi'], oy['t_hi'])
                               - max(ox['t_lo'], oy['t_lo']))
            if overlap <= 0.0:
                continue
            span_x = ox['t_hi'] - ox['t_lo']
            span_y = oy['t_hi'] - oy['t_lo']
            union = span_x + span_y - overlap
            score = overlap / max(union, 1e-9)   # IoU
            if score >= min_score:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)

    used_x, used_y = set(), set()
    pairs = []
    for score, i, j in candidates:
        if i in used_x or j in used_y:
            continue
        pairs.append((objects_x[i], objects_y[j], score))
        used_x.add(i)
        used_y.add(j)

    unmatched_x = [ox for i, ox in enumerate(objects_x) if i not in used_x]
    unmatched_y = [oy for j, oy in enumerate(objects_y) if j not in used_y]
    return pairs, unmatched_x, unmatched_y


def plot_xy_pairs_figure(pairs, unmatched_x, unmatched_y, run, subrun, event):
    """
    2-panel (X top, Y bottom) position-vs-time figure color-coded by pair.

    Matched pairs share the same colour in both panels.
    Tracks → scatter + dashed fit line.
    Single hits → star marker (filtered) or × marker (raw/isolated).
    Unmatched objects → light grey.
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.text(0.5, 0.99, 'X–Y paired objects',
             ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}',
             ha='center', va='top', fontsize=9)

    legend_handles = []

    def _draw_obj(ax, obj, color, alpha=1.0):
        if obj['type'] == 'track':
            ax.scatter(obj['time'], obj['pos'],
                       s=25, color=color, alpha=alpha, zorder=4)
            if len(obj['time']) >= 2:
                sl, ic = _fit_slope(obj['time'], obj['pos'])
                if sl is not None:
                    dt = (obj['time'].max() - obj['time'].min()) * 0.15
                    t_fit = np.linspace(obj['time'].min() - dt,
                                        obj['time'].max() + dt, 200)
                    ax.plot(t_fit, sl * t_fit + ic,
                            color=color, lw=1.5, ls='--',
                            alpha=alpha, zorder=3)
        else:
            mk = 'x' if obj['source'] == 'raw' else '*'
            ax.scatter(obj['time'], obj['pos'],
                       s=80, color=color, marker=mk,
                       linewidths=1.2, alpha=alpha, zorder=4)

    # Unmatched objects (grey, behind everything)
    for obj in unmatched_x:
        _draw_obj(axs[0], obj, color='lightgray', alpha=0.5)
    for obj in unmatched_y:
        _draw_obj(axs[1], obj, color='lightgray', alpha=0.5)

    # Matched pairs
    for k, (ox, oy, score) in enumerate(pairs):
        col = _track_color(k)
        n_x = len(ox['time'])
        n_y = len(oy['time'])
        lbl = (f'Pair {k + 1}: '
               f'X {ox["type"]}({n_x}h) / Y {oy["type"]}({n_y}h)  '
               f'score={score:.2f}')
        _draw_obj(axs[0], ox, color=col)
        _draw_obj(axs[1], oy, color=col)
        legend_handles.append(mpatches.Patch(color=col, label=lbl))

    axs[0].set_ylabel('X position (mm)', fontsize=11)
    axs[1].set_ylabel('Y position (mm)', fontsize=11)
    axs[1].set_xlabel('Time (μs)', fontsize=11)

    # if legend_handles:
    #     axs[0].legend(handles=legend_handles, fontsize=8,
    #                   loc='upper right', framealpha=0.85)

    axs[0].set_title(
        f'{len(pairs)} pair(s) found  |  '
        f'{len(unmatched_x)} unmatched X obj  |  {len(unmatched_y)} unmatched Y obj',
        fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.04)
    _set_window_title(fig, f'X–Y pairs — Evt {event} — {run}/{subrun}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _set_window_title(fig, title):
    """Set the OS window title bar text (no-op if backend has no manager)."""
    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass


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
            axs[row, col].set_xlabel('Time (μs)')
            axs[row, col].set_ylabel(f'{proj} position (mm)')

    plt.tight_layout()
    _set_window_title(fig, f'Hit overview — Evt {event} — {run}/{subrun}')
    return fig


def plot_track_figure(pos_filt, time_filt, tracks, noise_mask,
                      dead_strip_positions, strip_pitch, all_strips, proj_label,
                      run, subrun, event,
                      excl_mask=None, tracks_p2=None):
    """
    Large standalone figure for one projection showing track-finding result.

    Dead strip positions are drawn as horizontal bands.
    Fit lines are extended slightly beyond the hit range.
    excl_mask  : boolean array (same length as pos_filt) marking excluded hits.
                 Excluded noise hits → dark grey ×.
                 Excluded track hits → track colour ×.
    tracks_p2  : optional list of pass-2-only track index arrays.  Drawn with
                 dashed fit lines and triangle markers (continuing the colour
                 cycle after pass-1 tracks).
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

    has_excl = excl_mask is not None and excl_mask.any()

    # Noise hits (unassigned)
    if has_excl:
        ax.scatter(time_filt[noise_mask & ~excl_mask], pos_filt[noise_mask & ~excl_mask],
                   s=12, c='lightgray', zorder=2)
        ax.scatter(time_filt[noise_mask & excl_mask], pos_filt[noise_mask & excl_mask],
                   s=12, c='darkgrey', marker='x', linewidths=0.8, zorder=2)
    else:
        ax.scatter(time_filt[noise_mask], pos_filt[noise_mask],
                   s=12, c='lightgray', zorder=2)

    # Tracks
    for i, idx in enumerate(tracks):
        col = _track_color(i)
        t_tr = time_filt[idx]
        p_tr = pos_filt[idx]
        if has_excl:
            is_excl_hit = excl_mask[idx]
            ax.scatter(t_tr[~is_excl_hit], p_tr[~is_excl_hit],
                       s=30, color=col, zorder=4)
            ax.scatter(t_tr[is_excl_hit], p_tr[is_excl_hit],
                       s=20, color=col, marker='x', linewidths=1.2, zorder=4)
        else:
            ax.scatter(t_tr, p_tr, s=30, color=col, zorder=4)

        if len(idx) >= 2:
            slope, intercept = _fit_slope(t_tr, p_tr)
            dt = (t_tr.max() - t_tr.min()) * 0.1   # 10% extension each side
            t_fit = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 200)
            ax.plot(t_fit, slope * t_fit + intercept,
                    color=col, lw=1.5, ls='--', zorder=3,
                    label=f'Track {i+1}: {len(idx)} hits')

    # Pass-2 new tracks (dashed lines, triangle markers, continuing colour cycle)
    if tracks_p2:
        offset = len(tracks)   # continue numbering after pass-1 tracks
        for j, idx in enumerate(tracks_p2):
            col = _track_color(offset + j)
            t_tr = time_filt[idx]
            p_tr = pos_filt[idx]
            if has_excl:
                is_excl_hit = excl_mask[idx]
                ax.scatter(t_tr[~is_excl_hit], p_tr[~is_excl_hit],
                           s=25, color=col, marker='^', alpha=0.75, zorder=4)
                ax.scatter(t_tr[is_excl_hit], p_tr[is_excl_hit],
                           s=18, color=col, marker='x', linewidths=1.0,
                           alpha=0.75, zorder=4)
            else:
                ax.scatter(t_tr, p_tr, s=25, color=col, marker='^',
                           alpha=0.75, zorder=4)
            if len(idx) >= 2:
                slope, intercept = _fit_slope(t_tr, p_tr)
                dt = (t_tr.max() - t_tr.min()) * 0.1
                t_fit = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 200)
                ax.plot(t_fit, slope * t_fit + intercept,
                        color=col, lw=1.2, ls=':', alpha=0.75, zorder=3,
                        label=f'P2 Track {j+1}: {len(idx)} hits')

    # Dead strip bands
    if dead_strip_positions is not None and len(dead_strip_positions) > 0:
        hw = strip_pitch / 2.0
        for dp in dead_strip_positions:
            ax.axhspan(dp - hw, dp + hw, color='salmon', alpha=0.25, zorder=1)

    ax.set_xlabel('Time (μs)', fontsize=11)
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
        f'Excl: amp≤{EXCLUDE_AMP_MAX} & ToT≤{EXCLUDE_TOT_MAX}  |  '
        f'Road: {ROAD_WIDTH_MM} mm  |  '
        f'Max gap: {MAX_TIME_GAP_US} μs  |  '
        f'Max missed: {MAX_MISSED}  |  '
        f'Min hits: {MIN_TRACK_HITS}'
    )
    ax.annotate(param_text, xy=(0.01, 0.01), xycoords='axes fraction',
                fontsize=8, va='bottom',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    _set_window_title(fig, f'{proj_label} tracks — Evt {event} — {run}/{subrun}')
    return fig


def plot_hit_colorcoded_figure(
        pos_x, time_x, vals_x,
        pos_y, time_y, vals_y,
        var_label, cmap, run, subrun, event):
    """
    2×1 figure: X and Y projections stacked vertically sharing the time axis,
    hits colour-coded by one variable.  All hits shown (no isolation filter applied).
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.text(0.5, 0.99, var_label, ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}', ha='center', va='top', fontsize=9)

    for ax, proj, pos, time_arr, vals in [
        (axs[0], 'X', pos_x, time_x, vals_x),
        (axs[1], 'Y', pos_y, time_y, vals_y),
    ]:
        vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        sc = ax.scatter(time_arr, pos, s=15, c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax, zorder=3)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, label=var_label)
        cb.ax.tick_params(labelsize=8)

        ax.set_ylabel(f'{proj} position (mm)', fontsize=10)

    axs[1].set_xlabel('Time (μs)', fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.065, hspace=0.02)
    _set_window_title(fig, f'{var_label} — Evt {event} — {run}/{subrun}')
    return fig


def plot_amp_vs_tot_figure(amp_x, tot_x, amp_y, tot_y, run, subrun, event,
                           amp_max=None, tot_max=None):
    """
    1×2 hist2d: amplitude vs time-over-threshold for X and Y projections.
    All hits shown (no isolation filter).
    amp_max / tot_max : if given, clip the axis range and filter hits to that range.
    """
    range_label = ''
    if amp_max is not None or tot_max is not None:
        parts = []
        if amp_max is not None:
            parts.append(f'amp ≤ {amp_max}')
        if tot_max is not None:
            parts.append(f'ToT ≤ {tot_max}')
        range_label = f'  [{", ".join(parts)}]'

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.text(0.5, 0.99, f'Amplitude vs Time over threshold{range_label}',
             ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}',
             ha='center', va='top', fontsize=9)

    for ax, proj, amp, tot in [
        (axs[0], 'X', amp_x, tot_x),
        (axs[1], 'Y', amp_y, tot_y),
    ]:
        mask = np.ones(len(amp), dtype=bool)
        if amp_max is not None:
            mask &= amp <= amp_max
        if tot_max is not None:
            mask &= tot <= tot_max
        h = ax.hist2d(amp[mask], tot[mask], bins=50, cmap='viridis', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Hits')
        ax.set_xlabel('Amplitude (ADC)', fontsize=10)
        ax.set_ylabel('Time over threshold (ns)', fontsize=10)
        ax.set_title(f'{proj} projection', fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.3)
    win_title = f'Amp vs ToT{range_label} — Evt {event} — {run}/{subrun}'
    _set_window_title(fig, win_title)
    return fig


def plot_hit_segments_figure(
        pos_x, time_x, amp_x, tot_x,
        pos_y, time_y, amp_y, tot_y,
        run, subrun, event):
    """
    Position-vs-time figure.  For each hit:
      - Circle marker at the start time, coloured by amplitude (high zorder).
      - Faint thin horizontal segment from start time to start + ToT behind it
        (low alpha, low zorder) — only visible on zoom.
    """
    from matplotlib.collections import LineCollection

    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.text(0.5, 0.99, 'Hits coloured by amplitude  (faint bar = ToT extent)',
             ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}',
             ha='center', va='top', fontsize=9)

    for ax, proj, pos, time_arr, amp, tot in [
        (axs[0], 'X', pos_x, time_x, amp_x, tot_x),
        (axs[1], 'Y', pos_y, time_y, amp_y, tot_y),
    ]:
        tot_us = tot / 1000.0   # ns → μs
        vmin, vmax = np.nanpercentile(amp, 2), np.nanpercentile(amp, 98)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # ── ToT segments (behind, very faint) ──────────────────────────────
        segments = np.array([[[t, p], [t + dt, p]]
                              for t, p, dt in zip(time_arr, pos, tot_us)])
        lc = LineCollection(segments, cmap='viridis', norm=norm,
                            linewidths=0.8, alpha=0.25, zorder=2)
        lc.set_array(amp)
        ax.add_collection(lc)

        # ── Circle markers at start time, coloured by amplitude ─────────────
        sc = ax.scatter(time_arr, pos, s=15, c=amp, cmap='viridis', norm=norm,
                        zorder=4)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, label='Amplitude (ADC)')
        cb.ax.tick_params(labelsize=8)

        # Set limits from scatter (auto-scaled) then extend x for ToT tails
        ax.set_xlim(ax.get_xlim()[0],
                    max(ax.get_xlim()[1], (time_arr + tot_us).max() + 0.05))
        ax.set_ylabel(f'{proj} position (mm)', fontsize=10)

    axs[1].set_xlabel('Time (μs)', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.05)
    _set_window_title(fig, f'Hit segments — Evt {event} — {run}/{subrun}')
    return fig


def launch_region_explorer(
        pos_x, time_x, amp_x, tot_x,
        pos_y, time_y, amp_y, tot_y,
        run, subrun, event, amp_max=None, tot_max=None):
    """
    Interactive amp-vs-ToT region explorer.

    Shows a correlation figure (with optional range limits) and a companion
    position-vs-time figure.  Draw a rectangle on either correlation panel to
    highlight the matching hits on the position-vs-time figure.  Each projection
    has its own independent selection.  Selected range is printed to the console.

    Returns (fig_cor, fig_pos, selectors) — caller must keep the return value
    alive (assign it to a variable) so the widget callbacks are not GC'd.
    """
    from matplotlib.widgets import RectangleSelector

    # per-projection data, keyed lowercase
    data = {
        'x': (pos_x, time_x, amp_x, tot_x),
        'y': (pos_y, time_y, amp_y, tot_y),
    }

    # ── Correlation figure ─────────────────────────────────────────────────
    fig_cor, axs_cor = plt.subplots(1, 2, figsize=(12, 5))
    range_label = ''
    if amp_max is not None or tot_max is not None:
        parts = []
        if amp_max is not None:
            parts.append(f'amp ≤ {amp_max}')
        if tot_max is not None:
            parts.append(f'ToT ≤ {tot_max}')
        range_label = f'  [{", ".join(parts)}]'
    fig_cor.text(0.5, 0.99,
                 f'Amplitude vs Time over threshold{range_label}  —  draw rectangle to select',
                 ha='center', va='top', fontsize=11, fontweight='bold')
    fig_cor.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}',
                 ha='center', va='top', fontsize=9)

    for ax, proj in [(axs_cor[0], 'x'), (axs_cor[1], 'y')]:
        _, _, amp, tot = data[proj]
        mask = np.ones(len(amp), dtype=bool)
        if amp_max is not None:
            mask &= amp <= amp_max
        if tot_max is not None:
            mask &= tot <= tot_max
        h = ax.hist2d(amp[mask], tot[mask], bins=50, cmap='viridis', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Hits')
        ax.set_xlabel('Amplitude (ADC)', fontsize=10)
        ax.set_ylabel('Time over threshold (ns)', fontsize=10)
        ax.set_title(f'{proj.upper()} projection', fontsize=10)

    fig_cor.tight_layout()
    fig_cor.subplots_adjust(top=0.88, bottom=0.12, wspace=0.3)
    _set_window_title(fig_cor, f'Region selector — Evt {event} — {run}/{subrun}')

    # ── Companion position-vs-time figure ──────────────────────────────────
    fig_pos, axs_pos = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig_pos.text(0.5, 0.99, 'Selected hits on position vs time',
                 ha='center', va='top', fontsize=13, fontweight='bold')
    fig_pos.text(0.5, 0.965, f'Event {event}  |  {run}/{subrun}',
                 ha='center', va='top', fontsize=9)

    sc_bg  = {}   # all-hits background (grey)
    sc_sel = {}   # selected hits (crimson), updated on each rectangle draw

    for ax, proj in [(axs_pos[0], 'x'), (axs_pos[1], 'y')]:
        pos, time_arr, _, _ = data[proj]
        sc_bg[proj]  = ax.scatter(time_arr, pos, s=8,  c='lightgrey', zorder=2)
        sc_sel[proj] = ax.scatter([], [],          s=25, c='crimson',   zorder=3,
                                  label='Selected (0)')
        ax.set_ylabel(f'{proj.upper()} position (mm)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    axs_pos[1].set_xlabel('Time (μs)', fontsize=10)
    fig_pos.subplots_adjust(top=0.91, bottom=0.08, hspace=0.05)
    _set_window_title(fig_pos, f'Region hits — Evt {event} — {run}/{subrun}')

    # ── Selection state and callbacks ──────────────────────────────────────
    selections = {'x': None, 'y': None}

    def _update():
        for proj, ax in [('x', axs_pos[0]), ('y', axs_pos[1])]:
            pos, time_arr, amp, tot = data[proj]
            sel = selections[proj]
            if sel is not None:
                a0, a1, t0, t1 = sel
                mask = (amp >= a0) & (amp <= a1) & (tot >= t0) & (tot <= t1)
            else:
                mask = np.zeros(len(amp), dtype=bool)
            offsets = np.c_[time_arr[mask], pos[mask]] if mask.any() else np.empty((0, 2))
            sc_sel[proj].set_offsets(offsets)
            sc_sel[proj].set_label(f'Selected ({mask.sum()})')
            ax.legend(fontsize=8, loc='upper right')
        fig_pos.canvas.draw_idle()

    def _make_callback(proj):
        def _on_select(eclick, erelease):
            a0 = min(eclick.xdata, erelease.xdata)
            a1 = max(eclick.xdata, erelease.xdata)
            t0 = min(eclick.ydata, erelease.ydata)
            t1 = max(eclick.ydata, erelease.ydata)
            selections[proj] = (a0, a1, t0, t1)
            _, _, amp, tot = data[proj]
            n = int(((amp >= a0) & (amp <= a1) & (tot >= t0) & (tot <= t1)).sum())
            print(f'{proj.upper()}: amp=[{a0:.0f}, {a1:.0f}]  '
                  f'ToT=[{t0:.0f}, {t1:.0f}]  →  {n} hits selected')
            _update()
        return _on_select

    selectors = []
    for ax_cor, proj in [(axs_cor[0], 'x'), (axs_cor[1], 'y')]:
        rs = RectangleSelector(
            ax_cor, _make_callback(proj),
            useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='data',
            interactive=True,
        )
        selectors.append(rs)

    return fig_cor, fig_pos, selectors


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


def print_xy_pairs(pairs, unmatched_x, unmatched_y):
    print(f'\n── X–Y pairing ───────────────────────────────────────────')

    def _obj_str(obj):
        t0, t1 = obj['t_min'], obj['t_max']
        n = len(obj['time'])
        if obj['type'] == 'track':
            slope, _ = _fit_slope(obj['time'], obj['pos'])
            slope_str = f'slope={slope:.2f} mm/μs' if slope is not None else 'slope=n/a'
            return (f"track({n}h) t=[{t0:.3f},{t1:.3f}] μs "
                    f"span={t1-t0:.3f} μs {slope_str}")
        else:
            return (f"hit({obj['source']}) t={t0:.3f} μs "
                    f"pos={obj['pos'][0]:.1f} mm")

    if not pairs:
        print('  No pairs found.')
    else:
        for k, (ox, oy, score) in enumerate(pairs):
            print(f'  Pair {k+1}  IoU={score:.3f}')
            print(f'    X: {_obj_str(ox)}')
            print(f'    Y: {_obj_str(oy)}')

    if unmatched_x:
        print(f'  Unmatched X ({len(unmatched_x)}):')
        for obj in unmatched_x:
            print(f'    {_obj_str(obj)}')
    if unmatched_y:
        print(f'  Unmatched Y ({len(unmatched_y)}):')
        for obj in unmatched_y:
            print(f'    {_obj_str(obj)}')


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
    _set_window_title(fig, f'Track metrics — {run}/{subrun}')
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
    _set_window_title(fig, f'Track metrics [{label}] — {run}/{subrun}')
    return fig


if __name__ == '__main__':
    main()
