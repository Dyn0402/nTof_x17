#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_explorer.py

Interactive single-event browser + track finder for run_70/resist_final_790V.

Focuses on mx17_3:
  FEU 1 (X-running strips)  →  y_mm positions
  FEU 2 (Y-running strips)  →  x_mm positions

Figure layout (2 rows × 6 gridspec columns):
  Row 0: [FEU 1 pos-time (cols 0-2)] | [FEU 2 pos-time (cols 3-5)]
  Row 1: [X-Y scatter (0-1)] | [FEU 1 waveform (2-3)] | [FEU 2 waveform (4-5)]

Key bindings:
  n / →   next event         p / ←   previous event
  2       toggle 2-pair-only mode
  s       save figure        q       quit

Workflow: edit TRACKING PARAMETERS below, rerun to iterate.
"""

import pickle
import sys
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import uproot

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from common.Mx17StripMap import Mx17StripMap, Detector
from beam_track_finding import (
    remove_isolated_hits,
    find_tracks_1d,
    find_tracks_1d_pass2,
    _fit_slope,
    _track_color,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SUBRUN_DIR      = Path('/media/dylan/data/x17/may_beam/runs/run_70/resist_final_790V')
RUN_CONFIG_PATH = SUBRUN_DIR.parent / 'run_config.json'
MAP_CSV         = _ROOT / 'mx17_m1_map.csv'
DECODED_DIR     = SUBRUN_DIR / 'decoded_root'
OUT_DIR         = Path(__file__).parent / 'output' / 'run_70_resist_final_790V' / 'events'
WF_INDEX_CACHE  = OUT_DIR / 'wf_index.pkl'

# ── Hit filter ────────────────────────────────────────────────────────────────
AMP_THRESHOLD = 200    # ADC  — minimum hit amplitude
TIME_WIN_MIN  = 300.0  # ns   — beam drift window lower edge
TIME_WIN_MAX  = 1000.0 # ns   — beam drift window upper edge

# ── Dead strip detection ──────────────────────────────────────────────────────
DEAD_STRIP_AMP_THRESHOLD = 500   # ADC — amplitude used for dead-strip rate
DEAD_STRIP_RATE_FRACTION = 0.10  # strips below this × median rate are flagged dead

# ── Tracking parameters (edit to iterate) ────────────────────────────────────
ISO_POS_MM         = 5.0   # mm  — isolation neighbourhood radius in position
ISO_TIME_NS        = 200.0 # ns  — isolation neighbourhood radius in time
ROAD_WIDTH_MM      = 3.0   # mm  — road width once slope is established
ROAD_WIDTH_SEED_MM = 7.0   # mm  — road width before slope is known
MAX_TIME_GAP_NS    = 150.0 # ns  — max time gap between consecutive track hits
MIN_TRACK_HITS     = 3     #     — minimum hits to accept a track
MAX_MISSED         = 2     #     — max consecutive time-step misses before stopping
MAX_STRIP_GAP      = 2     #     — max skipped live strips between hits
PAIR_MIN_IOU       = 0.20  #     — minimum time-overlap IoU to accept an X-Y pair

# ── Waveform ──────────────────────────────────────────────────────────────────
NS_PER_SAMPLE = 20.0   # ns/sample (run_config: dream_daq_info.sample_period)

# ── Display ───────────────────────────────────────────────────────────────────
FIG_SIZE     = (22, 12)
AMP_SIZE_MIN = 8      # minimum scatter marker size
AMP_SIZE_MAX = 140    # maximum scatter marker size


# ─────────────────────────────────────────────────────────────────────────────
# Detector setup + data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_detector() -> Detector:
    cfg = json.loads(RUN_CONFIG_PATH.read_text())
    det_cfg = next(d for d in cfg['detectors'] if d['name'] == 'mx17_3')
    return Detector('mx17_3', det_cfg, Mx17StripMap(str(MAP_CSV)))


def build_ch_to_pos(det: Detector) -> tuple:
    """Precompute channel→position dicts for FEU 1 (y_mm) and FEU 2 (x_mm)."""
    ch_to_y = {}
    ch_to_x = {}
    for ch in range(512):
        p1 = det.map_hit(1, ch)
        if p1 and p1[1] is not None:
            ch_to_y[ch] = float(p1[1])
        p2 = det.map_hit(2, ch)
        if p2 and p2[0] is not None:
            ch_to_x[ch] = float(p2[0])
    return ch_to_y, ch_to_x


def load_and_map(det: Detector) -> pd.DataFrame:
    """Load all combined_hits (FEU 1+2) and add mm positions."""
    files = sorted(
        f for f in (SUBRUN_DIR / 'combined_hits_root').iterdir()
        if f.suffix == '.root' and '_datrun_' in f.name and 'feu-combined' in f.name
    )
    print(f'Loading {len(files)} combined_hits files …')
    df = uproot.concatenate([f'{f}:hits' for f in files], library='pd')
    df = df[df['feu'].isin([1, 2])].copy()

    xs, ys = [], []
    for feu, ch in zip(df['feu'].values, df['channel'].values):
        p = det.map_hit(int(feu), int(ch))
        xs.append(p[0] if p else np.nan)
        ys.append(p[1] if p else np.nan)
    df['x_mm'] = xs
    df['y_mm'] = ys
    print(f'  {len(df):,} hits  |  {df["eventId"].nunique():,} events')
    return df


def filter_candidates(df: pd.DataFrame) -> np.ndarray:
    """Event IDs with ≥1 qualifying hit in each FEU (same logic as beam_qa.py)."""
    qual = df[
        (df['amplitude'] > AMP_THRESHOLD) &
        (df['time'] >= TIME_WIN_MIN) &
        (df['time'] <= TIME_WIN_MAX)
    ]
    has1 = set(qual.loc[qual['feu'] == 1, 'eventId'])
    has2 = set(qual.loc[qual['feu'] == 2, 'eventId'])
    return np.array(sorted(has1 & has2))


# ─────────────────────────────────────────────────────────────────────────────
# Dead strip detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_dead_strips(df: pd.DataFrame) -> tuple:
    """
    Find strip positions with hit rate below DEAD_STRIP_RATE_FRACTION × median.
    Returns (dead1_y_mm, dead2_x_mm, strip_pitch_mm).
    """
    df_th  = df[df['amplitude'] >= DEAD_STRIP_AMP_THRESHOLD]
    n_evts = max(df['eventId'].nunique(), 1)

    all_y = np.sort(df[df['feu'] == 1]['y_mm'].dropna().unique())
    all_x = np.sort(df[df['feu'] == 2]['x_mm'].dropna().unique())

    y_cnt = (df_th[(df_th['feu'] == 1) & df_th['y_mm'].notna()]
             ['y_mm'].value_counts())
    x_cnt = (df_th[(df_th['feu'] == 2) & df_th['x_mm'].notna()]
             ['x_mm'].value_counts())

    y_rates = pd.Series({p: y_cnt.get(p, 0) / n_evts for p in all_y})
    x_rates = pd.Series({p: x_cnt.get(p, 0) / n_evts for p in all_x})

    def _dead(rates):
        active = rates[rates > 0]
        if len(active) == 0:
            return np.array([])
        thresh = float(np.median(active)) * DEAD_STRIP_RATE_FRACTION
        return np.sort(rates.index[rates < thresh].to_numpy(dtype=float))

    dead1 = _dead(y_rates)
    dead2 = _dead(x_rates)

    pitch1 = float(np.median(np.diff(all_y))) if len(all_y) >= 2 else 0.78
    pitch2 = float(np.median(np.diff(all_x))) if len(all_x) >= 2 else 0.78
    pitch  = float(0.5 * (pitch1 + pitch2))

    print(f'Dead strips: FEU1 y_mm = {len(dead1)},  '
          f'FEU2 x_mm = {len(dead2)}   (pitch ≈ {pitch:.3f} mm)')
    return dead1, dead2, pitch


# ─────────────────────────────────────────────────────────────────────────────
# Waveform index + loading
# ─────────────────────────────────────────────────────────────────────────────

def build_waveform_index(use_cache: bool = True) -> dict:
    """
    Build {event_id: {feu: (fpath_str, row_idx)}} for FEUs 1 and 2.
    Caches to WF_INDEX_CACHE on first run; subsequent calls load instantly.
    """
    if use_cache and WF_INDEX_CACHE.exists():
        print('Loading waveform index from cache …')
        with open(WF_INDEX_CACHE, 'rb') as f:
            return pickle.load(f)

    print('Building waveform index (scanning decoded_root) …')
    index = {}
    all_files = sorted(DECODED_DIR.iterdir(), key=lambda p: p.name)

    for feu in (1, 2):
        feu_str   = f'_{feu:02d}.'
        feu_files = [f for f in all_files
                     if f.suffix == '.root' and feu_str in f.name]
        for fpath in feu_files:
            try:
                with uproot.open(fpath) as uf:
                    if 'nt' not in uf:
                        continue
                    eids = uf['nt']['eventId'].array(library='np')
                    for row_i, eid in enumerate(eids):
                        eid_int = int(eid)
                        if eid_int not in index:
                            index[eid_int] = {}
                        index[eid_int][feu] = (str(fpath), int(row_i))
            except Exception as e:
                print(f'  [wf index] {fpath.name}: {e}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(WF_INDEX_CACHE, 'wb') as f:
        pickle.dump(index, f)
    print(f'  Indexed {len(index)} events → cached to {WF_INDEX_CACHE.name}')
    return index


def load_waveform(event_id: int, wf_index: dict) -> dict:
    """
    Load raw waveform arrays for one event.
    Returns {feu: (samples_arr, channels_arr, amps_arr)}.
    """
    result = {}
    if event_id not in wf_index:
        return result
    for feu, (fpath_str, row_i) in wf_index[event_id].items():
        try:
            with uproot.open(fpath_str) as uf:
                tree = uf['nt']
                s = np.asarray(tree['sample'].array(
                    library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                c = np.asarray(tree['channel'].array(
                    library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                a = np.asarray(tree['amplitude'].array(
                    library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                result[feu] = (s, c, a.astype(float))
        except Exception as e:
            print(f'  [wf load] event {event_id} FEU {feu}: {e}')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-event result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventResult:
    event_id: int
    # Filtered hits (amp > threshold, time in window, valid position), time in μs
    pos1:  np.ndarray   # FEU 1 → y_mm
    time1: np.ndarray   # μs
    amp1:  np.ndarray
    pos2:  np.ndarray   # FEU 2 → x_mm
    time2: np.ndarray   # μs
    amp2:  np.ndarray
    # Isolation masks (True = kept)
    keep1: np.ndarray
    keep2: np.ndarray
    # Track index arrays (into pos1[keep1] / pos2[keep2])
    tracks1: List[np.ndarray] = field(default_factory=list)
    tracks2: List[np.ndarray] = field(default_factory=list)
    # Pairs: (idx_in_tracks1, idx_in_tracks2, iou_score)
    pairs: list = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _pair_tracks(tracks1, time1f, tracks2, time2f, gap_us: float) -> list:
    """Greedy IoU-based pairing. Returns [(i, j, iou), …] best-first."""
    candidates = []
    for i, t1 in enumerate(tracks1):
        lo1 = time1f[t1].min() - gap_us
        hi1 = time1f[t1].max() + gap_us
        for j, t2 in enumerate(tracks2):
            lo2 = time2f[t2].min() - gap_us
            hi2 = time2f[t2].max() + gap_us
            overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            if overlap == 0:
                continue
            iou = overlap / max((hi1 - lo1) + (hi2 - lo2) - overlap, 1e-9)
            if iou >= PAIR_MIN_IOU:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used1, used2, pairs = set(), set(), []
    for iou, i, j in candidates:
        if i not in used1 and j not in used2:
            pairs.append((i, j, float(iou)))
            used1.add(i); used2.add(j)
    return pairs


def _cluster_extend(track: np.ndarray, posf: np.ndarray, timef: np.ndarray,
                    strip_pitch: float) -> np.ndarray:
    """
    After road-following, absorb adjacent-strip unassigned hits at the temporal
    ENDPOINTS of the track only (not from middle hits).  This catches charge
    sharing at the track edges without risking bridging two separate tracks.

    The road algorithm picks ONE hit per time step (closest in time, ignoring
    amplitude).  This pass adds missed adjacent-strip hits at the same cluster
    time as the endpoint hit.

    Constraint: within 3×strip_pitch in position AND 3 samples in time (~60 ns).
    """
    cluster_dt_us = 3.0 * NS_PER_SAMPLE / 1000.0   # 3 samples × 20 ns = 60 ns

    in_track = np.zeros(len(posf), bool)
    in_track[track] = True

    changed = True
    while changed:
        changed = False
        idx_in  = np.where(in_track)[0]
        # Only seed from the two temporal endpoints of the current track
        t_earliest = idx_in[np.argmin(timef[idx_in])]
        t_latest   = idx_in[np.argmax(timef[idx_in])]
        for ep in (t_earliest, t_latest):
            for i in range(len(posf)):
                if in_track[i]:
                    continue
                if (abs(posf[i] - posf[ep]) <= 3.0 * strip_pitch and
                        abs(timef[i] - timef[ep]) <= cluster_dt_us):
                    in_track[i] = True
                    changed = True
    return np.where(in_track)[0]


def process_event(df_evt: pd.DataFrame,
                  dead1: np.ndarray, dead2: np.ndarray,
                  strip_pitch: float) -> EventResult:
    """Full tracking pipeline for one event."""
    eid    = int(df_evt['eventId'].iloc[0])
    iso_us = ISO_TIME_NS  / 1000.0
    gap_us = MAX_TIME_GAP_NS / 1000.0

    def _hits(feu, pos_col):
        d = df_evt[
            (df_evt['feu'] == feu) &
            (df_evt['amplitude'] > AMP_THRESHOLD) &
            (df_evt['time'] >= TIME_WIN_MIN) &
            (df_evt['time'] <= TIME_WIN_MAX) &
            df_evt[pos_col].notna()
        ]
        return (d[pos_col].to_numpy(dtype=float),
                d['time'].to_numpy(dtype=float) / 1000.0,
                d['amplitude'].to_numpy(dtype=float))

    pos1, time1, amp1 = _hits(1, 'y_mm')
    pos2, time2, amp2 = _hits(2, 'x_mm')

    def _isolation(pos, time):
        if len(pos) < 2:
            return np.ones(len(pos), bool)
        return remove_isolated_hits(pos, time, ISO_POS_MM, iso_us)

    keep1 = _isolation(pos1, time1)
    keep2 = _isolation(pos2, time2)
    pos1f, time1f = pos1[keep1], time1[keep1]
    pos2f, time2f = pos2[keep2], time2[keep2]

    kw = dict(road_width_mm=ROAD_WIDTH_MM,
              road_width_seed_mm=ROAD_WIDTH_SEED_MM,
              max_time_gap_us=gap_us,
              min_hits=MIN_TRACK_HITS,
              max_missed=MAX_MISSED,
              max_strip_gap=MAX_STRIP_GAP,
              strip_pitch=strip_pitch)

    def _track(pos, time, dead):
        if len(pos) < MIN_TRACK_HITS:
            return []
        p1, _ = find_tracks_1d(pos, time, dead_strip_positions=dead, **kw)
        ext, new = find_tracks_1d_pass2(p1, pos, time, dead_strip_positions=dead, **kw)
        all_t = ext + new
        sets  = [set(t.tolist()) for t in all_t]
        raw   = [all_t[i] for i, s in enumerate(sets)
                 if not any(s < sets[j] for j in range(len(sets)) if j != i)]
        # Strip-cluster extension: absorb adjacent unassigned hits at track endpoints
        extended = [_cluster_extend(t, pos, time, strip_pitch) for t in raw]
        # Deduplicate and remove subsets again (cluster extension can merge tracks)
        seen, deduped = [], []
        for t in extended:
            fs = frozenset(t.tolist())
            if fs not in seen:
                seen.append(fs); deduped.append(t)
        ext_sets = [set(t.tolist()) for t in deduped]
        return [deduped[i] for i, s in enumerate(ext_sets)
                if not any(s < ext_sets[j] for j in range(len(ext_sets)) if j != i)]

    tracks1 = _track(pos1f, time1f, dead1)
    tracks2 = _track(pos2f, time2f, dead2)
    pairs   = _pair_tracks(tracks1, time1f, tracks2, time2f, gap_us)

    return EventResult(eid, pos1, time1, amp1, pos2, time2, amp2,
                       keep1, keep2, tracks1, tracks2, pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _amp_to_size(amp: np.ndarray) -> np.ndarray:
    """Amplitude → scatter marker size. Scale maxes out at ~1000 ADC above threshold
    so typical beam hits show clearly varying sizes (not all near the minimum)."""
    frac = np.clip((amp - AMP_THRESHOLD) / 800.0, 0.0, 1.0)
    return AMP_SIZE_MIN + frac * (AMP_SIZE_MAX - AMP_SIZE_MIN)


def _draw_projection(ax, r: EventResult, feu: int,
                     dead: Optional[np.ndarray] = None,
                     strip_pitch: float = 0.78):
    """Position vs drift time with amplitude-sized hits, dead strip bands, track fits."""
    if feu == 1:
        pos, time, amp, keep, tracks = r.pos1, r.time1, r.amp1, r.keep1, r.tracks1
        pos_lbl  = 'Y pos [mm]  (FEU 1 — X strips)'
        pair_idx = {i: k for k, (i, j, _) in enumerate(r.pairs)}
    else:
        pos, time, amp, keep, tracks = r.pos2, r.time2, r.amp2, r.keep2, r.tracks2
        pos_lbl  = 'X pos [mm]  (FEU 2 — Y strips)'
        pair_idx = {j: k for k, (i, j, _) in enumerate(r.pairs)}

    # Dead strip bands
    if dead is not None and len(dead):
        hw = strip_pitch / 2.0
        for dp in dead:
            ax.axhspan(dp - hw, dp + hw, color='salmon', alpha=0.2, zorder=0)

    if len(pos) == 0:
        ax.text(0.5, 0.5, 'No hits', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(pos_lbl); return

    posf  = pos[keep]
    timef = time[keep]
    ampf  = amp[keep]

    # Isolated hits — red × sized by amplitude
    if (~keep).any():
        ax.scatter(time[~keep], pos[~keep],
                   s=_amp_to_size(amp[~keep]),
                   c='tomato', marker='x', linewidths=0.9, zorder=2,
                   label=f'Isolated ({(~keep).sum()})')

    # Hits on track vs noise
    on_track = np.zeros(len(posf), bool)
    for t in tracks:
        on_track[t] = True

    # Noise hits — dark grey with faint edge so they're visible but clearly secondary
    if (~on_track).any():
        ax.scatter(timef[~on_track], posf[~on_track],
                   s=_amp_to_size(ampf[~on_track]),
                   c='#888888', alpha=0.85, zorder=3,
                   edgecolors='#444444', linewidths=0.25,
                   label=f'Noise ({(~on_track).sum()})')

    # Track hits — colored by pair, amplitude-sized
    for i, idx in enumerate(tracks):
        col  = _track_color(pair_idx[i]) if i in pair_idx else '#666666'
        t_tr = timef[idx]
        p_tr = posf[idx]
        lbl  = (f'Pair {pair_idx[i]+1} ({len(idx)} hits)'
                if i in pair_idx else f'Unmatched ({len(idx)} hits)')
        ax.scatter(t_tr, p_tr, s=_amp_to_size(ampf[idx]),
                   color=col, zorder=5, edgecolors='k', linewidths=0.4, label=lbl)
        if len(idx) >= 2:
            sl, ic = _fit_slope(t_tr, p_tr)
            if sl is not None:
                dt = (t_tr.max() - t_tr.min()) * 0.12
                tf = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 120)
                ax.plot(tf, sl * tf + ic, color=col, lw=1.5, ls='--', zorder=4, alpha=0.85)

    ax.axvspan(TIME_WIN_MIN / 1000, TIME_WIN_MAX / 1000, alpha=0.05, color='green', zorder=0)
    ax.set_xlabel('Drift time [μs]')
    ax.set_ylabel(pos_lbl)
    ax.set_title(f'{pos_lbl.split("[")[0].strip()}  |  {len(tracks)} track(s), {len(pos)} hits')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # Zoom to actual hit data — don't let dead strip axhspan bands drive the scale.
    # Prefer track hit range; fall back to all filtered hits if no tracks.
    if tracks and any(len(t) > 0 for t in tracks):
        zoom_pos  = np.concatenate([posf[t]  for t in tracks])
        zoom_time = np.concatenate([timef[t] for t in tracks])
    elif len(posf) > 0:
        zoom_pos, zoom_time = posf, timef
    else:
        zoom_pos, zoom_time = pos, time

    if len(zoom_pos) > 0:
        p_lo, p_hi = zoom_pos.min(),  zoom_pos.max()
        t_lo, t_hi = zoom_time.min(), zoom_time.max()
        p_pad = max(4.0,  (p_hi - p_lo) * 0.18)
        t_pad = max(0.02, (t_hi - t_lo) * 0.18)
        ax.set_ylim(p_lo - p_pad, p_hi + p_pad)
        ax.set_xlim(t_lo - t_pad, t_hi + t_pad)


def _draw_xy_scatter(ax, r: EventResult):
    """X-Y position scatter: earliest hit per track pair = reconstructed mesh crossing."""
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title('Reconstructed positions (earliest hit per pair)')
    ax.grid(True, alpha=0.2)

    if not r.pairs:
        ax.text(0.5, 0.5, 'No pairs found', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='grey')
        return

    posf1 = r.pos1[r.keep1]; tf1 = r.time1[r.keep1]
    posf2 = r.pos2[r.keep2]; tf2 = r.time2[r.keep2]

    for k, (i, j, iou) in enumerate(r.pairs):
        col   = _track_color(k)
        t1, t2 = r.tracks1[i], r.tracks2[j]
        y_mesh = posf1[t1[np.argmin(tf1[t1])]]
        x_mesh = posf2[t2[np.argmin(tf2[t2])]]
        ax.scatter(x_mesh, y_mesh, s=220, color=col, zorder=4,
                   edgecolors='k', linewidths=0.6,
                   label=f'Pair {k+1}  IoU={iou:.2f}')
        for p_y in posf1[t1]:
            ax.axhline(p_y, color=col, lw=0.4, alpha=0.3)
        for p_x in posf2[t2]:
            ax.axvline(p_x, color=col, lw=0.4, alpha=0.3)

    ax.legend(fontsize=8, loc='upper right')


def _draw_waveform(ax, wf_data: dict, feu: int, r: EventResult):
    """
    Per-channel waveform traces (amplitude vs time) from decoded_root.
    Each unique channel is drawn as a line, coloured by channel number.
    Track-hit channels are highlighted; time window and threshold are marked.
    Follows detector_qa.py _plot_waveform_hits_event style.
    """
    ax.set_xlabel('Time [μs]')
    ax.set_ylabel('Amplitude [ADC]')
    ax.set_title(f'FEU {feu} raw waveform')

    if feu not in wf_data or len(wf_data[feu][0]) == 0:
        ax.text(0.5, 0.5, f'No waveform data\n(FEU {feu})',
                transform=ax.transAxes, ha='center', va='center', color='grey')
        ax.axvline(TIME_WIN_MIN / 1000, color='red',    lw=1.2, ls='--')
        ax.axvline(TIME_WIN_MAX / 1000, color='orange', lw=1.2, ls='--')
        return

    samples, channels, amps = wf_data[feu]
    t_us      = samples.astype(float) * NS_PER_SAMPLE / 1000.0   # sample → μs
    unique_ch = np.unique(channels.astype(int))

    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=float(unique_ch.min()), vmax=float(unique_ch.max()))

    for ch in unique_ch:
        mask  = channels.astype(int) == ch
        t_ch  = t_us[mask]
        a_ch  = amps[mask]
        order = np.argsort(t_ch)
        ax.plot(t_ch[order], a_ch[order],
                lw=0.7, color=cmap(norm(ch)), alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Channel', pad=0.01)

    # Threshold line offset by +250 — raw waveform ADC is not directly comparable
    # to combined_hits amplitude (different baseline subtraction); this is approximate.
    wf_thr = AMP_THRESHOLD + 250
    ax.axhline(wf_thr, color='red', lw=1.0, ls='--',
               label=f'≈ hit thr = {wf_thr} ADC (approx)',
               zorder=5)
    ax.axvline(TIME_WIN_MIN / 1000, color='red',    lw=1.2, ls=':',
               label=f'{TIME_WIN_MIN:.0f} ns', zorder=5)
    ax.axvline(TIME_WIN_MAX / 1000, color='orange', lw=1.2, ls=':',
               label=f'{TIME_WIN_MAX:.0f} ns', zorder=5)

    # Strip count from combined_hits (amplitude-thresholded hits), not raw waveform
    hits_pos  = r.pos1 if feu == 1 else r.pos2
    n_strips  = len(np.unique(hits_pos)) if len(hits_pos) else 0
    ax.text(0.01, 0.99, f'{n_strips} strips fired (combined_hits, amp>{AMP_THRESHOLD})',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)


def plot_dead_strips(df: pd.DataFrame,
                     dead1: np.ndarray, dead2: np.ndarray,
                     strip_pitch: float) -> None:
    """
    Show per-strip hit rates for FEU 1 and FEU 2 with dead strips highlighted.
    Saves to output dir and shows non-blocking so it stays visible during the browser.
    """
    n_evts   = max(df['eventId'].nunique(), 1)
    df_th    = df[df['amplitude'] >= DEAD_STRIP_AMP_THRESHOLD]

    all_y = np.sort(df[df['feu'] == 1]['y_mm'].dropna().unique())
    all_x = np.sort(df[df['feu'] == 2]['x_mm'].dropna().unique())
    y_cnt = df_th[(df_th['feu'] == 1) & df_th['y_mm'].notna()]['y_mm'].value_counts()
    x_cnt = df_th[(df_th['feu'] == 2) & df_th['x_mm'].notna()]['x_mm'].value_counts()
    y_rates = np.array([y_cnt.get(p, 0) / n_evts for p in all_y])
    x_rates = np.array([x_cnt.get(p, 0) / n_evts for p in all_x])

    def _thresh(rates):
        active = rates[rates > 0]
        return float(np.median(active)) * DEAD_STRIP_RATE_FRACTION if len(active) else 0.0

    thresh_y = _thresh(y_rates)
    thresh_x = _thresh(x_rates)
    dead1_set = set(dead1.tolist())
    dead2_set = set(dead2.tolist())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(
        f'Dead-strip analysis — run_70/resist_final_790V\n'
        f'{n_evts} events  |  amp ≥ {DEAD_STRIP_AMP_THRESHOLD}  |  '
        f'dead threshold = {DEAD_STRIP_RATE_FRACTION:.0%} of median',
        fontsize=11
    )

    for ax, positions, rates, dead_set, thresh, label in [
        (ax1, all_y, y_rates, dead1_set, thresh_y, 'FEU 1 — Y position [mm]  (X strips)'),
        (ax2, all_x, x_rates, dead2_set, thresh_x, 'FEU 2 — X position [mm]  (Y strips)'),
    ]:
        colors = ['tomato' if p in dead_set else 'steelblue' for p in positions]
        ax.bar(positions, rates, width=strip_pitch, color=colors,
               align='center', zorder=2)
        median_active = thresh / max(DEAD_STRIP_RATE_FRACTION, 1e-9)
        ax.axhline(median_active, color='green', ls='--', lw=1.2,
                   label=f'Median active: {median_active:.4f}')
        ax.axhline(thresh, color='orange', ls='--', lw=1.2,
                   label=f'Dead threshold ({DEAD_STRIP_RATE_FRACTION:.0%}): {thresh:.4f}')
        n_dead  = len(dead_set)
        n_live  = len(positions) - n_dead
        ax.legend(fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel('Hits / event')
        ax.set_title(f'{n_dead} dead  |  {n_live} live', fontsize=10)

    fig.tight_layout()
    out_path = OUT_DIR.parent / 'dead_strips.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Dead strip plot saved → {out_path}')
    plt.show(block=False)
    plt.pause(0.5)


def draw_event(fig, gs_axes: dict, r: EventResult,
               wf_data: Optional[dict] = None,
               dead1: Optional[np.ndarray] = None,
               dead2: Optional[np.ndarray] = None,
               strip_pitch: float = 0.78):
    """Redraw all panels for one event."""
    # Remove colorbar axes created by previous draw (plt.colorbar adds new axes to fig)
    main_axes = set(gs_axes.values())
    for ax in list(fig.axes):
        if ax not in main_axes:
            ax.remove()
    for ax in gs_axes.values():
        ax.cla()

    wf = wf_data or {}
    d1 = dead1 if dead1 is not None else np.array([])
    d2 = dead2 if dead2 is not None else np.array([])

    _draw_projection(gs_axes['feu1'], r, feu=1, dead=d1, strip_pitch=strip_pitch)
    _draw_projection(gs_axes['feu2'], r, feu=2, dead=d2, strip_pitch=strip_pitch)
    _draw_xy_scatter(gs_axes['xy'], r)
    _draw_waveform(gs_axes['wf1'], wf, feu=1, r=r)
    _draw_waveform(gs_axes['wf2'], wf, feu=2, r=r)

    pair_colors = {0: 'grey', 1: 'darkorange', 2: 'green'}
    title_color = pair_colors.get(r.n_pairs, 'red')
    n_str   = f'{r.n_pairs} pair(s)' if r.n_pairs else 'no pairs'
    tr_str  = f'FEU1: {len(r.tracks1)} tr  FEU2: {len(r.tracks2)} tr'
    iso_str = f'isolated: {(~r.keep1).sum()}/{(~r.keep2).sum()} (FEU1/FEU2)'
    param_str = (f'Amp>{AMP_THRESHOLD}  Iso:{ISO_POS_MM}mm/{ISO_TIME_NS:.0f}ns  '
                 f'Road:{ROAD_WIDTH_MM}mm  Gap:{MAX_TIME_GAP_NS:.0f}ns  '
                 f'MinHits:{MIN_TRACK_HITS}  Dead:{len(d1)}/{len(d2)}')
    fig.suptitle(
        f'Event {r.event_id}  —  {n_str}  [{tr_str} | {iso_str}]\n{param_str}',
        fontsize=10, color=title_color, fontweight='bold'
    )


def _make_figure() -> tuple:
    """Create figure with 2×6 GridSpec. Returns (fig, axes_dict)."""
    fig = plt.figure(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.08,
                        wspace=0.38, hspace=0.45)
    gs  = gridspec.GridSpec(2, 6, figure=fig)
    axes = {
        'feu1': fig.add_subplot(gs[0, 0:3]),
        'feu2': fig.add_subplot(gs[0, 3:6]),
        'xy':   fig.add_subplot(gs[1, 0:2]),
        'wf1':  fig.add_subplot(gs[1, 2:4]),
        'wf2':  fig.add_subplot(gs[1, 4:6]),
    }
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Interactive browser
# ─────────────────────────────────────────────────────────────────────────────

class EventBrowser:
    def __init__(self, candidate_ids: np.ndarray, results: dict,
                 dead1, dead2, strip_pitch: float,
                 wf_index: dict):
        self.all_ids     = candidate_ids
        self.two_pair    = np.array([eid for eid in candidate_ids
                                     if results[eid].n_pairs == 2])
        self.results     = results
        self.dead1       = dead1
        self.dead2       = dead2
        self.strip_pitch = strip_pitch
        self.wf_index    = wf_index
        self._two_mode   = False
        self.active      = self.all_ids
        self.idx         = 0

        self.fig, self.axs = _make_figure()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    def _on_key(self, ev):
        k = ev.key
        if k in ('n', 'right'):
            self.idx = (self.idx + 1) % len(self.active); self._draw()
        elif k in ('p', 'left'):
            self.idx = (self.idx - 1) % len(self.active); self._draw()
        elif k == '2':
            self._two_mode = not self._two_mode
            self.active = self.two_pair if self._two_mode else self.all_ids
            self.idx = 0
            print(f'  Mode: {"2-pair only" if self._two_mode else "all candidates"} '
                  f'({len(self.active)} events)')
            self._draw()
        elif k == 's':
            self._save()
        elif k == 'q':
            plt.close('all')

    def _draw(self):
        if len(self.active) == 0:
            return
        eid     = self.active[self.idx]
        result  = self.results[eid]
        wf_data = load_waveform(eid, self.wf_index)
        draw_event(self.fig, self.axs, result, wf_data,
                   self.dead1, self.dead2, self.strip_pitch)
        mode = ' [2-pair]' if self._two_mode else ''
        print(f'  [{self.idx+1}/{len(self.active)}] event {eid}{mode}'
              f'  — {result.n_pairs} pair(s)')
        self.fig.canvas.draw_idle()

    def _save(self):
        eid  = self.active[self.idx]
        path = OUT_DIR / f'event_{eid:06d}_pairs{self.results[eid].n_pairs}.png'
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Saved {path}')

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    det  = build_detector()
    df   = load_and_map(det)
    dead1, dead2, pitch = compute_dead_strips(df)

    print('\nShowing dead strip plot — close or leave open, then the browser will launch …')
    plot_dead_strips(df, dead1, dead2, pitch)

    cids = filter_candidates(df)
    print(f'\n{len(cids)} track candidates')

    print('Running tracking on all candidates …')
    grouped = df.groupby('eventId')
    results = {eid: process_event(grouped.get_group(eid), dead1, dead2, pitch)
               for eid in cids if eid in grouped.groups}

    pair_counts = Counter(r.n_pairs for r in results.values())
    print('\nEvent classification:')
    for n in sorted(pair_counts):
        print(f'  {n} pair(s): {pair_counts[n]:4d} events')

    wf_index = build_waveform_index()

    print('\nKey bindings:')
    print('  n/→  next      p/←  prev      2  toggle 2-pair mode')
    print('  s    save      q    quit\n')

    browser = EventBrowser(cids, results, dead1, dead2, pitch, wf_index)
    browser.show()


if __name__ == '__main__':
    main()
