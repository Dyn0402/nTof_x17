#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_explorer.py

Interactive single-event browser + track finder.

Focuses on mx17_3:
  FEU 1 (X-measuring strips, run in Y)  →  x_mm positions
  FEU 2 (Y-measuring strips, run in X)  →  y_mm positions

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
# run = 'run_70'
# subrun = 'resist_final_790V'
# run = 'run_49'
# subrun = 'hv_scan_drift_600_resist_530'
# run = 'run_52'
# subrun = 'long_run'
# run = 'run_51'
# subrun = 'hv_scan_drift_600_resist_530'
run = 'run_67'
subrun = 'run_2'

SUBRUN_DIR      = Path(f'/media/dylan/data/x17/may_beam/runs/{run}/{subrun}')
RUN_CONFIG_PATH = SUBRUN_DIR.parent / 'run_config.json'
MAP_CSV         = _ROOT / 'mx17_m1_map.csv'
DECODED_DIR     = SUBRUN_DIR / 'decoded_root'
OUT_DIR         = Path(__file__).parent / 'output' / run / subrun / 'events'
WF_INDEX_CACHE  = OUT_DIR / 'wf_index.pkl'

# ── Hit filter ────────────────────────────────────────────────────────────────
AMP_THRESHOLD        = 50    # ADC  — minimum hit amplitude
TIME_WIN_MIN         = 300.0  # ns   — beam drift window lower edge
TIME_WIN_MAX         = 1100.0 # ns   — beam drift window upper edge
GAMMA_FLASH_MAX_HITS = 200    # hits per FEU per event — above this = gamma flash
GAMMA_FLASH_AMP      = 500    # ADC — amplitude threshold for gamma flash hit counting
GAMMA_FLASH_TIME_NS  = 1000.0 # ns  — only count hits before this time for gamma flash
HIGH_AMP_THRESHOLD   = 500   # ADC  — candidate events must have ≥1 hit above this

# ── Drift velocity ────────────────────────────────────────────────────────────
DRIFT_GAS = 'Ne/iC4H10 95/5'   # gas mixture used in this run
DRIFT_HV  = 600.0               # V — drift high voltage (set to match run)

# (gas, drift_field_V_per_mm) → full drift time in ns, eyeballed from long tracks
# drift_field = DRIFT_HV / drift_gap_mm   (rounded to nearest 0.5 V/mm)
DRIFT_TIME_LUT = {
    ('Ne/iC4H10 95/5', 37.5): 500.0,   # 600 V / 16 mm = 37.5 V/mm
}

# ── Dead strip detection ──────────────────────────────────────────────────────
DEAD_STRIP_AMP_THRESHOLD = 500   # ADC — amplitude used for dead-strip rate
DEAD_STRIP_RATE_FRACTION = 0.10  # strips below this × median rate are flagged dead

# ── Tracking parameters (edit to iterate) ────────────────────────────────────
ISO_POS_MM         = 5.0   # mm  — isolation neighbourhood radius in position
ISO_TIME_NS        = 200.0 # ns  — isolation neighbourhood radius in time
ROAD_WIDTH_MM      = 3.0   # mm  — road width once slope is established
ROAD_WIDTH_SEED_MM = 7.0   # mm  — road width before slope is known
MAX_TIME_GAP_NS    = 250.0 # ns  — max time gap between consecutive track hits
MIN_TRACK_HITS     = 2     #     — minimum hits to accept a track
MAX_MISSED         = 3     #     — max consecutive time-step misses before stopping
MAX_STRIP_GAP      = 3     #     — max skipped live strips between hits
PAIR_MIN_IOU       = 0.20  #     — minimum time-overlap IoU to accept an X-Y pair
SEED_AMP_THRESHOLD = 600   # ADC — minimum amplitude to seed a new track; lower-amp
                            #       hits can still join an existing track once seeded

# ── Cross-dimensional merge parameters ───────────────────────────────────────
# Fragments in one dimension are merged when: (a) multiple tracks overlap the
# partner-dimension track's time window, AND (b) all their hits together fit a
# single straight line with RMS < ROAD_WIDTH_MM (the primary quality gate that
# prevents merging genuinely-separate particles at different positions).
CROSS_DIM_STRIP_GAP_SCALE = 2   # max_strip_gap multiplier in the cross-dim pass
CROSS_DIM_MISSED_SCALE    = 2   # max_missed multiplier in the cross-dim pass

# ── Browser startup ───────────────────────────────────────────────────────────
START_EVENT_ID = None  # jump directly to this event ID when the browser opens

# ── Debug flag (set by --debug-event CLI; also toggleable at top) ─────────────
CROSS_DIM_DEBUG = False  # verbose per-pair output from _cross_dim_merge

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


def compute_drift_velocity() -> tuple:
    """
    Returns (drift_gap_mm, v_drift_mm_ns) from run_config + DRIFT_TIME_LUT.
    Prints the computed values.  v_drift_mm_ns = 0 if gas/field not in LUT.
    """
    cfg = json.loads(RUN_CONFIG_PATH.read_text())
    det_cfg = next(d for d in cfg['detectors'] if d['name'] == 'mx17_3')
    gap_str = det_cfg.get('drift_gap', '')
    try:
        drift_gap_mm = float(gap_str.split()[0])
    except (ValueError, IndexError):
        print(f'[warn] Cannot parse drift_gap: {gap_str!r}')
        return 16.0, 0.0

    drift_field_raw = DRIFT_HV / drift_gap_mm           # V/mm
    drift_field     = round(drift_field_raw * 2) / 2    # nearest 0.5 V/mm for LUT key

    key = (DRIFT_GAS, drift_field)
    if key not in DRIFT_TIME_LUT:
        # Fall back to nearest field entry for the same gas
        gas_keys = [(g, f) for (g, f) in DRIFT_TIME_LUT if g == DRIFT_GAS]
        if not gas_keys:
            print(f'[warn] Gas {DRIFT_GAS!r} not in DRIFT_TIME_LUT — drift velocity unknown')
            return drift_gap_mm, 0.0
        key = min(gas_keys, key=lambda k: abs(k[1] - drift_field))
        print(f'[warn] No LUT entry for {drift_field:.2f} V/mm; ' 
              f'using nearest {key[1]:.2f} V/mm')

    full_drift_ns = DRIFT_TIME_LUT[key]
    v_drift       = drift_gap_mm / full_drift_ns   # mm/ns

    print(f'Drift: gap={drift_gap_mm:.1f} mm  HV={DRIFT_HV:.0f} V  '
          f'field={drift_field_raw:.2f} V/mm  '
          f'full_drift={full_drift_ns:.0f} ns  '
          f'v_drift={v_drift:.4f} mm/ns')
    return drift_gap_mm, v_drift


def build_ch_to_pos(det: Detector) -> tuple:
    """Precompute channel→position dicts for FEU 1 (x_mm) and FEU 2 (y_mm)."""
    ch_to_x = {}
    ch_to_y = {}
    for ch in range(512):
        p1 = det.map_hit(1, ch)
        if p1 and p1[0] is not None:
            ch_to_x[ch] = float(p1[0])
        p2 = det.map_hit(2, ch)
        if p2 and p2[1] is not None:
            ch_to_y[ch] = float(p2[1])
    return ch_to_x, ch_to_y


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
    """Event IDs with ≥1 qualifying hit in each FEU, excluding gamma flash events."""
    # Gamma flash cut: amp > GAMMA_FLASH_AMP, time < GAMMA_FLASH_TIME_NS, per FEU per event
    gf = df[(df['amplitude'] > GAMMA_FLASH_AMP) & (df['time'] < GAMMA_FLASH_TIME_NS)]
    hits_per_feu = gf.groupby(['eventId', 'feu']).size().unstack(fill_value=0)
    gamma_mask = (hits_per_feu > GAMMA_FLASH_MAX_HITS).any(axis=1)
    gamma_ids = set(hits_per_feu.index[gamma_mask].tolist())
    n_flash = len(gamma_ids)

    qual = df[
        (df['amplitude'] > AMP_THRESHOLD) &
        (df['time'] >= TIME_WIN_MIN) &
        (df['time'] <= TIME_WIN_MAX)
    ]
    has1 = set(qual.loc[qual['feu'] == 1, 'eventId'])
    has2 = set(qual.loc[qual['feu'] == 2, 'eventId'])

    high = df[
        (df['amplitude'] > HIGH_AMP_THRESHOLD) &
        (df['time'] >= TIME_WIN_MIN) &
        (df['time'] <= TIME_WIN_MAX)
    ]
    has_high = set(high['eventId'])

    candidates = ((has1 & has2) - gamma_ids) & has_high
    n_low_amp  = len((has1 & has2) - gamma_ids) - len(candidates)
    print(f'  Gamma flash events removed: {n_flash}  '
          f'(amp>{GAMMA_FLASH_AMP}, t<{GAMMA_FLASH_TIME_NS:.0f}ns, >{GAMMA_FLASH_MAX_HITS} hits/FEU)')
    print(f'  Low-amplitude events removed: {n_low_amp}  '
          f'(no hit amp>{HIGH_AMP_THRESHOLD} in time window)')
    return np.array(sorted(candidates))


# ─────────────────────────────────────────────────────────────────────────────
# Dead strip detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_dead_strips(df: pd.DataFrame) -> tuple:
    """
    Find strip positions with hit rate below DEAD_STRIP_RATE_FRACTION × median.
    Returns (dead1_x_mm, dead2_y_mm, strip_pitch_mm).
    """
    df_th  = df[df['amplitude'] >= DEAD_STRIP_AMP_THRESHOLD]
    n_evts = max(df['eventId'].nunique(), 1)

    all_x = np.sort(df[df['feu'] == 1]['x_mm'].dropna().unique())
    all_y = np.sort(df[df['feu'] == 2]['y_mm'].dropna().unique())

    x_cnt = (df_th[(df_th['feu'] == 1) & df_th['x_mm'].notna()]
             ['x_mm'].value_counts())
    y_cnt = (df_th[(df_th['feu'] == 2) & df_th['y_mm'].notna()]
             ['y_mm'].value_counts())

    x_rates = pd.Series({p: x_cnt.get(p, 0) / n_evts for p in all_x})
    y_rates = pd.Series({p: y_cnt.get(p, 0) / n_evts for p in all_y})

    def _dead(rates):
        active = rates[rates > 0]
        if len(active) == 0:
            return np.array([])
        thresh = float(np.median(active)) * DEAD_STRIP_RATE_FRACTION
        return np.sort(rates.index[rates < thresh].to_numpy(dtype=float))

    dead1 = _dead(x_rates)
    dead2 = _dead(y_rates)

    pitch1 = float(np.median(np.diff(all_x))) if len(all_x) >= 2 else 0.78
    pitch2 = float(np.median(np.diff(all_y))) if len(all_y) >= 2 else 0.78
    pitch  = float(0.5 * (pitch1 + pitch2))

    print(f'Dead strips: FEU1 x_mm = {len(dead1)},  '
          f'FEU2 y_mm = {len(dead2)}   (pitch ≈ {pitch:.3f} mm)')
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
    pos1:  np.ndarray   # FEU 1 → x_mm
    time1: np.ndarray   # μs
    amp1:  np.ndarray
    pos2:  np.ndarray   # FEU 2 → y_mm
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


def _track_rms(pos: np.ndarray, time: np.ndarray) -> float:
    """RMS residual of a set of hits from their best-fit straight line."""
    if len(pos) < 2:
        return float('inf')
    sl, ic = _fit_slope(time, pos)
    if sl is None:
        return float('inf')
    return float(np.sqrt(np.mean((pos - (sl * time + ic)) ** 2)))


def _cross_dim_merge(anchor_tracks, anchor_time,
                     target_tracks, target_pos, target_time,
                     pairs_at, target_dead, strip_pitch, gap_us):
    """
    Cross-dimensional guided merge pass.

    Uses each paired anchor-track's time window to detect and merge fragmented
    target tracks.  For each (anchor_i, target_j, iou) in pairs_at:

      1. Counts how many anchor tracks overlap [t_lo, t_hi] — the multi-particle
         guard.  If N anchor tracks are present we only try to merge when the
         target has *more* than N tracks in the window.
      2. Pools all target-track hits in the window plus nearby unassigned hits.
      3. Quality gate: fits a single line to the pooled hits; only proceeds if
         RMS < ROAD_WIDTH_MM.  Genuinely separate particles at different positions
         will fail this check and are left alone.
      4. Re-tracks with a time-gap large enough to bridge the full anchor span
         and accepts only if the result has fewer tracks than before.

    anchor_tracks / anchor_time  : hit-index arrays and filtered times, anchor dim
    target_tracks / target_pos / target_time : same for the dimension being fixed
    pairs_at : [(anchor_idx, target_idx, iou), ...]
    target_dead, strip_pitch, gap_us : tracking geometry

    Returns updated target_tracks list.  Pairs are re-computed by the caller.
    """
    if not pairs_at or len(target_tracks) <= 1:
        return list(target_tracks)

    n_t = len(target_pos)
    on_target = np.zeros(n_t, bool)
    for t in target_tracks:
        on_target[t] = True

    # Pre-compute anchor time intervals for the multi-particle guard
    anchor_intervals = [(float(anchor_time[trk].min()), float(anchor_time[trk].max()))
                        for trk in anchor_tracks]

    replaced     = set()
    extra_tracks = []

    kw = dict(
        road_width_mm      = ROAD_WIDTH_MM,
        road_width_seed_mm = ROAD_WIDTH_SEED_MM,
        min_hits           = MIN_TRACK_HITS,
        max_missed         = MAX_MISSED    * CROSS_DIM_MISSED_SCALE,
        max_strip_gap      = MAX_STRIP_GAP * CROSS_DIM_STRIP_GAP_SCALE,
        dead_strip_positions = target_dead,
        strip_pitch        = strip_pitch,
    )

    for ai, _ti, _iou in pairs_at:
        if ai >= len(anchor_tracks):
            continue
        t_lo, t_hi = anchor_intervals[ai]
        t_span = t_hi - t_lo
        if t_span <= 0:
            continue

        # Multi-particle guard: count how many anchor tracks span this window
        n_anchor_in_win = sum(
            1 for (lo, hi) in anchor_intervals
            if lo <= t_hi and hi >= t_lo
        )

        # Find target tracks overlapping or abutting [t_lo, t_hi].
        # Using gap_us proximity so tracks that end/start just outside the anchor
        # window (due to tracking fragmentation) are still considered.
        overlapping = [
            k for k, trk in enumerate(target_tracks)
            if k not in replaced
            and float(target_time[trk].max()) >= t_lo - gap_us
            and float(target_time[trk].min()) <= t_hi + gap_us
        ]

        if CROSS_DIM_DEBUG:
            print(f'  [cdm] anchor t=[{t_lo*1000:.0f},{t_hi*1000:.0f}]ns '
                  f'n_anchor={n_anchor_in_win}  overlapping={len(overlapping)}')
            for k in overlapping:
                trk = target_tracks[k]
                t_t = target_time[trk]
                rk  = _track_rms(target_pos[trk], t_t)
                print(f'    target[{k}]: t=[{t_t.min()*1000:.0f},{t_t.max()*1000:.0f}]ns '
                      f'n_hits={len(trk)}  rms={rk:.3f}mm')

        # Only act when target has more tracks than anchor evidence warrants
        if len(overlapping) <= n_anchor_in_win:
            if CROSS_DIM_DEBUG:
                print(f'    → skip (overlapping {len(overlapping)} ≤ n_anchor {n_anchor_in_win})')
            continue

        # Pool: on-track hits from overlapping tracks + unassigned hits in window
        hit_set = set()
        for k in overlapping:
            hit_set.update(target_tracks[k].tolist())
        for h in range(n_t):
            if not on_target[h] and t_lo - gap_us <= target_time[h] <= t_hi + gap_us:
                hit_set.add(h)

        hit_arr  = np.array(sorted(hit_set))
        sub_pos  = target_pos[hit_arr]
        sub_time = target_time[hit_arr]

        # Quality gate: pooled hits must fit a straight line within the road width.
        # Two genuine particles at different positions will fail here (large RMS).
        merged_rms = _track_rms(sub_pos, sub_time)
        if CROSS_DIM_DEBUG:
            gate = 'PASS' if merged_rms < ROAD_WIDTH_MM else 'FAIL'
            print(f'    merged_rms={merged_rms:.3f}mm  gate<{ROAD_WIDTH_MM:.1f}mm → {gate}  '
                  f'(pooled {len(hit_arr)} hits, {len(hit_arr)-sum(on_target[hit_arr])} unassigned)')
        if merged_rms >= ROAD_WIDTH_MM:
            continue

        # Re-track pass 1 then pass 2.
        # Pass 1 uses exclusive hit assignment, so long fragments that share no
        # overlapping time can still end up as two tracks.  Pass 2 (find_tracks_1d_pass2)
        # then extends each track without the exclusivity constraint, allowing them to
        # absorb each other's hits and merge into one.
        big_gap = max(t_span * 1.5, gap_us)
        sub_p1, _ = find_tracks_1d(sub_pos, sub_time,
                                    max_time_gap_us=big_gap, **kw)
        sub_ext, sub_new = find_tracks_1d_pass2(sub_p1, sub_pos, sub_time,
                                                 max_time_gap_us=big_gap, **kw)
        all_sub = sub_ext + sub_new
        # Dedup + subset removal (pass2 can produce overlapping tracks)
        seen_fs, sub_deduped = [], []
        for t in all_sub:
            fs = frozenset(t.tolist())
            if fs not in seen_fs:
                seen_fs.append(fs); sub_deduped.append(t)
        sub_dsets = [set(t.tolist()) for t in sub_deduped]
        sub_tracks = [sub_deduped[i] for i, s in enumerate(sub_dsets)
                      if not any(s < sub_dsets[j]
                                 for j in range(len(sub_dsets)) if j != i)]

        # Dominance drop: if one sub-track fully covers another's time range with
        # strictly more hits, the smaller one is a fragmented duplicate — drop it.
        if len(sub_tracks) > 1:
            t_info = [(float(sub_time[st].min()), float(sub_time[st].max()), len(st))
                      for st in sub_tracks]
            dominated = set()
            for a, (lo_a, hi_a, n_a) in enumerate(t_info):
                for b, (lo_b, hi_b, n_b) in enumerate(t_info):
                    if a == b or b in dominated:
                        continue
                    if lo_b <= lo_a and hi_b >= hi_a and n_b > n_a:
                        dominated.add(a)
            if dominated:
                if CROSS_DIM_DEBUG:
                    print(f'    dominance drop: removing sub-tracks {sorted(dominated)}')
                sub_tracks = [st for i, st in enumerate(sub_tracks) if i not in dominated]

        if CROSS_DIM_DEBUG:
            n_st = len(sub_tracks)
            print(f'    re-track big_gap={big_gap*1000:.0f}ns → p1:{len(sub_p1)} p2+dom:{n_st} tracks  '
                  f'(need < {len(overlapping)} to improve)')
            for si, st in enumerate(sub_tracks):
                t_st = sub_time[st]
                print(f'      sub_track[{si}]: t=[{t_st.min()*1000:.0f},{t_st.max()*1000:.0f}]ns '
                      f'n_hits={len(st)}')

        if not sub_tracks or len(sub_tracks) >= len(overlapping):
            continue  # no improvement

        sep_rms_vals = [_track_rms(target_pos[target_tracks[k]],
                                    target_time[target_tracks[k]])
                        for k in overlapping]
        avg_sep_rms = float(np.nanmean([r for r in sep_rms_vals
                                        if np.isfinite(r)] or [0.0]))
        print(f'    [cross-dim] {len(overlapping)} → {len(sub_tracks)} tracks  '
              f'merged_rms={merged_rms:.2f} mm  avg_sep_rms={avg_sep_rms:.2f} mm')

        for st in sub_tracks:
            extra_tracks.append(hit_arr[st])
        for k in overlapping:
            replaced.add(k)

    if not replaced:
        return list(target_tracks)

    final = [t for k, t in enumerate(target_tracks) if k not in replaced]
    final.extend(extra_tracks)

    # Deduplicate and drop subsets (same pattern as in _track())
    seen, deduped = [], []
    for t in final:
        fs = frozenset(t.tolist())
        if fs not in seen:
            seen.append(fs)
            deduped.append(t)
    dsets = [set(t.tolist()) for t in deduped]
    return [deduped[i] for i, s in enumerate(dsets)
            if not any(s < dsets[j] for j in range(len(dsets)) if j != i)]


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

    pos1, time1, amp1 = _hits(1, 'x_mm')
    pos2, time2, amp2 = _hits(2, 'y_mm')

    def _isolation(pos, time):
        if len(pos) < 2:
            return np.ones(len(pos), bool)
        return remove_isolated_hits(pos, time, ISO_POS_MM, iso_us)

    keep1 = _isolation(pos1, time1)
    keep2 = _isolation(pos2, time2)
    pos1f,  time1f = pos1[keep1],  time1[keep1]
    pos2f,  time2f = pos2[keep2],  time2[keep2]
    amp1f,  amp2f  = amp1[keep1],  amp2[keep2]

    kw = dict(road_width_mm=ROAD_WIDTH_MM,
              road_width_seed_mm=ROAD_WIDTH_SEED_MM,
              max_time_gap_us=gap_us,
              min_hits=MIN_TRACK_HITS,
              max_missed=MAX_MISSED,
              max_strip_gap=MAX_STRIP_GAP,
              strip_pitch=strip_pitch)

    def _track(pos, time, amp, dead):
        if len(pos) < MIN_TRACK_HITS:
            return []
        # Hits below SEED_AMP_THRESHOLD cannot start a track but can join one.
        excl = (amp < SEED_AMP_THRESHOLD) if SEED_AMP_THRESHOLD > AMP_THRESHOLD else None
        p1, _ = find_tracks_1d(pos, time, dead_strip_positions=dead,
                                excl_mask=excl, **kw)
        ext, new = find_tracks_1d_pass2(p1, pos, time, dead_strip_positions=dead,
                                         excl_mask=excl, **kw)
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

    tracks1 = _track(pos1f, time1f, amp1f, dead1)
    tracks2 = _track(pos2f, time2f, amp2f, dead2)
    pairs   = _pair_tracks(tracks1, time1f, tracks2, time2f, gap_us)

    # Cross-dimensional guided merge: pass 1 — use X windows to fix Y fragmentation
    if pairs:
        tracks2 = _cross_dim_merge(
            tracks1, time1f,
            tracks2, pos2f, time2f,
            pairs, dead2, strip_pitch, gap_us)
        pairs = _pair_tracks(tracks1, time1f, tracks2, time2f, gap_us)

        # Pass 2 (symmetric) — use Y windows to fix X fragmentation
        pairs_yx = [(j, i, iou) for i, j, iou in pairs]
        tracks1 = _cross_dim_merge(
            tracks2, time2f,
            tracks1, pos1f, time1f,
            pairs_yx, dead1, strip_pitch, gap_us)
        pairs = _pair_tracks(tracks1, time1f, tracks2, time2f, gap_us)

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
                     strip_pitch: float = 0.78,
                     time_ns: bool = False):
    """Position vs drift time with amplitude-sized hits, dead strip bands, track fits.
    time_ns=True displays the x-axis in ns instead of μs."""
    t_scale = 1000.0 if time_ns else 1.0
    t_unit  = 'ns'   if time_ns else 'μs'
    win_lo  = TIME_WIN_MIN       if time_ns else TIME_WIN_MIN / 1000.0
    win_hi  = TIME_WIN_MAX       if time_ns else TIME_WIN_MAX / 1000.0

    if feu == 1:
        pos, time, amp, keep, tracks = r.pos1, r.time1, r.amp1, r.keep1, r.tracks1
        pos_lbl  = 'X pos [mm]  (FEU 1 — X strips)'
        pair_idx = {i: k for k, (i, j, _) in enumerate(r.pairs)}
    else:
        pos, time, amp, keep, tracks = r.pos2, r.time2, r.amp2, r.keep2, r.tracks2
        pos_lbl  = 'Y pos [mm]  (FEU 2 — Y strips)'
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
        ax.scatter(time[~keep] * t_scale, pos[~keep],
                   s=_amp_to_size(amp[~keep]),
                   c='tomato', marker='x', linewidths=0.9, zorder=2,
                   label=f'Isolated ({(~keep).sum()})')

    # Hits on track vs noise
    on_track = np.zeros(len(posf), bool)
    for t in tracks:
        on_track[t] = True

    # Noise hits — dark grey with faint edge so they're visible but clearly secondary
    if (~on_track).any():
        ax.scatter(timef[~on_track] * t_scale, posf[~on_track],
                   s=_amp_to_size(ampf[~on_track]),
                   c='#888888', alpha=0.85, zorder=3,
                   edgecolors='#444444', linewidths=0.25,
                   label=f'Noise ({(~on_track).sum()})')

    # Track hits — colored by pair, amplitude-sized
    for i, idx in enumerate(tracks):
        col  = _track_color(pair_idx[i]) if i in pair_idx else '#666666'
        t_tr = timef[idx]   # μs, used for fit
        p_tr = posf[idx]
        lbl  = (f'Pair {pair_idx[i]+1} ({len(idx)} hits)'
                if i in pair_idx else f'Unmatched ({len(idx)} hits)')
        ax.scatter(t_tr * t_scale, p_tr, s=_amp_to_size(ampf[idx]),
                   color=col, zorder=5, edgecolors='k', linewidths=0.4, label=lbl)
        if len(idx) >= 2:
            sl, ic = _fit_slope(t_tr, p_tr)   # sl in mm/μs, ic in mm
            if sl is not None:
                dt = (t_tr.max() - t_tr.min()) * 0.12
                tf = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 120)
                # x display scaled to chosen unit; y = sl*tf+ic stays in mm
                ax.plot(tf * t_scale, sl * tf + ic,
                        color=col, lw=1.5, ls='--', zorder=4, alpha=0.85)

    ax.axvspan(win_lo, win_hi, alpha=0.05, color='green', zorder=0)
    ax.set_xlabel(f'Drift time [{t_unit}]')
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
        zt_disp    = zoom_time * t_scale
        t_lo, t_hi = zt_disp.min(), zt_disp.max()
        p_pad = max(4.0,  (p_hi - p_lo) * 0.18)
        t_pad = max(20.0 if time_ns else 0.02, (t_hi - t_lo) * 0.18)
        ax.set_ylim(p_lo - p_pad, p_hi + p_pad)
        ax.set_xlim(t_lo - t_pad, t_hi + t_pad)


def _draw_xy_scatter(ax, r: EventResult, v_drift: float = 0.0):
    """X-Y position scatter: earliest hit = mesh crossing, arrow from late to early time.
    v_drift in mm/ns is used to convert track slopes to 3-D angles."""
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title('Reconstructed positions + track direction (arrow: drift top → mesh)')
    ax.grid(True, alpha=0.2)

    if not r.pairs:
        ax.text(0.5, 0.5, 'No pairs found', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='grey')
        return

    posf1 = r.pos1[r.keep1]; tf1 = r.time1[r.keep1]
    posf2 = r.pos2[r.keep2]; tf2 = r.time2[r.keep2]

    # v_drift in mm/ns → mm/μs for comparison with slopes (which are mm/μs)
    v_drift_mm_us = v_drift * 1000.0

    angle_lines = []   # collect angle text for the annotation box

    for k, (i, j, iou) in enumerate(r.pairs):
        col   = _track_color(k)
        t1, t2 = r.tracks1[i], r.tracks2[j]

        # Earliest hit → mesh crossing (reconstructed position, arrow head)
        x_mesh = posf1[t1[np.argmin(tf1[t1])]]
        y_mesh = posf2[t2[np.argmin(tf2[t2])]]

        # Latest hit → top of drift volume (arrow tail)
        x_top  = posf1[t1[np.argmax(tf1[t1])]]
        y_top  = posf2[t2[np.argmax(tf2[t2])]]

        ax.scatter(x_mesh, y_mesh, s=220, color=col, zorder=5,
                   edgecolors='k', linewidths=0.6,
                   label=f'Pair {k+1}  IoU={iou:.2f}')

        # Arrow from top-of-drift (late time) → mesh crossing (early time)
        if (x_mesh - x_top)**2 + (y_mesh - y_top)**2 > 0.01:
            ax.annotate('', xy=(x_mesh, y_mesh), xytext=(x_top, y_top),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=1.8, mutation_scale=16),
                        zorder=4)

        for p_x in posf1[t1]:
            ax.axvline(p_x, color=col, lw=0.4, alpha=0.3)
        for p_y in posf2[t2]:
            ax.axhline(p_y, color=col, lw=0.4, alpha=0.3)

        # ── Track angle ────────────────────────────────────────────────────
        sl_x, _ = _fit_slope(tf1[t1], posf1[t1])   # mm/μs
        sl_y, _ = _fit_slope(tf2[t2], posf2[t2])   # mm/μs

        if sl_x is not None and sl_y is not None and v_drift_mm_us > 0:
            # 3-D angles: slopes are dx/dt and dy/dt, drift sets dz/dt = v_drift_mm_us
            theta_x   = np.degrees(np.arctan(abs(sl_x) / v_drift_mm_us))
            theta_y   = np.degrees(np.arctan(abs(sl_y) / v_drift_mm_us))
            theta_tot = np.degrees(np.arctan(
                np.sqrt(sl_x**2 + sl_y**2) / v_drift_mm_us))
            phi       = np.degrees(np.arctan2(sl_y, sl_x))
            angle_lines.append(
                f'Pair {k+1}: θ={theta_tot:.1f}°  '
                f'(θx={theta_x:.1f}°, θy={theta_y:.1f}°, φ={phi:.1f}°)')
        elif sl_x is not None and sl_y is not None:
            # v_drift unknown: show raw slopes and 2-D projection angle
            phi = np.degrees(np.arctan2(sl_y, sl_x))
            angle_lines.append(
                f'Pair {k+1}: φ={phi:.1f}°  '
                f'(slX={sl_x:.1f} mm/μs, slY={sl_y:.1f} mm/μs)  [v_drift unknown]')

    if angle_lines:
        ax.text(0.02, 0.02, '\n'.join(angle_lines),
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

    ax.legend(fontsize=8, loc='upper right')


def _draw_waveform(ax, wf_data: dict, feu: int, r: EventResult, cax=None):
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
    plt.colorbar(sm, cax=cax, label='Channel') if cax is not None else None

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

    all_x = np.sort(df[df['feu'] == 1]['x_mm'].dropna().unique())
    all_y = np.sort(df[df['feu'] == 2]['y_mm'].dropna().unique())
    x_cnt = df_th[(df_th['feu'] == 1) & df_th['x_mm'].notna()]['x_mm'].value_counts()
    y_cnt = df_th[(df_th['feu'] == 2) & df_th['y_mm'].notna()]['y_mm'].value_counts()
    x_rates = np.array([x_cnt.get(p, 0) / n_evts for p in all_x])
    y_rates = np.array([y_cnt.get(p, 0) / n_evts for p in all_y])

    def _thresh(rates):
        active = rates[rates > 0]
        return float(np.median(active)) * DEAD_STRIP_RATE_FRACTION if len(active) else 0.0

    thresh_x = _thresh(x_rates)
    thresh_y = _thresh(y_rates)
    dead1_set = set(dead1.tolist())
    dead2_set = set(dead2.tolist())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(
        f'Dead-strip analysis — {run}/{subrun}\n'
        f'{n_evts} events  |  amp ≥ {DEAD_STRIP_AMP_THRESHOLD}  |  '
        f'dead threshold = {DEAD_STRIP_RATE_FRACTION:.0%} of median',
        fontsize=11
    )

    for ax, positions, rates, dead_set, thresh, label in [
        (ax1, all_x, x_rates, dead1_set, thresh_x, 'FEU 1 — X position [mm]  (X strips)'),
        (ax2, all_y, y_rates, dead2_set, thresh_y, 'FEU 2 — Y position [mm]  (Y strips)'),
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
               strip_pitch: float = 0.78,
               v_drift: float = 0.0):
    """Redraw all panels for one event."""
    for ax in gs_axes.values():
        ax.cla()

    wf = wf_data or {}
    d1 = dead1 if dead1 is not None else np.array([])
    d2 = dead2 if dead2 is not None else np.array([])

    _draw_projection(gs_axes['feu1'], r, feu=1, dead=d1, strip_pitch=strip_pitch, time_ns=True)
    _draw_projection(gs_axes['feu2'], r, feu=2, dead=d2, strip_pitch=strip_pitch, time_ns=True)
    _draw_xy_scatter(gs_axes['xy'], r, v_drift=v_drift)
    _draw_waveform(gs_axes['wf1'], wf, feu=1, r=r, cax=gs_axes['cax_wf1'])
    _draw_waveform(gs_axes['wf2'], wf, feu=2, r=r, cax=gs_axes['cax_wf2'])

    pair_colors = {0: 'grey', 1: 'darkorange', 2: 'green'}
    title_color = pair_colors.get(r.n_pairs, 'red')
    n_str   = f'{r.n_pairs} pair(s)' if r.n_pairs else 'no pairs'
    tr_str  = f'FEU1: {len(r.tracks1)} tr  FEU2: {len(r.tracks2)} tr'
    iso_str = f'isolated: {(~r.keep1).sum()}/{(~r.keep2).sum()} (FEU1/FEU2)'
    v_str   = f'  v_drift={v_drift:.4f}mm/ns' if v_drift > 0 else ''
    param_str = (f'Amp>{AMP_THRESHOLD}  Iso:{ISO_POS_MM}mm/{ISO_TIME_NS:.0f}ns  '
                 f'Road:{ROAD_WIDTH_MM}mm  Gap:{MAX_TIME_GAP_NS:.0f}ns  '
                 f'MinHits:{MIN_TRACK_HITS}  Dead:{len(d1)}/{len(d2)}{v_str}')
    fig.suptitle(
        f'Event {r.event_id}  —  {n_str}  [{tr_str} | {iso_str}]\n{param_str}',
        fontsize=10, color=title_color, fontweight='bold'
    )


def _make_figure() -> tuple:
    """
    Create figure.  Returns (fig, axes_dict).

    The two waveform axes get dedicated colorbar axes created once via
    make_axes_locatable — their sizes never change across redraws.
    All axes (including colorbar axes) are included in the returned dict
    so draw_event can cla() all of them cleanly without creating new ones.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.88, bottom=0.08,
                        wspace=0.42, hspace=0.45)
    gs = gridspec.GridSpec(2, 6, figure=fig)

    ax_feu1 = fig.add_subplot(gs[0, 0:3])
    ax_feu2 = fig.add_subplot(gs[0, 3:6])
    ax_xy   = fig.add_subplot(gs[1, 0:2])
    ax_wf1  = fig.add_subplot(gs[1, 2:4])
    ax_wf2  = fig.add_subplot(gs[1, 4:6])

    # Pre-allocate colorbar axes — sizes stay fixed for the figure lifetime
    div1     = make_axes_locatable(ax_wf1)
    cax_wf1  = div1.append_axes('right', size='5%', pad=0.06)
    div2     = make_axes_locatable(ax_wf2)
    cax_wf2  = div2.append_axes('right', size='5%', pad=0.06)

    return fig, {
        'feu1':    ax_feu1,
        'feu2':    ax_feu2,
        'xy':      ax_xy,
        'wf1':     ax_wf1,
        'cax_wf1': cax_wf1,
        'wf2':     ax_wf2,
        'cax_wf2': cax_wf2,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Companion figure (all hits, no zoom)
# ─────────────────────────────────────────────────────────────────────────────

def _make_companion_figure() -> tuple:
    """
    Two-column companion with shared x per column.
      Col 0 (sharex within col): FEU 1 pos-vs-time (top) | FEU 2 pos-vs-time (bottom)
      Col 1 (sharex within col): FEU 1 raw waveform (top) | FEU 2 raw waveform (bottom)
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 8),
        sharex='col',
        gridspec_kw={'hspace': 0.05, 'wspace': 0.38},
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.09)
    return fig, {
        'pos1': axes[0, 0],
        'pos2': axes[1, 0],
        'wf1':  axes[0, 1],
        'wf2':  axes[1, 1],
    }


def _draw_companion_pos(ax, df_evt: pd.DataFrame, feu: int, r: EventResult,
                        dead: Optional[np.ndarray] = None,
                        strip_pitch: float = 0.78,
                        is_bottom: bool = False):
    """
    All combined_hits for this FEU, colour-coded by filter stage — no zoom.

    Layers (back to front):
      light grey ·    amp ≤ threshold
      medium grey ·   above threshold, outside time window
      red ×           above threshold, in window, isolated
      dark grey ·     above threshold, in window, noise (amp-sized)
      coloured ·      track hits by pair (amp-sized)
    """
    pos_col  = 'x_mm' if feu == 1 else 'y_mm'
    pos_lbl  = ('X pos [mm]  (FEU 1 — X strips)' if feu == 1
                else 'Y pos [mm]  (FEU 2 — Y strips)')
    pos_r    = r.pos1  if feu == 1 else r.pos2
    time_r   = r.time1 if feu == 1 else r.time2
    amp_r    = r.amp1  if feu == 1 else r.amp2
    keep_r   = r.keep1 if feu == 1 else r.keep2
    tracks_r = r.tracks1 if feu == 1 else r.tracks2
    pair_idx = ({i: k for k, (i, j, _) in enumerate(r.pairs)} if feu == 1
                else {j: k for k, (i, j, _) in enumerate(r.pairs)})

    if dead is not None and len(dead):
        hw = strip_pitch / 2.0
        for dp in dead:
            ax.axhspan(dp - hw, dp + hw, color='salmon', alpha=0.15, zorder=0)

    ax.axvspan(TIME_WIN_MIN / 1000, TIME_WIN_MAX / 1000, alpha=0.06, color='green', zorder=0)
    ax.axvline(TIME_WIN_MIN / 1000, color='green', lw=0.9, ls=':', alpha=0.8, zorder=1)
    ax.axvline(TIME_WIN_MAX / 1000, color='green', lw=0.9, ls=':', alpha=0.8, zorder=1)

    d = df_evt[(df_evt['feu'] == feu) & df_evt[pos_col].notna()]
    n_all = len(d)

    if n_all:
        t_all  = d['time'].to_numpy(float) / 1000.0   # ns → μs
        p_all  = d[pos_col].to_numpy(float)
        a_all  = d['amplitude'].to_numpy(float)
        t_raw  = d['time'].to_numpy(float)             # ns for window comparison

        below   = a_all <= AMP_THRESHOLD
        out_win = ((a_all > AMP_THRESHOLD) &
                   ((t_raw < TIME_WIN_MIN) | (t_raw > TIME_WIN_MAX)))

        if below.any():
            ax.scatter(t_all[below], p_all[below], s=5, c='#e0e0e0',
                       linewidths=0, zorder=2,
                       label=f'amp ≤ {AMP_THRESHOLD}  ({below.sum()})')
        if out_win.any():
            ax.scatter(t_all[out_win], p_all[out_win], s=7, c='#aaaaaa',
                       linewidths=0, zorder=3,
                       label=f'above thr, out of window  ({out_win.sum()})')

    # Overlay EventResult hits (amp+time filtered)
    if len(pos_r):
        if (~keep_r).any():
            ax.scatter(time_r[~keep_r], pos_r[~keep_r],
                       s=_amp_to_size(amp_r[~keep_r]),
                       c='tomato', marker='x', linewidths=0.9, zorder=4,
                       label=f'Isolated ({(~keep_r).sum()})')

        posf  = pos_r[keep_r]
        timef = time_r[keep_r]
        ampf  = amp_r[keep_r]
        on_track = np.zeros(len(posf), bool)
        for t in tracks_r:
            on_track[t] = True

        if (~on_track).any():
            ax.scatter(timef[~on_track], posf[~on_track],
                       s=_amp_to_size(ampf[~on_track]),
                       c='#888888', alpha=0.85, zorder=5,
                       edgecolors='#444444', linewidths=0.25,
                       label=f'In-window noise ({(~on_track).sum()})')

        for i, idx in enumerate(tracks_r):
            col = _track_color(pair_idx[i]) if i in pair_idx else '#555555'
            lbl = (f'Pair {pair_idx[i]+1} ({len(idx)} hits)'
                   if i in pair_idx else f'Unmatched ({len(idx)} hits)')
            ax.scatter(timef[idx], posf[idx],
                       s=_amp_to_size(ampf[idx]),
                       color=col, zorder=6, edgecolors='k', linewidths=0.4, label=lbl)
    elif n_all == 0:
        ax.text(0.5, 0.5, f'No FEU {feu} hits', transform=ax.transAxes,
                ha='center', va='center', color='grey')

    ax.set_title(f'FEU {feu} — {n_all} combined_hits (all, no zoom)', fontsize=9)
    ax.set_ylabel(pos_lbl)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2)
    if is_bottom:
        ax.set_xlabel('Drift time [μs]')


def draw_companion(fig, axes: dict, r: EventResult, df_evt: pd.DataFrame,
                   wf_data: Optional[dict] = None,
                   dead1: Optional[np.ndarray] = None,
                   dead2: Optional[np.ndarray] = None,
                   strip_pitch: float = 0.78):
    """Redraw companion: all-hits position plots + waveforms, shared x per column."""
    for ax in axes.values():
        ax.cla()

    d1 = dead1 if dead1 is not None else np.array([])
    d2 = dead2 if dead2 is not None else np.array([])
    wf = wf_data or {}

    _draw_companion_pos(axes['pos1'], df_evt, feu=1, r=r, dead=d1,
                        strip_pitch=strip_pitch, is_bottom=False)
    _draw_companion_pos(axes['pos2'], df_evt, feu=2, r=r, dead=d2,
                        strip_pitch=strip_pitch, is_bottom=True)
    _draw_waveform(axes['wf1'], wf, feu=1, r=r, cax=None)
    axes['wf1'].set_xlabel('')   # bottom panel carries the x label (shared axis)
    _draw_waveform(axes['wf2'], wf, feu=2, r=r, cax=None)

    fig.suptitle(
        f'Companion — Event {r.event_id}  (all combined_hits, no zoom)',
        fontsize=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive browser
# ─────────────────────────────────────────────────────────────────────────────

class EventBrowser:
    def __init__(self, candidate_ids: np.ndarray, results: dict,
                 dead1, dead2, strip_pitch: float,
                 wf_index: dict, df: pd.DataFrame,
                 v_drift: float = 0.0,
                 start_event_id: Optional[int] = None):
        self.all_ids     = candidate_ids
        self.two_pair    = np.array([eid for eid in candidate_ids
                                     if results[eid].n_pairs == 2])
        self.results     = results
        self.dead1       = dead1
        self.dead2       = dead2
        self.strip_pitch = strip_pitch
        self.wf_index    = wf_index
        self.df_grouped  = df.groupby('eventId')
        self.v_drift     = v_drift
        self._two_mode   = False
        self.active      = self.all_ids
        self.idx         = 0
        self._goto_buf   = None   # None = normal; str = accumulating digits for go-to

        # Jump to requested start event
        if start_event_id is not None:
            self._jump_to_event(start_event_id, silent=True)

        self.fig, self.axs = _make_figure()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.comp_fig, self.comp_axs = _make_companion_figure()
        self.comp_fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw()

    def _jump_to_event(self, event_id: int, silent: bool = False):
        """Move self.idx to the position of event_id in self.active, if present."""
        hits = np.where(self.active == event_id)[0]
        if len(hits):
            self.idx = int(hits[0])
            if not silent:
                print(f'  → event {event_id}  [{self.idx+1}/{len(self.active)}]')
            return True
        else:
            if not silent:
                print(f'  Event {event_id} not in current list '
                      f'({"all" if not self._two_mode else "2-pair"} mode)')
            return False

    def _on_key(self, ev):
        # Go-to mode: accumulate digit key presses, confirm with Enter
        if self._goto_buf is not None:
            if ev.key.isdigit():
                self._goto_buf += ev.key
                print(f'\r  Go to event: {self._goto_buf}_', end='', flush=True)
                return
            elif ev.key in ('enter', 'return'):
                buf = self._goto_buf
                self._goto_buf = None
                print()
                if buf:
                    if self._jump_to_event(int(buf)):
                        self._draw()
                return
            elif ev.key == 'escape':
                self._goto_buf = None
                print('\n  Go-to cancelled')
                return
            else:
                return  # ignore other keys while in go-to mode

        k = ev.key
        if k in ('n', 'right'):
            self.idx = (self.idx + 1) % len(self.active); self._draw()
        elif k in ('p', 'left'):
            self.idx = (self.idx - 1) % len(self.active); self._draw()
        elif k == 'g':
            self._goto_buf = ''
            print('  Go to event: _', end='', flush=True)
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
                   self.dead1, self.dead2, self.strip_pitch,
                   v_drift=self.v_drift)
        df_evt = (self.df_grouped.get_group(eid)
                  if eid in self.df_grouped.groups else pd.DataFrame())
        draw_companion(self.comp_fig, self.comp_axs, result, df_evt, wf_data,
                       self.dead1, self.dead2, self.strip_pitch)
        mode = ' [2-pair]' if self._two_mode else ''
        print(f'\n  [{self.idx+1}/{len(self.active)}] event {eid}{mode}'
              f'  — {result.n_pairs} pair(s)')
        if not df_evt.empty:
            cols = [c for c in ['feu', 'channel', 'amplitude', 'time', 'x_mm', 'y_mm']
                    if c in df_evt.columns]
            print(df_evt[cols].sort_values(['feu', 'time']).to_string(index=False))
        self.fig.canvas.draw_idle()
        self.comp_fig.canvas.draw_idle()

    def _save(self):
        eid  = self.active[self.idx]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f'event_{eid:06d}_pairs{self.results[eid].n_pairs}.png'
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        comp_path = OUT_DIR / f'event_{eid:06d}_pairs{self.results[eid].n_pairs}_companion.png'
        self.comp_fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {path}')
        print(f'  Saved {comp_path}')

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Debug helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_debug_event(event_id: int, df: pd.DataFrame,
                     dead1: np.ndarray, dead2: np.ndarray, pitch: float) -> None:
    """
    Process a single event with CROSS_DIM_DEBUG=True and print a full summary.
    No matplotlib figures are created.
    """
    global CROSS_DIM_DEBUG
    CROSS_DIM_DEBUG = True

    grouped = df.groupby('eventId')
    if event_id not in grouped.groups:
        print(f'Event {event_id} not found in loaded data.')
        return

    print(f'\n{"="*70}')
    print(f'DEBUG: event {event_id}')
    print('='*70)

    r = process_event(grouped.get_group(event_id), dead1, dead2, pitch)

    pos1f  = r.pos1[r.keep1];   time1f = r.time1[r.keep1]
    pos2f  = r.pos2[r.keep2];   time2f = r.time2[r.keep2]

    print(f'\nX tracks ({len(r.tracks1)}):')
    for i, t in enumerate(r.tracks1):
        tp, tt = pos1f[t], time1f[t]
        print(f'  [{i}] n={len(t):3d}  t=[{tt.min()*1000:.0f},{tt.max()*1000:.0f}]ns  '
              f'pos=[{tp.min():.1f},{tp.max():.1f}]mm  '
              f'rms={_track_rms(tp, tt):.3f}mm')

    print(f'\nY tracks ({len(r.tracks2)}):')
    for j, t in enumerate(r.tracks2):
        tp, tt = pos2f[t], time2f[t]
        print(f'  [{j}] n={len(t):3d}  t=[{tt.min()*1000:.0f},{tt.max()*1000:.0f}]ns  '
              f'pos=[{tp.min():.1f},{tp.max():.1f}]mm  '
              f'rms={_track_rms(tp, tt):.3f}mm')

    print(f'\nPairs ({len(r.pairs)}):')
    for i, j, iou in r.pairs:
        print(f'  X[{i}] ↔ Y[{j}]  IoU={iou:.3f}')

    CROSS_DIM_DEBUG = False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Track explorer for mx17_3')
    parser.add_argument('--event', type=int, default=None,
                        help='Jump browser to this event ID on startup')
    parser.add_argument('--debug-event', type=int, default=None, metavar='N',
                        help='Process event N with verbose debug output, then exit')
    args = parser.parse_args()

    det  = build_detector()
    df   = load_and_map(det)
    _, v_drift = compute_drift_velocity()
    dead1, dead2, pitch = compute_dead_strips(df)

    # Debug-only mode: process one event, print results, no GUI
    if args.debug_event is not None:
        _run_debug_event(args.debug_event, df, dead1, dead2, pitch)
        return

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

    start_eid = args.event if args.event is not None else START_EVENT_ID

    print('\nKey bindings:')
    print('  n/→  next      p/←  prev      2  toggle 2-pair mode')
    print('  g    go to event ID            s  save      q  quit\n')

    browser = EventBrowser(cids, results, dead1, dead2, pitch, wf_index, df,
                           v_drift=v_drift, start_event_id=start_eid)
    browser.show()


if __name__ == '__main__':
    main()
