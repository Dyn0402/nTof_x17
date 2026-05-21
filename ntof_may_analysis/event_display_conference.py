#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_display_conference.py

Conference-quality display for a single cherry-picked beam event.

Lab coordinate system (matching the experimental layout):
  Y — beam direction (vertical, from floor through capsule)
  Z — connects capsule to detector (perpendicular to beam)
  X — horizontal, perpendicular to Z

  Capsule at Z = 0.
  MM cathode (top of drift gap) at Z = -CAPSULE_TO_CATHODE_MM.
  Drift direction is Z (ionisation electrons drift toward mesh at more negative Z).

Produces three figures saved to output/:
  1.  event_<ID>_2d.png      — technical 2-D display (pos vs time, X-Y, waveforms)
  2.  event_<ID>_3d_mm.png   — 3-D MM hit display with fitted track
  3.  event_<ID>_3d_geo.png  — artistic full-geometry display

Edit the CONFIG section below to change the event, run, or display options.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
import trimesh
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

EVENT_ID = 100

run    = 'run_51'
subrun = 'hv_scan_drift_600_resist_530'

SUPPRESS_NOISE_HITS = True   # hide off-track hits in all figures
SUPPRESS_OFF_PAIR   = True   # show only the best-matched track pair

# ── Hit filter (keep in sync with track_explorer.py) ─────────────────────────
AMP_THRESHOLD        = 50
TIME_WIN_MIN         = 300.0   # ns
TIME_WIN_MAX         = 1100.0  # ns
GAMMA_FLASH_MAX_HITS = 200
GAMMA_FLASH_AMP      = 500
GAMMA_FLASH_TIME_NS  = 1000.0

# ── Tracking (keep in sync with track_explorer.py) ───────────────────────────
ISO_POS_MM         = 5.0
ISO_TIME_NS        = 200.0
ROAD_WIDTH_MM      = 3.0
ROAD_WIDTH_SEED_MM = 7.0
MAX_TIME_GAP_NS    = 250.0
MIN_TRACK_HITS     = 2
MAX_MISSED         = 3
MAX_STRIP_GAP      = 3
PAIR_MIN_IOU       = 0.20
NS_PER_SAMPLE      = 20.0

# ── Drift velocity ────────────────────────────────────────────────────────────
DRIFT_GAS = 'Ne/iC4H10 95/5'
DRIFT_HV  = 600.0   # V
DRIFT_TIME_LUT = {
    ('Ne/iC4H10 95/5', 37.5): 500.0,   # 600 V / 16 mm
}

# ── Geometry (all mm, capsule at Z=0) ─────────────────────────────────────────
CAPSULE_TO_CATHODE_MM = 200.0   # front of drift volume → capsule
DRIFT_GAP_MM          = 16.0
DET_HALF_SIZE_MM      = 200.0   # half of 400 mm active area, to centre at (0,0)
MM_PCB_THICK_MM       = 20.0     # estimate for back-plate after mesh
CAPSULE_RADIUS_MM     = 10.0
CAPSULE_HEIGHT_MM     = 50.0
CAPSULE_Y_CENTER_MM   = 30.0   # lab Y offset of capsule centr (positive = up / beam direction)
SCINT_WALL_SIZE_MM    = 500.0
SCINT_WALL_THICK_MM   = 10.0
SCINT_GAP_MM          = 20.0    # gap between back of MM and front of scint wall
SCINT_STRIP_WIDTH_MM  = 25.0    # 2.5 cm / strip
N_SCINT_STRIPS        = 20
SCINT_WALL_SINGLE     = True    # True = single rectangle; False = alternating bars
LARGE_SCINT_SIZE_MM   = 300.0
LARGE_SCINT_THICK_MM  = 10.0
LARGE_SCINT_GAP_MM    = 30.0    # gap between scint wall and large scint

# computed z positions of key planes
Z_CATHODE    = -CAPSULE_TO_CATHODE_MM
Z_MESH       = Z_CATHODE - DRIFT_GAP_MM
Z_MM_BACK    = Z_MESH - MM_PCB_THICK_MM
Z_SCINT_FRONT = Z_MM_BACK - SCINT_GAP_MM
Z_SCINT_BACK  = Z_SCINT_FRONT - SCINT_WALL_THICK_MM
Z_LARGE_FRONT = Z_SCINT_BACK - LARGE_SCINT_GAP_MM
Z_LARGE_BACK  = Z_LARGE_FRONT - LARGE_SCINT_THICK_MM

# ── Output ────────────────────────────────────────────────────────────────────
SUBRUN_DIR  = Path(f'/media/dylan/data/x17/may_beam/runs/{run}/{subrun}')
OUT_DIR     = Path(__file__).parent / 'output' / run / subrun / 'event_display'
MAP_CSV     = _ROOT / 'mx17_m1_map.csv'
DECODED_DIR = SUBRUN_DIR / 'decoded_root'


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f'=== Event display  —  event {EVENT_ID}  ===')

    det, v_drift = build_detector()
    df_evt       = load_event(det)
    _diagnose(df_evt)
    wf_data      = load_waveform(EVENT_ID)

    r    = run_tracking(df_evt, v_drift)
    best = best_pair_data(r)

    print('\nPlotting 2-D display …')
    plot_2d_display(r, best, wf_data)

    print('Plotting 3-D MM hit display …')
    plot_3d_mm_display(r, best, v_drift)

    print('Plotting 3-D geometry display …')
    plot_3d_geometry(r, best, v_drift)
    export_blender_data(r, best, v_drift)

    print('Plotting X-Y projection …')
    plot_xy_projection(r, best, v_drift)

    print('Saving rotating GIF …')
    plot_3d_geometry_gif(r, best, v_drift)

    print(f'\nDone. All figures saved to {OUT_DIR}')
    plt.show()

def _diagnose(df_evt: pd.DataFrame):
    """Print a step-by-step breakdown of how many hits survive each filter."""
    print('\n── Hit filter diagnostics ──────────────────────────────────────')
    for feu, pos_col in [(1, 'x_mm'), (2, 'y_mm')]:
        d_all  = df_evt[df_evt['feu'] == feu]
        d_amp  = d_all[d_all['amplitude'] > AMP_THRESHOLD]
        d_win  = d_amp[(d_amp['time'] >= TIME_WIN_MIN) & (d_amp['time'] <= TIME_WIN_MAX)]
        d_pos  = d_win[d_win[pos_col].notna()]
        pos_vals = d_pos[pos_col]
        print(f'  FEU {feu} ({pos_col}):  '
              f'total={len(d_all)}  '
              f'amp>{AMP_THRESHOLD}={len(d_amp)}  '
              f'in_win={len(d_win)}  '
              f'pos_notna={len(d_pos)}', end='')
        if len(d_all):
            print(f'\n    amp range:  [{d_all["amplitude"].min():.0f}, {d_all["amplitude"].max():.0f}]  '
                  f'time range: [{d_all["time"].min():.0f}, {d_all["time"].max():.0f}] ns  '
                  f'{pos_col} nan={d_all[pos_col].isna().sum()}/{len(d_all)}  '
                  f'range=[{d_all[pos_col].min():.2f}, {d_all[pos_col].max():.2f}]')
        else:
            print()
    print('────────────────────────────────────────────────────────────────\n')


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_detector() -> Detector:
    cfg     = json.loads((SUBRUN_DIR.parent / 'run_config.json').read_text())
    det_cfg = next(d for d in cfg['detectors'] if d['name'] == 'mx17_3')
    gap_str = det_cfg.get('drift_gap', f'{DRIFT_GAP_MM} mm')
    try:
        gap = float(gap_str.split()[0])
    except (ValueError, IndexError):
        gap = DRIFT_GAP_MM
    field = round(DRIFT_HV / gap * 2) / 2
    key   = (DRIFT_GAS, field)
    if key in DRIFT_TIME_LUT:
        v_drift = gap / DRIFT_TIME_LUT[key]   # mm/ns
    else:
        print(f'[warn] {key} not in DRIFT_TIME_LUT; using 0.032 mm/ns')
        v_drift = 0.032
    print(f'Drift: gap={gap:.1f} mm  HV={DRIFT_HV:.0f} V  '
          f'field={DRIFT_HV/gap:.2f} V/mm  v_drift={v_drift:.4f} mm/ns')
    return Detector('mx17_3', det_cfg, Mx17StripMap(str(MAP_CSV))), v_drift


def load_event(det: Detector) -> pd.DataFrame:
    files = sorted(
        f for f in (SUBRUN_DIR / 'combined_hits_root').iterdir()
        if f.suffix == '.root' and '_datrun_' in f.name and 'feu-combined' in f.name
    )
    print(f'Loading {len(files)} combined_hits file(s) …')
    df = uproot.concatenate([f'{f}:hits' for f in files], library='pd')
    df = df[df['feu'].isin([1, 2])].copy()

    xs, ys = [], []
    for feu, ch in zip(df['feu'].values, df['channel'].values):
        p = det.map_hit(int(feu), int(ch))
        xs.append(p[0] if p else np.nan)
        ys.append(p[1] if p else np.nan)
    df['x_mm'] = xs
    df['y_mm'] = ys

    ev = df[df['eventId'] == EVENT_ID].copy()
    if ev.empty:
        raise RuntimeError(f'Event {EVENT_ID} not found in combined_hits.')
    print(f'Event {EVENT_ID}: {len(ev)} hits  '
          f'(FEU 1: {(ev["feu"]==1).sum()}  FEU 2: {(ev["feu"]==2).sum()})')
    return ev


def load_waveform(event_id: int) -> dict:
    """Return {feu: (samples, channels, amplitudes)} for FEUs 1 and 2."""
    result = {}
    for feu in (1, 2):
        tag = f'_{feu:02d}.'
        feu_files = sorted(f for f in DECODED_DIR.iterdir()
                           if f.suffix == '.root' and tag in f.name)
        for fpath in feu_files:
            try:
                with uproot.open(fpath) as uf:
                    if 'nt' not in uf:
                        continue
                    eids = uf['nt']['eventId'].array(library='np')
                    rows = np.where(eids == event_id)[0]
                    if not len(rows):
                        continue
                    row_i = int(rows[0])
                    tree  = uf['nt']
                    s = np.asarray(tree['sample'].array(
                        library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                    c = np.asarray(tree['channel'].array(
                        library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                    a = np.asarray(tree['amplitude'].array(
                        library='np', entry_start=row_i, entry_stop=row_i + 1)[0])
                    result[feu] = (s, c, a.astype(float))
                    break
            except Exception as e:
                print(f'  [wf] {fpath.name}: {e}')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Track finding (mirrors track_explorer.py pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(df_evt: pd.DataFrame, v_drift: float) -> dict:
    """Run the full tracking pipeline; return a results dict."""
    iso_us = ISO_TIME_NS / 1000.0
    gap_us = MAX_TIME_GAP_NS / 1000.0

    def _hits(feu, pos_col):
        d = df_evt[
            (df_evt['feu'] == feu) &
            (df_evt['amplitude'] > AMP_THRESHOLD) &
            (df_evt['time'] >= TIME_WIN_MIN) &
            (df_evt['time'] <= TIME_WIN_MAX) &
            df_evt[pos_col].notna()
        ]
        return (d[pos_col].to_numpy(float),
                d['time'].to_numpy(float) / 1000.0,
                d['amplitude'].to_numpy(float))

    pos1, time1, amp1 = _hits(1, 'x_mm')   # FEU 1 x-strips measure X
    pos2, time2, amp2 = _hits(2, 'y_mm')   # FEU 2 y-strips measure Y

    def _iso(pos, time):
        if len(pos) < 2:
            return np.ones(len(pos), bool)
        return remove_isolated_hits(pos, time, ISO_POS_MM, iso_us)

    keep1 = _iso(pos1, time1)
    keep2 = _iso(pos2, time2)

    dead = np.array([])   # skip dead-strip computation for single event
    kw = dict(road_width_mm=ROAD_WIDTH_MM, road_width_seed_mm=ROAD_WIDTH_SEED_MM,
              max_time_gap_us=gap_us, min_hits=MIN_TRACK_HITS,
              max_missed=MAX_MISSED, max_strip_gap=MAX_STRIP_GAP, strip_pitch=0.78)

    def _track(pos, time):
        if len(pos) < MIN_TRACK_HITS:
            return []
        p1, _ = find_tracks_1d(pos, time, dead_strip_positions=dead, **kw)
        ext, new = find_tracks_1d_pass2(p1, pos, time, dead_strip_positions=dead, **kw)
        all_t = ext + new
        sets  = [set(t.tolist()) for t in all_t]
        return [all_t[i] for i, s in enumerate(sets)
                if not any(s < sets[j] for j in range(len(sets)) if j != i)]

    posf1, tf1 = pos1[keep1], time1[keep1]
    posf2, tf2 = pos2[keep2], time2[keep2]

    tracks1 = _track(posf1, tf1)
    tracks2 = _track(posf2, tf2)

    def _pair(tracks1, t1f, tracks2, t2f):
        cands = []
        for i, tr1 in enumerate(tracks1):
            lo1, hi1 = t1f[tr1].min() - gap_us, t1f[tr1].max() + gap_us
            for j, tr2 in enumerate(tracks2):
                lo2, hi2 = t2f[tr2].min() - gap_us, t2f[tr2].max() + gap_us
                ov = max(0.0, min(hi1, hi2) - max(lo1, lo2))
                if ov == 0:
                    continue
                iou = ov / max((hi1 - lo1) + (hi2 - lo2) - ov, 1e-9)
                if iou >= PAIR_MIN_IOU:
                    cands.append((iou, i, j))
        cands.sort(reverse=True)
        used1, used2, pairs = set(), set(), []
        for iou, i, j in cands:
            if i not in used1 and j not in used2:
                pairs.append((i, j, float(iou)))
                used1.add(i); used2.add(j)
        return pairs

    pairs = _pair(tracks1, tf1, tracks2, tf2)

    print(f'Tracking: FEU1={len(tracks1)} track(s), FEU2={len(tracks2)} track(s), '
          f'{len(pairs)} pair(s)')

    return dict(pos1=pos1, time1=time1, amp1=amp1, keep1=keep1,
                pos2=pos2, time2=time2, amp2=amp2, keep2=keep2,
                posf1=posf1, tf1=tf1, posf2=posf2, tf2=tf2,
                tracks1=tracks1, tracks2=tracks2, pairs=pairs,
                v_drift=v_drift)


def best_pair_data(r: dict) -> Optional[dict]:
    """Extract arrays for the best-matched pair (highest IoU)."""
    if not r['pairs']:
        print('[warn] No paired tracks found for this event.')
        return None

    i, j, iou = r['pairs'][0]
    t1_idx = r['tracks1'][i]
    t2_idx = r['tracks2'][j]

    posf1, tf1 = r['posf1'], r['tf1']
    posf2, tf2 = r['posf2'], r['tf2']

    t_tr1 = tf1[t1_idx]   # μs — FEU1 track times
    p_tr1 = posf1[t1_idx]  # x_mm  (FEU 1 x-strips measure X)
    t_tr2 = tf2[t2_idx]
    p_tr2 = posf2[t2_idx]  # y_mm  (FEU 2 y-strips measure Y)

    sl_x, ic_x = _fit_slope(t_tr1, p_tr1)   # mm/μs — x position fit
    sl_y, ic_y = _fit_slope(t_tr2, p_tr2)   # mm/μs — y position fit

    if sl_x is None or sl_y is None:
        print('[warn] Could not fit slope for this track.')
        return None

    return dict(i=i, j=j, iou=iou,
                t_tr1=t_tr1, p_tr1=p_tr1, sl_x=sl_x, ic_x=ic_x,
                t_tr2=t_tr2, p_tr2=p_tr2, sl_y=sl_y, ic_y=ic_y)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: conference 2-D display
# ─────────────────────────────────────────────────────────────────────────────

def _amp_to_size(amp, lo=AMP_THRESHOLD, scale=800.0, smin=12, smax=160):
    frac = np.clip((np.asarray(amp) - lo) / scale, 0, 1)
    return smin + frac * (smax - smin)


def _draw_proj_2d(ax, r: dict, feu: int, best: Optional[dict],
                  track_color: str = '#2980b9'):
    """Position vs drift time — conference version."""
    if feu == 1:
        keep, posf, tf = r['keep1'], r['posf1'], r['tf1']
        amp_filt = r['amp1'][r['keep1']]
        track_idx = r['tracks1'][best['i']] if best else None
        sl, ic    = (best['sl_x'], best['ic_x']) if best else (None, None)
        pos_lbl   = 'X Position [mm]'
        ax_title  = 'X Hit Positions vs. Drift Time'
    else:
        keep, posf, tf = r['keep2'], r['posf2'], r['tf2']
        amp_filt = r['amp2'][r['keep2']]
        track_idx = r['tracks2'][best['j']] if best else None
        sl, ic    = (best['sl_y'], best['ic_y']) if best else (None, None)
        pos_lbl   = 'Y Position [mm]'
        ax_title  = 'Y Hit Positions vs. Drift Time'

    if track_idx is not None and len(track_idx):
        t_tr = tf[track_idx]
        p_tr = posf[track_idx]
        ax.scatter(t_tr * 1000, p_tr,
                   s=_amp_to_size(amp_filt[track_idx]),
                   color=track_color, zorder=5, edgecolors='k', linewidths=0.5,
                   label=f'{len(track_idx)} track hits')
        if sl is not None and len(track_idx) >= 2:
            dt = (t_tr.max() - t_tr.min()) * 0.30
            tf_l = np.linspace(t_tr.min() - dt, t_tr.max() + dt, 300)
            ax.plot(tf_l * 1000, sl * tf_l + ic,
                    color=track_color, lw=1.8, ls='--', zorder=4, alpha=0.80)

    ax.set_xlabel('Drift Time [ns]', fontsize=11)
    ax.set_ylabel(pos_lbl, fontsize=11)
    ax.set_title(ax_title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=10)

    if track_idx is not None and len(track_idx):
        p_tr   = posf[track_idx]
        t_ns   = tf[track_idx] * 1000
        p_pad  = max(12.0, (p_tr.max() - p_tr.min()) * 0.35)
        t_pad  = max(60.0,  (t_ns.max()  - t_ns.min())  * 0.35)
        ax.set_ylim(max(0,   p_tr.min() - p_pad), min(400, p_tr.max() + p_pad))
        ax.set_xlim(max(0,   t_ns.min()  - t_pad), t_ns.max() + t_pad)


def _draw_xy_panel(ax, r: dict, best: Optional[dict]):
    """Reconstructed track position — conference version."""
    ax.set_xlabel('X Position [mm]', fontsize=11)
    ax.set_ylabel('Y Position [mm]', fontsize=11)
    ax.set_title('Reconstructed Track Position', fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=10)

    if not best:
        ax.text(0.5, 0.5, 'No track pairs found', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='grey')
        ax.set_xlim(0, 400); ax.set_ylim(0, 400)
        return

    posf1, tf1 = r['posf1'], r['tf1']
    posf2, tf2 = r['posf2'], r['tf2']
    t1 = r['tracks1'][best['i']]
    t2 = r['tracks2'][best['j']]
    col = '#2980b9'

    x_mesh = posf1[t1[np.argmin(tf1[t1])]]
    y_mesh = posf2[t2[np.argmin(tf2[t2])]]
    x_top  = posf1[t1[np.argmax(tf1[t1])]]
    y_top  = posf2[t2[np.argmax(tf2[t2])]]

    # col_mesh  = '#2980b9'   # blue  — mesh exit
    # col_entry = '#c0392b'   # red   — drift entry
    col_mesh = 'red'  # blue  — mesh exit
    col_entry = 'green'  # red   — drift entry

    # ── Set axis limits first so we can convert mm → pts ─────────────────────
    all_x = posf1[t1]; all_y = posf2[t2]
    x_pad = max(8.0, (all_x.max() - all_x.min()) * 0.5 + 10)
    y_pad = max(8.0, (all_y.max() - all_y.min()) * 0.5 + 10)
    xc   = 0.5 * (all_x.min() + all_x.max())
    yc   = 0.5 * (all_y.min() + all_y.max())
    half = max(x_pad, y_pad)
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)

    # Convert 0.78 mm strip pitch to linewidth in pts for the current zoom.
    # Approximate panel size: set_aspect('equal') makes it square at ~row_height.
    # row_height ≈ (top-bottom)*fig_h / (n_rows + hspace) ≈ 2.87 in.
    x_range = 2.0 * half
    approx_ax_in = 2.87
    pts_per_mm = approx_ax_in * 72.0 / max(x_range, 1.0)
    max_lw = float(np.clip(0.78 * pts_per_mm, 0.8, 8.0))

    # Amplitudes for track hits in each FEU
    amp1_track = r['amp1'][r['keep1']][t1]
    amp2_track = r['amp2'][r['keep2']][t2]
    amp_global_max = max(float(amp1_track.max()), float(amp2_track.max()), 1.0)

    # X-measuring strips: vertical blue lines, width ∝ amplitude
    for x_pos, amp_val in zip(posf1[t1], amp1_track):
        lw = max(0.5, float(amp_val) / amp_global_max * max_lw)
        ax.axvline(x_pos, color='#2980b9', lw=lw, alpha=0.3, zorder=2)

    # Y-measuring strips: horizontal orange lines, width ∝ amplitude
    for y_pos, amp_val in zip(posf2[t2], amp2_track):
        lw = max(0.5, float(amp_val) / amp_global_max * max_lw)
        ax.axhline(y_pos, color='#e67e22', lw=lw, alpha=0.3, zorder=2)

    # Both open circles
    ax.scatter([x_top], [y_top], s=120, facecolors='none', zorder=6,
               edgecolors=col_entry, linewidths=2.2, label='Drift entry')
    ax.scatter([x_mesh], [y_mesh], s=120, facecolors='none', zorder=6,
               edgecolors=col_mesh,  linewidths=2.2, label='Mesh exit')

    # Arrow center-to-center
    if (x_mesh - x_top) ** 2 + (y_mesh - y_top) ** 2 > 0.1:
        # '#2c3e50'
        ax.annotate('', xy=(x_mesh, y_mesh), xytext=(x_top, y_top),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8,
                                    mutation_scale=14, shrinkA=0, shrinkB=0),
                    zorder=7)

    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)


def _draw_waveform_2d(ax, wf_data: dict, feu: int,
                      color: str = '#2980b9', hide_ylabel: bool = False):
    """Raw waveforms — all traces one solid colour (X=blue, Y=orange)."""
    feu_name = 'X' if feu == 1 else 'Y'
    ax.set_title(f'{feu_name}-Strip Raw Waveforms', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time [ns]', fontsize=11)
    if not hide_ylabel:
        ax.set_ylabel('Amplitude [ADC]', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0, 1950)

    if feu not in wf_data or len(wf_data[feu][0]) == 0:
        ax.text(0.5, 0.5, 'No waveform data', transform=ax.transAxes,
                ha='center', va='center', color='grey', fontsize=10)
        return

    samples, channels, amps = wf_data[feu]
    t_ns = samples.astype(float) * NS_PER_SAMPLE
    uch  = np.unique(channels.astype(int))

    for ch in uch:
        m     = channels.astype(int) == ch
        t_ch  = t_ns[m]; a_ch = amps[m]
        order = np.argsort(t_ch)
        ax.plot(t_ch[order], a_ch[order], lw=0.8, color=color, alpha=0.85)


def plot_2d_display(r: dict, best: Optional[dict], wf_data: dict):
    """Conference-quality 2-D display. Saves PNG and PDF."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(12, 8))
    # 10-col main grid; waveforms share a sub-gridspec with wspace=0
    fig.subplots_adjust(left=0.055, right=0.995, top=0.91, bottom=0.06,
                        wspace=0.0, hspace=0.3)
    gs = gridspec.GridSpec(2, 100, figure=fig)

    ax_f1 = fig.add_subplot(gs[0, 0:46])
    ax_f2 = fig.add_subplot(gs[0, 54:100])
    ax_xy = fig.add_subplot(gs[1, 0:28])   # 2 cols → near-square for 0–400 mm

    # Zero horizontal space between the two waveform panels
    gs_wf = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 35:100], wspace=0.0)
    ax_w1 = fig.add_subplot(gs_wf[0])
    ax_w2 = fig.add_subplot(gs_wf[1], sharey=ax_w1)

    _draw_proj_2d(ax_f1, r, feu=1, best=best, track_color='#2980b9')
    _draw_proj_2d(ax_f2, r, feu=2, best=best, track_color='#e67e22')
    _draw_xy_panel(ax_xy, r, best)
    _draw_waveform_2d(ax_w1, wf_data, feu=1, color='#2980b9')
    _draw_waveform_2d(ax_w2, wf_data, feu=2, color='#e67e22', hide_ylabel=True)
    # Fully remove y-axis tick space from the shared-y right panel
    ax_w2.tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.suptitle(
        'Reconstructed Beam Event — Micromegas Micro-TPC Tracker',
        fontsize=14, fontweight='bold',
    )
    for ext in ('png', 'pdf'):
        _save(fig, f'event_{EVENT_ID}_2d.{ext}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 3-D MM hit display
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_3d_hits(r: dict, best: dict, v_drift: float) -> dict:
    """
    Build 3-D hit arrays for the best pair using the paired-fit prediction.

    FEU 1 hits (X strips → x measured):
        x = measured strip position (x_mm, centred)
        y = predicted from FEU 2 (y) fit at same drift time
        z = drift_time × v_drift (0 at mesh, positive toward cathode)

    FEU 2 hits (Y strips → y measured):
        y = measured strip position (y_mm, centred)
        x = predicted from FEU 1 (x) fit at same drift time
        z = drift_time × v_drift

    Returns dict with arrays for FEU1 points, FEU2 points, track line, and amplitudes.
    """
    v_mm_us = v_drift * 1000.0   # mm/μs

    posf1, tf1 = r['posf1'], r['tf1']
    posf2, tf2 = r['posf2'], r['tf2']
    t1_idx = r['tracks1'][best['i']]
    t2_idx = r['tracks2'][best['j']]
    amp1   = r['amp1'][r['keep1']]
    amp2   = r['amp2'][r['keep2']]

    t_tr1 = tf1[t1_idx]   # μs
    p_tr1 = posf1[t1_idx]   # x_mm  (FEU 1 x-strips, same coords as 2D plots)
    t_tr2 = tf2[t2_idx]
    p_tr2 = posf2[t2_idx]   # y_mm  (FEU 2 y-strips, same coords as 2D plots)

    t0_us = min(t_tr1.min(), t_tr2.min())

    z1 = (t_tr1 - t0_us) * v_mm_us   # mm from mesh
    z2 = (t_tr2 - t0_us) * v_mm_us

    sl_x, ic_x = best['sl_x'], best['ic_x']
    sl_y, ic_y = best['sl_y'], best['ic_y']

    y1_pred = sl_y * t_tr1 + ic_y   # y predicted for FEU1 hits (from FEU2 fit)
    x2_pred = sl_x * t_tr2 + ic_x   # x predicted for FEU2 hits (from FEU1 fit)

    # Track line from mesh to cathode
    z_track_local = np.linspace(0, DRIFT_GAP_MM, 200)
    t_track = t0_us + z_track_local / v_mm_us
    x_track = sl_x * t_track + ic_x
    y_track = sl_y * t_track + ic_y

    return dict(
        # FEU 1 (X strips): x measured, y predicted
        x1=p_tr1,   y1=y1_pred, z1=z1, a1=amp1[t1_idx],
        # FEU 2 (Y strips): y measured, x predicted
        x2=x2_pred, y2=p_tr2,   z2=z2, a2=amp2[t2_idx],
        # Track line
        x_track=x_track, y_track=y_track, z_track=z_track_local,
        t0_us=t0_us,
    )


def _draw_mm_frame(ax):
    """Wire-frame outline of the MM active volume (0→400 mm in X and Y)."""
    sz = DET_HALF_SIZE_MM * 2   # 400 mm
    dg = DRIFT_GAP_MM
    for x in (0, sz):
        for y in (0, sz):
            ax.plot([x, x], [y, y], [0, dg], color='#666666', lw=0.6, alpha=0.5)
    for z in (0, dg):
        xs = [0, sz, sz, 0, 0]
        ys = [0, 0, sz, sz, 0]
        ax.plot(xs, ys, [z] * 5, color='#666666', lw=0.6, alpha=0.5)


def plot_3d_mm_display(r: dict, best: Optional[dict], v_drift: float):
    if not best:
        print('[skip] 3D MM display — no paired track.')
        return

    pts = _reconstruct_3d_hits(r, best, v_drift)

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection='3d')

    s1 = _amp_to_size(pts['a1'])
    s2 = _amp_to_size(pts['a2'])
    ax.scatter(pts['x1'], pts['y1'], pts['z1'],
               s=s1, color='#e74c3c', alpha=0.9, zorder=4,
               label=f'FEU 1 — X strips ({len(pts["a1"])} hits)')
    ax.scatter(pts['x2'], pts['y2'], pts['z2'],
               s=s2, color='#2980b9', alpha=0.9, zorder=4,
               label=f'FEU 2 — Y strips ({len(pts["a2"])} hits)')

    ax.plot(pts['x_track'], pts['y_track'], pts['z_track'],
            color='limegreen', lw=2.5, zorder=5, label='Fitted track')

    _draw_mm_frame(ax)

    ax.set_xlabel('X [mm]', labelpad=8)
    ax.set_ylabel('Y [mm]', labelpad=8)
    ax.set_zlabel('Drift distance from mesh [mm]', labelpad=8)
    ax.set_title(
        f'{run} / {subrun}  —  Event {EVENT_ID}\n'
        f'v_drift = {v_drift:.4f} mm/ns  |  drift gap = {DRIFT_GAP_MM:.0f} mm',
        fontsize=10,
    )
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=20, azim=-50)

    fig.tight_layout()
    _save(fig, f'event_{EVENT_ID}_3d_mm.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 3-D geometry display
# ─────────────────────────────────────────────────────────────────────────────


def export_blender_data(r: dict, best: Optional[dict], v_drift: float):
    """
    Exports the exact detector geometry to an OBJ file and the track to a JSON file
    for cinematic rendering in Blender.
    Uses True Lab Coordinates (X=horizontal, Y=beam, Z=drift).
    """

    print('\nExporting 3-D geometry to Blender formats …')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hw = DET_HALF_SIZE_MM
    sw = SCINT_WALL_SIZE_MM / 2
    ls = LARGE_SCINT_SIZE_MM / 2

    # Helper to generate a 3D box using True Lab Coordinates (X, Y, Z)
    def make_box(x0, x1, y0, y1, z0, z1):
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        cx, cy, cz = (x1 + x0) / 2, (y1 + y0) / 2, (z1 + z0) / 2
        box = trimesh.creation.box(extents=(dx, dy, dz))
        box.apply_translation([cx, cy, cz])
        return box

    meshes = []

    # 1. Carbon capsule (axis along Lab Y)
    # trimesh defaults to Z-axis. Rotate 90 deg around X to align with Y.
    capsule = trimesh.creation.cylinder(radius=CAPSULE_RADIUS_MM, height=CAPSULE_HEIGHT_MM)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    capsule.apply_transform(rot)
    meshes.append(capsule)

    # 2. MM drift volume
    drift_gap = make_box(-hw, hw, -hw, hw, Z_MESH, Z_CATHODE)
    meshes.append(drift_gap)

    # 3. Scintillator wall
    for k in range(N_SCINT_STRIPS):
        x0 = -sw + k * SCINT_STRIP_WIDTH_MM
        x1 = x0 + SCINT_STRIP_WIDTH_MM
        strip = make_box(x0, x1, -sw, sw, Z_SCINT_BACK, Z_SCINT_FRONT)
        meshes.append(strip)

    # 4. Large scintillator
    ls_y_bottom = -sw
    large_scint = make_box(-ls, ls, ls_y_bottom, ls_y_bottom + LARGE_SCINT_SIZE_MM, Z_LARGE_BACK, Z_LARGE_FRONT)
    meshes.append(large_scint)

    # Combine all meshes into a single scene and export
    scene = trimesh.Scene(meshes)
    geo_path = OUT_DIR / f'detector_geo_{EVENT_ID}.glb'
    scene.export(str(geo_path))
    print(f'  Saved Geometry → {geo_path}')

    # 5. Export Track Data
    if best:
        lx, ly, lz = _extrapolate_track_geo(
            best, v_drift, z_lab_start=250.0, z_lab_end=Z_LARGE_BACK - 20
        )
        # Combine the separate coordinate arrays into an array of [x, y, z] points
        track_points = np.column_stack((lx, ly, lz)).tolist()
        track_dict = {f"track_{EVENT_ID}": track_points}

        track_path = OUT_DIR / f'track_{EVENT_ID}.json'
        with open(track_path, 'w') as f:
            json.dump(track_dict, f, indent=4)
        print(f'  Saved Track    → {track_path}')
    else:
        print('  [skip] No best track found to export.')


def _box_faces(x0, x1, y0, y1, z0, z1):
    """Return the 6 faces of a box as a list of (4,3) vertex arrays."""
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]
    return verts


def _add_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=0.18, edge_alpha=0.5):
    faces = _box_faces(x0, x1, y0, y1, z0, z1)
    poly  = Poly3DCollection(faces, alpha=alpha,
                             facecolor=color, edgecolor=color, linewidth=0.4)
    poly._edge_alpha = edge_alpha
    ax.add_collection3d(poly)


def _add_cylinder_z(ax, radius, z0, z1, x_center=0.0, y_center=0.0,
                    color='dimgray', alpha=0.35, n=60):
    """Cylinder with axis along matplotlib Z (= lab Y = beam direction)."""
    theta  = np.linspace(0, 2 * np.pi, n)
    z_vals = np.array([z0, z1])
    Theta, Z_c = np.meshgrid(theta, z_vals)
    X_c = x_center + radius * np.cos(Theta)
    Y_c = y_center + radius * np.sin(Theta)
    ax.plot_surface(X_c, Y_c, Z_c, color=color, alpha=alpha,
                    rstride=1, cstride=1, linewidth=0)
    for z_cap in (z0, z1):
        T, R = np.meshgrid(theta, np.linspace(0, radius, 10))
        ax.plot_surface(x_center + R*np.cos(T), y_center + R*np.sin(T),
                        np.full_like(R, z_cap), color=color, alpha=alpha, linewidth=0)


def _add_glow_spot(ax, face_axis: str, face_val: float,
                   cx: float, cy: float, color: str,
                   radius: float = 42.0, max_alpha: float = 0.72,
                   nr: int = 28, nth: int = 64):
    """
    Radial gradient disc on a detector face — colour at centre, fades to
    transparent at the edge. face_axis is 'x', 'y', or 'z' (mpl axis that
    is constant); cx/cy are the spot centre in the two free mpl axes.
    """
    from matplotlib.colors import to_rgba
    r  = np.linspace(0, radius, nr + 1)
    th = np.linspace(0, 2 * np.pi, nth + 1)
    R, T = np.meshgrid(r, th)
    A = cx + R * np.cos(T)
    B = cy + R * np.sin(T)
    r_mid   = (r[:-1] + r[1:]) / 2.0
    a_1d    = max_alpha * (1.0 - r_mid / radius) ** 2   # quadratic falloff
    fcolors = np.zeros((nth, nr, 4))
    fcolors[:, :, :3] = to_rgba(color)[:3]
    fcolors[:, :, 3]  = a_1d[np.newaxis, :]
    kw = dict(facecolors=fcolors, rstride=1, cstride=1,
              linewidth=0, antialiased=False, shade=False)
    if face_axis == 'y':
        ax.plot_surface(A, np.full_like(A, face_val), B, **kw)
    elif face_axis == 'x':
        ax.plot_surface(np.full_like(A, face_val), A, B, **kw)
    elif face_axis == 'z':
        ax.plot_surface(A, B, np.full_like(A, face_val), **kw)


def _extrapolate_track_geo(best: dict, v_drift: float,
                            z_lab_start: float, z_lab_end: float) -> tuple:
    """
    Extrapolate fitted track across a lab-Z range (mm, capsule=0).
    Returns (lab_x, lab_y, lab_z) centred so detector is at (0,0).
    """
    v_mm_us = v_drift * 1000.0
    t0_us   = (best['t_tr1'].min() + best['t_tr2'].min()) / 2.0
    sl_x, ic_x = best['sl_x'], best['ic_x']
    sl_y, ic_y = best['sl_y'], best['ic_y']

    z_local_start = z_lab_start - Z_MESH
    z_local_end   = z_lab_end   - Z_MESH
    z_local = np.linspace(z_local_start, z_local_end, 400)
    lab_z   = z_local + Z_MESH

    t_us   = t0_us + z_local / v_mm_us
    lab_x  = sl_x * t_us + ic_x - DET_HALF_SIZE_MM   # centred
    lab_y  = sl_y * t_us + ic_y - DET_HALF_SIZE_MM   # centred

    return lab_x, lab_y, lab_z


def plot_3d_geometry(r: dict, best: Optional[dict], v_drift: float,
                     return_fig: bool = False):
    """
    Artistic geometry figure.
    Matplotlib axis convention (so beam = vertical in plot):
      mpl X = lab X  (horizontal)
      mpl Y = lab Z  (detector direction, left/right)
      mpl Z = lab Y  (beam direction, vertical = up/down)
    All _add_box calls use (x0,x1, lab_z0,lab_z1, lab_y0,lab_y1).
    """
    hw  = DET_HALF_SIZE_MM   # 200 mm — half-side of MM active area
    sw  = SCINT_WALL_SIZE_MM / 2   # 250 mm
    ls  = LARGE_SCINT_SIZE_MM / 2  # 150 mm

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # ── Carbon capsule: axis along beam (lab Y = mpl Z), offset upward ───
    _add_cylinder_z(ax,
                    radius=CAPSULE_RADIUS_MM,
                    z0=CAPSULE_Y_CENTER_MM - CAPSULE_HEIGHT_MM / 2,
                    z1=CAPSULE_Y_CENTER_MM + CAPSULE_HEIGHT_MM / 2,
                    x_center=0.0, y_center=0.0,   # lab Z=0 → mpl Y=0
                    color='#606060', alpha=0.55)

    # ─────────────────────────────────────────────────────────────────────
    # Z-facing detector stack (current, at lab_Z = -200 mm)
    # ─────────────────────────────────────────────────────────────────────
    _add_box(ax, -hw, hw, Z_MESH, Z_CATHODE, -hw, hw,
             color='#4477cc', alpha=0.05)
    _add_box(ax, -hw, hw, Z_CATHODE - 2, Z_CATHODE, -hw, hw,
             color='#4477cc', alpha=0.1)
    _add_box(ax, -hw, hw, Z_MESH - 2, Z_MESH, -hw, hw,
             color='#2244aa', alpha=0.1)

    if SCINT_WALL_SINGLE:
        _add_box(ax, -sw, sw, Z_SCINT_BACK, Z_SCINT_FRONT, -sw, sw,
                 color='#cc8833', alpha=0.15)
    else:
        for k in range(N_SCINT_STRIPS):
            xs = -sw + k * SCINT_STRIP_WIDTH_MM
            _add_box(ax, xs, xs + SCINT_STRIP_WIDTH_MM,
                     Z_SCINT_BACK, Z_SCINT_FRONT, -sw, sw,
                     color='#cc8833' if k % 2 == 0 else '#ee9944', alpha=0.15)

    ls_y_bottom = -sw
    _add_box(ax, -ls, ls, Z_LARGE_BACK, Z_LARGE_FRONT,
             ls_y_bottom, ls_y_bottom + LARGE_SCINT_SIZE_MM,
             color='#cc4444', alpha=0.15)

    # ─────────────────────────────────────────────────────────────────────
    # X-facing detector stack (flipped to -X side, no scint wall)
    # Cathode at X=-200, stack extends toward more negative X.
    # Forms an L with the Z-facing stack; both 200 mm from the capsule.
    # ─────────────────────────────────────────────────────────────────────
    xc2n  = -CAPSULE_TO_CATHODE_MM                                          # -200  cathode
    xm2n  = xc2n - DRIFT_GAP_MM                                             # -216  mesh
    xb2n  = xm2n - MM_PCB_THICK_MM                                          # -236  MM back
    xlf2n = xb2n - SCINT_GAP_MM - SCINT_WALL_THICK_MM - LARGE_SCINT_GAP_MM # -296  large front
    xlb2n = xlf2n - LARGE_SCINT_THICK_MM                                    # -306  large back

    _add_box(ax, xm2n, xc2n,   -hw, hw, -hw, hw, color='#4477cc', alpha=0.02)
    _add_box(ax, xc2n-2, xc2n, -hw, hw, -hw, hw, color='#4477cc', alpha=0.1)
    _add_box(ax, xm2n, xm2n+2, -hw, hw, -hw, hw, color='#2244aa', alpha=0.10)

    _add_box(ax, xlb2n, xlf2n, -ls, ls,
             ls_y_bottom, ls_y_bottom + LARGE_SCINT_SIZE_MM,
             color='#cc4444', alpha=0.1)

    # ── Glow spots at track–detector intersections (Z-facing stack) ──────
    if best:
        for z_face, glow_color in [
            (Z_CATHODE,     '#3366dd'),   # MM cathode  (saturated blue)
            (Z_SCINT_FRONT, '#cc7722'),   # Scint wall  (saturated orange)
            (Z_LARGE_FRONT, '#cc2222'),   # Large scint (saturated red)
        ]:
            hx, hy = _track_pos_at_z(best, v_drift, z_face)
            # offset by +1 mm along mpl_Y so the disc sits just in front of the face
            _add_glow_spot(ax, face_axis='y', face_val=z_face + 1.0,
                           cx=hx, cy=hy, color=glow_color)

    # ── Track: slightly extended past scintillators, less above capsule ──
    if best:
        lx, ly, lz = _extrapolate_track_geo(
            best, v_drift, z_lab_start=120.0, z_lab_end=Z_LARGE_BACK - 50)
        # Plot in mpl coords: (lab_x, lab_z, lab_y)
        ax.plot(lx, lz, ly, color='green', lw=2.5, zorder=6,
                label='Reconstructed track')

    # ── Neutron beam: cylinder shaft + 3D cone arrowhead ─────────────────
    beam_col      = '#d8d8d8'
    cone_h        = 40      # mm — cone height
    cone_r        = 10      # mm — cone base radius
    shaft_r       = 6       # mm — shaft radius
    beam_z_floor  = -(sw + 20)  # shaft base in mpl_Z (= lab_Y)
    beam_z_tip    = CAPSULE_Y_CENTER_MM - CAPSULE_HEIGHT_MM / 2 - 20
    beam_z_cbase  = beam_z_tip - cone_h

    # Shaft (cylinder along mpl_Z at mpl_X=mpl_Y=0)
    th_s = np.linspace(0, 2 * np.pi, 24)
    Z_s, T_s = np.meshgrid([beam_z_floor, beam_z_cbase], th_s)
    ax.plot_surface(shaft_r * np.cos(T_s), shaft_r * np.sin(T_s), Z_s,
                    color=beam_col, alpha=0.90, linewidth=0)

    # Cone surface (narrows to tip at beam_z_tip)
    th_c = np.linspace(0, 2 * np.pi, 36)
    z_c  = np.linspace(beam_z_cbase, beam_z_tip, 20)
    T_c, Z_c = np.meshgrid(th_c, z_c)
    R_c = cone_r * (1.0 - (Z_c - beam_z_cbase) / cone_h)
    ax.plot_surface(R_c * np.cos(T_c), R_c * np.sin(T_c), Z_c,
                    color=beam_col, alpha=0.90, linewidth=0)
    # Cone base cap
    r_cap = np.linspace(0, cone_r, 8)
    T_cap, R_cap = np.meshgrid(th_c, r_cap)
    ax.plot_surface(R_cap * np.cos(T_cap), R_cap * np.sin(T_cap),
                    np.full_like(R_cap, beam_z_cbase),
                    color=beam_col, alpha=0.90, linewidth=0)

    # ── Limits — X extends to cover both sides of the L-shaped config ───
    pad = 30
    ax.set_xlim(xlb2n - pad, sw + pad)
    ax.set_ylim(Z_LARGE_BACK - pad, 270)
    ax.set_zlim(-(sw + pad), sw + pad)
    ax.set_box_aspect([1, 1, 1])

    # ── Suppress axes ─────────────────────────────────────────────────────
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.set_label_text('')
        axis.line.set_linewidth(0)
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # ── Legend: placed between MM and capsule, above capsule Y level ──────
    # In axes-fraction coords at azim=30/elev=15, this region projects to
    # roughly the centre-right of the plot area.
    legend_entries = [
        plt.Line2D([0], [0], color='limegreen', lw=2.5, label='Reconstructed track'),
        plt.Line2D([0,1], [0,0], color=beam_col, lw=3,
                   marker='>', markersize=9, markerfacecolor=beam_col,
                   markeredgecolor='#999999', label='Neutron beam'),
        # plt.Rectangle((0,0),1,1, fc='#4477cc', alpha=0.3, label=f'MM drift volume ({DRIFT_GAP_MM:.0f} mm gap)'),
        plt.Rectangle((0,0),1,1, fc='#4477cc', alpha=0.3, label=f'MM drift volume'),
        plt.Rectangle((0,0),1,1, fc='#cc8833', alpha=0.5, label='SiPM scint. wall (50×50 cm)'),
        plt.Rectangle((0,0),1,1, fc='#cc4444', alpha=0.5, label='Large scint. (30×30 cm)'),
        plt.Rectangle((0,0),1,1, fc='#606060', alpha=0.5, label='Carbon target'),
    ]
    # fig.legend() renders in figure space (always on top of 3D geometry)
    leg = fig.legend(handles=legend_entries, loc='upper left', fontsize=9,
                     frameon=True, fancybox=False,
                     facecolor='white', edgecolor='#aaaaaa',
                     bbox_to_anchor=(0.45, 0.76))
    leg.get_frame().set_alpha(1.0)
    fig.text(0.54, 0.78, 'May Test Beam Configuration\nEvent Display',
             fontsize=11, fontweight='bold',
             ha='center', va='bottom', color='#2c3e50')

    # Side view (slight angle off the detector-direction axis) so all three
    # detector planes are visibly separated and the track traverses all of them
    ax.view_init(elev=20, azim=15)
    fig.tight_layout()
    if return_fig:
        return fig, ax
    _save(fig, f'event_{EVENT_ID}_3d_geo.png')



def _track_pos_at_z(best: dict, v_drift: float, z_lab: float) -> tuple:
    """Return centred (lab_x, lab_y) of track at specific lab Z (mm)."""
    v_mm_us = v_drift * 1000.0
    t0 = (best['t_tr1'].min() + best['t_tr2'].min()) / 2.0
    z_local = z_lab - Z_MESH
    t = t0 + z_local / v_mm_us
    x = best['sl_x'] * t + best['ic_x'] - DET_HALF_SIZE_MM
    y = best['sl_y'] * t + best['ic_y'] - DET_HALF_SIZE_MM
    return float(x), float(y)


def plot_3d_geometry_gif(r: dict, best: Optional[dict], v_drift: float):
    """Save a 360° rotating GIF of the geometry display."""
    print('  Building geometry for GIF …')
    fig, ax = plot_3d_geometry(r, best, v_drift, return_fig=True)

    n_frames = 120   # 3° per frame, 20 fps → 6 s loop
    def _update(frame):
        ax.view_init(elev=15, azim=frame * 3)
        return []

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=100, blit=False)
    path = OUT_DIR / f'event_{EVENT_ID}_3d_geo_rotating.gif'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=20), dpi=250)
    plt.close(fig)
    print(f'  Saved → {path}')


def plot_xy_projection(r: dict, best: Optional[dict], v_drift: float):
    """
    X-Y projection: outlines of all detector planes + track intersections.
    Centred coordinates (detector at ±200 mm, capsule at 0,0).
    """
    hw  = DET_HALF_SIZE_MM          # 200 mm — MM half-side
    sw  = SCINT_WALL_SIZE_MM / 2    # 250 mm — scint wall half-side
    ls  = LARGE_SCINT_SIZE_MM / 2   # 150 mm — large scint half-side
    ls_y_bot = -sw                  # large scint bottom = scint wall bottom
    ls_y_top = ls_y_bot + LARGE_SCINT_SIZE_MM

    fig, ax = plt.subplots(figsize=(8, 9))

    # MM active area
    ax.add_patch(mpatches.Rectangle((-hw, -hw), 2*hw, 2*hw,
                 fc='#4477cc', alpha=0.10, ec='#4477cc', lw=2,
                 label=f'MM active area ({2*hw:.0f}×{2*hw:.0f} mm)'))

    # Scintillator wall + strip boundaries
    ax.add_patch(mpatches.Rectangle((-sw, -sw), 2*sw, 2*sw,
                 fc='#cc8833', alpha=0.08, ec='#cc8833', lw=2, ls='--',
                 label=f'Scint. wall ({2*sw:.0f}×{2*sw:.0f} mm, {N_SCINT_STRIPS} strips)'))
    for k in range(N_SCINT_STRIPS + 1):
        xk = -sw + k * SCINT_STRIP_WIDTH_MM
        ax.plot([xk, xk], [-sw, sw], color='#cc8833', lw=0.4, alpha=0.4)

    # Large scintillator
    ax.add_patch(mpatches.Rectangle((-ls, ls_y_bot), 2*ls, LARGE_SCINT_SIZE_MM,
                 fc='#cc4444', alpha=0.10, ec='#cc4444', lw=2, ls=':',
                 label=f'Large scint. ({2*ls:.0f}×{LARGE_SCINT_SIZE_MM:.0f} mm)'))

    # Capsule cross-section
    ax.add_patch(mpatches.Circle((0, 0), CAPSULE_RADIUS_MM,
                 fc='#606060', alpha=0.40, ec='#606060', lw=1.5,
                 label=f'Capsule (r={CAPSULE_RADIUS_MM:.0f} mm)'))

    if best:
        posf1 = r['posf1'];  tf1 = r['tf1']
        posf2 = r['posf2'];  tf2 = r['tf2']
        t1_idx = r['tracks1'][best['i']]
        t2_idx = r['tracks2'][best['j']]

        # MM strip hits: vertical lines for X-measuring strips, horizontal for Y
        for xpos in (posf1[t1_idx] - hw):
            ax.plot([xpos, xpos], [-hw, hw], color='#4477cc', lw=1.2, alpha=0.55)
        for ypos in (posf2[t2_idx] - hw):
            ax.plot([-hw, hw], [ypos, ypos], color='#2980b9', lw=1.2, alpha=0.55)

        # Track intersection on each detector surface
        detectors = [
            ('MM mesh crossing',  Z_MESH,        'o', '#4477cc', 180),
            ('Scint. wall front', Z_SCINT_FRONT,  's', '#cc8833', 180),
            ('Large scint. front', Z_LARGE_FRONT, '^', '#cc4444', 180),
        ]
        for lbl, z_lab, mrk, col, ms in detectors:
            xi, yi = _track_pos_at_z(best, v_drift, z_lab)
            ax.scatter([xi], [yi], s=ms, marker=mrk, color=col, zorder=6,
                       edgecolors='k', linewidths=0.8, label=f'Track @ {lbl}')

    pad = 30
    ax.set_xlim(-sw - pad, sw + pad)
    ax.set_ylim(ls_y_bot - pad, sw + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X [mm]  (horizontal)')
    ax.set_ylabel('Y [mm]  (vertical / beam direction)')
    ax.set_title(
        f'X-Y detector plane projection  —  Event {EVENT_ID}\n'
        'Centred coordinates  |  Blue lines = X-strip hits  |  Cyan lines = Y-strip hits',
        fontsize=10,
    )
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _save(fig, f'event_{EVENT_ID}_xy_projection.png')


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'  Saved → {path}')


if __name__ == '__main__':
    main()
