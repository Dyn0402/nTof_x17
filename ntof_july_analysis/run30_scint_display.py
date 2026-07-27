#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_scint_display.py

Per-detector micro-TPC event displays for the run_30 SCINTILLATOR-triggered
blocks (2026-07-11).  A scint-tagged charged particle crossing a Micromegas'
30 mm drift gap at an angle draws a diagonal line in the (strip-position vs
drift-time) plane — that's the micro-TPC "track" we're hunting for in each of
detectors A/B/C independently (D is out of service).

For a chosen block (e.g. scintOff_A700_00) this renders a handful of individual
events as a 3-row (det A/B/C) x 2-col (X projection, Y projection) grid of
position-vs-time scatter plots, hits coloured by amplitude.  A real track shows
up as a tilted band; flash/pile-up shows up as a vertical (single-time) smear
across all strips.

Event picking: for each event count amp>=SEL_AMP hits per detector, and keep
events whose busiest detector has SEL_MIN..SEL_MAX hits (moderate multiplicity —
avoids empty events and thousand-hit flash pile-up).  Rendered newest-first up
to N_EVENTS.

Output -> July_HV_Scan/run30_scint_tracks/<block>/evt_*.png  (flask Analysis tab).

Run:  .venv/bin/python ntof_july_analysis/run30_scint_display.py [block ...]
      default blocks: scintOff_A700_00 scintOn_A700_00
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from common.Mx17StripMap import Detector, Mx17StripMap  # noqa: E402
from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, MAP_CSV_PATH, MX17_DETECTORS,
    load_config, load_hits, _save,
)

RUN = 'run_30'
DEFAULT_BLOCKS = ['scintOff_A700_00', 'scintOn_A700_00']

# Hit selection / display.
DISP_AMP = 150.0        # amplitude floor just to render a hit (colour = amplitude)
SEL_AMP = 300.0         # amplitude used for event-selection multiplicity counting
SEL_MIN, SEL_MAX = 6, 120   # busiest-detector hit count that qualifies an event
N_EVENTS = 16           # events rendered per block
AMP_CLIP = (150, 3000)  # colour scale clip (ADC)
# All activity is prompt: ~99% of hits fall in [-0.3, 0.6] us of the trigger, so
# a micro-TPC track (if present) is a diagonal INSIDE this narrow window.  Zoom
# the display accordingly — a wide axis squishes real diagonals into a vertical
# stripe and makes everything look like a flash.
TIME_LO_US, TIME_HI_US = -0.2, 0.9   # drift-time display window (us)


def build_detectors(cfg):
    """{det_name: Detector} for the live MX17 detectors, in A/B/C/D order."""
    strip_map = Mx17StripMap(MAP_CSV_PATH)
    dets = {}
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        dets[name] = Detector(name=name, det_cfg=det_cfg, strip_map=strip_map)
    return {n: dets[n] for n in MX17_DETECTORS if n in dets}


def map_positions(df, det):
    """Add x_mm / y_mm columns for one detector's hits (NaN where not this det)."""
    feus = set(det.feu_map.keys())
    sub = df[df['feu'].isin(feus)]
    xs = np.full(len(sub), np.nan)
    ys = np.full(len(sub), np.nan)
    feu_arr = sub['feu'].to_numpy()
    ch_arr = sub['channel'].to_numpy()
    for i in range(len(sub)):
        pos = det.map_hit(int(feu_arr[i]), int(ch_arr[i]))
        if pos is None:
            continue
        x, y = pos
        if x is not None:
            xs[i] = x
        if y is not None:
            ys[i] = y
    out = sub.copy()
    out['x_mm'] = xs
    out['y_mm'] = ys
    out['time_us'] = out['time'].to_numpy() / 1000.0
    return out


# --- diagonal (micro-TPC track) scoring ------------------------------------
TRK_MIN_N = 5           # min hits in a projection to consider a diagonal
TRK_MAX_N = 60          # above this it's a flash/noise smear, not a clean track
TRK_MIN_TSPAN = 0.3     # us: min time spread (vertical flash has ~0)
TRK_MIN_PSPAN = 4.0     # mm: min position spread (stuck strip has ~0)
TRK_MIN_R2 = 0.55       # linear pos-vs-time fit quality
TRK_FLASH_FRAC = 0.6    # if this frac of hits fall in a 0.15 us window -> flashy
TRK_FLASH_WIN = 0.15    # us


def _diagonal_score(t, p):
    """Return (is_track, r2, tspan, pspan, slope) for one projection's hits."""
    n = len(t)
    if n < TRK_MIN_N:
        return False, 0.0, 0.0, 0.0, np.nan
    tspan = float(t.max() - t.min())
    pspan = float(p.max() - p.min())
    if tspan < TRK_MIN_TSPAN or pspan < TRK_MIN_PSPAN:
        return False, 0.0, tspan, pspan, np.nan
    # flash guard: most hits piled into one narrow time slice
    tc = np.median(t)
    if np.mean(np.abs(t - tc) < TRK_FLASH_WIN) >= TRK_FLASH_FRAC:
        return False, 0.0, tspan, pspan, np.nan
    slope, inter = np.polyfit(t, p, 1)
    pred = slope * t + inter
    ss_res = np.sum((p - pred) ** 2)
    ss_tot = np.sum((p - p.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (r2 >= TRK_MIN_R2), float(r2), tspan, pspan, float(slope)


def pick_events(mapped_by_det, det_names):
    """
    Rank events by micro-TPC diagonal-ness.  For each (event, detector) look at
    the X and Y projections separately; an event scores by how many detector
    projections look like a clean tilted band (good pos-vs-time line, real time
    AND position spread, not a single-time flash).  Events with a track-like
    band in BOTH projections of some detector rank highest.
    """
    per_ev = {}   # eventId -> dict(det -> {'x':(...), 'y':(...)})
    for det in det_names:
        m = mapped_by_det[det]
        sel = m[(m['amplitude'] >= SEL_AMP) &
                (m['time_us'] >= TIME_LO_US) & (m['time_us'] <= TIME_HI_US)]
        for ev, g in sel.groupby('eventId'):
            gx = g[g['x_mm'].notna()]
            gy = g[g['y_mm'].notna()]
            sx = _diagonal_score(gx['time_us'].to_numpy(), gx['x_mm'].to_numpy())
            sy = _diagonal_score(gy['time_us'].to_numpy(), gy['y_mm'].to_numpy())
            if sx[0] or sy[0]:
                per_ev.setdefault(ev, {})[det] = {'x': sx, 'y': sy}

    def ev_score(ev):
        best = 0.0
        for det, d in per_ev[ev].items():
            both = 10.0 if (d['x'][0] and d['y'][0]) else 0.0
            r2sum = (d['x'][1] if d['x'][0] else 0) + (d['y'][1] if d['y'][0] else 0)
            best = max(best, both + r2sum)
        return best

    ranked = sorted(per_ev.keys(), key=ev_score, reverse=True)
    # console summary of the top candidates
    for ev in ranked[:N_EVENTS]:
        parts = []
        for det, d in per_ev[ev].items():
            tag = ''.join(ax for ax in ('x', 'y') if d[ax][0])
            r2 = max(d['x'][1], d['y'][1])
            parts.append(f'{det.split("_")[-1]}[{tag} r2={r2:.2f}]')
        print(f'      evt {ev}: ' + ' '.join(parts))
    return ranked[:N_EVENTS], per_ev


def render_event(ev, mapped_by_det, det_names, block, out_dir):
    fig, axes = plt.subplots(len(det_names), 2, figsize=(11, 3.0 * len(det_names)),
                             squeeze=False, sharex=True)
    any_hit = False
    for ri, det in enumerate(det_names):
        m = mapped_by_det[det]
        me = m[(m['eventId'] == ev) & (m['amplitude'] >= DISP_AMP) &
               (m['time_us'] >= TIME_LO_US) & (m['time_us'] <= TIME_HI_US)]
        for ci, (coord, lab) in enumerate([('x_mm', 'X'), ('y_mm', 'Y')]):
            ax = axes[ri][ci]
            p = me[me[coord].notna()]
            if len(p):
                any_hit = True
                sc = ax.scatter(p['time_us'], p[coord], c=p['amplitude'],
                                cmap='viridis', vmin=AMP_CLIP[0], vmax=AMP_CLIP[1],
                                s=16, alpha=0.85)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(TIME_LO_US, TIME_HI_US)
            if ri == 0:
                ax.set_title(f'{lab} projection', fontsize=10)
            if ri == len(det_names) - 1:
                ax.set_xlabel('drift time [us]', fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'{det}\n{lab} pos [mm]', fontsize=9)
            else:
                ax.set_ylabel(f'{lab} pos [mm]', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.text(0.02, 0.97, f'n={len(p)}', transform=ax.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    if any_hit:
        fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                     label='amplitude [ADC]')
    fig.suptitle(f'{RUN}/{block} — event {ev}   (micro-TPC: pos vs drift time)',
                 fontsize=12)
    _save(fig, out_dir, f'evt_{ev:06d}.png')


def process_block(block, det_names_ref=None):
    cfg = load_config(BASE_PATH, RUN)
    dets = build_detectors(cfg)
    det_names = [n for n in dets if n != 'mx17_D']  # D out of service
    all_feus = sorted({f for d in dets.values() for f in d.feu_map})

    df = load_hits(BASE_PATH, RUN, block, all_feus)
    if df is None or df.empty:
        print(f'  [skip] {block}: no combined hits')
        return
    # dedupe rows that can repeat across ROOT cycles/parts
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    print(f'  {block}: {len(df)} hit rows, {df["eventId"].nunique()} events')

    mapped = {det: map_positions(df, dets[det]) for det in det_names}
    events, counts = pick_events(mapped, det_names)
    print(f'    selected {len(events)} events '
          f'(busiest-det amp>={SEL_AMP:.0f} in [{SEL_MIN},{SEL_MAX}] hits): '
          f'{events}')

    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_scint_tracks', block)
    for ev in events:
        render_event(ev, mapped, det_names, block, out_dir)
    print(f'    figures -> {out_dir}')


if __name__ == '__main__':
    blocks = sys.argv[1:] or DEFAULT_BLOCKS
    for b in blocks:
        process_block(b)
    print('done')
