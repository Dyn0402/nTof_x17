#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_scint_neutron.py

Post-flash detector-A event displays for the run_30 scintillator blocks — the
thermal-neutron search.  Each ~1.2 s beam extraction gives a burst: a prompt
gamma-flash event (t=0) followed by ~30 ms of scintillator-triggered events.
Thermal neutrons thermalise and are captured over that 30 ms, so the GOOD events
are the delayed ones — hits arriving milliseconds after the flash.

For each event we compute Δt = (its trigger timestamp) − (its burst's flash-event
timestamp).  Bursts are found by clustering event timestamps with a 0.3 s gap;
the first event of a burst is the flash (t=0).

This renders detector-A 2-D (x,y) hit maps for post-flash events, each titled
with how long after the flash it fired (µs / ms), and a summary of Δt vs A hit
multiplicity.  Fired X strips are vertical lines, Y strips horizontal; a real
capture product is a compact cluster where a few X cross a few Y.

Outputs -> {ANALYSIS_DIR}July_HV_Scan/run30_neutron/<block>/:
  summary_dt.png             Δt distribution + A-hit multiplicity vs Δt
  A_dt_<ms>ms_evt_*.png       per-event 2-D hit maps, filename sorted by Δt

Run:  .venv/bin/python ntof_july_analysis/run30_scint_neutron.py [block ...]
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from common.Mx17StripMap import Detector, Mx17StripMap  # noqa: E402
from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, MAP_CSV_PATH, load_config, load_hits, _save,
)
from run30_scint_display import map_positions  # noqa: E402

RUN = 'run_30'
DET = 'mx17_A'
DEFAULT_BLOCKS = ['scintOff_A700_00', 'scintOn_A700_00']

AMP = 250.0             # amplitude floor for a hit
AMP_CLIP = (250, 3000)  # colour scale (ADC)
BURST_GAP_S = 0.3       # timestamp gap that starts a new beam-spill burst

# Which post-flash A events to render as displays.
DT_MIN_MS = 0.02        # skip the flash event itself (Δt=0); keep everything after
DT_MAX_MS = 32.0
MIN_A_HITS = 2          # need at least this many A hits (x+y) to be worth showing
MAX_A_HITS = 40         # above this it's not a single localized cluster
LOC_SPREAD_MM = 40.0    # X and Y spread must both be below this (compact cluster)
N_RENDER = 48           # cap on rendered event displays
# The (0,0) corner (connector-1 low channels) rings ~10 ms after every flash — a
# fixed-channel instrumental artifact, NOT a particle.  Reject events whose whole
# cluster sits in this corner box.
CORNER_MM = 18.0
# Δt bins to sample the gallery across the full post-flash window (ms).
DT_BINS_MS = [(0.02, 1), (1, 9), (9, 15), (15, 24), (24, 32)]
PER_BIN = 12            # up to this many rendered per Δt bin


def event_times(block):
    """eventId -> (flash_dt_s) using burst clustering; also returns burst id."""
    d = os.path.join(BASE_PATH, RUN, block, 'combined_hits_root')
    fs = [x for x in glob.glob(d + '/*.root')
          if 'feu-combined' in x and '_datrun_' in x and '_pedestals_' not in x]
    a = uproot.open(fs[0])['hits'].arrays(
        ['eventId', 'trigger_timestamp_ns'], library='np')
    eid, ts = a['eventId'], a['trigger_timestamp_ns'].astype(np.float64)
    ue, idx = np.unique(eid, return_index=True)
    t = ts[idx] / 1e9
    o = np.argsort(t); ue, t = ue[o], t[o]
    starts = np.concatenate([[0], np.where(np.diff(t) > BURST_GAP_S)[0] + 1])
    burst_of = np.searchsorted(starts, np.arange(len(t)), side='right') - 1
    flash_t = t[starts][burst_of]
    dt_s = t - flash_t
    return {int(e): (float(dt), int(b)) for e, dt, b in zip(ue, dt_s, burst_of)}


def fmt_dt(dt_s):
    return f'{dt_s*1e6:.0f} us' if dt_s < 1e-3 else f'{dt_s*1e3:.2f} ms'


def render(ev, gx, gy, dt_s, burst, block, out_dir, extent):
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    norm = plt.Normalize(*AMP_CLIP); cmap = plt.get_cmap('viridis')
    xs, ax_ = gx['x_mm'].to_numpy(), gx['amplitude'].to_numpy()
    ys, ay_ = gy['y_mm'].to_numpy(), gy['amplitude'].to_numpy()
    for x, a in zip(xs, ax_):
        ax.axvline(x, color=cmap(norm(a)), lw=1.6, alpha=0.9)
    for y, a in zip(ys, ay_):
        ax.axhline(y, color=cmap(norm(a)), lw=1.6, alpha=0.9)
    cx, cy = np.mean(xs), np.mean(ys)
    if len(xs) and len(ys):
        ax.plot(cx, cy, 'r+', ms=16, mew=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal'); ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label='amplitude [ADC]')

    # zoomed inset (±25 mm) around the cluster centroid so its shape is visible
    zw = 25.0
    axin = ax.inset_axes([0.60, 0.60, 0.38, 0.38])
    for x, a in zip(xs, ax_):
        axin.axvline(x, color=cmap(norm(a)), lw=2.0)
    for y, a in zip(ys, ay_):
        axin.axhline(y, color=cmap(norm(a)), lw=2.0)
    axin.set_xlim(cx - zw, cx + zw); axin.set_ylim(cy - zw, cy + zw)
    axin.set_aspect('equal')
    axin.tick_params(labelsize=6)
    axin.set_title('zoom ±25 mm', fontsize=7)
    for s in axin.spines.values():
        s.set_edgecolor('red')
    maxa = max(ax_.max() if len(xs) else 0, ay_.max() if len(ys) else 0)
    ax.set_title(f'{RUN}/{block} — detector A — event {ev}\n'
                 f'Δt = {fmt_dt(dt_s)} after flash   (burst {burst})\n'
                 f'nx={len(xs)} ny={len(ys)}  maxA={maxa:.0f} ADC', fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, f'A_dt_{dt_s*1e3:07.2f}ms_evt_{ev:06d}.png')


def summary(recs, block, out_dir):
    """recs: list of (ev, dt_s, n_a). Δt histogram + multiplicity vs Δt."""
    dt_ms = np.array([r[1] * 1e3 for r in recs])
    na = np.array([r[2] for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(dt_ms, bins=np.linspace(0, 32, 65), color='steelblue')
    axes[0].set_xlabel('Δt after flash [ms]'); axes[0].set_ylabel('detector-A events (>=1 hit)')
    axes[0].set_title('When do post-flash A hits occur?')
    axes[0].grid(alpha=0.3)
    sc = axes[1].scatter(dt_ms, na, s=14, c=na, cmap='viridis',
                         norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=max(na.max(), 2)))
    axes[1].set_xlabel('Δt after flash [ms]'); axes[1].set_ylabel('A hits (x+y, amp>=250)')
    axes[1].set_yscale('log'); axes[1].set_title('A hit multiplicity vs Δt')
    axes[1].grid(alpha=0.3, which='both')
    fig.colorbar(sc, ax=axes[1], label='A hits')
    fig.suptitle(f'{RUN}/{block} — detector A post-flash timing (amp>={AMP:.0f})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, 'summary_dt.png')


def process(block, dets):
    allf = sorted(dets[DET].feu_map.keys())
    df = load_hits(BASE_PATH, RUN, block, allf)
    if df is None or df.empty:
        print(f'  [skip] {block}'); return
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    times = event_times(block)
    m = map_positions(df, dets[DET])
    sel = m[m['amplitude'] >= AMP]
    xs = [xp for (ax, _, _), (xp, yp) in dets[DET].strip_map.map.items() if ax == 'x']
    ys = [yp for (ax, _, _), (xp, yp) in dets[DET].strip_map.map.items() if ax == 'y']
    extent = (min(xs), max(xs), min(ys), max(ys))
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_neutron', block)

    # per-event A hit multiplicity + Δt
    recs = []           # all post-flash A-hit events (for summary)
    candidates = []     # distributed compact events, gallery pool
    n_corner = 0
    for ev, g in sel.groupby('eventId'):
        if ev not in times:
            continue
        dt_s, burst = times[ev]
        gx = g[g['x_mm'].notna()]; gy = g[g['y_mm'].notna()]
        na = len(gx) + len(gy)
        if dt_s > 0:
            recs.append((ev, dt_s, na))
        if not (DT_MIN_MS <= dt_s * 1e3 <= DT_MAX_MS and MIN_A_HITS <= na <= MAX_A_HITS
                and len(gx) and len(gy)):
            continue
        xspread = float(gx['x_mm'].max() - gx['x_mm'].min())
        yspread = float(gy['y_mm'].max() - gy['y_mm'].min())
        if xspread > LOC_SPREAD_MM or yspread > LOC_SPREAD_MM:
            continue                # whole-plane / partial-flash, not a cluster
        mx, my = float(gx['x_mm'].mean()), float(gy['y_mm'].mean())
        if mx < CORNER_MM and my < CORNER_MM:
            n_corner += 1           # fixed-channel post-flash corner artifact
            continue
        candidates.append((ev, dt_s, burst, gx, gy, na))

    # stratified sample across Δt bins so the gallery spans the 30 ms window
    render_list = []
    for lo, hi in DT_BINS_MS:
        inb = sorted((c for c in candidates if lo <= c[1] * 1e3 < hi), key=lambda c: c[1])
        # even spread within the bin
        if len(inb) > PER_BIN:
            idx = np.linspace(0, len(inb) - 1, PER_BIN).round().astype(int)
            inb = [inb[i] for i in idx]
        render_list.extend(inb)
    render_list = render_list[:N_RENDER]

    print(f'  {block}: {len(recs)} post-flash A-hit events; '
          f'{len(candidates)} distributed compact candidates '
          f'(+{n_corner} corner-artifact rejected); rendering {len(render_list)} '
          f'across {[b[0] for b in DT_BINS_MS]}..{DT_BINS_MS[-1][1]}ms')
    if recs:
        summary(recs, block, out_dir)
    for ev, dt_s, burst, gx, gy, na in render_list:
        render(ev, gx, gy, dt_s, burst, block, out_dir, extent)
    print(f'    -> {out_dir}')


if __name__ == '__main__':
    blocks = sys.argv[1:] or DEFAULT_BLOCKS
    cfg = load_config(BASE_PATH, RUN)
    sm = Mx17StripMap(MAP_CSV_PATH)
    dets = {d['name']: Detector(d['name'], d, sm)
            for d in cfg['detectors'] if d['name'].startswith('mx17')}
    for b in blocks:
        process(b, dets)
    print('done')
