#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_scint_det_a.py

Detector-A-focused micro-TPC track hunt for the run_30 scintillator blocks.
A (mx17_A) is the clean "good M1" plane — no whole-plane common-mode bands that
swamp B/C — so it is the only detector where a genuine micro-TPC diagonal
(strip-position correlating with drift-time) can be trusted.

For a block it scans every event's A-only hits (FEU 3=x, 4=y), scores the X and
Y projections for a clean tilted band, and renders the best candidates as a
large 2-panel (X, Y) position-vs-drift-time display with the fit line drawn.

A projection qualifies as track-like when it has TRK_MIN_N..TRK_MAX_N hits, a
real time spread (not a single-instant vertical flash), a real position spread
(not a stuck strip), a physically-sane slope, a good linear fit, and is not
flash-piled into one narrow time slice.  Events track-like in BOTH X and Y rank
highest (a real 3-D segment fires both coordinates).

Run:  .venv/bin/python ntof_july_analysis/run30_scint_det_a.py [block ...]
      default: scintOff_A700_00 scintOn_A700_00
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
    BASE_PATH, ANALYSIS_DIR, MAP_CSV_PATH, load_config, load_hits, _save,
)
from run30_scint_display import map_positions  # noqa: E402

RUN = 'run_30'
DET = 'mx17_A'
DEFAULT_BLOCKS = ['scintOff_A700_00', 'scintOn_A700_00']

DISP_AMP = 150.0        # amplitude floor to render a hit
SEL_AMP = 250.0         # amplitude for track scoring
N_RENDER = 20           # best candidates rendered per block
AMP_CLIP = (150, 3000)
TIME_LO_US, TIME_HI_US = -0.2, 0.9

# Track-like projection criteria (compressed ~0.6 us drift window).
TRK_MIN_N, TRK_MAX_N = 6, 40
TRK_MIN_TSPAN = 0.20    # us: real depth spread (kills vertical flash band)
TRK_MIN_PSPAN = 6.0     # mm: real lateral spread (kills stuck strip)
TRK_MAX_PSPAN = 250.0   # mm: reject whole-plane smears
TRK_MIN_R2 = 0.85       # clean straight line
TRK_SLOPE_MIN = 10.0    # mm/us: |dp/dt|; below -> horizontal streak
TRK_SLOPE_MAX = 400.0   # mm/us: above -> effectively-vertical band
TRK_FLASH_WIN = 0.08    # us
TRK_FLASH_FRAC = 0.6    # >=this frac of hits within TRK_FLASH_WIN of median t -> flashy


def score_proj(t, p):
    """(is_track, r2, tspan, pspan, slope, n) for one projection's hits."""
    n = len(t)
    if n < TRK_MIN_N or n > TRK_MAX_N:
        return False, 0.0, 0.0, 0.0, np.nan, n
    tspan = float(t.max() - t.min())
    pspan = float(p.max() - p.min())
    if not (tspan >= TRK_MIN_TSPAN and TRK_MIN_PSPAN <= pspan <= TRK_MAX_PSPAN):
        return False, 0.0, tspan, pspan, np.nan, n
    if np.mean(np.abs(t - np.median(t)) < TRK_FLASH_WIN) >= TRK_FLASH_FRAC:
        return False, 0.0, tspan, pspan, np.nan, n   # flash-piled
    slope, inter = np.polyfit(t, p, 1)
    pred = slope * t + inter
    ss_tot = np.sum((p - p.mean()) ** 2)
    r2 = 1.0 - np.sum((p - pred) ** 2) / ss_tot if ss_tot > 0 else 0.0
    ok = (r2 >= TRK_MIN_R2 and TRK_SLOPE_MIN <= abs(slope) <= TRK_SLOPE_MAX)
    return bool(ok), float(r2), tspan, pspan, float(slope), n


def scan(mapped):
    """Return list of (event, dict) A-track candidates, best first."""
    sel = mapped[(mapped['amplitude'] >= SEL_AMP) &
                 (mapped['time_us'] >= TIME_LO_US) & (mapped['time_us'] <= TIME_HI_US)]
    cands = []
    for ev, g in sel.groupby('eventId'):
        gx = g[g['x_mm'].notna()]
        gy = g[g['y_mm'].notna()]
        sx = score_proj(gx['time_us'].to_numpy(), gx['x_mm'].to_numpy())
        sy = score_proj(gy['time_us'].to_numpy(), gy['y_mm'].to_numpy())
        if sx[0] or sy[0]:
            both = sx[0] and sy[0]
            r2 = (sx[1] if sx[0] else 0) + (sy[1] if sy[0] else 0)
            cands.append((ev, dict(x=sx, y=sy, both=both, score=(10 if both else 0) + r2)))
    cands.sort(key=lambda c: c[1]['score'], reverse=True)
    return cands


def render(ev, info, gx, gy, block, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, g, coord, lab, s in [
        (axes[0], gx, 'x_mm', 'X', info['x']),
        (axes[1], gy, 'y_mm', 'Y', info['y']),
    ]:
        d = g[(g['amplitude'] >= DISP_AMP) &
              (g['time_us'] >= TIME_LO_US) & (g['time_us'] <= TIME_HI_US)]
        if len(d):
            sc = ax.scatter(d['time_us'], d[coord], c=d['amplitude'], cmap='viridis',
                            vmin=AMP_CLIP[0], vmax=AMP_CLIP[1], s=26, alpha=0.85)
            fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label='amp [ADC]')
        ok, r2, tspan, pspan, slope, n = s
        # draw the fit line over the track hits (amp>=SEL_AMP)
        dfit = g[g['amplitude'] >= SEL_AMP]
        if ok and len(dfit) >= 2:
            m, b = np.polyfit(dfit['time_us'].to_numpy(), dfit[coord].to_numpy(), 1)
            tl = np.array([dfit['time_us'].min(), dfit['time_us'].max()])
            ax.plot(tl, m * tl + b, 'r--', lw=1.4)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(TIME_LO_US, TIME_HI_US)
        ax.set_xlabel('drift time [us]')
        ax.set_ylabel(f'{lab} position [mm]')
        flag = 'TRACK' if ok else '-'
        ax.set_title(f'{lab}: n={n}  {flag}\n'
                     f'R2={r2:.2f}  slope={slope:.0f} mm/us  '
                     f'dt={tspan:.2f}us  dp={pspan:.0f}mm', fontsize=9)
    tag = 'XY' if info['both'] else ('X' if info['x'][0] else 'Y')
    fig.suptitle(f'{RUN}/{block} — event {ev} — detector A  [{tag} track]',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, f'A_evt_{ev:06d}.png')


def process(block, dets, allf):
    df = load_hits(BASE_PATH, RUN, block, allf)
    if df is None or df.empty:
        print(f'  [skip] {block}: no hits')
        return
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    mapped = map_positions(df, dets[DET])
    cands = scan(mapped)
    n_both = sum(1 for _, i in cands if i['both'])
    print(f'  {block}: {mapped["eventId"].nunique()} events -> '
          f'{len(cands)} A-track candidates ({n_both} with X+Y)')
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_scint_tracks',
                           block + '_detA')
    for ev, info in cands[:N_RENDER]:
        g = mapped[mapped['eventId'] == ev]
        gx = g[g['x_mm'].notna()]
        gy = g[g['y_mm'].notna()]
        tag = 'XY' if info['both'] else ('X' if info['x'][0] else 'Y')
        r2 = max(info['x'][1], info['y'][1])
        print(f'      evt {ev}: {tag}  R2={r2:.2f}  score={info["score"]:.2f}')
        render(ev, info, gx, gy, block, out_dir)
    print(f'    figures -> {out_dir}')


if __name__ == '__main__':
    blocks = sys.argv[1:] or DEFAULT_BLOCKS
    cfg = load_config(BASE_PATH, RUN)
    sm = Mx17StripMap(MAP_CSV_PATH)
    dets = {d['name']: Detector(d['name'], d, sm)
            for d in cfg['detectors'] if d['name'].startswith('mx17')}
    allf = sorted(dets[DET].feu_map.keys())
    for b in blocks:
        process(b, dets, allf)
    print('done')
