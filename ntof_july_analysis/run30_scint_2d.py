#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_scint_2d.py

2-D (x,y) hit-map view of the run_30 scintillator blocks — hunting for events
that are something OTHER than the whole-plane gamma flash that fires with each
~1.2 s beam extraction.

Motivation: the scint trigger is PS(beam) OR Singles(scintillator).  The beam
line fires the gamma flash (whole plane lit at once); a scintillator single is a
real particle that should light only a few adjacent X strips and a few adjacent
Y strips at one location — a compact (x,y) CLUSTER.  This tool separates the two
by spatial compactness and renders the localized candidates.

Per detector (A/B/C; A is the clean plane) each event's hits are strip-mapped.
A fired X strip measures an X coordinate (drawn as a vertical line at that X); a
fired Y strip measures a Y coordinate (horizontal line).  A localized particle
shows up as a small box where a few X lines cross a few Y lines; a flash hatches
the whole plane.

Outputs -> July_HV_Scan/run30_scint_tracks/<block>_2d/:
  occupancy.png          all-events vs localized-events hit-position maps (per det)
  A_loc_evt_*.png        2-D hit maps of the most localized detector-A events

Run:  .venv/bin/python ntof_july_analysis/run30_scint_2d.py [block ...]
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
DEFAULT_BLOCKS = ['scintOff_A700_00', 'scintOn_A700_00']
LIVE = ['mx17_A', 'mx17_B', 'mx17_C']

AMP = 250.0             # amplitude floor for a hit
AMP_CLIP = (250, 3000)  # colour scale (ADC)

# "Localized cluster" (real-particle candidate) definition, per detector:
LOC_MAX_NX = 12         # at most this many X strips
LOC_MAX_NY = 12         # at most this many Y strips
LOC_SPREAD = 25.0       # mm: X and Y spread both below this
LOC_MIN_XY = 1          # need at least this many hits in EACH projection

N_RENDER = 24           # localized detector-A events rendered


def event_table(mapped):
    """eventId -> dict(nx,ny,xspread,yspread,xs,ys,ax,ay,gx,gy) for amp>=AMP."""
    sel = mapped[mapped['amplitude'] >= AMP]
    out = {}
    for ev, g in sel.groupby('eventId'):
        gx = g[g['x_mm'].notna()]
        gy = g[g['y_mm'].notna()]
        xs = gx['x_mm'].to_numpy(); ys = gy['y_mm'].to_numpy()
        out[ev] = dict(
            nx=len(xs), ny=len(ys),
            xspread=float(xs.max() - xs.min()) if len(xs) else 0.0,
            yspread=float(ys.max() - ys.min()) if len(ys) else 0.0,
            xs=xs, ys=ys,
            ax=gx['amplitude'].to_numpy(), ay=gy['amplitude'].to_numpy())
    return out


def is_localized(e):
    return (e['nx'] >= LOC_MIN_XY and e['ny'] >= LOC_MIN_XY and
            e['nx'] <= LOC_MAX_NX and e['ny'] <= LOC_MAX_NY and
            e['xspread'] < LOC_SPREAD and e['yspread'] < LOC_SPREAD)


def render_2d(ev, e, det, block, out_dir, extent):
    """2-D hit map: fired X strips = vertical lines, Y strips = horizontal."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    norm = plt.Normalize(*AMP_CLIP)
    cmap = plt.get_cmap('viridis')
    for x, a in zip(e['xs'], e['ax']):
        ax.axvline(x, color=cmap(norm(a)), lw=1.6, alpha=0.9)
    for y, a in zip(e['ys'], e['ay']):
        ax.axhline(y, color=cmap(norm(a)), lw=1.6, alpha=0.9)
    # mark the crossing region
    if e['nx'] and e['ny']:
        ax.plot(np.mean(e['xs']), np.mean(e['ys']), 'r+', ms=16, mew=2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label='amplitude [ADC]')
    ax.set_title(f'{RUN}/{block} — event {ev} — {det.split("_")[-1]}\n'
                 f'nx={e["nx"]} ny={e["ny"]}  spread {e["xspread"]:.0f}x{e["yspread"]:.0f} mm  '
                 f'maxA={max(e["ax"].max() if e["nx"] else 0, e["ay"].max() if e["ny"] else 0):.0f}',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, f'{det.split("_")[-1]}_loc_evt_{ev:06d}.png')


def occupancy(tables, dets_order, block, out_dir, extent):
    """All-events vs localized-events hit-position occupancy, per detector."""
    ncol = len(dets_order)
    fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 8.4), squeeze=False)
    bins = [np.linspace(extent[0], extent[1], 60), np.linspace(extent[2], extent[3], 60)]
    for ci, det in enumerate(dets_order):
        tbl = tables[det]
        for ri, (which, title) in enumerate([('all', 'all events'),
                                              ('loc', 'localized events')]):
            ax = axes[ri][ci]
            X, Y = [], []
            for ev, e in tbl.items():
                if which == 'loc' and not is_localized(e):
                    continue
                # pair every fired X with every fired Y (occupancy smear)
                if e['nx'] and e['ny']:
                    xx, yy = np.meshgrid(e['xs'], e['ys'])
                    X.append(xx.ravel()); Y.append(yy.ravel())
            if X:
                X = np.concatenate(X); Y = np.concatenate(Y)
                ax.hist2d(X, Y, bins=bins, cmap='inferno')
                ntag = f'{sum(1 for e in tbl.values() if (which=="all") or is_localized(e))} evt'
            else:
                ntag = '0 evt'
            ax.set_aspect('equal')
            if ri == 0:
                ax.set_title(f'{det.split("_")[-1]} — {title}\n({ntag})', fontsize=10)
            else:
                ax.set_title(f'{title} ({ntag})', fontsize=10)
            ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    fig.suptitle(f'{RUN}/{block} — hit-position occupancy (amp>={AMP:.0f})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir, 'occupancy.png')


def process(block, dets):
    allf = sorted({f for n in LIVE for f in dets[n].feu_map})
    df = load_hits(BASE_PATH, RUN, block, allf)
    if df is None or df.empty:
        print(f'  [skip] {block}'); return
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    # detector-plane extent from the strip map.  Axis-'x' strips carry the X
    # coordinate (x_position_mm), axis-'y' strips carry the Y coordinate.
    xs = [xp for (ax, _, _), (xp, yp) in dets['mx17_A'].strip_map.map.items() if ax == 'x']
    ys = [yp for (ax, _, _), (xp, yp) in dets['mx17_A'].strip_map.map.items() if ax == 'y']
    extent = (min(xs), max(xs), min(ys), max(ys))

    tables = {det: event_table(map_positions(df, dets[det])) for det in LIVE}
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_scint_tracks', block + '_2d')

    nloc = {det: sum(1 for e in tables[det].values() if is_localized(e)) for det in LIVE}
    print(f'  {block}: localized events per det -> '
          + '  '.join(f'{d.split("_")[-1]}={nloc[d]}' for d in LIVE))

    occupancy(tables, LIVE, block, out_dir, extent)

    # render the most localized detector-A events (fewest total hits, tie-broken
    # by highest amplitude — the cleanest, strongest single-particle candidates)
    A = tables['mx17_A']
    loc = [(ev, e) for ev, e in A.items() if is_localized(e)]
    loc.sort(key=lambda t: (t[1]['nx'] + t[1]['ny'],
                            -max(t[1]['ax'].max() if t[1]['nx'] else 0,
                                 t[1]['ay'].max() if t[1]['ny'] else 0)))
    for ev, e in loc[:N_RENDER]:
        render_2d(ev, e, 'mx17_A', block, out_dir, extent)
    print(f'    rendered {min(len(loc), N_RENDER)} localized A events -> {out_dir}')


if __name__ == '__main__':
    blocks = sys.argv[1:] or DEFAULT_BLOCKS
    cfg = load_config(BASE_PATH, RUN)
    sm = Mx17StripMap(MAP_CSV_PATH)
    dets = {d['name']: Detector(d['name'], d, sm)
            for d in cfg['detectors'] if d['name'].startswith('mx17')}
    for b in blocks:
        process(b, dets)
    print('done')
