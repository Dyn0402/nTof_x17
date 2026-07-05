#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_efficiency_maps.py

Reference-track based efficiency, the elegant way (per Dylan):
For every clean M3 single track, project to the aligned detector plane and record
  within_range : det1 has a reconstructed X+Y hit within R mm of the projection
  has_any      : det1 fired ANY strip in that event (detector responded at all)
A ray with NO DREAM event (DAQ saves only events with a valid hit) is a genuine
MISS (both False) and is kept in the denominator.

From the resulting ray list we make, for BOTH definitions:
  - color scatter (green hit / red miss, low alpha) at the projected positions
  - binned efficiency map (numerator / all rays per bin; self-normalises spatially)
  - integrated efficiency inside the aligned active area

Reco positions come from the no-veto per-event cache (sparks included — they
correctly fail within_range); alignment from the default veto50 run.

Usage: python 08_efficiency_maps.py ovn_det1 [--r=5]
"""
import os, sys, pickle
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

R = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--r=')), 5.0)
RECO_CACHE = 'event_results.pkl'              # no-veto: reco for ALL events incl sparks
ALIGN_DIR = 'alignment_tpc_veto50'            # default/best alignment
BINS = 40


def main():
    out_dir = CFG.out_dir('efficiency')
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, ALIGN_DIR, 'alignment.json'))

    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', RECO_CACHE), 'rb'))
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=5.0)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    # reco aligned position per event (from det1 micro-TPC fit)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # det1 "any hit" event set: >=1 hit on the detector FEUs (raw, sparks included)
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    det1_hit_events = set(int(e) for e in raw.loc[raw['feu'].isin(CFG.MX17_FEUS), 'eventId'].unique())

    # projection of every clean M3 ray at the aligned plane (ref frame, code sign convention)
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr)

    rows = []
    for e, x, y in zip((int(v) for v in evn), px, py):
        within = False
        if e in reco:
            within = (np.hypot(x - reco[e][0], y - reco[e][1]) <= R)
        rows.append((e, x, y, within, e in det1_hit_events))
    d = pd.DataFrame(rows, columns=['event_id', 'x', 'y', 'within', 'has_any'])
    d = d[np.isfinite(d['x']) & np.isfinite(d['y'])]
    print(f'clean M3 rays: {len(d)}  | within {R}mm: {d["within"].mean()*100:.1f}%  '
          f'has_any: {d["has_any"].mean()*100:.1f}%  (whole field of view)')

    # aligned active area: empirical detector footprint in the ref frame, from the
    # reconstructed (aligned) hit positions (get_active_det_bounds transform is flaky).
    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    inact = (d['x'] >= ax0) & (d['x'] <= ax1) & (d['y'] >= ay0) & (d['y'] <= ay1)
    da = d[inact]
    print(f'active area ref box: x[{ax0:.0f},{ax1:.0f}] y[{ay0:.0f},{ay1:.0f}]  rays inside: {len(da)}')
    print(f'INTEGRATED efficiency in active area:  within {R}mm = {da["within"].mean()*100:.1f}%   '
          f'has_any = {da["has_any"].mean()*100:.1f}%')

    box = dict(xy=(ax0, ay0), width=ax1 - ax0, height=ay1 - ay0)

    # ---- scatter plots (green hit / red miss) ----
    for col, name, ttl in [('within', f'within_{R:g}mm', f'hit within {R:g} mm'),
                           ('has_any', 'has_any', 'any hit on detector')]:
        fig, ax = plt.subplots(figsize=(7, 7))
        miss = d[~d[col]]; hit = d[d[col]]
        ax.scatter(miss['x'], miss['y'], s=6, c='red', alpha=0.12, linewidths=0, label=f'miss ({len(miss)})')
        ax.scatter(hit['x'], hit['y'], s=6, c='green', alpha=0.12, linewidths=0, label=f'hit ({len(hit)})')
        ax.add_patch(plt.Rectangle(**box, fill=False, ec='black', lw=1.5, label='active area'))
        ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]'); ax.set_aspect('equal')
        ax.set_title(f'{CFG.DET_NAME} efficiency scatter — {ttl}\n{CFG.RUN}/{CFG.SUB_RUN}')
        lg = ax.legend(loc='upper right', framealpha=0.9)
        for h in lg.legend_handles: h.set_alpha(1)
        fig.tight_layout(); fig.savefig(f'{out_dir}/scatter_{name}.png', dpi=150, bbox_inches='tight')

    # ---- binned efficiency maps ----
    rng = [[ax0 - 40, ax1 + 40], [ay0 - 40, ay1 + 40]]
    den, xe, ye = np.histogram2d(d['x'], d['y'], bins=BINS, range=rng)
    for col, name, ttl in [('within', f'within_{R:g}mm', f'hit within {R:g} mm'),
                           ('has_any', 'has_any', 'any hit')]:
        num, _, _ = np.histogram2d(d.loc[d[col], 'x'], d.loc[d[col], 'y'], bins=[xe, ye])
        with np.errstate(invalid='ignore', divide='ignore'):
            eff = np.where(den >= 5, num / den, np.nan)
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgrey')
        im = ax.imshow(eff.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       vmin=0, vmax=1, cmap=cmap, aspect='equal')
        plt.colorbar(im, ax=ax, label='efficiency')
        ax.add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.5))
        ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]')
        ax.set_title(f'{CFG.DET_NAME} efficiency map — {ttl}  (>=5 rays/bin)\n{CFG.RUN}/{CFG.SUB_RUN}')
        fig.tight_layout(); fig.savefig(f'{out_dir}/map_{name}.png', dpi=150, bbox_inches='tight')

    d.to_csv(f'{out_dir}/ray_hit_miss_list.csv', index=False)
    print(f'\nWritten to: {out_dir}')


if __name__ == '__main__':
    main()
