#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe.py — load ONE event, run the Det-A double-track finder, print the result
and render a per-plane micro-TPC display with the extracted lines overlaid.

Usage:
    .venv/bin/python ntof_july_analysis/detA_doubletrack/probe.py \
        run_58 sngPS_dr300_r555_041 1459
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dtrack_lib as D  # noqa: E402
from ntof_tracking.reco import io, noise, geometry as geo  # noqa: E402

OUT = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'detA_doubletrack', 'probe')
LINE_COL = ['crimson', 'royalblue', 'seagreen', 'darkorange', 'purple']


def draw(g_ev, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    for ax, plane, lines in zip(axes, ('x', 'y'),
                                (res['xlines'], res['ylines'])):
        gp = g_ev[(g_ev['det'] == D.DET_A) & (g_ev['plane'] == plane)]
        cl = gp[gp['clean']]
        nb = gp[~gp['clean']]
        ax.scatter(nb['time'], nb['pos_mm'], s=8, c='0.8', marker='.',
                   label='flagged noise', zorder=1)
        ax.scatter(cl['time'], cl['pos_mm'], s=18, c='0.35', marker='o',
                   label='clean (unassigned)', zorder=2)
        cl_idx = cl.index.to_numpy()
        pos = cl['pos_mm'].to_numpy()
        tim = cl['time'].to_numpy()
        # lines[i]['idx'] are POSITIONS into the plane clean array -> rebuild
        clp = g_ev[(g_ev['det'] == D.DET_A) & (g_ev['plane'] == plane)
                   & g_ev['clean']]
        cpos = clp['pos_mm'].to_numpy()
        ctim = clp['time'].to_numpy()
        for i, ln in enumerate(lines):
            c = LINE_COL[i % len(LINE_COL)]
            ii = ln['idx']
            ax.scatter(ctim[ii], cpos[ii], s=34, facecolors='none',
                       edgecolors=c, linewidths=1.6, zorder=4,
                       label=f"line {i}: n={ln['n_hits']} r2={ln['r2']:.2f} "
                             f"q={ln['q_sum']:.0f}")
            tt = np.array([ln['t0_ns'], ln['t1_ns']])
            ax.plot(tt, ln['slope_mm_ns'] * tt + ln['intercept_mm'],
                    c=c, lw=2, zorder=3)
        ax.set_title(f'Det A  plane {plane}   ({len(lines)} lines)')
        ax.set_xlabel('drift time [ns]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    axes[0].set_ylabel('strip position [mm, centred]')
    topo = res.get('topo', {})
    fig.suptitle(f"event {res['eventId']}  |  n_xline={res['n_xline']} "
                 f"n_yline={res['n_yline']} n_pair={res['n_pair']}  |  "
                 f"double={res['is_double']}  topo={topo.get('tag','-')}",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    run, subrun, evid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cfg = io.load_run_config(run)
    lut = io.build_channel_lut(cfg)
    hits = io.load_subrun_hits(run, subrun, lut,
                               columns=io.HIT_COLUMNS + ['trigger_timestamp_ns'])
    g = hits[hits['eventId'] == evid]
    if g.empty:
        sys.exit(f'event {evid} not found in {subrun}')
    g = noise.flag_noise(g)
    drift_hv = io.parse_drift_hv(subrun) or 800.0
    drift = geo.DriftModel.from_drift_hv(drift_hv)
    res = D.analyze_event(g, drift)
    if res is None:
        sys.exit('too few clean Det-A hits')
    print(f"event {evid}: n_xline={res['n_xline']} n_yline={res['n_yline']} "
          f"n_pair={res['n_pair']} double={res['is_double']} "
          f"topo={res['topo'].get('tag','-')}")
    for pl, lines in (('x', res['xlines']), ('y', res['ylines'])):
        for i, ln in enumerate(lines):
            print(f"  {pl} line {i}: n={ln['n_hits']} strips={ln['n_strips']} "
                  f"slope={ln['slope_mm_ns']*1000:.2f} um/ns r2={ln['r2']:.3f} "
                  f"pspan={ln['pspan_mm']:.1f} tspan={ln['tspan_ns']:.0f} "
                  f"q={ln['q_sum']:.0f}")
    if res['topo']:
        print('  topo:', {k: (round(v, 1) if isinstance(v, float) else v)
                           for k, v in res['topo'].items()})
    path = os.path.join(OUT, f'{run}_{subrun}_ev{evid}.png')
    draw(g, res, path)
    print('  ->', path)


if __name__ == '__main__':
    main()
