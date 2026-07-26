#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_flash_intensity.py

run_30 flash blocks split by BEAM PULSE INTENSITY. Each event is matched to
its PS pulse via pulse_match.py (beam_watcher per-pulse log + clock-offset
fit); the July pulses are bimodal (~410e10 and ~850e10), so events split into
LOW (< E10_SPLIT) and HIGH classes. Metrics per (subrun, detector, window,
intensity class) with the run_30 realigned windows; mesh On/Off kept separate.

Output -> July_HV_Scan/run30_flash_intensity/:
  intgrid_hits_per_event.png   rows = window, cols = det; 4 lines =
                               (low/high) x (mesh On/Off)
  intgrid_mean_amplitude.png   same for mean hit amplitude
  flashwin_intensity.png       flash-window row blown up

Run:  .venv/bin/python ntof_july_analysis/run30_flash_intensity.py
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import july_hv_scan as jhs  # noqa: E402

RUN30_EDGES_NS = [0, 1200, 2600, 4400, 6200, 8000]
jhs.TIME_WINDOWS[:] = list(zip(RUN30_EDGES_NS[:-1], RUN30_EDGES_NS[1:]))

from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, TIME_WINDOWS, WINDOW_METRICS, AMP_THRESHOLD,
    load_config, build_detector_info, load_hits, detector_window_metrics, _save,
)
from run9_mesh_scan import subrun_hvs, resist_hv  # noqa: E402
from pulse_match import match_subrun  # noqa: E402

RUN = 'run_30'
E10_SPLIT = 600.0    # pulses are bimodal ~410 vs ~850 e10
CLASSES = [('low', lambda v: v is not None and v < E10_SPLIT, 'tab:blue'),
           ('high', lambda v: v is not None and v >= E10_SPLIT, 'tab:red')]
MESH_TAGS = [('On', r'^flashOn_A', '-o'), ('Off', r'^flashOff_A', '--s')]
OUT = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_flash_intensity')
FLASH_WIN = (1200, 2600)


def build():
    cfg = load_config(BASE_PATH, RUN)
    det_info = build_detector_info(cfg)
    hv_map = subrun_hvs(cfg)
    det_names = list(det_info.keys())
    all_feus = sorted({f for di in det_info.values() for f in di['feus']})

    run_dir = os.path.join(BASE_PATH, RUN)
    series = {}          # (mesh, class) -> det -> list[(hv, metrics[n_window])]
    for mesh, pat_s, _ in MESH_TAGS:
        pat = re.compile(pat_s)
        subs = sorted(n for n in os.listdir(run_dir)
                      if os.path.isdir(os.path.join(run_dir, n)) and pat.search(n))
        for name in subs:
            df = load_hits(BASE_PATH, RUN, name, all_feus)
            if df is None or df.empty:
                continue
            pm = match_subrun(RUN, name)
            if pm is None or pm['match_frac'] < 0.5:
                print(f'  ! no pulse match for {name} — skipped')
                continue
            e10 = pm['event_e10']
            hvs = hv_map.get(name, {})
            for cls, sel, _ in CLASSES:
                ev_ids = {e for e, v in e10.items() if sel(v)}
                if len(ev_ids) < 10:
                    continue
                dsub = df[df['eventId'].isin(ev_ids)]
                for det in det_names:
                    hv = resist_hv(hvs, det_info[det]['resist'])
                    if hv is None:
                        continue
                    wm = detector_window_metrics(dsub, det_info[det]['feus'],
                                                 len(ev_ids))
                    series.setdefault((mesh, cls), {}).setdefault(det, []).append(
                        (float(hv), {k: np.array(wm[k]) for k in WINDOW_METRICS}))
            print(f'  {name}: matched {pm["match_frac"]:.0%}, '
                  + '  '.join(f'{c}={sum(1 for v in e10.values() if s(v))}'
                              for c, s, _ in CLASSES))
    for key, dd in series.items():
        for det, pts in dd.items():
            pts.sort(key=lambda t: t[0])
    return series, det_names, cfg.get('gas', '')


def plot_grid(series, det_names, gas, metric):
    label = WINDOW_METRICS[metric]
    nrow, ncol = len(TIME_WINDOWS), len(det_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.25 * nrow),
                             squeeze=False, sharex=True)
    for wi, (lo, hi) in enumerate(TIME_WINDOWS):
        for ci, det in enumerate(det_names):
            ax = axes[wi][ci]
            for mesh, _, ls in MESH_TAGS:
                for cls, _, col in CLASSES:
                    pts = series.get((mesh, cls), {}).get(det)
                    if not pts:
                        continue
                    hv = np.array([p[0] for p in pts])
                    y = np.array([p[1][metric][wi] for p in pts])
                    good = np.isfinite(hv) & np.isfinite(y)
                    ax.plot(hv[good], y[good], ls, color=col, ms=3.5, lw=1.2,
                            alpha=0.85, label=f'{cls} e10, mesh {mesh}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if (lo, hi) == FLASH_WIN:
                ax.set_facecolor('#fff6e6')
            if wi == 0:
                ax.set_title(det, fontsize=10)
            if wi == nrow - 1:
                ax.set_xlabel('Resist HV [V]', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{lo/1000:g}-{hi/1000:g} us', fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'run_30 flash blocks — {label} vs HV, split by pulse '
                 f'intensity (low <{E10_SPLIT:.0f}e10 ~410, high ~850)'
                 f'{"  —  " + gas if gas else ""}', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _save(fig, OUT, f'intgrid_{metric}.png')


def plot_flashwin(series, det_names, gas):
    wi = TIME_WINDOWS.index(FLASH_WIN)
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    for ci, det in enumerate(det_names):
        ax = axes[0][ci]
        for mesh, _, ls in MESH_TAGS:
            for cls, _, col in CLASSES:
                pts = series.get((mesh, cls), {}).get(det)
                if not pts:
                    continue
                hv = np.array([p[0] for p in pts])
                y = np.array([p[1]['hits_per_event'][wi] for p in pts])
                ax.plot(hv, y, ls, color=col, ms=4,
                        label=f'{cls} e10, mesh {mesh}')
        ax.grid(alpha=0.3)
        ax.set_title(det)
        ax.set_xlabel('Resist HV [V]')
        if ci == 0:
            ax.set_ylabel(f'Hits/event ({FLASH_WIN[0]/1000:g}-{FLASH_WIN[1]/1000:g} us)')
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f'run_30 — flash-window hits/event vs HV by pulse intensity'
                 f'{"  —  " + gas if gas else ""}')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, OUT, 'flashwin_intensity.png')


if __name__ == '__main__':
    series, det_names, gas = build()
    if not series:
        print('nothing matched')
    else:
        plot_grid(series, det_names, gas, 'hits_per_event')
        plot_grid(series, det_names, gas, 'mean_amplitude')
        plot_flashwin(series, det_names, gas)
        print('Figures ->', OUT)
