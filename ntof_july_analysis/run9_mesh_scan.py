#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run9_mesh_scan.py

run_9 packs several amplification-HV scans into one run, each taken with a
different mesh-circuit configuration set on the N1081B.  Subruns are named

    scan<NN>_dr<DRIFT>_A<HV>_<idx>

where ``scan<NN>`` selects the mesh config (scan01 = baseline, mesh circuit
OFF), ``dr<DRIFT>`` is the drift HV and ``A<HV>`` tracks the mx17_A resist HV
(each detector's own resist HV is read from run_config.json).

This plots **hits / event vs resist HV** in each coarse time-of-arrival window
(reusing TIME_WINDOWS from july_hv_scan), with **one series per scan** overlaid.
The point of interest is whether the *mid-window turn-off* — the resist HV at
which post-flash hits/event drops precipitously (the 1.15-3.5 us row) — shifts
with the mesh configuration.

Grid = rows: time window, cols: detector.  Every panel autoscales its own y so
the turn-off shape is legible in each window.

    python run9_mesh_scan.py [run]      # default run_9

Output -> {ANALYSIS_DIR}July_HV_Scan/{run}_mesh/  (flask "Analysis" tab).
"""

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from july_hv_scan import (  # noqa: E402  (reuse the July helpers)
    BASE_PATH, ANALYSIS_DIR, MX17_DETECTORS, TIME_WINDOWS, WINDOW_METRICS,
    AMP_THRESHOLD, load_config, build_detector_info, load_hits,
    get_total_events, detector_window_metrics, discover_run, _save,
)

# scan<NN>_dr<DRIFT>_A<HV>_<idx>
SCAN_RE = re.compile(r'scan(\d+)_dr(\d+)_A(\d+)_(\d+)$')

DEFAULT_RUN = 'run_9'


def subrun_hvs(cfg: dict) -> Dict[str, dict]:
    """Map sub_run_name -> its configured hvs dict."""
    return {s['sub_run_name']: s.get('hvs', {}) for s in cfg.get('sub_runs', [])}


def resist_hv(hvs: dict, resist_ch: Optional[Tuple[int, int]]) -> Optional[float]:
    if not resist_ch:
        return None
    card, ch = resist_ch
    return hvs.get(str(card), {}).get(str(ch))


def collect(run: str):
    """
    Walk run_9's subruns, group by scan number, and compute per-detector,
    per-time-window hits/event vs resist HV.

    Returns (series, det_names, drift_by_scan) where
        series[scan][det] = {'hv': [...], 'win': np.array[n_pts, n_window]}
    """
    cfg = load_config(BASE_PATH, run)
    gas = cfg.get('gas', '')
    det_info = build_detector_info(cfg)
    det_names = list(det_info.keys())
    all_feus = sorted({f for di in det_info.values() for f in di['feus']})
    hv_map = subrun_hvs(cfg)

    run_dir = os.path.join(BASE_PATH, run)
    subs = sorted(n for n in os.listdir(run_dir)
                  if os.path.isdir(os.path.join(run_dir, n)) and SCAN_RE.match(n))

    # scan -> det -> list of (resist_hv, {metric: [per window]})
    acc: Dict[int, Dict[str, List[Tuple[float, Dict[str, np.ndarray]]]]] = \
        defaultdict(lambda: defaultdict(list))
    drift_by_scan: Dict[int, int] = {}
    n_done = n_skip = 0

    for name in subs:
        m = SCAN_RE.match(name)
        scan, drift = int(m.group(1)), int(m.group(2))
        drift_by_scan[scan] = drift
        df = load_hits(BASE_PATH, run, name, all_feus)
        if df is None or df.empty:
            n_skip += 1
            continue
        n_total = get_total_events(BASE_PATH, run, name) or df['eventId'].nunique()
        hvs = hv_map.get(name, {})
        for det in det_names:
            hv = resist_hv(hvs, det_info[det]['resist'])
            if hv is None:
                continue
            wm = detector_window_metrics(df, det_info[det]['feus'], n_total)
            acc[scan][det].append(
                (float(hv), {k: np.array(wm[k]) for k in WINDOW_METRICS}))
        n_done += 1
        print(f'  {name}: scan{scan:02d} dr{drift} nev={n_total}')

    # sort each series by resist HV; stack each metric into [n_pts, n_window]
    series: Dict[int, Dict[str, dict]] = {}
    for scan, per_det in acc.items():
        series[scan] = {}
        for det, pts in per_det.items():
            pts.sort(key=lambda t: t[0])
            entry = {'hv': np.array([p[0] for p in pts])}
            for k in WINDOW_METRICS:
                entry[k] = np.vstack([p[1][k] for p in pts])  # [n_pts, n_window]
            series[scan][det] = entry
    print(f'\nProcessed {n_done} subruns, skipped {n_skip} (unprocessed).')
    return series, det_names, drift_by_scan, gas


def scan_label(scan: int, drift: int, base_drift: Optional[int]) -> str:
    tag = 'baseline, mesh off' if scan == 1 else f'cfg {scan}'
    if base_drift is not None and drift != base_drift:
        tag += f', dr{drift}'
    return f'scan{scan:02d} ({tag})'


def plot_mesh_windows(series, det_names, drift_by_scan, gas, run, out_dir,
                      metric='hits_per_event'):
    """rows = time window, cols = detector, one line per scan, indep. y."""
    label = WINDOW_METRICS[metric]
    scans = sorted(series)
    base_drift = drift_by_scan.get(1)
    cmap = plt.get_cmap('tab10')
    colors = {sc: cmap(i % 10) for i, sc in enumerate(scans)}

    nrow, ncol = len(TIME_WINDOWS), len(det_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.25 * nrow),
                             squeeze=False, sharex=True)
    for wi, (lo, hi) in enumerate(TIME_WINDOWS):
        for ci, det in enumerate(det_names):
            ax = axes[wi][ci]
            for sc in scans:
                s = series[sc].get(det)
                if s is None or s['hv'].size == 0:
                    continue
                y = s[metric][:, wi]
                good = np.isfinite(s['hv']) & np.isfinite(y)
                if not good.any():
                    continue
                ax.plot(s['hv'][good], y[good], '-o', ms=3, lw=1.2,
                        color=colors[sc],
                        label=scan_label(sc, drift_by_scan[sc], base_drift))
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            # highlight the mid-window turn-off row
            if (lo, hi) == (1150, 3500):
                ax.set_facecolor('#fff6e6')
            if wi == 0:
                ax.set_title(det, fontsize=10)
            if wi == nrow - 1:
                ax.set_xlabel('Resist HV [V]', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{lo/1000:g}-{hi/1000:g} us', fontsize=8)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(scans), 5),
               fontsize=7, title='Mesh config', title_fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'{label} vs resist HV, per time window & mesh config  '
                 f'—  {run}{gas_s}\nrows = time window (indep. y), '
                 f'shaded = mid-window turn-off,  amp >= {AMP_THRESHOLD} ADC',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, out_dir, f'mesh_timewin_grid_{metric}.png')


def plot_midwindow_focus(series, det_names, drift_by_scan, gas, run, out_dir):
    """A single row (the 1.15-3.5 us mid window) blown up, one panel per det."""
    wi = TIME_WINDOWS.index((1150, 3500))
    scans = sorted(series)
    base_drift = drift_by_scan.get(1)
    cmap = plt.get_cmap('tab10')
    colors = {sc: cmap(i % 10) for i, sc in enumerate(scans)}

    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0),
                             squeeze=False, sharex=True)
    axes = axes[0]
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        for sc in scans:
            s = series[sc].get(det)
            if s is None or s['hv'].size == 0:
                continue
            y = s['hits_per_event'][:, wi]
            good = np.isfinite(s['hv']) & np.isfinite(y)
            if not good.any():
                continue
            ax.plot(s['hv'][good], y[good], '-o', ms=4, lw=1.5,
                    color=colors[sc],
                    label=scan_label(sc, drift_by_scan[sc], base_drift))
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=10)
        ax.set_xlabel('Resist HV [V]')
        if ci == 0:
            ax.set_ylabel('Hits / event  (1.15-3.5 us)')
    axes[-1].legend(fontsize=7, title='Mesh config', title_fontsize=8)
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'Mid-window (1.15-3.5 us) turn-off vs resist HV & mesh config  '
                 f'—  {run}{gas_s}\namp >= {AMP_THRESHOLD} ADC', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, 'mesh_midwindow_turnoff.png')


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    if not run.startswith('run_'):
        run = f'run_{run}'
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', f'{run}_mesh')
    print(f'Run {run}   Output -> {out_dir}\n')

    series, det_names, drift_by_scan, gas = collect(run)
    if not series:
        print('No processed scan data found — nothing plotted.')
        return
    print(f'Scans with data: {sorted(series)}')

    plot_mesh_windows(series, det_names, drift_by_scan, gas, run, out_dir,
                      metric='hits_per_event')
    plot_mesh_windows(series, det_names, drift_by_scan, gas, run, out_dir,
                      metric='mean_amplitude')
    plot_midwindow_focus(series, det_names, drift_by_scan, gas, run, out_dir)
    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
