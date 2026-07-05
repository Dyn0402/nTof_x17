#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/compare_scans.py

General overlay tool for the July HV scans: plot **hits/event and mean hit
amplitude vs resist HV, in each coarse time window, with one series per
arbitrary (run, subrun-selection)** — so you can compare scans *across runs*
(e.g. a fresh baseline run vs a few scans from an earlier run).

Each series is one entry in ``SERIES`` below:
    {'label': ...,          # legend text
     'run':   'run_10',     # run directory (or just '10')
     'match': r'^dr800_'}   # regex; a subrun of that run joins the series if
                            # re.search(match, subrun_name) hits

Subruns are matched by name; resist HV per detector is read from that run's
run_config.json (card 5), so mixed naming schemes
(``scan01_dr800_A470_09`` and ``dr800_A470_09``) both work. Unprocessed
subruns are silently skipped, so this is safe to run on a live run.

Output -> {ANALYSIS_DIR}July_HV_Scan/{OUT_LABEL}/  (flask "Analysis" tab):
    compare_timewin_grid_hits_per_event.png   rows=time window, cols=detector
    compare_timewin_grid_mean_amplitude.png     (indep. y, mid-window shaded)
    compare_midwindow_turnoff.png             the 1.15-3.5 us row blown up

Edit SERIES / OUT_LABEL and run:  .venv/bin/python ntof_july_analysis/compare_scans.py
"""

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, TIME_WINDOWS, WINDOW_METRICS, AMP_THRESHOLD,
    load_config, build_detector_info, load_hits, get_total_events,
    detector_window_metrics, _save,
)
from run9_mesh_scan import subrun_hvs, resist_hv  # noqa: E402

# ---------------------------------------------------------------------------
# What to overlay — edit this.
# ---------------------------------------------------------------------------
SERIES = [
    {'label': 'run_10 (mesh disconnected)', 'run': 'run_10', 'match': r'^dr800_'},
    {'label': 'run_9 scan01 (mesh off)',    'run': 'run_9',  'match': r'^scan01_'},
    {'label': 'run_9 scan02 (cfg 2)',       'run': 'run_9',  'match': r'^scan02_'},
    {'label': 'run_9 scan03 (cfg 3)',       'run': 'run_9',  'match': r'^scan03_'},
]
OUT_LABEL = 'run10_vs_run9'          # output subdir under July_HV_Scan/
MID_WINDOW = (1150, 3500)            # the "mid-window turn-off" row (ns)


# ---------------------------------------------------------------------------
# Build the series
# ---------------------------------------------------------------------------

def build_series(specs: List[dict]):
    """
    Returns (series, det_names, gas):
        series = [(label, {det: {'hv': [...], metric: [n_pts, n_window], ...}}), ...]
    in the order given.  Detector list / gas come from the first run's config.
    """
    run_cache: Dict[str, tuple] = {}
    series, det_names, gas = [], None, ''

    for spec in specs:
        run = spec['run'] if str(spec['run']).startswith('run_') else f"run_{spec['run']}"
        if run not in run_cache:
            cfg = load_config(BASE_PATH, run)
            run_cache[run] = (cfg, build_detector_info(cfg), subrun_hvs(cfg))
        cfg, det_info, hv_map = run_cache[run]
        if det_names is None:
            det_names = list(det_info.keys())
            gas = cfg.get('gas', '')
        all_feus = sorted({f for di in det_info.values() for f in di['feus']})
        pat = re.compile(spec['match'])

        run_dir = os.path.join(BASE_PATH, run)
        subs = sorted(n for n in os.listdir(run_dir)
                      if os.path.isdir(os.path.join(run_dir, n)) and pat.search(n))

        acc: Dict[str, List[Tuple[float, Dict[str, np.ndarray]]]] = defaultdict(list)
        n_used = 0
        for name in subs:
            df = load_hits(BASE_PATH, run, name, all_feus)
            if df is None or df.empty:
                continue
            n_total = get_total_events(BASE_PATH, run, name) or df['eventId'].nunique()
            hvs = hv_map.get(name, {})
            for det in det_names:
                hv = resist_hv(hvs, det_info[det]['resist'])
                if hv is None:
                    continue
                wm = detector_window_metrics(df, det_info[det]['feus'], n_total)
                acc[det].append((float(hv), {k: np.array(wm[k]) for k in WINDOW_METRICS}))
            n_used += 1

        entry: Dict[str, dict] = {}
        for det, pts in acc.items():
            pts.sort(key=lambda t: t[0])
            e = {'hv': np.array([p[0] for p in pts])}
            for k in WINDOW_METRICS:
                e[k] = np.vstack([p[1][k] for p in pts])
            entry[det] = e
        series.append((spec['label'], entry))
        print(f'  {spec["label"]:36s} {run:7s} match={spec["match"]:12s} '
              f'-> {n_used}/{len(subs)} subruns used')

    return series, det_names, gas


# ---------------------------------------------------------------------------
# Plotting (series-keyed)
# ---------------------------------------------------------------------------

def _colors(n):
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n)]


def plot_grid(series, det_names, gas, out_dir, metric):
    """rows = time window, cols = detector, one line per series, indep. y."""
    label = WINDOW_METRICS[metric]
    cols = _colors(len(series))
    nrow, ncol = len(TIME_WINDOWS), len(det_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.25 * nrow),
                             squeeze=False, sharex=True)
    for wi, (lo, hi) in enumerate(TIME_WINDOWS):
        for ci, det in enumerate(det_names):
            ax = axes[wi][ci]
            for si, (slabel, entry) in enumerate(series):
                s = entry.get(det)
                if s is None or s['hv'].size == 0:
                    continue
                y = s[metric][:, wi]
                good = np.isfinite(s['hv']) & np.isfinite(y)
                if not good.any():
                    continue
                ax.plot(s['hv'][good], y[good], '-o', ms=3.5, lw=1.3,
                        color=cols[si], label=slabel)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if (lo, hi) == MID_WINDOW:
                ax.set_facecolor('#fff6e6')
            if wi == 0:
                ax.set_title(det, fontsize=10)
            if wi == nrow - 1:
                ax.set_xlabel('Resist HV [V]', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{lo/1000:g}-{hi/1000:g} us', fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(series), 4),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'{label} vs resist HV, per time window{gas_s}\n'
                 f'rows = time window (indep. y), shaded = mid-window turn-off,  '
                 f'amp >= {AMP_THRESHOLD} ADC', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, out_dir, f'compare_timewin_grid_{metric}.png')


def plot_midwindow(series, det_names, gas, out_dir):
    """The mid-window row blown up, one panel per detector."""
    wi = TIME_WINDOWS.index(MID_WINDOW)
    cols = _colors(len(series))
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0),
                             squeeze=False, sharex=True)
    axes = axes[0]
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        for si, (slabel, entry) in enumerate(series):
            s = entry.get(det)
            if s is None or s['hv'].size == 0:
                continue
            y = s['hits_per_event'][:, wi]
            good = np.isfinite(s['hv']) & np.isfinite(y)
            if not good.any():
                continue
            ax.plot(s['hv'][good], y[good], '-o', ms=4, lw=1.6,
                    color=cols[si], label=slabel)
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=10)
        ax.set_xlabel('Resist HV [V]')
        if ci == 0:
            lo, hi = MID_WINDOW
            ax.set_ylabel(f'Hits / event  ({lo/1000:g}-{hi/1000:g} us)')
    axes[-1].legend(fontsize=8)
    gas_s = f'  —  {gas}' if gas else ''
    lo, hi = MID_WINDOW
    fig.suptitle(f'Mid-window ({lo/1000:g}-{hi/1000:g} us) turn-off vs resist HV'
                 f'{gas_s}\namp >= {AMP_THRESHOLD} ADC', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, 'compare_midwindow_turnoff.png')


def main():
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', OUT_LABEL)
    print(f'Output -> {out_dir}\n')
    series, det_names, gas = build_series(SERIES)
    if not any(entry for _, entry in series):
        print('No processed data matched any series — nothing plotted.')
        return
    plot_grid(series, det_names, gas, out_dir, 'hits_per_event')
    plot_grid(series, det_names, gas, out_dir, 'mean_amplitude')
    plot_midwindow(series, det_names, gas, out_dir)
    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
