#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run38_41_target_compare.py

run_38 (target=Marex) vs run_41 (target=3He): identical flash-triggered
resist-HV scans (dr800, A560->A400, -5V steps, all 4 detectors together) —
turn-on/turn-off curves (hits/event vs resist HV, per TIME_WINDOWS bin),
split by target AND by per-pulse beam intensity (low ~410e10 parasitic vs
high ~850e10 dedicated, via pulse_match.py / run32_lib's E10_SPLIT=600
convention).

Every event in these runs is itself a gamma-flash trigger (trigger = PS/
flash line only).  TIME WINDOWS ARE FLASH-CENTRED PER RUN
(july_hv_scan.flash_scan_windows): the flash is measured from the clean
reference detector (mx17_A) at the top of the gain range and the coarse
windows are [pre-flash] [flash] [post-flash 1..3].  The flash row *is* the
flash turning on/off with gain -- it gets its own blown-up panel here
(target=color, intensity=linestyle), in addition to the full window-role grid
and the first post-flash (recovery) blowup.

Edit RUNS / OUT_LABEL and run:
  .venv/bin/python ntof_july_analysis/run38_41_target_compare.py
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, WINDOW_METRICS, AMP_THRESHOLD,
    WINDOW_ROLES, FLASH_ROLE_IDX, POST1_ROLE_IDX,
    load_config, build_detector_info, load_hits, detector_window_metrics,
    flash_scan_windows, _save,
)
from run9_mesh_scan import subrun_hvs, resist_hv  # noqa: E402
import pulse_match  # noqa: E402

# ---------------------------------------------------------------------------
RUNS = [
    {'run': 'run_38', 'target': 'Marex'},
    {'run': 'run_41', 'target': '3He'},
]
SUBRUN_MATCH = r'^flash_dr800_'
OUT_LABEL = 'run38_41_target_intensity'

E10_SPLIT = 600.0          # pulses are bimodal ~410e10 (parasitic) / ~850e10 (dedicated)
MIN_MATCH_FRAC = 0.5       # pulse_match quality gate (same as run32/33 lib)
MIN_N_CLASS = 10           # need at least this many matched events in a class to plot a point

# Per-run flash-centred windows + measurement meta, filled by build_*_series().
RUN_WINDOWS: dict = {}
RUN_META: dict = {}


def windows_caption():
    bits = []
    for spec in RUNS:
        m = RUN_META.get(spec['run'])
        if m:
            bits.append(f"{spec['run']} flash {m['flash_lo']:.0f}-{m['flash_hi']:.0f} ns")
    return 'flash-centred windows:  ' + ',  '.join(bits)

# Color groups by intensity (blue=low, red=high); marker+shade differentiate the
# run/target within a color group (Marex=dark/circle, 3He=light/triangle).
STYLE = {
    ('Marex', 'low'):  dict(color='#1a4fa0', marker='o',
                            tag='Marex, low e10 (parasitic, ~410)'),
    ('3He',   'low'):  dict(color='#8fc1f5', marker='^',
                            tag='3He, low e10 (parasitic, ~410)'),
    ('Marex', 'high'): dict(color='#9c0d0d', marker='o',
                            tag='Marex, high e10 (dedicated, ~850)'),
    ('3He',   'high'): dict(color='#f2948a', marker='^',
                            tag='3He, high e10 (dedicated, ~850)'),
}
PLOT_KW = dict(lw=1.5, ms=4.5, alpha=0.9)


# ---------------------------------------------------------------------------
def build_target_intensity_series():
    """
    Returns (series, det_names, gas, series_meta):
        series      = [(label, {det: {'hv':[...], metric:[n_pts,n_window]}}), ...]
        series_meta = [(target, cls), ...] aligned with series, for styling
    """
    series, series_meta = [], []
    det_names, gas = None, ''

    for spec in RUNS:
        run, target = spec['run'], spec['target']
        cfg = load_config(BASE_PATH, run)
        det_info = build_detector_info(cfg)
        hv_map = subrun_hvs(cfg)
        if det_names is None:
            det_names = list(det_info.keys())
            gas = cfg.get('gas', '')
        all_feus = sorted({f for di in det_info.values() for f in di['feus']})

        # Per-run flash-centred windows (measured from the clean det at top gain).
        windows, meta = flash_scan_windows(BASE_PATH, run)
        RUN_WINDOWS[run] = windows
        RUN_META[run] = meta
        print(f'  {target:6s} {run:7s} flash '
              f'{meta["flash_lo"]:.0f}-{meta["flash_hi"]:.0f} ns  windows='
              f'{[(int(a), int(b)) for a, b in windows]}')

        run_dir = os.path.join(BASE_PATH, run)
        pat = re.compile(SUBRUN_MATCH)
        subs = sorted(n for n in os.listdir(run_dir)
                      if os.path.isdir(os.path.join(run_dir, n)) and pat.search(n))

        # acc[cls][det] = [(hv, {metric: array}), ...]
        acc = {c: defaultdict(list) for c in ('low', 'high')}
        n_used, n_matchfail = 0, 0
        for name in subs:
            df = load_hits(BASE_PATH, run, name, all_feus)
            if df is None or df.empty:
                continue
            pm = pulse_match.match_subrun(run, name)
            if pm is None or pm['match_frac'] < MIN_MATCH_FRAC:
                n_matchfail += 1
                continue
            eids = np.array(list(pm['event_e10'].keys()), dtype=np.int64)
            e10 = np.array([v if v is not None else np.nan
                            for v in pm['event_e10'].values()])
            cls = np.full(eids.shape, '', dtype='<U4')
            cls[np.isfinite(e10) & (e10 < E10_SPLIT)] = 'low'
            cls[np.isfinite(e10) & (e10 >= E10_SPLIT)] = 'high'
            eid_to_cls = dict(zip(eids.tolist(), cls.tolist()))
            n_by_cls = {c: int((cls == c).sum()) for c in ('low', 'high')}

            df_cls = df['eventId'].map(eid_to_cls).fillna('')
            hvs = hv_map.get(name, {})
            for c in ('low', 'high'):
                if n_by_cls[c] < MIN_N_CLASS:
                    continue
                sub_df = df[df_cls.values == c]
                for det in det_names:
                    hv = resist_hv(hvs, det_info[det]['resist'])
                    if hv is None:
                        continue
                    wm = detector_window_metrics(sub_df, det_info[det]['feus'],
                                                 n_by_cls[c], windows=windows)
                    acc[c][det].append((float(hv), {k: np.array(wm[k]) for k in WINDOW_METRICS}))
            n_used += 1

        for c in ('low', 'high'):
            entry = {}
            for det, pts in acc[c].items():
                pts.sort(key=lambda t: t[0])
                e = {'hv': np.array([p[0] for p in pts])}
                for k in WINDOW_METRICS:
                    e[k] = np.vstack([p[1][k] for p in pts])
                entry[det] = e
            label = f'{target} ({run}) — {STYLE[(target, c)]["tag"]}'
            series.append((label, entry))
            series_meta.append((target, c))

        print(f'  {target:6s} {run:7s} -> {n_used}/{len(subs)} subruns used '
              f'({n_matchfail} failed pulse-match quality gate)')

    return series, det_names, gas, series_meta


# ---------------------------------------------------------------------------
def plot_grid_styled(series, series_meta, det_names, gas, out_dir, metric):
    """rows = window role, cols = detector, one line per (target, intensity) series."""
    label = WINDOW_METRICS[metric]
    nrow, ncol = len(WINDOW_ROLES), len(det_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.25 * nrow),
                             squeeze=False, sharex=True)
    for wi, role in enumerate(WINDOW_ROLES):
        for ci, det in enumerate(det_names):
            ax = axes[wi][ci]
            for (slabel, entry), (target, c) in zip(series, series_meta):
                s = entry.get(det)
                if s is None or s['hv'].size == 0:
                    continue
                y = s[metric][:, wi]
                good = np.isfinite(s['hv']) & np.isfinite(y)
                if not good.any():
                    continue
                st = STYLE[(target, c)]
                ax.plot(s['hv'][good], y[good], marker=st['marker'],
                        color=st['color'], mec=st['color'], label=st['tag'], **PLOT_KW)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if wi == POST1_ROLE_IDX:
                ax.set_facecolor('#fff6e6')
            if wi == 0:
                ax.set_title(det, fontsize=10)
            if wi == nrow - 1:
                ax.set_xlabel('Resist HV [V]', fontsize=8)
            if ci == 0:
                ax.set_ylabel(role, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(series), 4),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'{label} vs resist HV, per flash-centred window{gas_s}\n'
                 f'rows = window role (indep. y), shaded = post-flash recovery window,  '
                 f'amp >= {AMP_THRESHOLD} ADC\n{windows_caption()}', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _save(fig, out_dir, f'compare_timewin_grid_{metric}.png')


def plot_postflash_styled(series, series_meta, det_names, gas, out_dir):
    """The first post-flash (recovery decay) window blown up, one panel per det."""
    wi = POST1_ROLE_IDX
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False, sharex=True)
    axes = axes[0]
    for (slabel, entry), (target, c) in zip(series, series_meta):
        st = STYLE[(target, c)]
        for ci, det in enumerate(det_names):
            ax = axes[ci]
            s = entry.get(det)
            if s is None or s['hv'].size == 0:
                continue
            y = s['hits_per_event'][:, wi]
            good = np.isfinite(s['hv']) & np.isfinite(y)
            if not good.any():
                continue
            lab = st['tag'] if ci == len(det_names) - 1 else None
            ax.plot(s['hv'][good], y[good], marker=st['marker'],
                    color=st['color'], mec=st['color'], label=lab, **PLOT_KW)
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=10)
        ax.set_xlabel('Resist HV [V]')
        if ci == 0:
            ax.set_ylabel('Hits / event  (first post-flash window)')
    axes[-1].legend(fontsize=8)
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'First post-flash (recovery) window vs resist HV{gas_s}\n'
                 f'amp >= {AMP_THRESHOLD} ADC  —  {windows_caption()}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, 'compare_postflash_turnoff.png')


def plot_flash_turnon(series, series_meta, det_names, gas, out_dir):
    """Blown-up flash-window hits/event vs resist HV, one panel per detector;
    color = intensity group, marker/shade = target."""
    wi = FLASH_ROLE_IDX
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4), squeeze=False, sharex=True)
    axes = axes[0]
    for (slabel, entry), (target, c) in zip(series, series_meta):
        st = STYLE[(target, c)]
        for ci, det in enumerate(det_names):
            ax = axes[ci]
            s = entry.get(det)
            if s is None or s['hv'].size == 0:
                continue
            y = s['hits_per_event'][:, wi]
            good = np.isfinite(s['hv']) & np.isfinite(y)
            if not good.any():
                continue
            lab = st['tag'] if ci == len(det_names) - 1 else None
            ax.plot(s['hv'][good], y[good], marker=st['marker'],
                    color=st['color'], mec=st['color'], label=lab, **PLOT_KW)
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=11)
        ax.set_xlabel('Resist HV [V]')
        if ci == 0:
            ax.set_ylabel('Hits / event  (flash window)')
    axes[-1].legend(fontsize=7.5, loc='upper left')
    gas_s = f'  —  {gas}' if gas else ''
    fig.suptitle(f'Flash turn-on/turn-off vs resist HV, by target & beam '
                 f'intensity{gas_s}\ndrift 800 V,  amp >= {AMP_THRESHOLD} ADC  —  '
                 f'{windows_caption()}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, 'flash_turnon_by_target_intensity.png')


def main():
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', OUT_LABEL)
    print(f'Output -> {out_dir}\n')
    series, det_names, gas, series_meta = build_target_intensity_series()
    if not any(entry for _, entry in series):
        print('No processed data / pulse-matched events -- nothing plotted.')
        return

    plot_grid_styled(series, series_meta, det_names, gas, out_dir, 'hits_per_event')
    plot_grid_styled(series, series_meta, det_names, gas, out_dir, 'mean_amplitude')
    plot_postflash_styled(series, series_meta, det_names, gas, out_dir)  # first post-flash (recovery)
    plot_flash_turnon(series, series_meta, det_names, gas, out_dir)      # flash-window blowup

    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
