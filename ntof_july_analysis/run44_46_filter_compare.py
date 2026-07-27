#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run44_46_filter_compare.py

Pb-filter effect on the gamma-flash: two IDENTICAL flash-only HV scans, the
only difference a 20 mm Pb slab in the beamline.

  run_44  3He, Ar/Iso 95/5, NO filter        (A490->A390, -5V; det D 20V below)
  run_46  3He, Ar/Iso 95/5, 20 mm Pb filter   (A490->A400, -5V; det D 20V below)

Both are gamma-flash-only triggers (Mode 1, mesh disconnected), drift 800 V,
400 smp x 20 ns, same gas / target / geometry.  Because the two runs share gas,
target and drift, the resist HV is directly comparable point-for-point -- NO
gain-equivalence remap is needed.

TIME WINDOWS ARE FLASH-CENTRED PER RUN.  The gamma-flash trigger latency drifted
between these runs -- the flash lands at ~1.0-1.1 us in run_44 but ~1.5 us in
run_46 -- so a single fixed edge set would mis-bin one of them.  Each run's
windows are built by july_hv_scan.flash_scan_windows(): the flash is measured
from the clean reference detector (mx17_A) at the top of the gain range, and the
coarse windows are

    [pre-flash]  [flash]  [post-flash 1]  [post-flash 2]  [post-flash 3]

bin 0 = pre-flash baseline, bin 1 fully contains the flash (centred on it), and
the readout tail is split into three post-flash recovery windows.  Rows in the
grids are therefore keyed by ROLE, not by an absolute time -- the flash row
compares each run's own flash (see the per-run window caption on every figure).

Events are additionally split by per-pulse beam intensity (low ~410e10 parasitic
vs high ~850e10 dedicated, via pulse_match.py, E10_SPLIT=600) -- the Pb slab
attenuates the flash at the detector but does NOT change the upstream
beam-intensity measurement, so the matched-intensity overlay is the clean
comparison of the filter's effect.

The gamma-flash SATURATES the detector (whole plane lights up, ~950 hits/event
regardless of gain), so the flash window itself is nearly identical between the
two runs and its amplitude is pinned -- neither is a useful filter observable.
The signal of the 20 mm Pb slab lives in the POST-FLASH hits/event (the recovery
/ afterglow after the flash), which is the headline here.

Outputs (-> {ANALYSIS_DIR}July_HV_Scan/run44_46_filter_compare/):
  * compare_timewin_grid_hits_per_event[_<cls>].png  rows=window role, cols=det
  * postflash_recovery_[combined|<cls>].png     hits/event after the flash (HEADLINE)
  * postflash_recovery_ratio[_<cls>].png        run_46/run_44 post-flash recovery
  * flash_combined.png                           the (saturated) flash window, ref
  * metrics_run44_46_filter.csv                  per (filter,cls,det,hv,window)

Run:
  .venv/bin/python ntof_july_analysis/run44_46_filter_compare.py
"""
import os
import re
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
# Two runs, distinguished ONLY by the Pb filter. run_44 is the no-filter
# baseline; run_46 has 20 mm Pb upstream. Both 3He / Ar-Iso 95/5 / drift 800.
RUNS = [
    {'run': 'run_44', 'filter': 'none',      'label': 'no filter',
     'color': '#1a4fa0', 'marker': 'o'},
    {'run': 'run_46', 'filter': 'Pb 20 mm',  'label': '20 mm Pb',
     'color': '#c0392b', 'marker': 's'},
]
RUN_STYLE = {r['run']: r for r in RUNS}
BASELINE_RUN = 'run_44'          # denominator of the attenuation ratio
FILTER_RUN = 'run_46'

SUBRUN_MATCH = r'^flash_dr800_'
OUT_LABEL = 'run44_46_filter_compare'

E10_SPLIT = 600.0          # pulses bimodal ~410e10 (parasitic) / ~850e10 (dedicated)
MIN_MATCH_FRAC = 0.5       # pulse_match quality gate (same as run32/33 lib)
MIN_N_CLASS = 10           # need this many matched events in a class to plot a point
CLASSES = ('low', 'high')

# Intensity styling for the combined (both-class) plots.
INT_STYLE = {
    'low':  dict(ls='--', alpha=0.6, fill=False, tag='low e10 (parasitic ~410)'),
    'high': dict(ls='-',  alpha=0.95, fill=True, tag='high e10 (dedicated ~850)'),
}
PLOT_KW = dict(lw=1.5, ms=4.8)

XLABEL = 'Resist HV [V]'
XNOTE = ('same gas/target/drift -> resist HV directly comparable; '
         'run_46 has 20 mm Pb in the beamline')

# Per-run flash-centred windows + measurement meta, filled by build_series().
RUN_WINDOWS: dict = {}
RUN_META: dict = {}


def run_label(run):
    r = RUN_STYLE[run]
    return f"{run} — {r['label']}"


def windows_caption():
    """One-line note of each run's flash window (they differ per run)."""
    bits = []
    for spec in RUNS:
        m = RUN_META.get(spec['run'])
        if m:
            bits.append(f"{spec['run']} flash {m['flash_lo']:.0f}-{m['flash_hi']:.0f} ns")
    return 'flash-centred windows:  ' + ',  '.join(bits)


def role_label(role_idx, run=None):
    """Role name + (if a single run) its absolute window, e.g. 'flash 875-1425 ns'."""
    role = WINDOW_ROLES[role_idx]
    if run is not None and run in RUN_WINDOWS:
        lo, hi = RUN_WINDOWS[run][role_idx]
        return f'{role}\n{lo:.0f}-{hi:.0f} ns'
    return role


# ---------------------------------------------------------------------------
def build_series():
    """
    Returns (series, det_names):
        series = [(run, cls, entry), ...]
        entry  = {det: {'hv':[n_pts], metric:[n_pts, n_role]}}
    Also populates RUN_WINDOWS / RUN_META (per-run flash-centred windows).
    """
    series = []
    det_names = None

    for spec in RUNS:
        run = spec['run']
        cfg = load_config(BASE_PATH, run)
        det_info = build_detector_info(cfg)
        hv_map = subrun_hvs(cfg)
        if det_names is None:
            det_names = list(det_info.keys())
        all_feus = sorted({f for di in det_info.values() for f in di['feus']})

        # Per-run flash-centred windows (measured from the clean det at top gain).
        windows, meta = flash_scan_windows(BASE_PATH, run)
        RUN_WINDOWS[run] = windows
        RUN_META[run] = meta
        print(f'  {run:7s} {spec["label"]:10s} flash '
              f'{meta["flash_lo"]:.0f}-{meta["flash_hi"]:.0f} ns  windows='
              f'{[(int(a), int(b)) for a, b in windows]}')

        run_dir = os.path.join(BASE_PATH, run)
        pat = re.compile(SUBRUN_MATCH)
        subs = sorted(n for n in os.listdir(run_dir)
                      if os.path.isdir(os.path.join(run_dir, n)) and pat.search(n))

        # acc[cls][det] = [(hv, {metric: array}), ...]
        acc = {c: defaultdict(list) for c in CLASSES}
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
            n_by_cls = {c: int((cls == c).sum()) for c in CLASSES}

            df_cls = df['eventId'].map(eid_to_cls).fillna('')
            hvs = hv_map.get(name, {})
            for c in CLASSES:
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

        for c in CLASSES:
            entry = {}
            for det, pts in acc[c].items():
                pts.sort(key=lambda t: t[0])
                e = {'hv': np.array([p[0] for p in pts])}
                for k in WINDOW_METRICS:
                    e[k] = np.vstack([p[1][k] for p in pts])
                entry[det] = e
            series.append((run, c, entry))

        print(f'          -> {n_used}/{len(subs)} subruns used '
              f'({n_matchfail} failed pulse-match quality gate)')

    return series, det_names


def _line_style(run, cls, single_class):
    """Assemble matplotlib kwargs for one (run, cls) line."""
    rs = RUN_STYLE[run]
    if single_class:
        # split plot: one line per run (filter), solid + filled, label by filter
        return dict(color=rs['color'], mec=rs['color'], mfc=rs['color'],
                    marker=rs['marker'], ls='-', alpha=0.95,
                    label=run_label(run), **PLOT_KW)
    ist = INT_STYLE[cls]
    return dict(color=rs['color'], mec=rs['color'],
                mfc=rs['color'] if ist['fill'] else 'none',
                marker=rs['marker'], ls=ist['ls'], alpha=ist['alpha'],
                label=f'{run_label(run)} · {ist["tag"]}', **PLOT_KW)


def _select(series, classes):
    return [(run, cls, entry) for (run, cls, entry) in series if cls in classes]


# ---------------------------------------------------------------------------
def plot_grid(series, det_names, out_dir, metric, classes, fname, subtitle):
    """rows = window role, cols = detector. One line per selected (run, cls)."""
    single = len(classes) == 1
    label = WINDOW_METRICS[metric]
    nrow, ncol = len(WINDOW_ROLES), len(det_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.25 * nrow),
                             squeeze=False, sharex=True)
    for wi, role in enumerate(WINDOW_ROLES):
        for ci, det in enumerate(det_names):
            ax = axes[wi][ci]
            for run, cls, entry in _select(series, classes):
                s = entry.get(det)
                if s is None or s['hv'].size == 0:
                    continue
                x = s['hv']
                y = s[metric][:, wi]
                good = np.isfinite(x) & np.isfinite(y)
                if not good.any():
                    continue
                ax.plot(x[good], y[good], **_line_style(run, cls, single))
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if wi == POST1_ROLE_IDX:
                ax.set_facecolor('#fff6e6')
            if wi == 0:
                ax.set_title(det, fontsize=10)
            if wi == nrow - 1:
                ax.set_xlabel(XLABEL, fontsize=8)
            if ci == 0:
                ax.set_ylabel(role, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(handles), 2), fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{label} vs resist HV, per flash-centred window — {subtitle}\n'
                 f'rows = window role (indep. y), shaded = post-flash recovery window,  '
                 f'amp >= {AMP_THRESHOLD} ADC  —  {XNOTE}\n{windows_caption()}',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, out_dir, fname)


def plot_window_blowup(series, det_names, out_dir, role_idx, classes,
                       fname, subtitle, ylabel):
    """One detector per panel, a single window role blown up."""
    single = len(classes) == 1
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2),
                             squeeze=False, sharex=True)
    axes = axes[0]
    for run, cls, entry in _select(series, classes):
        for ci, det in enumerate(det_names):
            ax = axes[ci]
            s = entry.get(det)
            if s is None or s['hv'].size == 0:
                continue
            x = s['hv']
            y = s['hits_per_event'][:, role_idx]
            good = np.isfinite(x) & np.isfinite(y)
            if not good.any():
                continue
            kw = _line_style(run, cls, single)
            if ci != len(det_names) - 1:      # legend only on last panel
                kw.pop('label', None)
            ax.plot(x[good], y[good], **kw)
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=11)
        ax.set_xlabel(XLABEL)
        if ci == 0:
            ax.set_ylabel(ylabel)
    axes[-1].legend(fontsize=7.5, loc='upper left')
    fig.suptitle(subtitle + f'\ndrift 800 V,  amp >= {AMP_THRESHOLD} ADC  —  '
                 f'{XNOTE}\n{windows_caption()}', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, fname)


def plot_ratio(series, det_names, out_dir, role_idx, fname, subtitle, ylabel,
               classes):
    """
    run_46 / run_44 hits/event in a chosen window role, at matched HV, per det.
    Used for the post-flash recovery window (the observable that matters): a
    ratio != 1 is the Pb slab changing the post-flash afterglow/recovery.
    """
    single = len(classes) == 1
    wi = role_idx
    by_run = {r: {c: e for (rr, c, e) in series if rr == r} for r in
              (BASELINE_RUN, FILTER_RUN)}
    ncol = len(det_names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2),
                             squeeze=False, sharex=True, sharey=True)
    axes = axes[0]
    for cls in classes:
        base = by_run[BASELINE_RUN].get(cls, {})
        filt = by_run[FILTER_RUN].get(cls, {})
        for ci, det in enumerate(det_names):
            ax = axes[ci]
            b = base.get(det)
            f = filt.get(det)
            if b is None or f is None or b['hv'].size == 0 or f['hv'].size == 0:
                continue
            # ratio only where both runs have that exact HV point
            bmap = {round(float(h), 1): v
                    for h, v in zip(b['hv'], b['hits_per_event'][:, wi])}
            fmap = {round(float(h), 1): v
                    for h, v in zip(f['hv'], f['hits_per_event'][:, wi])}
            hv = sorted(set(bmap) & set(fmap))
            x, y = [], []
            for h in hv:
                bv, fv = bmap[h], fmap[h]
                if np.isfinite(bv) and np.isfinite(fv) and bv > 0:
                    x.append(h)
                    y.append(fv / bv)
            if not x:
                continue
            st = INT_STYLE[cls]
            ax.plot(x, y, marker='D', ms=5, lw=1.6,
                    ls='-' if single else st['ls'],
                    alpha=0.95 if single else st['alpha'],
                    color='#6a3d9a',
                    mfc='#6a3d9a' if (single or st['fill']) else 'none',
                    mec='#6a3d9a',
                    label=None if single else f'{INT_STYLE[cls]["tag"]}')
    for ci, det in enumerate(det_names):
        ax = axes[ci]
        ax.axhline(1.0, color='0.4', lw=1.0, ls=':')
        ax.grid(True, alpha=0.3)
        ax.set_title(det, fontsize=11)
        ax.set_xlabel(XLABEL)
        if ci == 0:
            ax.set_ylabel(ylabel)
    if not single:
        axes[-1].legend(fontsize=7.5, loc='best')
    fig.suptitle(subtitle + f'\n{WINDOW_ROLES[role_idx]} window (per-run, right '
                 f'after each flash),  ratio at matched resist HV\n'
                 f'{windows_caption()}', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, fname)


def write_csv(series, det_names, out_dir):
    path = os.path.join(out_dir, 'metrics_run44_46_filter.csv')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'filter', 'intensity_class', 'detector', 'resist_hv',
                    'window_role', 'win_lo_ns', 'win_hi_ns',
                    'hits_per_event', 'mean_amplitude'])
        for run, cls, entry in series:
            filt = RUN_STYLE[run]['filter']
            wins = RUN_WINDOWS[run]
            for det in det_names:
                s = entry.get(det)
                if s is None:
                    continue
                for i, hv in enumerate(s['hv']):
                    for wi, role in enumerate(WINDOW_ROLES):
                        lo, hi = wins[wi]
                        w.writerow([run, filt, cls, det, f'{hv:g}', role,
                                    f'{lo:g}', f'{hi:g}',
                                    f'{s["hits_per_event"][i, wi]:.5g}',
                                    ('' if not np.isfinite(s['mean_amplitude'][i, wi])
                                     else f'{s["mean_amplitude"][i, wi]:.5g}')])
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
def main():
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', OUT_LABEL)
    print(f'Output -> {out_dir}\n')
    series, det_names = build_series()
    if not any(entry for _, _, entry in series):
        print('No processed data / pulse-matched events -- nothing plotted.')
        return

    flash_ylabel = 'Hits / event  (flash window)'
    post_ylabel = 'Hits / event  (first post-flash window)'

    # hits/event only (amplitude is not a useful observable for the flash --
    # the flash saturates, so amplitude is pinned; the post-flash hits/event
    # recovery is what carries the Pb-filter signal).
    plot_grid(series, det_names, out_dir, 'hits_per_event', list(CLASSES),
              'compare_timewin_grid_hits_per_event.png',
              'no-filter vs 20 mm Pb x low/high intensity')
    for cls in CLASSES:
        plot_grid(series, det_names, out_dir, 'hits_per_event', [cls],
                  f'compare_timewin_grid_hits_per_event_{cls}.png',
                  f'{cls} intensity only (no filter vs 20 mm Pb)')

    # HEADLINE: post-flash recovery -- hits/event AFTER the flash, per det.
    # (The flash window itself saturates ~950 hits/event and is nearly identical
    #  between the runs; the Pb effect lives in the post-flash recovery.)
    plot_window_blowup(series, det_names, out_dir, POST1_ROLE_IDX, list(CLASSES),
                       'postflash_recovery_combined.png',
                       'Post-flash recovery (hits/event after the flash) — '
                       'no filter vs 20 mm Pb',
                       post_ylabel)
    for cls in CLASSES:
        plot_window_blowup(series, det_names, out_dir, POST1_ROLE_IDX, [cls],
                           f'postflash_recovery_{cls}.png',
                           f'Post-flash recovery (hits/event after the flash) — '
                           f'{cls} intensity (no filter vs 20 mm Pb)',
                           post_ylabel)
    plot_ratio(series, det_names, out_dir, POST1_ROLE_IDX,
               'postflash_recovery_ratio.png',
               '20 mm Pb post-flash recovery — low & high intensity',
               'post-flash hits/event ratio  run_46 / run_44', list(CLASSES))
    for cls in CLASSES:
        plot_ratio(series, det_names, out_dir, POST1_ROLE_IDX,
                   f'postflash_recovery_ratio_{cls}.png',
                   f'20 mm Pb post-flash recovery — {cls} intensity',
                   'post-flash hits/event ratio  run_46 / run_44', [cls])

    # Reference: the flash window itself (saturated; shown for completeness).
    plot_window_blowup(series, det_names, out_dir, FLASH_ROLE_IDX, list(CLASSES),
                       'flash_combined.png',
                       'Flash window (saturated) — no filter vs 20 mm Pb',
                       flash_ylabel)

    write_csv(series, det_names, out_dir)
    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
