#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run38_41_44_flash_compare.py

Three flash-triggered resist-HV scans compared:
  run_38  target=Marex, gas Ar/Iso 90/10   (A560->A400, -5V, all 4 dets same HV)
  run_41  target=3He,   gas Ar/Iso 90/10   (A560->A400, -5V, all 4 dets same HV)
  run_44  target=3He,   gas Ar/Iso 95/5    (A490->A390, -5V; det D 20V below)

All three are gamma-flash-only triggers (Mode 1, mesh disconnected), drift
800 V, 400 smp x 20 ns. Every event IS a flash trigger. Time windows are
FLASH-CENTRED PER RUN (july_hv_scan.flash_scan_windows): the flash time drifts
run-to-run, so each run's flash is measured from the clean detector (mx17_A) at
top gain and the coarse windows are [pre-flash] [flash] [post-flash 1..3]. The
flash row turns on/off with gain; rows are keyed by role, not absolute time.

Because run_41 and run_44 share the same target (3He), the series are keyed
and colored by RUN, not by target (run_44's extra distinction is the 95/5
quencher fraction). Each run's events are additionally split by per-pulse
beam intensity (low ~410e10 parasitic vs high ~850e10 dedicated, via
pulse_match.py, E10_SPLIT=600).

With three runs the 6-line (3 runs x 2 intensities) panels get busy, so in
addition to the usual combined grids this script also emits INTENSITY-SPLIT
versions: a low-intensity-only figure and a high-intensity-only figure (3
lines each) for the time-window grids, the flash-window blowup, and the
post-flash mid-window blowup.

Because run_44 is in a different gas (95/5) than run_38/41 (90/10), the same
resist HV is NOT the same gas gain across runs. Every figure is therefore
emitted twice:
  <name>.png         x = resist HV as set
  <name>_gaineq.png  x = gain-matched 95/5 volts (gain_map.py / garfield
                     hv_equivalence.json, CERN pressure) -- run_38/41's 90/10
                     HV mapped to the 95/5 voltage of equal simulated gain;
                     run_44 (95/5) unchanged. 95/5-equiv <400 V is shaded
                     (extrapolated beyond the 95/5 simulated span).

Run:
  .venv/bin/python ntof_july_analysis/run38_41_44_flash_compare.py
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

from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, WINDOW_METRICS, AMP_THRESHOLD,
    WINDOW_ROLES, FLASH_ROLE_IDX, POST1_ROLE_IDX,
    load_config, build_detector_info, load_hits, detector_window_metrics,
    flash_scan_windows, _save,
)
from run9_mesh_scan import subrun_hvs, resist_hv  # noqa: E402
import pulse_match  # noqa: E402
from gain_map import GainMap  # noqa: E402

# ---------------------------------------------------------------------------
# Color by RUN (run_41 and run_44 are both 3He, so target alone won't
# distinguish them). Marker also per run; intensity is shown by linestyle in
# the combined plots and is the whole content of the split plots.
RUNS = [
    {'run': 'run_38', 'target': 'Marex', 'gas': '90/10',
     'color': '#1b7837', 'marker': 'o'},
    {'run': 'run_41', 'target': '3He', 'gas': '90/10',
     'color': '#1a4fa0', 'marker': '^'},
    {'run': 'run_44', 'target': '3He', 'gas': '95/5',
     'color': '#c0392b', 'marker': 's'},
]
RUN_STYLE = {r['run']: r for r in RUNS}

SUBRUN_MATCH = r'^flash_dr800_'
OUT_LABEL = 'run38_41_44_flash_compare'

E10_SPLIT = 600.0          # pulses bimodal ~410e10 (parasitic) / ~850e10 (dedicated)
MIN_MATCH_FRAC = 0.5       # pulse_match quality gate (same as run32/33 lib)
MIN_N_CLASS = 10           # need this many matched events in a class to plot a point
CLASSES = ('low', 'high')

# Time windows are flash-centred PER RUN (july_hv_scan.flash_scan_windows): the
# gamma-flash trigger latency drifts run-to-run, so each run's flash is measured
# from the clean reference detector (mx17_A) at top gain and the coarse windows
# are [pre-flash] [flash] [post-flash 1..3].  Rows in the grids are keyed by
# ROLE; the flash row (FLASH_ROLE_IDX) compares each run's own flash.  Filled by
# build_series().
RUN_WINDOWS: dict = {}
RUN_META: dict = {}


def windows_caption():
    bits = []
    for spec in RUNS:
        m = RUN_META.get(spec['run'])
        if m:
            bits.append(f"{spec['run']} {m['flash_lo']:.0f}-{m['flash_hi']:.0f} ns")
    return 'flash windows:  ' + ',  '.join(bits)

# Intensity styling for the combined (both-class) plots.
INT_STYLE = {
    'low':  dict(ls='--', alpha=0.6, fill=False, tag='low e10 (parasitic ~410)'),
    'high': dict(ls='-',  alpha=0.95, fill=True, tag='high e10 (dedicated ~850)'),
}
PLOT_KW = dict(lw=1.5, ms=4.8)

# Garfield gain-equivalence: map each run's resist HV to the 95/5 voltage of
# equal simulated gas gain, so 90/10 (run_38/41) and 95/5 (run_44) share an
# x-axis. n_TOF is at CERN -> CERN_450m pressure condition.
GAINMAP = GainMap(pressure='CERN_450m')
REF_LO, REF_HI = GAINMAP.ref_range          # 95/5 simulated span (400-490 V)

# Plotted twice: 'raw' = resist HV as-set; 'gaineq' = gain-matched 95/5 volts.
X_MODES = {
    'raw':    dict(suffix='',        xlabel='Resist HV [V]',
                   note='resist HV as set'),
    'gaineq': dict(suffix='_gaineq',
                   xlabel='Gain-equiv. 95/5 resist HV [V]',
                   note=(f'x = garfield gain-matched 95/5 volts (CERN); '
                         f'95/5-equiv <{REF_LO:g} V extrapolated')),
}


def run_label(run):
    r = RUN_STYLE[run]
    return f"{run} — {r['target']}, {r['gas']}"


def series_x(run, hv, x_mode):
    """x-values for one run's HV array under the chosen x-axis mode."""
    if x_mode == 'gaineq':
        return GAINMAP.to_ref_voltage(RUN_STYLE[run]['gas'], hv)
    return hv


# ---------------------------------------------------------------------------
def build_series():
    """
    Returns (series, det_names):
        series = [(run, cls, entry), ...]
        entry  = {det: {'hv':[n_pts], metric:[n_pts, n_window]}}
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
        print(f'  {run:7s} flash {meta["flash_lo"]:.0f}-{meta["flash_hi"]:.0f} ns  '
              f'windows={[(int(a), int(b)) for a, b in windows]}')

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

        print(f'  {run:7s} {spec["target"]:5s} {spec["gas"]:6s} -> '
              f'{n_used}/{len(subs)} subruns used '
              f'({n_matchfail} failed pulse-match quality gate)')

    return series, det_names


def _line_style(run, cls, single_class):
    """Assemble matplotlib kwargs for one (run, cls) line."""
    rs = RUN_STYLE[run]
    if single_class:
        # split plot: one line per run, solid + filled, label by run
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
def plot_grid(series, det_names, out_dir, metric, classes, fname, subtitle,
              x_mode='raw'):
    """rows = window role, cols = detector. One line per selected (run, cls)."""
    single = len(classes) == 1
    xm = X_MODES[x_mode]
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
                x = series_x(run, s['hv'], x_mode)
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
                ax.set_xlabel(xm['xlabel'], fontsize=8)
            if ci == 0:
                ax.set_ylabel(role, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(handles), 3), fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{label} vs {xm["xlabel"].split(" [")[0].lower()}, per flash-'
                 f'centred window — {subtitle}\n'
                 f'rows = window role (indep. y), shaded = post-flash recovery window,  '
                 f'amp >= {AMP_THRESHOLD} ADC  —  {xm["note"]}\n{windows_caption()}',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, out_dir, fname)


def plot_window_blowup(series, det_names, out_dir, role_idx, classes,
                       fname, subtitle, ylabel, x_mode='raw'):
    """One detector per panel, a single window role blown up."""
    single = len(classes) == 1
    xm = X_MODES[x_mode]
    wi = role_idx
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
            x = series_x(run, s['hv'], x_mode)
            y = s['hits_per_event'][:, wi]
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
        if x_mode == 'gaineq':               # shade the extrapolated 95/5 region
            ax.axvspan(ax.get_xlim()[0], REF_LO, color='0.85', alpha=0.35, zorder=0)
        ax.set_title(det, fontsize=11)
        ax.set_xlabel(xm['xlabel'])
        if ci == 0:
            ax.set_ylabel(ylabel)
    axes[-1].legend(fontsize=7.5, loc='upper left')
    fig.suptitle(subtitle + f'\ndrift 800 V,  amp >= {AMP_THRESHOLD} ADC  —  '
                 f'{xm["note"]}\n{windows_caption()}', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, fname)


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

    # Each block emitted twice: raw resist HV, and garfield gain-matched 95/5 V.
    for xmode, xm in X_MODES.items():
        sfx = xm['suffix']
        for metric in ('hits_per_event', 'mean_amplitude'):
            # combined (all runs x both intensities)
            plot_grid(series, det_names, out_dir, metric, list(CLASSES),
                      f'compare_timewin_grid_{metric}{sfx}.png',
                      '3 runs x low/high intensity', x_mode=xmode)
            # intensity-split: one figure per class (3 lines)
            for cls in CLASSES:
                plot_grid(series, det_names, out_dir, metric, [cls],
                          f'compare_timewin_grid_{metric}_{cls}{sfx}.png',
                          f'{cls} intensity only (3 runs)', x_mode=xmode)

        # Flash-window blowup: combined + split
        plot_window_blowup(series, det_names, out_dir, FLASH_ROLE_IDX, list(CLASSES),
                           f'flash_turnon_combined{sfx}.png',
                           'Flash turn-on/turn-off — 3 runs x low/high',
                           flash_ylabel, x_mode=xmode)
        for cls in CLASSES:
            plot_window_blowup(series, det_names, out_dir, FLASH_ROLE_IDX, [cls],
                               f'flash_turnon_{cls}{sfx}.png',
                               f'Flash turn-on/turn-off — '
                               f'{cls} intensity (3 runs)',
                               flash_ylabel, x_mode=xmode)

        # First post-flash (recovery) window blowup: combined + split
        plot_window_blowup(series, det_names, out_dir, POST1_ROLE_IDX, list(CLASSES),
                           f'postflash_combined{sfx}.png',
                           'First post-flash window — 3 runs x low/high',
                           post_ylabel, x_mode=xmode)
        for cls in CLASSES:
            plot_window_blowup(series, det_names, out_dir, POST1_ROLE_IDX, [cls],
                               f'postflash_{cls}{sfx}.png',
                               f'First post-flash window — '
                               f'{cls} intensity (3 runs)',
                               post_ylabel, x_mode=xmode)

    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
