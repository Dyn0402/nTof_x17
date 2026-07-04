#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/july_hv_scan.py

Quick-look analysis for the July beam HV scans.

Unlike the May runs, a July run is a **2-D scan**: each subrun is named
``scan_drift<DRIFT>_resistdrop<DROP>`` and sweeps

    * drift HV     — {100, 400, 700, 1000, 1200} V   (electron transparency)
    * resist drop  — {0, 10, ... 190} V below the per-detector base resist HV
                     (the amplification / gain knob; larger drop => lower gain)

For every (drift, resist-drop) point and each MX17 detector (A/B/C/D) it
computes three metrics and writes them to the web-browsable Analysis folder,
both as 2-D heatmaps (drift vs resist-drop) and as line plots (metric vs resist
HV, one coloured line per drift value):

    1. Hits / event            — mean hits (amp >= AMP_THRESHOLD) in the signal
                                 time window, per triggered event.
    2. Mean hit amplitude      — mean amplitude of those hits (gain proxy).
    3. Gamma-flash hits/event  — events are split into gamma-flash vs thermal
                                 (as in ntof_july_analysis/beam_hv_scan.py); the
                                 mean hits/event of flash events in the prompt
                                 flash window is reported.

Output PNGs -> {ANALYSIS_DIR}July_HV_Scan/{RUN}/  (the flask "Analysis" tab).

IMPORTANT: only the real per-subrun scan data is used.  Each subrun's
combined_hits_root also contains a copy of the shared pedestal acquisition
(``Mx17_pedestals_datrun_...``); those files are identical across every subrun
and are explicitly skipped here.  If a subrun contains only pedestal-derived
combined hits it is reported as "not processed yet" and skipped, so this script
never silently plots pedestal data as if it were beam data.
"""

import os
import re
import sys
import json
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import uproot
from common.Mx17StripMap import Detector, Mx17StripMap

# ---------------------------------------------------------------------------
# Configuration — tune per campaign
# ---------------------------------------------------------------------------
BASE_PATH = '/mnt/data/x17/beam_july/runs/'

# Where plots are written — the flask "Analysis" tab browses this directory
# ({BASE_DATA_DIR}analysis in flask_app/app.py), NOT runs/.
ANALYSIS_DIR = '/mnt/data/x17/beam_july/analysis/'

# Run to analyse.  None -> newest run_<N> directory found in BASE_PATH.
RUN: Optional[str] = None

MX17_DETECTORS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
MAP_CSV_PATH = f'{_ROOT}/mx17_m1_map.csv'

# Pedestal copies share this token in their filename and must never be treated
# as scan data (see get_pedestals()/processor_watcher in the DAQ repo).
PEDESTAL_NAME_TOKEN = '_pedestals_'

# Hit selection.
AMP_THRESHOLD = 400            # ADC counts
SIGNAL_T0_NS  = -500           # signal time window for hits/event + amplitude
SIGNAL_T1_NS  = 3000

# Gamma-flash classification (mirrors beam_hv_scan.py).
FLASH_DETECT_WINDOW_NS = 2000  # ns
GAMMA_STRIP_FRACTION   = 0.40  # fraction-of-strips threshold
FLASH_PROMPT_BIN_NS    = (0, 2000)  # window for the reported flash hits/event

# Coarse time-of-arrival windows (ns) for the per-window multi-panel plots.
# Edges match plot_hits_vs_time_hv_scan.py: [0, 800, 1150, 3500, 7000, 10000].
# Each (lo, hi) becomes one coloured line (hits/event, mean amp) vs resist HV.
TIME_WINDOW_EDGES_NS = [0, 800, 1150, 3500, 7000, 10000]
TIME_WINDOWS = list(zip(TIME_WINDOW_EDGES_NS[:-1], TIME_WINDOW_EDGES_NS[1:]))

# Subrun-name parser for the 2-D scan grid.
SUBRUN_RE = re.compile(r'scan_drift(\d+)_resistdrop(\d+)')

# Resist / drift HV channels are read from run_config.json (card, channel).


# ---------------------------------------------------------------------------
# Run / config discovery
# ---------------------------------------------------------------------------

def discover_run(base_path: str) -> Optional[str]:
    """Newest run_<N> directory under base_path (highest N with a config)."""
    runs = []
    for name in os.listdir(base_path):
        m = re.fullmatch(r'run_(\d+)', name)
        if m and os.path.isfile(os.path.join(base_path, name, 'run_config.json')):
            runs.append((int(m.group(1)), name))
    return max(runs)[1] if runs else None


def load_config(base_path: str, run: str) -> dict:
    with open(os.path.join(base_path, run, 'run_config.json')) as f:
        return json.load(f)


def build_detector_info(cfg: dict) -> Dict[str, dict]:
    """
    Return {det_name: {'feus': [...], 'resist': (card, ch), 'drift': (card, ch)}}
    for each requested MX17 detector, in MX17_DETECTORS order.
    """
    strip_map = Mx17StripMap(MAP_CSV_PATH)
    info: Dict[str, dict] = {}
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if name not in MX17_DETECTORS:
            continue
        det = Detector(name=name, det_cfg=det_cfg, strip_map=strip_map)
        hv_ch = det_cfg.get('hv_channels', {})
        resist = hv_ch.get('resist')
        drift = hv_ch.get('drift')
        info[name] = {
            'feus': sorted(det.feu_map.keys()),
            'resist': (int(resist[0]), int(resist[1])) if resist else None,
            'drift': (int(drift[0]), int(drift[1])) if drift else None,
        }
    return {name: info[name] for name in MX17_DETECTORS if name in info}


def parse_scan_subruns(base_path: str, run: str, cfg: dict,
                       det_info: Dict[str, dict]) -> List[dict]:
    """
    Build one record per 2-D-scan subrun that exists on disk:
        {run, subrun, drift, drop, resist_hv{det}, drift_hv{det}}
    resist_hv / drift_hv are read from the subrun's configured HV values.
    """
    run_dir = os.path.join(base_path, run)
    on_disk = {n for n in os.listdir(run_dir)
               if os.path.isdir(os.path.join(run_dir, n))}
    records = []
    for sub in cfg.get('sub_runs', []):
        name = sub['sub_run_name']
        m = SUBRUN_RE.fullmatch(name)
        if not m or name not in on_disk:
            continue
        drift, drop = int(m.group(1)), int(m.group(2))
        hvs = sub.get('hvs', {})
        resist_hv, drift_hv = {}, {}
        for det, di in det_info.items():
            if di['resist']:
                card, ch = di['resist']
                resist_hv[det] = hvs.get(str(card), {}).get(str(ch))
            if di['drift']:
                card, ch = di['drift']
                drift_hv[det] = hvs.get(str(card), {}).get(str(ch))
        records.append({
            'run': run, 'subrun': name, 'drift': drift, 'drop': drop,
            'resist_hv': resist_hv, 'drift_hv': drift_hv,
        })
    return records


# ---------------------------------------------------------------------------
# Data loading (real scan data only — pedestals excluded)
# ---------------------------------------------------------------------------

def _real_files(directory: str, suffix: str) -> List[str]:
    """Files in directory ending with suffix, excluding pedestal copies."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(suffix) and '_datrun_' in f and PEDESTAL_NAME_TOKEN not in f
    )


def load_hits(base_path: str, run: str, subrun: str,
              feu_ids: List[int]) -> Optional[pd.DataFrame]:
    """Load real (non-pedestal) combined hits for one subrun."""
    hits_dir = os.path.join(base_path, run, subrun, 'combined_hits_root')
    sources = _real_files(hits_dir, '.root')
    if not sources:
        return None
    # A file may be truncated / still being written (no 'hits' tree yet); skip it.
    good = []
    for s in sources:
        try:
            with uproot.open(s) as f:
                if 'hits' in f:
                    good.append(s)
        except Exception:
            continue
    if not good:
        return None
    df = uproot.concatenate([f'{s}:hits' for s in good],
                            ['eventId', 'feu', 'channel', 'time', 'amplitude'],
                            library='pd')
    return df[df['feu'].isin(feu_ids)].copy()


def get_total_events(base_path: str, run: str, subrun: str) -> Optional[int]:
    """Total triggered events = max eventId over real decoded ROOT files."""
    decoded_dir = os.path.join(base_path, run, subrun, 'decoded_root')
    files = _real_files(decoded_dir, '.root')
    max_id = 0
    for fname in files:
        try:
            with uproot.open(fname) as f:
                if 'nt' not in f:
                    continue
                ids = f['nt']['eventId'].array(library='np')
                if len(ids):
                    max_id = max(max_id, int(ids.max()))
        except Exception:
            continue
    return max_id if max_id > 0 else None


def compute_total_strips(df: pd.DataFrame, all_feus: List[int]) -> int:
    """Unique (feu, channel) pairs — denominator for the flash fraction."""
    d = df[df['feu'].isin(all_feus)]
    return int(d[['feu', 'channel']].drop_duplicates().shape[0])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classify_flash_events(df: pd.DataFrame, total_strips: int) -> np.ndarray:
    """Return the eventIds classified as gamma flashes."""
    amp = df[df['amplitude'] >= AMP_THRESHOLD]
    early = amp[amp['time'].between(0, FLASH_DETECT_WINDOW_NS)]
    n_early = (early[['eventId', 'feu', 'channel']]
               .drop_duplicates().groupby('eventId').size())
    frac = n_early.reindex(df['eventId'].unique(), fill_value=0) / max(total_strips, 1)
    return frac[frac >= GAMMA_STRIP_FRACTION].index.values


def detector_metrics(df: pd.DataFrame, feu_ids: List[int], n_total: int,
                     flash_ids: np.ndarray) -> Dict[str, float]:
    """Compute the three scan metrics for a single detector."""
    n_total = max(n_total, 1)
    sel = df[df['feu'].isin(feu_ids) & (df['amplitude'] >= AMP_THRESHOLD)]
    sig = sel[sel['time'].between(SIGNAL_T0_NS, SIGNAL_T1_NS)]

    hits_per_event = len(sig) / n_total
    mean_amp = float(sig['amplitude'].mean()) if not sig.empty else np.nan

    n_flash = len(flash_ids)
    if n_flash:
        t0, t1 = FLASH_PROMPT_BIN_NS
        flash_sig = sel[sel['eventId'].isin(flash_ids) & sel['time'].between(t0, t1)]
        flash_hpe = len(flash_sig) / n_flash
    else:
        flash_hpe = np.nan

    return {
        'hits_per_event': hits_per_event,
        'mean_amplitude': mean_amp,
        'flash_hits_per_event': flash_hpe,
    }


def detector_window_metrics(df: pd.DataFrame, feu_ids: List[int],
                            n_total: int) -> Dict[str, List[float]]:
    """hits/event and mean amplitude for this detector in each TIME_WINDOWS bin."""
    n_total = max(n_total, 1)
    sel = df[df['feu'].isin(feu_ids) & (df['amplitude'] >= AMP_THRESHOLD)]
    hpe, amp = [], []
    for lo, hi in TIME_WINDOWS:
        w = sel[sel['time'].between(lo, hi, inclusive='left')]
        hpe.append(len(w) / n_total)
        amp.append(float(w['amplitude'].mean()) if not w.empty else np.nan)
    return {'hits_per_event': hpe, 'mean_amplitude': amp}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS = {
    'hits_per_event':       ('Hits / event',            'viridis'),
    'mean_amplitude':       ('Mean hit amplitude [ADC]', 'magma'),
    'flash_hits_per_event': ('Gamma-flash hits / event', 'cividis'),
}


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heatmaps(grid: Dict[str, np.ndarray], drifts: List[int], drops: List[int],
                  det_names: List[str], resist_hv_by_det: Dict[str, List[Optional[int]]],
                  run: str, gas: str, out_dir: str) -> None:
    """One figure per metric: a 2-D (drift x resist-drop) heatmap per detector."""
    for key, (label, cmap) in METRICS.items():
        vals = [grid[key][d] for d in det_names]
        finite = np.concatenate([v[np.isfinite(v)].ravel() for v in vals]) \
            if any(np.isfinite(v).any() for v in vals) else np.array([0.0])
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

        n = len(det_names)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
        axes = axes[0]
        for ax, det in zip(axes, det_names):
            arr = grid[key][det]
            im = ax.imshow(arr, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                           extent=[-0.5, len(drops) - 0.5, -0.5, len(drifts) - 0.5])
            ax.set_xticks(range(len(drops)))
            ax.set_xticklabels(drops, rotation=90, fontsize=7)
            ax.set_yticks(range(len(drifts)))
            ax.set_yticklabels(drifts, fontsize=8)
            ax.set_xlabel('Resist drop [V]  (higher = lower gain)')
            if ax is axes[0]:
                ax.set_ylabel('Drift HV [V]')
            base = next((h for h in resist_hv_by_det[det] if h is not None), None)
            sub = f'{det}' + (f'  (base resist {base} V)' if base else '')
            ax.set_title(sub, fontsize=10)
        fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02, label=label)
        fig.subplots_adjust(top=0.80)
        gas_s = f'  —  {gas}' if gas else ''
        fig.suptitle(f'{label}   vs   drift x resist-drop  —  {run}{gas_s}\n'
                     f'amp >= {AMP_THRESHOLD} ADC,  '
                     f't in [{SIGNAL_T0_NS/1000:.1f}, {SIGNAL_T1_NS/1000:.1f}] us',
                     fontsize=11, y=0.99)
        _save(fig, out_dir, f'heatmap_{key}.png')


def plot_lines(grid: Dict[str, np.ndarray], drifts: List[int], drops: List[int],
               det_names: List[str], resist_hv_by_det: Dict[str, List[Optional[int]]],
               run: str, gas: str, out_dir: str) -> None:
    """One figure per metric: metric vs resist HV, one line per drift, panel per det."""
    cmap = plt.get_cmap('plasma')
    dnorm = Normalize(vmin=min(drifts), vmax=max(drifts))
    for key, (label, _cmap) in METRICS.items():
        n = len(det_names)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False, sharey=True)
        axes = axes[0]
        for ax, det in zip(axes, det_names):
            arr = grid[key][det]                      # [n_drift, n_drop]
            x = np.array([h if h is not None else np.nan
                          for h in resist_hv_by_det[det]], dtype=float)
            order = np.argsort(x)
            for di, drift in enumerate(drifts):
                y = arr[di]
                good = np.isfinite(x) & np.isfinite(y)
                if not good.any():
                    continue
                col = cmap(dnorm(drift))
                ax.plot(x[order], y[order], '-o', ms=4, lw=1.4, color=col,
                        label=f'{drift} V')
            ax.set_xlabel('Resist HV [V]')
            if ax is axes[0]:
                ax.set_ylabel(label)
            ax.set_title(det, fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[-1].legend(title='Drift HV', fontsize=7, title_fontsize=8)
        gas_s = f'  —  {gas}' if gas else ''
        fig.suptitle(f'{label}   vs   resist HV  —  {run}{gas_s}\n'
                     f'amp >= {AMP_THRESHOLD} ADC,  '
                     f't in [{SIGNAL_T0_NS/1000:.1f}, {SIGNAL_T1_NS/1000:.1f}] us',
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, out_dir, f'lines_{key}.png')


WINDOW_METRICS = {
    'hits_per_event': 'Hits / event',
    'mean_amplitude': 'Mean hit amplitude [ADC]',
}


def plot_time_windows(wgrid: Dict[str, Dict[str, np.ndarray]], drifts: List[int],
                      drops: List[int], det_names: List[str],
                      resist_hv_by_det: Dict[str, List[Optional[int]]],
                      run: str, gas: str, out_dir: str) -> None:
    """
    Per-metric multi-panel figure: metric vs resist HV with one coloured line
    per coarse time window (TIME_WINDOWS).  Grid = drift rows x detector cols,
    so single-drift runs give a 1 x n_det strip.  wgrid[metric][det] has shape
    [n_drift, n_drop, n_window].
    """
    cmap = plt.get_cmap('viridis')
    nw = len(TIME_WINDOWS)
    wlabels = [f'{lo/1000:g}-{hi/1000:g} us' for lo, hi in TIME_WINDOWS]
    for key, label in WINDOW_METRICS.items():
        nrow, ncol = len(drifts), len(det_names)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                                 squeeze=False, sharex=True, sharey='row')
        for di, drift in enumerate(drifts):
            for ci, det in enumerate(det_names):
                ax = axes[di][ci]
                cube = wgrid[key][det]                 # [n_drift, n_drop, n_window]
                x = np.array([h if h is not None else np.nan
                              for h in resist_hv_by_det[det]], dtype=float)
                order = np.argsort(x)
                for wi in range(nw):
                    y = cube[di, :, wi]
                    good = np.isfinite(x) & np.isfinite(y)
                    if not good.any():
                        continue
                    col = cmap(wi / max(nw - 1, 1))
                    ax.plot(x[order], y[order], '-o', ms=3.5, lw=1.3, color=col,
                            label=wlabels[wi])
                ax.grid(True, alpha=0.3)
                if di == 0:
                    ax.set_title(det, fontsize=10)
                if di == nrow - 1:
                    ax.set_xlabel('Resist HV [V]')
                if ci == 0:
                    ylab = f'{label}\n(drift {drift} V)' if nrow > 1 else label
                    ax.set_ylabel(ylab)
        axes[0][-1].legend(title='Time window', fontsize=7, title_fontsize=8)
        gas_s = f'  —  {gas}' if gas else ''
        fig.suptitle(f'{label}   vs   resist HV, per time window  —  {run}{gas_s}\n'
                     f'amp >= {AMP_THRESHOLD} ADC', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93 if nrow == 1 else 0.96))
        _save(fig, out_dir, f'timewin_{key}.png')


def plot_time_windows_grid(wgrid: Dict[str, Dict[str, np.ndarray]], drifts: List[int],
                           drops: List[int], det_names: List[str],
                           resist_hv_by_det: Dict[str, List[Optional[int]]],
                           run: str, gas: str, out_dir: str) -> None:
    """
    Per-metric grid with **time windows stacked vertically**: rows = time
    windows, cols = detectors.  Every panel autoscales its own y-axis (windows
    span very different absolute levels) so the eye reads the *relative*
    variation with resist HV.  Multiple drifts are overlaid as coloured lines
    within each panel.
    """
    dcmap = plt.get_cmap('plasma')
    dnorm = Normalize(vmin=min(drifts), vmax=max(drifts))
    nw = len(TIME_WINDOWS)
    for key, label in WINDOW_METRICS.items():
        nrow, ncol = nw, len(det_names)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.2 * nrow),
                                 squeeze=False, sharex=True)  # independent y per panel
        for wi, (lo, hi) in enumerate(TIME_WINDOWS):
            for ci, det in enumerate(det_names):
                ax = axes[wi][ci]
                cube = wgrid[key][det]                 # [n_drift, n_drop, n_window]
                x = np.array([h if h is not None else np.nan
                              for h in resist_hv_by_det[det]], dtype=float)
                order = np.argsort(x)
                for di, drift in enumerate(drifts):
                    y = cube[di, :, wi]
                    good = np.isfinite(x) & np.isfinite(y)
                    if not good.any():
                        continue
                    col = dcmap(dnorm(drift)) if len(drifts) > 1 else 'C0'
                    ax.plot(x[order], y[order], '-o', ms=3, lw=1.2, color=col,
                            label=f'{drift} V')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
                if wi == 0:
                    ax.set_title(det, fontsize=10)
                if wi == nrow - 1:
                    ax.set_xlabel('Resist HV [V]', fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f'{lo/1000:g}-{hi/1000:g} us', fontsize=8)
        if len(drifts) > 1:
            axes[0][-1].legend(title='Drift HV', fontsize=6, title_fontsize=7)
        gas_s = f'  —  {gas}' if gas else ''
        fig.suptitle(f'{label}   vs   resist HV  (rows = time window, indep. y)  '
                     f'—  {run}{gas_s}\namp >= {AMP_THRESHOLD} ADC', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, out_dir, f'timewin_grid_{key}.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Optional CLI: `july_hv_scan.py [run]` where run is e.g. "run_6" or "6".
    run = RUN
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        run = arg if arg.startswith('run_') else f'run_{arg}'
    run = run or discover_run(BASE_PATH)
    if run is None:
        print(f'ERROR: no run_<N> with a config found under {BASE_PATH}')
        return
    cfg = load_config(BASE_PATH, run)
    gas = cfg.get('gas', '')
    det_info = build_detector_info(cfg)
    if not det_info:
        print(f'ERROR: none of {MX17_DETECTORS} found in {run} config')
        return
    det_names = list(det_info.keys())
    all_feus = sorted({f for di in det_info.values() for f in di['feus']})

    records = parse_scan_subruns(BASE_PATH, run, cfg, det_info)
    if not records:
        print(f'ERROR: no scan_drift*_resistdrop* subruns on disk for {run}')
        return

    drifts = sorted({r['drift'] for r in records})
    drops = sorted({r['drop'] for r in records})
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', run)

    print(f'Run {run}  ({gas})')
    print(f'Detectors: {det_names}   FEUs: {all_feus}')
    print(f'Grid: {len(drifts)} drift x {len(drops)} resist-drop  '
          f'({len(records)} subruns on disk)')
    print(f'Output -> {out_dir}\n')

    # resist HV per detector along the drop axis (from any record per drop).
    resist_hv_by_det = {det: [None] * len(drops) for det in det_names}
    for r in records:
        j = drops.index(r['drop'])
        for det in det_names:
            if resist_hv_by_det[det][j] is None:
                resist_hv_by_det[det][j] = r['resist_hv'].get(det)

    # Result grids: metric -> det -> [n_drift, n_drop].
    grid = {key: {det: np.full((len(drifts), len(drops)), np.nan)
                  for det in det_names} for key in METRICS}
    # Per-time-window cubes: metric -> det -> [n_drift, n_drop, n_window].
    wgrid = {key: {det: np.full((len(drifts), len(drops), len(TIME_WINDOWS)), np.nan)
                   for det in det_names} for key in WINDOW_METRICS}

    total_strips: Optional[int] = None
    n_done, n_skip = 0, 0
    for rec in records:
        sub = rec['subrun']
        df = load_hits(BASE_PATH, run, sub, all_feus)
        if df is None or df.empty:
            print(f'  [skip] {sub}: no real scan combined_hits yet '
                  f'(pedestal-only or unprocessed)')
            n_skip += 1
            continue
        if total_strips is None:
            total_strips = compute_total_strips(df, all_feus)
            print(f'  total strips (flash denom): {total_strips}\n')

        n_total = get_total_events(BASE_PATH, run, sub) or df['eventId'].nunique()
        flash_ids = classify_flash_events(df, total_strips)

        i, j = drifts.index(rec['drift']), drops.index(rec['drop'])
        for det in det_names:
            feus = det_info[det]['feus']
            m = detector_metrics(df, feus, n_total, flash_ids)
            for key in METRICS:
                grid[key][det][i, j] = m[key]
            wm = detector_window_metrics(df, feus, n_total)
            for key in WINDOW_METRICS:
                wgrid[key][det][i, j, :] = wm[key]
        n_done += 1
        print(f'  {sub}: drift={rec["drift"]} drop={rec["drop"]} '
              f'nev={n_total} nflash={len(flash_ids)}')

    print(f'\nProcessed {n_done} subruns, skipped {n_skip}.')
    if n_done == 0:
        print('No real scan data found — nothing plotted. '
              '(Has the processor reprocessed the scan datrun files?)')
        return

    plot_heatmaps(grid, drifts, drops, det_names, resist_hv_by_det, run, gas, out_dir)
    plot_lines(grid, drifts, drops, det_names, resist_hv_by_det, run, gas, out_dir)
    plot_time_windows(wgrid, drifts, drops, det_names, resist_hv_by_det, run, gas, out_dir)
    plot_time_windows_grid(wgrid, drifts, drops, det_names, resist_hv_by_det, run, gas, out_dir)
    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
