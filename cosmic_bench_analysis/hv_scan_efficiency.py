#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hv_scan_efficiency.py

Batch efficiency analysis for a HV (high-voltage) scan on a single run.

For each subrun the script:
  1. Loads detector hits and M3 reference tracks.
  2. Runs the standard micro-TPC event analysis pipeline.
  3. Uses a fixed z position and zero in-plane rotation (no scan) so that
     low-statistics subruns can still be processed.
  4. Runs translation alignment only so that the detector local frame is
     centred on the M3 reference frame and radial residuals are meaningful.
  5. Computes a 2-D efficiency map with wide bins and a single
     surface-averaged efficiency value (integrated hits / integrated tracks).
  6. Plots efficiency vs. HV.
"""

import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from cosmic_micro_tpc_analysis import (
    AlignmentParams,
    EventResult,
    analyse_event,
    translation_alignment,
    attach_reference_positions,
    plot_radial_residuals,
    _map_strip_positions,
    _progress,
    _build_2d_map_arrays,
    load_alignment,
)
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = '/media/dylan/data/x17/cosmic_bench/det_1/'
RUN       = 'mx17_det0_He_HV_Scan_4-1-26'
MX17_FEUS = [3, 4]   # active FEUs for this run

RUN_CONFIG_PATH = f'{BASE_PATH}{RUN}/run_config.json'
MAP_CSV_PATH    = f'{_ROOT}/mx17_m4_map.csv'

CSV_OUT_DIR = f'{BASE_PATH}Analysis/HV_Scan/'

# Alignment file produced by cosmic_micro_tpc_analysis.py.
# If this file exists it is used to seed z, theta, and centre for every subrun
# (translation is still re-run per subrun).  Set to None to skip.
ALIGNMENT_FILE = f'{BASE_PATH}Alignment/alignment.json'

# Fallback fixed alignment parameters used when no alignment file is found.
# Only translation is run to centre the detector on the reference frame.
FIXED_Z         = 728.0   # mm
FIXED_THETA_DEG = -0.625     # degrees
DET_CENTRE_X    = 200.0   # mm (rotation pivot, detector local frame)
DET_CENTRE_Y    = 200.0   # mm

M3_CHI2_CUT = 20

# Efficiency map parameters
EFF_BINS           = 10    # bins per axis (wide bins for low-stats subruns)
MIN_TRACKS_PER_BIN = 2     # mask bins with fewer reference tracks
RADIUS_CUT_MM      = 10.0  # radial residual cut for hit-matching


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_hv(subrun_name: str) -> Optional[int]:
    """Extract resistor HV value in volts from a name like 'resist_450V_drift_800V'."""
    m = re.search(r'resist_(\d+)V', subrun_name)
    return int(m.group(1)) if m else None


def find_subruns(base_path: str, run: str) -> List[Tuple[str, int]]:
    """
    Scan the run directory and return (subrun_name, hv_volts) pairs for all
    folders matching the 'resist_<NNN>V' pattern, sorted by HV descending.
    """
    run_dir = os.path.join(base_path, run)
    pairs = []
    for name in sorted(os.listdir(run_dir)):
        if not os.path.isdir(os.path.join(run_dir, name)):
            continue
        hv = extract_hv(name)
        if hv is not None and name.startswith('resist_'):
            pairs.append((name, hv))
    return sorted(pairs, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Active region helpers
# ---------------------------------------------------------------------------

def get_active_det_bounds(det, strip_map_csv: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Derive the bounding box of the plugged-in strips in detector local coordinates.

    Queries every (feu_id, feu_connector) entry in the detector config,
    looks up all strip positions in the CSV, and returns
        (x_min, x_max), (y_min, y_max)
    in detector local mm.

    For MX17:
      - X-axis strips run along X → they localise the hit in Y (y_position_mm)
      - Y-axis strips run along Y → they localise the hit in X (x_position_mm)
    """
    from common.Mx17StripMap import Mx17StripMap
    sm = Mx17StripMap(strip_map_csv)

    x_positions: List[float] = []   # from Y-axis strips
    y_positions: List[float] = []   # from X-axis strips

    for det_key, (feu_id, feu_connector) in det.dream_feus.items():
        axis = det_key[0]  # 'x' or 'y'
        for local_ch in range(sm.CHANNELS_PER_CONNECTOR):
            pos = sm.lookup(axis, feu_connector, local_ch)
            if pos is None:
                continue
            px, py = pos
            if axis == 'x':
                y_positions.append(py)
            else:
                x_positions.append(px)

    x_min = float(np.min(x_positions)) if x_positions else 0.0
    x_max = float(np.max(x_positions)) if x_positions else 0.0
    y_min = float(np.min(y_positions)) if y_positions else 0.0
    y_max = float(np.max(y_positions)) if y_positions else 0.0

    return (x_min, x_max), (y_min, y_max)


# ---------------------------------------------------------------------------
# Per-subrun pipeline
# ---------------------------------------------------------------------------

def analyse_subrun(
    subrun: str,
    det,
    fixed_params: AlignmentParams,
    det_active_x: Optional[Tuple[float, float]] = None,
    det_active_y: Optional[Tuple[float, float]] = None,
) -> Optional[dict]:
    """
    Run the full pipeline for one subrun.

    det_active_x / det_active_y are the active strip bounds in detector local
    coordinates.  After translation alignment they are converted to reference-
    frame coordinates and passed to _compute_efficiency to restrict the
    efficiency calculation to the plugged-in region.

    Returns a summary dict on success, or None if the subrun should be skipped.
    """
    hits_dir = f'{BASE_PATH}{RUN}/{subrun}/combined_hits_root/'
    rays_dir = f'{BASE_PATH}{RUN}/{subrun}/m3_tracking_root/'

    if not os.path.isdir(hits_dir) or not os.path.isdir(rays_dir):
        print(f'  [SKIP] {subrun}: missing hits or tracking directory')
        return None

    # ---- Load hits ----
    hit_files = [f for f in os.listdir(hits_dir)
                 if f.endswith('.root') and '_datrun_' in f]
    if not hit_files:
        print(f'  [SKIP] {subrun}: no hit files found')
        return None

    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(MX17_FEUS)].copy()
    df = _map_strip_positions(df, det)
    print(f'  Loaded {len(df)} hits over {df["eventId"].nunique()} events')

    # ---- Load M3 reference tracks ----
    rays = M3RefTracking(rays_dir, chi2_cut=M3_CHI2_CUT)
    x_ref_angles, y_ref_angles, angle_event_nums = get_xy_angles(rays.ray_data)
    x_ref_angles = -np.array(x_ref_angles)

    # ---- Event analysis ----
    results: List[EventResult] = []
    event_ids = df['eventId'].unique()
    for event_id in _progress(event_ids, desc='  Analysing events'):
        df_event = df[df['eventId'] == event_id]
        results.append(analyse_event(df_event, event_id=event_id, plot=False))

    n_valid = sum(r.has_both for r in results)
    print(f'  {n_valid}/{len(results)} events with valid X+Y hits')
    if n_valid < 20:
        print(f'  [SKIP] {subrun}: too few valid events ({n_valid})')
        return None

    # ---- Translation alignment only (z and rotation fixed) ----
    params = translation_alignment(results, rays, fixed_params)

    # ---- Convert active region from detector local → reference frame ----
    # With theta=0, ref = det + offset.
    ref_active_x = ref_active_y = None
    if det_active_x is not None:
        ref_active_x = (det_active_x[0] + params.x_offset,
                        det_active_x[1] + params.x_offset)
    if det_active_y is not None:
        ref_active_y = (det_active_y[0] + params.y_offset,
                        det_active_y[1] + params.y_offset)
    if ref_active_x is not None and ref_active_y is not None:
        print(f'  Active region (ref frame): '
              f'X [{ref_active_x[0]:.1f}, {ref_active_x[1]:.1f}]  '
              f'Y [{ref_active_y[0]:.1f}, {ref_active_y[1]:.1f}]')

    # ---- Attach reference positions ----
    attach_reference_positions(results, rays, params,
                               x_ref_angles, angle_event_nums)

    # ---- Efficiency ----
    return _compute_efficiency(results, rays, params, subrun,
                               active_x=ref_active_x, active_y=ref_active_y)


def _compute_efficiency(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    subrun: str,
    active_x: Optional[Tuple[float, float]] = None,
    active_y: Optional[Tuple[float, float]] = None,
) -> dict:
    """
    Build a wide-bin 2-D efficiency map and compute the surface-averaged
    efficiency as: sum(hits in active bins) / sum(tracks in active bins).

    If active_x / active_y are provided (in reference-frame coordinates),
    reference tracks outside those bounds are excluded from both the
    denominator and numerator before histogramming.

    This integrated estimator is more robust than averaging per-bin
    efficiencies when statistics are low.
    """
    ref_x_all, ref_y_all, ref_x_hit, ref_y_hit, x_edges, y_edges = \
        _build_2d_map_arrays(results, rays, params, EFF_BINS, RADIUS_CUT_MM)

    # ---- Restrict to active detector region ----
    if active_x is not None:
        mask_all = (ref_x_all >= active_x[0]) & (ref_x_all <= active_x[1])
        mask_hit = (ref_x_hit >= active_x[0]) & (ref_x_hit <= active_x[1])
        ref_x_all, ref_y_all = ref_x_all[mask_all], ref_y_all[mask_all]
        ref_x_hit, ref_y_hit = ref_x_hit[mask_hit], ref_y_hit[mask_hit]
    if active_y is not None:
        mask_all = (ref_y_all >= active_y[0]) & (ref_y_all <= active_y[1])
        mask_hit = (ref_y_hit >= active_y[0]) & (ref_y_hit <= active_y[1])
        ref_x_all, ref_y_all = ref_x_all[mask_all], ref_y_all[mask_all]
        ref_x_hit, ref_y_hit = ref_x_hit[mask_hit], ref_y_hit[mask_hit]

    # Recompute bin edges to span only the active region
    if active_x is not None or active_y is not None:
        all_x = np.concatenate([ref_x_all, ref_x_hit]) if len(ref_x_hit) else ref_x_all
        all_y = np.concatenate([ref_y_all, ref_y_hit]) if len(ref_y_hit) else ref_y_all
        x_edges = np.linspace(np.nanmin(all_x), np.nanmax(all_x), EFF_BINS + 1)
        y_edges = np.linspace(np.nanmin(all_y), np.nanmax(all_y), EFF_BINS + 1)

    total, _, _ = np.histogram2d(ref_x_all, ref_y_all, bins=[x_edges, y_edges])
    hits,  _, _ = np.histogram2d(ref_x_hit, ref_y_hit, bins=[x_edges, y_edges])

    in_mask = total >= MIN_TRACKS_PER_BIN
    n_active_bins = int(in_mask.sum())

    if n_active_bins == 0:
        return {'efficiency': np.nan, 'eff_err': np.nan,
                'n_tracks': 0, 'n_hits': 0, 'n_bins': 0,
                'subrun': subrun, 'params': params}

    n_tracks = int(total[in_mask].sum())
    n_hits   = int(hits[in_mask].sum())
    eff      = n_hits / n_tracks if n_tracks > 0 else np.nan
    eff_err  = float(np.sqrt(eff * (1 - eff) / n_tracks)) if n_tracks > 0 else np.nan

    return {
        'efficiency':  eff,
        'eff_err':     eff_err,
        'n_tracks':    n_tracks,
        'n_hits':      n_hits,
        'n_bins':      n_active_bins,
        'total_map':   total,
        'hits_map':    hits,
        'x_edges':     x_edges,
        'y_edges':     y_edges,
        'ref_x_all':   ref_x_all,
        'ref_y_all':   ref_y_all,
        'params':      params,
        'subrun':      subrun,
        'results':     results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_efficiency_map(result: dict, hv: int) -> None:
    """Plot 2-D efficiency map and reference track density for one subrun."""
    total   = result['total_map']
    hits    = result['hits_map']
    x_edges = result['x_edges']
    y_edges = result['y_edges']
    eff_mean = result['efficiency']

    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.where(total >= MIN_TRACKS_PER_BIN,
                              hits / total, np.nan)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    cmap_eff = plt.get_cmap('viridis').copy()
    cmap_eff.set_bad(color='lightgrey')
    cmap_den = plt.get_cmap('plasma').copy()
    cmap_den.set_bad(color='lightgrey')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im = axes[0].imshow(efficiency.T, origin='lower', aspect='auto',
                        extent=extent, cmap=cmap_eff, vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0], label='Efficiency')
    axes[0].set_xlabel('Reference X [mm]')
    axes[0].set_ylabel('Reference Y [mm]')
    axes[0].set_title(
        f'HV = {hv} V  —  mean eff = {eff_mean:.3f}\n'
        f'(r < {RADIUS_CUT_MM:.0f} mm, grey = < {MIN_TRACKS_PER_BIN} tracks)'
    )

    density = np.where(total >= MIN_TRACKS_PER_BIN, total, np.nan)
    im2 = axes[1].imshow(density.T, origin='lower', aspect='auto',
                         extent=extent, cmap=cmap_den)
    plt.colorbar(im2, ax=axes[1], label='Reference tracks per bin')
    axes[1].set_xlabel('Reference X [mm]')
    axes[1].set_ylabel('Reference Y [mm]')
    axes[1].set_title(
        f'Reference track density\n({len(result["ref_x_all"]):,} total tracks)'
    )

    fig.suptitle(result['subrun'], fontsize=11)
    fig.tight_layout()


def plot_efficiency_vs_hv(
    hv_values: List[int],
    efficiencies: List[float],
    eff_errors: List[float],
) -> None:
    """Plot surface-averaged efficiency vs. HV with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(hv_values, efficiencies, yerr=eff_errors,
                fmt='o-', color='steelblue', capsize=5, lw=2, ms=8)
    ax.set_xlabel('HV [V]')
    ax.set_ylabel('Surface-averaged Efficiency')
    ax.set_title(
        f'Detector Efficiency vs. HV  —  {RUN}\n'
        f'(r < {RADIUS_CUT_MM:.0f} mm cut, integrated over active bins)'
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_efficiency_map_csv(result: dict, hv: int, out_dir: str) -> None:
    """Save the per-bin efficiency map to a CSV for later comparison."""
    total   = result['total_map']
    hits    = result['hits_map']
    x_edges = result['x_edges']
    y_edges = result['y_edges']

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centres, y_centres, indexing='ij')

    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.where(total >= MIN_TRACKS_PER_BIN,
                              hits / total, np.nan)

    df = pd.DataFrame({
        'ref_x_mm':      xx.ravel(),
        'ref_y_mm':      yy.ravel(),
        'total_tracks':  total.ravel().astype(int),
        'hits':          hits.ravel().astype(int),
        'efficiency':    efficiency.ravel(),
        'hv_v':          hv,
        'radius_cut_mm': RADIUS_CUT_MM,
    })

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'efficiency_{hv}V.csv')
    df.to_csv(path, index=False)
    print(f'  Saved efficiency map → {path}')


def save_summary_csv(
    hv_values: List[int],
    efficiencies: List[float],
    eff_errors: List[float],
    n_tracks_list: List[int],
    n_hits_list: List[int],
    out_dir: str,
) -> None:
    """Save the per-HV efficiency summary to a single CSV."""
    df = pd.DataFrame({
        'hv_v':       hv_values,
        'efficiency': efficiencies,
        'eff_err':    eff_errors,
        'n_tracks':   n_tracks_list,
        'n_hits':     n_hits_list,
    })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'efficiency_vs_hv.csv')
    df.to_csv(path, index=False)
    print(f'\nSummary saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Detector config ----
    rc  = RunConfig(RUN_CONFIG_PATH, MAP_CSV_PATH)
    det = rc.get_detector('mx17_1')

    # ---- Active region in detector local coordinates ----
    (det_x_min, det_x_max), (det_y_min, det_y_max) = \
        get_active_det_bounds(det, MAP_CSV_PATH)
    print(f'Active detector region (local coords):')
    print(f'  X: [{det_x_min:.2f}, {det_x_max:.2f}] mm')
    print(f'  Y: [{det_y_min:.2f}, {det_y_max:.2f}] mm')

    # Seed alignment from file if available; otherwise fall back to hard-coded values.
    # x_offset / y_offset are always reset to 0 because translation_alignment()
    # re-derives them fresh for every subrun.
    if ALIGNMENT_FILE and os.path.exists(ALIGNMENT_FILE):
        _loaded = load_alignment(ALIGNMENT_FILE)
        fixed_params = AlignmentParams(
            z_x=_loaded.z_x,
            z_y=_loaded.z_y,
            theta_deg=_loaded.theta_deg,
            centre_x=_loaded.centre_x,
            centre_y=_loaded.centre_y,
        )
        print(f'Using alignment from file: {fixed_params}')
    else:
        if ALIGNMENT_FILE:
            print(f'Alignment file not found ({ALIGNMENT_FILE}); using hard-coded defaults.')
        fixed_params = AlignmentParams(
            z_x=FIXED_Z, z_y=FIXED_Z,
            theta_deg=FIXED_THETA_DEG,
            centre_x=DET_CENTRE_X,
            centre_y=DET_CENTRE_Y,
        )

    # ---- Discover subruns ----
    subruns = find_subruns(BASE_PATH, RUN)
    if not subruns:
        print(f'No HV-scan subruns found in {BASE_PATH}{RUN}/')
        return
    print(f'Found {len(subruns)} subruns:')
    for name, hv in subruns:
        print(f'  {name}  ({hv} V)')

    # ---- Process each subrun ----
    hv_values, efficiencies, eff_errors = [], [], []
    n_tracks_list, n_hits_list = [], []

    for subrun, hv in subruns:
        print(f'\n{"="*60}')
        print(f'Subrun: {subrun}  (HV = {hv} V)')
        print(f'{"="*60}')

        result = analyse_subrun(subrun, det, fixed_params,
                                det_active_x=(det_x_min, det_x_max),
                                det_active_y=(det_y_min, det_y_max))
        if result is None:
            continue

        eff = result['efficiency']
        err = result['eff_err']
        print(f'  Efficiency = {eff:.4f} ± {err:.4f}  '
              f'({result["n_hits"]}/{result["n_tracks"]} hits/tracks, '
              f'{result["n_bins"]} active bins)')

        # Radial residual diagnostic
        plot_radial_residuals(result['results'], radius_cut_mm=RADIUS_CUT_MM)

        hv_values.append(hv)
        efficiencies.append(eff)
        eff_errors.append(err)
        n_tracks_list.append(result['n_tracks'])
        n_hits_list.append(result['n_hits'])

        plot_efficiency_map(result, hv)
        save_efficiency_map_csv(result, hv, CSV_OUT_DIR)

    if not hv_values:
        print('\nNo valid subruns processed.')
        return

    # ---- Summary table ----
    print(f'\n{"HV [V]":>8}  {"Efficiency":>12}  {"± Err":>8}  '
          f'{"Tracks":>8}  {"Hits":>8}')
    for hv, eff, err, nt, nh in zip(hv_values, efficiencies, eff_errors,
                                     n_tracks_list, n_hits_list):
        print(f'{hv:>8}  {eff:>12.4f}  {err:>8.4f}  {nt:>8}  {nh:>8}')

    plot_efficiency_vs_hv(hv_values, efficiencies, eff_errors)
    save_summary_csv(hv_values, efficiencies, eff_errors,
                     n_tracks_list, n_hits_list, CSV_OUT_DIR)

    plt.show()


if __name__ == '__main__':
    main()
