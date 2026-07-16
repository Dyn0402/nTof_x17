#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_alignment_and_tpc.py

Step 3 of the June cosmic-bench QA: micro-TPC event analysis, geometric
alignment to the M3 reference tracker, and (optionally) the full physics
analysis.  Reuses cosmic_bench_analysis/cosmic_micro_tpc_analysis.py wholesale
as a namespace; this script only wires it to the June run and fixes the
run-specific geometry.

KEY RUN-SPECIFIC FIX
--------------------
The upstream main() is hard-coded for a detector near z~250 mm (the det4 bench
position).  THIS detector (mx17_1) sits at det_center z = 702 mm, between the
M3 stations (z = 24..1302 mm).  The z-alignment scan is therefore retargeted to
bracket ~702 mm; using the upstream 235-270 mm range would make alignment fail.

Usage
-----
    python 03_alignment_and_tpc.py          # checkpoint: analysis + alignment + quality plots
    python 03_alignment_and_tpc.py --full   # also efficiency/resolution maps + event display
    python 03_alignment_and_tpc.py --refit   # ignore cached per-event results, recompute

Products (output/<run>/alignment_tpc/):
  z_alignment_scan_iter_*.png, rotation_alignment_scan_iter_*.png,
  residuals.png, position_correlation.png, radial_residuals.png,
  ref_angle_distributions.png, alignment.json
  (+ efficiency / resolution / angle-correlation / 3-D display with --full)
"""

import os
import sys
import pickle
import concurrent.futures

import matplotlib
matplotlib.use('Agg')

import numpy as np

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles

# --- run-specific geometry (from the run registry) ---
DET_PLANE_Z = CFG.DET_PLANE_Z        # mx17_1 det_center z from run_config.json
# Fine iterative z and theta scans (defaults tuned for the Saclay cosmic bench).
# Z-scan window defaults to DET_PLANE_Z +/- 60 mm but can be overridden via the
# Z_LO / Z_HI environment variables (absolute mm) when the optimum rails the edge.
_z_lo = float(os.environ.get('Z_LO', DET_PLANE_Z - 60.0))
_z_hi = float(os.environ.get('Z_HI', DET_PLANE_Z + 60.0))
Z_SCAN = np.linspace(_z_lo, _z_hi, int(round(_z_hi - _z_lo)) + 1)  # 1 mm steps
CENTRE_XY = 200.0                    # strip-map centre (active area ~0..400 mm)
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
N_ITER = 3                           # iterative z -> rotation -> translation cycles

# ---------------------------------------------------------------------------
# Default alignment recipe for the SACLAY cosmic bench (override via flags):
#   * spark veto  : drop events with > VETO raw hits (full-detector discharges)
#   * cluster cut : determine the alignment only on events with <= MAXDROP strips
#                   sitting in competing clusters (n_dropped) — these otherwise
#                   bias the z scan
#   * base rotation: read from run_config det_orientation.z (the detector's
#                   in-plane mounting rotation vs the M3 reference, ~90 deg for the
#                   mx17 detectors).  The fine theta scan (+/-2 deg) absorbs the
#                   small residual misalignment on top of this.  Override with
#                   --rot0=<deg> if the config field is missing/wrong.
#   * clean handedness convention (REF_X_SIGN=+1): the raw M3 reference X is used
#                   directly (no x_ref negation) and the detector is NOT y-flipped;
#                   the full detector->M3 transform is carried by the base rotation.
#                   This replaces the old rot0=90 + --flipy + hardcoded x_ref=-x_ref
#                   recipe (two reflections cancelling) with a single rotation.
# ---------------------------------------------------------------------------
FULL = '--full' in sys.argv
REFIT = '--refit' in sys.argv
VETO = (None if '--no-veto' in sys.argv
        else next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50))
MAXDROP = (None if '--no-clustercut' in sys.argv
           else next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--maxdrop=')), 2))
# Base rotation: None -> take from run_config det_orientation.z; else --rot0 override.
ROT0_OVERRIDE = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--rot0=')), None)
REF_X_SIGN = +1.0   # clean convention: raw M3 reference X, rotation carries the frame


def _load_hits():
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    import uproot
    hit_files = [f for f in os.listdir(CFG.combined_hits_dir)
                 if f.endswith('.root') and '_datrun_' in f]
    sources = [f'{CFG.combined_hits_dir}{f}:hits' for f in hit_files]
    df = uproot.concatenate(sources, library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    if VETO is not None:
        hits_per_ev = df.groupby('eventId')['channel'].transform('size')
        n_before = df['eventId'].nunique()
        df = df[hits_per_ev <= VETO].copy()
        n_after = df['eventId'].nunique()
        print(f'Spark veto (>{VETO} hits): kept {n_after:,}/{n_before:,} events '
              f'({100*n_after/n_before:.1f}%)')
    df = cm._map_strip_positions(df, det)
    # No y-flip: the clean convention carries the detector->M3 frame entirely in
    # the base rotation (det_orientation.z) with raw M3 reference (REF_X_SIGN=+1).
    print(f'Loaded {len(df):,} hits over {df["eventId"].nunique():,} events.')
    return df, det, rc


def _analyse_events(df, cache_path):
    if os.path.exists(cache_path) and not REFIT:
        print(f'Loading cached per-event results from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    event_ids = df['eventId'].unique()
    grouped = df.groupby('eventId')
    event_args = [(grouped.get_group(eid).copy(), int(eid)) for eid in event_ids]
    n_workers = max(1, (os.cpu_count() or 1) - cm.N_FREE_THREADS)
    print(f'Analysing {len(event_args):,} events on {n_workers} workers ...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(cm._progress(pool.map(cm._analyse_event_worker, event_args),
                                    total=len(event_args), desc='Analysing events'))
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Cached per-event results to {cache_path}')
    return results


def main():
    # Cache/out tag keys only on the veto (the per-event fits are rotation- and
    # handedness-independent; no y-flip is applied anymore).
    tag = (f'_veto{VETO}' if VETO is not None else '')
    out_dir = CFG.out_dir(f'alignment_tpc{tag}')
    cache_dir = CFG.out_dir('cache')
    cache_path = os.path.join(cache_dir, f'event_results{tag}.pkl')

    df, det, rc = _load_hits()
    results = _analyse_events(df, cache_path)
    n_both = sum(r.has_both for r in results)
    print(f'Analysed {len(results):,} events: {n_both:,} with valid X+Y hits.')

    # ---- Base rotation from run_config det_orientation.z (override: --rot0) ----
    cfg_rot = float(det.orientation.get('z', 0.0) or 0.0)
    rot0 = ROT0_OVERRIDE if ROT0_OVERRIDE is not None else cfg_rot
    print(f'Base rotation: {rot0:.2f} deg '
          f'({"--rot0 override" if ROT0_OVERRIDE is not None else "run_config det_orientation.z"})')

    # ---- M3 reference ----
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    x_ref_angles, y_ref_angles, angle_event_nums = get_xy_angles(rays.ray_data)
    x_ref_angles = REF_X_SIGN * np.array(x_ref_angles)  # reference X handedness

    # ---- Cluster-quality subset for ALIGNMENT determination only ----
    # Events with strips in competing clusters (n_dropped) bias the z scan; the
    # geometry is fit on the clean subset, then applied to ALL events for the
    # efficiency/residual evaluation below.
    if MAXDROP is not None:
        align_results = [r for r in results if r.has_both
                         and (r.x_fit.n_dropped + r.y_fit.n_dropped) <= MAXDROP]
        print(f'Cluster-quality cut (n_dropped<={MAXDROP}): '
              f'{len(align_results):,} clean events used for alignment '
              f'(of {n_both:,} with X+Y).')
    else:
        align_results = results

    # ---- Iterative z + rotation + translation alignment (base rotation rot0) ----
    initial = cm.AlignmentParams(z_x=DET_PLANE_Z, z_y=DET_PLANE_Z, theta_deg=rot0,
                                 centre_x=CENTRE_XY, centre_y=CENTRE_XY,
                                 ref_x_sign=REF_X_SIGN)
    theta_scan = np.linspace(rot0 - 2.0, rot0 + 2.0, 81)  # 0.05 deg steps around the base
    best = cm.run_alignment(
        align_results, rays,
        initial_params=initial,
        n_iterations=N_ITER,
        z_values=Z_SCAN,
        theta_values=theta_scan,
        plot_each=False, plot_final=True,
        mask_to_active_region=False,
        out_dir=out_dir,
    )
    cm.save_alignment(best, os.path.join(out_dir, 'alignment.json'))

    # ---- Attach reference positions & alignment-quality plots ----
    cm.attach_reference_positions(results, rays, best, x_ref_angles, angle_event_nums)
    cm.plot_ref_angle_distributions(x_ref_angles, y_ref_angles, out_dir=out_dir)
    cm.plot_position_correlation(results, out_dir=out_dir)
    cm.plot_radial_residuals(results, radius_cut_mm=10.0, out_dir=out_dir)
    fit_x, fit_y = cm.plot_residuals(results, out_dir=out_dir)
    if fit_x:
        print(f'X resolution: {fit_x.resolution:.2f} +/- {fit_x.resolution_err:.2f} mm')
    if fit_y:
        print(f'Y resolution: {fit_y.resolution:.2f} +/- {fit_y.resolution_err:.2f} mm')

    if not FULL:
        print(f'\nAlignment checkpoint done. Plots in {out_dir}')
        print('Re-run with --full once alignment looks good.')
        return

    # ---- Full micro-TPC physics analysis ----
    v_dx, v_dy = cm.plot_angle_correlation(
        results, residual_cut_mm=10.0, min_strips=4, max_red_chi2=None,
        v_scan_min=25.0, v_scan_max=50.0, v_scan_steps=51, out_dir=out_dir,
        params=best)
    v_avg = float(np.nanmean([v_dx, v_dy]))
    print(f'Drift velocity: X={v_dx:.1f}  Y={v_dy:.1f}  avg={v_avg:.1f} um/ns')

    (xmn, xmx), (ymn, ymx) = cm.get_active_det_bounds(det, CFG.MAP_CSV_PATH)
    cx, cy = cm._det_to_ref(np.array([xmn, xmx, xmn, xmx]),
                            np.array([ymn, ymn, ymx, ymx]), best)
    active = (float(cx.min()), float(cx.max()), float(cy.min()), float(cy.max()))

    csv_dir = CFG.out_dir('alignment_tpc', 'Plot_Data')
    for rc_cut, name, title in [
        (None, 'efficiency_no_cut.csv', 'Detector Efficiency Map (any hit)'),
        (10.0, 'efficiency_r10mm_cut.csv', 'Detector Efficiency Map (r<10mm)'),
    ]:
        cm.plot_efficiency_map(results, rays, best, bins=40, min_tracks_per_bin=5,
                               radius_cut_mm=rc_cut, title=title,
                               csv_out_path=os.path.join(csv_dir, name),
                               active_region=active, out_dir=out_dir,
                               det_name=CFG.DET_NAME)

    cm.plot_resolution_map(results, rays, best, bins=20, min_hits_per_bin=20,
                           radius_cut_mm=None, out_dir=out_dir, det_name=CFG.DET_NAME)
    cm.plot_resolution_map_sliding(results, grid_points=100, kernel_radius_mm=50.0,
                                   min_hits=50, out_dir=out_dir, sigma_vmax=1.0,
                                   params=best, det_name=CFG.DET_NAME)
    print(f'\nFull micro-TPC analysis done. Plots in {out_dir}')


if __name__ == '__main__':
    main()
