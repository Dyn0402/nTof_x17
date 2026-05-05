#!/usr/bin/env python3
# -- coding: utf-8 --
"""
cosmic_micro_tpc_analysis.py

Refactored cosmic muon analysis for the MX17 micro-TPC Micromegas detector.

Pipeline overview
-----------------
1. Load detector hits (ROOT → pandas) and map FEU/channel → (x_mm, y_mm).
2. Load M3 reference tracker data (rays + positions).
3. Analyse each event with `analyse_event`:
      - Spatial clustering in X and Y strips (gap threshold)
      - Amplitude-weighted linear fit of strip position vs. hit time
        → slope  [ns/mm]  (dt/dx)
        → 1/slope [mm/ns] (drift velocity proxy, NOT a track angle)
      - Earliest-hit position in X and Y → mesh crossing position
      - Reduced χ² quality metrics
4. Z-alignment scan: iterate over candidate z values; at each z get the
   reference position, compute residuals vs. detector hit position, and
   minimise the variance of the residual distribution (separately for X and Y).
5. With the best z, compute per-event residuals and build the efficiency map.

Coordinate / angle conventions
-------------------------------
The M3 tracker reports angles as the arctangent of the track slope in the
xz (or yz) plane:  θ_x = atan(dx/dz),  θ_y = atan(dy/dz).
tan(θ_x) = dx/dz  [dimensionless].

The micro-TPC fit yields  slope_x = dt/dx  [ns/mm].
These are NOT directly comparable.  To correlate them you need the drift
velocity v_d [mm/ns]:
    dx/dz = (dx/dt) * (dt/dz) = (1/slope_x) / v_d_z
where v_d_z is the z-component of the drift velocity.  In a vertical
TPC with vertical drift,  dt/dz ≈ 1/v_d  so:
    tan(θ_x) ≈ (1/slope_x) / v_d

This script stores both raw slopes and the derived spatial slopes
(dx/dz estimator) so you can calibrate v_d from the correlation with the
reference tracker.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
import uproot
from scipy.optimize import curve_fit as cf, OptimizeWarning

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions
from common.Measure import Measure

# ---------------------------------------------------------------------------
# Configuration / tunables
# ---------------------------------------------------------------------------

GAP_THRESHOLD_MM: float = 12.0   # Max gap within a spatial cluster [mm]
MIN_STRIPS: int = 3               # Minimum strips for a valid fit
EPS: float = 1e-9                 # Guard against divide-by-zero

# Parallelism: leave this many CPU cores free; use the rest for event analysis
N_FREE_THREADS: int = 2

# Drift velocity (µm/ns) – initial estimate; refined by angle correlation
V_DRIFT_ESTIMATE: float = 50.0   # µm/ns — adjust after first pass

# Z-alignment scan parameters
# Z_SCAN_MIN: float = 200.0        # [mm]
# Z_SCAN_MAX: float = 300.0        # [mm]
# Z_SCAN_MIN: float = 500.0        # [mm]
# Z_SCAN_MAX: float = 800.0        # [mm]
Z_SCAN_MIN: float = 235.0        # [mm]
Z_SCAN_MAX: float = 270.0        # [mm]
Z_SCAN_STEPS: int = 101

# Rotation alignment scan parameters (rotation about detector centre in xy-plane)
# ROT_SCAN_MIN_DEG: float = -5.0  # [deg]
# ROT_SCAN_MAX_DEG: float = 5.0   # [deg]
ROT_SCAN_MIN_DEG: float = -2.0  # [deg]
ROT_SCAN_MAX_DEG: float = 2.0   # [deg]
ROT_SCAN_STEPS: int = 81

# Iterative alignment: number of z → rotation → z → ... cycles
ALIGNMENT_ITERATIONS: int = 2


# ---------------------------------------------------------------------------
# Main  (placed here for quick orientation; all helpers are defined below)
# ---------------------------------------------------------------------------

def main():
    # base_path = '/media/dylan/data/x17/cosmic_bench/det_1/'
    base_path = '/media/dylan/data/x17/cosmic_bench/det_3/'
    # run = 'mx17_det1_1-27-26'
    # sub_run = 'resist_scan_480V'
    # mx17_feus = [6]   # 4 = X strips, 6 = Y strips
    # run = 'mx17_det1_daytime_run_1-28-26'
    # sub_run = 'overnight_run'
    # mx17_feus = [4, 6]   # 4 = X strips, 6 = Y strips
    # run = 'mx17_det0_He_HV_Scan_4-1-26'
    # sub_run = 'resist_505V_drift_800V'
    # run = 'mx17_det1_Ar_CF4_HV_Scan_4-25-26'
    # sub_run = 'resist_550V_drift_1000V'
    # run = 'mx17_daq_det3_quick_test_5-5-26'
    # sub_run = 'quick_test'
    run = 'mx17_det3_HV_Scan_5-5-26'
    sub_run = 'resist_510V_drift_900V'
    mx17_feus = [3, 4]   # 4 = X strips, 6 = Y strips

    run_config_path = f'{base_path}{run}/run_config.json'
    map_csv_path = '../mx17_m4_map.csv'

    alignment_dir = f'{base_path}Alignment/'
    alignment_file = f'{alignment_dir}alignment.json'
    realign = True   # set True to re-run alignment and overwrite the saved file

    analysis_out_dir = f'{base_path}Analysis/'
    csv_out_dir = f'{analysis_out_dir}Plot_Data/'

    # ---- Load detector configuration ----
    rc = RunConfig(run_config_path, map_csv_path)
    det = rc.get_detector('mx17_1')

    # ---- Load raw hits ----
    hits_dir = f'{base_path}{run}/{sub_run}/combined_hits_root/'
    hit_files = [f for f in os.listdir(hits_dir) if f.endswith('.root') and '_datrun_' in f]
    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    df = df[df['feu'].isin(mx17_feus)].copy()

    # ---- Map (feu, channel) → (x_mm, y_mm) ----
    df = _map_strip_positions(df, det)
    print(f'Loaded {len(df)} hits over {df["eventId"].nunique()} events.')

    # ---- Load M3 reference tracker ----
    rays_dir = f'{base_path}{run}/{sub_run}/m3_tracking_root/'
    rays = M3RefTracking(rays_dir, chi2_cut=20)

    x_ref_angles, y_ref_angles, angle_event_nums = get_xy_angles(rays.ray_data)
    x_ref_angles = -np.array(x_ref_angles)  # sign convention

    # ---- Analyse each event (without reference position yet) ----
    event_ids = df['eventId'].unique()
    grouped = df.groupby('eventId')
    event_args = [(grouped.get_group(eid).copy(), int(eid)) for eid in event_ids]

    n_workers = max(1, (os.cpu_count() or 1) - N_FREE_THREADS)
    print(f'Analysing {len(event_args)} events on {n_workers} workers …')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results: List[EventResult] = list(
            _progress(pool.map(_analyse_event_worker, event_args),
                      total=len(event_args), desc='Analysing events')
        )

    print(f'Analysed {len(results)} events: '
          f'{sum(r.has_both for r in results)} with valid X+Y hits.')

    # ---- Iterative z + rotation + translation alignment ----
    if not realign and os.path.exists(alignment_file):
        best_params = load_alignment(alignment_file)
    else:
        initial_params = AlignmentParams(
            z_x=250.0, z_y=250.0, theta_deg=0.0,
            centre_x=200.0, centre_y=200.0,   # detector centre in strip-map coordinates
        )
        best_params = run_alignment(
            results, rays,
            initial_params=initial_params,
            n_iterations=ALIGNMENT_ITERATIONS,
            plot_each=False,
            plot_final=True,
            mask_to_active_region=False,
            active_region_margin_mm=10.0,
        )
        save_alignment(best_params, alignment_file)

    # ---- Attach aligned reference positions to EventResult objects ----
    attach_reference_positions(results, rays, best_params,
                               x_ref_angles, angle_event_nums)

    # ---- Plots ----
    plot_ref_angle_distributions(x_ref_angles, y_ref_angles)
    # Radial residual distribution (debug: check radius cut and M3 matching)
    plot_radial_residuals(results, radius_cut_mm=10.0)
    v_drift_x, v_drift_y = plot_angle_correlation(
        results,
        residual_cut_mm=10.0,
        min_strips=4,
        # max_red_chi2=5.0,
        max_red_chi2=None,
        v_scan_min=30.0,
        v_scan_max=50.0,
        v_scan_steps=41,
    )
    if not np.isnan(v_drift_x):
        print(f'Fitted drift velocity X: {v_drift_x:.1f} µm/ns')
    if not np.isnan(v_drift_y):
        print(f'Fitted drift velocity Y: {v_drift_y:.1f} µm/ns')
    v_avg = float(np.nanmean([v_drift_x, v_drift_y]))
    if np.isfinite(v_avg):
        print(f'Average drift velocity:  {v_avg:.1f} µm/ns')
    # ---- 3-D event display ----
    # x_max_mm restricts to the left (working) side of the detector.
    # Tune this and the other cuts based on the efficiency map and residual plots.
    if np.isfinite(v_avg):
        display_event_id = select_display_event(
            results,
            v_drift_um_per_ns=v_avg,
            max_spatial_residual_mm=3.0,
            max_angle_residual=0.1,
            max_red_chi2=None,
            min_strips=8,
            x_max_mm=150.0,   # left (working) side of detector — adjust as needed
        )
        if display_event_id is not None:
            df_display = df[df['eventId'] == display_event_id].copy()
            display_result = next(r for r in results if r.event_id == display_event_id)
            plot_event_display_3d(df_display, display_result, best_params,
                                  v_drift_um_per_ns=v_avg, event_id=display_event_id)
            gif_out = f'{analysis_out_dir}event_{display_event_id}_3d.gif'
            plot_event_display_3d_rotating(df_display, display_result, best_params,
                                           v_drift_um_per_ns=v_avg,
                                           event_id=display_event_id,
                                           drift_window_mm=30.0,
                                           gif_path=gif_out)
            _plot_event(df_display, display_result, display_event_id)

    # plt.show()
    plot_position_correlation(results)
    fit_x, fit_y = plot_residuals(results)
    if fit_x:
        print(f'X resolution: {fit_x.resolution:.2f} ± {fit_x.resolution_err:.2f} mm')
    if fit_y:
        print(f'Y resolution: {fit_y.resolution:.2f} ± {fit_y.resolution_err:.2f} mm')

    # Active detector region in reference frame (for efficiency box + annotation)
    (det_x_min, det_x_max), (det_y_min, det_y_max) = get_active_det_bounds(det, map_csv_path)
    corners_x = np.array([det_x_min, det_x_max, det_x_min, det_x_max])
    corners_y = np.array([det_y_min, det_y_min, det_y_max, det_y_max])
    ref_cx, ref_cy = _det_to_ref(corners_x, corners_y, best_params)
    active_region_ref = (float(ref_cx.min()), float(ref_cx.max()),
                         float(ref_cy.min()), float(ref_cy.max()))

    # Efficiency map — no radius cut (any valid hit counts)
    plot_efficiency_map(results, rays, best_params,
                        bins=40, min_tracks_per_bin=5,
                        radius_cut_mm=None,
                        title='Detector Efficiency Map (any hit)',
                        csv_out_path=f'{csv_out_dir}efficiency_no_cut.csv',
                        active_region=active_region_ref)

    # Efficiency map — with radius cut (hit must be within 5 mm of reference)
    plot_efficiency_map(results, rays, best_params,
                        bins=40, min_tracks_per_bin=5,
                        radius_cut_mm=10.0,
                        title='Detector Efficiency Map',
                        csv_out_path=f'{csv_out_dir}efficiency_r10mm_cut.csv',
                        active_region=active_region_ref)
    plt.show()

    # Radial residual distribution (debug: check radius cut and M3 matching)
    plot_radial_residuals(results, radius_cut_mm=10.0)

    # Spatial resolution map (coarse, non-overlapping bins)
    plot_resolution_map(results, rays, best_params,
                        bins=20, min_hits_per_bin=20,
                        radius_cut_mm=None)

    # Smooth sliding-window resolution map
    plot_resolution_map_sliding(results,
                                grid_points=100,
                                kernel_radius_mm=50.0,
                                min_hits=50)

    plot_resolution_map_sliding(results,
                                grid_points=100,
                                kernel_radius_mm=25.0,
                                min_hits=50)

    # ---- Export to DataFrame ----
    summary_df = pd.DataFrame([r.to_dict() for r in results])
    print(summary_df.describe())

    plt.show()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StripFitResult:
    """Result of fitting one axis (X or Y) in a single event."""
    slope_ns_per_mm: float          # dt/dx  [ns/mm]
    mesh_position_mm: float         # strip position of earliest hit [mm]
    earliest_time_ns: float         # time of earliest hit [ns]
    latest_time_ns: float           # time of latest hit [ns]
    n_strips: int                   # strips in the selected cluster
    n_dropped: int                  # strips in other clusters (noise)
    red_chi2: float                 # reduced χ²

    @property
    def slope_mm_per_ns(self) -> float:
        """Inverse slope: dx/dt  [mm/ns] (drift-velocity proxy)."""
        if abs(self.slope_ns_per_mm) < EPS:
            return np.nan
        return 1.0 / self.slope_ns_per_mm

    @property
    def tan_theta_estimate(self) -> float:
        """
        Estimated tan(θ) = (dx/dt) / v_drift_z.
        Uses the module-level V_DRIFT_ESTIMATE (µm/ns) as a placeholder.
        Replace with a calibrated value once the correlation plot is made.
        slope_mm_per_ns is in mm/ns; V_DRIFT_ESTIMATE is in µm/ns = mm/µs,
        so multiply slope by 1000 to convert to µm/ns before dividing.
        """
        if np.isnan(self.slope_mm_per_ns):
            return np.nan
        return (self.slope_mm_per_ns * 1000.0) / V_DRIFT_ESTIMATE

    @property
    def cluster_duration_ns(self) -> float:
        return self.latest_time_ns - self.earliest_time_ns

    def is_valid(self) -> bool:
        return (
            not np.isnan(self.slope_ns_per_mm)
            and not np.isnan(self.mesh_position_mm)
            and self.n_strips >= MIN_STRIPS
        )


@dataclass
class EventResult:
    """All analysis outputs for a single event."""
    event_id: int

    # Per-axis fit results (None if not enough hits)
    x_fit: Optional[StripFitResult] = None
    y_fit: Optional[StripFitResult] = None

    # Rotation-aligned detector positions (filled by attach_reference_positions)
    # Before alignment these equal det_x_mm / det_y_mm.
    det_x_aligned_mm: float = np.nan
    det_y_aligned_mm: float = np.nan

    # Reference tracker quantities (filled after matching)
    ref_x_mm: float = np.nan
    ref_y_mm: float = np.nan
    ref_tan_theta_x: float = np.nan
    ref_tan_theta_y: float = np.nan

    # Reference track mesh position in raw detector coordinates (inverse-alignment
    # of ref_x_mm / ref_y_mm); used for the reference track anchor in 3-D displays.
    ref_mesh_x_mm: float = np.nan
    ref_mesh_y_mm: float = np.nan

    @property
    def has_x(self) -> bool:
        return self.x_fit is not None and self.x_fit.is_valid()

    @property
    def has_y(self) -> bool:
        return self.y_fit is not None and self.y_fit.is_valid()

    @property
    def has_both(self) -> bool:
        return self.has_x and self.has_y

    @property
    def det_x_mm(self) -> float:
        """Raw (unrotated) mesh hit position in X."""
        return self.x_fit.mesh_position_mm if self.has_x else np.nan

    @property
    def det_y_mm(self) -> float:
        """Raw (unrotated) mesh hit position in Y."""
        return self.y_fit.mesh_position_mm if self.has_y else np.nan

    @property
    def det_x_for_residual(self) -> float:
        """Aligned X position used for residual; falls back to raw if not yet set."""
        if not np.isnan(self.det_x_aligned_mm):
            return self.det_x_aligned_mm
        return self.det_x_mm

    @property
    def det_y_for_residual(self) -> float:
        """Aligned Y position used for residual; falls back to raw if not yet set."""
        if not np.isnan(self.det_y_aligned_mm):
            return self.det_y_aligned_mm
        return self.det_y_mm

    @property
    def residual_x_mm(self) -> float:
        return self.ref_x_mm - self.det_x_for_residual

    @property
    def residual_y_mm(self) -> float:
        return self.ref_y_mm - self.det_y_for_residual

    @property
    def radial_residual_mm(self) -> float:
        """Radial distance between aligned detector hit and reference position [mm]."""
        dx, dy = self.residual_x_mm, self.residual_y_mm
        if np.isnan(dx) or np.isnan(dy):
            return np.nan
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def is_efficient(self, radius_cut_mm: Optional[float] = None) -> bool:
        """
        True if both axes had a valid hit.

        Parameters
        ----------
        radius_cut_mm : if given, also require the aligned detector hit to be
                        within this radius [mm] of the reference position.
                        If None, no spatial cut is applied.
        """
        if not self.has_both:
            return False
        if radius_cut_mm is not None:
            r = self.radial_residual_mm
            if np.isnan(r) or r > radius_cut_mm:
                return False
        return True

    def to_dict(self) -> dict:
        """Flat dictionary for DataFrame export."""
        d = {
            'event_id': self.event_id,
            'has_x': self.has_x,
            'has_y': self.has_y,
            'det_x_mm': self.det_x_mm,
            'det_y_mm': self.det_y_mm,
            'det_x_aligned_mm': self.det_x_for_residual,
            'det_y_aligned_mm': self.det_y_for_residual,
            'ref_x_mm': self.ref_x_mm,
            'ref_y_mm': self.ref_y_mm,
            'residual_x_mm': self.residual_x_mm,
            'residual_y_mm': self.residual_y_mm,
            'ref_tan_theta_x': self.ref_tan_theta_x,
            'ref_tan_theta_y': self.ref_tan_theta_y,
            'is_efficient': self.is_efficient(),
            'radial_residual_mm': self.radial_residual_mm,
        }
        for prefix, fit in [('x', self.x_fit), ('y', self.y_fit)]:
            if fit is not None:
                d[f'{prefix}_slope_ns_per_mm'] = fit.slope_ns_per_mm
                d[f'{prefix}_slope_mm_per_ns'] = fit.slope_mm_per_ns
                d[f'{prefix}_tan_theta_est'] = fit.tan_theta_estimate
                d[f'{prefix}_red_chi2'] = fit.red_chi2
                d[f'{prefix}_n_strips'] = fit.n_strips
                d[f'{prefix}_cluster_duration_ns'] = fit.cluster_duration_ns
            else:
                for k in ('slope_ns_per_mm', 'slope_mm_per_ns', 'tan_theta_est',
                          'red_chi2', 'n_strips', 'cluster_duration_ns'):
                    d[f'{prefix}_{k}'] = np.nan
        return d


@dataclass
class GaussFitResult:
    """
    Result of a robust iterative Gaussian peak fit to a residual distribution.
    The fitted σ is the detector spatial resolution estimate for that axis.
    """
    amplitude: float
    amplitude_err: float
    mean: float
    mean_err: float
    sigma: float           # Spatial resolution estimate
    sigma_err: float
    fit_window_lo: float   # Final fitting window lower edge [data units]
    fit_window_hi: float   # Final fitting window upper edge [data units]
    bin_width: float       # Bin width used in the final fit
    n_bins_used: int       # Number of bins inside the fit window
    n_iterations: int      # Convergence iterations taken
    converged: bool        # Whether the fit converged within tolerance

    @property
    def resolution(self) -> float:
        return abs(self.sigma)

    @property
    def resolution_err(self) -> float:
        return self.sigma_err

    def __str__(self) -> str:
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"GaussFit ({status}, {self.n_iterations} iters): "
                f"μ = {self.mean:.3f} ± {self.mean_err:.3f},  "
                f"σ = {self.resolution:.3f} ± {self.resolution_err:.3f}  "
                f"[window {self.fit_window_lo:.2f} → {self.fit_window_hi:.2f}]")


@dataclass
class AlignmentParams:
    """
    Full geometric alignment of the micro-TPC relative to the M3 reference frame.

    z_x, z_y : the z position [mm] at which the reference track is evaluated
                to match the detector's X and Y hit positions respectively.
                These may differ if the detector has a small tilt about x or y.
    theta_deg : rotation of the detector about the z-axis (i.e. in the xy-plane)
                relative to the reference frame, in degrees.
                Positive = counter-clockwise when viewed from +z.
    centre_x, centre_y : the pivot point for the rotation [mm], in detector
                coordinates.  Defaults to (0, 0); update to the detector centre
                once you know it from the strip map.
    x_offset, y_offset : translation applied after rotation to bring the detector
                local frame into the reference frame [mm].
                i.e. corrected_pos = rotated_det_pos + (x_offset, y_offset).
                Refined automatically by translation_alignment().
    """
    z_x: float = 250.0
    z_y: float = 250.0
    theta_deg: float = 0.0
    centre_x: float = 0.0
    centre_y: float = 0.0
    x_offset: float = 0.0
    y_offset: float = 0.0

    @property
    def z_mean(self) -> float:
        return 0.5 * (self.z_x + self.z_y)

    def __str__(self) -> str:
        return (f"AlignmentParams(z_x={self.z_x:.1f} mm, z_y={self.z_y:.1f} mm, "
                f"θ={self.theta_deg:.3f}°, "
                f"centre=({self.centre_x:.1f}, {self.centre_y:.1f}) mm, "
                f"offset=({self.x_offset:.2f}, {self.y_offset:.2f}) mm)")


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

def _progress(iterable, desc: str = '', total: int = None):
    """
    Wrap an iterable with tqdm if available, otherwise print a simple
    line-by-line counter that doesn't flood the terminal.
    """
    if _TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, total=total, ncols=80, leave=True)

    # Fallback: print progress every ~5% or at least every 10 steps
    items = list(iterable)
    n = len(items)
    report_every = max(1, min(10, n // 20))

    class _SimpleProgress:
        def __init__(self, items, desc, report_every):
            self._items = items
            self._desc = desc
            self._report_every = report_every

        def __iter__(self):
            n = len(self._items)
            start = desc_str = self._desc
            for i, item in enumerate(self._items):
                if i == 0 or (i + 1) % self._report_every == 0 or i == n - 1:
                    pct = 100.0 * (i + 1) / n
                    print(f'\r  {start}: {i + 1}/{n}  ({pct:.0f}%)', end='', flush=True)
                yield item
            print()  # newline after completion

    return _SimpleProgress(items, desc, report_every)


# ---------------------------------------------------------------------------
# Core per-event analysis
# ---------------------------------------------------------------------------

def _fit_single_axis(
    df_axis: pd.DataFrame,
    pos_col: str,
    gap_threshold: float = GAP_THRESHOLD_MM,
    min_strips: int = MIN_STRIPS,
) -> Optional[StripFitResult]:
    """
    Cluster strips along one axis, select the largest cluster, and fit
    strip position vs. hit time to a line anchored at the earliest hit.

    Parameters
    ----------
    df_axis : DataFrame with columns [pos_col, 'time', 'amplitude']
    pos_col : 'x_position_mm' or 'y_position_mm'
    gap_threshold : maximum gap [mm] within a cluster
    min_strips : minimum strips required for a valid fit

    Returns
    -------
    StripFitResult or None if insufficient data.

    Notes
    -----
    Fit model:  time = slope * (pos - pos_anchor) + time_anchor
    where (pos_anchor, time_anchor) is the earliest-hit strip.
    slope has units [ns/mm].
    """
    df_axis = df_axis[df_axis[pos_col].notna()].sort_values(pos_col).reset_index(drop=True)

    if len(df_axis) < min_strips:
        return None

    # --- Spatial clustering ---
    df_axis['_cluster'] = (
        df_axis[pos_col].diff().gt(gap_threshold)
    ).fillna(False).cumsum()

    largest_cluster_id = df_axis['_cluster'].value_counts().idxmax()
    df_cluster = df_axis[df_axis['_cluster'] == largest_cluster_id]
    n_dropped = len(df_axis) - len(df_cluster)

    if len(df_cluster) < min_strips:
        return None

    # --- Anchor point: earliest hit ---
    earliest_idx = df_cluster['time'].idxmin()
    pos_anchor = df_cluster.loc[earliest_idx, pos_col]
    time_anchor = df_cluster.loc[earliest_idx, 'time']
    latest_time = df_cluster['time'].max()

    # --- Amplitude-weighted fit ---
    positions = df_cluster[pos_col].to_numpy()
    times = df_cluster['time'].to_numpy()
    amps = df_cluster['amplitude'].to_numpy()
    sigma = 1.0 / np.sqrt(amps + EPS)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = cf(
                lambda pos, m: _line_anchored(pos, m, pos_anchor, time_anchor),
                positions,
                times,
                sigma=sigma,
                absolute_sigma=False,
            )
    except RuntimeError:
        return None

    slope_ns_per_mm = float(popt[0])

    # --- Reduced χ² ---
    t_pred = _line_anchored(positions, slope_ns_per_mm, pos_anchor, time_anchor)
    dof = len(df_cluster) - 1
    red_chi2 = float(np.sum(((times - t_pred) / sigma) ** 2) / dof) if dof > 0 else np.nan

    return StripFitResult(
        slope_ns_per_mm=slope_ns_per_mm,
        mesh_position_mm=float(pos_anchor),
        earliest_time_ns=float(time_anchor),
        latest_time_ns=float(latest_time),
        n_strips=len(df_cluster),
        n_dropped=n_dropped,
        red_chi2=red_chi2,
    )


def _analyse_event_worker(args: Tuple[pd.DataFrame, int]) -> EventResult:
    """Top-level wrapper so ProcessPoolExecutor can pickle the call."""
    df_event, event_id = args
    return analyse_event(df_event, event_id=event_id, plot=False)


def analyse_event(
    df_event: pd.DataFrame,
    event_id: int,
    gap_threshold: float = GAP_THRESHOLD_MM,
    plot: bool = False,
) -> EventResult:
    """
    Analyse one event's strip hits and return an EventResult.

    Parameters
    ----------
    df_event : DataFrame of hits for a single event, with columns
               [x_position_mm, y_position_mm, time, amplitude]
    event_id : event identifier
    gap_threshold : spatial clustering threshold [mm]
    plot : if True, draw position-vs-time scatter + fit lines
    """
    result = EventResult(event_id=event_id)

    df_x = df_event[df_event['x_position_mm'].notna()].copy()
    df_y = df_event[df_event['y_position_mm'].notna()].copy()

    result.x_fit = _fit_single_axis(df_x, 'x_position_mm', gap_threshold)
    result.y_fit = _fit_single_axis(df_y, 'y_position_mm', gap_threshold)

    if plot:
        _plot_event(df_event, result, event_id)

    return result


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _robust_sigma(vals: list) -> float:
    """Return robust Gaussian σ of a residual list, falling back to RMS."""
    if len(vals) < 50:
        return float(np.std(vals)) if vals else np.nan
    fit = fit_residual_peak(np.array(vals))
    if fit is not None and fit.converged:
        return fit.sigma
    return float(np.var(vals) ** 0.5)


def _rotate_det_positions(
    results: List[EventResult],
    theta_deg: float,
    centre_x: float,
    centre_y: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a rotation of `theta_deg` degrees about (centre_x, centre_y) followed
    by a translation (x_offset, y_offset) to bring detector positions into the
    reference frame.

        x' =  cos(θ)·(x−cx) − sin(θ)·(y−cy) + cx + x_offset
        y' =  sin(θ)·(x−cx) + cos(θ)·(y−cy) + cy + y_offset
    """
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_rot = np.full(len(results), np.nan)
    y_rot = np.full(len(results), np.nan)

    for i, r in enumerate(results):
        if r.has_x and r.has_y:
            dx = r.det_x_mm - centre_x
            dy = r.det_y_mm - centre_y
            x_rot[i] = cos_t * dx - sin_t * dy + centre_x + x_offset
            y_rot[i] = sin_t * dx + cos_t * dy + centre_y + y_offset
        elif r.has_x:
            x_rot[i] = r.det_x_mm + x_offset
        elif r.has_y:
            y_rot[i] = r.det_y_mm + y_offset

    return x_rot, y_rot


def _collect_residuals(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given alignment parameters, compute (dx, dy) residual arrays
    (ref_pos_at_z − rotated_det_pos) for all matched events.
    Returns two float arrays (dx_all, dy_all), NaN where no valid hit.
    """
    result_by_id = {r.event_id: r for r in results}

    x_refs_arr, y_refs_arr, event_nums = get_xy_positions(rays.ray_data, params.z_mean)
    x_refs_arr = -np.array(x_refs_arr)   # sign convention

    # We need per-event rotated+translated positions keyed by event_id
    x_rot, y_rot = _rotate_det_positions(
        results, params.theta_deg, params.centre_x, params.centre_y,
        params.x_offset, params.y_offset,
    )
    rot_x_by_id = {r.event_id: x_rot[i] for i, r in enumerate(results)}
    rot_y_by_id = {r.event_id: y_rot[i] for i, r in enumerate(results)}

    dx_all, dy_all = [], []
    for j, evn in enumerate(event_nums):
        if evn not in result_by_id:
            continue
        r = result_by_id[evn]
        rx = rot_x_by_id[evn]
        ry = rot_y_by_id[evn]
        dx_all.append(x_refs_arr[j] - rx if not np.isnan(rx) else np.nan)
        dy_all.append(y_refs_arr[j] - ry if not np.isnan(ry) else np.nan)

    return np.array(dx_all), np.array(dy_all)


# ---------------------------------------------------------------------------
# Z-alignment scan
# ---------------------------------------------------------------------------

def z_alignment_scan(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    z_values: Optional[np.ndarray] = None,
    plot: bool = False,
    label: str = '',
) -> AlignmentParams:
    """
    Scan over candidate z positions and find the z_x and z_y that minimise
    the robust Gaussian σ of the residual distribution.

    The current rotation in `params` is held fixed during the scan.

    Parameters
    ----------
    results   : list of EventResult (det positions filled)
    rays      : M3RefTracking instance
    params    : current AlignmentParams (z and theta used as starting point)
    z_values  : candidate z values [mm]; defaults to Z_SCAN_MIN..Z_SCAN_MAX
    plot      : draw σ-vs-z curves
    label     : string appended to plot titles (e.g. 'iter 1')

    Returns
    -------
    Updated AlignmentParams with best z_x and z_y.
    """
    if z_values is None:
        z_values = np.linspace(Z_SCAN_MIN, Z_SCAN_MAX, Z_SCAN_STEPS)

    result_by_id = {r.event_id: r for r in results}

    # Pre-compute rotated+translated det positions (held fixed during z scan)
    x_rot, y_rot = _rotate_det_positions(
        results, params.theta_deg, params.centre_x, params.centre_y,
        params.x_offset, params.y_offset,
    )
    rot_x_by_id = {r.event_id: x_rot[i] for i, r in enumerate(results)}
    rot_y_by_id = {r.event_id: y_rot[i] for i, r in enumerate(results)}

    sigma_x = np.full(len(z_values), np.nan)
    sigma_y = np.full(len(z_values), np.nan)

    desc = f'Z-scan{" " + label if label else ""}'
    for i, z in enumerate(_progress(z_values, desc=desc)):
        x_refs, y_refs, event_nums = get_xy_positions(rays.ray_data, z)
        x_refs = -np.array(x_refs)

        dx_vals, dy_vals = [], []
        for j, evn in enumerate(event_nums):
            if evn not in result_by_id:
                continue
            rx = rot_x_by_id[evn]
            ry = rot_y_by_id[evn]
            if not np.isnan(rx):
                dx_vals.append(float(x_refs[j]) - rx)
            if not np.isnan(ry):
                dy_vals.append(float(y_refs[j]) - ry)

        sigma_x[i] = _robust_sigma(dx_vals)
        sigma_y[i] = _robust_sigma(dy_vals)

    best_z_x = float(z_values[np.nanargmin(sigma_x)])
    best_z_y = float(z_values[np.nanargmin(sigma_y)])
    print(f'  Z-scan result:  z_x = {best_z_x:.1f} mm,  z_y = {best_z_y:.1f} mm')

    if plot:
        title_sfx = f'  [{label}]' if label else ''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(z_values, sigma_x, color='red')
        axes[0].axvline(best_z_x, color='red', ls='--', label=f'best z = {best_z_x:.1f} mm')
        axes[0].set_xlabel('z [mm]')
        axes[0].set_ylabel('Residual σ [mm]')
        axes[0].set_title(f'X: residual σ vs. z{title_sfx}')
        axes[0].legend()

        axes[1].plot(z_values, sigma_y, color='blue')
        axes[1].axvline(best_z_y, color='blue', ls='--', label=f'best z = {best_z_y:.1f} mm')
        axes[1].set_xlabel('z [mm]')
        axes[1].set_ylabel('Residual σ [mm]')
        axes[1].set_title(f'Y: residual σ vs. z{title_sfx}')
        axes[1].legend()
        fig.tight_layout()

    return AlignmentParams(
        z_x=best_z_x, z_y=best_z_y,
        theta_deg=params.theta_deg,
        centre_x=params.centre_x, centre_y=params.centre_y,
        x_offset=params.x_offset, y_offset=params.y_offset,
    )


# ---------------------------------------------------------------------------
# Rotation alignment scan
# ---------------------------------------------------------------------------

def rotation_alignment_scan(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    theta_values: Optional[np.ndarray] = None,
    plot: bool = False,
    label: str = '',
) -> AlignmentParams:
    """
    Scan over candidate rotation angles about the detector centre and find
    the θ that minimises the quadrature sum of σ_x and σ_y residuals.

    Because X and Y strips measure orthogonal coordinates, a rotation mixes
    them: the optimal θ minimises both simultaneously.  We use
    σ_combined = √(σ_x² + σ_y²) as the figure of merit.

    The z positions in `params` are held fixed during the scan.

    Parameters
    ----------
    results      : list of EventResult (det positions filled)
    rays         : M3RefTracking instance
    params       : current AlignmentParams (z and theta used as starting point)
    theta_values : candidate angles [deg]; defaults to ROT_SCAN_MIN/MAX_DEG
    plot         : draw σ-vs-θ curve
    label        : string appended to plot title

    Returns
    -------
    Updated AlignmentParams with best theta_deg.
    """
    if theta_values is None:
        theta_values = np.linspace(ROT_SCAN_MIN_DEG, ROT_SCAN_MAX_DEG, ROT_SCAN_STEPS)

    result_by_id = {r.event_id: r for r in results}

    # Reference positions at the current z (held fixed during rotation scan).
    # We evaluate at z_x for the X residuals and z_y for Y residuals.
    x_refs_raw, _, event_nums_x = get_xy_positions(rays.ray_data, params.z_x)
    x_refs_raw = -np.array(x_refs_raw)
    _, y_refs_raw, event_nums_y = get_xy_positions(rays.ray_data, params.z_y)

    # Build per-event ref lookup (use z_mean for combined metric)
    x_ref_by_id = {evn: float(x_refs_raw[j]) for j, evn in enumerate(event_nums_x)
                   if evn in result_by_id}
    y_ref_by_id = {evn: float(y_refs_raw[j]) for j, evn in enumerate(event_nums_y)
                   if evn in result_by_id}

    sigma_x_arr = np.full(len(theta_values), np.nan)
    sigma_y_arr = np.full(len(theta_values), np.nan)
    sigma_comb  = np.full(len(theta_values), np.nan)

    desc = f'Rotation-scan{" " + label if label else ""}'
    for i, theta in enumerate(_progress(theta_values, desc=desc)):
        cos_t = np.cos(np.deg2rad(theta))
        sin_t = np.sin(np.deg2rad(theta))
        cx, cy = params.centre_x, params.centre_y

        dx_vals, dy_vals = [], []
        for r in results:
            if not (r.has_x and r.has_y):
                continue
            # Rotate then translate det position
            dxr = r.det_x_mm - cx
            dyr = r.det_y_mm - cy
            rx = cos_t * dxr - sin_t * dyr + cx + params.x_offset
            ry = sin_t * dxr + cos_t * dyr + cy + params.y_offset

            if r.event_id in x_ref_by_id:
                dx_vals.append(x_ref_by_id[r.event_id] - rx)
            if r.event_id in y_ref_by_id:
                dy_vals.append(y_ref_by_id[r.event_id] - ry)

        sx = _robust_sigma(dx_vals)
        sy = _robust_sigma(dy_vals)
        sigma_x_arr[i] = sx
        sigma_y_arr[i] = sy
        if not (np.isnan(sx) or np.isnan(sy)):
            sigma_comb[i] = np.sqrt(sx ** 2 + sy ** 2)

    best_idx = int(np.nanargmin(sigma_comb))
    best_theta = float(theta_values[best_idx])
    print(f'  Rotation-scan result:  θ = {best_theta:.3f}°  '
          f'(σ_x={sigma_x_arr[best_idx]:.2f}, σ_y={sigma_y_arr[best_idx]:.2f} mm)')

    if plot:
        title_sfx = f'  [{label}]' if label else ''
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(theta_values, sigma_x_arr, color='red',  label='σ_x')
        ax.plot(theta_values, sigma_y_arr, color='blue', label='σ_y')
        ax.plot(theta_values, sigma_comb,  color='black', lw=2, label='σ_combined')
        ax.axvline(best_theta, color='black', ls='--',
                   label=f'best θ = {best_theta:.3f}°')
        ax.set_xlabel('Rotation θ [deg]')
        ax.set_ylabel('Residual σ [mm]')
        ax.set_title(f'Rotation alignment scan{title_sfx}')
        ax.legend()
        fig.tight_layout()

    return AlignmentParams(
        z_x=params.z_x, z_y=params.z_y,
        theta_deg=best_theta,
        centre_x=params.centre_x, centre_y=params.centre_y,
        x_offset=params.x_offset, y_offset=params.y_offset,
    )


# ---------------------------------------------------------------------------
# Active-region filter (partial detector support)
# ---------------------------------------------------------------------------

def _filter_to_active_region(
    results: List[EventResult],
    params: AlignmentParams,
    margin_mm: float = 10.0,
) -> List[EventResult]:
    """
    After a rough alignment, restrict the event list to those whose aligned
    detector hit falls within the spatial bounds of all observed hits.

    This removes "ghost" hits from unplugged strips (which land at positions
    outside the plugged region) so they don't bias subsequent alignment
    iterations.  The bounds are taken from the 1st–99th percentile of aligned
    hit positions to avoid sensitivity to outliers, then expanded by
    `margin_mm` on each side.

    Parameters
    ----------
    results    : full list of EventResult (detector positions filled)
    params     : current AlignmentParams (used to compute aligned positions)
    margin_mm  : extra margin beyond the observed hit extent [mm]

    Returns
    -------
    Filtered list of EventResult.
    """
    x_rot, y_rot = _rotate_det_positions(
        results, params.theta_deg, params.centre_x, params.centre_y,
        params.x_offset, params.y_offset,
    )

    valid_x = x_rot[~np.isnan(x_rot)]
    valid_y = y_rot[~np.isnan(y_rot)]

    if len(valid_x) < 10 or len(valid_y) < 10:
        print('  Active-region filter: too few hits to determine bounds — skipping.')
        return results

    x_min = float(np.percentile(valid_x, 5))  - margin_mm
    x_max = float(np.percentile(valid_x, 95)) + margin_mm
    y_min = float(np.percentile(valid_y, 5))  - margin_mm
    y_max = float(np.percentile(valid_y, 95)) + margin_mm

    print(f'  Active-region filter: '
          f'X = [{x_min:.1f}, {x_max:.1f}] mm, '
          f'Y = [{y_min:.1f}, {y_max:.1f}] mm  (margin = {margin_mm:.1f} mm)')

    filtered = []
    for i, r in enumerate(results):
        rx, ry = x_rot[i], y_rot[i]
        # For events with both axes, require both to be in bounds.
        # For single-axis hits, check only the available coordinate.
        if r.has_x and r.has_y:
            if x_min <= rx <= x_max and y_min <= ry <= y_max:
                filtered.append(r)
        elif r.has_x:
            if x_min <= rx <= x_max:
                filtered.append(r)
        elif r.has_y:
            if y_min <= ry <= y_max:
                filtered.append(r)

    print(f'  Active-region filter: {len(filtered)}/{len(results)} events kept')
    return filtered


# ---------------------------------------------------------------------------
# Translation alignment
# ---------------------------------------------------------------------------

def translation_alignment(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
) -> AlignmentParams:
    """
    Compute the robust median X/Y residual (after rotation) and absorb it
    into params.x_offset / params.y_offset.

    This corrects for a global shift between detector local coordinates and
    the reference frame (e.g. detector centred at (200, 200) mm vs. M3 at (0, 0)).
    The offset is additive: corrected_pos = rotated_det + (x_offset, y_offset).
    """
    dx_all, dy_all = _collect_residuals(results, rays, params)
    dx_clean = dx_all[~np.isnan(dx_all)]
    dy_clean = dy_all[~np.isnan(dy_all)]

    if len(dx_clean) == 0 or len(dy_clean) == 0:
        print('  Translation alignment: no matched events — skipping.')
        return params

    dx_shift = float(np.median(dx_clean))
    dy_shift = float(np.median(dy_clean))
    print(f'  Translation alignment:  Δx = {dx_shift:.2f} mm,  Δy = {dy_shift:.2f} mm  '
          f'({len(dx_clean)} matched events)')

    return AlignmentParams(
        z_x=params.z_x, z_y=params.z_y,
        theta_deg=params.theta_deg,
        centre_x=params.centre_x, centre_y=params.centre_y,
        x_offset=params.x_offset + dx_shift,
        y_offset=params.y_offset + dy_shift,
    )


# ---------------------------------------------------------------------------
# Iterative alignment
# ---------------------------------------------------------------------------

def run_alignment(
    results: List[EventResult],
    rays: M3RefTracking,
    initial_params: Optional[AlignmentParams] = None,
    n_iterations: int = ALIGNMENT_ITERATIONS,
    z_values: Optional[np.ndarray] = None,
    theta_values: Optional[np.ndarray] = None,
    plot_each: bool = False,
    plot_final: bool = True,
    mask_to_active_region: bool = False,
    active_region_margin_mm: float = 10.0,
) -> AlignmentParams:
    """
    Iteratively refine z-position, in-plane rotation, and translation alignment.

    Each iteration runs:
        1. z_alignment_scan          (z_x, z_y)  with current θ and offset fixed
        2. rotation_alignment_scan   (θ)          with updated z fixed
        3. translation_alignment     (x/y offset) from median residual

    If mask_to_active_region=True, after the first iteration a spatial filter
    is applied so that only events whose aligned detector hit falls within the
    observed hit bounds (± active_region_margin_mm) are used for all subsequent
    iterations.  This prevents ghost hits from unplugged strips from biasing
    the alignment when only part of the detector is instrumented.

    Parameters
    ----------
    results                 : list of EventResult
    rays                    : M3RefTracking
    initial_params          : starting AlignmentParams (default: z=250, θ=0°)
    n_iterations            : number of z → rotation → translation cycles
    z_values                : z scan grid (mm); None → use defaults
    theta_values            : rotation scan grid (deg); None → use defaults
    plot_each               : draw scan curves for every iteration
    plot_final              : draw scan curves for the final iteration only
    mask_to_active_region   : if True, filter to plugged-strip region after
                              iteration 1 (useful for partial detectors)
    active_region_margin_mm : margin added around observed hit bounds [mm]

    Returns
    -------
    Best AlignmentParams after all iterations.
    """
    if initial_params is None:
        initial_params = AlignmentParams()

    params = initial_params
    active_results = results   # may be replaced after first iteration
    print(f'\n{"="*60}')
    print(f'Starting iterative alignment ({n_iterations} iterations)')
    print(f'Initial: {params}')
    print(f'{"="*60}')

    history = [params]

    for it in range(1, n_iterations + 1):
        print(f'\n--- Iteration {it}/{n_iterations} ---')
        is_final = (it == n_iterations)
        do_plot = plot_each or (plot_final and is_final)
        lbl = f'iter {it}'

        # Step 1: z scan
        params = z_alignment_scan(
            active_results, rays, params,
            z_values=z_values, plot=do_plot, label=lbl,
        )

        # Step 2: rotation scan
        params = rotation_alignment_scan(
            active_results, rays, params,
            theta_values=theta_values, plot=do_plot, label=lbl,
        )

        # Step 3: translation (median residual → offset correction)
        params = translation_alignment(active_results, rays, params)

        # After the first iteration we have a rough translation — use it to
        # restrict subsequent iterations to the active (plugged) strip region.
        if mask_to_active_region and it == 1:
            print('\n  Applying active-region filter after first iteration:')
            active_results = _filter_to_active_region(
                results, params, margin_mm=active_region_margin_mm,
            )

        history.append(params)
        print(f'  → {params}')

    print(f'\n{"="*60}')
    print(f'Alignment converged to: {params}')
    print(f'{"="*60}\n')

    # ---- Convergence summary table ----
    print(f'{"Iter":>5}  {"z_x [mm]":>10}  {"z_y [mm]":>10}  {"θ [deg]":>10}  '
          f'{"x_off [mm]":>12}  {"y_off [mm]":>12}')
    for k, p in enumerate(history):
        tag = '(initial)' if k == 0 else f'iter {k}'
        print(f'{tag:>9}  {p.z_x:>10.2f}  {p.z_y:>10.2f}  {p.theta_deg:>10.4f}  '
              f'{p.x_offset:>12.2f}  {p.y_offset:>12.2f}')

    return params


# ---------------------------------------------------------------------------
# Alignment persistence
# ---------------------------------------------------------------------------

def save_alignment(params: AlignmentParams, path: str) -> None:
    """Serialise AlignmentParams to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        'z_x':       params.z_x,
        'z_y':       params.z_y,
        'theta_deg': params.theta_deg,
        'centre_x':  params.centre_x,
        'centre_y':  params.centre_y,
        'x_offset':  params.x_offset,
        'y_offset':  params.y_offset,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Alignment saved to {path}')


def load_alignment(path: str) -> AlignmentParams:
    """Load AlignmentParams from a JSON file saved by save_alignment()."""
    with open(path) as f:
        data = json.load(f)
    params = AlignmentParams(
        z_x=data['z_x'],
        z_y=data['z_y'],
        theta_deg=data['theta_deg'],
        centre_x=data['centre_x'],
        centre_y=data['centre_y'],
        x_offset=data.get('x_offset', 0.0),
        y_offset=data.get('y_offset', 0.0),
    )
    print(f'Alignment loaded from {path}: {params}')
    return params


# ---------------------------------------------------------------------------
# Reference position attachment (uses AlignmentParams)
# ---------------------------------------------------------------------------

def attach_reference_positions(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    x_ref_angles: np.ndarray,
    angle_event_nums: list,
) -> None:
    """
    Fill ref_x_mm, ref_y_mm, ref_tan_theta_x, ref_tan_theta_y on each
    EventResult using the given AlignmentParams.

    The rotation is applied to the *detector* positions before computing
    residuals, which is equivalent to rotating the reference frame the other
    way.  This function stores the rotated detector positions back onto the
    results so that downstream code (efficiency map, residual plots) uses the
    aligned coordinates consistently.

    Parameters
    ----------
    results           : list of EventResult (modified in-place)
    rays              : M3RefTracking
    params            : AlignmentParams from run_alignment
    x_ref_angles      : array of M3 x-angles (already sign-corrected)
    angle_event_nums  : event numbers corresponding to x_ref_angles
    """
    result_by_id = {r.event_id: r for r in results}

    # Evaluate reference at axis-specific z values
    x_pos_ref, _, pos_event_nums_x = get_xy_positions(rays.ray_data, params.z_x)
    _, y_pos_ref, pos_event_nums_y = get_xy_positions(rays.ray_data, params.z_y)
    x_pos_ref = -np.array(x_pos_ref)

    x_ref_by_id = {evn: float(x_pos_ref[j]) for j, evn in enumerate(pos_event_nums_x)
                   if evn in result_by_id}
    y_ref_by_id = {evn: float(y_pos_ref[j]) for j, evn in enumerate(pos_event_nums_y)
                   if evn in result_by_id}

    # Build angle lookup (y-angles come from the same call as x-angles in M3)
    _, y_ref_angles, _ = get_xy_angles(rays.ray_data)
    angle_by_id = {evn: (x_ref_angles[i], y_ref_angles[i])
                   for i, evn in enumerate(angle_event_nums)}

    # Apply rotation + translation to det positions
    x_rot, y_rot = _rotate_det_positions(
        results, params.theta_deg, params.centre_x, params.centre_y,
        params.x_offset, params.y_offset,
    )

    desc = 'Attaching reference positions'
    for i, r in enumerate(_progress(results, desc=desc)):
        # Store rotated hit positions (so residuals are in aligned frame)
        if r.has_x:
            r.det_x_aligned_mm = float(x_rot[i]) if not np.isnan(x_rot[i]) else np.nan
        if r.has_y:
            r.det_y_aligned_mm = float(y_rot[i]) if not np.isnan(y_rot[i]) else np.nan

        r.ref_x_mm = x_ref_by_id.get(r.event_id, np.nan)
        r.ref_y_mm = y_ref_by_id.get(r.event_id, np.nan)

        if r.event_id in angle_by_id:
            r.ref_tan_theta_x = float(np.tan(angle_by_id[r.event_id][0]))
            r.ref_tan_theta_y = float(np.tan(angle_by_id[r.event_id][1]))

        # Compute reference track mesh position in raw detector coordinates by
        # applying the inverse of the alignment transform to (ref_x_mm, ref_y_mm).
        #   Forward:  x' = cos(θ)·(x−cx) − sin(θ)·(y−cy) + cx + dx
        #             y' = sin(θ)·(x−cx) + cos(θ)·(y−cy) + cy + dy
        #   Inverse:  x  = cx + cos(θ)·(x'−cx−dx) + sin(θ)·(y'−cy−dy)
        #             y  = cy − sin(θ)·(x'−cx−dx) + cos(θ)·(y'−cy−dy)
        if not (np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm)):
            theta = np.deg2rad(params.theta_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cx, cy = params.centre_x, params.centre_y
            u = r.ref_x_mm - cx - params.x_offset
            v = r.ref_y_mm - cy - params.y_offset
            r.ref_mesh_x_mm = float(cx + cos_t * u + sin_t * v)
            r.ref_mesh_y_mm = float(cy - sin_t * u + cos_t * v)


# ---------------------------------------------------------------------------
# 2D map helpers
# ---------------------------------------------------------------------------

def _build_2d_map_arrays(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    bins: int,
    radius_cut_mm: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the raw arrays needed by both the efficiency and resolution maps.

    The denominator (total reference tracks) is built directly from the M3
    ray positions at the alignment z, completely independently of whether the
    detector found a hit.  This is the correct definition:

        Denominator = all reference tracks crossing the detector acceptance
        Numerator   = subset that produced a valid detector hit
                      (optionally within radius_cut_mm of the ref position)

    Holes in the denominator map indicate real gaps in reference track
    coverage (dead M3 scintillator tiles, beam halo, geometric acceptance),
    NOT missing detector hits.

    Returns
    -------
    ref_x_all, ref_y_all : reference positions for ALL M3 rays [mm]
    ref_x_hit, ref_y_hit : reference positions for rays that produced a hit
    x_edges, y_edges     : common histogram bin edges
    """
    # ---- All reference tracks at alignment z (denominator) ----
    x_refs_raw, y_refs_raw, _ = get_xy_positions(rays.ray_data, params.z_mean)
    ref_x_all = -np.array(x_refs_raw, dtype=float)  # sign convention
    ref_y_all  =  np.array(y_refs_raw, dtype=float)
    finite = np.isfinite(ref_x_all) & np.isfinite(ref_y_all)
    ref_x_all = ref_x_all[finite]
    ref_y_all  = ref_y_all[finite]

    # ---- Reference positions only for events with a valid detector hit ----
    ref_x_hit, ref_y_hit = [], []
    for r in results:
        if r.is_efficient(radius_cut_mm) and not (
            np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm)
        ):
            ref_x_hit.append(r.ref_x_mm)
            ref_y_hit.append(r.ref_y_mm)
    ref_x_hit = np.array(ref_x_hit)
    ref_y_hit = np.array(ref_y_hit)

    # ---- Common bin edges spanning both arrays ----
    all_x = np.concatenate([ref_x_all, ref_x_hit]) if len(ref_x_hit) else ref_x_all
    all_y = np.concatenate([ref_y_all, ref_y_hit]) if len(ref_y_hit) else ref_y_all
    x_edges = np.linspace(np.nanmin(all_x), np.nanmax(all_x), bins + 1)
    y_edges = np.linspace(np.nanmin(all_y), np.nanmax(all_y), bins + 1)

    return ref_x_all, ref_y_all, ref_x_hit, ref_y_hit, x_edges, y_edges


def _det_to_ref(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    params: 'AlignmentParams',
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply alignment transform (rotate about centre, then translate) to arrays of detector positions."""
    theta = np.deg2rad(params.theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = params.centre_x, params.centre_y
    dx = np.asarray(x_arr, dtype=float) - cx
    dy = np.asarray(y_arr, dtype=float) - cy
    x_ref = cos_t * dx - sin_t * dy + cx + params.x_offset
    y_ref = sin_t * dx + cos_t * dy + cy + params.y_offset
    return x_ref, y_ref


def get_active_det_bounds(det, strip_map_csv: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Bounding box of plugged-in strips in detector local coordinates.

    Returns (x_min, x_max), (y_min, y_max) in mm.
    X positions come from Y-axis strips; Y positions come from X-axis strips.
    """
    from common.Mx17StripMap import Mx17StripMap
    sm = Mx17StripMap(strip_map_csv)
    x_positions: List[float] = []
    y_positions: List[float] = []
    for det_key, (feu_id, feu_connector) in det.dream_feus.items():
        axis = det_key[0]
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


def plot_efficiency_map(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    bins: int = 100,
    min_tracks_per_bin: int = 2,
    radius_cut_mm: Optional[float] = None,
    title: str = 'Detector Efficiency Map',
    csv_out_path: Optional[str] = None,
    active_region: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Plot a 2-D efficiency map and track density side-by-side.

    The denominator is built directly from all M3 reference tracks, so the
    density plot shows the true reference track distribution.  Holes there
    mean the reference tracker had no coverage — not missing detector hits.

    Parameters
    ----------
    results           : list of EventResult with ref positions filled
    rays              : M3RefTracking (needed for the full denominator)
    params            : AlignmentParams (z used for reference position lookup)
    bins              : bins per axis (default 100)
    min_tracks_per_bin: bins with fewer reference tracks are masked grey
    radius_cut_mm     : if set, a hit only counts if the detector position is
                        within this radius of the reference track [mm]
    title             : plot title
    csv_out_path      : if given, write efficiency data to this CSV file path.
                        Columns: ref_x_mm, ref_y_mm, total_tracks, hits, efficiency,
                        radius_cut_mm.  Masked bins (< min_tracks_per_bin) are
                        included with efficiency = NaN so the grid is complete.
    """
    ref_x_all, ref_y_all, ref_x_hit, ref_y_hit, x_edges, y_edges = \
        _build_2d_map_arrays(results, rays, params, bins, radius_cut_mm)

    total, _, _ = np.histogram2d(ref_x_all, ref_y_all, bins=[x_edges, y_edges])
    hits,  _, _ = np.histogram2d(ref_x_hit, ref_y_hit, bins=[x_edges, y_edges])

    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.where(total >= min_tracks_per_bin, hits / total, np.nan)

    n_masked = int(np.sum(np.isnan(efficiency)))
    cut_str = f', r < {radius_cut_mm:.1f} mm' if radius_cut_mm is not None else ''
    print(f'Efficiency map{cut_str}: {n_masked}/{bins**2} bins masked '
          f'(< {min_tracks_per_bin} reference tracks)')

    # ---- Optional CSV export ----
    if csv_out_path is not None:
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x_centres, y_centres, indexing='ij')
        rows = pd.DataFrame({
            'ref_x_mm':       xx.ravel(),
            'ref_y_mm':       yy.ravel(),
            'total_tracks':   total.ravel().astype(int),
            'hits':           hits.ravel().astype(int),
            'efficiency':     efficiency.ravel(),
            'radius_cut_mm':  radius_cut_mm if radius_cut_mm is not None else np.nan,
        })
        os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
        rows.to_csv(csv_out_path, index=False)
        print(f'  Efficiency data written to {csv_out_path}')

    cmap_eff = plt.get_cmap('viridis').copy()
    cmap_eff.set_bad(color='lightgrey')
    cmap_den = plt.get_cmap('plasma').copy()
    cmap_den.set_bad(color='lightgrey')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Left: efficiency ----
    im = axes[0].imshow(
        efficiency.T, origin='lower', aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap_eff, vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=axes[0], label='Efficiency (hits / tracks)')
    axes[0].set_xlabel('Reference X [mm]')
    axes[0].set_ylabel('Reference Y [mm]')
    title_full = title + (f'\nRadius cut: r < {radius_cut_mm:.1f} mm' if radius_cut_mm else '')
    axes[0].set_title(f'{title_full}\n(grey = < {min_tracks_per_bin} tracks in bin)')

    # ---- Right: reference track density (denominator only) ----
    density = np.where(total >= min_tracks_per_bin, total, np.nan)
    im2 = axes[1].imshow(
        density.T, origin='lower', aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap_den,
    )
    plt.colorbar(im2, ax=axes[1], label='Reference tracks per bin')
    axes[1].set_xlabel('Reference X [mm]')
    axes[1].set_ylabel('Reference Y [mm]')
    axes[1].set_title(
        f'Reference track density (denominator)\n'
        f'Total M3 tracks: {len(ref_x_all):,}'
    )

    # Shrink axes to the bounding box of non-masked bins
    valid_xi, valid_yi = np.where(total >= min_tracks_per_bin)
    if len(valid_xi) > 0:
        x_lo = x_edges[valid_xi.min()]
        x_hi = x_edges[valid_xi.max() + 1]
        y_lo = y_edges[valid_yi.min()]
        y_hi = y_edges[valid_yi.max() + 1]
        for ax in axes:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

    # ---- Active region: red box + mean efficiency annotation ----
    if active_region is not None:
        from matplotlib.patches import Rectangle
        ar_x0, ar_x1, ar_y0, ar_y1 = active_region
        for ax in axes:
            ax.add_patch(Rectangle(
                (ar_x0, ar_y0), ar_x1 - ar_x0, ar_y1 - ar_y0,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            ))

        # Mean efficiency: integrated hits / tracks over bins inside active region
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x_centres, y_centres, indexing='ij')
        in_active = (xx >= ar_x0) & (xx <= ar_x1) & (yy >= ar_y0) & (yy <= ar_y1)
        active_mask = in_active & (total >= min_tracks_per_bin)
        n_active_tracks = int(total[active_mask].sum())
        n_active_hits = int(hits[active_mask].sum())
        if n_active_tracks > 0:
            mean_eff = n_active_hits / n_active_tracks
            eff_err = float(np.sqrt(mean_eff * (1.0 - mean_eff) / n_active_tracks))
            ann_text = f'Active region\neff = {mean_eff:.3f} ± {eff_err:.3f}'
        else:
            ann_text = 'Active region\neff = N/A'
        axes[0].annotate(
            ann_text,
            xy=(0.02, 0.03), xycoords='axes fraction',
            fontsize=9, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='red'),
        )

    fig.tight_layout()


# ---------------------------------------------------------------------------
# 2D spatial resolution map
# ---------------------------------------------------------------------------

def plot_resolution_map(
    results: List[EventResult],
    rays: M3RefTracking,
    params: AlignmentParams,
    bins: int = 20,
    min_hits_per_bin: int = 20,
    radius_cut_mm: Optional[float] = None,
    title: str = '2D Spatial Resolution Map',
) -> None:
    """
    Plot a 2-D map of the spatial resolution (Gaussian σ of the residual
    distribution) as a function of reference track position.

    Each 2D bin collects the X and Y residuals of all matched events whose
    reference track fell in that bin, then fits each with `fit_residual_peak`
    to extract σ_x and σ_y.  Bins with fewer than `min_hits_per_bin` matched
    events are masked.

    Note: the binning here is coarser than the efficiency map because each
    bin needs enough statistics to fit a Gaussian peak reliably.  The default
    of 20×20 bins is a reasonable starting point; reduce if bins are sparse.

    Parameters
    ----------
    results           : list of EventResult with residuals filled
    rays              : M3RefTracking (used to set axis ranges consistently)
    params            : AlignmentParams
    bins              : bins per axis (default 20 — coarser than efficiency map)
    min_hits_per_bin  : minimum matched hits to attempt a Gaussian fit
    radius_cut_mm     : optional pre-filter: only include hits within this
                        radius before computing per-bin resolution
    title             : plot title
    """
    # Collect per-event (ref_x, ref_y, dx, dy) for matched events
    ref_x_pts, ref_y_pts, dx_pts, dy_pts = [], [], [], []
    for r in results:
        if not r.is_efficient(radius_cut_mm):
            continue
        if np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm):
            continue
        dx = r.residual_x_mm
        dy = r.residual_y_mm
        if np.isnan(dx) or np.isnan(dy):
            continue
        ref_x_pts.append(r.ref_x_mm)
        ref_y_pts.append(r.ref_y_mm)
        dx_pts.append(dx)
        dy_pts.append(dy)

    ref_x_pts = np.array(ref_x_pts)
    ref_y_pts = np.array(ref_y_pts)
    dx_pts    = np.array(dx_pts)
    dy_pts    = np.array(dy_pts)

    if len(ref_x_pts) < min_hits_per_bin * 4:
        print('Resolution map: not enough matched events to build a useful map.')
        return

    x_edges = np.linspace(ref_x_pts.min(), ref_x_pts.max(), bins + 1)
    y_edges = np.linspace(ref_y_pts.min(), ref_y_pts.max(), bins + 1)

    sigma_x_map = np.full((bins, bins), np.nan)
    sigma_y_map = np.full((bins, bins), np.nan)
    count_map   = np.zeros((bins, bins), dtype=int)

    for ix in range(bins):
        for iy in range(bins):
            mask = (
                (ref_x_pts >= x_edges[ix]) & (ref_x_pts < x_edges[ix + 1]) &
                (ref_y_pts >= y_edges[iy]) & (ref_y_pts < y_edges[iy + 1])
            )
            n = int(mask.sum())
            count_map[ix, iy] = n
            if n < min_hits_per_bin:
                continue

            fit_x = fit_residual_peak(dx_pts[mask])
            fit_y = fit_residual_peak(dy_pts[mask])

            if fit_x is not None:
                sigma_x_map[ix, iy] = fit_x.resolution
            if fit_y is not None:
                sigma_y_map[ix, iy] = fit_y.resolution

    cut_str = f', r < {radius_cut_mm:.1f} mm' if radius_cut_mm is not None else ''
    print(f'Resolution map{cut_str}: '
          f'{int(np.sum(~np.isnan(sigma_x_map)))}/{bins**2} X bins fitted, '
          f'{int(np.sum(~np.isnan(sigma_y_map)))}/{bins**2} Y bins fitted')

    # Shared colour scale: 0 → 95th percentile of fitted values
    all_sigma = np.concatenate([
        sigma_x_map[~np.isnan(sigma_x_map)],
        sigma_y_map[~np.isnan(sigma_y_map)],
    ])
    vmax = float(np.nanpercentile(all_sigma, 95)) if len(all_sigma) > 0 else 1.0

    cmap_res = plt.get_cmap('jet').copy()
    cmap_res.set_bad(color='lightgrey')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    for ax, data, label in [
        (axes[0], sigma_x_map, 'σ_x [mm]'),
        (axes[1], sigma_y_map, 'σ_y [mm]'),
    ]:
        im = ax.imshow(
            data.T, origin='lower', aspect='auto', extent=extent,
            cmap=cmap_res, vmin=0, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('Reference X [mm]')
        ax.set_ylabel('Reference Y [mm]')
        ax.set_title(f'{title}\n{label}{cut_str}')

    # ---- Third panel: hit count per bin ----
    count_masked = np.where(count_map >= min_hits_per_bin, count_map, np.nan)
    cmap_cnt = plt.get_cmap('plasma').copy()
    cmap_cnt.set_bad(color='lightgrey')
    im3 = axes[2].imshow(
        count_masked.T, origin='lower', aspect='auto', extent=extent,
        cmap=cmap_cnt,
    )
    plt.colorbar(im3, ax=axes[2], label='Matched hits per bin')
    axes[2].set_xlabel('Reference X [mm]')
    axes[2].set_ylabel('Reference Y [mm]')
    axes[2].set_title(f'Hit statistics per bin\n(grey = < {min_hits_per_bin} hits)')

    # Shrink axes to the bounding box of non-masked bins
    valid_xi, valid_yi = np.where(count_map >= min_hits_per_bin)
    if len(valid_xi) > 0:
        x_lo = x_edges[valid_xi.min()]
        x_hi = x_edges[valid_xi.max() + 1]
        y_lo = y_edges[valid_yi.min()]
        y_hi = y_edges[valid_yi.max() + 1]
        for ax in axes:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# 2D sliding-window resolution map
# ---------------------------------------------------------------------------

def plot_resolution_map_sliding(
    results: List[EventResult],
    grid_points: int = 100,
    kernel_radius_mm: float = 5.0,
    min_hits: int = 50,
    title: str = '2D Sliding-Window Resolution Map',
) -> None:
    """
    Compute a smooth 2-D spatial resolution map using a sliding radial kernel.

    For each point on a regular (grid_points × grid_points) grid, all matched
    events whose reference track is within `kernel_radius_mm` of that grid
    point are collected.  The X and Y residual distributions are each fitted
    with `fit_residual_peak` (the same robust iterative Gaussian used elsewhere)
    to extract σ_x and σ_y at that location.

    Because adjacent grid points share statistics (overlapping kernels), the
    resulting maps are smooth even with a fine grid, unlike the non-overlapping
    bin approach in `plot_resolution_map`.

    Parameters
    ----------
    results          : list of EventResult with residuals filled
    grid_points      : number of grid points per axis (default 100)
    kernel_radius_mm : radius of the collection kernel [mm] (default 5.0)
    min_hits         : minimum events in kernel to attempt a Gaussian fit
    title            : plot title
    """
    # ---- Collect valid (ref_x, ref_y, dx, dy) arrays ----
    ref_x_pts, ref_y_pts, dx_pts, dy_pts = [], [], [], []
    for r in results:
        if not r.has_both:
            continue
        if np.isnan(r.ref_x_mm) or np.isnan(r.ref_y_mm):
            continue
        dx = r.residual_x_mm
        dy = r.residual_y_mm
        if np.isnan(dx) or np.isnan(dy):
            continue
        ref_x_pts.append(r.ref_x_mm)
        ref_y_pts.append(r.ref_y_mm)
        dx_pts.append(dx)
        dy_pts.append(dy)

    ref_x_arr = np.array(ref_x_pts)
    ref_y_arr = np.array(ref_y_pts)
    dx_arr    = np.array(dx_pts)
    dy_arr    = np.array(dy_pts)

    if len(ref_x_arr) < min_hits:
        print('Sliding resolution map: not enough matched events.')
        return

    print(f'Sliding resolution map: {len(ref_x_arr):,} matched events, '
          f'{grid_points}×{grid_points} grid, kernel r = {kernel_radius_mm:.1f} mm', flush=True)

    # ---- Build grid spanning the data extent ----
    x_grid = np.linspace(ref_x_arr.min(), ref_x_arr.max(), grid_points)
    y_grid = np.linspace(ref_y_arr.min(), ref_y_arr.max(), grid_points)

    sigma_x_map = np.full((grid_points, grid_points), np.nan)
    sigma_y_map = np.full((grid_points, grid_points), np.nan)
    count_map   = np.zeros((grid_points, grid_points), dtype=int)

    r2_cut = kernel_radius_mm ** 2

    # Outer loop over x (vectorise the inner y loop per x-row for speed)
    for i, xg in enumerate(_progress(x_grid, desc='Sliding resolution map')):
        dx2 = (ref_x_arr - xg) ** 2          # shape (N_events,), reused for all y
        for j, yg in enumerate(y_grid):
            dr2 = dx2 + (ref_y_arr - yg) ** 2
            mask = dr2 <= r2_cut
            n = int(mask.sum())
            count_map[i, j] = n
            if n < min_hits:
                continue

            fit_x = fit_residual_peak(dx_arr[mask])
            fit_y = fit_residual_peak(dy_arr[mask])
            if fit_x is not None:
                sigma_x_map[i, j] = fit_x.resolution
            if fit_y is not None:
                sigma_y_map[i, j] = fit_y.resolution

    n_fitted = int(np.sum(~np.isnan(sigma_x_map)))
    print(f'Sliding resolution map: {n_fitted}/{grid_points**2} grid points fitted')

    # ---- Shared colour scale: floor(min, 2 dp) → 95th percentile ----
    all_sigma = np.concatenate([
        sigma_x_map[~np.isnan(sigma_x_map)],
        sigma_y_map[~np.isnan(sigma_y_map)],
    ])
    if len(all_sigma) > 0:
        vmin = float(np.floor(np.nanmin(all_sigma) * 100) / 100)
        vmax = float(np.nanpercentile(all_sigma, 95))
    else:
        vmin, vmax = 0.0, 1.0

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    cmap_res = plt.get_cmap('jet').copy()
    cmap_res.set_bad(color='lightgrey')
    cmap_cnt = plt.get_cmap('plasma').copy()
    cmap_cnt.set_bad(color='lightgrey')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, data, label in [
        (axes[0], sigma_x_map, 'σ_x [mm]'),
        (axes[1], sigma_y_map, 'σ_y [mm]'),
    ]:
        im = ax.imshow(
            data.T, origin='lower', aspect='auto', extent=extent,
            cmap=cmap_res, vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('Reference X [mm]')
        ax.set_ylabel('Reference Y [mm]')
        ax.set_title(f'{title}\n{label}  (kernel r = {kernel_radius_mm:.1f} mm)')

    count_masked = np.where(count_map >= min_hits, count_map, np.nan)
    im3 = axes[2].imshow(
        count_masked.T, origin='lower', aspect='auto', extent=extent,
        cmap=cmap_cnt,
    )
    plt.colorbar(im3, ax=axes[2], label='Events in kernel')
    axes[2].set_xlabel('Reference X [mm]')
    axes[2].set_ylabel('Reference Y [mm]')
    axes[2].set_title(f'Events per kernel\n(grey = < {min_hits})')

    # Shrink axes to the bounding box of non-masked grid points
    valid_xi, valid_yi = np.where(count_map >= min_hits)
    if len(valid_xi) > 0:
        dx_half = (x_grid[1] - x_grid[0]) / 2 if len(x_grid) > 1 else 0
        dy_half = (y_grid[1] - y_grid[0]) / 2 if len(y_grid) > 1 else 0
        x_lo = x_grid[valid_xi.min()] - dx_half
        x_hi = x_grid[valid_xi.max()] + dx_half
        y_lo = y_grid[valid_yi.min()] - dy_half
        y_hi = y_grid[valid_yi.max()] + dy_half
        for ax in axes:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_ref_angle_distributions(
    x_ref_angles: np.ndarray,
    y_ref_angles: np.ndarray,
    n_bins: int = 80,
) -> None:
    """
    Plot histograms of the raw M3 reference track angles (in radians and in
    degrees) for the X and Y projections.  Useful for checking whether the
    angle arrays are sensible before looking at the correlation plots.
    """
    x_deg = np.degrees(x_ref_angles)
    y_deg = np.degrees(y_ref_angles)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, data, label, color in [
        (axes[0, 0], x_ref_angles, 'X angle [rad]', 'red'),
        (axes[0, 1], y_ref_angles, 'Y angle [rad]', 'blue'),
        (axes[1, 0], x_deg,        'X angle [deg]', 'red'),
        (axes[1, 1], y_deg,        'Y angle [deg]', 'blue'),
    ]:
        ax.hist(data, bins=n_bins, color=color, alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel('Counts')
        ax.set_title(
            f'{label}  —  mean={np.nanmean(data):.3f},  σ={np.nanstd(data):.3f}'
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle('M3 reference track angle distributions', fontsize=12)
    fig.tight_layout()


def _deming_fit_sigma_clip(
    x: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = 1.0,
    n_sigma: float = 2.5,
    max_iter: int = 15,
) -> Tuple[float, np.ndarray]:
    """
    Robust linear fit through the origin using Deming (orthogonal / total least
    squares) regression with iterative sigma-clipping on *perpendicular* residuals.

    Why this beats vertical-residual clipping for the angle correlation
    -------------------------------------------------------------------
    The outlier cloud sits at (large |θ_det|, θ_ref ≈ 0).  A line of slope
    s = 0.9 passes through the origin with gradient ~1, so the vertical
    distance from an outlier at (20°, 0°) to the fitted line is only
    |0 - 0.9*20| = 18° — identical to the vertical distance of an inlier at
    (20°, 18°).  Vertical clipping cannot separate them.

    The *perpendicular* distance from (20°, 0°) to the diagonal is
    20° * sin(45°) ≈ 14°, while a perfect inlier at (20°, 18°) has
    perpendicular distance of only 1° * sin(45°) ≈ 0.7°.  Orthogonal
    clipping cleanly separates them.

    Parameters
    ----------
    x, y    : paired angle arrays [any unit, typically deg]
    delta   : variance ratio σ_x²/σ_y² — 1.0 for equal-error Deming
    n_sigma : clipping threshold in units of the robust (MAD-based) σ
    max_iter: hard cap on iterations

    Returns
    -------
    slope      : fitted slope (ideal = 1.0 for a perfect detector)
    inlier_mask: boolean array aligned with input x, y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    for _ in range(max_iter):
        xm, ym = x[mask], y[mask]
        if xm.size < 5:
            break

        # Deming regression through origin
        # Minimises Σ [(y_i - b·x_i)² / (1 + b²·δ)] subject to slope = b
        # Closed form: b = (S_yy - δ·S_xx + √[(S_yy-δ·S_xx)² + 4δ·S_xy²]) / (2·S_xy)
        # where S_xx = Σx², S_yy = Σy², S_xy = Σxy  (sums, not deviations — through origin)
        sxx = float(np.dot(xm, xm))
        syy = float(np.dot(ym, ym))
        sxy = float(np.dot(xm, ym))
        if abs(sxy) < 1e-12:
            break
        disc = (syy - delta * sxx) ** 2 + 4.0 * delta * sxy ** 2
        slope = (syy - delta * sxx + np.sqrt(max(disc, 0.0))) / (2.0 * sxy)

        # Perpendicular (orthogonal) signed residual to  y = slope·x
        # Distance = (slope·x - y) / √(slope²+1)
        denom = np.sqrt(slope ** 2 + 1.0)
        resid = (slope * x - y) / denom          # all points, not just inliers

        # Robust σ from inlier residuals using MAD
        inlier_resid = resid[mask]
        med = float(np.median(inlier_resid))
        mad = float(np.median(np.abs(inlier_resid - med)))
        sigma_rob = mad / 0.6745
        if sigma_rob < 1e-9:
            break

        new_mask = np.abs(resid - med) <= n_sigma * sigma_rob
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # Final slope on converged inlier set
    xm, ym = x[mask], y[mask]
    if xm.size >= 2:
        sxx = float(np.dot(xm, xm))
        syy = float(np.dot(ym, ym))
        sxy = float(np.dot(xm, ym))
        disc = (syy - delta * sxx) ** 2 + 4.0 * delta * sxy ** 2
        slope = (syy - delta * sxx + np.sqrt(max(disc, 0.0))) / (2.0 * sxy)
    else:
        slope = np.nan

    return float(slope), mask


def _deg_from_slope(slope_mm_per_ns: np.ndarray, v_drift_um_per_ns: float) -> np.ndarray:
    """Convert raw strip-fit slope (mm/ns) to angle (deg) using a given drift velocity (µm/ns)."""
    tan_theta = (slope_mm_per_ns * 1000.0) / v_drift_um_per_ns
    return np.degrees(np.arctan(tan_theta))


def plot_angular_resolution(
    deg_ref: np.ndarray,
    deg_resid: np.ndarray,
    lim: float,
    color: str,
    proj: str,
    n_bins: int = 40,
    min_events: int = 15,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plot angular resolution (σ of Gaussian fit to Δθ slices) vs. reference angle.

    deg_ref   : reference angle for each event [deg]
    deg_resid : rotated residual Δθ = θ_ref − slope·θ_det [deg]
    ax        : if provided, draw into this axes instead of creating a new figure
    """
    bin_edges   = np.linspace(-lim, lim, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    sigmas, sigma_errs, centres_used = [], [], []
    for lo, hi, cen in zip(bin_edges[:-1], bin_edges[1:], bin_centres):
        mask = (deg_ref >= lo) & (deg_ref < hi)
        data = deg_resid[mask]
        if len(data) < min_events:
            continue
        fit = fit_residual_peak(data)
        if fit is not None:
            sigmas.append(fit.resolution)
            sigma_errs.append(fit.resolution_err)
            centres_used.append(cen)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    if sigmas:
        ax.errorbar(centres_used, sigmas, yerr=sigma_errs,
                    fmt='o-', color=color, capsize=4, lw=1.5, ms=6,
                    label=proj)
    ax.set_xlabel('θ reference [deg]')
    ax.set_ylabel('Angular resolution σ [deg]')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    if standalone:
        ax.set_xlabel(f'θ_{proj.lower()} reference [deg]')
        ax.set_title(f'{proj} angular resolution vs. track angle')
        ax.set_xlim(-lim, lim)
        fig.tight_layout()


def _fit_diagonal_peak(d: np.ndarray) -> Tuple[float, float]:
    """
    Fit a Gaussian to the central peak of  d = θ_ref − θ_det  using the
    iterative windowed fitter.  Returns (sigma, sigma_err) or (nan, nan).
    """
    result = fit_residual_peak(d[np.isfinite(d)], min_counts_in_window=20)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    # ax.hist(d[np.isfinite(d)], bins=200)
    # ax.set_xlabel('Residuals')
    # fig.tight_layout()
    if result is not None:
        return result.resolution, result.resolution_err
    return np.nan, np.nan


def plot_angle_correlation(
    results: List[EventResult],
    residual_cut_mm: Optional[float] = None,
    min_strips: int = 4,
    max_red_chi2: Optional[float] = 5.0,
    v_scan_min: float = 30.0,
    v_scan_max: float = 50.0,
    v_scan_steps: int = 41,
) -> Tuple[float, float]:
    """
    Plot angle_ref vs. angle_det (in degrees) and extract the drift velocity
    via a diagonal-projection scan.

    Steps
    -----
    1. Figure 1: raw correlation scatter (all quality-passing events) using
       the ballpark V_DRIFT_ESTIMATE.
    2. Apply track-quality cuts (radial residual + reduced-chi²).
    3. Scan drift velocities from v_scan_min to v_scan_max (µm/ns).  At each
       value recompute θ_det and project onto the diagonal:
           d = θ_ref − θ_det
       Fit a Gaussian to the d=0 peak using a half-mirror technique that
       ignores the skewed tail on one side.  Record σ.
    4. Figure 2: σ vs v_drift scan — minimum → best drift velocity.
    5. Figure 3: corrected correlation scatter using the best v_drift (average
       of X and Y) with a y=x reference line.
    6. Figure 4 (per projection): angular resolution σ(θ_ref) from sliced
       Gaussian fits on the corrected residuals.

    Returns (v_drift_x, v_drift_y) [µm/ns], or (nan, nan) on failure.
    """
    # ---- Collect per-event data: store raw slopes to allow v_drift recomputation ----
    deg_ref_x, slope_x, rad_res_x, rchi2_x = [], [], [], []
    deg_ref_y, slope_y, rad_res_y, rchi2_y = [], [], [], []

    for r in results:
        if (r.has_x
                and not np.isnan(r.ref_tan_theta_x)
                and not np.isnan(r.x_fit.slope_mm_per_ns)
                and r.x_fit.n_strips >= min_strips):
            deg_ref_x.append(np.degrees(np.arctan(r.ref_tan_theta_x)))
            slope_x.append(r.x_fit.slope_mm_per_ns)
            rad_res_x.append(r.radial_residual_mm)
            rchi2_x.append(r.x_fit.red_chi2)

        if (r.has_y
                and not np.isnan(r.ref_tan_theta_y)
                and not np.isnan(r.y_fit.slope_mm_per_ns)
                and r.y_fit.n_strips >= min_strips):
            deg_ref_y.append(np.degrees(np.arctan(r.ref_tan_theta_y)))
            slope_y.append(r.y_fit.slope_mm_per_ns)
            rad_res_y.append(r.radial_residual_mm)
            rchi2_y.append(r.y_fit.red_chi2)

    deg_ref_x = np.array(deg_ref_x); slope_x = np.array(slope_x)
    rad_res_x = np.array(rad_res_x); rchi2_x = np.array(rchi2_x)
    deg_ref_y = np.array(deg_ref_y); slope_y = np.array(slope_y)
    rad_res_y = np.array(rad_res_y); rchi2_y = np.array(rchi2_y)

    lim_deg = 30.0
    lim_x = min(1.5 * float(np.nanpercentile(np.abs(deg_ref_x), 99)), lim_deg * 1.5) \
            if len(deg_ref_x) else lim_deg
    lim_y = min(1.5 * float(np.nanpercentile(np.abs(deg_ref_y), 99)), lim_deg * 1.5) \
            if len(deg_ref_y) else lim_deg

    # ---- Figure 1: raw correlation with ballpark v_drift ----
    deg_det_x0 = _deg_from_slope(slope_x, V_DRIFT_ESTIMATE)
    deg_det_y0 = _deg_from_slope(slope_y, V_DRIFT_ESTIMATE)

    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xd, yd, lim, col, proj in [
        (axes1[0], deg_det_x0, deg_ref_x, lim_x, 'red',  'x'),
        (axes1[1], deg_det_y0, deg_ref_y, lim_y, 'blue', 'y'),
    ]:
        iv = (np.abs(xd) <= lim) & (np.abs(yd) <= lim)
        ax.scatter(xd[iv], yd[iv], alpha=0.25, s=4, color=col)
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        xs = np.array([-lim, lim])
        ax.plot(xs, xs, 'k--', lw=1.0, label='y = x')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'θ_{proj} detector [deg]  (v_drift={V_DRIFT_ESTIMATE:.0f} µm/ns)')
        ax.set_ylabel(f'θ_{proj} reference [deg]')
        ax.set_title(f'{proj.upper()} angle correlation — all, n_strips ≥ {min_strips}')
        ax.legend(fontsize=8)
    fig1.tight_layout()

    v_drift_x = v_drift_y = np.nan
    if residual_cut_mm is None:
        return v_drift_x, v_drift_y

    # ---- Quality mask ----
    chi2_ok_x = np.isfinite(rchi2_x) & ((rchi2_x <= max_red_chi2) if max_red_chi2 else True)
    chi2_ok_y = np.isfinite(rchi2_y) & ((rchi2_y <= max_red_chi2) if max_red_chi2 else True)
    qmask_x = (rad_res_x < residual_cut_mm) & chi2_ok_x & np.isfinite(deg_ref_x)
    qmask_y = (rad_res_y < residual_cut_mm) & chi2_ok_y & np.isfinite(deg_ref_y)

    ref_x_q  = deg_ref_x[qmask_x];  sl_x_q  = slope_x[qmask_x]
    ref_y_q  = deg_ref_y[qmask_y];  sl_y_q  = slope_y[qmask_y]

    # ---- Drift-velocity scan ----
    v_values = np.linspace(v_scan_min, v_scan_max, v_scan_steps)
    sigmas_x = np.full(len(v_values), np.nan)
    sigmas_y = np.full(len(v_values), np.nan)

    for i, v in enumerate(v_values):
        d_x = np.abs(ref_x_q) - np.abs(_deg_from_slope(sl_x_q, v))
        d_y = np.abs(ref_y_q) - np.abs(_deg_from_slope(sl_y_q, v))
        sigma_x, _ = _fit_diagonal_peak(d_x)
        sigma_y, _ = _fit_diagonal_peak(d_y)
        sigmas_x[i] = sigma_x
        sigmas_y[i] = sigma_y

    # Best v_drift = position of minimum sigma
    def _best_v(sigmas):
        valid = np.isfinite(sigmas)
        if not valid.any():
            return np.nan
        return float(v_values[np.nanargmin(sigmas)])

    v_drift_x = _best_v(sigmas_x)
    v_drift_y = _best_v(sigmas_y)

    # ---- Figure 2: sigma vs v_drift scan ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sigmas, v_best, color, proj in [
        (axes2[0], sigmas_x, v_drift_x, 'red',  'X'),
        (axes2[1], sigmas_y, v_drift_y, 'blue', 'Y'),
    ]:
        ax.plot(v_values, sigmas, '-', color=color, lw=1.8)
        if np.isfinite(v_best):
            sigma_best = sigmas[np.nanargmin(sigmas)]
            ax.axvline(v_best, color='k', ls='--', lw=1.2,
                       label=f'best v_drift = {v_best:.1f} µm/ns\nσ = {sigma_best:.3f}°')
            ax.legend(fontsize=9)
        ax.set_xlabel('Drift velocity [µm/ns]')
        ax.set_ylabel('Gaussian σ of diagonal projection [deg]')
        ax.set_title(f'{proj}: diagonal projection width vs drift velocity\n'
                     f'(r < {residual_cut_mm:.0f} mm, {int(qmask_x.sum() if proj=="X" else qmask_y.sum()):,} events)')
        ax.grid(True, alpha=0.3)
    fig2.tight_layout()

    # ---- Figure 3 + 4: corrected correlation and angular resolution ----
    v_avg = float(np.nanmean([v_drift_x, v_drift_y]))
    if not np.isfinite(v_avg):
        return v_drift_x, v_drift_y

    deg_det_x_corr = _deg_from_slope(sl_x_q, v_avg)
    deg_det_y_corr = _deg_from_slope(sl_y_q, v_avg)

    ang_res_data = []
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, deg_ref, deg_det_c, lim, color, proj in [
        (axes3[0], ref_x_q, deg_det_x_corr, lim_x, 'red',  'X'),
        (axes3[1], ref_y_q, deg_det_y_corr, lim_y, 'blue', 'Y'),
    ]:
        iv = (np.abs(deg_det_c) <= lim) & (np.abs(deg_ref) <= lim) \
             & np.isfinite(deg_det_c) & np.isfinite(deg_ref)
        ax.scatter(deg_det_c[iv], deg_ref[iv], alpha=0.3, s=4, color=color)

        xs = np.array([-lim, lim])
        ax.plot(xs, xs, 'k--', lw=1.2, label='y = x  (perfect correlation)')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'θ_{proj.lower()} detector [deg]  (v_drift={v_avg:.1f} µm/ns)')
        ax.set_ylabel(f'θ_{proj.lower()} reference [deg]')
        ax.set_title(f'{proj} corrected correlation  v_drift={v_avg:.1f} µm/ns\n'
                     f'(r < {residual_cut_mm:.0f} mm,  {int(iv.sum()):,} events)')
        ax.legend(fontsize=8)

        # Collect data for combined angular resolution plot
        deg_resid = deg_ref[iv] - deg_det_c[iv]   # d = 0 for perfect correlation
        ang_res_data.append((deg_ref[iv], deg_resid, lim, color, proj))

    # Angular resolution: both X and Y on one plot
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    lim_combined = max(d[2] for d in ang_res_data) if ang_res_data else 40
    for deg_ref_i, deg_resid_i, lim_i, color_i, proj_i in ang_res_data:
        plot_angular_resolution(deg_ref_i, deg_resid_i, lim_i, color_i, proj_i, ax=ax4)
    ax4.set_xlim(-lim_combined, lim_combined)
    ax4.set_title('Angular resolution vs. track angle')
    ax4.legend()
    fig4.tight_layout()

    fig3.tight_layout()
    return v_drift_x, v_drift_y


def plot_position_correlation(results: List[EventResult]) -> None:
    det_x = np.array([r.det_x_for_residual for r in results])
    det_y = np.array([r.det_y_for_residual for r in results])
    ref_x = np.array([r.ref_x_mm for r in results])
    ref_y = np.array([r.ref_y_mm for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(det_x, ref_x, alpha=0.4, s=6, color='red')
    axes[0].set_xlabel('Detector X [mm]')
    axes[0].set_ylabel('Reference X [mm]')
    axes[0].set_title('X position correlation')
    _add_diagonal(axes[0])

    axes[1].scatter(det_y, ref_y, alpha=0.4, s=6, color='blue')
    axes[1].set_xlabel('Detector Y [mm]')
    axes[1].set_ylabel('Reference Y [mm]')
    axes[1].set_title('Y position correlation')
    _add_diagonal(axes[1])

    fig.tight_layout()


def fit_residual_peak(
    data: np.ndarray,
    *,
    n_bins_initial: int = 200,
    window_nsigma: float = 2.5,
    max_iterations: int = 8,
    convergence_tol: float = 0.01,
    min_bins_in_window: int = 10,
    min_counts_in_window: int = 50,
) -> Optional[GaussFitResult]:
    """
    Robust iterative Gaussian fit to the central peak of a residual distribution
    that has large non-Gaussian tails.

    Algorithm
    ---------
    1. Seed: estimate the peak centre from the histogram mode (most-populated bin),
       and seed σ from the IQR / 1.35 (robust scale estimate, insensitive to tails).
    2. Iterate:
       a. Define a symmetric window  [μ − window_nsigma·σ,  μ + window_nsigma·σ].
       b. Re-bin the data within that window to have ~sqrt(N_window) bins, but at
          least min_bins_in_window and always an odd number so the centre is clean.
       c. Fit a Gaussian to the binned data inside the window.
       d. Update (μ, σ) from the fit.  Convergence is declared when both shift by
          less than convergence_tol * σ between iterations.
    3. Return GaussFitResult with fit parameters, window, and diagnostics.

    Parameters
    ----------
    data : 1-D array of residuals (NaNs silently dropped)
    n_bins_initial : bins for the seed histogram (covers full data range)
    window_nsigma : half-width of the fitting window in units of current σ
    max_iterations : hard cap on iterations
    convergence_tol : fractional σ change threshold to declare convergence
    min_bins_in_window : lower bound on bins inside the window
    min_counts_in_window : minimum events inside window to attempt a fit

    Returns
    -------
    GaussFitResult, or None if the fit could not be made.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < min_counts_in_window:
        return None

    # ---- Seed estimates ----
    hist0, edges0 = np.histogram(data, bins=n_bins_initial)
    centres0 = 0.5 * (edges0[:-1] + edges0[1:])
    mu = float(centres0[np.argmax(hist0)])          # mode → peak centre seed
    q25, q75 = np.percentile(data, [25, 75])
    sigma = float((q75 - q25) / 1.35)              # IQR-based σ seed
    if sigma < 1e-6:
        sigma = float(np.std(data))
    if sigma < 1e-6:
        return None

    popt_final = None
    pcov_final = None
    window_lo = window_hi = bin_width = 0.0
    n_iters = 0
    converged = False

    for iteration in range(max_iterations):
        n_iters = iteration + 1

        # ---- Define window ----
        window_lo = mu - window_nsigma * sigma
        window_hi = mu + window_nsigma * sigma
        in_window = data[(data >= window_lo) & (data <= window_hi)]

        if len(in_window) < min_counts_in_window:
            break

        # ---- Choose bin count: ~sqrt(N), clamped, always odd ----
        n_bins_window = max(min_bins_in_window, int(np.sqrt(len(in_window))))
        if n_bins_window % 2 == 0:
            n_bins_window += 1

        hist_w, edges_w = np.histogram(in_window, bins=n_bins_window,
                                       range=(window_lo, window_hi))
        centres_w = 0.5 * (edges_w[:-1] + edges_w[1:])
        bin_width = float(edges_w[1] - edges_w[0])

        # Poisson errors; floor at 1 to avoid zero-weight bins
        errors_w = np.where(hist_w > 0, np.sqrt(hist_w), 1.0)

        p0 = [float(np.max(hist_w)), mu, sigma]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, pcov = cf(
                    gaus, centres_w, hist_w,
                    p0=p0, sigma=errors_w, absolute_sigma=True,
                    bounds=([0, window_lo, 0], [np.inf, window_hi, window_hi - window_lo]),
                )
        except (RuntimeError, ValueError):
            break

        mu_new = float(popt[1])
        sigma_new = abs(float(popt[2]))

        # ---- Convergence check ----
        delta_mu = abs(mu_new - mu)
        delta_sigma = abs(sigma_new - sigma)
        mu, sigma = mu_new, sigma_new
        popt_final, pcov_final = popt, pcov

        if delta_mu < convergence_tol * sigma and delta_sigma < convergence_tol * sigma:
            converged = True
            break

    if popt_final is None:
        return None

    perr = np.sqrt(np.diag(pcov_final))
    in_window_final = data[(data >= window_lo) & (data <= window_hi)]
    n_bins_final = max(min_bins_in_window, int(np.sqrt(len(in_window_final))))

    return GaussFitResult(
        amplitude=float(popt_final[0]),
        amplitude_err=float(perr[0]),
        mean=float(popt_final[1]),
        mean_err=float(perr[1]),
        sigma=abs(float(popt_final[2])),
        sigma_err=float(perr[2]),
        fit_window_lo=float(window_lo),
        fit_window_hi=float(window_hi),
        bin_width=float(bin_width),
        n_bins_used=n_bins_final,
        n_iterations=n_iters,
        converged=converged,
    )


def plot_residuals(
    results: List[EventResult],
    n_bins_initial: int = 200,
    window_nsigma: float = 2.5,
    overview_nsigma: float = 6.0,
    plot_full_range: bool = True,
) -> Tuple[Optional[GaussFitResult], Optional[GaussFitResult]]:
    """
    Fit and plot the residual distributions for X and Y.

    Two panels per axis are shown:
      - Left column: full distribution with the fit window indicated
      - Right column: zoom into the fit window with the Gaussian overlay

    Parameters
    ----------
    results : list of EventResult objects with residuals filled
    n_bins_initial : initial bin count for seeding (covers full range)
    window_nsigma : half-window for fitting, in units of fitted σ
    overview_nsigma : half-width of the "overview" zoom panel, in units of σ
    plot_full_range : if True, also show the full-range histogram (log scale)

    Returns
    -------
    (fit_x, fit_y) — GaussFitResult for each axis, or None if fit failed.
    """
    dx = np.array([r.residual_x_mm for r in results if r.has_x and not np.isnan(r.residual_x_mm)])
    dy = np.array([r.residual_y_mm for r in results if r.has_y and not np.isnan(r.residual_y_mm)])

    fits = {}
    for data, axis, color in [(dx, 'X', 'red'), (dy, 'Y', 'blue')]:
        fit = fit_residual_peak(data, n_bins_initial=n_bins_initial,
                                window_nsigma=window_nsigma)
        fits[axis] = fit
        print(f'{axis} residual fit: {fit}')

    ncols = 3 if plot_full_range else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9))

    for row, (data, axis, color) in enumerate([(dx, 'X', 'red'), (dy, 'Y', 'blue')]):
        fit = fits[axis]
        col = 0

        # ---- Panel 1: full range, log scale ----
        if plot_full_range:
            ax = axes[row, col]
            hist_full, edges_full = np.histogram(data, bins=n_bins_initial)
            centres_full = 0.5 * (edges_full[:-1] + edges_full[1:])
            ax.bar(centres_full, hist_full, width=edges_full[1] - edges_full[0],
                   color=color, alpha=0.6, label='All events')
            ax.set_yscale('log')
            ax.set_xlabel(f'{axis} residual (ref − det) [mm]')
            ax.set_ylabel('Events (log)')
            ax.set_title(f'{axis}: full distribution ({len(data):,} events)')
            if fit is not None:
                ax.axvspan(fit.fit_window_lo, fit.fit_window_hi, alpha=0.15,
                           color='gold', label=f'Fit window (±{window_nsigma}σ)')
                ax.axvline(fit.mean, color='black', lw=1, linestyle='--')
            ax.legend(fontsize=8)
            col += 1

        # ---- Panel 2: overview zoom (±overview_nsigma · σ) ----
        ax = axes[row, col]
        if fit is not None:
            lo_ov = fit.mean - overview_nsigma * fit.sigma
            hi_ov = fit.mean + overview_nsigma * fit.sigma
            in_ov = data[(data >= lo_ov) & (data <= hi_ov)]
            n_bins_ov = max(40, int(np.sqrt(len(in_ov))))
            hist_ov, edges_ov = np.histogram(in_ov, bins=n_bins_ov, range=(lo_ov, hi_ov))
            centres_ov = 0.5 * (edges_ov[:-1] + edges_ov[1:])
            ax.bar(centres_ov, hist_ov, width=edges_ov[1] - edges_ov[0],
                   color=color, alpha=0.6)
            ax.axvspan(fit.fit_window_lo, fit.fit_window_hi, alpha=0.15,
                       color='gold', label=f'Fit window')
            ax.set_xlabel(f'{axis} residual (ref − det) [mm]')
            ax.set_ylabel('Events')
            ax.set_title(f'{axis}: overview (±{overview_nsigma:.0f}σ region)')
            ax.legend(fontsize=8)
        else:
            n_bins_ov = 80
            hist_ov, edges_ov = np.histogram(data, bins=n_bins_ov)
            centres_ov = 0.5 * (edges_ov[:-1] + edges_ov[1:])
            ax.bar(centres_ov, hist_ov, width=edges_ov[1] - edges_ov[0],
                   color=color, alpha=0.6)
            ax.set_title(f'{axis}: overview (fit failed)')
        col += 1

        # ---- Panel 3: fit window zoom with Gaussian overlay ----
        ax = axes[row, col]
        if fit is not None:
            in_win = data[(data >= fit.fit_window_lo) & (data <= fit.fit_window_hi)]
            hist_win, edges_win = np.histogram(in_win, bins=fit.n_bins_used,
                                               range=(fit.fit_window_lo, fit.fit_window_hi))
            centres_win = 0.5 * (edges_win[:-1] + edges_win[1:])
            errs_win = np.where(hist_win > 0, np.sqrt(hist_win), 1.0)

            ax.errorbar(centres_win, hist_win, yerr=errs_win,
                        fmt='o', color=color, ms=4, lw=1, zorder=3, label='Data')

            x_fine = np.linspace(fit.fit_window_lo, fit.fit_window_hi, 400)
            ax.plot(x_fine, gaus(x_fine, fit.amplitude, fit.mean, fit.sigma),
                    color='black', lw=2, zorder=4, label='Gaussian fit')

            fit_str = (f'μ = {fit.mean:.2f} ± {fit.mean_err:.2f} mm\n'
                       f'σ = {fit.resolution:.2f} ± {fit.resolution_err:.2f} mm\n'
                       f'iters = {fit.n_iterations}'
                       + ('' if fit.converged else ' (not converged)'))
            ax.annotate(fit_str, xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray'))
            ax.set_xlabel(f'{axis} residual (ref − det) [mm]')
            ax.set_ylabel('Events')
            ax.set_title(f'{axis}: fit window  σ = {fit.resolution:.2f} mm')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f'{axis}: fit window (no result)')

    fig.suptitle('Residual distributions — robust Gaussian peak fit', fontsize=13)
    fig.tight_layout()
    return fits.get('X'), fits.get('Y')


# ---------------------------------------------------------------------------
# Radial residual diagnostic plot
# ---------------------------------------------------------------------------

def plot_radial_residuals(
    results: List[EventResult],
    radius_cut_mm: float = 5.0,
    n_bins: int = 100,
) -> None:
    """
    Plot the distribution of radial residuals and print matching statistics.

    Useful for diagnosing why is_efficient(radius_cut_mm) may always return False:
      - If "events with M3 match" is 0, the event IDs don't overlap.
      - If the distribution peaks far above radius_cut_mm, the alignment is off.
    """
    n_no_hit = sum(1 for r in results if not r.has_both)
    radials, n_no_ref = [], 0
    for r in results:
        if not r.has_both:
            continue
        rr = r.radial_residual_mm
        if np.isnan(rr):
            n_no_ref += 1
        else:
            radials.append(rr)

    radials = np.array(radials)
    n_matched = len(radials)
    n_within = int(np.sum(radials <= radius_cut_mm)) if n_matched else 0

    print('\n--- Radial residual diagnostics ---')
    print(f'  Total events:                    {len(results)}')
    print(f'  Events without valid hit:        {n_no_hit}')
    print(f'  has_both, no M3 match (ref NaN): {n_no_ref}')
    print(f'  has_both, with M3 match:         {n_matched}')
    if n_matched:
        print(f'  Median radial residual:          {np.median(radials):.2f} mm')
        print(f'  Mean   radial residual:          {np.mean(radials):.2f} mm')
        print(f'  Within {radius_cut_mm:.1f} mm cut:           '
              f'{n_within}/{n_matched} ({100*n_within/n_matched:.1f}%)')
    print('-----------------------------------\n')

    if n_matched == 0:
        print('  No matched events — cannot plot radial residuals.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(radials, bins=n_bins, color='steelblue', alpha=0.7)
    axes[0].axvline(radius_cut_mm, color='red', lw=2, ls='--',
                    label=f'cut = {radius_cut_mm:.1f} mm')
    axes[0].set_xlabel('Radial residual [mm]')
    axes[0].set_ylabel('Events')
    axes[0].set_title(f'Radial residual — full range\n({n_matched:,} matched events)')
    axes[0].legend()

    zoom_max = max(3 * radius_cut_mm, float(np.percentile(radials, 90)))
    in_zoom = radials[radials <= zoom_max]
    axes[1].hist(in_zoom, bins=n_bins, color='steelblue', alpha=0.7)
    axes[1].axvline(radius_cut_mm, color='red', lw=2, ls='--',
                    label=f'cut = {radius_cut_mm:.1f} mm  '
                          f'({n_within} pass, {100*n_within/n_matched:.1f}%)')
    axes[1].set_xlabel('Radial residual [mm]')
    axes[1].set_ylabel('Events')
    axes[1].set_title(f'Radial residual — zoom to {zoom_max:.0f} mm')
    axes[1].legend()

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_strip_positions(df: pd.DataFrame, det) -> pd.DataFrame:
    """Add x_position_mm and y_position_mm columns by mapping (feu, channel)."""
    all_x, all_y = [], []
    for feu, channel in zip(df['feu'].to_numpy(), df['channel'].to_numpy()):
        pos = det.map_hit(feu, channel)
        if pos is not None:
            all_x.append(pos[0])
            all_y.append(pos[1])
        else:
            all_x.append(None)
            all_y.append(None)
    df = df.copy()
    df['x_position_mm'] = all_x
    df['y_position_mm'] = all_y
    return df


def _line_anchored(pos, slope, pos_anchor, time_anchor):
    """Line through a fixed point:  t = slope * (pos - pos_anchor) + time_anchor."""
    return slope * (pos - pos_anchor) + time_anchor


def _add_diagonal(ax):
    """Draw y=x reference line on a correlation plot."""
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.4, lw=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)


def _plot_event(df_event: pd.DataFrame, result: EventResult, event_id: int) -> None:
    """Diagnostic plot: strip position vs. hit time for one event."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, pos_col, fit, color, label in [
        (axes[0], 'x_position_mm', result.x_fit, 'red', 'X'),
        (axes[1], 'y_position_mm', result.y_fit, 'blue', 'Y'),
    ]:
        df_ax = df_event[df_event[pos_col].notna()]
        sc = ax.scatter(df_ax[pos_col], df_ax['time'],
                        c=df_ax['amplitude'], cmap='jet', alpha=0.8, zorder=3)

        if fit is not None and fit.is_valid():
            pos_range = np.linspace(df_ax[pos_col].min(), df_ax[pos_col].max(), 200)
            t_line = _line_anchored(pos_range, fit.slope_ns_per_mm,
                                    fit.mesh_position_mm, fit.earliest_time_ns)
            ax.plot(pos_range, t_line, color=color, lw=1.5,
                    label=f'slope={fit.slope_ns_per_mm:.3f} ns/mm\nχ²/dof={fit.red_chi2:.2f}')
            ax.axvline(fit.mesh_position_mm, color=color, linestyle=':', alpha=0.6,
                       label=f'mesh pos={fit.mesh_position_mm:.1f} mm')
            ax.legend(fontsize=8)

        ax.set_xlabel(f'{label} position [mm]')
        ax.set_ylabel('Time [ns]')
        ax.set_title(f'Event {event_id} – {label} strips')

    fig.tight_layout()


def plot_event_display_3d(
    df_event: pd.DataFrame,
    result: EventResult,
    params: AlignmentParams,
    v_drift_um_per_ns: float,
    event_id: int,
) -> None:
    """
    3-D event display for a single cosmic muon event.

    Coordinate system (detector local)
    -----------------------------------
    x  — strip position from X strips [mm]
    y  — strip position from Y strips [mm]
    z  — drift distance from the mesh [mm]
         z = (t - t0) * v_drift_um_per_ns / 1000
         where t0 is the earliest hit time across both axes.

    Hit reconstruction
    ------------------
    Because a single straight track produces one cluster on X strips and one
    on Y strips, we pair them using the fitted line from the complementary axis:

    • X-strip hits → x is directly measured; y is predicted from the Y fit at
      the same drift time → 3-D point: (x_measured, y_predicted, z).
    • Y-strip hits → y is directly measured; x is predicted from the X fit at
      the same drift time → 3-D point: (x_predicted, y_measured, z).

    This avoids nearest-neighbour matching and is exact for a straight track.

    Reference track
    ---------------
    Drawn in the same local frame using the M3 angles stored in EventResult:
        x_track(z) = ref_mesh_x_mm + z * ref_tan_theta_x
        y_track(z) = ref_mesh_y_mm + z * ref_tan_theta_y
    where ref_mesh_x_mm / ref_mesh_y_mm are the reference track positions at
    the mesh in raw detector coordinates (inverse-alignment of ref_x_mm /
    ref_y_mm).

    Parameters
    ----------
    df_event          : DataFrame of hits for this event (all strips, both axes)
    result            : EventResult with valid x_fit, y_fit and ref angles
    params            : AlignmentParams (z_x, z_y used in subtitle)
    v_drift_um_per_ns : calibrated drift velocity [µm/ns]
    event_id          : event number (for the plot title)
    """
    if not result.has_both:
        print(f'Event {event_id}: missing X or Y fit — cannot make 3-D display.')
        return

    x_fit = result.x_fit
    y_fit = result.y_fit

    # ---- Time → drift-distance conversion ----
    t0 = min(x_fit.earliest_time_ns, y_fit.earliest_time_ns)

    def _z(t_ns: np.ndarray) -> np.ndarray:
        return (np.asarray(t_ns, dtype=float) - t0) * v_drift_um_per_ns / 1000.0

    # ---- Split hits by axis ----
    df_x = df_event[df_event['x_position_mm'].notna()].copy()
    df_y = df_event[df_event['y_position_mm'].notna()].copy()

    x_meas = df_x['x_position_mm'].values
    t_x    = df_x['time'].values
    amp_x  = df_x['amplitude'].values
    z_x    = _z(t_x)

    y_meas = df_y['y_position_mm'].values
    t_y    = df_y['time'].values
    amp_y  = df_y['amplitude'].values
    z_y    = _z(t_y)

    # ---- Predict complementary coordinate from the fitted line ----
    # Y fit: t = slope_ns_per_mm * (y - y_anchor) + t_y_anchor
    # → y(t) = y_anchor + (t - t_y_anchor) * slope_mm_per_ns
    y_pred_at_x = (y_fit.mesh_position_mm
                   + (t_x - y_fit.earliest_time_ns) * y_fit.slope_mm_per_ns)
    x_pred_at_y = (x_fit.mesh_position_mm
                   + (t_y - x_fit.earliest_time_ns) * x_fit.slope_mm_per_ns)

    # ---- Reference track in detector-local coordinates ----
    z_max_track = max(x_fit.cluster_duration_ns, y_fit.cluster_duration_ns) * v_drift_um_per_ns / 1000.0
    z_track = np.linspace(0.0, z_max_track, 200)
    x_track = result.ref_mesh_x_mm + z_track * result.ref_tan_theta_x
    y_track = result.ref_mesh_y_mm + z_track * result.ref_tan_theta_y

    # ---- Plot ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # X-strip hits: x measured, y predicted
    sc_x = ax.scatter(x_meas, y_pred_at_x, z_x,
                      c=amp_x, cmap='Reds', s=50, alpha=0.85, zorder=4,
                      label='X strips (x meas., y pred.)')

    # Y-strip hits: x predicted, y measured
    sc_y = ax.scatter(x_pred_at_y, y_meas, z_y,
                      c=amp_y, cmap='Blues', s=50, alpha=0.85, zorder=4,
                      label='Y strips (x pred., y meas.)')

    # Reference track
    ax.plot(x_track, y_track, z_track,
            color='green', lw=2.5, zorder=5, label='Reference track (M3)')

    # Colorbars
    plt.colorbar(sc_x, ax=ax, label='Amplitude (X strips)', shrink=0.5, pad=0.1)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Drift distance [mm]')
    ax.set_title(
        f'Event {event_id} — 3-D display\n'
        f'v_drift = {v_drift_um_per_ns:.1f} µm/ns  |  '
        f'z_x = {params.z_x:.0f} mm,  z_y = {params.z_y:.0f} mm'
    )
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()


def plot_event_display_3d_rotating(
    df_event: pd.DataFrame,
    result: EventResult,
    params: AlignmentParams,
    v_drift_um_per_ns: float,
    event_id: int,
    drift_window_mm: float = 30.0,
    gif_path: Optional[str] = None,
    gif_fps: int = 20,
    gif_frames: int = 180,
) -> None:
    """
    Clean 3-D event display with amplitude-scaled point sizes, horizontal
    planes marking the mesh (z = 0) and drift window (z = drift_window_mm),
    no grid, and an optional rotating GIF output.

    Points are coloured by axis (X = crimson, Y = steelblue) with size
    proportional to strip amplitude.  No colourbar is shown.

    Parameters
    ----------
    df_event          : DataFrame of hits for this event
    result            : EventResult with valid x_fit, y_fit and ref angles
    params            : AlignmentParams (used in subtitle)
    v_drift_um_per_ns : calibrated drift velocity [µm/ns]
    event_id          : event number (for the plot title)
    drift_window_mm   : z position of the top horizontal plane [mm]
    gif_path          : if given, save a rotating GIF to this path
    gif_fps           : frames per second for the GIF
    gif_frames        : number of frames (360 / gif_frames degrees per frame)
    """
    from matplotlib.animation import FuncAnimation

    if not result.has_both:
        print(f'Event {event_id}: missing X or Y fit — cannot make 3-D display.')
        return

    x_fit = result.x_fit
    y_fit = result.y_fit

    # ---- Time → drift-distance conversion ----
    t0 = min(x_fit.earliest_time_ns, y_fit.earliest_time_ns)

    def _z(t_ns: np.ndarray) -> np.ndarray:
        return (np.asarray(t_ns, dtype=float) - t0) * v_drift_um_per_ns / 1000.0

    # ---- Split hits by axis ----
    df_x = df_event[df_event['x_position_mm'].notna()].copy()
    df_y = df_event[df_event['y_position_mm'].notna()].copy()

    x_meas = df_x['x_position_mm'].values
    t_x    = df_x['time'].values
    amp_x  = df_x['amplitude'].values
    z_x    = _z(t_x)

    y_meas = df_y['y_position_mm'].values
    t_y    = df_y['time'].values
    amp_y  = df_y['amplitude'].values
    z_y    = _z(t_y)

    y_pred_at_x = (y_fit.mesh_position_mm
                   + (t_x - y_fit.earliest_time_ns) * y_fit.slope_mm_per_ns)
    x_pred_at_y = (x_fit.mesh_position_mm
                   + (t_y - x_fit.earliest_time_ns) * x_fit.slope_mm_per_ns)

    # ---- Reference track ----
    z_max_track = max(x_fit.cluster_duration_ns, y_fit.cluster_duration_ns) * v_drift_um_per_ns / 1000.0
    z_track = np.linspace(0.0, z_max_track, 200)
    x_track = result.ref_mesh_x_mm + z_track * result.ref_tan_theta_x
    y_track = result.ref_mesh_y_mm + z_track * result.ref_tan_theta_y

    # ---- Amplitude → point size (20–200 range) ----
    all_amp = np.concatenate([amp_x, amp_y])
    a_min, a_max = all_amp.min(), all_amp.max()
    def _size(amp):
        if a_max == a_min:
            return np.full_like(amp, 80.0, dtype=float)
        return 20.0 + 180.0 * (amp - a_min) / (a_max - a_min)

    # ---- Horizontal plane extent (padded by 10 mm each side) ----
    all_x = np.concatenate([x_meas, x_pred_at_y, x_track])
    all_y = np.concatenate([y_pred_at_x, y_meas, y_track])
    pad = 10.0
    xlo, xhi = all_x.min() - pad, all_x.max() + pad
    ylo, yhi = all_y.min() - pad, all_y.max() + pad
    xx_p, yy_p = np.meshgrid([xlo, xhi], [ylo, yhi])

    # ---- Figure ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Mesh plane at z = 0
    ax.plot_surface(xx_p, yy_p, np.zeros_like(xx_p),
                    alpha=0.18, color='silver', zorder=1, linewidth=0)
    ax.text(xhi, yhi, 0.0, 'Mesh (z = 0)', fontsize=7, color='grey')

    # Drift window plane at z = drift_window_mm
    ax.plot_surface(xx_p, yy_p, np.full_like(xx_p, drift_window_mm),
                    alpha=0.12, color='lightsteelblue', zorder=1, linewidth=0)
    ax.text(xhi, yhi, drift_window_mm,
            f'Drift window ({drift_window_mm:.0f} mm)', fontsize=7, color='steelblue')

    # X-strip hits: crimson, amplitude-sized
    ax.scatter(x_meas, y_pred_at_x, z_x,
               color='crimson', s=_size(amp_x), alpha=0.85, zorder=4,
               label='X strips (x meas., y pred.)', edgecolors='none')

    # Y-strip hits: steelblue, amplitude-sized
    ax.scatter(x_pred_at_y, y_meas, z_y,
               color='steelblue', s=_size(amp_y), alpha=0.85, zorder=4,
               label='Y strips (x pred., y meas.)', edgecolors='none')

    # Reference track
    ax.plot(x_track, y_track, z_track,
            color='limegreen', lw=2.5, zorder=5, label='Reference track (M3)')

    # ---- Clean up panes and grid ----
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('lightgrey')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Drift distance [mm]')
    ax.set_zlim(0, max(drift_window_mm, z_max_track * 1.05))
    ax.set_title(
        f'Event {event_id}  |  v_drift = {v_drift_um_per_ns:.1f} µm/ns',
        fontsize=11,
    )
    ax.legend(loc='upper left', fontsize=9)

    # ---- Static view ----
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()

    # ---- Rotating GIF ----
    if gif_path is not None:
        def _animate(frame):
            ax.view_init(elev=20, azim=frame * (360.0 / gif_frames))
            return []

        anim = FuncAnimation(fig, _animate, frames=gif_frames,
                             interval=1000 // gif_fps, blit=False)
        try:
            anim.save(gif_path, writer='pillow', fps=gif_fps)
            print(f'Rotating GIF saved → {gif_path}')
        except Exception as exc:
            print(f'Could not save GIF ({exc}). Is Pillow installed?')


def select_display_event(
    results: List[EventResult],
    v_drift_um_per_ns: float,
    max_spatial_residual_mm: float = 3.0,
    max_angle_residual: float = 0.05,
    max_red_chi2: float = 3.0,
    min_strips: int = 4,
    x_max_mm: Optional[float] = None,
    x_min_mm: Optional[float] = None,
    y_max_mm: Optional[float] = None,
    y_min_mm: Optional[float] = None,
) -> Optional[int]:
    """
    Select the best event for a 3-D event display from a list of EventResults.

    "Best" means lowest combined score on four quality metrics, all evaluated
    with the calibrated drift velocity:

    1. Spatial X residual   : |ref_x − det_x_aligned|
    2. Spatial Y residual   : |ref_y − det_y_aligned|
    3. X angle residual     : |tan(θ_x)_det  − tan(θ_x)_ref|
    4. Y angle residual     : |tan(θ_y)_det  − tan(θ_y)_ref|

    Events failing the hard cuts (max_spatial_residual_mm, max_angle_residual,
    max_red_chi2, min_strips, position bounds) are excluded entirely.

    Parameters
    ----------
    results                : list of EventResult objects after alignment
    v_drift_um_per_ns      : calibrated drift velocity [µm/ns]
    max_spatial_residual_mm: hard cut on |residual_x| and |residual_y| [mm]
    max_angle_residual     : hard cut on |Δtan(θ)| for both X and Y
    max_red_chi2           : hard cut on reduced χ² for both fits
    min_strips             : minimum strips required on each axis
    x_max_mm               : upper bound on det_x_aligned_mm (left-side cut)
    x_min_mm               : lower bound on det_x_aligned_mm
    y_max_mm               : upper bound on det_y_aligned_mm
    y_min_mm               : lower bound on det_y_aligned_mm

    Returns
    -------
    event_id of the best matching event, or None if nothing passes.
    """
    candidates = []

    for r in results:
        # ---- Basic validity ----
        if not r.has_both:
            continue
        if r.x_fit.n_strips < min_strips or r.y_fit.n_strips < min_strips:
            continue
        if max_red_chi2 and (r.x_fit.red_chi2 > max_red_chi2 or r.y_fit.red_chi2 > max_red_chi2):
            continue

        # ---- Spatial residuals ----
        rx = r.residual_x_mm
        ry = r.residual_y_mm
        if np.isnan(rx) or np.isnan(ry):
            continue
        if abs(rx) > max_spatial_residual_mm or abs(ry) > max_spatial_residual_mm:
            continue

        # ---- Angle residuals using calibrated v_drift ----
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_tan_theta_y):
            continue
        det_tan_x = r.x_fit.slope_mm_per_ns * 1000.0 / v_drift_um_per_ns
        det_tan_y = r.y_fit.slope_mm_per_ns * 1000.0 / v_drift_um_per_ns
        dtan_x = det_tan_x - r.ref_tan_theta_x
        dtan_y = det_tan_y - r.ref_tan_theta_y
        if abs(dtan_x) > max_angle_residual or abs(dtan_y) > max_angle_residual:
            continue

        # ---- Position bounds (left/working side of detector) ----
        x_det = r.det_x_for_residual
        y_det = r.det_y_for_residual
        if x_max_mm is not None and x_det > x_max_mm:
            continue
        if x_min_mm is not None and x_det < x_min_mm:
            continue
        if y_max_mm is not None and y_det > y_max_mm:
            continue
        if y_min_mm is not None and y_det < y_min_mm:
            continue

        # ---- Composite score: sum of normalised squared deviations ----
        score = (
            (rx / max_spatial_residual_mm) ** 2
            + (ry / max_spatial_residual_mm) ** 2
            + (dtan_x / max_angle_residual) ** 2
            + (dtan_y / max_angle_residual) ** 2
        )
        candidates.append((score, r.event_id))

    if not candidates:
        print('select_display_event: no events passed all cuts.')
        return None

    candidates.sort()
    best_score, best_id = candidates[0]
    print(f'select_display_event: {len(candidates)} candidates; '
          f'best event {best_id}  score={best_score:.4f}')
    return best_id


# ---------------------------------------------------------------------------
# Utility functions kept from original script
# ---------------------------------------------------------------------------

def gaus(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def make_percentile_cuts(data, percentile_cuts=(None, None), return_what='data'):
    if len(data) == 0:
        return data
    if percentile_cuts[0] is not None and percentile_cuts[1] is not None:
        lo = np.nanpercentile(data, percentile_cuts[0])
        hi = np.nanpercentile(data, percentile_cuts[1])
        mask = (data > lo) & (data < hi)
    elif percentile_cuts[0] is not None:
        lo = np.nanpercentile(data, percentile_cuts[0])
        mask = data > lo
    elif percentile_cuts[1] is not None:
        hi = np.nanpercentile(data, percentile_cuts[1])
        mask = data < hi
    else:
        return data
    return mask if return_what == 'filter' else data[mask]


def fit_time_diffs(time_diffs, n_bins=100, min_events=100, nsigma_filter=None, return_hist=False):
    time_diffs = np.array(time_diffs)
    time_diffs = time_diffs[~np.isnan(time_diffs)]

    if nsigma_filter is not None:
        median, std = np.median(time_diffs), np.std(time_diffs)
        time_diffs = time_diffs[
            (time_diffs > median - nsigma_filter * std) &
            (time_diffs < median + nsigma_filter * std)
        ]

    meases = [Measure(np.nan, np.nan) for _ in range(3)]
    if time_diffs.size < min_events:
        return (meases, None, None, None) if return_hist else meases

    hist, bin_edges = np.histogram(time_diffs, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_err = np.where(hist > 0, np.sqrt(hist), 1)

    try:
        p0 = [np.max(hist), np.mean(time_diffs), np.std(time_diffs)]
        popt, pcov = cf(gaus, bin_centers, hist, p0=p0, sigma=hist_err, absolute_sigma=True)
        popt[2] = abs(popt[2])

        if nsigma_filter is not None:
            mask = (time_diffs > popt[1] - nsigma_filter * popt[2]) & \
                   (time_diffs < popt[1] + nsigma_filter * popt[2])
            time_diffs = time_diffs[mask]
            hist, bin_edges = np.histogram(time_diffs, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_err = np.where(hist > 0, np.sqrt(hist), 1)
            p0 = [np.max(hist), np.mean(time_diffs), np.std(time_diffs)]
            popt, pcov = cf(gaus, bin_centers, hist, p0=p0, sigma=hist_err, absolute_sigma=True)
            popt[2] = abs(popt[2])

        perr = np.sqrt(np.diag(pcov))
        meases = [Measure(v, e) for v, e in zip(popt, perr)]
    except RuntimeError:
        pass

    return (meases, hist, bin_centers, hist_err) if return_hist else meases


def extract_file_numbers_tuple(filename: str) -> Optional[Tuple[int, int]]:
    match = re.match(r'.*_(\d{3})_(\d{2})*', filename)
    if match:
        return tuple(int(x) for x in match.groups())
    return None


if __name__ == '__main__':
    main()
