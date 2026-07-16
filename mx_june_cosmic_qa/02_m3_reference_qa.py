#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_m3_reference_qa.py

Step 2 of the June cosmic-bench QA: check the M3 reference tracker on its own,
independent of the detector under test.  Reuses M3RefTracking from
cosmic_bench_analysis/.

Products (written to output/<run>/m3_reference_qa/):
  - m3_chi2_distributions.png   Chi2X / Chi2Y (raw, pre-single-track-cut)
  - m3_track_multiplicity.png   good tracks per event
  - m3_angles.png               theta_x / theta_y of clean single tracks
  - m3_beam_profile_detz.png    interpolated (x,y) at the detector plane
  - m3_up_down_positions.png    hit maps at the up + down tracker stations

The detector plane z is read from run_config.json (mx17_1 det_center z).
"""

import json
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import awkward as ak
from M3RefTracking import M3RefTracking, get_ray_data, get_xy_angles

from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)


def _det_plane_z() -> float:
    with open(CFG.run_config_path) as f:
        cfg = json.load(f)
    for d in cfg['detectors']:
        if d['name'] == CFG.DET_NAME:
            return float(d['det_center_coords']['z'])
    return 702.0


def plot_chi2(out_dir):
    """Raw chi2 distributions before any single-track selection."""
    raw = get_ray_data(CFG.m3_tracking_dir)
    chi2x = ak.to_numpy(ak.flatten(raw['Chi2X']))
    chi2y = ak.to_numpy(ak.flatten(raw['Chi2Y']))
    n_tracks = ak.to_numpy(ak.num(raw['Chi2X'], axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, lbl in [(axes[0], chi2x, 'Chi2X'), (axes[1], chi2y, 'Chi2Y')]:
        finite = data[np.isfinite(data)]
        ax.hist(finite[finite < 50], bins=100, color='indianred', edgecolor='none')
        ax.axvline(CHI2_CUT, color='k', ls='--', lw=1, label=f'cut = {CHI2_CUT:g}')
        ax.set_xlabel(lbl)
        ax.set_ylabel('Tracks')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'M3 raw track chi2 — {CFG.RUN}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/m3_chi2_distributions.png', dpi=150, bbox_inches='tight')

    # Track multiplicity
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_tracks, bins=range(0, int(n_tracks.max()) + 2),
            align='left', color='steelblue', edgecolor='white')
    ax.set_xlabel('Reconstructed tracks per event')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    ax.set_title(f'M3 track multiplicity — {CFG.RUN}\n'
                 f'{len(n_tracks):,} events, mean {n_tracks.mean():.2f} tracks/event')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/m3_track_multiplicity.png', dpi=150, bbox_inches='tight')
    return len(n_tracks)


def plot_angles_and_positions(out_dir, det_z):
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    n_clean = len(ak.to_numpy(rays.ray_data['X_Up']))

    # Angles
    x_ang, y_ang, _ = get_xy_angles(rays.ray_data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, lbl in [(axes[0], np.degrees(x_ang), r'$\theta_x$ [deg]'),
                          (axes[1], np.degrees(y_ang), r'$\theta_y$ [deg]')]:
        ax.hist(data, bins=120, range=(-40, 40), color='seagreen', edgecolor='none')
        ax.set_xlabel(lbl)
        ax.set_ylabel('Tracks')
        ax.set_title(f'{lbl}  (median {np.median(data):.2f}, std {np.std(data):.2f})')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'M3 single-track angles — {CFG.RUN}  ({n_clean:,} tracks)')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/m3_angles.png', dpi=150, bbox_inches='tight')

    # Beam profile at the detector plane
    xs, ys, _ = rays.get_xy_positions(det_z)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    h = ax.hist2d(xs, ys, bins=80, range=[[-300, 300], [-300, 300]], cmap='viridis')
    fig.colorbar(h[3], ax=ax, label='Tracks')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_aspect('equal')
    ax.set_title(f'M3 track projection at detector z = {det_z:.0f} mm\n{CFG.RUN}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/m3_beam_profile_detz.png', dpi=150, bbox_inches='tight')

    # Up and down station hit maps
    rd = rays.ray_data
    xu, yu = ak.to_numpy(rd['X_Up']), ak.to_numpy(rd['Y_Up'])
    xd, yd = ak.to_numpy(rd['X_Down']), ak.to_numpy(rd['Y_Down'])
    zu = float(np.mean(ak.to_numpy(rd['Z_Up'])))
    zd = float(np.mean(ak.to_numpy(rd['Z_Down'])))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, xx, yy, z in [(axes[0], xu, yu, zu), (axes[1], xd, yd, zd)]:
        h = ax.hist2d(xx, yy, bins=80, range=[[-300, 300], [-300, 300]], cmap='viridis')
        fig.colorbar(h[3], ax=ax, label='Tracks')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_aspect('equal')
        ax.set_title(f'Station at z = {z:.0f} mm')
    fig.suptitle(f'M3 station hit maps — {CFG.RUN}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/m3_up_down_positions.png', dpi=150, bbox_inches='tight')

    return n_clean


def main():
    out_dir = CFG.out_dir('m3_reference_qa')
    det_z = _det_plane_z()
    print(f'Detector plane z = {det_z:.1f} mm')
    n_events = plot_chi2(out_dir)
    n_clean = plot_angles_and_positions(out_dir, det_z)
    print(f'\nM3 events: {n_events:,}  |  clean single tracks: {n_clean:,} '
          f'({100 * n_clean / max(n_events, 1):.1f}%)')
    print(f'M3 reference QA written to: {out_dir}')


if __name__ == '__main__':
    main()
