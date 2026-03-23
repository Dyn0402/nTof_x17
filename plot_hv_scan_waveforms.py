#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 2026
Created in PyCharm
Created as nTof_x17/plot_hv_scan_waveforms.py

Plot waveforms (and optionally hits) for several subruns at different resist HVs
stitched horizontally so the evolution with voltage is visible side-by-side.

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plot_beam_hits import load_subrun, add_xy_pos
from plot_waveforms import load_decoded_waveforms, load_evtids_timestamps


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'

    feu_nums = {4: 'y', 5: 'x'}

    # Each entry: (run, subrun, file_num, event)
    # Add/remove entries to control how many columns appear.
    subruns = [
        ('run_33',  'resist_380V_drift_600V', 0, 14),
        ('run_33',  'resist_440V_drift_600V', 0, 14),
        ('run_33',  'resist_470V_drift_600V', 0, 14),
        ('run_33',  'resist_510V_drift_600V', 0, 14),
    ]

    show_orientations = 'y'  # 'both', 'x', 'y'

    xlim_us = (-0.5, 7)       # µs — applied to every column
    min_hit_amp = 400
    decoded_dir = 'decoded_root'

    plot_hv_scan_waveforms_hits(base_path, subruns, feu_nums, decoded_dir,
                                xlim_us=xlim_us, min_hit_amp=min_hit_amp, show=show_orientations)

    plot_hv_scan_waveforms_only(base_path, subruns, feu_nums, decoded_dir,
                                xlim_us=xlim_us, show=show_orientations)

    plt.show()
    print('donzo')


# ---------------------------------------------------------------------------
# 4-row version: X waveform / X hits / Y waveform / Y hits
# ---------------------------------------------------------------------------

def plot_hv_scan_waveforms_hits(base_path, subruns, feu_nums, decoded_dir,
                                xlim_us=(0, 8), min_hit_amp=400, channels=None,
                                show='both'):
    """
    Stitch subruns at different HVs horizontally.

    Layout rows depend on `show`:
        'both' — X waveforms / X hits / Y waveforms / Y hits  (4 rows)
        'x'    — X waveforms / X hits                         (2 rows)
        'y'    — Y waveforms / Y hits                         (2 rows)

    Parameters
    ----------
    base_path   : root data directory
    subruns     : list of (run, subrun, file_num, event) tuples, one per column
    feu_nums    : dict mapping FEU number → axis label ('x' or 'y')
    decoded_dir : name of decoded subdirectory inside each subrun folder
    xlim_us     : (xmin, xmax) in µs applied to all axes
    min_hit_amp : minimum amplitude threshold for hits
    channels    : optional array of channel numbers to restrict plotting
    show        : 'both' | 'x' | 'y' — which detector axes to include
    """
    ns_per_sample = 20
    n_cols = len(subruns)

    # Build ordered list of (row_key, y-label) to determine layout
    row_specs = []
    if show in ('both', 'x'):
        row_specs += [('x_wave', 'X Amplitude'), ('x_hits', 'X Hit Position (mm)')]
    if show in ('both', 'y'):
        row_specs += [('y_wave', 'Y Amplitude'), ('y_hits', 'Y Hit Position (mm)')]
    n_rows = len(row_specs)

    row_heights = [2 if 'wave' in k else 1 for k, _ in row_specs]
    fig, axs = plt.subplots(
        n_rows, n_cols,
        sharey='row', sharex=True,
        figsize=(4 * n_cols, sum(row_heights) * 1.5),
        gridspec_kw={'height_ratios': row_heights},
    )
    # Ensure axs is always 2-D
    axs = np.atleast_2d(axs) if n_cols > 1 else np.array(axs).reshape(n_rows, 1)

    for col, (run, subrun, file_num, event) in enumerate(subruns):
        dec_path = os.path.join(base_path, run, subrun, decoded_dir)
        hv_str = _resist_hv_str(subrun)

        # --- Load waveforms ---
        evt_samples, evt_channels, evt_amplitudes = load_decoded_waveforms(
            dec_path, feu_nums, file_num)
        evt_ids, _ = load_evtids_timestamps(dec_path, list(feu_nums.keys())[0], file_num)
        evt_index = np.where(evt_ids == event)[0][0]

        if show in ('both', 'x'):
            evt_ch_x   = evt_channels['x'][evt_index]
            evt_amp_x  = evt_amplitudes['x'][evt_index]
            evt_time_x = evt_samples['x'][evt_index] * ns_per_sample / 1000
        if show in ('both', 'y'):
            evt_ch_y   = evt_channels['y'][evt_index]
            evt_amp_y  = evt_amplitudes['y'][evt_index]
            evt_time_y = evt_samples['y'][evt_index] * ns_per_sample / 1000

        # --- Load hits (needed for any hit row) ---
        df, det = load_subrun(base_path, run, subrun, list(feu_nums.keys()))
        df = df[(df['eventId'] == event) & (df['amplitude'] >= min_hit_amp)]
        if channels is not None:
            df = df[df['channel'].isin(channels)]
        df = add_xy_pos(df, det)

        # --- Fill each row ---
        for row, (key, _) in enumerate(row_specs):
            ax = axs[row, col]
            if key == 'x_wave':
                unique_x = np.unique(evt_ch_x)
                cmap_x = plt.get_cmap('coolwarm')
                norm_x = mcolors.Normalize(vmin=unique_x.min(), vmax=unique_x.max())
                for ch in unique_x:
                    if channels is not None and ch not in channels:
                        continue
                    m = (evt_ch_x == ch)
                    ax.plot(evt_time_x[m], evt_amp_x[m], lw=0.7, color=cmap_x(norm_x(ch)))
                _annotate_hv(ax, hv_str)
            elif key == 'x_hits':
                df_x = df[df['x_position_mm'].notna()]
                ax.scatter(df_x['time'] / 3 / 1000, df_x['x_position_mm'],
                           c=df_x['local_max'], cmap='jet', s=10)
            elif key == 'y_wave':
                unique_y = np.unique(evt_ch_y)
                cmap_y = plt.get_cmap('coolwarm')
                norm_y = mcolors.Normalize(vmin=unique_y.min(), vmax=unique_y.max())
                for ch in unique_y:
                    if channels is not None and ch not in channels:
                        continue
                    m = (evt_ch_y == ch)
                    ax.plot(evt_time_y[m], evt_amp_y[m], lw=0.7, color=cmap_y(norm_y(ch)))
                _annotate_hv(ax, hv_str)
            elif key == 'y_hits':
                df_y = df[df['y_position_mm'].notna()]
                ax.scatter(df_y['time'] / 3 / 1000, df_y['y_position_mm'],
                           c=df_y['local_max'], cmap='jet', s=10)

        axs[-1, col].set_xlabel('Time (µs)')

    # --- Shared x limits and y-axis labels on left column only ---
    axs[0, 0].set_xlim(xlim_us)
    for row, (_, ylabel) in enumerate(row_specs):
        axs[row, 0].set_ylabel(ylabel)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.02)


# ---------------------------------------------------------------------------
# 2-row version: waveforms only (no hits)
# ---------------------------------------------------------------------------

def plot_hv_scan_waveforms_only(base_path, subruns, feu_nums, decoded_dir,
                                xlim_us=(0, 8), channels=None, show='both'):
    """
    Same horizontal HV-scan layout but showing only waveforms (no hit panels).

    Layout rows depend on `show`:
        'both' — X waveforms / Y waveforms  (2 rows)
        'x'    — X waveforms only           (1 row)
        'y'    — Y waveforms only           (1 row)

    Parameters
    ----------
    base_path   : root data directory
    subruns     : list of (run, subrun, file_num, event) tuples, one per column
    feu_nums    : dict mapping FEU number → axis label ('x' or 'y')
    decoded_dir : name of decoded subdirectory inside each subrun folder
    xlim_us     : (xmin, xmax) in µs applied to all axes
    channels    : optional array of channel numbers to restrict plotting
    show        : 'both' | 'x' | 'y' — which detector axes to include
    """
    ns_per_sample = 20
    n_cols = len(subruns)

    row_specs = []
    if show in ('both', 'x'):
        row_specs.append(('x_wave', 'X Amplitude'))
    if show in ('both', 'y'):
        row_specs.append(('y_wave', 'Y Amplitude'))
    n_rows = len(row_specs)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        sharey='row', sharex=True,
        figsize=(4 * n_cols, 2.5 * n_rows),
    )
    # Ensure axs is always 2-D
    axs = np.atleast_2d(axs) if n_cols > 1 else np.array(axs).reshape(n_rows, 1)

    for col, (run, subrun, file_num, event) in enumerate(subruns):
        dec_path = os.path.join(base_path, run, subrun, decoded_dir)
        hv_str = _resist_hv_str(subrun)

        evt_samples, evt_channels, evt_amplitudes = load_decoded_waveforms(
            dec_path, feu_nums, file_num)
        evt_ids, _ = load_evtids_timestamps(dec_path, list(feu_nums.keys())[0], file_num)
        evt_index = np.where(evt_ids == event)[0][0]

        if show in ('both', 'x'):
            evt_ch_x   = evt_channels['x'][evt_index]
            evt_amp_x  = evt_amplitudes['x'][evt_index]
            evt_time_x = evt_samples['x'][evt_index] * ns_per_sample / 1000
        if show in ('both', 'y'):
            evt_ch_y   = evt_channels['y'][evt_index]
            evt_amp_y  = evt_amplitudes['y'][evt_index]
            evt_time_y = evt_samples['y'][evt_index] * ns_per_sample / 1000

        for row, (key, _) in enumerate(row_specs):
            ax = axs[row, col]
            if key == 'x_wave':
                unique_x = np.unique(evt_ch_x)
                cmap_x = plt.get_cmap('coolwarm')
                norm_x = mcolors.Normalize(vmin=unique_x.min(), vmax=unique_x.max())
                for ch in unique_x:
                    if channels is not None and ch not in channels:
                        continue
                    m = (evt_ch_x == ch)
                    ax.plot(evt_time_x[m], evt_amp_x[m], lw=0.7, color=cmap_x(norm_x(ch)))
                _annotate_hv(ax, hv_str)
            elif key == 'y_wave':
                unique_y = np.unique(evt_ch_y)
                cmap_y = plt.get_cmap('coolwarm')
                norm_y = mcolors.Normalize(vmin=unique_y.min(), vmax=unique_y.max())
                for ch in unique_y:
                    if channels is not None and ch not in channels:
                        continue
                    m = (evt_ch_y == ch)
                    ax.plot(evt_time_y[m], evt_amp_y[m], lw=0.7, color=cmap_y(norm_y(ch)))
                _annotate_hv(ax, hv_str)

        axs[-1, col].set_xlabel('Time (µs)')

    axs[0, 0].set_xlim(xlim_us)
    for row, (_, ylabel) in enumerate(row_specs):
        axs[row, 0].set_ylabel(ylabel)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.02)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re

def _resist_hv_str(subrun):
    """Return just the resist HV as a string, e.g. '410 V'."""
    m = re.search(r'resist_(\d+)V', subrun)
    return f'{m.group(1)} V' if m else subrun


def _annotate_hv(ax, hv_str):
    """Place a yellow-box HV label in the top-right corner of ax."""
    ax.text(0.97, 0.97, hv_str,
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='none'))


if __name__ == '__main__':
    main()
