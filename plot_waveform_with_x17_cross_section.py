#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on December 11 6:21 PM 2025
Created in PyCharm
Created as nTof_x17/plot_waveform_with_x17_cross_section.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import uproot

from neutron_energy_vs_flight_time import energy_eV_to_time_s
from X17CalculationParser import X17CalculationParser

distance_m = 19.5


def main():
    run_dir = '/media/dylan/data/x17/nov_25_beam_test/dream_run/'
    run = 'run_69'
    feu_num = 5
    file_num = 0
    event = 8
    detector = 'strip'  # 'strip' or 'plein'

    xy_map = {
        'strip': {
            'x': [1, 2],
            'y': [3, 4],
        },
        'plein': {
            'x': [5],
            'y': [6, 7],
        }
    }

    run_dir = os.path.join(run_dir, run)

    # Find all _array.root files in the run directory
    array_files = [f for f in os.listdir(run_dir) if f.endswith('.root')]

    # Get feu_num padded to two digits and file_num padded to two digits
    feu_num_str = f'_{feu_num:02d}'
    file_num_str = f'_{file_num:03d}_'
    # Find the file that matches the feu_num and file_num
    file = None
    for f in array_files:
        if feu_num_str in f and file_num_str in f:
            file = f
            break

    file_path = os.path.join(run_dir, file)

    calculation_tables_dir = f'/media/dylan/data/x17/calculation_tables/'
    file_name = 'results_3He'
    parser = X17CalculationParser(calculation_tables_dir + file_name)
    df_x17 = parser.get_dataframe()
    print(df_x17.columns)

    flash_time = energy_eV_to_time_s(1000e9)
    print(f'Gamma flash time for 1000 GeV neutron over {distance_m} m: {flash_time * 1e9:.2f} ns')

    with uproot.open(file_path) as f:
        tree = f['nt']
        # Read all branches as NumPy arrays
        samples = tree["sample"].array(library="np")  # Jagged array
        channels = tree["channel"].array(library="np")  # Jagged array
        amplitudes = tree["amplitude"].array(library="np")

        evt_samples = samples[event]
        evt_channels = channels[event]
        evt_amplitudes = amplitudes[event]

        plot_waveforms_with_x17(
            evt_channels,
            evt_samples,
            evt_amplitudes,
            xy_map,
            df_x17,
            detector='strip',
            event_id=event,
            run=run,
            flash_time_s=flash_time,
            blind_time_s=550e-9,  # adjust by eye
            x17_data_time_offset=150e-9  # adjust by eye
        )


    print('donzo')


def plot_waveforms_with_x17(
        evt_channels, evt_samples, evt_amplitudes,
        xy_map, df_x17,
        detector='strip',
        event_id=None, run=None,
        flash_time_s=0.0,
        blind_time_s=None,
        x17_data_time_offset=0.0,
        sample_period_s=20e-9   # 20 ns
    ):
    """
    3-row plot:
      Row 1: X-strip waveforms
      Row 2: Y-strip waveforms
      Row 3: X17 cross section vs time

    Parameters
    ----------
    x17_data_time_offset : float
        Constant time shift applied to waveform time axis.
        Positive → shifts waveforms to the right.
    """

    # ---------------------------
    # Helper: expand connector lists
    # ---------------------------
    def expand(conn):
        out = []
        for c in conn:
            start = (c - 1) * 64
            out.extend(range(start, start + 64))
        return np.array(out)

    x_channels = expand(xy_map[detector]['x'])
    y_channels = expand(xy_map[detector]['y'])

    x_mask = np.isin(evt_channels, x_channels)
    y_mask = np.isin(evt_channels, y_channels)

    # ---------------------------
    # Convert sample # → time
    # ---------------------------
    evt_time = evt_samples * sample_period_s   # time in seconds

    # ---------------------------
    # Create 3-row figure
    # ---------------------------
    fig, (ax_cs, ax_x, ax_y) = plt.subplots(
        3, 1, figsize=(11, 9), sharex='all',
        gridspec_kw={'height_ratios': [0.8, 1, 1]}
    )

    # ============================================================
    # Row 1 — X17 cross section
    # ============================================================
    # Already computed in your script:
    # df_x17 contains: time_low_energy_s, time_high_energy_s, time_mid_s, time_err_s, X17

    # Convert energy bins to time bins
    E_low = df_x17["elow [eV]"].values
    E_up = df_x17["eup [eV]"].values
    t_low = energy_eV_to_time_s(E_up, distance=distance_m)  # note inversion of low/high
    t_up = energy_eV_to_time_s(E_low, distance=distance_m)
    t = (t_low + t_up) / 2.0  # mid time

    y = df_x17['X17 [1/day]'].values

    # Resort by time (in case energy bins were not sorted)
    sorted_indices = np.argsort(t)
    t = t[sorted_indices]
    t_low = t_low[sorted_indices]
    t_up = t_up[sorted_indices]
    y = y[sorted_indices]

    # --- Interpolation (log-log cubic spline) ---
    # Use log axes for physics spectra spanning orders of magnitude
    cs = CubicSpline(np.log(t), np.log(y))
    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 2000)
    y_smooth = np.exp(cs(np.log(t_smooth)))

    # Apply time offset to x17 and flash times
    t += x17_data_time_offset
    t_low += x17_data_time_offset
    t_up += x17_data_time_offset
    t_smooth += x17_data_time_offset
    flash_time_s += x17_data_time_offset

    # --- Plot raw points with horizontal error bars (exact bin widths) ---
    xerr = np.array([t - t_low, t_up - t])  # asymmetric error bars
    ax_cs.errorbar(
        t * 1e6, y, xerr=xerr * 1e6, fmt="o", markersize=6, capsize=3,
        label="Alberto's Calculation", color="C0"
    )

    # --- Plot the interpolated faint smooth line ---
    ax_cs.plot(
        t_smooth * 1e6, y_smooth,
        linewidth=1.2, alpha=0.35, color="C0",
        label="interpolation"
    )
    ax_cs.axhline(0, color="gray", linestyle="-", zorder=0)
    ax_cs.set_xlim(left=0, right=np.max(evt_time) * 1e6 * 1.05)
    ax_cs.set_ylabel("X17 Yield (x17/day)")

    # --- Plot vertical lines for flash, blind, readout times if provided ---
    if flash_time_s is not None:
        ax_cs.axvline(flash_time_s * 1e6, color='red', ls='--', label=f'Gamma Flash')
    if blind_time_s is not None:
        ax_cs.axvline((flash_time_s + blind_time_s) * 1e6, color='orange', ls='--',
                   label=f'End of Blind Time')

    ax_cs.legend()

    # ============================================================
    # Row 2 — X waveforms
    # ============================================================
    unique_x = np.unique(evt_channels[x_mask])
    for ch in unique_x:
        m = evt_channels == ch
        ax_x.plot(evt_time[m]*1e6, evt_amplitudes[m], lw=0.7)

    ax_x.set_ylabel("X Amplitude (ADC)")
    ax_x.axvline(flash_time_s*1e6, color='red', ls='--', lw=1.2, label="Gamma Flash")
    if blind_time_s is not None:
        ax_x.axvline((flash_time_s + blind_time_s)*1e6, color='orange', ls='--', lw=1.2)

    # ============================================================
    # Row 3 — Y waveforms
    # ============================================================
    unique_y = np.unique(evt_channels[y_mask])
    for ch in unique_y:
        m = evt_channels == ch
        ax_y.plot(evt_time[m]*1e6, evt_amplitudes[m], lw=0.7)

    ax_y.set_ylabel("Y Amplitude (ADC)")
    ax_y.axvline(flash_time_s*1e6, color='red', ls='--', lw=1.2)
    if blind_time_s is not None:
        ax_y.axvline((flash_time_s + blind_time_s)*1e6, color='orange', ls='--', lw=1.2)
    ax_y.set_xlabel("Time (µs)")

    # ---------------------------
    # Title
    # ---------------------------
    title = "X17 Yield + Micromegas Waveforms"
    if detector: title += f" ({detector})"
    if event_id is not None: title += f" — Event {event_id}"
    if run is not None: title += f" — Run {run}"
    fig.suptitle(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0)
    plt.show()


if __name__ == '__main__':
    main()
