#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 29 21:05 2025
Created in PyCharm
Created as nTof_x17/plot_timestamps

@author: Dylan Neff, dn277127
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import vector

from plot_waveforms_array import read_det_data_vars


def main():
    run_dir = '/local/home/dn277127/x17/dream_run/'
    run = 'run_66'
    feu_num = 5
    file_num = 0

    run_dir = os.path.join(run_dir, run)

    # Find all _array.root files in the run directory
    array_files = [f for f in os.listdir(run_dir) if f.endswith('_array.root')]

    # Get feu_num padded to two digits and file_num padded to two digits
    feu_num_str = f'_{feu_num:02d}_'
    file_num_str = f'_{file_num:03d}_'
    # Find the file that matches the feu_num and file_num
    file = None
    for f in array_files:
        if feu_num_str in f and file_num_str in f:
            file = f
            break

    file_path = os.path.join(run_dir, file)

    with uproot.open(file_path) as f:
        tree = f['nt']
        # Read all branches as NumPy arrays
        data = tree.arrays(library='np')

        timestamps = data['timestamp'] / 1e8  # Convert 100MHz clock to s
        event_num = data['eventId']
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, event_num, marker='o', ls='none')
        ax.set_ylabel('Event Number')
        ax.set_xlabel('Timestamp (s)')
        ax.set_title(f'Run {run} - FEU {feu_num} - File {file_num} - Event Timestamps')
        ax.grid(True)
        fig.tight_layout()

        delta_timestamps = data['delta_timestamp'] / 1e8  # Convert to s
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        beam_period = 1.2  # s
        dt_bin_width = beam_period / 16
        dt_bins = np.arange(-dt_bin_width / 2, np.max(delta_timestamps) + dt_bin_width / 2, dt_bin_width)  # Period
        ax2.hist(delta_timestamps, bins=dt_bins, histtype='stepfilled', zorder=10)
        ax2.set_xlabel('Delta Timestamp (s)')
        ax2.set_ylabel('Counts')
        ax2.set_title(f'Run {run} - FEU {feu_num} - File {file_num} - Delta Timestamps Histogram')
        ax2.grid(True)
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        t_bin_width = beam_period / 16
        t_bins = np.arange(np.min(timestamps) - t_bin_width / 2, np.max(timestamps) + t_bin_width / 2, t_bin_width)
        ax3.hist(timestamps, bins=t_bins, histtype='stepfilled', zorder=10)
        ax3.set_xlabel('Timestamp (s)')
        ax3.set_ylabel('Counts')
        ax3.set_title(f'Run {run} - FEU {feu_num} - File {file_num} - Timestamps Histogram')
        fig3.tight_layout()

        delta_timestamps_ns = data['delta_timestamp'] / 100  # in us
        fig_4, ax4 = plt.subplots(figsize=(12, 6))
        dt_small_bins = np.linspace(0, 10, 100)
        ax4.hist(delta_timestamps_ns, bins=dt_small_bins, histtype='stepfilled', zorder=10)
        ax4.set_xlabel('Delta Timestamp (us)')
        ax4.set_ylabel('Counts')
        ax4.set_title(f'Run {run} - FEU {feu_num} - File {file_num} - Delta Timestamps Histogram (0-10 us)')
        fig_4.tight_layout()

        plt.show()


    print('donzo')



if __name__ == '__main__':
    main()
