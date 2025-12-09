#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 28 16:02 2025
Created in PyCharm
Created as nTof_x17/plot_flash_time

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import uproot


def main():
    base_dir = '/local/home/dn277127/x17/dream_run'
    run = 'run_40'
    feus = [5, 6]                     # FEUs to plot
    file_num = 0                      # Which file index per FEU
    threshold = 2000                  # Amplitude crossing threshold
    sample_period = 20                # ns per sample

    # Connector map: which connectors are physically plugged per FEU
    plugged = {
        5: [1, 2, 3, 4, 5, 6, 7],
        6: [5, 6, 7],
    }

    feu_names = {
        5: 'Micromegas',
        6: 'Scintillators',
    }

    run_dir = os.path.join(base_dir, run)

    # Plot init
    fig, ax = plt.subplots(figsize=(12, 6))

    for feu in feus:
        flash_samples = np.array(get_feu_flash_indices(run_dir, feu, file_num, threshold, plugged))
        flash_times = flash_samples * sample_period / 1e3  # convert to us
        # plot time on the left y-axis (ns)
        ax.plot(
            np.arange(len(flash_samples)),
            flash_times,
            marker='o', ls='-', label=f'{feu_names.get(feu, f"FEU {feu}")} (FEU {feu})'
        )

    ax.set_xlabel('Event Number')
    ax.set_ylabel('Time from Waveform Start (us)')
    ax.set_title(f'Run {run} - Gamma Flash Time (Threshold = {threshold} ADC)')
    ax.legend()
    ax.grid(True)

    # secondary y-axis on the right to show sample index (converted from time)
    ax_right = ax.secondary_yaxis('right',
                                  functions=(lambda t: t / sample_period * 1000, lambda s: s * sample_period))
    ax_right.set_ylabel('Sample Index of First Threshold Crossing')
    fig.tight_layout()
    plt.show()

    print('donzo')


def get_feu_flash_indices(run_dir, feu, file_num, threshold, plugged):
    """Read amplitudes for a FEU and return array of first sample indices above threshold."""
    feu_tag = f'_{feu:02d}_'
    file_tag = f'_{file_num:03d}_'

    # find the matching file
    array_files = [f for f in os.listdir(run_dir) if f.endswith('_array.root')]
    file = next((f for f in array_files if feu_tag in f and file_tag in f), None)
    if file is None:
        raise FileNotFoundError(f"No ROOT array file found for FEU {feu}, file {file_num}")

    file_path = os.path.join(run_dir, file)

    with uproot.open(file_path) as f:
        tree = f['nt']
        amplitudes = tree['amplitude'].array(library='np')  # shape = (n_events, n_channels, n_samples)

    print(f'Loaded: {file_path}, amplitude shape = {amplitudes.shape}')

    plugged_idxs = []

    # Loop events
    for event_i in range(amplitudes.shape[0]):
        connector_idxs = []
        for conn in plugged.get(feu, []):
            begin = (conn - 1) * 64
            end = begin + 64

            waveforms = amplitudes[event_i, begin:end, :]
            idx = get_thresh_sample(waveforms, threshold)
            connector_idxs.append(idx)

        # FEU index per event = max connector index (if any valid)
        valid = [i for i in connector_idxs if i >= 0]
        feu_idx = max(valid) if valid else -1
        plugged_idxs.append(feu_idx)

    return np.array(plugged_idxs)


def get_thresh_sample(waveforms, threshold):
    """Return first sample index where any channel crosses threshold; -1 if none."""
    mask = waveforms > threshold           # (n_channels, n_samples)
    sample_cross = mask.any(axis=0)        # (n_samples,)
    if not sample_cross.any():
        return -1
    return np.argmax(sample_cross)


if __name__ == '__main__':
    main()
