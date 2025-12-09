#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 27 12:50 2025
Created in PyCharm
Created as nTof_x17/check_timestamps

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import vector


def main():
    run_dir = '/local/home/dn277127/x17/dream_run/'
    run = 'run_84'
    feu_num = 5
    file_num = 0
    event = 8
    detector = 'strip'  # 'strip' or 'plein'
    min_sample = 0
    max_sample = 60

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
        print(data.keys())
        print(data['timestamp'].shape)
        print(np.diff(data['timestamp']) / 1e8)  # Convert ns to s
        print(data['delta_timestamp'] / 1e8)

        # Write a list of timestamps in seconds to file
        # out_file = file_path.replace('.root', '_timestamps.txt')
        # timestamps_s = data['timestamp'] / 1e8  # Convert ns to s
        # np.savetxt(out_file, timestamps_s, fmt='%.8f')
        # print(f'Wrote timestamps to {out_file}')

        amplitudes = data['amplitude']
        print(f'Amplitudes shape: {amplitudes.shape}')
        print(f'{np.max(amplitudes, axis=2).shape}')
        print(f'{np.max(amplitudes, axis=(0, 2))}')
        max_per_event = np.max(amplitudes, axis=(1, 2))
        flat_idx = np.argmax(amplitudes.reshape(amplitudes.shape[0], -1), axis=1)
        n_samples = amplitudes.shape[2]
        channel_idx = flat_idx // n_samples
        sample_idx = flat_idx % n_samples
        print('Max amplitude per event with (channel, sample):')
        for i, (m, ch, s) in enumerate(zip(max_per_event, channel_idx, sample_idx)):
            print(f'Event {i}: max={m} at channel={ch}, sample={s}')
        print(f'Max amplitude over all events channels, samples: {np.max(amplitudes)}')

        # threshold = 2000
        # flash_samples = []
        # for event_i in range(amplitudes.shape[0]):
        #     plugged_connectors = plugged.get(feu_num, [])
        #     plugged_idxs = []
        #     for connector in plugged_connectors:
        #         start_channel = (connector - 1) * 64
        #         end_channel = start_channel + 64
        #         channel_waveforms = amplitudes[event_i, start_channel:end_channel, :]
        #         idx = get_thresh_sample(channel_waveforms, threshold)
        #         plugged_idxs.append(idx)
        #     # Get the maximum index among plugged connectors
        #     if plugged_idxs:
        #         max_idx = max(plugged_idxs)
        #     else:
        #         max_idx = -1
        #     flash_samples.append(max_idx)
        #     # idx = get_thresh_sample(channel_timing_waveforms[event_i], threshold)
        #     # flash_samples.append(idx)
        # flash_samples = np.array(flash_samples)
        #
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.hist(flash_samples[flash_samples >= 0], bins=50, range=(0, amplitudes.shape[2]))
        # ax.set_xlabel('Sample Index of First Crossing')
        # ax.set_ylabel('Number of Events')
        # ax.set_title(f'Histogram of First Sample Crossing {threshold} for Channels 0-255')
        #
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.axhline(0, color='gray', ls='-')
        # ax.plot(np.arange(flash_samples.shape[0]), flash_samples, marker='o', ls='-')
        # ax.set_xlabel('Event Index')
        # ax.set_ylabel('Sample Index of First Crossing')
        # ax.set_title(f'Sample Index of First Crossing {threshold} vs Event Index')
        #
        # plt.show()
        # Plot one waveform for one channel. (event, channel, sample)
        # channels = np.arange(384, 448, 1)
        # channel_sets = [np.arange(0, 64, 1), np.arange(64, 128, 1), np.arange(128, 192, 1),
        #                 np.arange(192, 256, 1), np.arange(256, 320, 1), np.arange(320, 384, 1),
        #                 np.arange(384, 448, 1)]
        # if amplitudes.shape[1] == 512:
        #     channel_sets.append(np.arange(448, 512, 1))
        # channel_sets = np.array(channel_sets)
        # if not show_unplugged:
        #     connectors = np.array(plugged.get(feu_num, []))
        #     channel_sets = channel_sets[connectors - 1]
        #     print(f'Showing only plugged connectors: {connectors}')
        # for channels in channel_sets:
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     # channels = np.arange(0, 448, 1)
        #     # Find sample index of first channel that goes above 2000
        #     channel_set_waveforms = amplitudes[event, channels, :]
        #     threshold = 2000
        #     idx = get_thresh_sample(channel_set_waveforms, threshold)
        #
        #     print(f'First sample index crossing {threshold} for event {event}, channels {channels[0]}-{channels[-1]}: {idx}')
        #
        #     for channel in channels:
        #         waveform = amplitudes[event, channel, :]
        #         time_axis = np.arange(waveform.shape[0])  # Sample indices as time
        #         ax.plot(time_axis, waveform, label=f'Event {event}, Channel {channel}')
        #     ax.set_xlabel('Sample Index')
        #     ax.set_ylabel('Amplitude')
        #     ax.set_title(f'Waveform for Run {run}, Event {event}, Channels {channels[0]}-{channels[-1]}')
        # ax.legend()

        if feu_num == 6:  # SiPms
            # Plot waveforms for all channels
            fig, ax = plt.subplots(figsize=(12, 6))
            for channel in range(amplitudes.shape[1]):
                waveform = amplitudes[event, channel, :]
                time_axis = np.arange(waveform.shape[0])  # Sample indices as time
                ax.plot(time_axis, waveform, lw=0.5)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'All Channel Waveforms for Run {run}, Event {event}')

        plot_flash_xy_waveforms(amplitudes, event, xy_map, detector, run=run,
                                min_sample=min_sample, max_sample=max_sample)

        plot_flash_xy_map(amplitudes[event], xy_map, min_sample, max_sample, detector=detector)

        plt.show()
        # for event in events:
        #     event_amplitudes = amplitudes[event]
        #     print(f'Event {event} amplitudes shape: {event_amplitudes.shape}')
        #     print(np.max(event_amplitudes, axis=0).shape)
        #     print(np.max(event_amplitudes, axis=0))
        #     # Plot



    print('donzo')


def plot_flash_xy_map(flash_samples, xy_map, min_sample=None, max_sample=None, detector='strip'):
    """
    Plot flash sample indices on an XY map based on channel grouping.
    """
    if max_sample is not None:
        flash_samples = flash_samples[:, :max_sample]
    if min_sample is not None:
        flash_samples = flash_samples[:, min_sample:]

    x_connectors, y_connectors = np.array(xy_map[detector]['x']), np.array(xy_map[detector]['y'])
    x_channels = []
    for connector in x_connectors:
        start_channel = (connector - 1) * 64
        end_channel = start_channel + 64
        x_channels.extend(range(start_channel, end_channel))
    y_channels = []
    for connector in y_connectors:
        start_channel = (connector - 1) * 64
        end_channel = start_channel + 64
        y_channels.extend(range(start_channel, end_channel))
    x_channels = np.array(x_channels)
    y_channels = np.array(y_channels)

    x_amps = flash_samples[x_channels]
    y_amps = flash_samples[y_channels]

    print(f'X amps shape: {x_amps.shape}, Y amps shape: {y_amps.shape}')

    # For each channel, get the max amplitude sample index and value
    x_max_amps = np.max(x_amps, axis=1)
    y_max_amps = np.max(y_amps, axis=1)

    # Plot 2 1D plots of max x-amps and y-amps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    ax1.plot(np.arange(len(x_max_amps)), x_max_amps)
    ax1.set_ylabel('X Strips Max Amplitude')
    ax2.plot(np.arange(len(y_max_amps)), y_max_amps)
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Y Strips Max Amplitude')
    plt.suptitle(f'Flash sample indices on {detector} detector')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    # Now plot in 2D. Each point on the grid should be the sum of the x and y amplitudes at
    # the corresponding channel indices.

    plot_flash_xy_grid(x_max_amps, y_max_amps, detector)
    plt.show()

    input()


def plot_flash_xy_grid(x_max_amps, y_max_amps, detector='strip'):
    """
    Plot a 2D grid where entry (i, j) = x_max_amps[i] + y_max_amps[j].
    """
    # Create 2D map: outer sum of x and y amplitudes
    xy_grid = x_max_amps[:, None] + y_max_amps[None, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(xy_grid.T, aspect='auto', origin='lower', cmap='jet', vmin=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('X + Y Max Amplitude')

    ax.set_xlabel('X Strip Index')
    ax.set_ylabel('Y Strip Index')
    ax.set_title(f'2D Flash Amplitude Map on {detector} detector')

    plt.tight_layout()



def get_thresh_sample(waveforms, threshold):
    """Get the first sample index where any channel crosses the threshold.

    Args:
        waveforms (np.ndarray): Array of shape (n_channels, n_samples).
        threshold (float): Amplitude threshold to detect crossing.

    Returns:
        int: Sample index of first crossing, or -1 if none cross.
    """
    # Mask where amplitude > threshold
    mask = waveforms > threshold  # shape: (n_channels, n_samples)

    # Collapse channels → per-sample flag
    sample_cross = mask.any(axis=0)  # shape: (n_samples,)

    # First index where True appears
    idx = np.argmax(sample_cross)

    # If no channel crosses, argmax returns 0 incorrectly, so detect no cross
    if not sample_cross.any():
        idx = -1  # or np.nan

    return idx


def plot_flash_xy_waveforms(amplitudes, event, xy_map, detector='strip', run=None, min_sample=None, max_sample=None):
    """
    Plot all X-strip waveforms in the top axis and all Y-strip waveforms in the bottom axis.

    amplitudes[event, channel, sample] is assumed.
    xy_map[detector]['x'] and ['y'] must contain connector numbers (1-indexed).
    """

    # Select connectors for detector
    x_connectors = np.array(xy_map[detector]['x'])
    y_connectors = np.array(xy_map[detector]['y'])

    # Expand connectors → channel indices
    def expand_channels(connectors):
        chans = []
        for c in connectors:
            start = (c - 1) * 64
            chans.extend(range(start, start + 64))
        return np.array(chans)

    x_channels = expand_channels(x_connectors)
    y_channels = expand_channels(y_connectors)

    # --- PLOTTING ---
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    time_axis = np.arange(amplitudes.shape[2])

    # Plot X waveforms
    for ch in x_channels:
        ax_x.plot(time_axis, amplitudes[event, ch, :], lw=0.7)
    ax_x.set_ylabel('X Strip Amplitudes')

    # Plot Y waveforms
    for ch in y_channels:
        ax_y.plot(time_axis, amplitudes[event, ch, :], lw=0.7)
    ax_y.set_ylabel('Y Strip Amplitude')
    ax_y.set_xlabel('Sample Index')

    if min_sample is not None:
        ax_x.axvline(min_sample, color='red', ls='-')
        ax_y.axvline(min_sample, color='red', ls='-')
    if max_sample is not None:
        ax_x.axvline(max_sample, color='red', ls='-')
        ax_y.axvline(max_sample, color='red', ls='-')

    fig.suptitle(f'Flash Amplitude Map on {detector} detector for Event {event}' + (f' in {run}' if run else ''))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)



def read_det_data_vars(file_path, variables, tree_name='nt', event_range=None):
    if hasattr(vector, "register_awkward") and callable(vector.register_awkward):
        vector.register_awkward()
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    if event_range is None:
        variable_data = tree.arrays(variables, library='np')
    else:
        variable_data = tree.arrays(variables, library='np', entry_start=event_range[0], entry_stop=event_range[-1])
    variable_data = {key: val.astype(np.float32) for key, val in variable_data.items()}
    root_file.close()

    return variable_data


if __name__ == '__main__':
    main()
