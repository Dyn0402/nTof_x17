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
    run_dir = '/media/dylan/data/x17/nov_25_beam_test/dream_run/'
    run = 'run_69'
    feu_num = 5
    file_num = 0
    event = 8
    detector = 'plein'  # 'strip' or 'plein'
    min_sample = 80
    max_sample = 110

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

    # plot_waveforms(file_path, event)
    # plt.show()

    with uproot.open(file_path) as f:
        tree = f['nt']
        # Read all branches as NumPy arrays
        evt_ids = tree["eventId"].array(library="np")
        samples = tree["sample"].array(library="np")  # Jagged array
        channels = tree["channel"].array(library="np")  # Jagged array
        amplitudes = tree["amplitude"].array(library="np")

        # timestamps = tree["timestamp"].array(library="np")
        # check_timestamps(timestamps)
        # input()

        # if feu_num == 6:  # SiPms
        #     # Plot waveforms for all channels
        #     fig, ax = plt.subplots(figsize=(12, 6))
        #     for channel in range(amplitudes.shape[1]):
        #         waveform = amplitudes[event, channel, :]
        #         time_axis = np.arange(waveform.shape[0])  # Sample indices as time
        #         ax.plot(time_axis, waveform, lw=0.5)
        #     ax.set_xlabel('Sample Index')
        #     ax.set_ylabel('Amplitude')
        #     ax.set_title(f'All Channel Waveforms for Run {run}, Event {event}')

        plot_flash_xy_waveforms(channels[event], samples[event], amplitudes[event], xy_map, detector, run=run,
                                min_sample=min_sample, max_sample=max_sample)

        plot_flash_xy_map(channels[event], samples[event], amplitudes[event], xy_map, detector, min_sample, max_sample)

        plt.show()
        # for event in events:
        #     event_amplitudes = amplitudes[event]
        #     print(f'Event {event} amplitudes shape: {event_amplitudes.shape}')
        #     print(np.max(event_amplitudes, axis=0).shape)
        #     print(np.max(event_amplitudes, axis=0))
        #     # Plot



    print('donzo')


def build_waveform_matrix(amp, chan, samp, n_channels=4096, n_samples=2000):
    """
    Convert flat arrays for one event into a dense matrix [channel, sample].
    Missing samples remain 0.

    amp, chan, samp: 1D arrays of equal length.
    """
    wf = np.zeros((n_channels, n_samples), dtype=amp.dtype)
    wf[chan, samp] = amp
    return wf


def plot_flash_xy_map(evt_channels, evt_samples, evt_amplitudes,
                      xy_map, detector='strip',
                      min_sample=None, max_sample=None):
    """
    evt_channels, evt_samples, evt_amplitudes come from the jagged event.
    """

    # Sample window cut
    if min_sample is not None:
        mask = evt_samples >= min_sample
        evt_channels = evt_channels[mask]
        evt_samples  = evt_samples[mask]
        evt_amplitudes = evt_amplitudes[mask]
    if max_sample is not None:
        mask = evt_samples < max_sample
        evt_channels = evt_channels[mask]
        evt_samples  = evt_samples[mask]
        evt_amplitudes = evt_amplitudes[mask]

    # Expand connectors → channels
    def expand(connectors):
        chans = []
        for c in connectors:
            start = (c - 1) * 64
            chans.extend(range(start, start + 64))
        return np.array(chans)

    x_channels = expand(xy_map[detector]['x'])
    y_channels = expand(xy_map[detector]['y'])

    # Split event hits into X and Y sides
    x_mask = np.isin(evt_channels, x_channels)
    y_mask = np.isin(evt_channels, y_channels)

    # Max amplitude PER STRIP (64 per connector)
    def max_per_strip(channels, amps, target_ch_list):
        out = []
        for ch in target_ch_list:
            hit_mask = channels == ch
            if np.any(hit_mask):
                out.append(np.max(amps[hit_mask]))
            else:
                out.append(0)
        return np.array(out)

    x_max_amps = max_per_strip(evt_channels, evt_amplitudes, x_channels)
    y_max_amps = max_per_strip(evt_channels, evt_amplitudes, y_channels)

    # 1D plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(x_max_amps); ax1.set_ylabel("X Max Amp")
    ax2.plot(y_max_amps); ax2.set_ylabel("Y Max Amp"); ax2.set_xlabel("Strip index")
    plt.tight_layout()

    # 2D grid of X+Y
    plot_flash_xy_grid(x_max_amps, y_max_amps, detector)
    plt.show()



def plot_waveforms(waveform_file, event_id):
    """
    waveform_file : str   path to ROOT file with the 'nt' tree (raw waveforms)
    event_id      : int   eventId to plot
    """

    # ---------------------------
    # Read waveform tree
    # ---------------------------
    with uproot.open(waveform_file) as f:
        nt = f["nt"]
        evt_ids = nt["eventId"].array(library="np")
        samples = nt["sample"].array(library="np")     # Jagged array
        channels = nt["channel"].array(library="np")   # Jagged array
        amplitudes = nt["amplitude"].array(library="np")

    # Find index in nt corresponding to the eventId
    match = np.where(evt_ids == event_id)[0]
    if len(match) == 0:
        raise ValueError(f"Event {event_id} not found in nt tree.")
    idx = match[0]

    # Extract this event
    evt_samples = samples[idx]
    evt_channels = channels[idx]
    evt_amplitudes = amplitudes[idx]


    waveform_sample_idx = evt_samples
    waveform_amp = evt_amplitudes

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(10,6))

    # Plot waveform for each channel
    unique_channels = np.unique(evt_channels)
    for ch in unique_channels:
        ch_mask = (evt_channels == ch)
        plt.plot(waveform_sample_idx[ch_mask], waveform_amp[ch_mask], lw=0.7)


    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_flash_xy_grid(x_max_amps, y_max_amps, detector='strip'):
    xy_grid = x_max_amps[:, None] + y_max_amps[None, :]

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(xy_grid.T, origin='lower', aspect='auto', cmap='jet', vmin=1)
    plt.colorbar(im, ax=ax).set_label("X + Y Max Amplitude")
    ax.set_xlabel("X Strip")
    ax.set_ylabel("Y Strip")
    ax.set_title(f"2D Flash Map ({detector})")
    plt.tight_layout()


def get_thresh_sample(evt_channels, evt_samples, evt_amplitudes, threshold):
    """
    Returns first sample index where ANY channel exceeds threshold.
    """

    mask = evt_amplitudes > threshold
    if not np.any(mask):
        return -1

    return int(np.min(evt_samples[mask]))




def plot_flash_xy_waveforms(evt_channels, evt_samples, evt_amplitudes,
                            xy_map, detector='strip',
                            min_sample=None, max_sample=None,
                            event_id=None, run=None):
    """
    Plot waveforms for X- and Y-side strips based on flat jagged input data.
    """

    # Expand connectors
    def expand(conn):
        out = []
        for c in conn:
            start = (c - 1) * 64
            out.extend(range(start, start + 64))
        return np.array(out)

    x_channels = expand(xy_map[detector]['x'])
    y_channels = expand(xy_map[detector]['y'])

    # Masks
    x_mask = np.isin(evt_channels, x_channels)
    y_mask = np.isin(evt_channels, y_channels)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # --- PLOT X WAVEFORMS ---
    unique_x = np.unique(evt_channels[x_mask])
    for ch in unique_x:
        m = (evt_channels == ch)
        ax_x.plot(evt_samples[m], evt_amplitudes[m], lw=0.7)
    ax_x.set_ylabel("X Amplitude")

    # --- PLOT Y WAVEFORMS ---
    unique_y = np.unique(evt_channels[y_mask])
    for ch in unique_y:
        m = (evt_channels == ch)
        ax_y.plot(evt_samples[m], evt_amplitudes[m], lw=0.7)
    ax_y.set_ylabel("Y Amplitude")
    ax_y.set_xlabel("Sample")

    if min_sample is not None:
        ax_x.axvline(min_sample, color='red', ls='-')
        ax_y.axvline(min_sample, color='red', ls='-')
    if max_sample is not None:
        ax_x.axvline(max_sample, color='red', ls='-')
        ax_y.axvline(max_sample, color='red', ls='-')

    title = f"Flash Waveforms ({detector})"
    if event_id is not None: title += f" — Event {event_id}"
    if run is not None: title += f" — Run {run}"
    fig.suptitle(title)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)


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


def check_timestamps(timestamps):
    """
    Check timestamps to see if they are tagged on the first sample after the trigger or on the trigger itself.
    Trigger comes in on some 10 ns clock cycle. If sample period is 20 ns, then the trigger has 50% chance of
    being in the middle of two samples. If timestamps are tagged on the first sample after the trigger, then we
    expect for 20 ns period that timestamps will be either only even or only odd.
    """
    remainders = timestamps % 2
    unique_remainders = np.unique(remainders)
    if len(unique_remainders) == 1:
        if unique_remainders[0] == 0:
            print("Timestamps are all even - likely tagged on first sample after trigger with 20 ns period.")
        else:
            print("Timestamps are all odd - likely tagged on first sample after trigger with 20 ns period.")
    else:
        print("Timestamps contain both even and odd values - likely tagged on the trigger itself.")
    # Plot remainder histogram
    plt.figure(figsize=(6,4))
    plt.hist(remainders, bins=[-0.5, 0.5, 1.5], align='mid', rwidth=0.5)
    plt.xticks([0, 1], ['Even', 'Odd'])
    plt.xlabel("Timestamp % 2")
    plt.ylabel("Count")
    plt.title("Timestamp Remainder Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
