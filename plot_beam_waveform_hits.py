#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 08 4:30 PM 2026
Created in PyCharm
Created as nTof_x17/plot_beam_waveform_hits.py

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
    # run = 'run_49'
    # subrun = 'resist_510V_drift_600V'
    # subrun = 'drift_600V_6'
    # run = 'run_18'
    # subrun = 'resist_440V_drift_600V'
    # run = 'run_19'
    # subrun = 'resist_440V_drift_600V'
    run = 'run_34'
    subrun = 'resist_425V_drift_600V'
    # run = 'run_60'
    # subrun = 'resist_580V_drift_600V'
    # run = 'run_59'
    # subrun = 'resist_550V_drift_600V'
    # run = 'run_84'
    # subrun = 'resist_490V_drift_1000V'
    # run = 'run_88'
    # subrun = 'resist_500V_drift_1000V'
    # run = 'run_88'
    # subrun = 'resist_530V_drift_1000V'
    # run = 'run_26'
    # subrun = 'resist_410V_drift_600V'
    # run = 'run_103'
    # subrun = 'resist_540V_drift_1000V'
    # run = 'run_71'
    # run = 'run_88'
    # subrun = 'resist_690V_drift_1000V'
    # run = 'run_124'
    # subrun = 'resist_710V_drift_1000V'
    # run = 'run_137'
    # subrun = 'resist_0V_drift_0V'
    # subrun = 'resist_530V_drift_1000V'
    # subrun = 'initial_resist_550V_drift_1000V'
    # run = 'run_140'
    # subrun = 'resist_0V_drift_0V'
    # run = 'run_74'
    # subrun = 'resist_650V_drift_1000V'
    # run = 'run_114'
    # subrun = 'resist_620V_drift_1000V'
    # run = 'run_124'
    # subrun = 'resist_680V_drift_1000V'
    # run = 'run_128'
    # subrun = 'resist_530V_drift_1000V'
    # run = 'run_139'
    # subrun = 'resist_490V_drift_1000V'
    # run = 'run_141'
    # subrun = 'resist_450V_drift_800V'


    feu_nums = {4: 'y', 5: 'x'}  # 4 goes in x and gives y position
    # feu_nums = {4: 'y', 6: 'x'}  # 4 goes in x and gives y position
    file_num = 0
    event = 14
    # file_num = 4
    # event = 1811
    channels = None  # To select specific channels to plot
    # channels = np.array([103, 104, 105])  # To select specific channels to plot
    # channels = np.array([4, 5, 6]) + 64  # To select specific channels to plot
    # channels = np.array([8, 9, 10]) + 64  # To select specific channels to plot
    # channels = np.array([17, 18, 19]) + 64  # To select specific channels to plot
    # channels = np.array([-30, 4, 5, 6, 16, 17, 18, 19, 8, 9, 10]) + 64  # To select specific channels to plot
    # channels = np.array([5, 18, 9]) + 64  # To select specific channels to plot
    # channels = np.array([16, 4]) + 64  # To select specific channels to plot
    # channels = np.arange(1 * 64, 2 * 64)  # To select specific channels to plot

    decoded_dir = 'decoded_root'

    plot_waveform_hits_figure(base_path, run, subrun, decoded_dir, feu_nums, file_num, event, channels=channels)
    # plot_waveform_hits_figure_att_test(base_path, run, subrun, decoded_dir, feu_nums, file_num, event, channels=channels)
    # plot_waveform_hits_figure_single_feu(base_path, run, subrun, decoded_dir, 6, file_num, event, channels=channels)
    plt.show()

    print('donzo')


def plot_waveform_hits_figure(base_path, run, subrun, decoded_dir, feu_nums, file_num, event, channels=None):
    """
    Plot combination waveform and hits figure. Waveforms on the top of the figure and hits on the bottom.
    """
    # min_hit_amp = 50
    min_hit_amp = 400
    ns_per_sample = 20

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 8))

    decoded_dir = os.path.join(base_path, run, subrun, decoded_dir)
    evt_samples, evt_channels, evt_amplitudes = load_decoded_waveforms(decoded_dir, feu_nums, file_num)
    evt_ids, timestamps = load_evtids_timestamps(decoded_dir, list(feu_nums.keys())[0], file_num)
    evt_index = np.where(evt_ids == event)[0][0]
    evt_channels_x, evt_amplitudes_x = evt_channels['x'][evt_index], evt_amplitudes['x'][evt_index]
    evt_channels_y, evt_amplitudes_y = evt_channels['y'][evt_index], evt_amplitudes['y'][evt_index]
    evt_times_x = evt_samples['x'][evt_index] * ns_per_sample / 1000  # Convert from samples to microseconds
    evt_times_y = evt_samples['y'][evt_index] * ns_per_sample / 1000

    # --- PLOT X WAVEFORMS ---
    unique_x = np.unique(evt_channels_x)
    cmap_x = plt.get_cmap("coolwarm")
    norm_x = mcolors.Normalize(vmin=unique_x.min(), vmax=unique_x.max())
    for ch in unique_x:
        if channels is not None and ch not in channels: continue
        m = (evt_channels_x == ch)
        axs[0].plot(evt_times_x[m], evt_amplitudes_x[m], lw=0.7, color=cmap_x(norm_x(ch)))
        # Superimpose derivative
        # d_evt_amplitudes_x = evt_amplitudes_x[m].astype(float)
        # evt_times_x_avg = (evt_times_x[m][:-1] + evt_times_x[m][1:]) / 2
        # axs[0].plot(evt_times_x_avg, d_evt_amplitudes_x[1:] - d_evt_amplitudes_x[:-1], lw=0.7, color='red')
    axs[0].set_ylabel("X Amplitude")

    # --- PLOT Y WAVEFORMS ---
    unique_y = np.unique(evt_channels_y)
    cmap_y = plt.get_cmap("coolwarm")
    norm_y = mcolors.Normalize(vmin=unique_y.min(), vmax=unique_y.max())
    for ch in unique_y:
        if channels is not None and ch not in channels: continue
        m = (evt_channels_y == ch)
        axs[2].plot(evt_times_y[m], evt_amplitudes_y[m], lw=0.7,
                  color=cmap_y(norm_y(ch)))
    axs[2].set_ylabel("Y Amplitude")
    # ax_y.set_xlabel("Sample")

    # if min_sample is not None:
    #     ax_x.axvline(min_sample, color='red', ls='-')
    #     ax_y.axvline(min_sample, color='red', ls='-')
    # if max_sample is not None:
    #     ax_x.axvline(max_sample, color='red', ls='-')
    #     ax_y.axvline(max_sample, color='red', ls='-')

    title = f"Event Waveforms and Hits"
    if event is not None: title += f" — Event {event}"
    if run is not None: title += f" — {run}"
    if subrun is not None: title += f' - {subrun}'
    fig.suptitle(title)

    # Get hits
    df, det = load_subrun(base_path, run, subrun, list(feu_nums.keys()))
    df = df[(df['eventId'] == event) & (df['amplitude'] >= min_hit_amp)]
    if channels is not None: df = df[df['channel'].isin(channels)]
    df = add_xy_pos(df, det)
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

    axs[1].scatter(df_x['time'] / 3 / 1000, df_x['x_position_mm'], c=df_x['local_max'], cmap='jet')
    axs[1].set_ylabel('X Hit Position (mm)')
    fig.tight_layout()

    axs[3].scatter(df_y['time'] / 3 / 1000, df_y['y_position_mm'], c=df_y['local_max'], cmap='jet')
    axs[3].set_ylabel('Y Hit Position (mm)')
    axs[3].set_xlabel('Time (μs)')
    fig.tight_layout()

    plt.subplots_adjust(hspace=0.0)


def plot_waveform_hits_figure_att_test(base_path, run, subrun, decoded_dir, feu_nums, file_num, event, channels=None):
    """
    Plot combination waveform and hits figure. Waveforms on the top of the figure and hits on the bottom.
    """
    # min_hit_amp = 50
    min_hit_amp = 200
    ns_per_sample = 20
    ch_map = {4: 'x2', 5: 'x2', 6: 'x2', 8: 'x5', 9: 'x5', 10: 'x5', 17: 'x10', 18: 'x10', 19: 'x10'}
    color_map = {'x2': 'blue', 'x5': 'green', 'x10': 'red', 'x0': 'black'}

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 5))

    decoded_dir = os.path.join(base_path, run, subrun, decoded_dir)
    evt_samples, evt_channels, evt_amplitudes = load_decoded_waveforms(decoded_dir, feu_nums, file_num)
    evt_ids, timestamps = load_evtids_timestamps(decoded_dir, list(feu_nums.keys())[0], file_num)
    evt_index = np.where(evt_ids == event)[0][0]
    evt_channels_x, evt_amplitudes_x = evt_channels['x'][evt_index], evt_amplitudes['x'][evt_index]
    evt_channels_y, evt_amplitudes_y = evt_channels['y'][evt_index], evt_amplitudes['y'][evt_index]
    evt_times_x = evt_samples['x'][evt_index] * ns_per_sample / 1000  # Convert from samples to microseconds
    evt_times_y = evt_samples['y'][evt_index] * ns_per_sample / 1000

    # --- PLOT X WAVEFORMS ---
    unique_x = np.unique(evt_channels_x)
    for ch in unique_x:
        if channels is not None and ch not in channels: continue
        ch -= 1
        m = (evt_channels_x == ch)
        ch_n = ch - 64 + 1
        ch_att = ch_map.get(ch_n, 'x0')
        ax.plot(evt_times_x[m], evt_amplitudes_x[m], lw=1.5, color=color_map[ch_att], label=f'Channel {ch_n} - {ch_att}')
        # Superimpose derivative
        # d_evt_amplitudes_x = evt_amplitudes_x[m].astype(float)
        # evt_times_x_avg = (evt_times_x[m][:-1] + evt_times_x[m][1:]) / 2
        # axs[0].plot(evt_times_x_avg, d_evt_amplitudes_x[1:] - d_evt_amplitudes_x[:-1], lw=0.7, color='red')
    ax.set_ylabel("X Amplitude")
    ax.legend()

    title = f"Event Waveforms and Hits"
    if event is not None: title += f" — Event {event}"
    if run is not None: title += f" — {run}"
    if subrun is not None: title += f' - {subrun}'
    fig.suptitle(title)

    ax.set_xlabel('Time (μs)')
    fig.tight_layout()

    plt.subplots_adjust(hspace=0.0)


def plot_waveform_hits_figure_single_feu(base_path, run, subrun, decoded_dir, feu_num, file_num, event, channels=None):
    """
    Plot combination waveform and hits figure. Waveforms on the top of the figure and hits on the bottom.
    """
    # min_hit_amp = 250
    min_hit_amp = 400
    ns_per_sample = 20

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    decoded_dir = os.path.join(base_path, run, subrun, decoded_dir)
    evt_samples, evt_channels, evt_amplitudes = load_decoded_waveforms(decoded_dir, {feu_num: 'feu'}, file_num)
    evt_ids, timestamps = load_evtids_timestamps(decoded_dir, feu_num, file_num)
    evt_index = np.where(evt_ids == event)[0][0]
    evt_channels, evt_amplitudes = evt_channels['feu'][evt_index], evt_amplitudes['feu'][evt_index]
    evt_times = evt_samples['feu'][evt_index] * ns_per_sample / 1000  # Convert from samples to microseconds

    # --- PLOT WAVEFORMS ---
    unique = np.unique(evt_channels)
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=unique.min(), vmax=unique.max())
    for ch in unique:
        if channels is not None and ch not in channels: continue
        m = (evt_channels == ch)
        axs[0].plot(evt_times[m], evt_amplitudes[m], lw=0.7, color=cmap(norm(ch)))
    axs[0].set_ylabel("Amplitude")

    # ax_y.set_xlabel("Sample")
    title = f"Event Waveforms and Hits"
    if event is not None: title += f" — Event {event}"
    if run is not None: title += f" — {run}"
    if subrun is not None: title += f' - {subrun}'
    fig.suptitle(title)

    # Get hits
    df, det = load_subrun(base_path, run, subrun, [feu_num])
    df = df[(df['eventId'] == event) & (df['amplitude'] >= min_hit_amp)]
    if channels is not None: df = df[df['channel'].isin(channels)]
    df = add_xy_pos(df, det)

    axs[1].scatter(df['time'] / 3 / 1000, df['channel'], c=df['local_max'], cmap='jet')
    axs[1].set_ylabel('Channel Number')
    fig.tight_layout()

    plt.subplots_adjust(hspace=0.0)


def plot_hits_xy_vs_time(base_path, run, sub_run, feus, event):
    """
    Plot hits on position vs time plot.
    """
    min_amp = 250
    print(f'Loading data for {run} - {sub_run} Event {event}...')
    df, det = load_subrun(base_path, run, sub_run, feus)
    print(f'Adding xy positions to dataframe...')
    df = df[(df['amplitude'] >= min_amp) & (df['eventId'] == event)]
    df = add_xy_pos(df, det)
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

    print(f"df_x:\n{df_x[['amplitude', 'x_position_mm']]}\n\ndf_y:\n{df_y[['amplitude', 'y_position_mm']]}\n")

    # For x and y separately, plot scatter of hit position on the y axis vs time on the x axis
    fig, ax = plt.subplots()
    ax.scatter(df_x['time'] / 3 / 1000, df_x['x_position_mm'], c=df_x['local_max'], cmap='jet')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('X Position (mm)')
    ax.set_title(f'Hits on X Position vs Time for {run} - {sub_run} Event {event}')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(df_y['time'] / 3 / 1000, df_y['y_position_mm'], c=df_y['local_max'], cmap='jet')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title(f'Hits on Y Position vs Time for {run} - {sub_run} Event {event}')
    fig.tight_layout()


if __name__ == '__main__':
    main()
