#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 17 12:08 PM 2026
Created in PyCharm
Created as nTof_x17/plot_hv_current.py

@author: Dylan Neff, dylan
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    # runs = ['run_19', 'run_23', 'run_25', 'run_26', 'run_84', 'run_88']
    # runs = ['run_64', 'run_71']
    runs = ['run_126']
    # sub_run = 'resist_720V_drift_600V'
    # base_path = '/media/dylan/data/x17/cosmic_bench/det_1/'
    # run = 'mx17_det1_daytime_run_1-28-26'
    # run = 'mx17_det1_1-27-26'
    # sub_run = 'overnight_run'

    hv_channel = '2:0'
    # hv_channel = '3:0'

    fig_i_v, ax_i_v = plt.subplots()
    cmap = plt.get_cmap('tab10')

    for i, run in enumerate(runs):
        run_dir = f'{base_path}{run}'
        sub_runs = [x for x in os.listdir(run_dir) if os.path.isdir(f'{run_dir}/{x}')]
        fig, axs = plt.subplots(nrows=2, sharex=True)
        for sub_run in sub_runs:
            print(f'{run}/{sub_run}')
            plot_hv_current_vs_time(base_path, run, sub_run, hv_channel, axs=axs)
            plot_current_vs_hv(base_path, run, sub_run, hv_channel, ax=ax_i_v, color=cmap(i), skip_ramps=True)

        fig.suptitle(f'HV vs Current for {run}')
        fig.tight_layout()

    # 1. Create a list of "proxy" lines for the legend
    legend_elements = [
        Line2D([0], [0], color=cmap(i), lw=2, label=run)
        for i, run in enumerate(runs)
    ]

    # 2. Add the legend manually to the plot
    ax_i_v.legend(handles=legend_elements, title="Runs")

    ax_i_v.set_xlabel('HV (V)')
    ax_i_v.set_ylabel('Current (A)')
    fig_i_v.suptitle(f'Current vs HV')
    fig_i_v.tight_layout()

    plt.show()

    print('donzo')


def plot_hv_current_vs_time(base_path, run, sub_run, hv_channel, axs=None):
    """
    Plot hv and current vs time
    """
    sub_run_dir = f'{base_path}{run}/{sub_run}'
    hv_file = f'{sub_run_dir}/hv_monitor.csv'
    hv_df = pd.read_csv(hv_file)
    hv_df['timestamp'] = pd.to_datetime(hv_df['timestamp'], unit='s', errors='coerce')

    # Plot hv and current vs time
    if axs is None:
        fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(hv_df['timestamp'], hv_df[f'{hv_channel} vmon'], label=sub_run)
    axs[1].plot(hv_df['timestamp'], hv_df[f'{hv_channel} imon'])
    axs[0].set_ylabel('HV (V)')
    axs[1].set_ylabel('Current (A)')


def plot_current_vs_hv(base_path, run, sub_run, hv_channel, ax=None, color=None, skip_ramps=False):
    """
    Plot current vs hv
    """
    sub_run_dir = f'{base_path}{run}/{sub_run}'
    hv_file = f'{sub_run_dir}/hv_monitor.csv'
    hv_df = pd.read_csv(hv_file)
    hv_df['timestamp'] = pd.to_datetime(hv_df['timestamp'], unit='s', errors='coerce')

    if skip_ramps:
        # Define a window size (e.g., 5-10 samples) and a threshold
        window_size = 5
        # Threshold: if HV stays within +/- 0.5V, it's "steady"
        threshold = 0.5

        # Calculate rolling standard deviation
        hv_df['hv_std'] = hv_df[f'{hv_channel} vmon'].rolling(window=window_size, center=True).std()

        # Filter for steady states
        hv_df = hv_df[hv_df['hv_std'] < threshold].copy()

    # Trim time off the start and end
    # trim_time = pd.Timedelta(seconds=30)  # s
    # min_time, max_time = hv_df['timestamp'].min() + trim_time, hv_df['timestamp'].max() - trim_time
    # hv_df = hv_df[(hv_df['timestamp'] >= min_time) & (hv_df['timestamp'] <= max_time)]

    # Plot hv and current vs time
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('HV (V)')
        ax.set_ylabel('Current (A)')
    if color is None:
        ax.scatter(hv_df[f'{hv_channel} vmon'], hv_df[f'{hv_channel} imon'], s=2)
    else:
        ax.scatter(hv_df[f'{hv_channel} vmon'], hv_df[f'{hv_channel} imon'], s=2, color=color)


if __name__ == '__main__':
    main()
