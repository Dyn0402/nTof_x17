#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on January 27 10:03 PM 2026
Created in PyCharm
Created as nTof_x17/plot_cosmic_hits.py

@author: Dylan Neff, dylan
"""

import os
import re
from typing import Tuple, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import uproot
from scipy.optimize import curve_fit as cf

from Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions


def main():
    # base_path = '/media/dylan/data/x17/cosmic_bench/det_1/mx17_1-27-26/'
    base_path = '/media/dylan/data/x17/cosmic_bench/det_1/'
    run = 'mx17_det1_daytime_run_1-28-26'
    sub_run = 'overnight_run'
    # base_path = '/mnt/data/x17/cosmic_bench/det_1/'
    # run = 'mx17_det1_overnight_run_1-27-26'
    # sub_run = 'overnight_run'
    # feus = {1: 'm3', 4: 'MX17 X Strips', 6: 'MX17 Y Strips'}
    file_nums = [0]

    run_config_path = f'{base_path}{run}/run_config.json'
    map_csv_path = './mx17_m4_map.csv'

    rc = RunConfig(run_config_path, map_csv_path)

    det = rc.get_detector('mx17_1')

    hits_dir = f'{base_path}{run}/{sub_run}/combined_hits_root/'
    hit_files = [f for f in os.listdir(hits_dir) if f.endswith('.root') and '_datrun_' in f]

    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')
    print(df)

    df = df[df['feu'].isin([4, 6])]

    feus_array = df['feu'].to_numpy()
    channels_array = df['channel'].to_numpy()

    x_positions, all_xs = [], []
    y_positions, all_ys = [], []

    for feu, channel in zip(feus_array, channels_array):
        pos = det.map_hit(feu, channel)
        if pos is not None:
            x_mm, y_mm = pos
            all_xs.append(x_mm)
            all_ys.append(y_mm)
            if x_mm is not None:
                x_positions.append(x_mm)
            if y_mm is not None:
                y_positions.append(y_mm)
        else:
            print(f'No mapping found for FEU {feu}, channel {channel}')
            all_xs.append(None)
            all_ys.append(None)

    # Append x and y positions to dataframe
    df['x_position_mm'] = all_xs
    df['y_position_mm'] = all_ys
    print(f'Minimum event number: {df["eventId"].min()}')

    # Load m3s
    rays_dir = f'{base_path}{run}/{sub_run}/m3_tracking_root/'
    for file in os.listdir(rays_dir):
        if file.endswith('.root'):
            with uproot.open(f'{rays_dir}{file}') as f:
                tree_name = f"{f.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in f.keys()])}"
                tree = f[tree_name]  # Get tree with max ;# at end
                new_data = tree.arrays(['evn'], library='ak')
                print(f'File {file} has {len(new_data)} events. Minimum is {min(new_data["evn"])}.')
    rays = M3RefTracking(rays_dir, chi2_cut=20)
    x_angles, y_angles, ray_event_nums = get_xy_angles(rays.ray_data)
    x_positions, y_positions, ray_pos_event_nums = get_xy_positions(rays.ray_data, 250)

    fig, ax = plt.subplots()
    ax.scatter(np.rad2deg(x_angles), np.rad2deg(y_angles), alpha=0.5, color='red')
    ax.set_xlabel('X Angle (deg)')
    ax.set_ylabel('Y Angle (deg)')
    ax.set_title('M3 Angles')
    fig.tight_layout()

    # Correlate tan(theta) with angles.
    x_slopes, y_slopes, x_pos, y_pos = [], [], [], []
    for event_num in ray_event_nums:
        print(f'\nEvent {event_num}:')
        if event_num > 83000:
            plot = True
        else:
            plot = False
        x_slope, y_slope, x, y = line_fit_test(df, event_number=event_num, plot=plot)
        x_slopes.append(1 / -x_slope)
        y_slopes.append(1 / y_slope)
        x_pos.append(-x)
        y_pos.append(y)

    # Plot correlation between slopes and tangent of angles
    fig, ax = plt.subplots()
    ax.scatter(x_slopes, np.tan(x_angles), alpha=0.5, color='red')
    ax.set_xlim(-5, 5)
    ax.set_xlabel('X Slope')
    ax.set_ylabel('Tangent of X Angle')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(y_slopes, np.tan(y_angles), alpha=0.5, color='blue')
    ax.set_xlim(-5, 5)
    ax.set_xlabel('Y Slope')
    ax.set_ylabel('Tangent of Y Angle')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(x_pos, x_positions, alpha=0.5, color='red')
    ax.set_xlabel('X Position Det (mm)')
    ax.set_ylabel('X Position Rays (mm)')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(y_pos, y_positions, alpha=0.5, color='red')
    ax.set_xlabel('Y Position Det (mm)')
    ax.set_ylabel('Y Position Rays (mm)')
    fig.tight_layout()

    plt.show()


def line_fit_test(df, event_number, plot=True):
    df = df[df['eventId'] == event_number]

    # df_x = df[df['x_position_mm'].notna()]
    # df_y = df[df['y_position_mm'].notna()]

    # Clustering
    # df_x = df_x.sort_values('x_position_mm').reset_index(drop=True)
    # df_y = df_y.sort_values('y_position_mm').reset_index(drop=True)

    # print(f'dfx:\n{df_x}\ndfy:\n{df_y}\n')

    gap_threshold = 3  # mm

    df_x = df[df["x_position_mm"].notna()].sort_values("x_position_mm").reset_index(drop=True)
    df_y = df[df["y_position_mm"].notna()].sort_values("y_position_mm").reset_index(drop=True)

    df_x["cluster"] = (df_x["x_position_mm"].diff().gt(gap_threshold).fillna(False)).cumsum()
    df_y["cluster"] = (df_y["y_position_mm"].diff().gt(gap_threshold).fillna(False)).cumsum()

    # print(f'dfx cluster:\n{df_x["cluster"]}\ndfy cluster:\n{df_y["cluster"]}\n')

    # If no clusters, return nans
    if df_x.empty or df_y.empty or not (df_x["cluster"].any() and df_y["cluster"].any()):
        return np.nan, np.nan, np.nan, np.nan

    # Get the largest cluster number
    largest_cluster_x = df_x['cluster'].value_counts().idxmax()
    largest_cluster_y = df_y['cluster'].value_counts().idxmax()

    # For now take only the largest cluster. Print how many strips are dropped outside of this cluster.
    non_largest_clusters_x = df_x[df_x['cluster'] != largest_cluster_x]
    non_largest_clusters_y = df_y[df_y['cluster'] != largest_cluster_y]
    print(f'Dropped {len(non_largest_clusters_x)} strips outside of largest cluster in X direction.')
    print(f'Dropped {len(non_largest_clusters_y)} strips outside of largest cluster in Y direction.')

    df_x_dropped = df_x[df_x['cluster'] != largest_cluster_x]
    df_y_dropped = df_y[df_y['cluster'] != largest_cluster_y]

    df_x = df_x[df_x['cluster'] == largest_cluster_x]
    df_y = df_y[df_y['cluster'] == largest_cluster_y]

    # For x and y separately, find the earliest entry in time.
    x_earliest_time = df_x['time'].min()
    x_pos_earliest_time = df_x[df_x['time'] == x_earliest_time]['x_position_mm'].values[0]
    y_earliest_time = df_y['time'].min()
    y_pos_earliest_time = df_y[df_y['time'] == y_earliest_time]['y_position_mm'].values[0]

    eps = 1e-9  # avoid divide-by-zero

    sigma_x = 1.0 / np.sqrt(df_x["amplitude"].to_numpy() + eps)
    sigma_y = 1.0 / np.sqrt(df_y["amplitude"].to_numpy() + eps)

    # Fit x and y positions to a line with x_earliest_time as the fixed point.
    popt_x, pcov_x = cf(
        lambda pos, m: line_fixed_point(pos, m, x_pos_earliest_time, x_earliest_time),
        df_x["x_position_mm"].to_numpy(),
        df_x["time"].to_numpy(),
        sigma=sigma_x,
        absolute_sigma=False,  # usually fine unless sigma is a true time-uncertainty in ns
    )

    popt_y, pcov_y = cf(
        lambda pos, m: line_fixed_point(pos, m, y_pos_earliest_time, y_earliest_time),
        df_y["y_position_mm"].to_numpy(),
        df_y["time"].to_numpy(),
        sigma=sigma_y,
        absolute_sigma=False,
    )

    if event_number == 84806:
        print(f'popt_x: {popt_x}\npopt_y: {popt_y}')
        print(f'Fit data:\nx: {df_x["x_position_mm"].to_numpy()}\nx_time= {df_x["time"].to_numpy()}'
              f'\ny: {df_y["y_position_mm"].to_numpy()}\ny_time= {df_y["time"].to_numpy()}')

    if plot:
        # Plot x position vs time. Do the same for y position.
        # Create a figure with two columns sharing the y-axis
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # --- Left Plot: X Position ---
        # Map amplitude to color (c=...) using 'jet' cmap instead of size (s=...)
        sc1 = axes[0].scatter(df_x['x_position_mm'], df_x['time'], c=df_x['amplitude'], cmap='jet', alpha=1)
        sc1 = axes[0].scatter(df_x_dropped['x_position_mm'], df_x_dropped['time'], c=df_x_dropped['amplitude'],
                              marker='^', cmap='jet', alpha=1)
        line_x_time = line_fixed_point(df['x_position_mm'], popt_x[0], x_pos_earliest_time, x_earliest_time)
        axes[0].plot(df['x_position_mm'], line_x_time, color='red')

        axes[0].set_ylabel('Time (ns)')
        axes[0].set_xlabel('X Position (mm)')
        axes[0].set_title(f'X Position vs Time for Event {event_number}')

        # --- Right Plot: Y Position ---
        sc2 = axes[1].scatter(df_y['y_position_mm'], df_y['time'], c=df_y['amplitude'], cmap='jet', alpha=1)
        sc2 = axes[1].scatter(df_y_dropped['y_position_mm'], df_y_dropped['time'], c=df_y_dropped['amplitude'],
                              marker='^', cmap='jet', alpha=1)
        line_y_time = line_fixed_point(df['y_position_mm'], popt_y[0], y_pos_earliest_time, y_earliest_time)
        axes[1].plot(df['y_position_mm'], line_y_time, color='blue')

        axes[1].set_xlabel('Y Position (mm)')
        axes[1].set_title(f'Y Position vs Time for Event {event_number}')

        # --- Adjustments ---
        # Scale y-axis based ONLY on the scatter points range
        y_min, y_max = df['time'].min(), df['time'].max()
        padding = (y_max - y_min) * 0.05  # 5% padding for better visibility
        axes[0].set_ylim(y_min - padding, y_max + padding)

        # Add a colorbar to indicate amplitude values
        # fig.colorbar(sc2, ax=axes, label='Amplitude')

        fig.tight_layout()

    return popt_x[0], popt_y[0], x_pos_earliest_time, y_pos_earliest_time

    # min_amp = 200
    # df = df[df['amplitude'] >= min_amp]



def extract_file_numbers_tuple(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extracts the file number (xxx) and FEU number (yy) from a filename
    following the pattern ..._xxx_yy.ext

    Args:
        filename (str): The name of the file.

    Returns:
        Optional[Tuple[int, int]]: A tuple (file_number, feu_number) as integers,
                                    or None if the pattern is not found.
    """
    # Pattern explanation:
    # r'.*': Matches any characters at the start (non-greedy)
    # '_': Matches the literal underscore
    # '(\d{3})': Capturing group 1: exactly three digits (xxx)
    # '_': Matches the literal underscore
    # '(\d{2})': Capturing group 2: exactly two digits (yy)
    # '*': Matches anything at the end
    pattern = r'.*_(\d{3})_(\d{2})*'

    match = re.match(pattern, filename)

    if match:
        # Group 1 is the file number (xxx), Group 2 is the FEU number (yy)
        file_num_str, feu_num_str = match.groups()
        # Convert the strings to integers and return
        return (int(file_num_str), int(feu_num_str))
    else:
        return None


def line_fixed_point(x, m, x0, y0):
    """
    Line but with one point fixed
    """
    return m * (x - x0) + y0


if __name__ == '__main__':
    main()
