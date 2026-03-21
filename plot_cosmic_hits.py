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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot

from common.Mx17StripMap import RunConfig


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

    min_amp = 200
    df = df[df['amplitude'] >= min_amp]

    amplitudes_array = df['amplitude'].to_numpy()

    # Plot amplitudes histogram in log scale
    fig, ax = plt.subplots()
    ax.hist(amplitudes_array, bins=100, color='orange', alpha=0.7, log=True)
    ax.set_title('Amplitudes Histogram')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Counts (log scale)')
    plt.tight_layout()


    # for hit_file in hit_files:
    #     with uproot.open(f'{hits_dir}{hit_file}') as f:
    #         tree = f['hits']
    #         feus_array = tree['feu'].array(library='np')
    #         channels_array = tree['channel'].array(library='np')

    # Plot 1D arrays for x and y positions
    pitch = 0.78
    bins = np.arange(-pitch / 2, 512 * pitch + pitch / 2, pitch)
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].hist(x_positions, bins=bins, color='blue', alpha=0.7)
    axs[0].set_title('X Positions Histogram')
    axs[0].set_xlabel('X Position (mm)')
    axs[0].set_ylabel('Counts')
    axs[1].hist(y_positions, bins=bins, color='green', alpha=0.7)
    axs[1].set_title('Y Positions Histogram')
    axs[1].set_xlabel('Y Position (mm)')
    axs[1].set_ylabel('Counts')
    plt.tight_layout()

    event_ids = df['eventId'].to_numpy()
    event_ids, counts = np.unique(event_ids, return_counts=True)
    fig, ax = plt.subplots()
    ax.hist(counts, bins=np.arange(-0.5, np.max(counts) + 0.5, 1), color='purple', alpha=0.7)
    ax.set_title('Hits per Event Histogram')
    ax.set_xlabel('Number of Hits in Event')
    ax.set_ylabel('Counts')
    plt.tight_layout()

    # Group by event number. Get average x and y positions per event, ignoring None values
    grouped = df.groupby('eventId')
    avg_x_positions = grouped['x_position_mm'].mean().to_numpy()
    avg_y_positions = grouped['y_position_mm'].mean().to_numpy()

    # Drop entries where either avg_x_positions or avg_y_positions is NaN
    valid_indices = ~np.isnan(avg_x_positions) & ~np.isnan(avg_y_positions)
    avg_x_positions = avg_x_positions[valid_indices]
    avg_y_positions = avg_y_positions[valid_indices]

    # Calculate the percentage drop due to NaN values
    total_events = len(grouped)
    valid_events = len(avg_x_positions)
    drop_percentage = (total_events - valid_events) / total_events * 100
    print(f'Dropped {drop_percentage:.2f}% of events due to NaN average positions.')

    # 2D histogram of average x and y positions
    fig, ax = plt.subplots()
    h = ax.hist2d(avg_x_positions, avg_y_positions, bins=100, norm=LogNorm(), cmap='jet')
    ax.set_title('2D Histogram of Average X and Y Positions per Event')
    ax.set_xlabel('Average X Position (mm)')
    ax.set_ylabel('Average Y Position (mm)')
    plt.colorbar(h[3], ax=ax, label='Counts')
    plt.tight_layout()

    # Scatter plot of average x and y positions
    fig, ax = plt.subplots()
    ax.scatter(avg_x_positions, avg_y_positions, alpha=0.3, color='green', s=1)
    ax.set_title('Scatter Plot of Average X and Y Positions per Event')
    ax.set_xlabel('Average X Position (mm)')
    ax.set_ylabel('Average Y Position (mm)')
    plt.tight_layout()

    plt.show()



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


if __name__ == '__main__':
    main()
