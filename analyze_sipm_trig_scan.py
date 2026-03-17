#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 23 3:35 PM 2026
Created in PyCharm
Created as nTof_x17/analyze_sipm_trig_scan.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_beam_hits import load_subrun, get_run_time, plot_2ds


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    run = 'run_93'
    sub_run = 'initial_resist_540V_drift_1000V'
    feus_map = {4: 'y', 5: 'x'}  # Which positions they give
    feus = list(feus_map.keys())

    # get_true_hits(base_path, run, sub_run, feus, feus_map, plot=False)

    run = 'run_94'
    out_csv_path = f'/media/dylan/data/x17/feb_beam/Analysis/Plot_Data/sipm_trig_calibration_{run}.csv'
    # out_csv_path = None
    run_dir = os.path.join(base_path, run)
    hvs, rates, all_rates = [], [], []
    for subrun in os.listdir(run_dir):
        print(subrun)
        subrun_dir = os.path.join(run_dir, subrun)
        if not os.path.isdir(subrun_dir):
            continue
        hv = int(subrun.split('_')[1].strip('V'))
        rate, all_rate = get_true_hits(base_path, run, subrun, feus, feus_map, plot=False)
        hvs.append(hv)
        rates.append(rate)
        all_rates.append(all_rate)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hvs, rates, marker='o', label='Filtered Rate')
    ax.plot(hvs, all_rates, marker='o', linestyle='--', label='All Event Rate')
    ax.axhline(0, color='gray', zorder=0)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))  # minor ticks every 10 V
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))  # major ticks every 20 V
    ax.grid(axis='x', which='minor', linestyle='-', linewidth=0.4, alpha=0.7, zorder=0)
    ax.grid(axis='x', which='major', linestyle='-', linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_ylabel('Rate (Hz)')
    ax.set_xlabel('Resist HV (V)')
    fig.suptitle(f'Event Rate for Beam Scintillator Trigger HV Scan -- {run}')
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, left=0.055, right=0.995, bottom=0.09)

    if out_csv_path is not None:
        df = pd.DataFrame({'HV': hvs, 'Rate': rates, 'All Rate': all_rates})
        df.to_csv(out_csv_path, index=False)

    plt.show()
    print('donzo')


def get_true_hits(base_path, run, sub_run, feus, feus_map, plot=False):
    """
    Plot general metrics for full subrun.
    """
    run_time = get_run_time(base_path, run, sub_run)
    df, det = load_subrun(base_path, run, sub_run, feus)
    pitch = 0.78
    df['axis'] = df['feu'].map(feus_map)
    df['channel_flipped'] = (df['channel'] // 64) * 64 + (63 - (df['channel'] % 64))
    df['position'] = df['channel_flipped'] * pitch

    # Create x_pos: if axis is 'x', use position, else None
    df['x_position_mm'] = np.where(df['axis'] == 'x', df['position'], float('nan'))

    # Create y_pos: if axis is 'y', use position, else None
    df['y_position_mm'] = np.where(df['axis'] == 'y', df['position'], float('nan'))

    min_amp = 100
    df = df[df['amplitude'] >= min_amp]

    # Get number of events total
    all_rate = df['eventId'].nunique() / run_time

    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

    # Count hits outside of min/max time range
    # min_time, max_time = 250, 750
    # min_time, max_time = 1, 2000
    # df_x_out_of_time = df_x[(df_x['time'] < min_time) | (df_x['time'] > max_time)]
    # df_y_out_of_time = df_y[(df_y['time'] < min_time) | (df_y['time'] > max_time)]
    #
    # df_x_out_of_time['out_of_time_hit_count'] = df_x_out_of_time.groupby('eventId')['eventId'].transform('count')
    # df_y_out_of_time['out_of_time_hit_count'] = df_y_out_of_time.groupby('eventId')['eventId'].transform('count')
    #
    # max_out_of_time_hits = 2
    # bad_events_x = df_x_out_of_time[df_x_out_of_time['out_of_time_hit_count'] > max_out_of_time_hits]['eventId'].unique()
    # bad_events_y = df_y_out_of_time[df_y_out_of_time['out_of_time_hit_count'] > max_out_of_time_hits]['eventId'].unique()
    # bad_events = np.intersect1d(bad_events_x, bad_events_y)
    #
    # print(f'Out of time events: {len(bad_events)}, total events: {len(df["eventId"].unique())}')
    # print(f'Percent of events out of time: {len(bad_events) / len(df["eventId"].unique()) * 100:.2f}%')
    #
    # df = df[~df['eventId'].isin(bad_events)]
    df_x = df[df['x_position_mm'].notna()]
    df_y = df[df['y_position_mm'].notna()]

    min_hits, max_hits = 1, 8
    # 1. Calculate hits per event and map it back to every row in the original df
    df_x['event_hit_count'] = df_x.groupby('eventId')['eventId'].transform('count')
    df_y['event_hit_count'] = df_y.groupby('eventId')['eventId'].transform('count')

    # Get event numbers where between min_hits and max_hits hits were found for both x and y
    df_x = df_x[(df_x['event_hit_count'] >= min_hits) & (df_x['event_hit_count'] <= max_hits)]
    df_y = df_y[(df_y['event_hit_count'] >= min_hits) & (df_y['event_hit_count'] <= max_hits)]

    event_nums_x = df_x['eventId'].unique()
    event_nums_y = df_y['eventId'].unique()

    event_nums = np.intersect1d(event_nums_x, event_nums_y)

    # 2. Filter the original dataframe
    df = df[df['eventId'].isin(event_nums)].copy()

    grouped = df.groupby('eventId')
    avg_x_positions = grouped['x_position_mm'].mean().to_numpy()
    avg_y_positions = grouped['y_position_mm'].mean().to_numpy()
    event_ids = grouped['eventId'].first().to_numpy()

    # 1. Define a list of "bad" regions
    bad_regions = [
        {'x': (237, 246), 'y': (382, 385)},
        {'x': (201, 203), 'y': (189, 190)},
        {'x': (120, 130), 'y': (0, 10)}
    ]

    # 2. Initialize a mask of False values (nothing is bad yet)
    combined_bad_mask = np.zeros(len(avg_x_positions), dtype=bool)

    # 3. Update the mask for each region
    for region in bad_regions:
        x_range = region['x']
        y_range = region['y']

        current_mask = (
                (avg_x_positions >= x_range[0]) & (avg_x_positions <= x_range[1]) &
                (avg_y_positions >= y_range[0]) & (avg_y_positions <= y_range[1])
        )

        # Combine using the 'OR' operator
        combined_bad_mask |= current_mask

    # 4. Get event IDs and filter the dataframe
    bad_events = event_ids[combined_bad_mask]
    df_filter = df[~df['eventId'].isin(bad_events)]

    times_array = df_filter['time'].to_numpy() / 3
    times_array = times_array[times_array > -500]

    if plot:
        fig, ax = plt.subplots()
        ax.hist(times_array, bins=100, color='orange', alpha=0.7)
        ax.set_title('Time of Arrival Histogram')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Counts')
        plt.tight_layout()

        plot_2ds(df_filter, pitch, run_time)

        print(df['eventId'].unique())
        plt.show()

    rate = df_filter['eventId'].nunique() / run_time
    print(f'Rate: {rate:.2f} events/s')
    return rate, all_rate


if __name__ == '__main__':
    main()
