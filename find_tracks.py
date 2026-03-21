#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 22 8:44 PM 2026
Created in PyCharm
Created as nTof_x17/find_tracks.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_beam_hits import load_subrun, get_run_time
from cosmic_bench_analysis.cosmic_micro_tpc_tests import line_fit_test


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    run = 'run_34'
    sub_run = 'final_resist_440V_drift_600V'
    # run = 'run_60'
    # sub_run = 'resist_405V_drift_600V'
    feus_map = {4: 'y', 5: 'x'}  # Which positions they give
    feus = list(feus_map.keys())
    find_tracks(base_path, run, sub_run, feus, feus_map)

    print('donzo')



def find_tracks(base_path, run, sub_run, feus, feus_map):
    """
    Plot general metrics for full subrun.
    """
    run_time = get_run_time(base_path, run, sub_run)
    print(f'Run time: {run_time / 60:.2f} minutes')
    df, det = load_subrun(base_path, run, sub_run, feus)
    pitch = 0.78
    df['axis'] = df['feu'].map(feus_map)
    df['channel_flipped'] = (df['channel'] // 64) * 64 + (63 - (df['channel'] % 64))
    df['position'] = df['channel_flipped'] * pitch
    df['time'] = df['time'] / 3

    # Create x_pos: if axis is 'x', use position, else None
    df['x_position_mm'] = np.where(df['axis'] == 'x', df['position'], float('nan'))

    # Create y_pos: if axis is 'y', use position, else None
    df['y_position_mm'] = np.where(df['axis'] == 'y', df['position'], float('nan'))

    min_amp = 400
    df = df[df['amplitude'] >= min_amp]

    event_ids = df['eventId'].to_numpy()
    event_ids, counts = np.unique(event_ids, return_counts=True)
    fig, ax = plt.subplots()
    ax.hist(counts, bins=np.arange(-0.5, np.max(counts) + 0.5, 1), color='purple', alpha=0.7)
    ax.set_title('Hits per Event Histogram')
    ax.set_xlabel('Number of Hits in Event')
    ax.set_ylabel('Counts')
    plt.tight_layout()

    max_hits = 800
    # 1. Calculate hits per event and map it back to every row in the original df
    df['event_hit_count'] = df.groupby('eventId')['eventId'].transform('count')

    # 2. Filter the original dataframe
    df = df[df['event_hit_count'] <= max_hits].copy()

    # Optional: Clean up by dropping the helper column
    df.drop(columns=['event_hit_count'], inplace=True)

    grouped = df.groupby('eventId')
    avg_x_positions = grouped['x_position_mm'].mean().to_numpy()
    avg_y_positions = grouped['y_position_mm'].mean().to_numpy()
    event_ids = grouped['eventId'].first().to_numpy()

    bad_xs = (230, 270)
    # bad_ys = (370, 390)
    bad_ys = (230, 390)

    # 1. Identify events that ARE valid AND fall in the 'bad' spatial box
    # Remove the ~ (not) from valid_indices
    bad_mask = (
            (avg_x_positions >= bad_xs[0]) & (avg_x_positions <= bad_xs[1]) &
            (avg_y_positions >= bad_ys[0]) & (avg_y_positions <= bad_ys[1])
    )

    # 2. Get the event IDs that fell into that bad mask
    bad_events = event_ids[bad_mask]

    # 3. Filter the original dataframe
    df_filter = df[~df['eventId'].isin(bad_events)]
    print(df_filter[['eventId', 'time', 'x_position_mm', 'y_position_mm']])

    max_chi2 = 1e8
    mx, my, x0, y0, times, chi2_xs, chi2_ys = [], [], [], [], [], [], []
    for event_id in df_filter['eventId'].unique():
        print(event_id)
        x_slope, y_slope, det_x, det_y, time, chi2_x, chi2_y = line_fit_test(df, event_number=event_id, plot=False,
                                                                             return_extra=True)
        if chi2_x < max_chi2 and chi2_y < max_chi2:
            mx.append(x_slope)
            my.append(y_slope)
            x0.append(det_x)
            y0.append(det_y)
            times.append(time)
            chi2_xs.append(chi2_x)
            chi2_ys.append(chi2_y)

        # if 1e7 < chi2_x < 1e9:
        #     line_fit_test(df, event_number=event_id, plot=True, return_extra=True)

    vd = 15 / 1000  # mm/ns

    mx = np.array(mx)
    my = np.array(my)
    mx = (1 / mx / vd)
    my = (1 / my / vd)

    # Histogram chi2s
    fig_chi2, ax_chi2 = plt.subplots()
    ax_chi2.hist(chi2_xs, bins=100, color='blue', alpha=0.7, label='Chi2 x')
    ax_chi2.hist(chi2_ys, bins=100, color='green', alpha=0.7, label='Chi2 y')
    ax_chi2.set_title('Chi2 Histogram')
    ax_chi2.set_xlabel('Chi2')
    ax_chi2.legend()
    plt.tight_layout()

    # Histogram x and y slope
    fig_mx, ax_mx = plt.subplots()
    ax_mx.hist(mx, bins=100, color='purple', alpha=0.7)
    ax_mx.set_title('x Slope Histogram')
    ax_mx.set_xlabel('x Slope')

    fig_my, ax_my = plt.subplots()
    ax_my.hist(my, bins=100, color='purple', alpha=0.7)
    ax_my.set_title('y Slope Histogram')
    ax_my.set_xlabel('y Slope')

    fig_mx_my, ax_mx_my = plt.subplots()
    ax_mx_my.scatter(mx, my, alpha=0.5)
    ax_mx_my.set_xlabel('x Slope')
    ax_mx_my.set_ylabel('y Slope')

    fig_times, ax_times = plt.subplots()
    ax_times.hist(times, bins=100, color='purple', alpha=0.7)
    ax_times.set_title('Cluster Time Histogram')
    ax_times.set_xlabel('Cluster Time (ns)')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the Z range
    z_start = -30
    z_end = 0
    z_points = np.array([z_start, z_end])

    for i in range(len(x0)):
        # Calculate x and y at the two ends of the Z range
        x_points = x0[i] + mx[i] * z_points
        y_points = y0[i] + my[i] * z_points

        # Plot the line
        ax.plot(x_points, y_points, z_points, label=f'Track {i}')

    # Formatting
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Detector Tracks')
    ax.invert_zaxis()  # Optional: if you want -30 at the 'top' visually

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the Z range
    z_start = -30 - 200
    z_end = 0
    z_points = np.array([z_start, z_end])

    for i in range(len(x0)):
        # Calculate x and y at the two ends of the Z range
        x_points = x0[i] + mx[i] * z_points
        y_points = y0[i] + my[i] * z_points

        # Plot the line
        ax.plot(x_points, y_points, z_points, label=f'Track {i}')

    # Formatting
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Detector Tracks')
    ax.invert_zaxis()  # Optional: if you want -30 at the 'top' visually


    plt.show()


if __name__ == '__main__':
    main()
