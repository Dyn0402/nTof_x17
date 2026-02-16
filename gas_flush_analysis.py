#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 12 6:23 PM 2026
Created in PyCharm
Created as nTof_x17/gas_flush_analysis.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    runs = ['run_48', 'run_49', 'run_50', 'run_51', 'run_52', 'run_53', 'run_54', 'run_55']
    hv_csv_file_name = 'hv_monitor.csv'
    time_col = 'timestamp'
    voltage_col = '2:0 vmon'
    current_col = '2:0 imon'

    data = {}
    for run in runs:
        run_dir = os.path.join(base_path, run)
        if not os.path.isdir(run_dir):
            print(f'{run_dir} is not a directory')
            continue
        sub_runs = os.listdir(run_dir)
        data.update({run: {'hvs': [], 'currents': [], 'times': []}})
        for sub_run in sub_runs:
            sub_run_dir = os.path.join(run_dir, sub_run)
            if not os.path.isdir(sub_run_dir):
                continue
            hv_csv_path = os.path.join(sub_run_dir, hv_csv_file_name)
            hv_df = pd.read_csv(hv_csv_path)
            subrun_hv = hv_df[voltage_col].values
            subrun_current = hv_df[current_col].values
            subrun_timestamp = hv_df[time_col].values

            # Take average values for last 90% of data
            avg_start_index = int(0.1 * len(subrun_timestamp))

            dt_series = pd.to_datetime(subrun_timestamp[avg_start_index:])
            subrun_time = dt_series.mean()
            subrun_hv = np.mean(subrun_hv[avg_start_index:])
            subrun_current = np.mean(subrun_current[avg_start_index:])
            # subrun_time = np.mean(subrun_timestamp[avg_start_index:])
            data[run]['hvs'].append(subrun_hv)
            data[run]['currents'].append(subrun_current)
            data[run]['times'].append(subrun_time)

            # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
            # axs[0].plot(subrun_timestamp, subrun_hv)
            # axs[1].plot(subrun_timestamp, subrun_current)
            # # axs[1].set_xlabel('Time')
            # axs[0].set_ylabel('HV (V)')
            # axs[1].set_ylabel('Current (A)')
            # fig.suptitle(f'{run} - {sub_run}')
            # plt.show()

    fig, ax = plt.subplots()
    for run, run_data in data.items():
        ax.plot(run_data['hvs'], run_data['currents'], marker='o', label=run)
    ax.legend()
    ax.set_xlabel('HV (V)')
    ax.set_ylabel('Current (A)')
    fig.tight_layout()

    hv = 610
    currents, times = [], []
    for run, run_data in data.items():
        for hv_val, current_val, time_val in zip(run_data['hvs'], run_data['currents'], run_data['times']):
            if hv - 2 <= hv_val <= hv + 2:
                currents.append(current_val)
                times.append(time_val)

    fig, ax = plt.subplots()
    ax.plot(times, currents, marker='o')
    ax.set_ylabel('Current (A)')
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle(f'Current Drift for HV = {hv} V')
    fig.tight_layout()

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
