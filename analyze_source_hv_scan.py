#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 17 11:21 AM 2026
Created in PyCharm
Created as nTof_x17/analyze_source_hv_scan.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot_beam_hits import load_subrun, get_run_time, get_run_json


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    # run = 'run_38'
    run = 'run_39'
    # run = 'run_72'
    # run = 'run_63'
    # run = 'run_98'
    out_csv_path = f'/media/dylan/data/x17/feb_beam/Analysis/Plot_Data/cs137_calibration_{run}.csv'
    # out_csv_path = None

    max_hits_dict = {
        'run_38': 400,
        'run_39': 400,
        'run_63': 400,
        'run_72': 400,
        'run_98': 400,
    }
    # feus = {1: 'm3', 4: 'MX17 X Strips', 6: 'MX17 Y Strips'}
    feus = [4, 5]
    # feus = [4, 6]
    run_dir = os.path.join(base_path, run)
    run_cfg = get_run_json(base_path, run)
    print(run_cfg)
    print(run_cfg['sub_runs'])
    sub_run_hvs = {x['sub_run_name']: x['hvs']['2']['0'] for x in run_cfg['sub_runs']}
    print(sub_run_hvs)
    hvs, rates, filtered_rates = [], [], []
    for subrun in os.listdir(run_dir):
        subrun_dir = os.path.join(run_dir, subrun)
        if not os.path.isdir(subrun_dir):
            continue
        print(subrun)
        df, det = load_subrun(base_path, run, subrun, feus)
        if len(df) == 0:
            continue
        run_time = get_run_time(base_path, run, subrun)
        # Get number of unique eventId
        n_events = df['eventId'].nunique()
        hvs.append(sub_run_hvs[subrun])
        rates.append(n_events / run_time)

        max_hits = max_hits_dict[run]
        # 1. Calculate hits per event and map it back to every row in the original df
        df['event_hit_count'] = df.groupby('eventId')['eventId'].transform('count')

        # 2. Filter the original dataframe
        df = df[df['event_hit_count'] <= max_hits].copy()

        # Optional: Clean up by dropping the helper column
        df.drop(columns=['event_hit_count'], inplace=True)

        n_events_filter = df['eventId'].nunique()
        filtered_rates.append(n_events_filter / run_time)

    # Sort everything on hvs
    sorted_idx = np.argsort(hvs)
    hvs = np.array(hvs)[sorted_idx]
    rates = np.array(rates)[sorted_idx]
    filtered_rates = np.array(filtered_rates)[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hvs, rates, marker='o', label='All Events')
    ax.plot(hvs, filtered_rates, marker='o', label='Max Hits = 400')
    ax.legend()
    ax.axhline(0, color='gray', zorder=0)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.grid(axis='x', which='minor', linestyle='-', linewidth=0.4, alpha=0.7, zorder=0)
    ax.grid(axis='x', which='major', linestyle='-', linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_xlabel('HV (V)')
    ax.set_ylabel('Event Rate (Hz)')
    fig.suptitle(f'Event Rate for Cs-137 HV Scan -- {run}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, left=0.055, right=0.995, bottom=0.1)

    if out_csv_path is not None:
        df = pd.DataFrame({'HV': hvs, 'Rate': rates, 'Filtered Rate': filtered_rates})
        df.to_csv(out_csv_path, index=False)
        print(f'Saved data to {out_csv_path}')

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
