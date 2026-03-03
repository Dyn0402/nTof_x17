#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 17 1:01 PM 2026
Created in PyCharm
Created as nTof_x17/plot_hits_vs_time_hv_scan.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_beam_hits import load_subrun, get_run_time, get_run_json


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    run = 'run_64'
    # feus = {1: 'm3', 4: 'MX17 X Strips', 6: 'MX17 Y Strips'}
    # feus = [4, 5]
    feus = [4, 6]
    min_amp = 400

    run_dir = os.path.join(base_path, run)
    run_cfg = get_run_json(base_path, run)
    print(run_cfg)
    print(run_cfg['sub_runs'])
    sub_run_hvs = {x['sub_run_name']: x['hvs']['2']['0'] for x in run_cfg['sub_runs']}
    print(sub_run_hvs)
    time_bins = np.arange(0, 10000, 50)
    coarse_time_bin_edges = [0, 800, 1150, 3500, 7000, 10000]
    coarse_time_bin_hits = {
        i: {'edges': (coarse_time_bin_edges[i], coarse_time_bin_edges[i + 1]), 'hits_per_event': []}
        for i in range(len(coarse_time_bin_edges) - 1)
    }
    hvs, rates, hits_per_event = [], [], []
    for subrun in os.listdir(run_dir):
        subrun_dir = os.path.join(run_dir, subrun)
        if not os.path.isdir(subrun_dir):
            continue
        print(subrun)
        df, det = load_subrun(base_path, run, subrun, feus)
        run_time = get_run_time(base_path, run, subrun)
        # Get number of unique eventId
        n_events = df['eventId'].nunique()
        hvs.append(sub_run_hvs[subrun])
        rates.append(n_events / run_time)

        df = df[df['amplitude'] >= min_amp]

        # Make a histogram of time
        df_time = df['time'] / 3
        hist, _ = np.histogram(df_time, bins=time_bins)
        hits_per_event.append(hist / n_events)

        for i in coarse_time_bin_hits:
            lower_edge, upper_edge = coarse_time_bin_hits[i]['edges']
            df_time_bin = df_time[(df_time >= lower_edge) & (df_time < upper_edge)]
            coarse_time_bin_hits[i]['hits_per_event'].append(df_time_bin.size / n_events)

    # Convert to numpy arrays
    hvs, rates, hits_per_event = np.array(hvs), np.array(rates), np.array(hits_per_event)
    # Sort hvs, rates, hits_per_event by hv
    sort_idx = np.argsort(hvs)
    hvs, rates, hits_per_event = hvs[sort_idx], rates[sort_idx], hits_per_event[sort_idx]

    fig, ax = plt.subplots()
    for hv, rate, hist in zip(hvs, rates, hits_per_event):
        # ax.plot(time_bins[:-1], rate, label=f'{hv} V')
        if hv == 0:
            lw = 3
            zo = 100
        else:
            lw = 1
            zo = 5
        ax.plot(time_bins[:-1], hist, label=f'{hv} V', lw=lw, zorder=zo)
    ax.legend()
    ax.axhline(0, color='gray', zorder=0)
    ax.set_xlabel('HV (V)')
    ax.set_ylabel('Event Rate (Hz)')
    fig.suptitle(f'Event Rate for Cs-137 HV Scan -- {run}')
    fig.tight_layout()

    fig_all, ax_all = plt.subplots(figsize=(10, 5))
    for i in coarse_time_bin_hits:
        fig, ax = plt.subplots()
        bin_low, bin_high = coarse_time_bin_hits[i]['edges']
        ax.plot(hvs, coarse_time_bin_hits[i]['hits_per_event'], label=f'{bin_low / 1000}-{bin_high / 1000} μs')
        ax.legend()
        ax.set_xlabel('HV (V)')
        ax.set_ylabel('Hits per Event')
        fig.suptitle(f'Hits per Event for Cs-137 HV Scan -- {run}')
        fig.tight_layout()
        ax_all.plot(hvs, coarse_time_bin_hits[i]['hits_per_event'], label=f'{bin_low / 1000}-{bin_high / 1000} μs')
    ax_all.legend()
    ax_all.set_xlabel('HV (V)')
    ax_all.set_ylabel('Hits per Event')
    fig_all.suptitle(f'Hits per Event for Cs-137 HV Scan -- {run}')
    fig_all.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
