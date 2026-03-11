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
from matplotlib.colors import LogNorm
import uproot
import pandas as pd

from Mx17StripMap import Mx17StripMap, RunConfig


def main():
    # base_path = '/media/dylan/data/x17/cosmic_bench/det_1/'
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    # run = 'run_63'
    # sub_run = 'resist_440V_drift_600V'
    # run = 'run_34'
    # sub_run = 'resist_425V_drift_600V'
    # run = 'run_60'
    # sub_run = 'resist_580V_drift_600V'
    # run = 'run_114'
    # sub_run = 'resist_665V_drift_1000V'
    # base_path = '/mnt/data/x17/cosmic_bench/det_1/'
    # run = 'mx17_det1_overnight_run_1-27-26'
    # run = 'mx17_det1_daytime_run_1-28-26'
    # sub_run = 'overnight_run'
    # feus = {1: 'm3', 4: 'MX17 X Strips', 6: 'MX17 Y Strips'}
    feus_map = {4: 'y', 5: 'x'}  # Which positions they give
    # feus_map = {4: 'y', 6: 'x'}  # Which positions they give
    feus = list(feus_map.keys())

    # hvs = [400, 425, 450, 475]
    # hvs = [300, 400, 430, 450, 480]
    hvs = list(np.arange(360, 820, 5))
    # runs = ['run_26', 'run_27', 'run_30', 'run_31']
    # runs = {
    #     'run_18': 'Ar/CF4/Iso 88/10/2 - Timepix - No Shielding',
    #     'run_35': 'Ar/CF4/Iso 88/10/2 - Carbon - Shielding'
    # }

    # runs = {
    #         'run_26': 'He - Empty - No Shielding',
    #         'run_27': 'He - Lead - No Shielding',
    #         'run_30': 'He - B4C - No Shielding',
    #         'run_32': 'He - Empty - Shielding',
    #         'run_33': 'He - Carbon - Shielding'
    # }
    #
    # runs = {
    #     'run_57': 'Ar/CF4 90/10 - Beam Cap - Shielding',
    #     'run_59': 'Ar/CF4 90/10 - B4C - Shielding'
    # }
    #
    # runs = {
    #     'run_43': 'Ar/CF4/CO2 45/40/15 - Carbon - Shielding',
    #     # 'run_47': 'Ar/CF4/CO2 45/40/15 - Carbon - Shielding SELF',
    # }

    # title = 'Ar/CF4/Iso 88/10/2 with 30 mm Drift Gap'
    # runs = {
    #     'run_18': 'run_18 - Timepix - Feb 3',
    #     'run_35': 'run_35 - Carbon - Feb 7 (Still flushing Helium)',
    # }

    # title = 'Helium/Ethane 96.5/3.5 with 30 mm Drift Gap'
    # runs = {
    #     'run_26': 'run_26 - Empty - No Shielding',
    #     'run_27': 'run_27 - Lead - No Shielding',
    #     # 'run_29': 'run_29 - B4C - No Shielding (Quick)',
    #     'run_30': 'run_30 - B4C - No Shielding',
    #     'run_32': 'run_32 - Empty - Shielding',
    #     'run_33': 'run_33 - Carbon - Shielding',
    # }

    # title = 'Argon/CF4/CO2 45/40/15 with 30 mm Drift Gap'
    # runs = {
    #     'run_43': 'run_43 - Carbon - No Shielding',
    # }

    # title = 'Argon/CF4 90/10 with 30 mm Drift Gap'
    # runs = {
    #     'run_50': 'run_50 - Carbon',
    #     'run_57': 'run_57 - Beam Cap',
    #     'run_59': 'run_59 - B4C',
    # }

    title = 'Argon/CO2 70/30 with 30 mm Drift Gap'
    runs = {
        'run_64': 'run_64 - B4C',
        'run_67': 'run_67 - None',
    }

    # runs = {
    #     'run_18': 'Ar/CF4/Iso 88/10/2 - Timepix - No Shielding',
    #     'run_35': 'Ar/CF4/Iso 88/10/2 - Carbon - Shielding',
    #     'run_26': 'He - Empty - No Shielding',
    #     'run_27': 'He - Lead - No Shielding',
    #     'run_30': 'He - B4C - No Shielding',
    #     'run_32': 'He - Empty - Shielding',
    #     'run_33': 'He - Carbon - Shielding',
    #     # 'run_57': 'Ar/CF4 90/10 - Beam Cap - Shielding',
    #     # 'run_59': 'Ar/CF4 90/10 - B4C - Shielding',
    #     # 'run_43': 'Ar/CF4/CO2 45/40/15 - Carbon - Shielding',
    #     # 'run_47': 'Ar/CF4/CO2 45/40/15 - Carbon - Shielding SELF',
    # }

    # runs = {
    #     'run_88_D500V': 'He/Ethane 96.5/3.5 - B4C - Drift 500V',
    #     'run_88_D1000V': 'He/Ethane 96.5/3.5 - B4C - Drift 1000V',
    #     'run_88_D1500V': 'He/Ethane 96.5/3.5 - B4C - Drift 1500V',
    # }
    # sub_run = 'resist_540V_drift_1000V'

    # runs = {
    #     'run_114_D500V': 'Ar/CO2 70/30 - B4C - Carbon Frame - Drift 500V',
    #     'run_114_D1000V': 'Ar/CO2 70/30 - B4C - Carbon Frame - Drift 1000V',
    #     'run_114_D1500V': 'Ar/CO2 70/30 - B4C - Carbon Frame - Drift 1500V',
    #     'run_71_D500V': 'Ar/CO2 70/30 - B4C - Aluminum Frame - Drift 500V',
    #     'run_71_D1000V': 'Ar/CO2 70/30 - B4C - Aluminum Frame - Drift 1000V',
    #     'run_74_D1000V': 'Ar/CO2 70/30 - Empty - Aluminum Frame - Drift 1000V',
    # }
    # sub_run = 'resist_540V_drift_1000V'
    #
    # runs = {
    #     'run_30_D600V': 'He - B4C - Al Frame - 30mm Drift at -600V',
    #     # 'run_88_D500V': 'He/Ethane 96.5/3.5 - B4C - Aluminum Frame - Drift 500V',
    #
    #     'run_88_D1000V': 'He - B4C - Al Frame - 6mm Drift at -1000V',
    #     # 'run_88_D1500V': 'He/Ethane 96.5/3.5 - B4C - Aluminum Frame - Drift 1500V',
    #     # 'run_126_D500V': 'He/Ethane 96.5/3.5 - Lead - Carbon Frame - 55 cm - Drift 500V',
    #     # 'run_126_D1000V': 'He/Ethane 96.5/3.5 - Lead - Carbon Frame - 55cm  - Drift 1000V',
    #     # 'run_126_D1500V': 'He/Ethane 96.5/3.5 - Lead - Carbon Frame - 55cm  - Drift 1500V',
    #     # 'run_131_D500V': 'He/Ethane 96.5/3.5 - B4C - Carbon Frame - Drift 500V',
    #
    #     'run_131_D1000V': 'He - B4C - Carbon Frame - 6mm Drift at -1000V',
    #     # 'run_131_D1500V': 'He/Ethane 96.5/3.5 - B4C - Carbon Frame - Drift 1500V',
    # }
    # sub_run = 'resist_510V_drift_1000V'

    # csv_out_dir = f'/media/dylan/data/x17/feb_beam/Analysis/hits_vs_time_csvs'
    # export_hits_vs_time_csvs_wrapper(base_path, hvs, csv_out_dir, feus)
    # input('Enter to continue...')

    # plot_amps_with_hv(base_path, run, sub_run, feus, hvs)
    # plot_hits_vs_hv(base_path, run, sub_run, feus, hvs)
    # plot_hits_vs_hv_runs(base_path, list(runs.keys()), feus, hvs, runs)
    plot_hits_vs_hv_runs_vertical(base_path, list(runs.keys()), feus, hvs, runs, title)
    plt.show()

    # runs = {
    #     'run_88': 'He/Ethane 96.5/3.5 - B4C - Resist 540V - Drift 1000V',
    # }
    # runs = {
    #     'run_114': 'Ar/CO2 70/30 - B4C - Carbon Frame - Resist 690V Drift 1000V',
    #     'run_71': 'Ar/CO2 70/30 - B4C - Aluminum Frame - Resist 690V Drift 1000V',
    #     'run_74': 'Ar/CO2 70/30 - Empty - Aluminum Frame - Resist 690V Drift 1000V',
    # }
    # sub_run = 'resist_690V_drift_1000V'

    # runs = {
    #     'run_30_D600V': 'He - B4C - Aluminum Frame - 30mm Drift - Drift 600V',
    #     'run_88_D1000V': 'He/Ethane 96.5/3.5 - B4C - Aluminum Frame - Drift 1000V',
    #     'run_131_D1000V': 'He/Ethane 96.5/3.5 - B4C - Carbon Frame - Drift 1000V',
    # }
    runs = {
        'run_30_D600V': 'He - B4C - Aluminum Frame - 30mm Drift - Drift 600V',
    }
    sub_run = 'resist_510V_drift_1000V'

    plot_hits_vs_time(base_path, list(runs.keys()), sub_run, feus, runs, save_csv_path='/home/dylan/Downloads/hists.csv')
    #
    # run_hits = 'run_88'
    # hvs = [540, 530, 520, 510, 500, 490]
    # plot_hits_vs_time_hvs(base_path, run_hits, sub_run, feus, hvs)
    #
    plt.show()

    # event = 6
    # run = 'run_33'
    # sub_run = 'initial_resist_440V_drift_600V'
    # run = 'run_19'
    # sub_run = 'resist_440V_drift_600V'
    # event = 16
    # run = 'run_34'
    # # sub_run = 'final_resist_440V_drift_600V'
    # sub_run = 'resist_475V_drift_600V'
    event = 16
    # run = 'run_52'
    # sub_run = 'final_resist_440V_drift_600V'
    # sub_run = 'resist_485V_drift_600V'
    # sub_run = 'initial_resist_610V_drift_600V'
    # sub_run = 'drift_600V_6'

    plot_hits_per_event(base_path, run, sub_run, feus)
    # plot_hits_xy_vs_time(base_path, run, sub_run, feus, event)

    # plot_general_metrics(base_path, run, sub_run, feus)
    # plot_general_metrics_fast(base_path, run, sub_run, feus, feus_map)

    plt.show()


def plot_times_with_hv(base_path, run, sub_run, feus, hvs):
    """
    Plot hit time distribution for several subruns
    """
    min_amp = 300
    fig, ax = plt.subplots()
    bins = np.arange(0, 10000, 100)
    for hv in hvs:
        sub_run_parts = sub_run.split('_')
        sub_run_parts[1] = f'{hv}V'
        if run == 'run_17':
            hv_sub_run = f'resist_hv_{hv}V'
        else:
            hv_sub_run = '_'.join(sub_run_parts)
        df, det = load_subrun(base_path, run, hv_sub_run, feus)
        df = df[df['amplitude'] >= min_amp]
        times_array = df['time'].to_numpy() / 3
        ax.hist(times_array, bins=bins, alpha=0.7, label=f'{hv}V', log=False)
    ax.set_title('Time of Arrival Histogram')
    ax.set_xlabel('Time of Arrival (ns)')
    ax.set_ylabel('Hits')
    ax.legend()
    fig.tight_layout()


def plot_hits_vs_hv(base_path, run, sub_run, feus, hvs):
    """
    Plot number of hits per event vs HV in several time of arrival bins.
    """
    min_amp = 300
    time_bins = [[0, 1000], [1000, 3000], [3000, 10000]]

    hits_per_event = [[] for _ in time_bins]
    hvs_plot = []
    for hv in hvs:
        print(f'Processing HV {hv}V')
        sub_run_parts = sub_run.split('_')
        sub_run_parts[1] = f'{hv}V'
        hv_sub_run = '_'.join(sub_run_parts)
        df, det = load_subrun(base_path, run, hv_sub_run, feus)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f'No data found for HV {hv}V!\n')
            continue
        df = df[df['amplitude'] >= min_amp]
        n_events = df['eventId'].nunique()
        for i in enumerate(time_bins):
            df_time_bin = df[(df['time'] / 3 >= i[1][0]) & (df['time'] / 3 < i[1][1])]
            n_hits = df_time_bin.shape[0]
            hits_per_event[i[0]].append(n_hits / n_events)
        hvs_plot.append(hv)

    fig, ax = plt.subplots()
    for i in range(len(time_bins)):
        ax.plot(hvs_plot, hits_per_event[i], label=f'{time_bins[i][0]/1000} μs - {time_bins[i][1]/1000} μs')
    ax.set_yscale('log')
    ax.axhline(0, color='gray', zorder=0)
    ax.annotate(f'Minimum Amplitude: {min_amp} ADC', xy=(0.05, 0.6), xycoords='axes fraction')
    ax.set_title(f'Hits per Event vs HV for {run} - {sub_run}')
    ax.set_xlabel('HV (V)')
    ax.set_ylabel('Hits per Event')
    ax.legend()
    fig.tight_layout()


def plot_hits_vs_hv_runs(base_path, runs, feus, hvs, run_name_map=None):
    """
    Plot number of hits per event vs HV in several time of arrival bins.
    """
    min_amp = 800
    # time_bins = [[0, 1000], [1000, 3000], [3000, 10000]]
    time_bins = [[0, 75 * 20], [75 * 20, 270 * 20], [270 * 20, 10000]]

    hits_per_event = [{run: [] for run in runs} for _ in time_bins]
    hits_per_event_std = [{run: [] for run in runs} for _ in time_bins]
    hvs_plot = [{run: [] for run in runs} for _ in time_bins]
    for hv in hvs:
        print(f'Processing HV {hv}V')
        for run in runs:
            if len(run.split('_')) == 3:  # Drift at end
                drift = run.split('_')[-1].strip('D')
                run_dir_name = '_'.join(run.split('_')[:-1])
            else:
                drift = None
                run_dir_name = run
            run_dir = os.path.join(base_path, run_dir_name)
            hv_sub_run = None
            for sub_run_dir in os.listdir(run_dir):
                if not os.path.isdir(os.path.join(run_dir, sub_run_dir)):
                    continue
                if f'resist_{hv}V_' in sub_run_dir:
                    if drift is not None and f'_drift_{drift}' not in sub_run_dir:
                        continue
                    hv_sub_run = sub_run_dir
                    break
            # sub_run_parts = sub_run.split('_')
            # sub_run_parts[1] = f'{hv}V'
            # if run == 'run_17':
            #     hv_sub_run = f'resist_hv_{hv}V'
            # else:
            #     hv_sub_run = '_'.join(sub_run_parts)
            if hv_sub_run is None:
                print(f'No data found for HV {hv}V in {run_dir_name}!\n')
                continue
            try:
                df, det = load_subrun(base_path, run_dir_name, hv_sub_run, feus)
            except FileNotFoundError:
                print(f'No data found for HV {hv}V in {run_dir_name}, {hv_sub_run}!\n')
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f'No data found for HV {hv}V!\n')
                continue
            df = df[df['amplitude'] >= min_amp]
            n_events = df['eventId'].nunique()
            if n_events == 0:
                continue

            for i in enumerate(time_bins):
                df_time_bin = df[(df['time'] / 3 >= i[1][0]) & (df['time'] / 3 < i[1][1])]
                n_hits = df_time_bin.shape[0]
                hits_per_event[i[0]][run].append(n_hits / n_events)
                hvs_plot[i[0]][run].append(hv)

            # With SD
            # all_event_ids = df["eventId"].unique()
            # for i, (t0, t1) in enumerate(time_bins):
            #     df_time_bin = df[(df["time"] / 3 >= t0) & (df["time"] / 3 < t1)]
            #
            #     # hits per event in this time bin
            #     hits_per_event_counts = (
            #         df_time_bin.groupby("eventId").size().reindex(all_event_ids, fill_value=0)
            #     )
            #
            #     mean_hits = hits_per_event_counts.mean()
            #     std_hits = hits_per_event_counts.std(ddof=1)  # sample std; use ddof=0 for population
            #
            #     hits_per_event[i][run].append(mean_hits)
            #     hits_per_event_std[i][run].append(std_hits)
            #     hvs_plot[i][run].append(hv)

    fig, axs = plt.subplots(ncols=len(time_bins), figsize=(19.5, 5), sharey=False)

    for i in range(len(time_bins)):
        for run in runs:
            label = run_name_map[run] if run_name_map is not None else run
            axs[i].plot(hvs_plot[i][run], hits_per_event[i][run], marker='o', label=label)
            # ax.errorbar(hvs_plot[i][run], hits_per_event[i][run], hits_per_event_std[i][run], marker='o', label=label)
        # ax.set_yscale('log')
        axs[i].axhline(0, color='gray', zorder=0)
        axs[i].set_title(f'Hits per Event vs HV for {time_bins[i][0]/1000} μs - {time_bins[i][1]/1000} μs')
        axs[i].set_xlabel('HV (V)')
    axs[0].annotate(f'Minimum Amplitude: {min_amp} ADC', xy=(0.05, 0.6), xycoords='axes fraction')
    axs[0].set_ylabel('Hits per Event')
    axs[0].legend(loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08)


def plot_hits_vs_hv_runs_vertical(base_path, runs, feus, hvs, run_name_map=None, title=None):
    """
    Plot number of hits per event vs HV in several time of arrival bins.
    Panels are stacked vertically with a shared x-axis.
    """
    min_amp = 800
    time_bins = [[0, 75 * 20], [75 * 20, 270 * 20], [270 * 20, 10000]]

    hits_per_event = [{run: [] for run in runs} for _ in time_bins]
    hvs_plot = [{run: [] for run in runs} for _ in time_bins]

    for hv in hvs:
        print(f'Processing HV {hv}V')
        for run in runs:
            if len(run.split('_')) == 3:  # Drift at end
                drift = run.split('_')[-1].strip('D')
                run_dir_name = '_'.join(run.split('_')[:-1])
            else:
                drift = None
                run_dir_name = run

            run_dir = os.path.join(base_path, run_dir_name)
            hv_sub_run = None
            for sub_run_dir in os.listdir(run_dir):
                if run == 'run_32' and hv > 470:
                    continue
                if not os.path.isdir(os.path.join(run_dir, sub_run_dir)):
                    continue
                if f'resist_{hv}V_' in sub_run_dir:
                    if drift is not None and f'_drift_{drift}' not in sub_run_dir:
                        continue
                    hv_sub_run = sub_run_dir
                    break

            if hv_sub_run is None:
                print(f'No data found for HV {hv}V in {run_dir_name}!\n')
                continue
            try:
                df, det = load_subrun(base_path, run_dir_name, hv_sub_run, feus)
            except FileNotFoundError:
                print(f'No data found for HV {hv}V in {run_dir_name}, {hv_sub_run}!\n')
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f'No data found for HV {hv}V!\n')
                continue

            df = df[df['amplitude'] >= min_amp]
            n_events = df['eventId'].nunique()
            if n_events == 0:
                continue

            for i, (t0, t1) in enumerate(time_bins):
                df_time_bin = df[(df['time'] / 3 >= t0) & (df['time'] / 3 < t1)]
                hits_per_event[i][run].append(df_time_bin.shape[0] / n_events)
                hvs_plot[i][run].append(hv)

    n_bins = len(time_bins)
    fig, axs = plt.subplots(nrows=n_bins, ncols=1, figsize=(12, 2 * n_bins), sharex=True)

    for i, ax in enumerate(axs):
        t0_us = time_bins[i][0] / 1000
        t1_us = time_bins[i][1] / 1000
        time_window_label = f'{t0_us:.2f}–{t1_us:.2f} μs'

        for run in runs:
            label = run_name_map[run] if run_name_map is not None else run
            ax.plot(hvs_plot[i][run], hits_per_event[i][run], marker='o', label=label)

        ax.axhline(0, color='gray', linewidth=0.8, zorder=0)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(10))  # minor ticks every 5 V
        ax.grid(axis='x', which='minor', linestyle='-', linewidth=0.4, alpha=0.7, zorder=0)
        ax.grid(axis='x', which='major', linestyle='-', linewidth=0.8, alpha=0.8, zorder=0)
        ax.set_ylabel('Hits per Event')
        ax.text(0.02, 0.3, time_window_label, transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=1.0, edgecolor='black'))

        if i == 2:
            ax.annotate(f'Min. Amplitude: {min_amp} ADC', xy=(0.05, 0.8),
                        xycoords='axes fraction', ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1.0, edgecolor='none'))
        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))

    axs[-1].set_xlabel('HV (V)')
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.075, left=0.06, right=0.995, hspace=0)


def plot_hits_vs_time(base_path, runs, sub_run, feus, run_name_map=None, save_csv_path=None):
    """
    Plots hits vs time and optionally saves the binned data to a CSV.
    """
    min_amp = 1000
    time_bins = np.arange(0, 10000, 100)
    time_bin_mids = (time_bins[:-1] + time_bins[1:]) / 2

    # Dictionary to store data for CSV export
    csv_data = {"time_bin_mid_us": time_bin_mids / 1000}

    fig, ax = plt.subplots(figsize=(9, 4))

    for run in runs:
        if len(run.split('_')) == 3:  # Drift at end
            drift = run.split('_')[-1].strip('D')
            run_dir_name = '_'.join(run.split('_')[:-1])
        else:
            drift = None
            run_dir_name = run
        sub_run_parts = sub_run.split('_')
        sub_run_parts[-1] = f'{drift}' if drift is not None else sub_run_parts[-1]
        sub_run_i = '_'.join(sub_run_parts)
        df, det = load_subrun(base_path, run_dir_name, sub_run_i, feus)
        df = df[df['amplitude'] >= min_amp]
        n_events = df['eventId'].nunique()

        # Binning logic
        bins = pd.cut(df["time"] / 3, bins=time_bins)
        hits_per_event_series = (
                df["time"]
                .groupby(bins, observed=False)
                .count()
                .reindex(bins.cat.categories, fill_value=0)
                / n_events
        )

        label = run_name_map[run] if run_name_map is not None else run
        ax.plot(time_bin_mids / 1000, hits_per_event_series.values, label=label)

        # Store results for CSV
        csv_data[label] = hits_per_event_series.values

    # Plot formatting
    ax.axhline(0, color='gray', zorder=0)
    ax.annotate(f'Minimum Amplitude: {min_amp} ADC', xy=(0.5, 0.6), xycoords='axes fraction', ha='center')
    ax.set_title(f'Hits per Event vs Time for {sub_run}')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Hits per Event')
    ax.set_ylim(bottom=1e-3)
    ax.legend()
    fig.tight_layout()

    # Export to CSV if a path is provided
    if save_csv_path:
        pd.DataFrame(csv_data).to_csv(save_csv_path, index=False)
        print(f"Binned data saved to: {save_csv_path}")

    return fig, ax


def export_hits_vs_time_csvs_wrapper(base_path, hvs, output_dir, feus):
    """
    Wrap the export_hits_vs_time_csvs function and put all run info and metadata here.
    """
    # ── run_name_map ──────────────────────────────────────────────────────────────
    # Maps run id → human-readable label (used as the CSV column header / filename)

    run_name_map = {
        # Ar:CF4:Iso 88/10/2, 30 mm drift
        'run_18': 'run_18 - Timepix - Feb 3',
        'run_35': 'run_35 - Carbon - Feb 7 (Still flushing Helium)',

        # He:Ethane 96.5/3.5, 30 mm drift
        'run_26': 'run_26 - Empty - No Shielding',
        'run_27': 'run_27 - Lead - No Shielding',
        'run_29': 'run_29 - B4C - No Shielding (Quick)',
        'run_30': 'run_30 - B4C - No Shielding',
        'run_32': 'run_32 - Empty - Shielding',
        'run_33': 'run_33 - Carbon - Shielding',

        # Ar:CF4:CO2 45/40/15, 30 mm drift
        'run_43': 'run_43 - Carbon - No Shielding',

        # Ar:CF4 90/10, 30 mm drift
        'run_50': 'run_50 - Carbon',
        'run_57': 'run_57 - Beam Cap',
        'run_59': 'run_59 - B4C',

        # Ar:CO2 70/30, 30 mm drift
        'run_64': 'run_64 - B4C',
        'run_67': 'run_67 - None',
    }

    # ── run_metadata ──────────────────────────────────────────────────────────────
    # Maps run id → dict of descriptive fields written into the CSV header block

    run_metadata = {
        'run_18': {
            'gas': 'Ar:CF4:Iso 88/10/2',
            'target': 'Timepix',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '10 / 520',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-03',
            'notes': 'HV scan; first 2 subruns 0V/0V and 0V/300V',
        },
        'run_35': {
            'gas': 'Ar:CF4:Iso 88/10/2',
            'target': 'Carbon',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '20 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-07',
            'notes': 'Gas swap He→Ar; flushing; 9.4 µs delay',
        },

        'run_26': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'None',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '10 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-04',
            'notes': 'Overnight HV scan; no target',
        },
        'run_27': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'Lead',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '20 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-05',
            'notes': 'Lead target; step scan then stats run',
        },
        'run_29': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'B4C',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '400 / 475',
            'drift_hv_V': '-600',
            'date': '2026-02-05',
            'notes': 'B4C target; back to 600 fC; quick scan',
        },
        'run_30': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'B4C',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '20 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-05',
            'notes': 'Overnight HV scan',
        },
        'run_32': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'None + PE+Pb shield',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '20 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-06',
            'notes': 'Long HV scan; new noise possibly introduced; capped at 470V',
        },
        'run_33': {
            'gas': 'He:Ethane 96.5/3.5',
            'target': 'Carbon',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '20 / 510',
            'drift_hv_V': '-600 / -300 / 0',
            'date': '2026-02-06',
            'notes': 'Overnight HV scan with carbon target',
        },

        'run_43': {
            'gas': 'Ar:CF4:CO2 45/40/15',
            'target': 'Carbon',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '560 / 700',
            'drift_hv_V': '-600 / 0',
            'date': '2026-02-09',
            'notes': 'Flush run; 2 hr',
        },

        'run_50': {
            'gas': 'Ar:CF4 90/10',
            'target': 'Carbon',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '350 / 620',
            'drift_hv_V': '-600',
            'date': '2026-02-10',
            'notes': 'Beam+carbon; overnight HV scan after flush',
        },
        'run_57': {
            'gas': 'Ar:CF4 90/10',
            'target': 'None (cap on beam!)',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '405 / 620',
            'drift_hv_V': '-600 / 0',
            'date': '2026-02-13',
            'notes': 'Cap left on beam pipe!',
        },
        'run_59': {
            'gas': 'Ar:CF4 90/10',
            'target': 'B4C 2mm',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '405 / 620',
            'drift_hv_V': '-600 / 0',
            'date': '2026-02-13',
            'notes': 'Cap removed; B4C installed; overnight',
        },

        'run_64': {
            'gas': 'Ar:CO2 70/30',
            'target': 'B4C 2mm',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '440 / 770',
            'drift_hv_V': '-600 / 0',
            'date': '2026-02-16',
            'notes': 'Beam returning; still flushing',
        },
        'run_67': {
            'gas': 'Ar:CO2 70/30',
            'target': 'None (holder only)',
            'drift_gap_mm': 30,
            'frame': 'Al',
            'dist_cm': '~20',
            'trigger': 'PS',
            'resist_hv_V': '520 / 730',
            'drift_hv_V': '-600 / 0',
            'date': '2026-02-17',
            'notes': 'No target HV scan',
        },
    }

    # ── Call site ─────────────────────────────────────────────────────────────────

    export_hits_vs_time_csvs(base_path, list(run_name_map.keys()), hvs, feus, output_dir,
                             run_name_map=run_name_map, run_metadata=run_metadata)


def export_hits_vs_time_csvs(base_path, runs, hvs, feus, output_dir,
                              run_name_map=None, run_metadata=None):
    """
    For each (run, HV) combination, bins hits/event vs time and saves a CSV.

    Parameters
    ----------
    base_path    : str  - root directory containing run folders
    runs         : list of str - run identifiers (e.g. ['run_32', 'run_33_5D'])
    hvs          : list of int - HV values to process
    feus         : list - FEU identifiers passed to load_subrun
    output_dir   : str  - directory where CSVs will be saved
    run_name_map : dict, optional - maps run id -> human-readable label
    run_metadata : dict, optional - maps run id -> dict of metadata fields
                   e.g. {'run_32': {'gas': 'Ar/CO2 70/30', 'drift_gap_mm': 10}}
    """
    min_amp = 800
    time_bins = np.arange(0, 10000, 10)
    time_bin_mids = (time_bins[:-1] + time_bins[1:]) / 2

    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        # Parse drift suffix (e.g. 'run_32_5D' -> drift='5', run_dir='run_32')
        parts = run.split('_')
        if len(parts) == 3:
            drift = parts[-1].strip('D')
            run_dir_name = '_'.join(parts[:-1])
        else:
            drift = None
            run_dir_name = run

        label = run_name_map[run] if run_name_map else run
        meta = run_metadata[run] if run_metadata else {}

        for hv in hvs:
            print(f'Processing {run}, HV={hv}V')

            # Locate the matching sub-run directory
            run_dir = os.path.join(base_path, run_dir_name)
            hv_sub_run = None
            for sub_run_dir in os.listdir(run_dir):
                if run == 'run_32' and hv > 470:
                    continue
                if not os.path.isdir(os.path.join(run_dir, sub_run_dir)):
                    continue
                if f'resist_{hv}V_' in sub_run_dir:
                    if drift is not None and f'_drift_{drift}' not in sub_run_dir:
                        continue
                    hv_sub_run = sub_run_dir
                    break

            if hv_sub_run is None:
                print(f'  No sub-run found for HV={hv}V — skipping.')
                continue

            try:
                df, det = load_subrun(base_path, run_dir_name, hv_sub_run, feus)
            except FileNotFoundError:
                print(f'  File not found for HV={hv}V, sub-run={hv_sub_run} — skipping.')
                continue

            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f'  Empty dataframe for HV={hv}V — skipping.')
                continue

            df = df[df['amplitude'] >= min_amp]
            n_events = df['eventId'].nunique()
            if n_events == 0:
                print(f'  No events after amplitude cut for HV={hv}V — skipping.')
                continue

            # Bin hits/event vs time
            bins_cut = pd.cut(df['time'] / 3, bins=time_bins)
            hits_per_event = (
                df['time']
                .groupby(bins_cut, observed=False)
                .count()
                .reindex(bins_cut.cat.categories, fill_value=0)
                / n_events
            )

            # Build CSV with a metadata header
            fname = f'{run}_R{hv}V.csv'
            fpath = os.path.join(output_dir, fname)

            with open(fpath, 'w') as f:
                # --- Header block ---
                f.write(f'# Run:          {label}\n')
                f.write(f'# Run ID:        {run}\n')
                f.write(f'# HV (V):        {hv}\n')
                f.write(f'# Sub-run dir:   {hv_sub_run}\n')
                f.write(f'# Drift gap:     {drift + " mm" if drift else "N/A"}\n')
                f.write(f'# Min amplitude: {min_amp} ADC\n')
                f.write(f'# N events:      {n_events}\n')
                for key, val in meta.items():
                    f.write(f'# {key:<14} {val}\n')
                f.write('#\n')

                # --- Data ---
                pd.DataFrame({
                    'time_bin_mid_us': time_bin_mids / 1000,
                    'hits_per_event':  hits_per_event.values,
                }).to_csv(f, index=False)

            print(f'  Saved: {fpath}')


def plot_hits_vs_time_hvs(base_path, run, sub_run, feus, hvs):
    """
    Pass
    """
    min_amp = 1000
    # time_bins = [[0, 1000], [1000, 3000], [3000, 10000]]
    time_bins = np.arange(0, 10000, 100)
    time_bin_mids = (time_bins[:-1] + time_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(9, 4))
    for hv in hvs:
        sub_run_parts = sub_run.split('_')
        sub_run_parts[1] = f'{hv}V'
        sub_run_i = '_'.join(sub_run_parts)
        df, det = load_subrun(base_path, run, sub_run_i, feus)
        df = df[df['amplitude'] >= min_amp]
        n_events = df['eventId'].nunique()
        # Count the number of entries with 'time' within each time_bin
        # hits_per_event_series = df['time'].groupby(pd.cut(df['time'] / 3, bins=time_bins)).count() / n_events
        bins = pd.cut(df["time"] / 3, bins=time_bins)
        hits_per_event_series = (
                df["time"]
                .groupby(bins)
                .count()
                .reindex(bins.cat.categories, fill_value=0)
                / n_events
        )
        label = f'{hv}V'
        ax.plot(time_bin_mids / 1000, hits_per_event_series.values, label=label)
    ax.axhline(0, color='gray', zorder=0)
    ax.annotate(f'Minimum Amplitude: {min_amp} ADC', xy=(0.5, 0.9), xycoords='axes fraction',
                ha='center')
    ax.set_title(f'Hits per Event vs Time for {sub_run}')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Hits per Event')
    # ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3)
    ax.legend()
    fig.tight_layout()


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

    # df_x['time_us'] = df_x['time'] / 3 / 1000
    # tracks_x = find_tracks_seed_grow(df_x, pos_col='x_position_mm')
    #
    # df_y['time_us'] = df_y['time'] / 3 / 1000
    # tracks_y = find_tracks_seed_grow(df_y, pos_col='y_position_mm')


def plot_hits_per_event(base_path, run, sub_run, feus):
    """
    Plot hits on position vs time plot.
    """
    min_amp = 250
    df, det = load_subrun(base_path, run, sub_run, feus)
    print(f'Adding xy positions to dataframe...')

    df = df[(df['amplitude'] >= min_amp)]

    # Count number of entries for each eventId
    event_counts = df['eventId'].value_counts()
    fig, ax = plt.subplots()
    ax.scatter(event_counts.index, event_counts.values)
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Number of Hits')
    ax.set_title(f'Hits per Event for {run} - {sub_run}')
    fig.tight_layout()

    # Plot timestamp difference between consecutive events vs eventId. Calculate a moving average to get a rate.
    # Get unique eventIds
    event_ids = df['eventId'].unique()
    # Get one entry per eventId and get the trigger_timestamp_ns
    timestamps = df['trigger_timestamp_ns'].unique() / 1e9

    fig, ax = plt.subplots()
    ax.scatter(timestamps, event_ids)
    ax.set_xlabel('Trigger Timestamp (ns)')
    ax.set_ylabel('Event ID')
    ax.set_title(f'Trigger Timestamp vs Event ID for {run} - {sub_run}')
    fig.tight_layout()

    d_timestamps = np.diff(timestamps)

    fig, ax = plt.subplots()
    ax.plot(list(event_ids)[:-1], d_timestamps)
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Timestamp Difference (s)')
    ax.set_title(f'Timestamp Difference per Event for {run} - {sub_run}')
    fig.tight_layout()



def find_tracks_seed_grow(
        df,
        time_col='time_us',
        pos_col='x_position_mm',
        min_neighbors=1,
        neighbor_dt=0.15,      # μs
        neighbor_dx=3.0,       # mm
        seed_time_window=0.4,  # μs, start from the end
        corridor=5.0,          # mm
        min_track_points=8,
):
    """
    Prototype micro-TPC tracker using:
    clean -> seed at late times -> grow backward.

    Returns list of tracks, each track = indices of df.
    Also produces plots of each stage.
    """

    work = df.copy().reset_index(drop=True)

    # ----------------------------------------------------------
    # STEP 0 — raw
    # ----------------------------------------------------------
    fig, ax = plt.subplots()
    ax.scatter(work[time_col], work[pos_col], s=10)
    ax.set_title("Step 0: Raw hits")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Position (mm)")
    fig.tight_layout()

    # ----------------------------------------------------------
    # STEP 1 — remove isolated hits
    # ----------------------------------------------------------
    t = work[time_col].to_numpy()
    x = work[pos_col].to_numpy()

    keep = np.zeros(len(work), dtype=bool)

    for i in range(len(work)):
        dt = np.abs(t - t[i])
        dx = np.abs(x - x[i])
        neigh = np.sum((dt < neighbor_dt) & (dx < neighbor_dx)) - 1
        if neigh >= min_neighbors:
            keep[i] = True

    clean = work[keep].copy().reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.scatter(clean[time_col], clean[pos_col], s=10)
    ax.set_title("Step 1: After neighbor cleaning")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Position (mm)")
    fig.tight_layout()

    # ----------------------------------------------------------
    # STEP 2+ — iterative seed & grow
    # ----------------------------------------------------------
    tracks = []
    remaining = clean.copy().reset_index(drop=True)

    track_id = 0

    while True:
        if len(remaining) < min_track_points:
            break

        # late time region
        tmax = remaining[time_col].max()
        seed_hits = remaining[remaining[time_col] > tmax - seed_time_window]

        if len(seed_hits) < 2:
            break

        # choose 2 furthest in position to define initial slope
        i1 = seed_hits.index[0]
        i2 = seed_hits.index[-1]

        t1, x1 = remaining.loc[i1, [time_col, pos_col]]
        t2, x2 = remaining.loc[i2, [time_col, pos_col]]

        if t2 == t1:
            break

        slope = (x2 - x1) / (t2 - t1)
        intercept = x1 - slope * t1

        # grow backward
        attached = []

        for i, row in remaining.iterrows():
            ti = row[time_col]
            xi = row[pos_col]
            pred = slope * ti + intercept

            if abs(xi - pred) < corridor:
                attached.append(i)

        if len(attached) < min_track_points:
            break

        # refit
        tt = remaining.loc[attached, time_col].to_numpy()
        xx = remaining.loc[attached, pos_col].to_numpy()
        p = np.polyfit(tt, xx, 1)
        slope, intercept = p

        tracks.append(remaining.loc[attached].copy())
        track_id += 1

        # remove
        remaining = remaining.drop(attached).reset_index(drop=True)

        # plot this extraction
        fig, ax = plt.subplots()
        ax.scatter(clean[time_col], clean[pos_col], s=5, alpha=0.2, label='all')
        ax.scatter(tt, xx, s=15, label=f'track {track_id}')
        ax.plot(clean[time_col], slope * clean[time_col] + intercept)
        ax.set_title(f"Step 2: Extracted track {track_id}")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Position (mm)")
        ax.legend()
        fig.tight_layout()

    # ----------------------------------------------------------
    # final leftover
    # ----------------------------------------------------------
    if len(remaining) > 0:
        fig, ax = plt.subplots()
        ax.scatter(remaining[time_col], remaining[pos_col], s=10)
        ax.set_title("Remaining hits (unassigned)")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Position (mm)")
        fig.tight_layout()

    return tracks


def plot_amps_with_hv(base_path, run, sub_run, feus, hvs):
    """
    Plot hit time distribution for several subruns
    """
    fig, ax = plt.subplots()
    # bins = np.arange(0, 10000, 100)
    for hv in hvs:
        sub_run_parts = sub_run.split('_')
        sub_run_parts[1] = f'{hv}V'
        hv_sub_run = '_'.join(sub_run_parts)
        df, det = load_subrun(base_path, run, hv_sub_run, feus)
        amplitudes_array = df['amplitude'].to_numpy()
        local_max_array = df['local_max'].to_numpy()
        bins = ax.hist(amplitudes_array, bins=100, alpha=0.4, label=f'{hv}V', log=True)
        # ax.hist(local_max_array, bins=bins[1], alpha=0.4, log=True, label=f'Max Sample ADC')
    ax.set_title('Amplitudes Histogram')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Hits')
    ax.legend()
    fig.tight_layout()


def load_subrun(base_path, run, sub_run, feus):
    """
    Load subrun
    """

    run_config_path = f'{base_path}{run}/run_config.json'
    map_csv_path = './mx17_m4_map.csv'

    rc = RunConfig(run_config_path, map_csv_path)

    det = rc.get_detector('mx17_1')

    hits_dir = f'{base_path}{run}/{sub_run}/combined_hits_root/'
    hit_files = [f for f in os.listdir(hits_dir) if f.endswith('.root') and '_datrun_' in f]

    file_sources = [f'{hits_dir}{hf}:hits' for hf in hit_files]
    df = uproot.concatenate(file_sources, library='pd')

    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df[df['feu'].isin(feus)]

    return df, det


def add_xy_pos(df, det):
    """
    Append x and y positions to dataframe
    """

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

    return df


def plot_general_metrics(base_path, run, sub_run, feus):
    """
    Plot general metrics for full subrun.
    """
    df, det = load_subrun(base_path, run, sub_run, feus)
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
    local_max_array = df['local_max'].to_numpy()

    # Plot amplitudes histogram in log scale
    fig, ax = plt.subplots()
    bins = ax.hist(amplitudes_array, bins=100, color='orange', alpha=1.0, log=True, label='Amplitude Estimation')[1]
    ax.hist(local_max_array, bins=bins, color='green', alpha=0.4, log=True, label='Max Sample ADC')
    ax.set_title('Amplitudes Histogram')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Hits (log scale)')
    ax.legend()
    plt.tight_layout()

    # Get amplitude sums for each event.
    event_sums = df.groupby('eventId')['amplitude'].sum().reset_index()
    fig, ax = plt.subplots()
    ax.hist(event_sums['amplitude'], bins=100, color='purple', alpha=0.7)
    ax.set_title('Event Amplitude Sums Histogram')
    ax.set_xlabel('Event Amplitude Sum')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(local_max_array, amplitudes_array, alpha=0.5)
    ax.set_xlabel('Local Max')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()

    times_array = df['time'].to_numpy() / 3

    fig, ax = plt.subplots()
    ax.hist(times_array, bins=100, color='orange', alpha=0.7)
    ax.set_title('Time of Arrival Histogram')
    ax.set_xlabel('Time of Arrival (ns)')
    ax.set_ylabel('Hits')
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
    hits_per_event = grouped['eventId'].count().to_numpy()

    print(avg_y_positions)

    # Drop entries where either avg_x_positions or avg_y_positions is NaN
    valid_indices = ~np.isnan(avg_x_positions) & ~np.isnan(avg_y_positions)
    avg_x_positions = avg_x_positions[valid_indices]
    avg_y_positions = avg_y_positions[valid_indices]
    hits_per_event = hits_per_event[valid_indices]

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
    sc = ax.scatter(avg_x_positions, avg_y_positions, alpha=0.3, c=hits_per_event, cmap='jet', s=1.5)
    ax.set_title('Scatter Plot of Average X and Y Positions per Event')
    ax.set_xlabel('Average X Position (mm)')
    ax.set_ylabel('Average Y Position (mm)')
    plt.colorbar(sc, ax=ax, label='N Hits')
    plt.tight_layout()

    # Scatter plot of average x and y positions converted to strip number
    fig, ax = plt.subplots()
    ax.scatter(avg_x_positions / pitch, avg_y_positions / pitch, alpha=0.3, color='green', s=1)
    ax.set_title('Scatter Plot of Average X and Y Positions per Event')
    ax.set_xlabel('Average X Position (Strip Number)')
    ax.set_ylabel('Average Y Position (Strip Number)')
    plt.tight_layout()


def get_run_time(base_path, run, sub_run):
    run_time_file = f'{base_path}{run}/{sub_run}/raw_daq_data/run_time.txt'
    with open(run_time_file, 'r') as f:
        lines = f.readlines()
        run_time = float(lines[0].replace('Run Time: ', '').replace(' seconds', ''))
    return run_time


def get_run_json(base_path, run):
    run_json_file = f'{base_path}{run}/run_config.json'
    with open(run_json_file, 'r') as f:
        run_json = json.load(f)
    return run_json


def plot_general_metrics_fast(base_path, run, sub_run, feus, feus_map):
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

    min_amp = 400
    df = df[df['amplitude'] >= min_amp]

    amplitudes_array = df['amplitude'].to_numpy()
    local_max_array = df['local_max'].to_numpy()

    # Plot amplitudes histogram in log scale
    fig, ax = plt.subplots()
    bins = ax.hist(amplitudes_array, bins=100, color='orange', alpha=1.0, log=True, label='Amplitude Estimation')[1]
    ax.hist(local_max_array, bins=bins, color='green', alpha=0.4, log=True, label='Max Sample ADC')
    ax.set_title('Amplitudes Histogram')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Hits (log scale)')
    ax.legend()
    plt.tight_layout()

    # # Get amplitude sums for each event.
    event_sums = df.groupby('eventId')['amplitude'].sum().reset_index()
    # fig, ax = plt.subplots()
    # ax.hist(event_sums['amplitude'], bins=100, color='purple', alpha=0.7)
    # ax.set_title('Event Amplitude Sums Histogram')
    # ax.set_xlabel('Event Amplitude Sum')
    # ax.set_ylabel('Hits')
    # plt.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(local_max_array, amplitudes_array, alpha=0.5)
    ax.set_xlabel('Local Max')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()

    times_array = df['time'].to_numpy() / 3

    # Plot 1D arrays for channel hits
    fig, axs = plt.subplots(nrows=2, ncols=1)
    for i, feu in enumerate(feus):
        df_feu = df[df['feu'] == feu]
        axs[i].hist(df_feu['channel'], bins=np.arange(-0.5, 512.5, 1), alpha=0.7)
        axs[i].set_title(f'Feu {feu} Histogram')
        axs[i].set_xlabel('Channel')
        axs[i].set_ylabel('Hits')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(times_array, bins=100, color='orange', alpha=0.7)
    ax.set_title('Time of Arrival Histogram')
    ax.set_xlabel('Time of Arrival (ns)')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    # Plot 1D arrays for x and y positions
    x_positions = df['x_position_mm'].dropna().to_numpy()
    y_positions = df['y_position_mm'].dropna().to_numpy()
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
    hits_per_event = grouped['eventId'].count().to_numpy()

    max_hits = 400
    # 1. Calculate hits per event and map it back to every row in the original df
    df['event_hit_count'] = df.groupby('eventId')['eventId'].transform('count')

    # 2. Filter the original dataframe
    df = df[df['event_hit_count'] <= max_hits].copy()

    # Optional: Clean up by dropping the helper column
    df.drop(columns=['event_hit_count'], inplace=True)

    # Get amplitude sums for each event.
    event_sums_post = df.groupby('eventId')['amplitude'].sum().reset_index()
    fig, ax = plt.subplots()
    hist = ax.hist(event_sums['amplitude'], bins=100, color='purple', alpha=0.7, label='Before NHits Cut')
    ax.hist(event_sums_post['amplitude'], bins=hist[1], color='green', alpha=0.7, label='After NHits Cuts')
    ax.legend()
    ax.set_title('Event Amplitude Sums Histogram')
    ax.set_xlabel('Event Amplitude Sum')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    # Get plot cut sums by themselves
    event_sums_post = df.groupby('eventId')['amplitude'].sum().reset_index()
    fig, ax = plt.subplots()
    ax.hist(event_sums_post['amplitude'], bins=100, color='green', alpha=0.7, label='After NHits Cuts')
    ax.legend()
    ax.set_title('Event Amplitude Sums Histogram')
    ax.set_xlabel('Event Amplitude Sum')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    # Plot 1D arrays for x and y positions
    x_positions = df['x_position_mm'].dropna().to_numpy()
    y_positions = df['y_position_mm'].dropna().to_numpy()
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

    times_array = df_filter['time'].to_numpy() / 3
    times_array = times_array[times_array > -500]

    fig, ax = plt.subplots()
    ax.hist(times_array, bins=100, color='orange', alpha=0.7)
    ax.set_title('Time of Arrival Histogram')
    ax.set_xlabel('Time of Arrival (ns)')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    plot_2ds(df, pitch, run_time)
    plot_2ds(df_filter, pitch, run_time)

    # # Plot 1D arrays for x and y positions
    # x_positions = df_filter['x_position_mm'].dropna().to_numpy()
    # y_positions = df_filter['y_position_mm'].dropna().to_numpy()
    # bins = np.arange(-pitch / 2, 512 * pitch + pitch / 2, pitch)
    # fig, axs = plt.subplots(nrows=2, ncols=1)
    # axs[0].hist(x_positions, bins=bins, color='blue', alpha=0.7)
    # axs[0].set_title('X Positions Histogram')
    # axs[0].set_xlabel('X Position (mm)')
    # axs[0].set_ylabel('Counts')
    # axs[1].hist(y_positions, bins=bins, color='green', alpha=0.7)
    # axs[1].set_title('Y Positions Histogram')
    # axs[1].set_xlabel('Y Position (mm)')
    # axs[1].set_ylabel('Counts')
    # plt.tight_layout()
    #
    # grouped = df_filter.groupby('eventId')
    # avg_x_positions = grouped['x_position_mm'].mean().to_numpy()
    # avg_y_positions = grouped['y_position_mm'].mean().to_numpy()
    # hits_per_event = grouped['eventId'].count().to_numpy()
    #
    # valid_indices = ~np.isnan(avg_x_positions) & ~np.isnan(avg_y_positions)
    # avg_x_positions = avg_x_positions[valid_indices]
    # avg_y_positions = avg_y_positions[valid_indices]
    # hits_per_event = hits_per_event[valid_indices]
    # event_nums = grouped.groups.keys()
    # print(f'Remaining event numbers: {", ".join(map(str, event_nums))}.')
    # print(f'Number of events remaining: {len(event_nums)}.')
    # print(f'Run time: {run_time:.2f} s.')
    # print(f'Rate: {len(event_nums) / run_time:.2f} Hz.')
    #
    # # Calculate the percentage drop due to NaN values
    # total_events = len(grouped)
    # valid_events = len(avg_x_positions)
    # if total_events == 0:
    #     drop_percentage = np.nan
    # else:
    #     drop_percentage = (total_events - valid_events) / total_events * 100
    # print(f'Dropped {drop_percentage:.2f}% of events due to NaN average positions.')
    #
    # try:
    #     # 2D histogram of average x and y positions
    #     fig, ax = plt.subplots()
    #     h = ax.hist2d(avg_x_positions, avg_y_positions, bins=100, norm=LogNorm(), cmap='jet')
    #     ax.set_title('2D Histogram of Average X and Y Positions per Event')
    #     ax.set_xlabel('Average X Position (mm)')
    #     ax.set_ylabel('Average Y Position (mm)')
    #     plt.colorbar(h[3], ax=ax, label='Counts')
    #     plt.tight_layout()
    #
    #     # Scatter plot of average x and y positions
    #     fig, ax = plt.subplots()
    #     sc = ax.scatter(avg_x_positions, avg_y_positions, alpha=0.3, c=hits_per_event, cmap='jet', s=1.5)
    #     ax.set_title('Scatter Plot of Average X and Y Positions per Event')
    #     ax.set_xlabel('Average X Position (mm)')
    #     ax.set_ylabel('Average Y Position (mm)')
    #     plt.colorbar(sc, ax=ax, label='N Hits')
    #     plt.tight_layout()
    #
    #     # Scatter plot of average x and y positions converted to strip number
    #     fig, ax = plt.subplots()
    #     ax.scatter(avg_x_positions / pitch, avg_y_positions / pitch, alpha=0.3, color='green', s=1)
    #     ax.set_title('Scatter Plot of Average X and Y Positions per Event')
    #     ax.set_xlabel('Average X Position (Strip Number)')
    #     ax.set_ylabel('Average Y Position (Strip Number)')
    #     plt.tight_layout()
    # except Exception as e:
    #     print(f'Error plotting positions: {e}')


def plot_2ds(df, pitch, run_time):
    # Plot 1D arrays for x and y positions
    x_positions = df['x_position_mm'].dropna().to_numpy()
    y_positions = df['y_position_mm'].dropna().to_numpy()
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

    grouped = df.groupby('eventId')
    avg_x_positions = grouped['x_position_mm'].mean().to_numpy()
    avg_y_positions = grouped['y_position_mm'].mean().to_numpy()
    hits_per_event = grouped['eventId'].count().to_numpy()

    valid_indices = ~np.isnan(avg_x_positions) & ~np.isnan(avg_y_positions)
    avg_x_positions = avg_x_positions[valid_indices]
    avg_y_positions = avg_y_positions[valid_indices]
    hits_per_event = hits_per_event[valid_indices]
    event_nums = grouped.groups.keys()
    print(f'Remaining event numbers: {", ".join(map(str, event_nums))}.')
    print(f'Remaining hits per event: {", ".join(map(str, hits_per_event))}.')
    print(f'Number of events remaining: {len(event_nums)}.')
    print(f'Run time: {run_time:.2f} s.')
    print(f'Rate: {len(event_nums) / run_time:.2f} Hz.')

    # Calculate the percentage drop due to NaN values
    total_events = len(grouped)
    valid_events = len(avg_x_positions)
    if total_events == 0:
        drop_percentage = np.nan
    else:
        drop_percentage = (total_events - valid_events) / total_events * 100
    print(f'Dropped {drop_percentage:.2f}% of events due to NaN average positions.')

    try:
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
        sc = ax.scatter(avg_x_positions, avg_y_positions, alpha=0.3, c=hits_per_event, cmap='jet', s=1.5)
        ax.set_title('Scatter Plot of Average X and Y Positions per Event')
        ax.set_xlabel('Average X Position (mm)')
        ax.set_ylabel('Average Y Position (mm)')
        plt.colorbar(sc, ax=ax, label='N Hits')
        plt.tight_layout()

        # Scatter plot of average x and y positions converted to strip number
        fig, ax = plt.subplots()
        ax.scatter(avg_x_positions / pitch, avg_y_positions / pitch, alpha=0.3, color='green', s=1)
        ax.set_title('Scatter Plot of Average X and Y Positions per Event')
        ax.set_xlabel('Average X Position (Strip Number)')
        ax.set_ylabel('Average Y Position (Strip Number)')
        plt.tight_layout()
    except Exception as e:
        print(f'Error plotting positions: {e}')



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
