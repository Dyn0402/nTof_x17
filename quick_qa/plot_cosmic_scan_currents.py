#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 02 12:44 PM 2026
Created in PyCharm
Created as nTof_x17/plot_cosmic_scan_currents.py

@author: Dylan Neff, dylan
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    base_path = '/media/dylan/data/x17/cosmic_bench/det_1/'
    # run = 'mx17_det1_daytime_run_1-28-26'
    run = 'mx17_det0_He_HV_Scan_4-1-26'

    plot_scan_currents(base_path, run)

    plt.show()
    print('donzo')


def plot_scan_currents(base_path, run):
    run_dir = f'{base_path}{run}'
    sub_runs = sorted([x for x in os.listdir(run_dir) if os.path.isdir(f'{run_dir}/{x}')])

    # Build combined dataframe across all subruns
    dfs = []
    for sub_run in sub_runs:
        hv_file = f'{run_dir}/{sub_run}/hv_monitor.csv'
        if not os.path.exists(hv_file):
            print(f'No hv_monitor.csv found in {sub_run}, skipping.')
            continue
        df = pd.read_csv(hv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['subrun'] = sub_run
        dfs.append(df)

    if not dfs:
        print('No HV data found.')
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Find all card:channel pairs from imon and vmon columns
    imon_cols = [c for c in combined.columns if re.match(r'^\d+:\d+ imon$', c)]
    vmon_cols = [c for c in combined.columns if re.match(r'^\d+:\d+ vmon$', c)]
    if not imon_cols:
        print('No imon columns found.')
        return

    # Group channels by card number
    cards = {}
    for col in imon_cols:
        card_ch = col.replace(' imon', '')
        card = card_ch.split(':')[0]
        cards.setdefault(card, []).append(card_ch)

    # Color map for subruns
    sub_run_list = combined['subrun'].unique()
    cmap = plt.get_cmap('tab10')
    subrun_colors = {sr: cmap(i % 10) for i, sr in enumerate(sub_run_list)}

    legend_elements = [Line2D([0], [0], color=subrun_colors[sr], lw=2, label=sr)
                       for sr in sub_run_list]

    def make_fig(card, channels, col_suffix, ylabel_suffix, title_suffix):
        n_channels = len(channels)
        fig, axs = plt.subplots(nrows=n_channels, sharex=True,
                                figsize=(10, 3 * n_channels))
        if n_channels == 1:
            axs = [axs]

        for ax, card_ch in zip(axs, sorted(channels)):
            col = f'{card_ch} {col_suffix}'
            if col not in combined.columns:
                continue
            for sub_run in sub_run_list:
                sub_df = combined[combined['subrun'] == sub_run]
                ax.plot(sub_df['timestamp'], sub_df[col],
                        color=subrun_colors[sub_run], label=sub_run)
            ax.set_ylabel(f'{card_ch}\n{ylabel_suffix}')

        axs[-1].set_xlabel('Time')
        axs[0].legend(handles=legend_elements, title='Subrun',
                      bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')

        fig.suptitle(f'{run}  —  HV Card {card} {title_suffix}')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

    # One current figure and one voltage figure per card
    for card, channels in sorted(cards.items()):
        make_fig(card, channels, 'imon', 'Current (A)', 'Currents vs Time')
        if vmon_cols:
            make_fig(card, channels, 'vmon', 'Voltage (V)', 'Voltages vs Time')


if __name__ == '__main__':
    main()
