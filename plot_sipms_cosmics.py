#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 12 9:32 AM 2026
Created in PyCharm
Created as nTof_x17/plot_sipms_cosmics.py

@author: Dylan Neff, dylan
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plot_beam_hits import load_subrun


def main():
    base_path = '/media/dylan/data/x17/feb_beam/runs/'
    run = 'run_52'
    sub_run = 'initial_resist_610V_drift_600V'

    feus = {4: 'MX17 X Strips', 5: 'MX17 Y Strips', 6: 'SiPMs'}

    print(f'Loading data for {run} - {sub_run}...')
    df, det = load_subrun(base_path, run, sub_run, feus=list(feus.keys()))
    df['time'] = df['time'] / 3  # Correct for 60ns samples assumed in reconstruction
    df = df[df['eventId'] < 10000]

    df_sipm = df[df['feu'] == 6]

    fig, ax = plt.subplots()
    channel_bins = np.arange(-0.5, 64.5, 1)
    ax.hist(df_sipm['channel'], bins=channel_bins, alpha=0.8, label='SiPMs')
    ax.legend()
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Hits')
    plt.tight_layout()

    fig, ax = plt.subplots()
    sns.barplot(data=df_sipm, x='channel', y='amplitude', ax=ax)
    # Rotate x axis labels
    plt.xticks(rotation=-40)
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Average Amplitude')
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.histplot(data=df_sipm, x='channel', y='local_max', bins=(channel_bins, 256), ax=ax)
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Amplitude')
    fig.tight_layout()

    fig, ax = plt.subplots()
    for channel in [11, 14]:
        sns.histplot(data=df_sipm[df_sipm['channel'] == channel], x='time', ax=ax, label=f'Channel {channel}')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Hits')
    ax.legend()
    fig.tight_layout()

    df_mms = df[df['feu'].isin([4, 5])]
    df_mms = df_mms[(df_mms['time'] > 0) & (df_mms['time'] < 10000)]

    fig, ax = plt.subplots()
    for feu in [4, 5]:
        sns.histplot(data=df_mms[df_mms['feu'] == feu], x='time', ax=ax, label=feus[feu])
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Hits')
    ax.legend()
    fig.tight_layout()

    df_mms = df_mms[(df_mms['amplitude'] > 200) & (df_mms['time'] < 1000)]
    fig, ax = plt.subplots()  # Plot hits per event
    for feu in [4, 5]:
        sns.histplot(data=df_mms[df_mms['feu'] == feu], x='eventId', ax=ax, label=feus[feu])
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Hits')
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    for feu in [4, 5]:
        sns.scatterplot(data=df_mms[df_mms['feu'] == feu], x='eventId', y='amplitude', ax=ax, label=feus[feu])
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Event Amplitudes')
    ax.legend()
    fig.tight_layout()

    df_mms_saturated = df_mms[df_mms['saturated'] == 1]
    # 1. Group by event and feu, then count the number of saturated hits
    # We use reset_index(name='hit_count') to turn the result back into a clean DataFrame
    sat_counts = df_mms_saturated.groupby(['eventId', 'feu']).size().reset_index(name='hit_count')

    fig, ax = plt.subplots()

    for feu in [4, 5]:
        # 2. Filter the counts for the specific FEU
        subset = sat_counts[sat_counts['feu'] == feu]

        sns.scatterplot(
            data=subset,
            x='eventId',
            y='hit_count',  # This is our new count column
            ax=ax,
            label=feus[feu],
            alpha=0.7  # Transparency helps if points overlap
        )

    ax.set_xlabel('Event ID')  # Or 'Time' if eventId corresponds to time
    ax.set_ylabel('Saturated Hits per Event')
    ax.legend()
    fig.tight_layout()


    plt.show()


    print('donzo')


if __name__ == '__main__':
    main()
