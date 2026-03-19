#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on March 19 7:11 PM 2026
Created in PyCharm
Created as nTof_x17/plot_beam_intensities.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    runs_dir = '/media/dylan/data/x17/feb_beam/runs/'
    run = 'run_141'

    beam_intensity_csv_name = 'beam_intensity.csv'
    run_dir = os.path.join(runs_dir, run)
    print(run_dir)

    dfs = []
    for sub_run in os.listdir(run_dir):
        if not os.path.isdir(os.path.join(run_dir, sub_run)):
            continue
        print(sub_run)
        try:
            df = pd.read_csv(os.path.join(run_dir, sub_run, beam_intensity_csv_name))
            dfs.append(df)
        except FileNotFoundError:
            print(f'Beam intensity csv file not found for {sub_run}')

    df = pd.concat(dfs)
    print(df.head())

    fig, ax = plt.subplots()
    ax.hist(df['beam_intensity'], bins=100)
    ax.set_title(f'Beam Intensity for {run}')
    ax.set_xlabel('Beam Intensity (x10^? protons)')
    fig.tight_layout()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
