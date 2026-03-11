#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 10 4:38 PM 2026
Created in PyCharm
Created as nTof_x17/plot_subrun_hit_metrics.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uproot


def main():
    run = 'run_88_testing'
    # run = 'run_88'
    subrun = 'resist_530V_drift_1000V'
    feu = 5
    file_num = 0
    runs_base = '/media/dylan/data/x17/feb_beam/runs/'
    hits_dir = 'hits_root'

    run_dir = os.path.join(runs_base, run)
    subrun_dir = os.path.join(run_dir, subrun)
    hits_dir_path = os.path.join(subrun_dir, hits_dir)

    hits_paths = [os.path.join(hits_dir_path, f) for f in os.listdir(hits_dir_path) if f.endswith('.root')]

    # Open all with uproot
    arrs = uproot.concatenate({f: "hits" for f in hits_paths}, library='pd')

    print(arrs)
    print(arrs.columns)

    fig, ax = plt.subplots()
    sns.histplot(data=arrs, x='time', y='local_max', bins=200, cbar=True, cmap='magma')
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.histplot(data=arrs, x='time', y='local_baseline', bins=200, cbar=True, cmap='magma')
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.histplot(data=arrs, x='time', y='time_over_threshold', bins=200, cbar=True, cmap='magma')
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.histplot(data=arrs, x='time_over_threshold', y='local_max', bins=200, cbar=True, cmap='magma')
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.histplot(data=arrs, x='time', bins=200)
    sns.histplot(data=arrs[arrs['amplitude'] > 400], x='time', bins=200)
    fig.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
