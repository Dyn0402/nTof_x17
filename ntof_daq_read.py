#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 27 3:19 PM 2026
Created in PyCharm
Created as nTof_x17/ntof_daq_read.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    base_path = '/media/dylan/data/x17/feb_beam/ntof_daq_data'
    print(os.listdir(base_path))
    for file in os.listdir(base_path):
        df = pd.read_csv(os.path.join(base_path, file), header=10, sep='\t', na_values='-')
        df = df.fillna(0).astype(int)
        print(df)
        print(df.columns)

        # Make columns to int values
        # df['Time (ns)'] = df['Time (ns)'].astype(int)
        # df['SILI/1: Amplitude (ADC ch)'] = df['SILI/1: Amplitude (ADC ch)'].astype(int)

        fig, ax = plt.subplots()
        ax.plot(df['Time (ns)'], df['MGAS/1: Amplitude (ADC ch)'])
        plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
