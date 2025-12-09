#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 30 13:55 2025
Created in PyCharm
Created as nTof_x17/plot_ntof_daq_waveform

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from io import StringIO


def main():
    file_path = "/local/home/dn277127/x17/X17_Wall_waveform"

    numeric_lines = []

    with open(file_path, "r") as f:
        for line in f:
            # Find the first line that starts with a digit -> start of waveform
            if re.match(r"^\s*\d+(\s+|\t+|-?\d|\.)", line):
                numeric_lines.append(line)
                break  # Found the start, now read the rest

        # Append remaining lines (all should be numeric waveform)
        for line in f:
            numeric_lines.append(line)

    # Load into DataFrame
    df = pd.read_csv(
        StringIO("".join(numeric_lines)),
        sep=r"\s+",
        names=["Time_ns", "Amplitude_V", "Amplitude_ADC"],
        engine="python"
    )

    print(df.head())  # For verification

    # ---- Plot Voltage waveform ----
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time_ns"], df["Amplitude_V"])
    plt.title("Waveform Voltage vs Time")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
