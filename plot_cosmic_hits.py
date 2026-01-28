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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot


def main():
    base_path = '/media/dylan/data/x17/cosmic_bench/det_1/mx17_1-27-26/'
    run = 'mx17_det1_1-27-26'
    sub_run = 'resist_scan_480V'
    feus = {1: 'm3',6: 'MX17'}
    # base_path = '/mnt/data/x17/cosmic_bench/det_1/'
    # run = 'mx17_det1_overnight_run_1-27-26'
    # sub_run = 'overnight_run'
    # feus = {1: 'm3', 4: 'MX17 X Strips', 6: 'MX17 Y Strips'}
    file_nums = [0]

    hits_dir = f'{base_path}{run}/{sub_run}/hits_root/'
    hit_files = [f for f in os.listdir(hits_dir) if f.endswith('.root') and '_datrun_' in f]

    print(hit_files)
    for hit_file in hit_files:
        file_path = os.path.join(hits_dir, hit_file)
        print(hit_file)
        file_num, feu_num = extract_file_numbers_tuple(hit_file)
        if file_num not in file_nums:
            continue
        print('File num:', file_num, 'FEU num:', feu_num)
        print(f'Processing {file_path}...')
        with uproot.open(file_path) as f:
            print(f.keys())
            tree = f['hits']
            hits_data = tree.arrays(library='np')
            channels = hits_data['channel']
            amps = hits_data['amplitude']

            # Histogram of hits per channel
            plt.figure(figsize=(10, 6))
            plt.hist(channels, bins=np.arange(channels.min(), channels.max()+2)-0.5, histtype='stepfilled', alpha=0.7, color='blue')
            plt.xlabel('Channel Number')
            plt.ylabel('Number of Hits')
            plt.title(f'Hit Distribution per Channel for {feus.get(feu_num, "Unknown FEU")}')
            plt.grid(True)
            plt.tight_layout()

            # Make 2D histogram of channel vs amplitude with log color scale, zeros white, cmap 'jet'
            plt.figure(figsize=(10, 6))
            x_bins = np.arange(channels.min(), channels.max() + 2) - 0.5
            y_bins = np.linspace(400, 4100, 100)
            H, x_edges, y_edges = np.histogram2d(channels, amps, bins=[x_bins, y_bins])
            H_masked = np.ma.masked_where(H == 0, H)
            # cmap = plt.cm.get_cmap('jet')
            cmap = plt.colormaps['jet']
            cmap.set_bad('white')
            mesh = plt.pcolormesh(x_edges, y_edges, H_masked.T, cmap=cmap, norm=LogNorm())
            plt.colorbar(mesh, label='Counts')
            plt.xlabel('Channel Number')
            plt.ylabel('Amplitude')
            plt.title(f'Channel vs Amplitude for {feus.get(feu_num, "Unknown FEU")}')
            plt.tight_layout()

            # Plot mean amplitude vs channel as line plot and std deviation as a shaded area
            mean_amps = []
            std_amps = []
            min_amp = 500
            channel_centers = (x_edges[:-1] + x_edges[1:]) / 2
            for i in range(len(x_edges) - 1):
                bin_mask = (channels >= x_edges[i]) & (channels < x_edges[i + 1]) & (amps >= min_amp)
                bin_amps = amps[bin_mask]
                if len(bin_amps) > 0:
                    mean_amps.append(np.mean(bin_amps))
                    std_amps.append(np.std(bin_amps))
                else:
                    mean_amps.append(0)
                    std_amps.append(0)
            mean_amps = np.array(mean_amps)
            std_amps = np.array(std_amps)
            plt.figure(figsize=(10, 6))
            plt.plot(channel_centers, mean_amps, label='Mean Amplitude', color='red')
            plt.fill_between(channel_centers, mean_amps - std_amps, mean_amps + std_amps, color='red', alpha=0.3, label='Std Deviation')
            plt.xlabel('Channel Number')
            plt.ylabel('Amplitude')
            plt.title(f'Mean Amplitude vs Channel for {feus.get(feu_num, "Unknown FEU")}')
            plt.legend()
            plt.tight_layout()
    plt.show()
    print('donzo')


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
