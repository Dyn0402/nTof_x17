#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 18:15 2025
Created in PyCharm
Created as nTof_x17/plot_waveform_hits

@author: Dylan Neff, dn277127
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt

def main():
    # hits_file = '/local/home/dn277127/CLionProjects/mm_strip_reconstruction/build/waveform_analysis/hits.root'
    # waveform_file = "/local/home/dn277127/x17/decoder_test/ftest.root"
    hits_file = '/home/dylan/CLionProjects/mm_strip_reconstruction/test/run_81/hits.root'
    waveform_file = "/home/dylan/CLionProjects/mm_strip_reconstruction/test/run_81/Mx17_run_datrun_251204_18H17_000_05.root"
    event_id = 22
    channel_id = 28

    # channel_id = 29

    plot_waveform_and_hits(waveform_file, hits_file, event_id, channel_id)

    print('donzo')


def plot_waveform_and_hits(waveform_file, hits_file, event_id, channel_id):
    """
    waveform_file : str   path to ROOT file with the 'nt' tree (raw waveforms)
    hits_file     : str   path to ROOT file with the 'hits' tree
    event_id      : int   eventId to plot
    channel_id    : int   channel to plot
    """

    # ---------------------------
    # Read waveform tree
    # ---------------------------
    with uproot.open(waveform_file) as f:
        nt = f["nt"]
        evt_ids = nt["eventId"].array(library="np")
        samples = nt["sample"].array(library="np")     # Jagged array
        channels = nt["channel"].array(library="np")   # Jagged array
        amplitudes = nt["amplitude"].array(library="np")

    # Find index in nt corresponding to the eventId
    match = np.where(evt_ids == event_id)[0]
    if len(match) == 0:
        raise ValueError(f"Event {event_id} not found in nt tree.")
    idx = match[0]

    # Extract this event
    evt_samples = samples[idx]
    evt_channels = channels[idx]
    evt_amplitudes = amplitudes[idx]

    # Select waveform entries where channel == channel_id
    mask = (evt_channels == channel_id)
    if not np.any(mask):
        raise ValueError(f"Channel {channel_id} not found in event {event_id} of nt tree.")

    waveform_sample_idx = evt_samples[mask]
    waveform_amp = evt_amplitudes[mask]

    # ---------------------------
    # Read hit tree
    # ---------------------------
    with uproot.open(hits_file) as f:
        hits = f["hits"]
        hit_evt   = hits["eventId"].array(library="np")
        hit_ch    = hits["channel"].array(library="np")
        hit_samp  = hits["sample"].array(library="np")
        hit_amp   = hits["amplitude"].array(library="np")
        hit_time  = hits["time"].array(library="np")

    # Select hits matching event and channel
    hit_mask = (hit_evt == event_id) & (hit_ch == channel_id)

    hit_sample_positions = hit_samp[hit_mask]
    hit_amplitudes = hit_amp[hit_mask]
    hit_times = hit_time[hit_mask]

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(10,6))

    plt.plot(waveform_sample_idx, waveform_amp, label="Waveform", marker='.', linewidth=1)
    # Get minimum value of y axis of plot
    y_min = np.min(waveform_amp)

    # Overlay hits as vertical lines
    for hs, ha in zip(hit_sample_positions, hit_amplitudes):
        plt.axvline(hs, color="red", linestyle="--", alpha=0.7)
        plt.annotate(f"{hs:.1f}", xy=(hs, y_min), xytext=(-4, 0), textcoords='offset pixels',
                     color="red", fontsize=11, rotation=90, va='center', ha='right')
        # plt.text(hs, np.min(waveform_amp), f"{ha:.1f}", color="blue", fontsize=11, rotation=90,
        #          verticalalignment="bottom", horizontalalignment="left")

    plt.title(f"Event {event_id}, Channel {channel_id}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
