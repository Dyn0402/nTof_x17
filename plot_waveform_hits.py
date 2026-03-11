#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 18:15 2025
Created in PyCharm
Created as nTof_x17/plot_waveform_hits

@author: Dylan Neff, dn277127
"""

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf


def main():
    run = 'run_88_testing'
    # run = 'run_88'
    subrun = 'resist_530V_drift_1000V'
    feu = 5
    file_num = 0
    runs_base = '/media/dylan/data/x17/feb_beam/runs/'
    waveform_dir = 'decoded_root'
    hits_dir = 'hits_root'

    run_dir = os.path.join(runs_base, run)
    subrun_dir = os.path.join(run_dir, subrun)
    waveform_dir_path = os.path.join(subrun_dir, waveform_dir)
    hits_dir_path = os.path.join(subrun_dir, hits_dir)

    # Find waveform file in run dir. Will end with _xxx_yy.root where xxx is file number and yy is feu number
    waveform_file = None
    for f in os.listdir(waveform_dir_path):
        if f.endswith(f'_{file_num:03d}_{feu:02d}.root'):
            waveform_file = os.path.join(waveform_dir_path, f)
            break
    if waveform_file is None:
        raise ValueError(f'Waveform file for feu {feu} and file number {file_num} not found in {run_dir}.')

    hits_file = None
    for f in os.listdir(hits_dir_path):
        if f.endswith(f'_{file_num:03d}_{feu:02d}_hits.root'):
            hits_file = os.path.join(hits_dir_path, f)
            break
    if hits_file is None:
        raise ValueError(f'Hits file for feu {feu} and file number {file_num} not found in {run_dir}.')

    event_id = 2
    # channel_ids = np.arange(0, 512)
    channel_ids = np.arange(10, 20)

    plot_waveform_and_hits(waveform_file, hits_file, event_id, channel_ids, min_amp=300, plot_derivative=True)

    plt.show()

    print('donzo')


def plot_waveform_and_hits(waveform_file, hits_file, event_id, channel_ids, min_amp=None, plot_derivative=False):
    """
    waveform_file : str   path to ROOT file with the 'nt' tree (raw waveforms)
    hits_file     : str   path to ROOT file with the 'hits' tree
    event_id      : int   eventId to plot
    channel_ids    : int   channel to plot
    min_amp       : float minimum amplitude to plot (if None, plot all)
    """

    # ---------------------------
    # Read waveform tree
    # ---------------------------
    with uproot.open(waveform_file) as f:
        nt = f["nt"]
        evt_ids = nt["eventId"].array(library="np")
        ftsts = nt["ftst"].array(library="np")
        samples = nt["sample"].array(library="np")     # Jagged array
        channels = nt["channel"].array(library="np")   # Jagged array
        amplitudes = nt["amplitude"].array(library="np")

    max_ftst = np.max(ftsts)
    sample_period = 10 * (max_ftst + 1)  # in ns

    # Find index in nt corresponding to the eventId
    match = np.where(evt_ids == event_id)[0]
    if len(match) == 0:
        raise ValueError(f"Event {event_id} not found in nt tree.")
    idx = match[0]

    if isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    # Plot max amplitude per event vs eventId
    max_amplitudes = [np.max(amp) if len(amp) > 0 else 0 for amp in amplitudes]
    plt.figure(figsize=(10,6))
    plt.plot(evt_ids, max_amplitudes, marker='o', linestyle='None', markersize=3)
    plt.title("Max Amplitude per Event")
    plt.xlabel("Event ID")
    plt.ylabel("Max Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Extract this event
    ftst = ftsts[idx]
    evt_samples = samples[idx]
    evt_channels = channels[idx]
    evt_amplitudes = amplitudes[idx]

    # ---------------------------
    # Read hit tree
    # ---------------------------
    pedestals_exist = False
    with uproot.open(hits_file) as f:
        hits = f["hits"]
        hit_evt   = hits["eventId"].array(library="np")
        hit_ch    = hits["channel"].array(library="np")
        hit_samp  = hits["sample"].array(library="np")
        hit_amp   = hits["amplitude"].array(library="np")
        hit_i_max = hits["max_sample"].array(library="np")
        hit_time  = hits["time"].array(library="np")
        hit_left  = hits["left_sample"].array(library="np")
        hit_right = hits["right_sample"].array(library="np")
        hit_base  = hits["local_baseline"].array(library="np")
        hit_tot   = hits["time_over_threshold"].array(library="np")

        if 'pedestals' in f:
            pedestals_exist = True
            peds = f["pedestals"]
            ped_ch = peds["channel"].array(library="np")
            ped_base = peds["mean"].array(library="np")

    channels, time_samples, max_samples, amplitudes, samples, waveforms, tots = [], [], [], [], [], [], []
    for channel_id in channel_ids:
        # Select waveform entries where channel == channel_id
        mask = (evt_channels == channel_id)
        if not np.any(mask):
            print(f"Channel {channel_id} not found in event {event_id} of nt tree.")
            continue

        print(f'ftst: {ftst}, max_ftst: {max_ftst}')
        waveform_sample_idx = evt_samples[mask] + (ftst / (max_ftst + 1))
        # waveform_sample_idx = evt_samples[mask]
        waveform_amp = evt_amplitudes[mask]

        if min_amp is not None and np.max(waveform_amp) < min_amp:
            print(f'Skipping channel {channel_id} in event {event_id} due to max amplitude {np.max(waveform_amp)} < {min_amp}')
            continue

        # Select hits matching event and channel
        hit_mask = (hit_evt == event_id) & (hit_ch == channel_id)

        hit_sample_positions = hit_samp[hit_mask]
        hit_amplitudes = hit_amp[hit_mask]
        hit_i_maxes = hit_i_max[hit_mask]
        hit_lefts = hit_left[hit_mask]
        hit_rights = hit_right[hit_mask]
        hit_bases = hit_base[hit_mask]
        hit_tots = hit_tot[hit_mask]

        # Get pedestal for this channel
        if pedestals_exist:
            ped_mask = (ped_ch == channel_id)
            if not np.any(ped_mask):
                raise ValueError(f"Pedestal for channel {channel_id} not found in pedestals tree.")
            pedestal = ped_base[ped_mask][0]
        else:
            pedestal = 0.0

        # ---------------------------
        # Plot
        # ---------------------------
        plt.figure(figsize=(10,6))

        waveform_amp = waveform_amp - pedestal  # Subtract pedestal for plotting
        plt.plot(waveform_sample_idx, waveform_amp, label="Waveform", marker='.', linewidth=1)
        # Get minimum value of y axis of plot
        y_min = np.min(waveform_amp)

        # Overlay hits as vertical lines
        for hs, ha, hi, hl, hr, hb, ht in zip(hit_sample_positions, hit_amplitudes, hit_i_maxes, hit_lefts, hit_rights, hit_bases, hit_tots):
            print(f'Hit at sample {hs}: amplitude={ha}, i_max={hi}, left={hl}, right={hr}, base={hb}')
            plt.axvline(hs, color="red", linestyle="--", alpha=0.7, zorder=2)
            plt.axvspan(hl, hr, color="gray", alpha=0.1)
            # Make horizontal line from (hl, hb) to (hr, hb)
            plt.hlines(hb, hl - 8, hr, color="green", linestyle="-", alpha=0.7)
            # Make horizontal line at max amplitude spanning 3 samples centered at hi
            plt.hlines(ha + hb, hi-1, hi+1, color="red", linestyle="-", lw=2, alpha=0.7)
            plt.scatter(hi, ha + hb, color="red", marker='x', s=20, zorder=3)

            plt.annotate(f"{hs:.1f}", xy=(hs, y_min), xytext=(-4, 0), textcoords='offset pixels',
                         color="red", fontsize=11, rotation=90, va='center', ha='right')
            # plt.text(hs, np.min(waveform_amp), f"{ha:.1f}", color="blue", fontsize=11, rotation=90,
            #          verticalalignment="bottom", horizontalalignment="left")
            channels.append(channel_id)
            time_samples.append(hs)
            amplitudes.append(ha)
            max_samples.append(hi)
            tots.append(ht)
        samples.append(waveform_sample_idx)
        waveforms.append(waveform_amp)

        # If channel is 94, do a fit
        # if channel_id == 94:
        #     print(f'Waveform samples: {waveform_sample_idx}')
        #     print(f'Waveform amplitudes: {waveform_amp}')
        #     # Fit a parabola to three points.
        #     fit_indices = np.array([2, 3, 7, 8])
        #     x_fit = waveform_sample_idx[fit_indices]
        #     y_fit = waveform_amp[fit_indices]
        #     try:
        #         popt, pcov = cf(parabola, x_fit, y_fit)
        #         a, b, c = popt
        #         print(f'Parabola fit parameters: a={a}, b={b}, c={c}')
        #         x_fine = np.linspace(np.min(x_fit)-2, np.max(x_fit)+2, 100)
        #         y_fine = parabola(x_fine, a, b, c)
        #         plt.plot(x_fine, y_fine, color="orange", linestyle="-", linewidth=2, alpha=0.7, label="Parabola Fit")
        #     except Exception as e:
        #         print(f'Parabola fit failed: {e}')

        plt.title(f"Event {event_id}, Channel {channel_id}")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if plot_derivative:
            fig, ax = plt.subplots(figsize=(10, 6))
            # ── derivative-trigger parameters (mirror C++ defaults) ──────────────────────
            DERIV_SMOOTH_HW = 3  # box-car half-width before differencing
            DERIV_THR_SIGMA = 3.0  # derivative peak threshold  [units: noise_rms / sample]
            DERIV_MERGE_DIST = 4  # merge seeds closer than this many samples
            AMP_THR_SIGMA = 5.0  # amplitude threshold for region boundaries [same as thresholdSigma]
            MIN_WIDTH_SAMPLES = 2  # minimum region width to keep
            noise_rms = 1.0  # replace with your actual per-channel noise RMS

            plt.plot(waveform_sample_idx, waveform_amp, label="Waveform", marker='.', linewidth=1)
            # Get minimum value of y axis of plot
            y_min = np.min(waveform_amp)
            N = len(waveform_amp)
            idx = np.asarray(waveform_sample_idx, dtype=float)
            amp = np.asarray(waveform_amp, dtype=float)

            # ── Step 1: box-car smooth ────────────────────────────────────────────
            hw = DERIV_SMOOTH_HW
            smooth = np.convolve(amp, np.ones(2 * hw + 1) / (2 * hw + 1), mode='same')
            # fix edges that convolve pads with zeros → use edge-aware average instead
            for i in range(hw):
                lo = max(0, i - hw);
                hi = min(N - 1, i + hw)
                smooth[i] = amp[lo:hi + 1].mean()
            for i in range(N - hw, N):
                lo = max(0, i - hw);
                hi = min(N - 1, i + hw)
                smooth[i] = amp[lo:hi + 1].mean()

            # ── Step 2: central-difference derivative ─────────────────────────────
            deriv = np.empty(N)
            deriv[1:-1] = 0.5 * (smooth[2:] - smooth[:-2])
            deriv[0] = smooth[1] - smooth[0]
            deriv[-1] = smooth[-1] - smooth[-2]

            # ── Step 3: local maxima of deriv above threshold ─────────────────────
            deriv_thr = DERIV_THR_SIGMA * noise_rms
            amp_thr = AMP_THR_SIGMA * noise_rms

            is_local_max = (deriv[1:-1] > deriv_thr) & \
                           (deriv[1:-1] >= deriv[:-2]) & \
                           (deriv[1:-1] >= deriv[2:])
            seeds = np.where(is_local_max)[0] + 1  # +1 to correct for sliced index

            # ── Step 4: merge seeds within DERIV_MERGE_DIST ───────────────────────
            merged = []
            for s in seeds:
                if merged and (s - merged[-1]) <= DERIV_MERGE_DIST:
                    merged[-1] = s if deriv[s] > deriv[merged[-1]] else merged[-1]
                else:
                    merged.append(s)

            # ── Steps 5-6: build pulse regions ────────────────────────────────────
            pulse_regions = []  # list of (start_idx, end_idx) in sample-index space
            for k, seed in enumerate(merged):
                # left boundary: walk left until below amp threshold
                start = seed
                while start > 0 and amp[start - 1] > amp_thr:
                    start -= 1

                # right boundary: walk right until below amp threshold
                end = seed
                while end + 1 < N and amp[end + 1] > amp_thr:
                    end += 1

                # pile-up cut: clip at local minimum before next seed if it overlaps
                if k + 1 < len(merged):
                    next_seed = merged[k + 1]
                    next_start = next_seed
                    while next_start > 0 and amp[next_start - 1] > amp_thr:
                        next_start -= 1

                    if next_start <= end:
                        split_at = seed + int(np.argmin(amp[seed:next_seed]))
                        end = split_at

                if end - start + 1 >= MIN_WIDTH_SAMPLES:
                    pulse_regions.append((start, end))

            # ── Plot derivative (scaled to waveform y-range for overlay) ──────────
            deriv_scale = np.ptp(amp) / (np.ptp(deriv) + 1e-9)
            waveform_amp_deriv = deriv * deriv_scale  # scaled for display
            deriv_offset = y_min  # anchor baseline to plot bottom

            plt.plot(idx, waveform_amp_deriv + deriv_offset,
                     label=f"Deriv ×{deriv_scale:.1f} (offset to y_min)",
                     color='orange', linewidth=1, linestyle='--', alpha=0.75)

            # threshold line in derivative space
            plt.axhline(deriv_thr * deriv_scale + deriv_offset,
                        color='orange', linestyle=':', linewidth=0.8,
                        label=f"Deriv thr ({DERIV_THR_SIGMA}σ)")

            # ── Mark seeds and region boundaries ──────────────────────────────────
            if len(merged) > 0:
                plt.scatter(idx[merged], amp[merged],
                            marker='^', s=80, color='red', zorder=5,
                            label="Rising-edge seeds")

            for i, (start, end) in enumerate(pulse_regions):
                label_start = "Pulse region start" if i == 0 else None
                label_end = "Pulse region end" if i == 0 else None
                plt.axvline(idx[start], color='green', linestyle='--',
                            linewidth=1.0, alpha=0.8, label=label_start)
                plt.axvline(idx[end], color='purple', linestyle='--',
                            linewidth=1.0, alpha=0.8, label=label_end)
                # shade the pulse region
                # plt.axvspan(idx[start], idx[end], alpha=0.08, color='blue')

            plt.title(f"Event {event_id}, Channel {channel_id}")
            plt.xlabel("Sample index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

    # #For channels, find clusters based on np.diff of channels. If gap > 2, new cluster
    # clusters = []
    # if len(channels) > 0:
    #     sorted_indices = np.argsort(channels)
    #     sorted_channels = np.array(channels)[sorted_indices]
    #     cluster = [sorted_channels[0]]
    #     for i in range(1, len(sorted_channels)):
    #         if sorted_channels[i] - sorted_channels[i-1] > 2:
    #             clusters.append(cluster)
    #             cluster = [sorted_channels[i]]
    #         else:
    #             cluster.append(sorted_channels[i])
    #     clusters.append(cluster)
    #
    #     print(f'Identified channel clusters: {clusters}')
    #
    #     # For each cluster, fit a line to time_samples vs channels. Save slope and intercept
    #     slopes, intercepts, left_ch, right_ch = [], [], [], []
    #     parab_a, parab_b, parab_c = [], [], []
    #     for cluster in clusters:
    #         cluster_indices = [i for i, ch in enumerate(channels) if ch in cluster]
    #         cluster_channels = np.array(channels)[cluster_indices]
    #         cluster_time_samples = np.array(time_samples)[cluster_indices]
    #         cluster_amplitudes = np.array(amplitudes)[cluster_indices]
    #
    #         if len(cluster_channels) < 2:
    #             print(f'Not enough points to fit line for cluster {cluster}. Skipping.')
    #             continue
    #
    #         # Prepare weights: use sigma = 1/sqrt(amplitude) so larger amplitudes get larger weight
    #         amp_safe = np.maximum(cluster_amplitudes, 1e-6)
    #         sigma = 1.0 / np.sqrt(amp_safe)
    #
    #         try:
    #             popt, pcov = cf(line, cluster_channels, cluster_time_samples, sigma=sigma, absolute_sigma=False)
    #             slope, intercept = popt
    #         except Exception as e:
    #             print(f'Weighted fit failed for cluster {cluster}: {e}. Falling back to numpy.polyfit with weights.')
    #             w = np.sqrt(amp_safe)  # polyfit expects weights proportional to 1/sigma
    #             p = np.polyfit(cluster_channels, cluster_time_samples, 1, w=w)
    #             slope, intercept = p[0], p[1]
    #
    #         slopes.append(slope)
    #         intercepts.append(intercept)
    #         left_ch.append(np.min(cluster_channels))
    #         right_ch.append(np.max(cluster_channels))
    #
    #         # Now also fit a parabola to the same points
    #         try:
    #             popt_parab, pcov_parab = cf(parabola, cluster_channels, cluster_time_samples, sigma=sigma, absolute_sigma=False)
    #             a, b, c = popt_parab
    #             parab_a.append(a)
    #             parab_b.append(b)
    #             parab_c.append(c)
    #         except Exception as e:
    #             print(f'Parabola fit failed for cluster {cluster}: {e}. Skipping parabola fit.')
    #             parab_a.append(0)
    #             parab_b.append(0)
    #             parab_c.append(0)
    #
    #
    # # Plot time samples of hits vs channel
    # plt.figure(figsize=(10,6))
    # plt.scatter(channels, time_samples, marker='o', color='blue', s=np.array(amplitudes) / 20, zorder=10)
    # plt.scatter(channels, max_samples, marker='x', color='red', s=np.array(amplitudes) / 20, zorder=10)
    # # Add secondary y-axis for time in ns, just scaling by sample_period
    # ax = plt.gca()
    # secax = ax.secondary_yaxis('right', functions=(lambda x: x * sample_period, lambda x: x / sample_period))
    # secax.set_ylabel("Time (ns)")
    #
    # # Plot fitted lines
    # for m, b, lch, rch in zip(slopes, intercepts, left_ch, right_ch):
    #     x_fit = np.array([lch, rch])
    #     y_fit = line(x_fit, m, b)
    #     plt.plot(x_fit, y_fit, color='green', linestyle='-', linewidth=2, alpha=0.7)
    #     # Convert slope to time (ns) per channel
    #     slope_ns_per_ch = m * sample_period
    #     plt.text((lch + rch) / 2, line((lch + rch) / 2, m, b), f"slope = {m:.2f}, {slope_ns_per_ch:.1f}ns / strip",
    #              color='green', fontsize=10, ha='center', va='bottom', rotation=45)
    #
    # # Plot fitted parabolas
    # for a, b, c, lch, rch in zip(parab_a, parab_b, parab_c, left_ch, right_ch):
    #     x_fit = np.linspace(lch, rch, 100)
    #     y_fit = parabola(x_fit, a, b, c)
    #     plt.plot(x_fit, y_fit, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    #     plt.text((lch + rch) / 2, parabola((lch + rch) / 2, a, b, c), f"parabola fit",
    #              color='orange', fontsize=10, ha='center', va='bottom', rotation=45)
    #
    # plt.title(f"Hit Sample Positions for Event {event_id}")
    # plt.xlabel("Channel ID")
    # plt.ylabel("Sample Index")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    #
    # # Plot waveforms of all channels, stacked vertically
    # plt.figure(figsize=(10,6))
    # offset = 0
    # for ch, samp, wave in zip(channel_ids, samples, waveforms):
    #     plt.plot(samp, wave + offset, label=f"Ch {ch}")
    #     offset += np.max(wave) + 100  # add space between waveforms
    # plt.title(f"Waveforms for Event {event_id}")
    # plt.xlabel("Sample index")
    # plt.ylabel("Amplitude + offset")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()


def line(x, m, b):
    return m * x + b


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


if __name__ == '__main__':
    main()
