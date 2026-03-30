#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on March 26 3:26 PM 2026
Created in PyCharm
Created as nTof_x17/analyze_ntof_daq_mm_flashes.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from scipy.optimize import curve_fit


MV_PER_ADC = 0.0016  # Conversion from ADC to mv
NS_PER_SAMPLE = 1  # Conversion from sample to ns
PULSE_THRESHOLD_MV = -1  # Minimum amplitude to consider a real pulse (mV)
CORRECTION_THRESHOLD_MV = 1.5  # Residual amplitude above which correction is problematic (mV)

RUN_START = pd.Timestamp('2026-02-28 15:02:06')
RUN_STOP = pd.Timestamp('2026-02-28 16:34:24')
HV_RUN_DIR = '/media/dylan/data/x17/feb_beam/runs/run_131'
HV_CHANNEL = '2:0'
DRIFT_CHANNEL = '5:0'


def main():
    run_dir = '/home/dylan/x17/feb_beam/ntof_daq_data/run223049'

    dedicated_files = []
    parasitic_files = []

    for segment in sorted(os.listdir(run_dir)):
        segment_dir = f'{run_dir}/{segment}'
        for file_name in sorted(os.listdir(segment_dir)):
            full_path = f'{segment_dir}/{file_name}'
            if file_name.startswith('D_'):
                dedicated_files.append(full_path)
            elif file_name.startswith('P_'):
                parasitic_files.append(full_path)

    print(f'Dedicated files: {len(dedicated_files)}')
    print(f'Parasitic files: {len(parasitic_files)}')

    all_files = dedicated_files + parasitic_files

    hv_df, subrun_starts = load_hv_data(HV_RUN_DIR, HV_CHANNEL, DRIFT_CHANNEL)
    plot_superposition_subrun(dedicated_files, parasitic_files, all_files, hv_df,
                              target_resist_hv=500, target_drift=1000)
    plot_superposition_subrun(dedicated_files, parasitic_files, all_files, hv_df,
                              target_resist_hv=530, target_drift=1000)
    plot_pulse_correction(dedicated_files, parasitic_files, all_files, hv_df,
                          target_resist_hv=530, target_drift=1000)

    plt.show()

    fit_single_pulse(dedicated_files)

    ded_results = fit_all_pulses(dedicated_files, label='Dedicated')
    par_results = fit_all_pulses(parasitic_files, label='Parasitic')
    plot_fit_distributions(ded_results, par_results)
    plot_amplitude_vs_trigger(ded_results, par_results)

    plot_hv_current_amplitude_vs_time(ded_results, par_results, hv_df, subrun_starts, all_files)
    plot_amplitude_and_current_vs_hv(ded_results, par_results, hv_df, all_files)

    debug_failed_fits(dedicated_files, label='Dedicated', n_show=6)

    plt.show()
    print('donzo')


def plot_superposition_subrun(ded_files, par_files, all_files, hv_df,
                              target_resist_hv, target_drift, baseline_samples=100, hv_tol=1, buffer_s=60):
    """Plot superposition of waveforms from a specific HV sub-run, ded and par on same plot."""
    ded_sub, trig_lo, trig_hi = _get_subrun_files(ded_files, all_files, hv_df,
                                                   target_resist_hv, target_drift, hv_tol, buffer_s)
    par_sub, _, _ = _get_subrun_files(par_files, all_files, hv_df,
                                      target_resist_hv, target_drift, hv_tol, buffer_s)
    print(f'Sub-run resist={target_resist_hv}V drift={target_drift}V: '
          f'trig range [{trig_lo:.0f}, {trig_hi:.0f}]')
    print(f'  Dedicated: {len(ded_sub)} files, Parasitic: {len(par_sub)} files')

    fig, ax = plt.subplots(figsize=(12, 5))
    for file_list, color, label in [(ded_sub, 'steelblue', 'Dedicated'),
                                     (par_sub, 'tomato', 'Parasitic')]:
        for path in file_list:
            t, v = load_pulse(path, baseline_samples)
            if np.min(v) > PULSE_THRESHOLD_MV:
                continue
            ax.plot(t, v, color=color, alpha=0.4, linewidth=0.7)
    # Add legend proxy
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color='steelblue', label='Dedicated'),
                        Line2D([0], [0], color='tomato', label='Parasitic')], fontsize=8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f'Waveforms — resist={target_resist_hv}V, drift={target_drift}V')
    fig.tight_layout()


def _get_subrun_files(file_list, all_files, hv_df, target_resist_hv, target_drift,
                      hv_tol, buffer_s):
    """Return files whose trigger numbers fall within the specified HV sub-run window."""
    all_trigs = [int(os.path.basename(p).split('_')[4]) for p in all_files]
    trig_min, trig_max = min(all_trigs), max(all_trigs)
    run_duration = (RUN_STOP - RUN_START).total_seconds()

    mask = (
        (hv_df[f'{HV_CHANNEL} vmon'] - target_resist_hv).abs() < hv_tol) & (
        (hv_df[f'{DRIFT_CHANNEL} vmon'] - target_drift).abs() < hv_tol
    )
    t_start = hv_df.loc[mask, 'timestamp'].min() + pd.Timedelta(seconds=buffer_s)
    t_end = hv_df.loc[mask, 'timestamp'].max() - pd.Timedelta(seconds=buffer_s)

    def time_to_trig(t):
        frac = (t - RUN_START).total_seconds() / run_duration
        return trig_min + frac * (trig_max - trig_min)

    trig_lo, trig_hi = time_to_trig(t_start), time_to_trig(t_end)
    return [p for p in file_list
            if trig_lo <= int(os.path.basename(p).split('_')[4]) <= trig_hi], trig_lo, trig_hi


def plot_pulse_correction(ded_files, par_files, all_files, hv_df,
                          target_resist_hv=530, target_drift=1000,
                          baseline_samples=100, hv_tol=1, buffer_s=60):
    """Fit average parasitic pulse, subtract from all pulses, and plot residuals."""
    ded_sub, _, _ = _get_subrun_files(ded_files, all_files, hv_df,
                                      target_resist_hv, target_drift, hv_tol, buffer_s)
    par_sub, _, _ = _get_subrun_files(par_files, all_files, hv_df,
                                      target_resist_hv, target_drift, hv_tol, buffer_s)

    # Average parasitic waveforms
    par_waveforms = []
    t_common = None
    for path in par_sub:
        t, v = load_pulse(path, baseline_samples)
        if np.min(v) > PULSE_THRESHOLD_MV:
            continue
        if t_common is None:
            t_common = t
        par_waveforms.append(v)

    if not par_waveforms:
        print('No parasitic pulses found for correction.')
        return

    avg_waveform = np.mean(par_waveforms, axis=0)

    # Fit average waveform
    i_min = np.argmin(avg_waveform)
    try:
        popt, _ = curve_fit(landau_moyal, t_common, avg_waveform,
                            p0=[-np.min(avg_waveform), t_common[i_min], 20.0],
                            bounds=([0, t_common[0], 1], [np.inf, t_common[-1], 500]))
        avg_landau = landau_moyal(t_common, *popt)
        print(f'Average Landau: amp={popt[0]:.2f} mV, mu={popt[1]:.1f} ns, sigma={popt[2]:.1f} ns')
    except RuntimeError as e:
        print(f'Average Landau fit failed: {e}, using raw average')
        avg_landau = avg_waveform

    avg_landau *= 2.8

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: average waveform and fit
    axes[0].plot(t_common, avg_waveform, color='gray', linewidth=1, label='Average parasitic')
    axes[0].plot(t_common, avg_landau, color='black', linewidth=1.5, linestyle='--',
                 label='Landau fit')
    axes[0].axhline(0, color='gray', linewidth=0.6, zorder=0)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].set_title(f'Average Parasitic Pulse — resist={target_resist_hv}V, drift={target_drift}V')
    axes[0].legend(fontsize=8)

    # Bottom panel: residuals after subtracting average Landau
    axes[1].axhline(0, color='gray', linewidth=0.6, zorder=0)
    axes[1].axhline(CORRECTION_THRESHOLD_MV, color='red', linewidth=1.0, linestyle='--',
                    zorder=0, label=f'±{CORRECTION_THRESHOLD_MV} mV')
    axes[1].axhline(-CORRECTION_THRESHOLD_MV, color='red', linewidth=1.0, linestyle='--', zorder=0)

    for file_list, color in [(ded_sub, 'steelblue'), (par_sub, 'tomato')]:
        for path in file_list:
            t, v = load_pulse(path, baseline_samples)
            if np.min(v) > PULSE_THRESHOLD_MV:
                continue
            axes[1].plot(t_common, v - avg_landau, color=color, alpha=0.3, linewidth=0.7)

    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([0], [0], color='steelblue', label='Dedicated'),
        Line2D([0], [0], color='tomato', label='Parasitic'),
        Line2D([0], [0], color='red', linestyle='--', label=f'±{CORRECTION_THRESHOLD_MV} mV'),
    ], fontsize=8)
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Residual (mV)')
    axes[1].set_title('Residual after Landau Correction')

    fig.tight_layout()


def plot_superposition(file_list, title='Waveforms', baseline_samples=100):
    fig, ax = plt.subplots(figsize=(12, 5))
    for path in file_list:
        with uproot.open(path) as f:
            key = list(f.keys())[0]
            h = f[key]
            edges = h.axis(0).edges()
            centers = 0.5 * (edges[:-1] + edges[1:])
            values = h.values()
            baseline = np.mean(values[:baseline_samples])
            t = centers[:-1] * NS_PER_SAMPLE
            v = (values[:-1] - baseline) * MV_PER_ADC
            if np.min(v) > PULSE_THRESHOLD_MV:
                continue
            ax.plot(t, v, color='steelblue', alpha=0.4, linewidth=0.7)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title)
    fig.tight_layout()


def landau_moyal(x, amp, mu, sigma):
    """Inverted Landau pulse via Moyal approximation.
    amp is the peak amplitude in mV (positive), pulse goes negative."""
    z = (x - mu) / sigma
    return -amp * np.exp(-0.5 * (z + np.exp(-z) - 1))


def load_pulse(path, baseline_samples=100):
    with uproot.open(path) as f:
        key = list(f.keys())[0]
        h = f[key]
        edges = h.axis(0).edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = h.values()
    baseline = np.mean(values[:baseline_samples])
    t = centers[:-1] * NS_PER_SAMPLE
    v = (values[:-1] - baseline) * MV_PER_ADC
    return t, v


def fit_single_pulse(file_list, baseline_samples=100):
    # Find first pulse that passes the threshold
    t, v = None, None
    for path in file_list:
        t_try, v_try = load_pulse(path, baseline_samples)
        if np.min(v_try) <= PULSE_THRESHOLD_MV:
            t, v = t_try, v_try
            print(f'Fitting pulse: {path}')
            break

    if t is None:
        print('No pulse found exceeding threshold.')
        return

    i_min = np.argmin(v)
    amp0 = -np.min(v)
    mu0 = t[i_min]
    sigma0 = 20.0  # ns, rough width guess

    try:
        popt, pcov = curve_fit(landau_moyal, t, v, p0=[amp0, mu0, sigma0],
                               bounds=([0, t[0], 1], [np.inf, t[-1], 500]))
        perr = np.sqrt(np.diag(pcov))
        print(f'  amp   = {popt[0]:.3f} +/- {perr[0]:.3f} mV')
        print(f'  mu    = {popt[1]:.3f} +/- {perr[1]:.3f} ns')
        print(f'  sigma = {popt[2]:.3f} +/- {perr[2]:.3f} ns')
    except RuntimeError as e:
        print(f'Fit failed: {e}')
        popt = None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, v, color='steelblue', linewidth=0.8, label='Data')
    if popt is not None:
        t_fine = np.linspace(t[0], t[-1], 5000)
        ax.plot(t_fine, landau_moyal(t_fine, *popt), color='tomato', linewidth=1.5,
                label=f'Landau fit\n$\\mu$={popt[1]:.1f} ns, $\\sigma$={popt[2]:.1f} ns')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title('Single Pulse Landau Fit')
    ax.legend()
    fig.tight_layout()


def fit_all_pulses(file_list, label='', baseline_samples=100):
    """Fit all pulses passing the threshold. Returns results meta dict."""
    n_total = len(file_list)
    n_fit_failed = 0
    results = []

    for i, path in enumerate(file_list):
        trig_num = int(os.path.basename(path).split('_')[4])
        t, v = load_pulse(path, baseline_samples)
        min_v = np.min(v)
        if min_v > PULSE_THRESHOLD_MV:
            continue
        i_min = np.argmin(v)
        try:
            popt, _ = curve_fit(landau_moyal, t, v, p0=[-min_v, t[i_min], 20.0],
                                bounds=([0, t[0], 1], [np.inf, t[-1], 500]))
            results.append({'amp': popt[0], 'mu': popt[1], 'sigma': popt[2], 'trig': trig_num})
        except RuntimeError:
            n_fit_failed += 1
        if (i + 1) % 100 == 0:
            print(f'  [{label}] {i + 1}/{n_total} processed, {len(results)} fits successful')

    print(f'\n[{label}] Summary:')
    print(f'  Total files   : {n_total}')
    print(f'  Fit succeeded : {len(results)} ({100 * len(results) / max(n_total, 1):.1f}%)')
    print(f'  Fit failed    : {n_fit_failed} ({100 * n_fit_failed / max(n_total, 1):.1f}%)')

    return {'results': results, 'n_total': n_total, 'n_fit_failed': n_fit_failed, 'label': label}


def plot_fit_distributions(ded_meta, par_meta):
    ded_results = ded_meta['results']
    par_results = par_meta['results']

    def make_label(meta, n):
        return f"{meta['label']} (n={n})"

    ded_amp = [r['amp'] for r in ded_results]
    par_amp = [r['amp'] for r in par_results]
    ded_sigma = [r['sigma'] for r in ded_results]
    par_sigma = [r['sigma'] for r in par_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    amp_bins = np.linspace(0, max(max(ded_amp, default=1), max(par_amp, default=1)), 30)
    axes[0].hist(ded_amp, bins=amp_bins, alpha=0.6, color='steelblue',
                 label=make_label(ded_meta, len(ded_amp)))
    axes[0].hist(par_amp, bins=amp_bins, alpha=0.6, color='tomato',
                 label=make_label(par_meta, len(par_amp)))
    axes[0].set_xlabel('Peak Amplitude (mV)')
    axes[0].set_ylabel('Counts')
    axes[0].set_title('Landau Amplitude Distribution')
    axes[0].legend(fontsize=8)

    sig_bins = np.linspace(0, max(max(ded_sigma, default=1), max(par_sigma, default=1)), 30)
    axes[1].hist(ded_sigma, bins=sig_bins, alpha=0.6, color='steelblue',
                 label=make_label(ded_meta, len(ded_sigma)))
    axes[1].hist(par_sigma, bins=sig_bins, alpha=0.6, color='tomato',
                 label=make_label(par_meta, len(par_sigma)))
    axes[1].set_xlabel('Sigma (ns)')
    axes[1].set_ylabel('Counts')
    axes[1].set_title('Landau Sigma Distribution')
    axes[1].legend(fontsize=8)

    fig.tight_layout()


def plot_amplitude_vs_trigger(ded_meta, par_meta):
    fig, ax = plt.subplots(figsize=(14, 5))

    for meta, color in [(ded_meta, 'steelblue'), (par_meta, 'tomato')]:
        trigs = [r['trig'] for r in meta['results']]
        amps = [r['amp'] for r in meta['results']]
        ax.scatter(trigs, amps, color=color, s=8, alpha=0.6, label=meta['label'])

    ax.set_xlabel('Trigger Number')
    ax.set_ylabel('Peak Amplitude (mV)')
    ax.set_title('Amplitude vs Trigger Number')
    ax.legend(fontsize=8)
    fig.tight_layout()


def load_hv_data(hv_run_dir, hv_channel, drift_channel):
    """Load and concatenate HV monitor CSVs from all sub-runs, sorted by time.
    Returns (dataframe, list of sub-run start timestamps)."""
    dfs = []
    subrun_starts = []
    for sub_run in sorted(os.listdir(hv_run_dir)):
        sub_run_path = f'{hv_run_dir}/{sub_run}'
        if not os.path.isdir(sub_run_path):
            continue
        hv_file = f'{sub_run_path}/hv_monitor.csv'
        if not os.path.exists(hv_file):
            continue
        df = pd.read_csv(hv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        subrun_starts.append(df['timestamp'].min())
        dfs.append(df[['timestamp',
                        f'{hv_channel} vmon', f'{hv_channel} imon',
                        f'{drift_channel} vmon', f'{drift_channel} imon']])
    if not dfs:
        return None, []
    combined = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)
    return combined, sorted(subrun_starts)


def plot_hv_current_amplitude_vs_time(ded_meta, par_meta, hv_df, subrun_starts, all_files):
    """Plot HV+current (shared axis) and Landau amplitude on a shared time axis.
    Trigger numbers are linearly mapped to timestamps between RUN_START and RUN_STOP."""
    all_trigs = [int(os.path.basename(p).split('_')[4]) for p in all_files]
    trig_min, trig_max = min(all_trigs), max(all_trigs)
    run_duration = (RUN_STOP - RUN_START).total_seconds()

    def trig_to_time(trig):
        frac = (trig - trig_min) / (trig_max - trig_min)
        return RUN_START + pd.Timedelta(seconds=frac * run_duration)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax_hv, ax_amp = axes

    # HV and drift on left y-axis, current on right y-axis
    ax_cur = ax_hv.twinx()
    if hv_df is not None:
        ax_hv.plot(hv_df['timestamp'], hv_df[f'{HV_CHANNEL} vmon'],
                   color='navy', linewidth=0.8, label='HV (resist)')
        ax_hv.plot(hv_df['timestamp'], hv_df[f'{DRIFT_CHANNEL} vmon'],
                   color='steelblue', linewidth=0.8, linestyle='--', label='HV (drift)')
        ax_cur.plot(hv_df['timestamp'], hv_df[f'{HV_CHANNEL} imon'],
                    color='darkred', linewidth=0.8, label='Current (resist)')
        ax_cur.plot(hv_df['timestamp'], hv_df[f'{DRIFT_CHANNEL} imon'],
                    color='tomato', linewidth=0.8, linestyle='--', label='Current (drift)')
    ax_hv.set_ylabel('HV (V)')
    ax_cur.set_ylabel('Current (A)')
    ax_hv.axhline(0, color='gray', linewidth=0.6, zorder=0)

    lines_hv, labels_hv = ax_hv.get_legend_handles_labels()
    lines_cur, labels_cur = ax_cur.get_legend_handles_labels()
    ax_hv.legend(lines_hv + lines_cur, labels_hv + labels_cur, fontsize=8)

    # Amplitude panel
    ax_amp.axhline(0, color='gray', linewidth=0.6, zorder=0)
    for meta, color in [(ded_meta, 'steelblue'), (par_meta, 'tomato')]:
        times = [trig_to_time(r['trig']) for r in meta['results']]
        amps = [r['amp'] for r in meta['results']]
        ax_amp.scatter(times, amps, color=color, s=6, alpha=0.6, label=meta['label'], zorder=3)
    ax_amp.set_ylabel('Peak Amplitude (mV)')
    ax_amp.set_xlabel('Time')
    ax_amp.legend(fontsize=8)

    # Sub-run boundary lines on all axes
    for t_start in subrun_starts:
        for ax in [ax_hv, ax_cur, ax_amp]:
            ax.axvline(t_start, color='gray', linewidth=0.6, linestyle='--', zorder=0)

    fig.suptitle('HV, Current, and Pulse Amplitude vs Time')
    fig.tight_layout()


def plot_amplitude_and_current_vs_hv(ded_meta, par_meta, hv_df, all_files):
    """Plot mean amplitude vs HV (ded/par) and mean current vs HV, separated by drift voltage."""
    all_trigs = [int(os.path.basename(p).split('_')[4]) for p in all_files]
    trig_min, trig_max = min(all_trigs), max(all_trigs)
    run_duration = (RUN_STOP - RUN_START).total_seconds()

    # Build lookup with explicit column assignment to avoid rename ambiguity
    hv_lookup = pd.DataFrame({
        'timestamp': hv_df['timestamp'],
        'hv': hv_df[f'{HV_CHANNEL} vmon'],
        'current': hv_df[f'{HV_CHANNEL} imon'],
        'drift': hv_df[f'{DRIFT_CHANNEL} vmon'],
    }).sort_values('timestamp').reset_index(drop=True)
    hv_lookup['hv_group'] = hv_lookup['hv'].round()
    hv_lookup['drift_group'] = hv_lookup['drift'].round()

    drift_counts = hv_lookup['drift_group'].value_counts()
    all_drift_vals = sorted(v for v in drift_counts.index if v >= 500 and drift_counts[v] > 10)
    cmap = plt.get_cmap('tab10')
    drift_colors = {d: cmap(i) for i, d in enumerate(all_drift_vals)}

    def build_pulse_df(meta):
        rows = []
        for r in meta['results']:
            frac = (r['trig'] - trig_min) / (trig_max - trig_min)
            t = RUN_START + pd.Timedelta(seconds=frac * run_duration)
            rows.append({'timestamp': t, 'amp': r['amp']})
        df = pd.DataFrame(rows).sort_values('timestamp')
        merged = pd.merge_asof(df, hv_lookup, on='timestamp', direction='nearest')
        return merged

    ded_df = build_pulse_df(ded_meta)
    par_df = build_pulse_df(par_meta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Amplitude vs HV: only drift=1000V, color = ded/par
    high_drift = max(all_drift_vals)
    for df, color, marker, pulse_label in [(ded_df, 'steelblue', 'o', ded_meta['label']),
                                            (par_df, 'tomato', 's', par_meta['label'])]:
        grp = df[df['drift_group'].isin(all_drift_vals) & (df['drift_group'] == high_drift)]
        grouped = grp.groupby('hv_group')['amp'].agg(['mean', 'std', 'count']).sort_index()
        grouped = grouped[grouped['count'] >= 10]
        axes[0].errorbar(grouped.index.values, grouped['mean'].values,
                         yerr=grouped['std'].values, fmt=f'{marker}-', color=color,
                         capsize=4, label=f'{pulse_label}')

    axes[0].axhline(0, color='gray', linewidth=0.6, zorder=0)
    axes[0].set_xlabel('Resist HV (V)')
    axes[0].set_ylabel('Mean Peak Amplitude (mV)')
    axes[0].set_title(f'Mean Amplitude vs Resist HV (drift={high_drift:.0f}V)')
    axes[0].legend(fontsize=8)

    # Current vs HV: one series per drift voltage, x = resist HV
    for drift_v, grp in hv_lookup[hv_lookup['drift_group'].isin(all_drift_vals)].groupby('drift_group'):
        grouped_cur = grp.groupby('hv_group')['current'].agg(['mean', 'std', 'count']).sort_index()
        grouped_cur = grouped_cur[grouped_cur['count'] >= 10]
        axes[1].errorbar(grouped_cur.index.values, grouped_cur['mean'].values,
                         yerr=grouped_cur['std'].values, fmt='o-',
                         color=drift_colors[drift_v], capsize=4,
                         label=f'drift={drift_v:.0f}V')

    axes[1].axhline(0, color='gray', linewidth=0.6, zorder=0)
    axes[1].set_xlabel('Resist HV (V)')
    axes[1].set_ylabel('Mean Current (A)')
    axes[1].set_title('Mean Current vs Resist HV')
    axes[1].legend(fontsize=8)

    fig.tight_layout()


def debug_failed_fits(file_list, label='', n_show=6, baseline_samples=100):
    """Plot a grid of failed fits to diagnose what's going wrong."""
    failed = []
    for path in file_list:
        if len(failed) >= n_show:
            break
        t, v = load_pulse(path, baseline_samples)
        if np.min(v) > PULSE_THRESHOLD_MV:
            continue
        i_min = np.argmin(v)
        try:
            curve_fit(landau_moyal, t, v, p0=[-np.min(v), t[i_min], 20.0],
                      bounds=([0, t[0], 1], [np.inf, t[-1], 500]))
        except RuntimeError:
            failed.append((t, v))

    if not failed:
        print(f'[{label}] No failed fits found in first files checked.')
        return

    ncols = 3
    nrows = int(np.ceil(len(failed) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (t, v) in zip(axes, failed):
        ax.plot(t, v, color='steelblue', linewidth=0.8)
        ax.axhline(PULSE_THRESHOLD_MV, color='gray', linestyle='--', linewidth=0.8, label=f'{PULSE_THRESHOLD_MV} mV')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude (mV)')

    for ax in axes[len(failed):]:
        ax.set_visible(False)

    fig.suptitle(f'{label} - Failed Fit Examples')
    fig.tight_layout()


if __name__ == '__main__':
    main()
