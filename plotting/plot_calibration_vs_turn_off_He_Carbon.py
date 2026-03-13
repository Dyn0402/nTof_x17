#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 12 6:58 PM 2026
Created in PyCharm
Created as nTof_x17/plot_calibration_vs_turn_off.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import curve_fit as cf


def main():
    plot_data_dir = '/media/dylan/data/x17/feb_beam/Analysis/Plot_Data/helium_comparison/'
    plot_title = 'Helium/Ethane DREAM Saturation Compared to Efficiency Curves with Carbon Target'
    plot_files = {
        'run_33': {
            'file_name': 'run_33_-_30mm_-_Aluminum_Frame.csv',
            'x_col': 'hv',
            'y_col': 'hits_per_event_1.50_5.40us',
            'norm': False,
            'label': '30 mm Drift Gap, Al Frame',
            'vline_label': '30 mm Drift Gap Turn Off at',
            'vline_label_y': 0.5,
        },
        'run_141': {
            'file_name': 'run_141_-_11mm_-_Carbon_Frame.csv',
            'x_col': 'hv',
            'y_col': 'hits_per_event_1.50_5.40us',
            'norm': False,
            'label': '11 mm Drift Gap, C Frame',
            'vline_label': '11 mm Drift Gap Turn Off at',
            'vline_label_y': 0.5,
        },
        'run_84': {
            'file_name': 'run_84_-_6mm_-_Aluminum_Frame.csv',
            'x_col': 'hv',
            'y_col': 'hits_per_event_1.50_5.40us',
            'norm': False,
            'label': '6 mm Drift Gap, Al Frame',
            'vline_label': '6 mm Drift Gap Turn Off at',
            'vline_label_y': 0.5,
        },
        'run_139': {
            'file_name': 'run_139_-_6mm_-_Carbon_Frame.csv',
            'x_col': 'hv',
            'y_col': 'hits_per_event_1.50_5.40us',
            'norm': False,
            'label': '6 mm Drift Gap, C Frame',
            'vline_label': '6 mm Drift Gap Turn Off at',
            'vline_label_y': 0.5,
        },
        'run_94': {
            'file_name': 'sipm_trig_calibration_run_94.csv',
            'x_col': 'HV',
            'y_col': 'Rate',
            'norm': True,
            'label': 'Beam Scintillator Calib. (Norm.)',
            'vline_label': 'Scintillator Max Efficiency at',
            'vline_label_y': 0.15,
        },
        'run_98': {
            'file_name': 'cs137_calibration_run_98.csv',
            'x_col': 'HV',
            'y_col': 'Filtered Rate',
            'norm': True,
            'label': 'Cs-137 Calib.',
            'vline_label': 'Cs-137 Max Efficiency at',
            'vline_label_y': 0.15,
        }
    }

    # Color palette: distinct blues/greens for beam data, warm tones for calibrations
    beam_colors = ['#2196F3', '#4CAF50', '#9C27B0', '#FF5722']
    calib_colors = ['#F44336', '#FF9800']

    beam_runs = [k for k, v in plot_files.items() if not v['norm']]
    calib_runs = [k for k, v in plot_files.items() if v['norm']]
    color_map = {run: beam_colors[i] for i, run in enumerate(beam_runs)}
    color_map.update({run: calib_colors[i] for i, run in enumerate(calib_runs)})

    # Load all data upfront
    data_cache = {}
    for run, run_dict in plot_files.items():
        data_cache[run] = pd.read_csv(os.path.join(plot_data_dir, run_dict['file_name']))

    # Max of calib runs (normalize calib curves only to each other)
    max_calib_val = max(
        data_cache[run][plot_files[run]['y_col']].max() for run in calib_runs
    )

    # ── Style setup ────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax2 = ax.twinx()

    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    for spine in ['top']:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # ── Grid ───────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.grid(which='major', axis='x', color='#CCCCCC', linewidth=0.9, zorder=0)
    ax.grid(which='minor', axis='x', color='#E5E5E5', linewidth=0.5, zorder=0)
    ax.grid(which='major', axis='y', color='#DDDDDD', linewidth=0.6, linestyle='--', zorder=0)

    # ── Plot lines + vertical max lines ───────────────────────────────────────
    # We need y-limits after plotting to place text sensibly; collect vline info first
    vline_info = []  # (target_ax, x_max, y_text_frac, text, color)

    for run, run_dict in plot_files.items():
        data = data_cache[run]
        x_vals = data[run_dict['x_col']]
        y_raw = data[run_dict['y_col']]
        is_calib = run_dict['norm']
        color = color_map[run]

        if is_calib:
            y_vals = (y_raw - y_raw.min()) * max_calib_val / (y_raw.max() - y_raw.min())
            target_ax = ax2
        else:
            y_vals = y_raw
            target_ax = ax

        target_ax.plot(
            x_vals, y_vals,
            label=run_dict['label'],
            color=color,
            linewidth=2.2,
            linestyle='--' if is_calib else '-',
            alpha=0.85 if is_calib else 1.0,
            zorder=3,
        )

        idx_max = y_vals.idxmax()
        x_max = float(x_vals[idx_max])
        vline_text = f'{run_dict["vline_label"]} {int(round(x_max))} V'
        vline_info.append((target_ax, x_max, vline_text, color, run_dict['vline_label_y']))

    ax.axhline(0, color='#555555', linewidth=1.0, zorder=2)

    # ── Draw vertical lines with text (after autoscale) ────────────────────────
    fig.canvas.draw()  # force autoscale so get_ylim() is accurate

    for target_ax, x_max, vline_text, color, vline_y in vline_info:
        y_lo, y_hi = target_ax.get_ylim()
        y_span = y_hi - y_lo
        # Place text starting at 10% from the bottom
        y_text = y_lo + vline_y * y_span

        target_ax.axvline(x=x_max, color=color, linewidth=1.5, linestyle=':', alpha=0.75, zorder=2)
        target_ax.text(
            x_max + 1.2, y_text,
            vline_text,
            color=color,
            fontsize=7.5,
            rotation=90,
            va='bottom',
            ha='left',
            alpha=0.95,
            zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.95, pad=0.1, boxstyle='round,pad=0.1')
        )

    # ── Axes labels & title ────────────────────────────────────────────────────
    ax.set_xlabel('Resist HV (V)', fontsize=13, labelpad=8)
    ax.set_ylabel('Hits per Event', fontsize=13, labelpad=8)
    ax2.set_ylabel('Normalized Event Rate [Hz]', fontsize=13, labelpad=8, color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax.set_title(plot_title, fontsize=15, fontweight='bold', pad=14)

    # ── Combined legend from both axes ─────────────────────────────────────────
    beam_handles, beam_labels_list = ax.get_legend_handles_labels()
    calib_handles, calib_labels_list = ax2.get_legend_handles_labels()

    from matplotlib.lines import Line2D
    sep = Line2D([], [], color='none')

    legend = ax.legend(
        handles=[sep] + beam_handles + [sep] + calib_handles,
        labels=['Beam Runs'] + beam_labels_list + ['Calibrations'] + calib_labels_list,
        fontsize=9.5,
        framealpha=0.9,
        edgecolor='#CCCCCC',
        loc='upper left',
        handlelength=2.2,
    )
    for text in legend.get_texts():
        if text.get_text() in ('Beam Runs', 'Calibrations'):
            text.set_fontweight('bold')
            text.set_fontsize(9)
            text.set_color('#555555')

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.set_xlim(361, 549)

    ax.annotate(r'Hits in 1.5-5.4$\mu$s window', xy=(365, 150), xytext=(365, 150), xycoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='wheat', ec='gray', alpha=0.65),
                zorder=5, )

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, left=0.05, right=0.95, bottom=0.1)

    drift_gaps = [30, 11, 6]
    turn_off_hvs = [430, 470, 500]
    efficiency_hvs = [535, 530]
    efficiency_labels = [r'$e^-$ Efficient (Cs-137)', r'$e^-$ Efficient (Scintillator)']
    efficiency_colors = ['blue', 'orange']

    # Fit turn_off_hvs vs drift_gaps with exponential
    # func = exp
    # p0 = [500, 0.11, 400]
    func = inv_x
    p0 = [200, 1, 400]
    popt, pcov = cf(func, drift_gaps, turn_off_hvs, p0=p0)

    func2 = exp
    p02 = [500, 0.11, 400]
    popt2, pcov2 = cf(func2, drift_gaps, turn_off_hvs, p0=p02)

    drift_gaps_fit = np.linspace(0, 100, 1000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(drift_gaps, turn_off_hvs, color='green', marker='o', label='Turn Off HV', zorder=10, s=100)
    for efficiency_hv, label, color in zip(efficiency_hvs, efficiency_labels, efficiency_colors):
        ax.axhline(efficiency_hv, linestyle='--', color=color, alpha=0.7, label=label, zorder=0)
    # ax.plot(drift_gaps_fit, func(drift_gaps_fit, *p0), color='gray', label='Guess')
    fit_label = rf'Fit: ${popt[0]:0.0f} / x^{{{popt[1]:0.1f}}} + {popt[2]:0.0f}$'
    ax.plot(drift_gaps_fit, func(drift_gaps_fit, *popt), color='red', lw=2.5, label=fit_label, zorder=3)
    fit_label2 = rf'Fit: ${popt2[0]:0.0f} * e^{{-{popt2[1]:0.1f} x}} + {popt2[2]:0.0f}$'
    ax.plot(drift_gaps_fit, func2(drift_gaps_fit, *popt2), color='salmon', lw=1.5, label=fit_label2, zorder=2)
    ax.axvline(x=3, color='purple', linestyle='-', alpha=1.0, lw=2, label='3 mm Drift Gap', zorder=1)
    ax.set_xlim(left=0, right=35)
    ax.set_ylim(bottom=420, top=590)
    ax.set_xlabel('Drift Gap (mm)')
    ax.set_ylabel('Resist HV (V)')
    ax.set_title('Turn Off HV vs. Drift Gap in Helium/Ethane Gas with Empty Carbon Capsule', fontsize=15,
                 fontweight='bold', pad=14)
    ax.legend(bbox_to_anchor=(1.0, 0.6), loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
    fig.tight_layout()

    plt.show()

    print('donzo')


def exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def inv_x(x, a, b, c):
    return a / x**b + c


if __name__ == '__main__':
    main()
