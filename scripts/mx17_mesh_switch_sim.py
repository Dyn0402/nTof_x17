#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03 10:19 AM 2026
Created in PyCharm
Created as nTof_x17/mx17_mesh_switch_sim.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # --- Constant System Parameters ---
    v_supply = 100.0
    c_sys = 10e-9  # 100nF Mesh + 100nF Cable
    t_trigger = 10e-6  # When discharge starts
    t_release = 30e-3  # When MOSFET turns off (10us duration for visibility)
    t_end = 1000e-3  # Extended to 100ms to see slow charging
    mosfet_r = 0.68  # ohm, internal resistance of mosfet

    flash_start, flash_end = 10e-6, 11e-6  # Start and end of gamma flash which we want to avoid

    # --- Variable Lists for Comparison ---
    r_fast_list = np.array([1, 2, 3, 10]) * 10  # Ohms (Tuning the discharge)
    r_slow_list = np.array([1e6, 2e6, 3e6]) * 10  # Ohms (5M, 10M, 20M for recharge)

    # --- Time Arrays ---
    t_fast = np.linspace(0, 40e-6, 1000)
    t_slow = np.linspace(0, t_end, 2000)

    def get_v_trace(t_array, r_tune, r_iso):
        v = np.full_like(t_array, v_supply)
        for i, time in enumerate(t_array):
            if t_trigger <= time < t_release:
                dt = time - t_trigger
                v[i] = v_supply * np.exp(-dt / (r_tune * c_sys))
            elif time >= t_release:
                v_at_release = v_supply * np.exp(-(t_release - t_trigger) / (r_tune * c_sys))
                dt = time - t_release
                v[i] = v_supply - (v_supply - v_at_release) * np.exp(-dt / (r_iso * c_sys))
        return v

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # Colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',]

    # 1. Top Plot: Discharge detail (Varying R_tune, keeping R_iso constant)
    for r_t, col in zip(r_fast_list, colors):
        r_t_total = r_t + mosfet_r
        v_trace = get_v_trace(t_fast, r_t_total, r_slow_list[1])  # Use middle R_iso
        current = v_supply / r_t_total
        ax1.plot(t_fast * 1e6, v_trace, color=col, lw=2, label=f'R_discharge = {r_t} $\Omega$, Current = {current:.2f} A')
    ax1.axvspan(flash_start * 1e6, flash_end * 1e6, color='orange', alpha=0.4, label='Gamma Flash')
    ax1.axhline(0, color='gray', linestyle='-', zorder=0)
    ax1.axhline(v_supply, color='gray', linestyle='-', zorder=0)
    ax1.set_xlim(5, 20)
    ax1.set_title("Fast Discharge -- Detector Turn On")
    ax1.set_ylabel("Mesh Voltage (V)")
    ax1.set_xlabel("Time ($\mu$s)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Annotations for ax1
    x_ann1_off = ax1.get_xlim()[0] + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.25
    x_ann1_on = ax1.get_xlim()[0] + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.42
    ax1.annotate('Detector Off', xy=(x_ann1_off, v_supply), xytext=(x_ann1_off - 2.5, v_supply - 18),
                 fontsize=9, color='dimgray', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
    ax1.annotate('Detector On', xy=(x_ann1_on, 0), xytext=(x_ann1_on + 1.5, 0 + 18),
                 fontsize=9, color='dimgray', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))

    # 2. Bottom Plot: Full Recovery (Varying R_iso, keeping R_tune constant)
    for r_i, col in zip(r_slow_list, colors):
        v_trace = get_v_trace(t_slow, r_fast_list[1], r_i)  # Use middle R_tune
        current = v_supply / r_i * 1e6
        ax2.plot(t_slow * 1e3, v_trace, color=col, lw=2, label=f'R_charge = {r_i / 1e6:.0f} M$\Omega$, Current = {current:.2f} $\mu$A')

    ax2.axhline(0, color='gray', linestyle='-', zorder=0)
    ax2.axhline(v_supply, color='gray', linestyle='-', zorder=0)
    ax2.set_title("Slow Recharge -- Detector Turn Off")
    ax2.set_ylabel("Mesh Voltage (V)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Annotations for ax2
    x_ann2_off = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.8
    x_ann2_on = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.055
    ax2.annotate('Detector Off', xy=(x_ann2_off, v_supply), xytext=(x_ann2_off + 30, v_supply - 18),
                 fontsize=9, color='dimgray', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
    ax2.annotate('Detector On', xy=(x_ann2_on, 0), xytext=(x_ann2_on + 200, 0 + 18),
                 fontsize=9, color='dimgray', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))

    # Parameters text box
    info_text = (f"V_bias: {v_supply}V\n"
                 f"C_total: {c_sys * 1e9:.0f} nF")
    fig.text(0.5, 0.2, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), va='center')

    fig.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
