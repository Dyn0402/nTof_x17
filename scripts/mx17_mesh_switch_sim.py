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
    V_supply = 100.0
    C_sys = 500e-12  # 100pF Mesh + 100pF Cable
    t_trigger = 10e-6  # When discharge starts
    t_release = 20e-6  # When MOSFET turns off (10us duration for visibility)
    t_end = 20e-3  # Extended to 50ms to see slow charging

    flash_start, flash_end = 10e-6, 11e-6  # Start and end of gamma flash which we want to avoid

    # --- Variable Lists for Comparison ---
    R_tune_list = [100, 500, 1000]  # Ohms (Tuning the discharge)
    R_iso_list = [5e6, 10e6, 20e6]  # Ohms (5M, 10M, 20M for recharge)

    # --- Time Arrays ---
    t_fast = np.linspace(0, 40e-6, 1000)
    t_slow = np.linspace(0, t_end, 2000)

    def get_v_trace(t_array, r_tune, r_iso):
        v = np.full_like(t_array, V_supply)
        for i, time in enumerate(t_array):
            if t_trigger <= time < t_release:
                dt = time - t_trigger
                v[i] = V_supply * np.exp(-dt / (r_tune * C_sys))
            elif time >= t_release:
                v_at_release = V_supply * np.exp(-(t_release - t_trigger) / (r_tune * C_sys))
                dt = time - t_release
                v[i] = V_supply - (V_supply - v_at_release) * np.exp(-dt / (r_iso * C_sys))
        return v

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # Colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. Top Plot: Discharge detail (Varying R_tune, keeping R_iso constant)
    for r_t, col in zip(R_tune_list, colors):
        v_trace = get_v_trace(t_fast, r_t, R_iso_list[1])  # Use middle R_iso
        ax1.plot(t_fast * 1e6, v_trace, color=col, lw=2, label=f'R_discharge = {r_t} $\Omega$')
    ax1.axvspan(flash_start * 1e6, flash_end * 1e6, color='orange', alpha=0.4, label='Gamma Flash')
    ax1.axhline(0, color='gray', linestyle='-', zorder=0)
    ax1.axhline(V_supply, color='gray', linestyle='-', zorder=0)
    ax1.set_xlim(5, 20)
    ax1.set_title("Fast Discharge -- Detector Turn On")
    ax1.set_ylabel("Mesh Voltage (V)")
    ax1.set_xlabel("Time ($\mu$s)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Bottom Plot: Full Recovery (Varying R_iso, keeping R_tune constant)
    for r_i, col in zip(R_iso_list, colors):
        v_trace = get_v_trace(t_slow, R_tune_list[1], r_i)  # Use middle R_tune
        ax2.plot(t_slow * 1e3, v_trace, color=col, lw=2, label=f'R_charge = {r_i / 1e6:.0f} M$\Omega$')

    ax2.axhline(0, color='gray', linestyle='-', zorder=0)
    ax2.axhline(V_supply, color='gray', linestyle='-', zorder=0)
    ax2.set_title("Slow Recharge -- Detector Turn Off")
    ax2.set_ylabel("Mesh Voltage (V)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Parameters text box
    info_text = (f"V_bias: {V_supply}V\n"
                 f"C_total: {C_sys * 1e12:.0f} pF")
    fig.text(0.5, 0.2, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), va='center')

    fig.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
