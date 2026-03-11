#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03 10:19 AM 2026
Created in PyCharm
Created as nTof_x17/mx17_mesh_switch_sim.py

@author: Dylan Neff, dylan

Extended with Ramo/Shockley-Ramo induced charge simulation on readout plane.

Physics:
    The induced charge on the readout plane is calculated using Ramo's theorem.
    For a parallel plate geometry (mesh at distance d from readout plane),
    the weighting potential is linear: phi_w(x) = x / d
    The mesh sits at x = d, so phi_w(mesh) = 1.

    When mesh charge changes by dQ = C_mesh * dV, the induced charge on readout is:
        Q_induced = -dQ * phi_w(x_mesh) = -C_mesh * dV

    In the continuous (time-resolved) case:
        dQ_induced/dt = -C_mesh * dV/dt
    i.e. the induced current mirrors the rate of change of mesh voltage,
    scaled by the mesh capacitance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    # plot_discharge_times()
    plot_current()
    print('donzo')


def plot_discharge_times():
    # --- Constant System Parameters ---
    v_supply = 100.0
    c_sys = 10e-9        # 10 nF total system capacitance (mesh + cable)
    c_mesh = 1e-9        # 1 nF  mesh-to-readout capacitance (parallel plate, see below)
    d_gap = 150e-6       # 150 µm amplification gap (mesh to readout plane)

    t_trigger = 10e-6    # When discharge starts (MOSFET fires)
    t_release = 30e-3    # When MOSFET turns off
    t_end = 1000e-3      # End of simulation window
    mosfet_r = 0.68      # Ohm, MOSFET on-resistance

    flash_start, flash_end = 10e-6, 11e-6  # Gamma flash window to avoid

    # --- Parallel plate capacitance note ---
    # For reference: C = epsilon_0 * A / d
    # With d=150µm and e.g. 10x10 cm² pad: C ~ 8.85e-12 * 0.01 / 150e-6 ~ 0.59 nF
    # Adjust c_mesh above to match your actual geometry.
    epsilon_0 = 8.854e-12  # F/m
    A_readout = 10e-2 * 10e-2   # Example: 10x10 cm² readout area (m²)
    c_mesh_geo = epsilon_0 * A_readout / d_gap
    print(f"Geometric estimate of mesh capacitance: {c_mesh_geo * 1e12:.2f} pF  "
          f"(for {A_readout*1e4:.0f} cm², d={d_gap*1e6:.0f} µm)")
    print(f"Using c_mesh = {c_mesh * 1e12:.0f} pF for induced charge calculation.\n")

    # --- Variable Lists for Comparison ---
    r_fast_list = np.array([1, 2, 3, 10]) * 10   # Ohms (discharge tuning)
    r_slow_list = np.array([1e6, 2e6, 3e6]) * 10  # Ohms (recharge isolation)

    # --- Time Arrays ---
    t_fast = np.linspace(0, 40e-6, 5000)   # Fine resolution for fast transient
    t_slow = np.linspace(0, t_end, 10000)  # Coarser for slow recharge

    # -----------------------------------------------------------------------
    # Mesh voltage as a function of time
    # -----------------------------------------------------------------------
    def get_v_trace(t_array, r_tune, r_iso):
        """
        Returns mesh voltage V(t).
          - Before trigger:  V = V_supply  (fully charged)
          - During discharge: V = V_supply * exp(-dt / tau_discharge)
          - After release:    V recovers toward V_supply with tau_recharge
        """
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

    # -----------------------------------------------------------------------
    # Ramo-induced charge on the readout plane
    # -----------------------------------------------------------------------
    def get_induced_charge(t_array, v_trace, c_ramo=c_mesh):
        """
        Induced charge on the readout plane via Ramo's theorem.

        For a parallel plate with mesh at x=d (readout at x=0):
            Weighting potential at mesh position: phi_w = x/d = 1

        Therefore:
            Q_induced(t) = -C_mesh * [V(t) - V(t=0)]

        The minus sign reflects that a positive voltage on the mesh induces
        a negative (mirror) charge on the readout plane.

        Parameters
        ----------
        t_array  : time axis (s)
        v_trace  : mesh voltage as a function of time (V)
        c_ramo   : capacitance between mesh and readout plane (F)

        Returns
        -------
        q_induced : induced charge on readout plane (C), array
        i_induced : induced current dQ/dt (A), array
        """
        v0 = v_trace[0]  # Initial mesh voltage
        q_induced = -c_ramo * (v_trace - v0)   # Ramo: Q = -C * delta_V

        # Induced current: dQ/dt = -C * dV/dt
        dv_dt = np.gradient(v_trace, t_array)
        i_induced = -c_ramo * dv_dt

        return q_induced, i_induced

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    fig = plt.figure(figsize=(13, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # Fast discharge — full width
    ax2 = fig.add_subplot(gs[1, :])   # Slow recharge  — full width
    ax3 = fig.add_subplot(gs[2, 0])   # Induced charge during fast discharge
    ax4 = fig.add_subplot(gs[2, 1])   # Induced charge during slow recharge

    # --- Top: Fast Discharge (varying R_tune) ---
    for r_t, col in zip(r_fast_list, colors):
        r_t_total = r_t + mosfet_r
        v_trace = get_v_trace(t_fast, r_t_total, r_slow_list[1])
        current = v_supply / r_t_total
        ax1.plot(t_fast * 1e6, v_trace, color=col, lw=2,
                 label=f'R_discharge = {r_t} Ω,  I_peak = {current:.2f} A')
    ax1.axvspan(flash_start * 1e6, flash_end * 1e6, color='orange', alpha=0.4, label='Gamma Flash')
    ax1.axhline(0, color='gray', ls='-', zorder=0)
    ax1.axhline(v_supply, color='gray', ls='-', zorder=0)
    ax1.set_xlim(5, 20)
    ax1.set_title("Fast Discharge — Detector Turn-Off", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Mesh Voltage (V)")
    ax1.set_xlabel("Time (µs)")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    _annotate(ax1, 'Detector Off', v_supply, frac_x=0.25, dy=-18)
    _annotate(ax1, 'Detector On',  0,        frac_x=0.42, dy=+18)

    # --- Middle: Slow Recharge (varying R_iso) ---
    for r_i, col in zip(r_slow_list, colors):
        v_trace = get_v_trace(t_slow, r_fast_list[1], r_i)
        current_ua = v_supply / r_i * 1e6
        ax2.plot(t_slow * 1e3, v_trace, color=col, lw=2,
                 label=f'R_charge = {r_i / 1e6:.0f} MΩ,  I_peak = {current_ua:.2f} µA')
    ax2.axhline(0, color='gray', ls='-', zorder=0)
    ax2.axhline(v_supply, color='gray', ls='-', zorder=0)
    ax2.set_title("Slow Recharge — Detector Turn-On", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Mesh Voltage (V)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    _annotate(ax2, 'Detector Off', v_supply, frac_x=0.80, dy=-18)
    _annotate(ax2, 'Detector On',  0,        frac_x=0.055, dy=+18)

    # --- Bottom-left: Induced charge during fast discharge ---
    for r_t, col in zip(r_fast_list, colors):
        r_t_total = r_t + mosfet_r
        v_trace = get_v_trace(t_fast, r_t_total, r_slow_list[1])
        q_ind, _ = get_induced_charge(t_fast, v_trace)
        ax3.plot(t_fast * 1e6, q_ind * 1e9, color=col, lw=2,
                 label=f'R_d = {r_t} Ω')
    ax3.axvline(flash_start * 1e6, color='orange', ls='--', lw=1.2, label='Flash start')
    ax3.set_xlim(5, 20)
    ax3.set_title("Ramo Induced Charge — Fast Discharge", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Induced Charge on Readout (nC)")
    ax3.set_xlabel("Time (µs)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    _add_ramo_annotation(ax3, c_mesh, d_gap)

    # --- Bottom-right: Induced charge during slow recharge ---
    for r_i, col in zip(r_slow_list, colors):
        v_trace = get_v_trace(t_slow, r_fast_list[1], r_i)
        q_ind, _ = get_induced_charge(t_slow, v_trace)
        ax4.plot(t_slow * 1e3, q_ind * 1e9, color=col, lw=2,
                 label=f'R_c = {r_i / 1e6:.0f} MΩ')
    ax4.set_title("Ramo Induced Charge — Slow Recharge", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Induced Charge on Readout (nC)")
    ax4.set_xlabel("Time (ms)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    _add_ramo_annotation(ax4, c_mesh, d_gap)

    # Parameters text
    info_text = (f"V_bias: {v_supply} V\n"
                 f"C_sys: {c_sys * 1e9:.0f} nF\n"
                 f"C_mesh (Ramo): {c_mesh * 1e12:.0f} pF\n"
                 f"Gap d: {d_gap * 1e6:.0f} µm")
    fig.text(0.5, 0.01, info_text, fontsize=9, ha='center',
             bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.suptitle("Micromegas Mesh Voltage & Ramo-Induced Charge Simulation",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/mx17_mesh_switch_sim.png', dpi=150, bbox_inches='tight')
    plt.show()


# -----------------------------------------------------------------------
# Helper annotation functions
# -----------------------------------------------------------------------
def _annotate(ax, label, y_val, frac_x=0.3, dy=-18):
    xlim = ax.get_xlim()
    x = xlim[0] + (xlim[1] - xlim[0]) * frac_x
    dx = -2.5 if dy < 0 else 1.5
    ax.annotate(label, xy=(x, y_val), xytext=(x + dx, y_val + dy),
                fontsize=9, color='dimgray', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))


def _add_ramo_annotation(ax, c_mesh, d_gap):
    """Add a small formula box to the Ramo plots."""
    q_max = c_mesh * 100  # At 100V swing, in Coulombs
    ax.text(0.03, 0.97,
            f"Q = −C_mesh · ΔV\nC_mesh = {c_mesh*1e12:.0f} pF\nQ_max = {q_max*1e9:.1f} nC",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(facecolor='white', edgecolor='steelblue', alpha=0.85))


def plot_current():
    # -----------------------------------------------------------------------
    # System Parameters
    # -----------------------------------------------------------------------
    v_supply    = 20.0       # V  - mesh bias voltage
    c_mesh      = 10e-9        # F  - mesh-to-readout capacitance (1 nF total)
    c_sys       = 11e-9       # F  - full system capacitance (discharge RC)
    mosfet_r    = 0.68        # Ω  - MOSFET on-resistance
    n_strips    = 512 * 2         # number of readout strips per orientation

    # DREAM CSA charge ranges (fC)
    dream_ranges_fC = np.array([50, 100, 200, 600])  # fC

    # DREAM peaking times (ns) — 16 values, 50–900 ns
    # dream_peaking_ns = np.array([76, 123, 180, 228, 283, 328, 388, 433, 578, 618, 675, 717, 781, 818, 880, 919])
    dream_peaking_ns = np.array([76, 123, 180, 228, 283, 328])

    # Discharge resistors to evaluate
    r_discharge_list = np.array([10, 20, 30, 100]) + mosfet_r  # Ω total

    t_trigger   = 10e-6       # s
    t_release   = 30e-3       # s (MOSFET on duration — long for analysis)

    # -----------------------------------------------------------------------
    # Derived quantities
    # -----------------------------------------------------------------------
    q_total_mesh = c_mesh * v_supply          # Total induced charge (C)
    q_per_strip  = q_total_mesh / n_strips    # Assuming uniform distribution

    print("=" * 60)
    print("  DREAM DAQ Charge Budget Analysis")
    print("=" * 60)
    print(f"\n  Mesh voltage swing:       {v_supply:.0f} V")
    print(f"  C_mesh (Ramo):            {c_mesh*1e12:.0f} pF")
    print(f"  Total induced charge:     {q_total_mesh*1e9:.1f} nC")
    print(f"  Strips per axis:          {n_strips}")
    print(f"  Charge per strip (total): {q_per_strip*1e12:.1f} pC")
    print(f"\n  DREAM max range:          {dream_ranges_fC[-1]:.0f} fC")
    print(f"  Overflow factor:          {q_per_strip / (dream_ranges_fC[-1]*1e-15):.0f}x  ← saturates!")

    # -----------------------------------------------------------------------
    # Time-resolved induced current and windowed charge integration
    # -----------------------------------------------------------------------
    dt     = 1e-9            # 1 ns resolution
    t_max  = 2e-6            # look at first 2 µs after trigger
    t_arr  = np.arange(0, t_max, dt)

    print("\n" + "=" * 60)
    print("  Charge per strip integrated within DREAM peaking window")
    print("  (assuming discharge starts at t_trigger, strip = total/N_strips)")
    print("=" * 60)
    print(f"\n  {'R_disc (Ω)':>12}  {'τ_disc (µs)':>12}  ", end="")
    for tp in dream_peaking_ns:
        print(f"  t_peak={tp:3d}ns", end="")
    print()

    fig = plt.figure(figsize=(14, 12))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])   # Induced current per strip
    ax2 = fig.add_subplot(gs[1])   # Integrated charge vs time
    ax3 = fig.add_subplot(gs[2])   # Windowed charge vs peaking time

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for r_d, col in zip(r_discharge_list, colors):
        tau = r_d * c_sys                    # discharge time constant

        # Mesh voltage after trigger
        v_mesh = v_supply * np.exp(-t_arr / tau)

        # dV/dt → induced current on ALL readout (Ramo: I = -C_mesh * dV/dt)
        dv_dt   = np.gradient(v_mesh, dt)
        i_total = -c_mesh * dv_dt            # total induced current (A)
        i_strip = i_total / n_strips         # current per strip

        # Cumulative charge per strip
        q_cumul = np.cumsum(i_strip) * dt    # running integral (C)

        r_label = f'R={r_d-mosfet_r:.0f}Ω, τ={tau*1e6:.2f}µs'

        ax1.plot(t_arr*1e6, i_strip*1e6, color=col, lw=2, label=r_label)
        ax2.plot(t_arr*1e6, q_cumul*1e15, color=col, lw=2, label=r_label)

        # Windowed charge: integrate from t=0 to t=t_peak
        print(f"  {r_d-mosfet_r:>12.0f}  {tau*1e6:>12.3f}  ", end="")
        q_windowed = []
        for tp_ns in dream_peaking_ns:
            tp = tp_ns * 1e-9
            n_bins = int(tp / dt)
            q_win = np.sum(i_strip[:n_bins]) * dt
            q_windowed.append(q_win)
            print(f"  {q_win*1e15:>14.1f} fC", end="")
        print()

        ax3.plot(dream_peaking_ns, np.array(q_windowed)*1e15, 'o-',
                 color=col, lw=2, markersize=6, label=r_label)

    # DREAM range lines on ax3
    for dr, ls in zip(dream_ranges_fC, [':', '-.', '--', '-']):
        ax3.axhline(dr, color='red', ls=ls, lw=1.3, alpha=0.7,
                    label=f'DREAM range: {dr} fC')

    # --- Formatting ---
    ax1.set_title("Induced Current per Strip (Ramo, uniform distribution)",
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time after trigger (µs)")
    ax1.set_ylabel("Current per strip (µA)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', lw=0.8)

    ax2.set_title("Cumulative Induced Charge per Strip vs Time",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time after trigger (µs)")
    ax2.set_ylabel("Charge per strip (fC)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Draw DREAM max range
    ax2.axhline(dream_ranges_fC[-1], color='red', ls='--', lw=1.5,
                label='DREAM max 600 fC')
    ax2.axhline(dream_ranges_fC[0], color='orange', ls='--', lw=1.5,
                label='DREAM min 50 fC')
    ax2.legend(fontsize=9)

    ax3.set_title("Charge Integrated within DREAM Peaking Window per Strip",
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel("DREAM Peaking Time (ns)")
    ax3.set_ylabel("Integrated charge per strip (fC)")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=1)

    # Shade safe zone
    ax3.axhspan(0, dream_ranges_fC[-1], alpha=0.08, color='green',
                label='Within DREAM range')
    ax3.text(55, dream_ranges_fC[-1]*1.1, 'DREAM 600 fC ceiling',
             color='red', fontsize=8)

    # Parameters box
    info = (f"V_mesh = {v_supply:.0f} V\n"
            f"C_mesh = {c_mesh*1e12:.0f} pF\n"
            f"Q_total = {q_total_mesh*1e9:.1f} nC\n"
            f"N_strips = {n_strips}\n"
            f"Q/strip (total) = {q_per_strip*1e12:.1f} pC")
    fig.text(0.76, 0.08, info, fontsize=9,
             bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.suptitle("DREAM DAQ: Mesh Discharge Charge Budget per Strip",
                 fontsize=13, fontweight='bold')
    # plt.savefig('/mnt/user-data/outputs/mx17_dream_daq_analysis.png',
    #             dpi=150, bbox_inches='tight')
    plt.show()
    print("\ndonzo")


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on March 03 10:19 AM 2026
# Created in PyCharm
# Created as nTof_x17/mx17_mesh_switch_sim.py
#
# @author: Dylan Neff, dylan
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def main():
#     # --- Constant System Parameters ---
#     v_supply = 100.0
#     c_sys = 10e-9  # 100nF Mesh + 100nF Cable
#     t_trigger = 10e-6  # When discharge starts
#     t_release = 30e-3  # When MOSFET turns off (10us duration for visibility)
#     t_end = 1000e-3  # Extended to 100ms to see slow charging
#     mosfet_r = 0.68  # ohm, internal resistance of mosfet
#
#     flash_start, flash_end = 10e-6, 11e-6  # Start and end of gamma flash which we want to avoid
#
#     # --- Variable Lists for Comparison ---
#     r_fast_list = np.array([1, 2, 3, 10]) * 10  # Ohms (Tuning the discharge)
#     r_slow_list = np.array([1e6, 2e6, 3e6]) * 10  # Ohms (5M, 10M, 20M for recharge)
#
#     # --- Time Arrays ---
#     t_fast = np.linspace(0, 40e-6, 1000)
#     t_slow = np.linspace(0, t_end, 2000)
#
#     def get_v_trace(t_array, r_tune, r_iso):
#         v = np.full_like(t_array, v_supply)
#         for i, time in enumerate(t_array):
#             if t_trigger <= time < t_release:
#                 dt = time - t_trigger
#                 v[i] = v_supply * np.exp(-dt / (r_tune * c_sys))
#             elif time >= t_release:
#                 v_at_release = v_supply * np.exp(-(t_release - t_trigger) / (r_tune * c_sys))
#                 dt = time - t_release
#                 v[i] = v_supply - (v_supply - v_at_release) * np.exp(-dt / (r_iso * c_sys))
#         return v
#
#     # --- Plotting ---
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
#
#     # Colors for consistency
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',]
#
#     # 1. Top Plot: Discharge detail (Varying R_tune, keeping R_iso constant)
#     for r_t, col in zip(r_fast_list, colors):
#         r_t_total = r_t + mosfet_r
#         v_trace = get_v_trace(t_fast, r_t_total, r_slow_list[1])  # Use middle R_iso
#         current = v_supply / r_t_total
#         ax1.plot(t_fast * 1e6, v_trace, color=col, lw=2, label=f'R_discharge = {r_t} $\Omega$, Current = {current:.2f} A')
#     ax1.axvspan(flash_start * 1e6, flash_end * 1e6, color='orange', alpha=0.4, label='Gamma Flash')
#     ax1.axhline(0, color='gray', linestyle='-', zorder=0)
#     ax1.axhline(v_supply, color='gray', linestyle='-', zorder=0)
#     ax1.set_xlim(5, 20)
#     ax1.set_title("Fast Discharge -- Detector Turn On")
#     ax1.set_ylabel("Mesh Voltage (V)")
#     ax1.set_xlabel("Time ($\mu$s)")
#     ax1.legend(loc='upper right')
#     ax1.grid(True, alpha=0.3)
#
#     # Annotations for ax1
#     x_ann1_off = ax1.get_xlim()[0] + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.25
#     x_ann1_on = ax1.get_xlim()[0] + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.42
#     ax1.annotate('Detector Off', xy=(x_ann1_off, v_supply), xytext=(x_ann1_off - 2.5, v_supply - 18),
#                  fontsize=9, color='dimgray', fontweight='bold',
#                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
#     ax1.annotate('Detector On', xy=(x_ann1_on, 0), xytext=(x_ann1_on + 1.5, 0 + 18),
#                  fontsize=9, color='dimgray', fontweight='bold',
#                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
#
#     # 2. Bottom Plot: Full Recovery (Varying R_iso, keeping R_tune constant)
#     for r_i, col in zip(r_slow_list, colors):
#         v_trace = get_v_trace(t_slow, r_fast_list[1], r_i)  # Use middle R_tune
#         current = v_supply / r_i * 1e6
#         ax2.plot(t_slow * 1e3, v_trace, color=col, lw=2, label=f'R_charge = {r_i / 1e6:.0f} M$\Omega$, Current = {current:.2f} $\mu$A')
#
#     ax2.axhline(0, color='gray', linestyle='-', zorder=0)
#     ax2.axhline(v_supply, color='gray', linestyle='-', zorder=0)
#     ax2.set_title("Slow Recharge -- Detector Turn Off")
#     ax2.set_ylabel("Mesh Voltage (V)")
#     ax2.set_xlabel("Time (ms)")
#     ax2.legend(loc='lower right')
#     ax2.grid(True, alpha=0.3)
#
#     # Annotations for ax2
#     x_ann2_off = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.8
#     x_ann2_on = ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.055
#     ax2.annotate('Detector Off', xy=(x_ann2_off, v_supply), xytext=(x_ann2_off + 30, v_supply - 18),
#                  fontsize=9, color='dimgray', fontweight='bold',
#                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
#     ax2.annotate('Detector On', xy=(x_ann2_on, 0), xytext=(x_ann2_on + 200, 0 + 18),
#                  fontsize=9, color='dimgray', fontweight='bold',
#                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.5))
#
#     # Parameters text box
#     info_text = (f"V_bias: {v_supply}V\n"
#                  f"C_total: {c_sys * 1e9:.0f} nF")
#     fig.text(0.5, 0.2, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), va='center')
#
#     fig.tight_layout()
#     plt.show()
#     print('donzo')
#
#
# if __name__ == '__main__':
#     main()
