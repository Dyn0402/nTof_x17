#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 01 16:06 2025
Created in PyCharm
Created as nTof_x17/neutron_energy_vs_flight_time

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


# Python code to compute neutron flight time vs energy (relativistic and non-relativistic consistent),
# provide conversion functions and plot the result for a flight path of 19.5 m.
# Saves a PNG and CSV to /mnt/data for download.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

from X17CalculationParser import X17CalculationParser

# Constants
c = 299792458.0  # speed of light, m/s
eV_to_J = 1.602176634e-19  # J per eV
m_n = 1.67492749804e-27  # neutron mass, kg
rest_energy_J = m_n * c**2  # neutron rest energy in J
rest_energy_eV = rest_energy_J / eV_to_J

# Distance (EAR2 example)
distance_m = 19.5


def main():
    # plot_flight_time_vs_energy()
    flash_time = energy_eV_to_time_s(1000e9)
    print(f'Gamma flash time for 1000 GeV neutron over {distance_m} m: {flash_time*1e9:.2f} ns')

    # blind_time = 1000e-9  # 1000 ns --> 1 us
    blind_time = 700e-9  # 1000 ns --> 1 us
    blind_energy = time_s_to_energy_eV(blind_time + flash_time)
    print(f'Blind energy for {blind_time*1e9:.0f} ns after flash over {distance_m} m: {blind_energy/1e6:.2f} MeV')

    # readout_time = 3000e-9  # 3000 ns --> 3 us
    readout_time = 10000e-9  # 6000 ns --> 6 us
    readout_low_energy = time_s_to_energy_eV(readout_time + blind_time + flash_time)
    print(f'Readout low energy for {readout_time*1e9:.0f} ns after blind over {distance_m} m: {readout_low_energy/1e3:.2f} keV')

    print(f'For {readout_time*1e6:.1f} µs readout after {blind_time*1e6:.1f} µs of blindness, neutron energies '
          f'from {readout_low_energy/1e6:.2f} - {blind_energy/1e6:.2f} MeV are recorded.')

    calculation_tables_dir = f'/local/home/dn277127/x17/calculation_tables/'
    file_name = 'results_3He'
    parser = X17CalculationParser(calculation_tables_dir + file_name)
    df = parser.get_dataframe()
    print(df.columns)

    # Plot errorbar graph. Energy on x axis and X17 on y. Use elow [eV] and eup [eV] for x range (point at middle with errorbar)
    # y should just be X17 [1/day], no errorbar

    fig, ax = plt.subplots(figsize=(8, 5))
    energies_eV = (df['elow [eV]'] + df['eup [eV]']) / 2.0
    energies_err_eV = (df['eup [eV]'] - df['elow [eV]']) / 2.0
    X17 = df['X17 [1/day]']
    ax.errorbar(energies_eV, X17, xerr=energies_err_eV, fmt='o', ecolor='gray', capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Neutron Energy (eV)')
    ax.set_ylabel('X17 (1/day)')
    ax.set_title('X17 vs Neutron Energy')
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    # Convert energies to time with function
    df['time_low_energy_s'] = energy_eV_to_time_s(df['eup [eV]'], distance=distance_m)  # note inversion of low/high
    df['time_high_energy_s'] = energy_eV_to_time_s(df['elow [eV]'], distance=distance_m)
    df['time_mid_s'] = ( df['time_low_energy_s'] + df['time_high_energy_s'] ) / 2.0
    df['time_err_s'] = ( df['time_high_energy_s'] - df['time_low_energy_s'] ) / 2.0

    times_s = df['time_mid_s'].to_numpy()
    times_s_err = df['time_err_s'].to_numpy()

    # Plot the same data but with time on x axis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(times_s * 1e6, X17, xerr=times_s_err * 1e6, fmt='o', ecolor='gray', capsize=3)
    ax.axvline(flash_time * 1e6, color='red', ls='--', label='Gamma Flash Time')
    ax.axvline((flash_time + blind_time) * 1e6, color='orange', ls='--', label='End of Blind Time')
    ax.axvline((flash_time + blind_time + readout_time) * 1e6, color='green', ls='--', label='End of Readout Time')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Neutron Flight Time (µs) over {:.2f} m'.format(distance_m))
    ax.set_ylabel('X17 (1/day)')
    ax.set_title('X17 vs Neutron Flight Time')
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    plot_spectrum_vs_energy(df, 'X17 [1/day]')
    plot_spectrum_vs_time(df, 'X17 [1/day]', distance_m=distance_m,
                          flash_time_s=flash_time, blind_time_s=blind_time, readout_time_s=readout_time)

    plt.show()

    print('donzo')


def plot_spectrum_vs_energy(df, ycol):
    # Compute geometric-mean energy for the x-axis
    E_low = df["elow [eV]"].values
    E_up  = df["eup [eV]"].values
    E = np.sqrt(E_low * E_up)  # geometric mean

    y = df[ycol].values

    # --- Interpolation (log-log cubic spline) ---
    # Use log axes for physics spectra spanning orders of magnitude
    cs = CubicSpline(np.log(E), np.log(y))
    E_smooth = np.logspace(np.log10(E.min()), np.log10(E.max()), 2000)
    y_smooth = np.exp(cs(np.log(E_smooth)))

    plt.figure(figsize=(10, 6))

    # --- Plot raw points with horizontal error bars (exact bin widths) ---
    xerr = np.array([E - E_low, E_up - E])  # asymmetric error bars
    plt.errorbar(
        E, y, xerr=xerr, fmt="o", markersize=6, capsize=3,
        label="data bins", color="C0"
    )

    # --- Plot the interpolated faint smooth line ---
    plt.plot(
        E_smooth, y_smooth,
        linewidth=1.2, alpha=0.35, color="C0",
        label="interpolation"
    )
    plt.axhline(0, color="gray", linestyle="-", zorder=0)

    # --- Axis formatting ---
    plt.xscale("log")
    plt.xlabel("Energy [eV]")
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs neutron energy")
    plt.legend()
    plt.tight_layout()


def plot_spectrum_vs_time(df, ycol, distance_m=distance_m, flash_time_s=None, blind_time_s=None, readout_time_s=None):
    """
    Plot spectrum data vs neutron flight time instead of energy.
    :param df: DataFrame with elow [eV], eup [eV], and ycol columns
    :param ycol: Column name for y-axis data
    :param distance_m: Flight distance in meters
    :param flash_time_s: Optional flash time in seconds to mark on plot
    :param blind_time_s: Optional blind time in seconds to mark on plot
    :param readout_time_s: Optional readout time in seconds to mark on plot
    """
    # Convert energy bins to time bins
    E_low = df["elow [eV]"].values
    E_up  = df["eup [eV]"].values
    t_low = energy_eV_to_time_s(E_up, distance=distance_m)  # note inversion of low/high
    t_up  = energy_eV_to_time_s(E_low, distance=distance_m)
    t = (t_low + t_up) / 2.0  # mid time

    y = df[ycol].values

    # Resort by time (in case energy bins were not sorted)
    sorted_indices = np.argsort(t)
    t = t[sorted_indices]
    t_low = t_low[sorted_indices]
    t_up = t_up[sorted_indices]
    y = y[sorted_indices]

    # --- Interpolation (log-log cubic spline) ---
    # Use log axes for physics spectra spanning orders of magnitude
    cs = CubicSpline(np.log(t), np.log(y))
    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 2000)
    y_smooth = np.exp(cs(np.log(t_smooth)))

    plt.figure(figsize=(10, 6))

    # --- Plot raw points with horizontal error bars (exact bin widths) ---
    xerr = np.array([t - t_low, t_up - t])  # asymmetric error bars
    plt.errorbar(
        t * 1e6, y, xerr=xerr * 1e6, fmt="o", markersize=6, capsize=3,
        label="data bins", color="C0"
    )

    # --- Plot the interpolated faint smooth line ---
    plt.plot(
        t_smooth * 1e6, y_smooth,
        linewidth=1.2, alpha=0.35, color="C0",
        label="interpolation"
    )
    plt.axhline(0, color="gray", linestyle="-", zorder=0)

    # --- Plot vertical lines for flash, blind, readout times if provided ---
    annote_str = ''
    if flash_time_s is not None:
        plt.axvline(flash_time_s * 1e6, color='red', ls='--', label=f'Gamma Flash Time: {flash_time_s*1e6:.2f} µs')
    if blind_time_s is not None:
        plt.axvline((flash_time_s + blind_time_s) * 1e6, color='orange', ls='--', label=f'End of Blind Time: {(flash_time_s + blind_time_s)*1e6:.2f} µs')
        annote_str += f'Blind Time: {blind_time_s*1e6:.1f} µs\n'
    if readout_time_s is not None:
        plt.axvline((flash_time_s + blind_time_s + readout_time_s) * 1e6, color='green', ls='--', label=f'End of Readout Time: {(flash_time_s + blind_time_s + readout_time_s)*1e6:.2f} µs')
        annote_str += f'Readout Time: {readout_time_s*1e6:.1f} µs'
    if annote_str:
        plt.annotate(annote_str, xy=(0.65, 0.55), xycoords='axes fraction', fontsize=14,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    # --- Axis formatting ---
    plt.xscale("log")
    plt.xlabel(f"Neutron flight time [µs] over {distance_m:.2f} m")
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs neutron flight time")
    plt.legend()
    plt.tight_layout()



def plot_flight_time_vs_energy(distance=distance_m):
    """Plot neutron flight time vs kinetic energy for given distance."""
    # Generate energies from thermal and below up to high-energy neutrons
    energies_eV = np.logspace(-3, 9, num=1000)  # from 1e-8 eV to 1e9 eV
    times_s = energy_eV_to_time_s(energies_eV, distance=distance_m)

    # Create a DataFrame and save CSV
    df = pd.DataFrame({
        "energy_eV": energies_eV,
        "energy_meV": energies_eV * 1e3,
        "time_s": times_s,
        "time_us": times_s * 1e6,
        "time_ns": times_s * 1e9
    })

    # Plot: log-log flight time vs energy
    plt.figure(figsize=(8, 5))
    plt.loglog(energies_eV, times_s * 1e9)
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    plt.xlabel("Neutron kinetic energy (eV)")
    plt.ylabel("Flight time (ns) over {:.2f} m".format(distance_m))
    plt.title("Neutron flight time vs kinetic energy (distance = {:.2f} m)".format(distance_m))
    plt.tight_layout()

    plt.show()

    # Additionally produce a small table of example conversions for typical energies
    examples_eV = np.array([1e-8, 2.5e-2, 1.0, 1e3, 1e6, 1e9])  # cold, thermal (~0.025 eV), 1 eV, keV, MeV, GeV
    examples_df = pd.DataFrame({
        "energy_eV": examples_eV,
        "time_s": energy_eV_to_time_s(examples_eV, distance=distance_m),
        "time_us": energy_eV_to_time_s(examples_eV, distance=distance_m) * 1e6,
        "beta": energy_eV_to_beta(examples_eV),
        "KE_over_rest_fraction": (examples_eV / rest_energy_eV)
    })


def energy_eV_to_beta(energy_eV):
    """Return beta = v/c for neutron with kinetic energy energy_eV (can be array)."""
    E_J = energy_eV * eV_to_J
    gamma = E_J / rest_energy_J + 1.0
    # numeric safety: gamma >= 1
    gamma = np.maximum(gamma, 1.0)
    beta = np.sqrt(1.0 - 1.0/(gamma**2))
    return beta

def energy_eV_to_time_s(energy_eV, distance=distance_m):
    """Flight time in seconds for neutron of kinetic energy energy_eV over distance (m)."""
    beta = energy_eV_to_beta(energy_eV)
    # protect zero beta (very tiny energies) -> use nonrelativistic approximation for small beta
    # but formula already handles it; however numerical underflow can occur, so clip beta min
    beta = np.clip(beta, 1e-12, 1.0-1e-12)
    v = beta * c
    return distance / v

def time_s_to_energy_eV(time_s, distance=distance_m):
    """Invert flight time -> kinetic energy in eV."""
    # compute beta from time: beta = distance / (c * time)
    beta = distance / (c * time_s)
    # physical bounds
    beta = np.clip(beta, 0.0, 1.0-1e-15)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    KE_J = (gamma - 1.0) * rest_energy_J
    KE_eV = KE_J / eV_to_J
    return KE_eV


if __name__ == '__main__':
    main()
