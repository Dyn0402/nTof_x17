#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 02 10:40 PM 2026
Created in PyCharm
Created as nTof_x17/aluminum_shielding.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Energy range: 1 keV to 10 MeV
energies = np.logspace(3, 7, 200)  # in eV
energies_mev = energies / 1e6

# Aluminum properties
density_al = 2.7  # g/cm^3

# --- PHOTON ATTENUATION (Mass Attenuation Coefficient mu/rho in cm^2/g) ---
# Approximated values for Aluminum based on NIST XCOM data
energy_pts = np.array([1e3, 1.5e3, 2e3, 1e4, 1e5, 1e6, 1e7])  # eV
mu_rho_pts = np.array([1180, 2600, 400, 26.2, 0.170, 0.061, 0.023])  # cm^2/g
# Log-log interpolation for photon attenuation
mu_rho = 10 ** np.interp(np.log10(energies), np.log10(energy_pts), np.log10(mu_rho_pts))


# --- ELECTRON RANGE (Katz-Penfold relation for Aluminum) ---
# Range R in g/cm^2
def calculate_electron_range(E_mev):
    # Katz-Penfold empirical formula
    return 0.412 * E_mev ** (1.265 - 0.0954 * np.log(E_mev))


# Simplified transmission model for electrons: T = exp(-(x/R)^p)
# This mimics the "shoulder" and drop-off of electron transmission
def electron_transmission(E_mev, thickness_cm):
    R = calculate_electron_range(E_mev)
    rho_x = thickness_cm * density_al
    # Empirical slope factor p ~ 2-3 for Al
    return np.exp(-(rho_x / (0.8 * R)) ** 3)


# Thicknesses to plot (in mm)
thicknesses_mm = [0.1, 0.3, 1]
colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(figsize=(10, 6))

for i, t_mm in enumerate(thicknesses_mm):
    t_cm = t_mm / 10.0

    # Photon transmission (Beer-Lambert)
    t_photon = np.exp(-mu_rho * density_al * t_cm)

    # Electron transmission
    t_electron = [electron_transmission(E, t_cm) for E in energies_mev]
    t_electron = np.clip(t_electron, 0, 1)  # Bound by physics

    ax.plot(energies_mev, t_photon, label=f'Photons ({t_mm} mm)', color=colors[i], linestyle='--')
    ax.plot(energies_mev, t_electron, label=f'Electrons ({t_mm} mm)', color=colors[i], linestyle='-')

ax.set_xscale('log')
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Transmission Fraction')
ax.set_title('Transmission of Electrons vs Photons through Aluminum')
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend()
plt.ylim(-0.05, 1.05)
plt.xlim(1e-3, 10)
plt.tight_layout()


# --- Physical Constants & Data ---
DENSITY_AL = 2.7  # g/cm^3

# Photon Mass Attenuation Coefficient (mu/rho in cm^2/g)
# Approximate values for Aluminum (NIST XCOM)
E_PTS_EV = np.array([1e3, 1.5e3, 2e3, 1e4, 1e5, 1e6, 1e7, 1e8])
MU_RHO_PTS = np.array([1180, 2600, 400, 26.2, 0.170, 0.061, 0.023, 0.023])

def get_mu_rho(energy_ev):
    """Interpolates mass attenuation coefficient in log-log space."""
    return 10**np.interp(np.log10(energy_ev), np.log10(E_PTS_EV), np.log10(MU_RHO_PTS))

def get_electron_range(E_mev):
    """Katz-Penfold empirical formula for electron range R in g/cm^2."""
    # Valid for roughly 0.01 to 3 MeV, used here as a general proxy
    return 0.412 * E_mev**(1.265 - 0.0954 * np.log(E_mev))

# --- Plot 2: Energy for 90% Transmission vs Thickness ---
thicknesses_mm = np.logspace(-2, 0.2, 100) # 0.01 mm to ~30 mm
thicknesses_cm = thicknesses_mm / 10.0
target_t = 0.99

e_90_photon = []
e_90_electron = []

for t_cm in thicknesses_cm:
    # 1. Photons: T = exp(-mu_rho * rho * x) = 0.9
    # Solve for E where mu_rho = -ln(0.9) / (rho * x)
    target_mu = -np.log(target_t) / (DENSITY_AL * t_cm)
    log_e_search = np.linspace(3, 8, 2000) # 1 keV to 100 MeV
    mu_search = get_mu_rho(10**log_e_search)
    idx = np.argmin(np.abs(mu_search - target_mu))
    e_90_photon.append(10**log_e_search[idx] / 1e6)

    # 2. Electrons: T = exp(-(x / (0.8*R))^3) = 0.9
    # Solve for E where R = x / (0.8 * (-ln(0.9))^(1/3))
    target_R = (t_cm * DENSITY_AL) / (0.8 * (-np.log(target_t))**(1/3))
    e_sol = fsolve(lambda E: get_electron_range(E) - target_R, 1.0)[0]
    e_90_electron.append(e_sol)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(thicknesses_mm, e_90_photon, label=f'Photons ({target_t * 100:.1f}% Transmission)', ls='--', color='tab:blue')
plt.plot(thicknesses_mm, e_90_electron, label=f'Electrons ({target_t * 100:.1f}% Transmission)', color='tab:orange')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Aluminum Thickness (mm)')
plt.ylabel('Required Particle Energy (MeV)')
plt.title(f'Energy Required for {target_t * 100:.1f}% Particle Transmission through Aluminum')
plt.grid(True, which="both", alpha=0.4)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(e_90_photon, thicknesses_mm, label=f'Photons ({target_t * 100:.1f}% Transmission)', ls='--', color='tab:blue')
plt.plot(e_90_electron, thicknesses_mm, label=f'Electrons ({target_t * 100:.1f}% Transmission)', color='tab:orange')

plt.ylabel('Aluminum Thickness (mm)')
plt.xlabel('Required Particle Energy (MeV)')
plt.title(f'Energy Required for {target_t * 100:.1f}% Particle Transmission through Aluminum')
plt.grid(True, which="both", alpha=0.4)
plt.legend()

plt.show()
