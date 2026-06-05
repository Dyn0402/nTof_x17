#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 16 17:26 2025
Created in PyCharm
Created as nTof_x17/e+e-_rel_angle_sim

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from matplotlib.transforms import blended_transform_factory


# Physical constants (GeV)
m_e = 0.000511  # electron mass


def main():
    m = 0.0168   # parent mass (GeV), e.g. X17
    E = 0.02058  # parent energy (GeV)
    n_decay_stats = 1_000_000

    # Analytic lepton energy limits in lab frame: E_lab = gamma*(E* +/- beta*p*)
    gamma = E / m
    beta = np.sqrt(E**2 - m**2) / E
    E_star = m / 2
    p_star = np.sqrt(E_star**2 - m_e**2)
    E_min_analytic = gamma * (E_star - beta * p_star) * 1000  # MeV
    E_max_analytic = gamma * (E_star + beta * p_star) * 1000  # MeV

    chunk_size = 1000
    n_chunks = (n_decay_stats + chunk_size - 1) // chunk_size
    chunks = [(min(chunk_size, n_decay_stats - i * chunk_size), m, E, i) for i in range(n_chunks)]

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results_nested = list(tqdm(executor.map(simulate_batch, chunks), total=n_chunks, desc='Simulating decays'))
    results = [item for sublist in results_nested for item in sublist]
    angles = np.array([r[0] for r in results])
    e_em = np.array([r[1] for r in results])  # e- lab energies
    e_ep = np.array([r[2] for r in results])  # e+ lab energies

    # Convert to degrees and MeV
    angles_deg = np.degrees(angles)
    e_em_mev = e_em * 1000
    e_ep_mev = e_ep * 1000
    E_high = np.maximum(e_em_mev, e_ep_mev)
    E_low  = np.minimum(e_em_mev, e_ep_mev)

    print("Mean opening angle (deg):", angles_deg.mean())
    print("Min opening angle (deg):", angles_deg.min())
    print(f"e- lab energy range: {e_em_mev.min():.2f} – {e_em_mev.max():.2f} MeV")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.hist(angles_deg, bins=100, histtype='stepfilled', alpha=0.7, color='k', zorder=10)
    ax.set_xlabel('Opening Angle between e+ and e- (degrees)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Opening Angle Distribution\n(m={m} GeV, E={E} GeV)')
    ax.grid(True)

    ax = axes[0, 1]
    ax.hist(e_em_mev, bins=80, histtype='step', color='blue', label='e-', zorder=10)
    ax.hist(e_ep_mev, bins=80, histtype='step', color='red', linestyle='--', label='e+', zorder=10)
    ax.axvline(E_min_analytic, color='green', linestyle=':', linewidth=1.5, zorder=5)
    ax.axvline(E_max_analytic, color='green', linestyle=':', linewidth=1.5, zorder=5)
    center_x = (E_min_analytic + E_max_analytic) / 2
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(center_x, 0.08,
            f'Min = {E_min_analytic:.2f} MeV\nMax = {E_max_analytic:.2f} MeV',
            transform=transform, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Lepton Energy in Lab Frame (MeV)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Lab-Frame Lepton Energy Distribution\n(m={m} GeV, E={E} GeV)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot([E_min_analytic, E_max_analytic], [E_max_analytic, E_min_analytic],
            'g-', linewidth=2.5, label=f'$E_{{e^-}}+E_{{e^+}}={E*1000:.2f}$ MeV')
    ax.set_xlabel('e- Energy in Lab Frame (MeV)')
    ax.set_ylabel('e+ Energy in Lab Frame (MeV)')
    ax.set_title(f'Lepton Energy Correlation\n(m={m} GeV, E={E} GeV)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Subsample for the E vs angle plot — curves are deterministic so 10k points is plenty
    rng = np.random.default_rng(0)
    idx = rng.choice(len(angles_deg), size=10_000, replace=False)
    ax = axes[1, 1]
    ax.scatter(angles_deg[idx], E_high[idx], s=2, color='red',  alpha=0.4, label='Higher-energy lepton', zorder=10)
    ax.scatter(angles_deg[idx], E_low[idx],  s=2, color='blue', alpha=0.4, label='Lower-energy lepton',  zorder=10)
    ax.axhline(E_min_analytic, color='green', linestyle=':', linewidth=1, zorder=5)
    ax.axhline(E_max_analytic, color='green', linestyle=':', linewidth=1, zorder=5)
    ax.axhline((E_min_analytic + E_max_analytic) / 2, color='grey', linestyle='--', linewidth=1, zorder=5)
    ax.set_xlabel('Opening Angle between e+ and e- (degrees)')
    ax.set_ylabel('Lepton Energy in Lab Frame (MeV)')
    ax.set_title(f'Lepton Energy vs Opening Angle\n(m={m} GeV, E={E} GeV)')
    ax.legend(markerscale=4, fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    print('donzo')


def simulate_batch(args):
    """Run a batch of decays with a fixed seed for independent RNG per worker."""
    n, m_parent, E_parent, seed = args
    np.random.seed(seed)
    return [simulate_decay(m_parent, E_parent) for _ in range(n)]


def random_unit_vector():
    """Generate a random isotropic unit vector."""
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, 2*np.pi)
    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

def lorentz_boost(p4, beta_vec):
    """
    Lorentz boost of a 4-vector.
    p4 = (E, px, py, pz)
    beta_vec = 3-velocity of boost (|beta| < 1)
    """
    beta = np.linalg.norm(beta_vec)
    if beta == 0:
        return p4.copy()

    gamma = 1.0 / np.sqrt(1 - beta**2)
    beta_hat = beta_vec / beta

    E, p = p4[0], p4[1:]
    p_par = np.dot(p, beta_hat) * beta_hat
    p_perp = p - p_par

    E_prime = gamma * (E + np.dot(beta_vec, p))
    p_par_prime = gamma * (p_par + E * beta_hat * beta)

    return np.concatenate(([E_prime], p_par_prime + p_perp))

def simulate_decay(m_parent, E_parent):
    """
    Simulate one decay. Returns (opening_angle_rad, E_em_lab, E_ep_lab).
    Parent is boosted along z-axis (direction doesn't affect the distributions).
    """

    # Parent momentum along z (direction is arbitrary by symmetry)
    p_parent = np.sqrt(E_parent**2 - m_parent**2)
    beta_vec = np.array([0.0, 0.0, p_parent / E_parent])

    # --- Decay in parent rest frame ---
    E_e = m_parent / 2
    p_e = np.sqrt(E_e**2 - m_e**2)

    decay_dir = random_unit_vector()

    p4_em_rest = np.array([E_e, *(-p_e * decay_dir)])
    p4_ep_rest = np.array([E_e, *( p_e * decay_dir)])

    # --- Boost to lab frame (inverse boost: rest frame -> lab) ---
    p4_em_lab = lorentz_boost(p4_em_rest, beta_vec)
    p4_ep_lab = lorentz_boost(p4_ep_rest, beta_vec)

    # Opening angle
    p_em = p4_em_lab[1:]
    p_ep = p4_ep_lab[1:]
    cos_angle = np.dot(p_em, p_ep) / (np.linalg.norm(p_em) * np.linalg.norm(p_ep))
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    return angle, p4_em_lab[0], p4_ep_lab[0]


if __name__ == '__main__':
    main()
