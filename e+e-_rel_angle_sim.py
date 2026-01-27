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


# Physical constants (GeV)
m_e = 0.000511  # electron mass


def main():
    m = 0.0168  # parent mass (GeV), e.g. X17
    E = 0.02  # parent energy (GeV)

    angles = np.array([simulate_decay(m, E) for _ in range(10000)])

    # Convert to degrees
    angles_deg = np.degrees(angles)

    print("Mean opening angle (deg):", angles_deg.mean())
    print("Min opening angle (deg):", angles_deg.min())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(angles_deg, bins=100, histtype='stepfilled', alpha=0.7, color='k', zorder=10)
    ax.set_xlabel('Opening Angle between e+ and e- (degrees)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Opening Angle Distribution for X17 Decay (m={m} GeV, E={E} GeV)')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print('donzo')


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
    Simulate one decay and return the opening angle (radians)
    between e+ and e- in the lab frame.
    """

    # Parent momentum magnitude
    p_parent = np.sqrt(E_parent**2 - m_parent**2)

    # Random parent direction
    parent_dir = random_unit_vector()
    p_parent_vec = p_parent * parent_dir

    # Parent beta vector
    beta_vec = p_parent_vec / E_parent

    # --- Decay in parent rest frame ---

    # Electron energy in rest frame
    E_e = m_parent / 2
    p_e = np.sqrt(E_e**2 - m_e**2)

    # Random decay direction
    decay_dir = random_unit_vector()

    # e- and e+ 4-vectors in rest frame
    p4_em_rest = np.array([E_e, *(-p_e * decay_dir)])
    p4_ep_rest = np.array([E_e, *( p_e * decay_dir)])

    # --- Boost to lab frame ---
    p4_em_lab = lorentz_boost(p4_em_rest, beta_vec)
    p4_ep_lab = lorentz_boost(p4_ep_rest, beta_vec)

    # Extract spatial momenta
    p_em = p4_em_lab[1:]
    p_ep = p4_ep_lab[1:]

    # Opening angle
    cos_angle = np.dot(p_em, p_ep) / (np.linalg.norm(p_em) * np.linalg.norm(p_ep))
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    return angle


if __name__ == '__main__':
    main()
