#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 16 18:04 2025
Created in PyCharm
Created as nTof_x17/e+e-_simple_decay_sim

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    m = 0.511e-3  # electron mass in GeV
    E_parent = 0.02  # GeV
    M_parent = 0.017  # GeV, e.g. X17 mass
    # theta_plane = np.pi / 4  # 45 degrees
    theta_plane = np.deg2rad(0)  # 45 degrees

    p4_1, p4_2, opening_angle = simulate_decay(m, E_parent, M_parent, theta_plane)

    print("Particle 1 (E, px, py, pz):", p4_1)
    print("Particle 2 (E, px, py, pz):", p4_2)
    print("Opening angle in lab (deg):", np.degrees(opening_angle))

    # Make a plot of opening angle vs theta_plane
    theta_planes = np.linspace(0, np.pi, 1000)
    opening_angles = []
    for theta in theta_planes:
        _, _, angle = simulate_decay(m, E_parent, M_parent, theta)
        opening_angles.append(np.degrees(angle)) # in degrees
    opening_angles = np.array(opening_angles)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(np.degrees(theta_planes), opening_angles, label='Opening Angle vs Decay Plane Angle')
    ax.plot(np.degrees(theta_planes), np.acos(np.cos(theta_planes) + 0.01) / (1 + 0.01 * np.cos(theta_planes)), 'r--', label='Cosine Approximation')
    ax.set_xlabel('Decay Plane Angle θ_plane (degrees)')
    ax.set_ylabel('Opening Angle (degrees)')
    ax.set_title('Opening Angle between e+ and e- vs Decay Plane Angle')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print('donzo')


# -------------------------
# Lorentz boost along z
# -------------------------
def boost_z(p4, beta):
    """
    Boost a 4-vector along z with velocity beta.
    p4 = (E, px, py, pz)
    """
    gamma = 1.0 / np.sqrt(1 - beta**2)

    E, px, py, pz = p4

    E_prime  = gamma * (E + beta * pz)
    pz_prime = gamma * (pz + beta * E)

    return np.array([E_prime, px, py, pz_prime])


# -------------------------
# Simulation
# -------------------------
def simulate_decay(m, E_parent, M_parent, theta_plane):
    """
    m            : mass of decay particles
    E_parent     : total energy of parent in lab frame
    M_parent     : mass of parent particle
    theta_plane  : angle (rad) between decay plane and boost direction
    """

    # Parent properties
    beta = np.sqrt(1 - (M_parent / E_parent)**2)
    gamma = 1 / np.sqrt(1 - beta**2)

    # Decay in parent rest frame
    E_star = M_parent / 2
    p_star = np.sqrt(E_star**2 - m**2)

    # Momentum direction in rest frame
    # Plane rotated by theta_plane relative to z
    p1_star = np.array([
        p_star * np.sin(theta_plane),
        0.0,
        p_star * np.cos(theta_plane)
    ])

    p2_star = -p1_star

    # 4-vectors in rest frame
    p4_1_star = np.array([E_star, *p1_star])
    p4_2_star = np.array([E_star, *p2_star])

    # Boost to lab frame
    p4_1_lab = boost_z(p4_1_star, beta)
    p4_2_lab = boost_z(p4_2_star, beta)

    # Compute opening angle in lab
    p1 = p4_1_lab[1:]
    p2 = p4_2_lab[1:]

    cos_angle = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    return p4_1_lab, p4_2_lab, angle


if __name__ == '__main__':
    main()
