#!/usr/bin/env python3
"""mcs_common.py -- materials, geometry and multiple-scattering formulas shared by make_pointing.py,
make_gas_scatter.py and make_material_budget.py (Paper I v2, 2026-11-10).

Material constants are PDG "Atomic and nuclear properties" values (radiation length in g/cm^2, density in g/cm^3 at
20 C and 1 atm for the gases). The chamber stack is docs/detector_spec/spec.py (the 9 um cathode copper is marked
"assumed" there) with the drift gap at its measured 28.2 mm (chamber A, section 6 of Paper I).

Highland (PDG eq. 34.15, the 2-digit 0.038 form) is quoted by the PDG as accurate to 11 % for 1e-3 < x/X0 < 100.
The drift gas is 2.4e-4 X0, below that range, and the formula describes only the Gaussian core: single (Rutherford)
scatters, which give the tails, are not modelled anywhere in these toys. lynch_dahl() is the Moliere-based Gaussian
width of Lynch & Dahl (NIM B58 (1991) 6, eq. 7, F = 0.98) used as a cross-check.
"""
from __future__ import annotations

import numpy as np

ME = 0.51099895          # electron mass [MeV]
MMU = 105.6584           # muon mass [MeV]
ALPHA = 1 / 137.035999

# --- materials: X0 [g/cm^2], density [g/cm^3], elemental composition by mass {Z: (A, w)} --------------------------
MAT = {
    'air':    dict(X0=36.62, rho=1.205e-3, label='Air (20 C, 1 atm)',
                   el={7: (14.007, 0.7553), 8: (15.999, 0.2318), 18: (39.948, 0.0128), 6: (12.011, 0.0001)}),
    'Ar':     dict(X0=19.55, rho=1.662e-3, el={18: (39.948, 1.0)}),
    'iC4H10': dict(X0=45.23, rho=2.489e-3, el={6: (12.011, 0.8266), 1: (1.008, 0.1734)}),
    'mylar':  dict(X0=39.95, rho=1.40, label='Mylar (PET)',
                   el={1: (1.008, 0.0420), 6: (12.011, 0.6250), 8: (15.999, 0.3330)}),
    'kapton': dict(X0=40.58, rho=1.42, label='Kapton (polyimide)',
                   el={1: (1.008, 0.0264), 6: (12.011, 0.6911), 7: (14.007, 0.0733), 8: (15.999, 0.2092)}),
    'Cu':     dict(X0=12.86, rho=8.96, label='Copper', el={29: (63.546, 1.0)}),
}


def gas_mix(f_ar=0.95, f_iso=0.05):
    """Ar/iC4H10 by volume (ideal gas): density, X0 (1/X0 = sum w_i/X0_i by mass) and composition."""
    rho = f_ar * MAT['Ar']['rho'] + f_iso * MAT['iC4H10']['rho']
    w_ar = f_ar * MAT['Ar']['rho'] / rho
    w_iso = 1 - w_ar
    X0 = 1.0 / (w_ar / MAT['Ar']['X0'] + w_iso / MAT['iC4H10']['X0'])
    el = {18: (39.948, w_ar)}
    for Z, (A, w) in MAT['iC4H10']['el'].items():
        el[Z] = (A, w * w_iso)
    return dict(X0=X0, rho=rho, el=el, w_ar=w_ar, w_iso=w_iso, label='Ar/iC$_4$H$_{10}$ 95/5 (20 C, 1 atm)')


MAT['gas'] = gas_mix()


def X0_cm(m):
    return MAT[m]['X0'] / MAT[m]['rho']


# --- geometry along the normal from the target [mm] ---------------------------------------------------------------
D_WIN = 200.0            # target -> window front face
T_WIN = 0.060            # Mylar
T_GAP5 = 5.0             # window -> cathode gas gap
T_KAP, T_CU = 0.050, 0.009   # cathode foil (Cu: assumed thickness, faces the drift gas)
G_DRIFT = 28.2           # measured drift gap (nominal 30.000)
Z_CATH = D_WIN + T_WIN + T_GAP5                    # cathode front face
Z_DRIFT0 = Z_CATH + T_KAP + T_CU                   # start of the drift gas
Z_READOUT = Z_DRIFT0 + G_DRIFT                     # mesh / readout plane: where the position is measured
ACT_X, ACT_Y = 399.4, 359.9                        # efficient active area [mm] (Paper I, readout figure)

# layers in front of the drift gas: (key, label, material, thickness mm, z_start mm, assumed?)
LAYERS = [
    ('air', 'Air, target to window', 'air', D_WIN, 0.0, False),
    ('window', 'Entrance window (Mylar)', 'mylar', T_WIN, D_WIN, False),
    ('gap5', 'Gas gap, window to cathode', 'gas', T_GAP5, D_WIN + T_WIN, False),
    ('kapton', 'Drift cathode, Kapton', 'kapton', T_KAP, Z_CATH, False),
    ('cu', 'Drift cathode, copper', 'Cu', T_CU, Z_CATH + T_KAP, True),
]
DRIFT = ('drift', 'Drift gas', 'gas', G_DRIFT, Z_DRIFT0, False)


def kin(T, m=ME):
    """Kinetic energy [MeV] -> (p [MeV/c], beta)."""
    E = T + m
    p = np.sqrt(E * E - m * m)
    return p, p / E


def highland(x_over_X0, T, m=ME, z=1):
    """PDG/Highland theta0 [rad], plane-projected rms, for kinetic energy T (MeV): momentum, not T, enters."""
    p, b = kin(T, m)
    x = np.asarray(x_over_X0, float)
    return 13.6 / (b * p) * abs(z) * np.sqrt(x) * (1 + 0.038 * np.log(x * z * z / (b * b)))


def lynch_dahl(mat, t_mm, T, m=ME, z=1, F=0.98):
    """Lynch & Dahl (1991) Moliere-based Gaussian width [rad] of the central fraction F (their eq. 7, mixtures by
    their eq. 8-9). mat: key of MAT. t_mm: thickness."""
    p, b = kin(T, m)
    X = MAT[mat]['rho'] * t_mm / 10.0           # g/cm^2
    el = MAT[mat]['el']
    chic2 = 0.0
    lnchia_num = 0.0
    for Z, (A, w) in el.items():
        c = 0.157 * Z * (Z + 1) * w * X / A * (z / (p * b)) ** 2
        chia2 = 2.007e-5 * Z ** (2 / 3) * (1 + 3.34 * (Z * z * ALPHA / b) ** 2) / p ** 2
        chic2 += c
        lnchia_num += c * np.log(chia2)
    chia2 = np.exp(lnchia_num / chic2)
    omega = chic2 / (1.167 * chia2)
    v = 0.5 * omega / (1 - F)
    return float(np.sqrt(chic2 / (1 + F * F) * ((1 + v) / v * np.log(1 + v) - 1))), float(omega)


def x_over_X0(mat, t_mm):
    return t_mm / (10.0 * X0_cm(mat))
