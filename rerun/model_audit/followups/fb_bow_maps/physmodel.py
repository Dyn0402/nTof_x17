#!/usr/bin/env python3
"""physmodel.py -- plausibility only: how far can an MX17 readout board bow under its own loads?

The readout board is a 1.70 mm PCB backed by 5 mm Rohacell, screwed between the 30 mm Al gas frame (410 mm aperture)
and an 8 mm Al support plate with a 402 mm through-aperture (MX17_Geant design/GEOMETRY_FROM_CAD.md): behind the active
area there is no aluminium, so the board spans ~0.40 m unsupported.  On the bench it lies horizontal, readout side
down, cathode up.  Kirchhoff plate, square side a, flexural rigidity D = E h^3 / 12 (1 - nu^2) of the PCB alone (the
Rohacell core has ~1/300 of FR4's modulus and no second skin, so it adds little; ignoring it overestimates the bow).
Loads, each with the sign it gives on the bench (+ = centre moves DOWN, away from the cathode):
  gravity            q = rho g t                          (board lying flat; at n_TOF the boards stand vertical: 0)
  gas overpressure   q = dp                               (pushes the board outward = down; varies with gas flow)
  mesh pre-tension   uniform moment m = N e on the mesh face (bulk lamination of a stretched mesh; e = h/2):
                     a free plate takes the spherical curvature kappa = m / (D (1 + nu)); clamped edges suppress it.
Plate coefficients (Timoshenko & Woinowsky-Krieger, square, nu = 0.3): centre deflection w = alpha q a^4 / D with
alpha = 0.00406 (simply supported) / 0.00126 (clamped).  Writes results/physmodel.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import RES

E_FR4, NU, RHO_FR4 = 20e9, 0.3, 1850.0        # Pa, -, kg/m^3 (FR4 in-plane flexural modulus 18-24 GPa)
H_PCB, H_ROH, RHO_ROH = 1.70e-3, 5.0e-3, 52.0
A = 0.402                                     # m, unsupported span (support-plate aperture)
G = 9.81


def main():
    D = E_FR4 * H_PCB ** 3 / (12 * (1 - NU ** 2))
    q_g = (RHO_FR4 * H_PCB + RHO_ROH * H_ROH) * G
    out = dict(D_Nm=D, span_m=A, q_gravity_Pa=q_g)
    for bc, alpha in (('simply_supported', 0.00406), ('clamped', 0.00126)):
        w_g = alpha * q_g * A ** 4 / D
        out[f'gravity_{bc}_mm'] = 1e3 * w_g
        out[f'dp_for_2mm_{bc}_mbar'] = 2e-3 * D / (alpha * A ** 4) / 100.0
        out[f'mm_per_mbar_{bc}'] = 1e3 * alpha * 100.0 * A ** 4 / D
    # mesh pre-tension: N [N/m]; stretched-mesh lamination tensions are ~5-15 N/cm
    for N_cm in (5, 10, 15):
        m = N_cm * 100.0 * H_PCB / 2
        kappa = m / (D * (1 + NU))                 # 1/m, free edges
        out[f'mesh_{N_cm}Ncm_kappa_per_km'] = kappa * 1e3
        out[f'mesh_{N_cm}Ncm_depth_rim150_mm'] = 1e3 * 0.5 * kappa * 0.15 ** 2
    # the measured depth is the dip between the centre and a point 150 mm out; plate shapes (1-D approximations):
    # simply supported ~ cos(pi x / a): dip(150) = 1 - cos(pi 150 / a); clamped ~ (1 + cos(2 pi x / a)) / 2
    import math
    f_ss = 1 - math.cos(math.pi * 0.150 / A); f_cl = 1 - (1 + math.cos(2 * math.pi * 0.150 / A)) / 2
    out['dip150_factor_ss'], out['dip150_factor_cl'] = f_ss, f_cl
    for bc, f in (('simply_supported', f_ss), ('clamped', f_cl)):
        out[f'gravity_{bc}_dip150_mm'] = out[f'gravity_{bc}_mm'] * f
        out[f'dip150_per_mbar_{bc}'] = out[f'mm_per_mbar_{bc}'] * f
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(f'{RES}/physmodel.json', 'w'), indent=1)
    for k, v in out.items():
        print(f'{k:36s} {v:.4g}')


if __name__ == '__main__':
    main()
