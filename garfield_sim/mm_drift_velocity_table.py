#!/usr/bin/env python3
"""
mm_drift_velocity_table.py — Magboltz drift velocity vs drift field
====================================================================
Computes electron drift velocity (and diffusion) for Ar/iC4H10 95/5 at
Saclay pressure over the DRIFT-field range (the existing gas tables cover
only the amplification-region interpolation grid).

Used to compare against the cosmic-bench micro-TPC drift-velocity
measurement from the 6-27 det3 drift scan (drift 100–1100 V over a
~20 mm gap → E ≈ 50–550 V/cm).

Output: results/drift_velocity_Ar_iC4H10_95_5_Saclay.json (+ per-point print)

Run with the system python (Garfield env vars are global):
    python3 mm_drift_velocity_table.py
"""
import os
import json
import time
import ctypes

E_MIN_VCM = 25.0
E_MAX_VCM = 1000.0
N_PTS = 18
NCOLL = 5              # ×10^7 collisions; drift velocity converges quickly
PRESSURE_TORR = 745.83  # Saclay 160 m
TEMP_K = 293.15

OUT = os.path.join(os.path.dirname(__file__), 'results',
                   'drift_velocity_Ar_iC4H10_95_5_Saclay.json')


def main():
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield  # noqa: F401

    gas = ROOT.Garfield.MediumMagboltz()
    gas.SetComposition('ar', 95.0, 'ic4h10', 5.0)
    gas.SetTemperature(TEMP_K)
    gas.SetPressure(PRESSURE_TORR)
    gas.SetFieldGrid(E_MIN_VCM, E_MAX_VCM, N_PTS, True)

    t0 = time.time()
    print(f'Running Magboltz: Ar/iC4H10 95/5, {PRESSURE_TORR} Torr, '
          f'{E_MIN_VCM}-{E_MAX_VCM} V/cm, {N_PTS} log pts, ncoll={NCOLL}', flush=True)
    gas.GenerateGasTable(NCOLL)
    print(f'Magboltz done in {(time.time()-t0)/60:.1f} min', flush=True)

    rows = []
    import numpy as np
    for e in np.logspace(np.log10(E_MIN_VCM), np.log10(E_MAX_VCM), 60):
        vx = ctypes.c_double(0.); vy = ctypes.c_double(0.); vz = ctypes.c_double(0.)
        ok = gas.ElectronVelocity(0., 0., -e, 0., 0., 0., vx, vy, vz)
        dl = ctypes.c_double(0.); dt = ctypes.c_double(0.)
        gas.ElectronDiffusion(0., 0., -e, 0., 0., 0., dl, dt)
        # Garfield velocity unit: cm/ns → convert to µm/ns (×1e4)
        v_um_ns = vz.value * 1e4
        rows.append(dict(E_Vcm=float(e), v_um_per_ns=float(v_um_ns),
                         dL_sqrtcm=float(dl.value), dT_sqrtcm=float(dt.value),
                         ok=bool(ok)))
        print(f'  E={e:8.1f} V/cm   v={v_um_ns:7.2f} um/ns', flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(dict(gas='Ar/iC4H10 95/5', pressure_torr=PRESSURE_TORR,
                       temp_K=TEMP_K, ncoll=NCOLL, points=rows), f, indent=1)
    print(f'Written {OUT}', flush=True)


if __name__ == '__main__':
    main()
