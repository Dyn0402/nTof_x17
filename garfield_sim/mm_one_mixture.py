#!/usr/bin/env python3
"""
mm_one_mixture.py <mixture_key> — run a SINGLE Magboltz gas table for one
contaminated Ar/iso 90/10 candidate and write drift_9010_<key>.json.

One mixture per process (one condor job) so a hang/crash in a single gas is
isolated and never blocks the others (the multiprocessing.Pool version
deadlocked when one O2 worker died). CERN pressure, v + eta + diffusion.
"""
import os
import sys
import json
import time
import ctypes

E_MIN, E_MAX, N_GRID = 40.0, 500.0, 14
NCOLL = 5
PRESSURE_TORR = 720.8
TEMP_K = 293.15

MIXTURES = {
    'Ar90_iso10':      [('ar', 90.0),  ('ic4h10', 10.0)],
    'Ar_iso10_H2O0.3': [('ar', 89.7),  ('ic4h10', 10.0), ('h2o', 0.3)],
    'Ar_iso10_H2O0.5': [('ar', 89.5),  ('ic4h10', 10.0), ('h2o', 0.5)],
    'Ar_iso10_H2O1.0': [('ar', 89.0),  ('ic4h10', 10.0), ('h2o', 1.0)],
    'Ar_iso10_H2O1.5': [('ar', 88.5),  ('ic4h10', 10.0), ('h2o', 1.5)],
    'Ar_iso10_H2O2.0': [('ar', 88.0),  ('ic4h10', 10.0), ('h2o', 2.0)],
    'Ar_iso10_H2O3.0': [('ar', 87.0),  ('ic4h10', 10.0), ('h2o', 3.0)],
    'Ar_iso10_air1':   [('ar', 89.01), ('ic4h10', 10.0), ('n2', 0.78), ('o2', 0.21)],
    'Ar_iso10_air2':   [('ar', 88.02), ('ic4h10', 10.0), ('n2', 1.56), ('o2', 0.42)],
    'Ar_iso10_air3':   [('ar', 87.03), ('ic4h10', 10.0), ('n2', 2.34), ('o2', 0.63)],
    'Ar_iso10_O2_0.5': [('ar', 89.5),  ('ic4h10', 10.0), ('o2', 0.5)],
    'Ar_iso10_O2_1.0': [('ar', 89.0),  ('ic4h10', 10.0), ('o2', 1.0)],
    'Ar_iso10_N2_1':   [('ar', 89.0),  ('ic4h10', 10.0), ('n2', 1.0)],
    'Ar_iso10_N2_2':   [('ar', 88.0),  ('ic4h10', 10.0), ('n2', 2.0)],
    'Ar_iso10_N2_5':   [('ar', 85.0),  ('ic4h10', 10.0), ('n2', 5.0)],
}


def main():
    key = sys.argv[1]
    comps = MIXTURES[key]
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    assert ROOT.gSystem.Load('libGarfield') >= 0
    import numpy as np

    gas = ROOT.Garfield.MediumMagboltz()
    flat = [x for pair in comps for x in pair]
    gas.SetComposition(*flat)
    gas.SetTemperature(TEMP_K)
    gas.SetPressure(PRESSURE_TORR)
    gas.SetFieldGrid(E_MIN, E_MAX, N_GRID, True)
    t0 = time.time()
    print(f'{key}: generating {comps}', flush=True)
    gas.GenerateGasTable(NCOLL)
    rows = []
    for e in np.logspace(np.log10(E_MIN), np.log10(E_MAX), 60):
        vx = ctypes.c_double(0.); vy = ctypes.c_double(0.); vz = ctypes.c_double(0.)
        gas.ElectronVelocity(0., 0., -e, 0., 0., 0., vx, vy, vz)
        eta = ctypes.c_double(0.)
        gas.ElectronAttachment(0., 0., -e, 0., 0., 0., eta)
        dl = ctypes.c_double(0.); dt = ctypes.c_double(0.)
        gas.ElectronDiffusion(0., 0., -e, 0., 0., 0., dl, dt)
        rows.append(dict(E_Vcm=float(e), v_um_per_ns=float(vz.value * 1e4),
                         eta_per_cm=float(eta.value),
                         dL_sqrtcm=float(dl.value), dT_sqrtcm=float(dt.value)))
    out = f'drift_9010_{key}.json'
    json.dump(dict(name=key, gas_base='Ar/iC4H10 90/10', pressure_torr=PRESSURE_TORR,
                   temp_K=TEMP_K, ncoll=NCOLL, comps=comps, points=rows),
              open(out, 'w'), indent=1)
    print(f'{key}: done in {(time.time()-t0)/60:.1f} min -> {out}', flush=True)


if __name__ == '__main__':
    main()
