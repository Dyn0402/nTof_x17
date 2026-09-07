#!/usr/bin/env python3
"""
wetco2_one_mixture.py <mixture_key> -- wet bracket for the SPS CO2-epoch
gas Ar/CO2/iC4H10 95/3/2 (water displacing argon), CERN pressure.
Writes drift_wetco2_<key>.json. Cloned 2026-08-11 from
mm_one_mixture.py to test whether ONE water fraction explains BOTH
beam gases measured v_drift (CF4 epoch already bracketed 1-2%).
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
    'co2dry':     [('ar', 95.0), ('co2', 3.0), ('ic4h10', 2.0)],
    'co2_H2O0.5': [('ar', 94.5), ('co2', 3.0), ('ic4h10', 2.0), ('h2o', 0.5)],
    'co2_H2O1.0': [('ar', 94.0), ('co2', 3.0), ('ic4h10', 2.0), ('h2o', 1.0)],
    'co2_H2O1.5': [('ar', 93.5), ('co2', 3.0), ('ic4h10', 2.0), ('h2o', 1.5)],
    'co2_H2O2.0': [('ar', 93.0), ('co2', 3.0), ('ic4h10', 2.0), ('h2o', 2.0)],
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
    out = f'drift_wetco2_{key}.json'
    json.dump(dict(name=key, gas_base='Ar/iC4H10 90/10', pressure_torr=PRESSURE_TORR,
                   temp_K=TEMP_K, ncoll=NCOLL, comps=comps, points=rows),
              open(out, 'w'), indent=1)
    print(f'{key}: done in {(time.time()-t0)/60:.1f} min -> {out}', flush=True)


if __name__ == '__main__':
    main()
