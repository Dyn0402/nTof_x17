#!/usr/bin/env python3
"""
mm_water_grid_lxplus.py — fine Magboltz water grid for the det3 gas fit.

Runs on lxplus with the LCG_107 view (Garfield++ loaded via the ROOT
dictionary, no python Garfield module needed):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
    nohup python3 mm_water_grid_lxplus.py > water_grid.log 2>&1 &

Purpose: the geometry-corrected det3 v(E) (extent-slope estimator, E = HV/3 cm)
matches Ar/iso 95/5 + 1 % H2O at RMS ~0.8 um/ns. This grid brackets the water
fraction finely and tests N2/air co-contamination of the 1 % point.

Output: results/water_grid.json (same schema as the other attachment tables).
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 40.0, 800.0, 12
NCOLL = 5
PRESSURE_TORR = 745.83
TEMP_K = 293.15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'water_grid.json')

MIXTURES = {
    'Ar_iso5_H2O0.6':      [('ar', 94.4), ('ic4h10', 5.0), ('h2o', 0.6)],
    'Ar_iso5_H2O0.8':      [('ar', 94.2), ('ic4h10', 5.0), ('h2o', 0.8)],
    'Ar_iso5_H2O1.2':      [('ar', 93.8), ('ic4h10', 5.0), ('h2o', 1.2)],
    'Ar_iso5_H2O1.5':      [('ar', 93.5), ('ic4h10', 5.0), ('h2o', 1.5)],
    'Ar_iso5_H2O1_N2_1':   [('ar', 93.0), ('ic4h10', 5.0), ('h2o', 1.0), ('n2', 1.0)],
    'Ar_iso5_H2O1_air2':   [('ar', 91.0), ('ic4h10', 5.0), ('h2o', 1.0),
                            ('n2', 1.56), ('o2', 0.42)],
}


def worker(args):
    name, comps = args
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
    gas.GenerateGasTable(NCOLL)
    rows = []
    for e in np.logspace(np.log10(E_MIN), np.log10(E_MAX), 50):
        vx = ctypes.c_double(0.); vy = ctypes.c_double(0.); vz = ctypes.c_double(0.)
        gas.ElectronVelocity(0., 0., -e, 0., 0., 0., vx, vy, vz)
        eta = ctypes.c_double(0.)
        gas.ElectronAttachment(0., 0., -e, 0., 0., 0., eta)
        dl = ctypes.c_double(0.); dt = ctypes.c_double(0.)
        gas.ElectronDiffusion(0., 0., -e, 0., 0., 0., dl, dt)
        rows.append(dict(E_Vcm=float(e), v_um_per_ns=float(vz.value * 1e4),
                         eta_per_cm=float(eta.value),
                         dL_sqrtcm=float(dl.value), dT_sqrtcm=float(dt.value)))
    print(f'{name}: done in {(time.time()-t0)/60:.1f} min', flush=True)
    return name, rows


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with mp.get_context('spawn').Pool(len(MIXTURES)) as pool:
        results = dict(pool.map(worker, list(MIXTURES.items())))
    with open(OUT, 'w') as f:
        json.dump(dict(pressure_torr=PRESSURE_TORR, temp_K=TEMP_K, ncoll=NCOLL,
                       mixtures=results), f, indent=1)
    print(f'Written {OUT}')


if __name__ == '__main__':
    main()
