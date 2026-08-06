#!/usr/bin/env python3
"""
mm_water_grid2_local.py — fine Magboltz water/iso grid to match the
FORWARD-FIT det3 v(E) curve (waveform_first_threading, 2026-07-25):

    HV [V]      300   500   700   900   1000  1100
    v [um/ns]   12.0  20.6  26.4  35.5  36.6  38.8   (E = HV / 3.0 cm)

The existing grid already brackets the optimum: 95/5 + 0.8 % H2O fits at
RMS 1.87 (all points) / 0.67 um/ns (>=700 V); 0.6 % and 1.0 % are clearly
worse. This grid refines 0.65-0.9 % and tests the iso-ratio degeneracy.

Runs LOCALLY (no lxplus); setup_garfield.sh picks up the laptop install:

    source setup_garfield.sh
    nohup python3 mm_water_grid2_local.py > water_grid2.log 2>&1 &

Output: results/water_grid2.json (same schema as water_grid.json).
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 30.0, 500.0, 14
NCOLL = 5
PRESSURE_TORR = 745.83     # same convention as water_grid.json / candidates
TEMP_K = 293.15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   'water_grid2.json')

MIXTURES = {
    # refine the water bracket at 95/5 (0.6 and 0.8 already exist)
    'Ar_iso5_H2O0.65':   [('ar', 94.35), ('ic4h10', 5.0), ('h2o', 0.65)],
    'Ar_iso5_H2O0.7':    [('ar', 94.3),  ('ic4h10', 5.0), ('h2o', 0.7)],
    'Ar_iso5_H2O0.75':   [('ar', 94.25), ('ic4h10', 5.0), ('h2o', 0.75)],
    'Ar_iso5_H2O0.85':   [('ar', 94.15), ('ic4h10', 5.0), ('h2o', 0.85)],
    'Ar_iso5_H2O0.9':    [('ar', 94.1),  ('ic4h10', 5.0), ('h2o', 0.9)],
    # iso-ratio degeneracy at fixed water
    'Ar_iso4_H2O0.6':    [('ar', 95.4),  ('ic4h10', 4.0), ('h2o', 0.6)],
    'Ar_iso4_H2O0.8':    [('ar', 95.2),  ('ic4h10', 4.0), ('h2o', 0.8)],
    'Ar_iso6_H2O0.6':    [('ar', 93.4),  ('ic4h10', 6.0), ('h2o', 0.6)],
    'Ar_iso6_H2O0.8':    [('ar', 93.2),  ('ic4h10', 6.0), ('h2o', 0.8)],
    'Ar_iso7_H2O0.8':    [('ar', 92.2),  ('ic4h10', 7.0), ('h2o', 0.8)],
    # mild N2 co-contamination of the best point
    'Ar_iso5_H2O0.8_N2_0.5': [('ar', 93.7), ('ic4h10', 5.0), ('h2o', 0.8),
                              ('n2', 0.5)],
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
    print(f'{name}: generating', flush=True)
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
    print(f'{name}: done in {(time.time()-t0)/60:.1f} min', flush=True)
    return name, rows


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with mp.Pool(len(MIXTURES)) as pool:
        results = dict(pool.map(worker, list(MIXTURES.items())))
    json.dump(dict(pressure_torr=PRESSURE_TORR, temp_K=TEMP_K, ncoll=NCOLL,
                   e_grid=[E_MIN, E_MAX, N_GRID],
                   comps={k: v for k, v in MIXTURES.items()},
                   mixtures=results),
              open(OUT, 'w'), indent=1)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
