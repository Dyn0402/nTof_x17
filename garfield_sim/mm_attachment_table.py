#!/usr/bin/env python3
"""
mm_attachment_table.py — Magboltz attachment coefficient for humid Ar/iso.

Motivation: the det3 micro-TPC data show the visible drift column ending at
~19.4 mm although the mechanical drift gap is ~30 mm. If the gas is humid
(the v(E) shape already points to 2-2.5 % H2O), electron ATTACHMENT during
drift would progressively kill the signal from the far part of the gap:
survival = exp(-eta * z). This computes eta(E) [1/cm] and v(E) for the
nominal and water-contaminated mixtures over the drift-field range so the
predicted attenuation length lambda = 1/eta can be compared with the
measured per-strip amplitude decay vs drift time.

One process per mixture. Run with system python3 (Garfield env is global):
    python3 mm_attachment_table.py
Output: results/attachment_Ar_iso_H2O.json
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 40.0, 800.0, 12
NCOLL = 5              # attachment is a rare process — needs more statistics
PRESSURE_TORR = 745.83
TEMP_K = 293.15
OUT = os.path.join(os.path.dirname(__file__), 'results', 'attachment_Ar_iso_H2O.json')

MIXTURES = {
    'Ar95_iso5':          [('ar', 95.0), ('ic4h10', 5.0)],
    'Ar94_iso5_H2O1':     [('ar', 94.0), ('ic4h10', 5.0), ('h2o', 1.0)],
    'Ar93_iso5_H2O2':     [('ar', 93.0), ('ic4h10', 5.0), ('h2o', 2.0)],
    'Ar92.5_iso5_H2O2.5': [('ar', 92.5), ('ic4h10', 5.0), ('h2o', 2.5)],
    'Ar92_iso5_H2O3':     [('ar', 92.0), ('ic4h10', 5.0), ('h2o', 3.0)],
}


def worker(args):
    name, comps = args
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    import Garfield  # noqa
    import numpy as np

    gas = ROOT.Garfield.MediumMagboltz()
    if len(comps) == 2:
        gas.SetComposition(comps[0][0], comps[0][1], comps[1][0], comps[1][1])
    else:
        gas.SetComposition(comps[0][0], comps[0][1], comps[1][0], comps[1][1],
                           comps[2][0], comps[2][1])
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
    with mp.get_context('spawn').Pool(len(MIXTURES)) as pool:
        results = dict(pool.map(worker, list(MIXTURES.items())))
    with open(OUT, 'w') as f:
        json.dump(dict(pressure_torr=PRESSURE_TORR, temp_K=TEMP_K, ncoll=NCOLL,
                       mixtures=results), f, indent=1)
    print(f'Written {OUT}')


if __name__ == '__main__':
    main()
