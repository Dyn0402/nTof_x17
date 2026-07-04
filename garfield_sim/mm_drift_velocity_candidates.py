#!/usr/bin/env python3
"""
mm_drift_velocity_candidates.py — test gas hypotheses against the measured
det3 drift-velocity curve.

The measured v(E) (6-27 drift scan, gap 19.4 mm) is far below the Magboltz
prediction for pure Ar/iC4H10 95/5 and rises monotonically where the pure mix
should already be past its peak. Candidate explanations, each run through
Magboltz over the drift-field range:

  - Ar/iC4H10 95/5 + H2O contamination (0.3 / 1 / 2 %)
  - Ar/CO2 90/10 and 80/20 (wrong bottle / mislabeled line)

One process per mixture (coarse ncoll=2 — drift velocity converges fast).
Output: results/drift_velocity_candidates.json

Run: python3 mm_drift_velocity_candidates.py
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 25.0, 1000.0, 15
NCOLL = 2
PRESSURE_TORR = 745.83
TEMP_K = 293.15
OUT = os.path.join(os.path.dirname(__file__), 'results', 'drift_velocity_candidates.json')

CANDIDATES = {
    'Ar95_iso5_H2O0.3': [('ar', 94.7), ('ic4h10', 5.0), ('h2o', 0.3)],
    'Ar94_iso5_H2O1':   [('ar', 94.0), ('ic4h10', 5.0), ('h2o', 1.0)],
    'Ar93_iso5_H2O2':   [('ar', 93.0), ('ic4h10', 5.0), ('h2o', 2.0)],
    'Ar90_CO2_10':      [('ar', 90.0), ('co2', 10.0)],
    'Ar80_CO2_20':      [('ar', 80.0), ('co2', 20.0)],
}

# Round 2: quencher-rich mixtures (mixing-station error hypothesis) + finer H2O
CANDIDATES2 = {
    'Ar90_iso10':         [('ar', 90.0), ('ic4h10', 10.0)],
    'Ar85_iso15':         [('ar', 85.0), ('ic4h10', 15.0)],
    'Ar80_iso20':         [('ar', 80.0), ('ic4h10', 20.0)],
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
        rows.append(dict(E_Vcm=float(e), v_um_per_ns=float(vz.value * 1e4)))
    print(f'{name}: done in {(time.time()-t0)/60:.1f} min', flush=True)
    return name, rows


def main():
    import sys
    cands, out = (CANDIDATES2, OUT.replace('.json', '2.json')) \
        if '--round2' in sys.argv else (CANDIDATES, OUT)
    with mp.get_context('spawn').Pool(len(cands)) as pool:
        results = dict(pool.map(worker, list(cands.items())))
    with open(out, 'w') as f:
        json.dump(dict(pressure_torr=PRESSURE_TORR, temp_K=TEMP_K, ncoll=NCOLL,
                       mixtures=results), f, indent=1)
    print(f'Written {out}')


if __name__ == '__main__':
    main()
