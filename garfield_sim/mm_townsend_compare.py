#!/usr/bin/env python3
"""
mm_townsend_compare.py — can the observed gain turn-on discriminate
Ar/iso 95/5 from Ar/CO2 90/10?

The det3 resist-HV scan shows efficiency turn-on at 425 V and plateau by
455 V over a 150 um amplification gap. This computes the effective Townsend
coefficient alpha(E) for both gases at amplification fields and the implied
mesh gain G = exp(alpha * gap), to check where each gas would turn on.

Run with system python3:  python3 mm_townsend_compare.py
Output: results/townsend_compare.json
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 15000.0, 60000.0, 10
NCOLL = 3
PRESSURE_TORR = 745.83
TEMP_K = 293.15
AMP_GAP_CM = 0.0150
OUT = os.path.join(os.path.dirname(__file__), 'results', 'townsend_compare.json')

MIXTURES = {
    'Ar95_iso5':   [('ar', 95.0), ('ic4h10', 5.0)],
    'Ar90_CO2_10': [('ar', 90.0), ('co2', 10.0)],
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
    flat = [x for pair in comps for x in pair]
    gas.SetComposition(*flat)
    gas.SetTemperature(TEMP_K)
    gas.SetPressure(PRESSURE_TORR)
    gas.SetFieldGrid(E_MIN, E_MAX, N_GRID, True)
    t0 = time.time()
    gas.GenerateGasTable(NCOLL)
    rows = []
    for e in np.logspace(np.log10(E_MIN), np.log10(E_MAX), 40):
        alpha = ctypes.c_double(0.)
        gas.ElectronTownsend(0., 0., -e, 0., 0., 0., alpha)
        eta = ctypes.c_double(0.)
        gas.ElectronAttachment(0., 0., -e, 0., 0., 0., eta)
        a_eff = alpha.value - eta.value
        rows.append(dict(E_Vcm=float(e), V_at_150um=float(e * AMP_GAP_CM),
                         alpha_per_cm=float(alpha.value), eta_per_cm=float(eta.value),
                         gain=float(np.exp(max(a_eff, 0.0) * AMP_GAP_CM))))
    print(f'{name}: done in {(time.time()-t0)/60:.1f} min', flush=True)
    return name, rows


def main():
    with mp.get_context('spawn').Pool(len(MIXTURES)) as pool:
        results = dict(pool.map(worker, list(MIXTURES.items())))
    with open(OUT, 'w') as f:
        json.dump(dict(pressure_torr=PRESSURE_TORR, temp_K=TEMP_K, ncoll=NCOLL,
                       amp_gap_cm=AMP_GAP_CM, mixtures=results), f, indent=1)
    for name, rows in results.items():
        for target in (100.0, 1000.0, 10000.0):
            v = next((r['V_at_150um'] for r in rows if r['gain'] >= target), None)
            print(f'{name}: gain {target:>7.0f} at V = {v if v else ">900"}')
    print(f'Written {OUT}')


if __name__ == '__main__':
    main()
