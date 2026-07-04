#!/usr/bin/env python3
"""
mm_attachment_air_candidates.py — air-leak and wrong-bottle hypotheses.

Under the 30 mm gap interpretation (E = HV/3 cm) the measured v(E) is best
matched by Ar/CO2 90/10 or Ar/iso 95/5 + ~1 % H2O. The measured amplitude
decay with drift depth (lambda ~ 13-15 mm) requires real electron attachment,
which discriminates them: clean Ar/CO2 attaches ~nothing at these fields,
while O2 (which enters together with H2O in any air leak) attaches strongly.

Mixtures: Ar/iso 95/5 + 1 / 2 % dry air (78/21 N2/O2), + 0.5 % O2 alone,
and Ar/CO2 90/10 (v + eta reference for the wrong-bottle hypothesis).

Run with system python3:  python3 mm_attachment_air_candidates.py
Output: results/attachment_air_candidates.json
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
OUT = os.path.join(os.path.dirname(__file__), 'results', 'attachment_air_candidates.json')

MIXTURES = {
    'Ar_iso5_air1':  [('ar', 94.0), ('ic4h10', 5.0), ('n2', 0.78), ('o2', 0.21)],
    'Ar_iso5_air2':  [('ar', 93.0), ('ic4h10', 5.0), ('n2', 1.56), ('o2', 0.42)],
    'Ar_iso5_O2half': [('ar', 94.5), ('ic4h10', 5.0), ('o2', 0.5)],
    'Ar90_CO2_10':   [('ar', 90.0), ('co2', 10.0)],
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
