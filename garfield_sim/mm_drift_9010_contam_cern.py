#!/usr/bin/env python3
"""
mm_drift_9010_contam_cern.py — contaminated-gas Magboltz suite for the JULY
BEAM detectors (Ar/iC4H10 90/10 at CERN EAR2 pressure).

Motivation: run_58 singles drift scan (analyze_drift.py) measures, on the clean
Det A with the nominal 30 mm gap, v_drift ~ 35.6-35.8 um/ns at E = 200-233 V/cm,
which is ~12-16% BELOW pure Ar/iso 90/10 Magboltz (40.5 / 42.6 um/ns) and
plateaus where the pure mix keeps climbing. This is the same suppression seen in
the June det3 cosmics, where the best fit was Ar/iso 95/5 + ~1% H2O. The line has
been flushing a long time, so the question is whether the residual is water
outgassing, an air leak (N2 + O2), or O2 permeation.

This is the 90/10 analogue of the 95/5 contamination trio
(mm_water_grid_lxplus / mm_attachment_air_candidates / mm_drift_velocity_candidates):
one Magboltz table per candidate, reporting v_drift, attachment eta, and
longitudinal/transverse diffusion vs field, so v(E) fixes the contaminant
FRACTION and eta(E) DISCRIMINATES water (weak attachment) from air/O2 (strong).

Pressure = CERN 450 m (720.8 Torr) to match the July beam and the existing
results/drift_velocity_Ar_iC4H10_90_10_CERN.json pure reference.

Run on lxplus:
    source setup_garfield.sh
    nohup python3 mm_drift_9010_contam_cern.py > contam_9010.log 2>&1 &

Output: results/drift_9010_contam_cern.json (same schema as water_grid.json).
"""
import os
import json
import time
import ctypes
import multiprocessing as mp

E_MIN, E_MAX, N_GRID = 40.0, 500.0, 14   # covers drift 200-700 V over a 3 cm gap
NCOLL = 5                                 # accurate enough for eta as well as v
PRESSURE_TORR = 720.8                     # CERN EAR2, 450 m
TEMP_K = 293.15
NWORKERS = 8    # match the condor request_cpus=8 slot
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                   'drift_9010_contam_cern.json')

# Every candidate keeps the quencher at iso 10%; the contaminant displaces Ar.
# Air is dry air split 78/21 N2/O2 (the ~1% Ar in air is folded into the bulk Ar).
MIXTURES = {
    # --- pure reference ---
    'Ar90_iso10':          [('ar', 90.0),  ('ic4h10', 10.0)],
    # --- water outgassing / permeation (primary hypothesis) ---
    'Ar_iso10_H2O0.3':     [('ar', 89.7),  ('ic4h10', 10.0), ('h2o', 0.3)],
    'Ar_iso10_H2O0.5':     [('ar', 89.5),  ('ic4h10', 10.0), ('h2o', 0.5)],
    'Ar_iso10_H2O1.0':     [('ar', 89.0),  ('ic4h10', 10.0), ('h2o', 1.0)],
    'Ar_iso10_H2O1.5':     [('ar', 88.5),  ('ic4h10', 10.0), ('h2o', 1.5)],
    'Ar_iso10_H2O2.0':     [('ar', 88.0),  ('ic4h10', 10.0), ('h2o', 2.0)],
    'Ar_iso10_H2O3.0':     [('ar', 87.0),  ('ic4h10', 10.0), ('h2o', 3.0)],
    # --- air leak (N2 + O2 together, 78/21) ---
    'Ar_iso10_air1':       [('ar', 89.01), ('ic4h10', 10.0), ('n2', 0.78), ('o2', 0.21)],
    'Ar_iso10_air2':       [('ar', 88.02), ('ic4h10', 10.0), ('n2', 1.56), ('o2', 0.42)],
    'Ar_iso10_air3':       [('ar', 87.03), ('ic4h10', 10.0), ('n2', 2.34), ('o2', 0.63)],
    # --- O2 permeation alone (attachment discriminator) ---
    'Ar_iso10_O2_0.5':     [('ar', 89.5),  ('ic4h10', 10.0), ('o2', 0.5)],
    'Ar_iso10_O2_1.0':     [('ar', 89.0),  ('ic4h10', 10.0), ('o2', 1.0)],
    # --- N2 alone (outgassing / permeation, no attachment) ---
    'Ar_iso10_N2_1':       [('ar', 89.0),  ('ic4h10', 10.0), ('n2', 1.0)],
    'Ar_iso10_N2_2':       [('ar', 88.0),  ('ic4h10', 10.0), ('n2', 2.0)],
    'Ar_iso10_N2_5':       [('ar', 85.0),  ('ic4h10', 10.0), ('n2', 5.0)],
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
    nproc = min(NWORKERS, len(MIXTURES))
    print(f'{len(MIXTURES)} mixtures on {nproc} workers, '
          f'Ar/iso 90/10 base @ {PRESSURE_TORR} Torr, ncoll={NCOLL}', flush=True)
    with mp.get_context('spawn').Pool(nproc) as pool:
        results = dict(pool.map(worker, list(MIXTURES.items())))
    with open(OUT, 'w') as f:
        json.dump(dict(gas_base='Ar/iC4H10 90/10', pressure_torr=PRESSURE_TORR,
                       temp_K=TEMP_K, ncoll=NCOLL, mixtures=results), f, indent=1)
    print(f'Written {OUT}')


if __name__ == '__main__':
    main()
