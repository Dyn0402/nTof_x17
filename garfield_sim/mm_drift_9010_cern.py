#!/usr/bin/env python3
"""Drift velocity vs field for Ar/iC4H10 90/10 at CERN pressure (nTOF EAR2).
Covers the DRIFT-field operating range for a 30 mm gap at -500..-900 V
(E = 167..300 V/cm), with margin. Output: results/drift_velocity_Ar_iC4H10_90_10_CERN.json
"""
import os, json, time, ctypes

E_MIN_VCM, E_MAX_VCM, N_PTS = 80.0, 500.0, 16
NCOLL = 5
PRESSURE_TORR = 720.8   # CERN 450 m
TEMP_K = 293.15
OUT = os.path.join(os.path.dirname(__file__), 'results',
                   'drift_velocity_Ar_iC4H10_90_10_CERN.json')

def main():
    import ROOT, numpy as np
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield  # noqa
    gas = ROOT.Garfield.MediumMagboltz()
    gas.SetComposition('ar', 90.0, 'ic4h10', 10.0)
    gas.SetTemperature(TEMP_K); gas.SetPressure(PRESSURE_TORR)
    gas.SetFieldGrid(E_MIN_VCM, E_MAX_VCM, N_PTS, True)
    t0 = time.time()
    print(f'Magboltz Ar/iso 90/10 @ {PRESSURE_TORR} Torr, {E_MIN_VCM}-{E_MAX_VCM} V/cm', flush=True)
    gas.GenerateGasTable(NCOLL)
    print(f'done in {(time.time()-t0)/60:.1f} min', flush=True)
    rows = []
    for e in np.logspace(np.log10(E_MIN_VCM), np.log10(E_MAX_VCM), 60):
        vx=ctypes.c_double(0.); vy=ctypes.c_double(0.); vz=ctypes.c_double(0.)
        gas.ElectronVelocity(0.,0.,-e,0.,0.,0.,vx,vy,vz)
        dl=ctypes.c_double(0.); dt=ctypes.c_double(0.)
        gas.ElectronDiffusion(0.,0.,-e,0.,0.,0.,dl,dt)
        rows.append(dict(E_Vcm=float(e), v_um_per_ns=float(vz.value*1e4),
                         dL_sqrtcm=float(dl.value), dT_sqrtcm=float(dt.value)))
        print(f'  E={e:7.1f} V/cm  v={vz.value*1e4:6.2f} um/ns', flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(gas='Ar/iC4H10 90/10', pressure_torr=PRESSURE_TORR, temp_K=TEMP_K,
                   ncoll=NCOLL, points=rows), open(OUT,'w'), indent=1)
    print('Written', OUT, flush=True)

if __name__ == '__main__':
    main()
