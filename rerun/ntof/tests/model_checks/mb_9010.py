#!/usr/bin/env python3
"""mb_9010.py -- Magboltz v, D_T, D_L for the n_TOF gas: Ar/iC4H10 90/10 with water replacing argon.

    source garfield_sim/setup_garfield.sh
    python3 mb_9010.py <water %> <E V/cm> [ncoll] [--p 720.8] [--out DIR]

One point per process (the cloud_basics magboltz_drift.py recipe: MediumMagboltz, 293.15 K, GenerateGasTable(ncoll)).
CERN pressure 720.8 Torr as garfield_sim/mm_drift_9010_contam_cern.py.  Output <out>/mb9010_w<water>_E<E>_c<n>.json.
"""
import argparse, ctypes, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument('water', type=float); ap.add_argument('E', type=float); ap.add_argument('ncoll', type=int, nargs='?', default=10)
ap.add_argument('--p', type=float, default=720.8)
ap.add_argument('--out', default=os.path.join(os.environ.get('WORK_DATA', '/tmp'), 'ntof_pass1/model_checks/magboltz'))
a = ap.parse_args()
import ROOT
ROOT.gROOT.SetBatch(True)
import Garfield  # noqa
t = time.time()
g = ROOT.Garfield.MediumMagboltz()
comp = [('ar', 90.0 - a.water), ('ic4h10', 10.0)] + ([('h2o', a.water)] if a.water > 0 else [])
args = []
for n, f in comp:
    args += [n, f]
g.SetComposition(*args)
g.SetTemperature(293.15)
g.SetPressure(a.p)
g.SetFieldGrid(a.E, a.E, 1, False)
g.GenerateGasTable(a.ncoll)
vx, vy, vz = (ctypes.c_double() for _ in range(3))
g.ElectronVelocity(0, 0, -a.E, 0, 0, 0, vx, vy, vz)
dl, dt = ctypes.c_double(), ctypes.c_double()
g.ElectronDiffusion(0, 0, -a.E, 0, 0, 0, dl, dt)
res = dict(water=a.water, E_Vcm=a.E, pressure_torr=a.p, ncoll=a.ncoll, comp=comp, v_um_ns=abs(vz.value) * 1e4,
           DT_um_rtcm=dt.value * 1e4, DL_um_rtcm=dl.value * 1e4, cpu_s=time.time() - t)
os.makedirs(a.out, exist_ok=True)
f = lambda x: f'{x:g}'.replace('.', 'p')
json.dump(res, open(os.path.join(a.out, f'mb9010_w{f(a.water)}_E{f(a.E)}_c{a.ncoll}.json'), 'w'), indent=1)
print(json.dumps(res))
