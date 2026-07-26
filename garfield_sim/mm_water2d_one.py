#!/usr/bin/env python3
"""
mm_water2d_one.py <tag> — one Magboltz table for the 2026-07-25 water/iso 2-D
grid matching the det3 forward-fit v(E). Tag encodes the mixture, e.g.

    iso5.0_h2o0.45_n20.0_o20.0

Ar = 100 - iso - h2o - n2 - o2. Output: water2d_<tag>.json.
Same conditions as water_grid.json: 745.83 torr, 293.15 K, NCOLL=5.
"""
import sys
import json
import time
import ctypes
import re

E_MIN, E_MAX, N_GRID = 30.0, 500.0, 14
NCOLL = 5
PRESSURE_TORR = 745.83
TEMP_K = 293.15


def parse_tag(tag):
    m = re.fullmatch(r'iso([\d.]+)_h2o([\d.]+)_n2([\d.]+)_o2([\d.]+)', tag)
    if not m:
        raise SystemExit(f'bad tag {tag}')
    iso, h2o, n2, o2 = (float(g) for g in m.groups())
    ar = 100.0 - iso - h2o - n2 - o2
    comps = [('ar', ar), ('ic4h10', iso)]
    if h2o > 0:
        comps.append(('h2o', h2o))
    if n2 > 0:
        comps.append(('n2', n2))
    if o2 > 0:
        comps.append(('o2', o2))
    return comps


def main():
    tag = sys.argv[1]
    comps = parse_tag(tag)
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
    print(f'{tag}: generating {comps}', flush=True)
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
    json.dump(dict(name=tag, pressure_torr=PRESSURE_TORR, temp_K=TEMP_K,
                   ncoll=NCOLL, comps=comps, points=rows),
              open(f'water2d_{tag}.json', 'w'), indent=1)
    print(f'{tag}: done in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
