#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heed_edep.py -- the ionisation TRUTH for the MM energy-deposit study
(ntof_calorimetry/MM_EDEP.md): what one 30 mm Ar/iC4H10 90/10 drift gap
(n_TOF gas, ~1 atm) collects from a charged particle, track by track.

Heed (Garfield++ TrackHeed) is used, not Geant4: the gap is ~5 mg/cm^2, where
the energy-loss straggling is set by the atomic shell structure (the Landau /
Vavilov / Bichsel regime), which Heed's photo-absorption-ionisation model
treats and G4 opt4's universal fluctuation model only approximates.

Per (particle, T, theta) it stores, per track:
  e_dep     energy lost in the gap (sum of cluster energies, eV)  [the truth]
  n_e       conduction electrons produced (delta rays included)
  n_clu     primary clusters
  prof      n_e per 0.5 mm of depth (60 bins), for the digitiser
Depth is measured from the drift cathode (z = 0) toward the mesh (z = 30 mm);
the track enters at x = 0 and moves along +x by tan(theta) per mm of depth.

Runs on lxplus (PyROOT + the pinned Garfield++):
    source garfield_sim/setup_garfield.sh
    python3 heed_edep.py --out <dir> [--n 5000] [--only e-_4]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

GAP_CM = 3.0
NBIN = 60
#: (particle, kinetic energy MeV): electrons across the X17 / background
#: range plus the slow end where dE/dx rises; cosmic muons as the reference
CONFIGS = ([('e-', T) for T in (0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)]
           + [('mu-', 3000.0)])
THETAS = (0.0, 30.0, 45.0)
#: continuous mode: |tan| of the track up to this (data tracks reach ~0.8)
TAN_MAX = 0.9


def setup(pressure_torr: float):
    import ROOT  # noqa: F401
    import Garfield  # noqa: F401
    G = ROOT.Garfield
    gas = G.MediumMagboltz('ar', 90., 'ic4h10', 10.)
    gas.SetTemperature(293.15)
    gas.SetPressure(pressure_torr)
    box = G.SolidBox(0, 0, GAP_CM / 2, 200, 200, GAP_CM / 2)
    geo = G.GeometrySimple()
    geo.AddSolid(box, gas)
    cmp_ = G.ComponentConstant()
    cmp_.SetGeometry(geo)
    cmp_.SetElectricField(0, 0, 300)
    sens = G.Sensor()
    sens.AddComponent(cmp_)
    tr = G.TrackHeed(sens)
    return G, gas, sens, tr, (box, geo, cmp_)


def run_config(tr, particle: str, T_mev: float, theta, n: int, seed_note: str):
    """``theta``: one angle in degrees, or None for a continuous angle per
    track (tan uniform in 0 .. TAN_MAX), stored, for matching data tracks."""
    tr.SetParticle(particle)
    tr.SetKineticEnergy(T_mev * 1e6)
    rng = np.random.default_rng(abs(hash((particle, T_mev, seed_note))) % (1 << 31))
    ths = (np.full(n, np.radians(theta)) if theta is not None
           else np.arctan(rng.uniform(0, TAN_MAX, n)))
    e_dep = np.zeros(n, np.float32)
    n_e = np.zeros(n, np.int32)
    n_clu = np.zeros(n, np.int32)
    prof = np.zeros((n, NBIN), np.int16)
    for i in range(n):
        tr.NewTrack(0., 0., 0., 0., np.sin(ths[i]), 0., np.cos(ths[i]))
        e, ne, nc = 0.0, 0, 0
        for c in tr.GetClusters():
            k = len(c.electrons)
            e += c.energy
            nc += 1
            ne += k
            for el in c.electrons:
                b = int(el.z / GAP_CM * NBIN)
                if 0 <= b < NBIN:
                    prof[i, b] += 1
        e_dep[i], n_e[i], n_clu[i] = e, ne, nc
    return dict(e_dep=e_dep, n_e=n_e, n_clu=n_clu, prof=prof, theta_trk=np.degrees(ths).astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--pressure', type=float, default=730.0,
                    help='torr; n_TOF hall ~ 970 mbar')
    ap.add_argument('--only', default='', help='e.g. e-_4 or mu-_3000 (comma list)')
    ap.add_argument('--theta', default='', help='comma list of theta in deg (default all)')
    ap.add_argument('--continuous', action='store_true', help='one random angle per track (file tag thc)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G, gas, sens, tr, keep = setup(a.pressure)
    import ROOT
    ROOT.gRandom.SetSeed(12345 + os.getpid())   # Garfield's default engine is ROOT's
    only = {s for s in a.only.split(',') if s}
    thetas = [float(x) for x in a.theta.split(',') if x] or THETAS
    if a.continuous:
        thetas = [None]
    for p, T in CONFIGS:
        tag = f'{p}_{T:g}'
        if only and tag not in only:
            continue
        for th in thetas:
            f = os.path.join(a.out, f'{tag}_th{"c" if th is None else f"{th:g}"}.npz')
            if os.path.exists(f):
                continue
            t = time.time()
            r = run_config(tr, p, T, th, a.n, tag)
            np.savez_compressed(f, particle=p, T=T, theta=np.nan if th is None else th,
                                pressure_torr=a.pressure, **r)
            print(f'{tag} th{th}: {a.n} tracks in {time.time() - t:.0f}s  '
                  f'<E> {r["e_dep"].mean() / 1e3:.2f} keV  med {np.median(r["e_dep"]) / 1e3:.2f}  '
                  f'<n_e> {r["n_e"].mean():.0f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
