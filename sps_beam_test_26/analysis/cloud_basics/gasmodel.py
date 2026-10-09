#!/usr/bin/env python3
"""gasmodel.py -- Magboltz transport on a (water, air) grid, interpolated.

Two sources, same interface:
  'air'     results/air/magboltz_<gas>_w<w>_a<a>.json          (5e7 collisions, 12 fields per file)
  'air_hs'  results/air_hs/magboltz_<gas>_w<w>_a<a>_E<E>_c30.json  (3e8 collisions, one field per file)
Gas tags: beam = Ar/CF4/iso 88/10/2, co2 = Ar/CO2/iso 95/3/2, bench = Ar/iso 95/5; water and air
(N2/O2/Ar 78.08/20.95/0.93) replace argon.

    G = GasGrid('beam', 'air_hs');  G(w, a, E) -> dict(v=um/ns, eta=1/cm, etav=1/ns, DL=um/sqrt(cm), DT=...)
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
KEYS = ('v', 'eta', 'DL', 'DT')


class GasGrid:
    def __init__(self, gas, source='air_hs'):
        self.gas = gas
        T = {}
        for p in glob.glob(os.path.join(RES, source, f'magboltz_{gas}_w*_a*.json')):
            m = re.search(rf'{gas}_w([0-9p]+)_a([0-9p]+)(?:_E[0-9.]+_c\d+)?\.json$', os.path.basename(p))
            if not m:
                continue
            w, a = (float(x.replace('p', '.')) for x in m.groups())
            for pt in json.load(open(p))['points']:
                T.setdefault((w, a), {})[float(pt['E_Vcm'])] = (
                    pt['v_true_um_ns'], pt['eta_per_cm'], pt['DL_um_rtcm'], pt['DT_um_rtcm'])
        if not T:
            raise SystemExit(f'no Magboltz tables for {gas} in results/{source}')
        self.W = np.array(sorted({k[0] for k in T})); self.A = np.array(sorted({k[1] for k in T}))
        self.E = np.array(sorted({e for d in T.values() for e in d}))
        # dense cube, NaN where a job has not landed
        self.cube = np.full((len(self.W), len(self.A), len(self.E), 4), np.nan)
        for (w, a), d in T.items():
            for e, vals in d.items():
                self.cube[np.searchsorted(self.W, w), np.searchsorted(self.A, a), np.searchsorted(self.E, e)] = vals
        self.complete = float(np.isfinite(self.cube[..., 0]).mean())

    def _lin(self, X, x):
        i = int(np.clip(np.searchsorted(X, x) - 1, 0, len(X) - 2))
        f = (x - X[i]) / (X[i + 1] - X[i])
        return i, f

    def __call__(self, w, a, E):
        iw, fw = self._lin(self.W, w); ia, fa = self._lin(self.A, a); ie, fe = self._lin(self.E, E)
        out = 0.0
        for dw, cw in ((0, 1 - fw), (1, fw)):
            for da, ca in ((0, 1 - fa), (1, fa)):
                for de, ce in ((0, 1 - fe), (1, fe)):
                    out = out + cw * ca * ce * self.cube[iw + dw, ia + da, ie + de]
        d = dict(zip(KEYS, out))
        d['etav'] = d['eta'] * d['v'] * 1e-4          # per ns
        return d
