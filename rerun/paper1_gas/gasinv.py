#!/usr/bin/env python3
"""gasinv.py -- bench drift velocity <-> water through Magboltz, on the fine bench grid (rerun/paper1_gas/README.md).

The grid is GasGrid('bench', 'air_merged'): Ar/iso 95/5 + water + air, 3e8 collisions per point, water 0-3 %
(0.05 % steps over 0.4-2.0 %), air 0 / 0.01 / 0.04 %, 22 fields 33-400 V/cm (MB_FINE_2026-10-31.md).

    water_of_v(v, E, air=0)  every water on the grid scan that gives v at E (v(w) need not be monotonic: near the
                             velocity maximum two waters give one v, the bench_water_e.invert trap of FINDINGS §31)
    v_of(w, E, air=0), dv_dw(w, E)   Magboltz v and its water derivative (um/ns per % water)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.abspath(os.path.join(HERE, '..', '..', 'sps_beam_test_26', 'analysis', 'cloud_basics'))
sys.path.insert(0, CB)
from gasmodel import GasGrid  # noqa: E402

SOURCE = 'air_merged'
G = GasGrid('bench', SOURCE)
WS = np.arange(0.0, float(G.W.max()) + 1e-9, 0.002)


def v_of(w, E, air=0.0):
    return float(G(w, air, E)['v'])


def dt_of(w, E, air=0.0):
    return float(G(w, air, E)['DT'])


def dv_dw(w, E, air=0.0, h=0.02):
    lo, hi = max(w - h, 0.0), min(w + h, float(G.W.max()))
    return (v_of(hi, E, air) - v_of(lo, E, air)) / (hi - lo)


def water_of_v(v, E, air=0.0):
    """All waters (in %) whose Magboltz v at E equals v, by linear interpolation along the fine scan."""
    vs = np.array([v_of(w, E, air) for w in WS]) - v
    out = []
    for i in np.nonzero(np.sign(vs[:-1]) != np.sign(vs[1:]))[0]:
        out.append(float(WS[i] - vs[i] * (WS[i + 1] - WS[i]) / (vs[i + 1] - vs[i])))
    return out


def water_err(v, v_err, E, air=0.0):
    """Water and its 1-sigma interval from v +- v_err (first branch); None where v is off the grid."""
    w = water_of_v(v, E, air)
    if not w:
        return None
    lo, hi = water_of_v(v + v_err, E, air), water_of_v(v - v_err, E, air)   # v falls with water on the bench branch
    return dict(water=w[0], lo=(lo[0] if lo else np.nan), hi=(hi[0] if hi else np.nan), branches=w)


if __name__ == '__main__':
    for E in (243.1, 333.3, 356.1):
        print(E, [round(v_of(w, E), 2) for w in (0, 0.5, 0.95, 1.5, 2, 3)], water_of_v(35.82, E))
