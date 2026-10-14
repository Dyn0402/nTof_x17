"""
beam.py -- the EAR2 neutron beam at the station, for analysis code.

Samples transverse positions and ray slopes from the optical-transport 2D
profiles (optical_transport.py -> <out>/profile_<variant>.npz and
summary_<variant>.csv).  Below ~1 keV the profile does not depend on energy
(it is set by the collimator geometry), so one band ('beam', 14-124 meV) is
used for every slow neutron.

    from ear2_beam import beam
    x, z, sx, sz = beam.sample(n, rng, L=19.5, centre=(0, 0))   # mm, rad

Coordinates are the transport's (x, z about the n_TOF beam axis, which is
centred to ~1 mm on the vacuum-pipe axis); how they map onto the chamber frame
is not known, so callers place the beam centre themselves.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('/media/dylan/data/x17/ear2_beam')
#: the station height the rest of the repo assumes (flight path, m)
L_DEFAULT = 19.5


@lru_cache(maxsize=None)
def _tables(variant: str, band: str):
    npz = np.load(OUT / f'profile_{variant}.npz')
    S = pd.read_csv(OUT / f'summary_{variant}.csv')
    S = S[S.band == band].set_index('L')
    return npz, S


def profile(L: float = L_DEFAULT, variant: str = 'july2021', band: str = 'beam'):
    """(edges mm, 2D histogram normalised to 1, summary row) at the tabulated
    height nearest to L."""
    npz, S = _tables(variant, band)
    Ls = np.array(sorted(S.index))
    Lk = float(Ls[np.argmin(abs(Ls - L))])
    if abs(Lk - L) > 0.13:
        raise ValueError(f'no profile within 13 cm of L={L} (have {list(Ls)})')
    h = npz[f'{band}_{Lk:g}']
    return 10.0 * npz['edges_cm'], h / h.sum(), S.loc[Lk]


def sample(n: int, rng, L: float = L_DEFAULT, centre=(0.0, 0.0), variant: str = 'july2021',
           band: str = 'beam'):
    """n rays: x, z (mm, beam centroid moved to `centre`) and slopes sx, sz
    (rad): the cone of the collimation (slope proportional to position) plus
    the local spread."""
    e, h, s = profile(L, variant, band)
    c = h.ravel().cumsum()
    k = np.searchsorted(c, rng.random(n) * c[-1])
    nz = h.shape[1]
    ix, iz = k // nz, k % nz
    w = e[1] - e[0]
    x = e[ix] + w * rng.random(n) - s.cx_mm
    z = e[iz] + w * rng.random(n) - s.cz_mm
    k_slope = s.slope_x_per_cm_mrad * 1e-3 / 10.0          # rad per mm
    sx = k_slope * x + s.local_spread_mrad * 1e-3 * rng.standard_normal(n)
    sz = k_slope * z + s.local_spread_mrad * 1e-3 * rng.standard_normal(n)
    return x + centre[0], z + centre[1], sx, sz
