"""
X neighbour-asymmetry position correction (opt-in, 2026-10-21).

The June-built boards carry an X-only field of unequal +-1 neighbours (DAQ n+1 > n-1 on det3 June-15) that varies
smoothly over the board and takes both signs (det2: bands in y that cancel on average). It is not in the wft
forward model, and the production position carries it: against the M3 reference, the x residual of the bench runs
follows the raw-ADC centroid bias map with K = -0.77 +- 0.13 (all boards, alignment-like terms removed). The
alignment absorbs the board mean and a rotation, but not the pattern and not an x scale.

    from wft import xasym
    df['x_p0_xasym'] = xasym.correct_x(df.x_p0, df.y_p0, build='det2')
    df['x_p0_xasym'] = xasym.correct_x(df.x_p0, df.y_p0, build='det3 June-15', scale=0.96)

The correction is x - K * dx(x, y) * scale [mm]. ``scale`` is the run's amplitude relative to the all-session map
(the field's geometry is fixed per board, its size follows the cloud width; det3's measured sessions are in
``session_scale``). Re-run the alignment on corrected positions: the correction contains a board-mean part.

Leave-one-out validation (cell chi2 of the M3 residual, offset + rotation removed): det2 136 -> 108 and 34 -> 23,
det3 111 -> 96, 99 -> 94, 97 -> 92. det4/6/7 are excluded: their maps are too sparse and made it worse or no better.
Maps and provenance: ``xasym_maps.json``; the analysis: MX17_Geant
``scratchpad/2026-10-19/asym_deck/FINDINGS.md``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xasym_maps.json')


@lru_cache(maxsize=1)
def maps() -> dict:
    with open(MAPS_PATH) as f:
        return json.load(f)


def builds() -> list:
    return sorted(maps()['builds'])


def dx_mm(x_mm, y_mm, build: str) -> np.ndarray:
    """Raw centroid-bias map of ``build`` at (x, y) [mm, wft frame]; empty cells are 0."""
    M = maps()
    if build not in M['builds']:
        why = M.get('excluded', {}).get(build, 'no map')
        raise KeyError(f'no X-asymmetry map for {build!r} ({why}); have {builds()}')
    G = np.array([[np.nan if v is None else v for v in row] for row in M['builds'][build]['dx_um']], float)
    n, cell = M['ncell'], M['cell_mm']
    i = np.clip(np.floor(np.asarray(x_mm, float) / cell).astype(int), 0, n - 1)
    j = np.clip(np.floor(np.asarray(y_mm, float) / cell).astype(int), 0, n - 1)
    v = G[i, j]
    return np.where(np.isfinite(v), v, 0.0) * 1e-3


def correct_x(x_mm, y_mm, build: str, scale: float = 1.0) -> np.ndarray:
    """Corrected X position [mm]: x - K dx(x, y) scale. NaN positions stay NaN."""
    x = np.asarray(x_mm, float)
    y = np.asarray(y_mm, float)
    ok = np.isfinite(x) & np.isfinite(y)
    out = x.copy()
    out[ok] = x[ok] - maps()['K'] * dx_mm(x[ok], y[ok], build) * float(scale)
    return out
