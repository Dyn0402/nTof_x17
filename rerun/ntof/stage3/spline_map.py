#!/usr/bin/env python3
"""spline_map.py -- the evaluator of the n_TOF in-situ spline angle map (r3 and later).  Pure numpy.

The map is fitted by rerun/ntof/improve/spline_map_fit.py and applied to the analysis view by map_view.py, which
recognises it by ``meta.model == "spline"``.  This module only evaluates a map file; it never fits.

Model
-----
For each chamber (A, C, D) and each view p = x, y the map describes how the measured tan departs from the true tan,
in the chamber's LOCAL tan t (the stage-3 columns tanx, tany):

    t_measured = (1 + s_p(x, y)) * t_true + a_p(x, y)      so      t_corrected = (t - a_p) / (1 + s_p)

(x, y) is the track's position (x_local, y_local) in mm.  a_p is the tilt (in tan units) and s_p the relative scale.
The tilt is the gradient of one potential phi (a_x = d phi / dx, a_y = d phi / dy, ``tilt.kind = "potential"``), or
two free fields (``"free"``).  The scale is one field per view (``scale.kind = "per_view"``) or one field common to
both views (``"common"``).  Every field is a tensor-product uniform cubic B-spline on [lo, hi]^2 with knot spacing h.
It has nb = (hi - lo) / h + 3 coefficients per axis, stored as an nb x nb array indexed [i_x][i_y].  The production
map uses h = 20 mm for phi and h = 40 mm for the scales, on [-200, 200] mm.

Frames and signs
----------------
The same correction acts on the GLOBAL slopes, which is the frame of the cosmic lines the map is fitted on and the
frame of the cell maps (map_view.FRAME).  The global slope of view p is S = g_p t, with g = (g_x, g_y):
A (+1, +1), C (+1, -1), D (-1, +1).  The global slopes are d_x/d_z and d_y/d_z on A and C, and d_z/d_x and d_y/d_x
on D (D's x strips measure global z).  The corrected global slope is S' = (S - g_p a_p) / (1 + s_p).  The sign g is
stored per arm in the file (``maps[arm].g``); map_view checks it against the rows of every table.

Clamps
------
1. A position is clamped to [lo, hi] before the evaluation, so a track outside +-200 mm takes the edge value.
2. Each field is then clamped to the interval in ``maps[arm].clamp`` (keys a_x, a_y, s_x, s_y).  The fit sets it to
   the [0.1, 99.9] percentile of the field over the positions of the clean training lines.  So the spline's values in
   corners without lines never reach a track.
3. A track with a NaN position is returned unchanged (no correction).  It is never moved by an extrapolated value.

JSON format
-----------
    {"meta": {"model": "spline", "version": ..., "inputs": ..., "frames": ..., "clamps": ..., ...},
     "maps": {"A": {"arm": "A", "g": [gx, gy], "frame": "...",
                    "tilt_basis":  {"lo": -200.0, "hi": 200.0, "h": 20.0, "nb": 23},
                    "scale_basis": {"lo": -200.0, "hi": 200.0, "h": 40.0, "nb": 13},
                    "tilt":  {"kind": "potential", "phi": [[...]]}      or {"kind": "free", "a_x": ..., "a_y": ...},
                    "scale": {"kind": "per_view", "s_x": [[...]], "s_y": [[...]]}   or {"kind": "common", "s": ...},
                    "clamp": {"a_x": [lo, hi], "a_y": [...], "s_x": [...], "s_y": [...]},
                    "support": {"grid_mm": [...], "mask": [[0/1]]},  "fit": {...}},
              "C": {...}, "D": {...}}}

An arm without an entry (B) is returned unchanged.

    from rerun.ntof.stage3 import spline_map as SM
    M = SM.load('rerun/ntof/stage3/angle_map/angle_map_r3.json')
    tx2, ty2 = SM.correct_tan(M, 'A', x_local, y_local, tanx, tany)
"""
from __future__ import annotations

import json

import numpy as np

MODEL = 'spline'
#: local tan -> global slope sign per arm, (x view, y view)
G = {'A': (1.0, 1.0), 'C': (1.0, -1.0), 'D': (-1.0, 1.0)}


def load(path):
    with open(path) as f:
        return json.load(f)


def is_spline(M) -> bool:
    """True for a spline map file (meta.model == 'spline'); a cell map has no model key."""
    return str((M.get('meta') or {}).get('model', '')).strip().lower() == MODEL


# ------------------------------------------------------------------------------------------------ B-spline basis
def local_basis(x, lo, hi, h, deriv=0):
    """Uniform cubic B-spline on [lo, hi] with knot spacing h.  For each x: the first index i0 of the 4 non-zero basis
    functions and their values, shape (n, 4).  x is clamped to [lo, hi]; deriv 0, 1 or 2 (d/dx in 1/mm units)."""
    x = np.clip(np.asarray(x, float), lo, hi)
    nint = int(round((hi - lo) / h))
    t = (x - lo) / h
    i = np.clip(np.floor(t).astype(int), 0, nint - 1)
    u = t - i
    if deriv == 0:
        B = np.stack([(1 - u) ** 3, 3 * u ** 3 - 6 * u ** 2 + 4, -3 * u ** 3 + 3 * u ** 2 + 3 * u + 1, u ** 3], 1) / 6
    elif deriv == 1:
        B = np.stack([-(1 - u) ** 2, 3 * u ** 2 - 4 * u, -3 * u ** 2 + 2 * u + 1, u ** 2], 1) / (2 * h)
    elif deriv == 2:
        B = np.stack([1 - u, 3 * u - 2, -3 * u + 1, u], 1) / h ** 2
    else:
        raise ValueError(deriv)
    return i, B


def tensor_eval(C, x, y, lo, hi, h, dx=0, dy=0):
    """sum_ij C[i, j] B_i^(dx)(x) B_j^(dy)(y), vectorised (16 terms per point)."""
    C = np.asarray(C, float)
    ix, Bx = local_basis(x, lo, hi, h, dx)
    iy, By = local_basis(y, lo, hi, h, dy)
    out = np.zeros(len(ix))
    for a in range(4):
        for b in range(4):
            out += C[ix + a, iy + b] * Bx[:, a] * By[:, b]
    return out


# ------------------------------------------------------------------------------------------------ the map
def raw_fields(m, x, y):
    """(a_x, a_y, s_x, s_y) of one arm entry at finite (x, y), positions clamped to the domain, fields NOT clamped."""
    bt, bs = m['tilt_basis'], m['scale_basis']
    T = m['tilt']
    if T['kind'] == 'potential':
        ax = tensor_eval(T['phi'], x, y, bt['lo'], bt['hi'], bt['h'], dx=1)
        ay = tensor_eval(T['phi'], x, y, bt['lo'], bt['hi'], bt['h'], dy=1)
    elif T['kind'] == 'free':
        ax = tensor_eval(T['a_x'], x, y, bt['lo'], bt['hi'], bt['h'])
        ay = tensor_eval(T['a_y'], x, y, bt['lo'], bt['hi'], bt['h'])
    else:
        raise ValueError(f"unknown tilt kind {T['kind']!r}")
    S = m['scale']
    if S['kind'] == 'common':
        sx = sy = tensor_eval(S['s'], x, y, bs['lo'], bs['hi'], bs['h'])
    elif S['kind'] == 'per_view':
        sx = tensor_eval(S['s_x'], x, y, bs['lo'], bs['hi'], bs['h'])
        sy = tensor_eval(S['s_y'], x, y, bs['lo'], bs['hi'], bs['h'])
    else:
        raise ValueError(f"unknown scale kind {S['kind']!r}")
    return ax, ay, sx, sy


def fields(M, arm, x, y):
    """(a_x, a_y, s_x, s_y) in the local tan convention at (x_local, y_local), with the clamps of the arm entry.
    NaN positions give a = 0, s = 0 (no correction)."""
    m = M['maps'][arm]
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    bad = ~(np.isfinite(x) & np.isfinite(y))
    xx, yy = np.where(bad, 0.0, x), np.where(bad, 0.0, y)
    ax, ay, sx, sy = raw_fields(m, xx, yy)
    c = m['clamp']
    ax, ay = np.clip(ax, *c['a_x']), np.clip(ay, *c['a_y'])
    sx, sy = np.clip(sx, *c['s_x']), np.clip(sy, *c['s_y'])
    z = np.zeros_like(ax)
    return (np.where(bad, z, ax), np.where(bad, z, ay), np.where(bad, z, sx), np.where(bad, z, sy))


def correct_tan(M, arm, x_local, y_local, tanx, tany):
    """Corrected local tans (stage-3 tanx, tany convention).  Arms without a map (B) are returned unchanged."""
    tanx, tany = np.asarray(tanx, float), np.asarray(tany, float)
    if arm not in M['maps']:
        return tanx, tany
    ax, ay, sx, sy = fields(M, arm, x_local, y_local)
    return (tanx - ax) / (1.0 + sx), (tany - ay) / (1.0 + sy)


def uncorrect_tan(M, arm, x_local, y_local, tanx, tany):
    """The inverse of correct_tan: the measured tans that a true (tanx, tany) at (x_local, y_local) would give."""
    tanx, tany = np.asarray(tanx, float), np.asarray(tany, float)
    if arm not in M['maps']:
        return tanx, tany
    ax, ay, sx, sy = fields(M, arm, x_local, y_local)
    return (1.0 + sx) * tanx + ax, (1.0 + sy) * tany + ay


def correct_global(M, arm, x_local, y_local, slope_x, slope_y):
    """Corrected GLOBAL slopes (A, C: d_x/d_z, d_y/d_z; D: d_z/d_x, d_y/d_x), the frame of the cosmic lines."""
    slope_x, slope_y = np.asarray(slope_x, float), np.asarray(slope_y, float)
    if arm not in M['maps']:
        return slope_x, slope_y
    gx, gy = G[arm]
    ax, ay, sx, sy = fields(M, arm, x_local, y_local)
    return (slope_x - gx * ax) / (1.0 + sx), (slope_y - gy * ay) / (1.0 + sy)
