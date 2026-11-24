#!/usr/bin/env python3
"""apply_r3.py -- evaluate the r3 candidate angle map (smooth spline tilt + scale) on n_TOF tracks.  Pure numpy.

Model (per chamber A, C, D and per view p = x, y), in the chamber's LOCAL tan t (the stage-3 columns tanx, tany):

    t_measured = (1 + s_p(x, y)) * t_true + a_p(x, y)          ->   t_corrected = (t - a_p) / (1 + s_p)

with (x, y) = (x_local, y_local) in mm.  a_p is the tilt (tan units), s_p the relative scale.  The tilt is either the
gradient of one potential phi (a_x = d phi/dx, a_y = d phi/dy; "potential") or two free fields ("free").  The scale is
one field common to both views ("common") or one per view ("per_view").  All fields are tensor-product uniform cubic
B-splines on [lo, hi]^2 with knot spacing h (nb = (hi - lo)/h + 3 coefficients per axis); coefficient arrays are
nb x nb, index [i_x, i_y].

The same correction on GLOBAL slopes (the frame of angle_map_r2.json and map_view.FRAME): global slope S = g_p t with
g = A (+1, +1), C (+1, -1), D (-1, +1) (x view, y view); S' = (S - g_p a_p) / (1 + s_p).  Frames of the global slopes:
A, C: x view d_x/d_z, y view d_y/d_z;  D: x view d_z/d_x, y view d_y/d_x.

Edges / extrapolation: positions are clamped to the spline domain [lo, hi] (so a track outside it gets the edge value),
then the tilt and the scale are clamped to [min, max] of each field over the region that had training lines
(per-view 'clamp' block).  A track with a NaN position is returned unchanged (no correction), so it is never moved
by an extrapolated value.

    from apply_r3 import load, correct_tan, correct_global
    M = load('angle_map_r3_candidate.json')
    tx2, ty2 = correct_tan(M, 'A', x_local, y_local, tanx, tany)
"""
from __future__ import annotations

import json

import numpy as np

G = {'A': (1.0, 1.0), 'C': (1.0, -1.0), 'D': (-1.0, 1.0)}


def load(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ B-spline basis
def local_basis(x, lo, hi, h, deriv=0):
    """Uniform cubic B-spline on [lo, hi] (knot spacing h): for each x the first index i0 of the 4 non-zero basis
    functions and their values (n, 4).  x is clamped to [lo, hi]; deriv 0, 1, 2 (d/dx in 1/mm units)."""
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
def fields(M, arm, x, y):
    """(a_x, a_y, s_x, s_y) in the local tan convention at (x_local, y_local), with the clamps of the meta block."""
    m = M['maps'][arm]
    bt, bs = m['tilt_basis'], m['scale_basis']
    x, y = np.asarray(x, float), np.asarray(y, float)
    bad = ~(np.isfinite(x) & np.isfinite(y))
    xx, yy = np.where(bad, 0.0, x), np.where(bad, 0.0, y)
    T = m['tilt']
    if T['kind'] == 'potential':
        ax = tensor_eval(T['phi'], xx, yy, bt['lo'], bt['hi'], bt['h'], dx=1)
        ay = tensor_eval(T['phi'], xx, yy, bt['lo'], bt['hi'], bt['h'], dy=1)
    else:
        ax = tensor_eval(T['a_x'], xx, yy, bt['lo'], bt['hi'], bt['h'])
        ay = tensor_eval(T['a_y'], xx, yy, bt['lo'], bt['hi'], bt['h'])
    S = m['scale']
    if S['kind'] == 'common':
        sx = sy = tensor_eval(S['s'], xx, yy, bs['lo'], bs['hi'], bs['h'])
    else:
        sx = tensor_eval(S['s_x'], xx, yy, bs['lo'], bs['hi'], bs['h'])
        sy = tensor_eval(S['s_y'], xx, yy, bs['lo'], bs['hi'], bs['h'])
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


def correct_global(M, arm, x_local, y_local, slope_x, slope_y):
    """Corrected GLOBAL slopes (A, C: d_x/d_z, d_y/d_z; D: d_z/d_x, d_y/d_x), the frame of angle_map_r2.json."""
    slope_x, slope_y = np.asarray(slope_x, float), np.asarray(slope_y, float)
    if arm not in M['maps']:
        return slope_x, slope_y
    gx, gy = G[arm]
    ax, ay, sx, sy = fields(M, arm, x_local, y_local)
    return (slope_x - gx * ax) / (1.0 + sx), (slope_y - gy * ay) / (1.0 + sy)
