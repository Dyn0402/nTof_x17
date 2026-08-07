#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_microtpc.py -- how an MX17 chamber measures a track angle.

The micro-TPC idea in one picture: a muon crosses the 30 mm drift gap at an
angle, every primary ionisation cluster drifts straight down at v_drift, so a
cluster's ARRIVAL TIME measures its DEPTH -- and the slope of arrival time
against strip position is the track angle, from a single chamber.

Everything with a number attached is measured or simulated for this detector,
not invented:

  drift gap        30 mm                     run_config 'drift_gap'
  drift field      333 V/cm                  bench point, 1000 V over the gap
  v_drift          ~34 um/ns                 garfield_sim/results/water_grid
                                             (Ar/iso 95/5 + ~1 % H2O, the
                                             mixture the bench actually runs;
                                             agrees with the 34 +- 1.5 um/ns
                                             measured by the geometry estimator)
  sigma_T          0.35 mm / sqrt(cm)        same table -> ~0.5 mm over 3 cm
  strip pitch      0.7785 mm                 398.58 mm / 512 strips
  cluster density  ~30 /cm                   Ar/isobutane at NTP

Frame: +Z up (the drift direction), X across the strips, Y along them.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

import geometry as G
import meshes as M
import style as S

# --- the operating point ----------------------------------------------------
DRIFT_MM = G.MX17_DRIFT_GAP_MM          # 30
E_DRIFT_V_CM = 1000.0 / (DRIFT_MM / 10.0)
V_DRIFT_UM_NS = 34.0
SIGMA_T_SQRTCM = 0.0351                 # -> sigma_T [cm] = this * sqrt(L[cm])
CLUSTERS_PER_CM = 30.0
STRIP_PITCH_MM = G.MX17_ACTIVE_MM / 512

# --- the window drawn -------------------------------------------------------
WIN_X = 46.0                            # across the strips
WIN_Y = 26.0                            # along them
MESH_Z = 0.0
CATHODE_Z = DRIFT_MM

TRACK_ANGLE_DEG = 32.0                  # from vertical, in the x-z plane


def drift_time_ns(z_mm):
    """Time for a cluster at height ``z_mm`` to reach the mesh."""
    return z_mm * 1000.0 / V_DRIFT_UM_NS


def sigma_t_mm(z_mm):
    """Transverse diffusion after drifting ``z_mm``."""
    return SIGMA_T_SQRTCM * np.sqrt(max(z_mm, 0.0) / 10.0) * 10.0


def make_event(angle_deg=TRACK_ANGLE_DEG, seed=7, x0=None):
    """One muon and its primary clusters, with Poisson statistics.

    Returns (entry, exit, clusters) with clusters as (x, y, z) at creation.
    """
    rng = np.random.default_rng(seed)
    t = np.tan(np.radians(angle_deg))
    x0 = (-t * DRIFT_MM / 2 if x0 is None else x0)

    # the track, top of the gap -> bottom
    a = np.array([x0, 0.0, CATHODE_Z])
    b = np.array([x0 + t * DRIFT_MM, 0.0, MESH_Z])
    path_mm = np.linalg.norm(b - a)

    n = rng.poisson(CLUSTERS_PER_CM * path_mm / 10.0)
    frac = np.sort(rng.uniform(0, 1, n))
    pts = a[None, :] + (b - a)[None, :] * frac[:, None]
    return a, b, pts


def add_chamber(p, theme='light'):
    """Cathode, gas volume, mesh and the strip plane, as a cut-out block."""
    # strip plane (the readout PCB, just under the mesh)
    p.add_mesh(M.slab((0, 0, -3.0), WIN_X, WIN_Y, 4.0, normal='z'),
               **S.mat('pcb', S.COL['pcb']))
    n_strips = int(WIN_X / STRIP_PITCH_MM)
    p.add_mesh(M.strip_lines(n_strips, (-WIN_X / 2, WIN_X / 2),
                             (-WIN_Y / 2, WIN_Y / 2), -0.95, along='v',
                             width=STRIP_PITCH_MM * 0.7),
               **S.mat('copper', S.COL['copper']))

    # micromesh, a woven grid just above the strips
    nm = 34
    for along in ('u', 'v'):
        p.add_mesh(M.strip_lines(nm, (-WIN_X / 2, WIN_X / 2),
                                 (-WIN_Y / 2, WIN_Y / 2),
                                 MESH_Z + (0.16 if along == 'u' else -0.16),
                                 along=along, width=WIN_X / nm * 0.16),
                   **S.mat('mesh', '#8f979f', ambient=0.5))

    # the drift volume itself and the cathode above it
    gas = M.slab((0, 0, DRIFT_MM / 2), WIN_X, WIN_Y, DRIFT_MM, normal='z')
    p.add_mesh(gas, **S.mat('gas', S.COL['gas'], opacity=0.055))
    p.add_mesh(gas.extract_feature_edges(), color=S.COL['gas'], line_width=1.8,
               lighting=False, opacity=0.55)
    p.add_mesh(M.slab((0, 0, CATHODE_Z + 0.8), WIN_X, WIN_Y, 1.6, normal='z'),
               **S.mat('plastic', '#dfe7ef', opacity=0.30))


def add_event(p, a, b, clusters, cmap='plasma', show_clouds=True):
    """The track, its primaries, their drift columns and the strip response.

    Drift columns are coloured by ARRIVAL TIME, which is the whole point: the
    colour is a direct read-out of how deep the cluster was created.
    """
    tmax = drift_time_ns(DRIFT_MM)

    # --- the muon, with an arrow head on the way out ------------------------
    d = (b - a) / np.linalg.norm(b - a)
    for mesh in M.tracks_with_heads([(a - d * 7.0, b + d * 7.0)], 0.30,
                                    head_len=3.2, head_radius=0.95):
        p.add_mesh(mesh, **S.mat('glow', S.COL['track_mu']))

    # --- drift columns, one per primary -------------------------------------
    lines, scal = [], []
    for (x, y, z) in clusters:
        seg = pv.Line((x, y, z), (x, y, MESH_Z + 0.2), resolution=1)
        lines.append(seg)
        scal.append(drift_time_ns(z))
    if lines:
        merged = lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]
        merged.cell_data['t_ns'] = np.repeat(np.array(scal), 1)
        tubes = merged.tube(radius=0.085, n_sides=10)
        p.add_mesh(tubes, scalars='t_ns', cmap=cmap, clim=(0, tmax),
                   show_scalar_bar=False, **S.mat('glow', None, ambient=0.6,
                                                  diffuse=0.5))

    # --- the primaries themselves -------------------------------------------
    for (x, y, z) in clusters:
        p.add_mesh(pv.Sphere(radius=0.28, center=(x, y, z), theta_resolution=14,
                             phi_resolution=14),
                   **S.mat('glow', '#ffe9a8'))

    # --- the diffused cloud arriving at the mesh -----------------------------
    if show_clouds:
        for (x, y, z) in clusters:
            s = sigma_t_mm(z)
            disc = pv.Disc(center=(x, y, MESH_Z + 0.45), inner=0.0,
                           outer=max(2.0 * s, 0.25), normal=(0, 0, 1),
                           r_res=1, c_res=28)
            p.add_mesh(disc, color='#ffd166', opacity=0.28, lighting=False)

    # --- which strips fire, coloured by first arrival ------------------------
    strip_hits = {}
    for (x, y, z) in clusters:
        s = max(sigma_t_mm(z), 1e-3)
        lo = int(np.floor((x - 2 * s + WIN_X / 2) / STRIP_PITCH_MM))
        hi = int(np.ceil((x + 2 * s + WIN_X / 2) / STRIP_PITCH_MM))
        for k in range(lo, hi + 1):
            t = drift_time_ns(z)
            if k not in strip_hits or t < strip_hits[k]:
                strip_hits[k] = t
    if strip_hits:
        polys, times = [], []
        w = STRIP_PITCH_MM * 0.7
        for k, t in strip_hits.items():
            xc = -WIN_X / 2 + (k + 0.5) * STRIP_PITCH_MM
            if abs(xc) > WIN_X / 2:
                continue
            polys.append([[xc - w / 2, -WIN_Y / 2], [xc + w / 2, -WIN_Y / 2],
                          [xc + w / 2, WIN_Y / 2], [xc - w / 2, WIN_Y / 2]])
            times.append(t)
        mesh = M.quads_mesh(np.array(polys), -0.55)
        mesh.cell_data['t_ns'] = np.array(times)
        p.add_mesh(mesh, scalars='t_ns', cmap=cmap, clim=(0, tmax),
                   show_scalar_bar=False, lighting=False)
    return strip_hits


def strip_ladder(strip_hits):
    """(x_mm, t_ns) per fired strip -- the micro-TPC measurement itself."""
    xs, ts = [], []
    for k, t in sorted(strip_hits.items()):
        xs.append(-WIN_X / 2 + (k + 0.5) * STRIP_PITCH_MM)
        ts.append(t)
    return np.array(xs), np.array(ts)
