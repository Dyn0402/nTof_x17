#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_chamber.py -- one MX17 chamber, pulled apart.

The "how a resistive-strip micro-TPC chamber works" figure: the layers of a
single MX17 stacked along the drift axis and separated so each one is visible,
with a muon crossing the drift volume and its ionisation drifting down to the
mesh.

Layer stack, bottom (readout) to top (cathode) -- what is measured and what is
nominal:

  X readout strips     512 strips, pitch 398.58/512 = 0.7785 mm   [measured]
  Y readout strips     512 strips, orthogonal                     [measured]
  resistive strips     over the readout, 'resist_type': 'strip'   [run config]
  amplification gap    150 um, mesh to resistive layer            [garfield_sim/
                                                                   mm_config.py]
  micromesh            woven mesh at the top of that gap
  drift volume         30 mm                                      [run config]
  drift cathode        mylar                                      [run config]

Only the *thicknesses along the drift axis* are exaggerated -- an exploded view
of a 150 um gap under a 30 mm one is otherwise invisible.  The exaggeration
factor is written into the figure so no one reads a scale off it.

Frame: +Z along the drift direction (up), X/Y transverse.
"""
from __future__ import annotations

import numpy as np

import geometry as G
import meshes as M
import style as S

RNG = np.random.default_rng(20260807)

# --- the real stack, mm ------------------------------------------------------
AMP_GAP_MM = 0.150            # garfield_sim/mm_config.py: GAP_CM = 0.0150
N_STRIPS = 512                # mx17_m1_map.csv
STRIP_PITCH_MM = G.MX17_ACTIVE_MM / N_STRIPS      # 0.7785 mm

# --- what the figure draws (a WINDOW on the chamber, not the whole 40 cm) ----
WIN_MM = 30.0                 # side of the cut-out square shown
DRAWN_STRIPS = int(round(WIN_MM / STRIP_PITCH_MM))   # keeps the real pitch

# exploded spacing along z, mm (drawn, not physical)
EXPLODE = 19.0


def layers(explode=EXPLODE, window=WIN_MM):
    """(name, z_centre, thickness, label) for each layer, bottom to top.

    ``z`` is the *drawn* height; the physical stack is in the label text.
    """
    t_pcb, t_strip, t_resist, t_mesh, t_cath = 7.0, 1.6, 1.2, 1.4, 1.0
    z = 0.0
    out = []

    out.append(('pcb', z, t_pcb,
                'Readout PCB'))
    z += t_pcb / 2 + explode + t_strip / 2
    out.append(('strips_y', z, t_strip,
                f'Y readout strips\n{N_STRIPS} strips, {STRIP_PITCH_MM:.3f} mm pitch'))
    z += t_strip / 2 + explode * 0.55 + t_strip / 2
    out.append(('strips_x', z, t_strip,
                f'X readout strips\n{N_STRIPS} strips, orthogonal'))
    z += t_strip / 2 + explode * 0.55 + t_resist / 2
    out.append(('resist', z, t_resist,
                'Resistive strips\n(spark protection, charge spreading)'))
    z += t_resist / 2 + explode * 0.55 + t_mesh / 2
    out.append(('mesh', z, t_mesh,
                f'Micromesh\n{AMP_GAP_MM * 1000:.0f} um amplification gap'))
    z_gas0 = z + t_mesh / 2 + explode
    out.append(('gas', z_gas0 + G.MX17_DRIFT_GAP_MM / 2, G.MX17_DRIFT_GAP_MM,
                f'Drift volume\n{G.MX17_DRIFT_GAP_MM:.0f} mm, Ar/isobutane 95/5'))
    z = z_gas0 + G.MX17_DRIFT_GAP_MM + t_cath / 2
    out.append(('cathode', z, t_cath, 'Drift cathode (mylar)'))
    return out


def build(p, explode=EXPLODE, window=WIN_MM, track=True):
    """Draw the exploded chamber; returns {layer: anchor point} for labelling."""
    L = {n: (z, t, lab) for n, z, t, lab in layers(explode, window)}
    anchors = {}
    half = window / 2

    def anchor(name, z):
        anchors[name] = (half * 0.98, -half * 0.98, z)

    # readout PCB
    z, t, _ = L['pcb']
    p.add_mesh(M.slab((0, 0, z), window, window, t), **S.mat('pcb', S.COL['pcb']))
    anchor('pcb', z)

    # Y then X strips, at the real pitch, orthogonal to each other
    for name, along in (('strips_y', 'v'), ('strips_x', 'u')):
        z, t, _ = L[name]
        p.add_mesh(M.slab((0, 0, z - t / 2), window, window, t * 0.28),
                   **S.mat('pcb', '#123f38'))
        strips = M.strip_lines(DRAWN_STRIPS, (-half, half), (-half, half),
                               z, along=along, width=STRIP_PITCH_MM * 0.72)
        p.add_mesh(strips, **S.mat('copper', S.COL['copper']))
        anchor(name, z)

    # resistive strips: same pitch as the X strips, above them
    z, t, _ = L['resist']
    p.add_mesh(M.slab((0, 0, z - t / 2), window, window, t * 0.3),
               **S.mat('plastic', '#4a3f52', opacity=0.85))
    p.add_mesh(M.strip_lines(DRAWN_STRIPS, (-half, half), (-half, half), z,
                             along='u', width=STRIP_PITCH_MM * 0.80),
               **S.mat('plastic', '#8f6fa8', specular=0.4))
    anchor('resist', z)

    # micromesh: a woven grid, drawn as two crossed sets of fine bars
    z, t, _ = L['mesh']
    n_mesh = 26
    for along in ('u', 'v'):
        p.add_mesh(M.strip_lines(n_mesh, (-half, half), (-half, half),
                                 z + (0.35 if along == 'u' else -0.35),
                                 along=along, width=window / n_mesh * 0.40),
                   **S.mat('mesh', '#6f7883'))
    anchor('mesh', z)

    # drift volume + cathode
    z, t, _ = L['gas']
    gas = M.slab((0, 0, z), window, window, t)
    p.add_mesh(gas, **S.mat('gas', S.COL['gas'], opacity=0.11))
    p.add_mesh(gas.extract_feature_edges(), color=S.COL['gas'], line_width=2.4,
               lighting=False, opacity=0.8)
    anchor('gas', z)

    zc, tc, _ = L['cathode']
    p.add_mesh(M.slab((0, 0, zc), window, window, tc),
               **S.mat('plastic', '#dfe7ef', opacity=0.42))
    anchor('cathode', zc)

    if track:
        add_track_and_drift(p, L, window)
    return anchors


def add_track_and_drift(p, L, window):
    """A muon through the drift volume, plus the ionisation drifting to the mesh.

    This is the micro-TPC picture the whole reconstruction rests on: each
    primary cluster drifts down at v_drift, so its arrival time measures its
    depth, and the depth-vs-position slope is the track angle.
    """
    half = window / 2
    z0, t0, _ = L['gas']
    z_bot, z_top = z0 - t0 / 2, z0 + t0 / 2
    z_mesh = L['mesh'][0]

    # the muon: enters the top of the drift volume at an angle
    tanx, tany = 0.42, 0.14
    x_top, y_top = -half * 0.30, -half * 0.15
    a = np.array([x_top, y_top, z_top])
    b = np.array([x_top + tanx * (z_bot - z_top),
                  y_top + tany * (z_bot - z_top), z_bot])
    ext = 0.28 * (z_top - z_bot)
    d = (b - a) / np.linalg.norm(b - a)
    p.add_mesh(M.tube(a - d * ext, b + d * ext, 0.9),
               **S.mat('glow', S.COL['track_mu']))

    # primary ionisation clusters along it, and their drift down to the mesh
    n = 11
    for f in np.linspace(0.06, 0.94, n):
        q = a + (b - a) * f
        p.add_mesh(M.cylinder(tuple(q), (0, 0, 1), radius=1.0, height=1.0),
                   **S.mat('glow', '#ffd166'))
        # drift line: straight down to the mesh plane
        p.add_mesh(M.tube(q, (q[0], q[1], z_mesh + 1.2), 0.28),
                   **S.mat('glow', '#8fd8ea', ambient=0.55))
