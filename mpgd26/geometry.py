#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py -- the single source of truth for the MPGD-conference 3-D scenes.

Every number here is copied from a run config or a measured record, with the
provenance in the comment next to it.  Nothing is invented silently: the few
quantities that are *not* measured are collected in ``ASSUMPTIONS`` at the
bottom and are flagged in the rendered figure captions.

Two scenes:

  SPS  -- the July/August 2026 H4 parasitic beam telescope in P2 (banco).
          Frame: beam along +Z (downstream), +Y vertical up, X horizontal,
          Y = 0 at the mechanical table top.

  BENCH -- the Saclay cosmic bench.
          Frame: +Z vertical up (this is the bench's own frame, as used by
          every run config and by the whole analysis chain), X/Y transverse.
          Z = 0 is the bench origin the run configs are written against.

Sources
-------
SPS z positions and detector list
    sps_beam_test_26/analysis/run_inventory.json  ->  run_configs/run_59
    (= lxplus:~/x17/p2_sps_july/runs/run_59/run_config.json)

P2 BASKET fan shape, apex, mounting, beam spot, trigger slab
    sps_beam_test_26/det4_sps_assessment/SPS_BEAM_GEOMETRY_2026-07-31.md
    sps_beam_test_26/det4_sps_assessment/sps_beam_data/P2_BASKET_mapping.csv

Cosmic-bench levels, M3 planes, MX17 slots
    lxplus:~/x17/cosmic_bench/june_tests/Run/mx17_det2_det3_overnight_6-22-26/
        run_config.json   ->  bench_geometry + detectors

Trigger scintillators (60 x 60 cm, one above and one below the stack)
    mx_june_cosmic_qa/engineer_package/make_bench_diagram.py

MX17 active area
    common/mx17_active_area.py
"""
from __future__ import annotations

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

P2_MAP_CSV = os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment',
                          'sps_beam_data', 'P2_BASKET_mapping.csv')

# --------------------------------------------------------------------------- #
# Shared detector dimensions
# --------------------------------------------------------------------------- #
MX17_PCB_MM = 449.68 - (-20.32)      # 470.0 mm square PCB   (14_board_map.py)
MX17_PCB_LO = -20.32                 # detector-local, PCB edge
MX17_ACTIVE_MM = 398.58              # metallised strip region (mx17_m1_map.csv)
MX17_Y_PASSIVATION = (18.0, 379.9)   # true efficient Y (common/mx17_active_area)
MX17_DRIFT_GAP_MM = 30.0             # nominal drift gap (run_config 'drift_gap')
MX17_FRAME_MM = 40.0                 # frame lip drawn outside the PCB (visual)

M3_ACTIVE_MM = 500.0                 # M3 reference Micromegas, 50 x 50 cm
M3_FRAME_MM = 560.0                  # drawn frame (visual)
M3_THICK_MM = 22.0                   # drawn thickness (visual)

URW_ACTIVE_MM = 127.4                # EIC uRWELL active (SPS_BEAM_GEOMETRY 3)
URW_PCB_MM = 200.0                   # drawn board (det_size is 130; frame visual)
URW_THICK_MM = 12.0

SCINT_MM = 600.0                     # 60 x 60 cm trigger paddles
SCINT_THICK_MM = 30.0

# --------------------------------------------------------------------------- #
# SPS  --  H4 parasitic telescope in P2
# --------------------------------------------------------------------------- #
# z along the rail, mm, from run_59/run_config.json det_center_coords.
SPS_STATIONS = [
    # (name, z_mm, kind, label)
    ('EIC_uRWELL_front', 0.0,    'urwell', 'EIC uRWELL\n(front reference)'),
    ('P2_IN',            320.0,  'p2',     'P2 BASKET  IN'),
    ('P2_MID',           630.0,  'p2',     'P2 BASKET  MID'),
    ('P2_OUT',           940.0,  'p2',     'P2 BASKET  OUT'),
    ('mx17_E',           1155.0, 'mx17',   'MX17 "Detector E"'),
    ('EIC_uRWELL_back',  1370.0, 'urwell', 'EIC uRWELL\n(back reference)'),
]
SPS_MX17_Z_IS_PLACEHOLDER = True     # run_config says so explicitly

# --- P2 BASKET fan, mechanical (SPS_BEAM_GEOMETRY_2026-07-31.md sect. 3) ------
P2_APEX_PAD = (-33.69, -20.58)       # fan apex in the pad frame, back-solved
P2_BISECTOR_DEG = 30.074             # azimuth of the fan bisector, pad frame
P2_R_ACTIVE = (150.7, 635.0)         # metallised area, radius from the apex
P2_PHI_ACTIVE = (2.30, 57.85)        # metallised area, azimuth, pad frame [deg]
P2_TABLE_GAP = 130.0                 # table top -> lowest point of active area
P2_APEX_HEIGHT = P2_TABLE_GAP + P2_R_ACTIVE[1]      # 765.0 mm above the table
P2_MIRROR = -1                       # beam's-eye view convention (16_ script)
P2_THICK_MM = 6.0                    # drawn board thickness
P2_INSTRUMENTED_SECTORS = (3, 6)     # sectors 3-6 of 10 are read out (ch 384-895)

# --- the H4 beam, as measured through the P2 telescope ------------------------
SPS_BEAM_CENTRE_PAD = (414.7, 232.4)  # illumination centroid, P2 pad frame
SPS_BEAM_HEIGHT = 250.0              # beam axis above the table [mm]
SPS_BEAM_SIGMA_H = 28.6              # horizontal (true beam) sigma [mm]
SPS_TRIGGER_SLAB = (186.0, 311.0)    # scintillator aperture, height above table
SPS_BEAM_DIVERGENCE_MRAD = 0.5       # < 0.5 mrad over 620 mm; beam is parallel
SPS_BEAM_LATERAL_OFFSET = 6.0        # off the fan centre-line [mm]


def sps_beam_centre_lab():
    """Beam centre as (x, height) in the lab, from the measured pad-frame
    centroid pushed through the same mounting transform as the board itself.

    Deriving it rather than hard-coding +-6 mm keeps the beam on the side of
    the fan centre-line that the ``P2_MIRROR`` view convention puts it on --
    the one quantity in the whole scene whose sign depends on that convention.
    """
    return p2_pad_to_lab(np.asarray(SPS_BEAM_CENTRE_PAD, float))

# --------------------------------------------------------------------------- #
# Saclay cosmic bench
# --------------------------------------------------------------------------- #
# bench_geometry + detectors, mx17_det2_det3_overnight_6-22-26/run_config.json
BENCH_P1_Z = 227.0                   # lower test level (board top face)
BENCH_P2_Z = 697.0                   # upper test level
BENCH_BOARD_THICKNESS = 5.0
BENCH_DUT_Z = {'P1': 232.0, 'P2': 702.0}     # det_center_coords.z as configured
BENCH_LEVEL_SPACING = 97.0           # rail level pitch
BENCH_BOTTOM_LEVEL_Z = 82.0          # lowest rail level

BENCH_M3_Z = {                       # det_center_coords.z, m3_* detectors
    'm3_bot_bot': 24.0,
    'm3_bot_top': 144.0,
    'm3_top_bot': 1185.0,
    'm3_top_top': 1302.0,
}

# NOT in any run config -- the paddles are hardware outside the DAQ geometry.
# These are the values the engineer-package side-view schematic uses ("drawn
# just outside the stack"); see ASSUMPTIONS.
BENCH_SCINT_Z = {'bottom': -110.0, 'top': 1420.0}

# Bench structure (drawn, not surveyed): four uprights carrying the rail levels.
BENCH_POST_XY = 340.0                # post centres: just outside the 60 cm
                                     # scintillators, so a plane rests on rails
                                     # spanning post to post
BENCH_POST_SECTION = 45.0            # extruded-profile cross-section
BENCH_POST_Z = (-230.0, 1560.0)      # post span in z


# --------------------------------------------------------------------------- #
# P2 fan helpers
# --------------------------------------------------------------------------- #
def load_p2_pads():
    """Return (dataframe, pad corner polygons (npad, 4, 2), apex) in the pad frame.

    Identical construction to
    ``sps_beam_test_26/det4_sps_assessment/15_sps_beam_board.py`` so the two
    never disagree: rotated rectangles from pad_cx/cy/w/h/angle, and the apex
    back-solved from the map's own radius/phi columns (which are built on x/y,
    not on the pad centroids).
    """
    import pandas as pd

    m = pd.read_csv(P2_MAP_CSV)
    a = np.radians(m.pad_angle.values)
    hw, hh = m.pad_w.values / 2, m.pad_h.values / 2
    cx, cy = m.pad_cx.values, m.pad_cy.values
    ux, uy = np.cos(a), np.sin(a)
    corners = [np.stack([cx + sx * hw * ux - sy * hh * uy,
                         cy + sx * hw * uy + sy * hh * ux], axis=1)
               for sx, sy in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))]
    polys = np.stack(corners, axis=1)

    ax = m.x.values - m.radius.values * np.cos(m.phi.values)
    ay = m.y.values - m.radius.values * np.sin(m.phi.values)
    apex = np.array([ax.mean(), ay.mean()])
    return m, polys, apex


def p2_pad_to_lab(p, apex=None, h_apex=P2_APEX_HEIGHT, mirror=P2_MIRROR):
    """P2 pad frame -> lab (x, height).  Same transform as ``16_..._lab_frame.py``.

    The fan bisector is rotated onto straight-down, the apex sits at
    ``h_apex`` above the table, and the fan is centred on x = 0.
    """
    if apex is None:
        apex = np.asarray(P2_APEX_PAD, float)
    th = math.radians(-90.0 - P2_BISECTOR_DEG)
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(p, float) - np.asarray(apex, float)
    x = d[..., 0] * c - d[..., 1] * s
    y = d[..., 0] * s + d[..., 1] * c
    return np.stack([mirror * x, y + h_apex], axis=-1)


def p2_arc(r, phi_lo_deg, phi_hi_deg, n=200, apex=None, h_apex=P2_APEX_HEIGHT):
    """One constant-radius arc of the fan, as an (n, 2) lab-frame polyline."""
    if apex is None:
        apex = np.asarray(P2_APEX_PAD, float)
    ph = np.radians(np.linspace(phi_lo_deg, phi_hi_deg, n))
    pts = np.stack([apex[0] + r * np.cos(ph), apex[1] + r * np.sin(ph)], axis=1)
    return p2_pad_to_lab(pts, apex=apex, h_apex=h_apex)


def p2_band(r_lo, r_hi, phi_lo_deg, phi_hi_deg, n=200, **kw):
    """(inner_arc, outer_arc) for the annulus sector between two radii."""
    return (p2_arc(r_lo, phi_lo_deg, phi_hi_deg, n, **kw),
            p2_arc(r_hi, phi_lo_deg, phi_hi_deg, n, **kw))


# --------------------------------------------------------------------------- #
# What is measured, and what is not
# --------------------------------------------------------------------------- #
ASSUMPTIONS = {
    'sps_transverse_alignment': (
        'Every SPS run_config carries det_center_coords x = y = 0 -- nominal, '
        'not surveyed.  The uRWELLs and MX17 are drawn centred on the beam '
        'axis (250 mm above the table), which is how they were mounted, but no '
        'transverse survey exists to confirm it.'),
    'sps_mx17_z': (
        'MX17 "Detector E" at z = 1155 mm is flagged PLACEHOLDER in the run '
        'config itself.'),
    'p2_mounting': (
        'Fan bisector vertical, apex up, centred, 130 mm from the lowest active '
        'point to the table top -- supplied by the P2 group, corroborated by the '
        'trigger aperture coming out horizontal, but not surveyed.'),
    'bench_scintillator_z': (
        'The 60 x 60 cm trigger paddles are not in any run config (they are '
        'hardware outside the DAQ geometry).  They are drawn just outside the '
        'stack at z = -110 / +1420 mm, matching the engineer-package schematic.'),
    'bench_structure': (
        'The bench uprights and rails are drawn for context and are not '
        'surveyed dimensions.'),
    'drawn_thicknesses': (
        'Chamber/frame thicknesses are chosen for legibility; only the 30 mm '
        'MX17 drift gap is a real dimension.'),
}
