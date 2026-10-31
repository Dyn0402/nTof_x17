#!/usr/bin/env python3
"""mx17_fiducial.py -- the fiducial regions every headline bench number is quoted in (one definition for 03_angles,
04_maps and rerun/effbudget; PASS_PLAN_2026-10-28 §4, LESSONS L9 + 2026-07-13 / 10-25).

A reference ray is described by its M3 point on the mesh and its tangents, both in the chamber's RAW strip frame
(cosmic_micro_tpc_analysis.attach_reference_positions: ref_mesh_x_mm / ref_mesh_y_mm; _rotate_ref_tangents). Its
footprint in a view is the interval the track covers through the 30 mm drift gap, [mesh, mesh + 30 tan].

region (per ray; the worse of the two views):
  interior     the footprint is >= 30 mm from the edge of an unpassivated coordinate (X on every chamber) and >= 8 mm
               from a passivated one (Y, ~18 mm passivated per edge), in BOTH views (common.mx17_active_area)
  edge-other   inside, but < 30 / 8 mm from the edge of the OTHER coordinate
  edge a-b     a-b mm from the edge of this coordinate (unpassivated: 0-5 ... 20-30; passivated: 0-5 = 0-8 mm)
  outside      the footprint leaves the measured active area (acceptance, never in a denominator)

good_x (chamber hardware the paper excludes by position, not by quality):
  det7 (mx17_7)  X channels 256-511 only: the footprint must not touch connector card 1 (ch 0-255, x < 199.3 mm;
                 M1 badly seated on g_det7_long, pogo shorts / pickup on connectors 0-3; det7-x-readout-faults).
  others         everywhere.
"""
import numpy as np

from common.mx17_active_area import TRUE_ACTIVE_BY_DET

GAP_MM = 30.0             # drift gap the footprint is extended through (mx17-drift-gap-settled)
EDGE_OWN_MM = 30.0        # interior distance from an unpassivated edge
EDGE_PASSIV_MM = 8.0      # interior distance from a passivated edge
BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30))
DET7_X_CARD1_MM = 199.3   # boundary between det7 X ch 255 and 256: (512 - 1) / 2 x 0.78 mm
RANK = {k: i for i, k in enumerate(['interior', 'edge-other', 'edge 20-30', 'edge 15-20', 'edge 10-15', 'edge 5-10',
                                    'edge 0-5', 'outside'])}
KINDS = ('all', 'active', 'interior', 'good', 'interior_good')


def footprint(mesh, tan, gap=GAP_MM):
    a = np.asarray(mesh, float); b = a + gap * np.asarray(tan, float)
    return np.minimum(a, b), np.maximum(a, b)


def ray_regions(mesh_x, mesh_y, tan_x, tan_y, det):
    """dict of arrays: region, region_x, region_y, dist_x, dist_y (mm inside the active area, < 0 outside),
    lo_x, hi_x, lo_y, hi_y (footprint).  det = qa_config DET_NAME (mx17_2 ... mx17_7)."""
    act = TRUE_ACTIVE_BY_DET[det]
    passiv = {'x': act['x'][0] > 1.0, 'y': act['y'][0] > 1.0}
    out = {}
    for v, m, t in (('x', mesh_x, tan_x), ('y', mesh_y, tan_y)):
        lo, hi = footprint(m, t)
        out[f'lo_{v}'], out[f'hi_{v}'] = lo, hi
        out[f'dist_{v}'] = np.minimum(lo - act[v][0], act[v][1] - hi)
    for v in 'xy':
        o = 'y' if v == 'x' else 'x'
        d, od = out[f'dist_{v}'], out[f'dist_{o}']
        r = np.full(len(d), 'interior', dtype=object)
        lim_o = EDGE_PASSIV_MM if passiv[o] else EDGE_OWN_MM
        r[od < lim_o] = 'edge-other'
        if not passiv[v]:
            for a, b in BANDS:
                r[(d >= a) & (d < b)] = f'edge {a}-{b}'
        else:
            r[(d >= 0) & (d < EDGE_PASSIV_MM)] = 'edge 0-5'
        r[(d < 0) | (od < 0) | ~np.isfinite(d) | ~np.isfinite(od)] = 'outside'
        out[f'region_{v}'] = r
    out['region'] = np.array([a if RANK[a] >= RANK[b] else b for a, b in zip(out['region_x'], out['region_y'])],
                             dtype=object)
    return out


def good_x(det, lo_x):
    """True where the ray's X footprint is on hardware the paper quotes (module docstring)."""
    lo_x = np.asarray(lo_x, float)
    if det == 'mx17_7':
        return lo_x >= DET7_X_CARD1_MM
    return np.ones(len(lo_x), bool)


def fiducial_mask(kind, mesh_x, mesh_y, tan_x, tan_y, det):
    """Boolean mask of the rays in a fiducial selection: all | active (not outside) | interior | good (det7 X
    ch 256-511, active) | interior_good (both)."""
    if kind == 'all':
        return np.ones(len(np.asarray(mesh_x)), bool)
    R = ray_regions(mesh_x, mesh_y, tan_x, tan_y, det)
    act = R['region'] != 'outside'
    g = good_x(det, R['lo_x'])
    return {'active': act, 'interior': R['region'] == 'interior', 'good': act & g,
            'interior_good': (R['region'] == 'interior') & g}[kind]
