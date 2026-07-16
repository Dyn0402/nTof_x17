#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mx17_active_area.py -- PERMANENT record of the MX17 detector active area.

The MX17 chambers are nominally 40x40 cm of active area (the metallised strip
region spans 0..398.58 mm on each coordinate -- see mx17_m1_map.csv). BUT the
strip plane is PASSIVATED for ~2 cm at each edge of the Y coordinate (the FEU-Y
plane), so the true efficient area is smaller in Y. This was measured directly
from the June cosmic data (efficiency turn-on/off vs detector-local position;
see mx_june_cosmic_qa/det3_recofar_analysis/report_v2.pdf and
M3_CUT_AND_ACTIVE_AREA_NOTE.md).

Frame: detector-LOCAL strip coordinates, millimetres, origin at the corner of
the strip map (X = FEU-X plane position, Y = FEU-Y plane position). To draw
these on a plot in the aligned / M3 frame, transform the corners with the run's
alignment (_rotate_det_positions in cosmic_micro_tpc_analysis.py); helper
`active_area_corners()` returns the polygon you can push through that transform.

Measured on det3 (June 2026), reproduced identically on two independently
aligned runs (g_det3_wknd and sat_det3): the passivation is a physical
construction property, not an alignment artefact.

2026-07-14: confirmed fleet-wide. Re-measured on det2/4/6/7 (edge_chi2_extract.py
+ edge_chi2_plots.py, generalised from det3-only to any qa_config run key) --
every chamber shows the same ~18-20 mm Y passivation band (17.9-20.4 mm across
all 5 detectors, a ~2.5 mm spread consistent with measurement noise, not a
per-chamber difference), confirming this is indeed a common construction
feature. See `TRUE_ACTIVE_BY_DET` below for the per-detector values and
`active_area_corners(det_name=...)` / `draw_outlines(..., det_name=...)` to use
them. X passivation was NOT reliably re-measurable on det4/6/7: their
efficiency-vs-X profiles are corrupted by genuine hardware pathologies (det6's
board-C mesh defect, det7's weak X-plane gain gradient, det4's low gain overall)
that fool the edge-crossing finder into reporting spurious tiny "active"
windows -- inspect fig3_edges_passivation_g_det{4,6,7}*.png panel (c) before
trusting any future X re-measurement attempt. X is left at the nominal sharp
edges (confirmed clean on det2 and det3) for every detector.
"""

# --- nominal (metallised strip region) --------------------------------------
STRIP_MIN_MM = 0.0
STRIP_MAX_MM = 398.58                 # from mx17_m1_map.csv (512 strips)
NOMINAL_SIZE_MM = 400.0               # the "40 x 40 cm" quoted size

# nominal active rectangle (what "40x40 cm" means), detector-local mm
NOMINAL_ACTIVE = {'x': (STRIP_MIN_MM, STRIP_MAX_MM),
                  'y': (STRIP_MIN_MM, STRIP_MAX_MM)}

# --- measured TRUE active area (det3, June 2026) -----------------------------
# X (FEU-X plane): full width, sharp geometric edges at 0 / 398.6 mm, NO passivation.
# Y (FEU-Y plane): passivated ~18 mm at each edge -> efficient Y in [18, 380] mm.
PASSIVATION_Y_LOW_MM = 18.0           # dead band at the low-Y edge  (measured 18.0)
PASSIVATION_Y_HIGH_MM = 18.7          # dead band at the high-Y edge (measured 18.7)

TRUE_ACTIVE = {
    'x': (0.0, 398.6),                                     # full width
    'y': (PASSIVATION_Y_LOW_MM, STRIP_MAX_MM - PASSIVATION_Y_HIGH_MM),   # (18.0, 379.9)
}
# convenience: efficient area in cm (X span x Y span) ~ 40.0 x 36.2 cm
TRUE_ACTIVE_SIZE_CM = ((TRUE_ACTIVE['x'][1] - TRUE_ACTIVE['x'][0]) / 10.0,
                       (TRUE_ACTIVE['y'][1] - TRUE_ACTIVE['y'][0]) / 10.0)

# --- fleet-wide measured Y passivation (2026-07-14) --------------------------
# X held at the nominal sharp-edge value for every detector (see module
# docstring -- not reliably re-measurable on det4/6/7 with June statistics).
# Keyed by qa_config CFG.DET_NAME. Source: edge_chi2_extract.py/_plots.py on
# each detector's headline hits-level run (g_det2, sat_det3, g_det4,
# g_det6_long, g_det7_long); fig3_edges_passivation_<key>.png +
# edge_chi2_meta_<key>.json in det3_recofar_analysis/.
TRUE_ACTIVE_BY_DET = {
    'mx17_2': {'x': (0.0, 398.6), 'y': (19.7, 378.8)},   # det2 (B): pass. 19.7/19.8 mm
    'mx17_3': TRUE_ACTIVE,                                # det3 (A): pass. 18.0/18.7 mm (reference)
    'mx17_4': {'x': (0.0, 398.6), 'y': (18.4, 378.6)},   # det4 (E): pass. 18.4/20.0 mm (noisier: gain-limited)
    'mx17_6': {'x': (0.0, 398.6), 'y': (17.9, 380.7)},   # det6 (C): pass. 17.9/17.9 mm
    'mx17_7': {'x': (0.0, 398.6), 'y': (20.4, 379.3)},   # det7 (D): pass. 20.4/19.3 mm
}
# fleet-wide passivation stats (mm): low mean=18.9 (17.9-20.4 range), high mean=19.1
# (17.9-20.0 range) -- consistent within ~2.5 mm across all 5 chambers.

# provenance
MEASURED_ON = 'det3 (g_det3_wknd + sat_det3), June 2026 cosmic bench'
METHOD = 'efficiency turn-on/off vs detector-local position (50% points)'
MEASURED_ON_FLEET = ('det2 (g_det2), det3 (sat_det3), det4 (g_det4), det6 (g_det6_long), '
                     'det7 (g_det7_long), June 2026 cosmic bench, 2026-07-14 pass')


def _corners(box):
    """Closed polygon (5 points) for a {'x':(lo,hi),'y':(lo,hi)} rectangle."""
    (x0, x1), (y0, y1) = box['x'], box['y']
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def nominal_corners():
    """40x40 cm nominal outline (detector-local mm), closed polygon."""
    return _corners(NOMINAL_ACTIVE)


def active_area_corners(det_name=None):
    """Measured true active outline (detector-local mm), closed polygon.

    det_name : qa_config CFG.DET_NAME (e.g. 'mx17_6') to use that detector's
               own measured Y passivation (see TRUE_ACTIVE_BY_DET); falls back
               to the det3 reference (TRUE_ACTIVE) if None or not yet measured
               for that detector.
    """
    box = TRUE_ACTIVE_BY_DET.get(det_name, TRUE_ACTIVE) if det_name else TRUE_ACTIVE
    return _corners(box)


def draw_outlines(ax, transform=None, nominal=True, true_area=True,
                  nominal_kw=None, true_kw=None, det_name=None):
    """
    Draw the nominal 40x40 cm and/or the measured true active outline on `ax`.

    transform : optional callable (xs, ys) -> (xs', ys') mapping detector-local
                mm into the axis frame (e.g. the run's alignment). If None the
                outlines are drawn in detector-local mm.
    det_name  : qa_config CFG.DET_NAME to draw that detector's OWN measured
                active-area outline (see TRUE_ACTIVE_BY_DET); defaults to the
                det3 reference outline if None or not yet measured.
    Returns the list of Line2D handles drawn (for legend control).
    """
    handles = []
    nominal_kw = {'color': '0.4', 'ls': '--', 'lw': 1.3, 'label': 'nominal 40x40 cm',
                  **(nominal_kw or {})}
    true_kw = {'color': 'red', 'ls': '-', 'lw': 1.6, 'label': 'true active area',
               **(true_kw or {})}
    for on, poly, kw in ((nominal, nominal_corners(), nominal_kw),
                         (true_area, active_area_corners(det_name), true_kw)):
        if not on:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        if transform is not None:
            xs, ys = transform(xs, ys)
        handles += ax.plot(xs, ys, **kw)
    return handles


def alignment_transform(params):
    """
    Build a detector-local-mm -> aligned/M3-frame transform closure suitable for
    draw_outlines(ax, transform=...), from a run's alignment.

    `params` may be an AlignmentParams instance (cosmic_micro_tpc_analysis.py),
    any object/namespace with .theta_deg/.centre_x/.centre_y/.x_offset/.y_offset
    attributes, or a plain dict with those same keys (e.g. json.load'd straight
    from an alignment.json). This module intentionally has no import dependency
    on cosmic_bench_analysis, so any script can build the adapter without a
    circular import.

    Reproduces _rotate_det_positions() in cosmic_micro_tpc_analysis.py exactly:
        x' = cosθ*(x-cx) - sinθ*(y-cy) + cx + x_offset
        y' = sinθ*(x-cx) + cosθ*(y-cy) + cy + y_offset
    This is the detector-local -> aligned/M3-frame forward map used for det hit
    positions. It does NOT apply ref_x_sign -- that flips the M3 REFERENCE, not
    the detector-local outline being transformed into that frame.
    """
    import math

    def _get(key):
        return params[key] if isinstance(params, dict) else getattr(params, key)

    theta = math.radians(_get('theta_deg'))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = _get('centre_x'), _get('centre_y')
    xoff, yoff = _get('x_offset'), _get('y_offset')

    def transform(xs, ys):
        xs_out, ys_out = [], []
        for x, y in zip(xs, ys):
            dx, dy = x - cx, y - cy
            xs_out.append(cos_t * dx - sin_t * dy + cx + xoff)
            ys_out.append(sin_t * dx + cos_t * dy + cy + yoff)
        return xs_out, ys_out

    return transform


def alignment_transform_from_json(path):
    """Load alignment.json (as written by cosmic_micro_tpc_analysis.save_alignment)
    and return the detector-local -> aligned-frame transform closure. See
    alignment_transform() for the exact map applied."""
    import json
    with open(path) as f:
        data = json.load(f)
    return alignment_transform(data)


if __name__ == '__main__':
    print('MX17 active area (detector-local mm)')
    print(f'  nominal (40x40 cm): x={NOMINAL_ACTIVE["x"]}  y={NOMINAL_ACTIVE["y"]}')
    print(f'  TRUE (det3 ref):    x={TRUE_ACTIVE["x"]}  y={TRUE_ACTIVE["y"]}')
    print(f'  passivated Y band:  {PASSIVATION_Y_LOW_MM} mm (low) + {PASSIVATION_Y_HIGH_MM} mm (high)')
    print(f'  efficient size:     {TRUE_ACTIVE_SIZE_CM[0]:.1f} x {TRUE_ACTIVE_SIZE_CM[1]:.1f} cm')
    print(f'  measured on:        {MEASURED_ON}')
    print(f'\nFleet-wide (2026-07-14, {MEASURED_ON_FLEET}):')
    for det, box in TRUE_ACTIVE_BY_DET.items():
        (x0, x1), (y0, y1) = box['x'], box['y']
        print(f'  {det}: x={box["x"]}  y={box["y"]}  '
              f'passivation lo={y0:.1f}/hi={STRIP_MAX_MM - y1:.1f} mm  '
              f'size={((x1-x0)/10):.1f}x{((y1-y0)/10):.1f} cm')
