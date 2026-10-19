"""gaps.py -- the one drift-gap table for cloud_basics (FINDINGS §36).

Measured by gap_extent.py without any timing: the half-height edges of the per-strip charge of inclined
bench tracks against depth z = (x - x_mesh) / tan_ref, window integral, |tan| >= 0.2, X view.  Toy closure
(gap_extent_toy.py) within +-0.15 mm at v = 26-38 um/ns.  Errors are bootstrap (statistical); the M3
slope scale enters as gap x (1 + scale error) and is not included.

det6 cannot be measured this way: its deep charge runs past the 32-sample window (last sample 40-86 % of
the deep strips' peak), so its column is a lower limit; it keeps the mechanical 30 mm with a +-2 mm
systematic.  Mechanical design: 30 mm, with a couple of mm of build tolerance (Dylan, 2026-10-19).
"""

GAP_GEOM = {'det2': 31.45, 'det3': 28.10, 'det4': 30.67, 'det7': 29.85}
GAP_GEOM_ERR = {'det2': 0.38, 'det3': 0.23, 'det4': 0.41, 'det7': 1.0, 'det6': 2.0}
GAP_NOMINAL = 30.0


def gap(det):
    """Drift gap [mm] of a chamber: the measured one, else the mechanical 30 mm."""
    return GAP_GEOM.get(det, GAP_NOMINAL)
