"""gaps.py -- the one drift-gap table for cloud_basics (FINDINGS §36, §39).

Measured by gap_extent.py without any timing: the half-height edges of the per-strip charge of inclined
bench tracks against depth z = (x - x_mesh) / tan_ref, window integral, |tan| >= 0.2, X view.  Toy closure
(gap_extent_toy.py) within +-0.15 mm at v = 26-38 um/ns.

Since §39 (2026-10-19, pm) each value is the mean of two selections: events with no saturated sample (the §36
default) and every event with only its clipped strips masked (gap_extent.py --satmask), with half their
difference added in quadrature to the statistical (bootstrap) error.  The flagged strips are the head-on
dead/weak lists plus the inclined-track hole test (strip_gain_audit.bad_set); that moved det7's long run
30.1 -> 28.2 (20 more dead X channels).  det7 combines its long and longer runs.  The M3 slope scale enters
as gap x (1 + scale error) and is not included.

det6 cannot be measured this way: its deep charge runs past the 32-sample window (last sample 40-86 % of
the deep strips' peak), so its column is a lower limit (26.3).  Adopted 28.0 +- 1.0 (Dylan, 2026-10-24) from
the two estimates that are not window-limited: the 7-30 GAP_STUDY map median 27.85 and the reco-ladder v x
stacked drift time 27.5-28 (rerun C19).  Mechanical design: 30 mm, with a couple of mm of build tolerance.
"""

GAP_GEOM = {'det2': 31.12, 'det3': 28.08, 'det4': 30.48, 'det7': 29.44, 'det6': 28.0}
GAP_GEOM_ERR = {'det2': 0.42, 'det3': 0.19, 'det4': 0.44, 'det7': 1.12, 'det6': 1.0}
GAP_NOMINAL = 30.0


def gap(det):
    """Drift gap [mm] of a chamber: the measured one, else the mechanical 30 mm."""
    return GAP_GEOM.get(det, GAP_NOMINAL)
