"""pass1_basis.py -- the FD toy (ladder.py, scan_a.py) on the pass-1 basis instead of the 10-27 audit basis (2026-10-31).

Switched on by FD_BASIS=pass (FD_INPUTS = toy_inputs_pass1.json, FD_PLACE = ig | full); without it the toy is the 10-27
one, bit for bit.  What moves, all MEASURED on the pass-1 calibration rows (toy_inputs_pass1.py):
  bundles     FD_AUDIT points at the pass-1 calib bundles (template, sigma_p0, dead mask, t0 prior, kernel)
  s0          the generated footprint of the ideal rungs = the pass bundle's sigma_p0_x (0.30), so L1 stays ideal
  q50         median x_q_sum of the row (sample = FD_PLACE)
  dead_extra  dead strips the pass mask still misses: det2 ch 69 (cloud_basics 39; ch 269 is in the pass mask),
              det3 ch 39 (cloud_basics 39); det4 none (over-masks); det6 none (the 15-strip 177-185 mm block that
              the audit mask missed, ws_dead_charge F12, is ch 208-224 of the pass mask); det7 none on interior_good
              (X ch 256-511 only: every chamber-side det7 fault is below ch 256), 18 random strips below ch 256 on
              'full' (F12's count missing from the golden mask)
  placement   ig: the track in chip blocks 1-6 (ch 64-447; det7 4-6), an approximation of interior_good (>= 30 mm
              from the X edges; det7 X ch 256-511); full: any of the 8 blocks (the 10-27 placement)
  vdrift      v_true = v * v_rel, v_rel drawn from the row's own per-event r(t) (the v(t) map; uniform |dv| below the
              gate) instead of a Gaussian of the FOLLOWUPS-1 rms
  ang_w       the data's |theta_ref| mix (weights of the toy's 3 / 9 / 16 deg points), used by ladder.collect for the
              weighted 'all_w' summary next to the equal-weight 'all' of 10-27
"""
import json
import os

import numpy as np

ON = os.environ.get('FD_BASIS') == 'pass'
PLACE = os.environ.get('FD_PLACE', 'ig')
TAG = f'pass1_{PLACE}' if ON else ''
_HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = None
if ON:
    INPUTS = json.load(open(os.environ.get('FD_INPUTS') or os.path.join(_HERE, 'results', 'toy_inputs_pass1.json')))

DEAD_EXTRA = dict(det2=[69], det3=[39], det4=[], det6=[], det7=[])
BLOCKS_IG = dict(det2=(1, 2, 3, 4, 5, 6), det3=(1, 2, 3, 4, 5, 6), det4=(1, 2, 3, 4, 5, 6), det6=(1, 2, 3, 4, 5, 6),
                 det7=(4, 5, 6))
BLOCKS = {d: (BLOCKS_IG[d] if (ON and PLACE == 'ig') else None) for d in BLOCKS_IG}
V_REL = {}


def apply(ladder_mod, ft):
    """override the toy's audit-basis constants in floor_toy (ft) and, if given, ladder.py's REALISM table."""
    if not ON:
        return
    for det, rec in INPUTS.items():
        c = ft.CHAMBERS[det]
        c['s0'] = rec['s0']
        c['q50'] = rec['q50'][PLACE]
        c['v'] = rec['v_drift'] / 1e3
        c['Dp'] = rec['Dp']
        V_REL[det] = np.asarray(rec['v_rel'], float)
        if ladder_mod is not None:
            R = ladder_mod.REALISM[det]
            if det == 'det7' and PLACE == 'full':
                rs = np.random.default_rng(778)
                pool = np.setdiff1d(np.arange(256), rec['dead_x'])
                R['dead_extra'] = sorted(int(x) for x in rs.choice(pool, 18, replace=False))
            else:
                R['dead_extra'] = list(DEAD_EXTRA[det])
            R['v_rms'] = rec['v_rms']


def ang_weights(det):
    return INPUTS[det]['ang_w'][PLACE] if ON else None


def wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def s68_w(d, ang, wmap):
    """FD's s68 (|d - median| 68th percentile over |d| <= 5) with each event weighted by its angle point's data weight
    divided by the point's event count, so the pooled sample has the data's angle mix."""
    d, ang = np.asarray(d, float), np.asarray(ang, float)
    k = np.isfinite(d) & (np.abs(d) <= 5)
    d, ang = d[k], ang[k]
    if len(d) < 6:
        return np.nan
    w = np.zeros(len(d))
    for a, wa in wmap.items():
        m = ang == a
        if m.any():
            w[m] = wa / m.sum()
    if w.sum() <= 0:
        return np.nan
    med = wquantile(d, w, 0.5)
    return wquantile(np.abs(d - med), w, 0.68)
