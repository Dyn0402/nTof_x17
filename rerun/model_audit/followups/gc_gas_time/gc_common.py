"""gc_common.py -- shared helpers of GC (gas change vs time, consequences for the reconstruction)."""
import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [os.path.join(HERE, '..', 'fa_gas_v_tilt')]
from fa_common import load_colsum, tmean  # noqa: E402,F401

LOCAL = os.path.expanduser('~/x17/cosmic_bench/model_audit/followups')
GC_LOCAL = os.path.join(LOCAL, 'gc')                       # local pulls of the GC condor products
FA_LOCAL = os.path.join(LOCAL, 'fa')
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)
TICK_S = 10e-9                                             # DREAM timestamp tick (FA: 15.006 min on 15-min rows)
# fixed categorical order (dataviz reference palette), colour follows the chamber: bench det2/3/4/6/7, and the n_TOF
# arm carries its bench chamber's colour (A = det3, B = det2, C = det6, D = det7)
COL = {'mx17_2': '#2a78d6', 'mx17_3': '#eb6834', 'mx17_4': '#1baf7a', 'mx17_6': '#eda100', 'mx17_7': '#6250d6',
       'A': '#eb6834', 'B': '#2a78d6', 'C': '#eda100', 'D': '#6250d6'}
NAME = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}


def stem_time(tag):
    """'260622_20H20_000' -> datetime of the acquisition start (minute resolution, local CERN time)."""
    d, hm = tag.split('_')[:2]
    return dt.datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]), int(hm[:2]), int(hm[3:5]))


def robust_blocks(t, y, width, nmin=100):
    """trimmed means of y in consecutive windows of `width` (same unit as t); (centres, means, errs, n)."""
    if len(t) == 0:
        return (np.array([]),) * 4
    edges = np.arange(t.min(), t.max() + width, width)
    if len(edges) < 2:
        edges = np.array([t.min(), t.max() + 1e-9])
    c, m, e, n = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (t >= lo) & (t < hi)
        if k.sum() < nmin:
            continue
        yy = y[k]
        c.append(float(np.median(t[k]))); m.append(tmean(yy))
        e.append(float(1.2533 * 1.4826 * np.median(np.abs(yy - np.median(yy))) / np.sqrt(k.sum())))
        n.append(int(k.sum()))
    return np.array(c), np.array(m), np.array(e), np.array(n)


def angle_bias_deg(eps, theta_deg):
    """angle error [deg] from a fractional tan-scale error eps at true angle theta: atan((1+eps) tan th) - th."""
    th = np.radians(theta_deg)
    return np.degrees(np.arctan((1 + eps) * np.tan(th)) - th)
