#!/usr/bin/env python3
"""charge_vs_depth.py -- template-free attachment test on inclined X tracks.

For |tan_x| in [0.15, 0.35] the track crosses 5-13 X strips over the 30 mm gap;
X has no resistive spreading, so each strip's time-integrated charge is the
charge of its depth segment (plus prompt sharing, which conserves the sum in the
interior).  Strips are ordered along the track by the M3 direction; the strip
whose segment is at the mesh is the one at the reference mesh position.  The
depth of strip k is (k pitches) / |tan|.  Per event, charge is normalised to the
event's median interior strip; the stack is the median per depth bin.
Attachment would make charge fall with depth; none means flat.

    charge_vs_depth.py
"""
import os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES, PITCH

BINS = np.arange(0, 33, 3.0)
for name in ('det2', 'det3', 'det4', 'det6', 'det7'):
    ev = pickle.load(open(SOURCES[name], 'rb'))
    dep, q = [], []
    n = 0
    for e in ev.values():
        if 'x' not in e:
            continue
        t = e['tan_x']
        if not 0.15 <= abs(t) <= 0.35:
            continue
        P = e['x']; pos = np.asarray(P['pos']); W = np.asarray(P['W'], float)
        Q = W.sum(1)
        # depth of each strip centre along the track: (pos - mesh_pos) / tan  (>= 0 inside the gap)
        z = (pos - e['ref_mesh_x']) / t
        inside = (z > 1.5) & (z < 28.5)          # skip the end strips (partial segments)
        if inside.sum() < 4:
            continue
        med = np.median(Q[inside])
        if med <= 0:
            continue
        dep += list(z[inside]); q += list(Q[inside] / med); n += 1
    dep, q = np.array(dep), np.array(q)
    row = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (dep >= lo) & (dep < hi)
        row.append(np.median(q[m]) if m.sum() > 30 else np.nan)
    row = np.array(row)
    zc = 0.5 * (BINS[:-1] + BINS[1:]); ok = np.isfinite(row)
    sl = np.polyfit(zc[ok], np.log(row[ok]), 1)[0]
    lam = -1 / sl if sl < 0 else np.inf
    print(f'{name} n={n:4d}  Q(z)/median, z = ' + ' '.join(f'{c:.0f}:{r:.2f}' for c, r in zip(zc, row)) +
          f'  | exp fit lambda = {lam:.0f} mm')
