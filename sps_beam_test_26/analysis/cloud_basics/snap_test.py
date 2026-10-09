#!/usr/bin/env python3
"""snap_test.py -- does X charge snap to the resistive-strip grid?

Board (mpgd26/scenes_chamber.py, gerbers): resistive film in strips 550 um wide on a
0.80 mm pitch, running along y; X readout strips 0.78 mm pitch, parallel to them; Y
readout strips across them.  If an avalanche's charge equalises across the width of the
resistive strip it lands on, X sees it centred on THAT strip's centre: X charge
centroids pile up on a 0.80 mm grid.  Readout-side non-linearity (the centroid's own
DNL) piles them up on the 0.78 mm readout grid instead.  Y (along the resistive strips)
can only show the readout period.

Near-head-on tracks (|tan| < TAN), charge centroid of the view's time-integrated
strip charges, Rayleigh periodogram of the centroid positions over trial periods
0.70-0.90 mm: power R(P) = |mean exp(2 pi i x / P)|^2 * N.  N random phases give R ~ 1.

    snap_test.py det2=<big_cache.pkl> ...
Output: results/snap_test.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAN = 0.05
PERIODS = np.linspace(0.70, 0.90, 801)


def centroids(ev, view):
    xs = []
    for e in ev.values():
        if view not in e or abs(e[f'tan_{view}']) > TAN:
            continue
        P = e[view]; pos = np.asarray(P['pos'], float); W = np.asarray(P['W'], float)
        q = W.sum(1)
        k = int(np.argmax(q))
        sel = slice(max(k - 2, 0), k + 3)
        qq = np.clip(q[sel], 0, None)
        if qq.sum() <= 0:
            continue
        xs.append(float((pos[sel] * qq).sum() / qq.sum()))
    return np.array(xs)


def periodogram(x):
    ph = 2j * np.pi * x[None, :] / PERIODS[:, None]
    return np.abs(np.exp(ph).mean(1)) ** 2 * len(x)


def main():
    res = {'periods': PERIODS.tolist()}
    for arg in sys.argv[1:]:
        det, path = arg.split('=', 1)
        ev = pickle.load(open(path, 'rb'))
        res[det] = {}
        for view in ('x', 'y'):
            x = centroids(ev, view)
            R = periodogram(x)
            i78 = np.argmin(abs(PERIODS - 0.78)); i80 = np.argmin(abs(PERIODS - 0.80))
            ipk = int(np.argmax(R))
            res[det][view] = dict(n=int(len(x)), R=R.tolist(), R078=float(R[i78]), R080=float(R[i80]),
                                  peak_period=float(PERIODS[ipk]), peak_R=float(R[ipk]))
            print(f'{det} {view}: n={len(x):4d}  R(0.78) {R[i78]:6.1f}  R(0.80) {R[i80]:6.1f}  '
                  f'peak at {PERIODS[ipk]:.4f} mm (R {R[ipk]:.1f})')
    json.dump(res, open(os.path.join(HERE, 'results', 'snap_test.json'), 'w'))


if __name__ == '__main__':
    main()
