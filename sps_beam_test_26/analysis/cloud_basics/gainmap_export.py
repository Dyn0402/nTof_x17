#!/usr/bin/env python3
"""gainmap_export.py -- save strip_gain_audit's 2-D gain maps (ln g at strip scale) for the decks.

strip_gain_audit.gain_map keeps the map in memory only; this writes it, with the dead/weak channel lists'
positions, to results/gain_maps.json (ln g rounded to 0.01, NaN -> null).

    gainmap_export.py [--keys sat_det3,o22_long_det2,g_det4,g_det6_long,g_det7_long]
"""
import argparse
import json
import os
import pickle

import numpy as np

import strip_gain_audit as sga
import x_width as xw

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ['sat_det3', 'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
# the strip-health entry (head-on cache) whose dead/weak lists go with each map
HEALTH = {'sat_det3': 'sat_det3__long_run_resist_490V_drift_1000V'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    a = ap.parse_args()
    audit = json.load(open(f'{HERE}/results/strip_gain_audit.json'))
    out = {}
    for key in a.keys.split(','):
        m = sga.gain_map(key)
        hk = HEALTH.get(key, key)
        # channel -> position, from the head-on cache
        ev = pickle.load(open(f'{xw.CACHES}/headon_{hk}.pkl', 'rb'))
        pos = {}
        for e in ev.values():
            if 'x' in e:
                for c, p in zip(e['x']['ch'], e['x']['pos']):
                    pos[int(c)] = float(p)
        h = audit[hk]['health']['x']
        G = np.where(m['N'] > 4.0, m['G'], np.nan)
        out[key] = dict(det=xw.det_of(key), n=m['n'], gx=m['gx'].tolist(), gy=m['gy'].tolist(),
                        lng=[[None if not np.isfinite(v) else round(float(v), 2) for v in r] for r in G.T],
                        dead_x=[pos[c] for c in h['dead'] if c in pos],
                        weak_x=[pos[c] for c in h['weak'] if c in pos])
        print(key, m['n'], G.shape, 'dead', len(out[key]['dead_x']), 'weak', len(out[key]['weak_x']), flush=True)
    json.dump(out, open(f'{HERE}/results/gain_maps.json', 'w'))


if __name__ == '__main__':
    main()
