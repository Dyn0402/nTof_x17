#!/usr/bin/env python3
"""Batch v2 free fits on the test split.
Usage: wf14_freefit2.py [--joint] [--n N] [--out F] [--hyper hyper_v2.json]
       [--all-angles]
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
argv = sys.argv
JOINT = '--joint' in argv
N_MAX = int(argv[argv.index('--n') + 1]) if '--n' in argv else 2000
OUT_F = argv[argv.index('--out') + 1] if '--out' in argv else (
    'freefit2_joint.pkl' if JOINT else 'freefit2.pkl')
HYPER_F = argv[argv.index('--hyper') + 1] if '--hyper' in argv else 'hyper_v2.json'
ALL_ANGLES = '--all-angles' in argv

hj = json.load(open(os.path.join(BASE, HYPER_F)))
HYPER = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
print('hypers', HYPER, 'v', V, 'joint', JOINT, flush=True)

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
if ALL_ANGLES:
    trainset = set(split['train'])
    test = [e for e in events if e not in trainset][:N_MAX]
else:
    test = [e for e in split['test'] if e in events][:N_MAX]
print('test events', len(test), flush=True)


def fit_one(eid):
    ev = events[eid]
    out = dict(eid=eid)
    try:
        gx = fm2.init_guess(ev['x'], 'x', ev['tan_x'], ev['ref_mesh_x'], V * 1e-3)
        gy = fm2.init_guess(ev['y'], 'y', ev['tan_y'], ev['ref_mesh_y'], V * 1e-3)
        if JOINT:
            r = fm2.fit_joint(ev, gx[0], gx[1], gy[0], gy[1], gx[2], hyper=HYPER)
            for plane, p0k, wk in (('x', 'p0x', 'wx'), ('y', 'p0y', 'wy')):
                tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
                p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
                out[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r[p0k], w=r[wk],
                                  t0=r['t0'], chi2=r['chi2'], dof=r['dof'],
                                  amax=float(ev[plane]['W'].max()))
        else:
            for plane in ('x', 'y'):
                g = gx if plane == 'x' else gy
                tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
                p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
                r = fm2.fit_plane(ev[plane], plane, *g, hyper=HYPER)
                out[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r['p0'], w=r['w'],
                                  t0=r['t0'], chi2=r['chi2'], dof=r['dof'],
                                  q=r['q'], amax=float(ev[plane]['W'].max()))
    except Exception as ex:
        out['error'] = str(ex)
    return out

if __name__ == '__main__':
    t0_ = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        res = list(pool.map(fit_one, test, chunksize=4))
    print(f'{len(res)} events in {time.time()-t0_:.0f}s', flush=True)
    pickle.dump(res, open(os.path.join(BASE, OUT_F), 'wb'))
    print('saved', OUT_F, flush=True)
