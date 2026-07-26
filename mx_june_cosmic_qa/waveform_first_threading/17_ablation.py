#!/usr/bin/env python3
"""Ablation study on model v2: disable one component at a time (no
recalibration — measures marginal contribution at the v2 operating point).
Each ablation fits the same N_AB test events; results saved as
freefit2_ab_<name>.pkl for wf15_benchmark.
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
N_AB = 800

hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
test = [e for e in split['test'] if e in events][:N_AB]

ABLATIONS = {
    'nogain':   dict(patch='gain'),
    'onetmpl':  dict(patch='template'),
    'nocensor': dict(patch='censor'),
    'noc2':     dict(hyper=dict(c2=0.0)),
    'nosmear':  dict(hyper=dict(sigma_s=0.0)),
    'nodiff':   dict(hyper=dict(Dp=0.0)),
    'kY1':      dict(hyper=dict(kY=1.0)),
}

CUR = dict(hyper=H0, censor=True)

def fit_one(eid):
    ev = events[eid]
    out = dict(eid=eid)
    try:
        for plane in ('x', 'y'):
            tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
            p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
            g = fm2.init_guess(ev[plane], plane, tn, p0r, V * 1e-3)
            r = fm2.fit_plane(ev[plane], plane, *g, hyper=CUR['hyper'])
            out[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r['p0'], w=r['w'],
                              t0=r['t0'], chi2=r['chi2'], dof=r['dof'],
                              amax=float(ev[plane]['W'].max()))
    except Exception as ex:
        out['error'] = str(ex)
    return out

if __name__ == '__main__':
    orig_gain = {k: v.copy() for k, v in fm2.GAIN.items()}
    orig_tmpl = {k: v.copy() for k, v in fm2.TMPL.items()}
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, ab in ABLATIONS.items():
        if only and name != only:
            continue
        # reset
        fm2.GAIN.update({k: v.copy() for k, v in orig_gain.items()})
        fm2.TMPL.update({k: v.copy() for k, v in orig_tmpl.items()})
        fm2._smear_cache.clear()
        hyper = dict(H0)
        censor = True
        if ab.get('patch') == 'gain':
            fm2.GAIN.update(x=np.ones(512), y=np.ones(512))
        elif ab.get('patch') == 'template':
            comb = 0.5 * (orig_tmpl['x'] + orig_tmpl['y'])
            fm2.TMPL.update(x=comb, y=comb)
        elif ab.get('patch') == 'censor':
            censor = False
        hyper.update(ab.get('hyper', {}))
        CUR['hyper'] = hyper

        if not censor:
            orig = fm2.chi2_plane
            def chi2_nocensor(plane, W, noise, pos, sat, p0, w, t0, hyper,
                              censor=True, _o=orig):
                return _o(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=False)
            fm2.chi2_plane = chi2_nocensor
        t_ = time.time()
        with ProcessPoolExecutor(max_workers=14) as pool:
            res = list(pool.map(fit_one, test, chunksize=4))
        if not censor:
            fm2.chi2_plane = orig
        pickle.dump(res, open(os.path.join(BASE, f'freefit2_ab_{name}.pkl'), 'wb'))
        print(f'{name}: {len(res)} events in {time.time()-t_:.0f}s', flush=True)
