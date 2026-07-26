#!/usr/bin/env python3
"""Toy closure test: generate waveforms from the calibrated model with known
tracks, fit them back. Measures (a) noise-only intrinsic resolution of the
method, (b) the per-bin transverse centroid jitter needed to reproduce the
observed ~1 deg scatter (stochastic diffusion / delta rays / avalanche
weighting — physics floor candidates).
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
N_EV = 500
JITTERS = [0.0, 0.15, 0.30, 0.50]   # mm, per 60-ns-bin centroid displacement

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
res1 = pickle.load(open(os.path.join(BASE, 'freefit.pkl'), 'rb'))
qbank = {}
for r in res1:
    for p in ('x', 'y'):
        if p in r and 'error' not in r[p] and 'q' in r[p]:
            qbank[(r['eid'], p)] = r[p]['q']

split = json.load(open(os.path.join(BASE, 'split_ref.json')))
test = [e for e in split['test'] if e in events and (e, 'x') in qbank
        and (e, 'y') in qbank][:N_EV]
print('toy events', len(test))


def build_matrix_jitter(plane, pos, p0, w, t0, hyper, dpk):
    """Same as fm2.build_matrix but with per-bin centroid displacement dpk."""
    c1 = hyper['c1'] * (hyper['kY'] if plane == 'y' else 1.0)
    c2 = hyper['c2'] * (hyper['kY'] if plane == 'y' else 1.0)
    tau = hyper['tau_s']
    tmpl, sm = fm2._templates(plane, hyper['sigma_s'])
    n = len(pos)
    M = np.zeros((n, 32, fm2.K))
    for k in range(fm2.K):
        ua, ub = k * fm2.DT, (k + 1) * fm2.DT
        pa, pb = p0 + w * ua, p0 + w * ub
        pc = 0.5 * (pa + pb) + dpk[k]
        half = 0.5 * abs(pb - pa)
        sig = np.sqrt(hyper['sigma_p0'] ** 2 + hyper['Dp'] ** 2 * fm2.UK[k]
                      + half ** 2 / 3.0)
        Fk = 0.5 * (erf((pos + fm2.PITCH / 2 - pc) / (np.sqrt(2) * sig))
                    - erf((pos - fm2.PITCH / 2 - pc) / (np.sqrt(2) * sig)))
        h0 = np.interp(fm2.TS - (t0 + fm2.UK[k]), fm2.TGRID, tmpl, left=0, right=0)
        h1 = np.interp(fm2.TS - (t0 + fm2.UK[k] + tau), fm2.TGRID, sm, left=0, right=0)
        h2 = np.interp(fm2.TS - (t0 + fm2.UK[k] + 2 * tau), fm2.TGRID, sm, left=0, right=0)
        M[:, :, k] += Fk[:, None] * h0[None, :]
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
        M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * 32, fm2.K)


def toy_one(args):
    eid, jit, seed = args
    rng = np.random.default_rng(seed)
    ev = events[eid]
    out = dict(eid=eid, jit=jit)
    for plane in ('x', 'y'):
        P = ev[plane]
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        wtrue = tn * V * 1e-3
        t0true = 450.0
        q = qbank[(eid, plane)]
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        dpk = rng.normal(0, jit, fm2.K) if jit > 0 else np.zeros(fm2.K)
        M = build_matrix_jitter(plane, pos, p0r, wtrue, t0true, H0, dpk)
        W = (M @ q).reshape(len(pos), 32)
        W += rng.normal(0, 1, W.shape) * noise[:, None]
        W = np.minimum(W, fm2.SAT + 100)   # mimic clipping just above mask level
        # package a fake P dict (gain=1: waveforms are already 'gain-true')
        Pf = dict(W=W.astype(np.float16), pos=P['pos'], noise=P['noise'],
                  ch=np.zeros(len(pos), np.int16))  # ch->gain[0]=1? ensure 1.0
        try:
            g = fm2.init_guess(Pf, plane, tn, p0r, V * 1e-3)
            r = fm2.fit_plane(Pf, plane, *g, hyper=H0)
            out[plane] = dict(tan_ref=tn, p0_ref=p0r, w=r['w'], p0=r['p0'],
                              chi2=r['chi2'], dof=r['dof'])
        except Exception as ex:
            out[plane] = dict(error=str(ex), tan_ref=tn, p0_ref=p0r)
    return out


def rs(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))

if __name__ == '__main__':
    # neutralize gain map for toys (waveforms generated gain-free, ch=0)
    fm2.GAIN['x'] = np.ones(512)
    fm2.GAIN['y'] = np.ones(512)
    all_out = {}
    with ProcessPoolExecutor(max_workers=14) as pool:
        for jit in JITTERS:
            args = [(e, jit, 1000 + i) for i, e in enumerate(test)]
            t_ = time.time()
            rr = list(pool.map(toy_one, args, chunksize=4))
            all_out[jit] = rr
            for p in ('x', 'y'):
                dth = np.array([
                    np.degrees(np.arctan(r[p]['w'] * 1e3 / V))
                    - np.degrees(np.arctan(r[p]['tan_ref']))
                    for r in rr if 'error' not in r[p]])
                dp = np.array([r[p]['p0'] - r[p]['p0_ref']
                               for r in rr if 'error' not in r[p]])
                print(f'jit={jit:.2f}mm {p}: angle sig {rs(dth):.2f} deg  '
                      f'p0 sig {rs(dp)*1e3:.0f} um  ({time.time()-t_:.0f}s)',
                      flush=True)
    pickle.dump(all_out, open(os.path.join(BASE, 'toy_closure.pkl'), 'wb'))
    print('saved toy_closure.pkl')
