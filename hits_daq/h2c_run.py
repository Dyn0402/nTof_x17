#!/usr/bin/env python3
"""h2c_run.py <window pickle(s)> --ref REF --gap MM --out OUT.parquet [--families dream,peak] [--part i/n] [--nmax N]

Runs H2c (h2c.py) per plane and family, next to the hit ladder (u_h) and W, and writes one row per plane:
event_id, view, tan_ref, tan_W, per family u_<f> (hit ladder) and tan_h2c_<f>, cost_<f>.  tan from w with the bundle's
angle constants, as for W: tan = (1e3 w - w0) / (kw v).  Windows from boot_extract.py are gain corrected (model
compared without gain); windows from extract_windows.py / sps_extract.py are in ADC (model times the gain).
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from h2c import H2C  # noqa: E402
from tiers import ladder_u, load_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--ref', help='M3 reference (bench); omitted for SPS')
    ap.add_argument('--gap', type=float, required=True)
    ap.add_argument('--bundle')
    ap.add_argument('--out', required=True)
    ap.add_argument('--families', default='dream,peak')
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--nsig', type=float, default=5.0)
    ap.add_argument('--qmode', default='amp', choices=('amp', 'uniform'))
    ap.add_argument('--lam', type=float, default=3.0)
    ap.add_argument('--on-model', action='store_true', help='closure: fit the hits of the pass model window M (truth = W)')
    a = ap.parse_args()
    ip, npart = (int(x) for x in a.part.split('/'))
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    gain_corrected = 'raw' not in planes[0]                 # boot_extract windows
    cal = CalibrationBundle.load(a.bundle or meta['bundle'])
    wm.use_calibration(cal)
    hyper = dict(cal.hyper)
    if meta.get('sps'):
        # the SPS row's adaptation (sps_reco._init): v, K, nsamp; no bench w0
        cal.v_drift = meta['v_drift']
        wm.set_nsamp(meta['nsamp'])
    K = len(planes[0]['q']) if planes[0].get('q') is not None else wm.K
    wm.set_depth_bins(K)
    ref = load_ref(a.ref) if a.ref else {}
    fams = a.families.split(',')
    ideal = {v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in 'xy'}
    fit = {(v, f): H2C(wm, hyper, v, f, a.gap, meta['v_drift'], K, nsig=a.nsig, ideal=ideal[v], qmode=a.qmode, lam=a.lam) for v in 'xy' for f in fams}
    for (v, f), h in fit.items():
        h.learn_offsets([p for p in planes if p['view'] == v], n=200)
    sel = [p for i, p in enumerate(planes) if i % npart == ip and (not ref or (p['event_id'], p['view']) in ref)]
    if a.nmax:
        sel = sel[:a.nmax]
    vv = meta['v_drift']
    rows, t_start = [], time.time()
    for i, p in enumerate(sel):
        v = p['view']
        Wobs = (p['M'] if a.on_model else p['W']).astype(float)
        nz = p['noise'].astype(float)
        g = np.ones(len(nz)) if gain_corrected else p['gain'].astype(float)
        pos = p['pos'].astype(float)
        kw, w0 = meta['kw'][v], meta['w0'][v]
        r = dict(event_id=p['event_id'], view=v, tan_ref=ref.get((p['event_id'], v), np.nan),
                 tan_W=(p['w'] * 1e3 - w0) / (kw * vv), w_W=p['w'])
        for f in fams:
            h = fit[(v, f)]
            obs = h.emulate(Wobs, nz)
            u, _n = ladder_u(pos, obs['t'], obs['a'], weighted=f != 'binary')
            r[f'u_{f}'] = u
            res = h.fit(pos, g, nz, obs, u)
            if res is None:
                r[f'tan_h2c_{f}'] = np.nan
                continue
            r[f'w_h2c_{f}'] = res['w']
            r[f'tan_h2c_{f}'] = (res['w'] * 1e3 - w0) / (kw * vv)
            r[f'cost_{f}'], r[f'nfev_{f}'] = res['cost'], res['nfev']
        rows.append(r)
        if i % 200 == 0:
            print(f'[h2c] {i}/{len(sel)} planes, {time.time() - t_start:.0f} s', flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    T.to_parquet(a.out)
    print(f'[h2c] wrote {a.out}: {len(T)} planes in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    main()
