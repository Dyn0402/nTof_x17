#!/usr/bin/env python3
"""sps_dp_scan.py <sps window pickle> --bundle B --out OUT.parquet [--dp 0.003,...] [--nmax N] [--jobs N]
-- which transverse diffusion do the SPS rows prefer?

The SPS pass rows are fitted with the bench bundle's Dp (chamber E: 0.020 mm/sqrt(ns)), a gas property that does not
transfer (model_audit F20): Magboltz for the beam's wet Ar/CO2/iso 93/5/2 (the ATLAS NSW gas, docs/GAS_MIXTURES.md) at
the SPS gas-fit composition (1.7 % water, 280-320 ppm O2) gives D_T = 169-179 um/sqrt(cm) at v = 7.5-10 um/ns, i.e.
Dp = D_T sqrt(v) = 0.0049-0.0054 (sps_resim.gas_dp; 'gas' in --dp is that row's value).  Until 2026-11-20 this said
0.0055-0.006, from a Magboltz grid for 95/3/2, a wrong mixture label.  This script refits every plane of a row
with the pass fitter (wft.reco.fit_plane, the row's adaptation as sps_reco._init) at a list of Dp values, everything else
as the pass, and stores chi2, dof and the fit (p0, w, t0) per plane, view and Dp.  The X view (normal incidence, no RC
term) measures the diffusion directly from the growth of the lateral width with drift time; the Y view (inclined, RC)
shows what the choice does to the fitted slope.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'rerun', 'model_audit')]
G = {}


def init(bundle, meta, K, dps):
    from wft import model as wm, reco as wr
    from wft.calib import CalibrationBundle
    import sps_mask
    cal = CalibrationBundle.load(bundle)
    sps_mask.apply(cal, 'h4')
    cal.v_drift = meta['v_drift']
    cal.t0_abs, cal.t0_prior_sigma = {}, 0.0
    cal.bow, cal.v_map, cal.w0 = {}, {}, {}
    wm.use_calibration(cal)
    wm.set_depth_bins(K)
    wm.set_nsamp(meta['nsamp'])
    G.update(wm=wm, wr=wr, cal=cal, dps=dps, key=meta['key'])


def one(p):
    wr, cal = G['wr'], G['cal']
    P = dict(W=p['W'], noise=p['noise'], pos=p['pos'], ch=p['ch'])
    rows = []
    for dp in G['dps']:
        hy = dict(cal.hyper, Dp=dp)
        try:
            f = wr.fit_plane(P, p['view'], cal, hyper=hy)
        except Exception:
            f = None
        rows.append(dict(row=G['key'], event_id=p['event_id'], view=p['view'], Dp=dp, w_pass=p['w'], chi2_pass=p['chi2'],
                         chi2=getattr(f, 'chi2', np.nan), dof=getattr(f, 'dof', 0), w=getattr(f, 'w', np.nan),
                         p0=getattr(f, 'p0', np.nan), t0=getattr(f, 't0', np.nan)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickle')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dp', default='0.003,0.006,0.009,0.012,0.016,0.020,0.025')
    ap.add_argument('--nmax', type=int, default=400, help='planes per view')
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    D = pickle.load(open(a.pickle, 'rb'))
    meta = D['meta']
    dps = []
    for x in a.dp.split(','):
        if x == 'gas':
            import sps_resim
            g = sps_resim.gas_dp(meta, D['planes'])
            print(f"   gas Dp {meta['key']}: {g}", flush=True)
            dps.append(round(g['Dp'], 5))
        else:
            dps.append(float(x))
    sel = []
    for v in 'xy':
        sel += [p for p in D['planes'] if p['view'] == v and p.get('q') is not None][:a.nmax]
    K = len(sel[0]['q'])
    out = []
    with mp.Pool(a.jobs, initializer=init, initargs=(a.bundle, meta, K, dps)) as pool:
        for rows in pool.imap_unordered(one, sel, chunksize=4):
            out += rows
    T = pd.DataFrame(out)
    T.to_parquet(a.out)
    print(f"== {meta['key']}: {len(sel)} planes")
    for v in 'xy':
        g = T[T.view == v]
        piv = g.pivot(index='event_id', columns='Dp', values='chi2').dropna()
        wp = g.pivot(index='event_id', columns='Dp', values='w').loc[piv.index]
        ref = piv[0.020] if 0.020 in piv else piv.iloc[:, -1]
        print(f'  [{v}] {len(piv)} planes; per Dp: sum chi2 - sum chi2(0.020), median per-plane dchi2, fraction of planes '
              f'with their best chi2 here, median |w| [um/ns]')
        best = piv.idxmin(axis=1)
        for dp in piv.columns:
            print(f'     Dp {dp:.4f}  dsum {piv[dp].sum() - ref.sum():+10.0f}  med {np.median(piv[dp] - ref):+7.1f}'
                  f'  best {np.mean(best == dp):.2f}  |w| {np.median(np.abs(wp[dp])) * 1e3:.3f}', flush=True)


if __name__ == '__main__':
    main()
