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
    ap.add_argument('--qmode', default='amp', choices=('amp', 'uniform', 'mean', 'ampmean'))
    ap.add_argument('--lam', type=float, default=3000.0, help='amp-mode smoothness weight (scan 2026-11-06: 3000)')
    ap.add_argument('--qcal', default='parity', help="mean/ampmean shape: 'parity' = mean normalised waveform-fit q of"
                    " the even event ids (odd ids scored), or an .npz with arrays x and y")
    ap.add_argument('--t-rise', type=float, default=700.0, help='ns; per-strip time error S_T (+) t_rise noise/a (h2c.py);'
                    ' 700 chosen 2026-11-06 (results/h2c_trise_sat_det3.json)')
    ap.add_argument('--t-robust', type=float, default=0.0, help='pseudo-Huber scale c of the time pulls (0 = quadratic)')
    ap.add_argument('--s-t', type=float, default=15.0, help='ns; constant term of the per-strip time error')
    ap.add_argument('--v-drift', type=float, default=0.0, help='um/ns; overrides the window meta (SPS rows: the row v, sps_score v_stage)')
    ap.add_argument('--odd-only', action='store_true', help='score odd event ids only (as the parity-calibrated modes)')
    ap.add_argument('--model-noise', default='', choices=('', 'white'),
                    help='with --on-model: add white noise at each strip\'s sigma')
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
    cal = CalibrationBundle.load(E.bundle_path(meta, a.bundle))
    wm.use_calibration(cal)
    hyper = dict(cal.hyper)
    if a.v_drift > 0:
        meta['v_drift'] = a.v_drift
    if meta.get('sps'):
        # the SPS row's adaptation (sps_reco._init): v, K, nsamp; no bench w0
        cal.v_drift = meta['v_drift']
        wm.set_nsamp(meta['nsamp'])
    K = len(planes[0]['q']) if planes[0].get('q') is not None else wm.K
    wm.set_depth_bins(K)
    ref = load_ref(a.ref) if a.ref else {}
    fams = a.families.split(',')
    ideal = {v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in 'xy'}
    qshape = {'x': None, 'y': None}
    if a.qmode in ('mean', 'ampmean'):
        if a.qcal == 'parity':
            for v in 'xy':
                Q = np.array([p['q'] for p in planes if p['view'] == v and p.get('q') is not None
                              and p['event_id'] % 2 == 0 and np.sum(p['q']) > 0], float)
                qshape[v] = (Q / Q.sum(1, keepdims=True)).mean(0)
            planes = [p for p in planes if p['event_id'] % 2 == 1]
        else:
            Z = np.load(a.qcal)
            qshape = {v: Z[v] for v in 'xy'}
        print('[h2c] qshape x', np.round(qshape['x'] * K, 2), '\n[h2c] qshape y', np.round(qshape['y'] * K, 2))
    fit = {(v, f): H2C(wm, hyper, v, f, a.gap, meta['v_drift'], K, nsig=a.nsig, ideal=ideal[v], qmode=a.qmode, lam=a.lam,
                       qshape=qshape[v], t_rise=a.t_rise, t_robust=a.t_robust, s_t=a.s_t) for v in 'xy' for f in fams}
    for (v, f), h in fit.items():
        h.learn_offsets([p for p in planes if p['view'] == v], n=200)
    if a.odd_only:
        planes = [p for p in planes if p['event_id'] % 2 == 1]
    sel = [p for i, p in enumerate(planes) if i % npart == ip and (not ref or (p['event_id'], p['view']) in ref)]
    if a.nmax:
        sel = sel[:a.nmax]
    vv = meta['v_drift']
    rows, t_start = [], time.time()
    for i, p in enumerate(sel):
        v = p['view']
        Wobs = (p['M'] if a.on_model else p['W']).astype(float)
        if a.on_model and a.model_noise == 'white':
            Wobs = Wobs + np.random.default_rng(int(p['event_id']) * 2 + (v == 'y')).normal(size=Wobs.shape) * p['noise'].astype(float)[:, None]
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
