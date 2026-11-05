#!/usr/bin/env python3
"""resim.py <window pickle(s)> --variants SPEC --out OUT.parquet [--families ...] [--part i/n] [--refit] -- Arm S.

Re-simulation on real events.  Every plane keeps what the data fixed -- its fitted track (p0, w, t0), its depth
profile q (the event's own ionisation and attachment), its strip layout and its noise -- and only the detector physics
is changed: the window is regenerated through the pass model (wft.model.model_waveforms) with a modified kernel and the
event's own fit residual R = W - M (noise + whatever the model misses) is added back:

    W' = M(p0, w, t0, q; hyper') + R

The truth is then exact: the generating w.  Per variant the hit families (emulate.py) give ladders u, scored as u / w
(the ladder compression) and as angle residuals against the true tan; with --refit the waveform fit is redone on W' with
the generating kernel (fit_plane, hyper'), i.e. W calibrated to the detector it reads.  The window is padded by PAD
strips on each side (noise-only rows, Gaussian at the window's median noise) so that wide spreading is not cut off.

Variant SPEC: comma-separated 'name:key=value;key=value', values may be 'x<factor>' (times the bundle value), e.g.
  'base:,rc0:rc_D_y=0,rc2:rc_D_y=x2,sheet:rc_D_x=@rc_D_y,metal:rc_D_y=0;sigma_p0=0.07'
('@k' = the bundle value of k).  rc_D_<v> = 0 is replaced by 1e-12 (keeps the RC kernel path).
"""
import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from tiers import ladder_u  # noqa: E402

PAD = 4


def parse_variants(spec, hyper):
    out = {}
    for item in spec.split(','):
        name, _, kv = item.partition(':')
        h = dict(hyper)
        for pair in filter(None, kv.split(';')):
            k, _, val = pair.partition('=')
            if val.startswith('x'):
                h[k] = float(hyper.get(k, 0.0)) * float(val[1:])
            elif val.startswith('@'):
                h[k] = float(hyper.get(val[1:], 0.0))
            else:
                h[k] = float(val)
            if k == 'sigma_p0':
                for v in 'xy':
                    h[f'sigma_p0_{v}'] = h[k]
        for v in 'xy':
            if f'rc_D_{v}' in h and float(h[f'rc_D_{v}']) <= 0:
                h[f'rc_D_{v}'] = 1e-12
        out[name] = h
    return out


def padded(p, rng):
    pos = p['pos'].astype(float)
    n = len(pos)
    pitch = np.median(np.diff(pos)) if n > 1 else 0.78
    pos2 = np.r_[pos[0] - pitch * np.arange(PAD, 0, -1), pos, pos[-1] + pitch * np.arange(1, PAD + 1)]
    nz = p['noise'].astype(float)
    nz2 = np.r_[np.full(PAD, np.median(nz)), nz, np.full(PAD, np.median(nz))]
    g = p['gain'].astype(float)
    g2 = np.r_[np.full(PAD, np.median(g)), g, np.full(PAD, np.median(g))]
    return pos2, nz2, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle')
    ap.add_argument('--variants', required=True)
    ap.add_argument('--families', default='dream,peak,thr,fd,ideal')
    ap.add_argument('--out', required=True)
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--nsig', type=float, default=5.0)
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--views', default='xy')
    a = ap.parse_args()
    ip, npart = (int(x) for x in a.part.split('/'))
    from wft import model as wm, reco as wr
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    gain_corrected = 'raw' not in planes[0]                  # boot_extract windows: W, M gain corrected
    cal = CalibrationBundle.load(a.bundle or meta['bundle'])
    wm.use_calibration(cal)
    K = len(planes[0]['q'])
    wm.set_depth_bins(K)
    V = parse_variants(a.variants, dict(cal.hyper))
    fams = a.families.split(',')
    ideal = {v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in 'xy'}
    sel = [p for i, p in enumerate(planes) if i % npart == ip and p['view'] in a.views]
    if a.nmax:
        sel = sel[:a.nmax]
    vv = meta['v_drift']
    rows, t0c = [], time.time()
    for i, p in enumerate(sel):
        v = p['view']
        rng = np.random.default_rng(p['event_id'] * 2 + (v == 'y'))
        g = p['gain'].astype(float)
        Wadc = p['W'].astype(float) * (g[:, None] if gain_corrected else 1.0)
        Madc = p['M'].astype(float) * (g[:, None] if gain_corrected else 1.0)
        R = Wadc - Madc
        pos2, nz2, g2 = padded(p, rng)
        R2 = np.vstack([rng.normal(0, 1, (PAD, R.shape[1])) * nz2[0], R,
                        rng.normal(0, 1, (PAD, R.shape[1])) * nz2[-1]])
        q = p['q'].astype(float)
        kw, w0 = meta['kw'][v], meta['w0'][v]
        base = dict(event_id=p['event_id'], view=v, w_true=p['w'], tan_true=(p['w'] * 1e3 - w0) / (kw * vv))
        for name, hy in V.items():
            M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], q, hy) * g2[:, None]
            W2 = M2 + R2
            r = dict(base, variant=name)
            for f in fams:
                if f == 'dream':
                    h = E.dream(W2, nz2, nsig=a.nsig)
                elif f == 'fd':
                    h = E.fd(W2, nz2, nsig=a.nsig)
                elif f == 'ideal':
                    h = ideal[v](W2, nz2, nsig=a.nsig)
                elif f == 'binary':
                    h = E.binary(W2, nz2, nsig=a.nsig, rng=rng)
                else:
                    h = getattr(E, f)(W2, nz2, nsig=a.nsig)
                r[f'u_{f}'], r[f'n_{f}'] = ladder_u(pos2, h['t'], h['a'], weighted=f != 'binary')
                r[f'ur_{f}'], _ = ladder_u(pos2, h['t'], h['a'], weighted=f != 'binary', robust=True)
            if a.refit:
                P = dict(W=W2.astype(np.float32), noise=nz2.astype(np.float32), pos=pos2.astype(np.float32),
                         ch=np.r_[np.full(PAD, -1), p['ch'], np.full(PAD, -1)].astype(int))
                # padded rows have no channel: give them the nearest real channel's gain via a private GAIN copy
                P['ch'] = np.r_[np.full(PAD, p['ch'][0]), p['ch'], np.full(PAD, p['ch'][-1])].astype(int)
                try:
                    f_ = wr.fit_plane(P, v, cal, hyper=hy)
                    r['w_fit'] = f_.w if f_ is not None else np.nan
                except Exception:
                    r['w_fit'] = np.nan
            rows.append(r)
        if i % 100 == 0:
            print(f'[resim] {i}/{len(sel)} planes, {time.time() - t0c:.0f} s', flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    T.to_parquet(a.out)
    print(f'[resim] wrote {a.out}: {len(T)} rows ({len(sel)} planes x {len(V)} variants) in {time.time() - t0c:.0f} s')


if __name__ == '__main__':
    main()
