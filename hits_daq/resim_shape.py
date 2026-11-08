#!/usr/bin/env python3
"""resim_shape.py <window pickles> --out T.parquet [--tp 25,50,100,200] [--part i/n] [--nmax N] -- Arm S, front-end shaping.

The resim engine with the electronics response replaced: the pass kernel (footprint, diffusion, RC) is kept, the template
becomes a CR-RC^2 pulse of peaking time tp [ns] convolved with the avalanche current (10 % prompt electrons, 90 % ions
uniform over ION_NS).  A 60 ns sampled window cannot hold a 25 ns pulse, so every window is built on a 10 ns grid by
interleaving model evaluations at shifted t0 (M(t_s + j dt; t0) = M(t_s; t0 - j dt)) and the depth basis is refined to 20 ns
(depth_subdiv 3).  The event's fit residual cannot be reused at another shaping: the noise is white noise through the same
shaper, scaled to each strip's measured rms, i.e. the signal-to-noise of DREAM is held at every tp.  (The real ENC changes
with tp; not modelled: stated limitation.)  Variants: <det>_<tp> with det = base | metal, tp = DREAM (the measured template,
the anchor against the 60 ns resim) or the peaking time.  Families: peak and thr (time-at-peak and time-at-threshold, VMM3a's modes).
Scored by resim_score.py.
"""
import argparse
import os
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from resim import padded, parse_variants  # noqa: E402
from tiers import ladder_u  # noqa: E402

DT_FINE = 10.0
NI = int(60 / DT_FINE)
ION_NS = 100.0
DETS = 'base:,metal:rc_D_y=0;sigma_p0=0.07'


def crrc2(t, tp):
    tau = tp / 2.0
    x = np.clip(t, 0, None) / tau
    return x ** 2 * np.exp(-x)


def pulse(tp, dt=1.0, tmax=8000.0):
    """CR-RC^2(tp) convolved with the avalanche current; (t, y) with peak 1"""
    t = np.arange(0.0, tmax, dt)
    cur = np.zeros(int(ION_NS / dt) + 1)
    cur[1:] = 0.9 / (len(cur) - 1)
    cur[0] = 0.1
    y = fftconvolve(crrc2(t, tp), cur)[:len(t)]
    return t, y / y.max()


def template_from(tp, grid, ref):
    """the shaped pulse on the bundle's template grid, starting where the measured template starts, with its peak height"""
    t, y = pulse(tp)
    i_on = int(np.argmax(ref > 0.02 * ref.max()))
    return np.interp(grid - grid[i_on], t, y, left=0.0, right=0.0) * float(ref.max())


def noise_kernel(tmpl, grid, dt=2.0):
    t = np.arange(grid[0], grid[-1], dt)
    h = np.interp(t, grid, tmpl)
    return h[h != 0] if (h != 0).any() else h


def shaped_noise(rng, n, ns_fine, h, sig):
    """n strips of white noise through h, unit rms x sig, on the 10 ns grid (generated at 2 ns)"""
    d = int(DT_FINE / 2.0)
    N = ns_fine * d + len(h)
    x = fftconvolve(rng.normal(size=(n, N)), h[None, :], axes=1, mode='same')[:, len(h) // 2:][:, :ns_fine * d]
    x /= x.std(axis=1, keepdims=True)
    return (x * sig[:, None])[:, ::d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle')
    ap.add_argument('--tp', default='25,50,100,200')
    ap.add_argument('--out', required=True)
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--every', type=int, default=3, help='use every N-th plane')
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    ip, npart = (int(x) for x in a.part.split('/'))
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    cal = CalibrationBundle.load(E.bundle_path(meta, a.bundle))
    cal.depth_subdiv = 3
    wm.use_calibration(cal)
    K0 = len(planes[0]['q'])
    wm.set_depth_bins(K0)
    orig = {v: wm.TMPL[v].copy() for v in 'xy'}
    tmpl = {'DREAM': orig['x']}
    for tp in a.tp.split(','):
        tmpl[f'tp{tp}'] = template_from(float(tp), wm.TGRID, orig['x'])
    DV = parse_variants(DETS, dict(cal.hyper))
    h_n = {n: noise_kernel(T, wm.TGRID) for n, T in tmpl.items()}
    planes = planes[::a.every]
    sel = [p for i, p in enumerate(planes) if i % npart == ip]
    if a.nmax:
        sel = sel[:a.nmax]
    E.SNS = DT_FINE
    vv = meta['v_drift']
    rows, t_s = [], time.time()
    for name, T in tmpl.items():
        wm.TMPL = {'x': T, 'y': T if name != 'DREAM' else orig['y']}
        wm._rc_cache.clear()
        for i, p in enumerate(sel):
            v = p['view']
            rng = np.random.default_rng(p['event_id'] * 2 + (v == 'y') + zlib.crc32(name.encode()) % 1000)
            pos2, nz2, g2 = padded(p, rng)
            q3 = np.repeat(p['q'].astype(float), 3) / 3.0
            kw, w0 = meta['kw'][v], meta['w0'][v]
            base = dict(event_id=p['event_id'], view=v, w_true=p['w'], tan_true=(p['w'] * 1e3 - w0) / (kw * vv))
            for det, hy in DV.items():
                Ms = [wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'] - j * DT_FINE, q3, hy) for j in range(NI)]
                M = np.stack(Ms, axis=2).reshape(len(pos2), -1) * g2[:, None]
                W2 = M + shaped_noise(rng, len(pos2), M.shape[1], h_n[name], nz2)
                r = dict(base, variant=f'{det}_{name}')
                for f in ('peak', 'thr'):
                    h = getattr(E, f)(W2, nz2, nsig=5.0)
                    r[f'u_{f}'], r[f'n_{f}'] = ladder_u(pos2, h['t'], h['a'])
                    r[f'ur_{f}'], _ = ladder_u(pos2, h['t'], h['a'], robust=True)
                rows.append(r)
            if len(wm._rc_cache) > 400:
                wm._rc_cache.clear()
            if i % 50 == 0:
                print(f'[shape] {name} {i}/{len(sel)} {time.time() - t_s:.0f} s', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out)
    print(f'[shape] wrote {a.out}: {len(rows)} rows in {time.time() - t_s:.0f} s')


if __name__ == '__main__':
    main()
