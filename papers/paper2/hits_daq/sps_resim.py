#!/usr/bin/env python3
"""sps_resim.py <sps window pickle(s)> --bundle B --out OUT.parquet [--arms SPEC] [--nmax N] [--jobs N]
-- Arm S (resim.py) on the SPS known-angle rows: why the beam rows read steeper than the bench law.

Chamber E on the 25.4 deg stage in Ar/CO2/iso reads a hits-ladder / waveform-fit slope ratio of 1.07-1.13 on the inclined
(Y) view, against 1.03-1.05 from the bench law 1 + C/N^2 with the bench C_Y (DAG_SCORES_2026-11-13.md).  The gas is slow
there (v = 7.5-10 um/ns against 33-36 on the bench), so the charge column lasts 2.8-3.7 us instead of 0.8 us, and the RC
spreading along the resistive strips (sigma^2 += 2 D_RC s, s the time since landing) has that much longer to act.  This
script tests that reading on the planes themselves: every inclined plane is regenerated through the pass model with a
modified kernel, in two modes,

  res    W' = M(p0, w, t0, q; hyper') + (W - M(hyper))    the plane's own residual added back; 'base' is then the data
  noise  W' = M(p0, w, t0, q; hyper') + white noise       the pure model: does the kernel alone produce the ratio?
  zs     the 'noise' window with the data's suppressed samples (W == 0 on these ZS rows) set to 0

and the DREAM leading-edge ladder (emulate.dream, tiers.ladder_u, as the V0 tier) is fitted on W'.  The score is the
median u / w over the planes (w the generating slope, so no reference and no drift velocity), per row, arm and mode.
The windows are padded by PAD strips as in resim.py.  Arms: resim.parse_variants SPEC (default: the kernel taken apart).
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from resim import PAD, padded, parse_variants  # noqa: E402
from tiers import ladder_u  # noqa: E402

# 'gas' = the Magboltz diffusion of the beam's wet Ar/CO2/iso (D_T 180-190 um/sqrt(cm) at 9-11 um/ns: Dp = 0.006), in
# place of the bench bundle's 0.020 the pass rows were fitted with (sps_dp_scan.py)
ARMS = ('base:,rc0:rc_D_y=0,rc2:rc_D_y=x2,dp0:Dp=1e-6,gas:Dp=0.006,gasrc0:Dp=0.006;rc_D_y=0,'
        'rc0dp0:rc_D_y=0;Dp=1e-6,point:rc_D_y=0;Dp=1e-6;sigma_p0=0.07')
G = {}


def init(bundle, meta, K, spec, nsig):
    from wft import model as wm
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(bundle)
    cal.v_drift = meta['v_drift']
    cal.bow, cal.v_map, cal.w0 = {}, {}, {}          # as sps_reco._init: bench mounting terms never apply at H4
    wm.use_calibration(cal)
    wm.set_nsamp(meta['nsamp'])
    wm.set_depth_bins(K)
    E.SNS = float(meta.get('sns', 60.0))
    G.update(wm=wm, V=parse_variants(spec, dict(meta.get('hyper') or cal.hyper)), nsig=nsig, key=meta['key'])


def one(p):
    wm, v = G['wm'], p['view']
    rng = np.random.default_rng(p['event_id'] * 2 + (v == 'y'))
    g = p['gain'].astype(float)
    R = p['W'].astype(float) - p['M'].astype(float)          # sps_extract windows are in ADC (model times the gain)
    pos2, nz2, g2 = padded(p, rng)
    ns = R.shape[1]
    R2 = np.vstack([rng.normal(0, 1, (PAD, ns)) * nz2[0], R, rng.normal(0, 1, (PAD, ns)) * nz2[-1]])
    N2 = rng.normal(0, 1, (len(pos2), ns)) * nz2[:, None]
    q = p['q'].astype(float)
    Z2 = np.vstack([np.ones((PAD, ns), bool), p['W'] == 0, np.ones((PAD, ns), bool)])
    rows = []
    for name, hy in G['V'].items():
        M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], q, hy) * g2[:, None]
        for mode, W2 in (('res', M2 + R2), ('noise', M2 + N2), ('zs', np.where(Z2, 0.0, M2 + N2))):
            h = E.dream(W2, nz2, nsig=G['nsig'])
            u, n = ladder_u(pos2, h['t'], h['a'])
            ur, nr = ladder_u(pos2, h['t'], h['a'], robust=True)
            rows.append(dict(row=G['key'], event_id=p['event_id'], view=v, arm=name, mode=mode, w_true=p['w'], u=u, ur=ur,
                             n_hits=n, n_rob=nr, t_span=float(np.nanmax(h['t']) - np.nanmin(h['t'])) if n > 1 else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default=ARMS)
    ap.add_argument('--views', default='y')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--nsig', type=float, default=5.0)
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    out = []
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = D['meta']
        sel = [p for p in D['planes'] if p['view'] in a.views and p.get('q') is not None]
        if a.nmax:
            sel = sel[:a.nmax]
        K = len(sel[0]['q'])
        with mp.Pool(a.jobs, initializer=init, initargs=(a.bundle, meta, K, a.arms, a.nsig)) as pool:
            for rows in pool.imap_unordered(one, sel, chunksize=20):
                out += rows
        T = pd.DataFrame([r for r in out if r['row'] == meta['key']])
        print(f"== {meta['key']}  {len(sel)} planes, K {K}, hyper {meta.get('hyper')}")
        for (mode, arm), g in T.groupby(['mode', 'arm'], sort=False):
            k = np.isfinite(g.u) & (np.abs(g.w_true) > 0)
            print(f'   {mode:5s} {arm:8s} ratio {np.median(g.u[k] / g.w_true[k]):.3f}  robust {np.nanmedian(g.ur[k] / g.w_true[k]):.3f}'
                  f'  n_hits {np.mean(g.n_hits[k]):.1f}  dropped by robust {np.mean(g.n_hits[k] - g.n_rob[k]):.2f}  cov {k.mean():.2f}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(out).to_parquet(a.out)
    print(f'[sps_resim] wrote {a.out}')


if __name__ == '__main__':
    main()
