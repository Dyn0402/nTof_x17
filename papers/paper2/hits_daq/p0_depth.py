#!/usr/bin/env python3
"""p0_depth.py <boot pickle> <bundle> --out X.parquet [--part i/n] [--min-tan 0.15] -- where in depth is the ladder compressed?

LITERATURE_MIXING.md §5 (reviewer note) claims our single-time compression comes largely from the gap ends, where the
same-time mixing is one-sided (no charge beyond the mesh or the cathode), so a long drift far from both ends would
not have it.  This tests it on the P0 inputs (the residual-bootstrap planes of p0_xbias.py).

Per plane and per hit strip i (V0-like hit, 5 sigma peak, 30 % leading edge, p0_xbias.hits_v0):
  u_i  = (pos_i - p0) / w            the true drift time of the track at the strip's centre [ns since t0]
  f_i  = (u_i - u_lo) / (u_hi - u_lo) fractional depth in the plane's charge column (0 = mesh, 1 = cathode), the column
                                     ends taken from the fitted q (first/last bin above 10 % of its max)
  dt_i = t_hit_i - u_i - c           c = the plane's median (t_hit - u) over 0.3 <= f <= 0.7 (t0 + template delay)
on the data W and on the model regenerated at the plane's own (p0, w, t0, q) for the arms of p0_xbias.variants
('model', 'metallic', 'point').  If the compression is a gap-end effect, dt(f) is flat in the middle and bends at the
ends (late at the mesh end, early at the cathode end); if it is bulk, dt(f) has a constant slope
d(dt)/du = 1/ratio - 1 all through.  The table is long (one row per hit); the scoring is in --summary.
Only planes with |tan| >= --min-tan, so that a strip maps onto a narrow depth range.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
sys.path.insert(0, HERE)

from p0_xbias import hits_v0, variants  # noqa: E402

ARMS = ('model', 'metallic', 'point')
FB = np.linspace(-0.1, 1.1, 13)


def run(a):
    import pandas as pd
    from wft import model as wm
    from wft.calib import CalibrationBundle
    D = pickle.load(open(a.pickle, 'rb'))
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    hyper = dict(cal.hyper)
    v = D['meta']['v_drift'] / 1000.0
    ip, npart = (int(x) for x in a.part.split('/'))
    P = [p for j, p in enumerate(D['planes']) if j % npart == ip]
    if a.nmax:
        P = P[:a.nmax]
    rows = []
    for view in ('x', 'y'):
        V = variants(hyper, view)
        for p in P:
            if p['view'] != view or abs(p['w'] / v) < a.min_tan:
                continue
            pos, nz = p['pos'].astype(float), p['noise'].astype(float)
            u = (pos - p['p0']) / p['w']
            # the charge column's ends from the plane's own depth profile: q has margin bins (K DT = 1440 ns
            # against a 784 ns gap), so u = 0 is not the mesh.  Ends = first/last bin above 10 % of the max.
            qq = p['q'].astype(float)
            nzq = np.flatnonzero(qq > 0.1 * qq.max()) if qq.max() > 0 else []
            if len(nzq) < 3:
                continue
            ulo, uhi = nzq[0] * wm.DT, (nzq[-1] + 1) * wm.DT
            T = uhi - ulo
            arms = {'data': p['W'].astype(float)}
            for k in ARMS:
                arms[k] = wm.model_waveforms(view, pos, p['p0'], p['w'], p['t0'], p['q'].astype(float), V[k])
            for k, Wk in arms.items():
                h = hits_v0(Wk, nz)
                r = h[:, 0] - u
                f = (u - ulo) / T
                ok = np.isfinite(r) & (f > -0.2) & (f < 1.2)
                mid = ok & (f >= 0.3) & (f <= 0.7)
                if mid.sum() < 2:
                    continue
                r = r - np.median(r[mid])
                for i in np.flatnonzero(ok):
                    rows.append((view, k, int(p.get('event_id', -1)), float(p['w'] / v), float(f[i]), float(r[i]),
                                 float(h[i, 2] / nz[i]), float(u[i]), ulo, T))
    df = pd.DataFrame(rows, columns=['view', 'arm', 'event_id', 'tan', 'f', 'dt', 'snr', 'u', 'ulo', 'T'])
    df.to_parquet(a.out)
    print(f'[p0_depth] wrote {a.out}: {len(df)} hits, {df.groupby("view").event_id.nunique().to_dict()} planes')


def summary(a):
    import pandas as pd
    import glob
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(a.summary))], ignore_index=True)
    out = {}
    for (view, arm), g in df.groupby(['view', 'arm']):
        b = np.digitize(g.f, FB) - 1
        prof = []
        for j in range(len(FB) - 1):
            k = b == j
            prof.append(dict(lo=float(FB[j]), hi=float(FB[j + 1]), n=int(k.sum()),
                             dt_med=float(np.median(g.dt[k])) if k.sum() > 30 else None))
        # interior slope (0.2-0.8) vs end-to-end slope (0-1), ns of dt per unit f -> as a ladder ratio u_h/w = 1/(1 + s/T)
        def slope(lo, hi):
            xs = [0.5 * (r['lo'] + r['hi']) for r in prof if r['dt_med'] is not None and r['lo'] >= lo - 1e-9
                  and r['hi'] <= hi + 1e-9]
            ys = [r['dt_med'] for r in prof if r['dt_med'] is not None and r['lo'] >= lo - 1e-9 and r['hi'] <= hi + 1e-9]
            return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else None
        out[f'{view}/{arm}'] = dict(profile=prof, slope_inner=slope(0.2, 0.8), slope_full=slope(0.0, 1.0),
                                    n_hits=len(g), n_planes=int(g.event_id.nunique()))
        o = out[f'{view}/{arm}']
        meds = ' '.join(f'{r["dt_med"]:+6.1f}' if r['dt_med'] is not None else '     -' for r in prof)
        fs = lambda x: '      -' if x is None else f'{x:+7.1f}'  # noqa: E731
        if arm == 'data':
            Tm = g.groupby('event_id').T.first()
            print(f'{view} column length median {Tm.median():.0f} ns (gap/v = {28.08 / 0.0358:.0f} for det3)')
        print(f'{view} {arm:9s} planes {o["n_planes"]:5d}  slope inner {fs(o["slope_inner"])}  full '
              f'{fs(o["slope_full"])} ns/gap   dt(f) {meds}')
    print('f bins: ' + ' '.join(f'{0.5 * (l + h):+.2f}' for l, h in zip(FB[:-1], FB[1:])))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickle', nargs='?')
    ap.add_argument('bundle', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--min-tan', type=float, default=0.15)
    ap.add_argument('--summary', help='glob of p0_depth tables: print and (--json) write the depth profiles')
    ap.add_argument('--json')
    a = ap.parse_args()
    summary(a) if a.summary else run(a)


if __name__ == '__main__':
    main()
