#!/usr/bin/env python3
"""chambers_C.py --out results/chambers_C.json -- the strip-end coefficient C on every bench chamber with a bootstrap extraction.

For each chamber's residual-bootstrap windows (rerun/paper1_floor/boot_extract.py; clean planes of the golden row) the V0
leading-edge hits are formed on the data window W, on the stored pass model M and on the model regenerated with one
ingredient off (p0_xbias.variants: metallic = floor footprint + no RC, point = no diffusion either, Dp0 = no diffusion, rc0 =
no RC on Y).  Per plane the amplitude-weighted ladder gives u = dx/dt; the truth is the fit's own w (the kw angle scale, as
for the re-simulation arms of universal.py), tan from the bundle's (w0, kw, v).  Per chamber, view and arm the median ratio
u/w in |tan| bins is fitted with Delta tan = a / tan + b tan; C = a (L/p)^2 with L = 28.08 mm for every chamber (the design
gap; the chambers' charge-visible columns differ by a few mm).  Errors by bootstrap over planes.  The toy law of the review,
C = 1.1 + 11 (sigma_rms/p)^2 with sigma_rms^2 = sigma0^2 + Dp^2 T/2 (T = L/v), is evaluated with each chamber's Dp.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
from p0_xbias import hits_v0, ladder_u, variants  # noqa: E402

G = {}
L_GAP, PITCH = 28.08, 0.78
EDGES = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17, 0.20, 0.24, 0.28, 0.33, 0.40, 0.50])
ROWS = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det7': 'g_det7_long'}
NAME = {'det2': 'B', 'det3': 'A', 'det4': 'E', 'det6': 'C', 'det7': 'D'}
ARMS = ('model', 'metallic', 'point', 'Dp0', 'rc0')


def one(p):
    wm = G['wm']
    v = p['view']
    pos, nz = p['pos'].astype(float), p['noise'].astype(float)
    W, M = p['W'].astype(float), p['M'].astype(float)
    out = dict(event_id=int(p['event_id']), view=v, w=float(p['w']))
    h = hits_v0(W, nz)
    out['u_data'] = ladder_u(pos, h[:, 0], h[:, 2])
    h = hits_v0(M, nz)
    out['u_stored'] = ladder_u(pos, h[:, 0], h[:, 2])
    V = variants(G['hyper'], v)
    q = p['q'].astype(float)
    for k in ARMS:
        if k not in V:
            continue
        Mk = wm.model_waveforms(v, pos, p['p0'], p['w'], p['t0'], q, V[k])
        h = hits_v0(Mk, nz)
        out[f'u_{k}'] = ladder_u(pos, h[:, 0], h[:, 2])
    return out


COARSE = np.array([0.03, 0.08, 0.14, 0.22, 0.35, 0.6])


def binned_fit(tan, ratio, rng=None, nboot=200, edges=EDGES, nmin=25):
    def fit(tan, ratio):
        t, y, e = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = np.isfinite(ratio) & (tan >= lo) & (tan < hi)
            if k.sum() < nmin:
                continue
            r = ratio[k]
            med = float(np.median(r))
            err = max(float(1.2533 * 1.4826 * np.median(np.abs(r - med)) / np.sqrt(k.sum())), 1e-4)
            tc = float(np.median(tan[k]))
            t.append(tc), y.append((med - 1) * tc), e.append(err * tc)
        t, y, e = np.array(t), np.array(y), np.array(e)
        if len(t) < 4:
            return None, None, t, y
        A = np.vstack([1 / t, t]).T / e[:, None]
        c, *_ = np.linalg.lstsq(A, y / e, rcond=None)
        return float(c[0]), float(c[1]), t, y
    a, b, t, y = fit(tan, ratio)
    if a is None:
        if edges is not COARSE:          # few planes (chamber D): coarser bins, smaller minimum
            return binned_fit(tan, ratio, rng, nboot, COARSE, 12)
        return None
    res = dict(a=a, b=b, C=a * (L_GAP / PITCH) ** 2, tan=t.tolist(), ratio=(1 + y / t).tolist(), n=int(np.isfinite(ratio).sum()),
               coarse=bool(edges is COARSE))
    if rng is not None and nboot:
        aa, bb = [], []
        n = len(tan)
        for _ in range(nboot):
            k = rng.integers(0, n, n)
            a2, b2, _, _ = fit(tan[k], ratio[k])
            if a2 is not None:
                aa.append(a2), bb.append(b2)
        res.update(a_err=float(np.std(aa)), b_err=float(np.std(bb)), C_err=float(np.std(aa) * (L_GAP / PITCH) ** 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'chambers_C.json'))
    ap.add_argument('--boot', default='/media/ucla/work/paper2_hits/boot')
    ap.add_argument('--bundles', default='/media/ucla/work/paper2_hits/bundles')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--nmax', type=int, default=0)
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    out = dict(L=L_GAP, pitch=PITCH, edges=EDGES.tolist(), chambers={})
    for det, key in ROWS.items():
        D = pickle.load(open(os.path.join(a.boot, key + '.pkl'), 'rb'))
        meta, planes = D['meta'], D['planes']
        if a.nmax:
            planes = planes[:a.nmax]
        cal = CalibrationBundle.load(os.path.join(a.bundles, det, 'calib_bundle'))
        wm.use_calibration(cal)
        wm.set_depth_bins(len(planes[0]['q']))
        G.update(wm=wm, hyper=dict(cal.hyper))
        with mp.Pool(a.jobs) as P:
            rows = P.map(one, planes, chunksize=16)
        v = meta['v_drift']
        ch = dict(row=key, name=NAME[det], v=v, kw=meta['kw'], w0=meta['w0'], hyper={k: cal.hyper[k] for k in ('sigma_p0', 'Dp', 'rc_D_y')},
                  n_planes=len(planes), views={})
        T = L_GAP / (v / 1000.0)
        s_rms = np.sqrt(cal.hyper['sigma_p0'] ** 2 + cal.hyper['Dp'] ** 2 * T / 2)
        ch['toy_law_C_X'] = 1.1 + 11 * (s_rms / PITCH) ** 2
        ch['sigma_rms_over_p'] = s_rms / PITCH
        rng = np.random.default_rng(7)
        for view in 'xy':
            R = [r for r in rows if r['view'] == view]
            w = np.array([r['w'] for r in R])
            tan = np.abs(w * 1e3 - meta['w0'][view]) / (meta['kw'][view] * v)
            V = dict(n=len(R), arms={})
            for arm in ('data', 'stored') + ARMS:
                col = f'u_{arm}'
                if not all(col in r for r in R):
                    continue
                u = np.array([r[col] for r in R])
                f = binned_fit(tan, u / w, rng if arm == 'data' else None)
                if f:
                    V['arms'][arm] = f
            ch['views'][view] = V
            line = '  '.join(f"{arm}: C={f['C']:.2f}" + (f"+-{f['C_err']:.2f}" if 'C_err' in f else '') + f" b={f['b']:+.3f}" for arm, f in V['arms'].items())
            print(f"{det} ({NAME[det]}) {view}: n {len(R)}  {line}", flush=True)
        print(f"   toy law C_X (sigma_rms/p = {ch['sigma_rms_over_p']:.3f}): {ch['toy_law_C_X']:.2f};  Dp {cal.hyper['Dp']}, rc_D_y {cal.hyper['rc_D_y']:.2e}, v {v}")
        out['chambers'][det] = ch
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
