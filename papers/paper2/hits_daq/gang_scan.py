#!/usr/bin/env python3
"""gang_scan.py <window pickles> --ref REF --out T.parquet [--gang 1,2,3,4,6,8] -- does the single-time compression depend
on the strip pitch?  (Dylan 2026-11-08: "can we also blame the fine strips? with larger strips or pads, the same problem?")

Adjacent strips are summed into readout elements of k pitches (fixed to the board: element = floor(pos / (k pitch))), on
the real windows ('data', against M3) and on re-simulated windows ('base', 'metal', 'point', against the exact generating
track).  Noise adds in quadrature; the extra capacitance of a wider strip is not modelled.  Each element gives one hit
(emulate.py), the ladder is the same amplitude-weighted line as everywhere (tiers.ladder_u, >= 3 elements).  The waveform
fit is not redone at wider pitch (its kernel is per strip).

--score '<glob>' scores tables: ratio = median u / (v tan_ref) (data) or u / w_true (resim), for |tan| > 0.1 and per
|tan| bin (the bin dependence is the lever-arm dependence: column width on the strips = gap x tan), with coverage.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
from resim import padded, parse_variants  # noqa: E402
from tiers import ladder_u, load_ref  # noqa: E402

ARMS = 'base:,metal:rc_D_y=0;sigma_p0=0.07,point:rc_D_y=0;sigma_p0=0.07;Dp=0'
BINS = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5))


def gang(W, noise, pos, k, pitch):
    """(summed window, quadrature noise, element centres)"""
    if k == 1:
        return W, noise, pos
    e = np.floor(pos / (k * pitch)).astype(int)
    ue = np.unique(e)
    Wg = np.array([W[e == x].sum(0) for x in ue])
    ng = np.array([np.sqrt(np.sum(noise[e == x] ** 2)) for x in ue])
    return Wg, ng, (ue + 0.5) * k * pitch


def score(pattern, jout):
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    out = {}
    for (view, arm, k), g in T.groupby(['view', 'arm', 'k']):
        data = arm == 'data'
        den = (g.v_mm_ns * g.tan_ref if data else g.w_true).to_numpy(float)
        tt = (g.tan_ref if data else g.tan_true).to_numpy(float)
        o = {}
        for f in ('dream', 'ideal'):
            u = g[f'u_{f}'].to_numpy(float)
            m = np.isfinite(u) & np.isfinite(den) & (np.abs(tt) > 0.1)
            r = dict(ratio=float(np.median(u[m] / den[m])) if m.sum() > 30 else None,
                     coverage=float(np.isfinite(u[np.abs(tt) > 0.1]).mean()), n=int(m.sum()))
            for lo, hi in BINS:
                b = np.isfinite(u) & np.isfinite(den) & (np.abs(tt) >= lo) & (np.abs(tt) < hi)
                r[f'{lo}-{hi}'] = float(np.median(u[b] / den[b])) if b.sum() > 30 else None
            o[f] = r
        out.setdefault(view, {}).setdefault(arm, {})[int(k)] = o
    for view, A in out.items():
        print(f'[{view}]  ratio (V0, |tan|>0.1) / coverage; then per |tan| bin at each k')
        for arm, K in A.items():
            print(f'  {arm:6s} ' + '  '.join(f'k{k}: {K[k]["dream"]["ratio"] or np.nan:.2f}/{K[k]["dream"]["coverage"]:.2f}' for k in K))
            for lo, hi in BINS:
                print(f'     {lo}-{hi:<5} ' + '  '.join(f'k{k}: {K[k]["dream"].get(f"{lo}-{hi}") or np.nan:.2f}' for k in K))
    if jout:
        json.dump(out, open(jout, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='*')
    ap.add_argument('--ref')
    ap.add_argument('--bundle')
    ap.add_argument('--gang', default='1,2,3,4,6,8')
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--score')
    ap.add_argument('--json')
    a = ap.parse_args()
    if a.score:
        return score(a.score, a.json)
    from wft import model as wm
    from wft.calib import CalibrationBundle
    ip, npart = (int(x) for x in a.part.split('/'))
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    cal = CalibrationBundle.load(E.bundle_path(meta, a.bundle))
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    ref = load_ref(a.ref)
    V = parse_variants(ARMS, dict(cal.hyper))
    ks = [int(x) for x in a.gang.split(',')]
    ideal = {v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in 'xy'}
    sel = [p for i, p in enumerate(planes) if i % npart == ip and (p['event_id'], p['view']) in ref]
    if a.nmax:
        sel = sel[:a.nmax]
    vd = meta['v_drift'] / 1000.0
    rows = []

    def read(W, nz, pos, base, arm, pitch, view):
        for k in ks:
            Wg, ng, pg = gang(W, nz, pos, k, pitch)
            r = dict(base, arm=arm, k=k)
            for f, h in (('dream', E.dream(Wg, ng)), ('ideal', ideal[view](Wg, ng, nsig=5.0))):
                r[f'u_{f}'], r[f'n_{f}'] = ladder_u(pg, h['t'], h['a'])
            rows.append(r)

    for i, p in enumerate(sel):
        v = p['view']
        rng = np.random.default_rng(p['event_id'] * 2 + (v == 'y'))
        pos = p['pos'].astype(float)
        pitch = float(np.median(np.diff(pos)))
        tr = ref[(p['event_id'], v)]
        kw, w0 = meta['kw'][v], meta['w0'][v]
        base = dict(event_id=p['event_id'], view=v, tan_ref=tr, v_mm_ns=vd, w_true=p['w'],
                    tan_true=(p['w'] * 1e3 - w0) / (kw * meta['v_drift']))
        Wd, nz = p['W'].astype(float), p['noise'].astype(float)
        read(Wd, nz, pos, base, 'data', pitch, v)
        g = p['gain'].astype(float)
        R = Wd - p['M'].astype(float)
        pos2, nz2, g2 = padded(p, rng)
        R2 = np.vstack([rng.normal(0, 1, (4, R.shape[1])) * nz2[0], R, rng.normal(0, 1, (4, R.shape[1])) * nz2[-1]])
        for arm, hy in V.items():
            M2 = wm.model_waveforms(v, pos2, p['p0'], p['w'], p['t0'], p['q'].astype(float), hy) * g2[:, None]
            read(M2 + R2, nz2, pos2, base, arm, pitch, v)
        if i % 200 == 0:
            print(f'[gang] {i}/{len(sel)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out)
    print(f'[gang] wrote {a.out}: {len(rows)} rows')


if __name__ == '__main__':
    main()
