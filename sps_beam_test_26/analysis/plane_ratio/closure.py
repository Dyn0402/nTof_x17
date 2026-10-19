#!/usr/bin/env python3
"""closure.py -- posterior-predictive closure of the sharing kernel, per view.

For each near-vertical event in a calibration cache, fit the forward model at
the REFERENCE track (p0, w from M3; t0 by grid search, charge profile by NNLS,
exactly as wft.calibrate._event_chi2 does), rebuild the model waveforms, and
run the SAME neighbour estimator on data and model: leading strip as centre,
peak-aligned, normalised to the leading peak, 20 %-trimmed mean with absent
strips as zero.  The observable is the per-offset area and centroid delay
(d = +-1, +-2) relative to the centre.

A kernel parameterisation that is right reproduces the data's neighbour
pattern on BOTH views.  The hyper ``c2/c1`` is not this observable (most of
the measured +-1 signal is carried by sigma_p0, which is prompt and shared by
the two views), which is why the closure is needed at all.

    closure.py --cache <calib_cache.pkl> --bundle <dir> [--set k=v ...]
               [--tan-max 0.05] [--label NAME] [--json out.json]

``--set`` overrides hypers (e.g. ``--set c2_over_c1_x=0.15``); values that are
not numbers are rejected.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from wft import model as wm                      # noqa: E402
from wft.calib import CalibrationBundle          # noqa: E402

PITCH = 0.78
SNS = 60.0
NREL = 12
OFFS = (1, -1, 2, -2)

_G = {}


def _init(bundle, hyper, v, p0_free=False, k_bins=None, t0_rng=(150.0, 900.0)):
    os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')   # diag arms
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal)
    wm.MODEL_FRAC = float(os.environ.get('WFT_MODEL_FRAC', 0.0))
    if k_bins:
        wm.set_depth_bins(k_bins)
    _G['t0_rng'] = t0_rng
    h = dict(cal.hyper)
    h.update(hyper)
    _G.update(hyper=h, v=cal.v_drift if v is None else v, p0_free=p0_free)


def _fit_plane(ev, plane):
    P = ev[plane]
    if np.asarray(P['W']).shape[1] != wm.NSAMP:
        wm.set_nsamp(np.asarray(P['W']).shape[1])
    h, v = _G['hyper'], _G['v']
    wline = ev[f'tan_{plane}'] * v * 1e-3
    p0 = ev[f'ref_mesh_{plane}']
    W, noise, pos, sat = wm.prep_plane(P, plane)
    grid = np.arange(_G['t0_rng'][0], _G['t0_rng'][1], 30.0)
    chis = [wm.chi2_plane(plane, W, noise, pos, sat, p0, wline, float(t), h)[0]
            for t in grid]
    j = int(np.argmin(chis))
    t0c = grid[j]
    if _G.get('p0_free'):
        # profile p0 (the reference's pointing error is ~0.3-0.45 mm on the
        # bench); the angle stays pinned to the reference
        best = (np.inf, p0)
        for dp in np.arange(-1.2, 1.21, 0.1):
            for t in (t0c - 30.0, t0c, t0c + 30.0):
                c = wm.chi2_plane(plane, W, noise, pos, sat, p0 + dp, wline,
                                  float(t), h)[0]
                if c < best[0]:
                    best = (c, p0 + dp)
        p0 = best[1]
    fine = np.arange(t0c - 30.0, t0c + 31.0, 5.0)
    res = [wm.chi2_plane(plane, W, noise, pos, sat, p0, wline, float(t), h)
           for t in fine]
    k = int(np.argmin([r[0] for r in res]))
    chi, q = res[k]
    if q is None or not np.isfinite(chi):
        return None
    M = wm.model_waveforms(plane, pos, p0, wline, float(fine[k]), q, h)
    return dict(W=W, M=M, pos=pos, chi2=float(chi), dof=int((~sat).sum()),
                dp0=float(p0 - ev[f'ref_mesh_{plane}']))


def _rows(W, pos):
    """The bench_kernel estimator on one (n_strip, n_samp) block."""
    W = W - W[:, :3].mean(axis=1)[:, None]
    pk = W.max(axis=1)
    i0 = int(np.argmax(pk))
    s0 = int(np.argmax(W[i0]))
    if not (NREL <= s0 < W.shape[1] - NREL // 2):
        return None
    sidx = np.round((pos - pos[i0]) / PITCH).astype(int)
    cols = s0 + np.arange(-NREL, NREL + 1)
    ok = (cols >= 0) & (cols < W.shape[1])
    out = {}
    for d in (0,) + OFFS:
        j = np.flatnonzero(sidx == d)
        r = np.zeros(2 * NREL + 1)          # absent strip = zero
        if len(j):
            r[ok] = W[j[0], cols[ok]] / pk[i0]
        out[d] = r
    return out, pk[i0]


def _one(payload):
    ev, planes = payload
    out = {}
    for plane in planes:
        if plane not in ev:
            continue
        f = _fit_plane(ev, plane)
        if f is None:
            continue
        rd = _rows(f['W'], f['pos'])
        rm = _rows(f['M'], f['pos'])
        if rd is None or rm is None:
            continue
        out[plane] = dict(data=rd[0], model=rm[0], q0=float(rd[1]),
                          chi2=f['chi2'], dof=f['dof'], dp0=f['dp0'])
    return ev['eid'], out


def trim20(A):
    A = np.sort(np.asarray(A, float), axis=0)
    k = int(0.2 * len(A))
    return A[k:len(A) - k].mean(axis=0) if len(A) > 2 * k else A.mean(axis=0)


def summarise(stacks):
    t = (np.arange(2 * NREL + 1) - NREL) * SNS
    s0 = trim20(stacks[0])
    p0 = np.clip(s0, 0, None)
    c0 = (t * p0).sum() / p0.sum()
    o = {}
    for d in OFFS:
        s = trim20(stacks[d])
        p = np.clip(s, 0, None)
        o[f'area_{d:+d}'] = float(s.sum() / s0.sum())
        o[f'dt_{d:+d}'] = float((t * p).sum() / max(p.sum(), 1e-12) - c0)
    o['r21'] = (o['area_+2'] + o['area_-2']) / (o['area_+1'] + o['area_-1'])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--v', type=float, default=None)
    ap.add_argument('--tan-max', type=float, default=0.05)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--label', default='')
    ap.add_argument('--json', default=None)
    ap.add_argument('--depth-bins', type=int, default=None,
                    help='charge-basis length K (beam gas: the column is ~2 us)')
    ap.add_argument('--t0-lo', type=float, default=150.0)
    ap.add_argument('--t0-hi', type=float, default=900.0)
    ap.add_argument('--model-frac', type=float, default=0.0)
    ap.add_argument('--p0-free', action='store_true',
                    help='profile p0 per event instead of pinning it to M3')
    ap.add_argument('--arm-json', default=None,
                    help='take the hypers from a recal.py output')
    a = ap.parse_args()
    os.environ['WFT_MODEL_FRAC'] = str(a.model_frac)
    hyper = {}
    if a.arm_json:
        hyper.update({k: float(q) for k, q in json.load(open(a.arm_json))['hyper'].items()})
    for s in a.set:
        k, v = s.split('=', 1)
        hyper[k] = float(v)

    with open(a.cache, 'rb') as f:
        events = pickle.load(f)
    work = []
    for eid in sorted(events):
        ev = events[eid]
        planes = [p for p in ('x', 'y')
                  if p in ev and abs(ev[f'tan_{p}']) <= a.tan_max]
        if planes:
            work.append((ev, planes))

    acc = {p: {'data': {d: [] for d in (0,) + OFFS},
               'model': {d: [] for d in (0,) + OFFS}, 'chi2': [], 'dof': [],
               'dp0': []}
           for p in 'xy'}
    with ProcessPoolExecutor(a.jobs, initializer=_init,
                             initargs=(a.bundle, hyper, a.v, a.p0_free, a.depth_bins,
                                       (a.t0_lo, a.t0_hi))) as pool:
        for eid, out in pool.map(_one, work, chunksize=4):
            for p, r in out.items():
                for side in ('data', 'model'):
                    for d in (0,) + OFFS:
                        acc[p][side][d].append(r[side][d])
                acc[p]['chi2'].append(r['chi2'])
                acc[p]['dof'].append(r['dof'])
                acc[p]['dp0'].append(r['dp0'])

    rng = np.random.default_rng(20261009)
    res = {'label': a.label, 'bundle': a.bundle, 'set': hyper,
           'tan_max': a.tan_max, 'cache': a.cache}
    for p in 'xy':
        n = len(acc[p]['chi2'])
        if n < 10:
            continue
        st = {s: {d: np.array(acc[p][s][d]) for d in acc[p][s]} for s in ('data', 'model')}
        sd, sm = summarise(st['data']), summarise(st['model'])
        # paired bootstrap of model - data on every observable
        bs = []
        for _ in range(a.nboot):
            i = rng.integers(0, n, n)
            bd = summarise({d: st['data'][d][i] for d in st['data']})
            bm = summarise({d: st['model'][d][i] for d in st['model']})
            bs.append({k: bm[k] - bd[k] for k in bd})
        err = {k: float(np.std([b[k] for b in bs])) for k in sd}
        res[p] = dict(n=n, data=sd, model=sm, diff_err=err,
                      chi2_dof=float(np.sum(acc[p]['chi2']) / np.sum(acc[p]['dof'])))
        dp = np.array(acc[p]['dp0'])
        res[p]['dp0_rsig'] = float(0.7413 * np.subtract(*np.percentile(dp, [75, 25])))
        print(f'[{a.label}] {p.upper()} n={n} chi2/dof={res[p]["chi2_dof"]:.3f}'
              f'  p0-ref rsig {res[p]["dp0_rsig"]:.3f} mm')
        for k in ('area_+1', 'area_-1', 'area_+2', 'area_-2', 'r21',
                  'dt_+1', 'dt_-1', 'dt_+2', 'dt_-2'):
            dd = sm[k] - sd[k]
            print(f'   {k:8s} data {sd[k]:8.3f}  model {sm[k]:8.3f}  '
                  f'm-d {dd:+8.3f} ± {err[k]:.3f}  ({dd / max(err[k], 1e-9):+5.1f}σ)')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
