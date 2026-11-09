#!/usr/bin/env python3
"""remedies.py <window pickles> --ref FRAMES --bundle B --out OUT.json [--table T.parquet] [--jobs N] -- the literature's hit-level
remedies, and the charge-weighted time, scored on the same planes as the waveform fit (det3 Runs 1 and 2 against M3).

Per plane the strip waveforms W give, per family, one hit (t, a) per strip (emulate.py):
  V0  dream   DREAM leading edge (30 % of the peak)              V7  cw     charge-weighted mean time of the pulse
  V6  ideal   per-strip template fit (matched-filter time)       V8  totmid midpoint of the threshold crossings
Corrections, applied to the hit list of each family (the catalogue of LITERATURE_MIXING.md / strip_microtpcs.md):
  plain   amplitude-weighted line t(x) through all hits                             (H-naive of tiers.py)
  rob     iterative 3-sigma trimming of strips (Hough-like)                          NSW / APV25 chains
  e50/e25/e17  drop the first/last strip when its amplitude < 1/2 (Rinnagel), 1/4 (APV25), 1/6 (Sessa) of the inner neighbour
  shift   e17, then move the first/last strip toward the inside by (1 - a_edge/a_inner) p/2: the centroid of a partial
          column (Ntekas's geometric edge term; Farinelli's last-strip shift)
  sub07/sub15/sub20/sub30  time-bin-wise neighbour subtraction before the emulator, W_i - f (W_{i-1} + W_{i+1})
          (Lösel/Flierl, f = 0.15 simulated, 0.20 and 0.30 tuned; 0.07 = the same-time floor measured on chamber A)
  shift+sub15  both
Scores (tiers.score): d = atan(tan) - atan(tan_ref); naive tan = u / v; H0 (global remap, cross-validated); per |tan| bin
s68, bias and steep.  Everything is on planes with an M3 reference and a waveform fit (same set as tiers.py).
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402
from tiers import cv_fit, implied_v, load_ref, score  # noqa: E402

G = {}
FAMS = ('dream', 'cw', 'ideal', 'totmid')
SUBS = {'sub07': 0.07, 'sub15': 0.15, 'sub20': 0.20, 'sub30': 0.30}


def subtract(W, f):
    Wp = W.copy()
    Wp[1:] -= f * W[:-1]
    Wp[:-1] -= f * W[1:]
    return Wp


def edge_fix(pos, t, a, frac, shift):
    """drop the first/last hit strip below frac x the inner neighbour; optionally move a kept edge strip inward by
    (1 - a_edge/a_inner) p/2 (the centroid of the partial drift column it collects).  Returns (pos', t')."""
    k = np.flatnonzero(np.isfinite(t) & np.isfinite(a) & (a > 0))
    pos, t = pos.copy(), t.copy()
    if len(k) < 4:
        return pos, t
    k = k[np.argsort(pos[k])]
    pitch = np.median(np.diff(np.sort(pos)))
    for edge, inner, sgn in ((k[0], k[1], +1), (k[-1], k[-2], -1)):
        r = a[edge] / a[inner]
        if r < frac:
            t[edge] = np.nan
        elif shift and r < 1.0:
            pos[edge] += sgn * (1.0 - r) * pitch / 2
    return pos, t


def line_u(pos, t, a, weighted=True, robust=False, min_hits=3):
    k = np.isfinite(t) & np.isfinite(a) & ((a > 0) if weighted else True)
    if k.sum() < min_hits:
        return np.nan
    x, y = pos[k].astype(float), t[k].astype(float)
    wt = a[k].astype(float) if weighted else np.ones(k.sum())
    for _ in range(4 if robust else 1):
        xm, ym = np.average(x, weights=wt), np.average(y, weights=wt)
        sxx = np.sum(wt * (x - xm) ** 2)
        if sxx <= 0:
            return np.nan
        s = np.sum(wt * (x - xm) * (y - ym)) / sxx
        if not robust or len(x) <= min_hits:
            break
        r = y - (ym + s * (x - xm))
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r) <= 3 * max(sig, 10.0)
        if keep.all() or keep.sum() < min_hits:
            break
        x, y, wt = x[keep], y[keep], wt[keep]
    return 1.0 / s if abs(s) > 1e-9 else np.nan


def emulate(fam, W, nz):
    if fam == 'dream':
        return E.dream(W, nz)
    if fam == 'ideal':
        return G['ideal'][G['view']](W, nz)
    return getattr(E, fam)(W, nz)


def one(p):
    v = p['view']
    tr = G['ref'].get((int(p['event_id']), v))
    if tr is None:
        return None
    G['view'] = v
    W, nz, pos = p['W'].astype(float), p['noise'].astype(float), p['pos'].astype(float)
    meta = G['meta']
    r = dict(event_id=int(p['event_id']), view=v, tan_ref=tr,
             tan_W=(p['w'] * 1e3 - meta['w0'][v]) / (meta['kw'][v] * meta['v_drift']))
    variants = {'': W}
    for name, f in SUBS.items():
        variants[name] = subtract(W, f)
    for fam in FAMS:
        for vn, Wv in variants.items():
            h = emulate(fam, Wv, nz)
            t, a = h['t'], h['a']
            tag = f'{fam}{("_" + vn) if vn else ""}'
            r[f'u_{tag}'] = line_u(pos, t, a)
            if vn == '' or vn == 'sub15':
                r[f'u_{tag}_rob'] = line_u(pos, t, a, robust=True)
                for frac, lab in ((0.5, 'e50'), (0.25, 'e25'), (1 / 6, 'e17')):
                    pp, tt = edge_fix(pos, t, a, frac, False)
                    r[f'u_{tag}_{lab}'] = line_u(pp, tt, a, robust=True)
                pp, tt = edge_fix(pos, t, a, 1 / 6, True)
                r[f'u_{tag}_shift'] = line_u(pp, tt, a, robust=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--table')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--nmax', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    if a.nmax:
        planes = planes[:a.nmax]
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    G.update(ref=load_ref(a.ref), meta=meta, ideal={v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in 'xy'})
    with mp.Pool(a.jobs) as P:
        rows = [r for r in P.imap_unordered(one, planes, chunksize=32) if r is not None]
    T = pd.DataFrame(rows)
    if a.table:
        T.to_parquet(a.table)
    v = meta['v_drift'] / 1000.0
    res = dict(meta=dict(key=meta['key'], v_drift=meta['v_drift'], n_planes=len(T)), views={})
    ucols = [c for c in T.columns if c.startswith('u_')]
    for view in ('x', 'y'):
        g = T[T.view == view].reset_index(drop=True)
        tr, eid = g.tan_ref.to_numpy(), g.event_id.to_numpy()
        V = dict(n=len(g), W=score(g.tan_W.to_numpy(), tr), tiers={})
        for c in ucols:
            u = g[c].to_numpy()
            V['tiers'][c[2:]] = dict(implied_v=implied_v(u, tr), coverage=float(np.isfinite(u).mean()),
                                     naive=score(u / v, tr), H0=score(cv_fit(u, tr, eid, 'H0'), tr))
        res['views'][view] = V
        print(f'[{view}] {len(g)} planes; W s68 {V["W"]["all"]["s68"]:.3f}')
        for name, t in V['tiers'].items():
            nv, h0 = t['naive']['all'], t['H0']['all']
            bins = ' '.join(f"{(t['naive'][b]['steep'] if t['naive'][b]['steep'] is not None else float('nan')):+5.2f}" for b in ('0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5'))
            print(f'   {name:18s} cov {t["coverage"]:.2f} v {t["implied_v"] or float("nan"):5.1f}  naive s68 {nv["s68"] or float("nan"):.2f} steep {nv["steep"] or float("nan"):+.2f} [{bins}]  H0 s68 {h0["s68"] or float("nan"):.2f}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
