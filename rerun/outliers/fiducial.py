#!/usr/bin/env python3
"""fiducial.py -- every angle/position metric split by region and by selection category (C22 follow-up, Dylan 10-25:
characterize the edges separately; keep the competing-cluster events as their own category).

Region of a (event, view), from the M3 footprint [ref_mesh, ref_mesh + 30 tan_ref] of that view:
  outside      the footprint leaves the measured active area in either coordinate
  edge 0-5 / 5-10 / 10-15 / 15-20 / 20-30   distance [mm] of the footprint to the nearest active edge of the view's OWN coordinate
  edge-other   own coordinate >= 30 mm inside, but within 8 mm of the other coordinate's active edge
  interior     everything else
  (det7 X card 1 and det4 dead bands are interior/edge like the rest but reported apart: `hw`.)
Category: accepted = the production selection (fit ok, n_dropped <= 2); competing = fit ok but n_dropped > 2 (the
cluster-quality cut rejects it). Their union = all fitted.
Metrics per cell: n; angle s68 (|dθ| < 5°, ladder-scaled as finalize_compare) and far share; signed angle bias with
the sign folded so that + = deep charge displaced toward the chamber centre (edge cells), k = fitted tan / M3 tan;
position residual s68 / far (|res| > 3 mm) and the inward-folded position bias.
Writes report/fiducial.json and prints the interior and edge tables.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from olib import BASE, DETS, HEADON, FAR, load  # noqa: E402
from classify import DETNAME  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

REGIONS = ['interior', 'edge-other', 'edge 20-30', 'edge 15-20', 'edge 10-15', 'edge 5-10', 'edge 0-5', 'outside']


def regions(d, det, v):
    act = TRUE_ACTIVE_BY_DET[DETNAME[det]]; o = 'y' if v == 'x' else 'x'
    fp = lambda c: (np.minimum(d[f'ref_mesh_{c}'], d[f'ref_mesh_{c}'] + 30 * d[f'ref_tan_{c}']),
                    np.maximum(d[f'ref_mesh_{c}'], d[f'ref_mesh_{c}'] + 30 * d[f'ref_tan_{c}']))
    lo, hi = fp(v); olo, ohi = fp(o)
    d_lo, d_hi = lo - act[v][0], act[v][1] - hi
    dist = np.minimum(d_lo, d_hi)
    inward = np.where(d_lo < d_hi, 1.0, -1.0)          # + = toward the centre along this coordinate
    od = np.minimum(olo - act[o][0], act[o][1] - ohi)
    r = np.full(len(d), 'interior', dtype=object)
    r[od < 8] = 'edge-other'
    for a, b in ((20, 30), (15, 20), (10, 15), (5, 10), (0, 5)):
        r[(dist >= a) & (dist < b)] = f'edge {a}-{b}'
    r[(dist < 0) | (od < 0)] = 'outside'
    hw = np.zeros(len(d), bool)
    if det == 'det7' and v == 'x':
        hw = (lo < 199.3).to_numpy()
    if det == 'det4':
        L = pd.read_parquet(os.path.join(BASE, 'an', 'live_det4.parquet')).set_index('event_id').live
        hw = (d.event_id.map(L) < 0.3).to_numpy()
    return r, inward, dist.to_numpy(), hw


def metrics(u, v, inward):
    dl = u[f'dl_{v}'].to_numpy(); t = u[f'ref_tan_{v}'].to_numpy(); f = u[f'tl_{v}'].to_numpy()
    rr = u[f'rres_{v}'].to_numpy(); n = len(u)
    if n < 15:
        return dict(n=n)
    far = np.abs(dl) > FAR; pf = np.abs(rr) > 3; ho = np.abs(t) < HEADON
    m = ~far & np.isfinite(dl)
    k = np.nan
    mi = (np.abs(t) > 0.1) & ~far
    if mi.sum() > 15:
        k = float(np.sum(t[mi] * f[mi]) / np.sum(t[mi] ** 2))
    # edge characterization without the far cut (the far cut removes most of an edge band): robust k through all
    # tracks with |tan| < 0.5, and the median inward-folded tilt of head-on tracks
    ka = np.nan
    mk = np.isfinite(t) & np.isfinite(f) & (np.abs(t) < 0.5)
    for _ in range(5):
        if mk.sum() < 15:
            break
        ka = float(np.sum(t[mk] * f[mk]) / np.sum(t[mk] ** 2))
        res = f - ka * t; mk &= np.abs(res) < max(3 * np.percentile(np.abs(res[mk]), 68), 0.02)
    tilt = float(np.median((inward * (f - t))[ho])) if ho.sum() > 10 else np.nan
    return dict(n=n, k_all=ka, tilt_in_headon=tilt, s68=float(np.percentile(np.abs(dl[m]), 68)) if m.any() else np.nan,
                far=float(100 * far.mean()), far_headon=float(100 * far[ho].mean()) if ho.sum() > 10 else np.nan,
                bias_in=float(np.median((inward * (f - t))[ho & ~far])) if (ho & ~far).sum() > 10 else np.nan,
                k=k, pos_s68=float(np.percentile(np.abs(rr[~pf] - np.median(rr[~pf])), 68)) if (~pf).any() else np.nan,
                pos_far=float(100 * pf.mean()), pos_bias_in=float(np.median((inward * rr)[~pf])) if (~pf).any() else np.nan)


def main():
    out = {}
    for det in DETS:
        d = load(det, quality=False)
        for v in 'xy':
            r, inward, dist, hw = regions(d, det, v)
            fitted = d[f'{v}_ok'] & np.isfinite(d[f'ref_tan_{v}']) & np.isfinite(d[f'{v}_w'])
            comp = d[f'{v}_n_dropped'] > 2
            for cat, sel in (('accepted', fitted & ~comp), ('competing', fitted & comp), ('all fitted', fitted)):
                for reg in REGIONS:
                    for hwlab, hm in (('', ~hw), ('hw', hw)):
                        m = (sel & (r == reg) & hm).to_numpy()
                        if hwlab and not hw.any():
                            continue
                        out[f'{det}|{v}|{cat}|{reg}{"|hw" if hwlab else ""}'] = metrics(d[m], v, inward[m])
                tot = int(sel.sum())
                out[f'{det}|{v}|{cat}|_n'] = tot
    json.dump(out, open(os.path.join(BASE, 'report', 'fiducial.json'), 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    rows = []
    for det in DETS:
        for v in 'xy':
            for cat in ('accepted', 'competing'):
                for reg in ('interior', 'edge 20-30', 'edge 15-20', 'edge 10-15', 'edge 5-10', 'edge 0-5', 'edge-other'):
                    e = out.get(f'{det}|{v}|{cat}|{reg}', {})
                    if e.get('n', 0) >= 15:
                        rows.append(dict(det=det, v=v, cat=cat, region=reg, **{k: round(x, 3) if isinstance(x, float) else x for k, x in e.items()}))
    T = pd.DataFrame(rows); print(T.to_string(index=False))
    T.to_csv(os.path.join(BASE, 'report', 'fiducial.csv'), index=False)


if __name__ == '__main__':
    main()
