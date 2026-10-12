#!/usr/bin/env python3
"""x_width_map.py -- where on the chamber does the extra X width growth sit?  (FINDINGS §29)

Head-on X events (|tan_x| < 0.03) binned by the reference's mesh position, across the X strips
(ref_mesh_x) and along them (ref_mesh_y).  Per bin:
  * model-free: the stack's second moment over offsets, m2(t) = sum_o (o p)^2 M_o / sum_o M_o, averaged
    early (EARLY ns after the threshold crossing: shallow charge) and late (LATE: deep charge), and
    their difference, the growth;
  * x_width's template fit, sigma_0 and Dd free (they trade), and with sigma_0, T pinned at the pooled
    values (D_T then carries any difference).
det3 pools the long run with the amplification-scan sub-runs at the same drift field (425-505 V,
where D_T is gain-flat; above 505 V the stacks saturate).

    x_width_map.py [--nbins 5] [--nboot 30] -> results/x_width_map.json
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x_width as XW                                       # noqa: E402
from rc_diffusion import stack, GRID, OFFS                 # noqa: E402
from width_vs_time import PITCH                            # noqa: E402
from bench_attach import template, GAP                     # noqa: E402

EARLY, LATE = (0.0, 240.0), (600.0, 900.0)


def det3_pool():
    fs = [f'{XW.CACHES}/headon_sat_det3__long_run_resist_490V_drift_1000V.pkl']
    for f in sorted(glob.glob(f'{XW.CACHES}/headon_sat_det3__hv_scan*_drift_1000V.pkl')):
        if int(re.search(r'resist_(\d+)V', f).group(1)) <= 505:
            fs.append(f)
    ev = {}
    for i, f in enumerate(fs):
        for k, e in pickle.load(open(f, 'rb')).items():
            ev[(i, k)] = e                                  # event ids collide between sub-runs
    return ev, [os.path.basename(f) for f in fs]


POOLS = {'det3': det3_pool,
         'det4': lambda: (pickle.load(open(f'{XW.CACHES}/headon_g_det4.pkl', 'rb')), ['headon_g_det4.pkl'])}


def m2_curve(M):
    w = np.nan_to_num(M)
    x2 = ((OFFS * PITCH) ** 2)[:, None]
    den = w.sum(0)
    return np.where(den > 0.05 * den.max(), (w * x2).sum(0) / np.where(den > 0, den, 1), np.nan)


def m2_one(args):
    ev, seed = args
    if seed is not None:
        ks = list(ev); rng = np.random.default_rng(seed)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    m2 = m2_curve(stack(ev, 'x', tan_max=XW.TAN)[1])
    sel = lambda w: np.nanmean(m2[(GRID >= w[0]) & (GRID < w[1])])
    return sel(EARLY), sel(LATE)


def m2_measure(pool, ev, nboot):
    rs = list(pool.map(m2_one, [(ev, None)] + [(ev, 2000 + b) for b in range(nboot)]))
    e, l = rs[0]
    be = np.array([r[0] for r in rs[1:]]); bl = np.array([r[1] for r in rs[1:]])
    return dict(m2_early=float(e), m2_late=float(l), growth=float(l - e),
                m2_early_err=float(be.std()), m2_late_err=float(bl.std()), growth_err=float((bl - be).std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbins', type=int, default=5)
    ap.add_argument('--nboot', type=int, default=30)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    out = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for det, load in POOLS.items():
            ev, files = load()
            xo = {k: e for k, e in ev.items() if abs(e['tan_x']) < XW.TAN}
            h, _ = template(det)
            T0 = GAP / (XW.V_BUNDLE[det] * 1e-3)
            allf = XW.measure(pool, h, xo, T0, a.nboot)
            allm = m2_measure(pool, xo, a.nboot)
            r = dict(files=files, n=len(xo), all=dict(allf, **allm), bins={})
            print(f'{det}: n {len(xo)}  sig0 {allf["sig0"]:.3f} D_T {allf["DT"]:.0f}±{allf["DT_err"]:.0f}  '
                  f'm2 early {allm["m2_early"]:.3f} late {allm["m2_late"]:.3f} growth {allm["growth"]:.3f}±{allm["growth_err"]:.3f}',
                  flush=True)
            for coord in ('ref_mesh_x', 'ref_mesh_y'):
                vals = np.array([e[coord] for e in xo.values()])
                edges = np.quantile(vals, np.linspace(0, 1, a.nbins + 1)); edges[-1] += 1e-6
                rows = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sub = {k: e for k, e in xo.items() if lo <= e[coord] < hi}
                    f = XW.measure(pool, h, sub, T0, a.nboot)
                    fx = XW.measure(pool, h, sub, allf['T'], a.nboot, sig0_fix=allf['sig0'])
                    m = m2_measure(pool, sub, a.nboot)
                    rows.append(dict(lo=float(lo), hi=float(hi), mid=float(np.median([e[coord] for e in sub.values()])),
                                     n=len(sub), free=f, fixed=fx, **m))
                    print(f'  {coord} {lo:5.0f}-{hi:5.0f} n {len(sub):4d} | m2 early {m["m2_early"]:.3f}±{m["m2_early_err"]:.3f} '
                          f'late {m["m2_late"]:.3f}±{m["m2_late_err"]:.3f} growth {m["growth"]:.3f}±{m["growth_err"]:.3f} | '
                          f'free sig0 {f["sig0"]:.3f}±{f["sig0_err"]:.3f} D_T {f["DT"]:4.0f}±{f["DT_err"]:3.0f} | '
                          f'pinned D_T {fx["DT"]:4.0f}±{fx["DT_err"]:3.0f}', flush=True)
                r['bins'][coord] = rows
            out[det] = r
    json.dump(out, open(f'{HERE}/results/x_width_map.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
