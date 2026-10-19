#!/usr/bin/env python3
"""xw_amp.py -- faint against bright head-on events: what differs?  (FINDINGS §33 follow-up)

§32/§33: on det4 and det7 the unsaturated X D_T falls by ~100 um/sqrt(cm) as the window-maximum cut
tightens, and the xw_sim closure toy (same estimator, real layouts, tuned amplitude spectra) shows no
such dependence.  Cumulative cuts mix classes, so here the unsaturated head-on events (|tan_x| < 0.03,
window max < SAT) are split into exclusive amplitude terciles and compared directly:

  where       reference position (mean, spread, fraction near the edges of the hit area), |tan_x|
  fit         x_width.fit_x free; T pinned at the all-unsat T; sigma_0 and T pinned (then only Dd moves)
  model-free  the stack's second moment over offsets, early (0-240 ns) and late (600-900 ns), and growth
  toy         the same terciles on xw_sim 'base' replicas (truth known), for the expected values

    xw_amp.py [--sets det4,det7R,det7H,det3L] [--nboot 20] [--nrep 10]   -> results/xw_amp.json
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x_width as xw                                         # noqa: E402
import xw_sim as X                                           # noqa: E402
from x_width_map import m2_curve, EARLY, LATE                # noqa: E402
from rc_diffusion import stack, GRID                         # noqa: E402

SAT = 3500.0


def head(ev):
    return {k: e for k, e in ev.items() if 'x' in e and abs(e['tan_x']) < xw.TAN and e['x']['W'].max() < SAT}


def classes(ev, edges):
    mx = {k: e['x']['W'].max() for k, e in ev.items()}
    return [{k: e for k, e in ev.items() if lo <= mx[k] < hi} for lo, hi in zip(edges[:-1], edges[1:])]


def m2(ev):
    c = m2_curve(stack(ev, 'x', tan_max=xw.TAN)[1])
    s = lambda w: float(np.nanmean(c[(GRID >= w[0]) & (GRID < w[1])]))
    return s(EARLY), s(LATE)


def one(args):
    h, ev, T0, mode, pin, seed = args
    if seed is not None:
        rng = np.random.default_rng(seed); ks = list(ev)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    n, M = stack(ev, 'x', tan_max=xw.TAN)
    if mode == 'free':
        f = xw.fit_x(h, M, T0)
    elif mode == 'Tpin':
        f = xw.fit_x(h, M, pin['T'], free_T=False)
    else:
        f = xw.fit_x(h, M, pin['T'], sig0_fix=pin['sig0'])
    e, l = m2(ev)
    return dict(f, m2_early=e, m2_late=l, growth=l - e)


def measure(pool, h, ev, T0, mode, pin, nboot):
    rs = list(pool.map(one, [(h, ev, T0, mode, pin, s) for s in [None] + [7000 + b for b in range(nboot)]]))
    c, bs = rs[0], rs[1:]
    return dict(c, **{k + '_err': float(np.std([b[k] for b in bs])) for k in ('DT', 'v', 'sig0', 'm2_early', 'm2_late', 'growth')})


def where(ev, allev):
    x = np.array([e['ref_mesh_x'] for e in ev.values()]); y = np.array([e['ref_mesh_y'] for e in ev.values()])
    ax = np.array([e['ref_mesh_x'] for e in allev.values()]); ay = np.array([e['ref_mesh_y'] for e in allev.values()])
    xl, xh = np.quantile(ax, [0.02, 0.98]); yl, yh = np.quantile(ay, [0.02, 0.98])
    edge = ((x - xl < 20) | (xh - x < 20) | (y - yl < 20) | (yh - y < 20)).mean()
    return dict(x_mean=float(x.mean()), x_sd=float(x.std()), y_mean=float(y.mean()), y_sd=float(y.std()),
                frac_edge20=float(edge), abs_tan=float(np.mean([abs(e['tan_x']) for e in ev.values()])),
                total=float(np.median([e['x']['W'].sum() for e in ev.values()])))


def toy_one(args):
    S, seed, edges = args
    ev = head(X.Gen(S, 'base', seed).sample())
    out = []
    for sub in classes(ev, edges):
        if sum(1 for _ in sub) < 40:
            out.append(None); continue
        n, M = stack(sub, 'x', tan_max=xw.TAN)
        f = xw.fit_x(S['h_fit'], M, S['T0'])
        e, l = m2(sub)
        out.append(dict(DT=f['DT'], v=f['v'], sig0=f['sig0'], m2_early=e, m2_late=l, growth=l - e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='det4,det7R,det7H,det3L')
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--nrep', type=int, default=10)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    inp = pickle.load(open(X.INPUTS, 'rb'))
    out_path = f'{HERE}/results/xw_amp.json'
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    with ProcessPoolExecutor(a.workers) as pool:
        for name in a.sets.split(','):
            S = inp[name]
            allev = {k: e for k, e in pickle.load(open(f'{xw.CACHES}/headon_{S["key"]}.pkl', 'rb')).items() if 'x' in e}
            ev = head(allev)
            mx = np.array([e['x']['W'].max() for e in ev.values()])
            edges = np.r_[0.0, np.quantile(mx, [1 / 3, 2 / 3]), SAT]
            pooled = measure(pool, S['h_fit'], ev, S['T0'], 'free', None, a.nboot)
            r = dict(set=name, truth_DT=S['DT'], edges=edges.tolist(), pooled=pooled, classes=[])
            print(f'{name}: unsat head-on {len(ev)}, terciles of window max {np.round(edges)}; pooled D_T {pooled["DT"]:.0f} '
                  f'v {pooled["v"]:.2f} sig0 {pooled["sig0"]:.3f}', flush=True)
            toys = list(pool.map(toy_one, [(S, 800000 + k, edges) for k in range(a.nrep)]))
            for i, sub in enumerate(classes(ev, edges)):
                c = dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=len(sub), where=where(sub, allev))
                for mode in ('free', 'Tpin', 'pin'):
                    c[mode] = measure(pool, S['h_fit'], sub, S['T0'], mode, pooled, a.nboot)
                tv = [t[i] for t in toys if t[i] is not None]
                c['toy'] = {k: dict(mean=float(np.mean([t[k] for t in tv])), rms=float(np.std([t[k] for t in tv], ddof=1)))
                            for k in ('DT', 'v', 'sig0', 'm2_early', 'm2_late', 'growth')} if len(tv) > 2 else {}
                r['classes'].append(c)
                f, tp, p, w = c['free'], c['Tpin'], c['pin'], c['where']
                ty = c['toy']
                print(f'  max {edges[i]:5.0f}-{edges[i + 1]:5.0f} n {len(sub):4d} | free D_T {f["DT"]:4.0f}±{f["DT_err"]:3.0f} '
                      f'v {f["v"]:5.2f}±{f["v_err"]:.2f} sig0 {f["sig0"]:.3f}±{f["sig0_err"]:.3f} | Tpin D_T {tp["DT"]:4.0f}±{tp["DT_err"]:3.0f} '
                      f'sig0 {tp["sig0"]:.3f} | pin D_T {p["DT"]:4.0f}±{p["DT_err"]:3.0f} | m2 early {f["m2_early"]:.3f}±{f["m2_early_err"]:.3f} '
                      f'growth {f["growth"]:.3f}±{f["growth_err"]:.3f}', flush=True)
                if ty:
                    print(f'      toy: D_T {ty["DT"]["mean"]:4.0f} (rms {ty["DT"]["rms"]:3.0f}) v {ty["v"]["mean"]:5.2f} sig0 {ty["sig0"]["mean"]:.3f} '
                          f'm2 early {ty["m2_early"]["mean"]:.3f} growth {ty["growth"]["mean"]:.3f} (rms {ty["growth"]["rms"]:.3f})', flush=True)
                print(f'      where: x {w["x_mean"]:6.1f}±{w["x_sd"]:5.1f} y {w["y_mean"]:6.1f}±{w["y_sd"]:5.1f} edge20 {w["frac_edge20"]:.2f} '
                      f'|tan| {w["abs_tan"]:.4f} median total {w["total"]:.0f}', flush=True)
            out[name] = r
            json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
