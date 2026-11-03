#!/usr/bin/env python3
"""ytilt_tests.py -- why does the fitted Y-view w0 gradient read 1.2-2.1x the X view's? (2026-11-03)
Tests on the final tables (bow_final parquets): tan(theta) distributions, angle-route refits with symmetric/matched tan
windows and without the t*U, t*W terms, and the cluster-width (sharing proxy) gradient across the own axis.
usage: ytilt_tests.py [dir]   (default ~/work_data/claude-jobs/bow_final)"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbcommon as F
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/work_data/claude-jobs/bow_final')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def fit(T, p, v, tmax=0.45, match=None, tU=True, nb=0):
    m = F.view_mask(T, p, tmax=tmax)
    t = T[f'{p}_t'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
    m &= np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
    if match is not None:           # keep a random subset so |t| histogram equals `match` (edges, probs)
        edges, pr = match
        idx = np.where(m)[0]; b = np.clip(np.digitize(np.abs(t[idx]), edges) - 1, 0, len(pr) - 1)
        rng = np.random.default_rng(5)
        keep = rng.random(len(idx)) < pr[b]
        mm = np.zeros_like(m); mm[idx[keep]] = True; m = mm
    t, wv = t[m], wv[m]
    U = (T.u.to_numpy()[m] - 200) / 100; W = (T.w.to_numpy()[m] - 200) / 100
    cols = [F.surf_terms(U, W), t] + ([t * U, t * W] if tU else [])
    X = np.column_stack(cols)
    b = F.huber(X, wv, 0.03 * v)
    return dict(cu=b[1], cw=b[2], B=b[6], n=int(m.sum())), t


def main():
    out = {}
    print('chamber | view: n, <t>, rms t, frac|t|>0.25 | own-axis w0 gradient (um/ns per 100 mm) full / |t|<0.2 / |t|<0.3 / matched-|t| / no tU,tW')
    for d in DETS:
        T = F.load_path(f'{D}/{d}.parquet', json.load(open(f'{D}/{d}.bundle.json')), d)
        v = T.attrs['v']; out[d] = {}
        # reference |t| histogram: pooled both views
        tx = np.abs(T.x_t.dropna().to_numpy()); ty = np.abs(T.y_t.dropna().to_numpy())
        edges = np.linspace(0, 0.45, 10)
        hx = np.histogram(tx, edges)[0].astype(float); hy = np.histogram(ty, edges)[0].astype(float)
        hmin = np.minimum(hx / hx.sum(), hy / hy.sum())
        for p in 'xy':
            h = hx if p == 'x' else hy
            pr = np.clip(hmin / (h / h.sum()), 0, 1)
            r = {}
            for lab, kw in (('full', {}), ('t02', dict(tmax=0.2)), ('t03', dict(tmax=0.3)), ('match', dict(match=(edges, pr))), ('notU', dict(tU=False))):
                f, t = fit(T, p, v, **kw); r[lab] = f['cu'] if p == 'x' else f['cw']
            t = T[f'{p}_t'].dropna().to_numpy()
            r['stat'] = dict(mean=float(t.mean()), rms=float(t.std()), f25=float((np.abs(t) > .25).mean()))
            out[d][p] = r
            print(f'{d} {p}: <t>={t.mean():+.3f} rms={t.std():.3f} f>0.25={(np.abs(t)>.25).mean():.2f} | ' +
                  ' '.join(f'{r[k]:+.3f}' for k in ('full', 't02', 't03', 'match', 'notU')))
        # cluster width (n_strips) vs own coordinate, tracks |t|<0.1 : sharing proxy gradient
        for p in 'xy':
            if f'{p}_n_strips' not in T: continue
            m = F.view_mask(T, p, tmax=0.1) & np.isfinite(T[f'{p}_n_strips'].to_numpy())
            pos = ((T.u if p == 'x' else T.w).to_numpy()[m] - 200) / 100
            ns = T[f'{p}_n_strips'].to_numpy()[m].astype(float)
            sl = np.polyfit(pos, ns, 1)[0]; mean = ns.mean()
            q = T[f'{p}_q_sum'].to_numpy()[m]; ok = np.isfinite(q)
            slq = np.polyfit(pos[ok], q[ok] / np.nanmedian(q[ok]), 1)[0]
            out[d][p]['nstrips_mean'] = float(mean); out[d][p]['nstrips_slope_per100mm'] = float(sl); out[d][p]['qsum_relslope'] = float(slq)
            print(f'   {d} {p}: n_strips mean {mean:.2f}  slope {sl:+.3f}/100mm   q_sum rel slope {slq:+.3f}/100mm')
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_final', 'ytilt_tests.json'), 'w'), indent=1, default=float)


def window_test():
    """(e) edge zone: own-axis w0 gradient with the track position restricted to central windows (both coordinates)."""
    print('\n(e) own-axis w0 gradient (um/ns per 100 mm), X | Y, track u,w restricted to [lo,hi] mm; ratio Y/X')
    res = {}
    for d in DETS:
        T = F.load_path(f'{D}/{d}.parquet', json.load(open(f'{D}/{d}.bundle.json')), d); v = T.attrs['v']
        line = []
        for lo, hi in ((25, 375), (60, 340), (100, 300)):
            g = {}
            for p in 'xy':
                m = F.view_mask(T, p) & (T.u > lo).to_numpy() & (T.u < hi).to_numpy() & (T.w > lo).to_numpy() & (T.w < hi).to_numpy()
                t = T[f'{p}_t'].to_numpy(); wv = T[f'{p}_wv'].to_numpy(); m &= np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
                U = (T.u.to_numpy()[m] - 200) / 100; W = (T.w.to_numpy()[m] - 200) / 100
                X = np.column_stack([F.surf_terms(U, W), t[m], t[m] * U, t[m] * W]); b = F.huber(X, wv[m], 0.03 * v)
                g[p] = b[1] if p == 'x' else b[2]
            line.append(f'[{lo},{hi}] {g["x"]:+.2f}|{g["y"]:+.2f} r={g["y"]/g["x"]:.2f}')
            res.setdefault(d, {})[f'{lo}_{hi}'] = g
        print(d, ' '.join(line))
    return res


if __name__ == '__main__':
    main()
    window_test()
