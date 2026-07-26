#!/usr/bin/env python3
"""det2 (detector B): freefit benchmark + endpoint + gap map (standalone)."""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2
import forward_model3 as fm3

OUT_DIR = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_6-22-26/'
           'long_run/mx17_2/waveform_first')
events = pickle.load(open(os.path.join(OUT_DIR, 'wfcache.pkl'), 'rb'))['events']
hj = json.load(open(os.path.join(OUT_DIR, 'hyper_det2.json')))
H = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V2 = hj['v']
DT = 60.0

ns2 = next(iter(events.values()))['x']['W'].shape[1]
fm2.set_nsamp(ns2)
tz = np.load(os.path.join(OUT_DIR, 'templates_perplane.npz'))
fm2.TGRID = tz['grid']
fm2.TMPL = {'x': tz['tmpl_x'], 'y': tz['tmpl_y']}
fm2.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
fm2._smear_cache.clear()
fm3._TT_CACHE.clear()

good = [e for e, ev in events.items()
        if ev['x']['W'].size and ev['y']['W'].size]
cand = [e for e in good
        if 0.10 < np.hypot(events[e]['tan_x'], events[e]['tan_y']) < 0.45
        and events[e]['x']['W'].max() > 250 and events[e]['y']['W'].max() > 250]
rng = np.random.default_rng(2222)
rng.shuffle(cand)
test = cand[130:130 + 800]
epsel = [e for e in good
         if max(abs(events[e]['tan_x']), abs(events[e]['tan_y'])) > 0.08][:5000]
print(f'det2: {len(cand)} candidates, freefit test {len(test)}, endpoint {len(epsel)}',
      flush=True)


def fit_free(eid):
    ev = events[eid]
    o = dict(eid=eid)
    try:
        for plane in ('x', 'y'):
            tn = ev[f'tan_{plane}']
            p0r = ev[f'ref_mesh_{plane}']
            g = fm2.init_guess(ev[plane], plane, tn, p0r, V2 * 1e-3)
            r = fm3.fit_plane(ev[plane], plane, *g, hyper=H)
            o[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r['p0'], w=r['w'],
                            chi2=r['chi2'], dof=r['dof'])
    except Exception as ex:
        o['error'] = str(ex)
    return o


def fit_q(args):
    eid, K = args
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = events[eid]
    out = []
    for plane in ('x', 'y'):
        tn = ev[f'tan_{plane}']
        if abs(tn) < 0.08 or ev[plane]['W'].size == 0:
            continue
        W, noise, pos, sat = fm2.prep_plane(ev[plane], plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat,
                                  ev[f'ref_mesh_{plane}'], tn * V2 * 1e-3,
                                  float(t0), H)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append((ev['ref_mesh_x'], ev['ref_mesh_y'], q / q.sum()))
    return out


def U50g(uk, prof, pb=(1, 6)):
    plat = np.median(prof[pb[0]:pb[1]])
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= pb[1] - 1]
    if len(below) == 0 or plat <= 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1


def rs(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=14) as pool:
        resf = list(pool.map(fit_free, test, chunksize=4))
        for p in ('x', 'y'):
            tan, w, p0, p0r = [], [], [], []
            for r in resf:
                if p in r and 'error' not in r:
                    d = r[p]
                    tan.append(d['tan_ref']); w.append(d['w'])
                    p0.append(d['p0']); p0r.append(d['p0_ref'])
            tan, w, p0, p0r = map(np.array, (tan, w, p0, p0r))
            tanf = w * 1e3 / V2
            dth = np.degrees(np.arctan(tanf)) - np.degrees(np.arctan(tan))
            dp = p0 - p0r
            at = np.abs(tan)
            vimp = [round(float(np.nanmedian((w * 1e3 / tan)[(at >= a) & (at < b)])), 1)
                    for a, b in ((0.08, 0.15), (0.15, 0.25), (0.25, 0.45))]
            print(f'det2 {p}: angle med {np.nanmedian(dth):+.2f} sig {rs(dth):.2f} deg; '
                  f'mesh sig {rs(dp):.3f} mm; v(bins) {vimp}', flush=True)
        pickle.dump(resf, open(os.path.join(OUT_DIR, 'freefit_det2.pkl'), 'wb'))

        rows = []
        for o in pool.map(fit_q, [(e, 22) for e in epsel], chunksize=8):
            rows.extend(o)
    X = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows])
    L = max(len(r[2]) for r in rows)
    P = np.zeros((len(rows), L))
    for i, r in enumerate(rows):
        P[i, :len(r[2])] = r[2]
    uk = (np.arange(L) + 0.5) * DT
    U = U50g(uk, P.mean(axis=0))
    print(f'det2 overall column: U50={U:.0f} ns -> {V2*U*1e-3:.1f} mm (n={len(P)})',
          flush=True)
    xq = np.percentile(X, [0, 33, 66, 100]); yq = np.percentile(Y, [0, 33, 66, 100])
    M = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            m = (X >= xq[i]) & (X < xq[i + 1]) & (Y >= yq[j]) & (Y < yq[j + 1])
            if m.sum() > 120:
                M[i, j] = U50g(uk, P[m].mean(axis=0)) * V2 * 1e-3
    print('det2 column map [mm]:'); print(np.round(M, 1))
    json.dump(dict(U=float(U), col=float(V2 * U * 1e-3), map=M.tolist()),
              open(os.path.join(OUT_DIR, 'det2_column.json'), 'w'))
