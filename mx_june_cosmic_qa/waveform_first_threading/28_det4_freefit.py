#!/usr/bin/env python3
"""det4 freefit + benchmark using the saved calibration (standalone)."""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REPO = '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading'
sys.path.insert(0, REPO)
import forward_model2 as fm2
import forward_model3 as fm3

OUT_DIR = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det4_day_6-24-26/'
           'long_run/mx17_4/waveform_first')
events = pickle.load(open(os.path.join(OUT_DIR, 'wfcache.pkl'), 'rb'))['events']
hj = json.load(open(os.path.join(OUT_DIR, 'hyper_det4.json')))
H = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V4 = hj['v']

ns_det = next(iter(events.values()))['x']['W'].shape[1]
fm2.set_nsamp(ns_det)
tz = np.load(os.path.join(OUT_DIR, 'templates_perplane.npz'))
fm2.TGRID = tz['grid']
fm2.TMPL = {'x': tz['tmpl_x'], 'y': tz['tmpl_y']}
fm2.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
fm2._smear_cache.clear()
fm3._TT_CACHE.clear()

cand = []
for eid, ev in events.items():
    if ev['x']['W'].size == 0 or ev['y']['W'].size == 0:
        continue
    t3 = np.hypot(ev['tan_x'], ev['tan_y'])
    if 0.10 < t3 < 0.40 and ev['x']['W'].max() > 250 and ev['y']['W'].max() > 250:
        cand.append(eid)
rng = np.random.default_rng(4242)
rng.shuffle(cand)
test = cand[130:130 + 800]
print(f'test {len(test)}', flush=True)


def fit_free(eid):
    ev = events[eid]
    o = dict(eid=eid)
    try:
        for plane in ('x', 'y'):
            tn = ev[f'tan_{plane}']
            p0r = ev[f'ref_mesh_{plane}']
            g = fm2.init_guess(ev[plane], plane, tn, p0r, V4 * 1e-3)
            r = fm3.fit_plane(ev[plane], plane, *g, hyper=H)
            o[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r['p0'], w=r['w'],
                            t0=r['t0'], chi2=r['chi2'], dof=r['dof'])
    except Exception as ex:
        o['error'] = str(ex)
    return o

if __name__ == '__main__':
    t_ = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        resf = list(pool.map(fit_free, test, chunksize=4))
    print(f'{len(resf)} in {time.time()-t_:.0f}s', flush=True)
    pickle.dump(resf, open(os.path.join(OUT_DIR, 'freefit_det4.pkl'), 'wb'))

    def rs(a):
        a = a[np.isfinite(a)]
        return 1.4826 * np.median(np.abs(a - np.median(a)))
    for p in ('x', 'y'):
        tan, w, p0, p0r = [], [], [], []
        for r in resf:
            if p in r and 'error' not in r:
                d = r[p]
                tan.append(d['tan_ref']); w.append(d['w'])
                p0.append(d['p0']); p0r.append(d['p0_ref'])
        tan, w, p0, p0r = map(np.array, (tan, w, p0, p0r))
        tanf = w * 1e3 / V4
        dth = np.degrees(np.arctan(tanf)) - np.degrees(np.arctan(tan))
        dp = p0 - p0r
        dev = np.maximum(np.abs(dp), np.abs(dp + (tanf - tan) * 29))
        at = np.abs(tan)
        vimp = [round(float(np.nanmedian((w * 1e3 / tan)[(at >= a) & (at < b)])), 1)
                for a, b in ((0.08, 0.15), (0.15, 0.25), (0.25, 0.45))]
        print(f'det4 {p}: angle med {np.nanmedian(dth):+.2f} sig {rs(dth):.2f} deg; '
              f'mesh sig {rs(dp):.3f} mm; <1mm {np.mean(dev<1)*100:.0f}%; '
              f'v(angle bins) {vimp}', flush=True)
