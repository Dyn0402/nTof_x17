#!/usr/bin/env python3
"""run_fitters.py --cache c.pkl --bundle b --fitters prod,pivot2 --out o.json [--n 400] [--jobs 8]
Held-out events (cache ids 180+) through each fitter; stores per-event p0, w [um/ns], t0, chi2, nfev and tan_ref."""
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
_EV = None; _FIT = None


def _init(cache, bundle, k, fitters):
    global _EV, _FIT
    sys.path.insert(0, REPO); sys.path.insert(0, HERE)
    from wft.calib import CalibrationBundle
    from wft import model as wm
    import fitters as F
    _EV = pickle.load(open(cache, 'rb'))
    wm.use_calibration(CalibrationBundle.load(bundle)); wm.set_depth_bins(k); wm.COL_GUARD = 0.01
    _FIT = {n: F.FITTERS[n] for n in fitters}


def one(eid):
    from wft import model as wm
    cal = wm.CAL; ev = _EV[eid]; out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        ft = ev.get(f'ftst_{plane}')
        t0p = (cal.t0_abs or {}).get(plane, {}).get(str(ft), (cal.t0_abs or {}).get(plane, {}).get(ft))
        row = dict(tr=float(ev[f'tan_{plane}']), ref=float(ev[f'ref_mesh_{plane}']))
        for n, f in _FIT.items():
            t = time.time()
            try:
                r = f(P, plane, t0p, tr=row['tr'], ref=row['ref'])
                row[n] = dict(p0=float(r['p0']), w=float(r['w'] * 1e3), t0=float(r['t0']), chi2=float(r['chi2']),
                              nfev=int(r['nfev']), sec=time.time() - t)
            except Exception as ex:
                row[n] = dict(err=repr(ex))
        out[plane] = row
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--fitters', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=400); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ids = sorted(pickle.load(open(a.cache, 'rb')))[a.start:a.start + a.n]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k, a.fitters.split(','))) as pool:
        res = dict(pool.map(one, ids, chunksize=2))
    json.dump(dict(fitters=a.fitters, bundle=a.bundle, events=res), open(a.out, 'w'))


if __name__ == '__main__':
    main()
