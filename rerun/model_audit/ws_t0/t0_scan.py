#!/usr/bin/env python3
"""t0_scan.py --cache c.pkl --bundle b --out o.npz [--C free,200:460:20] [--start 180 --n 300] [--jobs 4]

Pin every plane's t0 at  C - 10 ns * ftst  (one trigger-latency constant per plane, the trigger phase from the fine
timestamp) and refit (p0, w, charge profile) through the pass's own fitter, for a grid of C; plus one free-t0 arm
(C = nan).  Rows: (eid, view, C, tan_ref, ref_mesh, w[um/ns], p0, t0, chi2, dof, q_uend, q_u50, sigma_p0_used, ftst).

What it answers (workstream D, model audit 2026-10-27): is there ONE latency constant per plane that the data prefer
(total chi2 minimum), is it the same across chambers on the same FEU, and what pinning at it does to the ladder, the
angle resolution and the position at the mesh compared with the free t0.
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..', '..'))
_EV = None; _CS = None


def _init(cache, bundle, k, Cs):
    global _EV, _CS
    sys.path.insert(0, REPO)
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    cal.t0_prior_sigma = 0.0             # the bundle's own (stale) prior off: every arm here sets t0 explicitly
    wm.use_calibration(cal); wm.set_depth_bins(k)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01
    wr.PIVOT_START = True
    _CS = Cs


def one(eid):
    from wft import model as wm, reco as wr
    cal = wm.CAL; ev = _EV[eid]; rows = []
    for vi, plane in enumerate('xy'):
        if plane not in ev or ev.get(f'ftst_{plane}') is None:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        ft = int(ev[f'ftst_{plane}'])
        if ft < 0:
            continue
        tr = float(ev[f'tan_{plane}']); ref = float(ev.get(f'ref_mesh_{plane}', np.nan))
        for C in _CS:
            prior = None if not np.isfinite(C) else (float(C - 10.0 * ft), 0.05)
            try:
                f = wr.fit_plane(P, plane, cal, t0_prior=prior)
            except Exception:
                f = None
            if f is None:
                continue
            rows.append((int(eid), vi, C, tr, ref, f.w * 1e3, f.p0, f.t0, f.chi2, f.dof, f.q_uend, f.q_u50,
                         float(getattr(f, '_sigma_p0', np.nan)), ft))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--C', default='200:460:20', help='comma list of latency constants [ns]: values, lo:hi:step ranges, and/or free (t0 unpinned)')
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    Cs = []
    for tok in a.C.split(','):
        if tok == 'free':
            Cs.append(np.nan)
        elif ':' in tok:
            lo, hi, st = (float(x) for x in tok.split(':'))
            Cs += list(np.arange(lo, hi + 1e-9, st))
        else:
            Cs.append(float(tok))
    C = pickle.load(open(a.cache, 'rb'))
    ids = sorted(C)[a.start:a.start + a.n]
    del C
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k, Cs)) as pool:
        rows = [r for rs in pool.map(one, ids, chunksize=2) for r in rs]
    per = np.array(rows, float)
    env = {k: v for k, v in os.environ.items() if k.startswith('WFT_')}
    np.savez(a.out, per=per, bundle=a.bundle, env=json.dumps(env), args=json.dumps(vars(a)))
    print(a.out, per.shape)


if __name__ == '__main__':
    main()
