#!/usr/bin/env python3
"""lm_ab.py -- A/B of the outer solver (wft.model.FITTER 'nm' vs 'lm', branch lm-varpro) through the production entry
point wft.reco.fit_plane (pivot start, dead mask, per-ftst t0 prior, errors), on the vladder bench caches of the five
chambers (M3 reference tan in the cache).

Arms per plane:
  nm, lm            the bench configuration (t0 prior from the bundle's t0_abs, sigma 5 ns)
  nm_free, lm_free  no t0 prior (the beam situation: t0 free), --free
  lm_nohop, lm_free_nohop   the same without the lattice hops (LM_HOPS = 0), to isolate them
  lm_grid           --grid: LM from the nm grid's best point, no hops (LM_START = 'grid'), the minimal-change form

    .venv/bin/python rerun/model_audit/algo_review/lm_ab.py --dets det3 det2 --n 300 --jobs 4 [--free]
Writes lm_ab_rows_<det>.pkl next to --out; lm_ab_summary.py makes the tables.
"""
import os, sys, time, pickle, argparse
import numpy as np
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
VL = '/media/dylan/data/x17/cosmic_bench/vladder'
BUNDLE = {'det3': 'bundle_phys_mask'}           # others: bundle_phys

_ST = {}


def _init(det, free, grid_only=False):
    from wft.calib import CalibrationBundle
    from wft import model as wm
    cal = CalibrationBundle.load(os.path.join(VL, det, BUNDLE.get(det, 'bundle_phys')))
    wm.use_calibration(cal)
    _ST.update(cal=cal, ev=pickle.load(open(os.path.join(VL, det, 'cache.pkl'), 'rb')), free=free, det=det, grid_only=grid_only)
    orig = wm._solve_nnls
    _ST['n'] = 0

    def counted(*a, **kw):
        _ST['n'] += 1
        return orig(*a, **kw)
    wm._solve_nnls = counted


def _arm(P, plane, cal, fitter, prior, hops=3, start='ladder'):
    from wft import model as wm, reco as wr
    wm.FITTER, wm.LM_HOPS, wm.LM_START = fitter, hops, start
    _ST['n'] = 0
    t = time.perf_counter()
    try:
        f = wr.fit_plane(P, plane, cal, t0_prior=prior)
    except Exception as e:                        # noqa: BLE001
        f = None
        print('fail', plane, fitter, repr(e), flush=True)
    dt, n = time.perf_counter() - t, _ST['n']
    wm.FITTER, wm.LM_HOPS, wm.LM_START = 'nm', 3, 'ladder'
    if f is None:
        return None
    return dict(p0=f.p0, w=f.w, t0=f.t0, tan=f.tan_theta, chi2=f.chi2, dof=f.dof, p0_err=f.p0_err, w_err=f.w_err,
                t0_err=f.t0_err, tan_err=f.tan_err, q_sum=f.q_sum, q_uend=f.q_uend, time=dt, nsolve=n)


def one(eid):
    from wft import reco as wr
    cal, ev = _ST['cal'], _ST['ev'][eid]
    out = []
    for plane in 'xy':
        if plane not in ev:
            continue
        P = ev[plane]
        row = dict(det=_ST['det'], eid=int(eid), plane=plane, tan_ref=float(ev.get(f'tan_{plane}', np.nan)),
                   ref_mesh=float(ev.get(f'ref_mesh_{plane}', np.nan)))
        prior = wr.t0_prior_for(cal, plane, ev.get(f'ftst_{plane}'))
        row['t0_pred'] = prior[0] if prior is not None else np.nan
        if _ST['grid_only']:
            arms = [('nm', 'nm', prior, 3), ('lm_grid', 'lm', prior, 0, 'grid')]
        else:
            arms = [('nm', 'nm', prior, 3), ('lm', 'lm', prior, 3), ('lm_nohop', 'lm', prior, 0)]
        if _ST['free']:
            arms += [('nm_free', 'nm', None, 3), ('lm_free', 'lm', None, 3), ('lm_free_nohop', 'lm', None, 0)]
        for name, fitter, pr, hops, *st in arms:
            r = _arm(P, plane, cal, fitter, pr, hops, *st)
            if r is not None:
                row.update({f'{name}_{k}': v for k, v in r.items()})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', nargs='*', default=['det3', 'det2', 'det4', 'det6', 'det7'])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--free', action='store_true')
    ap.add_argument('--grid', action='store_true', help='arms nm and lm_grid only (LM_START=grid, no hops)')
    ap.add_argument('--out', default='/media/dylan/data/x17/cosmic_bench/model_audit/lm_ab')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for det in a.dets:
        C = pickle.load(open(os.path.join(VL, det, 'cache.pkl'), 'rb'))
        eids = sorted(C)[:a.n]
        del C
        t = time.time()
        with Pool(a.jobs, initializer=_init, initargs=(det, a.free, a.grid)) as pool:
            rows = [r for rr in pool.imap_unordered(one, eids, chunksize=2) for r in rr]
        tag = ('_free' if a.free else '') + ('_grid' if a.grid else '')
        fn = os.path.join(a.out, f'lm_ab_rows_{det}{tag}.pkl')
        pickle.dump(rows, open(fn, 'wb'))
        print(f'{det}: {len(rows)} plane rows from {len(eids)} events in {time.time() - t:.0f} s -> {fn}', flush=True)


if __name__ == '__main__':
    main()
