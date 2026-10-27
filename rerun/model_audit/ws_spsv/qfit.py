#!/usr/bin/env python3
"""qfit.py --cache c.pkl --bundle b --out o.npz [--k 54] [--n 600] [--jobs 4] [--v V] [--scale Dp=0.7]

The audit harness's fit (heldout_arm.py: pivot start, sigma_p0 profile, COL_GUARD 0.01, bench bundle with the dead list
dropped) but keeping each plane's NNLS depth profile q[K], so the column end can be measured on the STACKED profile,
away from the depth-grid end (q_uend alone sits on the grid end: NNLS parks charge in the last bin).
WFT_ZS_CENSOR is read from the environment (1 for ZS caches, and for the RAW cache, whose dropped samples are stored as
exact zeros).  per rows: (eid, view, tan_ref, w[um/ns], chi2, dof, t0, p0, q_uend, q_sum, sigma_p0_used); q: (nrows, K).
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
_EV = None


def _init(cache, bundle, k, v, scale, setv):
    global _EV
    sys.path.insert(0, REPO)
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    if v:
        cal.v_drift = float(v)
    for k_, f_ in (scale or {}).items():
        cal.hyper[k_] = cal.hyper[k_] * float(f_)
    cal.hyper.update(setv or {})
    cal.dead, cal.hot = {}, {}          # bench channel numbering does not apply to FEU 03 at H4
    wm.use_calibration(cal); wm.set_depth_bins(k)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01
    wr.PIVOT_START = True


def one(eid):
    from wft import model as wm, reco as wr
    ev = _EV[eid]; rows = []
    for vi, plane in enumerate('xy'):
        if plane not in ev:
            continue
        try:
            f = wr.fit_plane(ev[plane], plane, wm.CAL)
        except Exception:
            f = None
        if f is None:
            continue
        rows.append(((int(eid), vi, float(ev[f'tan_{plane}']), f.w * 1e3, f.chi2, f.dof, f.t0, f.p0, f.q_uend, f.q_sum,
                      float(getattr(f, '_sigma_p0', np.nan))), np.asarray(f._q, float)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=54); ap.add_argument('--n', type=int, default=600); ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--v', type=float, default=None)
    ap.add_argument('--scale', default='', help='k=f[,k=f] multiplicative hyper factors')
    ap.add_argument('--set', default='', help='k=v[,k=v] hyper overrides; empty value = [] (e.g. sigma_p0_prof_y=)')
    a = ap.parse_args()
    sc = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.scale.split(',') if kv}
    st = {}
    for kv in a.set.split(','):
        if kv:
            k_, v_ = kv.split('='); st[k_] = [] if v_ == '' else float(v_)
    ids = sorted(pickle.load(open(a.cache, 'rb')))[:a.n]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k, a.v, sc, st)) as pool:
        res = [r for rs in pool.map(one, ids, chunksize=4) for r in rs]
    per = np.array([r[0] for r in res], float); q = np.array([r[1] for r in res], float)
    np.savez(a.out, per=per, q=q, k=a.k, args=json.dumps(vars(a)),
             env=json.dumps({k: v for k, v in os.environ.items() if k.startswith('WFT_')}))
    print(a.out, per.shape, flush=True)


if __name__ == '__main__':
    main()
