#!/usr/bin/env python3
"""Joint x-y fit (one shared depth profile, per-plane scale, t0 tied through dt_xy) as a pairing statistic.

Right pairs: an event's own x and y windows. Wrong pairs: event i's x with event j's y (next event with fits).
Both seeded from the production per-plane fits. The statistic is dchi2 = chi2_joint - chi2_separate, where
chi2_separate solves the two profiles independently under the same weighting (so dchi2 >= 0 is the cost of sharing).
"""
import os, sys, pickle, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algo_review as ar

_ROWS = None


def _init2(rows_path):
    global _ROWS
    ar._init(13)
    from wft import model as wm
    rows = pickle.load(open(rows_path, 'rb'))
    ev = {}
    for r in rows:
        ev.setdefault(r['eid'], {})[r['plane']] = r
    # dt_xy per ftst-difference class from the production t0 medians of this sample (the bundle's are stale, F5)
    cls = {}
    for e in ev.values():
        if 'x' in e and 'y' in e and 'prod_t0' in e['x'] and 'prod_t0' in e['y']:
            cls.setdefault(int(e['x']['ftst']) - int(e['y']['ftst']), []).append(e['x']['prod_t0'] - e['y']['prod_t0'])
    wm.DT_XY = {k: float(np.median(v)) for k, v in cls.items()}
    _ROWS = ev


def _chi_sep(evx, evy, fd, p0x, wx, p0y, wy, t0, hyper, t0y=None):
    """chi2 with two independent profiles, same weighting/censoring as fit_joint (plain NNLS on live rows).
    ``t0y`` None: y at the tied t0 - dt_xy; else y at its own t0y (the untied reference)."""
    from wft import model as wm
    from scipy.optimize import nnls
    dt = wm.DT_XY.get(int(fd), -18.8)
    tot = 0.0
    for P, plane, p0, w, tt in ((evx, 'x', p0x, wx, t0), (evy, 'y', p0y, wy, (t0 - dt) if t0y is None else t0y)):
        W, n, pos, sat = wm.prep_plane(P, plane)
        M = wm.build_matrix(plane, pos, p0, w, tt, hyper)
        ok = ~sat.reshape(-1)
        A = (M * np.repeat(1.0 / n, wm.NSAMP)[:, None])[ok]
        y = (W / n[:, None]).reshape(-1)[ok]
        q, rn = nnls(A, y, maxiter=50 * wm.K)
        tot += rn * rn
    return tot


def one(pair):
    from wft import model as wm
    ei, ej = pair
    ex, ey = _ROWS[ei]['x'], _ROWS[ej]['y']
    Px, Py = ar._EV[ei]['x'], ar._EV[ej]['y']
    fd = int(ex['ftst']) - int(ey['ftst'])
    hyper = dict(wm.CAL.hyper)
    for P in (Px, Py):
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
    t = time.perf_counter()
    try:
        r = wm.fit_joint(Px, Py, fd, ex['prod_p0'], ex['prod_w'], ey['prod_p0'], ey['prod_w'], ex['prod_t0'], hyper=hyper)
    except Exception:
        return None
    sep0 = _chi_sep(Px, Py, fd, ex['prod_p0'], ex['prod_w'], ey['prod_p0'], ey['prod_w'], ex['prod_t0'], hyper,
                    t0y=ey['prod_t0'])          # each plane at its own production (p0, w, t0): the untied reference
    sep1 = _chi_sep(Px, Py, fd, r['p0x'], r['wx'], r['p0y'], r['wy'], r['t0'], hyper)
    return dict(ei=ei, ej=ej, right=ei == ej, fd=fd, time=time.perf_counter() - t,
                chi2_joint=r['chi2'], dof=r['dof'], chi2_sep_at_prod=sep0, chi2_sep_at_joint=sep1,
                p0x=r['p0x'], wx=r['wx'], p0y=r['p0y'], wy=r['wy'], t0=r['t0'], alpha=r['alpha'],
                tan_ref_x=ex['tan_ref'], tan_ref_y=ey['tan_ref'],
                prod_wx=ex['prod_w'], prod_wy=ey['prod_w'], prod_p0x=ex['prod_p0'], prod_p0y=ey['prod_p0'],
                prod_chi2x=ex['prod_chi2'], prod_chi2y=ey['prod_chi2'], prod_dofx=ex['prod_dof'], prod_dofy=ey['prod_dof'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=os.path.join(os.path.dirname(__file__), 'rows800.pkl'))
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'joint_rows.pkl'))
    a = ap.parse_args()
    rows = pickle.load(open(a.rows, 'rb'))
    ev = {}
    for r in rows:
        ev.setdefault(r['eid'], {})[r['plane']] = r
    good = sorted(e for e, d in ev.items() if 'x' in d and 'y' in d and 'prod_w' in d['x'] and 'prod_w' in d['y'])[:a.n]
    pairs = [(e, e) for e in good] + [(good[i], good[(i + 1) % len(good)]) for i in range(len(good))]
    t = time.time()
    with Pool(a.jobs, initializer=_init2, initargs=(a.rows,)) as pool:
        out = [r for r in pool.imap_unordered(one, pairs, chunksize=2) if r is not None]
    print(f'{len(out)} joint fits in {time.time() - t:.0f} s')
    pickle.dump(out, open(a.out, 'wb'))


if __name__ == '__main__':
    main()
