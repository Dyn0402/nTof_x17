#!/usr/bin/env python3
"""resid_map.py --cache c.pkl --bundle b --out o.npz [--n 400] [--start 180] [--jobs 8]

Residual anatomy of the waveform fit: held-out events through the pivot fitter (rerun/seedfit/fitters.py), then the
data - model residual in noise units, z = (W - M) / noise, on a grid of strip offset from the charge-maximum strip
(-OFF..OFF, in POSITION order, so the sign is the detector axis) x sample offset from that strip's peak sample
(-TB..TA).  Accumulated per class: view x {head-on |tan_ref| < 0.1, inclined |tan_ref| > 0.3} x {unsaturated,
saturated}; inclined events are folded so tan_ref > 0 (offset and the sign of the residual's spatial structure flip
with the track); head-on events keep the detector axis, so a board-fixed asymmetry shows.  Saturated samples are left out.  Also per event: chi2, dof, window max, tan_ref, fitted w.

Sums kept: n, sum z, sum z^2 and sum of the data and model in ADC (to see whether a cell is a shape or a scale miss).
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
OFF, TB, TA = 5, 8, 16
_EV = None


def _init(cache, bundle, k):
    global _EV
    sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'seedfit')]
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    wm.use_calibration(CalibrationBundle.load(bundle)); wm.set_depth_bins(k); wm.COL_GUARD = 0.01


def one(eid):
    from wft import model as wm
    import fitters as F
    cal = wm.CAL; ev = _EV[eid]; out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        ft = ev.get(f'ftst_{plane}')
        t0p = (cal.t0_abs or {}).get(plane, {}).get(str(ft), (cal.t0_abs or {}).get(plane, {}).get(ft))
        tr = float(ev[f'tan_{plane}'])
        try:
            r = F.fit_pivot(P, plane, t0p)
        except Exception:
            continue
        W, noise, pos, sat = wm.prep_plane(P, plane)
        chi, q = wm.chi2_plane(plane, W, noise, pos, sat, r['p0'], r['w'], r['t0'], wm.HYPER, snap_t0=False)
        if q is None:
            continue
        M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], q, wm.HYPER)
        o = np.argsort(pos); W, M, noise, sat, pos = W[o], M[o], noise[o], sat[o], pos[o]
        if tr <= -0.1:                               # fold inclined tracks to tan > 0; head-on keep the board axis
            W, M, noise, sat, pos = W[::-1], M[::-1], noise[::-1], sat[::-1], -pos[::-1]
        c = int(np.argmax(W.sum(1)))
        tp = int(np.argmax(W[c]))
        z = (W - M) / noise[:, None]
        cells = []
        for so in range(-OFF, OFF + 1):
            i = c + so
            if not 0 <= i < len(W):
                continue
            for to in range(-TB, TA + 1):
                t = tp + to
                if 0 <= t < W.shape[1] and not sat[i, t]:
                    cells.append((so + OFF, to + TB, float(z[i, t]), float(W[i, t]), float(M[i, t])))
        out[plane] = dict(tr=tr, w=float(r['w'] * 1e3), chi2=float(chi), dof=int((~sat).sum()), wmax=float(W.max()),
                          nsat=int(sat.sum()), cells=cells)
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=400); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ids = sorted(pickle.load(open(a.cache, 'rb')))[a.start:a.start + a.n]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k)) as pool:
        res = dict(pool.map(one, ids, chunksize=2))
    shape = (2, 2, 2, 2 * OFF + 1, TB + TA + 1)       # view, class (head-on, inclined), sat, strip, sample
    n, sz, sz2, sw, sm = (np.zeros(shape) for _ in range(5))
    per = []
    for eid, d in res.items():
        for vi, plane in enumerate('xy'):
            if plane not in d:
                continue
            e = d[plane]; at = abs(e['tr'])
            ci = 0 if at < 0.1 else (1 if at > 0.3 else None)
            per.append((int(eid), vi, e['tr'], e['w'], e['chi2'], e['dof'], e['wmax'], e['nsat']))
            if ci is None:
                continue
            si = int(e['nsat'] > 0)
            for so, to, z, w, m in e['cells']:
                idx = (vi, ci, si, so, to)
                n[idx] += 1; sz[idx] += z; sz2[idx] += z * z; sw[idx] += w; sm[idx] += m
    np.savez(a.out, n=n, sz=sz, sz2=sz2, sw=sw, sm=sm, per=np.array(per), off=OFF, tb=TB, ta=TA, bundle=a.bundle)
    print('wrote', a.out, len(per))


if __name__ == '__main__':
    main()
