#!/usr/bin/env python3
"""sat_residual.py -- what does real DREAM saturation do beyond clipping?  (FINDINGS §33: the data's
saturation bias is 2-6x a hard-clip toy's; chi2_split.py: saturated events carry 4-6x the chi2/dof of
unsaturated ones in every kernel, with the clipped samples already censored.)

Held-out bench events with at least one sample >= SAT, fitted ref-start with one arm (wft.model, censored
as in production).  Residuals (data - model) / noise are stacked against the sample offset from the end
of the clipped run, k = sample - last_saturated_sample, for
  * the saturated strip itself (k > 0: the recovery),
  * its +-1 and +-2 neighbours (same k axis),
and, as a control, the same strips/offsets of the event's peak in unsaturated events (k from the peak).
Also the length of the clipped run and the samples just below the ceiling (onset).

    sat_residual.py <det> [--arm rcms25] [--max-held 1500]  -> results/sat_residual_<det>.json
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
import chi2_split as CS                                      # noqa: E402
PB = CS.PB

K = np.arange(-6, 13)


def one(payload):
    eid, hyper, v, t0abs = payload
    from wft import model as wm
    ev = PB._EV[eid]
    out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        ft = ev[f'ftst_{plane}']
        if ft not in t0abs[plane]:
            continue
        t0p = t0abs[plane][ft]
        p0r, tr = ev[f'ref_mesh_{plane}'], ev[f'tan_{plane}']
        try:
            r = wm.fit_plane_raw(P, plane, p0r, tr * v * 1e-3, t0p, hyper=hyper, t0_prior=(t0p, PB.T0_SIGMA))
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        model = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], hyper)
        z = (Wp - model) / noise[:, None]
        rel = (Wp - model) / np.maximum(model.max(), 1.0)          # in units of the event's model peak
        rows = []
        if sat.any():
            for c in np.unique(np.where(sat)[0]):
                ks = np.where(sat[c])[0]
                last, first = ks.max(), ks.min()
                for o in (-2, -1, 0, 1, 2):
                    if 0 <= c + o < len(Wp):
                        for k in K:
                            j = last + k
                            if 0 <= j < Wp.shape[1] and not sat[c + o, j]:
                                rows.append((1, o, int(k), float(z[c + o, j]), float(rel[c + o, j]), int(last - first + 1)))
        else:
            c, j0 = np.unravel_index(np.argmax(Wp), Wp.shape)
            if Wp[c, j0] > 1500:
                for o in (-2, -1, 0, 1, 2):
                    if 0 <= c + o < len(Wp):
                        for k in K:
                            j = j0 + k
                            if 0 <= j < Wp.shape[1]:
                                rows.append((0, o, int(k), float(z[c + o, j]), float(rel[c + o, j]), 0))
        out[plane] = rows
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('det')
    ap.add_argument('--arm', default='rcms25')
    ap.add_argument('--max-held', type=int, default=1500)
    ap.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    d, bun, train = CS.SETS[a.det]
    W = f'{CS.A}/{d}/wft'
    cache = f'{W}/plane_ratio/big_cache_3000.pkl'
    tcache = f'{W}/calib_work/calib_cache.pkl' if train == 'calib' else cache
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle
    os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')
    cal = CalibrationBundle.load(f'{W}/{bun}')
    v = float(cal.v_drift)
    tev = pickle.load(open(tcache, 'rb'))
    train_ids = sorted(tev)[:180]
    trainset = {e: tev[e] for e in train_ids}
    big = pickle.load(open(cache, 'rb'))
    held = [e for e in sorted(big) if e not in set(train_ids)][:a.max_held]
    del big
    h = ({k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'} if a.arm == 'production' else
         {k: float(q) for k, q in json.load(open(f'{CS.ARMS}/arm_{a.det}_{a.arm}.json'))['hyper'].items()})
    t0abs, _ = wc.measure_t0_abs(trainset, f'{W}/{bun}', h, v)
    rows = {'x': [], 'y': []}
    with ProcessPoolExecutor(a.jobs, initializer=PB._init, initargs=(cache, f'{W}/{bun}', 0.0)) as pool:
        for e, o in pool.map(one, [(e, h, v, t0abs) for e in held], chunksize=8):
            for p, rr in o.items():
                rows[p].extend(rr)
    res = {}
    for p in ('x', 'y'):
        R = np.array(rows[p])
        if not len(R):
            continue
        res[p] = {}
        for s, lab in ((1, 'sat'), (0, 'unsat_bright')):
            res[p][lab] = {}
            for o in (-2, -1, 0, 1, 2):
                line = []
                for k in K:
                    m = (R[:, 0] == s) & (R[:, 1] == o) & (R[:, 2] == k)
                    line.append(dict(k=int(k), n=int(m.sum()), z=float(np.mean(R[m, 3])) if m.sum() > 10 else None,
                                     rel=float(np.mean(R[m, 4])) if m.sum() > 10 else None))
                res[p][lab][str(o)] = line
                print(f'{a.det} {p} {lab:12s} o {o:+d}: ' + ' '.join(
                    f'{q["k"]:+d}:{q["rel"]:+.3f}' for q in line if q['rel'] is not None and q['k'] in (-3, -1, 1, 2, 3, 5, 8, 12)), flush=True)
        Rs = R[(R[:, 0] == 1) & (R[:, 1] == 0) & (R[:, 2] == 1)]
        res[p]['clip_len_quantiles'] = [float(q) for q in np.quantile(Rs[:, 5], [0.1, 0.5, 0.9])] if len(Rs) else None
    json.dump(dict(arm=a.arm, K=K.tolist(), res=res), open(f'{HERE}/results/sat_residual_{a.det}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
