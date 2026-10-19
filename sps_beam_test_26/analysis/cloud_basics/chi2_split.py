#!/usr/bin/env python3
"""chi2_split.py -- where does the RC kernel's worse X chi2 come from?  (FINDINGS §12: X chi2/dof worse than
production on every chamber, det3 58 -> 79, det2 110 -> 189.)

plane_bench.py's held-out ref-start fit, production vs one RC arm, keeping per event and plane: chi2, dof,
the number of censored (saturated) samples, the window maximum, |tan_ref|, and the per-strip chi2 by
offset from the cluster centre.  Then the chi2 difference is split by saturated / unsaturated events,
by inclination and by strip offset.

    chi2_split.py <det> [--max-held 1000]   -> results/chi2_split_<det>.json
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(HERE, '..', 'plane_ratio')]
import plane_bench as PB                                     # noqa: E402

A = '/media/dylan/data/x17/cosmic_bench/Analysis'
ARMS = '/home/dylan/x17/cosmic_bench/cloud_basics/arms'
SETS = {'det3': ('mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3', 'calib_bundle_r06', 'calib'),
        'det2': ('mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 'calib_bundle_r06', 'big'),
        'det7': ('mx17_det6_det7_overnight_6-26-26/long_run/mx17_7', 'calib_bundle_r06', 'big'),
        'det4': ('mx17_det4_day_6-24-26/long_run/mx17_4', 'calib_bundle_lp_t0p', 'calib'),
        'det6': ('mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 'calib_bundle_lp', 'big')}
OFF = 6
PER_EVENT = '/home/dylan/x17/cosmic_bench/cloud_basics/chi2_split'     # per-event fits (MBs): outside the repo


def geo(payload):
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
        # chi2 and signed residual (data - model, in noise units) by strip offset from the charge centre,
        # censored samples left out
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        model = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], hyper)
        z = (Wp - model) / noise[:, None]
        z[sat] = np.nan
        c = int(np.argmax(Wp.sum(1)))
        by = {}
        for o in range(-OFF, OFF + 1):
            if 0 <= c + o < len(Wp):
                zz = z[c + o][np.isfinite(z[c + o])]
                by[o] = (float((zz ** 2).sum()), float(zz.sum()), int(len(zz)))
        out[plane] = dict(tr=float(tr), tf=float(r['w'] * 1e3 / v), chi2=float(r['chi2']), dof=int(r['dof']),
                          nsat=int(sat.sum()), wmax=float(Wp.max()), c=c, n=len(Wp), by=by)
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('det')
    ap.add_argument('--max-held', type=int, default=1000)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--arms', default='rcm', help='comma list of arm tags (arm_<det>_<tag>.json); production first')
    a = ap.parse_args()
    d, bun, train = SETS[a.det]
    W = f'{A}/{d}/wft'
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
    arms = {'production': {k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'}}
    for t in a.arms.split(','):
        arms[t] = {k: float(q) for k, q in json.load(open(f'{ARMS}/arm_{a.det}_{t}.json'))['hyper'].items()}
    res = {}
    for name, h in arms.items():
        t0abs, _ = wc.measure_t0_abs(trainset, f'{W}/{bun}', h, v)
        with ProcessPoolExecutor(a.jobs, initializer=PB._init, initargs=(cache, f'{W}/{bun}', 0.0)) as pool:
            res[name] = {int(e): o for e, o in pool.map(geo, [(e, h, v, t0abs) for e in held], chunksize=8)}
        print(name, 'done', flush=True)
    rows = {}
    names = list(arms)
    for p in ('x', 'y'):
        ids = [e for e in held if all(p in res[n].get(e, {}) for n in names)]
        P = [res['production'][e][p] for e in ids]
        C = {n: np.array([res[n][e][p]['chi2'] / res[n][e][p]['dof'] for e in ids]) for n in names}
        nsat = np.array([q['nsat'] for q in P]); wmax = np.array([q['wmax'] for q in P])
        at = np.abs([q['tr'] for q in P])
        cls = dict(all=np.ones(len(ids), bool), unsat=nsat == 0, sat=nsat > 0,
                   head=at < 0.05, incl=at >= 0.2,
                   unsat_head=(nsat == 0) & (at < 0.05), unsat_incl=(nsat == 0) & (at >= 0.2),
                   faint=wmax < 1500)
        rows[p] = {k: dict(n=int(m.sum()), **{n: float(np.median(C[n][m])) for n in names})
                   for k, m in cls.items() if m.sum() >= 20}
        print(f'{a.det} {p}: ' + '  '.join(f'{k} n{q["n"]} ' + '/'.join(f'{q[n]:.0f}' for n in names)
                                           for k, q in rows[p].items()), flush=True)
    tag = a.det + ('' if a.arms == 'rcm' else '_' + a.arms.replace(',', '_'))
    json.dump(dict(rows=rows), open(f'{HERE}/results/chi2_split_{tag}.json', 'w'), indent=1, default=float)
    os.makedirs(PER_EVENT, exist_ok=True)
    json.dump({n: {str(e): o for e, o in r.items()} for n, r in res.items()},
              open(f'{PER_EVENT}/chi2_split_{tag}_events.json', 'w'), default=float)


if __name__ == '__main__':
    main()
