#!/usr/bin/env python3
"""
18_ladder_bench.py -- score the ladder-kernel arms against production on the
bench, the only judge that counts (KERNEL_ARMS doc §35: chi2 alone has bought
-23 % with zero geometry gain before).

Each arm is scored the same way and cold:
  * ref-pinned chi2 on train and held-out, cold-started for every arm (a
    warm-started refit flatters itself by ~20 %),
  * its OWN absolute-t0 table, re-measured under its own kernel -- a table
    measured with a different tau_s puts the pulse somewhere else and the
    sigma = 5 ns prior then pulls the fit to the wrong place,
  * a free (p0, w, t0) fit on held-out events against the M3 reference, giving
    sigma_theta and the angle-compression slope,
  * a PAIRED bootstrap over the held-out events for the differences, because
    the arms share events and the unpaired errors are ~3x too big.

The kernel FORM lives on the bundle (model.SHARE_MODE is read once, in
use_calibration), so ladder arms need their own share_mode='lp' bundle; each
arm therefore gets its own worker pool.

    ../.venv/bin/python mx_june_wft/18_ladder_bench.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/ladder_bench.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

T0_SIGMA = 5.0
N_TRAIN, N_HELD, NBOOT = 180, 220, 2000

#: the beam-pinned DELAY arm from 2026-08-17 -- same c1/c2 the H4 beam measured,
#: but in the shipped kernel form.  Kept as the middle rung: it shows how much
#: of the gain is the pinning and how much is the FORM.
BEAM_DELAY = dict(c1=0.281, c2=0.111, kY=1.28573, tau_s=79.35,
                  sigma_s=147.72001, sigma_p0=0.25365, Dp=0.0089)

_EV = None


def _init(cache, bundle):
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))


def _geo(payload):
    eid, hyper, v, t0abs = payload
    from wft import model as wm
    ev = _EV[eid]
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
            r = wm.fit_plane_raw(P, plane, p0r, tr * v * 1e-3, t0p, hyper=hyper,
                                 t0_prior=(t0p, T0_SIGMA))
        except Exception:
            continue
        out[plane] = (float(tr), float(r['w'] * 1e3 / v), float(p0r),
                      float(r['p0']), float(r['chi2'] / max(r['dof'], 1)))
    return eid, out


def rsig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(0.7413 * (np.percentile(x, 75) - np.percentile(x, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--jobs', type=int, default=9)
    ap.add_argument('--bundle', default='calib_bundle_lp2_t0p',
                    help='the production bundle to score against; the frozen '
                         'MPGD26 set is calib_bundle_lp2_t0p for det3 and '
                         'calib_bundle_lp for det2/4/6/7')
    ap.add_argument('--skip-beam-delay', action='store_true',
                    help='the beam_delay arm carries det3-specific numbers')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    out_dir = cfg.out_dir('wft', 'kernel_arms')
    delay_bundle = os.path.join(cfg.OUT_BASE, 'wft', args.bundle)
    lp_bundle = os.path.join(out_dir, 'ladder_provisional_bundle')
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')

    cal = CalibrationBundle.load(delay_bundle)
    v = float(cal.v_drift)
    prod = {k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'}
    with open(cache, 'rb') as f:
        events = pickle.load(f)
    eids = sorted(events)
    train, held = eids[:N_TRAIN], eids[N_TRAIN:][:N_HELD]

    arms = {'production': (delay_bundle, prod)}
    if not args.skip_beam_delay:
        arms['beam_delay'] = (delay_bundle, dict(BEAM_DELAY, share_lp=1.0))
    lad_path = os.path.join(out_dir, 'ladder_recal.json')
    if os.path.exists(lad_path):
        for name, rec in json.load(open(lad_path)).items():
            arms[name] = (lp_bundle,
                          {k: float(q) for k, q in rec['hyper'].items()
                           if not k.startswith('c2_effective')})
    # the c2-slaved arms live in the SHIPPED kernel form, so they score on the
    # production bundle -- only the hyper dict changes
    rat_path = os.path.join(out_dir, 'ratio_recal.json')
    if os.path.exists(rat_path):
        for name, rec in json.load(open(rat_path)).items():
            arms[name] = (delay_bundle,
                          {k: float(q) for k, q in rec['hyper'].items()
                           if k != 'c2_implied'})

    rows, resid = {}, {}
    for name, (bundle, h) in arms.items():
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=wc._init_hyper,
                                 initargs=(cache, bundle)) as cpool, \
                ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                    initargs=(cache, bundle)) as gpool:
            def cold(ids):
                return sum(c for _e, c, _t in cpool.map(
                    wc._event_chi2, [(e, h, v, {}) for e in ids], chunksize=6))
            c_tr, c_he = cold(train), cold(held)
            t0abs, _ = wc.measure_t0_abs({e: events[e] for e in train},
                                         bundle, h, v)
            got = {'x': [], 'y': []}
            keys = {'x': [], 'y': []}
            for e, o in gpool.map(_geo, [(e, h, v, t0abs) for e in held],
                                  chunksize=6):
                for p, tup in o.items():
                    got[p].append(tup)
                    keys[p].append(e)
        g = {}
        for p in ('x', 'y'):
            a = np.array(got[p], float)
            tr, tf, p0r, p0f, cd = a.T
            d = tf - tr
            k = np.abs(d) < 0.15
            g[p] = dict(n=int(k.sum()),
                        sig_theta=float(np.degrees(np.arctan(rsig(d[k])))),
                        slope=float(np.polyfit(tr[k], tf[k], 1)[0]),
                        bias=float(np.median(d[k])),
                        sig_p0=rsig((p0f - p0r)[k]),
                        chi2dof=float(np.median(cd[k])),
                        out=float(1 - k.mean()))
        resid[name] = {p: (np.array(keys[p]), np.array(got[p], float))
                       for p in ('x', 'y')}
        rows[name] = dict(hyper=h, bundle=os.path.basename(bundle),
                          chi2_train_cold=c_tr, chi2_held_cold=c_he, geo=g)
        print(f'{name:13} chi2 train {c_tr:.4e} held {c_he:.4e}  '
              f'({time.time() - t0:.0f} s)')
        for p in ('x', 'y'):
            q = g[p]
            print(f'   {p}: sigma_theta {q["sig_theta"]:.3f} deg  '
                  f'slope {q["slope"]:.4f}  bias {q["bias"]:+.4f}  '
                  f'sig_p0 {q["sig_p0"]:.3f} mm  chi2/dof {q["chi2dof"]:.1f}  '
                  f'out {100 * q["out"]:.1f} %', flush=True)

    # ---- paired bootstrap, on the events every arm reconstructed ----------
    names = list(arms)
    rng = np.random.default_rng(20260818)
    stat = {}
    for p in ('x', 'y'):
        common = set(resid[names[0]][p][0])
        for n in names[1:]:
            common &= set(resid[n][p][0])
        common = np.array(sorted(common))
        keep = np.ones(len(common), bool)
        pick = {}
        for n in names:
            ids, arr = resid[n][p]
            idx = {e: i for i, e in enumerate(ids)}
            a = arr[[idx[e] for e in common]]
            pick[n] = a
            keep &= np.abs(a[:, 1] - a[:, 0]) < 0.15
        nk = int(keep.sum())
        bi = rng.integers(0, nk, size=(NBOOT, nk))
        for n in names:
            a = pick[n][keep]
            tr, tf = a[:, 0], a[:, 1]
            d = tf - tr
            stat[(n, p)] = (
                np.array([np.degrees(np.arctan(rsig(d[i]))) for i in bi]),
                np.array([np.polyfit(tr[i], tf[i], 1)[0] for i in bi]))
        rows.setdefault('_paired', {})[p] = dict(n_common=int(nk))

    print('\nPaired difference vs production (negative sigma = BETTER):')
    print(f'{"arm":13}{"plane":6}{"d sigma_theta [deg]":>28}{"d slope":>22}')
    for n in names:
        if n == 'production':
            continue
        for p in ('x', 'y'):
            ds = stat[(n, p)][0] - stat[('production', p)][0]
            dl = stat[(n, p)][1] - stat[('production', p)][1]
            print(f'{n:13}{p:6}{ds.mean():+10.3f} +- {ds.std():.3f} '
                  f'({abs(ds.mean()) / max(ds.std(), 1e-9):4.1f}s)'
                  f'{dl.mean():+10.4f} +- {dl.std():.4f} '
                  f'({abs(dl.mean()) / max(dl.std(), 1e-9):4.1f}s)')
            rows[n].setdefault('vs_production', {})[p] = dict(
                d_sig=float(ds.mean()), d_sig_err=float(ds.std()),
                d_slope=float(dl.mean()), d_slope_err=float(dl.std()))

    path = os.path.join(out_dir, 'ladder_bench.json')
    with open(path, 'w') as f:
        json.dump(rows, f, indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
