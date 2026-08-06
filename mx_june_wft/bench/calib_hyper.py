#!/usr/bin/env python3
"""
calib_hyper.py — the portable half of `wft.calibrate`: everything downstream of
the corridor cache. No qa_config, no ROOT, no bench data — it needs only
`calib_work/calib_cache.pkl` (2 MB) — so the expensive ref-pinned hyper fit runs
on a condor worker.

Stages (identical calls to wft.calibrate, so the bundle is the same object the
local path produces): templates -> hyper fit -> w0/kw -> dt_xy.

    calib_hyper.py --cache calib_cache.pkl --out calib_bundle_lp \
                   --detector mx17_6 --run-key g_det6_long --share-lp \
                   --fix-v 26.7 [--tmpl-tan-min 0.10 --tmpl-min-amp 250]
                   [--jobs 4] [--train 180]

The corridor cache itself is built locally (it reads decoded waveforms); this
step is the one that costs CPU.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='calib_work/calib_cache.pkl')
    ap.add_argument('--out', required=True, help='bundle directory to write')
    ap.add_argument('--work', default=None,
                    help='scratch dir for the provisional bundle '
                         '(default: alongside --out)')
    ap.add_argument('--detector', default='')
    ap.add_argument('--run-key', default='')
    ap.add_argument('--run', default='')
    ap.add_argument('--sub-run', default='')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--train', type=int, default=180)
    ap.add_argument('--maxiter', type=int, default=130)
    ap.add_argument('--fix-v', type=float, default=None)
    ap.add_argument('--seed-bundle', default=None)
    ap.add_argument('--share-lp', action='store_true')
    ap.add_argument('--ktau-y', type=float, default=1.78)
    ap.add_argument('--tmpl-tan-min', type=float, default=None)
    ap.add_argument('--tmpl-min-amp', type=float, default=None)
    args = ap.parse_args()

    from wft import calibrate as wc
    from wft.calib import CalibrationBundle, HYPER_NAMES

    if args.tmpl_tan_min is not None:
        wc.TEMPLATE_TAN_MIN = float(args.tmpl_tan_min)
    if args.tmpl_min_amp is not None:
        wc.TEMPLATE_MIN_AMP = float(args.tmpl_min_amp)

    events = pickle.load(open(args.cache, 'rb'))
    print(f'[calib] {len(events):,} corridor events', flush=True)

    extra = {'share_lp': 1.0, 'kTauY': float(args.ktau_y)} if args.share_lp else None
    grid, tmpl = wc.measure_templates(events)

    seed = CalibrationBundle.load(args.seed_bundle) if args.seed_bundle else None
    cal = CalibrationBundle(
        hyper=(dict(zip(HYPER_NAMES, wc.HYPER_X0[:7])) if seed is None
               else dict(seed.hyper)),
        v_drift=float(wc.HYPER_X0[7]) if seed is None else seed.v_drift,
        grid=grid, tmpl=tmpl,
        gain={'x': np.ones(512), 'y': np.ones(512)},
        detector=args.detector, run_key=args.run_key,
        conditions=dict(run=args.run, sub_run=args.sub_run))

    work = args.work or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                     'calib_work_condor')
    os.makedirs(work, exist_ok=True)
    prov = os.path.join(work, 'provisional_bundle')
    if extra:
        cal.hyper.update(extra)
    cal.save(prov, note='templates measured (condor)')

    train = sorted(events)[:args.train]
    print(f'[calib] hyper fit on {len(train)} events, {args.jobs} jobs',
          flush=True)
    x0 = None
    if seed is not None:
        x0 = np.array([seed.hyper[k] for k in HYPER_NAMES] + [seed.v_drift])
    elif extra:
        x0 = wc.HYPER_X0_LP.copy()
    if args.fix_v is not None and x0 is not None:
        x0[7] = args.fix_v
    hj = wc.fit_hypers(args.cache, prov, train, jobs=args.jobs,
                       maxiter=args.maxiter, x0=x0, v_fixed=args.fix_v,
                       extra_hyper=extra)
    if args.fix_v is not None:
        hj['v'] = float(args.fix_v)
    cal.hyper = {k: hj[k] for k in HYPER_NAMES}
    if extra:
        cal.hyper.update(extra)
    cal.v_drift = hj['v']
    cal.provenance.update(n_train=hj['n_train'], chi2=hj['chi2'],
                          chi2_init=hj['chi2_init'],
                          fitted='mx_june_wft/bench/calib_hyper.py (condor)',
                          gain_map='unit (not fitted)')
    cal.save(args.out, note='hypers fitted ref-pinned (condor)')

    # corridor w0/kw (pass 1). The production retrofit (bench/set_w0.py) still
    # has to run locally against a reco table -- see HANDOFF_2026-07-30.
    try:
        w0, kw = wc.measure_w0(args.cache, args.out, train, cal.v_drift,
                               jobs=args.jobs)
        cal.w0, cal.kw = w0, kw
    except Exception as e:
        print(f'[calib] w0 stage skipped: {e}')
    try:
        cal.dt_xy = wc.measure_dt_xy(events, args.out, cal.hyper, cal.v_drift)
    except Exception as e:
        print(f'[calib] dt_xy stage skipped: {e}')
    cal.save(args.out, note='hypers + w0/kw + dt_xy (condor)')
    print(cal.summary())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
