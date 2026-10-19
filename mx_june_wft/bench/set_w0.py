#!/usr/bin/env python3
"""
set_w0.py — measure the per-plane transverse-speed offset w0 for an existing
reconstruction and (optionally) write it into the calibration bundle.

w0 = median(w_fit*1e3 - v_cal * tan_ref) over M3 tracks with |tan_ref| < 0.3
and a good XY point (|r| < 5 mm). It needs no refit: w0 only changes how w is
converted to an angle. See wft.calibrate.measure_w0 for the from-scratch
calibration path.

    ../../.venv/bin/python mx_june_wft/bench/set_w0.py sat_det3 [--write]
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--write', action='store_true',
                    help='write w0 into the calibration bundle')
    ap.add_argument('--bundle', default=None,
                    help='bundle to read v from / write into '
                         '(default <OUT_BASE>/wft/calib_bundle)')
    ap.add_argument('--events', default=None,
                    help='reco table to measure from, absolute or relative to '
                         '<OUT_BASE>/wft (default events.parquet). Use this to '
                         'measure a tagged arm without touching the live one.')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    ev = args.events or 'events.parquet'
    ev = ev if os.path.isabs(ev) else os.path.join(W, ev)
    if not os.path.exists(ev):
        raise SystemExit(f'no reco table at {ev}')
    print(f'measuring from {ev}')
    p = pd.read_parquet(ev)
    for tname in ('bench_cache.pkl', 'bench_truth.pkl'):
        tpath = os.path.join(W, tname)
        if os.path.exists(tpath):
            break
    else:
        raise SystemExit('no truth table: run bench/build_cache.py '
                         f'{args.run_key} [--truth-only] first')
    c = pickle.load(open(tpath, 'rb'))
    tr = (c['events'] if 'events' in c else None)
    if tr is not None:
        t = pd.DataFrame([{'event_id': e, **d['truth']} for e, d in tr.items()])
    else:
        t = pd.DataFrame([{'event_id': e, **d} for e, d in c['truth'].items()])
    m = p.merge(t, on='event_id')

    bundle = args.bundle or os.path.join(W, 'calib_bundle')
    cal = CalibrationBundle.load(bundle)
    v = cal.v_drift
    good = (m.x_ok & m.y_ok
            & np.isfinite(m.ref_x) & np.isfinite(m.ref_y)
            & (np.hypot(m.x_p0 - m.ref_x, m.y_p0 - m.ref_y) < 5.0))
    w0, kw = {}, {}
    for plane in ('x', 'y'):
        fin = good & np.isfinite(m[f'tan_{plane}'])
        s = fin & (np.abs(m[f'tan_{plane}']) < 0.30)
        d = (m.loc[s, f'{plane}_w'] * 1e3 - v * m.loc[s, f'tan_{plane}'])
        w0[plane] = float(np.median(d))
        s1 = fin & (np.abs(m[f'tan_{plane}']) > 0.10) \
            & (np.abs(m[f'tan_{plane}']) < 0.40)
        if s1.sum() >= 30:
            kw[plane] = float(np.median(
                (m.loc[s1, f'{plane}_w'] * 1e3 - w0[plane])
                / (v * m.loc[s1, f'tan_{plane}'])))
        print(f'{plane}: w0 = {w0[plane]:+.3f} um/ns  '
              f'(n={int(s.sum())}, rsig {1.4826*np.median(np.abs(d-np.median(d))):.2f}) '
              f'-> angle bias if uncorrected {np.degrees(w0[plane]/v):+.2f} deg'
              + (f'   kw = {kw[plane]:.3f}' if plane in kw else ''))
    if args.write:
        cal.w0, cal.kw = w0, kw
        # The staleness this flag warns about is exactly what we just fixed --
        # leave it set and every downstream guard keeps refusing the bundle.
        if cal.provenance.get('w0_kw_stale'):
            cal.provenance = dict(cal.provenance)
            cal.provenance['w0_kw_stale'] = False
            cal.provenance['w0_kw_note'] = (
                f'w0/kw re-measured from {os.path.basename(ev)} '
                f'({args.run_key}), a reco run with THIS bundle')
            print('  cleared the w0_kw_stale flag')
        cal.save(bundle,
                 note=f'w0/kw measured from {os.path.basename(ev)} '
                      f'({args.run_key})')
        print(f'wrote w0/kw into {bundle}')


if __name__ == '__main__':
    main()
