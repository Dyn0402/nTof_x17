#!/usr/bin/env python3
"""
apply_w0.py -- re-apply a bundle's w0/kw to an existing reco table.

w0/kw enter the reconstruction ONLY as a post-hoc map from the fitted
transverse speed w to an angle (wft/reco.py, `tan = (w*1e3 - w0) / (kw * v)`).
Nothing upstream of that sees them, so re-measuring w0/kw does NOT require a
second reco pass -- the stored `<plane>_w` column is untouched by them and
this script recomputes everything downstream of it:

    <plane>_tan_theta, <plane>_theta_deg, <plane>_slope_reliable

`slope_reliable` is included on purpose: it is a cut on |tan|, so it moves
with w0/kw, and the sigma-theta *population* moves with it (FREEZE §7).

    ../../.venv/bin/python mx_june_wft/bench/apply_w0.py sat_det3 \
        --events events_r06.parquet --bundle .../calib_bundle_r06 [--write]

Without --write it prints what would change and writes nothing.
"""
import argparse
import json
import os
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
    ap.add_argument('--events', default='events.parquet')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default=None, help='default: in place')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft import reco as wr

    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    ev = args.events if os.path.isabs(args.events) \
        else os.path.join(W, args.events)
    df = pd.read_parquet(ev)
    cal = CalibrationBundle.load(args.bundle)
    v = cal.v_drift
    print(f'{ev}\n  {len(df)} events, bundle {os.path.basename(args.bundle)}: '
          f'v={v:.2f}, w0={ {k: round(x, 4) for k, x in cal.w0.items()} }, '
          f'kw={ {k: round(x, 4) for k, x in cal.kw.items()} }')

    for plane in ('x', 'y'):
        w = df[f'{plane}_w'].to_numpy(float)
        tan = ((w * 1e3 - cal.w0.get(plane, 0.0))
               / (cal.kw.get(plane, 1.0) * v))
        rel = np.abs(tan) >= wr.TAN_MIN_SLOPE
        ok = df[f'{plane}_ok'].to_numpy(bool)
        d = np.degrees(np.arctan(tan)) - df[f'{plane}_theta_deg'].to_numpy(float)
        n_flip = int((rel != df[f'{plane}_slope_reliable'].to_numpy(bool))[ok].sum())
        print(f'  {plane}: median dtheta {np.nanmedian(d[ok]):+.4f} deg, '
              f'max |dtheta| {np.nanmax(np.abs(d[ok])):.4f}, '
              f'slope_reliable {df[f"{plane}_slope_reliable"][ok].mean():.4f} '
              f'-> {rel[ok].mean():.4f} ({n_flip} events flip)')
        df[f'{plane}_tan_theta'] = tan
        df[f'{plane}_theta_deg'] = np.degrees(np.arctan(tan))
        df[f'{plane}_slope_reliable'] = rel

    if not args.write:
        print('  (dry run -- pass --write to save)')
        return
    out = args.out or ev
    df.to_parquet(out, index=False)
    print(f'  wrote {out}')

    meta = os.path.splitext(out)[0] + '.meta.json'
    alt = out.replace('.parquet', '.meta.json')
    for m in (meta, alt):
        if os.path.exists(m):
            j = json.load(open(m))
            j['angle_constants'] = dict(
                applied=True, w0=dict(cal.w0), kw=dict(cal.kw),
                reapplied_from=os.path.basename(args.bundle),
                note='w0/kw re-applied post-hoc by bench/apply_w0.py; the fits '
                     'themselves are unchanged')
            json.dump(j, open(m, 'w'), indent=1)
            print(f'  stamped {m}')
            break


if __name__ == '__main__':
    main()
