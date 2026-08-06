#!/usr/bin/env python3
"""
gap_matrix.py — merge the condor sweep and build the data x bundle matrix.

The sweep fits every dataset with every dataset's calibration bundle. Reading
the matrix:

  * DOWN a column (one bundle, different data): the chamber-to-chamber and
    run-to-run differences at fixed calibration — this is the physics.
  * ACROSS a row (one dataset, different bundles): the calibration systematic
    on the absolute column — this is the error bar that
    GAP_STUDY_2026-07-30.md does not currently quote.

    gap_matrix.py [--shards DIR] [--out DIR] [--plane x]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

STAGE = '/home/dylan/x17/cosmic_bench/condor_wft'
BUNDLE_NAME = {
    'sat_det3': 'calib_bundle_lp2',
    'g_det3_wknd': 'calib_bundle_lp',
    'g_det3': 'calib_bundle_lp',
    'o22_long_det2': 'calib_bundle_lp',
    'g_det2': 'calib_bundle_lp',
    'g_det4': 'calib_bundle_lp',
    'g_det6_long': 'calib_bundle_lp',
    'g_det7_long': 'calib_bundle_lp',
}
SHORT = {'sat_det3': 'det3 6-27 sat', 'g_det3_wknd': 'det3 6-28 P2',
         'g_det3': 'det3 6-22 bot', 'o22_long_det2': 'det2 6-22 longer',
         'g_det2': 'det2 6-22 long', 'g_det4': 'det4 6-24',
         'g_det6_long': 'det6 6-26', 'g_det7_long': 'det7 6-26'}


def bundle_path(key):
    """Local bundle dir; falls back to a condor-produced bundle if that is the
    only one (det4/6/7 were calibrated on the grid)."""
    from qa_config import get_config
    p = os.path.join(get_config(key).OUT_BASE, 'wft', BUNDLE_NAME[key])
    if os.path.exists(os.path.join(p, 'bundle.json')):
        return p
    alt = os.path.join(STAGE, 'out', f'bundle_{key}')
    return alt if os.path.exists(os.path.join(alt, 'bundle.json')) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default=os.path.join(STAGE, 'shards'))
    ap.add_argument('--out', default=os.path.join(STAGE, 'merged'))
    ap.add_argument('--plane', default='x')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    labels = sorted({os.path.basename(f).replace('profiles_', '').rsplit('_', 1)[0]
                     for f in os.listdir(args.shards) if f.startswith('profiles_')})
    print(f'{len(labels)} fits to merge')

    rows = []
    for lab in labels:
        data_key, bundle_key = (lab.split('__with__') if '__with__' in lab
                                else (lab, lab))
        out_dir = os.path.join(args.out, lab)
        js = os.path.join(out_dir, 'gap_study.json')
        if not os.path.exists(js):
            cmd = [sys.executable,
                   os.path.join(REPO, 'mx_june_wft', 'bench', 'gap_merge.py'),
                   '--dir', args.shards, '--label', lab,
                   '--bundle', bundle_path(bundle_key), '--out', out_dir]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  {lab}: merge FAILED\n{r.stderr[-400:]}')
                continue
        d = json.load(open(js))
        pl = d['planes'].get(args.plane)
        if not pl:
            continue
        sh = pl['fits'].get('sharp', {})
        rows.append(dict(data=data_key, bundle=bundle_key,
                         n=pl['n_contained'], v_geom=round(pl['v_geom'], 2),
                         T_end=round(sh.get('T_end', np.nan)),
                         gap=round(sh.get('gap_mm', np.nan), 2),
                         err=round(sh.get('gap_err', np.nan), 2)))
        print(f"  {lab:34} gap {rows[-1]['gap']:6.2f} mm  "
              f"(v_geom {rows[-1]['v_geom']:.2f}, n={pl['n_contained']:,})")

    if not rows:
        raise SystemExit('nothing merged')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'gap_matrix.csv'), index=False)

    piv = df.pivot(index='data', columns='bundle', values='gap')
    order = [k for k in SHORT if k in piv.index]
    piv = piv.reindex(index=order, columns=[k for k in SHORT if k in piv.columns])
    named = piv.rename(index=SHORT, columns=SHORT)
    print('\n== charge-visible column [mm], rows = data, cols = calibration bundle ==')
    print(named.to_string(float_format=lambda v: f'{v:6.2f}'))

    print('\n== spread per dataset ACROSS bundles (the calibration systematic) ==')
    for k in piv.index:
        v = piv.loc[k].dropna()
        if len(v) > 1:
            print(f'  {SHORT[k]:20} {v.min():6.2f} - {v.max():6.2f} mm   '
                  f'(spread {v.max()-v.min():.2f}, rms {v.std():.2f})')

    print('\n== chamber contrast per bundle (det2 mean - det3 mean, fixed calib) ==')
    d3 = [k for k in piv.index if k.startswith(('sat_det3', 'g_det3'))
          and k != 'g_det3']          # g_det3 = pathological drift, excluded
    d2 = [k for k in piv.index if 'det2' in k]
    for b in piv.columns:
        a, c = piv.loc[d3, b].dropna(), piv.loc[d2, b].dropna()
        if len(a) and len(c):
            print(f'  bundle {SHORT[b]:20} det3 {a.mean():6.2f}  '
                  f'det2 {c.mean():6.2f}  ->  contrast {c.mean()-a.mean():+.2f} mm')
    with open(os.path.join(args.out, 'gap_matrix.txt'), 'w') as f:
        f.write(named.to_string(float_format=lambda v: f'{v:6.2f}') + '\n')
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
