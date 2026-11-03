#!/usr/bin/env python3
"""sps_patches.py -- is region R (the Paper I chamber-E resolution patch) representative?  The SAME code path as the quoted
numbers (rerun/sps_region/region_analysis.py: align, resolution(), the common f of region.json) on a few other patches of
similar size (10-11 x 24 mm) in live X bands, run 63.  X from the p_mid estimator, Y from p0 (as in the paper).  Does not
change any quoted number.   WORK_DATA/sps_region/tel_run63.parquet (EOS x17rerun/sps_region/).
  python sps_patches.py [--out results/sps_patches.json]"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sps_region'))
import region_analysis as ra  # noqa: E402
from common import load_tel, align  # noqa: E402

PATCHES = {
    'R':  dict(x=(150.0, 160.0), y=(222.0, 246.0)),     # the quoted region
    'R2': dict(x=(202.0, 213.0), y=(222.0, 246.0)),     # band 200-215, same Y (the sps_region cross-check)
    'R3': dict(x=(150.0, 160.0), y=(198.0, 222.0)),     # R's band, lower Y
    'R4': dict(x=(202.0, 212.0), y=(198.0, 222.0)),     # band 200-215, lower Y
    'R5': dict(x=(181.0, 191.0), y=(222.0, 246.0)),     # band 178-193 (lower efficiency), same Y
    'R6': dict(x=(181.0, 191.0), y=(198.0, 222.0)),     # band 178-193, lower Y
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'sps_patches.json'))
    a_ = ap.parse_args()
    reg = json.load(open(os.path.join(HERE, '..', 'sps_region', 'results', 'region.json')))
    f, f_rng = reg['f_common']['f'], tuple(reg['f_common']['range'])
    d = load_tel('run63'); a = align(d)
    out = dict(f=f, f_range=f_rng, patches={})
    for nm, r in PATCHES.items():
        row = dict(x=r['x'], y=r['y'])
        for est in ('p0', 'p_mid'):
            o = ra.resolution(d, a, est, f, f_rng, r)
            row[est] = {k: o[k] for k in ('n', 'n_region')} if 'X' not in o else dict(
                n=o['n'], n_region=o['n_region'],
                X=dict(sigma_det4=o['X']['sigma_det4'], stat=o['X']['stat'], sys_range=o['X']['sys_range'], sigma_res=o['X']['sigma_res'],
                       zone=o['X']['zone'], zone_frac=o['X']['zone_frac'], upper_b2f2=o['X']['res_b2f2'][0]),
                Y=dict(sigma_det4=o['Y']['sigma_det4'], stat=o['Y']['stat'], sys_range=o['Y']['sys_range'], sigma_res=o['Y']['sigma_res'],
                       zone=o['Y']['zone'], zone_frac=o['Y']['zone_frac'], upper_b2=o['Y']['res_b2'][0]))
        out['patches'][nm] = row
        if 'X' in row['p_mid']:
            X, Y = row['p_mid']['X'], row['p0']['Y']
            print(f"{nm:3s} x {r['x']} y {r['y']} n {row['p0']['n']:5d}: X(p_mid) {1e3*X['sigma_det4']:.0f}+-{1e3*X['stat']:.0f} "
                  f"(res {1e3*X['sigma_res']:.0f}, zone {X['zone']} {X['zone_frac']:.2f}) | Y(p0) {1e3*Y['sigma_det4']:.0f}+-{1e3*Y['stat']:.0f} "
                  f"(res {1e3*Y['sigma_res']:.0f}, zone {Y['zone']} {Y['zone_frac']:.2f})", flush=True)
        else:
            print(nm, 'too few tracks', row['p0'])
    os.makedirs(os.path.dirname(a_.out), exist_ok=True)
    json.dump(out, open(a_.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
