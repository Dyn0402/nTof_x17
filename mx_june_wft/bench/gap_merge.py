#!/usr/bin/env python3
"""
gap_merge.py — combine gap_fit.py shards into the drift-gap result.

Reads every profiles_<label>_*.parquet/.csv.gz in a directory, and reproduces
what bench/gap_study.py writes locally:

  * stacked normalised charge-arrival profile of contained tracks, per plane
  * endpoint fits (sharp erfc edge, and edge x attachment)
  * geometric drift speed v_geom = median((w*1e3 - w0) / tan_ref)
  * gap_study.json + event_profiles.parquet, ready for gap_compare.py

    gap_merge.py --dir results/ --label det3_sat --bundle calib_bundle_lp2 \
                 [--out <dir>]
"""
import argparse
import glob
import re
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

DT_NS = 60.0          # depth-bin width of the charge basis
# The basis depth is whatever the fit was run with (gap_fit --k-bins), so it is
# read from the data rather than assumed: a slow chamber needs a deeper basis
# than the default 18 bins or its endpoint falls off the end of the model.
def basis(df):
    cols = sorted((c for c in df.columns if re.fullmatch(r'q\d+', c)),
                  key=lambda c: int(c[1:]))
    return cols, (np.arange(len(cols)) + 0.5) * DT_NS


def sharp(u, A, T, sig):
    return A * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))


def attach(u, A, T, sig, tau):
    return A * np.exp(-u / tau) * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))


def endpoint_fits(u, prof, prof_err):
    u_end = float(u[-1] + 0.5 * DT_NS)          # the basis end
    sel = u < u_end - 30.0
    res = {}
    try:
        p, c = curve_fit(sharp, u[sel], prof[sel], p0=[prof[:5].mean(), 700, 60],
                         sigma=prof_err[sel], absolute_sigma=True, maxfev=40000,
                         bounds=([0, 200, 10],
                                 [np.inf, u_end, max(400.0, 0.4 * u_end)]))
        res['sharp'] = dict(A=p[0], T_end=p[1], sig_e=p[2],
                            T_err=float(np.sqrt(c[1, 1])),
                            chi2=float((((sharp(u[sel], *p) - prof[sel])
                                         / prof_err[sel]) ** 2).sum()))
    except Exception as e:
        res['sharp'] = dict(error=str(e))
    try:
        p, c = curve_fit(attach, u[sel], prof[sel],
                         p0=[prof[:5].mean(), 700, 60, 5000],
                         sigma=prof_err[sel], absolute_sigma=True,
                         bounds=([0, 300, 10, 300], [np.inf, u_end, 300, 1e6]),
                         maxfev=40000)
        res['attach'] = dict(A=p[0], T_end=p[1], sig_e=p[2], tau_att=p[3],
                             T_err=float(np.sqrt(c[1, 1])),
                             chi2=float((((attach(u[sel], *p) - prof[sel])
                                          / prof_err[sel]) ** 2).sum()))
    except Exception as e:
        res['attach'] = dict(error=str(e))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--label', default='*')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--v-geom', type=float, default=None,
                    help='override the geometric drift speed [um/ns]')
    args = ap.parse_args()

    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(args.bundle)
    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    # `[0-9]` matters: a bare `_*` also matches the cross-bundle labels
    # (profiles_<label>__with__<other>_000), and merging fits made with
    # different calibrations into one result would silently mix a 1.7 mm
    # systematic into the answer. The shard index is always numeric.
    files = sorted(glob.glob(os.path.join(args.dir,
                                          f'profiles_{args.label}_[0-9]*.parquet'))
                   + glob.glob(os.path.join(args.dir,
                                            f'profiles_{args.label}_[0-9]*.csv.gz')))
    if not files:
        raise SystemExit(f'no shards in {args.dir}')
    parts = [pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
             for f in files]
    df = pd.concat(parts, ignore_index=True).drop_duplicates(['eid', 'plane'])
    print(f'{len(files)} shards -> {len(df):,} plane-rows, '
          f'{df.eid.nunique():,} events')

    summary = {'label': args.label, 'bundle': args.bundle,
               'v_drift': cal.v_drift, 'kw': dict(cal.kw), 'w0': dict(cal.w0),
               'n_shards': len(files), 'planes': {}}
    npz = {}
    for plane in ('x', 'y'):
        g = df[(df.plane == plane) & (df.chi2dof < 250)]
        gc = g[g.contained]
        if len(gc) < 50:
            continue
        QCOLS, U = basis(gc)
        Q = gc[QCOLS].to_numpy()
        Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-9)
        m, e = Q.mean(axis=0), np.maximum(Q.std(axis=0) / np.sqrt(len(Q)), 1e-5)
        w0 = cal.w0.get(plane, 0.0)
        s = (gc.tan.abs() > 0.10) & (gc.tan.abs() < 0.40)
        v_geom = args.v_geom or float(np.median(
            (gc.loc[s, 'w'] * 1e3 - w0) / gc.loc[s, 'tan']))
        fits = endpoint_fits(U, m, e)
        for f in fits.values():
            if 'T_end' in f:
                f['gap_mm'] = f['T_end'] * v_geom / 1000.0
                f['gap_err'] = f['T_err'] * v_geom / 1000.0
        summary['planes'][plane] = dict(
            n_contained=int(len(gc)), n_edge=int((~g.contained).sum()),
            v_geom=v_geom, u=U.tolist(), profile=m.tolist(),
            profile_err=e.tolist(), fits=fits)
        npz[f'prof_{plane}'] = m
        npz[f'err_{plane}'] = e
        sh = fits.get('sharp', {})
        if 'T_end' in sh:
            print(f'  {plane}: n={len(gc):,} v_geom={v_geom:.2f}  '
                  f'T_end {sh["T_end"]:.0f}+-{sh["T_err"]:.0f} ns  '
                  f'-> gap {sh["gap_mm"]:.2f} mm')

    with open(os.path.join(out_dir, 'gap_study.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez(os.path.join(out_dir, 'profiles.npz'), u=U, **npz)
    df.to_parquet(os.path.join(out_dir, 'event_profiles.parquet'), index=False)
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
