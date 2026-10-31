#!/usr/bin/env python3
"""toy_inputs_pass1.py --out F.json -- the measured inputs of the FD toy on the pass-1 calibration rows (2026-10-31).

What the 10-27 toy took from the audit bundles / tables and the pass-1 basis replaces (pass1_basis.py reads this file):
  s0        sigma_p0_x of the pass bundle (0.30 mm, the physical sigma0 of the calib basis study; audit 0.238)
  q50       median x_q_sum of the row's M3-matched events (sample 'full') and of interior_good ('ig')
  ang_w     fraction of |theta_ref| (X) in [0, 6), [6, 12.5), [12.5, 90) deg -> weights of the toy's 3 / 9 / 16 deg
            points, per sample: the data's angle mix, which the 10-27 budget pooled with equal weights
  v_rel     the row's own r(t) per event (pass arm: the uncorrected drift), normalised to its mean, thinned to <= 4000
            values -- the vdrift rung draws v_true = v * v_rel instead of a Gaussian of the FOLLOWUPS-1 rms; rows with
            no v(t) map (below the gate) get a uniform spread of the survey's |dv| (bench_vt.csv)
Run on lxplus with RES_CODE=<code archive with rerun/paper1_res> (lock venv), like fd_pass1.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

CODE = os.environ['RES_CODE']
sys.path[:0] = [os.path.join(CODE, 'rerun', 'paper1_res')]
import res_rows as rr  # noqa: E402

CALIB = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det6='g_det6_long', det7='g_det7_long')
EDGES = (0.0, 6.0, 12.5, 90.0)


def weights(theta):
    a = np.abs(np.asarray(theta, float))
    a = a[np.isfinite(a)]
    h = np.histogram(a, bins=EDGES)[0].astype(float)
    return (h / h.sum()).tolist(), int(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'pass_registry.csv')))}
    vtrows = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'vt_all', 'results',
                                                                    'vt_corrected_rows.csv')))}
    survey = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'vt_all', 'results', 'bench_vt.csv')))}
    rs = np.random.default_rng(5)
    out = {}
    for det, key in CALIB.items():
        r = reg[key]
        wdir = os.path.join(rr.P1, rr.rel(r), 'wft')
        b = json.load(open(os.path.join(wdir, 'calib_bundle', 'bundle.json')))
        df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
        df = df[~df.event_id.duplicated(keep=False)].set_index('event_id')
        rec = dict(key=key, wdir=wdir, s0=float(b['hyper'].get('sigma_p0_x', b['hyper']['sigma_p0'])),
                   v_drift=float(b['v_drift']), Dp=float(b['hyper']['Dp']), dead_x=sorted(b['dead'].get('x', [])),
                   ang_w={}, n_ang={}, q50={})
        for smp, src in (('full', os.path.join(wdir, 'angles', 'per_event.npz')),
                         ('ig', os.path.join(rr.EFF, 'angles', rr.rel(r), 'per_event.npz'))):
            Z = np.load(src)
            e = Z['x_eid'].astype(int)
            rec['ang_w'][smp], rec['n_ang'][smp] = weights(np.degrees(np.arctan(Z['x_tan_ref'])))
            q = df['x_q_sum'].reindex(e).to_numpy()
            rec['q50'][smp] = float(np.nanmedian(q))
        if key in vtrows:
            v = pd.read_parquet(vtrows[key]['table']).drop_duplicates('event_id').set_index('event_id')['v_rel']
            v = v.reindex(df.index).to_numpy()
            v = v[np.isfinite(v)]
            v = v / v.mean()
            rec['v_source'] = 'v(t) map (vt_corrected_rows.csv), per event'
        else:
            dv = abs(float(survey[key]['dv_pct'] or 0)) / 100
            v = 1 + rs.uniform(-dv / 2, dv / 2, 4000)
            rec['v_source'] = f'uniform |dv| = {dv:.4f} (bench_vt.csv, below the v(t) gate)'
        if len(v) > 4000:
            v = rs.choice(v, 4000, replace=False)
        rec['v_rel'] = [round(float(x), 5) for x in v]
        rec['v_rms'] = float(np.std(v))
        out[det] = rec
        print(det, key, 's0', rec['s0'], 'q50', rec['q50'], 'ang_w', rec['ang_w'], 'v_rms', round(rec['v_rms'], 4),
              rec['v_source'], flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
