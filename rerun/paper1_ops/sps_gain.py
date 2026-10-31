#!/usr/bin/env python3
"""sps_gain.py [--root R] [--registry rerun/sps_registry.csv] [--out-dir D] -- det4's charge per SPS row (Paper I §4).

The bench 6-23 det3/det4 night cannot give det4 a gain curve (rerun/paper1_charge/README.md "Excluded"), so det4's
curve comes from the SPS mesh-voltage scans, with the bench estimator (rerun/paper1_charge/make_report.py):
q = (q_x + q_y) cos(theta_mount), cens_25 / cens_50 / median_unsat, saturated = any strip at the 3550 rule.
Read-only on Analysis_pass1/sps.

Selection per row, in order (counts kept):
  base     event_id once in the table, x_ok and y_ok, not spark
  beam     run60 ov07: t_wall >= 91170 s (beam off before; rerun/sps_followup/README.md part iii)
  sane     q_x, q_y finite and > 0, both below 1e7 (the NNLS runaway tail, `Large q_sum tracks` handoff section)
Two charges per event: the fit's q_sum (as the bench) and the raw cluster sum c_q_sum (x + y; strip ADC around the
lead strip, no fit), which carries the same ZS but no model.  cos(theta_mount) = 1 / sqrt(1 + tan_mount^2).
Also the beam-spot position (median x_p0 / y_p0), so a scan whose spot moved across det4's stripes shows up.

Writes <out-dir>/rows/<key>.json and <out-dir>/events/<key>.npz.
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps'
Q_SANE = 1e7
BEAM_ON = {'run60_flush__ov07': 91170.0}


def one(r, root, out_dir):
    d = os.path.join(root, r['key'])
    p = os.path.join(d, 'events.parquet')
    if not os.path.exists(p):
        return None
    m = json.load(open(os.path.join(d, 'events.meta.json')))
    df = pd.read_parquet(p)
    cnt = dict(all=int(len(df)))
    once = df.event_id.value_counts()
    df = df[df.event_id.isin(once.index[once == 1])]
    df = df[df.x_ok.astype(bool) & df.y_ok.astype(bool) & ~df.spark.astype(bool)]
    cnt['base'] = int(len(df))
    if r['key'] in BEAM_ON:
        df = df[df.t_wall >= BEAM_ON[r['key']]]
    cnt['beam'] = int(len(df))
    qx, qy = df.x_q_sum.to_numpy(float), df.y_q_sum.to_numpy(float)
    ok = np.isfinite(qx) & np.isfinite(qy) & (qx > 0) & (qy > 0) & (qx < Q_SANE) & (qy < Q_SANE)
    df = df[ok]
    cnt['sane'] = int(len(df))
    tm = df.tan_mount.to_numpy(float) if 'tan_mount' in df else np.zeros(len(df))
    cos = 1.0 / np.sqrt(1.0 + np.nan_to_num(tm) ** 2)
    sat_x = df.x_n_sat_strips.to_numpy() > 0
    sat_y = df.y_n_sat_strips.to_numpy() > 0
    out = dict(key=r['key'], dataset=r['dataset'], run=r['run'], plateau=r['plateau'], gas=r['gas'],
               mount=r['mount'], resist_V=float(r['resist_V']), drift_V=float(r['drift_V']),
               field_Vcm=float(r['field_Vcm']), raw=r['raw'], n_samples=r['n_samples'], cls=r['cls'],
               code=m.get('code'), zeros=m.get('zeros'), counts=cnt,
               f_sat=dict(x=float(sat_x.mean()) if len(df) else None, y=float(sat_y.mean()) if len(df) else None,
                          any=float((sat_x | sat_y).mean()) if len(df) else None),
               spot=dict(x=float(np.median(df.x_p0)) if len(df) else None,
                         y=float(np.median(df.y_p0)) if len(df) else None))
    os.makedirs(os.path.join(out_dir, 'rows'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'events'), exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, 'rows', r['key'] + '.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out_dir, 'events', r['key'] + '.npz'), event_id=df.event_id.to_numpy(),
                        qx=qx[ok] * cos, qy=qy[ok] * cos, cqx=df.x_c_q_sum.to_numpy(float) * cos,
                        cqy=df.y_c_q_sum.to_numpy(float) * cos, sat_x=sat_x, sat_y=sat_y,
                        nsx=df.x_n_sat_strips.to_numpy(), nsy=df.y_n_sat_strips.to_numpy(),
                        x_p0=df.x_p0.to_numpy(float), y_p0=df.y_p0.to_numpy(float),
                        t_wall=df.t_wall.to_numpy(float) if 't_wall' in df else np.zeros(len(df)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'sps_registry.csv'))
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    n = 0
    for r in csv.DictReader(open(a.registry)):
        o = one(r, a.root, a.out_dir)
        if o:
            n += 1
            print(r['key'], o['counts'], 'f_sat', o['f_sat']['any'])
    print(f'{n} rows')


if __name__ == '__main__':
    main()
