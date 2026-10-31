#!/usr/bin/env python3
"""dp_score.py --out OUT.json -- score the X-width (diffusion) arms of dp_arms.py against the pass table, table level,
on exactly the pass's samples (PASS1_RESOLUTION_2026-10-31.md §4).  READ-ONLY on Analysis_pass1.

Per arm, on the events both tables reconstructed (same G1 / pass-table events):
  angle     03_angles' interior_good sample of the pass (paper1_eff/angles/<rel>/per_event.npz: eids and tan_ref), the
            arm's tan_theta against the pass's (s68, 03_angles' 68th percentile, both on the same events); and the arm
            with w0/kw re-folded (set_w0kw's ladder, as a difference to the same measurement on the pass table)
  position  the effbudget rays' interior_good sample (spark veto, |r| < 15 mm): the arm's p0 - M3 against the pass's,
            core sigma (res_rows.stats)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('RES_CODE') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'rerun', 'vt_within_rows')]
import res_rows as rr  # noqa: E402
from dp_arms import ARMS, ROOT, rel  # noqa: E402
from wft import reco as wr  # noqa: E402
from wft.seed import SPARK_VETO_HITS  # noqa: E402
from common.mx17_fiducial import fiducial_mask  # noqa: E402
from vt_common import ladder, bow_aS  # noqa: E402


def tp_map(df, cal, z, p):
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    e = z[f'{p}_eid'].astype(int)
    k = np.isin(e, df.index.to_numpy())
    e = e[k]
    pos = df.index.get_indexer(e)
    a, S = bow_aS(cal, p, U[pos], W[pos])
    return e, (z[f'{p}_tan_ref'][k] + a) * S, a, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    out = {}
    for arm, (key, d, k) in ARMS.items():
        r = reg[key]
        t = os.path.join(ROOT, rel(r), 'wft', f'arm_{arm}', 'events.parquet')
        if not os.path.exists(t):
            print(arm, 'no arm table yet'); continue
        wdir = os.path.join(rr.P1, rel(r), 'wft')
        m, cal = rr.load_calib(wdir)
        B = pd.read_parquet(os.path.join(wdir, 'events.parquet')).drop_duplicates('event_id', keep=False).set_index('event_id')
        A = pd.read_parquet(t).drop_duplicates('event_id', keep=False).set_index('event_id')
        common = B.index.intersection(A.index)
        B, A = B.loc[common], A.loc[common]
        ab = json.load(open(os.path.join(ROOT, rel(r), 'wft', f'bundle_{arm}', 'bundle.json')))
        res = dict(key=key, det=d, shift_100um_rtcm=k, n_common=int(len(common)),
                   Dp=[float(m['bundle']['hyper']['Dp']), float(ab['hyper']['Dp'])])
        ANG = np.load(os.path.join(rr.EFF, 'angles', rel(r), 'per_event.npz'))
        Z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
        rays = pd.read_parquet(os.path.join(rr.EFF, 'budget', rel(r), f'rays_{d}.parquet')).drop_duplicates('event_id', keep=False)
        ig = fiducial_mask('interior_good', rays.mesh_x.to_numpy(), rays.mesh_y.to_numpy(), rays.tan_x.to_numpy(),
                           rays.tan_y.to_numpy(), r['det'])
        R = rays[ig].set_index('event_id')
        for p in 'xy':
            # re-fold on the arm: the ladder on the arm's w minus the same on the pass's, over the full-coverage events
            e, tp, aa, S = tp_map(B, cal, Z, p)
            okc = A[f'{p}_ok'].reindex(e).astype('boolean').fillna(False).to_numpy() & B[f'{p}_ok'].reindex(e).astype('boolean').fillna(False).to_numpy()
            wb = B[f'{p}_w'].reindex(e).to_numpy() * 1e3
            wa = A[f'{p}_w'].reindex(e).to_numpy() * 1e3
            w0b, kwb, _, _ = ladder(cal, np.where(okc, wb, np.nan), tp)
            w0a, kwa, _, _ = ladder(cal, np.where(okc, wa, np.nan), tp)
            w0n, kwn = cal.w0[p] + (w0a - w0b), cal.kw[p] * kwa / kwb
            tan_refold = pd.Series((wa - w0n) / (kwn * cal.v_drift * S) - aa, index=e)
            ea = ANG[f'{p}_eid'].astype(int)
            sel = np.isin(ea, common) & A[f'{p}_ok'].reindex(ea).astype('boolean').fillna(False).to_numpy()
            ea, tha = ea[sel], np.degrees(np.arctan(ANG[f'{p}_tan_ref'][sel]))
            db = np.degrees(np.arctan(B.loc[ea, f'{p}_tan_theta'].to_numpy())) - tha
            da = np.degrees(np.arctan(A.loc[ea, f'{p}_tan_theta'].to_numpy())) - tha
            dr = np.degrees(np.arctan(tan_refold.reindex(ea).to_numpy())) - tha
            sb, sa, sr = (rr.stats(x, rr.ANG_Q) for x in (db, da, dr))
            # positions
            ep = R.index.intersection(common)
            okp = (A.loc[ep, f'{p}_ok'].astype(bool) & B.loc[ep, f'{p}_ok'].astype(bool)).to_numpy()
            ep = ep[okp]
            spark = R.loc[ep, 'mult'].to_numpy() > SPARK_VETO_HITS
            mesh = R.loc[ep, f'mesh_{p}'].to_numpy()
            pb = B.loc[ep, f'{p}_p0'].to_numpy() - mesh
            pa = A.loc[ep, f'{p}_p0'].to_numpy() - mesh
            keep = ~spark & np.isfinite(pb) & np.isfinite(pa) & (np.abs(pb) < rr.RES_MAX_MM) & (np.abs(pa) < rr.RES_MAX_MM)
            qb, qa = rr.stats(pb[keep]), rr.stats(pa[keep])
            res[p] = dict(n=sb['n'], s68_base=sb['s68'], s68_arm=sa['s68'], s68_arm_refold=sr['s68'],
                          core_base=sb['core'], core_arm=sa['core'], bias_base=sb['bias'], bias_arm=sa['bias'],
                          kw_ratio=float(kwa / kwb), n_pos=qb['n'], pos_base=qb['core'], pos_arm=qa['core'],
                          pos_s68_base=qb['s68'], pos_s68_arm=qa['s68'],
                          frac_tan_changed=float(np.mean(np.abs(da - db) > 1e-6)) if len(da) else np.nan)
        out[arm] = res
        print(arm, json.dumps({p: {k2: round(v, 4) if isinstance(v, float) else v for k2, v in res[p].items()} for p in 'xy'}))
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
