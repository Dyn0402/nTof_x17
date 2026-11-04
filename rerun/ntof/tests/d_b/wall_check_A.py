#!/usr/bin/env python3
"""wall_check_A.py -- the wall and plastic estimators of wall_kw.py on chamber A, whose scale the A-C line fixes.

A's fits: the held-out run_149 cos_0001 v1k reconstruction (kw measured on cos_0000; the A-C line closes at
1.003 / 0.994 on cos_0001).  If the estimators are unbiased they return s = true / fitted = 1 here.  Output
d_b/wall_kw/check_A.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]
W = '/media/ucla/work/ntof_pass1'


def main():
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_scint_stack import ana as SA
    from sept26_prelim_analysis.det_a_scint import layer_geometry
    import line_kw as LK
    import wall_kw as WK
    CWS.ARM = 'A'
    geo = layer_geometry('run_149', 'A')
    L = geo['w_wall'] - geo['w_strip']
    H = pd.read_parquet('/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/wall_hits_A.parquet')
    pk = CWS.peaks(H)
    sub = 'cosbounce_cos_0001'
    Wd = CWS.wide(H[H.subrun == sub], pk).assign(subrun=sub)
    trig = H[H.subrun == sub][['subrun', 'event_id', 'trig_arm']].drop_duplicates(['subrun', 'event_id'])
    T = LK.fitted(f'{W}/heldout/r149s1_A_v1k.parquet', 'A', LK.transforms()['mx17_A'], single=False).assign(subrun=sub)
    T = T.merge(trig, on=['subrun', 'event_id']).merge(Wd, on=['subrun', 'event_id'], how='left')
    T['u_mm'] = T.x_local + geo['u_mm']
    T['tan_raw_x'] = T.tanx
    x = CWS._cosmic_singles(T, T.x_quality_ok.astype(bool) & T.x_plausible.astype(bool))
    w = SA.fit_wall_u(x, L, 1.0)
    P = CWS.profile(x, L, np.round(np.arange(0.70, 1.501, 0.0125), 4))
    inside = P[P.dnll <= 0.5]
    WK.OUT = f'{W}/d_b/wall_kw'
    pl = WK.plastic(T, 'A', 'checkA')
    res = dict(arm='A', subrun=sub, n_single_x=int(len(x)), wall_s=float(w['s']), wall_s_err=float(w['s_err']),
               wall_prof_best=float(P.s[P.dnll.idxmin()]), wall_prof_lo=float(inside.s.min()),
               wall_prof_hi=float(inside.s.max()), plastic_s=pl['s'], plastic_boot_lo=pl['boot_lo'],
               plastic_boot_hi=pl['boot_hi'])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'{W}/d_b/wall_kw/check_A.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
