#!/usr/bin/env python3
"""sat_compare.py -- model check 2b: does the saturation clip (bundle sat_adc 3550 vs 3700) move n_TOF tracks?

    sat_compare.py --kind line --a35 A.parquet --a37 A.parquet --c35 C.parquet --c37 C.parquet --run-config RC --out J
    sat_compare.py --kind point --arm A --r35 R.parquet --r37 R.parquet --run-config RC --out J

Both reconstructions are on the same events (ntof_reco.py --events: the events of sat_census.py's ``events_hi``, i.e.
every trigger with a non-coherent sample >= 3500 ADC in the arm's FEUs -- an event with no sample >= 3550 is fitted
bit-identically under both rules, so these are the only events the choice can change).

  per plane   the share of fitted planes whose p0 moves > 0.1 mm or tan > 0.01, and the 68 % of |dp0|, |dtan|
  line        rerun/ntof/ac_line.py on each pair (A-C cosmic line: residual-angle 68 % per arm, line count)
  point       rerun/ntof/capsule_point.py on each (tracks pointing within 15 mm of the capsule axis)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'ntof')]


def per_plane(p35, p37):
    a, b = pd.read_parquet(p35), pd.read_parquet(p37)
    m = a.merge(b, on='event_id', suffixes=('_35', '_37'))
    out = dict(n_events=int(len(m)))
    for p in 'xy':
        ok = m[f'{p}_ok_35'].astype(bool) & m[f'{p}_ok_37'].astype(bool)
        dp = (m[f'{p}_p0_37'] - m[f'{p}_p0_35'])[ok].abs()
        dt = (m[f'{p}_tan_theta_37'] - m[f'{p}_tan_theta_35'])[ok].abs()
        mv = (dp > 0.1) | (dt > 0.01)
        out[p] = dict(n_ok_both=int(ok.sum()), ok_35_only=int((m[f'{p}_ok_35'] & ~m[f'{p}_ok_37']).sum()),
                      ok_37_only=int((~m[f'{p}_ok_35'] & m[f'{p}_ok_37']).sum()),
                      n_moved=int(mv.sum()), frac_moved=float(mv.mean()) if ok.any() else np.nan,
                      dp0_68_moved=float(np.percentile(dp[mv], 68)) if mv.any() else 0.0,
                      dtan_68_moved=float(np.percentile(dt[mv], 68)) if mv.any() else 0.0,
                      dtan_68_all=float(np.percentile(dt, 68)) if ok.any() else np.nan,
                      n_sat_planes_35=int((m.loc[ok, f'{p}_n_sat_strips_35'] > 0).sum()),
                      n_sat_planes_37=int((m.loc[ok, f'{p}_n_sat_strips_37'] > 0).sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=('line', 'point'), required=True)
    ap.add_argument('--a35'); ap.add_argument('--a37'); ap.add_argument('--c35'); ap.add_argument('--c37')
    ap.add_argument('--arm'); ap.add_argument('--r35'); ap.add_argument('--r37')
    ap.add_argument('--run-config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(kind=a.kind)
    if a.kind == 'line':
        import ac_line
        out['plane_A'] = per_plane(a.a35, a.a37)
        out['plane_C'] = per_plane(a.c35, a.c37)
        s35, M35 = ac_line.score(a.a35, a.c35, a.run_config, label='3550')
        s37, M37 = ac_line.score(a.a37, a.c37, a.run_config, label='3700')
        out['line_3550'], out['line_3700'] = s35, s37
        com = sorted(set(M35.event_id) & set(M37.event_id))
        out['n_common_lines'] = len(com)
        # residual angle on the common line events, per arm, for both rules
        def resid(M, s_):
            L = M[[f'p0_x_c', f'p0_y_c', f'p0_z_c']].to_numpy(float) - M[[f'p0_x_a', f'p0_y_a', f'p0_z_a']].to_numpy(float)
            L /= np.linalg.norm(L, axis=1, keepdims=True)
            d = M[[f'd_x_{s_}', f'd_y_{s_}', f'd_z_{s_}']].to_numpy(float)
            return pd.Series(np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', d, L)).clip(0, 1))), index=M.event_id)
        for s_ in 'ac':
            r35 = resid(M35, s_).loc[com]
            r37 = resid(M37, s_).loc[com]
            out[f'res68_{s_.upper()}'] = dict(r3550=float(np.percentile(r35, 68)) if len(com) else np.nan,
                                             r3700=float(np.percentile(r37, 68)) if len(com) else np.nan,
                                             n_changed=int((np.abs(r37.to_numpy() - r35.to_numpy()) > 0.05).sum()),
                                             mean_abs_change=float(np.mean(np.abs(r37.to_numpy() - r35.to_numpy())))
                                             if len(com) else np.nan)
    else:
        import capsule_point
        out['plane'] = per_plane(a.r35, a.r37)
        s35, T35 = capsule_point.score(a.r35, a.arm, a.run_config, '3550')
        s37, T37 = capsule_point.score(a.r37, a.arm, a.run_config, '3700')
        out['point_3550'], out['point_3700'] = s35, s37
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(('line_', 'point_'))}, indent=1, default=float))
    for k in ('line_3550', 'line_3700', 'point_3550', 'point_3700'):
        if k in out:
            print(k, {kk: vv for kk, vv in out[k].items() if kk != 'bins'})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
