"""Phase-corrected, tooth-invariant timing of each bench chamber's charge column (workstream D, 2026-10-27).

The per-plane fitted t0 is only defined up to the depth-bin tooth (shift t0 by one 60 ns bin, slide the NNLS profile
one bin, the model barely changes), so its median is not a physical number.  The ARRIVAL TIMES of the profile's
quantiles, t0 + u50 and t0 + u90, are invariant under that shift.  For a through-going track (uniform ionisation over
the gap) the median charge arrives T/2 after the mesh charge and the 90 % quantile 0.9 T after it, so

    T      = (u90 - u50) / 0.4           column duration -> v_col = gap / T
    t_mesh = (t0 + u50) - T / 2          arrival of the mesh-level charge (what the trigger latency fixes)

Every time here is phase-corrected with the trigger's fine timestamp, t + 10 ns * ftst (the DREAM clock is free-
running against the trigger; ftst is the trigger phase in 10 ns units, PLAN_42).

Usage: t0_invariants.py [--base ~/x17/cosmic_bench/model_audit] [--out ws_t0/t0_invariants.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
# DREAM FEU of each view in the golden run (run_config.json 'dream_feus')
FEU = {'det2': (6, 8), 'det3': (7, 8), 'det4': (6, 8), 'det6': (3, 4), 'det7': (6, 8)}
RUN = {'det2': 'det2_det3 overnight 6-22', 'det3': 'det3 saturday 6-27', 'det4': 'det4 day 6-24',
       'det6': 'det6_det7 overnight 6-26', 'det7': 'det6_det7 overnight 6-26'}
GAP_MM = 30.0


def rsig(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return np.nan
    q = np.percentile(a, [16, 84])
    return 0.5 * (q[1] - q[0])


def plane_rows(d, p, tan_max=0.3):
    ok = (d[p + '_ok'].astype(bool) & d[p + '_quality_ok'].astype(bool) & (d[p + '_ftst'] >= 0)
          & (d[p + '_tan_theta'].abs() < tan_max) & (d[p + '_q_uend'] > 300) & (d[p + '_n_strips'] >= 3))
    s = d[ok]
    ph = 10.0 * s[p + '_ftst']
    return pd.DataFrame(dict(
        ftst=s[p + '_ftst'].astype(int),
        t0c=s[p + '_t0'] + ph,
        t50c=s[p + '_t0'] + s[p + '_q_u50'] + ph,
        t90c=s[p + '_t0'] + s[p + '_q_u90'] + ph,
        u50=s[p + '_q_u50'], u90=s[p + '_q_u90'], uend=s[p + '_q_uend'],
        tan=s[p + '_tan_theta'], q=s[p + '_q_sum'], event_id=s['event_id']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.expanduser('~/x17/cosmic_bench/model_audit'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 't0_invariants.json'))
    a = ap.parse_args()
    res = {}
    for det in DETS:
        d = pd.read_parquet(f'{a.base}/{det}/final_physmask/events.parquet')
        b = json.load(open(f'{a.base}/{det}/calib_bundle/bundle.json'))
        v = float(b['v_drift'])
        res[det] = dict(run=RUN[det], v_bundle=v, feu=dict(x=FEU[det][0], y=FEU[det][1]))
        for i, p in enumerate('xy'):
            r = plane_rows(d, p)
            T = (r.u90.median() - r.u50.median()) / 0.4
            tm = r.t50c.median() - T / 2
            # ftst ladder slope of the raw t0 (expected -10 ns / class)
            g = (r.t0c - 10.0 * r.ftst).groupby(r.ftst).median()
            slope = float(np.polyfit(g.index.values, g.values, 1)[0])
            # tooth structure of the phase-corrected t0: fraction within +-30 ns of the mode
            h, e = np.histogram(r.t0c, bins=np.arange(-200, 800, 5))
            mode = float(e[np.argmax(h)] + 2.5)
            dev = r.t0c - mode
            res[det][p] = dict(
                n=int(len(r)), feu=FEU[det][i],
                t0c_median=float(r.t0c.median()), t0c_mode=mode, t0c_rsig=float(rsig(r.t0c)),
                frac_central_tooth=float((dev.abs() < 30).mean()),
                frac_early_tooth=float(((dev < -30) & (dev > -90)).mean()),
                frac_late_tooth=float(((dev > 30) & (dev < 90)).mean()),
                t50c_median=float(r.t50c.median()), t50c_rsig=float(rsig(r.t50c)),
                t90c_median=float(r.t90c.median()), t90c_rsig=float(rsig(r.t90c)),
                T_col=float(T), v_col=float(GAP_MM / T * 1e3), t_mesh=float(tm),
                ftst_slope_raw_t0=slope)
            print(f'{det} {p} FEU{FEU[det][i]} n={len(r):5d}  t0c med {r.t0c.median():6.1f} mode {mode:6.1f} '
                  f'rsig {rsig(r.t0c):5.1f} central {res[det][p]["frac_central_tooth"]:.2f}  '
                  f't50c {r.t50c.median():6.1f} ({rsig(r.t50c):4.1f})  t90c {r.t90c.median():6.1f}  '
                  f'T {T:6.1f} v_col {GAP_MM / T * 1e3:5.1f} (bundle {v:5.1f})  t_mesh {tm:6.1f}  '
                  f'ftst slope {slope:+.1f}')
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
