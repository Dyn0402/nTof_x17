#!/usr/bin/env python3
"""t0_position.py --pairs key=row.npz,rays.parquet ... --out results/t0_position.json
-- does the trigger-time error reach the reconstructed position?  (Paper I §7 consequence.)

The pass pins each plane's t0 to the trigger (t0_abs[ftst], prior sigma 0.05 ns).  If the true crossing time is off by
dt, the fitted track moves along the drift: p0 (the mesh position) shifts by tan x v x dt, the angle does not.  The
chamber measures dt itself: the column start (timing_row.py, mean of X and Y, minus the row median).  So the residual
p0 - M3 (effbudget rays table, raw strip frame, the M3 ray at the alignment z) is regressed on tan_ref x dt:
  slope = v_eff x f,  f = var(trigger part of dt) / var(dt)
On a row with a clean trigger the slope is ~0; with a trigger jitter it is positive, and the corrected residual
res - slope x tan x dt shows what the jitter costs.  Fixed-window core sigma as in paper1_res.
"""
import argparse
import json

import numpy as np
import pandas as pd


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    q = np.percentile(np.abs(a - np.median(a)), 68.27)
    return float(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = {}
    for p in a.pairs:
        key, files = p.split('=', 1); fn, fr = files.split(',')
        z = np.load(fn); tags = list(z['acq_tags'])
        T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], eid=z['eid'], tevt=(z['tx'] + z['ty']) / 2,
                              tend=(z['ex'] + z['ey']) / 2))
        R = pd.read_parquet(fr)
        print(key, 'rays columns:', [c for c in R.columns][:40])
        idcol = 'event_id' if 'event_id' in R.columns else 'eid'
        # the rays table carries the event id and (if several acquisitions) the acquisition tag
        on = [c for c in ('acq_tag',) if c in R.columns]
        if on:
            m = R.merge(T, left_on=on + [idcol], right_on=['acq_tag', 'eid'])
        else:
            dup = T.eid.duplicated(keep=False)
            m = R.merge(T[~dup], left_on=idcol, right_on='eid')
        res = dict(n=int(len(m)))
        dt = m.tevt - np.nanmedian(m.tevt)
        dmid = (m.tevt + m.tend) / 2 - np.nanmedian((m.tevt + m.tend) / 2)
        for v in 'xy':
            rc, tc = f'res_{v}', f'tan_{v}'
            if rc not in m.columns:
                continue
            s = (np.isfinite(m[rc]) & (m[rc].abs() < 15) & np.isfinite(dt) & np.isfinite(dmid) & np.isfinite(m[tc])
                 & m[f'{v}_ok'].astype(bool))
            for nm, d_ in (('start', dt), ('mid', dmid)):
                X = (m[tc] * d_)[s].to_numpy(); Y = m[rc][s].to_numpy()
                core = np.abs(Y - np.median(Y)) < 3 * s68(Y)
                A = np.vstack([X[core], np.ones(core.sum())]).T
                coef, *_ = np.linalg.lstsq(A, Y[core], rcond=None)
                yc = Y - coef[0] * X
                # error of the slope by bootstrap
                rng = np.random.default_rng(1); bs = []
                ii_all = np.flatnonzero(core)
                for _ in range(100):
                    ii = rng.choice(ii_all, len(ii_all))
                    bs.append(np.linalg.lstsq(np.vstack([X[ii], np.ones(len(ii))]).T, Y[ii], rcond=None)[0][0])
                res[f'{v}_{nm}'] = dict(n=int(s.sum()), slope_mm_per_ns=float(coef[0]), slope_err=float(np.std(bs)),
                                        s68_before=s68(Y), s68_after=s68(yc),
                                        s68_before_big_tan=s68(Y[np.abs(m[tc][s].to_numpy()) > 0.2]),
                                        s68_after_big_tan=s68(yc[np.abs(m[tc][s].to_numpy()) > 0.2]))
        # position width against |tan_ref| per view: a trigger jitter sigma_t adds (|tan| v sigma_t) in quadrature
        for v in 'xy':
            rc, tc = f'res_{v}', f'tan_{v}'
            s = np.isfinite(m[rc]) & (m[rc].abs() < 15) & np.isfinite(m[tc]) & m[f'{v}_ok'].astype(bool)
            tt, yy = m[tc][s].abs().to_numpy(), m[rc][s].to_numpy()
            edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45]
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b_ = (tt >= lo) & (tt < hi)
                if b_.sum() >= 60:
                    bins.append(dict(tan=float(np.median(tt[b_])), n=int(b_.sum()), s68=s68(yy[b_])))
            # sigma^2 = a^2 + (c tan)^2 by least squares on the binned s68^2
            if len(bins) >= 3:
                Tq = np.array([b['tan'] for b in bins]) ** 2; Sq = np.array([b['s68'] for b in bins]) ** 2
                (cq, aq), *_ = np.linalg.lstsq(np.vstack([Tq, np.ones_like(Tq)]).T, Sq, rcond=None)
                res[f'{v}_vs_tan'] = dict(bins=bins, a_mm=float(np.sqrt(max(aq, 0))), c_mm=float(np.sign(cq) * np.sqrt(abs(cq))))
        out[key] = res
        print(key, json.dumps(res, indent=None)[:1500])
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
