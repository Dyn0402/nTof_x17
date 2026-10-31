#!/usr/bin/env python3
"""angle.py --pairs key=row.npz,events.parquet ... --out results/angle.json
-- the column-start time against the track angle (Paper I §7: is the estimator a physical time at every angle?).

Joins timing_row.py's per-trigger times (acq tag, eventId) with the pass-1 events table of the same row for the fitted
angles (x/y_tan_theta, both views ok, one track).  Per bin of the larger |tan|: median event time (a depth-dependent
estimator would drift with the angle), s68 of the event time and of (t_X - t_Y)/sqrt 2.  The head-on bin (|tan| < 0.05
in both views) is the SPS situation.
"""
import argparse
import json

import numpy as np
import pandas as pd

BINS = [0.0, 0.05, 0.15, 0.3, 0.5, 1.0]


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 30:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {}
    for p in a.pairs:
        key, files = p.split('=', 1); fn, fe = files.split(',')
        z = np.load(fn); tags = list(z['acq_tags'])
        T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], tx=z['tx'], ty=z['ty'],
                              ex=z['ex'], ey=z['ey']))
        ev = pd.read_parquet(fe, columns=['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks', 'x_tan_theta', 'y_tan_theta'])
        m = T.merge(ev, on=['acq_tag', 'event_id'])
        m = m[m.x_ok & m.y_ok & (m.n_tracks == 1)]
        M = (m.tx + m.ty) / 2; D = m.tx - m.ty
        big = np.maximum(m.x_tan_theta.abs(), m.y_tan_theta.abs())
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (big >= lo) & (big < hi)
            rows.append(dict(lo=lo, hi=hi, n=int(s.sum()), median=float(np.nanmedian(M[s])) if s.sum() else np.nan,
                             sigma_evt=s68(M[s]), sigma_xy=s68(D[s]) / np.sqrt(2)))
        # the M3-matched single-track sample as a whole: widths, and cov(start, end) of the column (trigger term)
        E_ = (m.ex + m.ey) / 2
        ok = np.isfinite(M) & np.isfinite(E_)
        sa, sb = M[ok].to_numpy(), E_[ok].to_numpy()
        qa, qb = np.percentile(sa, [5, 95]), np.percentile(sb, [5, 95])
        c = (sa > qa[0]) & (sa < qa[1]) & (sb > qb[0]) & (sb < qb[1])
        cov = float(np.cov(sa[c], sb[c])[0, 1])
        sxy, sev = s68(D) / np.sqrt(2), s68(M)
        allm = dict(n=int(len(m)), sigma_xy=sxy, sigma_evt=sev, sigma_x=s68(m.tx), sigma_y=s68(m.ty),
                    sigma_common=float(np.sqrt(max(sev ** 2 - sxy ** 2 / 2, 0))), trig_cov_ns2=cov)
        res[key] = dict(n_joined=int(len(m)), all=allm, bins=rows)
        print(key, 'M3-matched all:', {k: round(v, 1) for k, v in allm.items()})
        print(key, len(m), ' '.join(f'[{r["lo"]}-{r["hi"]}: n {r["n"]} med {r["median"]:.0f} evt {r["sigma_evt"]:.1f} '
                                    f'xy {r["sigma_xy"]:.1f}]' for r in rows))
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
