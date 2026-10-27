#!/usr/bin/env python3
"""position_scale.py — is the angle scale a function of WHERE the track crosses the chamber (v(x, y), field
non-uniformity), and does the telescope acceptance tie position to angle so that a position-dependent scale looks like
an angle-dependent one?

For each chamber/view: (1) the scale r = (w − w0)/(v·tan_ref) of inclined tracks (0.10 < |tan| < 0.30) in bins of the
track position along the SAME axis (p0) and along the OTHER axis; (2) the mean position of the tracks per signed tan bin
(the acceptance correlation); (3) S(|tan|) recomputed after dividing each event's r by its position-bin scale (a
position-corrected response: if the angle dependence disappears, it was v(x, y) seen through the acceptance).
Writes ws_widthtilt/position_scale.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, decompose, ROOT, ABS


def main():
    out = {}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        b, T = load(det); v = b['v_drift']
        # cross-view position: join on eid
        px = T['x'].set_index('eid').p0; py = T['y'].set_index('eid').p0
        for p in 'xy':
            df = T[p].copy()
            other = py if p == 'x' else px
            df['p_other'] = other.reindex(df.eid).to_numpy()
            inc = df[(np.abs(df.t) > 0.10) & (np.abs(df.t) < 0.30)]
            res = {}
            for col in ('p0', 'p_other'):
                e = np.nanpercentile(inc[col], np.linspace(0, 100, 7))
                sc = []
                for lo, hi in zip(e[:-1], e[1:]):
                    s = inc[(inc[col] >= lo) & (inc[col] < hi)]
                    # signed-symmetric scale (w0-free)
                    sp, sn = s[s.t > 0], s[s.t < 0]
                    sc.append(dict(lo=float(lo), hi=float(hi), S=float(0.5 * (np.median(sp.r) + np.median(sn.r))) if len(sp) > 5 and len(sn) > 5 else np.nan,
                                   n=int(len(s))))
                res[col] = sc
            # acceptance: mean position per signed tan bin
            acc = []
            for lo, hi in [(-0.45, -0.28), (-0.28, -0.14), (-0.14, -0.03), (0.03, 0.14), (0.14, 0.28), (0.28, 0.45)]:
                s = df[(df.t >= lo) & (df.t < hi)]
                acc.append(dict(lo=lo, hi=hi, p0=float(s.p0.median()), p_other=float(s.p_other.median()), n=int(len(s))))
            res['acceptance'] = acc
            # position-corrected response: divide r by a smooth scale(p0) (bin-wise, 6 bins)
            e = np.nanpercentile(df.p0, np.linspace(0, 100, 7)); e[-1] += 1e-6
            fac = np.ones(len(df))
            for c in res['p0']:
                m = (df.p0 >= c['lo']) & (df.p0 < c['hi'] + 1e-6)
                if np.isfinite(c['S']):
                    fac[m.to_numpy()] = c['S']
            dfc = df.copy(); dfc['r'] = df.r / fac
            S0, E0, _ = decompose(df, v); S1, E1, _ = decompose(dfc, v)
            res['S_raw'] = S0; res['E_raw'] = E0; res['S_poscorr'] = S1; res['E_poscorr'] = E1
            out[f'{det}_{p}'] = res
            f = lambda xs: ' '.join(f'{x:6.3f}' for x in xs)
            print(f'{det} {p}  S(p0 bins) {f([c["S"] for c in res["p0"]])}  S(other) {f([c["S"] for c in res["p_other"]])}')
            print(f'       accept p0 {f([a["p0"] for a in acc])}   other {f([a["p_other"] for a in acc])}')
            print(f'       S raw {f(S0)} -> poscorr {f(S1)}   E raw {f(E0)} -> {f(E1)}')
    json.dump(out, open(f'{ROOT}/ws_widthtilt/position_scale.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
