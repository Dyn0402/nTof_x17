#!/usr/bin/env python3
"""chi2_split.py -- GB test (a) on all five chambers: the FD measured resolution (s68 vs M3) in quintiles of the M3
track chi2 of the coordinate the view is built from, inside the production selection (chi2 < 1, NClus 4: the tables
hold no reconstructed events outside it).  A reference-limited resolution worsens with chi2 (the 4-point chi2 sees
part of any kink and every bad point); a chamber-limited one is flat.

    chi2_split.py   -> results/chi2_split.json
"""
import json
import os

import numpy as np

from common import RES, RAYS, load_dut, load_rays, fit_map, deg, s68


def main():
    out = {}
    cache = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        if RAYS[det] not in cache:
            cache[RAYS[det]] = load_rays(RAYS[det])
        rays = cache[RAYS[det]]
        for v in ('x', 'y'):
            d, _, _ = load_dut(det, v)
            j = d.join(rays, how='inner')
            c, rms = fit_map(j['tan_ref'], j)
            src = 'chi2X' if abs(c[0]) > abs(c[1]) else 'chi2Y'
            dd = deg(j['tan_fit']) - deg(j['tan_ref'])
            ch = j[src].to_numpy()
            q = np.percentile(ch, [0, 20, 40, 60, 80, 100])
            rows = [dict(lo=float(lo), hi=float(hi), n=int(((ch >= lo) & (ch <= hi)).sum()),
                         s68=s68(dd[(ch >= lo) & (ch <= hi)])) for lo, hi in zip(q[:-1], q[1:])]
            both = np.maximum(j['chi2X'], j['chi2Y']).to_numpy()
            lo_ = both < np.percentile(both, 25)
            out[f'{det}_{v}'] = dict(n=int(len(j)), map_rms=rms, m3_coord=src, s68_all=s68(dd), quintiles=rows,
                                     s68_best_quarter_both=s68(dd[lo_]), s68_rest=s68(dd[~lo_]),
                                     slope_deg_per_unit_chi2=float(np.polyfit([r['s68'] for r in rows][:0] or
                                                                            [0.5 * (r['lo'] + r['hi']) for r in rows],
                                                                            [r['s68'] for r in rows], 1)[0]))
            print(det, v, src, f"all {s68(dd):.2f}", ' '.join(f"{r['s68']:.2f}" for r in rows),
                  f"| best quarter (both coords) {s68(dd[lo_]):.2f} rest {s68(dd[~lo_]):.2f}")
    json.dump(out, open(os.path.join(RES, 'chi2_split.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
