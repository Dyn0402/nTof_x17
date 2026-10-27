#!/usr/bin/env python3
"""unified_checks.py <gate_full> -- two questions the gate table cannot answer, unified arm against production:

1. Do the tracks production already had get worse?  Angle residual against M3 (robust width, deg) on the SAME events:
   interior M3 rays that production detects within 5 mm in both views, the unified fit of those events.  A change in
   the gate table's s68 that does not show here is the population the new pieces add (head-on rescues etc.).
2. The two-track contract: share of clean single muons (interior, production within 3 mm in both views) whose selected
   candidate the unified reconstruction splits, per view.  Contract: <= 1 %.
"""
import sys
import numpy as np
import pandas as pd

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
rw = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))


def main(root, arm='unified'):
    for d in DETS:
        try:
            u = pd.read_parquet(f'{root}/{d}/{arm}/events.parquet').set_index('event_id')
        except FileNotFoundError:
            print(f'{d}: no {arm} table'); continue
        r = pd.read_parquet(f'{root}/{d}/prod/rays_{d}.parquet').drop_duplicates('event_id').set_index('event_id')
        p = pd.read_parquet(f'{root}/{d}/prod/events.parquet').set_index('event_id')
        com = r.index[(r.region == 'interior') & (r.cat_xy_5 == 'detected')].intersection(u.index).intersection(p.index)
        out = [f'{d}: n {len(com)}']
        for v in 'xy':
            ref = r.loc[com, f'tan_{v}'].to_numpy()
            res = {}
            for nm, t in (('prod', p), (arm, u)):
                f = t.loc[com, f'{v}_tan_theta'].to_numpy()
                # the M3 tangent's sign convention per view: take the one the fit follows
                s = 1.0 if rw(f - ref) <= rw(f + ref) else -1.0
                res[nm] = np.degrees(rw(np.arctan(f) - np.arctan(s * ref)))
            out.append(f'{v.upper()} {res["prod"]:.3f} -> {res[arm]:.3f} deg ({res[arm] - res["prod"]:+.3f})')
        clean = r.index[(r.region == 'interior') & (r.cat_xy_3 == 'detected')].intersection(u.index)
        try:
            sp = pd.read_parquet(f'{root}/{d}/{arm}/events.splits.parquet')
            a = sp[sp.accepted & (sp['rank'] == 0) & sp.event_id.isin(clean)]
            out.append('splits on clean singles ' + ' / '.join(
                f'{v.upper()} {100 * (a.plane == v).sum() / len(clean):.2f} %' for v in 'xy') + f' (n {len(clean)})')
        except FileNotFoundError:
            out.append('no splits table')
        print(' | '.join(out))


if __name__ == '__main__':
    main(*sys.argv[1:])
