#!/usr/bin/env python3
"""satgate_lost.py ROOT ARM DET... -- why interior M3 rays move detected -> far between sat_base and ARM in the gate
tree ROOT (pull_gate.sh layout): per view, how many lost rays have the fit's p0 moved > 5 mm (a different position,
not a refit), whether the base had an accepted two-track split there, and the mean n_tracks in each arm."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model_audit'))
from gate_compare import ROWS  # noqa: E402


def main(root, arm, dets):
    for det in dets:
        d = os.path.join(root, ROWS[det], 'wft')
        ra = pd.read_parquet(f'{d}/sat_base/budget/rays_{det}.parquet'); rb = pd.read_parquet(f'{d}/{arm}/budget/rays_{det}.parquet')
        ea = pd.read_parquet(f'{d}/sat_base/events.parquet').set_index('event_id')
        eb = pd.read_parquet(f'{d}/{arm}/events.parquet').set_index('event_id')
        m = ra.merge(rb[['event_id', 'cat_xy_5', 'cat_x_5', 'cat_y_5']], on='event_id', suffixes=('_a', '_b'))
        m = m[m['region'] == 'interior']
        for lab, sel in (('det->far', (m.cat_xy_5_a == 'detected') & (m.cat_xy_5_b == 'far')),
                         ('far->det', (m.cat_xy_5_a == 'far') & (m.cat_xy_5_b == 'detected'))):
            ev = m.loc[sel, 'event_id'].to_numpy()
            A, B = ea.loc[ev], eb.loc[ev]
            out = {}
            for v in 'xy':
                vb = m.loc[sel, f'cat_{v}_5_b'].to_numpy() == 'far'
                dp = np.abs(A[f'{v}_p0'].to_numpy() - B[f'{v}_p0'].to_numpy())
                out[v] = (int(vb.sum()), int((vb & (dp > 5)).sum()))
            spl = lambda E: int((E['n_splits'] > 0).sum()) if 'n_splits' in E else 0
            print(f"{det} {arm} {lab}: {len(ev)} rays; far view X {out['x'][0]} (p0 moved > 5 mm: {out['x'][1]}), "
                  f"Y {out['y'][0]} ({out['y'][1]}); split in base {spl(A)}, in arm {spl(B)}; "
                  f"n_tracks base/arm {A['n_tracks'].mean():.2f}/{B['n_tracks'].mean():.2f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
