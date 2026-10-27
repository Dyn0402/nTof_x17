#!/usr/bin/env python3
"""legacy_identity.py <gate_full> -- does the refactored wft, with every new piece switched off through wft.diag (gate arm
'legacy'), reproduce the 2026-10-27 production arm event by event?  Same fit = p0, tan, t0 equal to 1e-6 in both views
(float round-off between condor nodes is ~1e-11).  The production arm ran code 440a76c, before the F7 column window, so
the differences left are the F7 change itself: a different candidate selected where the old window called the muon
implausible (det6/det4 the most), and n_tracks, which keys on plausibility.
"""
import sys
import numpy as np
import pandas as pd

COLS = ['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_t0', 'y_t0']


def main(root):
    for d in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = pd.read_parquet(f'{root}/{d}/prod/events.parquet').set_index('event_id')
        q = pd.read_parquet(f'{root}/{d}/legacy/events.parquet').set_index('event_id')
        j = p.join(q, rsuffix='_l', how='inner')
        same = np.ones(len(j), bool)
        for c in COLS:
            a, b = j[c].to_numpy(), j[c + '_l'].to_numpy()
            same &= np.isclose(a, b, rtol=1e-6, atol=1e-6) | (np.isnan(a) & np.isnan(b))
        print(f'{d}: {len(p)} / {len(q)} events, same fit {100 * same.mean():.2f} %, '
              f'same n_tracks {100 * (j.n_tracks == j.n_tracks_l).mean():.2f} %')


if __name__ == '__main__':
    main(sys.argv[1])
