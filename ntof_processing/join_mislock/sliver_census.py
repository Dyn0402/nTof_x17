#!/usr/bin/env python3
"""Census of the 78 non-OK sliver segments: which are really 'sliver-class'?

Input: the campaign's inventory.csv (one row per attempted segment).

Two results (2026-08-12):

1. The 'n_TOF run START' hypothesis is DEAD. Orienting every straddling
   segment (smaller ntof_run = the sub-run's early slice = that n_TOF run's
   END; larger = the next run's START), failures split exactly 33/33
   between run_END and run_START. Both orientations also fit 20/21 times.
   There is nothing special about a run's first minutes.

2. The 78 non-OK sliver segments are TWO populations, split by whether the
   same DREAM sub-run fitted on its other side (which pins the shared
   pulse_match offset):
   - SIBLING_OK (42): offset proven right, join proven right, still no
     sharp coincidence. All small: <= 402 joined bunches, median 158,
     overlap median 0.26. The true mystery class (run_79/0002 x 224573 is
     one). NOT recovered by the pulse_match fix.
   - no OK sibling (36 segments on 24 sub-runs): nothing on the sub-run
     fitted, so the offset was never verified. Includes every LARGE
     'sliver' failure (up to 1101 bunches / 55.6 min) and all four dark
     DREAM runs that appear among slivers (run_126/128/150/156). Cached v1
     pulse_match offsets betray mislocks: run_132/0007 locked -68.08 s
     where its run's OK sub-run locked +48.32 (-68.08 + 3x39.6 = +50.7);
     run_139/0007 locked -48.88 vs 0003's +67.5 (+118.8 apart). These are
     the SAME supercycle mislock as the whole-hour class, classified
     'sliver' only because the sub-run straddles a boundary -- the fixed
     pulse_match should recover them by plain re-run.

Usage: sliver_census.py [inventory.csv]
"""
import sys
from pathlib import Path

import pandas as pd

INV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/media/dylan/data/x17/slim_campaign_2026-08-12/inventory.csv')


def main():
    df = pd.read_csv(INV)
    sl = df[df['kind'] == 'sliver'].copy()

    rows = []
    for (dr, sr), g in sl.groupby(['dream_run', 'sub_run']):
        g = g.sort_values('ntof_run')
        for i, (_, r) in enumerate(g.iterrows()):
            ori = ('single' if len(g) == 1
                   else 'run_END' if i == 0 else 'run_START')
            sib = g[g.ntof_run != r.ntof_run]
            sib_ok = (sib.status == 'OK').any()
            rows.append(dict(dream_run=dr, sub_run=sr, ntof_run=r.ntof_run,
                             status=r.status, orientation=ori,
                             ovl=r.overlap_frac, nb=r.joined_bunches,
                             sibling='SIBLING_OK' if sib_ok else
                             ('NO_SIBLING' if len(sib) == 0
                              else 'SIBLING_ALSO_FAILED')))
    t = pd.DataFrame(rows)
    f = t[t.status != 'OK']

    print('failures by orientation (run-start hypothesis test):')
    print(f.groupby('orientation').size().to_string(), '\n')
    print('failures by sibling status:')
    print(f.groupby('sibling').size().to_string(), '\n')

    myst = f[f.sibling == 'SIBLING_OK']
    mis = f[f.sibling != 'SIBLING_OK']
    print(f'mystery class (sibling fitted): n={len(myst)}, '
          f'bunches median {myst.nb.median():.0f} max {myst.nb.max()}, '
          f'overlap median {myst.ovl.median():.2f}')
    print(f'mislock candidates (no fitted sibling): n={len(mis)} on '
          f'{mis.groupby(["dream_run", "sub_run"]).ngroups} sub-runs, '
          f'bunches median {mis.nb.median():.0f} max {mis.nb.max()}')
    print('\nmislock-candidate sub-runs (feed these to the fixed re-slim):')
    for (dr, sr), g in mis.groupby(['dream_run', 'sub_run']):
        print(f'  {dr}/{sr}: ' + ', '.join(
            f'x{r.ntof_run} {r.status} ovl {r.ovl:.2f} nb {r.nb}'
            for _, r in g.iterrows()))
    print('\nmystery-class segments (NOT recovered by the pulse_match fix):')
    for _, r in myst.sort_values(['dream_run', 'sub_run']).iterrows():
        print(f'  {r.dream_run}/{r.sub_run} x {r.ntof_run}: {r.status} '
              f'ovl {r.ovl:.2f} nb {r.nb} ({r.orientation})')


if __name__ == '__main__':
    main()
