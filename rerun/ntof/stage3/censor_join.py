#!/usr/bin/env python3
"""censor_join.py -- flag tracks whose fit window is made only of censored strips, and take them out of `gated`.

Found 2026-11-20 (rerun/ntof/improve/README.md, "D edge fakes"): a fit window whose every strip is a dead/hot wildcard
of the bundle (`n_flagged_strips == n_strips`) carries no charge measurement at all.  The fit puts unbounded charge
there (q_sum up to 1e22), chi2 ~ 0, and the candidate can still win the plane's ranking.  On D x this is 21 % of the
gated tracks of run_145: they sit on the last connector's edge strips (x_p0 ~ 395 mm, x_local ~ -196), have
tan x ~ 0 and a 3-strip seed, and the wall confirms them at the chance rate (0.24 against 0.47).  A, B, C have none.

This adds to each view table, from the canonical stage-2 candidates of the same rows (ntof_stage3.canonical):

    x_n_flagged, y_n_flagged   the winner's flagged (dead/hot) strip count per view (track_id >= 0 rows)
    x_cens_frac, y_cens_frac   their fraction of the window
    censored_window            x window all flagged, or y window >= 50 % flagged (X_FULL, Y_FRAC below)
    gated_uncensored           `gated` before this step
    gated                      gated & ~censored_window

It edits the view in place (atomic replace per table) and records itself in the sidecar.  Re-running is a no-op
for tables that already have the columns (--force redoes them from `gated_uncensored`).

    censor_join.py [--view $WORK_DATA/ntof_pass1/stage3_vt] [--jobs 12] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NT = HERE.parent
sys.path[:0] = [str(HERE), str(NT)]


#: the rule (chosen on D's late-window wall purity, run_145; rerun/ntof/improve/README.md): an x window that is ALL
#: censored, or a y window at least half censored.  x is not cut at 50 %: D x has two dead connectors (~100 strips)
#: that the seeder bridges by design (WFT_DEAD_SEED), and real tracks there need it.  Late D: keeps 96 % of the
#: wall-confirmed tracks, removes 16 % of all, purity 0.69 -> 0.79.  A/B/C gated tracks essentially never reach it.
X_FULL = 1.0
Y_FRAC = 0.5


def flags_for(run, sub, rows, canon, roots):
    import pandas as pd
    import ntof_ledger as L  # noqa: F401
    from ntof_stage3 import canonical
    parts = []
    for r in rows:
        for arm in ('A', 'B', 'C', 'D'):
            d, stem = canonical(r, arm, canon, roots)
            f = d / f'{stem}.candidates.parquet'
            if not f.exists():
                continue
            c = pd.read_parquet(f, columns=['event_id', 'plane', 'track_id', 'n_strips', 'n_flagged_strips'])
            c = c[c.track_id >= 0]
            w = c.pivot_table(index=['event_id', 'track_id'], columns='plane',
                              values=['n_strips', 'n_flagged_strips'], aggfunc='first')
            w.columns = [f'{p}_{v}' for v, p in w.columns]
            w = w.reset_index().assign(tag=r['tag'], arm=arm)
            parts.append(w)
    return pd.concat(parts, ignore_index=True) if parts else None


def one(path, rows, canon, roots, force):
    import pandas as pd
    path = Path(path)
    t = pd.read_parquet(path)
    if 'censored_window' in t.columns and not force:
        return path.name, 'done'
    if 'gated_uncensored' in t.columns:
        t['gated'] = t['gated_uncensored']
        t = t.drop(columns=[c for c in ('x_n_flagged', 'y_n_flagged', 'x_cens_frac', 'y_cens_frac', 'censored_window',
                                        'gated_uncensored') if c in t.columns])
    run, sub = str(t['run'].iloc[0]).replace('run_', ''), t['subrun'].iloc[0]
    F = flags_for(run, sub, rows, canon, roots)
    if F is None:
        raise FileNotFoundError(f'no candidates for run_{run} {sub}')
    F = F.rename(columns={'x_n_flagged_strips': 'x_n_flagged', 'y_n_flagged_strips': 'y_n_flagged',
                          'x_n_strips': '_xn', 'y_n_strips': '_yn'})
    m = t.merge(F, on=['tag', 'event_id', 'arm', 'track_id'], how='left', validate='many_to_one')
    if len(m) != len(t):
        raise AssertionError('join changed the row count')
    miss = int(m['_xn'].isna().sum())
    if miss:
        raise AssertionError(f'{path.name}: {miss} rows have no candidate')
    if not (np.array_equal(m['_xn'].to_numpy(float), m['x_n_strips'].to_numpy(float))
            and np.array_equal(m['_yn'].to_numpy(float), m['y_n_strips'].to_numpy(float))):
        raise AssertionError(f'{path.name}: candidate n_strips disagree with the table (wrong join)')
    fx = (m['x_n_flagged'] / m['x_n_strips'].where(m['x_n_strips'] > 0)).to_numpy(float)
    fy = (m['y_n_flagged'] / m['y_n_strips'].where(m['y_n_strips'] > 0)).to_numpy(float)
    with np.errstate(invalid='ignore'):
        cen = (fx >= X_FULL) | (fy >= Y_FRAC)
    t['x_cens_frac'], t['y_cens_frac'] = fx.astype('float32'), fy.astype('float32')
    t['x_n_flagged'] = m['x_n_flagged'].astype('int32').to_numpy()
    t['y_n_flagged'] = m['y_n_flagged'].astype('int32').to_numpy()
    t['censored_window'] = cen
    t['gated_uncensored'] = t['gated']
    t['gated'] = t['gated'].fillna(False).astype(bool).to_numpy() & ~cen
    tmp = path.with_suffix('.part')
    t.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, path)
    mj = path.with_suffix('.meta.json')
    meta = json.load(open(mj)) if mj.exists() else {}
    by = t.groupby('arm').agg(n=('censored_window', 'size'), cens=('censored_window', 'sum'),
                              gated_before=('gated_uncensored', 'sum'), gated=('gated', 'sum'))
    meta['censor'] = dict(rule=f'censored_window = x window all dead/hot (>= {X_FULL}) or y window >= {Y_FRAC} dead/hot; gated &= ~it',
                          by_arm={a: {k: int(v) for k, v in r.items()} for a, r in by.iterrows()},
                          built_by='rerun/ntof/stage3/censor_join.py')
    json.dump(meta, open(mj, 'w'), indent=1)
    return path.name, ' '.join(f'{a} {int(r.gated_before)}->{int(r.gated)}' for a, r in by.iterrows())


def main():
    import ntof_ledger as L
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view', default=f'{wd}/ntof_pass1/stage3_vt')
    ap.add_argument('--stage2', default=f'{wd}/ntof_pass1/stage2')
    ap.add_argument('--side', default=f'{wd}/ntof_pass1/run104')
    ap.add_argument('--tests', default=f'{wd}/ntof_pass1/tests')
    ap.add_argument('--rerun-dbeam', default=f'{wd}/ntof_pass1/rerun_dbeam_L1')
    ap.add_argument('--canon', default=str(NT / 'ledger_canon.json'))
    ap.add_argument('--only', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    canon = L.load_canon(a.canon)
    roots = {'stage2': a.stage2, 'side': a.side, 'tests': a.tests, 'rerun_dbeam_L1': a.rerun_dbeam}
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    fs = sorted(Path(a.view).glob(a.only or 'tracks_run_*.parquet'))
    with ProcessPoolExecutor(a.jobs) as ex:
        jobs = {}
        for f in fs:
            run, sub = f.stem.split('_')[2], '_'.join(f.stem.split('_')[3:])
            jobs[ex.submit(one, f, groups[(run, sub)], canon, roots, a.force)] = f
        for i, j in enumerate(as_completed(jobs)):
            name, msg = j.result()
            print(f'[{i + 1}/{len(fs)}] {name}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
