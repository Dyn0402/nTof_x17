#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regen_stage3.py -- rebuild stage-3 tables on the fixed y convention, from their
own sidecars, and PROVE the rebuild is the 2026-09-07 mirror undone and nothing
else (HANDOFF_Y_FLIP_FIX.md §3).

Each ``tracks_<run>_<sub>.meta.json`` records what built it: the reco
directory, the stage-1 and allowlist inputs, the k applied per arm and the
stage-3 acceptance.  This re-runs ``build_tracks.build`` with exactly those,
then compares against the old table row by row:

  * x/z columns, gates, charges, times: identical;
  * p0_y, d_y, tany, target_y_mm, y_local: exactly negated;
  * pred_* and in_bore: may change (the scintillators and the capsule are not
    y-symmetric) -- counted and reported, not required to match.

Only when that holds for EVERY table in the directory are the old tables
replaced, all at once (kept under ``<dir>/pre_yfix/`` unless ``--drop-backup``);
otherwise nothing in the directory changes.  Constant columns a wrapper
added after the build (``k_source`` from cosmic_tracks) are carried over.

    python -m sept26_prelim_analysis.regen_stage3 <stage3 dir> [--jobs 8] [--combine]

``--combine`` re-concatenates ``tracks_campaign.parquet`` afterwards with the
same ``k_source`` label the old combined table had.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NEGATED = ('p0_y', 'd_y', 'tany', 'target_y_mm', 'y_local')
MAY_CHANGE = ('in_bore', 'pred_sipm_bar', 'pred_plastic', 'pred_ls', 'pred_n_cross',
              'pred_sipm_s_mm', 'gated', 'in_acceptance')
SKIP = ('built',)
KEY = ['tag', 'event_id', 'arm', 'track_id']
STAGING = '_yfix_staging'


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        x, y = a.to_numpy(float), b.to_numpy(float)
        return np.isclose(x, y, rtol=1e-9, atol=1e-7) | (np.isnan(x) & np.isnan(y))
    return (a.astype(object).to_numpy() == b.astype(object).to_numpy()) | (a.isna() & b.isna()).to_numpy()


def compare(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Row-matched comparison of an old (mirrored) and a new (up) table."""
    o = old.sort_values(KEY).reset_index(drop=True)
    n = new.sort_values(KEY).reset_index(drop=True)
    if len(o) != len(n) or not (o[KEY].values == n[KEY].values).all():
        return dict(ok=False, why=f'row sets differ: {len(o)} old vs {len(n)} new')
    bad, changed = {}, {}
    for c in n.columns:
        if c not in o.columns or c in SKIP:
            continue
        if c in NEGATED:
            m = _same(-o[c].astype(float), n[c].astype(float))
            if not m.all():
                bad[c] = int((~m).sum())
        elif c in MAY_CHANGE:
            m = _same(o[c], n[c])
            changed[c] = int((~m).sum())
        else:
            m = _same(o[c], n[c])
            if not m.all():
                bad[c] = int((~m).sum())
    y_ok = not any(c in bad for c in NEGATED)
    drift = {c: v for c, v in bad.items() if c not in NEGATED}
    if changed.get('gated', 0):
        drift['gated'] = changed['gated']
    return dict(ok=y_ok and not drift, y_ok=y_ok, n=len(n), mismatched=bad, drift=drift,
                changed=changed)


def one(table: str, allow_drift: bool = False) -> dict:
    from sept26_prelim_analysis import build_tracks as BT
    p = Path(table)
    meta = json.loads(p.with_name(p.name.replace('.parquet', '.meta.json')).read_text())
    if meta.get('y_convention') == BT.Y_CONVENTION:
        return dict(table=p.name, skipped='already up')
    old = pd.read_parquet(p)
    k = {a: float(v) for a, v in ((meta.get('k_arm') or {}).get('applied') or {}).items()}
    new, nmeta = BT.build(meta['run'], meta['subrun'], Path(meta['reco_dir']),
                          stage1=Path(meta['stage1']) if meta.get('stage1') else None,
                          allow=Path(meta['allowlist']) if meta.get('allowlist') else None,
                          k_arm=k, tan_max_true=meta.get('stage3_tan_max_true'),
                          write=False)
    for c in old.columns:                  # wrapper-added constants (k_source)
        if c not in new.columns and old[c].nunique(dropna=False) == 1:
            new[c] = old[c].iloc[0]
    cmp = compare(old, new)
    res = dict(table=p.name, **{k_: v for k_, v in cmp.items()},
               y_guard=nmeta.get('y_guard', {}).get('verdict'))
    if not (cmp['ok'] or (allow_drift and cmp['y_ok'])):
        return res                          # nothing staged
    # Stage only. The swap happens in main(), for the whole directory at once,
    # and only if every table passed: a directory half on each convention is
    # the one state a reader that ignores the flag cannot survive.
    res['staged'] = True
    st = p.parent / STAGING
    st.mkdir(exist_ok=True)
    new = new[[c for c in old.columns if c in new.columns]
              + [c for c in new.columns if c not in old.columns]]
    new.to_parquet(st / p.name, index=False)
    nmeta['regenerated_from'] = dict(built=meta.get('built'), schema=meta.get('schema'),
                                     check='exact y mirror of the previous table',
                                     changed=cmp['changed'], drift=cmp['drift'])
    (st / p.name.replace('.parquet', '.meta.json')).write_text(
        json.dumps(nmeta, indent=1, default=str))
    return res


def swap(src: Path, names: list, drop_backup: bool) -> None:
    """Move every old table to pre_yfix/ (or delete it) and the staged one in."""
    st, bdir = src / STAGING, src / 'pre_yfix'
    bdir.mkdir(exist_ok=True)
    for n in names:
        m = n.replace('.parquet', '.meta.json')
        for f in (n, m):
            if drop_backup:
                (src / f).unlink()
            else:
                (src / f).rename(bdir / f)
            (st / f).rename(src / f)
    st.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('src', type=Path)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--drop-backup', action='store_true')
    ap.add_argument('--allow-drift', action='store_true',
                    help='replace even if non-y columns differ from the old table (code '
                         'changed since it was built); y must still be an exact mirror. '
                         'The differences are recorded in regen_yfix.csv and the sidecar.')
    a = ap.parse_args()
    tables = sorted(str(p) for p in a.src.glob('tracks_run_*.parquet'))
    print(f'{len(tables)} table(s) under {a.src}')
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, t, a.allow_drift): t for t in tables}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:                                # noqa: BLE001
                r = dict(table=Path(futs[f]).name, ok=False, why=traceback.format_exc(limit=3))
            rows.append(r)
            if r.get('skipped'):
                msg = r['skipped']
            elif r.get('staged'):
                msg = (f"staged; changed {r.get('changed')}; drift {r.get('drift') or 'none'}; "
                       f"guard {r.get('y_guard')}")
            else:
                msg = f"!! FAILED: {r.get('why') or r.get('mismatched')}"
            print(f"  {r['table']}: {msg}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(a.src / 'regen_yfix.csv', index=False)
    n_fail = sum(1 for r in rows if not (r.get('skipped') or r.get('staged')))
    staged = [r['table'] for r in rows if r.get('staged')]
    if n_fail:
        print(f'\n{n_fail} table(s) FAILED: NOTHING replaced in {a.src} (staged copies in '
              f'{a.src / STAGING}; details in regen_yfix.csv). Fix, or --allow-drift if only '
              f'non-y columns differ.')
        return 1
    swap(a.src, staged, a.drop_backup)
    print(f'\nall {len(rows)} ok: {len(staged)} replaced, {len(rows) - len(staged)} already up'
          f' (old tables in {a.src / "pre_yfix"})')
    if a.combine and (a.src / 'tracks_campaign.parquet').exists():
        import pyarrow.parquet as pq
        ks = pq.read_table(a.src / 'tracks_campaign.parquet', columns=['k_source']).column(0)
        ks = sorted(set(ks.to_pylist()))
        if len(ks) != 1:
            print(f'!! old combined table has k_source {ks}; combine by hand')
            return 1
        import subprocess
        cmd = [sys.executable, '-m', 'sept26_prelim_analysis.campaign_tracks',
               '--out', str(a.src), '--combine-only', '--k-source', ks[0]]
        tm = json.loads((a.src / 'tracks_campaign.meta.json').read_text()).get('stage3_tan_max_true')
        if tm is not None:
            cmd += ['--tan-max-true', str(tm)]
        print(' '.join(cmd))
        return subprocess.call(cmd, cwd=REPO)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
