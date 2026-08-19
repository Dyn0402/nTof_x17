#!/usr/bin/env python3
"""
merge_beam_tags.py — put the per-tag condor outputs back together.

The condor campaign runs one job per (arm, file tag); the local chain
(run145_target_imaging, make_run145_note) wants one events_prelim.parquet per
arm, exactly as the single-process driver would have written it.  This merges
them:

    <results>/beam_<arm>_<tag>.tar.gz   (or already-unpacked out/ trees)
        -> <analysis>/<run>/<subrun>/mx17_<arm>/events_prelim.parquet
                                                events_prelim.candidates.parquet
                                                events_prelim.meta.json
                                                calib_bundle_prelim/

It refuses to overwrite a table that is not already parked, and it refuses to
merge tags that were reconstructed with different bundles -- a table stitched
from two calibrations is the exact failure this whole re-run is fixing.

    ../../.venv/bin/python ntof_tracking/condor/merge_beam_tags.py \
        --results ~/x17/wft_beam145/results --arms A,B,D
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tarfile


def unpack(results, work):
    """Unpack any beam_<arm>_<tag>.tar.gz into <work>/; return <work>."""
    os.makedirs(work, exist_ok=True)
    for t in sorted(glob.glob(os.path.join(results, 'beam_*.tar.gz'))):
        with tarfile.open(t) as f:
            f.extractall(work)
    return work


def merge_arm(work, arm, out_dir, park_dir=None):
    import pandas as pd
    src = os.path.join(work, 'out', f'mx17_{arm}')
    tabs = sorted(glob.glob(os.path.join(src, 'events_*.parquet')))
    tabs = [t for t in tabs if not t.endswith('.candidates.parquet')]
    if not tabs:
        print(f'  arm {arm}: nothing to merge in {src}')
        return None

    metas = [json.load(open(t.replace('.parquet', '.meta.json'))) for t in tabs]
    hypers = {json.dumps(m['bundle']['hyper'], sort_keys=True) for m in metas}
    if len(hypers) != 1:
        sys.exit(f'FATAL: arm {arm}: {len(hypers)} different bundles across '
                 'tags — refusing to stitch one table out of two calibrations')

    dfs = [pd.read_parquet(t) for t in tabs]
    df = pd.concat(dfs, ignore_index=True).sort_values('event_id')
    dup = int(df['event_id'].duplicated().sum())
    df = df.reset_index(drop=True)

    cands = [pd.read_parquet(t.replace('.parquet', '.candidates.parquet'))
             for t in tabs
             if os.path.exists(t.replace('.parquet', '.candidates.parquet'))]

    os.makedirs(out_dir, exist_ok=True)
    if park_dir and os.path.exists(os.path.join(out_dir,
                                                'events_prelim.parquet')):
        if not os.path.isdir(park_dir):
            sys.exit(f'FATAL: {out_dir} already holds a table and it is not '
                     f'parked in {park_dir} — park it before overwriting '
                     '(mx_june_wft/FREEZE_MPGD26_2026-08-12.md §7)')

    df.to_parquet(os.path.join(out_dir, 'events_prelim.parquet'), index=False)
    if cands:
        pd.concat(cands, ignore_index=True).sort_values(
            ['event_id', 'plane', 'rank']).reset_index(drop=True).to_parquet(
            os.path.join(out_dir, 'events_prelim.candidates.parquet'),
            index=False)

    meta = dict(metas[0])
    meta.update(n_events=int(len(df)),
                n_seeded=int(sum(m['n_seeded'] for m in metas)),
                partial=False,
                status='PRELIMINARY',
                tags_done=sorted(t for m in metas for t in m['tags_done']),
                merged_from=[os.path.basename(t) for t in tabs],
                merged_by='ntof_tracking/condor/merge_beam_tags.py',
                n_duplicate_event_ids=dup)
    meta['run'] = dict(meta['run'], file_tags=meta['tags_done'])
    with open(os.path.join(out_dir, 'events_prelim.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    b = os.path.join(src, 'calib_bundle_prelim')
    if os.path.isdir(b):
        shutil.rmtree(os.path.join(out_dir, 'calib_bundle_prelim'),
                      ignore_errors=True)
        shutil.copytree(b, os.path.join(out_dir, 'calib_bundle_prelim'))

    h = meta['bundle']['hyper']
    r = h.get('c2_over_c1')
    c2 = float(r) * h['c1'] if r is not None else h['c2']
    print(f'  arm {arm}: {len(tabs)} tags -> {len(df):,} events '
          f'(seeded {meta["n_seeded"]:,}), c2/c1={c2 / h["c1"]:.3f}, '
          f'v={meta["bundle"]["v_drift"]}' +
          (f', {dup} duplicate event_ids' if dup else ''))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/home/dylan/x17/wft_beam145/results')
    ap.add_argument('--work', default=None, help='unpack dir (default '
                                                 '<results>/unpacked)')
    ap.add_argument('--analysis',
                    default='/media/dylan/data/x17/beam_july/analysis/wft/')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arms', default='A,B,D')
    ap.add_argument('--park', default='pre_r06_backup_20260819')
    a = ap.parse_args()

    work = a.work or os.path.join(a.results, 'unpacked')
    unpack(a.results, work)
    print(f'merging {a.run}/{a.subrun} from {a.results}')
    for arm in a.arms.split(','):
        out_dir = os.path.join(a.analysis, a.run, a.subrun, f'mx17_{arm}')
        merge_arm(work, arm, out_dir,
                  park_dir=os.path.join(out_dir, a.park) if a.park else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
