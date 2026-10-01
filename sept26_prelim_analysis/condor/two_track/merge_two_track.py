#!/usr/bin/env python3
"""
merge_two_track.py -- bring the two-track condor results home and merge them.

    .venv/bin/python sept26_prelim_analysis/condor/two_track/merge_two_track.py [--no-fetch]

1. rsync lxplus:/eos/user/d/dneff/x17/two_track_limit/results/ -> <pkg>/results/
2. check every job of <pkg>/jobs.txt has its tarball (lists the missing ones)
3. per variant, concatenate the shards into ~/x17/sept26_prelim/intra_bench/:
     bench    <variant>/                 overlays, candidates, splits, donors,
                                         build.meta.json (tags, shard count)
     split-ab split_ab_<variant>_7tags/  events, tracks, summary.csv (the same
                                         split_ab_summary as a local run), plus
                                         summary_by_tag.csv
   Incomplete variants are merged anyway and marked (``complete: false``).
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
from sept26_prelim_analysis import intra_bench as ib, paths  # noqa: E402

EOS_RESULTS = '/eos/user/d/dneff/x17/two_track_limit/results/'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default=str(paths.spell('x17', 'two_track_condor')))
    ap.add_argument('--no-fetch', action='store_true')
    a = ap.parse_args()
    res = os.path.join(a.pkg, 'results')
    os.makedirs(res, exist_ok=True)
    if not a.no_fetch:
        subprocess.run(['rsync', '-a', f'lxplus:{EOS_RESULTS}', res + '/'], check=True)

    jobs = [l.split() for l in open(os.path.join(a.pkg, 'jobs.txt')) if l.strip()]
    groups = defaultdict(list)            # (kind, variant) -> [(tag, outname)]
    for kind, arm, tag, outname, _flav, *extra in jobs:
        variant = extra[extra.index('--variant') + 1]
        groups[(kind, variant)].append((tag, outname))

    ext = os.path.join(res, 'extracted')
    os.makedirs(ext, exist_ok=True)
    for (kind, variant), items in sorted(groups.items()):
        have = [(t, o) for t, o in items if os.path.exists(os.path.join(res, f'{o}.tar.gz'))]
        missing = [o for t, o in items if (t, o) not in have]
        print(f'[merge] {kind} {variant}: {len(have)}/{len(items)} shards'
              + (f'  MISSING {missing[:4]}{" ..." if len(missing) > 4 else ""}' if missing else ''))
        if not have:
            continue
        for _t, o in have:
            if not os.path.isdir(os.path.join(ext, o)):
                with tarfile.open(os.path.join(res, f'{o}.tar.gz')) as tf:
                    tf.extractall(ext)
        dirs = [(t, os.path.join(ext, o)) for t, o in have]
        if kind == 'bench':
            od = ib.out_dir(variant)
            for name in ('overlays', 'candidates', 'splits'):
                parts = [pd.read_parquet(os.path.join(d, f'{name}.parquet'))
                         for _t, d in dirs if os.path.exists(os.path.join(d, f'{name}.parquet'))]
                if parts:
                    pd.concat(parts, ignore_index=True).to_parquet(od / f'{name}.parquet',
                                                                    index=False)
            pd.read_parquet(os.path.join(dirs[0][1], 'donors.parquet')).to_parquet(
                od / 'donors.parquet', index=False)
            meta = json.load(open(os.path.join(dirs[0][1], 'build.meta.json')))
            metas = [json.load(open(os.path.join(d, 'build.meta.json'))) for _t, d in dirs]
            meta.update(only_tag='', merged_from_tags=sorted(t for t, _d in dirs),
                        complete=not missing, n_overlays=sum(m['n_overlays'] for m in metas),
                        n_split_attempts=sum(m['n_split_attempts'] for m in metas),
                        n_splits=sum(m['n_splits'] for m in metas),
                        ran_on='lxplus condor (two_track.sub)',
                        note='noise-control rows use per-tag generators (only_tag shards)')
            (od / 'build.meta.json').write_text(json.dumps(meta, indent=1))
            print(f'         -> {od}  ({meta["n_overlays"]:,} overlays)')
        else:
            od = ib.out_dir(f'split_ab_{variant}_7tags')
            E = pd.concat([pd.read_parquet(os.path.join(d, 'events.parquet')) for _t, d in dirs],
                          ignore_index=True)
            T = pd.concat([pd.read_parquet(os.path.join(d, 'tracks.parquet')) for _t, d in dirs],
                          ignore_index=True)
            E.to_parquet(od / 'events.parquet', index=False)
            T.to_parquet(od / 'tracks.parquet', index=False)
            ib.split_ab_summary(E, T, od)
            rows = []
            (od / '_tmp').mkdir(exist_ok=True)
            for tag, e in E.groupby('tag'):
                s = ib.split_ab_summary(e, T[T.tag == tag], od / '_tmp').assign(tag=tag)
                rows.append(s)
            pd.concat(rows).to_csv(od / 'summary_by_tag.csv', index=False)
            (od / '_tmp' / 'summary.csv').unlink(missing_ok=True)
            (od / '_tmp').rmdir()
            (od / 'merge.meta.json').write_text(json.dumps(dict(
                variant=variant, shards=len(have), expected=len(items), complete=not missing,
                ran_on='lxplus condor (two_track.sub)'), indent=1))
            print(f'         -> {od}')


if __name__ == '__main__':
    main()
