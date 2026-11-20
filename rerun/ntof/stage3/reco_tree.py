#!/usr/bin/env python3
"""reco_tree.py -- the pass-1 stage-2 output in the layout the downstream analysis reads (merged per sub-run and arm).

The analysis code of the prod / is2_v2 era (det_a_intra, acceptance.Chambers, source_imaging.plane_occupancy,
campaign_imaging.subruns_of, k_arm, ...) reads one merged table per (sub-run, arm):

    <tree>/run_<r>/<sub>/mx17_<arm>/events_prelim.parquet      (every tag, a `tag` column; = merge_fullpass.py)

This builds that tree from the pass's CANONICAL stage-2 outputs, chosen exactly as stage 3 chooses them
(ntof_stage3.canonical over ledger_canon.json: D run_104 from side/run104_D_L1, D L2-era beam from rerun_dbeam_L1).
Event ids are unique within a sub-run across tags, so the concatenation needs no re-keying.  A sub-run whose arm has
no canonical rows (e.g. run_104 stat090_0016 A/C, 0-byte acquisitions) gets no table for that arm, which the old
readers already treat as "not usable".  A sidecar `reco_tree.json` records the canon and the inputs.

    reco_tree.py [--stage2 $WORK_DATA/ntof_pass1/stage2] [--side $WORK_DATA/ntof_pass1/run104]
                 [--out $WORK_DATA/ntof_pass1/reco_tree] [--runs 145,149] [--jobs 8] [--force]
"""
import argparse, csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [HERE, NT]

import ntof_ledger as L       # noqa: E402
from ntof_stage3 import canonical  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


def one(run, sub, rows, canon, roots, out, force):
    made, missing = [], []
    for arm in ARMS:
        dst = Path(out) / f'run_{run}' / sub / f'mx17_{arm}' / 'events_prelim.parquet'
        if dst.exists() and not force:
            continue
        parts = []
        for r in rows:
            d, stem = canonical(r, arm, canon, roots)
            f = d / f'{stem}.parquet'
            if not f.exists():
                missing.append(str(f))
                continue
            parts.append(pd.read_parquet(f).assign(tag=r['tag']))
        if not parts:
            continue
        T = pd.concat(parts, ignore_index=True)
        if 'acq_tag' in T.columns:
            T = T.drop(columns='acq_tag')
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix('.part')
        T.to_parquet(tmp, index=False)
        os.replace(tmp, dst)
        made.append(arm)
    return run, sub, made, missing


def main():
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage2', default=f'{wd}/ntof_pass1/stage2')
    ap.add_argument('--side', default=f'{wd}/ntof_pass1/run104')
    ap.add_argument('--tests', default=f'{wd}/ntof_pass1/tests')
    ap.add_argument('--rerun-dbeam', default=f'{wd}/ntof_pass1/rerun_dbeam_L1')
    ap.add_argument('--out', default=f'{wd}/ntof_pass1/reco_tree')
    ap.add_argument('--canon', default=os.path.join(NT, 'ledger_canon.json'))
    ap.add_argument('--runs', default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    canon = L.load_canon(a.canon)
    roots = {'stage2': a.stage2, 'side': a.side, 'tests': a.tests, 'rerun_dbeam_L1': a.rerun_dbeam}
    reg = [r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv'))) if r['excluded'] != 'True']
    if a.runs:
        reg = [r for r in reg if r['run'] in a.runs.split(',')]
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    os.makedirs(a.out, exist_ok=True)
    miss_all = []
    with ProcessPoolExecutor(a.jobs) as ex:
        fs = [ex.submit(one, run, sub, rows, canon, roots, a.out, a.force) for (run, sub), rows in sorted(groups.items())]
        for i, f in enumerate(as_completed(fs)):
            run, sub, made, missing = f.result()
            miss_all += missing
            print(f'[{i + 1}/{len(fs)}] run_{run} {sub}: {"".join(made) or "-"}'
                  + (f'  MISSING {len(missing)}' if missing else ''), flush=True)
    json.dump(dict(canon=canon, roots=roots, registry='rerun/ntof/ntof_registry.csv', n_subruns=len(groups),
                   missing=miss_all, layout='run_<r>/<sub>/mx17_<arm>/events_prelim.parquet (tag column)'),
              open(os.path.join(a.out, 'reco_tree.json'), 'w'), indent=1)
    print(f'{len(miss_all)} canonical files missing' if miss_all else 'all canonical files present')
    return 1 if miss_all else 0


if __name__ == '__main__':
    sys.exit(main())
