#!/usr/bin/env python3
"""
make_matched_lists.py — compute each row's M3-matched event ids LOCALLY, so the
condor job never has to read the rays files itself.

Why this exists: LCG_105 (python 3.9, uproot 4.3, awkward 1.10) mis-resolves
the NClus branches of **v1** rays files. The branches are there — a modern
uproot reads NClusX/NClusY out of them fine — but on the worker stack the
recipe silently degrades toward chi2-only and the job reconstructs a DIFFERENT
event set than every local accounting (g_det3_wknd: 36,745 job events vs 26,670
recipe-passing locally, 5.3 % of good rays absent; the det2 long_run row shows
the same disagreement with a 2.8 % surplus). 184 of the 214 manifest rows have
no m3_tracking_root_v2, so this is not a corner case.

Reco needs nothing from M3 except which event ids to fit — reference positions
are attached downstream by 01_alignment/03_angles, which run locally — so
shipping a list is a complete fix and removes the worker's stack dependency
entirely.

    ../../.venv/bin/python mx_june_wft/condor/make_matched_lists.py \
        --rows 0,3,7 --out <dir>
    ../../.venv/bin/python mx_june_wft/condor/make_matched_lists.py \
        --tier A --v1-only --out <dir>

Writes <out>/matched_row<NNN>.json, consumed by run_reco_job.py --matched-list.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS, setup_paths  # noqa: E402
setup_paths()
from M3RefTracking import M3RefTracking, get_xy_angles        # noqa: E402

BENCH = '/home/dylan/x17/cosmic_bench'


def m3_dir_for(row):
    """The M3 tracking dir as the campaign job would resolve it, but local."""
    base = os.path.join(BENCH, row['tree'], row['run'], row['subrun'])
    name = ('m3_tracking_root_v2' if row['has_m3v2'] == '1'
            else 'm3_tracking_root')
    return os.path.join(base, name) + os.sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',
                    default=os.path.join(HERE, 'campaign_manifest.csv'))
    ap.add_argument('--rows', default=None, help='comma-separated row indices')
    ap.add_argument('--tier', default=None)
    ap.add_argument('--v1-only', action='store_true',
                    help='restrict to rows with no m3_tracking_root_v2')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    idx = (list(range(len(rows))) if args.rows is None
           else [int(x) for x in args.rows.split(',')])
    if args.tier:
        idx = [i for i in idx if rows[i]['tier'] == args.tier]
    if args.v1_only:
        idx = [i for i in idx if rows[i]['has_m3v2'] != '1']

    os.makedirs(args.out, exist_ok=True)
    recipe = f'chi2<{M3_CHI2_CUT} & NClus>={M3_MIN_NCLUS}'
    print(f'{len(idx)} rows; recipe {recipe}\n')
    ok, bad = 0, []
    for i in idx:
        row = rows[i]
        d = m3_dir_for(row)
        tag = f'row {i:3d}  {row["det"]:8s} {row["run"]}/{row["subrun"]}'
        if not os.path.isdir(d):
            print(f'{tag}  SKIP: no M3 dir {d}')
            bad.append((i, 'no m3 dir'))
            continue
        try:
            rays = M3RefTracking(d, chi2_cut=M3_CHI2_CUT,
                                 min_nclus=M3_MIN_NCLUS)
            _xa, _ya, evn = get_xy_angles(rays.ray_data)
        except Exception as e:                     # noqa: BLE001
            print(f'{tag}  FAIL: {type(e).__name__}: {e}')
            bad.append((i, f'{type(e).__name__}: {e}'))
            continue
        ids = sorted({int(e) for e in evn})
        if not ids:
            # Rows 90/91 of the original campaign: telescope not recording.
            # An empty list would crash reco at reco.py:575 on an empty match
            # list — refuse here where the reason is legible.
            print(f'{tag}  FAIL: 0 matched events (telescope off?)')
            bad.append((i, '0 matched'))
            continue
        if not bool(getattr(rays, 'has_nclus', True)):
            print(f'{tag}  FAIL: local stack also lacks NClus')
            bad.append((i, 'no nclus locally'))
            continue
        p = os.path.join(args.out, f'matched_row{i:03d}.json')
        with open(p, 'w') as f:
            json.dump(dict(row=i, key=row['key'], det=row['det'],
                           run=row['run'], subrun=row['subrun'],
                           recipe=recipe, source=d, n=len(ids),
                           event_ids=ids), f)
        print(f'{tag}  {len(ids):,} matched -> {os.path.basename(p)}')
        ok += 1
    print(f'\n{ok} lists written, {len(bad)} failed')
    for i, why in bad:
        print(f'  row {i}: {why}')
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
