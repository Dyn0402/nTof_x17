#!/usr/bin/env python3
"""make_ntof_jobs.py -- job list for the n_TOF pass-1 reconstruction (NTOF_PASS_PLAN P6), one line per (acquisition, arm).

    make_ntof_jobs.py --bundles EOS_DIR --out EOS_DIR [--arms ABCD] [--rclass production,cosmic] [--runs 145,149]
                      [--subruns-per-run N] [--extra "--limit-per-tag 300"] > jobs.txt

Rows come from rerun/ntof/ntof_registry.csv (excluded rows dropped).  Bundle per arm: <bundles>/<arm>; output dir
per row: <out>/run_<run>/<subrun>/<arm>/.  Feed the file to ntof.sub (jobfile=).
"""
import argparse, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--bundles', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--arms', default='ABCD'); ap.add_argument('--rclass', default=None)
ap.add_argument('--runs', default=None); ap.add_argument('--subruns-per-run', type=int, default=None)
ap.add_argument('--tags-per-subrun', type=int, default=None)
ap.add_argument('--extra', default='')
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(os.path.join(HERE, '..', 'ntof_registry.csv'))) if r['excluded'] != 'True']
if a.rclass:
    rows = [r for r in rows if r['rclass'] in a.rclass.split(',')]
if a.runs:
    rows = [r for r in rows if r['run'] in a.runs.split(',')]
seen_sub, seen_tag = {}, {}
for r in sorted(rows, key=lambda r: (int(r['run']), r['subrun'], r['tag'])):
    subs = seen_sub.setdefault(r['run'], [])
    if r['subrun'] not in subs:
        if a.subruns_per_run and len(subs) >= a.subruns_per_run:
            continue
        subs.append(r['subrun'])
    tags = seen_tag.setdefault((r['run'], r['subrun']), [])
    if a.tags_per_subrun and len(tags) >= a.tags_per_subrun:
        continue
    tags.append(r['tag'])
    for arm in a.arms:
        print(f"{r['run']} {r['subrun']} {r['tag']} {arm} {a.bundles}/{arm} "
              f"{a.out}/run_{r['run']}/{r['subrun']}/{arm} {a.extra}".rstrip())
