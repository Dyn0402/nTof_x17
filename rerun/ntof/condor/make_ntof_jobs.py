#!/usr/bin/env python3
"""make_ntof_jobs.py -- job list for the n_TOF pass-1 reconstruction (NTOF_PASS_PLAN P6), one line per (acquisition, arm).

    make_ntof_jobs.py --bundles EOS_DIR --out EOS_DIR [--arms ABCD] [--rclass production,cosmic] [--runs 145,149]
                      [--subruns-per-run N] [--subruns a,b] [--per-row [--t0-class free]] [--extra "--limit-per-tag 300"]
                      > jobs.txt

Rows come from rerun/ntof/ntof_registry.csv (excluded rows dropped).  Bundle: <bundles>/<arm>, or with --per-row
<bundles>/<arm>_<E1|E2>_<cos|beam> chosen by the row's mask_era and rclass (cosmic rows -> the cosmic latency,
production -> the beam latency; make_pf_bundles.sh); --t0-class forces the class (e.g. free).  Output dir per row:
<out>/run_<run>/<subrun>/<arm>/.  Feed the file to ntof.sub (jobfile=).
"""
import argparse, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--bundles', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--arms', default='ABCD'); ap.add_argument('--rclass', default=None)
ap.add_argument('--runs', default=None); ap.add_argument('--subruns-per-run', type=int, default=None)
ap.add_argument('--tags-per-subrun', type=int, default=None)
ap.add_argument('--subruns', default=None, help='comma list of sub-runs to keep')
ap.add_argument('--per-row', action='store_true', help='bundle per (arm, mask era, trigger class)')
ap.add_argument('--t0-class', default=None, choices=['cos', 'beam', 'free'])
ap.add_argument('--extra', default='')
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(os.path.join(HERE, '..', 'ntof_registry.csv'))) if r['excluded'] != 'True']
if a.rclass:
    rows = [r for r in rows if r['rclass'] in a.rclass.split(',')]
if a.runs:
    rows = [r for r in rows if r['run'] in a.runs.split(',')]
if a.subruns:
    rows = [r for r in rows if r['subrun'] in a.subruns.split(',')]
CLASS = {'cosmic': 'cos', 'production': 'beam'}


def bundle(r, arm):
    if not a.per_row:
        return f'{a.bundles}/{arm}'
    era = r['mask_era'].split('_')[0]
    if era not in ('E1', 'E2'):
        raise SystemExit(f"row {r['run']} {r['subrun']} {r['tag']}: no mask_era")
    lc = a.t0_class or CLASS.get(r['rclass'])
    if lc is None:
        raise SystemExit(f"row {r['run']} {r['subrun']} {r['tag']}: rclass {r['rclass']} has no trigger class")
    return f'{a.bundles}/{arm}_{era}_{lc}'


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
        print(f"{r['run']} {r['subrun']} {r['tag']} {arm} {bundle(r, arm)} "
              f"{a.out}/run_{r['run']}/{r['subrun']}/{arm} {a.extra}".rstrip())
