#!/usr/bin/env python3
"""vtc_manifest.py [--root R] [--out CSV] -- the one list of bench rows that carry a v(t) correction (VT_PLAN §4).

Scans <R>/<run>/<subrun>/<det>/vt_map.json (R = /eos/user/d/dneff/x17rerun/vt_correction, the single location of the
corrected tables) and writes one line per row: registry key, model, estimator, clock, source, the v_rel range, whether
the table also carries the timing v, and the survey's verdict (rerun/vt_all/results/bench_vt.csv).  The repo copy,
rerun/vt_all/results/vt_corrected_rows.csv, is the source of truth for "is this row corrected"; MANIFEST.csv sits
next to the tables.
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ROOT = '/eos/user/d/dneff/x17rerun/vt_correction'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default=os.path.join(REPO, 'rerun', 'vt_all', 'results', 'vt_corrected_rows.csv'))
    a = ap.parse_args()
    reg = {(r['run'], r['subrun'], r['det']): r['key'] for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    sv = {r['key']: r['verdict'] for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'vt_all', 'results', 'bench_vt.csv')))}
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, '*', '*', '*', 'vt_map.json'))):
        run, sub, det = f.split(os.sep)[-4:-1]
        m = json.load(open(f))
        key = reg.get((run, sub, det), '')
        rows.append(dict(key=key, run=run, subrun=sub, det=det, model=m['model'],
                         estimator=m.get('estimator', 'M3 ladder (angle scale)'), clock=m['clock'],
                         source=m['source'], v_rel_min=round(min(m['v_rel']), 4), v_rel_max=round(max(m['v_rel']), 4),
                         timing_v=bool(m.get('timing')), survey_verdict=sv.get(key, ''),
                         table=os.path.join(os.path.dirname(f), 'events_vt.parquet')))
    for out in (a.out, os.path.join(a.root, 'MANIFEST.csv')):
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} corrected rows -> {a.out}')


if __name__ == '__main__':
    main()
