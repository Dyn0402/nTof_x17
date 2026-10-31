#!/usr/bin/env python3
"""make_jobs.py <gc_rows.txt> > jobs.txt -- one line per bench row with a colsum.npz: colsum path, the row's pass-1
calib_bundle, output stem and the registry meta.  Sources, in order of preference (a row appears once):
  pass  colsum run with the PASS bundle (S4 golden rows, the v(t) cross-check rows)
  gc    colsum run with the audit bundle (the GC follow-up, 150 rows).  S(t) is a raw strip sum: the bundle only set the
        dead mask of the cluster finder; this script fits it with the row's own pass template.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = '/eos/experiment/ntof/data/x17/cosmic_bench'
A = f'{X}/Analysis_pass1'
OUT = f'{X}/x17rerun/paper1_timing/rows'
PASS_COLSUM = [
    f'{X}/x17rerun/s4_inputs/Analysis/{p}/fa/colsum.npz' for p in (
        'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'mx17_det4_day_6-24-26/long_run/mx17_4',
        'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7')] + [
    f'{X}/x17rerun/vt_check/Analysis/{p}/fa/colsum.npz' for p in (
        'mx17_det2_det3_overnight_6-22-26/long_run/mx17_2',
        'mx17_det3_det4_overnight_6-23-26/long_run/mx17_4',
        'mx17_det6_det7_overnight_6-26-26/longer_run/mx17_6',
        'mx17_det6_det7_overnight_6-26-26/longer_run/mx17_7')]

reg = {(r['run'], r['subrun'], r['det']): r for r in csv.DictReader(open(os.path.join(HERE, '..', 'pass_registry.csv')))}
by_key = {r['key']: r for r in reg.values()}
seen = set()
rows = []
for c in PASS_COLSUM:
    run, sub, det = c.split('/Analysis/')[1].split('/')[:3]
    rows.append((c, run, sub, det, 'pass'))
for line in open(sys.argv[1]):
    run, sub, det = line.strip().split('/')[:3]
    rows.append((f'{X}/x17rerun/followups/gc/Analysis/{run}/{sub}/{det}/gc/colsum.npz', run, sub, det, 'gc'))
for c, run, sub, det, src in rows:
    if (run, sub, det) in seen:
        continue
    r = reg.get((run, sub, det))
    if r is None:
        print(f'# no registry row: {run}/{sub}/{det}', file=sys.stderr); continue
    if r['cls'] == 'excluded':
        print(f'# excluded: {r["key"]}', file=sys.stderr); continue
    seen.add((run, sub, det))
    cr = by_key[r['calib_key']]          # rows without their own calibration read their calibration row's bundle
    meta = (f'key={r["key"]} det={det} run={run} subrun={sub} resist_V={r["resist_V"]} drift_V={r["drift_V"]} '
            f'cls={r["cls"]} source={src} calib_key={r["calib_key"]}')
    print(f'{c} {A}/{cr["run"]}/{cr["subrun"]}/{cr["det"]}/wft/calib_bundle {OUT}/{r["key"]} {meta}')
