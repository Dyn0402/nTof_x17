#!/usr/bin/env python3
"""make_jobs.py -- stack_jobs.in: <key> <row> <alignment row> <bundle row> for every bench row the gas study stacks (pg_stack.sh).

The rows: det3's 6-27 drift scan (300-1100 V), every det6/det7 row of 6-26 (HV scan and overnight), det4 6-23 (three
runs) and 6-24 (g_det4), det2 6-22 (three runs).  A row without its own alignment (align_key another row: det4 6-23
short_run, det6 400/410 V) has no stage-1 products either and takes the align_key row's alignment and the calib_key
row's bundle.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, '..', '..', 'pass_registry.csv')


def want(r):
    if r['cls'] == 'excluded':
        return False
    if 'saturday_scan' in r['run']:
        return 'drift_scan' in r['subrun'] and r['drift_V'] != '100'
    if '6-26' in r['run']:
        return r['det'] in ('mx17_6', 'mx17_7')
    if 'det3_det4_overnight_6-23' in r['run'] or '6-24' in r['run']:
        return r['det'] == 'mx17_4' and r['subrun'].endswith('run')
    if '6-22' in r['run']:
        return r['det'] == 'mx17_2' and r['subrun'].endswith('run')
    return False


R = list(csv.DictReader(open(REG)))
K = {r['key']: r for r in R}
path = lambda r: f"{r['run']}/{r['subrun']}/{r['det']}"

def line(r):
    own = r.get('align_key') in (None, '', r['key']) or r['align_key'] not in K
    return f"{r['key']} {path(r)} {path(r if own else K[r['align_key']])} {path(r if own else K[r['calib_key']])}"


L = [line(r) for r in R if want(r)]
open(os.path.join(HERE, 'stack_jobs.in'), 'w').write('\n'.join(L) + '\n')
print(len(L), 'rows;', sum(l.split()[1] != l.split()[2] for l in L), "take another row's alignment")
