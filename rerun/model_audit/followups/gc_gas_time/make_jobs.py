#!/usr/bin/env python3
"""make_jobs.py -- GC condor job list: colsum.py on every non-excluded bench row of the pass registry for det2/3/4/6/7,
except the nine rows FA already ran (their colsum.npz is read from the FA tree)."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, '..', '..', '..', 'pass_registry.csv')
CODE = '/eos/user/d/dneff/x17rerun/followups/fa/code_2ce1dcc'
EOSB = '/eos/user/d/dneff/x17rerun/followups/fa/bundles'
FA_DONE = {'o22_long_det2', 'sat_det3', 'g_det4', 'g_det6_long', 'g_det7_long'} | {
    f'c26__mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_{v}V__mx17_3' for v in (500, 700, 900, 1100)}
rows = [r for r in csv.DictReader(open(REG)) if r['cls'] != 'excluded' and r['det'] != 'mx17_1']
L = []
for r in rows:
    if r['key'] in FA_DONE:
        continue
    det = 'det' + r['det'][-1]
    L.append(f"{CODE} {r['key']} {EOSB}/{det}_base")
open(os.path.join(HERE, 'gc_jobs.txt'), 'w').write('\n'.join(L) + '\n')
print(len(L), 'jobs -> gc_jobs.txt')
