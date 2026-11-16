#!/usr/bin/env python3
"""make_vt_jobs.py -- P7a job list, one line per sub-run: <run> <subrun> <arms> <tags,...>  (vt.sub / vt_stage.sh).

A and C on every non-excluded registry row (beam and cosmic rows are kept apart at table level); D only on cosmic rows
(D's beam column is not a box: NTOF_PASS_PLAN §3 P7a); B never (no box).

    make_vt_jobs.py > vt_jobs.txt
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
reg = [r for r in csv.DictReader(open(os.path.join(HERE, '..', 'ntof_registry.csv'))) if r['excluded'] != 'True']
groups = {}
for r in reg:
    g = groups.setdefault((int(r['run']), r['subrun']), dict(rclass=set(), tags=[]))
    g['rclass'].add(r['rclass'])
    g['tags'].append(r['tag'])
for (run, sub), g in sorted(groups.items()):
    assert len(g['rclass']) == 1, (run, sub, g['rclass'])
    arms = 'ACD' if 'cosmic' in g['rclass'] else 'AC'
    print(run, sub, arms, ','.join(sorted(g['tags'])))
