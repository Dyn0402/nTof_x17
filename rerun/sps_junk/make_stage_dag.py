#!/usr/bin/env python3
"""make_stage_dag.py <census table.csv> <code dir> <stage dir> > stage.dag

The SPS rows whose events the census (census.py / census_table.py) says change, as sps.sub nodes that write into a
staging dir (never Analysis_pass1): the same node as make_pass_dag.py sps (8 cpus, 16 GB, the pass's det4 bundle)."""
import csv, sys

BUNDLE = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
table, code, stage = sys.argv[1:4]
for r in csv.DictReader(open(table)):
    if r['changes'] != 'True':
        continue
    k = r['key']; n = f'SPS_{k}'
    print(f'JOB {n} sps.sub')
    print(f'VARS {n} code="{code}" args="{k} {stage}/{k} --bundle {BUNDLE}" cpus="8" mem="16000" node="{n}"')
    print(f'RETRY {n} 3')
