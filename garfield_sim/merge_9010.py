#!/usr/bin/env python3
"""Merge per-mixture drift_9010_<key>.json files (from the condor array) into the
combined results/drift_9010_contam_cern.json schema {mixtures: {name: points}}."""
import os
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'results', 'contam_9010_parts')
OUT = os.path.join(HERE, 'results', 'drift_9010_contam_cern.json')

mixtures = {}
meta = {}
for p in sorted(glob.glob(os.path.join(SRC, 'drift_9010_*.json'))):
    d = json.load(open(p))
    mixtures[d['name']] = d['points']
    meta = dict(gas_base=d.get('gas_base'), pressure_torr=d.get('pressure_torr'),
                temp_K=d.get('temp_K'), ncoll=d.get('ncoll'))
meta['mixtures'] = mixtures
json.dump(meta, open(OUT, 'w'), indent=1)
print(f'merged {len(mixtures)} mixtures -> {OUT}')
print('  ' + ', '.join(sorted(mixtures)))
