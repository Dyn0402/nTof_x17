#!/usr/bin/env python3
"""ped_guard_fleet.py <code dir> <bench_root> <out.json> -- F11 fleet scan: wft.reco.pedestal_guard's connector ratios
(hit noise / data-run waveform noise) on every non-excluded pass_registry row, first acquisition of the rr1 inputs.
Lists the rows and connectors the guard corrects in the pass."""
import csv, glob, json, os, sys, types
sys.path.insert(0, sys.argv[1])
import numpy as np, uproot
from wft import reco

root, out = sys.argv[2], sys.argv[3]
man = {r['key']: r for r in csv.DictReader(open(os.path.join(sys.argv[1], 'mx_june_wft/condor/campaign_manifest.csv')))}
reg = [r for r in csv.DictReader(open(os.path.join(sys.argv[1], 'rerun/pass_registry.csv'))) if r['cls'] != 'excluded']
res = {}
for r in reg:
    m = man.get(r['key'])
    if m is None:          # golden rows carry short keys
        m = next((x for x in man.values() if x['run'] == r['run'] and x['subrun'] == r['subrun'] and x['det'] == r['det']), None)
    sd = os.path.join(root, 'june_tests', r['run'], r['subrun'])
    comb = sorted(glob.glob(os.path.join(sd, 'combined_hits_root_rr1_cm0', '*_datrun_*.root')))
    if not comb or m is None:
        res[r['key']] = 'no inputs'; print(r['key'], 'no inputs', flush=True); continue
    fx, fy = int(m['feu_x']), int(m['feu_y'])
    stem = os.path.basename(comb[0]).split('_feu-combined')[0]
    files = {f: sorted(glob.glob(os.path.join(sd, 'decoded_root_fa80c06', f'{stem}_{f:02d}.root'))) for f in (fx, fy)}
    h = uproot.open(comb[0])['hits'].arrays(['eventId', 'feu', 'channel', 'amplitude', 'significance'], library='pd')
    h = h[h['feu'].isin((fx, fy))]
    try:
        _, rep = reco.pedestal_guard(h, types.SimpleNamespace(), files_by_feu=files)
    except Exception as e:
        res[r['key']] = f'error {e}'; print(r['key'], 'error', e, flush=True); continue
    res[r['key']] = rep
    print(r['key'], 'CORRECTED ' + json.dumps(rep['corrected']) if rep['corrected'] else 'ok',
          {f: min(v.values()) if v else None for f, v in rep['ratios'].items()}, flush=True)
json.dump(res, open(out, 'w'), indent=1)
