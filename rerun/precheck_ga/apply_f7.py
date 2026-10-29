#!/usr/bin/env python3
"""apply_f7.py -- write the F7 verdicts (README.md) into the bench masks the pass bundles read
(model_audit/followups/fe_unify2/masks/bench_det{3,4}.json): GA channels found healthy leave the dead / hot lists;
confirmed and inconclusive ones stay.  Idempotent."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MASKS = os.path.join(HERE, '..', 'model_audit', 'followups', 'fe_unify2', 'masks')
DROP = {3: {'x': {'hot': [62]}, 'y': {'dead': [41]}},
        4: {'x': {'dead': [125, 196, 376, 455, 498], 'hot': [274]}}}
KEEP = {3: {'y': {'dead': [510]}}, 4: {'x': {'dead': [94, 400, 435, 439, 472]}}}
NOTE = ('F7 2026-10-29 (rerun/precheck_ga/README.md): flank fired fraction + amplitude ratio (two-sided and mirror) on '
        'rr1_cm0 hits and fa80c06 waveforms of the golden row, waveform display each. Healthy -> dropped; confirmed weak '
        'or inconclusive -> kept censored (a censored healthy strip costs little, a weak strip fitted as live biases).')

for det, views in DROP.items():
    p = os.path.join(MASKS, f'bench_det{det}.json')
    m = json.load(open(p))
    for view in set(views) | set(KEEP.get(det, {})):
        pl = m['planes'][view]
        for kind, chs in views.get(view, {}).items():
            pl[kind] = [c for c in pl[kind] if c not in chs]
        ga = pl['ga_added']
        ga['f7_dropped'] = views.get(view, {})
        ga['f7_kept'] = KEEP.get(det, {}).get(view, {})
        ga['verify'] = NOTE
    json.dump(m, open(p, 'w'), indent=1)
    print(p, {v: (m['planes'][v]['ga_added']['f7_dropped'], m['planes'][v]['ga_added']['f7_kept']) for v in m['planes'] if 'ga_added' in m['planes'][v]})
