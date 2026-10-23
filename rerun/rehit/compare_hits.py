#!/usr/bin/env python3
"""compare_hits.py <jobs.txt> <out.json> -- W9 re-hit against the production hits, and cm5 against cm0.

Per acquisition (combined hits): entries and events in the production `combined_hits_root/`, in `_rr1_cm0` and in
`_rr1_cm5`; whether cm0 equals production exactly (expected on nights whose decode did not change), and if not, how
many events differ.  Per run the cm5/cm0 hit ratio.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import uproot

COLS = ['eventId', 'channel', 'amplitude', 'time']


def load(p):
    if not os.path.exists(p):
        return None
    for k in range(4):
        try:
            return uproot.open(p)['hits'].arrays(COLS, library='np')
        except OSError:
            import time
            time.sleep(3 + 3 * k)
    return None


def main():
    out = dict(acq={}, run=defaultdict(lambda: defaultdict(int)))
    for line in open(sys.argv[1]):
        sd, tag, _ = line.split(' ', 2)
        name = None
        for p in glob.glob(os.path.join(sd, 'combined_hits_root_rr1_cm0', f'*_{tag}_feu-combined_hits.root')):
            name = os.path.basename(p)
        if name is None:
            out['acq'][f'{sd}/{tag}'] = 'no cm0 output'
            continue
        old, c0, c5 = (load(os.path.join(sd, d, name)) for d in
                       ('combined_hits_root', 'combined_hits_root_rr1_cm0', 'combined_hits_root_rr1_cm5'))
        r = {}
        for lab, a in (('old', old), ('cm0', c0), ('cm5', c5)):
            r[lab] = None if a is None else dict(hits=int(len(a['eventId'])), events=int(len(np.unique(a['eventId']))))
        if old is not None and c0 is not None:
            same = len(old['eventId']) == len(c0['eventId']) and all(np.array_equal(old[k], c0[k]) for k in COLS)
            r['cm0_equals_old'] = bool(same)
            if not same:
                eo, ec = np.unique(old['eventId']), np.unique(c0['eventId'])
                r['events_only_old'] = int(len(np.setdiff1d(eo, ec)))
                r['events_only_cm0'] = int(len(np.setdiff1d(ec, eo)))
        run = os.path.basename(os.path.dirname(sd))
        R = out['run'][run]
        R['acq'] += 1
        for lab in ('old', 'cm0', 'cm5'):
            if r[lab]:
                R[f'{lab}_hits'] += r[lab]['hits']
                R[f'{lab}_events'] += r[lab]['events']
        R['cm0_equals_old'] += int(r.get('cm0_equals_old', False))
        R['events_only_old'] += r.get('events_only_old', 0)
        R['events_only_cm0'] += r.get('events_only_cm0', 0)
        out['acq'][f'{sd}/{tag}'] = r
    out['run'] = {k: dict(v) for k, v in out['run'].items()}
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
    print(f"{'run':40s} acq  cm0==old  hits old / cm0 / cm5            cm5/cm0  events only-old only-cm0")
    for run, R in sorted(out['run'].items()):
        print(f"{run:40s} {R['acq']:3d}  {R['cm0_equals_old']:4d}   {R.get('old_hits', 0):10d} {R.get('cm0_hits', 0):10d}"
              f" {R.get('cm5_hits', 0):10d}  {R.get('cm5_hits', 0) / max(R.get('cm0_hits', 1), 1):6.3f}"
              f"   {R['events_only_old']:6d} {R['events_only_cm0']:6d}")


if __name__ == '__main__':
    main()
