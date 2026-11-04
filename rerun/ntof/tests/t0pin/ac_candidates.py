#!/usr/bin/env python3
"""ac_candidates.py -- list the run_149 triggers with hits in all four A/C planes (candidate A-C through-goers).

    ac_candidates.py SUBRUN_DIR OUT.txt [--min 3] [--max 150]

Hits decide only which triggers to fit (CLAUDE.md: candidate finding); geometry comes from the waveform fit.
A x/y = FEU 3/4, C x/y = FEU 7/8 (run_config.json).  A plane counts if it has min..max hits (max = the seeder's
busy-plane limit, 150).
"""
import argparse, glob, os
import numpy as np, uproot

ap = argparse.ArgumentParser()
ap.add_argument('subrun'); ap.add_argument('out')
ap.add_argument('--min', type=int, default=3); ap.add_argument('--max', type=int, default=150)
a = ap.parse_args()
ids_all = []
for f in sorted(glob.glob(os.path.join(a.subrun, 'combined_hits_root', '*_feu-combined_hits.root'))):
    t = uproot.open(f)['hits']
    ev, feu = (t[b].array(library='np') for b in ('eventId', 'feu'))
    ok = None
    for fe in (3, 4, 7, 8):
        u, n = np.unique(ev[feu == fe], return_counts=True)
        s = set(u[(n >= a.min) & (n <= a.max)].tolist())
        ok = s if ok is None else ok & s
    ids_all += sorted(ok)
    print(os.path.basename(f), len(np.unique(ev)), 'triggers ->', len(ok))
np.savetxt(a.out, np.array(sorted(set(ids_all)), dtype=int), fmt='%d')
print('total', len(set(ids_all)))
