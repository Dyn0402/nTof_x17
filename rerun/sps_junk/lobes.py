#!/usr/bin/env python3
"""lobes.py <registry key> <out.npz> [n]: per event and channel of an SPS row's first n plateau events (the cache's event
selection, seeded as the 3ac0e79 pass), the peak and the number of LOBES -- runs of samples above LOBE_ADC separated by at
least LOBE_GAP samples below it.  An oscillation is periodic (a lobe every ~15 samples: 3-4 lobes in a 64-sample window);
a real pulse is one run.  Read-only diagnostic for the oscillator rule (rerun/sps_junk/README.md)."""
import csv, os, sys
import numpy as np

CODE = os.environ['WFT_REPO']
sys.path[:0] = [CODE, os.path.join(CODE, 'rerun', 'model_audit')]
import sps_build_cache as sbc   # noqa: E402

LOBE_ADC, LOBE_GAP = 100.0, 3
key, outp = sys.argv[1], sys.argv[2]
n_want = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
row = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'sps_registry.csv')))}[key]
nsmp = int(row['n_samples'])
PK, LB = [], []
orig = sbc.event_record


def lobes_of(ch, sm, am):
    W = np.zeros((512, nsmp), np.float32); m = (sm >= 0) & (sm < nsmp); W[ch[m], sm[m]] = am[m]
    above = W > LOBE_ADC
    nl = np.zeros(512, np.int8)
    for c in np.flatnonzero(above.any(1)):
        idx = np.flatnonzero(above[c])
        nl[c] = 1 + int((np.diff(idx) > LOBE_GAP).sum())
    return W.max(1), nl


def wrapped(ch, sm, am, raw, ns, *a, **k):
    rec = orig(ch, sm, am, raw, ns, *a, **k)
    if rec is not None:
        p, l = lobes_of(ch, sm, sbc._baseline(ch, sm, am, raw))
        PK.append(p.astype(np.float32)); LB.append(l)
    return rec


sbc.event_record = wrapped
tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
n = 0
for sub in row['subruns'].split():
    got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=n_want - n, tan=tan, only=[row['plateau']],
                          verbose=False, hot_scan=0).get(row['plateau'], {})
    n += len(got)
    if n >= n_want:
        break
np.savez_compressed(outp, peak=np.array(PK), lobes=np.array(LB))
print(key, len(PK))
