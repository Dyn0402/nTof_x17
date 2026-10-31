#!/usr/bin/env python3
"""look_probe.py <probe.pkl>: is the residual 155-160 mm Y signal coupled to the oscillating channel?  Per channel
356-395: P(peak > 300 | oscillator active) against P(peak > 300 | quiet), and a few waveforms of junk-seeded events."""
import sys, pickle
import numpy as np

CODE = '/afs/cern.ch/user/d/dneff/x17rerun/code_3ac0e79'
sys.path[:0] = [CODE + '/sps_beam_test_26/det4_sps_assessment']
from det4_sps_map import POSITION_MM   # noqa: E402

d = pickle.load(open(sys.argv[1], 'rb')); E = d['evs']
osc = [int(c) for c in sys.argv[2].split(',')] if len(sys.argv) > 2 else [372, 373]
print('hot', d['hot'], 'oscillators', osc)
chs = np.arange(356, 396)
pk = np.zeros((len(E), len(chs))); seed = np.zeros(len(E), int); tpk = np.full((len(E), len(chs)), -1)
for i, (e, r) in enumerate(sorted(E.items())):
    c, s, a = r['y']
    for j, ch in enumerate(chs):
        m = c == ch
        if m.any():
            pk[i, j] = a[m].max(); tpk[i, j] = s[m][np.argmax(a[m])]
    seed[i] = r['seed_y']
oi = [list(chs).index(c) for c in osc]
act = (pk[:, oi] > 300).any(1)
print(f'oscillator active (any of {osc} > 300 ADC) in {act.mean():.3f} of {len(E)} events')
print(' ch   pos     P(>300|osc)  P(>300|quiet)  ratio   seeds')
for j, ch in enumerate(chs):
    a1, a0 = (pk[act, j] > 300).mean(), (pk[~act, j] > 300).mean()
    print(f'{ch:4d} {POSITION_MM[ch]:7.2f}   {a1:8.3f}     {a0:8.3f}    {a1 / max(a0, 1e-3):6.1f}  {(seed == ch).sum():5d}')
# junk-seeded events: seeds within 4 strips of an oscillator
junk = np.isin(seed, [c for o in osc for c in range(o - 4, o + 5) if c not in osc])
print(f'junk-seeded events: {junk.sum()}, of them oscillator active: {(junk & act).sum()}')
# peak time of the oscillator vs the junk strip
for i in np.flatnonzero(junk)[:8]:
    e = sorted(E)[i]; r = E[e]; c, s, a = r['y']
    print(f'-- event {e} seed {seed[i]}')
    for ch in sorted(set(c) & set(range(366, 382))):
        m = c == ch; w = np.zeros(64); w[s[m]] = a[m]
        print(f'   {ch:3d} ' + ''.join(f'{int(v):5d}' for v in w[::3]))
