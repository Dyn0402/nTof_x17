#!/usr/bin/env python3
"""rule_on_probe.py <probe.pkl> <dev dir>: apply sps_build_cache.coupled_channels (from the dev dir) to a probe pickle of
probe.py (Y peaks of ch 356-395 only) with the row's hot channels; print the coupling found."""
import ast, pickle, sys
import numpy as np

sys.path.insert(0, sys.argv[2])
import sps_build_cache as sbc   # noqa: E402

d = pickle.load(open(sys.argv[1], 'rb'))
hot = sorted({c for v in d['hot'].values() for c in v})
osc = sbc.SEED_BAD.copy(); osc[hot] = True
E = d['evs']; pk = np.zeros((len(E), 512), np.float32)
for i, (e, r) in enumerate(sorted(E.items())):
    c, s, a = r['y']
    np.maximum.at(pk[i], c.astype(int), a)
print('hot', hot, '->', sbc.coupled_channels(pk, osc))
