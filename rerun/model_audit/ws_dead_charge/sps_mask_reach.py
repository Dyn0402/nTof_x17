"""How often does the H4 mask touch a fit window on the ZS caches?  (ZS: a dead strip has no samples, so it is usually
not in the window at all, and exactly-zero samples are censored by WFT_ZS_CENSOR anyway.)"""
import json, os, pickle
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
M = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
for f in ('run63_rot25_d425', 'run63_rot25_d325'):
    E = pickle.load(open(f'/media/dylan/data/x17/cosmic_bench/model_audit/sps/{f}.pkl', 'rb'))
    for p in 'xy':
        d = set(M[p]['dead']); n = nz = 0
        for e in E.values():
            ch = [int(c) for c in e[p]['ch']]
            hit = [i for i, c in enumerate(ch) if c in d]
            if hit:
                n += 1
                nz += bool(np.abs(np.asarray(e[p]['W'])[hit]).sum() > 0)
        print(f, p, len(E), 'windows containing a masked channel', n, 'with non-zero samples on it', nz)
