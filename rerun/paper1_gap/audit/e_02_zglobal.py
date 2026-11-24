"""e_02_zglobal.py [tag] -- global z_k(t) from the 'first' centroid tables (both runs, both views) -> zk_global.json (zpred for the real pass)"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, e_lib as L
tag = sys.argv[1] if len(sys.argv) > 1 else 'first'
edges = np.arange(-60, 901, 60.)
Z = []
for run in ('run1', 'run2'):
    for v in 'xy':
        d = pd.read_parquet(f'{L.OUT}/cen{tag}_{run}_{v}.parquet'); d = d[~d.drop_mask & ~d.sat_ev]
        p = L.profile_region(d, edges); Z.append((p['tmid'], p['z']))
t = np.mean([a for a, b in Z], 0); z = np.mean([b for a, b in Z], 0)
z = np.maximum.accumulate(z)
json.dump(dict(t=[-200.] + t.tolist() + [1200.], z=[0.5] + z.tolist() + [z[-1] + 0.1]), open(f'{L.OUT}/zk_global.json', 'w'))
print(np.c_[t, z])
