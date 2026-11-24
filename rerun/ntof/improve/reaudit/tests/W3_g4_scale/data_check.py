"""Reproduce the data scale with w3lib.scale (n-weighted bins) on the T5 cache, same selection as W2b q2c/q2a."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map'); sys.path.insert(0, '/media/ucla/work/claude-jobs/eb8200b3/W2b_chamber_D')
import pm, hmap, q2lib, w3lib as L
d = pm.load()
rows, prof = [], []
for arm in 'ACD':
    g = d[(d.arm == arm) & d.gated.astype(bool) & (d.ms >= 25)]
    A = pm.arm_arrays(g, arm); hx = hmap.holes(A['x'], A['y']); A['e_normal'] = hmap.normal_halfwidth(A['sh'], A['y']) + 0.06
    tk = hmap.track_ok(A, hx) & (np.abs(A['y']) <= 125)
    e = A['e'][tk]
    for col in ('sh', 'shm'):
        k_q2 = q2lib.scale(pd.DataFrame(dict(e=e, v=A[col][tk])), 'v')[0]
        k, a, B = L.scale(e, A[col][tk], return_bins=True)
        rows.append(dict(arm=arm, col=col, k_q2lib=k_q2, k_w3=k, n=int(tk.sum()), e_normal=A['e_normal']))
        B['arm'], B['col'] = arm, col; prof.append(B)
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('data_check.csv', index=False); pd.concat(prof).to_csv('data_profiles.csv', index=False)
