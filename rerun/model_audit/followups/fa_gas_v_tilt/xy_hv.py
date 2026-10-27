#!/usr/bin/env python3
"""xy_hv.py -- det3 X/Y ladder ratio against the mesh (resist) HV of the 6-27 scan rows (xy_scale.json): constant, or a
gain effect?  Weighted mean, chi2 about it, and a weighted line in HV.  Writes results/xy_hv.json."""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import RES  # noqa: E402

X = json.load(open(os.path.join(RES, 'xy_scale.json')))
pts = []
for r in X['rows']:
    m = re.search(r'hv_scan2?_resist_(\d+)V', r['subrun'])
    if r['det'] == 'mx17_3' and m:
        pts.append((float(m.group(1)), r['r'][0], r['r'][1], r['n']))
pts = np.array(sorted(pts))
V, y, e = pts[:, 0], pts[:, 1], pts[:, 2]
w = 1 / e ** 2
mu = float(np.sum(w * y) / w.sum()); emu = float(1 / np.sqrt(w.sum())); chi = float(np.sum(w * (y - mu) ** 2))
A = np.column_stack([np.ones_like(V), V - 475.0])
cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * y))
chil = float(np.sum(w * (y - A @ b) ** 2))
out = dict(n_rows=len(V), mean=mu, mean_err=emu, chi2=chi, ndf=len(V) - 1, slope_pct_per_10V=float(10 * b[1]),
           slope_err=float(10 * np.sqrt(cov[1, 1])), chi2_line=chil, rows=pts.tolist())
json.dump(out, open(os.path.join(RES, 'xy_hv.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
