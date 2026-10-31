#!/usr/bin/env python3
"""dev_edgeabs.py <colsum.npz> <bundle dir> -- does a noise-set edge cut beat the max-set one? (development check)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimators as E  # noqa: E402
from timing_row import s68  # noqa: E402

z = np.load(sys.argv[1]); d = {k: z[k] for k in z.files}
b = np.load(os.path.join(sys.argv[2], 'arrays.npz'))
kern = E.kernel(b['grid'], b['tmpl_x'])
clean = (d['x_has'] & d['y_has'] & (d['x_nsig'] >= 5) & (d['y_nsig'] >= 5) & (d['x_ncl'] == 1) & (d['y_ncl'] == 1)
         & ~d['x_sat'] & ~d['y_sat'])
idx = np.flatnonzero(clean)[:8000]
ns = d['x_S'].shape[1]
print('pre-sample rms of S (x):', np.median(np.sqrt(np.mean(d['x_S'][idx, :3].astype(float) ** 2, 1))))
cands = [('edge25', None)] + [(f'abs n{ns_} +{na}', (ns_, na)) for ns_ in (3, 5, 8) for na in (1, 2, 3)]
for nm, par in cands:
    if par is None:
        E.EDGE_FRAC = 0.25; est = E.Edge(kern, ns)
    else:
        est = E.EdgeAbs(kern, ns, nsig=par[0], nadd=par[1])
    t = {p: np.array([est.fit(s) for s in d[f'{p}_S'][idx].astype(float)]) + 10.0 * d[f'{p}_ftst'][idx] for p in 'xy'}
    D = t['x'] - t['y']; M = (t['x'] + t['y']) / 2
    sxy = s68(D) / np.sqrt(2); sev = s68(M)
    print(f'{nm:12s} xy {sxy:5.1f}  evt {sev:5.1f}  x {s68(t["x"]):5.1f}  common {np.sqrt(max(sev**2 - sxy**2/2, 0)):5.1f}'
          f'  fail {np.isnan(M).mean()*100:.1f} %')
