#!/usr/bin/env python3
"""Validation: recompute per-coordinate Chi2 from emitted per-plane positions and
compare to the producer's stored Chi2X/Y. If they agree, the per-plane branches
faithfully reproduce the fit input and the downstream analysis is trustworthy."""
import numpy as np, uproot, awkward as ak

F = 'sat_perplane_0to3.root'
t = uproot.open(F)
tn = sorted([k for k in t.keys() if k.startswith('T;')])[-1]
tr = t[tn]
br = ['Chi2X','Chi2Y','NClusX','NClusY',
      'mX0','mX1','mX2','mX3','mY0','mY1','mY2','mY3',
      'zX0','zX1','zX2','zX3','zY0','zY1','zY2','zY3']
a = tr.arrays(br, library='ak')
# flatten over rays
flat = {k: ak.to_numpy(ak.flatten(a[k])) for k in br}
n = len(flat['Chi2X'])
print(f'{n} rays')

def refit_chi2(coord):
    M = np.stack([flat[f'm{coord}{L}'] for L in range(4)], axis=1)   # (n,4) positions
    Z = np.stack([flat[f'z{coord}{L}'] for L in range(4)], axis=1)   # (n,4) z
    chi2 = np.full(n, np.nan)
    hit = np.isfinite(M) & np.isfinite(Z)
    for i in range(n):
        m = hit[i]
        if m.sum() < 2:
            chi2[i] = 0.0 if m.sum() < 2 else chi2[i]
            continue
        z = Z[i, m]; y = M[i, m]
        # unweighted straight-line least squares (matches TGraph::Fit no-errors)
        A = np.vstack([np.ones_like(z), z]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        chi2[i] = np.sum(res**2)
    return chi2

for coord in ['X','Y']:
    stored = flat[f'Chi2{coord}']
    nclus = flat[f'NClus{coord}']
    recomp = refit_chi2(coord)
    m = np.isfinite(stored) & (stored >= 0) & (nclus >= 2)
    d = recomp[m] - stored[m]
    print(f'\n[{coord}] compared on {m.sum()} rays with NClus>=2')
    print(f'  max|recomputed - stored Chi2| = {np.nanmax(np.abs(d)):.3e} mm^2')
    print(f'  median|diff|                  = {np.nanmedian(np.abs(d)):.3e} mm^2')
    # per-NClus breakdown of median stored chi2
    for nc in [2,3,4]:
        mm = m & (nclus == nc)
        if mm.sum():
            print(f'    NClus={nc}: {mm.sum():6d} rays, median Chi2={np.median(stored[mm]):.4f}, '
                  f'max|diff|={np.nanmax(np.abs(recomp[mm]-stored[mm])):.2e}')
