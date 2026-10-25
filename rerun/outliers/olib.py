"""Shared loaders for the outlier study: joined tables + the ladder-scaled angle residual of finalize_compare.py."""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser('~/x17/cosmic_bench/outliers')
DETS = ('det3', 'det4', 'det7', 'det6', 'det2')
HEADON, FAR = 0.1, 5.0


def slope(tr, w):
    m = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < 0.7)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.isfinite(r) & (np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27))
    return s, c


def load(det, quality=True):
    """Joined table with dl_<v> (ladder-scaled residual, deg), use_<v> (the production selection: ok and
    n_dropped <= 2, as 03_angles / finalize_compare), and the ladder (s, c) per view in attrs."""
    d = pd.read_parquet(os.path.join(BASE, f'joined_{det}.parquet'))
    lad = {}
    for v in 'xy':
        use = d[f'{v}_ok'] & np.isfinite(d[f'ref_tan_{v}']) & np.isfinite(d[f'{v}_w'])
        if quality:
            use &= d[f'{v}_n_dropped'] <= 2
        d[f'use_{v}'] = use
        tr, w = d[f'ref_tan_{v}'].to_numpy(), d[f'{v}_w'].to_numpy() * 1e3
        s, c = slope(tr[use], w[use]); lad[v] = (s, c)
        d[f'dl_{v}'] = np.degrees(np.arctan((w - c) / s) - np.arctan(tr))
        d[f'tl_{v}'] = (w - c) / s                       # ladder-scaled fitted tan
        d[f'chi2dof_{v}'] = d[f'{v}_chi2'] / d[f'{v}_dof'].clip(lower=1)
        d[f'rres_{v}'] = d[f'{v}_p0'] - d[f'ref_mesh_{v}']   # raw strip frame (res_* is the aligned frame)
    d.attrs['ladder'] = lad
    return d
