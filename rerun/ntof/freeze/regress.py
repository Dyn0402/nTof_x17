#!/usr/bin/env python3
"""regress.py NEW_DIR REF_DIR -- compare a frozen-bundle re-run with the scored reference on the same events.

For every <name>.parquet in NEW_DIR (a smoke run with --limit-per-tag) the reference <name>.parquet in REF_DIR is
restricted to the same event ids; per view, the ok flags must agree and p0 / tan / t0 must match within a tolerance
(pass outputs differ in the last digits between worker nodes: never compare bytes)."""
import glob, os, sys
import numpy as np, pandas as pd

new_dir, ref_dir = sys.argv[1], sys.argv[2]
TOL = dict(p0=1e-3, tan_theta=1e-4, t0=0.05)        # mm, tan, ns
bad = 0
for f in sorted(glob.glob(f'{new_dir}/**/*[0-9A-D].parquet', recursive=True)):
    if f.endswith(('.candidates.parquet', '.splits.parquet')):
        continue
    name = os.path.basename(f)
    r = glob.glob(f'{ref_dir}/**/{name}', recursive=True)
    if not r:
        print(f'{name}: no reference'); continue
    N = pd.read_parquet(f).set_index('event_id')
    R = pd.read_parquet(r[0]).set_index('event_id').reindex(N.index)
    out = [name, f'n={len(N)}', f'ref_missing={int(R.x_ok.isna().sum())}']
    for v in 'xy':
        okN, okR = N[f'{v}_ok'].fillna(False).astype(bool), R[f'{v}_ok'].fillna(False).astype(bool)
        flip = int((okN != okR).sum())
        both = okN & okR
        worst = {k: float(np.nanmax(np.abs(N.loc[both, f'{v}_{k}'] - R.loc[both, f'{v}_{k}']))) if both.any() else 0.0
                 for k in TOL}
        over = {k: int((np.abs(N.loc[both, f'{v}_{k}'] - R.loc[both, f'{v}_{k}']) > t).sum()) for k, t in TOL.items()}
        out.append(f'{v}: ok {int(okN.sum())}/{int(okR.sum())} flips {flip} max|d| ' +
                   ' '.join(f'{k} {worst[k]:.2g}' for k in TOL) + f' over_tol {sum(over.values())}')
        bad += flip + sum(over.values())
    print('  '.join(out))
print('REGRESSION', 'PASS' if bad == 0 else f'DIFFERENCES ({bad})')
