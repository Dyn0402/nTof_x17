#!/usr/bin/env python3
"""
residual_merge.py — combine residual_audit.py shards and draw the mismatch.

    residual_merge.py --dir <out> [--png mx_june_wft/residual_audit.png]

Produces, per plane: the mean residual image (data - model, as a fraction of
the event's peak model amplitude) and the mean pull image, both in the fit's own
frame — rows = strips relative to the fitted mesh position, columns = samples.
A perfect model gives noise; structure localises what the model gets wrong.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--png', default=None)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    stacks = sorted(glob.glob(os.path.join(args.dir, 'residual_stack_*.npz')))
    rows = sorted(glob.glob(os.path.join(args.dir, 'residual_rows_*.parquet')))
    if not stacks:
        raise SystemExit(f'no residual_stack_*.npz in {args.dir}')
    acc = {}
    for s in stacks:
        z = np.load(s)
        for k in z.files:
            acc[k] = acc.get(k, 0) + z[k]
    out = {'n_shards': len(stacks)}
    for plane in ('x', 'y'):
        n = float(acc.get(f'{plane}_all_n', acc.get(f'{plane}_n', 0)))
        C = acc.get(f'{plane}_cnt')
        if not n or C is None:
            continue
        # per-cell mean: outer strips are seen by far fewer events than the
        # central one, so the normalisation has to be per cell, not per event
        with np.errstate(invalid='ignore', divide='ignore'):
            R = np.where(C > 0, acc[f'{plane}_res'] / np.maximum(C, 1), np.nan)
            P = np.where(C > 0, acc[f'{plane}_pull'] / np.maximum(C, 1), np.nan)
        out[plane] = dict(
            n_plane_fits=int(n),
            mean_abs_residual_frac=float(np.nanmean(np.abs(R))),
            max_abs_residual_frac=float(np.nanmax(np.abs(R))),
            argmax=[int(i) for i in np.unravel_index(
                int(np.nanargmax(np.abs(R))), R.shape)],
            mean_abs_pull=float(np.nanmean(np.abs(P))),
            max_abs_pull=float(np.nanmax(np.abs(P))),
            cell_counts=[int(np.nanmin(C[C > 0])), int(np.nanmax(C))])
        acc[f'{plane}_res_mean'], acc[f'{plane}_pull_mean'] = R, P

    if rows:
        import pandas as pd
        df = pd.concat([pd.read_parquet(f) for f in rows], ignore_index=True)
        for plane in ('x', 'y'):
            d = df[df.plane == plane]
            if len(d) and plane in out:
                out[plane]['chi2dof_p5_50_95'] = [
                    float(np.percentile(d.chi2dof, q)) for q in (5, 50, 95)]
                out[plane]['n_rows'] = int(len(d))

    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j, plane in enumerate(('x', 'y')):
            for i, (key, title, cmap) in enumerate(
                    ((f'{plane}_res_mean', 'mean (data - model) / peak', 'RdBu_r'),
                     (f'{plane}_pull_mean', 'mean pull', 'RdBu_r'))):
                A = acc.get(key)
                ax = axes[i][j]
                if A is None:
                    continue
                v = np.nanpercentile(np.abs(A), 99)
                ns = (A.shape[0] - 1) // 2
                im = ax.imshow(A, aspect='auto', cmap=cmap, vmin=-v, vmax=v,
                               extent=[0, A.shape[1], ns + 0.5, -ns - 0.5])
                ax.set_title(f'{plane.upper()} plane: {title}')
                ax.set_xlabel('sample')
                ax.set_ylabel('strip - fitted position')
                fig.colorbar(im, ax=ax)
        fig.suptitle('Forward-model residual audit — where the model fails')
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
        print('wrote', args.png)


if __name__ == '__main__':
    main()
