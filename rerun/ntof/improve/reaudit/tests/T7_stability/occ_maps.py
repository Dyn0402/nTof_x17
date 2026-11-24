#!/usr/bin/env python3
"""Occupancy ratio maps: per production sub-run (time order) / position bin, observed over the E2 campaign profile."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = pd.read_parquet(HERE / 'stability.parquet')
P = S[S.rclass == 'production'].drop_duplicates(['run', 'subrun']).sort_values('t')
keys = [f'{r[4:]}_{s}' for r, s in zip(P.run, P.subrun)]
occ = {k: dict(np.load(HERE / 'cache/occ' / f'{k}.npz')) for k in keys}
E1 = [k.startswith(('79_', '81_')) for k in keys]
edges = np.linspace(-200, 200, 101)
rows = []
for sel in ('all', 'lg'):
    fig, axs = plt.subplots(2, 4, figsize=(10.5, 7), sharey=True)
    for j, arm in enumerate('ABCD'):
        for i, p in enumerate(('x_local', 'y_local')):
            key = f'{arm}_{sel}_{p}'
            M = np.array([occ[k][key] for k in keys], float)
            ref = M[~np.array(E1)].sum(0)
            exp = ref[None, :] * M.sum(1, keepdims=True) / ref.sum()
            with np.errstate(all='ignore'):
                R = M / exp
            R[M.sum(1) < 2000] = np.nan
            ax = axs[i, j]
            im = ax.imshow(R, aspect='auto', origin='lower', extent=[-200, 200, 0, len(keys)], vmin=0, vmax=2,
                           cmap='RdBu')
            ax.set_title(f'{arm} {p}', fontsize=8.5)
            if j == 0:
                first = P.reset_index(drop=True).groupby('run').apply(lambda g: g.index.min())
                sel_ = first.sort_values().iloc[::3]
                ax.set_yticks(sel_.values)
                ax.set_yticklabels([r[4:] for r in sel_.index], fontsize=6.5)
                ax.set_ylabel('run (sub-runs in time order)')
            for c in range(R.shape[1]):
                for r_ in range(R.shape[0]):
                    pass
            # windowed time series for the record: per sub-run, the fraction of tracks in each 4 mm bin vs ref
            for r_, k in enumerate(keys):
                rows.append(dict(key=k, sample=sel, arm=arm, view=p, **{f'b{b:03d}': R[r_, b] for b in range(R.shape[1])}))
    fig.colorbar(im, ax=axs, label='observed / campaign (E2) profile')
    fig.suptitle(f'Occupancy relative to the campaign profile, sample={sel} '
                 f'({"all tracks, any time" if sel == "all" else "late >= 25 ms gated"}); 4 mm bins')
    fig.savefig(HERE / f'fig_occupancy_ratio_{sel}.png')
    plt.close(fig)
pd.DataFrame(rows).to_parquet(HERE / 'occupancy_ratio.parquet', index=False)
