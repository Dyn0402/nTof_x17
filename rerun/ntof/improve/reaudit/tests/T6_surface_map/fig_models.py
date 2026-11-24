"""fig_models.py -- held-out clean sigma against number of parameters, A and C (both views), and D; summary_table.csv"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows = []
fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, arm in zip(axs, 'ACD'):
    T = pd.read_csv(f'models_{arm}.csv')
    if arm != 'D':
        T = pd.concat([T, pd.read_csv(f'finer_{arm}.csv')]).drop_duplicates(['model', 'view'], keep='last')
    for v, mk in (('x', 'o'), ('y', 's')):
        S = T[T.view == v]
        for fam, col in (('pot', 'C0'), ('ind', 'C1'), ('cellA', 'C2')):
            s = S[S.model.str.startswith(fam) & ~S.model.str.contains('_rot|off')]
            ax.scatter(s.npar, s.sig_clean * 1e3, c=col, marker=mk, s=14, label=f'{fam} {v}')
        for ref, ls in (('free50' if arm != 'D' else 'free100', '-'), ('none', ':')):
            ax.axhline(S[S.model == ref].sig_clean.iloc[0] * 1e3, color='k' if v == 'x' else 'grey', ls=ls, lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('parameters (both views)'); ax.set_ylabel('held-out clean sigma (1e-3 tan)')
    ax.set_title(f'{arm}: solid = free cells, dotted = no map', fontsize=9)
    T['arm'] = arm; rows.append(T)
axs[0].legend(fontsize=7, ncol=2)
fig.savefig('models_sigma_vs_npar.png', dpi=110)
pd.concat(rows).to_csv('summary_table.csv', index=False)
