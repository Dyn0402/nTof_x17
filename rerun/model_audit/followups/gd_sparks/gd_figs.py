#!/usr/bin/env python3
"""gd_figs.py -- GD figures: the ROC-like scan per chamber (detected vs bad fraction of accepted fits, hit-count
thresholds as a line, the other criteria as points) and waveform displays of vetoed events by class.
Reads gd_score.py's scored.parquet + results/roc_<det>.csv and lx_sparks.py's wf_samples.npz.
Writes ~/x17/cosmic_bench/model_audit/gd_sparks/figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.expanduser('~/x17/cosmic_bench/model_audit/gd_sparks')
FIG = os.path.join(L, 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def roc():
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.2))
    for ax, det in zip(axs, DETS):
        p = os.path.join(HERE, 'results', f'roc_{det}.csv')
        if not os.path.exists(p):
            continue
        T = pd.read_csv(p).set_index('criterion')
        h = T[T.index.str.startswith('n_hits>') & ~T.index.str.contains('&')]
        h = h.iloc[np.argsort([int(s.split('>')[1]) for s in h.index])]
        ax.plot(h.bad, h.detected, 'o-', color='C0', label='n_hits > t (event)')
        for s, r in h.iterrows():
            ax.annotate(s.split('>')[1], (r.bad, r.detected), fontsize=7, xytext=(3, -8), textcoords='offset points')
        for pre, c, lab in (('plane_max>', 'C1', 'max plane > t'), ('plane_both>', 'C2', 'both planes > t'),
                            ('rail>', 'C3', 'strips on rail > t'), ('on6>', 'C4', 'strips SNR>6 > t'),
                            ('cm_step>', 'C5', 'CM step > t'), ('neg10>', 'C6', 'strips < -10σ > t')):
            q = T[T.index.str.startswith(pre)]
            ax.plot(q.bad, q.detected, 's--', color=c, ms=4, lw=0.8, label=lab)
        for s in ('none', 'discharge(rail>20|on6>150|neg10>100)'):
            if s in T.index:
                ax.plot(T.loc[s, 'bad'], T.loc[s, 'detected'], '*', ms=12, label=s.split('(')[0])
        ax.set_title(det); ax.set_xlabel('bad among accepted fits (%)'); ax.grid(alpha=.3)
    axs[0].set_ylabel('M3 interior rays detected (%)')
    axs[-1].legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'roc.png'), dpi=110); plt.close(fig)


def displays(det, per=3):
    p = os.path.join(L, det, 'wf_samples.npz')
    s = os.path.join(L, det, 'scored.parquet')
    if not (os.path.exists(p) and os.path.exists(s)):
        return
    Z = np.load(p); X = pd.read_parquet(s)
    ids = list(Z['event_id'])
    V = X[X.index.isin(ids) & (X.n_hits_eff > 50)].copy()
    dis = (V.n_rail > 20) | (V.n_on6 > 150) | (V.n_neg10 > 100)
    V['cls'] = np.select([dis & V.det_nv, dis & ~V.det_nv, ~dis & V.det_nv], ['discharge, found', 'discharge, lost',
                                                                               'no discharge, found'], 'no discharge, lost')
    picks = [(c, e) for c, g in V.groupby('cls') for e in g.index[:per]]
    if not picks:
        return
    fig, axs = plt.subplots(len(picks), 2, figsize=(12, 2.3 * len(picks)), squeeze=False)
    for (c, e), row in zip(picks, axs):
        k = ids.index(e); r = X.loc[e]
        for ax, v in zip(row, 'xy'):
            W = Z[v][k].astype(np.float32); pos = Z[f'pos_{v}']
            o = np.argsort(np.where(np.isfinite(pos), pos, 1e9))
            ax.imshow(W[o].T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-800, vmax=800,
                      extent=[0, len(o), 0, W.shape[1]])
            m = np.searchsorted(np.sort(pos[np.isfinite(pos)]), r[f'mesh_{v}'])
            ax.axvline(m, color='k', lw=0.8, ls='--')
            if np.isfinite(r.get(f'{v}_p0', np.nan)):
                ax.axvline(np.searchsorted(np.sort(pos[np.isfinite(pos)]), r[f'{v}_p0']), color='g', lw=0.8)
            ax.set_title(f'{det} ev {e} {v}: {c}; n_hits {int(r.n_hits_eff)}, rail {int(r[f"{v}_n_rail"])}, '
                         f'on6 {int(r[f"{v}_n_on6"])}, r {r.r:.1f} mm', fontsize=8)
            ax.set_yticks([]); ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f'displays_{det}.png'), dpi=90); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    roc()
    for d in DETS:
        displays(d)
    print('figures in', FIG)


if __name__ == '__main__':
    main()
