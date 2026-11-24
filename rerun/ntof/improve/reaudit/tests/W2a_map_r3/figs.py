#!/usr/bin/env python3
"""figs.py -- figures for W2a_map_r3: fields of the candidate per chamber, held-out sigma per model, learning curve."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apply_r3 as AP  # noqa: E402

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 10.5, 'figure.dpi': 130})


def fields_fig(M, arm, out):
    m = M['maps'][arm]
    g = np.arange(-200, 200.1, 5.0)
    X, Y = np.meshgrid(g, g, indexing='ij')
    bt = m['tilt_basis']
    phi = AP.tensor_eval(m['tilt']['phi'], X.ravel(), Y.ravel(), bt['lo'], bt['hi'], bt['h']).reshape(X.shape)
    ax_, ay_, sx, sy = (f.reshape(X.shape) for f in AP.fields(M, arm, X.ravel(), Y.ravel()))
    sup = np.array(m['support']['mask'], bool)
    gs = np.array(m['support']['grid_mm'])
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    ext = [-200, 200, -200, 200]
    pp = phi - phi.mean()
    lo_, hi_ = np.percentile(pp, [2, 98])
    im = axs[0].imshow(pp.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=lo_, vmax=hi_)
    q = 4
    axs[0].quiver(X[::q * 2, ::q * 2], Y[::q * 2, ::q * 2], ax_[::q * 2, ::q * 2], ay_[::q * 2, ::q * 2],
                  scale=0.6, width=0.004)
    fig.colorbar(im, ax=axs[0], label='potential phi [mm]')
    axs[0].set_title(f'{arm}: potential phi, arrows = tilt')
    lim = np.nanpercentile(np.abs(np.concatenate([sx.ravel(), sy.ravel()])), 99)
    for k, (S, lab) in enumerate(((sx, 's_x'), (sy, 's_y'))):
        im = axs[k + 1].imshow(S.T, origin='lower', extent=ext, cmap='PuOr_r', vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=axs[k + 1], label=f'scale {lab}')
        axs[k + 1].set_title(f'{arm}: scale {lab} (local tan)')
    for a in axs:
        a.contour(gs, gs, sup.T.astype(float), levels=[0.5], colors='k', linewidths=0.8, linestyles='--')
        a.set_xlabel('x_local [mm]'); a.set_ylabel('y_local [mm]')
    fig.savefig(out); plt.close(fig)


def sigma_fig(out):
    rows = []
    for arm in 'AC':
        T = pd.read_csv(HERE / f't1_models_{arm}.csv')
        rows.append(T.assign(arm=arm))
        T2 = pd.read_csv(HERE / f't2_models_{arm}.csv')
        T2 = T2[(T2.region == 'all') & T2.model.isin(['joint_f1'])].rename(columns={'sig': 'sig_clean'})
        rows.append(T2.assign(model='spl_pot_sep + pointing (f1)'))
    T2 = pd.read_csv(HERE / 't2_models_D.csv')
    T2 = T2[T2.region == 'all'].rename(columns={'sig': 'sig_clean'})
    T2['model'] = T2.model.replace({'cells_current': 'free50',  # D: the current algorithm with 100 mm cells 'lines': 'spl_pot_sep',
                                    'joint_f1': 'spl_pot_sep + pointing (f1)'})
    rows.append(T2)
    T = pd.concat(rows, ignore_index=True)
    order = ['none', 'free50', 'r2_file_insample', 'T6_pot10_s3', 'T6_cell33_s3', 'spl_free_common',
             'spl_free_sep', 'spl_pot_common', 'spl_pot_sep', 'spl_pot_sep + pointing (f1)']
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=False, constrained_layout=True)
    for a, arm in zip(axs, 'ACD'):
        S = T[T.arm == arm]
        ms = [m for m in order if m in set(S.model)]
        for k, (vw, c) in enumerate((('x', '#3b6ea8'), ('y', '#c0613a'))):
            v = [S[(S.model == m) & (S.view == vw)].sig_clean.iloc[0] * 1e3 for m in ms]
            a.barh(np.arange(len(ms)) + (k - 0.5) * 0.38, v, 0.38, color=c, label=f'{vw} view')
        a.set_yticks(np.arange(len(ms)))
        a.set_yticklabels([('free100 (current alg.)' if (arm == 'D' and m == 'free50') else m) for m in ms])
        a.set_xlabel('held-out sigma [1e-3]'); a.set_title(f'chamber {arm} (run-parity held-out)')
        lo = min(S.sig_clean) * 1e3
        a.set_xlim(lo * 0.85, max(S.sig_clean) * 1e3 * 1.02)
    axs[0].legend(loc='lower right')
    fig.savefig(out); plt.close(fig)


def lc_fig(out):
    fig, ax = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    for arm, col in (('A', '#3b6ea8'), ('C', '#c0613a')):
        f = HERE / f'lc_{arm}.csv'
        if not f.exists():
            continue
        T = pd.read_csv(f)
        T = T[T.view.notna()]
        for vw, ls in (('x', '-'), ('y', '--')):
            S = T[T.view == vw].groupby('n_train_lines_mean').sig.mean()
            ax.plot(S.index, S.values * 1e3, ls, marker='o', color=col, label=f'{arm} {vw}')
    ax.set_xscale('log'); ax.set_xlabel('training lines (one run parity)')
    ax.set_ylabel('held-out clean sigma [1e-3]'); ax.legend(ncol=2)
    fig.savefig(out); plt.close(fig)


if __name__ == '__main__':
    M = AP.load(HERE / (sys.argv[1] if len(sys.argv) > 1 else 'angle_map_r3_candidate.json'))
    for arm in M['maps']:
        fields_fig(M, arm, HERE / f'fig_fields_{arm}.png')
    sigma_fig(HERE / 'fig_sigma_vs_model.png')
    lc_fig(HERE / 'fig_learning_curve.png')
