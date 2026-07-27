#!/usr/bin/env python3
"""
Det A alone — the RAW (un-smoothed) 2-D drift x resist optimum, per time group.

Why a dedicated script: `optimize.py` kernel-smooths the surface, which is the
right default here because run_61's FEU dropouts (see feu_presence.py) leave the
per-cell statistics BIMODAL — a cell is either fully live (~650-770 events in the
4 ms group) or dropout-hit (~30-80). On a raw grid the dropout-hit cells produce
the largest *apparent* efficiencies purely from small denominators: the top raw
cell for Det A / 4 ms is drift 700 / resist 525 at 96 +- 41 per mille, which is
k = 5 pairs out of n = 52. It is not a measurement of an optimum.

So this script shows the raw numbers, but refuses to hide the statistics:
  * every cell is annotated with p +- err AND its n;
  * cells below N_RELIABLE are hatched and excluded from the argmax;
  * the best RELIABLE cell is starred.

Figures (Det A only):
  detA_2d_raw.png       3 panels (4 ms / 13 ms / 27+ ms), raw heat map + n
  detA_profiles_raw.png 1-D slices with error bars: p vs drift (one curve per
                        resist) and p vs resist (one curve per drift), reliable
                        cells only — this is the honest way to read the 2-D
                        structure without smoothing.

Run: .venv/bin/python ntof_july_analysis/run61_scan/detA_2d.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run61_scan/detA/
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'detA')
STATS = os.path.join(L.OUT_BASE, 'tracks', 'per_cell_stats.csv')
DET = 'A'
N_RELIABLE = 200          # clean split: live cells ~650-770, dropout cells ~30-80
CLASSES = ['early', 'mid', 'late']
CLS_TITLE = {'early': '4 ms group  (4 triggers @ ~4.1 ms)',
             'mid': '13 ms group  (2 triggers @ ~13.5 ms)',
             'late': '27+ ms integrated  (27/41/55/69 ms)'}


def grids(st, cls):
    a = st[(st.det == DET) & (st.probe_class == cls)]
    P = a.pivot_table(index='resist', columns='drift', values='p_pair')
    E = a.pivot_table(index='resist', columns='drift', values='e_pair')
    N = a.pivot_table(index='resist', columns='drift', values='n')
    return P, E, N


def fig_heatmaps(st):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    for ax, cls in zip(axes, CLASSES):
        P, E, N = grids(st, cls)
        Z = P.to_numpy(float) * 1000.0
        Nv = np.nan_to_num(N.to_numpy(float))
        ok = Nv >= N_RELIABLE
        Zshow = np.where(ok, Z, np.nan)          # colour only reliable cells
        im = ax.imshow(Zshow, origin='lower', aspect='auto', cmap='viridis')
        drifts = list(P.columns)
        resists = list(P.index)
        ax.set_xticks(range(len(drifts)))
        ax.set_xticklabels([f'{int(d)}' for d in drifts])
        ax.set_yticks(range(len(resists)))
        ax.set_yticklabels([f'{int(r)}' for r in resists])
        ax.set_xlabel('drift HV [V]')
        ax.set_ylabel('resist HV [V]  (A/B/C setpoint)')
        for i in range(len(resists)):
            for j in range(len(drifts)):
                if not np.isfinite(Z[i, j]):
                    continue
                if ok[i, j]:
                    ax.text(j, i + 0.16, f'{Z[i, j]:.0f}±{E.iloc[i, j] * 1000:.0f}',
                            ha='center', va='center', fontsize=7.5,
                            color='white', weight='bold')
                    ax.text(j, i - 0.22, f'n={int(Nv[i, j])}', ha='center',
                            va='center', fontsize=6, color='0.85')
                else:
                    # dropout-hit cell: show it, but make clear it is not usable
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor='0.85', edgecolor='0.6',
                                               hatch='///', lw=.5))
                    ax.text(j, i + 0.16, f'{Z[i, j]:.0f}±{E.iloc[i, j] * 1000:.0f}',
                            ha='center', va='center', fontsize=7, color='0.35')
                    ax.text(j, i - 0.22, f'n={int(Nv[i, j])}', ha='center',
                            va='center', fontsize=6, color='0.45')
        # star the best RELIABLE cell
        Zr = np.where(ok, Z, -np.inf)
        if np.isfinite(Zr).any() and Zr.max() > -np.inf:
            bi, bj = np.unravel_index(np.nanargmax(Zr), Zr.shape)
            ax.plot(bj, bi, marker='*', ms=26, color='gold',
                    markeredgecolor='black', zorder=5)
            ax.set_title(f'{CLS_TITLE[cls]}\nbest reliable: drift {int(drifts[bj])} V, '
                         f'resist {int(resists[bi])} V  '
                         f'= {Zr[bi, bj]:.0f}±{E.iloc[bi, bj] * 1000:.0f}',
                         fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='P(3D x/y pair) x1000')
    fig.suptitle('run_61 Det A — RAW (un-smoothed) 2-D drift x resist efficiency, '
                 f'per time group.  Hatched = n < {N_RELIABLE} (FEU-dropout cell: '
                 'large apparent p is a small-denominator artefact, excluded from '
                 'the argmax).', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(OUT, 'detA_2d_raw.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_profiles(st):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for col, cls in enumerate(CLASSES):
        P, E, N = grids(st, cls)
        ok = np.nan_to_num(N.to_numpy(float)) >= N_RELIABLE
        drifts, resists = list(P.columns), list(P.index)
        # --- top: p vs drift, one curve per resist ---
        ax = axes[0, col]
        cmap = plt.cm.viridis
        for i, r in enumerate(resists):
            m = ok[i]
            if m.sum() < 2:
                continue
            x = np.array(drifts, float)[m]
            y = P.to_numpy(float)[i][m] * 1000
            ye = E.to_numpy(float)[i][m] * 1000
            ax.errorbar(x, y, ye, marker='o', ms=4, capsize=2, lw=1.2,
                        color=cmap(i / max(1, len(resists) - 1)),
                        label=f'{int(r)} V')
        ax.set_xlabel('drift HV [V]'); ax.set_ylabel('P(3D pair) x1000')
        ax.set_title(f'{CLS_TITLE[cls]}\nvs drift, one curve per resist',
                     fontsize=9)
        ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(fontsize=6, title='resist', ncol=2)
        # --- bottom: p vs resist, one curve per drift ---
        ax = axes[1, col]
        cmap = plt.cm.plasma
        for j, d in enumerate(drifts):
            m = ok[:, j]
            if m.sum() < 2:
                continue
            x = np.array(resists, float)[m]
            y = P.to_numpy(float)[:, j][m] * 1000
            ye = E.to_numpy(float)[:, j][m] * 1000
            ax.errorbar(x, y, ye, marker='s', ms=4, capsize=2, lw=1.2,
                        color=cmap(j / max(1, len(drifts) - 1)),
                        label=f'{int(d)} V')
        ax.set_xlabel('resist HV [V]'); ax.set_ylabel('P(3D pair) x1000')
        ax.set_title('vs resist, one curve per drift', fontsize=9)
        ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(fontsize=6, title='drift', ncol=2)
    fig.suptitle('run_61 Det A — raw 1-D slices through the 2-D scan '
                 f'(reliable cells only, n >= {N_RELIABLE}; error bars = binomial). '
                 'No smoothing.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, 'detA_profiles_raw.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    st = pd.read_csv(STATS)
    print('  ->', fig_heatmaps(st))
    print('  ->', fig_profiles(st))
    for cls in CLASSES:
        P, E, N = grids(st, cls)
        Z = P.to_numpy(float) * 1000
        Nv = np.nan_to_num(N.to_numpy(float))
        ok = Nv >= N_RELIABLE
        print(f'\n=== Det A / {cls}: raw cells, reliable only (n>={N_RELIABLE}) ===')
        rows = []
        for i, r in enumerate(P.index):
            for j, d in enumerate(P.columns):
                if ok[i, j] and np.isfinite(Z[i, j]):
                    rows.append((Z[i, j], E.iloc[i, j] * 1000, int(d), int(r),
                                 int(Nv[i, j])))
        rows.sort(reverse=True)
        for v, e, d, r, n in rows[:6]:
            print(f'   drift {d:3d} resist {r:3d}: {v:5.1f} +- {e:4.1f}  (n={n})')
        n_drop = int((~ok & np.isfinite(Z)).sum())
        print(f'   [{n_drop} of {int(np.isfinite(Z).sum())} cells hatched as '
              f'dropout-hit]')


if __name__ == '__main__':
    main()
