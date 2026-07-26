#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Det A — the RAW (un-smoothed) 2-D drift x resist efficiency, annotated with
p +- err AND n, per plastic threshold and per time-since-flash window.

This is the run_67 analogue of run64_scan/detA_2d.py (the operator's preferred
view). Det A is the clean-M1 reference detector. The script shows the raw
numbers but refuses to hide the statistics:
  * every cell is annotated with p +- err (per mille) AND its n;
  * cells below N_RELIABLE are hatched and excluded from the argmax;
  * the best RELIABLE cell is starred.

Unlike run_64, run_67 was taken AFTER the processor-watcher dropout fixes, so
the per-cell statistics are NOT bimodal — every cell is one full 10-min sub-run
(~300-1900 events per fine window). N_RELIABLE is therefore a modest floor, not
a dropout discriminator; it will rarely bite. Any hatched cell is flagged.

Figures, per threshold (mip) and per window set:
  detA/detA_2d_raw_{set}_m{mip}.png       heat map + p +- e + n, one panel/window
  detA/detA_profiles_{set}_m{mip}.png     1-D slices with error bars

Run: .venv/bin/python ntof_july_analysis/run67_scan/detA_2d.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run67_scan/detA/
Requires: analyze_tracks.py has written tracks/per_cell_stats_{set}.csv.
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
import stats as S  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'detA')
DET = 'A'
# Reliability floor. run_67 was taken after the dropout fixes, so a cell backed by
# a FULL 10-min sub-run always clears this easily: n ~ 1300-5400 (broad) and
# ~700-2500 (fine). The floor exists for ONE case -- the drift-400 block, which
# was truncated when the run was stopped after a single short sub-run: its cells
# carry n ~ 145, i.e. ~3% of a full sub-run, and at 28 +- 14 per mille it was
# winning the 30-80 ms argmax on noise alone. 500 hatches exactly those cells and
# nothing legitimate.
N_RELIABLE = 500


def stats_path(setname):
    return os.path.join(L.OUT_BASE, 'tracks', f'per_cell_stats_{setname}.csv')


def grids(st, mip, wl):
    a = st[(st.det == DET) & (st.mip == mip) & (st.window == wl)]
    P = a.pivot_table(index='resist', columns='drift', values='p_pair')
    E = a.pivot_table(index='resist', columns='drift', values='e_pair')
    N = a.pivot_table(index='resist', columns='drift', values='n')
    return P, E, N


def _panel(ax, P, E, N, title):
    Z = P.to_numpy(float) * 1000.0
    Nv = np.nan_to_num(N.to_numpy(float))
    ok = Nv >= N_RELIABLE
    Zshow = np.where(ok, Z, np.nan)
    im = ax.imshow(Zshow, origin='lower', aspect='auto', cmap='viridis')
    drifts, resists = list(P.columns), list(P.index)
    ax.set_xticks(range(len(drifts)))
    ax.set_xticklabels([f'{int(d)}' for d in drifts])
    ax.set_yticks(range(len(resists)))
    ax.set_yticklabels([f'{int(r)}' for r in resists])
    ax.set_xlabel('drift HV [V]')
    ax.set_ylabel('resist HV [V]')
    for i in range(len(resists)):
        for j in range(len(drifts)):
            if not np.isfinite(Z[i, j]):
                continue
            if ok[i, j]:
                ax.text(j, i + 0.17, f'{Z[i, j]:.0f}±{E.iloc[i, j] * 1000:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='white', weight='bold')
                ax.text(j, i - 0.24, f'n={int(Nv[i, j])}', ha='center',
                        va='center', fontsize=6.5, color='0.9')
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor='0.85', edgecolor='0.6',
                                           hatch='///', lw=.5))
                ax.text(j, i + 0.17, f'{Z[i, j]:.0f}±{E.iloc[i, j] * 1000:.0f}',
                        ha='center', va='center', fontsize=7.5, color='0.35')
                ax.text(j, i - 0.24, f'n={int(Nv[i, j])}', ha='center',
                        va='center', fontsize=6.5, color='0.45')
    Zr = np.where(ok, Z, -np.inf)
    star = None
    if np.isfinite(Zr).any() and Zr.max() > -np.inf:
        bi, bj = np.unravel_index(np.nanargmax(Zr), Zr.shape)
        ax.plot(bj, bi, marker='*', ms=22, color='gold',
                markeredgecolor='black', zorder=5)
        star = (int(drifts[bj]), int(resists[bi]), Zr[bi, bj],
                E.iloc[bi, bj] * 1000)
    ax.set_title(title + (f'\nbest: drift {star[0]} V, resist {star[1]} V '
                          f'= {star[2]:.0f}±{star[3]:.0f}' if star else ''),
                 fontsize=9.5)
    return im, star


def fig_heatmaps(st, mip, setname):
    wins = L.WINDOW_SETS[setname]
    n = len(wins)
    ncol = 3 if n <= 3 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 5.6 * nrow),
                             squeeze=False)
    im = None
    for k, (lo, hi) in enumerate(wins):
        ax = axes[k // ncol][k % ncol]
        P, E, N = grids(st, mip, L.win_label(lo, hi))
        if P.empty:
            ax.axis('off')
            continue
        im, _ = _panel(ax, P, E, N, f'{lo:g}-{hi:g} ms')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='P(3D pair) x1000')
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'run_67 Det A @ {L.MIP_LABEL[mip]} — RAW 2-D drift x resist '
                 f'efficiency, per time-since-flash window\n'
                 f'annotation = P(3D x/y pair) x1000 ± binomial err;  n = events;  '
                 f'star = best cell;  hatched = n < {N_RELIABLE}.',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, f'detA_2d_raw_{setname}_m{mip}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_profiles(st, mip, setname):
    wins = L.WINDOW_SETS[setname]
    n = len(wins)
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 8.4), squeeze=False)
    for col, (lo, hi) in enumerate(wins):
        P, E, N = grids(st, mip, L.win_label(lo, hi))
        if P.empty:
            axes[0, col].axis('off'); axes[1, col].axis('off'); continue
        ok = np.nan_to_num(N.to_numpy(float)) >= N_RELIABLE
        drifts, resists = list(P.columns), list(P.index)
        ax = axes[0, col]
        cmap = plt.cm.viridis
        for i, r in enumerate(resists):
            m = ok[i]
            if m.sum() < 2:
                continue
            ax.errorbar(np.array(drifts, float)[m], P.to_numpy(float)[i][m] * 1000,
                        E.to_numpy(float)[i][m] * 1000, marker='o', ms=4,
                        capsize=2, lw=1.2, color=cmap(i / max(1, len(resists) - 1)),
                        label=f'{int(r)} V')
        ax.set_xlabel('drift HV [V]'); ax.set_title(f'{lo:g}-{hi:g} ms', fontsize=9)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel('P(3D pair) x1000\nvs drift, per resist')
        if col == n - 1:
            ax.legend(fontsize=6, title='resist', ncol=2)
        ax = axes[1, col]
        cmap = plt.cm.plasma
        for j, d in enumerate(drifts):
            m = ok[:, j]
            if m.sum() < 2:
                continue
            ax.errorbar(np.array(resists, float)[m], P.to_numpy(float)[:, j][m] * 1000,
                        E.to_numpy(float)[:, j][m] * 1000, marker='s', ms=4,
                        capsize=2, lw=1.2, color=cmap(j / max(1, len(drifts) - 1)),
                        label=f'{int(d)} V')
        ax.set_xlabel('resist HV [V]'); ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel('P(3D pair) x1000\nvs resist, per drift')
        if col == n - 1:
            ax.legend(fontsize=6, title='drift', ncol=2)
    fig.suptitle(f'run_67 Det A @ {L.MIP_LABEL[mip]} — raw 1-D slices through the '
                 f'2-D scan ({setname} windows; reliable cells n >= {N_RELIABLE}; '
                 f'binomial error bars). No smoothing.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, f'detA_profiles_{setname}_m{mip}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    for setname in L.WINDOW_SETS:
        sp = stats_path(setname)
        if not os.path.exists(sp):
            sys.exit(f'missing {sp} — run analyze_tracks.py first')
        st = pd.read_csv(sp)
        for mip in S.complete_mips(st, verbose=(setname == 'broad')):
            print('  ->', fig_heatmaps(st, mip, setname))
            print('  ->', fig_profiles(st, mip, setname))
            # console: top raw reliable cells
            for lo, hi in L.WINDOW_SETS[setname]:
                P, E, N = grids(st, mip, L.win_label(lo, hi))
                if P.empty:
                    continue
                Z = P.to_numpy(float) * 1000
                Nv = np.nan_to_num(N.to_numpy(float))
                ok = Nv >= N_RELIABLE
                rows = []
                for i, r in enumerate(P.index):
                    for j, d in enumerate(P.columns):
                        if ok[i, j] and np.isfinite(Z[i, j]):
                            rows.append((Z[i, j], E.iloc[i, j] * 1000, int(d),
                                         int(r), int(Nv[i, j])))
                rows.sort(reverse=True)
                if rows:
                    v, e, d, r, nn = rows[0]
                    print(f'     {L.MIP_LABEL[mip]} {setname:5} {lo:g}-{hi:g}ms: '
                          f'best drift {d} resist {r} = {v:.0f}±{e:.0f} (n={nn})')


if __name__ == '__main__':
    main()
