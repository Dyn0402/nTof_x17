#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 2 (all detectors) — run_67 track-yield vs drift/resist HV, per plastic
threshold, in HAND-DEFINED time-since-flash windows.

run_67 has no deadtime comb (see scan_lib): the SINGLES trigger fills the
N93B 1-81 ms gate continuously, so the time axis is binned by the operator's
window sets (WINDOW_SETS in scan_lib): a BROAD set (1-10 / 10-30 / 30-80 ms)
and a FINE set (nine bins from 1 to 80 ms). This script:

  * caches per_cell_stats for both window sets ->
      tracks/per_cell_stats_broad.csv, tracks/per_cell_stats_fine.csv
    (consumed by detA_2d.py and compare_thresholds.py);
  * draws, per threshold:
      yield_vs_hv_m{mip}.png    P(3D pair) vs resist and vs drift, one row per
                                broad window, 4 detectors — the HV dependence;
      recovery_vs_dt_m{mip}.png P(3D pair) and blind-fraction vs FINE dt window,
                                curves per resist, faceted by detector — the
                                post-flash recovery shape at this threshold.

Efficiency metric = P(3D x/y pair) per recorded trigger (noise-robust; Det A is
the clean reference). See stats.py.

Output -> <ANALYSIS_DIR>/July_HV_Scan/run67_scan/tracks/
Run: .venv/bin/python ntof_july_analysis/run67_scan/analyze_tracks.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402
import stats as S  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'tracks')
DET_COL = S.DET_COL


def fig_yield_vs_hv(st, mip):
    """P(3D pair) vs resist (left) and vs drift (right); one row per broad
    window, all four detectors overlaid. HV dependence at one threshold."""
    windows = [L.win_label(lo, hi) for lo, hi in L.WINDOW_SETS['broad']]
    nrow = len(windows)
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.2 * nrow), squeeze=False)
    for wi, wl in enumerate(windows):
        for ax, (base, xlab) in zip(
                axes[wi],
                [('resist', 'resist HV [V]'), ('drift', 'drift HV [V]')]):
            g = S.agg_yield(st, base, 'p_pair', window=wl, mip=mip)
            for Ld in 'ABCD':
                gd = g[g.det == Ld].sort_values(base)
                if gd.empty:
                    continue
                x = (L.resist_for_det(gd[base].to_numpy(), Ld) if base == 'resist'
                     else gd[base].to_numpy())
                ax.errorbar(x, gd.p, gd.e, color=DET_COL[Ld], marker='o', ms=5,
                            capsize=2, label=f'Det {Ld}' if wi == 0 else None)
            ax.set_xlabel(xlab)
            ax.set_ylabel(f'P(3D pair)\n{wl}', fontsize=9)
            ax.grid(alpha=0.3)
        axes[wi][0].annotate(wl, xy=(0, 1.02), xycoords='axes fraction',
                             fontsize=10, fontweight='bold', color='#444')
    axes[0][0].legend(fontsize=8, ncol=2)
    axes[0][0].set_title('vs resist', fontsize=10)
    axes[0][1].set_title('vs drift', fontsize=10)
    fig.suptitle(f'run_67 @ {L.MIP_LABEL[mip]} — 3D-pair track yield vs HV, '
                 f'per broad time-since-flash window\n(drift-pooled left / '
                 f'resist-pooled right; error bars binomial)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, f'yield_vs_hv_m{mip}.png')


def fig_recovery_vs_dt(st_fine, mip):
    """P(3D pair) and blind-fraction vs FINE dt window, curves per resist,
    faceted by detector. The post-flash recovery shape at this threshold."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
    resists = sorted(st_fine.resist.unique())
    cmap = plt.cm.viridis
    xlab = [f'{int(lo)}-{int(hi)}' for lo, hi in L.WINDOW_SETS['fine']]
    xc = [0.5 * (lo + hi) for lo, hi in L.WINDOW_SETS['fine']]
    # balanced first: these curves pool over drift per (resist, window), and the
    # ragged drift-400 block would land only on resist 550 and bias that curve.
    s = S.balanced_grid(st_fine)
    s = s[s.mip == mip]
    for di, Ld in enumerate('ABCD'):
        ax_p = axes[0, di]
        ax_b = axes[1, di]
        for i, r in enumerate(resists):
            cell = s[(s.det == Ld) & (s.resist == r)].copy()
            # pool over drift per (resist, window): binomial
            cell['k'] = np.round(cell.p_pair * cell.n)
            gg = (cell.groupby('window', observed=True)
                  .agg(k=('k', 'sum'), n=('n', 'sum'),
                       blind=('blind_frac', 'mean')).reindex(
                      [L.win_label(lo, hi) for lo, hi in L.WINDOW_SETS['fine']]))
            p, e = S.binom_err(gg.k.to_numpy(float), gg.n.to_numpy(float))
            c = cmap(i / max(1, len(resists) - 1))
            ax_p.errorbar(xc, p, e, color=c, marker='o', ms=3, capsize=1.5, lw=1,
                          label=f'{int(r)} V' if di == 3 else None)
            ax_b.plot(xc, gg.blind.to_numpy(), color=c, marker='s', ms=3, lw=1)
        ax_p.set_title(f'Det {Ld}' + (' (clean M1)' if Ld == 'A' else ''),
                       fontsize=10, color=DET_COL[Ld])
        ax_p.grid(alpha=0.3)
        ax_b.grid(alpha=0.3)
        ax_b.set_xticks(xc)
        ax_b.set_xticklabels(xlab, fontsize=7, rotation=45, ha='right')
        ax_b.set_xlabel('time since flash [ms]', fontsize=9)
    axes[0, 0].set_ylabel('P(3D x/y pair) per trigger')
    axes[1, 0].set_ylabel('blind fraction\n(read out, 0 hits)')
    axes[0, 3].legend(fontsize=7, title='resist HV', ncol=2)
    fig.suptitle(f'run_67 @ {L.MIP_LABEL[mip]} — post-flash recovery vs time '
                 f'(fine windows, drift-pooled, per resist)\n'
                 f'top: track-finding efficiency;  bottom: front-end blindness',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, f'recovery_vs_dt_m{mip}.png')


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, _ = S.load()
    st = {}
    for setname, wins in L.WINDOW_SETS.items():
        st[setname] = S.per_cell_stats(ev, wins)
        st[setname].to_csv(os.path.join(OUT, f'per_cell_stats_{setname}.csv'),
                           index=False)
        print(f'  wrote per_cell_stats_{setname}.csv '
              f'({len(st[setname])} rows)')

    mips = S.complete_mips(st['broad'])
    if not mips:
        sys.exit('no threshold block has a complete enough HV grid yet')

    figs = []
    for mip in mips:
        figs.append(fig_yield_vs_hv(st['broad'], mip))
        figs.append(fig_recovery_vs_dt(st['fine'], mip))
    for f in figs:
        print('  ->', f)

    # console: best (drift,resist) per det per broad window per threshold
    for mip in mips:
        print(f'\n=== {L.MIP_LABEL[mip]} — best HV per det per broad window ===')
        for lo, hi in L.WINDOW_SETS['broad']:
            wl = L.win_label(lo, hi)
            line = []
            for base in ('resist', 'drift'):
                g = S.agg_yield(st['broad'], base, 'p_pair', window=wl, mip=mip)
                for Ld in 'A':      # keep it short: reference det only
                    gd = g[g.det == Ld]
                    if not gd.p.notna().any():
                        continue
                    b = gd.loc[gd.p.idxmax()]
                    v = (L.resist_for_det(b[base], Ld) if base == 'resist'
                         else b[base])
                    line.append(f'{base} {v:.0f}V')
            print(f'  {wl:>10}: DetA peak @ ' + ', '.join(line))


if __name__ == '__main__':
    main()
