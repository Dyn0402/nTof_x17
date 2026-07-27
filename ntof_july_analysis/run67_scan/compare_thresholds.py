#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 2 (synthesis) — compare the three plastic thresholds.

The run's central question (memory run67-hv-mesh-thresh-scan): dropping the
plastic threshold records MORE triggers per spill (part 1), but does the
per-trigger tracking efficiency HOLD UP, or does the extra rate just add junk?
Two complementary views, per time-since-flash window:

  1. EFFICIENCY  P(3D x/y pair | window) per recorded trigger vs threshold
     (HV-pooled, and at each detector's best HV cell). Flat/rising = efficiency
     holds as the threshold drops; falling = the cheaper triggers are worse.

  2. THROUGHPUT  good tracks / spill = efficiency x events/spill, vs threshold.
     This is what actually matters for collecting IPC: sum(k_pair)/n_spill in
     the window. If it keeps rising as the threshold drops, lower is better even
     at equal efficiency, because we bank more usable tracks per pulse.

"Best points": the highest-throughput (threshold, drift, resist) per detector
per window, tabulated in compare/best_points_{set}.csv and summarised in
compare/recommendation.md.

Output -> <ANALYSIS_DIR>/July_HV_Scan/run67_scan/compare/
Run: .venv/bin/python ntof_july_analysis/run67_scan/compare_thresholds.py
Requires: analyze_tracks.py (per_cell_stats CSVs).
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

OUT = os.path.join(L.OUT_BASE, 'compare')
DET_COL = S.DET_COL


def spills_per_mip():
    """Confirmed-flash spills per threshold, pooled over HV."""
    ev, _, _ = L.load_all()
    ev = ev[ev.flash_ok].copy()
    return ev.groupby('mip').apply(
        lambda d: d.drop_duplicates(['subrun', 'burst']).shape[0],
        include_groups=False)


def pooled_eff(st, mip, det, wl):
    """HV-pooled efficiency + binomial error for one (mip, det, window).

    Balanced first: pooling over a ragged HV grid would compare thresholds over
    different HV sets (see stats.balanced_grid).
    """
    s = S.balanced_grid(st)
    s = s[(s.mip == mip) & (s.det == det) & (s.window == wl)]
    k = np.round(s.p_pair * s.n).sum()
    n = s.n.sum()
    p, e = S.binom_err(k, n)
    return float(p), float(e), int(k), int(n)


def fig_eff_vs_threshold(st, setname, n_spill):
    """Grid: rows = windows, cols = [efficiency, throughput]; curve per det."""
    wins = L.WINDOW_SETS[setname]
    nrow = len(wins)
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3.0 * nrow), squeeze=False)
    # Only thresholds with a complete HV grid: an incomplete block would be
    # "HV-pooled" over a single cell and compared against a 21-cell average,
    # which confounds threshold with HV (see stats.balanced_grid).
    mips = S.complete_mips(st, verbose=False)
    xt = [L.MIP_LABEL[m] for m in mips]
    for wi, (lo, hi) in enumerate(wins):
        wl = L.win_label(lo, hi)
        ax_e, ax_t = axes[wi]
        for det in 'ABCD':
            eff, err, thr = [], [], []
            for m in mips:
                p, e, k, n = pooled_eff(st, m, det, wl)
                eff.append(p); err.append(e)
                thr.append(k / max(int(n_spill.get(m, 1)), 1))   # tracks/spill
            ax_e.errorbar(range(len(mips)), eff, err, color=DET_COL[det],
                          marker='o', ms=5, capsize=2,
                          label=f'Det {det}' if wi == 0 else None)
            ax_t.plot(range(len(mips)), thr, color=DET_COL[det], marker='s',
                      ms=5, lw=1.5)
        for ax in (ax_e, ax_t):
            ax.set_xticks(range(len(mips)))
            ax.set_xticklabels(xt, fontsize=8)
            ax.grid(alpha=0.3)
        ax_e.set_ylabel(f'P(3D pair)\n{wl}', fontsize=9)
        ax_t.set_ylabel('good tracks / spill', fontsize=9)
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].set_title('efficiency per recorded trigger', fontsize=10)
    axes[0, 1].set_title('throughput = eff x events/spill', fontsize=10)
    axes[-1, 0].set_xlabel('plastic threshold')
    axes[-1, 1].set_xlabel('plastic threshold')
    fig.suptitle(f'run_67 — efficiency & throughput vs plastic threshold, per '
                 f'time-since-flash window ({setname}; HV-pooled)\n'
                 f'left: does per-trigger efficiency survive lowering the '
                 f'threshold?  right: do we bank more usable tracks per spill?',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, f'eff_throughput_vs_threshold_{setname}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def best_points(st, setname, n_spill):
    """Per (det, window): the (threshold, drift, resist) with the most good
    tracks/spill among reliable cells, plus its efficiency."""
    rows = []
    for det in 'ABCD':
        for lo, hi in L.WINDOW_SETS[setname]:
            wl = L.win_label(lo, hi)
            cells = st[(st.det == det) & (st.window == wl) & (st.n >= 100)].copy()
            if cells.empty:
                continue
            cells['trk_per_spill'] = cells.apply(
                lambda r: r.k_pair / max(int(n_spill.get(r.mip, 1)), 1), axis=1)
            b = cells.loc[cells.trk_per_spill.idxmax()]
            # also HV-pooled efficiency per threshold at this window, for context
            rows.append(dict(
                det=det, window=wl, win_lo=lo, win_hi=hi,
                best_mip=int(b.mip), best_mip_label=L.MIP_LABEL[int(b.mip)],
                drift=int(b.drift), resist=int(L.resist_for_det(b.resist, det)),
                p_pair=round(float(b.p_pair), 4), e_pair=round(float(b.e_pair), 4),
                n=int(b.n), k_pair=int(b.k_pair),
                trk_per_spill=round(float(b.trk_per_spill), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f'best_points_{setname}.csv'), index=False)
    return df


def recommendation(st_broad, n_spill):
    lines = ['# run_67 — plastic-threshold comparison (broad windows)', '',
             'Metric definitions: **efficiency** = P(3D x/y pair) per recorded '
             'trigger (noise-robust; Det A is the clean reference). '
             '**throughput** = efficiency x events/spill = good tracks banked '
             'per proton pulse. HV-pooled unless stated.', '',
             'Reading: if efficiency is FLAT across thresholds, the extra '
             'triggers a lower threshold records are as good as the expensive '
             'ones, and throughput simply follows the rate — take the lowest '
             'threshold. If efficiency FALLS as the threshold drops, the cheaper '
             'triggers are junk and the optimum is higher.', '']
    mips = S.complete_mips(st_broad, verbose=False)
    skipped = [m for m in L.MIP_LEVELS if m not in mips]
    if skipped:
        lines += ['> **Thresholds still processing and therefore EXCLUDED:** '
                  + ', '.join(L.MIP_LABEL[m] for m in skipped)
                  + '. An incomplete block would be "HV-pooled" over a single '
                    'HV cell and compared against a full 21-cell average, which '
                    'confounds threshold with HV.', '']
    for lo, hi in L.WINDOW_SETS['broad']:
        wl = L.win_label(lo, hi)
        lines.append(f'## {wl}')
        for det in 'A':      # reference detector narrative; full table in CSV
            cells = []
            for m in mips:
                p, e, k, n = pooled_eff(st_broad, m, det, wl)
                tps = k / max(int(n_spill.get(m, 1)), 1)
                cells.append((m, p, e, tps, n))
            for m, p, e, tps, n in cells:
                lines.append(f'- Det {det} @ {L.MIP_LABEL[m]}: eff '
                             f'{p * 1000:.1f}±{e * 1000:.1f} /1000, '
                             f'{tps:.2f} good-tracks/spill (n={n})')
            if cells:
                trend = np.polyfit(range(len(cells)), [c[1] for c in cells], 1)[0]
                verdict = ('efficiency roughly FLAT — take the lowest threshold '
                           'for rate' if abs(trend) < cells[0][2]
                           else ('efficiency RISES toward lower threshold'
                                 if trend > 0 else 'efficiency FALLS toward '
                                 'lower threshold — do not over-lower'))
                lines.append(f'  -> Det A verdict: {verdict}.')
        lines.append('')
    text = '\n'.join(lines)
    with open(os.path.join(OUT, 'recommendation.md'), 'w') as f:
        f.write(text)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    n_spill = spills_per_mip()
    print('spills per threshold:', {int(k): int(v) for k, v in n_spill.items()})
    figs = []
    st_broad = None
    for setname in L.WINDOW_SETS:
        sp = os.path.join(L.OUT_BASE, 'tracks', f'per_cell_stats_{setname}.csv')
        if not os.path.exists(sp):
            sys.exit(f'missing {sp} — run analyze_tracks.py first')
        st = pd.read_csv(sp)
        if setname == 'broad':
            st_broad = st
        figs.append(fig_eff_vs_threshold(st, setname, n_spill))
        bp = best_points(st, setname, n_spill)
        print(f'  wrote best_points_{setname}.csv ({len(bp)} rows)')
    text = recommendation(st_broad, n_spill)
    for f in figs:
        print('  ->', f)
    print('\n' + text)


if __name__ == '__main__':
    main()
