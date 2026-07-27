#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zoomed + rebinned blind-fraction map: 0-30 ms time-since-flash in UNIFORM 2 ms
bins, so the early post-flash recovery is resolved. 2D counterpart of the bottom
row of analyze_tracks.fig_recovery_vs_dt, but re-aggregated from the EVENT-level
table (the cached per_cell_stats CSVs only carry the coarse WINDOW_SETS['fine']
windows, which are non-uniform).

blind_frac definition traced from stats.per_cell_stats / fig_recovery_vs_dt:
  * denominator = events READ OUT for that det (readout_{Ld}); in run_67 that is
    all flash-ok events (readout fraction == 1.000).
  * per cell (mip, drift, resist, det, window):
        blind_frac = 1 - mean(live_{Ld})            (1 - hit-producing fraction)
  * the map pools over drift as the UNWEIGHTED MEAN of the per-cell blind_frac
    (== fig_recovery_vs_dt's blind=('blind_frac','mean')), on the balanced grid.

Only the time binning changes here; everything else matches recovery_vs_dt_2d_m90.
mip == 90.  Output -> tracks/recovery_vs_dt_2d_30ms_m90.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})

RUN67 = '/home/mx17/PycharmProjects/nTof_x17/ntof_july_analysis/run67_scan'
sys.path.insert(0, RUN67)
import scan_lib as L      # noqa: E402
import stats as S         # noqa: E402

TRACKS = '/home/mx17/beam_july/analysis/July_HV_Scan/run67_scan/tracks'
MIP = 90
DETS = list('ABCD')
DET_TAG = {'A': 'A (clean M1 ref)', 'B': 'B (no-mesh ctrl)',
           'C': 'C', 'D': 'D (no-mesh ctrl)'}
REF_RESIST = 525

# uniform 2 ms bins, 1 -> 31 ms (15 bins); map ends at 30 ms
BIN_EDGES = np.arange(1.0, 32.0, 2.0)          # [1,3,5,...,31]
BIN_LO = BIN_EDGES[:-1]
BIN_HI = BIN_EDGES[1:]
BIN_LABELS = [f'{lo:g}-{hi:g}' for lo, hi in zip(BIN_LO, BIN_HI)]
NB = len(BIN_LABELS)


def balanced_drifts(ev):
    """Drifts carrying the full resist ladder (== stats.balanced_grid logic)."""
    cnt = (ev.drop_duplicates(['drift', 'resist']).groupby('drift').size())
    return sorted(cnt[cnt == cnt.max()].index)


def main():
    ev, _ = S.load()
    ev = ev[ev.mip == MIP].copy()
    kept = balanced_drifts(ev)
    ev = ev[ev.drift.isin(kept)].copy()
    resists = sorted(ev.resist.unique())
    nr = len(resists)
    ridx = {r: i for i, r in enumerate(resists)}

    # bin index per event (0..NB-1; -1 outside [1,31))
    b = np.digitize(ev['dt_ms'].to_numpy(), BIN_EDGES) - 1
    ev['tbin'] = b
    ev = ev[(ev.tbin >= 0) & (ev.tbin < NB)].copy()

    # per (drift, resist, det, bin): blind_frac = 1 - mean(live), n = #readout
    # then pool over drift as unweighted mean of per-cell blind_frac.
    blind = {Ld: np.full((nr, NB), np.nan) for Ld in DETS}
    n_pool = {Ld: np.zeros((nr, NB), int) for Ld in DETS}   # summed over drift
    ncell = {Ld: np.zeros((nr, NB), int) for Ld in DETS}    # #drift cells pooled
    bsum = {Ld: np.zeros((nr, NB)) for Ld in DETS}          # sum of per-cell blind

    for (dr, r), grp in ev.groupby(['drift', 'resist']):
        if r not in ridx:
            continue
        i = ridx[r]
        for Ld in DETS:
            ro = grp[grp[f'readout_{Ld}']]
            if ro.empty:
                continue
            g = ro.groupby('tbin')
            cnt = g.size()
            live_mean = g[f'live_{Ld}'].mean()
            for tb in cnt.index:
                nn = int(cnt[tb])
                if nn <= 0:
                    continue
                bf = 1.0 - float(live_mean[tb])
                n_pool[Ld][i, tb] += nn
                ncell[Ld][i, tb] += 1
                bsum[Ld][i, tb] += bf
    for Ld in DETS:
        with np.errstate(invalid='ignore'):
            m = ncell[Ld] > 0
            blind[Ld][m] = bsum[Ld][m] / ncell[Ld][m]

    # ---------------- figure ----------------
    vals = np.concatenate([blind[d][np.isfinite(blind[d])].ravel() for d in DETS])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#e8e8e8')
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 4.6), sharey=True)
    ref_i = resists.index(REF_RESIST)
    im = None
    for ax, Ld in zip(axes, DETS):
        arr = np.ma.masked_invalid(blind[Ld])
        im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(NB))
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(resists)
        ax.axhline(ref_i, color='white', lw=1.4, ls='--', alpha=0.9)
        ax.axhline(ref_i, color='black', lw=0.6, ls='--', alpha=0.5)
        ax.set_title(DET_TAG[Ld], fontsize=10)
        ax.set_xlabel('time since flash [ms]', fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('resist HV setpoint [V]', fontsize=10)
    axes[-1].annotate('525 V', xy=(1.005, ref_i / (nr - 1)),
                      xycoords=('axes fraction', 'axes fraction'),
                      va='center', ha='left', fontsize=8, color='#333')
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.22, top=0.84, wspace=0.12)
    cax = fig.add_axes([0.925, 0.22, 0.015, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('blind fraction (read out, 0 hits)', fontsize=9)
    fig.suptitle('run_67 @ 0.90 MIP - front-end blind fraction, 0-30 ms zoom '
                 '(uniform 2 ms bins)\n(mean over drift '
                 f'{kept} V, balanced grid; empty = no data)', fontsize=12)
    p = os.path.join(TRACKS, 'recovery_vs_dt_2d_30ms_m90.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)

    # ---------------- console ----------------
    print('=' * 74)
    print('mip                :', MIP, '(0.90 MIP)')
    print('drifts pooled      :', kept, '(balanced grid)')
    print('resist setpoints   :', resists, 'V')
    print('bin edges [ms]     :', list(BIN_EDGES))
    print('bin labels         :', BIN_LABELS, f'({NB} uniform 2 ms bins)')
    print('-' * 74)
    print('PER-DETECTOR pooled n per (resist, bin)  [summed over drift '
          f'{kept}]:')
    for Ld in DETS:
        print(f'\nDet {Ld}:')
        print('  resist\\bin | ' + ' '.join(f'{lbl:>7}' for lbl in BIN_LABELS))
        for i, r in enumerate(resists):
            print(f'      {r} | ' + ' '.join(f'{n_pool[Ld][i,j]:>7d}'
                                             for j in range(NB)))
    print('-' * 74)
    print('EARLY (1-12 ms) region min pooled-n per detector '
          '(first 6 bins x all resist):')
    for Ld in DETS:
        sub = n_pool[Ld][:, :6]
        print(f'  Det {Ld}: min={int(sub.min())}, median={int(np.median(sub))}, '
              f'max={int(sub.max())}')
    print('-' * 74)
    print('blind_frac at 525 V, early bins (recovery check):')
    for Ld in DETS:
        i = ridx[REF_RESIST]
        row = ' '.join(f'{blind[Ld][i,j]:.3f}' if np.isfinite(blind[Ld][i,j])
                       else '  -- ' for j in range(6))
        print(f'  Det {Ld} @525V, bins 1-12ms: {row}')
    print('=' * 74)
    print('WROTE', p)


if __name__ == '__main__':
    main()
