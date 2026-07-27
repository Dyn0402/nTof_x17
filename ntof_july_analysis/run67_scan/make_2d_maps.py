#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone 2D heatmap counterparts of analyze_tracks.py's 1D scan plots, for
the deck. Reads the cached per_cell_stats CSVs (does NOT re-run the reco or
touch analyze_tracks.py). mip == 90 (0.90 MIP) only.

Conventions matched to analyze_tracks.py / stats.py / scan_lib.py:
  * balanced_grid: keep only drifts carrying the FULL resist ladder before
    marginalising over drift (drift-400 is fragmentary; at mip90 it is absent
    entirely, so drifts 500/600/700 are pooled).
  * yield map  -> P(3D x/y pair) pooled over drift per (resist, window) as a
    binomial: p = sum(round(p_pair*n)) / sum(n)   [exactly fig_recovery_vs_dt].
  * blind map  -> blind_frac averaged (unweighted mean) over drift per
    (resist, window)                              [exactly fig_recovery_vs_dt].
  * resist_for_det is the identity in run_67 (no det-D offset), so y = resist.
  * FINE windows, in scan_lib.WINDOW_SETS['fine'] order.
  * empty cell (pooled n == 0) -> NaN, left blank (never painted 0).

Outputs (into the tracks/ dir):
  yield_vs_hv_2d_m90.png       recovery_vs_dt_2d_m90.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})

TRACKS = '/home/mx17/beam_july/analysis/July_HV_Scan/run67_scan/tracks'
FINE_CSV = os.path.join(TRACKS, 'per_cell_stats_fine.csv')
MIP = 90

# fine window order, verbatim from scan_lib.WINDOW_SETS['fine']
FINE_WINS = [(1, 4), (4, 8), (8, 12), (12, 20), (20, 30),
             (30, 40), (40, 50), (50, 60), (60, 80)]
WIN_LABELS = [f'{lo}-{hi}' for lo, hi in FINE_WINS]
WIN_ORDER = [f'{lo}-{hi} ms' for lo, hi in FINE_WINS]
DETS = list('ABCD')
DET_TAG = {'A': 'A (clean M1 ref)', 'B': 'B (no-mesh ctrl)',
           'C': 'C', 'D': 'D (no-mesh ctrl)'}
REF_RESIST = 525


def balanced_grid(st):
    """Keep only drifts that carry the full resist ladder (== stats.balanced_grid)."""
    cnt = st.drop_duplicates(['drift', 'resist']).groupby('drift').size()
    keep = sorted(cnt[cnt == cnt.max()].index)
    return st[st.drift.isin(keep)], keep


def build_grids(st, resists):
    """Return {det: (yield_grid, blind_grid, n_grid)} shaped [resist, window].

    yield pooled over drift as a binomial; blind as unweighted mean over drift.
    """
    st = st.copy()
    st['k'] = np.round(st['p_pair'] * st['n'])
    out = {}
    nr, nw = len(resists), len(WIN_ORDER)
    ridx = {r: i for i, r in enumerate(resists)}
    widx = {w: i for i, w in enumerate(WIN_ORDER)}
    for Ld in DETS:
        yg = np.full((nr, nw), np.nan)
        bg = np.full((nr, nw), np.nan)
        ng = np.zeros((nr, nw))
        d = st[st.det == Ld]
        g = d.groupby(['resist', 'window'], observed=True).agg(
            k=('k', 'sum'), n=('n', 'sum'), blind=('blind_frac', 'mean'))
        for (r, w), row in g.iterrows():
            if r not in ridx or w not in widx:
                continue
            i, j = ridx[r], widx[w]
            ng[i, j] = row.n
            if row.n > 0:
                yg[i, j] = row.k / row.n
                bg[i, j] = row.blind
        out[Ld] = (yg, bg, ng)
    return out


def heatmap_fig(grids, resists, key, cmap_name, cbar_label, suptitle, scale=1.0):
    """1x4 detector panels sharing one color scale + colorbar. key: 0=yield,1=blind."""
    vals = np.concatenate([grids[d][key][np.isfinite(grids[d][key])].ravel()
                           for d in DETS])
    vals = vals * scale
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('#e8e8e8')  # no-data cells: neutral light gray, clearly not 0

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 4.6), sharey=True)
    ref_i = list(resists).index(REF_RESIST)
    im = None
    for ax, Ld in zip(axes, DETS):
        arr = np.ma.masked_invalid(grids[Ld][key] * scale)
        im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(len(WIN_LABELS)))
        ax.set_xticklabels(WIN_LABELS, rotation=45, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(resists)))
        ax.set_yticklabels(resists)
        # 525 V reference line + label
        ax.axhline(ref_i, color='white', lw=1.4, ls='--', alpha=0.9)
        ax.axhline(ref_i, color='black', lw=0.6, ls='--', alpha=0.5)
        ax.set_title(DET_TAG[Ld], fontsize=10)
        ax.set_xlabel('time since flash [ms]', fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('resist HV setpoint [V]', fontsize=10)
    axes[-1].annotate('525 V', xy=(1.005, ref_i / (len(resists) - 1)),
                      xycoords=('axes fraction', 'axes fraction'),
                      va='center', ha='left', fontsize=8, color='#333')

    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.2, top=0.84, wspace=0.12)
    cax = fig.add_axes([0.925, 0.2, 0.015, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    return fig


def colnorm_fig(grids, resists, suptitle):
    """Per-column-normalized yield map: each time-window column divided by its
    own max over resist HV, per panel. vmin/vmax fixed 0..1. viridis."""
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e8e8e8')
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 4.6), sharey=True)
    ref_i = list(resists).index(REF_RESIST)
    im = None
    norm_grids = {}
    for Ld in DETS:
        yg = grids[Ld][0]
        col_max = np.nanmax(np.where(np.isfinite(yg), yg, np.nan), axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ng = yg / col_max[None, :]
        # guard: all-NaN or non-positive-max columns -> stay NaN (blank)
        bad_cols = ~(np.isfinite(col_max) & (col_max > 0))
        ng[:, bad_cols] = np.nan
        norm_grids[Ld] = ng
    for ax, Ld in zip(axes, DETS):
        arr = np.ma.masked_invalid(norm_grids[Ld])
        im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap,
                       vmin=0.0, vmax=1.0, interpolation='nearest')
        ax.set_xticks(range(len(WIN_LABELS)))
        ax.set_xticklabels(WIN_LABELS, rotation=45, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(resists)))
        ax.set_yticklabels(resists)
        ax.axhline(ref_i, color='white', lw=1.4, ls='--', alpha=0.9)
        ax.axhline(ref_i, color='black', lw=0.6, ls='--', alpha=0.5)
        ax.set_title(DET_TAG[Ld], fontsize=10)
        ax.set_xlabel('time since flash [ms]', fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('resist HV setpoint [V]', fontsize=10)
    axes[-1].annotate('525 V', xy=(1.005, ref_i / (len(resists) - 1)),
                      xycoords=('axes fraction', 'axes fraction'),
                      va='center', ha='left', fontsize=8, color='#333')
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.2, top=0.84, wspace=0.12)
    cax = fig.add_axes([0.925, 0.2, 0.015, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('P(3D pair), each column scaled to its own max', fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    return fig


def main():
    st = pd.read_csv(FINE_CSV)
    st = st[st.mip == MIP].copy()
    st['window'] = pd.Categorical(st['window'], categories=WIN_ORDER, ordered=True)
    st, kept_drifts = balanced_grid(st)
    resists = sorted(st.resist.unique())

    grids = build_grids(st, resists)

    # ---------- figure 1: yield ----------
    f1 = heatmap_fig(
        grids, resists, key=0, cmap_name='viridis',
        cbar_label='P(3D x/y pair) per trigger  [x10$^{-3}$]',
        suptitle=('run_67 @ 0.90 MIP - P(3D pair) vs resist HV per time-since-flash '
                  'window (2D form)\n(fine windows; pooled over drift '
                  f'{kept_drifts} V, balanced grid; empty = no data)'),
        scale=1000.0)
    p1 = os.path.join(TRACKS, 'yield_vs_hv_2d_m90.png')
    f1.savefig(p1, dpi=130)
    plt.close(f1)

    # ---------- figure 2: blind fraction ----------
    f2 = heatmap_fig(
        grids, resists, key=1, cmap_name='magma',
        cbar_label='blind fraction (read out, 0 hits)',
        suptitle=('run_67 @ 0.90 MIP - front-end blind fraction vs resist HV per '
                  'time-since-flash window (2D form)\n(fine windows; mean over drift '
                  f'{kept_drifts} V, balanced grid; empty = no data)'),
        scale=1.0)
    p2 = os.path.join(TRACKS, 'recovery_vs_dt_2d_m90.png')
    f2.savefig(p2, dpi=130)
    plt.close(f2)

    # ---------- figure 3: per-column-normalized yield ----------
    f3 = colnorm_fig(
        grids, resists,
        suptitle=('run_67 @ 0.90 MIP - HV optimum of P(3D pair) vs time-since-flash '
                  'window (per-column normalized)\n(fine windows; pooled over drift '
                  f'{kept_drifts} V; each column scaled to its own max over resist)'))
    p3 = os.path.join(TRACKS, 'yield_vs_hv_2d_norm_m90.png')
    f3.savefig(p3, dpi=130)
    plt.close(f3)

    # ================= sanity-check console output =================
    print('=' * 70)
    print('mip                :', MIP, '(0.90 MIP)')
    print('drifts pooled      :', kept_drifts, '(balanced grid)')
    print('resist setpoints   :', resists, 'V')
    print('fine window bins    :', WIN_ORDER)
    print('detectors shown    :', DETS)
    print('-' * 70)
    for Ld in DETS:
        yg, bg, ng = grids[Ld]
        filled = int(np.isfinite(yg).sum())
        print(f'Det {Ld}: {filled}/{yg.size} cells with data; '
              f'pooled n range [{int(ng[ng>0].min())}, {int(ng.max())}]')
    print('-' * 70)

    def locate(grid, resists, label, scale=1.0, want='max'):
        a = grid * scale
        flat = np.nanargmax(a) if want == 'max' else np.nanargmin(a)
        i, j = np.unravel_index(flat, a.shape)
        return (f'{label} {want} = {a[i, j]:.4g} at resist {resists[i]} V, '
                f'window {WIN_LABELS[j]} ms')

    for Ld in DETS:
        yg, bg, _ = grids[Ld]
        print(f'Det {Ld}  ' + locate(yg, resists, 'P(3D pair)x1e3',
                                     scale=1000, want='max'))
        print(f'Det {Ld}  ' + locate(yg, resists, 'P(3D pair)x1e3',
                                     scale=1000, want='min'))
        print(f'Det {Ld}  ' + locate(bg, resists, 'blind_frac', want='max'))
        print(f'Det {Ld}  ' + locate(bg, resists, 'blind_frac', want='min'))
    print('-' * 70)

    # explicit 525 V ridge check: best resist in the EARLY (1-4 ms) window
    print('EARLY-window (1-4 ms) resist of peak P(3D pair) per det '
          '(is the ridge at 525 V?):')
    for Ld in DETS:
        yg, _, ng = grids[Ld]
        col = yg[:, 0]
        if np.isfinite(col).any():
            bi = np.nanargmax(col)
            print(f'  Det {Ld}: peak @ resist {resists[bi]} V '
                  f'(P={col[bi]*1000:.3g}e-3, n={int(ng[bi,0])})  |  '
                  'col: ' + ', '.join(
                      f'{r}:{(v*1000):.2g}' if np.isfinite(v) else f'{r}:--'
                      for r, v in zip(resists, col)))
    # per-column argmax-resist table (does the HV optimum walk up with time?)
    print('PER-COLUMN argmax-resist HV [V] (HV that maximizes P(3D pair) in '
          'each time window):')
    hdr = '  det | ' + ' '.join(f'{w:>7}' for w in WIN_LABELS)
    print(hdr)
    for Ld in DETS:
        yg = grids[Ld][0]
        cells = []
        for j in range(len(WIN_ORDER)):
            col = yg[:, j]
            if np.isfinite(col).any() and np.nanmax(col) > 0:
                cells.append(f'{resists[int(np.nanargmax(col))]:>7}')
            else:
                cells.append(f'{"--":>7}')
        print(f'  {Ld}   | ' + ' '.join(cells))
    print('=' * 70)
    print('WROTE', p1)
    print('WROTE', p2)
    print('WROTE', p3)


if __name__ == '__main__':
    main()
