#!/usr/bin/env python3
"""make_figures.py <gate_full> <overlay_tt dir> <gate_tree> <out dir> -- slide figures for the unified reconstruction.

1. split_rate.png: share of clean bench single muons the joint two-track fit splits, per chamber and view, at the old
   separation floor (1.2 mm) and at four strip pitches (3.11 mm, replayed from the recorded attempts), with the 1 %
   contract line.
2. split_sep.png: separation of the two children -- det6 splits on clean single muons against det3 overlays of two
   real muons (true two-track pairs, satstudy overlay bench).
3. gate_delta.png (if gate_tree has the unified arm): unified minus production, per chamber -- detected, far, s68.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
FLOOR_OLD, FLOOR_NEW = 1.2, 4 * 398.58 / 512
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def clean_splits(root, det):
    r = pd.read_parquet(os.path.join(root, det, 'prod', f'rays_{det}.parquet'))
    sp = pd.read_parquet(os.path.join(root, det, 'twotrack', 'events.splits.parquet'))
    clean = r[(r.region == 'interior') & (r.cat_xy_3 == 'detected')].event_id.unique()
    return sp[sp.accepted & (sp['rank'] == 0) & sp.event_id.isin(clean)], len(clean)


def fig_rate(root, out):
    labels, old, new = [], [], []
    for det in DETS:
        a, n = clean_splits(root, det)
        for v in 'xy':
            b = a[a.plane == v]
            labels.append(f'{det}\n{v.upper()}'); old.append(100 * len(b) / n); new.append(100 * (b.sep >= FLOOR_NEW).sum() / n)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - w / 2 - 0.01, old, w, color=ORANGE, label='floor 1.2 mm (until 10-27)')
    ax.bar(x + w / 2 + 0.01, new, w, color=BLUE, label='floor 4 strips = 3.1 mm')
    ax.axhline(1.0, color=MUTED, lw=1.2, ls='--'); ax.text(len(x) - 0.5, 1.08, 'contract: ≤ 1 %', ha='right', color=MUTED)
    ax.set_xticks(x, labels); ax.set_ylabel('clean single muons split [%]')
    ax.set_title('One muon read as two: the two-track fit on clean bench singles', loc='left', color=INK)
    ax.legend(frameon=False, loc='upper left'); ax.grid(axis='y', color='#e5e4df', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'split_rate.png'), dpi=150); plt.close(fig)


def fig_sep(root, ov, out):
    a, _ = clean_splits(root, 'det6')
    o = pd.read_parquet(os.path.join(ov, 'overlays.parquet')).set_index('oid')
    c = pd.read_parquet(os.path.join(ov, 'candidates.parquet'))
    k = c[c.split_child.astype(bool) & (c['rank'] >= 0)]
    true = []
    for (oid, p), g in k.groupby(['oid', 'plane']):
        if len(g) == 2 and o.loc[oid, 'mode'] == 'overlay' and o.loc[oid, f'sep_{p}'] < 30:
            m = (g.p0 + 15 * g.tan_theta).to_numpy(); true.append(abs(m[0] - m[1]))
    bins = np.arange(0, 16.5, 0.5)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.hist(a.sep.clip(upper=16), bins, color=ORANGE, label=f'det6: splits of clean single muons (n={len(a)})',
            density=True, rwidth=0.9)
    ax.hist(np.clip(true, 0, 16), bins, color=BLUE, histtype='step', lw=2.2, density=True,
            label=f'det3 overlay: two real muons (n={len(true)})')
    ax.axvline(FLOOR_NEW, color=INK, lw=1.4); ax.text(FLOOR_NEW + 0.15, ax.get_ylim()[1] * 0.92, 'new floor: 4 strips', color=INK)
    ax.axvline(FLOOR_OLD, color=MUTED, lw=1.2, ls='--'); ax.text(FLOOR_OLD + 0.15, ax.get_ylim()[1] * 0.80, 'old', color=MUTED)
    ax.set_xlabel('separation of the two fitted tracks [mm]'); ax.set_ylabel('density')
    ax.set_title('False splits sit inside one sharing kernel; real pairs spread out', loc='left', color=INK)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'split_sep.png'), dpi=150); plt.close(fig)


def fig_gate(tree, out):
    import gate_compare as G
    rows = []
    for det in DETS:
        B, U = G.read(tree, det, 'prod'), G.read(tree, det, 'unified')
        if not U or U.get('det5') is None:
            return
        rows.append((det, U['det5'] - B['det5'], U['far5'] - B['far5'], U['s68_x'] - B['s68_x'], U['s68_y'] - B['s68_y']))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    x = np.arange(len(rows)); names = [r[0] for r in rows]
    axs[0].bar(x, [r[1] for r in rows], 0.6, color=BLUE); axs[0].set_title('detected (5 mm) [pt]', loc='left')
    axs[1].bar(x, [r[2] for r in rows], 0.6, color=BLUE); axs[1].set_title('far tracks [pt]', loc='left')
    axs[2].bar(x - 0.2, [r[3] for r in rows], 0.38, color=BLUE, label='X')
    axs[2].bar(x + 0.2, [r[4] for r in rows], 0.38, color=ORANGE, label='Y')
    axs[2].set_title('angular resolution s68 [deg]', loc='left'); axs[2].legend(frameon=False)
    for ax in axs:
        ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(x, names); ax.grid(axis='y', color='#e5e4df', lw=0.8); ax.set_axisbelow(True)
    fig.suptitle('Unified reconstruction minus production, 8000 M3 muons per chamber', x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'gate_delta.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    root, ov, tree, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    fig_rate(root, out); fig_sep(root, ov, out); fig_gate(tree, out)
    print('wrote', sorted(os.listdir(out)))
