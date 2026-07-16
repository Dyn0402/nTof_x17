"""
04_channel_mapping.py — Empirical channel geography from coincidence excesses.

Uses cached histograms only (no root-file read):
  - 03 cache: within-arm (wall channel 1-8) x (back-scint bar 1-2) dt histograms
    -> excess matrix -> which wall channels sit in front of which bar (u-halves).
  - 02 cache: tree-level 4x4 wall-arm x pss-arm excess matrix -> cross-arm
    correlation strength (adjacent vs opposite arm hypothesis).

Usage: python 04_channel_mapping.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '04_channel_mapping'
OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ['A', 'B', 'C', 'D']
d3 = np.load(BASE / 'cache' / f'03_offsets_hists_{RUN_STEM}.npz')
d2 = np.load(BASE / 'cache' / f'02_coinc_{RUN_STEM}.npz')
DT_CEN = 0.5 * (d3['dt_edges'][:-1] + d3['dt_edges'][1:])
SIDE = np.abs(DT_CEN) > 100


def excess(hh):
    base = hh[SIDE].mean()
    win = np.abs(DT_CEN - DT_CEN[np.argmax(hh - base)]) <= 10
    return max(hh[win].sum() - base * win.sum(), 0.0)


fig, axes = plt.subplots(2, 4, figsize=(17, 7), height_ratios=[1, 1.6])

# Row 1: within-arm wall-channel vs bar excess fractions
for ax, st in zip(axes[0], STATIONS):
    h = d3[st]  # (8, 2, ndt)
    exc = np.array([[excess(h[wc, pc]) for pc in range(2)] for wc in range(8)])
    frac = exc / exc.sum(axis=1, keepdims=True)
    im = ax.imshow(frac.T, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    for wc in range(8):
        for pc in range(2):
            ax.text(wc, pc, f'{frac[wc, pc]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(frac[wc, pc] - 0.5) > 0.25 else 'black')
    ax.set_xticks(range(8), [str(i + 1) for i in range(8)])
    ax.set_yticks([0, 1], ['bar 1', 'bar 2'])
    ax.set_xlabel(f'WAL{st} channel')
    ax.set_title(f'arm {st}: share of wall-ch excess per bar')

# Row 2: cross-arm tree-level excess matrix, normalized by geometric mean of diagonals
mat = np.zeros((4, 4))
for i, w in enumerate(STATIONS):
    for j, p in enumerate(STATIONS):
        hh = d2[f'WAL{w}_PSS{p}'].sum(axis=(0, 1))
        mat[i, j] = excess(hh)
norm = mat / np.sqrt(np.outer(np.diag(mat), np.diag(mat)))

ax = axes[1][0]
im = ax.imshow(np.log10(mat), cmap='viridis')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{mat[i, j]:,.0f}', ha='center', va='center', fontsize=8, color='white')
ax.set_xticks(range(4), [f'PSS{s}' for s in STATIONS])
ax.set_yticks(range(4), [f'WAL{s}' for s in STATIONS])
ax.set_title('excess pairs (log color)')

ax = axes[1][1]
off = norm.copy()
np.fill_diagonal(off, np.nan)
im = ax.imshow(off, cmap='magma')
for i in range(4):
    for j in range(4):
        if i != j:
            ax.text(j, i, f'{100 * norm[i, j]:.1f}%', ha='center', va='center', fontsize=8,
                    color='white')
ax.set_xticks(range(4), [f'PSS{s}' for s in STATIONS])
ax.set_yticks(range(4), [f'WAL{s}' for s in STATIONS])
ax.set_title('cross-arm excess / geo-mean of diagonals')

# symmetrized arm-arm coupling (wall_i x pss_j + wall_j x pss_i)
ax = axes[1][2]
sym = np.full((4, 4), np.nan)
for i in range(4):
    for j in range(4):
        if i != j:
            sym[i, j] = norm[i, j] + norm[j, i]
im = ax.imshow(sym, cmap='magma')
for i in range(4):
    for j in range(4):
        if i != j:
            ax.text(j, i, f'{100 * sym[i, j]:.1f}%', ha='center', va='center', fontsize=8,
                    color='white')
ax.set_xticks(range(4), STATIONS)
ax.set_yticks(range(4), STATIONS)
ax.set_title('symmetrized arm-arm coupling')

axes[1][3].axis('off')
pairs = sorted(((sym[i, j], STATIONS[i], STATIONS[j]) for i in range(4) for j in range(i + 1, 4)),
               reverse=True)
txt = 'arm-pair coupling ranking:\n\n' + '\n'.join(
    f'  {a}-{b}: {100 * s:.1f}%' for s, a, b in pairs)
txt += ('\n\nMX17: 4 arms at +X,-X,+Z,-Z\n(beam = +Y). Opposite arms see\n'
        'through-going tracks & e+e- pairs;\nadjacent arms mostly showers.\n'
        'Strongest couplings above are the\ncandidate opposite-arm pairs.')
axes[1][3].text(0.02, 0.98, txt, va='top', family='monospace', fontsize=10)

fig.suptitle(f'{RUN_STEM}: empirical channel/arm mapping from coincidence excesses')
fig.tight_layout()
fig.savefig(OUT / 'channel_mapping.png', dpi=140)
print(OUT / 'channel_mapping.png')
