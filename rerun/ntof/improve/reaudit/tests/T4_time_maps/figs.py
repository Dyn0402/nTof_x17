#!/usr/bin/env python3
"""T4 figures from tmaps.npz, results_maps_*.json, results_s.json, s_boot.npz."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10.5})
Z = dict(np.load(HERE / 'tmaps.npz'))
R = {}
for f in HERE.glob('results_maps_*.json'):
    R.update(json.load(open(f)))
S = json.load(open(HERE / 'results_s.json'))
SB = dict(np.load(HERE / 's_boot.npz'))
EXT = [-200, 200, -200, 200]


def show(ax, m, title, lim=15, cmap='RdBu_r'):
    im = ax.imshow(100 * (m.T - 1) if lim else m.T, origin='lower', extent=EXT, cmap=cmap,
                   vmin=-lim if lim else None, vmax=lim if lim else None)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x_local [mm]')
    ax.set_ylabel('y_local [mm]')
    return im


# 1. combined maps
fig, axs = plt.subplots(2, 2, figsize=(9, 8))
for ax, arm in zip(axs.flat, 'ACDB'):
    im = show(ax, Z[f'{arm}_xy_T'], f'{arm}: T/T_med - 1 [%], x-y mean')
fig.colorbar(im, ax=axs, shrink=0.8, label='T / T_med - 1 [%]')
fig.savefig(FIG / 'maps_xy.png', dpi=110)
plt.close(fig)

# 2. views
fig, axs = plt.subplots(4, 3, figsize=(10.5, 13))
for r, arm in enumerate('ACDB'):
    show(axs[r, 0], Z[f'{arm}_x_T'], f'{arm} x view [%]')
    im = show(axs[r, 1], Z[f'{arm}_y_T'], f'{arm} y view [%]')
    im2 = show(axs[r, 2], Z[f'{arm}_x_T'] / Z[f'{arm}_y_T'], f'{arm} x / y - 1 [%]', lim=8)
fig.colorbar(im, ax=axs[:, :2], shrink=0.5, label='%')
fig.colorbar(im2, ax=axs[:, 2], shrink=0.5, label='%')
fig.savefig(FIG / 'maps_views.png', dpi=100)
plt.close(fig)

# 3. raw vs corrected, t0 map, railed fraction
fig, axs = plt.subplots(4, 4, figsize=(13, 12.5))
for r, arm in enumerate('ACDB'):
    show(axs[r, 0], Z[f'{arm}_xy_T'], f'{arm} corrected [%]')
    show(axs[r, 1], Z[f'{arm}_xy_T_raw'], f'{arm} raw (table only) [%]')
    im = axs[r, 2].imshow(Z[f'{arm}_t0_map'].T, origin='lower', extent=EXT, cmap='viridis')
    axs[r, 2].set_title(f'{arm} median t0 [ns]', fontsize=10)
    fig.colorbar(im, ax=axs[r, 2])
    im = axs[r, 3].imshow(Z[f'{arm}_railed_frac'].T, origin='lower', extent=EXT, cmap='viridis', vmin=0, vmax=0.6)
    axs[r, 3].set_title(f'{arm} fraction q_uend >= 1170 ns', fontsize=10)
    fig.colorbar(im, ax=axs[r, 3])
fig.tight_layout()
fig.savefig(FIG / 'diagnostics.png', dpi=90)
plt.close(fig)

# 4. corrections
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
tb = np.arange(-0.5, 0.501, 0.05)
tc = 0.5 * (tb[1:] + tb[:-1])
t0b = np.r_[-160, np.arange(-150.0, 150.1, 20.0), 160]
t0c = 0.5 * (t0b[1:] + t0b[:-1])
for arm, col in zip('ACDB', ['C0', 'C1', 'C2', 'C3']):
    p = R[arm]['xy']['prof']
    for ax, k, xc in ((axs[0], 'tanx', tc), (axs[1], 'tany', tc), (axs[2], 't0', t0c), (axs[3], 'q', np.arange(20))):
        if k in p:
            ii = np.array([int(i) for i in p[k]])
            vv = np.array([p[k][i] for i in p[k]])
            ii, vv = ii[(ii >= 0) & (ii < len(xc))], vv[(ii >= 0) & (ii < len(xc))]
            ax.plot(xc[ii], 100 * vv, 'o-', ms=3, color=col, label=arm)
for ax, t in zip(axs, ['tan x (mapped)', 'tan y (mapped)', 't0 (x-y mean) [ns]', 'charge (5 % quantile bins)']):
    ax.set_xlabel(t)
    ax.axhline(0, color='k', lw=0.5)
axs[0].set_ylabel('ln T correction [%]')
axs[0].legend()
fig.tight_layout()
fig.savefig(FIG / 'corrections.png', dpi=110)
plt.close(fig)

# 5. stability
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, arm in zip(axs, 'ACDB'):
    a, b = Z[f'{arm}_xy_T_halves_a'], Z[f'{arm}_xy_T_halves_b']
    ok = np.isfinite(a) & np.isfinite(b)
    ax.plot(100 * (a[ok] - 1), 100 * (b[ok] - 1), '.', ms=3)
    lim = [-25, 30]
    ax.plot(lim, lim, 'k-', lw=0.5)
    st = R[arm]['stab_halves']
    ax.set_title(f"{arm}: r = {st['corr']:.3f}, rms diff {st['rms_diff_pct']:.1f} %", fontsize=10)
    ax.set_xlabel('first half T/T_med - 1 [%]')
axs[0].set_ylabel('second half [%]')
fig.tight_layout()
fig.savefig(FIG / 'stability.png', dpi=110)
plt.close(fig)

# 6. s comparison
fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, key in zip(axs.flat, ['A_x', 'A_y', 'C_x', 'C_y', 'D_x', 'D_y']):
    arm, view = key.split('_')
    tag, nmin = ('50', 80) if arm in 'AC' else ('100', 60)
    m = np.log(Z[f'{arm}_xy_T{tag}'])
    e = Z[f'{arm}_xy_err{tag}']
    s, se, ns = SB[f'{key}_s'], SB[f'{key}_s_err'], SB[f'{key}_n']
    ok = (ns >= nmin) & np.isfinite(m) & (Z[f'{arm}_xy_n{tag}'] >= 100)
    ax.errorbar(100 * m[ok], 100 * np.log1p(s[ok]), xerr=100 * e[ok], yerr=100 * se[ok] / (1 + s[ok]), fmt='o', ms=3)
    f = S[key]['T_xy']['all']
    xx = np.linspace(100 * m[ok].min(), 100 * m[ok].max(), 10)
    ax.plot(xx, 100 * f['a'] + f['b'] * xx, 'r-', label=f"slope {f['b']:.2f} ± {f['b_err']:.2f}, r = {f['r']:.2f}")
    ax.plot(xx, 100 * f['a'] - 0.4 * xx, 'k--', lw=0.8, label='gap expectation, slope -0.4')
    ax.set_title(f'{key}: angle-map scale vs T', fontsize=10)
    ax.set_xlabel('ln(T/T_med) [%], x-y mean')
    ax.set_ylabel('ln(1 + s) [%]')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 's_compare.png', dpi=110)
plt.close(fig)

# 7. C side by side: T map vs s maps (50 mm)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
im = show(axs[0], Z['C_xy_T50'], 'C: T/T_med - 1 [%] (50 mm)')
fig.colorbar(im, ax=axs[0])
for ax, v in zip(axs[1:], 'xy'):
    s = SB[f'C_{v}_s'].copy()
    s[SB[f'C_{v}_n'] < 80] = np.nan
    im = ax.imshow(100 * s.T, origin='lower', extent=EXT, cmap='RdBu_r', vmin=-15, vmax=15)
    ax.set_title(f'C {v}: cosmic angle-map s [%]', fontsize=10)
    fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(FIG / 'C_T_vs_s.png', dpi=110)
plt.close(fig)
print('figures written')
