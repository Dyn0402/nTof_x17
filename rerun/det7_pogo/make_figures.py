"""Figures: (1) the pogo field with the M1 wiring (FEU channel per pad), measured shorts, noisy,
dead; (2) measured vs best rigid-fit prediction; (3) single-pin contact map vs displacement."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import box
from shapely import affinity
import model as M

C = np.load('corr_feu6.npz')['C']
cl = M.Cv.CL['x']; noisy = set(cl['noisy']); dead = set(cl['dead']); weak = set(cl['weak'])
best = json.load(open('refine.json'))[0]
PAD = affinity.rotate(box(-0.05, -0.15, 0.05, 0.15).buffer(0.25, quad_segs=8), 45, origin=(0, 0))
padxy = np.array(PAD.exterior.coords)


def draw_field(ax, title):
    for (c, r) in M.GRID:
        role = M.ROLE[(c, r)]
        fc = {'S': '#dfe6ee', 'G': '#8a8a8a', 'N': 'white'}[role]
        ax.add_patch(Polygon(padxy + (c, r), closed=True, fc=fc, ec='#9aa', lw=0.4))
        ch = M.CH_AT.get((c, r))
        if ch is not None:
            col = 'crimson' if ch in noisy else 'black'
            ax.text(c, r, str(ch), ha='center', va='center', fontsize=4.2, color=col,
                    fontweight='bold' if ch in dead else 'normal')
            if ch in dead:
                ax.plot(c, r, 'x', color='k', ms=7, mew=0.8)
    for k in range(1, 4):
        pass
    ax.set_xlim(-0.8, 9.8); ax.set_ylim(29.8, -0.8); ax.set_aspect('equal')
    ax.set_xticks(range(10)); ax.set_yticks(range(0, 30, 2)); ax.tick_params(labelsize=6)
    ax.set_xlabel('pogo column (1 mm, radial)', fontsize=7); ax.set_ylabel('pogo row (1 mm, along the edge)', fontsize=7)
    ax.set_title(title, fontsize=8)


def edges(ax, lab, color, lw, off=0.0):
    for (a, b, k), on in zip(M.PAIRS, lab):
        if on:
            (x0, y0), (x1, y1) = M.POS[a], M.POS[b]
            ax.plot([x0 + off, x1 + off], [y0 + off, y1 + off], color=color, lw=lw, alpha=0.85, solid_capstyle='round')


meas = C[M._PA, M._PB] > 0.9
l = M.fast_labels(best['dx'], best['dy'], best['th'], best['rp'], best['mode'])
pred = (l[0][M._PA] == l[0][M._PB]) & ~l[1][M._PA]

fig, axs = plt.subplots(1, 2, figsize=(10, 12))
draw_field(axs[0], 'Measured: neighbour pairs with amplitude correlation > 0.9 (blue)\nred number = noisy (rms > 3x median), x = dead')
edges(axs[0], meas, '#1f5fbf', 2.2)
for k, (lo, hi) in enumerate([(0, 9), (11, 19), (21, 30)]):
    pass
draw_field(axs[1], f"Best rigid fit ({best['mode']}): dx {best['dx']:+.2f} dy {best['dy']:+.2f} mm, "
                   f"theta {best['th']:+.1f} deg, r_pin {best['rp']:.3f}\n"
                   f"green = reproduced ({best['tp']}), orange = predicted only ({best['fp']}), blue = missed ({best['fn']})")
edges(axs[1], pred & meas, '#2a9d4a', 2.4)
edges(axs[1], pred & ~meas, '#f08c00', 1.4)
edges(axs[1], ~pred & meas, '#1f5fbf', 2.2)
for ax in axs:
    for x, y, t in [(2, 4.5, 'conn 2'), (7, 11.5, 'conn 1'), (2, 24.5, 'conn 3'), (7, 24.5, 'conn 4')]:
        ax.text(x, y, t, ha='center', va='center', fontsize=16, color='#888', alpha=0.35, zorder=0)
fig.tight_layout(); fig.savefig('figures/pogo_field_measured_vs_fit.png', dpi=170)

# single-pin contact map
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
g = np.linspace(-0.6, 0.6, 241)
X, Y = np.meshgrid(g, g)
for ax, rp in zip(axs, [0.15, 0.25, 0.325]):
    code = np.zeros_like(X)
    for (a, b), bit in [((0, 0), 1), ((0, 1), 2), ((0, -1), 2), ((1, 0), 4), ((-1, 0), 4), ((1, 1), 8), ((-1, -1), 8), ((-1, 1), 16), ((1, -1), 16)]:
        code += bit * (M.pad_dist(X, Y, a, b) < rp) * (np.floor(code / bit) % 2 == 0)
    im = ax.imshow(code, origin='lower', extent=[g[0], g[-1], g[0], g[-1]], cmap='tab20', vmin=0, vmax=31, interpolation='nearest')
    ax.add_patch(Polygon(padxy, closed=True, fc='none', ec='k', lw=0.8))
    ax.set_title(f'pin radius {rp} mm: contacts vs pin offset\n(1 own pad, +2 vertical nb, +4 horizontal nb, +8/16 diagonals)', fontsize=8)
    ax.set_xlabel('pin offset x (mm)'); ax.set_ylabel('pin offset y (mm)')
    for v in np.unique(code):
        yy, xx = np.argwhere(code == v)[len(np.argwhere(code == v)) // 2]
        ax.text(g[xx], g[yy], int(v), fontsize=6, ha='center')
fig.tight_layout(); fig.savefig('figures/pin_contact_map.png', dpi=150)
print('ok')
