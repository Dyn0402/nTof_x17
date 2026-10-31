#!/usr/bin/env python3
"""fine_maps.py [--root OUTROOT] [--rows results/rows.csv] [--out report/figures] -- efficiency maps with the fine
(~8 mm) kernel (LESSONS 2026-06-28: a 25 mm kernel hides det4's ~12 mm stripes; PASS_PLAN §4), every chamber, from the
budget's per-ray table (rays_<det>.parquet): the denominator is every measured M3 ray inside the active area, the
numerator the budget's 'detected' (both views fitted, |r| <= 5 mm raw frame, n_dropped <= 2). Position = the M3 ray
at mid-gap (mesh + 15 tan) in the raw strip frame.

Map at a point = detected / rays within a disk of radius r (8 mm; 25 mm for comparison), on a 2 mm grid; points with
fewer than MIN_N rays are blank. Two samples per chamber: the golden row alone, and the 'plateau pool' (every row of
the chamber at the golden drift field whose interior efficiency is within 3 points of the golden row's: same
hardware, more rays). A profile along x in 2 mm bins (interior in y) shows the stripes without any kernel.
Writes <out>/effmap_<det>.png, effprof_<det>.png and <root>/maps/effmap_<det>.npz.
"""
import argparse, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTROOT = '/eos/user/d/dneff/x17rerun/paper1_eff'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
STEP, MIN_N = 2.0, 15


def disk(r):
    k = int(np.ceil(r / STEP)); y, x = np.mgrid[-k:k + 1, -k:k + 1] * STEP
    return (x ** 2 + y ** 2 <= r ** 2).astype(float)


def emap(R, r):
    from scipy.signal import fftconvolve
    e = np.arange(-10, 412 + STEP, STEP)
    px = R.mesh_x + 15 * R.tan_x; py = R.mesh_y + 15 * R.tan_y
    N, _, _ = np.histogram2d(px, py, bins=[e, e]); D, _, _ = np.histogram2d(px[R.det], py[R.det], bins=[e, e])
    K = disk(r)
    n = np.rint(fftconvolve(N, K, mode='same')); d = np.rint(fftconvolve(D, K, mode='same'))
    m = np.where(n >= MIN_N, d / np.maximum(n, 1), np.nan)
    return e, m, n


def load(root, rows):
    out = []
    for _, r in rows.iterrows():
        rel = f"{r.run}/{r.subrun}/mx17_{r.det[3:]}"
        p = os.path.join(root, 'budget', rel, f'rays_{r.det}.parquet')
        if os.path.exists(p):
            R = pd.read_parquet(p, columns=['mesh_x', 'mesh_y', 'tan_x', 'tan_y', 'region', 'not_measured', 'cat_xy_5', 'region_y'])
            R = R[~R.not_measured & (R.region != 'outside')].copy()
            R['det'] = R.cat_xy_5 == 'detected'
            out.append(R)
    return pd.concat(out) if out else None


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=OUTROOT); ap.add_argument('--rows', default=os.path.join(HERE, 'results', 'rows.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'report', 'figures'))
    a = ap.parse_args()
    T = pd.read_csv(a.rows); os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.root, 'maps'), exist_ok=True)
    pools = {}
    for d, g in T.groupby('det'):
        G = g[g.cls == 'golden_calib']
        if not len(G):
            continue
        G = G.iloc[0]
        pool = g[(g.drift_V == G.drift_V) & (g.eff_interior >= G.eff_interior - 3)]
        pools[d] = dict(golden=G.key, pool=pool.key.tolist())
        samples = {'golden row': load(a.root, g[g.key == G.key]), f'plateau pool ({len(pool)} rows)': load(a.root, pool)}
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        save = {}
        for i, (nm, R) in enumerate(samples.items()):
            if R is None:
                continue
            for j, r in enumerate((8.0, 25.0)):
                e, m, n = emap(R, r); save[f'{i}_{r:g}'] = m
                ax = axs[i, j]
                im = ax.imshow(100 * m.T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], vmin=0 if d == 'det4' else 50, vmax=100, cmap='viridis')
                ax.set_title(f'{d} {nm}: r = {r:g} mm disk (n = {len(R):,} rays)', fontsize=9)
                ax.set_xlabel('x raw strip frame, mid-gap [mm]'); ax.set_ylabel('y [mm]')
                if d == 'det7':
                    ax.axvline(199.3, color='w', ls='--', lw=1); ax.text(100, 5, 'X card 1 (ch 0-255)', color='w', ha='center', fontsize=8)
                plt.colorbar(im, ax=ax, label='efficiency [%]')
            # profile along x: interior in y (8 mm from the passivated edge), 2 mm bins, no kernel
            ax = axs[i, 2]
            px = R.mesh_x + 15 * R.tan_x
            sel = R.region_y.isin(['interior', 'edge-other'])
            b = np.arange(0, 400 + STEP, STEP); k = np.digitize(px[sel], b)
            s = pd.DataFrame(dict(k=k, det=R.det[sel])).groupby('k').det.agg(['mean', 'size'])
            s = s[(s.index > 0) & (s.index < len(b))]
            c = b[s.index - 1] + STEP / 2; p = s['mean']; err = np.sqrt(p * (1 - p) / s['size'])
            ax.errorbar(c, 100 * p, yerr=100 * err, fmt='.', ms=3, lw=0.6)
            ax.set_ylim(0, 105); ax.set_xlabel('x mid-gap [mm] (2 mm bins)'); ax.set_ylabel('efficiency [%]')
            ax.set_title(f'{d} {nm}: profile along x, no kernel', fontsize=9); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f'effmap_{d}.png'), dpi=90); plt.close(fig)
        np.savez_compressed(os.path.join(a.root, 'maps', f'effmap_{d}.npz'), **save)
        print(d, pools[d])
    import json
    json.dump(pools, open(os.path.join(HERE, 'results', 'map_pools.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
