#!/usr/bin/env python3
"""resid_show.py tag1 [tag2 ...] [--fig out.png] -- summarise resid_map outputs (~/x17/cosmic_bench/det4dig/rm_<tag>.npz).

Per view: angle slope (fitted w against tan_ref, robust), s68 of the angle residual at that slope, median chi2/dof;
then for unsaturated head-on and inclined events the mean signed residual by strip offset (summed over samples, in
noise units per event) and the chi2 share by strip offset; with --fig the full (strip x sample) mean-z maps.
"""
import argparse, os
import numpy as np

D = os.path.expanduser('~/x17/cosmic_bench/det4dig/')


def slope(tr, w):
    m = np.isfinite(w) & (np.abs(tr) < 0.7)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27)
    return s, c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tags', nargs='+'); ap.add_argument('--fig'); ap.add_argument('--v', type=float, default=None)
    a = ap.parse_args()
    Z = {t: np.load(D + f'rm_{t}.npz') for t in a.tags}
    off = int(Z[a.tags[0]]['off']); tb = int(Z[a.tags[0]]['tb'])
    for vi, view in enumerate('xy'):
        print(f'===== {view} view')
        for t, z in Z.items():
            P = z['per']; P = P[P[:, 1] == vi]
            tr, w, chi, dof, wmax, nsat = P[:, 2], P[:, 3], P[:, 4], P[:, 5], P[:, 6], P[:, 7]
            s, c = slope(tr, w)
            v = a.v or s
            d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
            print(f'  {t:12s} ladder {s:6.2f} (offset {c:+.2f}) far {100 * far.mean():4.1f}% s68 {np.percentile(np.abs(d[~far]), 68):.3f} '
                  f'chi2/dof med {np.median(chi / dof):6.1f}  unsat {np.median((chi / dof)[nsat == 0]):6.1f}  n {len(P)}')
        for ci, cl in enumerate(('head-on', 'inclined')):
            print(f'  -- {cl}, unsaturated: mean z summed over samples per event (centre at 0) | chi2 share %')
            for t, z in Z.items():
                n, sz, sz2 = z['n'][vi, ci, 0], z['sz'][vi, ci, 0], z['sz2'][vi, ci, 0]
                nev = max(n[off].max(), 1)
                rz = sz.sum(1) / nev; share = 100 * sz2.sum(1) / sz2.sum()
                print(f'    {t:12s} ' + ' '.join(f'{x:+6.1f}' for x in rz) + ' | ' + ' '.join(f'{x:4.0f}' for x in share))
    if a.fig:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(Z) * 2, 2, figsize=(11, 3.0 * 2 * len(Z)))
        for r, (t, z) in enumerate(Z.items()):
            for vi, view in enumerate('xy'):
                for ci, cl in enumerate(('head-on', 'inclined')):
                    ax = axs[2 * r + vi, ci]
                    n = z['n'][vi, ci, 0]; m = np.where(n > 20, z['sz'][vi, ci, 0] / np.maximum(n, 1), np.nan)
                    im = ax.imshow(m, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, origin='lower',
                                   extent=[-tb - 0.5, m.shape[1] - tb - 0.5, -off - 0.5, off + 0.5])
                    ax.set_title(f'{t} {view} {cl} (unsat): mean z', fontsize=9)
                    ax.set_xlabel('sample - peak sample of centre strip'); ax.set_ylabel('strip offset')
                    plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(a.fig, dpi=100)


if __name__ == '__main__':
    main()
