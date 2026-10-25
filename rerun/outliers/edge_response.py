#!/usr/bin/env python3
"""edge_response.py [--out png] -- the angle response vs distance to the plane edge, every chamber, both views.

In bins of the reference footprint's distance to the nearest edge of the view's own coordinate (the X view against
the X edges, Y against the measured Y active edges), on the production selection:
  k       slope of the ladder-scaled fitted tan against tan_ref (1 = honest; < 1 = the tracks read too vertical,
          as a drift field bent toward the cage wall would make them)
  far     share with |dθ| > 5 deg
  s68     of |dθ| inside 5 deg
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from olib import BASE, DETS, load  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402
from classify import DETNAME  # noqa: E402

BINS = [0, 5, 10, 15, 20, 30, 45, 70, 400]


def robust_k(t, f):
    m = np.isfinite(t) & np.isfinite(f)
    for _ in range(5):
        k = np.sum(t[m] * f[m]) / np.sum(t[m] ** 2)
        r = f - k * t; m &= np.abs(r) < max(3 * np.percentile(np.abs(r[m]), 68), 0.02)
    return k


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(BASE, 'an', 'edge_response.png'))
    a = ap.parse_args(); res = {}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for det in DETS:
        d = load(det); act = TRUE_ACTIVE_BY_DET[DETNAME[det]]
        for i, v in enumerate('xy'):
            u = d[d[f'use_{v}']]
            lo = np.minimum(u[f'ref_mesh_{v}'], u[f'ref_mesh_{v}'] + 30 * u[f'ref_tan_{v}'])
            hi = np.maximum(u[f'ref_mesh_{v}'], u[f'ref_mesh_{v}'] + 30 * u[f'ref_tan_{v}'])
            de = np.minimum(lo - act[v][0], act[v][1] - hi).to_numpy()
            t, f, dl = u[f'ref_tan_{v}'].to_numpy(), u[f'tl_{v}'].to_numpy(), u[f'dl_{v}'].to_numpy()
            ks, fa, c = [], [], []
            for b0, b1 in zip(BINS[:-1], BINS[1:]):
                m = (de >= b0) & (de < b1) & (np.abs(t) < 0.5)
                if m.sum() < 40:
                    ks.append(np.nan); fa.append(np.nan); c.append(0.5 * (b0 + b1)); continue
                ks.append(robust_k(t[m], f[m])); fa.append(100 * np.mean(np.abs(dl[m]) > 5)); c.append(0.5 * (b0 + b1))
            res[f'{det}|{v}'] = dict(bins=BINS, k=ks, far=fa)
            axs[0, i].plot(range(len(c)), ks, 'o-', label=det); axs[1, i].plot(range(len(c)), fa, 'o-', label=det)
    for i, v in enumerate('xy'):
        for j, (yl, lab) in enumerate((('k (fitted tan / tan_ref)', 'k'), ('far %', 'far'))):
            ax = axs[j, i]; ax.set_xticks(range(len(BINS) - 1)); ax.set_xticklabels([f'{a}-{b}' for a, b in zip(BINS[:-1], BINS[1:])])
            ax.set_xlabel(f'{v} view: footprint distance to the nearest {v} edge [mm]'); ax.set_ylabel(yl); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(a.out, dpi=80)
    json.dump(res, open(a.out.replace('.png', '.json'), 'w'), indent=1)
    for k, r in res.items():
        print(k, 'k', np.round(r['k'], 2).tolist(), 'far', np.round(r['far'], 1).tolist())


if __name__ == '__main__':
    main()
