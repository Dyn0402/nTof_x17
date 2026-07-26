#!/usr/bin/env python3
"""Endpoint re-measurement with sparsity-aware estimators.

NNLS charge profiles are sparse (adjacent bins degenerate under the 350ns
template) — a per-bin median underestimates the profile tail wherever
occupancy < 50%. Compare estimators on the 1000V sample:
  - per-bin mean (filtered)
  - 10% trimmed mean
  - median after pairwise rebinning (120 ns bins)
  - per-bin occupancy (fraction of events with q>2% of event total)
and recompute the endpoint for each. Also refit 500V (K=26) with the mean.
"""
import os, sys, pickle, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
DT = 60.0

def U50_generic(uk, prof, plat_bins=(1, 6)):
    plat = np.median(prof[plat_bins[0]:plat_bins[1]])
    if plat <= 0:
        return np.nan
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= plat_bins[1] - 1]
    if len(below) == 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    if y1 == y0:
        return x1
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0)

def boot_est(uk, P, est, n=200):
    rng = np.random.default_rng(2)
    vals = []
    for _ in range(n):
        vals.append(U50_generic(uk, est(P[rng.integers(0, len(P), len(P))])))
    return np.nanmedian(vals), np.nanstd(vals)

def load_1000V():
    res2 = pickle.load(open(os.path.join(BASE, 'freefit2.pkl'), 'rb'))
    profs = []
    for r in res2:
        for p in ('x', 'y'):
            if p not in r or 'error' in r[p] or 'q' not in r[p]:
                continue
            q = np.asarray(r[p]['q'], float)
            if q.sum() <= 0 or q.max() >= 2e4:
                continue
            profs.append(q / q.sum())
    return np.array(profs)

def rebin2(P):
    L = P.shape[1] // 2 * 2
    return P[:, :L].reshape(len(P), -1, 2).sum(axis=2)

if __name__ == '__main__':
    P = load_1000V()
    uk = (np.arange(P.shape[1]) + 0.5) * DT
    print(f'1000V: {len(P)} profiles, K={P.shape[1]}')

    ests = {
        'median': lambda A: np.median(A, axis=0),
        'mean': lambda A: A.mean(axis=0),
        'trim10': lambda A: np.mean(np.sort(A, axis=0)[
            int(0.05 * len(A)):int(0.95 * len(A))], axis=0),
    }
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, est in ests.items():
        U, dU = boot_est(uk, P, est)
        prof = est(P)
        axs[0].plot(uk, prof / prof[1:6].mean(), 'o-', ms=3,
                    label=f'{name}: U50={U:.0f}±{dU:.0f} ns -> {V*U*1e-3:.1f} mm')
        print(f'{name:8s}: U50 = {U:.0f} ± {dU:.0f} ns -> column {V*U*1e-3:.1f} mm')
    # rebinned median
    P2 = rebin2(P)
    uk2 = (np.arange(P2.shape[1]) + 0.5) * 2 * DT
    U, dU = boot_est(uk2, P2, lambda A: np.median(A, axis=0))
    prof2 = np.median(P2, axis=0)
    axs[0].plot(uk2, prof2 / prof2[1:3].mean(), 's--', ms=4,
                label=f'median 120ns bins: U50={U:.0f}±{dU:.0f} -> {V*U*1e-3:.1f} mm')
    print(f'median-120ns: U50 = {U:.0f} ± {dU:.0f} -> column {V*U*1e-3:.1f} mm')
    axs[0].axvline(25000 / V, ls=':', color='k', alpha=0.6)
    axs[0].axvline(29000 / V, ls='--', color='k', alpha=0.6)
    axs[0].set_title('1000 V profile estimators (dotted 25mm, dashed 29mm)')
    axs[0].set_xlabel('u [ns]'); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)

    # occupancy
    occ = (P > 0.02).mean(axis=0)
    axs[1].plot(uk, occ, 'o-')
    axs[1].axvline(25000 / V, ls=':', color='k'); axs[1].axvline(29000 / V, ls='--', color='k')
    axs[1].set_title('per-bin occupancy (q_k > 2% of event charge)')
    axs[1].set_xlabel('u [ns]'); axs[1].grid(alpha=0.3)

    # cumulative charge fraction vs u (integral view, sparsity-immune)
    cum = np.cumsum(P.mean(axis=0))
    axs[2].plot(uk, cum, 'o-')
    for f in (0.9, 0.95, 0.99):
        j = np.argmax(cum >= f)
        axs[2].axhline(f, color='gray', lw=0.5)
        print(f'u at {f*100:.0f}% of charge: {uk[j]:.0f} ns -> {V*uk[j]*1e-3:.1f} mm')
    axs[2].axvline(25000 / V, ls=':', color='k'); axs[2].axvline(29000 / V, ls='--', color='k')
    axs[2].set_title('cumulative mean charge fraction')
    axs[2].set_xlabel('u [ns]'); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, 'endpoint_robust.png'), dpi=110)
    print('saved endpoint_robust.png')
