#!/usr/bin/env python3
"""fig_summary.py — slide figures from the ws_widthtilt JSONs.

fig_resolution_map.png : angle s68 per chamber/view, constant mapping vs the cross-validated position map (w0map_xval.json)
fig_toys.png           : toy closure — fitted/true slope vs |tan| for the fitter on its own model, and for a generator
                         whose footprint is narrower than the fitter assumes (the width–tilt trade) (toy_summary.json)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_widthtilt')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def resolution():
    X = json.load(open(f'{ROOT}/w0map_xval.json'))
    labs, a, b = [], [], []
    for d in DETS:
        for p in 'xy':
            r = X[f'{d}_{p}']; labs.append(f'{d}\n{p.upper()}'); a.append(r['const']['all']); b.append(r['map']['all'])
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(x - 0.2, a, 0.4, color='#bbbbbb', label='constant w0, kw (production)')
    ax.bar(x + 0.2, b, 0.4, color='#2ca02c', label='position map w0(p), s(p) (measured on the other half)')
    for i in range(len(x)):
        ax.text(x[i] + 0.2, b[i] + 0.02, f'{100 * (b[i] / a[i] - 1):+.0f}%', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('angle resolution s68 [deg]')
    ax.set_title('Correcting the position-dependent drift bend: angle resolution, all angles (far > 5° excluded)')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{ROOT}/fig_resolution_map.png', dpi=130); plt.close(fig)


def toys():
    T = json.load(open(f'{ROOT}/toy_summary.json'))
    ctr = [0.12, 0.17, 0.24, 0.36]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    sty = {'base': ('fitter on its own model', '-', 'k'), 'landau': ('+ Landau charge fluctuations', ':', '#1f77b4'),
           'nar08': ('true footprint 0.8 × model', '--', '#d62728'), 'dp050': ('true diffusion 0.5 × model', '-.', '#9467bd'),
           's0w': ('true prompt footprint wide (0.42)', '-', '#ff7f0e')}
    for k, p in enumerate('xy'):
        ax = axs[k]
        for tag, (lab, ls, c) in sty.items():
            ys = [T[f'{d}_{tag}'][p]['S'] for d in ('det3', 'det4', 'det6') if f'{d}_{tag}' in T]
            if not ys:
                continue
            ax.plot(ctr, np.nanmean(np.array(ys, float), axis=0), ls, color=c, lw=2, marker='o', label=lab)
        ax.axhline(1, color='gray', lw=0.7); ax.set_title(f'{p.upper()} view (toys on det3/det4/det6 windows)')
        ax.set_xlabel('|tan θ| (truth)'); ax.grid(alpha=0.3)
    axs[0].set_ylabel('fitted / true slope'); axs[0].set_ylim(0.95, 1.02); axs[0].legend(fontsize=8)
    fig.suptitle('Toy closure: the fitter is unbiased on its own model; a too-wide model compresses by ~2 % — mostly at SHALLOW angles',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f'{ROOT}/fig_toys.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    resolution(); toys(); print('ok')
