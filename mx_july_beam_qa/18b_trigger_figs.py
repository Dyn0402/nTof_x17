"""18b_trigger_figs.py — Figures + per-wall threshold recommendations from the
18 cache. One threshold per wall (constraint): recommended = highest threshold
keeping the WEAKEST group of that wall at >= TARGET_EFF MIP efficiency; the
eff x purity optimum is reported alongside.

Usage: python 18b_trigger_figs.py [run_stem]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224460'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / RUN_STEM / '18_trigger'
OUT.mkdir(parents=True, exist_ok=True)
TARGET_EFF = 0.95

d = np.load(BASE / 'cache' / f'18_trigsum_{RUN_STEM}.npz')
edges = d['sum_edges']
cen = 0.5 * (edges[:-1] + edges[1:])
sb = float(d['sb_scale'])
nb = float(d['n_bunches'])
kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
kern /= kern.sum()
COL = ['#3b82f6', '#0d9488', '#f59e0b', '#c2410c']

def group_quantities(st, g):
    cand = d[f'{st}_cand'][g]
    raw = d[f'{st}_cand_raw'][g]
    sub = d[f'{st}_tag'][g, 0] - sb * d[f'{st}_tag'][g, 1]
    sm = np.convolve(sub, kern, mode='same')
    m = cen > 15
    pk = cen[m][np.argmax(sm[m])]
    hi = (cen >= 1.3 * pk) & (cen <= 2.5 * pk)
    eps = sub[hi].sum() / max(cand[hi].sum(), 1)
    mip = sub / max(eps, 1e-9)
    eff = np.cumsum(sub[::-1])[::-1] / max(sub.sum(), 1)
    pur = np.minimum(np.cumsum(mip[::-1])[::-1] /
                     np.maximum(np.cumsum(cand[::-1])[::-1], 1), 1)
    rate = np.cumsum(cand[::-1])[::-1] / nb
    rate_raw = np.cumsum(raw[::-1])[::-1] / nb
    return dict(cand=cand, mip=mip, pk=pk, eps=eps, eff=eff, pur=pur,
                rate=rate, rate_raw=rate_raw)

Q = {(st, g): group_quantities(st, g) for st in 'ABCD' for g in range(4)}

# ---- recommendations
rec = {}
for st in 'ABCD':
    min_eff = np.min([Q[(st, g)]['eff'] for g in range(4)], axis=0)
    avg_pur = np.mean([Q[(st, g)]['pur'] for g in range(4)], axis=0)
    i_rec = np.nonzero(min_eff >= TARGET_EFF)[0]
    i_rec = i_rec[-1] if len(i_rec) else 0
    i_opt = int(np.argmax(min_eff * avg_pur))
    rec[st] = (cen[i_rec], i_rec, cen[i_opt], i_opt)

# ---- fig 1: sum spectra (log + linear variants)
for scale in ('log', 'linear'):
    fig, axes = plt.subplots(4, 4, figsize=(19, 13), sharex=True)
    for r, st in enumerate('ABCD'):
        for g in range(4):
            ax = axes[r, g]
            q = Q[(st, g)]
            ax.step(cen, q['cand'], where='mid', color='#6b7280', lw=1.3,
                    label='trigger candidates (dup-vetoed)' if (r, g) == (0, 0) else None)
            ax.step(cen, q['mip'], where='mid', color='#c2410c', lw=1.3,
                    label=r'MIP content (tagged / $\epsilon_p$)' if (r, g) == (0, 0) else None)
            ax.fill_between(cen, 0, q['mip'], step='mid', color='#fb923c', alpha=0.4, lw=0)
            ax.axvline(rec[st][0], color='#1d4ed8', ls='--', lw=1.4,
                       label='recommended wall threshold' if (r, g) == (0, 0) else None)
            if scale == 'log':
                ax.set_xlim(0, 150)
                ax.set_yscale('log')
                ax.set_ylim(1, None)
            else:
                ax.set_xlim(0, 130)
                ax.annotate(f'{rec[st][0]:.0f} mV', (rec[st][0] + 2, 0.9 * q['cand'].max()),
                            fontsize=9, color='#1d4ed8')
            ax.set_title(f'WAL{st} g{g + 1}: MIP-sum pk {q["pk"]:.0f} mV, '
                         f'$\\epsilon_p$={q["eps"]:.2f}', fontsize=10)
            ax.grid(alpha=0.2)
            if g == 0:
                ax.set_ylabel('pairs / mV')
            if r == 3:
                ax.set_xlabel('top+bottom sum [mV]')
    fig.legend(loc='upper right', ncols=3, fontsize=10)
    fig.suptitle(f'{RUN_STEM}: top-bottom SUM spectra per group (late tof, duplication-vetoed) '
                 'with MIP content and per-wall recommended threshold', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / ('trigsum_spectra.png' if scale == 'log'
                       else 'trigsum_spectra_linear.png'), dpi=130)
    plt.close(fig)

# ---- fig 2: purity vs efficiency
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, st in zip(axes, 'ABCD'):
    for g in range(4):
        q = Q[(st, g)]
        ax.plot(q['eff'], q['pur'], color=COL[g], lw=1.6, label=f'g{g + 1}')
        i = rec[st][1]
        ax.plot(q['eff'][i], q['pur'][i], 'o', color=COL[g], ms=7,
                mec='k', mew=0.6)
    ax.set_xlabel('MIP efficiency')
    ax.set_ylabel('purity of accepted triggers')
    ax.set_xlim(0.55, 1.005)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_title(f'WAL{st}  (dots: single wall thr = {rec[st][0]:.0f} mV)', fontsize=11)
fig.suptitle(f'{RUN_STEM}: purity vs efficiency per group as the sum threshold varies',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / 'purity_vs_eff.png', dpi=130)
plt.close(fig)

# ---- fig 3: threshold scan per wall
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, st in zip(axes, 'ABCD'):
    min_eff = np.min([Q[(st, g)]['eff'] for g in range(4)], axis=0)
    avg_pur = np.mean([Q[(st, g)]['pur'] for g in range(4)], axis=0)
    rate = np.sum([Q[(st, g)]['rate'] for g in range(4)], axis=0)
    rate_raw = np.sum([Q[(st, g)]['rate_raw'] for g in range(4)], axis=0)
    ax.plot(cen, min_eff, color='#c2410c', lw=1.6, label='min-group MIP eff')
    ax.plot(cen, avg_pur, color='#3b82f6', lw=1.6, label='mean purity')
    ax2 = ax.twinx()
    ax2.plot(cen, rate, color='#6b7280', lw=1.3, label='trigger pairs / bunch')
    ax2.plot(cen, rate_raw, color='#6b7280', lw=1.0, ls=':',
             label='rate if duplication unfixed')
    ax2.set_yscale('log')
    ax2.set_ylabel('accepted pairs / bunch', fontsize=9)
    t, i, topt, _ = rec[st]
    ax.axvline(t, color='#1d4ed8', ls='--', lw=1.3)
    ax.annotate(f'{t:.0f} mV', (t, 0.35), fontsize=10, color='#1d4ed8',
                rotation=90, ha='right')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('wall threshold on top+bottom sum [mV]')
    ax.set_ylabel('efficiency / purity')
    ax.grid(alpha=0.25)
    if st == 'A':
        ax.legend(fontsize=8, loc='center right')
        ax2.legend(fontsize=8, loc='upper right')
    ax.set_title(f'WAL{st}', fontsize=11)
fig.suptitle(f'{RUN_STEM}: single-threshold-per-wall scan '
             f'(dashed: recommended = weakest group at {100 * TARGET_EFF:.0f}% eff)',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / 'threshold_scan.png', dpi=130)
plt.close(fig)

# ---- table
print(f'{"wall":5s} {"rec thr":>8s} {"eff g1-g4 @thr":>28s} {"pur g1-g4 @thr":>28s} '
      f'{"pairs/bunch":>12s} {"(unfixed)":>10s}')
for st in 'ABCD':
    t, i, topt, iopt = rec[st]
    effs = ' '.join(f'{Q[(st, g)]["eff"][i]:.3f}' for g in range(4))
    purs = ' '.join(f'{Q[(st, g)]["pur"][i]:.2f}' for g in range(4))
    rate = sum(Q[(st, g)]['rate'][i] for g in range(4))
    rraw = sum(Q[(st, g)]['rate_raw'][i] for g in range(4))
    print(f'WAL{st}  {t:7.0f}  {effs:>28s} {purs:>28s} {rate:12.1f} {rraw:10.1f}')
    print(f'      (eff x purity optimum at {topt:.0f} mV)')
print(f'\nFigures -> {OUT}')
