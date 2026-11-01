"""attach_fig.py -- the Paper I §6 'attachment' figure: what gas each setup had (make_figs.py attachment).

(a) water: the bench chambers' golden rows (rerun/paper1_gas/results/summary.json, timing v on Magboltz 'air_merged')
    next to the SPS groups (rerun/sps_gasfit/results/verdict.json, stacks fitted for water and O2 on the same grid);
(b) O2: the SPS groups; the bench is attachment-free at steady state (no O2 term needed) except det6's first ~4 h of
    6-26 (paper1_gas att_626, ~600 -> 370 ppm O2-equivalent, x2 model factor), drawn as a band.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
BENCH_C, CO2_C, CF4_C = '#4a4a4a', '#b0341d', '#2a78d6'


def draw(verdict_path, save):
    V = json.load(open(verdict_path))
    B = json.load(open(os.path.join(REPO, 'rerun', 'paper1_gas', 'results', 'summary.json')))['golden_water']
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw=dict(wspace=0.28, width_ratios=[1.35, 1]))
    ax = axs[0]; labels = []; x = 0
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        b = B[det]
        ax.errorbar(x, b['water'], yerr=[[b['water'] - b['water_lo']], [b['water_hi'] - b['water']]], fmt='o', color=BENCH_C, capsize=2)
        labels.append(f'bench\n{det}'); x += 1
    x += 0.5
    summ = dict(gap_quoted_mm=V['gap_quoted'], groups={})
    for r in V['table']:
        c = CF4_C if r['gas'] == 'beam' else CO2_C
        ax.errorbar(x, r['water'], yerr=r['err_w'], fmt='s', color=c, capsize=2, mfc='white' if r['edge'] else c)
        axs[1].errorbar(x - 5.5, r['o2_ppm'], yerr=r['err_o2'], fmt='s', color=c, capsize=2, mfc='white' if r['edge'] else c)
        g = r['group'].replace('cf4_', '').replace('co2_', '').replace('_ladder', ' ldr').replace('_', ' ')
        labels.append(('CF₄\n' if r['gas'] == 'beam' else 'CO₂\n') + g)
        summ['groups'][r['group']] = dict(water=round(r['water'], 3), err_w=round(r['err_w'], 3), o2_ppm=round(r['o2_ppm']),
                                          err_o2=round(r['err_o2']), edge=r['edge'])
        x += 1
    xs = list(range(5)) + list(np.arange(5.5, 5.5 + len(V['table'])))
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha='right')
    ax.set_ylabel('water [%]'); ax.set_ylim(0, None)
    ax.set_title(f"water: bench golden rows vs SPS (gap {V['gap_quoted']} mm)", fontsize=9)
    ax = axs[1]
    ax.axhspan(370, 600, color='#cccccc', alpha=0.5, lw=0)
    ax.text(len(V['table']) - 0.6, 590, 'bench det6, first ~4 h of 6-26\n(O₂-equivalent, ×2 model)', fontsize=6.5, ha='right', va='top')
    ax.axhline(0, color=BENCH_C, lw=1); ax.text(-0.3, 15, 'bench, steady state: attachment-free', fontsize=6.5, color=BENCH_C)
    ax.set_xticks(range(len(V['table']))); ax.set_xticklabels(labels[5:], fontsize=6.5, rotation=45, ha='right')
    ax.set_ylabel('O₂ [ppm]'); ax.set_ylim(-20, None)
    ax.set_title('O₂ (attachment) at SPS', fontsize=9)
    save(fig, 'attachment')
    return summ
