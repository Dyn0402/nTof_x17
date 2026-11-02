"""attach_fig.py -- the Paper I 'attachment' figure: what gas det4 had at SPS (make_figs.py attachment).

Restyled 2026-11-02 (review deck: legend for the gas colours, readable unrotated ticks, bench kept apart):
(a) water of every SPS group (rerun/sps_gasfit/results/verdict.json: head-on stacks fitted for water and O2 on the
    Magboltz grid, at the gas fit's gap 27.9 mm), CO2 and CF4 mixtures in ps.GAS colours.  The bench is not plotted
    point by point (its water is §6's gas_vE_timing / gas_drying); its golden-row range is a grey band for scale
    (rerun/paper1_gas/results/summary.json, det4 there at the old 30.48 mm gap).
(b) O2 of the same groups; the bench is attachment-free at steady state, except det6's first ~4 h of 6-26
    (paper1_gas att_626, ~600 -> 370 ppm O2-equivalent, x2 model factor), drawn as a band.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

GASNAME = {'co2': 'CO2', 'beam': 'CF4'}
GASLABEL = {'CO2': 'Ar/CO$_2$ (run 55–59)', 'CF4': 'Ar/CF$_4$/iC$_4$H$_{10}$ 88/10/2'}


def _tick(group):
    run = group.split('_run')[1].split('_')[0]
    tag = group.split('_run')[1][len(run):].strip('_')
    return run + ({'ladder': '\nscan', 'flat': '\nflat', 'raw': '\nraw'}.get(tag, ''))


def draw(verdict_path, save=None):
    import pandas as pd
    ps.use()
    V = json.load(open(verdict_path))
    B = json.load(open(os.path.join(REPO, 'rerun', 'paper1_gas', 'results', 'summary.json')))['golden_water']
    T = V['table']
    fig, axs = plt.subplots(1, 2, figsize=(ps.FULL[0], 2.6), gridspec_kw=dict(wspace=0.30))
    nco2 = sum(GASNAME[r['gas']] == 'CO2' for r in T)
    x = np.array([i if i < nco2 else nco2 + 0.5 + 1.4 * (i - nco2) for i in range(len(T))], float)
    bw = [B[d]['water'] for d in ('det2', 'det3', 'det4', 'det6', 'det7')]
    ax = axs[0]
    ax.axhspan(min(bw), max(bw), color=ps.GRID, lw=0, zorder=0)
    ax.text(x[-1] + 0.4, max(bw) - 0.03, f'bench golden rows\n{min(bw):.2f}–{max(bw):.2f} % (§6)', fontsize=6.5,
            color=ps.MUTED, ha='right', va='top')
    rows = []
    for xi, r in zip(x, T):
        g = GASNAME[r['gas']]; c = ps.GAS[g]
        mk = dict(fmt='s', ms=4.5, color=c, mec=c, mfc='white' if r['edge'] else c, lw=0.9)
        axs[0].errorbar(xi, r['water'], yerr=r['err_w'], **mk)
        axs[1].errorbar(xi, r['o2_ppm'], yerr=r['err_o2'], **mk)
        rows.append(dict(group=r['group'], gas=g, water=r['water'], err_w=r['err_w'], o2_ppm=r['o2_ppm'],
                         err_o2=r['err_o2'], gap_mm=r['gap'], edge=r['edge']))
    for a in axs:
        a.set_xticks(x); a.set_xticklabels([_tick(r['group']) for r in T], fontsize=7.5)
        a.set_xlabel('SPS run'); a.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylabel('water (%)'); ax.set_ylim(0, 2.8)
    ax.set_title(f"(a) water (gap {V['gap_quoted']} mm)")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker='s', ls='', color=ps.GAS[g]) for g in ('CO2', 'CF4')],
              [GASLABEL[g] for g in ('CO2', 'CF4')], loc='lower left', fontsize=6.8, borderaxespad=0.2)
    ax = axs[1]
    ax.axhspan(370, 600, color=ps.GRID, lw=0, zorder=0)
    ax.text(x[-1] + 0.4, 590, 'bench det6, first ~4 h of 6-26\n(O$_2$-equivalent, ×2 model)', fontsize=6.5,
            color=ps.MUTED, ha='right', va='top')
    ax.axhline(0, color=ps.MUTED, lw=0.8)
    ax.text(x[0] - 0.4, 12, 'bench, steady state: no attachment', fontsize=6.5, color=ps.MUTED, va='bottom')
    ax.set_ylabel('O$_2$ (ppm)'); ax.set_ylim(-20, 640)
    ax.set_title('(b) O$_2$ (attachment)')
    ps.save(fig, os.path.join(HERE, 'figures', 'attachment'), data=pd.DataFrame(rows))
    print('wrote', os.path.join(HERE, 'figures', 'attachment.pdf'))
    return dict(gap_quoted_mm=V['gap_quoted'], bench_band=[min(bw), max(bw)],
                groups={r['group']: dict(water=round(r['water'], 3), err_w=round(r['err_w'], 3), o2_ppm=round(r['o2_ppm']),
                                         err_o2=round(r['err_o2']), edge=r['edge']) for r in T})
