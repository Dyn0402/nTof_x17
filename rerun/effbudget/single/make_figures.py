#!/usr/bin/env python3
"""make_figures.py -- Paper I figures from results/ (run after radius.py and xy_pairing.py).

figures/eff_vs_radius.{pdf,png,csv}   det3 efficiency (production pick, multi-track credit) and the accidental rate
                                       against the matching radius, both golden rows
figures/xy_pairing.{pdf,png,csv}      ln(q_X/q_Y) of the true and the swapped pairings of 552 real two-particle
                                       events, against the single-track distribution (reference band)
House style: sept26_prelim_analysis/figstyle.py (document figures, light ink, every figure ships its CSV).
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sept26_prelim_analysis'))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')


def eff_vs_radius():
    d = json.load(open(os.path.join(RES, 'radius.json')))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax2 = ax.twinx()
    rows = []
    for key, ls, lab in (('p2_det3', '-', '6-27 long run'), ('sat_det3', '--', 'sat\\_det3')):
        r = d[key]; R = np.array(r['radii'])
        ax.plot(R, r['eff'], ls, color='#0072B2', lw=1.4, label=f'production pick, {lab}'.replace('\\', ''))
        ax.plot(R, r['credit'], ls, color='#009E73', lw=1.4, label=f'multi-track credit, {lab}'.replace('\\', ''))
        ax2.plot(R, r['accidental'], ls, color='#999999', lw=1.0)
        rows += [dict(row=key, R_mm=x, eff=e, credit=c, accidental=a)
                 for x, e, c, a in zip(R, r['eff'], r['credit'], r['accidental'])]
    ax.axvline(5.0, color='#D55E00', lw=0.9, ls=':')
    ax.text(5.3, 63, 'R = 5 mm', color='#D55E00', fontsize=9)
    ax.set_xscale('log'); ax.set_xlim(0.25, 40); ax.set_ylim(60, 100)
    ax.set_xlabel('matching radius R at the mesh [mm]'); ax.set_ylabel('det3 efficiency [%]')
    ax2.grid(False); ax2.set_ylabel('accidental match [%]', color='#777777'); ax2.set_ylim(0, 5); ax2.tick_params(colors='#777777')
    ax.legend(fontsize=8, frameon=False, loc='center right'); ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    fs.save(fig, os.path.join(OUT, 'eff_vs_radius'), data=pd.DataFrame(rows))
    fig.savefig(os.path.join(OUT, 'eff_vs_radius.pdf'))


def xy_pairing():
    E = pd.read_parquet(os.path.join(RES, 'xy_pairing_events.parquet'))
    ref = json.load(open(os.path.join(RES, 'xy_pairing.json')))
    med = np.mean([ref[k]['reference']['lq_med'] for k in ('sat_det3', 'p2_det3')])
    rs = np.mean([ref[k]['reference']['lq_rsig'] for k in ('sat_det3', 'p2_det3')])
    true = np.r_[E.lq_true_m, E.lq_true_o]; swap = np.r_[E.lq_swap_1, E.lq_swap_2]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    b = np.linspace(-2.5, 2.5, 101)
    ax.axvspan(med - rs, med + rs, color='#E69F00', alpha=0.18, lw=0, label='single tracks (±1σ)')
    ax.hist(true, b, histtype='step', color='#0072B2', lw=1.4, label='true pairing (M3-tagged)')
    ax.hist(swap, b, histtype='step', color='#D55E00', lw=1.2, label='swapped pairing')
    ax.set_xlabel(r'$\ln(q_X/q_Y)$ of an X/Y segment pair'); ax.set_ylabel('segment pairs')
    ok = ref['pooled']['all']
    ax.text(0.02, 0.97, f"{ok['n']} real two-particle events\ncorrect: charge {100 * ok['charge']['frac']:.1f} %, "
            f"charge + time {100 * ok['charge_time']['frac']:.1f} %", transform=ax.transAxes, va='top', fontsize=8.5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
    ax.legend(fontsize=8, frameon=False, loc='upper right'); ax.grid(alpha=0.25)
    fig.tight_layout()
    fs.save(fig, os.path.join(OUT, 'xy_pairing'), data=pd.DataFrame(dict(lq_true=pd.Series(true), lq_swap=pd.Series(swap))))
    fig.savefig(os.path.join(OUT, 'xy_pairing.pdf'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fs.use()
    eff_vs_radius(); xy_pairing()
    print('wrote', OUT)
