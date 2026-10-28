#!/usr/bin/env python3
"""gc_figures.py -- GC figures (figures/, from results/*.json)."""
import datetime as dt
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gc_common import COL, FIG, NAME, RES, angle_bias_deg  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e4e0', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'legend.frameon': False})
INK2 = '#5f5e5a'
J = lambda f: json.load(open(os.path.join(RES, f)))  # noqa: E731
D = lambda t: dt.datetime.fromtimestamp(t)  # noqa: E731
MK = {'golden': 'o', 'v': 's', 'none': '^'}


def fig_bench_ladder():
    rows = [r for r in J('ladder_rows.json') if r['valid']]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axs[0]
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        rr = sorted([r for r in rows if r['det'] == det and r['drift_V'] in (600, 700, 1000)], key=lambda r: r['t_start'])
        if not rr:
            continue
        c = COL['mx17_' + det[-1]]
        for r in rr:
            ax.plot([D(r['t_start']), D(r['t_end'])], [r['v_lad']] * 2, color=c, lw=3, solid_capstyle='round')
            ax.plot(D(0.5 * (r['t_start'] + r['t_end'])), r['v_lad'], MK[r['refit']], color=c, ms=6,
                    mec='white', mew=1)
        ax.plot([D(0.5 * (r['t_start'] + r['t_end'])) for r in rr], [r['v_lad'] for r in rr], color=c, lw=0.8,
                alpha=0.6, label=f"{det} ({rr[0]['drift_V']:.0f} V drift)")
    ax.set_ylabel('angle scale of the fit: ladder v = w / tan_M3  [um/ns]')
    ax.set_title('Bench: the reconstruction\'s own angle scale per row (FA ladder, M3)', loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.01, 0.02, 'o golden (calibration)   s own v refit   ^ inherits the group kw/v', transform=ax.transAxes,
            color=INK2, fontsize=8)
    ax = axs[1]
    sc = [r for r in rows if r['det'] == 'det3' and 'hv_scan' in r['subrun']]
    B = {r['key']: r for r in J('bench_rows.json') if not r.get('missing')}
    q = np.array([r['q'] for r in sc]); el = np.array([r['eps_lad_pct'] for r in sc])
    et = np.array([B[r['key']]['eps_group_pct'] for r in sc])
    o = np.argsort(q)
    ax.plot(q[o], -et[o], '-o', color=COL['mx17_3'], ms=5, mec='white', label='column duration: T_row / T_calib - 1')
    ax.plot(q[o], el[o], '-s', color=INK2, ms=5, mec='white', label='angle scale (ladder): v_row / v_calib - 1')
    ax.set_xscale('log'); ax.axhline(0, color='#b4b2a9', lw=0.8)
    ax.set_xlabel('median strip-sum charge of unsaturated tracks [ADC] (gain)'); ax.set_ylabel('%')
    ax.set_title('det3 6-27 mesh scans: T follows gain, angle scale flat', loc='left', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'gc_bench_ladder.png'), dpi=130); plt.close(fig)


def fig_bench_within():
    B = J('bench_blocks.json'); R = {r['key']: r for r in J('bench_rows.json') if not r.get('missing')}
    keys = ['o22_long_det2', 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'sat_det3',
            'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3',
            'c26__mx17_det3_day_6-25-26__long_run__mx17_3', 'g_det4',
            'c26__mx17_det3_det4_overnight_6-23-26__long_run__mx17_4', 'g_det6_long', 'g_det7_long']
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for ax, det in zip(axs, ['mx17_2', 'mx17_3', 'mx17_4', 'mx17_6', 'mx17_7']):
        for k, ls in zip([k for k in keys if B[k]['det'] == det], ['-', '--', ':']):
            b = B[k]; t = np.array(b['t']); v = np.array(b['v_rel']); e = np.array(b['err'])
            h = (t - t[0]) / 3600
            ax.plot(h, (v - 1) * 100, ls, color=COL[det], label=f"{R[k]['run'].split('_')[-1]} {R[k]['subrun'][:16]}")
            ax.fill_between(h, (v - e - 1) * 100, (v + e - 1) * 100, color=COL[det], alpha=0.15, lw=0)
        ax.set_title(NAME[det], loc='left'); ax.set_xlabel('hours since row start')
        ax.axhline(0, color='#b4b2a9', lw=0.8); ax.legend(fontsize=6.5, loc='upper left')
    axs[0].set_ylabel('v(t) / v_row - 1  [%]  (1 / column duration)')
    fig.suptitle('Bench: drift velocity within each long row, 10-min blocks of every clean trigger', x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'gc_bench_within.png'), dpi=130); plt.close(fig)


def fig_ntof():
    kb = J('ntof_kband.json')['runs']; camp = J('ntof_colsum_campaign.json'); within = J('ntof_colsum_runs.json')
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, arm in zip(axs, 'AC'):
        k = [x for r in kb if r['arm'] == arm for x in r['subs']]
        kr = np.array([x['k'] for x in k]); ref = np.median(kr)
        ax.plot([D(x['t']) for x in k], (ref / kr - 1) * 100, '.', color='#b4b2a9', ms=4,
                label='capsule-band k per sub-run (production kcal): v / v_med = k_med / k')
        c = [x for x in camp if x['arm'] == arm]
        Tm = np.median([x['T'] for x in c])
        ax.plot([D(x['t']) for x in c], (Tm / np.array([x['T'] for x in c]) - 1) * 100, 'o', color=COL[arm], ms=6,
                mec='white', label='column end of the stacked strip sum, run median (this study)')
        for w in [x for x in within if x['arm'] == arm]:
            cr = next(x for x in c if x['run'] == w['run'])
            off = Tm / cr['T']
            ax.plot([D(t) for t in w['t']], (np.array(w['v_rel']) * off - 1) * 100, '-', color=COL[arm], lw=1.2)
        ax.set_ylabel(f'arm {arm}: v / v_campaign - 1 [%]'); ax.axhline(0, color='#b4b2a9', lw=0.8)
        ax.axvline(dt.datetime(2026, 7, 27, 12), color=INK2, lw=0.8, ls=':')
        ax.legend(fontsize=8, loc='lower left')
    axs[0].set_title('n_TOF: drift velocity across the campaign (lines: 30-min bins inside the six long runs studied)',
                     loc='left')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axs[1].text(dt.datetime(2026, 7, 27, 14), axs[1].get_ylim()[1] * 0.85, '27 Jul access', color=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'gc_ntof_campaign.png'), dpi=130); plt.close(fig)


def fig_consequence():
    th = np.linspace(0, 60, 121)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    for eps, c in zip([0.01, 0.03, 0.10, 0.30], ['#86b6ef', '#3987e5', '#1c5cab', '#104281']):
        ax.plot(th, angle_bias_deg(eps, th), color=c, label=f'tan scale +{eps * 100:.0f} %')
    ax.axhspan(1.06, 1.76, color='#eda100', alpha=0.15, lw=0)
    ax.text(1, 1.15, 'bench angle resolution s68 1.06-1.76 deg', fontsize=8, color=INK2)
    ax.axhline(0.2, color=INK2, ls=':', lw=0.8); ax.text(1, 0.24, '0.2 deg', fontsize=8, color=INK2)
    ax.set_yscale('log'); ax.set_ylim(0.02, 12); ax.set_xlabel('true track angle [deg]')
    ax.set_ylabel('angle error [deg]'); ax.legend(fontsize=8, loc='lower right')
    ax.set_title('What a tan-scale error does to the angle', loc='left')
    ax = axs[1]
    C = J('consequences.json')
    groups = [('bench: within a row', lambda d: d['kind'] == 'bench row' and d['dataset'].split()[2] in
               ('longer_run', 'long_run', 'long_run_resist_490V_drift_1000V', 'long_run_p2_det1_sanity_check'),
               'h_eps_pct'),
              ('bench: inherited row vs its group (ladder)', lambda d: d['kind'] == 'ladder', 'eps_group_pct'),
              ('n_TOF: within a run (column end)', lambda d: d['kind'] == 'n_TOF run (column end)', 'h_eps_pct'),
              ('n_TOF: within a run (band k, noisy)', lambda d: d['kind'].startswith('n_TOF run (capsule'), 'h_eps_pct')]
    for i, (lab, sel, f) in enumerate(groups):
        v = np.array([abs(d[f]) for d in C if sel(d) and np.isfinite(d[f])])
        ax.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), 'o', ms=5,
                color=['#eb6834', '#e34948', '#1baf7a', '#b4b2a9'][i], mec='white')
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups], fontsize=8)
    ax.axvline(0.8, color=INK2, ls=':', lw=0.8); ax.text(0.85, 3.3, '0.2 deg at 30 deg', fontsize=8, color=INK2)
    ax.set_xscale('log'); ax.set_xlabel('worst |tan-scale error| with one v per run [%]')
    ax.set_title('Measured, per dataset', loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'gc_consequence.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    fig_bench_ladder(); fig_bench_within(); fig_ntof()
    if os.path.exists(os.path.join(RES, 'consequences.json')):
        fig_consequence()
    print('figures ->', FIG)
