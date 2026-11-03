"""attach_fig.py -- the Paper I 'attachment' figure: what gas det4 had at SPS (make_figs.py attachment).

Restyled 2026-11-02 (review deck: legend for the gas colours, readable unrotated ticks, bench kept apart):
(a) water of every SPS group (rerun/sps_gasfit/results/verdict.json: head-on stacks fitted for water and O2 on the
    Magboltz grid, at the gas fit's gap 27.9 mm), CO2 and CF4 mixtures in ps.GAS colours.  The bench is not plotted
    point by point (its water is §6's gas_vE_timing / gas_drying); its golden-row range is a grey band for scale
    (rerun/paper1_gas/results/summary.json, det4 there at the old 30.48 mm gap).
(b) O2 of the same groups.  2026-11-03 (Dylan): the det6 6-26 early-night O2-equivalent band (370-600 ppm) is DROPPED:
    it is an inference from the truncated column against an anchored ladder with a model that has no attachment term
    (x2), on one chamber, and was never confirmed by a charge-vs-depth tilt; see the 5(b) note in figures.toml.
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


#: registry key prefix of each gas-fit group (rerun/sps_gasfit/gasfit.py GROUPS) and the date of its clock: the registry's
#: t_lo_s / t_hi_s are seconds on that day's clock (run62/63 on Sun 08-02, run63 flat runs on past midnight; run71 is
#: Mon 08-03).  The README table of rerun/sps_gasfit gives the same start times (Sat 14:25 ... Mon 05:22).
PREFIX = {'co2_run55': 'run55_flatdrift', 'co2_run56': 'run56_m70V', 'co2_run57': 'run57_rot25_co2', 'co2_run59': 'run59_co2',
          'cf4_run62_ladder': 'run62_rot25_ladder', 'cf4_run63_ladder': 'run63_rot25', 'cf4_run63_flat': 'run63_flat',
          'cf4_run71_raw': 'run71_raw'}
DAY0 = {'run55': 1, 'run56': 1, 'run57': 1, 'run59': 1, 'run62': 2, 'run63': 2, 'run71': 3}      # August 2026


def _when(group):
    """(hours since Sat 2026-08-01 00:00, 'Sat 14:25') of the middle of the group's registry window."""
    import csv
    R = [r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv'))) if r['key'].startswith(PREFIX[group] + '__')]
    lo = min(float(r['t_lo_s']) for r in R); hi = max(float(r['t_hi_s']) for r in R)
    h = (DAY0[group.split('_')[1]] - 1) * 24 + 0.5 * (lo + hi) / 3600.0
    return h


def _clock(h):
    d, hh = divmod(h, 24)
    return ['Sat', 'Sun', 'Mon', 'Tue'][int(d)] + '\n%02d:00' % int(round(hh))


def _xpos(h):
    """Hours -> axis position: 22 idle hours between Sat evening and Sun night are cut (axis break)."""
    return h if h < 24 else h - 22.0


def draw(verdict_path, save=None):
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ps.use()
    V = json.load(open(verdict_path))
    B = json.load(open(os.path.join(REPO, 'rerun', 'paper1_gas', 'results', 'summary.json')))['golden_water']
    T = V['table']
    fig, axs = ps.fig_for('full', height=2.5, ncols=2)
    hrs = np.array([_when(r['group']) for r in T]); x = np.array([_xpos(h) for h in hrs])
    bw = [B[d]['water'] for d in ('det2', 'det3', 'det4', 'det6', 'det7')]
    BAND, BEDGE = '#aab2bf', '#6a7583'
    ax = axs[0]
    ax.axhspan(min(bw), max(bw), fc=BAND, alpha=0.55, ec=BEDGE, lw=0.8, ls=':', zorder=0)
    rows = []
    for xi, h, r in zip(x, hrs, T):
        g = GASNAME[r['gas']]; c = ps.GAS[g]
        mk = dict(fmt='s', ms=4.5, color=c, mec=c, mfc='white' if r['edge'] else c, lw=0.9, zorder=4)
        axs[0].errorbar(xi, r['water'], yerr=r['err_w'], **mk)
        axs[1].errorbar(xi, r['o2_ppm'], yerr=r['err_o2'], **mk)
        rows.append(dict(group=r['group'], gas=g, hours_since_Sat_0000=h, clock=_clock(h).replace('\n', ' '), water=r['water'],
                         err_w=r['err_w'], o2_ppm=r['o2_ppm'], err_o2=r['err_o2'], gap_mm=r['gap'], edge=r['edge']))
    ticks = [14, 17, 20, 46, 49, 52, 55]          # Sat 14:00 ... Sun 22:00 ... Mon 07:00 (hours since Sat 00:00)
    for a in axs:
        a.set_xticks([_xpos(t) for t in ticks]); a.set_xticklabels([_clock(t) for t in ticks], fontsize=7.5)
        a.set_xlabel('time of the run (2026-08-01 to 08-03)')
        a.set_xlim(_xpos(ticks[0]) - 1.2, _xpos(ticks[-1]) + 1.2)
        a.text(22.0, 0.015, '//', transform=a.get_xaxis_transform(), ha='center', va='bottom', color=ps.MUTED, fontsize=8,
               backgroundcolor='white', zorder=6)
    ax.set_ylabel('H$_2$O fraction (%)'); ax.set_ylim(0, 2.8)
    ax.text(0.97, 0.705 / 2.8, 'bench golden rows, A-D: %.2f-%.2f %%' % (min(bw), max(bw)), transform=ax.transAxes,
            ha='right', va='center', fontsize=7.5, color='#2b3340')
    ps.legend_above(fig, [Line2D([], [], marker='s', ls='', color=ps.GAS[g]) for g in ('CO2', 'CF4')]
                    + [Patch(fc=BAND, alpha=0.55, ec=BEDGE, ls=':', lw=0.8)],
                    [GASLABEL[g] for g in ('CO2', 'CF4')] + ['bench, for scale (see caption)'], ncol=3)
    ax = axs[1]
    ax.axhline(0, color=ps.MUTED, lw=0.8)
    ax.set_ylabel('O$_2$ concentration (ppm)'); ax.set_ylim(-20, 760)
    ps.panel_labels(axs)
    ps.save(fig, os.path.join(HERE, 'figures', 'attachment'), data=pd.DataFrame(rows))
    print('wrote', os.path.join(HERE, 'figures', 'attachment.pdf'))
    return dict(gap_quoted_mm=V['gap_quoted'], bench_band=[min(bw), max(bw)],
                groups={r['group']: dict(water=round(r['water'], 3), err_w=round(r['err_w'], 3), o2_ppm=round(r['o2_ppm']),
                                         err_o2=round(r['err_o2']), edge=r['edge']) for r in T})
