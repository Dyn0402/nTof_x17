#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- figures and report.html for the HV trade-off.

    ../../.venv/bin/python make_report.py

Writes ``figures/*.png`` and ``report.html`` beside this file.  Every number in
the HTML comes from ``hv_tradeoff.results()``, so re-running after the inputs
move updates the tables, the figures and the verdict together.
"""
from __future__ import annotations

import html
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hv_tradeoff as T                                    # noqa: E402
import plotstyle as P                                      # noqa: E402

FIG = os.path.join(HERE, 'figures')
DECK = os.path.normpath(os.path.join(HERE, '..', '..', 'mpgd26', 'slides',
                                     'assets', 'img'))
OP, GAIN = T.OP_V, T.GAIN_V

# The report wants a headline burned into each canvas; the DECK does not --
# its own rule is that a figure never repeats its slide's title, and a backup
# slide sets that title in HTML.  Same figures, one flag.
TITLES = True
DECK_NAMES = {'gas_map.png': 'hv_gas_map.png',
              'bench_mapped.png': 'hv_bench_mapped.png',
              'ntof_ladders.png': 'hv_ladders.png'}


def _title(ax, headline, sub=None):
    if TITLES:
        P.title(ax, headline, sub)


def save(fig, name):
    if not TITLES:
        os.makedirs(DECK, exist_ok=True)
        path = os.path.join(DECK, DECK_NAMES[name])
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  -> {path}')
        return name
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  -> {path}')
    return name


# --------------------------------------------------------------------------- #
# 1. the map itself
# --------------------------------------------------------------------------- #

def fig_gas_map(r):
    from gain_map import GainMap
    gm = GainMap(pressure='CERN_450m')
    v = np.arange(400, 620, 1.0)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))

    for gas, col, lab in (('Ar/Iso 95/5', P.DET_COLOR['C'], 'Ar/iso 95/5 — the bench'),
                          ('Ar/Iso 90/10', P.DET_COLOR['A'], 'Ar/iso 90/10 — n_TOF')):
        ax.plot(v, np.exp(gm.ln_gain(gas, v)), color=col, lw=2.2, label=lab)
    ax.set_yscale('log')
    ax.set_xlim(400, 620)
    ax.set_ylim(1e2, 1e6)

    # the equal-gain arrow at the bench's own optimum
    vb = r['bench']['eff_max_V']
    g = float(np.exp(gm.ln_gain('Ar/Iso 95/5', vb)))
    vn = vb + r['shift']['run_55']['gas']
    ax.plot([vb, vn], [g, g], color=P.BAND_DEAD, lw=1.4, ls='--', zorder=4)
    ax.annotate('', xy=(vn, g), xytext=(vb, g),
                arrowprops=dict(arrowstyle='-|>', color=P.BAND_DEAD, lw=1.6))
    ax.text((vb + vn) / 2, g * 1.55, f'+{vn - vb:.0f} V for the same gain',
            ha='center', color=P.BAND_DEAD, fontsize=11, fontweight='bold')
    for x, c in ((vb, P.DET_COLOR['C']), (vn, P.DET_COLOR['A'])):
        ax.plot([x], [g], 'o', ms=7, color=c, zorder=5)
    ax.axvspan(520, 560, color=P.ACCENT, alpha=0.10, lw=0, zorder=0)
    ax.text(540, 1.5e2, 'where n_TOF ran', ha='center', color=P.ACCENT,
            fontsize=10.5, fontweight='bold')
    ax.set_xlabel('amplification voltage  [V]')
    ax.set_ylabel('simulated gas gain')
    ax.legend(loc='upper left')
    _title(ax, 'The gas costs 73 volts',
           'Garfield++ / Magboltz, CERN pressure — garfield_sim/results/hv_equivalence.json')
    P.strip(ax)

    # the bracket
    b = r['bracket']
    rows = [('equivalence table\n(the one used)', b['equivalence_table']['dV'], P.ACCENT),
            ('T6 meshfield ×\nmeasured slope', b['meshfield_530V']['dV_meas_slope'], P.MUTED),
            ('T6 meshfield ×\nsimulated slope', b['meshfield_530V']['dV_sim_slope'], P.MUTED),
            ('uniform field ×\nmeasured slope', b['uniform_field']['dV_meas_slope'], P.MUTED),
            ('uniform field ×\nsimulated slope', b['uniform_field']['dV_sim_slope'], P.MUTED)]
    y = np.arange(len(rows))[::-1]
    ax2.barh(y, [x[1] for x in rows], color=[x[2] for x in rows], height=0.55)
    for yy, (lab, val, _) in zip(y, rows):
        ax2.text(val + 1.5, yy, f'{val:.0f} V', va='center', fontsize=11,
                 color=P.INK, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels([x[0] for x in rows], fontsize=10)
    ax2.set_xlim(0, 118)
    ax2.set_xlabel('95/5 → 90/10 shift  [V]')
    _title(ax2, 'and the sim is only worth ±20 V of it',
           'three determinations of one ratio, two slopes to divide by')
    ax2.grid(axis='y', visible=False)
    P.strip(ax2, left=False)
    fig.tight_layout()
    return save(fig, 'gas_map.png')


# --------------------------------------------------------------------------- #
# 2. the bench curve, on the n_TOF axis
# --------------------------------------------------------------------------- #

def fig_bench_mapped(r):
    """Both det3 efficiency scans, on the n_TOF axis, in both noise eras.

    Rebuilt 2026-08-25 when the deck panel switched to the saturday scan.
    **Re-derived 2026-08-28** -- both scans were rebuilt on the current chain
    (mx_june_cosmic_qa/10b_hv_scan_efficiency.py); the files this used to read
    were written on 29 June and plateaued at 81 %.  See BENCH_EFF_SAT in
    hv_tradeoff.py for the three reasons and the parked originals.

    The figure carries three facts, and the correction changed two of them:

      * **The plateau is 93-95 %**, and it matches det3's published headline
        (93.1 % hits / 93.5 % wft at 490 V on this run's own long run).
      * **The two scans now AGREE.**  They used to differ by ~10 points and
        that gap was explained by the top slot's M3 lever arm; the gap is gone
        and the explanation is withdrawn.  The lever arm survives where it
        belongs, in the core residual (0.34-0.41 mm bottom slot against
        0.44-0.59 mm top), and never cost efficiency at a 5 mm match.
      * **There is NO turn-on inside either scan.**  The 27 June scan reaches
        425 V and reads 89.6 % there, not 49 %.  What the figure shows below
        the plateau is a ~4-point sag, not a turn-on, and the collapse at the
        top is discharges.  So the honest statement is "flat across the whole
        n_TOF window, and we never found the low edge", NOT "we measured the
        turn-on".

    Everything is placed with the PRODUCTION ledger except the one deliberate
    ghost, so the top axis is unambiguous.
    """
    sv, se, sde, ssp = T.bench_efficiency_saturday()
    bv, be, bde, _bsp = T.bench_efficiency()
    s55 = r['shift']['run_55']['total']
    sprod = r['shift']['production']['total']

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.fill_between(sv + sprod, 0, ssp * 100, color=P.BAND_DEAD, alpha=0.20,
                    lw=0, label='discharge fraction (27 June)')
    # the deck's curve, exactly as the slide places it
    ax.errorbar(sv + sprod, se * 100, yerr=sde * 100, fmt='o-',
                color=P.DET_COLOR['A'], ms=5, lw=1.8, capsize=3,
                label='27 June scan, top slot — THE DECK CURVE')
    # the same points in the July configuration: the gap IS the 23 July step
    ax.plot(sv + s55, se * 100, '--', color=P.MUTED, lw=1.3, alpha=0.85,
            label='the same, in the quieter July noise (before 23 July)')
    # the independent scan: same chamber, bottom slot, five days earlier
    ax.errorbar(bv + sprod, be * 100, yerr=bde * 100, fmt='s-', color=P.ACCENT,
                ms=4, lw=1.3, alpha=0.85, capsize=2,
                label='22 June scan, bottom slot (independent, agrees)')

    ax.set_ylim(0, 100)
    ax.set_xlim(sv[0] + sprod - 14, sv[-1] + sprod + 6)
    for x, col, lab in ((OP, P.INK, f'{OP} V — where we ran'),
                        (GAIN, P.BAND_DEAD, f'{GAIN} V')):
        ax.axvline(x, color=col, lw=1.6, ls=':' if x != OP else '-', zorder=5)
        ax.text(x, 96, ' ' + lab, color=col, fontsize=11, fontweight='bold',
                ha='left', va='top', rotation=90)
    ax.axvspan(517, 563, color=P.ACCENT, alpha=0.08, lw=0, zorder=0)

    ax.set_xlabel('n_TOF-equivalent amplification voltage, Ar/iso 90/10  [V]'
                  '   (production ledger)')
    ax.set_ylabel('per cent')
    ax.legend(loc='lower right', fontsize=10.0)
    if TITLES:
        ax.set_title('The bench curves, moved onto the n_TOF voltage axis '
                     '— flat at 93–95 % across the whole window',
                     loc='left', color=P.INK, pad=44)

    top = ax.twiny()
    top.set_xlim(*(np.array(ax.get_xlim()) - sprod))
    top.set_xlabel('the bench voltage it came from, Ar/iso 95/5  [V]', labelpad=7)
    for side in ('right', 'left', 'bottom'):
        top.spines[side].set_visible(False)
    P.strip(ax)
    fig.tight_layout()
    return save(fig, 'bench_mapped.png')


# --------------------------------------------------------------------------- #
# 2b. how much gain we actually had
# --------------------------------------------------------------------------- #

def fig_gain_scale(r):
    """The charge ladder as a per cent of the gain that fills the readout.

    Added 2026-08-28 when the deck's top-left panel swapped from efficiency to
    gain; re-anchored and re-scaled 2026-08-29 (Dylan: *"can we put 100 % at
    497 V instead? Then also make it linear y-axis"*).

    100 % is ``T.saturating_voltage(0.5)`` -- the median track's peak strip
    just filling the 12-bit sample of the **200 fC** DREAM the scan ran, bench
    497 V, n_TOF 565 V.  n_TOF itself ran **600 fC**, which needs 3x the charge
    and would put 100 % at n_TOF 586 V; that scale is in the table beside this
    figure and in ``results()['gain_scale']['ntof600']``.

    Only gas and pressure move the voltage (-67.85 V), so the top axis is the
    plain gas equivalence: n_TOF 560 V is bench 492 V.
    """
    g = r['gain_scale']
    d = T.bench_gain_on_ntof_axis('q_sum')
    dw = T.bench_gain_on_ntof_axis('q_win')
    vn, pct, v_opt = d['v'], d['pct'], d['v_opt']
    shift = d['shift']

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    y1 = 165.0
    ax.axhspan(100, y1, color=P.COPPER, alpha=0.10, lw=0, zorder=0)
    ax.axhline(100, color=P.COPPER, lw=1.6, ls='--', zorder=2)

    ax.plot(dw['v'], dw['pct'], 's--', color=P.MUTED, ms=4, lw=1.2, alpha=0.85,
            label='raw window sum (no threshold, no model)')
    ax.plot(vn, pct, 'o-', color=P.DET_COLOR['A'], ms=5, lw=1.8,
            label='deconvolved forward-fit charge — THE DECK CURVE')
    ax.plot([v_opt], [100], 'o', ms=10, mfc=P.SURFACE, color=P.COPPER,
            markeredgewidth=2.0, zorder=6)

    ax.text(float(vn.min()) + 2.0, 78.0,
            f'100 % = the median track\'s peak strip just fills the 12-bit '
            f'sample\nn_TOF {v_opt:.0f} $\\pm$ 20 V  (bench '
            f'{g["v_sat50_bench"]:.0f} V, on the 200 fC range the scan ran)',
            ha='left', va='center', color=P.COPPER, fontsize=10.5,
            linespacing=1.35)
    ax.text(0.99, 0.955, 'over 100 % — the median track is clipping',
            transform=ax.transAxes, ha='right', va='top', color=P.COPPER,
            fontsize=10.5)

    for x, col, lab in ((OP, P.INK, f'{OP} V — where we ran'),
                        (GAIN, P.BAND_DEAD, f'{GAIN} V')):
        ax.axvline(x, color=col, lw=1.6, ls='-' if x == OP else ':', zorder=5)
        yy = float(np.interp(x, vn, pct))
        ax.text(x, 3.0, ' ' + lab, color=col, fontsize=11, fontweight='bold',
                ha='left', va='bottom', rotation=90)
        ax.annotate(f'{yy:.0f} %', xy=(x, yy), xytext=(-9, 3),
                    textcoords='offset points', ha='right', va='center',
                    color=col, fontsize=12, fontweight='bold')
    ax.axvspan(517, 563, color=P.ACCENT, alpha=0.08, lw=0, zorder=0)

    ax.set_xlim(float(vn.min()) - 2, float(vn.max()) + 2)
    ax.set_ylim(0.0, y1)
    ax.set_yticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_xlabel('n_TOF amplification voltage, Ar/iso 90/10  [V]'
                  f'   (the bench ladder, read at V − {shift:.1f} V)')
    ax.set_ylabel('collected charge  [% of optimal gain]')
    ax.legend(loc='upper left', fontsize=10.0, framealpha=0.92)
    if TITLES:
        ax.set_title('We ran at about a third of the gain that fills the '
                     'readout — and the last 20 V is where it runs away',
                     loc='left', color=P.INK, pad=44)

    top = ax.twiny()
    top.set_xlim(*(np.array(ax.get_xlim()) - shift))
    top.set_xlabel('the bench voltage it came from, Ar/iso 95/5  [V]', labelpad=7)
    for side in ('right', 'left', 'bottom'):
        top.spines[side].set_visible(False)
    P.strip(ax)
    fig.tight_layout()
    return save(fig, 'gain_scale.png')


# --------------------------------------------------------------------------- #
# 3. the two n_TOF ladders
# --------------------------------------------------------------------------- #

def fig_ntof_ladders(r):
    v, b1, b2 = T.ntof_yield()
    rv, rq, rms = T.recovery_ladder()

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.plot(v, b2, 'o-', color=P.DET_COLOR['A'], lw=2.0, ms=7,
            label='MIP-track rate, 16–28 ms  (clear of the recovery — use this)')
    ax.plot(v, b1, 's--', color=P.MUTED, lw=1.4, ms=5.5, alpha=0.85,
            label='MIP-track rate, 8–12 ms  (inside the recovery above 550 V)')
    ax.set_ylabel('tracks per trigger  [%]')
    ax.set_xlabel('amplification voltage, Ar/iso 90/10  [V]')
    ax.set_ylim(0, 17.5)
    ax.set_xlim(516, 564)

    ax2 = ax.twinx()
    ax2.plot(rv, rms, '^', color=P.BAND_DEAD, ms=6, alpha=0.55, lw=0)
    vv = np.linspace(516, 564, 100)
    ax2.plot(vv, T.recovery_at(vv), color=P.BAND_DEAD, lw=2.0,
             label='post-flash recovery (run_57)')
    ax2.set_yscale('log')
    ax2.set_ylim(0.5, 40)
    ax2.set_ylabel('post-flash recovery  [ms]', color=P.BAND_DEAD)
    ax2.tick_params(axis='y', colors=P.BAND_DEAD)
    ax2.axhspan(4.46, 14.1, color=P.BAND_DEAD, alpha=0.10, lw=0)
    ax2.text(517, 6.2, 'the thermal neutrons arrive in here',
             color=P.BAND_DEAD, fontsize=10)
    ax2.grid(False)

    ax.axvline(OP, color=P.INK, lw=1.6, zorder=5)
    ax.text(OP - 0.6, 0.35, 'where we ran ', color=P.INK, fontsize=11,
            fontweight='bold', ha='right', va='bottom')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10.5)
    _title(ax, 'Both halves of the trade, measured on the same chamber',
           'yield: run_55, 18 July · recovery: run_57 — chamber A, drift 600 V in both')
    P.strip(ax)
    fig.tight_layout()
    return save(fig, 'ntof_ladders.png')


# --------------------------------------------------------------------------- #
# 4. the product
# --------------------------------------------------------------------------- #

def fig_tradeoff(r):
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    v, vis, rel, prod = T.figure_of_merit('b2')
    _, _, rel1, prod1 = T.figure_of_merit('b1')

    ax.plot(v, vis / vis.max(), 'o-', color=P.BAND_DEAD, lw=2.0, ms=6,
            label='X17 rate that arrives after the chamber is alive')
    ax.plot(v, rel, 'o-', color=P.DET_COLOR['A'], lw=2.0, ms=6,
            label='tracks reconstructed per trigger (16–28 ms)')
    ax.plot(v, prod / prod.max(), 'o-', color=P.ACCENT, lw=3.0, ms=8,
            label='their product — what the campaign was optimising')
    ax.plot(v, prod1 / prod1.max(), '--', color=P.ACCENT, lw=1.3, alpha=0.6,
            label='the same product on the 8–12 ms window (biased — see text)')

    kb = int(np.argmax(prod))
    ax.annotate(f'optimum {v[kb]:.0f} V', xy=(v[kb], 1.0), xytext=(v[kb], 1.10),
                ha='center', color=P.ACCENT, fontweight='bold', fontsize=12,
                arrowprops=dict(arrowstyle='-|>', color=P.ACCENT, lw=1.5))
    ax.axvline(OP, color=P.INK, lw=1.6)
    ax.text(OP - 0.7, 0.06, 'we ran here ', color=P.INK, fontsize=11.5,
            fontweight='bold', ha='right')
    ax.set_ylim(0, 1.52)
    ax.set_xlim(516, 564)
    ax.set_xlabel('amplification voltage, Ar/iso 90/10  [V]')
    ax.set_ylabel('relative to its own maximum')
    ax.legend(loc='upper left', fontsize=10.5)
    _title(ax, f'The trade has a maximum, and it is {v[kb]:.0f} V',
           f'{OP} V delivers {prod[list(v).index(OP)] / prod.max() * 100:.0f} % '
           'of it on the unbiased window')
    P.strip(ax)
    fig.tight_layout()
    return save(fig, 'tradeoff.png')


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

CSS = """
:root{--ink:#23373b;--muted:#5d7176;--line:#d3dadb;--accent:#c66a0f;
       --dead:#a8402f;--good:#2e8b57;--paper:#fff;--ground:#f4f6f6;}
*{box-sizing:border-box}
body{margin:0;padding:0 1.5rem 5rem;background:var(--ground);color:var(--ink);
     font:16px/1.6 "Noto Sans","Segoe UI",Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto}
header{padding:2.6rem 0 1rem}
h1{font-size:2.1rem;line-height:1.12;margin:0 0 .5rem}
h2{font-size:1.35rem;margin:2.6rem 0 .3rem}
h3{font-size:1.02rem;margin:1.6rem 0 .3rem}
.eyebrow{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
         color:var(--muted);font-weight:700;margin:0 0 .8rem}
.dek{color:var(--muted);max-width:70ch;margin:0}
.verdict{background:var(--paper);border-left:4px solid var(--accent);
         padding:1.1rem 1.3rem;margin:1.6rem 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.verdict p{margin:0 0 .6rem}.verdict p:last-child{margin:0}
.tiles{display:flex;flex-wrap:wrap;gap:1rem;margin:1.4rem 0}
.tile{background:var(--paper);padding:.9rem 1.1rem;flex:1 1 190px;
      box-shadow:0 1px 3px rgba(0,0,0,.06)}
.tile .n{font-size:1.7rem;font-weight:700;line-height:1.1;
         font-variant-numeric:tabular-nums}
.tile .l{font-size:.8rem;color:var(--muted);margin-top:.25rem}
table{border-collapse:collapse;width:100%;background:var(--paper);font-size:.92rem;
      box-shadow:0 1px 3px rgba(0,0,0,.06);margin:.8rem 0}
th,td{padding:.5rem .8rem;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;
         color:var(--muted)}
tbody td{font-variant-numeric:tabular-nums;font-family:"Noto Sans Mono",monospace}
tbody td:first-child{font-family:inherit}
tr.op td{background:#fdf1e2;font-weight:700}
figure{margin:1.4rem 0;background:var(--paper);padding:.8rem;
       box-shadow:0 1px 3px rgba(0,0,0,.06)}
figure img{width:100%;display:block}
figcaption{font-size:.87rem;color:var(--muted);margin-top:.6rem}
figcaption b{color:var(--ink)}
code{font-family:"Noto Sans Mono",monospace;font-size:.87em;background:var(--ground);
     padding:.05em .3em}
.warn{border-left:4px solid var(--dead)}
.warn h3{color:var(--dead);margin-top:0}
ul{max-width:74ch}li{margin-bottom:.45rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
       font-size:.83rem;color:var(--muted)}
"""


def esc(s):
    return html.escape(str(s))


def build_html(r, figs):
    b = r['bench']
    s55, sp = r['shift']['run_55'], r['shift']['production']
    m = r['mapped']
    sat = r['saturday']
    f2, f1 = r['fom']['b2'], r['fom']['b1']
    v, vis, rel, prod = T.figure_of_merit('b2')
    _, _, rel1, prod1 = T.figure_of_merit('b1')

    ledger = ''.join(
        f'<tr><td>{esc(n)}</td><td>{a:+.1f}</td><td>{c:+.1f}</td>'
        f'<td>{esc(why)}</td></tr>'
        for n, a, c, why in (
            ('Gas — Ar/iso 95/5 → 90/10', s55['gas'], sp['gas'],
             'more quencher, less gain at the same field'),
            ('Pressure — Saclay 160 m → CERN 450 m', s55['pressure'],
             sp['pressure'], 'thinner air at CERN, so more gain'),
            ('Electronics — CSA range and noise', s55['electronics'],
             sp['electronics'],
             '200 → 600 fC full scale; σ 6.85 → 3.90 / 9.80 ADC'),
            ('TOTAL', s55['total'], sp['total'], '')))

    g = r['gain_scale']
    onset_row = ''.join(f'<td>{g["onset"][vv] * 100:.0f}</td>'
                        for vv in sorted(g['onset']) if vv >= 475)
    onset_hdr = ''.join(f'<th>{vv}</th>'
                        for vv in sorted(g['onset']) if vv >= 475)
    gain_rows = ''.join(
        f'<tr class="{"op" if vv == OP else ""}"><td>{vv} V</td>'
        f'<td>{g["pct"][vv]:.1f}</td><td>{g["pct_qwin"][vv]:.1f}</td>'
        f'<td>{g["ntof600"]["pct"][vv]:.1f}</td>'
        f'<td>{vv - g["shift"]:.0f} V</td></tr>'
        for vv in sorted(g['pct']))
    adc_ledger = ''.join(
        f'<tr><td>{esc(n)}</td><td>{x:+.1f}</td><td>{esc(why)}</td></tr>'
        for n, x, why in (
            ('Gas — Ar/iso 95/5 → 90/10', g['adc_shift']['gas'],
             'the same term as above'),
            ('Pressure — Saclay → CERN', g['adc_shift']['pressure'],
             'the same term as above'),
            ('CSA range — 200 → 600 fC', g['adc_shift']['csa'],
             '3× less ADC per electron, so 3× the avalanche to reach the rail'),
            ('Per-channel noise', 0.0,
             'excluded on purpose — the rail is a fixed ADC count'),
            ('TOTAL', g['adc_shift']['total'], '')))

    trade = ''.join(
        f'<tr class="{"op" if vv == OP else ""}"><td>{vv:.0f} V</td>'
        f'<td>{T.recovery_at(vv):.2f}</td><td>{vis[i] * 100:.1f}</td>'
        f'<td>{rel[i]:.2f}</td><td>{prod[i] / prod.max():.2f}</td>'
        f'<td>{rel1[i]:.2f}</td><td>{prod1[i] / prod1.max():.2f}</td></tr>'
        for i, vv in enumerate(v))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What the amplification voltage bought — chamber A</title>
<style>{CSS}</style></head><body><div class="wrap">

<header>
<p class="eyebrow">n_TOF July analysis · HV trade-off · chamber A</p>
<h1>What we gave up by running at {OP} V</h1>
<p class="dek">The flash recovery time says how high the amplification voltage
may go. This is the other half: what the voltage buys, from the June cosmic
bench at Ar/iso 95/5, carried onto the n_TOF voltage axis with the Garfield
equivalence map, and checked against n_TOF's own resist scan in the production
gas.</p>
</header>

<div class="verdict">
<p><b>{OP} V cost essentially no efficiency, in either noise era.</b> The
27 June bench scan puts the chamber's plateau at
<b>{sat['plateau'] * 100:.1f} %</b> over {sat['plateau_lo_V']:.0f}–{sat['plateau_hi_V']:.0f} V
(95/5) — the same chamber's published headline. Carried onto the n_TOF axis by
the July ledger, {OP} V is worth {sat['eff_at_op']['run_55'] * 100:.0f} %
against {GAIN} V's {sat['eff_at_gain']['run_55'] * 100:.0f} %; carried by the
<em>production</em> ledger, after the 23 July noise step,
<b>{sat['eff_at_op']['production'] * 100:.0f} %</b> against
{sat['eff_at_gain']['production'] * 100:.0f} %. <b>Both placements put the
setpoint on the plateau</b>, and the {OP}→{GAIN} V difference is ~1 point in
each — inside the scan's own point-to-point scatter. So <b>the honest cost of
the {OP} V decision, in detection efficiency, is not measurable here</b>: what
the extra volts bought was dead time, and what they cost was nothing the bench
can see.
<em>Superseded 2026-08-28:</em> this paragraph used to read that the noise step
pushed {OP} V "onto the shoulder of the turn-on, not the plateau", worth 69 %
against 81 %, and called that ~12-point gap the cost of the decision. That came
from the 29 June CSVs; there is no turn-on and there was no 12-point gap.
The product of visible X17 rate and reconstructed-track yield — which uses the
n_TOF MIP ladder, not this curve, and is unaffected — still peaks at
<b>{f2['best_V']:.0f} V</b>, with {OP} V within {f2['at_op'] / f2['best'] * 100:.0f} % of it.</p>
<p><b>The usable window is about 20 V wide and it has a ceiling on both
sides.</b> Below ~535 V the chamber falls off its own detection turn-on in the
production noise, and stops reconstructing tracks well before that; above ~555 V
the recovery passes the thermal peak and the rate goes to nothing; and the
bench's own discharge onset maps to <b>{m['spark_10pct_ntof_V']:.0f} V</b>, a
few volts above where the dead time already ended the argument.</p>
<p><b>The 23 July noise step moved the operating point, not just the noise.</b>
In the run_55 configuration {OP} V is worth bench {m['op_bench_V_run55']:.0f} V;
in the production configuration it is worth bench
{m['op_bench_V_prod']:.0f} V — {sp['electronics'] - s55['electronics']:.0f} V
lower, for the same setpoint, because the threshold rides on a doubled noise
floor.</p>
</div>

<div class="tiles">
<div class="tile"><div class="n">+{s55['gas']:.0f} V</div>
  <div class="l">the gas costs this much voltage, 95/5 → 90/10</div></div>
<div class="tile"><div class="n">{m['op_bench_V_run55']:.0f} V</div>
  <div class="l">bench-equivalent of {OP} V (run_55 configuration)</div></div>
<div class="tile"><div class="n">{f2['best_V']:.0f} V</div>
  <div class="l">optimum of rate × yield, unbiased window</div></div>
<div class="tile"><div class="n">±20 V</div>
  <div class="l">what the map is actually worth</div></div>
</div>

<h2>1. The gas costs 73 volts</h2>
<p>The bench ran Ar/iso 95/5 and n_TOF ran 90/10, so the bench's efficiency
curve cannot be read on the n_TOF voltage axis without a gain map. The
repository already has one —
<code>garfield_sim/results/hv_equivalence.json</code>, per-mixture
ln&nbsp;G&nbsp;=&nbsp;a&nbsp;+&nbsp;bV&nbsp;+&nbsp;c₂V² fits at two site
pressures — and it is the authority used here, inverted so that the match is
made on <em>gain</em> rather than by a constant offset.</p>
<figure><img src="figures/{figs['gas']}" alt="Left: simulated gas gain against voltage for Ar/iso 95/5 and 90/10, with an arrow showing the 73 volt shift at equal gain. Right: five determinations of that shift as a bar chart, spanning 63 to 103 volts.">
<figcaption><b>Left:</b> the two mixtures at CERN pressure. The shift is flat to
±0.6 V across 400–590 V, which is why everything downstream treats it as a
constant. <b>Right:</b> the same ratio from three independent Garfield products,
each divided by either the simulated or the measured gain slope. <b>The spread
— 63 to 103 V — is the honest uncertainty on the map</b>, and it is dominated
by the known factor-1.4 disagreement between the bench's measured gain slope
({b['slope10']:.3f} per 10 V) and Garfield's ({b['sim_slope10']:.3f}).</figcaption>
</figure>

<h2>2. The ledger</h2>
<p>Three terms carry a bench voltage onto the n_TOF axis. Only the first is
about the gas; the third is the one that is easy to forget, and it is the
largest single change between the run_55 scan and the production period.</p>
<table><thead><tr><th>Term</th><th>run_55 era</th><th>production</th>
<th>why</th></tr></thead><tbody>{ledger}</tbody></table>
<p>The CSA range is not an assumption: every saved bench configuration carries
<code>Feu * Dream * 6 0xAAAA</code> = 200 fC, and all 56 n_TOF pedestal contexts
from 1 July to 10 August carry <code>0xffff</code> = 600 fC. <b>That also
settles the deck's open question about which range production ran on.</b></p>

<h2>3. The bench curve, on the n_TOF axis</h2>
<figure><img src="figures/{figs['bench']}" alt="Reconstruction efficiency and discharge fraction from the June bench scan, plotted against the equivalent n_TOF voltage, with the operating point marked.">
<figcaption>det3 <em>is</em> chamber A, so this is the same physical detector.
Efficiency is M3-referenced, drift 1000 V, 22 June. The dashed grey copy is the
same measurement placed by the production ledger instead of the run_55 one —
the gap between the two curves is the 23 July noise step, worth
{sp['electronics'] - s55['electronics']:.0f} V.</figcaption>
</figure>

<h2>4. How much gain did we actually have?</h2>
<p>Efficiency answers <em>were you detecting?</em> and the answer is a flat
yes. It does not answer <em>how much gain did you have</em>, and that is the
quantity the recovery time is being paid for. Scale it against the gain the
readout could actually take: <b>100 % is the voltage at which the peak strip of
the median track just fills the 12-bit sample</b>.</p>
<p><code>peak_amp</code> is the tallest <em>sample</em> of the tallest <em>strip</em> of
the event — the max strip, which is the thing that clips first. It starts clipping in a few per
cent of tracks well before the median does:</p>
<table><thead><tr><th>bench V</th>{onset_hdr}</tr></thead>
<tbody><tr><td>max strip railed [%]</td>{onset_row}</tr></tbody></table>
<p><b>5 % of tracks by {g['onset_5pct']:.0f} V, a quarter by {g['onset_quartile']:.0f} V,
half by {g['v_sat50_bench']:.0f} V, 90 % by {g['onset_90pct']:.0f} V.</b></p>

<p><b>Reading ~500 V off the gain plot is also right, and it is a different statement.</b>
At the 50 % point half the sample is still below the rail, so the <em>median amplitude</em>
only reaches the nominal 3871.5 ADC near <b>500 V</b> — which is where the p50 marker visibly
lies on the rail line in <code>gain_vs_hv.png</code> (67 % of events clipped there, median at
97 % of the rail). Anchoring at 500 V instead of 497 would lower every percentage on this
scale by ~13 %.</p>

<p>The 50 % point barely cares where the clipping line is drawn:
<b>496.4 / 497.0 / 497.1 V</b> for a cut at 0.88 / 0.92 / 0.95 of the rail. A cut at 0.98 gives
508.6 V, but that is no longer a clipping test — per-channel pedestal subtraction spreads the
railed population over ~3700–3900 ADC, so it asks whether a channel's rail landed high rather
than whether the event clipped. The spark veto is not what sets it either: 496.8 V on the full
M3-golden fiducial set against 497.0 spark-free.</p>

<p>That point is measured, not modelled — the saturated fraction goes 0.39 at
495 V to 0.66 at 500 V in <em>both</em> views, so the 0.5 crossing is bracketed
by two points 5 V apart at <b>bench {g['v_sat50_bench']:.0f} V</b>, which is
<b>n_TOF {g['v_opt_ntof']:.0f} V</b>. Plotted is
the <b>total collected charge</b>, not the peak sample: the deconvolved forward
fit censors railed samples and so keeps measuring where the peak sample cannot.
The model-free window sum agrees to 3 %.</p>

<div class="verdict warn"><h3>Full scale of <em>what</em>? It is worth a factor 3</h3>
<p>Bench 497 V fills the <b>200 fC</b> DREAM the scan was taken with. n_TOF ran
<b>600 fC</b> — 3× less ADC per electron — so filling <em>that</em> needs three times the
avalanche, at bench ~518 V, <b>n_TOF {g['ntof600']['v_opt_ntof']:.0f} V</b>. Both are honest;
they answer different questions, and every percentage here moves by 3× between them (the table
carries both).</p>
<p><b>This page leads with the 200 fC one</b>, for three reasons: it is the scan's own
<em>measured</em> saturation point; it leaves nothing on the curve extrapolated (bench
425–505 V covers n_TOF 493–573 V, and 565 V is inside it, where 586 V is 13 V past the last
trustworthy bench point); and the 600 fC setting was forced by the gamma flash — 668 pC on a
single strip, 1113× the DREAM range — rather than chosen for tracking, so referring a
<em>tracking</em> gain to it asks for 3× more avalanche than a MIP measurement needs. Say which
one a number came from; never mix them.</p></div>

<p><b>How a bench voltage becomes an n_TOF one here.</b> Only the gas and the site
pressure move the <em>voltage</em>: n_TOF <i>W</i> is read off the bench ladder at
<b><i>W</i> − {g['shift']:.1f} V</b>, so <b>n_TOF 560 V is bench 492 V</b> — the
plain gas equivalence. The readout change then <b>divides the ADC by three</b>
(200 → 600 fC full scale); it is a factor, not a voltage. Written as one shift
that is +94.1 V, and that form is fine for <em>saying</em> which bench voltage makes
the same ADC — but it must not be used to <em>evaluate</em> the ladder, because it is
exact only for a straight ladder and this one is curved (0.33 per 10 V near 440 V,
0.52 near 495). Doing it that way read the wrong part of the ladder, by −13 % at
520 V and +6 % at 560 V. <b>Corrected 2026-08-28</b>, after Dylan queried the
mapping.</p>
<figure><img src="figures/{figs['gain']}" alt="Collected charge as a percentage of the gain that just saturates the median track's peak strip, on a logarithmic axis against the n_TOF-equivalent voltage. A straight rising line passes through 11 per cent at 540 volts and 25 per cent at 560 volts and reaches 100 per cent at 591 volts, above a shaded band marked as over-gain.">
<figcaption><b>Every setpoint in the deck build is measured</b> — bench
425–505 V covers n_TOF {g['v_meas_ntof'][0]:.0f}–{g['v_meas_ntof'][1]:.0f} V.
The dashed top is not: reaching 100 % needs three times the charge that rails the
<em>bench</em> readout, about 13 V more ladder than exists, so it is continued off
the top five points at {g['slope10']:.2f} per 10 V (the whole-ladder slope would put
the crossing at {g['v_opt_ntof_alt']:.0f} V instead). The gas term's ±20 V slides
the whole curve sideways and takes the crossing with it, without touching the ratios
between setpoints.</figcaption>
</figure>
<table><thead><tr><th>n_TOF</th>
<th>% of optimal gain<br>(forward fit)</th><th>% (window sum)</th>
<th>% if referred to<br>the 600 fC range</th>
<th>bench V it is<br>read at</th></tr></thead>
<tbody>{gain_rows}</tbody></table>
<p><b>Section 2's ledger does not apply here.</b> It carries the per-channel noise,
which is right for an efficiency and wrong for a rail: a rail sits at a fixed number
of ADC counts however noisy the channel is. Three different questions, three
different answers, and all three are correct:</p>
<table><thead><tr><th>&ldquo;the bench equivalent of n_TOF 560 V&rdquo;</th>
<th>shift</th><th>bench V</th><th>what it answers</th></tr></thead><tbody>
<tr><td>same <b>gas gain</b></td><td>+{g['shift']:.1f}</td><td><b>492.1</b></td>
    <td>the plain gas map — what the curve above is evaluated on</td></tr>
<tr><td>same <b>ADC counts</b></td><td>+{g['adc_shift']['total']:.1f}</td>
    <td>465.9</td><td>&hellip;plus the 600 fC CSA, written as volts</td></tr>
<tr><td>same <b>signal-to-noise</b></td><td>+{sp['total']:.1f}</td><td>457.3</td>
    <td>&hellip;plus the 23 July noise step — the efficiency panel's map</td></tr>
</tbody></table>
<table><thead><tr><th>Term</th><th>volts</th><th>why</th></tr></thead>
<tbody>{adc_ledger}</tbody></table>

<h2>5. What n_TOF measured in the production gas</h2>
<figure><img src="figures/{figs['ladders']}" alt="MIP track rate per trigger against amplification voltage in two time windows, with the post-flash recovery time on a second axis.">
<figcaption>Both halves of the trade on one axis, and both measured on chamber A
at drift 600 V two days apart. <b>The 16–28 ms window is the one to trust:</b>
above 550 V the recovery reaches into the 8–12 ms window, so that ladder's top
points are suppressed by the very quantity being traded against.</figcaption>
</figure>

<h2>6. The trade</h2>
<figure><img src="figures/{figs['trade']}" alt="Three curves against voltage: the visible X17 rate falling, the track yield rising, and their product peaking near 550 volts.">
<figcaption>Both factors are relative, so the product is too — it has a maximum
and no units. <b>{f2['best_V']:.0f} V on the unbiased window;
{f1['best_V']:.0f} V on the biased one.</b></figcaption>
</figure>

<table><thead><tr><th>Voltage</th><th>recovery [ms]</th><th>X17 rate left [%]</th>
<th>yield 16–28 ms</th><th>product</th><th>yield 8–12 ms</th><th>product</th>
</tr></thead><tbody>{trade}</tbody></table>

<h2>What this does not rule out</h2>
<div class="verdict warn"><h3>Read these before quoting a number</h3>
<ul>
<li><b>The n_TOF ladder is not an efficiency.</b> Its denominator is a
doubles-trigger whose geometric ceiling per arm is ~50 %, and its numerator
needs a 3–20 strip, ≤25 mm MIP-like cluster in <em>both</em> views. A cluster
loses strips over threshold as the gain falls, so it turns on much later than
detection does. Only its <b>shape</b> is used here.</li>
<li><b>Which is why it can disagree with the mapped bench curve, and does.</b>
The mapped curve says the chamber was on its detection plateau across the whole
n_TOF scan; the ladder rises ×9 over the same span. Both can be true: what the
voltage bought at n_TOF was <em>reconstructability</em>, not detection. If you
want a single sentence, it is that one.</li>
<li><b>Withdrawn 2026-08-25: "the bench never measured below 450 V".</b> It
did — the <b>27 June saturday det3 scan</b> runs 425–525 V with the
<em>same</em> efficiency definition. <b>Only the 520 V frame of the deck build
is extrapolated</b> (it maps to bench 417 V).</li>
<li><b>Withdrawn 2026-08-28: "the 27 June scan shows the turn-on directly".</b>
It does not, and neither scan does. Both were re-derived on the current chain
(<code>mx_june_cosmic_qa/10b_hv_scan_efficiency.py</code>, closure-checked
against this chamber's published breakdown to the third decimal). What this
page used to read — 49 % at 425 V rising to a plateau near 81 % — came from
CSVs written on <b>29 June</b>, before the golden M3 recipe, the significance
floor and the matched-filter reprocessing. Corrected, the plateau is
<b>93–95 %</b> and <b>425 V reads 89.6 %</b>: a ~4-point sag, not a turn-on.
The chamber's own gain ladder, from these same sub-runs, says why — at 425 V
the peak strip carries 69 ADC in the weakest 2 % of events, ~10 σ over the
6.85 ADC bench pedestal, so nothing is near threshold. <b>The scan never
reaches the chamber's low edge.</b></li>
<li><b>Also withdrawn 2026-08-28: the two scans' ~10-point gap and its
explanation.</b> The gap was an artefact of the stale CSVs; on the current
chain the 22 June and 27 June scans agree. The M3 lever arm that was invoked to
explain it is real but shows up in the <em>core residual</em> instead
(0.34–0.41 mm bottom slot against 0.44–0.59 mm top) and never cost efficiency
at a 5 mm match.</li>
<li><b>&ldquo;100 % of optimal gain&rdquo; is a READOUT limit, not a physics
optimum.</b> It says the ADC would be full, nothing more. The same 27 June scan
measures the chamber's angular resolution against M3 over every fitted plane,
and that is best at <b>bench 445&ndash;460 V</b> (1.02&ndash;1.06&deg;) &mdash;
already 1.11&ndash;1.15&deg; by bench 497 V, where the median track saturates,
and 1.31&ndash;1.38&deg; by 515 V. A resolution is a threshold quantity, so the ledger that
carries it across is the signal-to-noise one (+{sp['total']:.1f} V), which puts that optimum at
<b>n_TOF 548&ndash;563 V</b> &mdash; at or a little above where we ran, and well inside the
map's own &plusmn;20 V. So the honest reading of the vertical scale is <em>how much of the
readout's range we were using</em>, not <em>how far from best we were</em>.</li>
<li><b>The vertical scale is a choice, and it is worth 3×.</b> 100 % here is the
gain that fills the <em>200 fC</em> range the bench scan ran. Referred to the 600 fC range
n_TOF actually ran, every number divides by three and 100 % moves to n_TOF
{g['ntof600']['v_opt_ntof']:.0f} V. Neither is more correct; they answer different questions.
The table carries both.</li>
<li><b>Corrected 2026-08-28: how the CSA range enters.</b> The first version folded the factor
of three into the <em>voltage axis</em> as ln 3 / slope = +26.3 V, which is exact only for a
straight ladder — and this one is curved, so it read the wrong part of it (−13 % at n_TOF
520 V, +6 % at 560 V) and hid a 26 V extrapolation inside what looked like a measured curve.
The ladder is now evaluated at the gas-equivalent voltage, full stop, and the range enters as
a factor.</li>
<li><b>The percentages are gain RATIOS.</b> There is no ADC&rarr;electron
calibration for the June bench range, so nothing here is an absolute gas gain,
and the vertical scale means nothing outside its own normalisation. The ratios
themselves survive the map: the gas term's &plusmn;20 V slides the whole curve
sideways without changing them.</li>
<li><b>The flash charge climbs faster than the cosmic gain does.</b> Over
520&ndash;560 V the HV-current flash charge goes 35 &rarr; 277 nC, a factor
7.8, where the bench gain ladder gives 4.7 over the same span
(0.51 against 0.42 per 10 V). A 23 % slope difference across two very different
measurements &mdash; a MIP's avalanche against a gamma flash read off a supply
current, in different gases, mapped between them &mdash; is inside the known
scatter, but it is not zero, and the imon integration systematic is still
open.</li>
<li><b>The gas mixture is an operator label.</b> <code>run_config.json</code>
says "Ar/Iso 95/5" as free text; the mixer that sets and logs isobutane was
commissioned on 7 July, ten days after the bench scan. No certificate exists
for June.</li>
<li><b>The drift fields differ.</b> Bench 1000 V, n_TOF 600 V on chamber A in
both run_55 and run_57, production 700 V. The bench's own drift control puts
that at roughly −15 % in peak amplitude, ~4 V — inside the map's uncertainty,
but not zero.</li>
<li><b>The optimum is soft on the low side and a cliff on the high side.</b>
The recovery scatters ×1.33 rms about its fit, which moves the cliff by several
volts; the low side is flat, so nothing here says {OP} V was a mistake.</li>
</ul>
</div>

<footer>
Generated by <code>ntof_july_analysis/hv_tradeoff/make_report.py</code>;
numbers from <code>hv_tradeoff.py</code> → <code>results.json</code>.
Inputs: bench <code>mx17_det2_det3_overnight_6-22-26</code> and
<code>mx17_det3_saturday_scan_6-27-26</code>; map
<code>garfield_sim/results/hv_equivalence.json</code>; n_TOF
<code>run_55</code> (<code>calib/25_hv_scan_summary.json</code>) and
<code>run_57</code>; pedestals <code>ntof_pedestal_qa/</code> and the bench
pedestal of 22 June.
</footer>
</div></body></html>
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--deck', action='store_true',
                    help='render the three deck copies (no burned-in titles) '
                         'straight into mpgd26/slides/assets/img and stop')
    a = ap.parse_args()
    r = T.results()
    P.use()
    if a.deck:
        global TITLES
        TITLES = False
        fig_gas_map(r), fig_bench_mapped(r), fig_ntof_ladders(r)
        return
    figs = dict(gas=fig_gas_map(r), bench=fig_bench_mapped(r),
                gain=fig_gain_scale(r),
                ladders=fig_ntof_ladders(r), trade=fig_tradeoff(r))
    path = os.path.join(HERE, 'report.html')
    with open(path, 'w') as fh:
        fh.write(build_html(r, figs))
    print(f'  -> {path}')


if __name__ == '__main__':
    main()
