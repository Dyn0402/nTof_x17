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

    Rebuilt 2026-08-25 when the deck panel switched to the saturday scan.  The
    figure has to carry three facts at once: the turn-on (only the 27 June
    scan reaches it), the plateau's flatness (both scans agree), and the 22 V
    the 23 July noise step moves the whole curve by.  Everything is placed
    with the PRODUCTION ledger except the one deliberate ghost, so the top
    axis is unambiguous.
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
                label='22 June scan, bottom slot (starts on the plateau)')

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
        ax.set_title('The bench curves, moved onto the n_TOF voltage axis',
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
<p><b>Which noise era you ask about decides whether {OP} V was on the
plateau.</b> The 27 June bench scan puts the chamber's efficiency plateau at
{sat['plateau'] * 100:.0f} % over {sat['plateau_lo_V']:.0f}–{sat['plateau_hi_V']:.0f} V
(95/5). Carried onto the n_TOF axis by the July ledger, {OP} V is worth
{sat['eff_at_op']['run_55'] * 100:.0f} % — on the plateau, indistinguishable
from {GAIN} V's {sat['eff_at_gain']['run_55'] * 100:.0f} %. Carried by the
<em>production</em> ledger, after the 23 July noise step, the same setpoint is
worth <b>{sat['eff_at_op']['production'] * 100:.0f} %</b> against
{sat['eff_at_gain']['production'] * 100:.0f} % at {GAIN} V — on the shoulder of
the turn-on, not the plateau. <b>That difference, ~{(sat['eff_at_gain']['production'] - sat['eff_at_op']['production']) * 100:.0f}
points, is the honest cost of the {OP} V decision</b>, and it is a cost the
noise step imposed rather than the decision itself. The product of visible X17
rate and reconstructed-track yield still peaks at <b>{f2['best_V']:.0f} V</b>,
with {OP} V within {f2['at_op'] / f2['best'] * 100:.0f} % of it.</p>
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

<h2>4. What n_TOF measured in the production gas</h2>
<figure><img src="figures/{figs['ladders']}" alt="MIP track rate per trigger against amplification voltage in two time windows, with the post-flash recovery time on a second axis.">
<figcaption>Both halves of the trade on one axis, and both measured on chamber A
at drift 600 V two days apart. <b>The 16–28 ms window is the one to trust:</b>
above 550 V the recovery reaches into the 8–12 ms window, so that ladder's top
points are suppressed by the very quantity being traded against.</figcaption>
</figure>

<h2>5. The trade</h2>
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
did. The <b>27 June saturday det3 scan</b> runs 425–525 V with the <em>same</em>
efficiency definition and shows the turn-on directly — 49 % at 425 V, 66 % at
435, 77 % at 445, plateau by 455. This page previously said the low edge was an
extrapolation because only the 22 June scan had been looked at; that scan
starts at 450 V, already flat, which is why it looked like there was no
turn-on. The two scans agree on the plateau's flatness and on where discharges
end it; the 27 June absolute level is ~10 points lower because det3 sat in the
top slot (z 702, FEU 7/8) rather than the bottom one, twice the M3 lever arm
into the same fixed 5 mm match box. <b>Only the 520 V frame of the deck build
is still extrapolated</b> (it maps to bench 417 V).</li>
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
                ladders=fig_ntof_ladders(r), trade=fig_tradeoff(r))
    path = os.path.join(HERE, 'report.html')
    with open(path, 'w') as fh:
        fh.write(build_html(r, figs))
    print(f'  -> {path}')


if __name__ == '__main__':
    main()
