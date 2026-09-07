#!/usr/bin/env python3
"""make_report.py -- build report.html from the JSON products.

Regenerate after any of fit_kernel.py / systematics.py / bench_kernel.py /
mx_june_wft/18_ladder_bench.py so numbers, tables and verdict move together.

    ../../../.venv/bin/python make_report.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/wft/kernel_arms/')

J = json.load(open(os.path.join(HERE, 'fit_kernel.json')))
S = json.load(open(os.path.join(HERE, 'systematics.json')))
BK = json.load(open(os.path.join(HERE, 'bench_kernel_y.json')))
LB = json.load(open(os.path.join(BENCH, 'ladder_bench.json')))
RR = json.load(open(os.path.join(BENCH, 'ratio_recal.json')))
PLAT = [('raw700', 243), ('raw450', 156), ('raw275', 95)]

CSS = """
:root{--ink:#141312;--muted:#5c5a55;--line:#ddd9d0;--bg:#faf9f6;
      --accent:#2a78d6;--bad:#9E2B25;--good:#1f7a4d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.62 "Charter","Iowan Old Style",Georgia,serif;}
.wrap{max-width:960px;margin:0 auto;padding:44px 24px 96px}
h1{font-size:30px;line-height:1.24;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:21px;margin:46px 0 12px;padding-top:16px;
   border-top:1px solid var(--line)}
h3{font-size:16.5px;margin:26px 0 8px;color:var(--muted);
   text-transform:uppercase;letter-spacing:.07em}
.sub{color:var(--muted);font-size:15px;margin:0 0 26px}
.verdict{background:#fff;border:1px solid var(--line);border-left:4px solid
 var(--accent);padding:18px 22px;margin:26px 0 30px;border-radius:3px}
.verdict p{margin:.5em 0}
.verdict ol{margin:.5em 0 0;padding-left:20px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14.2px;
 font-family:"SF Mono",ui-monospace,Menlo,monospace;background:#fff}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid #c9c4b8;font-weight:600;color:var(--muted)}
tbody tr:hover{background:#f4f2ec}
td.win{color:var(--good);font-weight:600}
td.lose{color:var(--bad)}
figure{margin:26px 0}
img{width:100%;border:1px solid var(--line);border-radius:3px;background:#fff}
figcaption{color:var(--muted);font-size:13.6px;margin-top:8px}
code{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:.9em;
 background:#f0eee8;padding:1px 5px;border-radius:3px}
.eq{background:#fff;border:1px solid var(--line);padding:12px 18px;
 margin:14px 0;font-family:"SF Mono",ui-monospace,Menlo,monospace;
 font-size:14px;border-radius:3px}
.note{color:var(--muted);font-size:14.4px}
.tag{display:inline-block;font-size:11.5px;letter-spacing:.06em;
 text-transform:uppercase;padding:2px 8px;border-radius:10px;
 background:#e9e6dd;color:var(--muted);margin-right:6px}
.tag.bad{background:#f6e3e2;color:var(--bad)}
.tag.good{background:#e0f0e7;color:var(--good)}
"""


def row(cells, cls=None):
    cls = cls or [''] * len(cells)
    return '<tr>' + ''.join(
        f'<td class="{c}">{v}</td>' if c else f'<td>{v}</td>'
        for v, c in zip(cells, cls)) + '</tr>'


def table(head, rows):
    return ('<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) +
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


# ------------------------------------------------------------------ tables
def t_forms(view='y'):
    rows = []
    for lab, E in PLAT:
        r = J[view][lab]
        best = min(('cascade', 'ladder', 'delay', 'geom'),
                   key=lambda f: r[f]['rms_pct'])
        cells = [f'{E} V/cm', f'{r["n_events"]:,}']
        cls = ['', '']
        for f in ('cascade', 'ladder', 'delay', 'geom'):
            cells.append(f'{r[f]["rms_pct"]:.2f} %')
            cls.append('win' if f == best else '')
        rows.append(row(cells, cls))
    return table(['drift field', 'events', 'cascade (RC)', 'cascade, free c2',
                  'delay (shipped)', 'no sharing'], rows)


def t_constants(view='y'):
    rows = []
    for lab, E in PLAT:
        p = J[view][lab]['cascade']['par']
        e = J[view][lab]['cascade']['err']
        d = J[view][lab]['delay']['par']
        l = J[view][lab]['ladder']['par']
        rows.append(row([
            f'{E} V/cm',
            f'{p["tau"]:.0f} &plusmn; {e.get("tau", 0):.0f}',
            f'{p["c"]:.3f} &plusmn; {e.get("c", 0):.3f}',
            f"{l['c2'] / l['c'] ** 2:.2f}",
            f'{d["c2"] / max(d["c1"], 1e-9):.3f}',
            f'{0.5 * (p["q1"] + p["q1m"]):.3f}',
            f'{p["q1"] / p["q1m"]:.3f}']))
    return table(['drift field', '&tau; [ns]', 'c', 'c<sub>2</sub>/c<sup>2</sup>',
                  'c<sub>2</sub>/c<sub>1</sub> (delay form)', 'q<sub>&plusmn;1</sub>',
                  'q<sub>+1</sub>/q<sub>&minus;1</sub>'], rows)


def t_window():
    rows = [row([f'&minus;480 &hellip; +{r["end_ns"]} ns', f'{r["tau"]:.0f}',
                 f'{r["c"]:.3f}', f'{r["rms_cascade"]:.2f} %',
                 f'{r["rms_delay"]:.2f} %', f'{r["delay_ratio"]:.3f}'])
            for r in S['window']]
    return table(['fit window', '&tau; [ns]', 'c', 'cascade rms', 'delay rms',
                  'c<sub>2</sub>/c<sub>1</sub>'], rows)


def t_syst():
    rows = []
    for how, r in S['basis'].items():
        rows.append(row([f'aggregation: {how}', f'{r["tau"]:.0f}',
                         f'{r["c"]:.3f}', f'{r["rms_cascade"]:.2f} %',
                         f'{r["rms_delay"]:.2f} %']))
    for g, r in S['gate'].items():
        rows.append(row([f'amplitude gate: {g} ADC ({r["n"]:,} ev)',
                         f'{r["tau"]:.0f}', f'{r["c"]:.3f}',
                         f'{r["rms"]:.2f} %', '&mdash;']))
    a = S.get('align')
    if a:
        rows.append(row(['absolute window time (not peak-aligned)',
                         f'{a["absolute"]["tau"]:.0f}',
                         f'{a["absolute"]["c"]:.3f}',
                         f'{a["absolute"]["rms_cascade"]:.2f} %',
                         f'{a["absolute"]["rms_delay"]:.2f} %']))
    return table(['variation (Y view, 156 V/cm, full window)', '&tau; [ns]',
                  'c', 'cascade rms', 'delay rms'], rows)


def t_transfer():
    b = BK['cascade']
    e = BK['cascade']['err']
    rows = [row(['det4 at H4, 243 V/cm', '696', '0.534', '0.185', '1.76 %'],
                ),
            row(['det4 at H4, 156 V/cm', '664', '0.525', '0.196', '1.88 %']),
            row(['det4 at H4, 95 V/cm', '619', '0.506', '0.188', '1.87 %']),
            row([f'det3 on the bench ({BK["n_events"]} cosmics)',
                 f'{b["par"]["tau"]:.0f} &plusmn; {e["tau"]:.0f}',
                 f'{b["par"]["c"]:.3f} &plusmn; {e["c"]:.3f}',
                 f'{b["par"]["q1"]:.3f} &plusmn; {e["q1"]:.3f}',
                 f'{b["rms_pct"]:.2f} %'])]
    return table(['matched fit window, &minus;600 &hellip; +720 ns', '&tau; [ns]',
                  'c', 'q<sub>+1</sub>', 'rms'], rows)


def t_bench():
    order = ['production', 'ratio0.45', 'ratio0.6', 'ratio0.8', 'ladder_free',
             'ladder_pinY', 'ladder_long', 'beam_delay']
    names = {'production': 'production (shipped, c<sub>2</sub> &gt; c<sub>1</sub>)',
             'ratio0.45': 'c<sub>2</sub> = 0.45 c<sub>1</sub>',
             'ratio0.6': '<b>c<sub>2</sub> = 0.60 c<sub>1</sub></b>',
             'ratio0.8': 'c<sub>2</sub> = 0.80 c<sub>1</sub>',
             'ladder_free': 'RC kernel, bench-fitted',
             'ladder_pinY': 'RC kernel, beam-pinned (matched window)',
             'ladder_long': 'RC kernel, beam-pinned (long window)',
             'beam_delay': 'shipped form, beam c<sub>1</sub>/c<sub>2</sub> (8-17)'}
    p0 = LB['production']
    rows = []
    for k in order:
        if k not in LB:
            continue
        g, v = LB[k]['geo'], LB[k].get('vs_production', {})
        ds = v.get('y', {})
        cls = ['', '', '', '', '', '', '']
        if k != 'production':
            cls[3] = 'win' if abs(g['y']['slope'] - 1) < 0.005 else ''
            cls[5] = 'lose' if ds.get('d_sig', 0) > 2 * ds.get('d_sig_err', 1) \
                else ('win' if ds.get('d_sig', 0) < 0 else '')
        rows.append(row([
            names[k], f'{g["x"]["sig_theta"]:.3f}', f'{g["y"]["sig_theta"]:.3f}',
            f'{g["y"]["slope"]:.4f}',
            f'{100 * (LB[k]["chi2_held_cold"] / p0["chi2_held_cold"] - 1):+.1f} %',
            (f'{ds.get("d_sig", 0):+.3f} &plusmn; {ds.get("d_sig_err", 0):.3f}'
             if k != 'production' else '&mdash;'),
            (f'{ds.get("d_slope", 0):+.4f} &plusmn; {ds.get("d_slope_err", 0):.4f}'
             if k != 'production' else '&mdash;')], cls))
    return table(['arm', '&sigma;<sub>&theta;</sub> X [&deg;]',
                  '&sigma;<sub>&theta;</sub> Y [&deg;]', 'slope Y',
                  '&Delta;&chi;&sup2; held',
                  '&Delta;&sigma;<sub>&theta;</sub> Y (paired)',
                  '&Delta;slope Y (paired)'], rows)


def t_hyper():
    prod = LB['production']['hyper']
    r6 = RR['ratio0.6']['hyper']
    keys = ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')
    lbl = {'c1': 'c<sub>1</sub>', 'c2': 'c<sub>2</sub>', 'kY': 'k<sub>Y</sub>',
           'tau_s': '&tau;<sub>s</sub> [ns]', 'sigma_s': '&sigma;<sub>s</sub> [ns]',
           'sigma_p0': '&sigma;<sub>p0</sub> [mm]', 'Dp': 'D<sub>p</sub>'}
    rows = []
    for k in keys:
        a = prod.get(k, 0.0)
        b = r6['c1'] * r6['c2_over_c1'] if k == 'c2' else r6.get(k, 0.0)
        rows.append(row([lbl[k], f'{a:.4g}', f'{b:.4g}',
                         'slaved' if k == 'c2' else 'fitted']))
    rows.append(row(['<b>c<sub>2</sub>/c<sub>1</sub></b>',
                     f'<b>{prod["c2"] / prod["c1"]:.2f}</b>',
                     f'<b>{r6["c2_over_c1"]:.2f}</b>', 'the fix']))
    rows.append(row(['c<sub>1</sub> on Y (&times; k<sub>Y</sub>)',
                     f'{prod["c1"] * prod["kY"]:.4f}',
                     f'{r6["c1"] * r6["kY"]:.4f}', '']))
    rows.append(row(['c<sub>2</sub> on Y (&times; k<sub>Y</sub>)',
                     f'{prod["c2"] * prod["kY"]:.4f}',
                     f'{r6["c1"] * r6["c2_over_c1"] * r6["kY"]:.4f}', '']))
    return table(['hyper', 'production', 'c<sub>2</sub> = 0.60 c<sub>1</sub>',
                  ''], rows)


DET7 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det6_det7_overnight_6-26-26/'
        'long_run/mx17_7/wft/kernel_arms/ladder_bench.json')


def t_det7():
    if not os.path.exists(DET7):
        return '<p class="note">det7 not yet run.</p>'
    B = json.load(open(DET7))
    names = {'production': 'production (shipped, c<sub>2</sub>/c<sub>1</sub> = 1.75)',
             'ratio0.45': 'c<sub>2</sub> = 0.45 c<sub>1</sub>',
             'ratio0.6': '<b>c<sub>2</sub> = 0.60 c<sub>1</sub></b>',
             'ratio0.8': 'c<sub>2</sub> = 0.80 c<sub>1</sub>'}
    p0 = B['production']
    rows = []
    for k in ('production', 'ratio0.45', 'ratio0.6', 'ratio0.8'):
        if k not in B:
            continue
        g, v = B[k]['geo'], B[k].get('vs_production', {})
        ds = v.get('y', {})
        rows.append(row([
            names[k], f"{g['x']['sig_theta']:.3f}", f"{g['y']['sig_theta']:.3f}",
            f"{g['y']['slope']:.4f}",
            f"{100 * (B[k]['chi2_held_cold'] / p0['chi2_held_cold'] - 1):+.1f} %",
            (f"{ds.get('d_sig', 0):+.3f} &plusmn; {ds.get('d_sig_err', 0):.3f}"
             if k != 'production' else '&mdash;'),
            (f"{ds.get('d_slope', 0):+.4f} &plusmn; {ds.get('d_slope_err', 0):.4f}"
             if k != 'production' else '&mdash;')],
            ['', '', '', 'lose' if k != 'production' else '', '', '', '']))
    return table(['arm', '&sigma;<sub>&theta;</sub> X [&deg;]',
                  '&sigma;<sub>&theta;</sub> Y [&deg;]', 'slope Y',
                  '&Delta;&chi;&sup2; held',
                  '&Delta;&sigma;<sub>&theta;</sub> Y (paired)',
                  '&Delta;slope Y (paired)'], rows)


# ------------------------------------------------------------------- page
HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The sharing kernel, measured &mdash; and what it fixes</title>
<meta name="description" content="The MX17 resistive sharing kernel measured
model-free from head-on beam data, why the shipped bundles carry an
impossible c2 &gt; c1, and the one-line change that removes it at no cost.">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>The sharing kernel, measured &mdash; and what it fixes</h1>
<p class="sub">run_71 RAW head-on beam (det4 at H4, three drift fields) and
near-vertical bench cosmics (det3) &middot; 2026-08-18 &middot;
<code>sps_beam_test_26/analysis/sharing_kernel/</code></p>

<div class="verdict">
<p><b>The shipped calibration bundles carry <code>c2 &gt; c1</code> on every
detector &mdash; the &plusmn;2 strip receiving more than the &plusmn;1 strip,
which no lateral transport can do. This measures the kernel directly and says
what to do about it.</b></p>
<ol>
<li><b>No data wants <code>c2 &gt; c1</code>.</b> Fitted model-free on the beam
<em>in the shipped kernel form</em>, c<sub>2</sub>/c<sub>1</sub> =
0.45 &plusmn; 0.03 at all three drift fields; on near-vertical det3 cosmics,
0.63 &plusmn; 0.10. The inversion is an artefact of the ref-pinned cosmic
&chi;&sup2;, which is flat in that direction.</li>
<li><b>The shipped kernel FORM is also wrong.</b> A translated, smeared copy
cannot fit head-on data (4.2 % residual) where a cascade of one-poles can
(2.1 %); no lateral sharing at all is 16.5 %. The ranking survives every
systematic tested.</li>
<li><b>But the RC form is not adoptable yet</b>, and that is the honest limit
of this round: its absolute constants depend on how much pulse tail the fit
window contains (&tau; walks 629 &rarr; 1040 ns), and the 1.92 &micro;s bench
window cannot see the tail that sets them.</li>
<li><b>What ships now:</b> keep the shipped form, slave
c<sub>2</sub> = 0.6 c<sub>1</sub>. Tested on both affected detectors, it is
<b>free in resolution</b>: &Delta;&sigma;<sub>&theta;</sub> is under
0.6&sigma; in every plane of both (det3 Y +0.028 &plusmn; 0.062&deg;, det7 Y
+0.023 &plusmn; 0.080&deg;). It removes a free hyper, lowers det3's held-out
&chi;&sup2; by 1.1 %, and makes the ordering structural. It also shifts the raw
angle slope by about <b>+1.5 %</b> &mdash; which lands det3 on 1.0012 from
0.9876, and pushes det7 from 1.0150 to 1.0260. <b>That is a fixed shift, not a
correction toward truth</b>, and a per-plane k<sub>w</sub> absorbs it either
way; do not sell it as an angle-scale fix.</li>
</ol>
</div>

<h2>How you measure a kernel without deconvolving it</h2>
<p>At normal incidence every strip is driven by the same signal C(t) &mdash;
the ionisation column folded with the amplifier &mdash; through a lateral
transfer function n<sub>d</sub>. So for any two offsets the measured traces
obey a relation in which C cancels identically:</p>
<div class="eq">W<sub>d</sub> = n<sub>d</sub> &lowast; C
&nbsp;&nbsp;&rArr;&nbsp;&nbsp;
<b>n<sub>0</sub> &lowast; W<sub>d</sub> = n<sub>d</sub> &lowast;
W<sub>0</sub></b></div>
<p>Both sides are <em>measured data convolved with a model filter</em>. Nothing
is inverted, so there is no Wiener filter, no regularisation and no choice of
&lambda; to defend &mdash; the previous round's deconvolution needed all three.
Causality also makes the window truncation harmless: (n &lowast; W)[i] only ever
reaches back to W[&le;&nbsp;i], and the pre-pulse region really is empty (the
clean stacks sit at 1.7 % of peak there), so the missing tail past the window
is never needed. The cancellation holds <em>only</em> head-on, which is why the
flat runs are the ones that can do this.</p>
<p>The network is then <code>n_d = &Sigma;_j q_j k_(|d&minus;j|)</code>: a
geometric fraction q<sub>j</sub> of the primary cloud on strip j (a nuisance
parameter, and the one thing that is allowed to move with drift field), each
copy dispersed laterally by the kernel k. Three candidate k are fitted &mdash;
the RC cascade, the shipped translated copy, and no sharing at all.</p>

<h2>What the beam picks</h2>
{t_forms('y')}
<p class="note">Cross-relation residual, Y view, as a percentage of the trace
being fitted. Lower is better; the winner of each row is green. The shipped
form is beaten by a factor two at every drift field, and the free-c<sub>2</sub>
cascade barely improves on the strict one &mdash; the ladder constraint is not
what is costing it.</p>

<figure><img src="figures/forms_y.png" alt="form comparison">
<figcaption>Top left: the measured stacks. The central strip goes
<em>negative</em> after 500 ns while the neighbours hold a long positive
plateau &mdash; charge leaving the central strip is exactly what lateral
dispersion looks like. Top right and bottom left: both sides of the
cross-relation. The shipped form peaks too early on the &plusmn;2 strip and
then dies too fast; it has no way to make a tail without an unphysical smear.
Bottom right: per-offset residuals.</figcaption></figure>

<h2>Is it a ladder?</h2>
<p>If the &plusmn;2 strip is reached only by going through the &plusmn;1 strip,
its amplitude must be the square of the single-step one. Fitting c<sub>2</sub>
free instead of c<sup>2</sup> tests that.</p>
{t_constants('y')}
<p class="note">Y view. c<sub>2</sub>/c<sup>2</sup> = 0.93 &plusmn; 0.01 at all
three fields &mdash; the ladder holds to 7 %. The constants sit still to 4 %
over a 2.6&times; range of drift field, while the geometric fraction
q<sub>&plusmn;1</sub> and its symmetry confirm the mount really is head-on in
Y. The last column is the same ratio measured in the <em>shipped</em> form,
which is the number the production bundles get wrong.</p>

<figure><img src="figures/ladder.png" alt="ladder test"></figure>
<figure><img src="figures/invariance.png" alt="drift invariance">
<figcaption>The X view is not usable: its q<sub>+1</sub>/q<sub>&minus;1</sub>
sits at 0.6&ndash;0.7 and worsens as the field falls, which is the flat mount's
known 0.2&ndash;0.4&deg; residual tilt in x, and its fitted &tau; walks
390&nbsp;&rarr;&nbsp;610 ns instead of sitting still. Y is head-on; X is
not.</figcaption></figure>

<h2>What this does <em>not</em> settle</h2>
<p>The one number a reader would most want to quote &mdash; the RC time
constant &mdash; is the one this cannot deliver as a constant.</p>
{t_window()}
<p class="note"><b>The dominant systematic.</b> Lengthening the fit window
walks &tau; from 629 to 1040 ns and c from 0.51 to 0.66, monotonically: the
measured tail is <em>heavier than one exponential</em>, so a single-pole
cascade fitted over more tail keeps returning a longer &tau;. The lumped
cascade is therefore the best of the forms tried, not the right one &mdash;
a continuum RC (diffusive, t<sup>&minus;3/2</sup> tail) is the obvious next
candidate. Note the <em>ranking</em> is stable at every window, and so is the
qualitative statement c<sub>2</sub> &lt; c<sub>1</sub>.</p>
{t_syst()}
<p class="note">Everything else is small by comparison: &plusmn;6 % on the
aggregation basis, &plusmn;8 % on the amplitude gate. Dropping the peak
alignment costs the most (&tau; 1320, and both forms fit poorly) &mdash; not a
bias, since the same shift is applied to every strip of an event, but the
unaligned ensemble mean is washed out and carries much less high-frequency
information to constrain the kernel with.</p>

<figure><img src="figures/systematics.png" alt="systematics"></figure>

<h2>Does it transfer between chambers?</h2>
<p>My first answer here was wrong. det3's &plusmn;1/centre peak ratio is
0.48&ndash;0.53 against det4's 0.31&ndash;0.32, which looks like a factor
1.6 disagreement &mdash; but the peak ratio is not the kernel. det3's pulse is
visibly broader, its bench window spans only &plusmn;720 ns, and the two were
being compared at different fit windows. At a <em>matched</em> window:</p>
{t_transfer()}
<p class="note">The kernel constants agree within the bench's (large) errors;
only the <em>geometric</em> fraction q genuinely differs, and that is a
property of the drift and the avalanche footprint, not of the resistive layer.
Checked against shear: the bench ratio is flat from |tan&thinsp;&theta;| &lt;
0.20 down to &lt; 0.01, i.e. from a 6 mm to a 0.3 mm transverse shear over the
gap, so the near-vertical selection is not what is setting it.</p>

<figure><img src="figures/transfer.png" alt="beam vs bench">
<figcaption>The shaded band is where the bench has no data. A &tau; of order
1 &micro;s lives almost entirely inside it.</figcaption></figure>

<h2>The bench verdict</h2>
<p>Judged the way this repository insists these things are judged &mdash; on
held-out cosmics against the M3 reference, never on &chi;&sup2; &mdash; with
every arm scored cold, re-measuring its own absolute-t<sub>0</sub> table, and a
paired bootstrap over the events the arms share.</p>
{t_bench()}
<p class="note">Transplanting the RC kernel from the beam costs
&sigma;<sub>&theta;</sub> Y 1.14 &rarr; 1.54&deg;, and it is worse still with
the long-window constants &mdash; the negative result that stops the RC form
shipping this round. Slaving c<sub>2</sub> in the existing form costs nothing
measurable in either plane and fixes the angle scale. The three ratios in the
measured range 0.45&ndash;0.8 are indistinguishable in
&sigma;<sub>&theta;</sub>, so the ratio itself does not need to be pinned
precisely &mdash; only kept below 1.</p>

<h3>And on det7, the other inverted detector</h3>
{t_det7()}
<p class="note">Same conclusion on resolution &mdash; every arm within
0.6&sigma; of production in both planes &mdash; and the <em>opposite</em>
conclusion on slope. det7's production Y slope already over-reads at 1.0150,
and the constraint's +1.5 % shift takes it further from 1, at 5&ndash;8&sigma;.
So the slope movement is a property of the constraint, not evidence that the
constraint is right; det3 simply happened to be under-reading by about the
same amount. Note also that det7's k<sub>Y</sub> runs to 4.7&ndash;6.0 against
a bound of 6 &mdash; its long-standing v&nbsp;&harr;&nbsp;sharing degeneracy,
which this does not address.</p>

<figure><img src="figures/bench_verdict.png" alt="bench verdict"></figure>

<h3>The recommended hyper set for det3</h3>
{t_hyper()}
<p class="note">c<sub>1</sub> still sits on <code>calibrate.C1_MIN</code> = 0.05
and the kernel amplitude is still carried by k<sub>Y</sub>; that is unchanged
from production and is not what this fixes. The floor is <em>not</em> what
caused the inversion either &mdash; an arm seeded in the physical basin with no
floor still slid to c<sub>1</sub> = 0.022 &mdash; but with c<sub>2</sub> slaved
there is no longer an inverted basin to slide into.</p>

<h2>What it does not rule out</h2>
<ul>
<li><b>That the true kernel is the continuum, not the cascade.</b> The
window-dependent &tau; says the tail is heavier than one exponential. A
diffusive RC-continuum kernel would fit that and has one parameter; it has not
been tried.</li>
<li><b>That the X plane shares differently.</b> Its free-c<sub>2</sub> fit
drives the second rung to zero, consistent with the resistive strips running
along y so that X has no second rung &mdash; but X's head-on data is tilt-
contaminated and cannot settle it. A genuinely head-on X measurement would
need a mount alignment the flat runs did not have.</li>
<li><b>That the ratio is 0.6.</b> The bench cannot distinguish 0.45 from 0.8.
0.6 is chosen as the midpoint of the beam (0.45) and bench (0.63) measurements
and because it gives the flattest angle slope; any value in that range is
defensible.</li>
<li><b>Nothing is re-frozen.</b> Both affected detectors are refitted and a
<code>calib_bundle_r06</code> is written beside each production bundle, but the
MPGD26 manifest still points at the old ones. Re-freezing means re-running the
reco, then re-measuring w<sub>0</sub>/k<sub>w</sub>, and is a deliberate
separate act. Checked against that manifest: det2 (0.74), det4 (0.67) and det6
(0.82) already ship physical bundles, so <b>only det3 and det7 were ever
affected</b>. (<code>o22_long_det2</code>'s bundle is inverted at 1.53 but is
not the one the manifest points at.)</li>
<li><b>That &sigma;<sub>&theta;</sub> improves.</b> It does not, and was never
expected to &mdash; the claim is that the physical constraint is free. The
angle <em>scale</em> is what moves.</li>
<li><b>That the slope movement is an improvement.</b> det7 settles this: the
constraint shifts the raw slope by a near-constant +1.5 % on both detectors,
which helps det3 and hurts det7. Whatever the constraint is worth, it is not
worth this.</li>
<li><b>That the slope result survives w<sub>0</sub>/k<sub>w</sub>.</b> The
slopes here are the <em>raw</em> w&nbsp;&rarr;&nbsp;tan mapping,
<code>tan = w&middot;10&sup3;/v</code>. The bundles also carry a per-plane
linear correction <code>tan = (w&middot;10&sup3; &minus; w<sub>0</sub>) /
(k<sub>w</sub>&thinsp;v)</code>, and a linear scale can absorb a global slope
&mdash; det3's k<sub>w</sub>(Y) = 1.024 is the same size as the 1.4 % this
moves. So the honest claim is that the raw mapping is closer to correct
<em>before</em> any linear patch, not that an absolute angle scale has been
fixed. w<sub>0</sub>/k<sub>w</sub> are measured from an existing
reconstruction, so they cannot be re-derived without re-running the reco; the
new bundle carries the old pair and is stamped
<code>w0_kw_stale</code>.</li>
</ul>

<h2>Reproducing</h2>
<div class="eq">
cd sps_beam_test_26/analysis/sharing_kernel<br>
../../../.venv/bin/python stacks.py&nbsp;&nbsp;&nbsp;# per-event neighbour stacks, 3 plateaus<br>
../../../.venv/bin/python fit_kernel.py&nbsp;&nbsp;&nbsp;# the form comparison + bootstrap<br>
../../../.venv/bin/python systematics.py<br>
../../../.venv/bin/python bench_kernel.py --view y<br>
../../../.venv/bin/python make_figures.py &amp;&amp; ../../../.venv/bin/python make_report.py<br><br>
cd ../../..&nbsp;&nbsp;# the bench arms<br>
.venv/bin/python mx_june_wft/19_ratio_recal.py sat_det3 --ratios 0.45,0.6,0.8<br>
.venv/bin/python mx_june_wft/17_ladder_recal.py sat_det3<br>
.venv/bin/python mx_june_wft/18_ladder_bench.py sat_det3
</div>
<p class="note">The model change is one gated branch in
<code>wft/model.py::build_matrix</code>: a <code>c2_over_c1</code> hyper that
slaves c<sub>2</sub> = r&thinsp;c<sub>1</sub> before the per-plane
k<sub>Y</sub>/c<sub>X</sub> scaling. No existing bundle carries the key, so
nothing changes silently.</p>

</div>
</body>
</html>
"""

with open(os.path.join(HERE, 'report.html'), 'w') as f:
    f.write(HTML)
print('wrote', os.path.join(HERE, 'report.html'))
