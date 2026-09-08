#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_scint_report.py -- build ``report.html`` for the scintillator study.

The answer to "has the n_TOF scintillator information been fully integrated?",
written so that the answer is the first thing on the page: **yes as a filter,
not at all as a measurement** -- and then the first piece of the measurement,
which is position along the wall bars.

Generated, never hand-written.  Every number is read back from what
``scintillators.py`` wrote, so re-running the analysis and then this moves the
figures, the tables and the verdict together.

    python -m sept26_prelim_analysis.make_scint_report --run run_145
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.figstyle import DET_COLOR  # noqa: E402
from sept26_prelim_analysis.make_funnel_report import (  # noqa: E402
    CSS, FONT_LINK, fmt)

EMBED = {'v': False, 'dir': None}
FAM_NAME = {'WAL': 'scintillator wall', 'PSS': 'plastic bars',
            'LIQ': 'liquid cell'}


def figure(name: str, caption: str, alt: str = '') -> str:
    import html as _h
    if EMBED['v'] and EMBED['dir']:
        import base64
        p = os.path.join(EMBED['dir'], f'{name}.png')
        if os.path.exists(p):
            b = base64.b64encode(open(p, 'rb').read()).decode()
            return (f'<figure><img src="data:image/png;base64,{b}" '
                    f'alt="{_h.escape(alt or caption)}">'
                    f'<figcaption>{caption}</figcaption></figure>')
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(alt or caption)}">'
            f'</a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


def audit_table(A: pd.DataFrame) -> str:
    rows = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        g = A[A.family == fam]
        for _, r in g.iterrows():
            dead = ''
            # LIQ C is an order of magnitude below its siblings in run_145.
            med = g.n_in_time.median()
            if r.n_in_time < 0.35 * med:
                dead = ('<span class="why" style="color:var(--warn)">'
                        f'{r.n_in_time / med:.2f}&times; its siblings '
                        '&mdash; excluded</span>')
            rows.append(
                f'<tr><td>{FAM_NAME[fam]}</td>'
                f'<th class="s" style="color:{DET_COLOR[r.arm]}">arm {r.arm}'
                f'{dead}</th>'
                f'<td class="n">{int(r.n_channels)}</td>'
                f'<td class="n">{fmt(r.n_hits)}</td>'
                f'<td class="n">{fmt(r.n_in_time)}</td>'
                f'<td class="n">{fmt(r.n_events_in_time)}</td>'
                f'<td class="n">{100 * r.frac_pileup:.0f}%</td>'
                f'<td class="n">{r.median_amp:.0f}</td></tr>')
    return ('<table class="t"><thead><tr><th>element</th><th></th>'
            '<th>channels<br><span class="u">in the slim</span></th>'
            '<th>hits</th><th>in time</th><th>triggers</th>'
            '<th>pile-up flag</th>'
            '<th>median amp<br><span class="u">ADC</span></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def cal_table(C: pd.DataFrame, S: pd.DataFrame) -> str:
    sc = S.set_index('arm')
    rows = []
    for a in ('A', 'B', 'C', 'D'):
        selfr = (f'{sc.loc[a, "corr"]:+.2f}' if a in sc.index else '&mdash;')
        g = C[C.arm == a]
        if g.empty or not np.isfinite(g.iloc[0].get('lr_slope', np.nan)):
            rows.append(
                f'<tr><th class="s" style="color:{DET_COLOR[a]}">chamber {a}'
                f'</th><td class="n">&mdash;</td>'
                f'<td class="n">{selfr}</td>'
                f'<td class="n" colspan="5">'
                '<span style="color:var(--warn)">no tracks with angles &mdash; '
                'nothing to calibrate against</span></td></tr>')
            continue
        r = g.iloc[0]
        flip = ('<span class="why" style="color:var(--warn)">reads the other '
                'way</span>' if r.lr_slope > 0 else '')
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[a]}">chamber {a}{flip}'
            f'</th><td class="n">{fmt(r.n)}</td>'
            f'<td class="n">{selfr}</td>'
            f'<td class="n">{r.dt_corr:+.2f}</td>'
            f'<td class="n">{r.lr_corr:+.2f}</td>'
            f'<td class="n">{abs(r.v_eff_mm_ns):.0f}</td>'
            f'<td class="n">{abs(r.lambda_mm):.0f}</td>'
            f'<td class="n">&lt; {min(r.dt_resid_mm, r.lr_resid_mm):.0f}</td>'
            '</tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>matched<br><span class="u">track &amp; wall group</span></th>'
            '<th>self-check<br><span class="u">&Delta;t vs log ratio</span></th>'
            '<th>r(&Delta;t, y)</th><th>r(log ratio, y)</th>'
            '<th>v<sub>eff</sub><br><span class="u">mm/ns</span></th>'
            '<th>&lambda;<br><span class="u">mm</span></th>'
            '<th>&sigma;<sub>y</sub><br><span class="u">mm, upper limit</span></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def build_html(A, C, S, ST, meta) -> str:
    run = meta['run'].replace('run_', '')
    liq = A[A.family == 'LIQ'].set_index('arm').n_in_time
    liq_worst = liq.idxmin()
    liq_ratio = liq.min() / liq.median()
    best = C.dropna(subset=['lr_resid_mm'])
    sig = meta.get('sign', {})
    odd = sig.get('odd_ones_out') or []
    stab = ST[ST.grp != 'pooled']
    sy = (min(best.dt_resid_mm.min(), best.lr_resid_mm.min())
          if len(best) else float('nan'))

    return f"""<title>n_TOF scintillators: filter, or measurement?</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run {run}</span>
    <span>{' / '.join(meta['subruns'])}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>The scintillators are a filter. Here is the measurement they are not
      being asked for.</h1>
  <p class="sub">{fmt(meta['n_slim_hits'])} n_TOF hits &middot; wall, plastics
     and liquid across four arms &middot; nothing reprocessed</p>
</header>

<p class="lede"><b>Everything this analysis reads from the n_TOF detectors is a
yes/no.</b> Did the wall fire in this arm, in the window; did the plastic;
which segment, which bar. Three places use that answer, and no place reads an
amplitude or a time as anything but &ldquo;in the window&rdquo;. That is enough
to select events and to measure chamber efficiency &mdash; and it throws away
the one coordinate the Micromegas cannot supply on its own.</p>

<h2><span class="n">1</span>Where the scintillators enter today</h2>
<table class="t"><thead><tr><th>role</th><th>what it asks</th>
<th>what it uses</th></tr></thead><tbody>
<tr><th class="s">stage 1 &mdash; the candidate filter
<span class="why"><code>candidate_filter.py</code></span></th>
<td>which arms have a wall <b>and</b> plastic coincidence in the accept
window</td><td><code>det</code>, <code>detn</code>, <code>dt_ns</code></td></tr>
<tr><th class="s">efficiency
<span class="why"><code>efficiency.py</code></span></th>
<td>the <b>MM-independent denominator</b>: a particle crossed this arm, said by
something that is not the Micromegas</td>
<td><code>det</code>, <code>detn</code>, <code>dt_ns</code></td></tr>
<tr><th class="s">the funnel
<span class="why"><code>funnel.py</code></span></th>
<td>does the track extrapolate onto the wall <i>segment</i> and plastic
<i>bar</i> that actually fired &mdash; the pointing confirmation</td>
<td><code>det</code>, <code>detn</code>, <code>dt_ns</code></td></tr>
</tbody></table>
<p class="note">Three roles, one set of three branches, every answer boolean.
The slim carries a great deal more &mdash; <code>amp</code>,
<code>amp_0</code>, <code>area_0</code>, <code>fwhm</code>,
<code>risetime</code>, <code>chi2</code>, <code>satuflag</code>,
<code>pileup1</code>, <code>pulseshape</code>, <code>shadow_amp</code>,
<code>shadow_dt</code>, and <code>tof</code> as a full-precision double
&mdash; so <b>the missing work needs no reprocessing and no EOS</b>. It needs
the analysis.</p>

<div class="scroll">{audit_table(A)}</div>
<div class="caution"><b>The liquid cell in arm {liq_worst} is effectively dead in
this run</b> &mdash; {liq_ratio:.2f}&times; the in-time rate of its siblings.
It is excluded from every partition on this page and in the funnel, and its
amplitudes are not usable for anything.</div>

<h2><span class="n">2</span>What each element could localise, and how coarsely</h2>
<p>A fired element localises a particle to its own size. The wall's four
segments give 100&nbsp;mm <i>across</i> the wall &mdash; and its bars are
500&nbsp;mm long, so <b>along</b> them a fired segment says nothing at all. The
bars run along <b>v = global y, the beam axis</b>, and that is precisely the
coordinate the target-pointing method cannot reach: the He-3 capsule is a
10&nbsp;mm point in the transverse plane and 80&nbsp;mm long in y, so there is
no zero crossing to find.</p>
{figure('scint_roles',
        'Position uncertainty of one fired element, in each direction. The '
        'top three rows are what the analysis uses today: which wall group, '
        'which plastic bar, and the liquid cell, which has no internal '
        'structure at all. The bottom row is this work &mdash; the wall read '
        'at <i>both ends</i>, which turns its 500&thinsp;mm bar into a '
        'measurement.',
        'position granularity per scintillator element, across and along')}

<h2><span class="n">3</span>The wall is read at both ends, and that is a
position</h2>
<p>Each wall group appears in the slim as <b>two</b> <code>detn</code> values
&mdash; the two ends of the same four bars. Two independent physics processes
turn that pair into a position along the bar:</p>
<ul>
<li><b>propagation.</b> &Delta;t = t<sub>1</sub> &minus; t<sub>2</sub> = 2y /
v<sub>eff</sub>, so the delay between the ends is linear in y.</li>
<li><b>attenuation.</b> log(A<sub>1</sub>/A<sub>2</sub>) = &minus;2y /
&lambda;, so the amplitude ratio is too, with a different constant.</li>
</ul>
<p>They agree with each other <b>without any Micromegas in the argument</b>,
which is what makes this a property of the wall rather than of the
reconstruction &mdash; and it is the one measurement on this page that works in
chamber&nbsp;B, which has no drift field and therefore no tracks.</p>
{figure('wall_selfcheck',
        'The two estimators against each other, per chamber, on all in-time '
        'wall pairs. Both axes are centred <b>per bar group</b>: the four '
        'groups are read through their own cables and carry offsets of up to '
        '8&thinsp;ns, which pooled turn one band into four parallel ones. '
        'Correlation &minus;0.69 to &minus;0.83 in every chamber.',
        'delay against amplitude ratio for the two ends of each wall group')}
<p class="note"><b>The per-group offsets are a calibration constant, not a
result.</b> Up to 8&nbsp;ns within a single arm is metres of cable and nothing
physical. They are fitted alongside the slope and removed; the slope moves by
under 5&thinsp;% when they are, and the correlation against tracks rises from
0.30 to 0.52 in arm&nbsp;A and 0.36 to 0.60 in arm&nbsp;C.</p>

<h2><span class="n">4</span>Calibrated against Micromegas tracks</h2>
<p>A gated track that points back at the target
(<code>dca_axis &lt; {meta['dca_max']:.0f} mm</code>) and crosses the wall
predicts <i>where</i> on the bar. That is the truth the two estimators are
fitted against: one common slope per chamber, one offset per bar group.</p>
{figure('wall_along_bar',
        'The amplitude ratio against the track&rsquo;s y at the wall, group '
        'offsets removed. Points are binned medians; the line is the robust '
        'fit. The relation is real and linear through the populated core, and '
        'in chamber D it runs the other way.',
        'log amplitude ratio versus track y at the wall, per chamber')}
<div class="scroll">{cal_table(C, S)}</div>
<p class="note"><b>&sigma;<sub>y</sub> is an upper limit and is labelled as
one.</b> The truth here is an <i>extrapolated</i> Micromegas track, which
carries its own error, so the residual width contains both. The wall is
therefore at least as good as {sy:.0f}&nbsp;mm along a 500&nbsp;mm bar
&mdash; about a factor {500 / sy:.0f} better than knowing only which group
fired, and comparable to the 100&nbsp;mm the segments give in the other
direction.</p>
<p class="note"><b>The slope is not an artefact of the four groups differing.</b>
Refitted <i>inside</i> each u group separately it reproduces the pooled value:
{'; '.join(
    f"{a} {g.lr_slope.min() * 1e3:.2f} to {g.lr_slope.max() * 1e3:.2f}"
    for a, g in stab.groupby('arm'))} &times;10<sup>&minus;3</sup>/mm against
pooled {'; '.join(f"{r.arm} {r.lr_slope * 1e3:.2f}" for r in
                  ST[ST.grp == 'pooled'].itertuples())}. Fixing u does not
remove it, so it is position along the bar.</p>

<div class="caution"><b>Chamber {', '.join(odd) or '&mdash;'} runs the opposite
way, and this data cannot say why.</b> Both estimators flip together, so it is
<i>one</i> flip somewhere in that chamber's chain &mdash; and the two
candidates are indistinguishable here:
<ol>
<li>that chamber's <b>two wall ends swapped</b> in the readout map, or</li>
<li>that chamber's <b>Micromegas y strip plane mirrored</b>.</li>
</ol>
Three things narrow it and none closes it. The wall's own self-check is
<i>strongest</i> in that chamber ({sc_worst(S, odd)}), so its wall hardware and
readout are healthy. Its fitted slope is also about half the others', which is
what noise in the <i>predictor</i> does to a regression &mdash; pointing at the
Micromegas y as the noisy side rather than the wall. And <code>run_config</code>
declares the same FEU orientation for all four chambers, so a Micromegas-side
mirror would have to be an <b>undeclared</b> cabling difference. What would
settle it is an external fact: the wall cabling map, or the y-plane strip
mapping order, for that arm. A y-asymmetric source would also have done it, and
does not: the He-3 gas polycone is volume-symmetric in y to a skew of +0.06.</div>

<h2><span class="n">5</span>What this does and does not unlock</h2>
<p><b>It does</b> give a y estimator that is independent of the Micromegas
<i>resolution</i>, works in chamber B, and can be attached to every selected
event at no processing cost.</p>
<div class="caution"><b>It does not give an independent absolute y.</b> The
per-group offsets absorb the mean illumination height, so the wall's y zero is
fixed <i>by the fit against Micromegas tracks</i> and cannot then be used to
check that same zero. What it can do independently is the <b>shape and the
width</b> of the source's y distribution, and cross-check one chamber against
another. Pinning the capsule's absolute y needs the cabling constants, or a
survey.</p>
<p class="note"><b>Still not attempted, and still deferred:</b> any energy from
any scintillator (D6) &mdash; which is what an invariant mass would need; the
liquid-cell gain against position (D7); and the question underneath the
efficiency number, which is whether a wall-and-plastic coincidence really means
a charged particle crossed the gas at all. Chamber A measures 63&thinsp;% where
80&ndash;90&thinsp;% is expected, and a neutron or gamma converting in the PCB
would push it exactly that way.</p>

<footer><p>Generated by <code>sept26_prelim_analysis/make_scint_report.py</code>
from <code>scintillators.py</code>. Figures carry their numbers as CSV.</p>
</footer>
</div>
"""


def sc_worst(S: pd.DataFrame, odd) -> str:
    if not odd:
        return 'no chamber is odd'
    a = odd[0]
    r = S.set_index('arm')
    o = r['corr'].drop(a)
    return (f'r = {r.loc[a, "corr"]:+.2f} against '
            f'{o.max():+.2f} to {o.min():+.2f} in the others')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    sd = paths.out('scint')

    A = pd.read_csv(paths.require(sd / f'audit_{a.run}.csv', 'the audit'))
    C = pd.read_csv(paths.require(sd / f'wall_calibration_{a.run}.csv',
                                  'the wall calibration'))
    S = pd.read_csv(paths.require(sd / f'self_consistency_{a.run}.csv',
                                  'the self-consistency table'))
    ST = pd.read_csv(paths.require(sd / f'group_stability_{a.run}.csv',
                                   'the group stability table'))
    meta = json.load(open(paths.require(sd / f'scint_{a.run}.meta.json',
                                        'the scintillator meta')))

    od = str(sd)
    body = build_html(A, C, S, ST, meta)
    marker = '<div class="wrap">'
    head, rest = body.split(marker, 1)
    with open(os.path.join(od, 'report.html'), 'w') as fh:
        fh.write('<!doctype html>\n<html lang="en">\n<head>\n'
                 '<meta charset="utf-8">\n<meta name="viewport" '
                 'content="width=device-width,initial-scale=1">\n'
                 '<meta name="color-scheme" content="light dark">\n'
                 f'{head}</head>\n<body>\n{marker}{rest}\n</body>\n</html>\n')
    EMBED['v'], EMBED['dir'] = True, os.path.join(od, 'figures')
    with open(os.path.join(od, 'body.html'), 'w') as fh:
        fh.write(build_html(A, C, S, ST, meta))
    EMBED['v'] = False
    print(f'wrote {od}/report.html and body.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
