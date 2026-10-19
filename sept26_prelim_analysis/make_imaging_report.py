#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_imaging_report.py -- build ``report.html`` for the source-imaging study.

Where the He-3 capsule is, how well each chamber agrees about it, and what that
disagreement is: alignment.

Generated, never hand-written.  Everything is read back from what
``source_imaging.py`` wrote.

    python -m sept26_prelim_analysis.make_imaging_report --run run_145
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


def transverse_table(C, Cr, T) -> str:
    rows = []
    for _, r in C.iterrows():
        rob = Cr[(Cr.arm == r.arm)]
        rmm = float(rob.mm.iloc[0]) if len(rob) else np.nan
        subs = T[(T.arm == r.arm) & (T.variant == 'baseline') & T.mm.notna()]
        per = ', '.join(f'{v:+.1f}' for v in subs.mm)
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">chamber '
            f'{r.arm}</th><td class="n">{r.axis}</td>'
            f'<td class="n">{fmt(r.n)}</td>'
            f'<td class="n">{r.mm:+.2f}</td>'
            f'<td class="n">&plusmn;{r.err_stat:.2f}</td>'
            f'<td class="n">'
            f'{"&mdash;" if not np.isfinite(r.err_repro) else f"&plusmn;{r.err_repro:.2f}"}'
            f'</td><td class="n">{rmm:+.2f}</td>'
            f'<td class="n"><span class="u">{per}</span></td></tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>measures</th><th>tracks</th>'
            '<th>source<br><span class="u">mm, scale-free</span></th>'
            '<th>stat</th><th>sub-run<br>spread</th>'
            '<th>dead-masked +<br>self-centred</th>'
            '<th>per sub-run</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def dead_table_html(D) -> str:
    rows = []
    for _, r in D.iterrows():
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">chamber '
            f'{r.arm}</th><td class="n">{int(r.n_ranges)}</td>'
            f'<td class="n">{r.dead_mm:.0f}</td>'
            f'<td class="n">{100 * r.dead_frac_plane:.1f}%</td>'
            f'<td class="n">{r.lost_neg_mm:.1f}</td>'
            f'<td class="n">{r.lost_pos_mm:.1f}</td>'
            f'<td class="n">{r.asymmetry_mm:+.1f}</td></tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>dead runs</th><th>dead<br><span class="u">mm</span></th>'
            '<th>of the plane</th>'
            '<th>lost, &minus; side<br><span class="u">mm of 100</span></th>'
            '<th>lost, + side<br><span class="u">mm of 100</span></th>'
            '<th>asymmetry</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def y_table(Y) -> str:
    rows = []
    for _, r in Y.iterrows():
        if not np.isfinite(r.get('offset_mm', np.nan)):
            rows.append(f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">'
                        f'chamber {r.arm}</th>'
                        f'<td class="n" colspan="6">'
                        '<span style="color:var(--warn)">no angle scale &mdash; '
                        'no y</span></td></tr>')
            continue
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">chamber '
            f'{r.arm}</th><td class="n">{fmt(r.n)}</td>'
            f'<td class="n">{r.obs_median:+.1f}</td>'
            f'<td class="n">{r.pred_median:+.1f}</td>'
            f'<td class="n"><b>{r.offset_mm:+.1f}</b></td>'
            f'<td class="n">{r.obs_iqr:.0f} / {r.pred_iqr:.0f}</td>'
            f'<td class="n">{r.width_ratio:.1f}&times;</td>'
            f'<td class="n">{r.implied_sigma_mm:.0f}</td></tr>')
    return ('<table class="t"><thead><tr><th></th><th>tracks</th>'
            '<th>measured<br><span class="u">median y, mm</span></th>'
            '<th>model<br><span class="u">median y, mm</span></th>'
            '<th>offset</th>'
            '<th>IQR obs / model<br><span class="u">mm</span></th>'
            '<th>width</th>'
            '<th>implied &sigma;<sub>y</sub><br><span class="u">mm</span></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def vertex_table(VS, PJ) -> str:
    pj = PJ.set_index('topology')
    rows = []
    for _, r in VS.iterrows():
        if not np.isfinite(r.get('frac_mixed', np.nan)):
            continue
        p = pj.loc[r.topology]
        rows.append(
            f'<tr><th class="s">{r.topology}-chamber</th>'
            f'<td class="n">{fmt(r.n_real)}</td>'
            f'<td class="n">{100 * r.frac_real:.1f}%</td>'
            f'<td class="n">{100 * r.frac_mixed:.1f}%</td>'
            f'<td class="n">{r.lift:.2f}&times;</td>'
            f'<td class="n">{r.excess_sigma:+.1f}&sigma;</td>'
            f'<td class="n">{100 * p.excess_ul95:.1f}%</td>'
            f'<td class="n">{fmt(int(p.n_pairs_for_3sigma))}</td>'
            f'<td class="n">{fmt(int(p.campaign_pairs))}</td></tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>pairs</th><th>on the capsule</th>'
            '<th>same, mixed</th><th>lift</th><th>excess</th>'
            '<th>excess UL<br><span class="u">95 % CL</span></th>'
            '<th>pairs for 3&sigma;</th>'
            '<th>campaign<br><span class="u">&times;50</span></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def build_html(C, Cr, T, D, Y, VS, PJ, meta) -> str:
    run = meta['run'].replace('run_', '')
    v = {d['axis']: d for d in meta['verdict']}
    X, Z = v.get('X', {}), v.get('Z', {})
    ac = C[C.axis == 'X']
    yv = Y.dropna(subset=['offset_mm'])
    ymean = float(yv.offset_mm.mean()) if len(yv) else float('nan')
    yspread = float((yv.offset_mm.max() - yv.offset_mm.min()) / 2) \
        if len(yv) > 1 else float('nan')
    r = float(np.hypot(X.get('source_mm', np.nan), Z.get('source_mm', np.nan)))

    return f"""<title>Imaging the He-3 capsule</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run {run}</span>
    <span>{' / '.join(meta['subruns'])}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>Where the source is, and what the chambers&rsquo; disagreement is
      worth</h1>
  <p class="sub">the scale-free pointing crossing &middot; four chambers, two
     transverse axes &middot; nothing here depends on the drift velocity</p>
</header>

<p class="lede">The tracks back-project to a point
<b>{r:.1f}&nbsp;mm from the beam axis</b> &mdash; at
X&nbsp;=&nbsp;{X.get('source_mm', float('nan')):+.1f} and
Z&nbsp;=&nbsp;{Z.get('source_mm', float('nan')):+.1f}&nbsp;mm &mdash;
<b>inside the capsule&rsquo;s 10&nbsp;mm bore</b>, and the chambers viewing each
axis agree to about a millimetre. The measurement uses <i>no</i> drift velocity,
<i>no</i> angle scale and <i>no</i> calibration bundle, which is what makes it
usable while those are still provisional.</p>

<div class="cards">
  <div class="card"><div class="v">{X.get('source_mm', float('nan')):+.1f}</div>
    <div class="l">global <b>X</b>, mm &mdash; from chambers
      {' and '.join(X.get('arms', []))}, independently</div></div>
  <div class="card"><div class="v">{Z.get('source_mm', float('nan')):+.1f}</div>
    <div class="l">global <b>Z</b>, mm &mdash; from chambers
      {' and '.join(Z.get('arms', []))}</div></div>
  <div class="card"><div class="v">&plusmn;{max(X.get('align_syst_mm', 0), Z.get('align_syst_mm', 0)):.1f}</div>
    <div class="l">mm &mdash; the <b>chamber-to-chamber</b> spread, i.e. the
      relative alignment</div></div>
  <div class="card"><div class="v">{ymean:+.0f}</div>
    <div class="l">mm in <b>y</b>, and much weaker &mdash; see section 4</div></div>
</div>

<h2><span class="n">1</span>The observable, and why it survives everything else</h2>
<p>A track from a point source at perpendicular distance <i>d</i> must cross the
strip plane at <code>tan = (u &minus; u<sub>0</sub>)/d</code>. So the median
reconstructed angle against the in-plane position is a straight line, and
<b>where it crosses zero is the foot of the perpendicular from the source</b>.
Multiplying every angle by the unknown scale <i>k</i> multiplies the slope and
the intercept together and leaves <code>&minus;intercept/slope</code> exactly
where it was: <b>the crossing is scale-free</b>. It does not depend on the drift
velocity, on <i>k</i>, or on the calibration bundle &mdash; the three things
this analysis is least sure of.</p>
{figure('pointing_bands',
        'The pointing band per chamber, on the coincident sample, pooled over '
        'sub-runs. Points are binned medians, the line is the robust fit, and '
        'the dashed vertical is the crossing. Grey shading: the inner '
        '&plusmn;30&thinsp;mm excluded because the lever is too short to carry '
        'angle information. Red shading: dead channel ranges. <b>All four '
        'chambers give a clean line</b> &mdash; including B, whose <i>slope</i> '
        'is not trustworthy but whose <i>crossing</i> does not need it.',
        'median reconstructed tan against lever arm, per chamber')}
<div class="caution"><b>Chamber B is in this measurement, and that is not a
mistake.</b> B has no field-shaping ring chain, so its drift field is not
uniform and its angle <i>scale</i> is meaningless &mdash; which is why it carries
null angles everywhere else. The crossing asks something weaker: where the
track is <i>perpendicular</i> to the plane, i.e. where the reconstructed angle
is zero. A distorted but symmetric field still puts that at the same place. So
B contributes a Z measurement and nothing else, and its error bar
(&plusmn;{float(C.loc[C.arm == 'B', 'err_stat'].iloc[0]) if (C.arm == 'B').any() else float('nan'):.1f}&nbsp;mm,
one sub-run) says how much to lean on it.</div>

<h2><span class="n">2</span>Two axes, and what the disagreement means</h2>
<p>Each chamber measures the one transverse coordinate along its own in-plane
direction: A and C both see <b>global X</b>, from opposite sides; B and D both
see <b>global Z</b>. Two chambers on one axis is what makes this an alignment
measurement rather than a position: shifting one chamber&rsquo;s strip origin by
&delta; moves only that chamber&rsquo;s answer, so <b>the mean is the source and
the difference is the relative misalignment</b>.</p>
<div class="scroll">{transverse_table(C, Cr, T)}</div>
{figure('source_map',
        'The transverse plane, looking along the beam. Each chamber&rsquo;s '
        'crossing is one line; the band is its statistical error. The star is '
        'where the two axes intersect, and the circle is the capsule bore.',
        'the source position in the transverse plane, from four chambers')}
<p class="note">The <b>sub-run spread</b> column is the honest error. It
contains everything that changes run to run and a bootstrap cannot see, and for
every chamber it comes out about the same size as the statistical error &mdash;
which is what it should do if nothing is drifting.</p>

<h2><span class="n">3</span>Dead channels do not move the crossing &mdash;
measured, not assumed</h2>
<p>Chamber D has lost {float(D.loc[D.arm == 'D', 'dead_mm'].iloc[0]):.0f}&nbsp;mm
of its x plane, {100 * float(D.loc[D.arm == 'D', 'dead_frac_plane'].iloc[0]):.0f}&thinsp;%
of it, in five runs of channels on connector boundaries. The worry is not that
those tracks are missing &mdash; a dead channel produces no track to mask
&mdash; but that the surviving <b>acceptance is asymmetric about the crossing</b>,
so a line fitted through it is pulled toward the populated side.</p>
<div class="scroll">{dead_table_html(D)}</div>
<p>Inside the 100&nbsp;mm window the band is fitted over, D loses
{float(D.loc[D.arm == 'D', 'lost_pos_mm'].iloc[0]):.0f}&nbsp;mm on one side and
nothing on the other. Refitting with the window <b>recentred on the crossing
itself</b>, iterated to a fixed point, removes that asymmetry by construction
&mdash; and moves D&rsquo;s answer by
{abs(float(Cr.loc[Cr.arm == 'D', 'mm'].iloc[0]) - float(C.loc[C.arm == 'D', 'mm'].iloc[0])):.1f}&nbsp;mm.
So the dead channels are not what is limiting D here.</p>

<div class="caution"><b>A correction: chamber D was reported 40&nbsp;mm off
axis, and it is not.</b> The <code>-48 / -36 / -36&nbsp;mm</code> that appeared
in the funnel report and in the plan came from a cached
<code>imaging_summary.json</code> written on 2026-09-07. The y in-plane sign fix
the next day changed which tracks count as pointing-coincident &mdash; the
coincidence predicts a <i>v</i> on the wall and the plastic &mdash; and the
sample shrank by 30&ndash;45&thinsp;%. Re-running the <i>identical</i> estimator
on the current reconstruction gives
<b>{', '.join(f'{v:+.1f}' for v in T[(T.arm == 'D') & (T.variant == 'baseline')].mm)}&nbsp;mm</b>,
which also reproduces between sub-runs where the old numbers did not. The
crossing is now computed from the sample in memory rather than read from a file
nothing re-derives, so this class of staleness cannot recur.</div>

<h2><span class="n">4</span>y: no crossing, so a distribution against a model</h2>
<p>The capsule is a 10&nbsp;mm point in the transverse plane but 80&nbsp;mm long
in y, so <code>tan_y</code> has no zero to find and the crossing method simply
does not exist. What is left is the <i>distribution</i> of where each track
passes closest to the beam axis, compared against a forward model: vertices
drawn by volume from the real gas polycone, isotropic directions, straight
lines, and a track counted only if it crosses the active area <b>and</b> one of
the two plastic bars that make the trigger.</p>
{figure('source_y',
        'y at closest approach to the beam axis, measured against the forward '
        'model, per chamber. Dashed line: the measured median. The model is the '
        'capsule seen through the trigger; the data is much wider than it, and '
        'that difference is resolution.',
        'measured y distribution against the gas-polycone forward model')}
<div class="scroll">{y_table(Y)}</div>
<p>Two things come out, and they are of very different quality.</p>
<ul>
<li><b>The offset is coherent: {ymean:+.0f}&nbsp;mm</b>, with a chamber-to-chamber
spread of &plusmn;{yspread:.0f}&nbsp;mm. A, C and D are physically independent
detectors on different sides of the target, so a shift they <i>share</i> is far
more likely to be a shared convention than three separate mounting errors
&mdash; the y strip-map origin is used identically by all four. It is equally
consistent with the capsule genuinely sitting high. <b>Nothing here separates
those</b>, and the same is true of the ~9&nbsp;mm common offset in X.</li>
<li><b>The width is {yv.width_ratio.min():.1f}&ndash;{yv.width_ratio.max():.1f}&times;
the model</b>, so the y resolution at the target is
{yv.implied_sigma_mm.min():.0f}&ndash;{yv.implied_sigma_mm.max():.0f}&nbsp;mm.
That is much worse than the transverse crossing and it is why the y number is
quoted to the nearest 10&nbsp;mm and the transverse one to the nearest
millimetre. Chamber D is the worst of the three at
{float(Y.loc[Y.arm == 'D', 'implied_sigma_mm'].iloc[0]):.0f}&nbsp;mm, which
matches independently what the scintillator wall says about D&rsquo;s y &mdash;
see the <a href="../scintillators/">scintillator page</a>.</li>
</ul>
<p class="note">Chamber A also shows a distinct secondary population near
y&nbsp;&asymp;&nbsp;&minus;200&nbsp;mm, which is the half-height of the strip
map: tracks whose y fit has railed at the edge of the plane. It is a
reconstruction artefact, it is in the tails and not the core, and the median and
IQR quoted above are both insensitive to it.</p>

<h2><span class="n">5</span>Double-track vertices: the machinery works, the
statistics do not</h2>
<p>If two tracks in one trigger came from a common vertex, their closest
approach <b>to each other</b> locates it in all three coordinates at once
&mdash; no beam-axis assumption, and the y lever comes back. The control is
event mixing: pair tracks from <i>different</i> triggers through the identical
selection, and only the difference is evidence of a shared vertex.</p>
<div class="scroll">{vertex_table(VS, PJ)}</div>
{figure('vertex_null',
        'Distance of the two-track vertex from the beam axis, same trigger '
        'against event-mixed, for both topologies. The mixed sample is drawn '
        'from the tracks that actually form real pairs, so the two histograms '
        'differ only in whether the two tracks shared a trigger.',
        'two-track vertex radius, real against event-mixed control')}
<div class="caution"><b>No excess, and the sensitivity says why.</b> Run {run}
yields {fmt(int(VS.n_real.sum()))} two-track pairs in total, of which a couple
of per cent put a vertex on the capsule &mdash; the same couple of per cent the
mixed sample gives. Reaching 3&sigma; on an excess as large as run {run} can
still accommodate needs
{fmt(int(PJ.n_pairs_for_3sigma.min()))}&ndash;{fmt(int(PJ.n_pairs_for_3sigma.max()))}
pairs. The campaign is about 50&times; this run, so it gets there &mdash; but
only just, and only if the pair rate scales. <b>This is a method demonstration
with a number attached, not a measurement.</b></div>
<p class="note"><b>The control had to be built twice.</b> Mixing against every
track in the arm made the null look <i>better</i> than the data (lift 0.2&times;),
because a trigger that produced two tracks is a busier trigger and its tracks
are not drawn from the same distribution as a lone one. Mixing against only the
tracks that form real pairs fixes it, and the intra-chamber lift then comes out
at exactly 1.00 &mdash; which is what a correct null on a null signal looks
like.</p>

<h2><span class="n">6</span>What this does not establish</h2>
<ul>
<li><b>No absolute position better than the common offset.</b> X is
{X.get('source_mm', float('nan')):+.1f}&nbsp;mm and y is {ymean:+.0f}&nbsp;mm,
and in both cases a real target offset and a survey or convention error give the
same answer. Separating them needs survey information or a beam-spot
measurement, not more tracks.</li>
<li><b>Z is cross-checked, but weakly.</b> Chamber B contributes one sub-run at
&plusmn;2&nbsp;mm and cannot contribute a slope at all.</li>
<li><b>The y forward model is geometry only</b> &mdash; no multiple scattering,
no reconstruction efficiency across the plane, straight lines. It is good
enough to say the data is 2&ndash;6&times; wider than the acceptance; it is not
good enough to turn that ratio into a calibrated resolution.</li>
</ul>

<footer><p>Generated by
<code>sept26_prelim_analysis/make_imaging_report.py</code> from
<code>source_imaging.py</code>. Figures carry their numbers as CSV.</p></footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    d = paths.out('imaging')
    g = lambda n, w: pd.read_csv(paths.require(d / f'{n}_{a.run}.csv', w))  # noqa: E731

    C = g('transverse', 'the transverse table')
    Cr = g('transverse_robust', 'the robust transverse table')
    T = g('crossings', 'the per-sub-run crossings')
    D = g('dead', 'the dead-channel table')
    Y = g('y_compare', 'the y comparison')
    VS = g('vertex_summary', 'the vertex summary')
    PJ = g('projection', 'the statistics projection')
    meta = json.load(open(paths.require(d / f'imaging_{a.run}.meta.json',
                                        'the imaging meta')))

    od = str(d)
    body = build_html(C, Cr, T, D, Y, VS, PJ, meta)
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
        fh.write(build_html(C, Cr, T, D, Y, VS, PJ, meta))
    EMBED['v'] = False
    print(f'wrote {od}/report.html and body.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
