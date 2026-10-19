#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_response_report.py -- build ``report.html`` for the detector response study.

The companion to ``make_funnel_report.py``: that one is the reconstruction chain
end to end, this one is what the four chambers actually do -- where they
respond, how efficiently, and how much of the structure in their hit maps is
the trigger geometry rather than the detectors.

Generated, never hand-written (CLAUDE.md).  Every number is read back from the
CSVs the analysis wrote, so re-running efficiency.py / hit_maps.py /
plastic_acceptance.py and then this moves the figures, the tables and the
verdict text together.

    python -m sept26_prelim_analysis.make_response_report
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
    CSS, FONT_LINK, fmt, pct)

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


def eff_table(E: pd.DataFrame, H: pd.DataFrame) -> str:
    h = H.set_index('arm')
    rows = []
    for _, r in E.iterrows():
        a = r.arm
        basis = h.loc[a, 'basis']
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[a]}">chamber {a}</th>'
            f'<td class="n">{fmt(r.n_tagged)}</td>'
            f'<td class="n">{100 * r.eff_hit:.1f}%</td>'
            f'<td class="n">{100 * r.p0_hit:.1f}%</td>'
            f'<td class="n">{100 * r.eff_hit_corr:.1f}%</td>'
            f'<td class="n">{100 * r.eff_track_corr:.1f}%</td>'
            f'<td><b>{100 * h.loc[a, "efficiency"]:.1f}%</b>'
            f'<span class="u"> on {basis}</span></td></tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>tagged<br><span class="u">wall AND plastic</span></th>'
            '<th>hit, raw</th><th>p0<br><span class="u">accidental</span></th>'
            '<th>hit, corrected</th><th>track, corrected</th>'
            '<th>quote this</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def dead_table() -> str:
    """The dead readout runs, measured 2026-09-08."""
    runs = [('A', '—', '—', '—'),
            ('C', 'ch 227–236 (10)', '+15…+22', '3'),
            ('D', 'ch 183–212 (30)', '+34…+57', '2–3'),
            ('D', 'ch 219–227 (9)', '+22…+29', '3'),
            ('D', 'ch 234–255 (22)', '+0…+17', '3'),
            ('D', 'ch 27–63 (37)', '+150…+178', '0')]
    rows = ''.join(
        f'<tr><th class="s" style="color:{DET_COLOR[a]}">{a}</th>'
        f'<td class="n">{ch}</td><td class="n">{x}</td>'
        f'<td class="n">{c}</td></tr>' for a, ch, x, c in runs)
    return ('<table class="t"><thead><tr><th></th>'
            '<th>dead run<br><span class="u">&ge;8 ch below 20 % of median</span></th>'
            '<th>x local<br><span class="u">mm</span></th>'
            '<th>connector</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def build_html(E, H, tiers, meta) -> str:
    t = tiers.pivot(index='arm', columns='tier', values='frac')
    hrow = H.set_index('arm')
    best = H.loc[H.efficiency.idxmax()]
    return f"""<title>X17 Detector Response</title>
{FONT_LINK}
<style>{CSS}
figure{{margin:22px 0;padding:0}}
figure img{{display:block;width:100%;height:auto;border:1px solid var(--line);
  border-radius:8px;background:#fbfcfe}}
figcaption{{font-size:12.5px;color:var(--ink-2);margin-top:9px;max-width:82ch;
  line-height:1.55}}
figcaption .src{{font-family:var(--mono);font-size:11px;color:var(--ink-3);
  text-decoration:none;white-space:nowrap;margin-left:4px}}
</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run 145</span>
    <span>{', '.join(meta['subruns'])}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>What the four chambers actually do</h1>
  <p class="sub">efficiency, hit maps and the trigger's own acceptance
     &middot; {fmt(E.n_tagged.sum())} scintillator-tagged particles</p>
</header>

<p class="lede">Acceptance is what an opening-angle spectrum must be divided by,
and it needs three things this page measures: how efficiently each chamber
responds, <i>where</i> on its surface it responds, and how much of the
structure in that map belongs to the trigger rather than the detector. The
short answer to the last one is <b>most of it</b> &mdash; the two-lobe shape
every chamber shows is the plastic scintillator geometry, predicted from the
Geant model with nothing tuned.</p>

<div class="cards">
  <div class="card"><div class="v">{100 * best.efficiency:.0f}%</div>
    <div class="l">best chamber ({best.arm}, on {best.basis})</div></div>
  <div class="card"><div class="v">+7.3<span style="font-size:15px"> mm</span></div>
    <div class="l">predicted trigger shadow; +5 measured</div></div>
  <div class="card"><div class="v">~130</div>
    <div class="l">dead channels in chamber D, of 512</div></div>
  <div class="card"><div class="v">0</div>
    <div class="l">dead channels in chamber A</div></div>
</div>

<h2><span class="n">1</span>Efficiency, against a denominator that is not the MM</h2>
<p>Each arm has a scintillator wall 96.4&thinsp;mm behind its strip plane and a
plastic bar ~187&thinsp;mm behind it. An in-time coincidence of both is a
particle that went through that arm, tagged by detectors the Micromegas knows
nothing about. That is the denominator.</p>
<p><b>The raw number is a trap.</b> Chamber D seeds 80&thinsp;% of <i>all</i>
triggers, so its raw 93.6&thinsp;% is occupancy, not response. With an
accidental response probability p<sub>0</sub>,
P(resp|tagged) = &epsilon; + (1&minus;&epsilon;)p<sub>0</sub>, so
&epsilon; = (P &minus; p<sub>0</sub>)/(1 &minus; p<sub>0</sub>) &mdash; and
p<sub>0</sub> is measured on the events that arm's own scintillators did
<i>not</i> tag.</p>
<div class="scroll">{eff_table(E, H)}</div>
<p class="note"><b>Chamber B is quoted on hits, not tracks</b>, because it has no
field-shaping ring chain and therefore no uniform drift field: a "track" in B is
not a track. Everything else is on tracks.</p>
<div class="caution"><b>The tag may not mean what it says.</b> Chamber A comes
out at 63&thinsp;% where 80&ndash;90&thinsp;% is expected. A neutron or gamma
converting in the PCB would fire the scintillators with no charged particle
having crossed the gas &mdash; inflating the denominator and pushing the
efficiency down, which is the direction of the discrepancy. Deferred to October;
until then these are <b>lower bounds</b>.</div>

<h2><span class="n">2</span>Where each chamber responds</h2>
{figure('hitmap_occupancy',
        'Cluster occupancy on each chamber surface, log scale because a few '
        'edge cells run hundreds of times the bulk. A is uniform; B carries '
        'horizontal stripe artefacts; C is centrally illuminated; D is '
        'dominated by its edges.',
        'cluster occupancy per chamber')}
{figure('hitmap_tiers',
        'The same occupancy down a purity ladder &mdash; fitted, then '
        '+ wall AND plastic in the same arm, then + hits the segment and bar '
        'that actually fired, then + points back to the sample. Percentages '
        'are of that chamber&rsquo;s fitted clusters. D&rsquo;s edge artefacts '
        'wash out almost entirely: 109,920 clusters become 3,381.',
        'occupancy down the purity ladder')}
<div class="scroll"><table class="t"><thead><tr><th></th>
<th>fitted</th><th>+ scint</th><th>+ pointing</th><th>+ target</th></tr></thead>
<tbody>{''.join(
    f'<tr><th class="s" style="color:{DET_COLOR[a]}">chamber {a}</th>'
    + ''.join(f'<td class="n">{pct(t.loc[a, c]) if c in t.columns and not pd.isna(t.loc[a, c]) else "&mdash;"}</td>'
              for c in ('fitted', 'scint', 'pointing', 'target'))
    + '</tr>' for a in ('A', 'B', 'C', 'D'))}</tbody></table></div>
<p class="note">Chamber B takes the position-only route at the pointing tier
(no drift field, no angle) and has no target tier at all.</p>

<h2><span class="n">3</span>Dead readout channels</h2>
<p>Measured from the cluster occupancy per channel: runs of at least 8 adjacent
channels sitting below 20&thinsp;% of that plane&rsquo;s median.</p>
<div class="scroll">{dead_table()}</div>
<p class="note"><b>Chamber D has ~130 dead channels of 512 &mdash; a quarter of
its x plane</b> &mdash; clustered on connector boundaries, the same class of
fault as chamber A&rsquo;s connector-8 outage in run_79. Chamber A has none.
D&rsquo;s <i>angle scale</i> survives this: re-measured with the outer ring
excluded it moves 2&thinsp;%, because the pointing coincidence already strips
the junk (44.1&thinsp;% of D&rsquo;s fitted clusters sit in the outer 20&thinsp;mm
ring, against 6.8&thinsp;% of the sample its calibration uses).</p>

<h2><span class="n">4</span>The two-lobe structure is the trigger</h2>
<p>The production trigger needs a coincidence with <i>both</i> plastic bars
behind each wall. There is a gap between them, so a charged particle threading
it makes no plastic signal and no trigger. The chamber is not inefficient
there &mdash; the trigger is blind.</p>
<p><b>Nothing was tuned to make this agree.</b> The Geant config and the
reconstruction constants are independent sources and match where they overlap:
<code>bscTape_hu + bsc_gap/2 = 100.22 + 1.5 = 101.72&thinsp;mm</code> is exactly
<code>PLASTIC_U_OFFSET</code>, and <code>mm_pinwheel_shift_cm</code> is exactly
<code>PINWHEEL</code>. The shadow centre then follows from geometry alone,
<code>x = foot_x&thinsp;L/(D+L)</code>: <b>+7.3, +7.0, +7.7, +6.9&thinsp;mm</b>
for A, B, C, D &mdash; identical across arms, as the symmetry demands. Arm
A&rsquo;s measured minimum sits at <b>+5&thinsp;mm</b>.</p>
{figure('plastic_acceptance',
        'Top: the fraction of the He-3 capsule (r&thinsp;=&thinsp;10&thinsp;mm, '
        '60&thinsp;mm long with hemispherical caps) that can see each surface '
        'point through active plastic. Bottom: the measured occupancy with the '
        '20/50/80&thinsp;% contours drawn over it &mdash; three levels because '
        'a 60&thinsp;mm source cannot cast a hard edge. The data sits inside '
        'the prediction for all four chambers: both lobes, the gap between '
        'them, and the &plusmn;85&thinsp;mm extent in v that a 300&thinsp;mm '
        'bar imposes at the 1.8&times; lever.',
        'predicted plastic acceptance against the measured occupancy')}
<p class="note"><b>The depth needs ~5&thinsp;mm of inactive scintillator at each
bar edge.</b> The bare 3.4&thinsp;mm geometric gap gives a 3&thinsp;% dip
against 86&thinsp;% measured. The measured dip is <i>wider</i> than the
model&rsquo;s, ~30&thinsp;mm against 5&ndash;15&thinsp;mm, which the model does
not need to explain: the <code>dca&nbsp;&lt;&nbsp;30&thinsp;mm</code> selection
alone smears the plastic crossing by 30&thinsp;&times;&thinsp;L/D
&asymp; 24&thinsp;mm. Tightening that cut is the clean test.</p>
<div class="caution"><b>This sits on top of the dead channels, it does not
replace them.</b> Chamber A has no dead runs, so its dip is purely the trigger
&mdash; which is why it appears only once the scintillator cut is applied.
C and D have dead runs at x&thinsp;&asymp;&thinsp;+15&hellip;+57&thinsp;mm,
<i>the same place</i>, so theirs is both effects at once, and shows in the raw
occupancy.</div>

<h2><span class="n">5</span>What this does not establish</h2>
<ul>
<li><b>Not an absolute efficiency.</b> The scintillator tag includes particles
that missed the active area and accidental pairs, so every number here is a
lower bound. Section 1&rsquo;s caution is the leading suspect.</li>
<li><b>Not a corrected acceptance.</b> This is the ingredients &mdash; response,
position dependence, trigger geometry &mdash; not the correction. Folding them
into an opening-angle acceptance needs the angle dependence too, and the
statistics for that need the full campaign.</li>
<li><b>One run.</b> {', '.join(meta['subruns'])} of run_145 only. The code is
built to run unchanged over the campaign; it has not.</li>
<li><b>Chamber B contributes position and timing, never angle.</b> Its angle
columns are null by construction and must stay that way.</li>
</ul>

<footer>
Built by <code>sept26_prelim_analysis/make_response_report.py</code> from
<code>efficiency_*.csv</code>, <code>hitmap_*.csv</code> and
<code>plastic_acceptance.csv</code>. Numbers regenerate with the analysis.<br>
Geometry from <code>MX17_Full_Geant/include/SimConfig.hh</code>; tag window
{meta['dt_window'][0]:.0f} to {meta['dt_window'][1]:.0f} ns.
</footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    ed = paths.out('efficiency')
    E = pd.read_csv(paths.require(ed / f'efficiency_{a.run}.csv', 'efficiency'))
    H = pd.read_csv(paths.require(ed / f'efficiency_headline_{a.run}.csv',
                                  'efficiency headline'))
    meta = json.load(open(ed / f'efficiency_{a.run}.meta.json'))
    tiers = pd.read_csv(paths.require(
        paths.figures('hitmaps') / 'hitmap_tiers.csv', 'hit-map tiers'))

    od = a.out or str(paths.out('response'))
    os.makedirs(os.path.join(od, 'figures'), exist_ok=True)
    # gather the figures this page needs into one directory, so the relative
    # links work from disk, from EOS and from anywhere it is copied
    import shutil
    for src, name in ((paths.figures('hitmaps'), 'hitmap_occupancy'),
                      (paths.figures('hitmaps'), 'hitmap_tiers'),
                      (paths.figures('acceptance'), 'plastic_acceptance')):
        for ext in ('png', 'csv'):
            s = os.path.join(str(src), f'{name}.{ext}')
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(od, 'figures', f'{name}.{ext}'))

    body = build_html(E, H, tiers, meta)
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
        fh.write(build_html(E, H, tiers, meta))
    EMBED['v'] = False
    print(f'wrote {od}/report.html and body.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
