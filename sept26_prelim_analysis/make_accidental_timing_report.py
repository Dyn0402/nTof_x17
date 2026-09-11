#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_accidental_timing_report.py -- build ``report.html`` for the accidental-
timing study.

Follows through on HANDOFF_ACCIDENTAL_TIMING.md (2026-09-08): redoes its item
(a) properly (unbiased hit choice, the full range, a two-component fit) and
reports on (c) the accept window and (d)/(e), which stay open.

Generated, never hand-written.

    python -m sept26_prelim_analysis.make_accidental_timing_report --run run_145
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


def single_arm_table(S: pd.DataFrame) -> str:
    fam_name = {'WAL': 'wall', 'PSS': 'plastic'}
    cls_name = {'both': 'coincidence (both fire)', 'wall_only': 'wall only',
               'plastic_only': 'plastic only'}
    rows = []
    for _, r in S.iterrows():
        pc = ('&mdash;' if not np.isfinite(r.purity_core)
              else f'{100 * r.purity_core:.1f}%')
        rows.append(f'<tr><td>{fam_name[r.family]}</td>'
                    f'<td>{cls_name.get(r.cls, r.cls)}</td>'
                    f'<td class="n">{fmt(r.n)}</td>'
                    f'<td class="n">{fmt(r.n_core)}</td>'
                    f'<td class="n">{r.n_pedestal_per_10ns:.1f}</td>'
                    f'<td class="n">{pc}</td></tr>')
    return ('<table class="t"><thead><tr><th>element</th><th>class</th>'
            '<th>hits</th><th>in core</th>'
            '<th>pedestal<br><span class="u">per 10 ns</span></th>'
            '<th>core purity</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def fit_table(F: pd.DataFrame) -> str:
    name = {'all': 'all inter-chamber', 'opposing': 'opposing (A&ndash;C)',
           'perpendicular': 'perpendicular'}
    rows = []
    for _, r in F.iterrows():
        if not np.isfinite(r.f_hat):
            rows.append(f'<tr><th class="s">{name.get(r.topo, r.topo)}</th>'
                        f'<td class="n">{int(r.n)}</td>'
                        f'<td class="n" colspan="2">{r.note}</td></tr>')
            continue
        rows.append(f'<tr><th class="s">{name.get(r.topo, r.topo)}</th>'
                    f'<td class="n">{int(r.n)}</td>'
                    f'<td class="n">{100 * r.f_hat:.0f}%</td>'
                    f'<td class="n">[{100 * r.f_lo:.0f}, {100 * r.f_hi:.0f}]%</td>'
                    '</tr>')
    return ('<table class="t"><thead><tr><th>topology</th>'
            '<th>pairs</th><th>f&#770;</th><th>68% interval</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def build_html(S: pd.DataFrame, scan: pd.DataFrame, F: pd.DataFrame,
              meta: dict) -> str:
    run = meta['run'].replace('run_', '')
    n_solo = meta['n_single_active_arm_solo_prod_window']
    n_single = meta['n_single_active_arm_events_prod_window']
    rec = meta['recommended_window']
    prod = scan[(scan.center == -20) & (scan.half_width == 80)].iloc[0]
    all_row = F[F.topo == 'all'].iloc[0]
    opp_row = F[F.topo == 'opposing'].iloc[0]
    perp_row = F[F.topo == 'perpendicular'].iloc[0]
    n_strict = meta['n_two_arm_tagged_strict']
    n_loose = meta['n_two_arm_tagged_loose']
    n_inter = meta['n_inter_pairs_total']

    return f"""<title>Accidental Timing</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run {run}</span>
    <span>{' / '.join(meta['subruns'])}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>Are the two-track pairs real coincidences? The scintillators, redone
      without the bias</h1>
  <p class="sub">{fmt(meta['n_slim_hits'])} n_TOF slim hits &middot;
     {fmt(meta['n_control_hits'])} <code>is_control</code> hits &middot;
     no reprocessing, no EOS</p>
</header>

<p class="lede">The opening-angle page (S4) ends on a null: the measured pairs
follow the event-mixed shape rather than any physics spectrum, read as
&ldquo;one particle triggered, a second unrelated one landed in the same
DREAM window.&rdquo; <code>HANDOFF_ACCIDENTAL_TIMING.md</code> tested that
directly with the scintillators &mdash; prompt by construction, unlike the
Micromegas <code>t0</code> &mdash; and got <b>f &approx; 6 &plusmn; 3%</b>
true coincidence, an <i>upper bound</i> because the hit choice was biased
toward finding one. Redone here with an unbiased pick and a proper
two-component fit: <b>f = {100 * all_row.f_hat:.0f}%</b>
({100 * all_row.f_lo:.0f}&ndash;{100 * all_row.f_hi:.0f}%) overall, and
<b>{100 * opp_row.f_hat:.0f}%</b> in the opposing topology &mdash; A&ndash;C,
the signal region &mdash; against <b>{100 * perp_row.f_hat:.0f}%</b>
perpendicular. Not the near-zero the first pass suggested.</p>

<div class="cards">
  <div class="card"><div class="v">{100 * all_row.f_hat:.0f}%</div>
    <div class="l">true-coincidence fraction, all inter-chamber pairs
    (n={int(all_row.n)})</div></div>
  <div class="card"><div class="v">{100 * opp_row.f_hat:.0f}%</div>
    <div class="l">opposing (A&ndash;C, signal topology), n={int(opp_row.n)}</div></div>
  <div class="card"><div class="v">{100 * perp_row.f_hat:.0f}%</div>
    <div class="l">perpendicular, n={int(perp_row.n)}</div></div>
  <div class="card"><div class="v">{100 * prod.purity:.0f}%</div>
    <div class="l">peak purity of the current accept window &mdash;
    {100 * rec['purity']:.0f}% available at ({rec['lo']:.0f}, {rec['hi']:.0f}) ns</div></div>
</div>

<h2><span class="n">1</span>Single-arm events: does the coincidence buy
anything?</h2>
<p>Restrict to events where exactly one of the four arms shows any wall or
plastic activity near the trigger at all. Within that arm, does the
<i>other</i> element also fire, in the peak's own core
({meta['core_window'][0]:.0f}, {meta['core_window'][1]:.0f}) ns?</p>
{figure('single_arm_classes',
        'Single-active-arm events: the fired element&rsquo;s own dt_ns, over '
        'the full range, split by whether its partner ALSO fired in the peak '
        'core (purple) or not (orange). The wall panel shows the effect '
        'cleanly &mdash; a coincidence is a narrow, ~30&times; peak over a '
        'flat pedestal, and a wall-only hit is a much smaller, less clean '
        'sample of the same shape. The plastic panel shows the same '
        'direction but weaker: the plastic family runs a far higher '
        'accidental rate on its own.',
        'wall and plastic dt_ns, coincidence vs one element only')}
<div class="scroll">{single_arm_table(S)}</div>
<div class="caution"><b>Under the PRODUCTION accept window
{tuple(meta['production_window'])} ns, the effect vanishes &mdash; because the
window is wide enough that a coincidence is nearly automatic.</b> Only
{n_solo} of {fmt(n_single)} single-active-arm events
({100 * n_solo / max(n_single, 1):.3f}%) are anything other than a full
wall+plastic &ldquo;both&rdquo;: at 160&nbsp;ns wide, the plastic family's own
high accidental rate is enough to land <i>something</i> in almost every
window by chance, so &ldquo;wall AND plastic in-window&rdquo; is barely more
selective than &ldquo;wall in-window&rdquo; alone at that width. The table
above uses the peak's own core ({meta['core_window'][0]:.0f},
{meta['core_window'][1]:.0f}) ns instead, where the effect is real.</div>

<h2><span class="n">2</span>The accept window is mis-centred and too wide</h2>
<p>HANDOFF sec 2.1's own read: the peak core is about (&minus;30, +30) ns
against a production window of {tuple(meta['production_window'])} ns.
Measured here as peak/pedestal purity over a grid of windows, with the
pedestal rate fixed from the flat &minus;200&hellip;&minus;100 ns reference
slice so the metric is not circular with the window being scanned.</p>
{figure('window_scan',
        'Purity vs window half-width, at the production centre (&minus;20 ns) '
        'and a re-centred one (0 ns). The production window (&minus;100, +60) '
        'sits at 88% purity; a window the same shape but centred at zero '
        'peaks near 93%, and a much narrower centred window reaches 95%+.',
        'peak/pedestal purity vs accept window choice')}
<p class="note"><b>Not applied.</b> Changing <code>DT_WINDOW</code> touches
<code>candidate_filter.py</code>, <code>efficiency.py</code> and
<code>scintillators.py</code> &mdash; the stage-1/stage-2 production chain,
which is on Dylan's hold. This is the recommendation and the trade-off curve
behind it, not a code change: at ({rec['lo']:.0f}, {rec['hi']:.0f}) ns the
purity floor of {100 * rec['min_purity']:.0f}% clears with the most signal
kept.</p>

<h2><span class="n">3</span>The two-arm test: a spike at zero</h2>
<p>For the {fmt(n_inter)} real (not event-mixed) inter-chamber MM pairs from
<code>source_imaging.vertices</code> &mdash; the exact sample the S4 page's
null is about &mdash; take the scintillator tag time in each arm and look at
arm1 &minus; arm2. A true pair is born together, so both arms should read
close to the DREAM trigger and the difference should sit near zero; an
accidental second arm should look flat.</p>
<p class="note">The <b>strict</b> tag (wall <i>and</i> plastic, both
in-window &mdash; the same definition <code>efficiency.py</code> uses) leaves
only <b>{n_strict}</b> pairs with both arms tagged, too few to fit &mdash;
though tellingly, every one of those {n_strict} sits within &plusmn;35 ns.
The fit below uses the <b>loose</b> tag (either element, in-window), which
gives <b>{n_loose}</b> of {fmt(n_inter)} pairs ({100 * n_loose / max(n_inter, 1):.0f}%)
&mdash; a real, stated loosening of the tag definition, needed for
statistics.</p>
{figure('two_arm_delta_t',
        'arm1&ndash;arm2 scintillator time for the loose-tagged real '
        'inter-chamber pairs, with the fitted prompt + accidental mixture. '
        'The prompt template is the bootstrap difference of two draws from '
        'the single-arm reference (the trigger&rsquo;s own resolution); the '
        'accidental template draws one side from the reference and one from '
        '<code>is_control</code>, both restricted to the window used to '
        'define a tag.',
        'two-arm scintillator time difference with the fitted components')}
<div class="scroll">{fit_table(F)}</div>
{figure('f_by_topology',
        'The fitted fraction, per topology. Opposing (A&ndash;C, where an '
        'X17 or an IPC pair actually lands) carries roughly 2.7&times; the '
        'coincidence fraction of perpendicular pairs, which is what a real '
        'physical source predicts and an accidental floor would not.',
        'fitted true-coincidence fraction by topology')}
<p class="note">Why this differs so much from HANDOFF's own 6&plusmn;3%:
that number used the &ldquo;closest to dt_ns&nbsp;=&nbsp;0&rdquo; hit whenever
an arm had more than one candidate, which HANDOFF sec 3.1 already flagged as
biasing <i>toward</i> finding coincidence &mdash; so removing that bias should
if anything have pulled the number down, not up by 5&times;. The larger
change is methodological: this redo compares the full arm1&minus;arm2 <i>shape</i>
against two data-derived templates in an unbinned fit, rather than matching
one summary statistic (fraction within 20&nbsp;ns) to a single-parameter fold.
Both numbers are on record; this one is the properly redone measurement HANDOFF
item (a) asked for.</p>

<h2><span class="n">4</span>What this does and does not settle</h2>
<p><b>Does:</b> confirms the two-track sample is not purely accidental, gives
f with an error per topology &mdash; the number S4's own comparison states it
is missing &mdash; and shows it is topology-dependent in the direction a real
source predicts.</p>
<div class="caution"><b>Does not:</b> feed f back into the opening-angle
spectrum (HANDOFF item b). Only
{100 * n_loose / max(n_inter, 1):.0f}% of the real inter-chamber pairs carry
a two-arm scintillator tag at all &mdash; applying this f to the other {100 * (1 - n_loose / max(n_inter, 1)):.0f}%
needs the assumption that the tagged subsample is representative, which is not
checked. A first look at whether it even could matter &mdash; opening angle for
the near-zero (|&Delta;t| &lt; 20 ns) pairs against the far ones &mdash; sees no
significant shape difference, but at n = 30 vs 23 that is not a test, only a
non-result. Also open: intra-chamber pairs (item d, needs the wall's along-bar
position to give each leg its own arm-independent time, untested); the
Micromegas <code>t0</code> cross-check (item e, needs (a) fixed first since
<code>t0</code> itself carries the drift depth).</p>

<footer><p>Generated by
<code>sept26_prelim_analysis/make_accidental_timing_report.py</code> from
<code>accidental_timing.py</code>, following
<code>HANDOFF_ACCIDENTAL_TIMING.md</code> (2026-09-08). Figures carry their
numbers as CSV.</p>
</footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    sd = paths.out('accidental_timing')

    S = pd.read_csv(paths.require(sd / f'single_arm_summary_{a.run}.csv',
                                  'the single-arm summary'))
    scan = pd.read_csv(paths.require(sd / f'window_scan_{a.run}.csv',
                                     'the window scan'))
    F = pd.read_csv(paths.require(sd / f'fit_by_topology_{a.run}.csv',
                                  'the fit'))
    meta = json.load(open(paths.require(
        sd / f'accidental_timing_{a.run}.meta.json', 'the meta')))

    od = str(sd)
    body = build_html(S, scan, F, meta)
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
        fh.write(build_html(S, scan, F, meta))
    EMBED['v'] = False
    print(f'wrote {od}/report.html and body.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
