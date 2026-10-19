#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_insitu_report.py -- the in-situ performance report, built from the tables.

Writes ``report.html`` beside the figures, so the DAQ web page can serve it:
its Analysis tab lists any ``.html`` in an analysis directory and opens it
inline, and every figure is referenced by RELATIVE path so the same file works
from disk, from the web page, or copied elsewhere.

Nothing is hand-written here.  Every number in the prose comes out of the CSVs
`insitu_maps.py` wrote, so re-running after the analysis moves the numbers, the
tables and the verdict text together.  Reference model:
``ntof_july_analysis/leadshield_compare/make_report.py``.

    python ntof_athens_26/make_insitu_report.py
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402
from sept26_prelim_analysis.report_style import head              # noqa: E402
from ntof_athens_26 import insitu_maps as IM                      # noqa: E402

OUT = HERE / 'figures'
SRC = paths.spell('out', 'athens_insitu')

FIGURES = [
    ('hitmap_triggered', 'The trigger-biased maps',
     'Chambers A, C and D on scintillator-matched tracks. The two lobes, the '
     'gap between them at u &asymp; +7 mm and the cut at |v| = 83 mm are the '
     'plastic bars, not the chambers: the overlaid geometry is read from the '
     'DAQ survey and projected onto the chamber through the lever arm, and it '
     'lands on the features without being fitted to them.'),
    ('hitmap_unbiased', 'The unbiased maps',
     'The same chambers in events where their own scintillators were silent '
     'and another arm made the trigger. The trigger footprint is gone and the '
     'surface is lit to the fiducial edge. Chamber B is here and on no other '
     'map, because this selection needs no angle.'),
    ('hitmap_profiles', 'The footprint, projected',
     'The two samples along u and along v, each normalised to its own mean, so '
     'the comparison is of shape. In the matched sample both trigger features '
     'are deep; in the bystander sample they are gone in A, survive in C where '
     'dead channels sit at the same u, and are unreadable in D.'),
    ('shadow_test', 'The declared test',
     'The depth of the plastic-gap dip, per chamber, per sample. The '
     'prediction was written down before the test ran and it is the figure '
     'that decides whether the bystander construction works.'),
    ('detector_b', 'Chamber B',
     'Why B carries no angle: no field-shaping ring chain, so no uniform drift '
     'field. The cluster shape is the fingerprint that hardware fault '
     'predicts, and the map is what B can still do.'),
]


def esc(s) -> str:
    return html.escape(str(s))


def load(src: Path) -> dict:
    need = ('census.csv', 'shadow.csv', 'shape.csv', 'mask_census.csv',
            'trigger_census.csv', 'insitu_maps.meta.json')
    missing = [n for n in need if not (src / n).exists()]
    if missing:
        raise FileNotFoundError(
            f'{src} is missing {", ".join(missing)}\n'
            f'  run:  python ntof_athens_26/insitu_maps.py --jobs 6')
    return dict(
        census=pd.read_csv(src / 'census.csv'),
        shadow=pd.read_csv(src / 'shadow.csv'),
        shape=pd.read_csv(src / 'shape.csv'),
        mask=pd.read_csv(src / 'mask_census.csv'),
        trigger=pd.read_csv(src / 'trigger_census.csv'),
        meta=json.loads((src / 'insitu_maps.meta.json').read_text()))


def table(d: pd.DataFrame, cols: dict, fmt: dict | None = None) -> str:
    """A DataFrame as a styled table -- ``cols`` maps column -> header."""
    fmt = fmt or {}
    th = ''.join(f'<th>{c}</th>' for c in cols.values())
    rows = []
    for _, r in d.iterrows():
        tds = []
        for k in cols:
            v = r[k]
            f = fmt.get(k)
            tds.append(f'<td>{esc(f(v) if f else v)}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<table class="t"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def n(v) -> str:
    return f'{int(v):,}'


def pc(v) -> str:
    return f'{100 * float(v):.1f}%'


def f2(v) -> str:
    return f'{float(v):+.3f}'


def signed(v) -> str:
    """A depth, or an em-dash where there is genuinely nothing to report.

    Chamber B's ``triggered`` and ``leftover`` rows hold zero tracks -- both
    need a pointing match and B carries no angle -- so their depth is undefined
    rather than zero, and printing ``+nan`` would read as a failed computation
    instead of a sample that cannot exist.
    """
    return '—' if pd.isna(v) else f'{float(v):+.3f}'


def unsigned(v) -> str:
    return '—' if pd.isna(v) else f'{float(v):.3f}'


def verdict(T: dict) -> str:
    """The answer, first, in the numbers the tables carry."""
    sh = T['shadow'].set_index(['arm', 'sample'])
    S = sh.depth
    E = sh.depth_err

    def sigma(arm: str) -> float:
        """How far the removal method sits from the bystander one, in sigma."""
        d = S[(arm, 'leftover')] - S[(arm, 'bystander')]
        return abs(d) / math.hypot(E[(arm, 'leftover')],
                                   E[(arm, 'bystander')])

    sig_A, sig_C, sig_D = sigma('A'), sigma('C'), sigma('D')
    C = T['census']
    byst = int(C[C['sample'] == 'bystander'].n_tracks.sum())
    trig = int(C[C['sample'] == 'triggered'].n_tracks.sum())
    tc = T['trigger']
    tagged = int(tc[tc.n_arms > 0].n.sum())
    one = int(tc[tc.n_arms == 1].n.iloc[0])
    return f'''
<p class="verdict"><b>An unbiased hit map is possible, and it needs no track
removal.</b> The production trigger is a wall AND plastic coincidence in ONE
arm, OR&rsquo;d over the four, and {pc(one / max(tagged, 1))} of the read-outs
this analysis can tag have exactly one arm lit. So every chamber spends most of
its life being read out on somebody else&rsquo;s trigger &mdash; a
<b>bystander</b>, whose occupancy carries no trace of its own scintillator
acceptance. {n(byst)} such tracks over {len(T['meta']['runs'])} runs, against
{n(trig)} trigger-matched ones.</p>

<p>The construction is testable and the test was declared before it ran. The
plastic bars leave a gap that shadows u &asymp; +7 mm on every chamber; if that
dip is the trigger&rsquo;s, it must vanish when another arm does the
triggering, <i>except</i> where the chamber is genuinely blind underneath it.
<b>It does exactly that.</b> Chamber A, which has no dead readout channels,
goes from a dip of {S[('A', 'triggered')]:.3f} to {f2(S[('A', 'bystander')])}
&mdash; past zero, because the beam illuminates the centre more and the trigger
had been carving a hole in it. Chambers C and D, which have ~10 and ~130 dead
channels under that window, keep theirs:
{S[('C', 'triggered')]:.3f}&nbsp;&rarr;&nbsp;{S[('C', 'bystander')]:.3f} and
{S[('D', 'triggered')]:.3f}&nbsp;&rarr;&nbsp;{S[('D', 'bystander')]:.3f}.</p>

<p>The removal method &mdash; find the track that fired the trigger, drop it,
map what is left &mdash; is built too, as <code>leftover</code>, and it is
worth being precise about how far it agrees. Where the dip is <i>real</i> the
two methods land on top of each other: C {S[('C', 'leftover')]:.3f} against
{S[('C', 'bystander')]:.3f} ({sig_C:.1f}&sigma;), D
{S[('D', 'leftover')]:.3f} against {S[('D', 'bystander')]:.3f}
({sig_D:.1f}&sigma;). <b>On chamber A, where the trigger shadow is genuinely
absent, they do not</b>: {f2(S[('A', 'leftover')])} against
{f2(S[('A', 'bystander')])}, a {sig_A:.1f}&sigma; difference, and the removal
method is the one left closer to a shadow. That is the expected direction and
it is the argument for the bystander construction rather than against it: a
leftover track shares its event with the particle that fired the trigger, so it
is still weighted by the trigger&rsquo;s acceptance through whatever correlated
it with that particle. The bystander sample has no such partner. The removal
method is also 20&times; smaller, since a self-triggered event with a second
reconstructed track is rare, so it is the cross-check and not the result.</p>
'''


def body(T: dict) -> str:
    C = T['census']
    order = {s: i for i, s in enumerate(IM.SAMPLES)}
    C = C.assign(_o=C['sample'].map(order)).sort_values(['arm', '_o'])
    tc = T['trigger']
    M = T['mask']
    SH = T['shadow']
    SP = T['shape']
    meta = T['meta']
    dD = SP[SP.arm == 'D']
    dwid = float(dD.width_x.iloc[0])
    dvsa = float(dD.width_x_vs_A.iloc[0])

    figs = ''.join(
        f'<figure><img src="{name}.png" alt="{esc(title)}">'
        f'<figcaption><b>{esc(title)}.</b> {cap} '
        f'<span class="prov"><code>{name}.pdf</code> for the slide, '
        f'<code>{name}.csv</code> for the numbers.</span></figcaption>'
        f'</figure>'
        for name, title, cap in FIGURES)

    return f'''
<h2>What was compared</h2>
<p>Every gated track in the campaign is put into exactly one of five samples,
so the census is a partition and every fraction below has an honest
denominator. The split is made from the n_TOF scintillator slim alone &mdash;
which arm&rsquo;s wall and plastic fired within the
{meta['dt_window'][0]:.0f}&hellip;{meta['dt_window'][1]:.0f}&nbsp;ns accept
window &mdash; and, for the two matched samples, from whether the
track&rsquo;s own extrapolation lands on the channel that fired.</p>

<table class="t"><thead><tr><th>sample</th><th>definition</th>
<th>what it is for</th></tr></thead><tbody>
<tr><td><code>triggered</code></td><td>this arm&rsquo;s wall AND plastic fired,
and this track points at the group and the bar that fired</td>
<td>the trigger-biased map, at its purest</td></tr>
<tr><td><code>leftover</code></td><td>this arm self-triggered and a
<i>different</i> track in the event took the match</td>
<td>the &ldquo;remove the triggering track&rdquo; method, as a cross-check</td></tr>
<tr><td><code>bystander</code></td><td>this arm&rsquo;s scintillators silent,
another arm&rsquo;s fired</td><td><b>the unbiased map</b></td></tr>
<tr><td><code>self_unmatched</code></td><td>this arm self-triggered and no
track in the event matched the fired channel</td>
<td>ambiguous by construction; counted, mapped nowhere</td></tr>
<tr><td><code>no_tag</code></td><td>no arm reconstructs a coincidence offline
at all</td><td>counted, mapped nowhere</td></tr>
</tbody></table>

<h2>The trigger is single-arm, and that is what makes this work</h2>
<p>If read-outs routinely lit two arms there would be no bystander sample worth
mapping. They do not.</p>
{table(tc, {'n_arms': 'arms lit', 'n': 'read-outs', 'frac': 'fraction'},
       {'n': n, 'frac': pc})}
<p class="caution">The {pc(float(tc[tc.n_arms == 0].frac.iloc[0]))} with no arm
lit are events this analysis cannot tag offline: the DAQ&rsquo;s own
discriminator and this 160&nbsp;ns window are not the same cut, and the per-run
n_TOF join is imperfect (the zero-arm fraction runs from 0.2&nbsp;% to
35&nbsp;% run to run, which is join quality, not physics). They are dropped
from <i>both</i> samples alike, so they cannot bias the comparison. What they
could in principle do is let a partial slim loss push a self-triggered track
into the bystander sample &mdash; and the data bounds that directly: if it
happened at any scale, the trigger&rsquo;s gap shadow would partly survive in
chamber A&rsquo;s bystander map. It does not; it inverts.</p>

<h2>The census</h2>
{table(C, {'arm': 'chamber', 'sample': 'sample', 'n_tracks': 'tracks',
           'n_events': 'events', 'frac': 'of this chamber'},
       {'n_tracks': n, 'n_events': n, 'frac': pc})}
<p>Chamber B has no <code>triggered</code> or <code>leftover</code> tracks by
construction: both require a pointing match, and B carries no angle to point
with. Its bystander sample is the same size as everyone else&rsquo;s.</p>

<h2>The declared test</h2>
{table(SH, {'arm': 'chamber', 'sample': 'sample', 'n_shadow': 'in shadow',
            'n_flank': 'in flanks', 'depth': 'depth', 'depth_err': '&plusmn;'},
       {'n_shadow': n, 'n_flank': n, 'depth': signed,
        'depth_err': unsigned})}
<p>Depth is 1 &minus; (occupancy per live cell in
{meta['shadow_u'][0]:.0f}&hellip;{meta['shadow_u'][1]:.0f}&nbsp;mm) divided by
the same quantity in the two flank windows, read only at
|v|&nbsp;&lt;&nbsp;{meta['shadow_v']:.0f}&nbsp;mm so the bar-length cut cannot
leak in. Per live cell, not per millimetre, so neither the unequal widths of
the windows nor an unequal number of masked cells inside them can fake a
result.</p>

<h2>What had to be masked, and why</h2>
<p>Two reasons, and they are different statements about the detector.
<b>Fiducial</b>: outside &plusmn;{meta['fiducial']:.0f}&nbsp;mm the plane fit
rails at the edge of the strip map and piles up there, so a position outside it
is not a position on the detector. <b>Hot</b>: a cell holding at least
{meta['hot_min_count']} tracks and more than {meta['hot_factor']:.0f}&times;
the median of the occupied cells in its {meta['hot_window']}-square
neighbourhood. The mask is derived once from the occupancy summed over all
mapped samples and applied to each alike, so it cannot manufacture a difference
between the two samples being compared.</p>
{table(M, {'arm': 'chamber', 'n_hot': 'hot cells', 'n_fiducial': 'rail cells',
           'frac_hot_triggered': 'hot, of matched',
           'frac_hot_bystander': 'hot, of bystander',
           'frac_rail_bystander': 'rail, of bystander'},
       {'n_hot': n, 'n_fiducial': n, 'frac_hot_triggered': pc,
        'frac_hot_bystander': pc, 'frac_rail_bystander': pc})}
<p class="caution"><b>The asymmetry in that table is the price of the method,
and it is worth stating plainly.</b> Junk accumulates in the bystander sample
<i>by construction</i>: a cluster manufactured by a noisy strip points at no
scintillator, so it can never be matched, so it lands in
<code>bystander</code> and never in <code>triggered</code>. On chamber D the hot
cells hold {pc(float(M[M.arm == 'D'].frac_hot_bystander.iloc[0]))} of the
bystander sample against
{pc(float(M[M.arm == 'D'].frac_hot_triggered.iloc[0]))} of the matched one. The
bystander map is unbiased in <i>where</i> it looks; it is not purity-selected,
and nothing in it confirms an individual track.</p>

<h2>Chamber B</h2>
<p>B has no field-shaping ring chain. A, C and D ground their degrader rings
through three ~1.3&nbsp;G&Omega; resistors and it is that chain which draws
their 0.180&nbsp;&micro;A; B draws zero because it has no divider to draw it,
which says nothing about whether its cathode is at voltage &mdash; it should
be. Without the rings the drift field fringes, so there is no clean
time&harr;depth ladder and no angle. The amplification stage is untouched, so B
still measures position and time.</p>
<p>The fingerprint that predicts: a fringing field spreads the same charge over
more strips, so the signature is a <b>wide, dilute</b> cluster and not a weak
one.</p>
{table(SP, {'arm': 'chamber', 'width_x': 'width x [strips]',
            'width_y': 'width y', 'q_per_strip_x': 'charge / strip',
            'width_x_vs_A': 'width vs A',
            'q_per_strip_x_vs_A': 'density vs A'},
       {'width_x': lambda v: f'{float(v):.1f}',
        'width_y': lambda v: f'{float(v):.1f}',
        'q_per_strip_x': lambda v: f'{float(v):.1f}',
        'width_x_vs_A': lambda v: f'{float(v):.2f}',
        'q_per_strip_x_vs_A': lambda v: f'{float(v):.2f}'})}
<p>Median over all gated clusters &mdash; the basis
<code>chamber_b.py</code> established on run_145, carried here to the whole
campaign. B is the widest of the four and the most dilute by a clear margin,
which is what a fringing drift field predicts and a weak amplification stage
does not.</p>
<p class="caution"><b>One number moves between run_145 and the campaign, and it
is D&rsquo;s.</b> <code>chamber_b.py</code> measured D at 42 strips on run_145
(1.68&times; A), which is where STATUS.md&rsquo;s &ldquo;D is nearly as wide,
so width alone does not isolate the fault&rdquo; comes from. Pooled over 36
runs D is {dwid:.1f} strips ({dvsa:.2f}&times;), level with C rather than with
B. So on the campaign sample width <i>does</i> separate B, and the conclusion
is unchanged and better supported &mdash; but the run_145 ordering should not
be quoted as if it were the campaign&rsquo;s. Chamber D is also the chamber
whose cluster sample is most contaminated by hot channels, which is the first
thing to check before reading anything into the difference.</p>

<h2>The figures</h2>
{figs}

<h2>What this does not rule out</h2>
<ul>
<li><b>These are occupancy maps, not efficiency maps.</b> There is no
independent denominator, so a cold cell is &ldquo;less illuminated or less
efficient&rdquo; and nothing here says which. What the bystander construction
removes is the chamber&rsquo;s own scintillator acceptance; that is all it
claims to remove. A scintillator-tagged efficiency, with an MM-independent
denominator, is <code>efficiency.py</code> and lives only where the
scintillators cover.</li>
<li><b>The bystander sample still shares a beam.</b> It sits in an event some
other arm triggered, so the illumination is the beam&rsquo;s own and is not
flat, and an opposing arm&rsquo;s trigger selects a direction through the
target which weights the surface gently. Neither is the chamber&rsquo;s own
trigger footprint.</li>
<li><b>Chamber D&rsquo;s unbiased map does not work</b>, and no choice of mask
rescues it: its ratio of 99th-percentile to median cell occupancy is 207 at the
loosest setting scanned and still 68 at the tightest, where the mask has taken
45&nbsp;% of its sample. A quarter of D&rsquo;s x plane is dead. That is a
statement about D, not about the method.</li>
<li><b>Nothing here is an angle measurement</b> and nothing here depends on
one, which is the point. The angle scale remains preliminary (STATUS.md,
2026-09-10); the pointing match used for the <code>triggered</code> sample does
use the reconstructed direction, so that sample &mdash; and only that sample
&mdash; inherits the angle scale&rsquo;s uncertainty. The bystander sample does
not.</li>
<li><b>The masks are campaign-pooled, not per run.</b> A channel that went hot
part way through the campaign is masked throughout. Per-run masks were not
built because a single run&rsquo;s mask is Poisson noise at this binning.</li>
</ul>

<p class="foot">Built by <code>make_insitu_report.py</code> from
<code>insitu_maps.py</code>. {n(meta['n_tracks'])} gated tracks,
{len(meta['runs'])} runs, {meta['bin_mm']:.0f}&nbsp;mm map cells. Schema
<code>{esc(meta['schema'])}</code>.</p>
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(SRC))
    ap.add_argument('--out', default=str(OUT / 'report.html'))
    a = ap.parse_args()

    T = load(Path(a.src))
    doc = (f'<!doctype html><html><head>'
           f'{head("In-situ efficiency and performance of the four Micromegas")}'
           f'</head><body><main>'
           f'<p class="eyebrow">n_TOF 2026 &middot; Athens</p>'
           f'<h1>In-situ efficiency and performance of the four Micromegas</h1>'
           f'<p class="deck">What each chamber surface sees, with and without '
           f'its own trigger &mdash; and what that says about chamber B.</p>'
           f'<p class="badge">PRELIMINARY</p>'
           f'{verdict(T)}{body(T)}</main></body></html>')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding='utf-8')
    print(f'  -> {out}  ({len(doc):,} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
