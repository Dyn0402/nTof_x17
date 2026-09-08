#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_angle_report.py -- build ``report.html`` for the opening-angle study.

The final observable, and the first time it is put against an expectation
rather than shown on its own.

Generated, never hand-written.

    python -m sept26_prelim_analysis.make_angle_report --run run_145
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
TOPO = ('intra', 'perpendicular', 'opposing')
TOPO_DESC = {
    'intra': ('both legs in one chamber', '&theta; &lesssim; 90&deg;',
              'the IPC continuum measured directly &mdash; no X17 can appear '
              'here, since its minimum is 109&deg;'),
    'perpendicular': ('neighbouring chambers', '&theta; ~ 60&ndash;120&deg;',
                      'the transition region, and the only topology where the '
                      'physics models differ enough to be separated'),
    'opposing': ('facing chambers, A&ndash;C only', '&theta; &gtrsim; 110&deg;',
                 'the signal region &mdash; and B&ndash;D is lost entirely '
                 'with chamber B'),
}


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


def topo_table(real, C) -> str:
    rows = []
    for t in TOPO:
        g = C[C.topology == t]
        if g.empty:
            continue
        n = int(g.n_obs.iloc[0])
        best = g.loc[g.chi2dof.idxmin()]
        ipc = g[g.model.str.startswith('IPC')]
        arms = ', '.join(f'{a}&ndash;{b}' for a, b in
                         real[real.topo == t].groupby(['arm1', 'arm2']).size().index)
        what, span, why = TOPO_DESC[t]
        rows.append(
            f'<tr><th class="s">{t}<span class="why">{what}<br>{arms}</span></th>'
            f'<td class="n">{fmt(n)}</td><td class="n">{span}</td>'
            f'<td class="n">{100 * g.frac_above_x17_obs.iloc[0]:.1f}%</td>'
            f'<td class="n">{100 * ipc.frac_above_x17_pred.min():.1f}'
            f'&ndash;{100 * ipc.frac_above_x17_pred.max():.1f}%</td>'
            f'<td class="n">{100 * g.frac_above_x17_mixed.iloc[0]:.1f}%</td>'
            f'<td>{best.model}<span class="why">'
            f'&chi;&sup2;/dof {best.chi2dof:.1f}</span></td></tr>')
    return ('<table class="t"><thead><tr><th>topology</th>'
            '<th>pairs</th><th>angles it can hold</th>'
            '<th>above 109&deg;<br><span class="u">measured</span></th>'
            '<th>above 109&deg;<br><span class="u">IPC &times; acceptance</span></th>'
            '<th>above 109&deg;<br><span class="u">accidentals</span></th>'
            '<th>best-fitting model</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def chi2_table(C) -> str:
    models = sorted(C.model.unique(),
                    key=lambda m: (not m.startswith('event'), m))
    head = ''.join(f'<th>{m}</th>' for m in models)
    rows = []
    for t in list(TOPO) + ['all']:
        g = C[C.topology == t].set_index('model')
        if g.empty:
            continue
        cells = []
        best = g.chi2dof.min()
        for m in models:
            if m not in g.index:
                cells.append('<td class="n">&mdash;</td>')
                continue
            v = g.loc[m, 'chi2dof']
            w = ('font-weight:600;color:var(--good)' if v == best else '')
            cells.append(f'<td class="n" style="{w}">{v:.1f}</td>')
        rows.append(f'<tr><th class="s">{t}</th>'
                    f'<td class="n">{fmt(int(g.n_obs.iloc[0]))}</td>'
                    + ''.join(cells) + '</tr>')
    return ('<table class="t"><thead><tr><th></th><th>pairs</th>' + head
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def acc_table(ACC) -> str:
    rows = []
    for _, r in ACC.iterrows():
        rows.append(f'<tr><th class="s">{r.arm1}&ndash;{r.arm2}'
                    f'<span class="why">{r.topology}</span></th>'
                    f'<td class="n">{fmt(r.n_obs)}</td>'
                    f'<td class="n">{r.n_acc:.0f}</td>'
                    f'<td class="n">{r.n_acc / max(r.n_obs, 1):.1f}&times;</td>'
                    f'<td class="n">{r.sigma:+.1f}&sigma;</td></tr>')
    return ('<table class="t"><thead><tr><th>arms</th><th>pairs seen</th>'
            '<th>Poisson prediction</th><th>over-predicts by</th>'
            '<th>discrepancy</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def build_html(real, C, R, ACC, V, P, meta, amet) -> str:
    run = meta['run'].replace('run_', '')
    n = int(len(real))
    mixed_best = C[C.model.str.startswith('event-mixed')]
    best_pair = C[~C.model.str.startswith('event-mixed')]
    off = amet['source_offset_mm']

    return f"""<title>The opening angle, against an expectation</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run {run}</span>
    <span>{' / '.join(meta['subruns'])}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>The final observable, and the first thing to compare it to</h1>
  <p class="sub">{fmt(n)} two-track pairs &middot; three topologies &middot;
     physics &times; acceptance, with the physics carried as a band</p>
</header>

<p class="lede">A measured opening-angle spectrum is
<code>physics(&theta;) &times; acceptance(&theta;)</code>, and only the second
factor is ours. This page builds both &mdash; a pair generator validated against
the full Geant4 simulation to <b>KS = {V.ks.max():.3f}</b>, and a straight-line
acceptance toy through the as-built geometry &mdash; and puts run {run}&rsquo;s
{fmt(n)} pairs against them. <b>The answer is that the pairs look like
accidentals.</b> The event-mixed shape fits every topology at
&chi;&sup2;/dof&nbsp;{mixed_best.chi2dof.min():.1f}&ndash;{mixed_best.chi2dof.max():.1f};
the best genuine pair spectrum manages
{best_pair.chi2dof.min():.1f}&ndash;{best_pair.chi2dof.max():.0f}. That is what a
null looks like at these statistics, and it agrees with the two other null
results already on the board.</p>

<h2><span class="n">1</span>What a pair is born with</h2>
<p>⁴He* de-excites at 20.58&nbsp;MeV. An X17 at 16.8&nbsp;MeV is <i>slow</i>
(&gamma;&nbsp;=&nbsp;1.225), so it cannot collimate its daughters and the pair
comes out at <b>109&deg; or more</b>, piling up at 110&ndash;140&deg;. An
internal-pair-creation virtual photon can have any mass down to
2m<sub>e</sub>, and a light one is fast, so the IPC continuum falls steeply from
small angles &mdash; and the X17 peak would sit on its tail.</p>
{figure('angle_physics',
        'Birth opening angle for both channels. The dashed curves are the full '
        'Geant4 truth (300 000 events) and they lie under the toy&rsquo;s solid '
        'curves, which is what makes the toy usable. The shaded band is the '
        'IPC continuum across four modelling assumptions; its lower edge is the '
        'deliberately extreme 1/M³ mass spectrum, included to bracket rather '
        'than to claim.',
        'X17 and IPC opening-angle distributions at birth, with Geant4 overlay')}
<p class="note"><b>The band is where the honesty is.</b> The Geant4 generator
samples <code>dN/dM<sub>ee</sub> ~ 1/M<sub>ee</sub></code> and decays the virtual
photon isotropically. Neither is the internal-pair matrix element. In particular
the 20.21&nbsp;MeV state of ⁴He is <b>0&#8314; &rarr; 0&#8314;</b>, which cannot
emit a real photon at all &mdash; the transition is E0 and proceeds only through
a <i>longitudinally polarised</i> virtual photon, which decays as
sin&sup2;&theta;*, not isotropically. So the alternatives are generated and the
spread between them is quoted, rather than one curve being presented as the
prediction.</p>

<h2><span class="n">2</span>What the geometry lets through</h2>
<p>Four chambers at 90&deg; means the opening angle a pair can have is decided
almost entirely by <i>which two chambers it lands in</i>. The acceptance toy
throws pairs <b>flat in opening angle</b> &mdash; so what comes out is an
acceptance and not a prediction &mdash; from the capsule at the position the
pointing crossing measured
(<a href="../source-imaging/">X&nbsp;=&nbsp;{off[0]:+.1f}, Z&nbsp;=&nbsp;{off[2]:+.1f}&nbsp;mm</a>),
and applies, in order: the active area, the dead channels found in the data, the
two-plastic-bar trigger ray-traced through its own gap, and the
scintillator-tagged reconstruction efficiency per chamber.</p>
{figure('angle_acceptance',
        'Acceptance against opening angle, per topology. Small angles are '
        'intra-chamber and large angles are opposing; between them the '
        'acceptance falls by roughly six times, and the X17 signal region sits '
        'on the rising edge where opposing pairs take over.',
        'geometric acceptance per topology against opening angle')}
<div class="scroll">{topo_table(real, C)}</div>
<div class="caution"><b>Chamber B costs half the signal topology.</b> B has no
field cage and therefore no angle, so the B&ndash;D opposing channel does not
exist at all and every B leg is invisible to this measurement. What is left is
A&ndash;C, which is {fmt(int(C[C.topology == 'opposing'].n_obs.iloc[0]))} pairs
in this run.</div>

<h2><span class="n">3</span>The measured spectrum</h2>
{figure('angle_spectrum',
        'Measured pairs per topology against the folded expectations. The band '
        'is IPC &times; acceptance across the model spread, the dashed line is '
        'X17 &times; acceptance at the same normalisation, and the grey step is '
        'the event-mixed sample &mdash; two tracks that did not share a '
        'trigger, put through the identical selection.',
        'measured opening-angle spectrum per topology against expectations')}
<div class="scroll">{chi2_table(C)}</div>
<p><b>The event-mixed shape wins in every topology, and not narrowly.</b> That
is the result: at run {run}&rsquo;s statistics the two-track sample is
consistent with two unrelated tracks that happened to share a trigger, and is
not described by any pair spectrum from a common vertex. It is the same
conclusion the two-chamber rate reached
(<a href="../reco-funnel/">excess 1.5 &plusmn; 11.2 events</a>) and the same one
the double-track vertex reached
(<a href="../source-imaging/">lift 1.00&times;</a>), by a third route.</p>

<h2><span class="n">4</span>The background normalisation is the honest gap</h2>
<p>Nothing is subtracted on this page, and that is a decision with a measurement
behind it. Two candidate normalisations for the accidental component were tried
and <b>both fail</b>:</p>
<ul>
<li>The <b>event-mixed sample carries no rate</b>. It is built pair for pair
with the data, so subtracting all of it subtracts the signal too. It is a shape
template and is used as one.</li>
<li>The <b>Poisson rate</b> <code>N<sub>trig</sub> &times; p<sub>i</sub>
&times; p<sub>j</sub></code> over-predicts the observed pair count <b>in every
arm combination</b>, by a factor {ACC.eval('n_acc/n_obs').min():.1f}&ndash;{ACC.eval('n_acc/n_obs').max():.1f}:</li>
</ul>
<div class="scroll">{acc_table(ACC)}</div>
<p>The reason is not subtle: the production trigger is a wall-and-plastic
coincidence in <i>one</i> arm, so the arms are not independent given a trigger
and the product of single rates is not the pair rate. The funnel report hit the
same wall for the two-chamber rate and solved it with a <b>control chamber</b>
rather than a rate calculation. <b>Doing the same here is the single highest-value
next step for this observable</b>, because without it the accidental component
can only be shown, not removed.</p>

<h2><span class="n">5</span>The test that depends least on the model</h2>
<p>Comparing shapes <i>within</i> a topology is weak, because the acceptance
almost entirely determines what angles that topology can hold &mdash; the intra
sample cannot exceed 109&deg; and the opposing sample almost cannot fall below
it, whatever the physics. What does discriminate is the <b>fraction above the
X17 threshold</b>, compared across topologies, where the acceptance
normalisation, the vertex model and the efficiency scale are largely common and
divide out.</p>
{figure('angle_summary',
        'Fraction of pairs above 109&deg; per topology: measured, against the '
        'IPC &times; acceptance band and against the accidental shape. The '
        'measured points sit on the accidentals in all three.',
        'fraction of pairs above the X17 threshold, per topology')}
<div class="caution"><b>One number does not sit quietly, and it is not a
signal.</b> The perpendicular topology measures
{100 * float(R.loc[R.topology == 'perpendicular', 'frac_obs'].iloc[0]):.1f}
&plusmn; {100 * float(R.loc[R.topology == 'perpendicular', 'err'].iloc[0]):.1f}&thinsp;%
above 109&deg; against an IPC&nbsp;&times;&nbsp;acceptance band of
{100 * C[(C.topology == 'perpendicular') & C.model.str.startswith('IPC')].frac_above_x17_pred.min():.1f}&ndash;{100 * C[(C.topology == 'perpendicular') & C.model.str.startswith('IPC')].frac_above_x17_pred.max():.1f}&thinsp;%.
Before that means anything, four things have to be true and none of them is
established: the acceptance would have to be right in exactly the region where
it falls fastest and the toy is least reliable; the accidental component &mdash;
which predicts {100 * float(C[(C.topology == 'perpendicular') & C.model.str.startswith('event-mixed')].frac_above_x17_mixed.iloc[0]):.1f}&thinsp;%,
i.e. the measured value &mdash; would have to be small; multiple scattering and
the leptons&rsquo; energies would have to be negligible; and chamber&nbsp;B
would have to not be missing. <b>The accidental shape already explains it, so
the correct reading is that it is background.</b></div>

<h2><span class="n">6</span>What the campaign would need</h2>
<p>The separation between two shapes over N pairs grows as N, so the numbers
below are not a hope but arithmetic on what is measured here. The campaign is
about 50&times; run {run}.</p>
<div class="scroll"><table class="t"><thead><tr><th>topology</th>
<th>pairs now</th><th>campaign, &times;50</th>
<th>IPC band above 109&deg;</th>
<th>pairs to resolve a 10&thinsp;% excess at 3&sigma;</th>
</tr></thead><tbody>
{''.join(f'<tr><th class="s">{r.topology}</th>'
         f'<td class="n">{fmt(int(r.n_now))}</td>'
         f'<td class="n">{fmt(int(r.campaign_n))}</td>'
         f'<td class="n">{100 * r.band_lo:.1f}&ndash;{100 * r.band_hi:.1f}%</td>'
         f'<td class="n">{"&mdash;" if not np.isfinite(r.n_for_3sigma_10pct) or r.n_for_3sigma_10pct <= 0 else fmt(int(r.n_for_3sigma_10pct))}</td></tr>'
         for r in P.itertuples() if r.topology != 'all')}
</tbody></table></div>
<p class="note">The perpendicular topology is the one that pays: it needs about
{fmt(int(P.loc[P.topology == 'perpendicular', 'n_for_3sigma_10pct'].iloc[0]))}
pairs and the campaign should deliver
{fmt(int(P.loc[P.topology == 'perpendicular', 'campaign_n'].iloc[0]))}. The
opposing topology has almost no shape information in it &mdash; its acceptance
forces essentially every pair above 109&deg; regardless of physics &mdash; so
what it contributes is a <i>rate</i>, and a rate needs the background
normalisation of section 4.</p>

<h2><span class="n">7</span>What this is not</h2>
<ul>
<li><b>Not a measurement of anything.</b> It is the machinery, exercised end to
end on one run, with the numbers the campaign will need.</li>
<li><b>No invariant mass.</b> That needs the energy sharing, which needs
scintillator calorimetry, which needs a calibration that does not exist.</li>
<li><b>The acceptance is an over-estimate</b>: straight lines, no multiple
scattering, no energy loss, no lepton energy dependence, and no double-track
finding efficiency at small separations. Every one of those removes pairs.</li>
<li><b>The physics band is generated, not derived.</b> It brackets the
assumptions in the simulation&rsquo;s generator; it is not a calculation of the
internal-pair matrix element.</li>
</ul>

<footer><p>Generated by
<code>sept26_prelim_analysis/make_angle_report.py</code> from
<code>pair_physics.py</code>, <code>acceptance.py</code> and
<code>opening_angle.py</code>. Figures carry their numbers as CSV.</p></footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    d = paths.out('angle')
    g = lambda n, w: pd.read_csv(paths.require(d / n, w))  # noqa: E731

    C = g(f'compare_{a.run}.csv', 'the comparison')
    R = g(f'ratio_{a.run}.csv', 'the ratio test')
    ACC = g(f'accidentals_{a.run}.csv', 'the accidental estimate')
    V = g('physics_validation.csv', 'the physics validation')
    P = g(f'angle_projection_{a.run}.csv', 'the projection')
    real = pd.read_parquet(d / f'pairs_{a.run}.parquet')
    meta = json.load(open(paths.require(d / f'angle_{a.run}.meta.json',
                                        'the angle meta')))
    amet = json.load(open(paths.require(d / f'acceptance_{a.run}.meta.json',
                                        'the acceptance meta')))

    od = str(d)
    body = build_html(real, C, R, ACC, V, P, meta, amet)
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
        fh.write(build_html(real, C, R, ACC, V, P, meta, amet))
    EMBED['v'] = False
    print(f'wrote {od}/report.html and body.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
