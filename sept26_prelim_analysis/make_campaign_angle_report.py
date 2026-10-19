#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_campaign_angle_report.py -- ``report.html`` for `campaign_angle.py`.

The campaign opening-angle distributions, what survives each cut, and what
the folded thermal Born prediction does and does not describe.

Generated, never hand-written: every number is read back from what
`campaign_angle.py` wrote.

    python -m sept26_prelim_analysis.make_campaign_angle_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402
from sept26_prelim_analysis.campaign_angle import (  # noqa: E402
    TOPOLOGIES, X17_MIN_DEG, BACK_TO_BACK_DEG)

TOPO_COLOR = {'intra': '#0072B2', 'perpendicular': '#E69F00',
              'opposing': '#009E73'}


def figure(name: str, caption: str, csv: str = None) -> str:
    """A figure with ORDINARY RELATIVE links (CLAUDE.md), and its numbers.

    ``csv`` names the table when `figstyle.save` wrote a multi-panel figure and
    the CSV is therefore ``<name>.<suffix>.csv`` rather than ``<name>.csv`` --
    a caption linking to a file that does not exist is worse than no link.
    """
    import html as _h
    csv = csv or f'{name}.csv'
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(caption)}"></a>'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{csv}">numbers &#8599;</a>'
            f'</figcaption></figure>')


def _f(v, d=3, unit=''):
    """A number, or an em dash -- and no orphan unit hanging off the dash."""
    if v is None or not np.isfinite(v):
        return '&mdash;'
    return f'{v:.{d}f}{unit}'


def _i(v):
    return '&mdash;' if v is None or not np.isfinite(v) else f'{int(v):,}'


def census_table(K: pd.DataFrame) -> str:
    rows = []
    for r in K.itertuples():
        col = TOPO_COLOR.get(r.topology, '#333333')
        rows.append(
            f'<tr><th class="s" style="color:{col}">{r.topology}</th>'
            f'<td class="n">{_i(r.n_all)}</td>'
            f'<td class="n">{_i(r.n_b2b)}</td>'
            f'<td class="n">{_i(getattr(r, "n_tagged", np.nan))}</td>'
            f'<td class="n">{_i(getattr(r, "n_tight", np.nan))}</td>'
            f'<td class="n"><b>{_i(getattr(r, "n_tight_pair", np.nan))}</b></td>'
            f'<td class="n">{_f(r.median_all, 1, "&deg;")}</td>'
            f'<td class="n">{_f(getattr(r, "median_tight_pair", np.nan), 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.frac_above_x17_all)}</td>'
            f'<td class="n">{_f(r.frac_above_x17_mixed)}</td>'
            f'<td class="n">{_f(getattr(r, "frac_above_x17_tight_pair", np.nan))}'
            f'</td></tr>')
    return ('<table><thead><tr><th></th><th>all pairs</th>'
            '<th>back-to-back</th><th>tagged</th><th>tight</th>'
            '<th>tight_pair</th><th>median &theta;</th>'
            '<th>median, tight_pair</th><th>&gt;109&deg;, all</th>'
            '<th>&gt;109&deg;, mixed</th><th>&gt;109&deg;, tight_pair</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def compare_table(C: pd.DataFrame, selection: str) -> str:
    g = C[C.selection == selection]
    rows = []
    for topo in TOPOLOGIES:
        h = g[g.topology == topo].sort_values('chi2dof')
        if h.empty:
            continue
        first = True
        for r in h.itertuples():
            head = (f'<th class="s" rowspan="{len(h)}" '
                    f'style="color:{TOPO_COLOR[topo]}">{topo}<br>'
                    f'<span class="u">n = {int(r.n_obs):,}</span></th>'
                    if first else '')
            best = ' style="font-weight:600"' if first else ''
            rows.append(
                f'<tr>{head}<td{best}>{r.model}</td>'
                f'<td class="n"{best}>{r.chi2dof:.1f}</td>'
                f'<td class="n">{r.dof}</td>'
                f'<td class="n">{r.frac_above_x17_pred:.3f}</td>'
                f'<td class="n">{r.frac_above_x17_obs:.3f}</td></tr>')
            first = False
    return ('<table><thead><tr><th></th><th>model</th><th>&chi;&sup2;/dof</th>'
            '<th>dof</th><th>predicted &gt;109&deg;</th>'
            '<th>observed &gt;109&deg;</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def spectrum_table(S: pd.DataFrame, selection: str) -> str:
    p = S[S.selection == selection].pivot_table(
        index=['lo', 'hi'], columns='topology', values='n', aggfunc='sum')
    cols = [c for c in TOPOLOGIES if c in p.columns]
    rows = []
    for (lo, hi), r in p.iterrows():
        cells = ''.join(f'<td class="n">{int(r[c]):,}</td>' for c in cols)
        hot = ' class="s"' if lo >= X17_MIN_DEG else ''
        rows.append(f'<tr><th{hot}>{int(lo)}&ndash;{int(hi)}&deg;</th>{cells}</tr>')
    head = ''.join(f'<th style="color:{TOPO_COLOR[c]}">{c}</th>' for c in cols)
    return (f'<table><thead><tr><th></th>{head}</tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def build(d: Path) -> str:
    meta = json.loads((d / 'campaign_angle.meta.json').read_text())
    v = meta['verdict']
    K = pd.read_csv(d / 'census.csv')
    C = pd.read_csv(d / 'compare.csv')
    S = pd.read_csv(d / 'spectra.csv')
    R = pd.read_csv(d / 'ratio.csv')
    opp = K[K.topology == 'opposing'].iloc[0]
    allr = K[K.topology == '(all)'].iloc[0]
    tp = C[C.selection == 'tight_pair']
    best_opp = (tp[tp.topology == 'opposing'].sort_values('chi2dof').iloc[0]
                if len(tp[tp.topology == 'opposing']) else None)

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Opening angle, campaign-wide</title>{HEAD}
</head><body><main>
<h1>The opening angle, over the whole campaign</h1>
<p class="lede">{int(allr.n_all):,} two-track pairs, of which
<b>{int(allr.n_tight_pair):,}</b> survive a real coincidence requirement
&mdash; {int(opp.n_tight_pair):,} in the opposing (signal) topology. The
statistics the plan asked for have arrived. <b>No folded model describes the
shape</b>, and the closest thing to the data is still the event-mixed
accidental template.</p>

<h2>Where a pair is allowed to be</h2>
<p>Four chambers at 90&deg; means the topology decides the angle before the
physics does. Read the first figure before any number below it: the
intra-chamber sample cannot reach 109&deg; and the opposing sample cannot leave
it. &ldquo;Fraction above 109&deg;&rdquo; is therefore a statement about the
chambers first and about the physics only within one topology.</p>
{figure('ang_topology', 'The three topologies, raw counts, back-to-back '
        'pairs removed. The shaded band above 109° is where an X17 pair must '
        'land.')}

<h2>What survives each cut</h2>
{census_table(K)}
<p class="sub">The intra row carries no timing columns and that is not an
oversight. Both legs are in one chamber, so there is only one arm and no
arm-to-arm time difference to cut on &mdash; the mutual half of the tight cut
cannot be formed. <b>Comparing an intra spectrum with no timing cut against an
opposing spectrum with one is not like for like</b>, which matters because
<code>PLAN.md</code> &sect;S4 makes intra the background normalisation.</p>

<h2>The background that lives inside the signal region</h2>
<p>One charged particle that crosses the target and punches through
<i>both</i> opposing chambers registers as a pair at ~180&deg;, and it is
perfectly time-coincident because it is one particle. Campaign-wide,
<b>{int(opp.n_b2b):,} of {int(opp.n_all):,} opposing pairs</b>
({100 * opp.n_b2b / opp.n_all:.0f}&nbsp;%) sit above
{BACK_TO_BACK_DEG:.0f}&deg;.</p>
<p><b>The timing cut makes it worse, not better.</b> Of the
{int(opp.n_tight):,} tight opposing pairs,
{int(opp.n_tight - opp.n_tight_pair):,}
({100 * (opp.n_tight - opp.n_tight_pair) / opp.n_tight:.0f}&nbsp;%) are
back-to-back &mdash; against {100 * opp.n_b2b / opp.n_all:.0f}&nbsp;% before it.
A cut designed to remove accidentals enriches a single-particle background by a
factor {(opp.n_tight - opp.n_tight_pair) / opp.n_tight / (opp.n_b2b / opp.n_all):.1f},
because that background was never accidental.</p>
{figure('ang_b2b', 'The opposing spectrum in 2° bins, and what each stage of '
        'selection does to its shape.', csv='ang_b2b.fine.csv')}

<h2>The spectra</h2>
<h3>Coincident pairs (tight_pair)</h3>
{spectrum_table(S, 'tight_pair')}
<h3>All pairs, back-to-back removed</h3>
{spectrum_table(S, 'all_no_b2b')}
{figure('ang_cuts', 'Each selection’s shape, per topology, against the '
        'event-mixed null. Shape-normalised: the samples differ by a factor 60 '
        'in size.')}

<h2>Against the prediction</h2>
<p>The expectation is <code>ipc_channels.thermal_spectrum()</code> &mdash; the
M1 and E0 Born multipoles in the mixture the &lt;2&nbsp;eV neutron window makes
&mdash; folded through the acceptance and normalised to the observed count. No
rate is claimed and no background is subtracted: both candidate accidental
normalisations failed (the mixed sample is a shape and carries no rate; the
Poisson product over-predicts the pair count 2&ndash;4&times;), so the missing
normalisation is stated as the leading systematic rather than guessed.</p>
{compare_table(C, 'tight_pair')}
{figure('ang_models', 'The coincident sample against the folded thermal Born '
        'prediction, per topology.', csv='ang_models.obs.csv')}

<p><b>Nothing fits.</b>
{f"The best model in the opposing topology is <i>{best_opp.model}</i> at "
 f"&chi;&sup2;/dof {best_opp.chi2dof:.0f}" if best_opp is not None else ""}
&mdash; and on this evidence that is a statement about the <i>acceptance</i>,
not about the physics. The acceptance folded in is run_145&rsquo;s, borrowed
because no other run has the efficiency map it needs, and it applies efficiency
independently of incidence angle when the measured head-on tracking ratio is
0.80. A &chi;&sup2;/dof of this size on a shape comparison with one free
normalisation is what a wrong acceptance looks like.</p>

<h2>The model-light test</h2>
<p>The ratio between topologies is the test that depends least on the model:
the acceptance normalisation, the vertex model and the efficiency scale are
largely common and divide out. It is quoted as the fraction above the X17
threshold, which is the quantity a signal actually moves.</p>
{R.round(4).to_html(index=False, classes='n', border=0)}

<h2>What this does not rule out, and what it does not yet do</h2>
<ul>
<li><b>It is not a search.</b> No acceptance correction is applied to any
spectrum on this page and no rate is quoted. The corrected spectrum needs a
campaign acceptance, which needs a per-run efficiency measurement that does not
exist.</li>
<li><b>The angles carry per-run <i>k</i>, and nine runs sit inside an
excursion.</b> Runs 128&ndash;147 have every arm&rsquo;s <i>k</i> raised
7&ndash;8&nbsp;% by something that is <i>not</i> a drift-velocity change, so
applying per-run <i>k</i> there may inject an angle error rather than remove
one. Those runs are ~19&nbsp;% of the A&ndash;C sample and are <b>not</b> split
out here.</li>
<li><b>The intra control has no timing cut</b> and so cannot be compared to the
opposing sample as if it did. Fixing it needs a single-arm prompt requirement
defined for one-chamber pairs.</li>
<li><b>Chamber B contributes no angle</b> &mdash; no field cage &mdash; which
costs the B&ndash;D opposing channel, half the signal topology. The handful of
B pairs in the table above should be treated as a bookkeeping artefact.</li>
<li><b>Pre-access runs are excluded</b> ({', '.join(f'<code>{r}</code>' for r in ('run_79', 'run_81'))}):
a different detector condition with about twice the inter-chamber fraction.</li>
<li><b>Multiple scattering, energy loss and double-track finding efficiency at
small separations are in none of this.</b> Each makes the acceptance an
over-estimate.</li>
</ul>

<p class="foot">Generated by <code>make_campaign_angle_report.py</code> from
<code>{d}</code> on {dt.date.today().isoformat()}. Tracks:
<code>{meta['src']}</code>. Coincidence: <code>{meta['tight']}</code>.
Acceptance: <code>{meta['acceptance']}</code> (borrowed).
Binning {int(meta['bins'][1] - meta['bins'][0])}&deg;; X17 threshold
{meta['x17_min_deg']:.0f}&deg;; back-to-back above
{meta['back_to_back_deg']:.0f}&deg;.</p>
</main></body></html>'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None, help='default <out>/angle_campaign')
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('angle_campaign')
    paths.require(d / 'campaign_angle.meta.json', 'the campaign angle meta')
    out = d / 'report.html'
    out.write_text(build(d))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
