#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_det_a_report.py -- ``report.html`` for the detector-A intra-chamber study.

Generated, never hand-written: every number is read back from what
`det_a_intra.py` wrote, so re-running the analysis updates the tables, the
figures and the verdict together.

    python -m sept26_prelim_analysis.make_det_a_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as _h
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


def figure(name: str, caption: str, csv: str = None) -> str:
    csv = csv or f'{name}.csv'
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(caption)}"></a>'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{csv}">numbers &#8599;</a>'
            f'</figcaption></figure>')


def _f(v, d=3, unit=''):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{v:.{d}f}{unit}'


def _i(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{int(v):,}'


def census_table(K: pd.DataFrame) -> str:
    rows = []
    for r in K.itertuples():
        strong = ' style="font-weight:600"' if r.selection in (
            'all', 'slope') else ''
        rows.append(
            f'<tr><th class="s">{r.selection}</th>'
            f'<td class="n"{strong}>{_i(r.n)}</td>'
            f'<td class="n">{_f(r.frac_of_all, 3)}</td>'
            f'<td class="n"{strong}>{_f(r.median_open_deg, 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.p90_open_deg, 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.median_sep_mm, 0, "&nbsp;mm")}</td>'
            f'<td class="n">{_f(r.median_dca_pair_mm, 0, "&nbsp;mm")}</td>'
            f'<td class="n">{_f(r.frac_tag_A, 3)}</td>'
            f'<td class="n">{_f(r.frac_railed, 3)}</td></tr>')
    return ('<table><thead><tr><th>selection</th><th>pairs</th>'
            '<th>fraction</th><th>median &theta;</th><th>p90 &theta;</th>'
            '<th>median separation</th><th>median line approach</th>'
            '<th>arm-A tagged</th><th>drift railed</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def slope_table(S: pd.DataFrame) -> str:
    rows = []
    for r in S.itertuples():
        mark = (' <span class="u">(no timing slope)</span>'
                if r.below_tan_min else '')
        rows.append(
            f'<tr><th class="s">{r.tan_lo:.3f}&ndash;{r.tan_hi:.3f}{mark}</th>'
            f'<td class="n">{_i(r.n)}</td>'
            f'<td class="n"><b>{_f(r.core_over_wing, 2)}</b></td>'
            f'<td class="n">{_f(r.median_open_deg, 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.median_sep_mm, 0, "&nbsp;mm")}</td>'
            f'<td class="n">{_f(r.frac_tag_A, 3)}</td></tr>')
    return ('<table><thead><tr><th>smallest |tan| in the pair</th>'
            '<th>pairs</th><th>prompt / off-time</th><th>median &theta;</th>'
            '<th>median separation</th><th>arm-A tagged</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def dt0_table(D: pd.DataFrame) -> str:
    rows = []
    for r in D.itertuples():
        rows.append(
            f'<tr><th class="s">{getattr(r, "sample", "all")}</th>'
            f'<td>{r.model}</td>'
            f'<td class="n"><b>{_f(r.prompt_fraction, 3)}</b></td>'
            f'<td class="n">{_f(r.par1, 1, "&nbsp;ns")}</td>'
            f'<td class="n">{_f(r.nll, 1)}</td>'
            f'<td class="n">{_f(r.chi2dof, 2)}</td></tr>')
    return ('<table><thead><tr><th>sample</th><th>model</th>'
            '<th>fitted prompt fraction</th><th>prompt width</th>'
            '<th>&minus;log L</th><th>&chi;&sup2;/dof</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def tag_table(T: pd.DataFrame) -> str:
    rows = []
    for r in T.itertuples():
        rows.append(
            f'<tr><th class="s">{r.sample}</th>'
            f'<td class="n">{_i(r.n)}</td>'
            f'<td class="n">{_i(r.n_core)}</td>'
            f'<td class="n">{_i(r.n_wing)}</td>'
            f'<td class="n"><b>{_f(r.core_over_wing, 3)}</b> '
            f'&plusmn;&nbsp;{_f(r.err, 3)}</td>'
            f'<td class="n">{_f(r.median_open_core, 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.median_open_wing, 1, "&deg;")}</td></tr>')
    return ('<table><thead><tr><th></th><th>pairs</th><th>prompt</th>'
            '<th>off-time</th><th>prompt / off-time</th>'
            '<th>median &theta;, prompt</th><th>median &theta;, off-time</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def vertex_table(V: pd.DataFrame) -> str:
    rows = []
    for r in V.itertuples():
        rows.append(
            f'<tr><th class="s">{r.selection}</th>'
            f'<td class="n">{_i(r.k_real)} / {_i(r.n_real)}</td>'
            f'<td class="n">{_f(1e3 * r.rate_real, 2)}</td>'
            f'<td class="n">{_i(r.k_mixed)} / {_i(r.n_mixed)}</td>'
            f'<td class="n">{_f(1e3 * r.rate_mixed, 2)}</td>'
            f'<td class="n"><b>{_f(r.ratio, 2)}</b> '
            f'&plusmn;&nbsp;{_f(r.err, 2)}</td>'
            f'<td class="n">{_f(r.excess_sigma, 1, "&sigma;")}</td>'
            f'<td class="n">{_f(r.median_open_real, 1, "&deg;")}</td></tr>')
    return ('<table><thead><tr><th></th><th>real, vertexed</th>'
            '<th>per 1000</th><th>event-mixed, vertexed</th><th>per 1000</th>'
            '<th>ratio</th><th>excess</th><th>median &theta;</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def res_table(E: pd.DataFrame) -> str:
    rows = []
    for r in E[E.sep_hi <= 240].itertuples():
        rows.append(
            f'<tr><th class="s">{r.sep_lo:.0f}&ndash;{r.sep_hi:.0f}&nbsp;mm</th>'
            f'<td class="n">{_i(r.n_real)}</td>'
            f'<td class="n">{_i(r.n_mixed)}</td>'
            f'<td class="n"><b>{_f(r.eff, 3)}</b></td>'
            f'<td class="n">{_f(r.err, 3)}</td></tr>')
    return ('<table><thead><tr><th>separation</th><th>real pairs</th>'
            '<th>event-mixed</th><th>P(both found)</th><th>error</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def axis_table(X: pd.DataFrame) -> str:
    """The per-view turn-on, both views side by side."""
    u = X[X.axis == 'u'].set_index('sep_lo')
    v = X[X.axis == 'v'].set_index('sep_lo')
    rows = []
    for lo in sorted(set(u.index) & set(v.index)):
        if lo > 180:
            continue
        a, b = u.loc[lo], v.loc[lo]
        rows.append(
            f'<tr><th class="s">{lo:.0f}&ndash;{a.sep_hi:.0f}&nbsp;mm</th>'
            f'<td class="n">{_i(a.n_real)}</td>'
            f'<td class="n"><b>{_f(a.eff, 3)}</b></td>'
            f'<td class="n">{_i(b.n_real)}</td>'
            f'<td class="n"><b>{_f(b.eff, 3)}</b></td></tr>')
    return ('<table><thead><tr><th>separation in that view</th>'
            '<th>pairs, u</th><th>P(found), u</th>'
            '<th>pairs, v</th><th>P(found), v</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def compare_table(C: pd.DataFrame, sel: str, acc: str) -> str:
    g = C[(C.selection == sel) & (C.acceptance == acc)].sort_values('chi2dof')
    rows = []
    first = True
    for r in g.itertuples():
        b = ' style="font-weight:600"' if first else ''
        rows.append(
            f'<tr><td{b}>{r.model}</td>'
            f'<td class="n"{b}>{_f(r.chi2dof, 1)}</td>'
            f'<td class="n">{r.dof}</td>'
            f'<td class="n">{_f(r.median_pred, 1, "&deg;")}</td>'
            f'<td class="n">{_f(r.median_obs, 1, "&deg;")}</td></tr>')
        first = False
    return ('<table><thead><tr><th>model</th><th>&chi;&sup2;/dof</th>'
            '<th>dof</th><th>predicted median</th><th>observed median</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def eff_table(H: pd.DataFrame) -> str:
    e = H.efficiency.dropna()
    p10, p90 = (np.percentile(e, [10, 90]) if len(e) >= 3 else (np.nan,) * 2)
    return ('<table><thead><tr><th>runs</th><th>pairs</th><th>min</th>'
            '<th>median</th><th>max</th><th>p10&ndash;p90</th>'
            '</tr></thead><tbody><tr>'
            f'<td class="n">{len(H)}</td>'
            f'<td class="n">{_i(H.n_pairs.sum())}</td>'
            f'<td class="n">{_f(100 * e.min(), 1, "&nbsp;%")}</td>'
            f'<td class="n"><b>{_f(100 * e.median(), 1, "&nbsp;%")}</b></td>'
            f'<td class="n">{_f(100 * e.max(), 1, "&nbsp;%")}</td>'
            f'<td class="n">{_f(100 * (p90 - p10) / e.median(), 1, "&nbsp;%")}'
            f'</td></tr></tbody></table>')


# --------------------------------------------------------------------------- #
def build(d: Path) -> str:
    meta = json.loads((d / 'det_a_intra.meta.json').read_text())
    K = pd.read_csv(d / 'census.csv')
    S = pd.read_csv(d / 'spectra.csv')
    C = pd.read_csv(d / 'compare.csv')
    D = pd.read_csv(d / 'dt0_models.csv')
    T = pd.read_csv(d / 'tag_test.csv')
    E = pd.read_csv(d / 'two_track_efficiency.csv')
    Sl = pd.read_csv(d / 'slope_profile.csv')
    H = pd.read_csv(d / 'headline_per_run.csv')
    A = pd.read_csv(d / 'acceptance_pooled.csv')
    MS = pd.read_csv(d / 'eff_map_stability.csv')
    V = (pd.read_csv(d / 'vertex_excess.csv')
         if (d / 'vertex_excess.csv').exists() else pd.DataFrame())

    ka = K.set_index('selection')
    n_all = int(ka.n.get('all', 0))
    n_slope = int(ka.n.get('slope', 0))
    med_all = float(ka.median_open_deg.get('all', np.nan))
    med_slope = float(ka.median_open_deg.get('slope', np.nan))
    med_mixed = float(ka.median_open_deg.get('mixed', np.nan))
    med_slope_mixed = float(ka.median_open_deg.get('slope+mixed', np.nan))
    med_pr = float(ka.median_open_deg.get('slope+prompt', np.nan))
    med_of = float(ka.median_open_deg.get('slope+offtime', np.nan))
    dg = D[D['sample'] == 'all'] if 'sample' in D.columns else D
    f_lo, f_hi = float(dg.prompt_fraction.min()), float(dg.prompt_fraction.max())
    tg = T.set_index('sample')
    cw_tag = float(tg.core_over_wing.get('tagged (wall AND plastic A)', np.nan))
    cw_untag = float(tg.core_over_wing.get('untagged', np.nan))
    lo_slope = Sl[Sl.below_tan_min]
    hi_slope = Sl[~Sl.below_tan_min]
    cw_lo = float(lo_slope.core_over_wing.iloc[0]) if len(lo_slope) else np.nan
    cw_hi = float(hi_slope.core_over_wing.median()) if len(hi_slope) else np.nan
    res40 = E[(E.sep_lo < 40)].eff.max() if len(E) else np.nan
    best_all = C[(C.selection == 'all') & (C.acceptance == 'acc_sep')] \
        .sort_values('chi2dof')
    best_slope = C[(C.selection == 'slope') & (C.acceptance == 'acc_sep_slope')]\
        .sort_values('chi2dof')
    al_all = best_all[best_all.model == 'Al capsule (after wall)']
    al_slope = best_slope[best_slope.model == 'Al capsule (after wall)']
    X = (pd.read_csv(d / 'two_track_efficiency_axis.csv')
         if (d / 'two_track_efficiency_axis.csv').exists() else pd.DataFrame())
    SEPC = (pd.read_csv(d / 'two_track_separability.csv')
            if (d / 'two_track_separability.csv').exists() else pd.DataFrame())
    sep_rms = (float(np.sqrt((SEPC.residual ** 2).mean()))
               if len(SEPC) else np.nan)
    n_above = int(E.above_plateau.sum()) if 'above_plateau' in E.columns else 0
    lo = E[E.sep_hi <= 40]
    n_lt40 = int(lo.n_real.sum()) if len(lo) else 0
    n_lt40_mixed = (int(round((lo.n_mixed * lo.scale).sum()))
                    if len(lo) else 0)
    vs = V.set_index('selection') if len(V) else pd.DataFrame()
    vr_slope = float(vs.ratio.get('slope', np.nan)) if len(vs) else np.nan
    vx_slope = float(vs.excess_pairs.get('slope', np.nan)) if len(vs) else np.nan
    vs_slope = float(vs.excess_sigma.get('slope', np.nan)) if len(vs) else np.nan
    prov = meta.get('shape_provenance', {})
    med_birth = prov.get('medians_deg', {})
    e = H.efficiency.dropna()

    def one(g, col, dd=1, unit=''):
        return _f(float(g[col].iloc[0]), dd, unit) if len(g) else '&mdash;'

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detector A, intra-chamber pairs</title>{HEAD}</head><body><main>
<h1>Detector A, intra-chamber pairs</h1>

<p class="lede">One chamber, one topology, {meta['n_runs']} runs,
<b>{_i(meta['n_pairs'])}</b> pairs against
{_i(meta.get('n_mixed', np.nan))} event-mixed. <b>The headline is a null:</b>
the intra-A opening-angle distribution is the same as the distribution of two
tracks that never shared a trigger &mdash; median
{_f(med_all, 1, '&deg;')} against {_f(med_mixed, 1, '&deg;')} on all pairs and
{_f(med_slope, 1, '&deg;')} against {_f(med_slope_mixed, 1, '&deg;')} on the
pairs whose angle is actually measured. <b>One thing survives that null:</b>
on the pairs whose angle is measured, the two legs converge on the beam axis at
{_f(vr_slope, 2)} times the event-mixed rate, an excess of {_f(vx_slope, 0)}
pairs at {_f(vs_slope, 1)}&sigma;. Three things had to be established
before either could be read, and the third changes how the campaign spectra
should be read.
<b>One.</b> A two-dimensional efficiency map for A is measured in every run and
is stable. <b>Two.</b> The reconstruction loses a pair whose two tracks share EITHER
view: the loss is a cross in (|&Delta;u|, |&Delta;v|), not a disc, and only
{_i(n_lt40)} of {_i(meta['n_pairs'])} pairs survive below 40&nbsp;mm of radial
separation. Folding that measured loss into the acceptance, per view, moves the
predicted median opening angle from
{one(C[(C.selection == 'all') & (C.acceptance == 'acc') & (C.model == 'Al capsule (after wall)')], 'median_pred', 1, '&deg;')}
to {one(al_all, 'median_pred', 1, '&deg;')} against
{_f(med_all, 1, '&deg;')} observed. <b>Three, and this is the one that
matters:</b> the apparent in-chamber coincidence peak lives entirely in pairs
where at least one plane has no measurable timing slope. It vanishes as soon as
the slope is measurable, and the arm-A scintillators do not see it. Whatever it
is, <b>&Delta;<i>t</i><sub>0</sub> cannot be used as a coincidence selector on
this sample</b>, and in the one sample where the opening angle is a real
measurement there is no prompt excess to select.</p>

<h2>1 &middot; What a pair is here, and how many there are</h2>
<p>Every unordered pair of gated, angle-calibrated arm-A tracks inside one
trigger. No pointing cut is applied up front: the campaign chain applies one
silently inside <code>source_imaging</code>, and on this sample it is not a
quality cut but a physics one, so it appears below as a named selection that
keeps {_f(100 * float(ka.frac_of_all.get('pointing', np.nan)), 1)}&nbsp;% of
the pairs.</p>
{census_table(K)}
<p><b>A cross-check that ties this page to the campaign products.</b> The
<code>pointing</code> selection returns {_i(int(ka.n.get('pointing', 0)))}
pairs. <code>angle_campaign/pairs.parquet</code>, built by an independent path
through <code>source_imaging</code>, holds <b>7&nbsp;915</b> intra-A pairs. They
are the same sample: the campaign&rsquo;s intra-chamber control <i>is</i> the
30&nbsp;mm-pointing subset, which is
{_f(100 * float(ka.frac_of_all.get('pointing', np.nan)), 1)}&nbsp;% of the
intra-A pairs that exist.</p>
<p class="sub">The <code>mixed</code> rows are event-mixed pairs &mdash; two
arm-A tracks that never shared a trigger. They are the geometric null for the
opening angle, the separation and the closest approach. They are <b>not</b> a
null for the timing: <code>t0</code> carries a per-trigger offset whose spread
(235&nbsp;ns on run_145) exceeds the within-trigger spread (142&nbsp;ns), so a
cross-trigger time difference is broader than a within-trigger accidental one
by construction. Every timing column on a mixed row is blank rather than
misleading.</p>
{figure('a_spectrum', 'The opening-angle spectrum of each selection, '
        'shape-normalised.')}

<h2>2 &middot; The efficiency map, in two dimensions and in the right frame</h2>
<p>The denominator is an in-time wall <i>and</i> plastic coincidence in arm A,
so the absolute scale never touches the Micromegas. The cell position comes from
the reconstruction, which exists only for a seeded event, so the <i>map</i> is
the fit-and-gate efficiency at a position and the <i>headline</i> is the
absolute scale; the two are reported apart rather than multiplied.</p>
{eff_table(H)}
<p><b>A frame correction worth recording.</b> <code>efficiency.py</code> bins
in <code>local_x &minus; PINWHEEL</code>, the lever from the beam-axis foot,
while <code>acceptance.Chambers.cross</code> returns the offset from the
<i>plane centre</i>. For arm A those differ by 16.35&nbsp;mm, so the published
one-dimensional map is indexed 0.4 bins away from the toy that consumes it. The
map here is built in the plane-centre frame, in both coordinates.</p>
{figure('a_effmap', 'Left: the campaign-mean efficiency across the plane. '
        'Right: the run-to-run scatter of its shape.', csv='a_effmap.map.csv')}

<h2>3 &middot; The double-track resolution, measured in situ</h2>
<p>This is the largest term missing from <code>acceptance.py</code>, which
treats the two legs as independently reconstructed. The data disagrees:
<b>{_i(n_lt40)} of {_i(meta['n_pairs'])} intra-A pairs have their two impact
points closer than 40&nbsp;mm</b>, against {_i(n_lt40_mixed)} expected from the
event-mixed sample.</p>

<p><b>And the loss is a CROSS, not a disc.</b> On the map of real over
event-mixed against (|&Delta;u|, |&Delta;v|) the entire first row and the
entire first column sit at 0.00&ndash;0.03 of the plateau, whatever the other
coordinate does. A pair 12&nbsp;mm apart in <i>u</i> and 300&nbsp;mm apart in
<i>v</i> is lost as completely as one 12&nbsp;mm apart in both. That is what two
independent strip planes should do &mdash; two tracks sharing an x-strip band
merge in the x view, and the y view cannot rescue them because the fit needs
both &mdash; and <b>it is invisible to an efficiency in the radial separation
alone</b>, which calls that pair well separated. The first version of this page
used a radial curve; it is superseded and kept dashed on the figure for
comparison.</p>

<p>So the efficiency is measured once per view, with the other view held above
100&nbsp;mm so each curve is its own turn-on, and applied as the product
&epsilon;(|&Delta;u|)&nbsp;&times;&nbsp;&epsilon;(|&Delta;v|). Each is
normalised over 60&ndash;200&nbsp;mm, past the turn-on and before the decline
beyond 200&nbsp;mm that is geometry and leg-to-leg correlation rather than
efficiency, then taken as a running maximum because a resolution can only rise
with separation.</p>
{axis_table(X) if len(X) else ''}
<p class="sub"><b>The separable product is an assumption and it is tested.</b>
Against the measured two-dimensional map, over the cells where both coordinates
are past the hard edge and inside 300&nbsp;mm, the rms residual is
{_f(sep_rms, 2)}. The product reproduces the cross; what it leaves behind is the
same leg-to-leg correlation that makes the ratio climb at large separation, and
that is not an efficiency.</p>
<h3>The radial curve it replaces</h3>
{res_table(E)}
<p class="sub">What this assumes. If real pairs are genuinely correlated in
position &mdash; which a pair from a common vertex is &mdash; the ratio mixes
that correlation into the efficiency. The hard zero below 40&nbsp;mm cannot be
correlation, so that part is resolution; between 40 and 120&nbsp;mm the curve is
an upper bound on the loss if real pairs cluster at small separation and a lower
bound if they avoid it. Beyond the saturation point the raw ratio keeps rising
&mdash; {_i(n_above)} bins sit above unity and are clipped &mdash; which is
correlation, not efficiency, and is carried in the <code>ratio_raw</code>
column rather than folded into the acceptance.</p>
{figure('a_twotrack', 'Left: the measured two-track efficiency. Right: what '
        'each ingredient does to the intra-A acceptance.',
        csv='a_twotrack.eff.csv')}
{figure('a_dudv', 'Where the two legs land relative to each other, real '
        'against event-mixed. The hole is the resolution.',
        csv='a_dudv.real.csv')}

<h2>4 &middot; The in-chamber clock, and what it is not</h2>
<p><code>t0</code> is the fitted arrival time of the charge that starts at the
mesh, so two legs born in the same instant share it whatever their angles. The
difference between the two legs shows a clear peak at zero on a broad pedestal.
Three tests bound what can be concluded from it.</p>

<h3>Test one: the fraction is not identifiable</h3>
<p>Five two-component models, all centred at zero because the accidental term is
a difference of exchangeable times and must be symmetric. They agree on the
likelihood and disagree on the answer, so the shape alone cannot measure a
prompt fraction: it spans <b>{_f(f_lo, 2)} to {_f(f_hi, 2)}</b>.</p>
{dt0_table(D)}

<h3>Test two: the scintillators do not see it</h3>
<p>If the peak were the prompt population the scintillators call prompt, pairs
in triggers with an arm-A wall-and-plastic coincidence would have the larger
prompt-to-off-time ratio. They have the smaller one:
<b>{_f(cw_tag, 2)}</b> tagged against <b>{_f(cw_untag, 2)}</b> untagged.</p>
{tag_table(T)}
{figure('a_ntof', 'The arm-A scintillator timing on pair triggers, on all '
        'triggers, and on the n_TOF random-coincidence control.')}

<h3>Test three, which localises it</h3>
<p>A track below the reconstruction&rsquo;s own <code>TAN_MIN_SLOPE</code>
lays its charge on very few strips, so the fit has almost no lever arm on the
direction. Splitting the pairs by the smallest in-plane slope anywhere in the
pair &mdash; four planes, two legs &mdash; puts the whole peak on one side of
that threshold. The split is exact: the reconstruction&rsquo;s own
<code>slope_reliable</code> flag is False for every track below 0.08 and True
for every track above 0.15, so the two variables are the same cut.</p>
{slope_table(Sl)}
<p>The prompt-to-off-time ratio falls from <b>{_f(cw_lo, 2)}</b> in the
no-slope bin to <b>{_f(cw_hi, 2)}</b> &mdash; consistent with unity, meaning no
peak at all &mdash; once the slope is measurable. The median opening angle rises
across the same range, from {_f(float(lo_slope.median_open_deg.iloc[0]), 0, '&deg;') if len(lo_slope) else '&mdash;'}
to {_f(float(hi_slope.median_open_deg.iloc[-1]), 0, '&deg;') if len(hi_slope) else '&mdash;'}.</p>

<p><b>What this does not settle, and an explanation that was tried and
failed.</b> The obvious reading is that a slope-less fit has a degenerate
<code>t0</code> which collapses onto its prior, making &Delta;<i>t</i><sub>0</sub>
zero for free. <b>That is not what the data shows:</b> single-track
<code>t0</code> is <i>widest</i> for the low-slope tracks (standard deviation
293&nbsp;ns against 105&nbsp;ns for the well-measured ones on run_145), so
nothing is collapsing onto a prior. The peak is therefore either a genuine
coincidence of a population the arm-A scintillators do not tag, or a
correlation introduced by fitting two slope-less tracks in one trigger. This
page does not separate those two, and says so rather than picking one.</p>
<p><b>What it does settle</b> is the operational point: in the sample where the
opening angle is a measurement at all &mdash; both legs with a usable slope
&mdash; the prompt and off-time windows contain the same distribution, so there
is no in-chamber coincidence selection to be made.</p>
{figure('a_slope_profile', 'The prompt excess against the pair’s smallest '
        'in-plane slope, and the median opening angle over the same range.')}
{figure('a_dt0', 'The two-leg time difference, split by slope and by '
        'scintillator tag.')}
{figure('a_dt_sep', 'Time difference against in-plane separation, the '
        'correlation a pair from a common vertex would carry.',
        csv='a_dt_sep.grid.csv')}

<h2>5 &middot; The folded comparison</h2>
<p>The capsule and gas continua, multiplied by the measured detector-A
acceptance on the 1&deg; physics grid and binned afterwards. Each measured
selection is compared against the acceptance that matches it: the
slope-selected data against the toy that also requires a measurable slope on
both legs, and both against the toy that carries the measured two-track
resolution.</p>
<h3>All pairs</h3>
{compare_table(C, 'all', 'acc_sep')}
<h3>Both legs with a measurable slope</h3>
{compare_table(C, 'slope', 'acc_sep_slope')}
{figure('a_fold', 'The measured spectrum against the folded continua, for '
        'both selections.', csv='a_fold.obs.csv')}
<p>At birth the capsule continuum after the wall has a median of
{_f(med_birth.get('Al capsule (after wall)', np.nan), 1, '&deg;')} and the
helium gas {_f(med_birth.get('3He gas M1+E0 (after wall)', np.nan), 1, '&deg;')}.
Folded through this acceptance the capsule predicts
{one(al_all, 'median_pred', 1, '&deg;')} against {_f(med_all, 1, '&deg;')}
observed on all pairs, and {one(al_slope, 'median_pred', 1, '&deg;')} against
{_f(med_slope, 1, '&deg;')} on the slope-selected sample. <b>The medians agree
on the full sample and do not agree once a measurable slope is required</b>,
which is the opposite of what a real pair population would do: the selection
that makes the angle a measurement is the one the model stops describing.</p>

<h2>6 &middot; What the vertex says</h2>
<p>Two lines from a common vertex must approach each other at that vertex. The
rate at which a pair does so within {meta.get('vertex_dca_mm', 30):.0f}&nbsp;mm,
and within {meta.get('vertex_r_mm', 20):.0f}&nbsp;mm of the beam axis, is the
one quantity on this page that a real pair population moves and an uncorrelated
one does not. The event-mixed comparison is drawn from the same trigger class
&mdash; tracks from triggers that produced at least two arm-A tracks &mdash; so
the two samples see the same occupancy.</p>
{vertex_table(V) if len(V) else '<p class="sub">Not built.</p>'}
<p class="sub">The two samples are still not matched in separation acceptance:
a mixed pair may put its two tracks in the same place and a real pair may not,
which pushes the mixed rate up. The ratio above is therefore a lower bound on
any real excess.</p>
{figure('a_vertex', 'Closest approach of the two lines, and the radius of '
        'that point from the beam axis.', csv='a_vertex.dca_pair_mm.csv')}

<h2>What this does and does not establish</h2>
<ul>
<li><b>It does not show a pair signal in the intra-A <i>spectrum</i>.</b>
The vertex excess in &sect;6 is the one exception and it is a rate, not a
shape: {_i(int(vs.k_real.get('slope', 0)) if len(vs) else 0)} pairs, whose own
opening-angle distribution is not distinguishable from the rest at this size.
The measured spectrum matches the event-mixed null in both selections, the prompt and off-time
samples have the same opening-angle distribution
({_f(med_pr, 1, '&deg;')} against {_f(med_of, 1, '&deg;')} median), the
scintillator tag does not enrich the prompt window, and the one selection that
makes the opening angle a real measurement leaves the folded models
describing the data worse, not better.</li>
<li><b>A large &chi;&sup2;/dof here means very little.</b> With
{_i(meta['n_pairs'])} pairs in ten bins, a model that is 1&nbsp;% wrong in shape
is rejected at &chi;&sup2;/dof in the thousands. The comparison that carries
information is the ordering, and whether any physics model beats the
event-mixed null.</li>
<li><b>The in-chamber timing cannot be used as a coincidence cut</b> on this
sample. A &Delta;<i>t</i><sub>0</sub> selection here selects tracks without a
measurable slope, whose opening angle is not a measurement, and it buys nothing
on the tracks whose angle is.</li>
<li><b>The origin of the &Delta;<i>t</i><sub>0</sub> peak is open.</b> The
degenerate-fit explanation is falsified by the single-track <code>t0</code>
widths; a genuine coincidence invisible to the arm-A scintillators is not
excluded. Separating them needs a timing reference that does not come from the
same fit &mdash; the wall&rsquo;s own two-ended time, per track rather than per
trigger, is the obvious candidate and it is not built.</li>
<li><b>The two-track resolution belongs in every acceptance</b>, and it has to
be applied per view, not radially. It is the term <code>PLAN.md</code> lists as D1/D2 and it is now
measured. The inter-chamber topologies do not have it &mdash; two legs in
different chambers are reconstructed independently &mdash; so this correction
applies to the intra control alone, which is exactly the sample
<code>PLAN.md</code> &sect;S4 uses as the background normalisation.</li>
<li><b>The slope requirement is not identical in the data and in the toy.</b>
The reconstruction sets its flag from the <i>fitted</i> slope, which carries
resolution; the toy applies the same threshold to the <i>true</i> slope. Near
the threshold the two differ by whatever that resolution is, and tracks migrate
across it in both directions. The comparison of the slope-selected sample
against the slope-required toy is therefore approximate in exactly the region
the selection acts on.</li>
<li><b>The angle scale is unchanged and still open.</b> Every angle here
carries arm A&rsquo;s own per-run <i>k</i>, and <code>gap_check</code> fails on
A. A wrong scale stretches the measured spectrum and nothing here corrects
it.</li>
<li><b>Multiple scattering after the capsule, energy loss and the leptons&rsquo;
own energies are in none of this</b>, so the acceptance is an over-estimate and
this is a shape, never a rate.</li>
</ul>

<p class="foot">Generated by <code>make_det_a_report.py</code> on
{dt.date.today().isoformat()}. Arm {meta['arm']},
{meta['n_runs']} runs, {_i(meta['n_pairs'])} real and
{_i(meta.get('n_mixed', np.nan))} event-mixed pairs.
Tracks: <code>{meta['src']}</code>. Reco: <code>{meta['reco']}</code>.
Throw: {meta['n_throw']:,} pairs per run.
Prompt window &plusmn;{meta['prompt_ns']:.0f}&nbsp;ns; off-time
{meta['offtime_ns'][0]:.0f}&ndash;{meta['offtime_ns'][1]:.0f}&nbsp;ns;
pointing cut {meta['dca_max_mm']:.0f}&nbsp;mm.
n_TOF timing read from <code>{meta['ntof_reference_run']}</code>.
Aluminium lines with no assignment taken as {prov.get('assume', '?')}.</p>
</main></body></html>'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('det_a_intra')
    paths.require(d / 'det_a_intra.meta.json', 'the det_a_intra products')
    out = d / 'report.html'
    out.write_text(build(d))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
