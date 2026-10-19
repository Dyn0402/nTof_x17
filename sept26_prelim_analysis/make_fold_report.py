#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fold_report.py -- ``report.html`` for the per-run acceptance and the fold.

Answers, in order: is the borrowed acceptance the reason nothing fits (no), is
the efficiency applied in the right variable (no), and does the aluminium
capsule continuum describe the measured opening angles (better than the gas,
not well enough).

Generated, never hand-written: every number is read back from what
`campaign_efficiency.py`, `campaign_acceptance.py` and `campaign_fold.py`
wrote, so re-running the chain updates the tables, the figures and the verdict
together.

    python -m sept26_prelim_analysis.make_fold_report
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
from sept26_prelim_analysis.campaign_angle import TOPOLOGIES, X17_MIN_DEG  # noqa: E402

TOPO_COLOR = {'intra': '#0072B2', 'perpendicular': '#E69F00',
              'opposing': '#009E73'}


def figure(name: str, caption: str, csv: str = None) -> str:
    """A figure with ordinary relative links (CLAUDE.md), and its numbers."""
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


# --------------------------------------------------------------------------- #
def stability_table(S: pd.DataFrame, H: pd.DataFrame) -> str:
    r145 = H[H.run == 'run_145'].set_index('arm').efficiency
    rows = []
    for r in S.itertuples():
        d = (100 * (r145[r.arm] / r.median - 1)
             if r.arm in r145.index and r.median else np.nan)
        rows.append(
            f'<tr><th class="s">{r.arm}</th>'
            f'<td class="n">{r.n_runs}</td>'
            f'<td class="n">{_f(100 * r.min, 1, "&nbsp;%")}</td>'
            f'<td class="n"><b>{_f(100 * r.median, 1, "&nbsp;%")}</b></td>'
            f'<td class="n">{_f(100 * r.max, 1, "&nbsp;%")}</td>'
            f'<td class="n">{_f(100 * r.p10_p90_frac, 1, "&nbsp;%")}</td>'
            f'<td class="n">{_f(100 * r.block_shift_frac, 1, "&nbsp;%")}</td>'
            f'<td class="n">{_f(d, 1, "&nbsp;%")}</td></tr>')
    return ('<table><thead><tr><th></th><th>runs</th><th>min</th>'
            '<th>median</th><th>max</th><th>p10&ndash;p90</th>'
            '<th>128&ndash;147 shift</th><th>run_145 vs median</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def dip_table(D: pd.DataFrame) -> str:
    rows = []
    for arm, g in D.groupby('arm'):
        rows.append(
            f'<tr><th class="s">{arm}</th>'
            f'<td class="n">{g.run.nunique()}</td>'
            f'<td class="n"><b>{g.track_ratio.median():.3f}</b></td>'
            f'<td class="n">{g.track_ratio.min():.3f}</td>'
            f'<td class="n">{g.track_ratio.max():.3f}</td>'
            f'<td class="n">{g.seed_ratio.median():.3f}</td>'
            f'<td class="n">{100 * g.below_both.mean():.0f}&nbsp;%</td></tr>')
    return ('<table><thead><tr><th></th><th>runs</th>'
            '<th>track ratio, median</th><th>min</th><th>max</th>'
            '<th>seed ratio</th><th>below both neighbours</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def borrow_table(B: pd.DataFrame) -> str:
    rows = []
    for r in B.itertuples():
        col = TOPO_COLOR.get(r.group, '#333333')
        rows.append(
            f'<tr><th class="s" style="color:{col}">{r.group}</th>'
            f'<td class="n">{r.integral_ratio:.3f}</td>'
            f'<td class="n">{r.rms_shape_dev:.3f}</td>'
            f'<td class="n">{r.max_shape_dev:.3f}</td>'
            f'<td class="n">{r.frac109_ref:.4f}</td>'
            f'<td class="n">{r.frac109_campaign:.4f}</td></tr>')
    return ('<table><thead><tr><th></th><th>integral, r145 / campaign</th>'
            '<th>rms shape deviation</th><th>max</th>'
            '<th>&gt;109&deg;, run_145</th><th>&gt;109&deg;, campaign</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def variant_table(V: pd.DataFrame) -> str:
    rows = []
    for group, g in V[V.group.isin(TOPOLOGIES)].groupby('group'):
        col = TOPO_COLOR.get(group, '#333333')
        first = True
        for r in g.sort_values('variant').itertuples():
            head = (f'<th class="s" rowspan="{len(g)}" '
                    f'style="color:{col}">{group}</th>' if first else '')
            rows.append(
                f'<tr>{head}<td>{r.variant}</td>'
                f'<td class="n">{r.integral_ratio:.3f}</td>'
                f'<td class="n">{r.median_deg:.1f}&deg;</td>'
                f'<td class="n">{r.median_shift_deg:+.1f}&deg;</td>'
                f'<td class="n">{r.frac_above_109:.4f}</td>'
                f'<td class="n">{_f(r.frac109_ratio, 3)}</td></tr>')
            first = False
    return ('<table><thead><tr><th></th><th>variant</th>'
            '<th>integral / u_map</th><th>median</th><th>&Delta; median</th>'
            '<th>&gt;109&deg;</th><th>ratio to u_map</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def distortion_table(D: pd.DataFrame) -> str:
    rows = []
    for model, g in D[D.topology == 'all'].groupby('model'):
        r = g.iloc[0]
        rows.append(
            f'<tr><th class="s">{model}</th>'
            f'<td class="n">{r.vertex}</td>'
            f'<td class="n">{r.birth_median_deg:.1f}&deg;</td>'
            f'<td class="n">{r.folded_median_deg:.1f}&deg;</td>'
            f'<td class="n">{r.median_shift_deg:+.1f}&deg;</td>'
            f'<td class="n">{r.birth_frac109:.4f}</td>'
            f'<td class="n">{r.folded_frac109:.4f}</td>'
            f'<td class="n">{_f(r.frac109_ratio, 1, "&times;")}</td></tr>')
    return ('<table><thead><tr><th></th><th>vertex</th><th>median at birth</th>'
            '<th>folded</th><th>shift</th><th>&gt;109&deg; at birth</th>'
            '<th>folded</th><th>ratio</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def corrected_table(U: pd.DataFrame) -> str:
    h = U[U.theta.isna()] if 'median_deg' in U.columns else U.head(0)
    rows = []
    for r in h.itertuples():
        col = TOPO_COLOR.get(r.topology, '#333333')
        rows.append(
            f'<tr><th class="s">{r.selection}</th>'
            f'<td style="color:{col}">{r.topology}</td>'
            f'<td class="n">{_i(r.n_obs)}</td>'
            f'<td class="n"><b>{_f(r.median_deg, 1, "&deg;")}</b></td>'
            f'<td class="n">{_f(r.frac_above_109, 4)}</td>'
            f'<td class="n">{_f(100 * r.frac_dropped, 1, "&nbsp;%")}</td>'
            f'</tr>')
    return ('<table><thead><tr><th>selection</th><th>topology</th><th>pairs</th>'
            '<th>implied median at birth</th><th>&gt;109&deg;</th>'
            '<th>counts dropped</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def agreement_table(G: pd.DataFrame, selection: str) -> str:
    rows = []
    for topo in TOPOLOGIES:
        h = G[(G.selection == selection) & (G.topology == topo)] \
            .sort_values('chi2dof_sub')
        if h.empty:
            continue
        first = True
        for r in h.itertuples():
            head = (f'<th class="s" rowspan="{len(h)}" '
                    f'style="color:{TOPO_COLOR[topo]}">{topo}</th>'
                    if first else '')
            b = ' style="font-weight:600"' if first else ''
            rows.append(
                f'<tr>{head}<td{b}>{r.model}</td>'
                f'<td class="n"{b}>{r.chi2dof_sub:.1f}</td>'
                f'<td class="n">{r.dof_sub}</td>'
                f'<td class="n">{_f(r.ratio_above, 2, "&times;")}</td>'
                f'<td class="n">{_f(r.excess_sigma, 1, "&sigma;")}</td></tr>')
            first = False
    return ('<table><thead><tr><th></th><th>birth model</th>'
            '<th>&chi;&sup2;/dof below 109&deg;</th><th>dof</th>'
            '<th>data / model above 109&deg;</th><th>excess</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def ratio_table(R: pd.DataFrame) -> str:
    """Observed fraction above 109 deg beside every folded model's."""
    cols = [c for c in R.columns if c.startswith('frac[')]
    # nearest the observation first: the reader wants the ordering, not the
    # DataFrame's column order
    ref = R[R.topology == 'perpendicular']
    key = (lambda c: abs(float(ref[c].iloc[0]) - float(ref.frac_obs.iloc[0]))
           if len(ref) else lambda c: c)
    order = sorted(cols, key=key)
    head = ''.join(f'<th>{c[5:-1]}</th>' for c in order)
    rows = []
    for _, r in R.iterrows():
        col = TOPO_COLOR.get(r.topology, '#333333')
        cells = ''.join(f'<td class="n">{_f(float(r[c]), 3)}</td>'
                        for c in order)
        rows.append(
            f'<tr><th class="s">{r.selection}</th>'
            f'<td style="color:{col}">{r.topology}</td>'
            f'<td class="n">{_i(r.n)}</td>'
            f'<td class="n"><b>{_f(r.frac_obs, 3)}</b> '
            f'&plusmn;&nbsp;{_f(r.err, 3)}</td>{cells}</tr>')
    return ('<table><thead><tr><th>selection</th><th>topology</th><th>n</th>'
            f'<th>observed &gt;109&deg;</th>{head}</tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def edge_table(E: pd.DataFrame) -> str:
    rows = []
    for arm, g in E.groupby('arm'):
        rows.append(f'<tr><th class="s">{arm}</th>'
                    f'<td class="n">{g.u_edge_mm.iloc[0]:.0f}&nbsp;mm</td>'
                    f'<td class="n">{100 * g.frac_outside.mean():.1f}&nbsp;%</td>'
                    f'<td class="n">{100 * g.frac_outside.min():.1f}&nbsp;%</td>'
                    f'<td class="n">{100 * g.frac_outside.max():.1f}&nbsp;%</td>'
                    f'</tr>')
    return ('<table><thead><tr><th></th><th>map edge</th>'
            '<th>legs beyond it, mean</th><th>min</th><th>max</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def one_comp_table(C: pd.DataFrame, selection: str) -> str:
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
            b = ' style="font-weight:600"' if first else ''
            rows.append(
                f'<tr>{head}<td{b}>{r.model}</td>'
                f'<td class="n"{b}>{r.chi2dof:.1f}</td>'
                f'<td class="n">{r.dof}</td>'
                f'<td class="n">{r.frac_above_x17_pred:.3f}</td>'
                f'<td class="n">{r.frac_above_x17_obs:.3f}</td></tr>')
            first = False
    return ('<table><thead><tr><th></th><th>model</th><th>&chi;&sup2;/dof</th>'
            '<th>dof</th><th>predicted &gt;109&deg;</th>'
            '<th>observed &gt;109&deg;</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def two_comp_table(T: pd.DataFrame) -> str:
    rows = []
    for r in T.itertuples():
        # None/NaN means the topology has no independent timing measurement to
        # be consistent WITH -- not that it passed.  NaN is truthy, so this
        # cannot be a bare `if`.
        tc = r.timing_consistent
        ok = ('&mdash;' if not isinstance(tc, (bool, np.bool_))
              and (tc is None or (isinstance(tc, float) and not np.isfinite(tc)))
              else '<span style="color:#009E73">yes</span>' if bool(tc)
              else '<span style="color:#D55E00">NO</span>')
        rows.append(
            f'<tr><th class="s">{r.selection}</th>'
            f'<td style="color:{TOPO_COLOR.get(r.topology, "#333")}">'
            f'{r.topology}</td><td>{r.pair_model}</td>'
            f'<td class="n"><b>{r.f_acc:.2f}</b> '
            f'[{r.f_lo:.2f}, {r.f_hi:.2f}]</td>'
            f'<td class="n">{r.chi2dof:.2f}</td>'
            f'<td class="n">{r.chi2dof_pure_pair:.1f}</td>'
            f'<td class="n">{r.chi2dof_pure_acc:.1f}</td>'
            f'<td class="n">{_f(r.f_timing, 2)} '
            f'[{_f(r.f_timing_lo, 2)}, {_f(r.f_timing_hi, 2)}]</td>'
            f'<td class="n">{ok}</td></tr>')
    return ('<table><thead><tr><th>selection</th><th>topology</th>'
            '<th>pair shape</th><th>fitted accidental share</th>'
            '<th>&chi;&sup2;/dof</th><th>pure pair</th><th>pure accidental</th>'
            '<th>timing measurement</th><th>consistent</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


# --------------------------------------------------------------------------- #
def build(ed: Path, ad: Path, fd: Path) -> str:
    em = json.loads((ed / 'campaign_efficiency.meta.json').read_text())
    am = json.loads((ad / 'campaign_acceptance.meta.json').read_text())
    fm = json.loads((fd / 'campaign_fold.meta.json').read_text())
    H = pd.read_csv(ed / 'headline_per_run.csv')
    S = pd.read_csv(ed / 'stability.csv')
    Dip = pd.read_csv(ed / 'head_on_dip_per_run.csv')
    B = pd.read_csv(ad / 'borrow_test.csv')
    V = pd.read_csv(ad / 'variant_shift.csv')
    RS = pd.read_csv(ad / 'run_spread.csv')
    D = pd.read_csv(fd / 'distortion.csv')
    C = pd.read_csv(fd / 'one_component.csv')
    T = pd.read_csv(fd / 'two_component.csv')
    R = pd.read_csv(fd / 'ratio.csv')
    U = (pd.read_csv(fd / 'corrected.csv')
         if (fd / 'corrected.csv').exists() else pd.DataFrame())
    E = (pd.read_csv(ad / 'edge_cost.csv')
         if (ad / 'edge_cost.csv').exists() else pd.DataFrame())
    G = (pd.read_csv(fd / 'agreement.csv')
         if (fd / 'agreement.csv').exists() else pd.DataFrame())

    prov = fm.get('shape_provenance', {})
    unrel = prov.get('capsule_unreliable', float('nan'))
    esc = prov.get('capsule_escape', float('nan'))
    med = prov.get('medians_deg', {})
    f109 = prov.get('frac_above_109', {})
    acd = S.set_index('arm')
    dipm = Dip.groupby('arm').track_ratio.median()

    # the headline comparisons, read back rather than retyped
    tp = C[C.selection == 'tight_pair']
    opp = tp[tp.topology == 'opposing'].sort_values('chi2dof')
    al_opp = opp[opp.model == 'Al capsule (after wall)']
    gas_opp = opp[opp.model == '3He gas M1+E0']
    mix_opp = opp[opp.model == 'event-mixed (accidental shape)']
    perp = tp[tp.topology == 'perpendicular'].sort_values('chi2dof')
    al_perp = perp[perp.model == 'Al capsule (after wall)']
    gas_perp = perp[perp.model == '3He gas M1+E0']
    tt = T[(T.selection == 'tight_pair')
           & (T.pair_model == 'Al capsule (after wall)')]
    tt_perp = tt[tt.topology == 'perpendicular']
    tt_opp = tt[tt.topology == 'opposing']

    def one(df, col, d=1):
        return _f(float(df[col].iloc[0]), d) if len(df) else '&mdash;'

    sh_med = RS[RS.group.isin(TOPOLOGIES) & (RS.shape_mean > 1e-4)]
    _h_ = U[U.theta.isna()] if len(U) and 'median_deg' in U.columns \
        else pd.DataFrame(columns=['topology', 'selection', 'median_deg'])
    _ab = _h_[_h_.selection == 'all_no_b2b'].set_index('topology')
    med_int = float(_ab.median_deg.get('intra', np.nan))
    med_opp = float(_ab.median_deg.get('opposing', np.nan))
    edge_block = ('''<h3>One reason to expect that: the map has an edge</h3>
<p>The measured efficiency map spans |u| &le; 160&nbsp;mm because the
scintillators stop covering the plane beyond about 150&nbsp;mm, while the active
area runs to 190&nbsp;mm. In the <code>u_map</code> variant a leg landing in the
outer strip is given <b>zero</b> efficiency, which is an acceptance cut wearing
an efficiency&rsquo;s clothes. It is reproduced here rather than corrected,
because it is what every published number so far did.</p>
''' + edge_table(E)) if len(E) else ''
    corr_block = (corrected_table(U) if len(U)
                  else '<p class="sub">Not built.</p>')
    corr_fig = (figure('fold_corrected', 'The corrected spectrum per topology, '
                       'against the two birth continua.',
                       csv='fold_corrected.corrected.csv') if len(U) else '')
    gt = G[(G.selection == 'tight_pair') & (G.topology == 'perpendicular')]
    g_al = gt[gt.model == 'Al capsule (after wall)']
    g_gas = gt[gt.model == '3He gas M1+E0']
    agree_block = ('''<h3>And where it does agree, quantified</h3>
<p>The perpendicular topology is the only one whose acceptance does not decide
the answer, and its corrected spectrum <b>follows the capsule curve over the
sub-threshold range and rises above it past 109&deg;</b>. Both halves are
below, with the normalisation fitted <i>below</i> the threshold only &mdash; a
normalisation fitted over the whole range would let an excess above 109&deg;
pull the level beneath it and hide itself.</p>
''' + agreement_table(G, 'tight_pair')
        + '''<p class="sub">The intra rows are not usable and are shown for
completeness: intra loses ''' + _f(100 * float(
            U[(U.topology == 'intra') & U.theta.isna()].frac_dropped.iloc[0]
            if len(U[(U.topology == 'intra') & U.theta.isna()]) else np.nan),
            0, '&nbsp;%') + ''' of its counts to bins the acceptance
barely reaches, and it carries no timing cut at all.</p>''') if len(G) else ''

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-run acceptance and the aluminium fold</title>{HEAD}</head><body><main>
<h1>The acceptance, run by run &mdash; and the capsule folded through it</h1>

<p class="lede"><b>The borrowed acceptance was not the problem.</b> The tagged
efficiency is now measured in all {em['n_runs']} runs and its p10&ndash;p90
spread is {_f(100 * acd.p10_p90_frac.get('A', np.nan), 1)}&nbsp;% on&nbsp;A,
{_f(100 * acd.p10_p90_frac.get('C', np.nan), 1)}&nbsp;% on&nbsp;C and
{_f(100 * acd.p10_p90_frac.get('D', np.nan), 1)}&nbsp;% on&nbsp;D, with
run_145 within 1.5&nbsp;% of the campaign median on every angle arm; the
acceptance curve&rsquo;s own run-to-run shape scatter has a median coefficient
of variation of {_f(sh_med.shape_cv.median(), 3)}. <b>The
<i>head-on dip</i> is real</b> &mdash; a track arriving perpendicular to the
strips is reconstructed only
{_f(dipm.get('C', np.nan), 2)}&ndash;{_f(dipm.get('A', np.nan), 2)} times as
often as an oblique one, below both its positional neighbours in
{100 * Dip.below_both.mean():.0f}&nbsp;% of run&ndash;arm pairs &mdash; but
folding it in as an incidence factor barely moves A(&theta;). What <i>does</i>
move it is the treatment already in use: the <code>u_map</code> variant is the
outlier of the three, and the reason is an <b>edge artefact</b>, not physics.
<b>And the aluminium capsule continuum fits better than the helium gas
does</b> &mdash; &chi;&sup2;/dof {one(al_opp, 'chi2dof')} against
{one(gas_opp, 'chi2dof')} on the coincident opposing sample,
{one(al_perp, 'chi2dof')} against {one(gas_perp, 'chi2dof')} on perpendicular
&mdash; but the event-mixed accidental shape still beats both, and after
correction the three topologies do not recover the same spectrum.</p>

<h2>1 &middot; The efficiency, in every run</h2>
<p>The tag is an in-time wall <i>and</i> plastic coincidence in the same arm, so
the denominator never touches the Micromegas. Read campaign-wide off the
exported n_TOF slim rather than the ROOT slim, which exists for run_145 only on
this machine; validated to give the identical tag set on run_145 arm&nbsp;A.
The map that the acceptance shape uses is built from the <b>single-track</b>
events, because an event the chamber turned into two or three tracks has no one
in-plane position to bin.</p>
{stability_table(S, H)}
<p class="sub">B is a <b>hit</b> efficiency, not a track efficiency &mdash; no
field cage &mdash; and it is the only arm that moves: its spread is
{_f(100 * acd.p10_p90_frac.get('B', np.nan), 0)}&nbsp;%. B contributes no
angle, so nothing downstream depends on it.</p>
{figure('eff_per_run', 'Headline efficiency against run number. The shaded '
        'band is the 128–147 k excursion; the dotted line is run_145.')}
<p><b>This retires a standing suspicion.</b>
<code>campaign_angle.py</code> stamped its acceptance
<code>run_145 (BORROWED)</code> and called it the leading systematic behind a
set of fits that nothing described. On the efficiency scale, the borrowing was
worth about 1.5&nbsp;%. Whatever is wrong with the acceptance, it is not that
one run stood in for thirty-six.</p>

<h2>2 &middot; The head-on dip, which the toy never had &mdash; and which
turns out not to matter much</h2>
<p>A track that crosses the strip plane perpendicular delivers its charge to
every strip at the same time, so its timing carries no slope information and
the fit is more likely to fail. The four wall groups sit at four different
incidences and one of them lands inside the head-on band, which makes this
measurable with an abscissa that is entirely external to the chamber.</p>
{dip_table(Dip)}
<p class="sub">The confound is that the four groups also sit at four different
places on the chamber. The head-on group is an interior one with a neighbour on
each side: a surface effect would interpolate between them, an angle effect
sits below both. It sits below both in
{100 * Dip.below_both.mean():.0f}&nbsp;% of run&ndash;arm pairs.</p>
{figure('eff_incidence', 'Left: tracking rate against the wall group’s '
        'expected incidence, median over runs. Right: the head-on group over '
        'its two neighbours, run by run.', csv='eff_incidence.curve.csv')}

<h2>3 &middot; What that does to the acceptance &mdash; and the surprise</h2>
<p>The same measured efficiency, entered three ways into the same toy: uniform
(<code>flat</code>), as a shape in the in-plane position (<code>u_map</code>,
what every published number so far used), and as the measured response versus
incidence (<code>incidence</code>). <b>The last two must never be
multiplied</b> &mdash; the wall groups that supply the incidence abscissa are
themselves four bands of <code>u</code>, so applying both would count the dip
twice.</p>
<p><b>The result is not the one section&nbsp;2 sets up.</b>
<code>incidence</code> lands almost exactly on <code>flat</code>: the head-on
band is narrow in |tan| and a pair from a 20&nbsp;mm source samples a wide
range of it, so a factor of 0.7 in a narrow band averages away. It is
<code>u_map</code> that is the odd one out, and section&nbsp;7 says why.</p>
{variant_table(V)}
{figure('acc_variants', 'The three variants, shape-normalised, per topology.')}
<p>And the run-to-run picture, which is the other half of the borrowing
question: the normalised acceptance curve has a median coefficient of variation
of {_f(sh_med.shape_cv.median(), 3)} across the runs.</p>
{borrow_table(B)}
{figure('acc_runs', 'Every run’s acceptance in grey, the pair-weighted '
        'campaign mean in colour.', csv='acc_runs.per_run.csv')}
<p class="sub">The capsule components fold through a <b>wall</b> vertex
distribution &mdash; the skin of the He-3 polycone, where an aluminium pair is
actually born &mdash; and the gas components through the gas volume. The
difference is about 2&nbsp;% in the integral and nothing in the shape, which is
what a 10&nbsp;mm radius against a 235&nbsp;mm lever should give. It is applied
rather than argued about. Vertex models thrown:
{', '.join(f'<code>{v}</code>' for v in am.get('vertices', []))}.</p>

<h2>4 &middot; The capsule continuum, and what it looks like at birth</h2>
<p>The wide-angle pairs the capsule makes are <b>not</b> the 7.7&nbsp;MeV
primaries: they are the 2&ndash;4&nbsp;MeV E1 lines that feed
<sup>28</sup>Al&rsquo;s negative-parity levels, and a 3&nbsp;MeV pair is far
less collimated than a 7.7&nbsp;MeV one. Carbon in the fibre adds about a fifth
of the wide-angle yield, because both of its strong primaries are E1 and soft.
So the capsule background is a <i>low-energy</i> background, and it is wider
than the gas signal it sits under.</p>
<table><thead><tr><th></th><th>median at birth</th>
<th>fraction &gt;109&deg;</th></tr></thead><tbody>
{''.join(f'<tr><th class="s">{k}</th><td class="n">{_f(v, 1, "&deg;")}</td>'
         f'<td class="n">{_f(f109.get(k, np.nan), 4)}</td></tr>'
         for k, v in med.items())}
</tbody></table>
{figure('fold_shapes', 'The two continua before any apparatus, at birth '
        '(dotted) and after the capsule wall (solid).')}
<p class="sub">The after-wall curves carry escape ({_f(100 * esc, 1)}&nbsp;% of
the weight survives) and a Highland multiple-scattering Gaussian that is
<b>not trustworthy for {_f(100 * unrel, 0)}&nbsp;%</b> of the capsule weight,
where the scattering angle came out above one radian and a Gaussian is not a
description. Those curves are an indication, not a prediction.</p>

<h2>5 &middot; Folded, against the data</h2>
<p>The acceptance dominates the observed shape. Every model, whatever it looks
like at birth, is dragged into its topology&rsquo;s own band:</p>
{distortion_table(D)}
{figure('fold_distortion', 'Median at birth against median after the '
        'acceptance. The dashed line is no distortion.')}
<h3>One free normalisation, coincident sample</h3>
{one_comp_table(C, 'tight_pair')}
{figure('fold_models', 'The coincident sample against the folded models, per '
        'topology.', csv='fold_models.obs.csv')}
<h3>All pairs, back-to-back removed</h3>
{one_comp_table(C, 'all_no_b2b')}

<p><b>The aluminium wins over the helium, and loses to the accidentals.</b>
On the coincident opposing sample the capsule continuum after the wall gives
&chi;&sup2;/dof {one(al_opp, 'chi2dof')} against the gas&rsquo;s
{one(gas_opp, 'chi2dof')}; on perpendicular, {one(al_perp, 'chi2dof')} against
{one(gas_perp, 'chi2dof')}. That ordering is the right sign and it is what the
rate argument predicted &mdash; but the event-mixed accidental template is
still the best single description of the opposing sample, at
{one(mix_opp, 'chi2dof')}.</p>

<h2>6 &middot; The two-component fit, tested against a measurement that never
saw the angles</h2>
<p>A single-shape &chi;&sup2; asks the wrong question, because nobody claims
the sample is pure pairs. Floating the accidental share and profiling
&chi;&sup2; over it turns the comparison into a test: the arm-to-arm
scintillator timing already measured that share, without using the opening
angle at all.</p>
{two_comp_table(T)}
{figure('fold_two_comp', 'The accidental share the shape wants (circles) '
        'against the share the timing measured (squares).')}
<p><b>The shape wants more accidentals than the timing found, and on the
opposing topology it wants nothing else.</b> On the coincident opposing
sample the fit runs to the boundary,
f = {one(tt_opp, 'f_acc', 2)}, against the timing&rsquo;s 0.42&nbsp;[0.26,
0.58]. On perpendicular it lands at {one(tt_perp, 'f_acc', 2)} with
&chi;&sup2;/dof {one(tt_perp, 'chi2dof', 2)} &mdash; an acceptable fit at a
share that the timing excludes. Two independent handles on the same quantity
disagree, and the opening angle is the one with a model in it.</p>

<h2>7 &middot; The data divided by the acceptance</h2>
<p><code>PLAN.md</code> &sect;S4 asks for the corrected spectrum beside the raw
one. Folding the model forward &mdash; which is what every &chi;&sup2; above
does &mdash; is the better test, because dividing a handful of counts by a
small acceptance manufactures error bars. This is here for the one thing
folding cannot give: a birth-level median the data can be quoted at, next to
the {_f(med.get('Al capsule (after wall)', np.nan), 0, '&deg;')} the capsule
predicts and the {_f(med.get('3He gas M1+E0 (after wall)', np.nan), 0, '&deg;')}
the gas does. Bins below 2&nbsp;% of the group&rsquo;s peak acceptance are
dropped, and how much of the sample that costs travels with the table.</p>
{corr_block}
{corr_fig}
<p><b>The three topologies do not agree with each other after correction, and
they must.</b> The same source and the same physics feed all three, so a
correct acceptance would collapse them onto one curve. Their implied medians
run from {_f(med_int, 0, '&deg;')} to {_f(med_opp, 0, '&deg;')}. That is the
sharpest statement on this page about the acceptance &mdash; sharper than any
&chi;&sup2;, because it needs no model at all.</p>
{agree_block}
<p><b>This is the closest thing on the page to agreement, and it is worth being
precise about.</b> On the coincident perpendicular sample the capsule continuum
after the wall describes the corrected shape below the X17 threshold at
&chi;&sup2;/dof {one(g_al, 'chi2dof_sub')} &mdash; better than the gas at
{one(g_gas, 'chi2dof_sub')} &mdash; and above the threshold the data sits
{one(g_al, 'ratio_above', 2)}&times; the capsule prediction, an excess of
{one(g_al, 'excess_sigma', 1)}&sigma;. So the aluminium accounts for the bulk
of the continuum and <b>not</b> for the wide-angle tail. What lives in that
tail is the accidental population and the residual single-particle background,
both of which the event-mixed template already describes.</p>
<p class="sub">The &sigma; counts the corrected bins&rsquo; own errors and not
the uncertainty on a normalisation fitted to four sub-threshold bins, which is
a few per cent and would soften the number rather than change its sign. Read it
as a size, not as a significance.</p>
{edge_block}

<h2>8 &middot; The model-light test</h2>
<p>The ratio between topologies divides out the acceptance normalisation, the
vertex model and the efficiency scale; the shape of the physics survives.</p>
{ratio_table(R)}

<h2>What this does not settle</h2>
<ul>
<li><b>It does not identify the pairs as aluminium.</b> The one handle that
separates a 2&ndash;4&nbsp;MeV capsule pair from a 20.6&nbsp;MeV gas pair is
the total pair energy, and this setup does not measure it. Opening angle alone
cannot do it: after the wall the two medians are
{_f(med.get('Al capsule (after wall)', np.nan), 0, '&deg;')} and
{_f(med.get('3He gas M1+E0 (after wall)', np.nan), 0, '&deg;')}, separable in
principle and not in a thousand pairs.</li>
<li><b>The angle scale is still the open blocker.</b> Every measured angle on
this page carries its run&rsquo;s own <i>k</i>, and
<code>gap_check</code> fails on A and C: the pointing estimators want a smaller
<i>k</i> and the geometric bound wants a larger one. A wrong angle scale
distorts the measured spectrum, and nothing here corrects for it.</li>
<li><b>The incidence response is four points per arm</b>, with a positional
confound that the neighbour test controls but does not remove. A real
two-dimensional efficiency in position <i>and</i> incidence is the fix, and it
does not exist yet.</li>
<li><b>Multiple scattering after the capsule, energy loss, the leptons&rsquo;
own energies, pile-up and double-track finding at small separations are in none
of this.</b> Every one of them makes the acceptance an over-estimate, so this
is a shape and never a rate.</li>
<li><b>The intra control still has no timing cut</b> and cannot have one: both
legs in one chamber means one arm and no arm-to-arm time difference.</li>
<li><b>The capsule&rsquo;s own missing physics</b> is listed in
<code>IPC_MISSING.md</code>: neutron transport in the wall, the multipolarity
of the secondary cascade, and the hydrogen content of the fibre binder.</li>
</ul>

<p class="foot">Generated by <code>make_fold_report.py</code> on
{dt.date.today().isoformat()}.
Efficiency: <code>{ed}</code> ({em['n_runs']} runs, tag
&ldquo;{em['tag']}&rdquo;).
Acceptance: <code>{ad}</code> ({am['n_thrown']:,} pairs per run per variant,
variants {', '.join(am['variants'])}).
Fold: <code>{fd}</code> (acceptance variant
<code>{fm['variant']}</code>, aluminium lines unassigned taken as
{prov.get('assume', '?')}, {prov.get('lines_smeared', '?')} lines smeared).
Binning {int(fm['bins'][1] - fm['bins'][0])}&deg;; X17 threshold
{X17_MIN_DEG:.0f}&deg;.</p>
</main></body></html>'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--eff', default=None)
    ap.add_argument('--acceptance', default=None)
    ap.add_argument('--fold', default=None)
    a = ap.parse_args()
    ed = Path(a.eff) if a.eff else paths.out('efficiency_campaign')
    ad = Path(a.acceptance) if a.acceptance else paths.out(
        'acceptance_campaign')
    fd = Path(a.fold) if a.fold else paths.out('fold_campaign')
    for p, what in ((ed / 'campaign_efficiency.meta.json', 'campaign_efficiency'),
                    (ad / 'campaign_acceptance.meta.json', 'campaign_acceptance'),
                    (fd / 'campaign_fold.meta.json', 'campaign_fold')):
        paths.require(p, f'the {what} products -- run {what}.py first')
    out = fd / 'report.html'
    out.write_text(build(ed, ad, fd))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
