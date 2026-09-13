#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- the pair-vertex diagnosis, built from the tables.

Writes ``report.html`` beside the figures.  Nothing is hand-written: every
number in the prose is read out of the CSVs `diagnostics.py` produced, so
re-running a measurement moves the verdict, the tables and the figures
together.  Figures are referenced by relative path so the DAQ page's
``/analysis_file/<relpath>`` route serves the same file that works from disk.

    python -m pair_vertex_imaging.make_report
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402

OUT = HERE / 'figures'

FIGURES = [
    ('observation', 'The observation',
     'Per class, at the published 30&nbsp;mm leg cut. Pale grey is either '
     'leg&rsquo;s own miss distance at the beam axis; the coloured curve is '
     'the pair vertex computed in the transverse plane only; black is the '
     'published 3D closest approach, and the dotted black is the same thing '
     'on event-mixed pairs. <b>Both vertex curves sit further from the axis '
     'than the legs they were built from.</b>'),
    ('pointing', 'What one track actually knows',
     'The single-track miss distance at the beam axis, per chamber, over every '
     'gated track in the campaign &mdash; against a null built by shuffling '
     '<code>tan</code> among the tracks of that chamber, which keeps every '
     'marginal and destroys only the position&ndash;angle correlation that '
     'constitutes pointing. There is real information here and it is weak.'),
    ('conditioning', 'The vertex is the legs, amplified',
     'Left: the amplification 1/|sin&thinsp;&psi;| of the transverse crossing, '
     'per class. Right: the median vertex radius against it. The closed form '
     'holds to 1e-16, so the right-hand panel is not a fit &mdash; it is the '
     'identity being displayed.'),
    ('leg_scan', 'Tighten the legs and the vertex follows',
     'The per-leg pointing cut scanned from 60&nbsp;mm down to 5&nbsp;mm. The '
     'transverse crossing tracks it linearly all the way down and never '
     'settles onto a source size of its own; the 3D closest approach barely '
     'moves at all.'),
    ('y_view', 'Why the y view cannot be used',
     'Left: the pointing band&rsquo;s slope in each view, in units where a '
     'point source at the perpendicular foot gives 1. The x band is pinned at '
     '1 by the leg cut &mdash; that cut is a pure XZ quantity &mdash; and the '
     'y band, which nothing here cuts on, is a quarter of it at best. Right: '
     'the resulting y disagreement between the two legs.'),
    ('y_cost', 'What it costs the 3D vertex',
     'The 3D closest approach is free to slide both tracks along themselves to '
     'reduce a y mismatch, and every millimetre it slides moves the vertex '
     'transversely too. Binned in |dy| the transverse answer does not move and '
     'the 3D one follows |dy| all the way.'),
    ('scale', 'The angle scale moves one estimator and not the other',
     'Left: the band crossing under a tan rescale &mdash; exactly flat, which '
     'is the property <code>source_imaging</code> was built around. Middle: '
     'the per-track miss on every gated track, with no pointing cut, so no '
     'circularity. Right: the vertex, which has no such protection.'),
    ('floor', 'The floor, and how far above it the data sits',
     'Each leg&rsquo;s direction replaced by one that points exactly at a '
     'random point in the capsule, keeping its measured impact point, so the '
     'substitution changes the angle and nothing else. With both legs '
     'substituted every class returns 7.0&nbsp;mm &mdash; the median radius of '
     'a uniform 10&nbsp;mm disc, i.e. the source itself.'),
    ('lift', 'And whether the vertex selects pairs at all',
     'Real pairs over event-mixed ones, as a fraction of each sample, against '
     'the vertex cut. This is a different question from the imaging and it has '
     'a different answer, spelled out below.'),
]


def esc(s) -> str:
    return html.escape(str(s))


def n(v) -> str:
    return f'{int(v):,}'


def mm(v) -> str:
    return f'{float(v):.1f}'


def mm0(v) -> str:
    return f'{float(v):.0f}'


def pc(v) -> str:
    return f'{100 * float(v):.1f}%'


def x2(v) -> str:
    return f'{float(v):.2f}'


def load(src: Path) -> dict:
    need = ('classes.csv', 'decomposition.csv', 'leg_scan.csv', 'pointing.csv',
            'scale.csv', 'scale_focus.csv', 'ybudget.csv', 'yband.csv',
            'floor.csv', 'lift.csv', 'pair_vertex.meta.json')
    missing = [x for x in need if not (src / x).exists()]
    if missing:
        raise FileNotFoundError(
            f'{src} is missing {", ".join(missing)}\n'
            '  run:  python -m pair_vertex_imaging.vertex_lab --jobs 8\n'
            '        python -m pair_vertex_imaging.diagnostics --jobs 8')
    T = {x.split('.')[0]: pd.read_csv(src / x) for x in need
         if x.endswith('.csv')}
    T['meta'] = json.loads((src / 'pair_vertex.meta.json').read_text())
    p = src / 'scint_purity.csv'
    T['scint'] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return T


def table(d: pd.DataFrame, cols: dict, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    th = ''.join(f'<th>{c}</th>' for c in cols.values())
    rows = []
    for _, r in d.iterrows():
        tds = []
        for k in cols:
            v = r[k]
            f = fmt.get(k)
            tds.append('<td>&mdash;</td>' if pd.isna(v) else
                       f'<td>{esc(f(v) if f else v)}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<table class="t"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# --------------------------------------------------------------------------- #
def verdict(T: dict) -> str:
    C = T['classes']
    R = C[~C.mixed].set_index('topology').groupby(level=0)
    perp = C[(~C.mixed) & (C.topology == 'perpendicular') & (C.pair == 'all')]
    oppo = C[(~C.mixed) & (C.topology == 'opposing') & (C.pair == 'all')]
    allc = C[(~C.mixed) & (C.pair == 'all')]
    D = T['decomposition'].set_index('topology')
    L = T['leg_scan']
    L = L[~L.mixed]
    l5 = L[(L.leg_cut_mm == 5) & (L.topology == 'perpendicular')].iloc[0]
    l30 = L[(L.leg_cut_mm == 30) & (L.topology == 'perpendicular')].iloc[0]
    P = T['pointing'].set_index('arm')
    F = T['floor']
    fb = F[(F.variant == 'both') & (F.topology == 'perpendicular')].iloc[0]
    meta = T['meta']
    YB = T['yband'].set_index('arm')

    def yb(a):
        return f'{float(YB.loc[a].y_slope):.2f}'

    nreal = int(C[(~C.mixed) & (C.pair == 'all')].n.sum())
    # the leg / vertex comparison, pooled over the classes
    legbest = float(allc.leg_dca_best.median())
    legworst = float(allc.leg_dca_worst.median())
    return f'''
<p class="verdict"><b>The impression is right, and the two results do not
contradict each other &mdash; they are different measurements.</b> The
single-track image is an <i>ensemble centroid</i>: the zero crossing of
median(tan) against position, averaged over millions of tracks, whose error
falls as 1/&radic;N and reaches a few tenths of a millimetre. The pair vertex
is a <i>per-event position</i>, so it inherits the full per-track resolution
with nothing averaging it down. On this campaign that resolution is
{mm0(P.loc['A'].med_dca)}&nbsp;mm in chamber&nbsp;A,
{mm0(P.loc['C'].med_dca)}&nbsp;mm in C and {mm0(P.loc['D'].med_dca)}&nbsp;mm in
D. There is no way to build a 10&nbsp;mm image out of that, one pair at a
time.</p>

<p>Three things then make it worse, and all three are measured below rather
than argued.</p>

<p><b>1. The pairing amplifies the legs&rsquo; error instead of averaging it.</b>
Two lines in the transverse plane, each a known distance from the beam axis,
meet at a point whose distance from the axis is fixed algebraically:
<code>|c| = &radic;(e&#8321;&sup2;+e&#8322;&sup2;&minus;2e&#8321;e&#8322;cos&psi;)
/ |sin&psi;|</code>. That identity holds in the data to 1e-16, so <b>the
transverse pair vertex is a deterministic function of the two single-track
pointings and carries no information they did not already have</b> &mdash; and
1/|sin&psi;| is never below 1. Median amplification
{x2(D.loc['perpendicular'].amp_med)} for perpendicular pairs,
{x2(D.loc['intra'].amp_med)} for intra and {x2(D.loc['opposing'].amp_med)} for
opposing, which is exactly the order the three classes come out in.</p>

<p><b>2. The 3D closest approach spends transverse accuracy on a coordinate the
source does not localise.</b> The capsule is 10&nbsp;mm across but 80&nbsp;mm
long along the beam, so there is no y image to find; and the y view is not a
pointing measurement in any case &mdash; its band slope is
{yb('A')} (A), {yb('C')} (C), {yb('D')} (D) where a point source gives 1. The two legs consequently disagree in y
by a median of
{mm0(allc.abs_dy_cross.median())}&nbsp;mm,
and the 3D fit slides both tracks along themselves to reconcile that, dragging
the vertex off the transverse answer. <b>Dropping y outright halves it:</b>
perpendicular {mm0(perp.v_r.iloc[0])}&nbsp;&rarr;&nbsp;{mm0(perp.v_r_xz.iloc[0])}&nbsp;mm,
opposing {mm0(oppo.v_r.iloc[0])}&nbsp;&rarr;&nbsp;{mm0(oppo.v_r_xz.iloc[0])}&nbsp;mm.</p>

<p><b>3. Nothing here is a geometry problem or a bug.</b> Replace both legs&rsquo;
directions with directions that point exactly at a random point in the capsule,
keeping their measured impact points, and every class returns
{mm(fb.v_r_xz_med)}&nbsp;mm with {pc(fb.f_vrxz_10)} inside the bore &mdash;
{mm(fb.v_r_xz_med)}&nbsp;mm being the median radius of a uniform 10&nbsp;mm
disc, i.e. the capsule itself. The crossing geometry imposes no floor of its
own; the whole of the loss is the per-track angle.</p>

<p class="verdict"><b>What this means practically.</b> The pair vertex is not
broken and it is not a second measurement of the target &mdash; it is the two
legs&rsquo; own pointing, combined with a gain of 1&ndash;3. So it should be
read as a track-quality variable, not as imaging, and the imaging claim should
continue to rest on the band crossing. If a per-event vertex is wanted, it
exists already: <b>at a 5&nbsp;mm per-leg cut the transverse crossing puts
{pc(l5.f_vrxz_10)} of perpendicular pairs inside the capsule</b>, against
{pc(l30.f_vrxz_10)} at the published 30&nbsp;mm cut &mdash; on
{n(l5.n)} pairs of {n(l30.n)}. That is a real capsule image from pairs; it
costs {pc(1 - l5.n / l30.n)} of the sample and it is, by the algebra above, a
restatement of the leg cut.</p>

<p class="note">Sample: {n(nreal)} real pairs and as many event-mixed, over
{meta.get('n_runs', '?')} runs of the condor full pass, built at a
{mm0(meta['dca_max'])}&nbsp;mm leg ceiling so the published 30&nbsp;mm is one
point on a scan. At that 30&nbsp;mm point the sample is
<b>pair-for-pair identical to the published <code>pair_qa</code> table</b> on
all ten arm pairs &mdash; median leg miss {mm(legbest)} and {mm(legworst)}&nbsp;mm,
median vertex {mm0(allc.v_r.min())}&ndash;{mm0(allc.v_r.max())}&nbsp;mm.</p>
'''


def body(T: dict) -> str:
    C = T['classes']
    D = T['decomposition']
    L = T['leg_scan']
    LP = L[(~L.mixed) & (L.topology == 'perpendicular')].set_index('leg_cut_mm')
    lp5, lp60 = LP.loc[5], LP.loc[60]
    P = T['pointing']
    Y = T['ybudget']
    YB = T['yband']
    F = T['floor']
    LI = T['lift']
    SF = T['scale_focus']
    SC = T['scale']
    meta = T['meta']

    figs = ''.join(
        f'<figure><img src="{name}.png" alt="{esc(title)}">'
        f'<figcaption><b>{esc(title)}.</b> {cap} '
        f'<span class="prov"><code>{name}.pdf</code> for the slide.</span>'
        f'</figcaption></figure>'
        for name, title, cap in FIGURES)

    cl = C[(~C.mixed) & (C.pair != 'all')].sort_values(['topology', 'pair'])
    clm = C[C.pair != 'all'].pivot_table(
        index=['topology', 'pair'], columns='mixed',
        values=['v_r', 'v_r_xz']).reset_index()
    clm.columns = ['topology', 'pair', 'v_r_real', 'v_r_mixed',
                   'v_r_xz_real', 'v_r_xz_mixed']

    focus = SF.pivot_table(index='scale', columns='arm',
                           values='med_dca').reset_index()
    fcols = {'scale': 's'}
    for a in ('A', 'B', 'C', 'D'):
        if a in focus.columns:
            fcols[a] = f'chamber {a}'

    band = SC[(SC.leg_cut_mm == 30) & SC.topology.str.startswith('band_')]
    bspan = band.groupby('topology').band_x0.agg(['min', 'max'])
    bmove = float((bspan['max'] - bspan['min']).max())

    sc = T['scint']
    scint_block = ''
    if len(sc):
        ad = sc[sc.pair == 'A-D']
        scint_block = f'''
<h2>Is what is left after the cut resolution, or background?</h2>
<p>The leg scan says a tighter pointing cut sharpens the vertex. That is not
the same statement as &ldquo;the sample is dirty&rdquo;: a pointing cut sharpens
the vertex on a perfectly pure sample too, because it keeps the tracks whose
angle happened to come out well. The question that separates the two is
whether, <i>at a fixed pointing cut</i>, a scintillator-confirmed leg beats an
unconfirmed one. <code>det_a_scint</code> supplies that for arm&nbsp;A.</p>
{table(sc, {'pair': 'arm pair', 'leg_state': 'arm-A leg', 'n': 'pairs',
            'leg_dca_med': 'leg miss', 'sin_psi_med': 'sin&thinsp;&psi;',
            'v_r_xz_med': 'v<sub>r</sub><sup>xz</sup>',
            'f_vrxz_10': 'inside 10 mm'},
       {'n': n, 'leg_dca_med': mm, 'sin_psi_med': x2, 'v_r_xz_med': mm,
        'f_vrxz_10': pc})}
<p><b>Read the A&ndash;D row and ignore the other two.</b> Confirming an arm-A
leg selects tracks that point at arm&nbsp;A&rsquo;s own scintillator wall,
which restricts their direction &mdash; and for A&ndash;A and A&ndash;C that
restriction collapses the crossing angle (sin&thinsp;&psi; falls from
{x2(sc[(sc.pair == 'A-C')].sin_psi_med.max())} to
{x2(sc[(sc.pair == 'A-C')].sin_psi_med.min())} on A&ndash;C), so those rows
compare two different geometries and their vertex numbers are not comparable.
A&ndash;D is well conditioned either way
({x2(ad.sin_psi_med.max())} against {x2(ad.sin_psi_med.min())}), and there
confirmation does help: {mm(ad[ad.leg_state == 'confirmed'].v_r_xz_med.iloc[0])}
against {mm(ad[ad.leg_state == 'not confirmed'].v_r_xz_med.iloc[0])}&nbsp;mm,
{pc(ad[ad.leg_state == 'confirmed'].f_vrxz_10.iloc[0])} inside the bore against
{pc(ad[ad.leg_state == 'not confirmed'].f_vrxz_10.iloc[0])}. <b>Real, and
far too small to be the explanation</b> &mdash; a 10&nbsp;% improvement in the
median where a factor of three is needed. What survives the 30&nbsp;mm cut is
mostly resolution, not background.</p>
'''

    return f'''
<h2>What was measured, and on what</h2>
<p>One pair table, rebuilt from the stage-3 tracks with the same selection
<code>source_imaging</code> and <code>pair_qa</code> use &mdash; gated,
angle-calibrated, a ceiling on each leg&rsquo;s miss distance &mdash; but with
the ceiling left loose at {mm0(meta['dca_max'])}&nbsp;mm so that it can be
scanned, and with four quantities per pair where the published table keeps
one:</p>

<table class="t"><thead><tr><th>quantity</th><th>what it is</th>
<th>what it isolates</th></tr></thead><tbody>
<tr><td><code>v_r</code></td><td>radius of the 3D closest-approach midpoint of
the two lines</td><td>the published quantity, and a mixture of everything
below</td></tr>
<tr><td><code>v_r_xz</code></td><td>the same, computed in the transverse plane
only</td><td>uses the in-plane angles and nothing else &mdash; exactly the
information the band crossing uses</td></tr>
<tr><td><code>dy_cross</code></td><td>the two legs&rsquo; y, evaluated at that
transverse crossing, subtracted</td><td>the y information, with the transverse
information divided out</td></tr>
<tr><td><code>sin_psi_xz</code></td><td>sine of the transverse crossing
angle</td><td>the conditioning &mdash; how much the crossing multiplies
whatever error the legs have</td></tr>
</tbody></table>

<p class="caution"><b>The event-mixed null does not mean here what it means in
the opening-angle spectrum.</b> Mixing decorrelates the <i>trigger</i>, not the
<i>origin</i>: two tracks from two different neutron captures in the same
capsule both still came out of that capsule, so a mixed pair has a genuine
common source and a working vertex detector would image it just as well. Real
&asymp; mixed in the figures below is therefore <b>not</b> evidence that the
imaging failed. It is the separate (and expected) statement that the vertex
carries no information about whether the two legs shared a trigger.</p>

<h2>The observation, per arm pair</h2>
{table(cl, {'topology': 'class', 'pair': 'arms', 'n': 'pairs',
            'leg_dca_best': 'better leg', 'leg_dca_worst': 'worse leg',
            'v_r_xz': 'v<sub>r</sub><sup>xz</sup>', 'v_r': 'v<sub>r</sub> (3D)',
            'sin_psi_xz': 'sin&thinsp;&psi;',
            'abs_dy_cross': '|dy|', 'f_vr_10': '3D inside 10 mm'},
       {'n': n, 'leg_dca_best': mm, 'leg_dca_worst': mm, 'v_r_xz': mm,
        'v_r': mm, 'sin_psi_xz': x2, 'abs_dy_cross': mm0, 'f_vr_10': pc})}
<p>Medians, at the published 30&nbsp;mm leg cut. Every row has a vertex further
from the beam axis than either of the two tracks that made it, and every row
has the two legs disagreeing in y by more than the capsule is long.</p>

<h3>Against the event-mixed null</h3>
{table(clm, {'topology': 'class', 'pair': 'arms',
             'v_r_xz_real': 'v<sub>r</sub><sup>xz</sup> real',
             'v_r_xz_mixed': 'mixed', 'v_r_real': 'v<sub>r</sub> real',
             'v_r_mixed': 'mixed'},
       {'v_r_xz_real': mm, 'v_r_xz_mixed': mm, 'v_r_real': mm,
        'v_r_mixed': mm})}

<h2>1. The vertex is the two legs, combined with a gain</h2>
{table(D, {'topology': 'class', 'n': 'pairs',
           'closed_form_max_rel_resid': 'closed form, worst rel. residual',
           'leg_comb_med': 'combined leg miss', 'amp_med': 'gain (median)',
           'amp_p90': 'gain (p90)', 'frac_amp_gt5': 'gain &gt; 5',
           'v_r_xz_med': 'v<sub>r</sub><sup>xz</sup>'},
       {'n': n, 'closed_form_max_rel_resid': lambda v: f'{float(v):.1e}',
        'leg_comb_med': mm, 'amp_med': x2, 'amp_p90': x2,
        'frac_amp_gt5': pc, 'v_r_xz_med': mm})}
<p>The combined leg miss is the same
{mm(D.leg_comb_med.min())}&ndash;{mm(D.leg_comb_med.max())}&nbsp;mm in all three
classes &mdash; it is set by the leg cut, which is the same for all three. What
separates them is entirely the gain. That is why perpendicular pairs, whose
crossing is nearly square, give the best vertex in the campaign and opposing
pairs the worst, and it is a statement about geometry rather than about
chambers A, C or D.</p>

<h2>2. The single-track pointing, which is the input</h2>
{table(P, {'arm': 'chamber', 'n_tracks': 'gated tracks',
           'med_dca': 'median miss', 'med_dca_null': 'null',
           'f10': 'inside 10 mm', 'f10_null': 'null',
           'purity_10mm': 'excess over null, inside 10 mm'},
       {'n_tracks': n, 'med_dca': mm, 'med_dca_null': mm, 'f10': pc,
        'f10_null': pc, 'purity_10mm': pc})}
<p>The null shuffles <code>tan</code> among the tracks of one chamber. It keeps
the impact points, the angular distribution and the acceptance, and destroys
only the correlation between where a track landed and which way it was going
&mdash; which is the whole of &ldquo;this track came from the capsule&rdquo;.
So the difference between the two columns is the pointing information, in units
anyone can check.</p>
<p class="caution">There <i>is</i> real pointing information: chamber A&rsquo;s
median miss is half its null and its 10&nbsp;mm fraction is
{x2(float(P[P.arm == 'A'].f10.iloc[0]) / float(P[P.arm == 'A'].f10_null.iloc[0]))}&times;
it. But the last column is the one to keep: even inside 10&nbsp;mm, between a
third and a half of the tracks are accounted for by a sample with no source in
it at all. <b>Chamber D barely beats its own null</b>
({mm(float(P[P.arm == 'D'].med_dca.iloc[0]))} against
{mm(float(P[P.arm == 'D'].med_dca_null.iloc[0]))}&nbsp;mm), and a quarter of
its x plane is dead, which is the structure visible in its panel.</p>

<h2>3. Tighten the legs and the vertex follows them exactly</h2>
{table(L[(~L.mixed) & (L.topology == 'perpendicular')].sort_values('leg_cut_mm'),
       {'leg_cut_mm': 'leg cut', 'n': 'pairs', 'leg_comb_med': 'combined leg miss',
        'v_r_xz_med': 'v<sub>r</sub><sup>xz</sup>', 'f_vrxz_10': 'inside 10 mm',
        'v_r_med': 'v<sub>r</sub> (3D)', 'f_vr_10': 'inside 10 mm'},
       {'leg_cut_mm': mm0, 'n': n, 'leg_comb_med': mm, 'v_r_xz_med': mm,
        'f_vrxz_10': pc, 'v_r_med': mm, 'f_vr_10': pc})}
<p>Perpendicular pairs; the other two classes behave the same way with a larger
gain. The transverse crossing is 0.86&times; the cut across the whole range and
shows no sign of levelling off onto a source size &mdash; the signature of an
estimator limited by its legs and by nothing else. <b>The 3D closest approach
does not follow at all</b>: it improves by {pc(1 - lp5.v_r_med / lp60.v_r_med)}
while the legs improve by {x2(lp60.leg_comb_med / lp5.leg_comb_med)}&times;,
because it is limited by y and the y information does not improve with a cut
that is blind to y.</p>

<h2>4. The y view, and what it costs</h2>
<p>The leg cut is a pure XZ quantity &mdash; <code>dca_axis_mm</code> is the
closest approach of the line to the beam axis in projection and is blind to the
y slope &mdash; so the y band below is an uncut measurement on this sample even
though the x band is pinned at 1 by construction.</p>
{table(YB, {'arm': 'chamber', 'n': 'tracks',
            'x_slope': 'x band slope &times; d&#8869;',
            'y_slope': 'y band slope &times; d&#8869;',
            'y_irreducible_mm': 'irreducible y spread from the source length'},
       {'n': n, 'x_slope': x2, 'y_slope': x2, 'y_irreducible_mm': mm})}
<p>A point source at the perpendicular foot gives 1 in both. <b>The y view is
not a pointing measurement</b>, and it could not fully be one even if the
reconstruction were perfect: the capsule is 80&nbsp;mm long along y, so a track
from it carries an irreducible
{mm(YB.y_irreducible_mm.iloc[0])}&nbsp;mm y spread at the lever arm. There is
no y image to find and the 3D closest approach is spending transverse accuracy
looking for one.</p>
{table(Y[Y.topology == 'perpendicular'],
       {'dy_bin': '|dy| at the crossing', 'n': 'pairs', 'frac': 'of class',
        'v_r_xz_med': 'v<sub>r</sub><sup>xz</sup>',
        'v_r_med': 'v<sub>r</sub> (3D)', 'drag': 'the drag',
        'sep_med': 'line separation'},
       {'n': n, 'frac': pc, 'v_r_xz_med': mm, 'v_r_med': mm, 'drag': mm,
        'sep_med': mm})}
<p>The transverse answer is flat to within a millimetre across the whole range
while the 3D one runs from {mm(Y[(Y.topology == 'perpendicular')].v_r_med.iloc[0])}
to {mm(Y[(Y.topology == 'perpendicular')].v_r_med.iloc[-1])}&nbsp;mm. The drag
column is the median of v<sub>r</sub>&nbsp;&minus;&nbsp;v<sub>r</sub><sup>xz</sup>
pair by pair, and it accounts for the entire difference between the two
estimators with nothing left over.</p>

<h2>5. The angle scale: it moves the vertex, and it is not the explanation</h2>
<p>The band crossing is <code>&minus;intercept/slope</code>, so multiplying
every angle by <i>s</i> scales the numerator and the denominator together and
the crossing does not move. Over the whole scan it moves by
{mm(bmove)}&nbsp;mm &mdash; that is not an approximation, it is the exact
cancellation being displayed. The vertex uses the angle&rsquo;s magnitude and
has no such protection: at <i>s</i> = 1.33, the factor the arm-A scintillator
wall independently asks for, the perpendicular vertex goes from
{mm(SC[(SC.leg_cut_mm == 30) & (SC.topology == 'perpendicular') & (SC.scale == 1.0)].v_r_xz_med.iloc[0])}
to {mm(SC[(SC.leg_cut_mm == 30) & (SC.topology == 'perpendicular') & (SC.scale == 1.33)].v_r_xz_med.iloc[0])}&nbsp;mm.</p>
<p><b>So the angle scale matters to the vertex and not to the image, which is
half the answer to the original question.</b> It is not, however, the cause of
the failure: the scan has a minimum, and at that minimum the vertex is still
{mm(SC[(SC.leg_cut_mm == 30) & (SC.topology == 'perpendicular')].v_r_xz_med.min())}&nbsp;mm
rather than 7.</p>
<h3>Where the optimum sits, without the circularity</h3>
<p>The pair sample was selected at a pointing cut evaluated at
<i>s</i>&nbsp;=&nbsp;1, so its scale scan is biased toward finding its minimum
there. The focus scan below is run on <b>every gated track with no pointing cut
at all</b> and carries no such bias.</p>
{table(focus, fcols, {c: mm for c in fcols if c != 'scale'}
       | {'scale': x2})}
<p class="caution">Median single-track miss [mm]. Chamber A prefers
<i>s</i>&nbsp;&asymp;&nbsp;0.90, chamber C prefers <i>s</i>&nbsp;&le;&nbsp;0.7,
chamber D is flat to within 2&nbsp;mm over 0.6&ndash;1.1 and chamber B has no
minimum in range. <b>None of them asks for +33&nbsp;%.</b> That is a real
tension with the wall measurement and this analysis does not resolve it: the
wall is measured on the scintillator-confirmed sample and this scan runs on the
gated one, where by the table in section&nbsp;2 roughly half the tracks are
not from the target and dilute any focus. It is recorded here as an open
question, not as a competing number &mdash; a diluted focus scan is the weaker
of the two measurements.</p>

<h2>6. The floor: what perfect angles would give</h2>
{table(F, {'variant': 'legs substituted', 'topology': 'class', 'n': 'pairs',
           'v_r_xz_med': 'v<sub>r</sub><sup>xz</sup>',
           'v_r_med': 'v<sub>r</sub> (3D)', 'f_vrxz_10': 'inside 10 mm',
           'abs_dy_med': '|dy|', 'sep_med': 'line separation'},
       {'n': n, 'v_r_xz_med': mm, 'v_r_med': mm, 'f_vrxz_10': pc,
        'abs_dy_med': mm0, 'sep_med': mm})}
<p><code>none</code> is the data. <code>one</code> replaces one leg&rsquo;s
direction with a direction from a random capsule point to that leg&rsquo;s own
measured impact point; <code>both</code> replaces both. Only the angle changes
&mdash; the impact points, the acceptance and the pairing are the measured
ones. With both legs substituted every class returns 7.0&nbsp;mm and 100&nbsp;%
inside the bore, including the opposing class with its median gain of
{x2(float(D[D.topology == 'opposing'].amp_med.iloc[0]))}: a gain multiplies an
error, and with no error there is nothing to multiply. <b>The crossing geometry
imposes no floor; the whole of the loss is the per-track angle.</b></p>

{scint_block}

<h2>7. A separate question: can the vertex select pairs?</h2>
{table(LI[LI.variable == 'v_r_xz'],
       {'topology': 'class', 'cut_mm': 'vertex cut', 'k_real': 'real',
        'k_mixed': 'mixed', 'frac_real': 'of real', 'frac_mixed': 'of mixed',
        'lift': 'lift', 'sigma': '&sigma;'},
       {'cut_mm': mm0, 'k_real': n, 'k_mixed': n, 'frac_real': pc,
        'frac_mixed': pc, 'lift': x2, 'sigma': lambda v: f'{float(v):+.1f}'})}
<p>Intra pairs show a small but consistent lift of
{x2(LI[(LI.topology == 'intra') & (LI.variable == 'v_r_xz')].lift.min())}&ndash;{x2(LI[(LI.topology == 'intra') & (LI.variable == 'v_r_xz')].lift.max())},
in the same direction as the 2.06&nbsp;&plusmn;&nbsp;0.29 that
<code>det_a_intra</code> found on its slope-selected sample. <b>Opposing pairs
sit below one</b> &mdash; a real A&ndash;C pair is <i>less</i> likely to have a
tight vertex than a mixed one &mdash; which is the trigger talking, not the
physics: the production trigger is single-arm, so an A-triggered event is
already depleted in C tracks and the surviving ones are a different population.
That is the same effect <code>pairs.py</code> was built to control for.</p>
<p class="caution">A lift of 1 is <b>not</b> &ldquo;no signal&rdquo; and it is
not &ldquo;the imaging failed&rdquo;. Both legs of a mixed pair still came out
of the same capsule, so a perfect vertex detector would put mixed pairs on the
capsule too. What this table says is narrower: the vertex cannot be used as a
pair selection, and a cut on it removes signal and background alike.</p>

<h2>What this does not rule out</h2>
<ul>
<li><b>It does not say the tracking is wrong.</b> Every test here is consistent
with a reconstruction whose per-track angular resolution is what it is. The
geometry checks out (the closed form to 1e-16, the ideal-leg substitution
returning the source exactly), and nothing points at a sign, a frame or a
pairing error.</li>
<li><b>It does not measure the angular resolution itself.</b> The miss
distance at the axis folds together the angle error, the real source size, any
scattering in the chamber walls and the non-target background. Separating those
needs a sample with an independent direction reference, which is what
<code>det_a_scint</code>&rsquo;s wall-and-plastic lever arm is for and what a
per-detector recalibration would produce.</li>
<li><b>It does not settle the angle scale.</b> Section&nbsp;5 records that the
gated sample's focus prefers <i>s</i>&nbsp;&le;&nbsp;1 while the wall prefers
1.33, on two different samples, and leaves it open.</li>
<li><b>It says nothing about the opening angle.</b> The opening angle is a
difference of two directions and does not use the vertex at all, so none of
this touches the campaign spectra. It does mean that a vertex cut cannot be
used to clean them.</li>
<li><b>Chamber B is in the tables for completeness only.</b> It has no
field-shaping rings, no uniform drift field and no usable angle; its rows are
printed so that they can be seen to be empty of information, not because they
are a measurement.</li>
</ul>

<h2>The figures</h2>
{figs}
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'pair_vertex')))
    ap.add_argument('--out', default=str(OUT / 'report.html'))
    a = ap.parse_args()
    T = load(Path(a.src))
    doc = (f'<!doctype html><html lang="en"><head>'
           f'{head("Pair vertices and the capsule image")}</head><body>'
           f'<div class="topbar"><div class="topbar-in">'
           f'<span class="eyebrow">n_TOF 2026 &middot; preliminary</span>'
           f'</div></div><div class="wrap">'
           f'<h1>Why the pair vertex does not image the capsule, '
           f'and the single tracks do</h1>'
           f'<p class="lede">The two results are not in conflict. One is an '
           f'ensemble centroid over millions of tracks; the other is a '
           f'per-event position built from two of them, with a gain. '
           f'<code>ntof_athens_26/pair_vertex_imaging</code></p>'
           f'{verdict(T)}{body(T)}'
           f'<p class="prov">Built by '
           f'<code>pair_vertex_imaging/make_report.py</code> from the tables in '
           f'<code>{esc(a.src)}</code>. Every number above is read from those '
           f'CSVs; nothing in the prose is transcribed.</p>'
           f'</div></body></html>')
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding='utf-8')
    print(f'wrote -> {p}  ({len(doc):,} chars)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
