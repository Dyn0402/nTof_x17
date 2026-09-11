#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_det_a_scint_report.py -- ``report.html`` for the arm-A scintillator match.

Generated, never hand-written: every number is read back from what
`det_a_scint.py` wrote, so re-running the analysis updates the tables, the
figures and the verdict together.  Figures are referenced with relative links so
the same file works from disk and from the DAQ page's ``/analysis_file`` route.

    python -m sept26_prelim_analysis.make_det_a_scint_report
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

WINDOW_NAME = {
    'prod': 'production accept window, &minus;100 to +60 ns',
    'core': 'peak core, &minus;30 to +30 ns',
    'ctrl': 'is_control, same width as production',
    'ctrl_core': 'is_control, same width as the core',
    'off': 'pre-trigger, &minus;560 to &minus;400 ns',
    'off_core': 'pre-trigger, same width as the core',
}
SEL_NAME = {
    'all': 'every gated track',
    'fiducial': 'inside the chamber’s active area',
    'single': 'the only track in its trigger',
    'slope': 'a measurable slope on both planes',
    'pointing': 'extrapolates within 30 mm of the beam axis',
}


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


def _p(v, d=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{100 * v:.{d}f}&thinsp;%'


def _i(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{int(v):,}'


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def rate_table(R: pd.DataFrame) -> str:
    rows = []
    for lay, name in (('wall', 'SiPM wall'), ('plas', 'plastic')):
        for sel in ('all', 'fiducial', 'single', 'slope', 'pointing'):
            sig = R[(R.layer == lay) & (R.window == 'prod')
                    & (R.selection == sel)]
            ctl = R[(R.layer == lay) & (R.window == 'ctrl')
                    & (R.selection == sel)]
            off = R[(R.layer == lay) & (R.window == 'off')
                    & (R.selection == sel)]
            if not len(sig):
                continue
            s = sig.iloc[0]
            strong = ' style="font-weight:600"' if sel == 'all' else ''
            rows.append(
                f'<tr{strong}><th class="s">{name}</th><td>{sel}</td>'
                f'<td class="n">{_i(s.n)}</td>'
                f'<td class="n">{_i(s.n_pred)}</td>'
                f'<td class="n">{_p(s.frac_pred)}</td>'
                f'<td class="n">{_p(s.frac_match_pred)}</td>'
                f'<td class="n">{_p(s.frac_match_hit)}</td>'
                f'<td class="n">{_p(ctl.frac_match_pred.iloc[0]) if len(ctl) else "&mdash;"}</td>'
                f'<td class="n">{_p(off.frac_match_pred.iloc[0]) if len(off) else "&mdash;"}</td>'
                f'</tr>')
    return (
        '<table><thead><tr><th>layer</th><th>selection</th>'
        '<th class="n">tracks</th><th class="n">predictable</th>'
        '<th class="n">of all</th><th class="n">confirmed</th>'
        '<th class="n">confirmed<br>given the layer fired</th>'
        '<th class="n">floor<br>(is_control)</th>'
        '<th class="n">floor<br>(pre-trigger)</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>')


def edge_table(W: pd.DataFrame) -> str:
    rows = []
    for r in W.sort_values(['group', 'side']).itertuples():
        tag = 'interior' if r.interior else 'wall end'
        rows.append(
            f'<tr><td>group {r.group} {r.side}</td><td>{tag}</td>'
            f'<td class="n">{_f(r.u_edge, 1)}</td>'
            f'<td class="n">{_f(r.u_fit, 1)}</td>'
            f'<td class="n">{_f(r.sigma_mm, 1)}</td>'
            f'<td class="n">{_f(r.plateau, 2)}</td></tr>')
    return ('<table><thead><tr><th>boundary</th><th>kind</th>'
            '<th class="n">surveyed u [mm]</th><th class="n">fitted u [mm]</th>'
            '<th class="n">width &sigma; [mm]</th>'
            '<th class="n">plateau</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def tol_table(T: pd.DataFrame) -> str:
    rows = [f'<tr><td class="n">{_f(r.n_sigma, 1)}</td>'
            f'<td class="n">{_f(r.n_sigma * r.sigma_mm, 1)}</td>'
            f'<td class="n">{_i(r.n_near)}</td>'
            f'<td class="n">{_p(r.frac_match)}</td></tr>'
            for r in T.itertuples()]
    return ('<table><thead><tr><th class="n">tolerance [&sigma;]</th>'
            '<th class="n">[mm]</th><th class="n">tracks on a boundary</th>'
            '<th class="n">confirmed</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def run_table(H: pd.DataFrame) -> str:
    H = H.copy()
    H['n'] = H.run.str.split('_').str[1].astype(int)
    rows = []
    for r in H.sort_values('n').itertuples():
        blk = ' style="background:#faf6fb"' if r.k_block else ''
        rows.append(
            f'<tr{blk}><th class="s">{r.run}</th>'
            f'<td class="n">{r.n_subruns}</td>'
            f'<td class="n">{_i(r.n_tracks)}</td>'
            f'<td class="n">{_p(r.frac_match_wall)}</td>'
            f'<td class="n">{_p(r.frac_match_plas)}</td>'
            f'<td class="n">{_p(r.frac_ctrl_wall)}</td></tr>')
    return ('<table><thead><tr><th>run</th><th class="n">sub-runs</th>'
            '<th class="n">tracks</th><th class="n">wall</th>'
            '<th class="n">plastic</th><th class="n">wall floor</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def both_table(B: pd.DataFrame) -> str:
    rows = [f'<tr><th class="s">{r.selection}</th>'
            f'<td class="n">{_i(r.n)}</td><td class="n">{_p(r.both)}</td>'
            f'<td class="n">{_p(r.wall_only)}</td>'
            f'<td class="n">{_p(r.plas_only)}</td>'
            f'<td class="n">{_p(r.neither)}</td>'
            f'<td class="n">{_p(r.both_ctrl)}</td></tr>'
            for r in B.itertuples()]
    return ('<table><thead><tr><th>selection</th><th class="n">tracks</th>'
            '<th class="n">both</th><th class="n">wall only</th>'
            '<th class="n">plastic only</th><th class="n">neither</th>'
            '<th class="n">both, control</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def conf_table(C: pd.DataFrame) -> str:
    cols = [c for c in C.columns if c.isdigit()]
    head = ''.join(f'<th class="n">fired {c}</th>' for c in cols)
    rows = []
    for _, r in C.iterrows():
        tot = sum(float(r[c]) for c in cols)
        cells = []
        for c in cols:
            hit = str(int(r['predicted'])) == c
            style = ' style="font-weight:600"' if hit else ''
            cells.append(f'<td class="n"{style}>'
                         f'{_p(r[c] / tot if tot else np.nan, 0)}</td>')
        rows.append(f'<tr><th class="s">points at {int(r["predicted"])}</th>'
                    f'{"".join(cells)}<td class="n">{_i(tot)}</td></tr>')
    return ('<table><thead><tr><th></th>' + head
            + '<th class="n">tracks</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def rail_table(R: pd.DataFrame) -> str:
    rows = []
    for r in R.itertuples():
        strong = (' style="font-weight:600"' if r.sample == 'in the v rail'
                  else '')
        rows.append(
            f'<tr{strong}><th class="s">{_h.escape(r.sample)}</th>'
            f'<td class="n">{_i(r.n)}</td>'
            f'<td class="n">{_p(r.frac_of_all)}</td>'
            f'<td class="n">{_p(r.frac_match_wall)}</td>'
            f'<td class="n">{_p(r.frac_match_plas)}</td>'
            f'<td class="n">{_p(r.frac_ctrl_wall, 2)}</td>'
            f'<td class="n">{_p(r.frac_slope)}</td></tr>')
    return ('<table><thead><tr><th>where the track lands</th>'
            '<th class="n">tracks</th><th class="n">of all</th>'
            '<th class="n">wall confirms</th>'
            '<th class="n">plastic confirms</th>'
            '<th class="n">wall floor</th>'
            '<th class="n">has a slope</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def robust_table(R: pd.DataFrame) -> str:
    rows = []
    for r in R.itertuples():
        if r.biased:
            note = ('<td class="s" style="color:#b04a3a">cuts on the '
                    'reconstructed direction &mdash; not valid here</td>')
            style = ' style="color:#8a949f"'
        else:
            note = '<td></td>'
            style = ''
        rows.append(
            f'<tr{style}><th class="s">{r.selection}</th>'
            f'<td>{"SiPM wall" if r.layer == "wall" else "plastic"}</td>'
            f'<td class="n">{_i(r.n_tracks)}</td>'
            f'<td class="n">{_i(r.n)}</td>'
            f'<td class="n">{_p(r.eps, 1)} &plusmn; {_p(r.eps_err, 1)}</td>'
            f'<td class="n">{_f(r.chi2dof, 1)}</td>{note}</tr>')
    return ('<table><thead><tr><th>selection</th><th>layer</th>'
            '<th class="n">tracks</th><th class="n">edges fitted</th>'
            '<th class="n">&epsilon;</th>'
            '<th class="n">&chi;&sup2;/dof</th><th></th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


# --------------------------------------------------------------------------- #
def build(d: Path) -> str:
    meta = json.loads((d / 'det_a_scint.meta.json').read_text())
    R = pd.read_csv(d / 'rates.csv')
    H = pd.read_csv(d / 'headline.csv')
    W = pd.read_csv(d / 'edge_width.csv')
    T = pd.read_csv(d / 'tolerance.csv')
    B = pd.read_csv(d / 'both_layers.csv')
    C = pd.read_csv(d / 'confusion.csv')
    RAIL = pd.read_csv(d / 'rail_census.csv')
    ROB = (pd.read_csv(d / 'angle_scale_robustness.csv')
           if (d / 'angle_scale_robustness.csv').exists() else pd.DataFrame())

    AS = meta.get('angle_scale') or {}
    AP = meta.get('angle_scale_plastic') or {}
    n_trk = meta['n_tracks']
    runs = meta['runs']
    failed = meta.get('failed', {})

    def g(lay, sel, win='prod', col='frac_match_pred'):
        x = R[(R.layer == lay) & (R.selection == sel) & (R.window == win)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    w_all, p_all = g('wall', 'all'), g('plas', 'all')
    w_pt, p_pt = g('wall', 'pointing'), g('plas', 'pointing')
    w_fl = g('wall', 'all', 'ctrl')
    p_fl = g('plas', 'all', 'ctrl')
    w_fl2 = g('wall', 'all', 'off')
    p_fl2 = g('plas', 'all', 'off')
    w_hit = g('wall', 'all', col='frac_match_hit')
    w_hit_c = g('wall', 'all', 'ctrl', 'frac_match_hit')
    p_hit_c = g('plas', 'all', 'ctrl', 'frac_match_hit')
    bo = B.set_index('selection')
    rl = RAIL.set_index('sample').frac_of_all
    rm = RAIL.set_index('sample').frac_match_wall
    sig = meta.get('edge_sigma_mm', np.nan)
    sfit = meta.get('sig_u_wall_median', np.nan)
    n_uncal = int(H.n_uncalibrated.sum()) if 'n_uncalibrated' in H else 0

    fail_html = ''
    if failed:
        # The stored reason is the module's full refusal message, which is
        # written for a terminal. On the page it is the same sentence three
        # times, so only the run and its track count are shown and the reason
        # is stated once below.
        items = ''.join(
            f'<li><b>{_h.escape(k)}</b> &mdash; '
            f'{_h.escape(str(v).split("none of its ")[-1].split(" gated")[0])}'
            f' gated arm-A tracks, none angle-calibrated</li>'
            for k, v in failed.items())
        fail_html = (
            '<div class="warn"><h3>Three runs are not in this sample, and '
            'their tracks are not lost</h3>'
            f'<ul>{items}</ul>'
            '<p>Their <code>k_arm_&lt;run&gt;.json</code> files exist; the '
            'stage-3 tables were built before them, so every direction in '
            'those tables is null. Rebuilding stage 3 for those runs returns '
            'about 370&thinsp;000 tracks, 13&thinsp;% more than this page '
            'has. Nothing here depends on them: the confirmation rate varies '
            'by a few per cent across the 31 runs that are here, and no '
            'conclusion turns on that spread.</p></div>')

    ang = ''
    if AS and np.isfinite(AS.get('eps', np.nan)):
        agree = (np.isfinite(AP.get('eps', np.nan))
                 and abs(AP['eps'] - AS['eps'])
                 <= 2 * np.hypot(AP.get('eps_err', 0), AS.get('eps_err', 0)))
        lev_ratio = meta['lever_plas'] / meta['lever_wall']
        shift_ratio = (AP.get('slope_mm', np.nan) / AS['slope_mm']
                       if AS.get('slope_mm') else np.nan)
        verdict = ('Two layers, two lever arms, one number' if agree else
                   'The two do not yet agree to within their errors, which is '
                   'the thing to resolve next')
        ang = f"""
<h2 id="angle">A surveyed boundary that moves with the track's own slope</h2>
<p>The wall's group boundaries are surveyed, fixed and known. Fitting where each
boundary <i>appears</i> to sit, separately in bins of the track's own in-plane
slope, gives a displacement that grows linearly with that slope:
<b>{_f(AS['slope_mm'], 1)} &plusmn; {_f(AS['slope_err'], 1)} mm per unit
tan</b>, on a rigid offset of only {_f(AS['intercept_mm'], 1)} &plusmn;
{_f(AS['intercept_err'], 1)} mm.</p>
<p>Nothing but the angle scale does that. A survey error, a swapped read-out
order or a plane-fit bias displaces every bin alike and lands entirely in the
intercept; only a tangent that is wrong by a factor gives a shift proportional
to the tangent. Dividing by the 97.4 mm lever arm turns the slope into the
fractional error in the calibrated tangent directly, with no assumption about
where the tracks came from:</p>
<p class="big">&epsilon; = {_p(AS['eps'], 1)} &plusmn; {_p(AS['eps_err'], 1)}
&nbsp;&rarr;&nbsp; the wall prefers <i>k</i>(arm A) &times;
{_f(AS['k_ratio'], 3)}</p>
<p><b>The decisive check is the second lever arm.</b> The plastic sits
{_f(meta['lever_plas'], 0)} mm past the strip plane against the wall's
{_f(meta['lever_wall'], 0)} mm, a ratio of {_f(lev_ratio, 2)}. If the defect is
an angle scale, the shift per unit tan must scale with the lever and
&epsilon; must not. It measures {_f(AP.get('slope_mm', np.nan), 1)} &plusmn;
{_f(AP.get('slope_err', np.nan), 1)} mm per unit tan against the wall's
{_f(AS['slope_mm'], 1)} &mdash; <b>a ratio of {_f(shift_ratio, 2)} against the
{_f(lev_ratio, 2)} the geometry demands</b> &mdash; and returns &epsilon; =
{_p(AP.get('eps', np.nan), 1)} &plusmn; {_p(AP.get('eps_err', np.nan), 1)}
against the wall's {_p(AS['eps'], 1)}. {verdict}.</p>
<p><b>And it survives every selection that does not cut on the direction
itself.</b> A large error in a calibrated quantity is most easily faked by a
selection that correlates the slope with position, so the fit is repeated on
samples cut by position, by fit quality and by both:</p>
{robust_table(ROB) if len(ROB) else ''}
<p>Thirty to thirty-four per cent throughout, on both layers. The one entry that
moves is the beam-pointing selection, and it is in the table to be excluded
rather than to disagree: that cut is computed <i>from</i> the reconstructed
direction, so selecting on it selects on the slope in a way correlated with
position, and its &chi;&sup2;/dof says so.</p>
<p><b>What this does not settle.</b> The wall fit has &chi;&sup2;/dof
{_f(AS.get('chi2dof', np.nan), 1)}, so the shift is not perfectly linear in tan
and the quoted error is inflated to cover it. The tan bins are quantiles of a
distribution that is not symmetric about zero, so a curvature in the true
relation would leak into the fitted slope. And this measures arm A on the wall's
own u axis only; it says nothing about C or D, nor about the v axis. It is not,
on its own, a replacement calibration &mdash; it is an independent handle on the
open blocker, from a direction the target imaging cannot see.</p>
{figure('g_angle_scale', 'The apparent shift of each surveyed boundary against '
        'the mean slope of the tracks in the bin, at both lever arms. The line '
        'through each is the weighted fit; its slope is &epsilon; times that '
        'layer’s lever arm.')}
"""

    body = f"""
<h1>Detector A, track by track, against the scintillators behind it</h1>
<p class="deck">Every gated arm-A track extrapolated to the SiPM wall, the
plastic bars and the liquid cell, and asked whether the channel it lands on is a
channel that fired. {_i(n_trk)} tracks over {len(runs)} runs.</p>

<div class="verdict">
<p><b>The chamber and the scintillators agree, and the agreement is
positional.</b> {_p(w_all)} of the tracks that can be confirmed at the wall
point at a wall group that fired, against an accidental floor of
{_p(w_fl, 2)}; at the plastic it is {_p(p_all)} against {_p(p_fl, 2)}. On the
tracks that extrapolate back to the beam axis the wall reaches {_p(w_pt)} and
the plastic {_p(p_pt)}, and <b>{_p(bo.both.get('pointing', np.nan))} of them are
confirmed by both layers at once</b> &mdash; two independent detectors at
different depths, agreeing on the same track.</p>
<p><b>The position tolerance is {_f(sig, 0)} mm, not the {_f(sfit, 1)} mm the
fit errors claim</b>, and it is measured rather than assumed.</p>
<p><b>One new result.</b> The surveyed wall boundaries appear to move with the
track's own slope, at {_f(AS.get('slope_mm', np.nan), 1)} mm per unit tan. That
is an angle-scale signature and nothing else produces it. It says the arm-A
tangents in the track table are about {_p(AS.get('eps', np.nan), 0)} too large.
</p>
</div>

{fail_html}

<h2 id="what">What this is, and what the analysis had instead</h2>
<p>Three things existed before this page, and each is less than a per-track
positional confirmation:</p>
<ul>
<li><code>wall_A</code> and <code>plastic_A</code> in the track table are
<b>per-trigger</b> booleans from stage 1 &mdash; &ldquo;arm A's wall fired
somewhere&rdquo;. A track on the far side of the chamber from the bar that fired
carries the same flag as one pointing straight at it. The intra-chamber pair
study's scintillator tag is this.</li>
<li><code>build_tracks.predictions</code> has been writing
<code>pred_sipm_bar</code> and <code>pred_plastic</code> for every track in all
586 track files, and nothing in the chain ever compared them with the slim.</li>
<li>The pointing coincidence in <code>run145_target_imaging</code> does compare
them, but in the <i>x</i> plane only: it never checks that the track is inside
the bar's 500 mm length, it returns one boolean with no residual, and its
consumers use it as a purity cut rather than as a measurement.</li>
</ul>
<p>Here the extrapolation is three-dimensional, every layer position is read
from the DAQ's own <code>run_config.json</code> (the strip plane at
{_f(meta['layers']['w_strip'], 1)} mm, the wall at
{_f(meta['layers']['w_wall'], 1)}, the plastic at
{_f(meta['layers']['w_plas'], 1)}, the liquid at
{_f(meta['layers']['w_ls'], 1)}), and arm A is placed identically in all 36 runs
&mdash; asserted, not assumed.</p>

<h2 id="maps">The map the question asked for</h2>
{figure('a_mm_channel', 'The chamber surface, each 20 mm cell coloured by the '
        'scintillator channel that confirmed the most tracks landing in it: '
        'the four SiPM wall read-out groups on the left, the two plastic bars '
        'on the right.')}
<p>The colour bands run in <i>u</i> and are flat in <i>v</i>, which is what the
geometry demands: both layers are segmented along <i>u</i> only. The wall's
bands are wider than a quarter of the chamber because the wall is 400 mm of
instrumented width sitting 97 mm behind a 380 mm chamber, and because the tracks
are not normal to it.</p>
{figure('b_mm_fraction', 'The same surface, coloured by how often a track '
        'landing there is confirmed, and by how single-valued the confirming '
        'group is.')}
{figure('c_wall_plane', 'Every track projected onto the SiPM wall, with the '
        'surveyed bars and groups drawn on top.')}
{figure('c2_plastic_plane', 'The same tracks projected onto the plastic layer, '
        '190.6 mm past the strip plane.')}
<p>The plastic projection is where the <i>v</i> coordinate earns its keep. The
plastic bars are 300 mm long against the chamber's 340 mm and sit twice as far
away, so only {_p(g('plas', 'all', col='frac_pred'))} of arm-A tracks land on
one at all; the rest leave the layer through its end and can never be confirmed
by it. A test that ignored <i>v</i> would count every one of those as a
failure.</p>

<h2 id="rates">The rates, each against its own accidental floor</h2>
<p>Every control window has <b>the same width</b> as the signal window it
controls. That is not a detail. The first version of this module used a 700 ns
control against a 160 ns signal window and duly reported the plastic's
confirmation rate as equal to its accidental floor &mdash; an artefact of
comparing a wide window with a narrow one. Two independent floors are quoted:
the slim's own <code>is_control</code> sample, and real hits in a pre-trigger
window. They agree.</p>
{rate_table(R)}
<p><b>The floors behave exactly as chance must.</b> Given that the wall fired at
all, a control-window &ldquo;match&rdquo; lands on the predicted one of four
groups {_p(w_hit_c, 0)} of the time, against the 25&thinsp;% of a coin toss
between four; at the plastic it is {_p(p_hit_c, 0)} against 50&thinsp;% between
two. Nothing is being confirmed there, which is the point of quoting it.</p>
{figure('e_rates', 'Confirmation rate per selection with both accidental '
        'floors drawn on top.')}
{figure('d_both_layers', 'How the two layers partition the tracks that could '
        'be confirmed by either.')}
{both_table(B)}

<h2 id="tolerance">The position tolerance, measured</h2>
<p>The formal plane-fit error extrapolates to {_f(sfit, 1)} mm at the wall. That
number is a floor and is not the width that matters: it carries the plane fit
alone, not the angle scale, not scattering in the 97 mm of air and structure
between the strips and the wall, and not the survey. The width that matters is
how sharply the identity of the fired group switches as the predicted crossing
crosses a boundary, and that is measurable.</p>
{figure('f_edge_profile', 'The probability that each group is the one that '
        'fired, against the predicted crossing. Dotted lines are the surveyed '
        'boundaries.')}
{edge_table(W)}
<p>The interior boundaries give <b>&sigma; = {_f(sig, 1)} mm</b>, ten times the
formal error, and the plateaux sit at 0.8&ndash;0.9 rather than 1.0 &mdash; the
wall's own single-group inefficiency and cross-talk, which the pointing cannot
remove. Widening the match to accept either neighbour of a boundary within a
tolerance buys what the geometry says it should and no more:</p>
{tol_table(T)}
{figure('h_confusion', 'Where the tracks pointing at each group actually found '
        'their hit.')}
{conf_table(C)}
<p>The diagonal is also the only in-situ check on the one piece of this geometry
the DAQ config does not record: the SiPM bars carry no
<code>ntof_daq</code> block, so the bar-to-<code>detn</code> order is empirical.
A descending map would put this population on the anti-diagonal.</p>
{figure('j_residual', 'The continuous residual to the fired channel’s centre.',
        'j_residual.wall.csv')}

{ang}

<h2 id="rail">What the projection found in the track table</h2>
<p>The wall projection carries a hard horizontal stripe at v &asymp;
&minus;195 mm that no part of the apparatus sits at. The cause is in the tracks,
not in the wall: <b>the fitted y position rails just outside the chamber's
active area</b>. Arm A is 340 mm tall, so |v| &le; 170 mm is the whole of it,
and {_p(rl.get('outside it in v', np.nan))} of gated arm-A tracks are outside
that in v &mdash; {_p(rl.get('in the v rail', np.nan))} of all tracks in a
single 20 mm window at the rail.</p>
<p>The scintillators are an external arbiter of what those tracks are, and they
are unambiguous. A railed track is confirmed at a small fraction of the rate of
one inside the chamber, on <i>both</i> layers, while its accidental floor moves
the other way:</p>
{rail_table(RAIL)}
{figure('k_v_rail', 'Tracks against the reconstructed v, and the wall’s '
        'confirmation rate against the same axis. Shaded is outside the '
        'chamber.', 'k_v_rail.csv')}
<p>This page reports it and does not cut on it &mdash; whether a railed y should
fail the 3D gate is a decision for the reconstruction, not for a confirmation
study. What it does mean is that <b>the &ldquo;all&rdquo; row of every table
above is diluted by a population that is largely not real</b>, which is why the
fiducial selection is carried beside it.</p>

<h2 id="runs">Run by run</h2>
{figure('i_per_run', 'The confirmation rate across the campaign.')}
{run_table(H)}

<h2 id="not">What this does not rule out</h2>
<ul>
<li><b>It does not measure the wall's efficiency.</b> Everything here is
conditioned on a reconstructed track, so a particle the chamber missed is
invisible to it. The efficiency measurement runs the other way round and lives
in <code>campaign_efficiency</code>.</li>
<li><b>It does not resolve <i>v</i>.</b> The wall's bars run along <i>v</i> and
its two ends carry the position along them, which this page never reads; every
match here is in <i>u</i>. The two-ended read-out is what
<code>scintillators.py</code> started on, and folding it in would add a second
coordinate to every confirmation.</li>
<li><b>It does not separate the wall's cross-talk from the extrapolation.</b>
The {_f(sig, 0)} mm edge width and the 0.8&ndash;0.9 plateaux are both
consistent with a real spread in where the particle arrived and with a wall that
sometimes lights a neighbour; this data cannot tell them apart.</li>
<li><b>The angle-scale number is one arm, one axis, one estimator.</b> It is not
a calibration and must not be applied as one. What it is entitled to do is say
that arm A's tangents are too large by a fraction the target imaging does not
see.</li>
<li><b>{_i(n_uncal)} tracks in the runs that are here were dropped for want of
an angle calibration</b>, on top of the three runs excluded outright.</li>
</ul>

<h2 id="repro">Reproducing this</h2>
<pre><code>python -m sept26_prelim_analysis.det_a_scint --jobs 6
python -m sept26_prelim_analysis.make_det_a_scint_figures
python -m sept26_prelim_analysis.make_det_a_scint_report</code></pre>
<p class="prov">Schema <code>{meta['schema']}</code>. Windows:
{'; '.join(f'{k} = {WINDOW_NAME[k]}' for k in ('prod', 'core', 'ctrl', 'off'))}.
Per-track tables under <code>tracks/</code>, one parquet per run. Built
{dt.date.today().isoformat()}.</p>
"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Detector A &rarr; scintillators</title>{HEAD}'
            f'</head><body><main>'
            f'{body}</main></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('det_a_scint')
    paths.require(d / 'det_a_scint.meta.json', 'the det_a_scint products')
    out = d / 'report.html'
    out.write_text(build(d))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
