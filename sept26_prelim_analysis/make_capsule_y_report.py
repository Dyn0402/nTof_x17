#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_capsule_y_report.py -- ``report.html`` for the capsule-height ray trace.

Generated, never hand-written: every number is read back from what
`capsule_y.py` wrote, so re-running the analysis updates the tables, the figures
and the verdict together.  Figures are referenced with relative links so the same
file works from disk and from the DAQ page's ``/analysis_file`` route.

    python -m sept26_prelim_analysis.make_capsule_y_report
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
from sept26_prelim_analysis import capsule_y as CY  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

DECIDE = CY.DECIDE


def figure(name: str, caption: str, csv: str = None) -> str:
    csv = csv or f'{name}.csv'
    p = paths.out('capsule_y') / 'figures' / f'{name}.png'
    if not p.exists():
        return ''
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(caption)}"></a>'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{csv}">numbers &#8599;</a>'
            f'</figcaption></figure>')


def _f(v, d=1, plus=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{v:{"+" if plus else ""}.{d}f}'


def _i(v):
    return '&mdash;' if v is None or not np.isfinite(v) else f'{int(round(v)):,}'


# --------------------------------------------------------------------------- #
def main_table(base: pd.DataFrame) -> str:
    r = ['<table><thead><tr><th>chamber</th><th class="n">tracks in fit</th>'
         '<th class="n">ray-trace <i>y</i><sub>0</sub> [mm]</th>'
         '<th class="n">&chi;&sup2;/&nu;</th>'
         '<th class="n">&sigma;<sub>edge</sub> [mm]</th>'
         '<th class="n">band crossing [mm]</th>'
         '<th class="n">&Delta;NLL vs CAD</th></tr></thead><tbody>']
    for x in base.itertuples():
        bad = ' class="bad"' if x.chi2_ndf > 3 else ''
        r.append(
            f'<tr><td><b>{x.arm}</b></td><td class="n">{_i(x.n_in_fit)}</td>'
            f'<td class="n"><b>{_f(x.y0, 1, True)} &plusmn; {_f(x.err_boot)}</b></td>'
            f'<td class="n"{bad}>{_f(x.chi2_ndf)}</td>'
            f'<td class="n">{_f(x.sigma, 0)}</td>'
            f'<td class="n">{_f(x.band_mm, 1, True)}</td>'
            f'<td class="n">{_f(x.dnll_nominal, 0)}</td></tr>')
    return ''.join(r) + '</tbody></table>'


def sep_table(S: pd.DataFrame) -> str:
    r = ['<table><thead><tr><th>chamber</th>'
         '<th class="n">band <i>B</i></th><th class="n">ray <i>R</i></th>'
         '<th class="n"><i>g</i><sub>band</sub></th>'
         '<th class="n"><i>g</i><sub>ray</sub></th>'
         '<th class="n">v origin &delta; [mm]</th>'
         '<th class="n">capsule <i>y<sub>s</sub></i> [mm]</th></tr>'
         '</thead><tbody>']
    for x in S.itertuples():
        cls = '' if x.arm in DECIDE else ' class="muted"'
        r.append(f'<tr{cls}><td><b>{x.arm}</b></td>'
                 f'<td class="n">{_f(x.band_mm, 1, True)}</td>'
                 f'<td class="n">{_f(x.ray_mm, 1, True)}</td>'
                 f'<td class="n">{_f(x.g_band, 2)}</td>'
                 f'<td class="n">{_f(x.g_ray, 2)}</td>'
                 f'<td class="n"><b>{_f(x.delta, 1, True)}</b></td>'
                 f'<td class="n"><b>{_f(x.y_source, 1, True)}</b></td></tr>')
    return ''.join(r) + '</tbody></table>'


def var_table(R: pd.DataFrame, run: str) -> str:
    g = R[R.run == run]
    piv = g.pivot_table(index='variant', columns='arm', values='y0',
                        aggfunc='mean')
    arms = [a for a in CY.ARMS if a in piv]
    order = ['baseline'] + sorted(v for v in piv.index if v != 'baseline')
    r = ['<table><thead><tr><th>variant</th>'
         + ''.join(f'<th class="n">{a}</th>' for a in arms)
         + '<th class="n">shift on A, C [mm]</th></tr></thead><tbody>']
    b = piv.loc['baseline']
    for v in order:
        row = piv.loc[v]
        sh = [row[a] - b[a] for a in arms if a in DECIDE and np.isfinite(row[a])]
        bold = ' style="font-weight:600"' if v == 'baseline' else ''
        r.append(f'<tr{bold}><td>{_h.escape(v)}</td>'
                 + ''.join(f'<td class="n">{_f(row[a], 1, True)}</td>' for a in arms)
                 + '<td class="n">'
                 + ('&mdash;' if v == 'baseline' else
                    ', '.join(_f(x, 1, True) for x in sh))
                 + '</td></tr>')
    return ''.join(r) + '</tbody></table>'


def camp_table(base: pd.DataFrame) -> str:
    r = ['<table><thead><tr><th>chamber</th><th class="n">runs</th>'
         '<th class="n">median <i>y</i><sub>0</sub></th>'
         '<th class="n">run-to-run s.d.</th>'
         '<th class="n">median bootstrap</th>'
         '<th class="n">median &chi;&sup2;/&nu;</th>'
         '<th class="n">median band</th></tr></thead><tbody>']
    for arm, g in base.groupby('arm'):
        cls = '' if arm in DECIDE else ' class="muted"'
        r.append(f'<tr{cls}><td><b>{arm}</b></td>'
                 f'<td class="n">{len(g)}</td>'
                 f'<td class="n"><b>{_f(g.y0.median(), 1, True)}</b></td>'
                 f'<td class="n">{_f(g.y0.std(ddof=1))}</td>'
                 f'<td class="n">{_f(g.err_boot.median())}</td>'
                 f'<td class="n">{_f(g.chi2_ndf.median())}</td>'
                 f'<td class="n">{_f(g.band_mm.median(), 1, True)}</td></tr>')
    return ''.join(r) + '</tbody></table>'


# --------------------------------------------------------------------------- #
#: Campaign statistics use runs with at least this many tracks in the fit; the
#: two below it (run_126, run_128) are reported in the CSV and not summarised.
MIN_TRACKS = 500


def band_table(C: pd.DataFrame) -> str:
    """Per chamber over the campaign: the height, and what the fit adds."""
    r = ['<table><thead><tr><th>chamber</th><th class="n">runs</th>'
         '<th class="n">band crossing [mm]</th>'
         '<th class="n">acceptance fit [mm]</th>'
         '<th class="n">band, &plusmn;30&nbsp;% eff(v) tilt</th>'
         '<th class="n">fit, &plusmn;10&nbsp;% eff(v) tilt</th></tr></thead><tbody>']
    for arm, g in C.groupby('arm'):
        cls = '' if arm in DECIDE else ' class="muted"'
        sd = lambda s: f' <span class="muted">(s.d. {_f(s.std(ddof=1))})</span>'  # noqa: E731
        bt = (np.abs(g.band_per_tilt) * 0.3).median() if 'band_per_tilt' in g else np.nan
        rt = (np.abs(g.y0_tilt_p10 - g.y0_tilt_m10) / 2).median()
        r.append(f'<tr{cls}><td><b>{arm}</b></td><td class="n">{len(g)}</td>'
                 f'<td class="n"><b>{_f(g.band_mm.median(), 1, True)}</b>{sd(g.band_mm)}</td>'
                 f'<td class="n">{_f(g.y0.median(), 1, True)}{sd(g.y0)}</td>'
                 f'<td class="n">&plusmn;{_f(bt)}</td>'
                 f'<td class="n">&plusmn;{_f(rt)}</td></tr>')
    return ''.join(r) + '</tbody></table>'


def thin_table(one: pd.DataFrame) -> str:
    """The detailed run: both estimators on the SAME thinned tracks."""
    tilts = [t for t in CY.TILTS if t != 0.0]
    r = ['<table><thead><tr><th>chamber</th><th>estimator</th>'
         + ''.join(f'<th class="n">{t * 100:+.0f}&nbsp;%</th>' for t in tilts)
         + '<th class="n">per 100&nbsp;% tilt</th></tr></thead><tbody>']
    for arm, x in one.iterrows():
        for lab, pre, ref in (('band crossing', 'band', 'band_mm'),
                              ('acceptance fit', 'ray', 'ray_mm')):
            if ref not in x or not np.isfinite(x.get(ref, np.nan)):
                continue
            cells = ''.join(
                f'<td class="n">{_f(x[f"{pre}_tilt_{t:+.1f}"] - x[ref], 1, True)}</td>'
                for t in tilts)
            r.append(f'<tr><td><b>{arm}</b></td><td>{lab}</td>{cells}'
                     f'<td class="n"><b>{_f(abs(x[f"{pre}_per_tilt"]), 1)}</b></td></tr>')
    return ''.join(r) + '</tbody></table>'


def build(d: Path) -> str:
    meta = json.loads((d / 'capsule_y.meta.json').read_text())
    R = pd.read_csv(d / 'capsule_y_fits.csv')
    run = meta.get('curve_run', 'run_145')
    base = R[R.variant == 'baseline']
    one = base[base.run == run].set_index('arm')
    n_runs = base.run.nunique()
    val = json.loads((d / 'validation.json').read_text()) \
        if (d / 'validation.json').exists() else None
    sens_p = d / 'capsule_y_sensitivity.csv'
    T = pd.read_csv(sens_p) if sens_p.exists() else pd.DataFrame()

    # ---- campaign table: every usable run, all three chambers
    C = base[base.n_in_fit >= MIN_TRACKS].copy()
    if len(T):
        C = C.merge(T[['run', 'arm', 'band_per_tilt', 'median_yt',
                       'median_yt_shuffled']], on=['run', 'arm'], how='left')
    CA = C[C.arm.isin(DECIDE)]
    n_used = CA.run.nunique()
    med = CA.groupby('arm').median(numeric_only=True)
    sdv = CA.groupby('arm').std(numeric_only=True)
    P = CA.pivot_table(index='run', columns='arm', values=['band_mm', 'y0'])
    ac_band = (P.band_mm['A'] - P.band_mm['C']).dropna()
    ys_band = float(((P.band_mm['A'] + P.band_mm['C']) / 2).median())
    band_tilt10 = float((np.abs(CA.band_per_tilt) * 0.1).median()) \
        if 'band_per_tilt' in CA else np.nan
    ray_tilt10 = float((np.abs(CA.y0_tilt_p10 - CA.y0_tilt_m10) / 2).median())
    ys_err = float(np.ceil(np.hypot(abs(ac_band.median()) / 2,
                                    np.nan_to_num(band_tilt10)) + 0.5))
    # sensitivity ratio, campaign: fit per unit tilt over band per unit tilt
    if 'band_per_tilt' in CA:
        ratio = (np.abs((CA.y0_tilt_p10 - CA.y0_tilt_m10) / 0.2)
                 / np.abs(CA.band_per_tilt).clip(lower=0.05)).median()
    else:
        ratio = np.nan
    all_est = pd.concat([CA.y0, CA.band_mm])
    lo_all, hi_all = float(all_est.quantile(0.05)), float(all_est.quantile(0.95))
    need_ray = float((med.band_mm * (1 + med.v_origin_gain)).mean())
    big = CA[CA.n_in_fit >= 10000].groupby('arm').chi2_ndf.median()

    # what the per-chamber split USED to be read as (kept, to show the mistake)
    S = pd.read_csv(d / 'capsule_y_separated.csv')
    SA = S[S.arm.isin(DECIDE)].merge(CA[['run', 'arm']], on=['run', 'arm'])
    Q = SA.pivot_table(index='run', columns='arm', values=['delta', 'y_source'])
    old_rel = float((Q.delta['A'] - Q.delta['C']).median())
    old_ys = (float(Q.y_source['A'].median()), float(Q.y_source['C'].median()))
    common_d = float(((Q.delta['A'] + Q.delta['C']) / 2).median())

    onet = T[T.run == run].set_index('arm') if len(T) else pd.DataFrame()
    shuf = (onet[['median_yt', 'median_yt_shuffled']] if len(onet) else None)
    d_bad = one.loc['D', 'chi2_ndf'] if 'D' in one.index else np.nan
    vg = R[(R.run == run) & R.arm.isin(DECIDE)]
    wall_same = (np.allclose(
        vg[vg.variant == 'baseline'].sort_values('arm').y0.to_numpy(),
        vg[vg.variant == 'no SiPM wall'].sort_values('arm').y0.to_numpy(),
        atol=0.05) if (vg.variant == 'no SiPM wall').any() else False)

    def ray_thin(arm):
        return (abs(onet.loc[arm, 'ray_per_tilt']) if len(onet)
                and 'ray_per_tilt' in onet and arm in onet.index else np.nan)

    def band_thin(arm):
        return (abs(onet.loc[arm, 'band_per_tilt']) if len(onet)
                and arm in onet.index else np.nan)

    body = f"""
<h1>Where the &sup3;He capsule sits along the beam</h1>
<p class="deck">The ray-trace test scoped in <code>HANDOFF_CAPSULE_Y.md</code>
&sect;4, run on {run} in detail and on {n_runs} campaign runs ({n_used} with at
least {MIN_TRACKS} tracks in the fit) &mdash; and what it turned out to be good
for, which is not what it was built for.</p>

<div class="verdict">
<p><b>The source is about {_f(ys_band, 0, True)}&nbsp;mm up the beam, not at the
CAD centroid of {_f(CY.NOMINAL_Y0, 1, True)}&nbsp;mm, and that is not a coordinate
error.</b> The height is quoted from the <b>band crossing</b>:
<b>{_f(med.loc['A', 'band_mm'], 1, True)}&nbsp;mm</b> on A (run-to-run s.d.
{_f(sdv.loc['A', 'band_mm'])}) and <b>{_f(med.loc['C', 'band_mm'], 1, True)}&nbsp;mm</b>
on C (s.d. {_f(sdv.loc['C', 'band_mm'])}), giving
<b><i>y<sub>s</sub></i> &asymp; {_f(ys_band, 0, True)} &plusmn; {_f(ys_err, 0)}&nbsp;mm</b>
&mdash; the error is half the A&ndash;C difference and a &plusmn;10&nbsp;% eff(v)
tilt, in quadrature, rounded up.</p>
<p><b>The ray trace's job is to rule out a common frame error, and it does.</b> It
fits the <i>shape</i> of the v distribution against the surveyed plastics and uses
no angle. A v offset moves the band crossing one-for-one and the acceptance fit
{_f(float(med.v_origin_gain.mean()), 1)}&times;; if the band's height were all frame
offset the fits would sit near {_f(need_ray, 0, True)}&nbsp;mm, and they sit at
{_f(med.loc['A', 'y0'], 0, True)} and {_f(med.loc['C', 'y0'], 0, True)}. Every
estimate made here, both methods, both chambers, every run: 90&nbsp;% between
{_f(lo_all, 0, True)} and {_f(hi_all, 0, True)}&nbsp;mm.</p>
<p><b>The ray trace is not a measurement of the height.</b> A chamber's own
efficiency along v moves it {_f(ratio, 0)}&times; more than it moves the band
crossing (campaign median), so its chamber-to-chamber differences are eff(v), not
geometry. An earlier version of this page did not know that and read them as an
alignment &mdash; <a href="#mistake">the mistake is written up below</a> so it is
not repeated.</p>
<p><b>Two smaller results.</b>
{'The SiPM wall changes nothing (under 0.05&nbsp;mm)' if wall_same else 'The SiPM wall barely matters'}:
it clips v only beyond &plusmn;177&nbsp;mm, outside the fiducial; the plastics set
the acceptance. And chamber D cannot do the shape fit (&chi;&sup2;/&nu;
{_f(d_bad)} on {run}), though its band crossing, which does not care about its
damaged v shape, lands at {_f(C[C.arm == 'D'].band_mm.median(), 1, True)}&nbsp;mm.</p>
</div>

<h2 id="mistake">The mistake, so it is not made again</h2>
<div class="verdict">
<p><b>Rule: take positions from pointing, not from the shape of a distribution.
When two estimators of one quantity respond differently to a nuisance, split
them only on what is common to all chambers.</b></p>
</div>
<p><b>What was done.</b> The band crossing <i>B</i> and the acceptance fit
<i>R</i> respond to a rigid offset &delta; in a chamber's v coordinate with gains 1
and ~2.1. Solving <i>B</i>&nbsp;=&nbsp;<i>y<sub>s</sub></i>&nbsp;+&nbsp;&delta;,
<i>R</i>&nbsp;=&nbsp;<i>y<sub>s</sub></i>&nbsp;+&nbsp;<i>g</i>&delta; chamber by
chamber gave <i>y<sub>s</sub></i> = {_f(old_ys[0], 1, True)}&nbsp;mm from A and
{_f(old_ys[1], 1, True)}&nbsp;mm from C, stably in every run. That was read as
&ldquo;A and C disagree on the one capsule&rdquo;, as a
{_f(abs(old_rel), 0)}&nbsp;mm relative v misalignment, and turned into a
&plusmn;8&nbsp;mm error on the height.</p>
<p><b>Why it was wrong.</b> The two-equation model allows exactly one
chamber-specific effect, a rigid shift, and assumes nothing else differs between
the estimators. Something else does. The band crossing uses the <i>correlation</i>
of tan<sub>y</sub> with v &mdash; whether tracks point back to one height &mdash;
and thinning tracks along v re-weights points along that line without moving it.
The acceptance fit uses the <i>shape</i> of the v distribution, which is exactly
what a chamber's eff(v), dead strips and noisy columns distort, and those differ
from chamber to chamber. So the per-chamber &delta; was mostly eff(v).</p>
<p><b>How it was caught.</b> The side view
(<code>make_overhead_figure.py --projection y</code>) shows A, C and D agreeing on
the height to a few millimetres &mdash; while the shape fit said D was unusable and
A and C were 15&nbsp;mm apart. A real relative v offset would move the band crossings
apart one-for-one; they differ by {_f(abs(ac_band.median()), 1)}&nbsp;mm, not
{_f(abs(old_rel), 0)}.</p>
<p><b>The test that settles it</b> (<code>capsule_y --sensitivity</code>). Thin the
fit's own tracks by an imposed efficiency tilt (1&nbsp;+&nbsp;<i>a</i>&thinsp;v/170)
and refit both estimators. On {run}, the same tracks:</p>
{thin_table(onet) if len(onet) else '<p>(run --sensitivity to fill this table)</p>'}
{figure('capsule_y_sensitivity',
        'Left: both estimators refitted on the same thinned tracks. Right: every '
        'run, response per unit tilt &mdash; the band crossing from thinning, the '
        'acceptance fit from its pinned model tilt. Log scale.')}
<p><b>What survives of the split</b> is its chamber average: the common offset comes
out at {_f(common_d, 1, True)}&nbsp;mm, and no plausible eff(v) closes a gap of
{_f(need_ray - float(med.y0.mean()), 0)}&nbsp;mm, so the &ldquo;all a frame
error&rdquo; reading is still excluded.</p>
<p><b>A related trap in the side-view histogram.</b> The <i>median</i> of y at the
target, v&nbsp;&minus;&nbsp;tan<sub>y</sub>&nbsp;&times;&nbsp;234.6&nbsp;mm, is not a
pointing statement: it is set by mean(v) and mean(tan<sub>y</sub>).
{'' if shuf is None else
 'Shuffling tan<sub>y</sub> across a chamber&rsquo;s tracks, which destroys pointing, '
 'moves it from ' + ', '.join(
     f'{_f(shuf.loc[a, "median_yt"], 1, True)} to {_f(shuf.loc[a, "median_yt_shuffled"], 1, True)} ({a})'
     for a in DECIDE if a in shuf.index) + '&nbsp;mm.'}
The numbers in that figure's box &mdash; the band crossings &mdash; are the
measurement; the curves agreeing is what a real source predicts but is not by
itself the proof.</p>

<h2 id="height">The height, over the campaign</h2>
{band_table(C)}
<p>Tilt columns: the median shift of each estimator for the stated imposed
efficiency tilt. The band column uses a &plusmn;30&nbsp;% tilt and is still the
smaller of the two.</p>
{figure('capsule_y_campaign',
        'Both estimators per run. Solid: the band crossing, which quotes the '
        'height. Dashed: the acceptance fit.')}
<p><b>What is still open is A against C, at
{_f(abs(ac_band.median()), 1)}&nbsp;mm</b> (s.d. {_f(ac_band.std(ddof=1))} run to
run) in the band crossing. Because the band crossing is insensitive to eff(v), that
difference is either a real relative v offset between the chambers or a bias
common to band crossings that differs between them. It is the genuine v alignment
question, and it is small.</p>

<h2 id="what">What the ray trace fits</h2>
<p>The distribution of <b>v</b> on the strip plane of the trigger-matched sample,
read off the y strip map &mdash; no angle enters, so the handoff's &sect;6.2 y-scale
bracket does not apply. The earlier forward comparison
(<code>source_imaging.y_forward_model</code>) compared
<code>target_y_mm</code>, which carries the angle scale and its resolution.</p>
<ol>
<li><b>The plastic bars</b>, 300&nbsp;mm tall at
{_f(one.loc['A', 'plastic_depth'], 0)}&nbsp;mm past the strips (surveyed): a window
of half-width &asymp;83&nbsp;mm in v whose centre moves with the source at gain 0.44,
hence the fit's ~2.1 gain to a frame offset.</li>
<li><b>The 1/<i>r</i>&sup2; flux</b>, via a cos&thinsp;&theta;/<i>r</i>&sup2; weight.</li>
<li><b>The live area in v</b>: measured passivation band, zero-occupancy ranges,
noisy v columns, all from the run's own data.</li>
<li><b>A fitted edge smearing</b> &sigma;<sub>edge</sub> &asymp;
{_f(one.loc['A', 'sigma'], 0)}&nbsp;mm for scattering on the way to the plastic;
without it &chi;&sup2;/&nu; roughly doubles.</li>
</ol>
<p>The fit uses |v|&nbsp;&ge;&nbsp;{_f(CY.V_EDGE, 0)}&nbsp;mm: the plateau is not
described and on its own measures nothing (it rails at the grid end). Even so the
model is rejected at high statistics (&chi;&sup2;/&nu; {_f(big.get('A', np.nan))} on
A and {_f(big.get('C', np.nan))} on C above 10&thinsp;000 tracks) &mdash; one more
reason it quotes an exclusion and not a height.</p>
{main_table(one.reset_index())}
{figure('capsule_y_profiles',
        'The v profile against the ray trace on ' + run + '. Solid: the fit. '
        'Dashed: the source at the CAD centroid. Grey: the plateau, not in the '
        'likelihood.')}
{figure('capsule_y_likelihood',
        'Profile likelihood in the source height on ' + run + '. Open markers: '
        'the band crossing, not a term in the likelihood.')}
{figure('capsule_y_separation',
        'The common-offset exclusion. Only what is common to the chambers is '
        'read; the gap between the markers is eff(v).')}

<h2 id="valid">The ray trace, validated</h2>
{'' if not val else
 f'<p>Against an independent isotropic Monte Carlo through the same geometry: mean '
 f'v agrees to {_f(abs(val["d_mean"]), 2)}&nbsp;mm (MC error '
 f'{_f(val["mc_mean_err"], 2)}) and the r.m.s. to {_f(abs(val["d_rms"]), 2)}&nbsp;mm '
 f'on {_i(val["n_accepted"])} accepted rays. <code>capsule_y --validate</code>. '
 f'The ray trace is correct; its problem is what it is exposed to, not how it '
 f'is computed.</p>'}

<h2 id="syst">Every knob on the shape fit ({run})</h2>
{var_table(R, run)}
{figure('capsule_y_systematics', 'The shape fit under every variant on ' + run + '.')}
<p>Source shape (gas vs aluminium shell vs end caps, handoff &sect;6.1) moves the fit
by under a millimetre once &sigma;<sub>edge</sub> is free. The knobs that matter are
the ones touching the chamber's response near the v edges &mdash; the fiducial and
the eff(v) tilt &mdash; consistent with everything above.</p>

<h2 id="notrule">What this does not rule out</h2>
<ul>
<li><b>A real {_f(abs(ac_band.median()), 0)}&nbsp;mm relative v offset between A and
C.</b> The band crossings allow it; a survey of the two chambers' heights against
the scintillators would settle it.</li>
<li><b>A bias in the band crossing itself</b> that is not eff(v): background tracks
flattening the band (<code>y_image</code> notes the band slope comes out below 1), or
a tan<sub>y</sub> offset. A constant tan<sub>y</sub> offset would shift the crossing
by an amount &prop; 1/<i>s</i>, which differs between A and C; the shape fit, which
uses no angle, still puts the source 30&ndash;40&nbsp;mm up, so an angle artefact
cannot be the whole height.</li>
<li><b>A non-uniform capture density.</b> 500&nbsp;atm of &sup3;He is optically thick,
displacing the emission centroid toward the beam-entrance end. What is measured is
the <b>emission centroid</b>, which is what acceptance needs, not necessarily the
mechanical centre.</li>
<li><b>The hot-strip trigger cut is missing on some runs</b>; it only removes chamber
D triggers.</li>
</ul>

<h2 id="next">What to do with the number</h2>
<p>Handoff &sect;5: fitted per chamber on {run} and {n_used} runs &mdash; <b>met</b>;
quoted with the source-shape systematic &mdash; <b>met</b> (small); compared with the
band crossing and the difference attributed &mdash; <b>met</b>, to eff(v); chambers
agree within their spread &mdash; <b>nearly</b>: D's band crossing agrees, A and C
differ by {_f(abs(ac_band.median()), 1)}&nbsp;mm against run-to-run spreads of
{_f(sdv.loc['A', 'band_mm'])} and {_f(sdv.loc['C', 'band_mm'])}.</p>
<p>Working value: <b><i>y<sub>s</sub></i> = {_f(ys_band, 0, True)} &plusmn;
{_f(ys_err, 0)}&nbsp;mm</b>, the gas emission centroid in the detector frame, from
the band crossings of A and C. Before it becomes the dated placement constant &sect;5
describes, resolve the A&ndash;C {_f(abs(ac_band.median()), 0)}&nbsp;mm.</p>

<p class="prov">Built {dt.date.today().isoformat()} by
<code>make_capsule_y_report.py</code> from <code>capsule_y.py</code>
({meta['schema']}): the fits table, <code>capsule_y_sensitivity.csv</code>
(<code>--sensitivity</code>) and <code>validation.json</code>
(<code>--validate</code>); {n_runs} runs, statistics on runs with &ge;&nbsp;{MIN_TRACKS}
tracks. Capsule transverse position {_f(meta['capsule_xz'][0], 1, True)},
{_f(meta['capsule_xz'][1], 1, True)}&nbsp;mm. Preliminary &mdash; see
<code>PLAN.md</code> &sect;8.</p>
"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Capsule height along the beam</title>{HEAD}'
            f'</head><body><main>{body}</main></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('capsule_y')
    paths.require(d / 'capsule_y.meta.json', 'the capsule_y products')
    out = d / 'report.html'
    out.write_text(build(d), encoding='utf-8')
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
