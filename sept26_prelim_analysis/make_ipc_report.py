#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ipc_report.py -- build ``report.html`` for the IPC modelling deep dive.

Answers three questions, in this order: what does the expected internal-pair
spectrum look like, does it depend on when the neutron arrives, and how much of
what we will see is the capsule rather than the gas.  Generated from
:mod:`ipc_born`, :mod:`ipc_channels` and :mod:`ipc_aluminium`, never
hand-written, so re-running them moves the prose, the tables and the verdict
together.

THE SPECTRUM IS THE DELIVERABLE.  The previous version of this page led with
tables of "fraction beyond 90 / 109 / 130 degrees".  That is three numbers
where there is a curve, and it invites an argument about where to put the
threshold instead of about the physics.  Every table here is now either a
spectrum (dN/dtheta, with a running fraction-above column so the reader picks
the threshold) or a yield.

    python -m sept26_prelim_analysis.make_ipc_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402
from sept26_prelim_analysis import ipc_channels as IC  # noqa: E402
from sept26_prelim_analysis import ipc_aluminium as AL  # noqa: E402
from sept26_prelim_analysis import ipc_diagrams as DG  # noqa: E402
from sept26_prelim_analysis.make_funnel_report import CSS, FONT_LINK  # noqa: E402

N = 2_000_000

#: The coarse axis the HTML tables use.  The CSVs keep the full 1 deg spectrum;
#: a 180-row table in a browser is not a table, it is a scroll.
COARSE = np.arange(0.0, 181.0, 15.0)


def figure(name: str, caption: str, alt: str = '') -> str:
    import html as _h
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(alt or caption)}">'
            f'</a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


def _rebin(y, bins=None):
    """A 1 deg density onto the coarse axis, preserving the integral."""
    if bins is None:
        bins = COARSE
    frac = np.asarray(y, float) * np.diff(IB.THETA_BINS)
    idx = np.digitize(IB.THETA_MID, bins) - 1
    out = np.zeros(len(bins) - 1)
    np.add.at(out, idx, frac)
    return out / np.diff(bins), out


def spectrum_table(curves: dict, note: str = '') -> str:
    """One row per 15 deg bin, one column per curve, plus running total.

    ``curves`` maps a column label to a 1 deg density.  The last two columns
    are the fraction of the FIRST curve in the bin and beyond it, so a reader
    who wants "how much is past 120 degrees" reads it off rather than being
    told what threshold to care about.
    """
    lo, hi = COARSE[:-1], COARSE[1:]
    dens = {k: _rebin(v)[0] for k, v in curves.items()}
    first = _rebin(list(curves.values())[0])[1]
    above = first.sum() - np.cumsum(first)
    rows = []
    for i in range(len(lo)):
        cells = ''.join(f'<td class="n">{dens[k][i]:.4f}</td>' for k in curves)
        band = (' style="background:color-mix(in srgb,var(--accent) 7%,transparent)"'
                if lo[i] >= 105 and hi[i] <= 150 else '')
        rows.append(f'<tr{band}><th class="s">{lo[i]:.0f}&ndash;{hi[i]:.0f}&deg;</th>'
                    f'{cells}<td class="n">{100 * first[i]:.2f}%</td>'
                    f'<td class="n">{100 * above[i]:.2f}%</td></tr>')
    heads = ''.join(f'<th>{k}</th>' for k in curves)
    lab = list(curves)[0]
    return ('<table class="t"><thead><tr><th>opening angle</th>' + heads
            + f'<th>{lab}<br><span class="u">in the bin</span></th>'
            + f'<th>{lab}<br><span class="u">beyond it</span></th>'
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
            + (f'<p class="note">{note}</p>' if note else ''))


def _ratio(x: float) -> str:
    """A capture-count ratio to one significant figure, as a power of ten.

    The three variants in section 8 span two orders of magnitude because one of
    their inputs is unverified, so printing 1256542 would be claiming six
    digits of a number whose first digit is in question.
    """
    e = int(np.floor(np.log10(x)))
    m = round(x / 10.0 ** e)
    if m == 10:          # 9626 is 1e4, not 10e3
        m, e = 1, e + 1
    sup = str(e).translate(str.maketrans('0123456789-', '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b'))
    return f'{m:d}&times;10{sup}' if e >= 3 else f'{x:,.0f}'


def val_table(V) -> str:
    rows = []
    for _, r in V.iterrows():
        ok = ('<span style="color:var(--good);font-weight:600">passes</span>'
              if r['pass'] else
              '<span style="color:var(--warn);font-weight:600">FAILS</span>')
        rows.append(f'<tr><th class="s">{r.check}<span class="why">{r.quantity}'
                    f'</span></th><td class="n">{r.a:.5g}</td>'
                    f'<td class="n">{r.b:.5g}</td>'
                    f'<td class="n">{r.delta:.2e}</td>'
                    f'<td class="n">{r.tol:.2e}</td><td class="n">{ok}</td></tr>')
    return ('<table class="t"><thead><tr><th>check</th><th>this module</th>'
            '<th>the reference</th><th>difference</th><th>tolerance</th>'
            '<th></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def multipole_table(T) -> str:
    rows = []
    for _, r in T.iterrows():
        ap = '&mdash;' if r.alpha_pair != r.alpha_pair else f'{r.alpha_pair:.2e}'
        y = IB.grid_spectrum(r.multipole, IB.E_TRANSITION)
        q = np.interp([0.25, 0.5, 0.75],
                      np.cumsum(y * np.diff(IB.THETA_BINS)), IB.THETA_MID)
        rows.append(f'<tr><th class="s">{r.multipole}'
                    f'<span class="why">{r.what}</span></th>'
                    f'<td class="n">{ap}</td>'
                    f'<td class="n">{q[0]:.0f}&deg;</td>'
                    f'<td class="n"><b>{q[1]:.0f}&deg;</b></td>'
                    f'<td class="n">{q[2]:.0f}&deg;</td>'
                    f'<td class="n">{100 * IB.frac_above(y, 109.0):.1f}%</td></tr>')
    return ('<table class="t"><thead><tr><th>multipole</th>'
            '<th>pairs per photon<br><span class="u">&alpha;<sub>pair</sub>'
            '</span></th><th>lower quartile</th><th>median &theta;</th>'
            '<th>upper quartile</th><th>beyond 109&deg;</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def channel_table(T) -> str:
    rows = []
    for _, r in T.iterrows():
        rows.append(f'<tr><th class="s">{r.channel} &nbsp;<code>{r.entrance}</code>'
                    f'<span class="why">{r.note}</span></th>'
                    f'<td class="n">{r.sigma_pair_ub:.3f}</td>'
                    f'<td class="n">{100 * r.share_of_pairs:.0f}%</td>'
                    f'<td class="n">{r.median_deg:.0f}&deg;</td>'
                    f'<td class="n"><b>{100 * r.share_gt109:.0f}%</b></td></tr>')
    return ('<table class="t"><thead><tr><th>channel</th>'
            '<th>&sigma;<sub>pair</sub> [&mu;b]</th><th>share of all pairs</th>'
            '<th>its median angle</th>'
            '<th>share of the pairs beyond 109&deg;</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def sens_table(S) -> str:
    rows = []
    for _, r in S.iterrows():
        rows.append(f'<tr><th class="s">{r.variation}</th>'
                    f'<td class="n">{r.sigma_e0_ub:.3f}</td>'
                    f'<td class="n">{100 * r.e0_share:.0f}%</td>'
                    f'<td class="n">{r.ipc_per_capture:.2e}</td></tr>')
    return ('<table class="t"><thead><tr><th>if instead&hellip;</th>'
            '<th>&sigma;<sub>E0</sub> [&mu;b]</th><th>E0 share of pairs</th>'
            '<th>IPC per radiative capture</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def viviani_table(V) -> str:
    rows = []
    for _, r in V.iterrows():
        rows.append(f'<tr><th class="s">{r.En_MeV:.2f} MeV</th>'
                    f'<td class="n">{r.sigma_pair_ub:.4f}</td>'
                    f'<td class="n">{r.sigma_gamma_ub:.1f}</td>'
                    f'<td class="n">{r.sigma_pair_ub / r.sigma_gamma_ub:.2e}</td>'
                    f'</tr>')
    return ('<table class="t"><thead><tr><th>E<sub>n</sub></th>'
            '<th>&sigma;(n,e&#8314;e&#8315;) [&mu;b]</th>'
            '<th>&sigma;(n,&gamma;) [&mu;b]</th><th>ratio</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def time_table(E) -> str:
    rows = []
    for _, r in E.iterrows():
        rows.append(f'<tr><th class="s">{r.t_ms:g} ms</th>'
                    f'<td class="n">{r.En_eV:.3g}</td>'
                    f'<td class="n">{r.dW_over_W:.1e}</td>'
                    f'<td class="n">exactly 1</td><td class="n">exactly 1</td>'
                    f'<td class="n">{r.p_wave_over_s_wave:.1e}</td></tr>')
    return ('<table class="t"><thead><tr><th>arrival time</th>'
            '<th>E<sub>n</sub> [eV]</th>'
            '<th>&Delta;W/W<br><span class="u">transition energy</span></th>'
            '<th>E0:M1<br><span class="u">relative to 1 ms</span></th>'
            '<th>Al:&sup3;He<br><span class="u">relative to 1 ms</span></th>'
            '<th>p-wave<br><span class="u">relative to 0.17 MeV</span></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def al_lines_table(T) -> str:
    rows = []
    for _, r in T.iterrows():
        assign = (f'<code>{r.final_jpi}</code> &rarr; {r.multipole}'
                  if r.multipole != 'unassigned'
                  else f'<span style="color:var(--ink-3)">unplaced, taken as '
                       f'{r.mult_used}</span>')
        rows.append(f'<tr><th class="s">{r.e_gam:.1f} keV'
                    f'<span class="why">{assign}</span></th>'
                    f'<td class="n">{100 * r.intensity:.2f}%</td>'
                    f'<td class="n">{r.alpha_pair:.2e}</td>'
                    f'<td class="n">{100 * r.frac_gt109:.1f}%</td>'
                    f'<td class="n">{1e6 * r.pairs_gt109_per_capture:.2f}</td>'
                    f'<td class="n"><b>{100 * r.share_of_gt109:.1f}%</b></td></tr>')
    return ('<table class="t"><thead><tr><th>&gamma; line</th>'
            '<th>per capture</th><th>&alpha;<sub>pair</sub></th>'
            '<th>beyond 109&deg;</th>'
            '<th>wide pairs per capture [&times;10&#8315;&#8310;]</th>'
            '<th>share of the wide-angle yield</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def capsule_table(C) -> str:
    """Aluminium against carbon fibre: captures in, wide-angle pairs out."""
    rows = []
    for _, r in C.iterrows():
        rows.append(f'<tr><th class="s">{r.species}</th>'
                    f'<td class="n">{100 * r.share_of_captures:.0f}%</td>'
                    f'<td class="n">{r.pairs_per_capture:.2e}</td>'
                    f'<td class="n">{r.median_deg:.0f}&deg;</td>'
                    f'<td class="n">{1e6 * r.gt109_per_capture:.0f}</td>'
                    f'<td class="n"><b>{100 * r.share_of_gt109:.0f}%</b></td></tr>')
    return ('<table class="t"><thead><tr><th>in the wall</th>'
            '<th>share of the wall\'s captures</th>'
            '<th>pairs per capture</th><th>median &theta;</th>'
            '<th>wide pairs per capture [&times;10&#8315;&#8310;]</th>'
            '<th>share of the wall\'s wide-angle pairs</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def rate_table(R) -> str:
    rows = []
    for _, r in R.iterrows():
        rows.append(f'<tr><th class="s">{r.variant}<span class="why">{r.note}'
                    f'</span></th>'
                    f'<td class="n">{r.al_captures:.2e}</td>'
                    f'<td class="n">{r.he3_radiative:.2e}</td>'
                    f'<td class="n"><b>{_ratio(r.ratio)}</b></td></tr>')
    return ('<table class="t"><thead><tr><th>how the capture counts are got</th>'
            '<th>Al+C captures<br><span class="u">per neutron</span></th>'
            '<th>&sup3;He radiative captures<br><span class="u">per neutron'
            '</span></th><th>wide-angle Al pairs per wide-angle &sup3;He pair</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def scat_table(S) -> str:
    """The capsule optical-depth bookkeeping, analytic against the rate table."""
    rows = []
    for _, r in S.iterrows():
        rows.append(f'<tr><th class="s">{r.quantity}</th>'
                    f'<td class="n">{r.value:.4g}</td></tr>')
    return ('<table class="t"><thead><tr><th>quantity</th><th>value per '
            'neutron entering the cell</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def missing_table(M) -> str:
    rows = []
    for _, r in M.iterrows():
        rows.append(f'<tr><th class="s">{r["what is missing"]}</th>'
                    f'<td class="n" style="white-space:nowrap">'
                    f'<b>{r["how much it moves"]}</b></td>'
                    f'<td style="font-size:13px;line-height:1.55">'
                    f'{r["why it matters"]}</td></tr>')
    return ('<table class="t"><thead><tr><th>what is missing</th>'
            '<th>how much it could move</th><th>why it matters</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def _number_sections(html: str) -> str:
    """Renumber the ``<h2>`` counters in document order.

    The numbers used to be typed into the template, and inserting a section in
    the middle silently produced 1 2 3 5 4 6. They are a navigation aid, not
    content, so they are computed here from the order the headings actually
    appear in and nowhere else.
    """
    import re as _re
    n = [0]

    def sub(m):
        n[0] += 1
        return f'<h2><span class="n">{n[0]}</span>'
    return _re.sub(r'<h2><span class="n">\d+</span>', sub, html)


def build_html(V, T, CH, S, SUM, E, P, LINES, TOP, YLD, BK, SCAT, RATE,
               SC, CS, MISS) -> str:
    m1 = T[T.multipole == 'M1'].iloc[0]
    e0 = T[T.multipole == 'E0'].iloc[0]
    e1 = T[T.multipole == 'E1'].iloc[0]
    y_m1 = IB.grid_spectrum('M1', IB.E_TRANSITION)
    y_e0 = IB.grid_spectrum('E0', IB.E_TRANSITION)
    y_e1 = IB.grid_spectrum('E1', IB.E_TRANSITION)
    tot = P.total.to_numpy()
    wdeg = np.diff(IB.THETA_BINS)
    med = float(np.interp(0.5, np.cumsum(tot * wdeg), IB.THETA_MID))
    e_win = IC.window_energy(1.0)
    tv = E.attrs['spectrum_tv_across_window']

    yld_m1 = YLD[YLD.unassigned_taken_as == 'M1'].iloc[0]
    yld_e1 = YLD[YLD.unassigned_taken_as == 'E1'].iloc[0]
    cap_gt109 = float((CS.share_of_captures * CS.gt109_per_capture).sum())
    cap_per_cap = float((CS.share_of_captures * CS.pairs_per_capture).sum())
    he_per_radcap = RATE.attrs['he_pairs_per_radcap']
    he_gt109_per_radcap = RATE.attrs['he_gt109_per_radcap']
    ratio_lo, ratio_hi = RATE.ratio.min(), RATE.ratio.max()
    p_abs = BK.attrs['p_abs_he3']
    al_after = SC.capsule_after_wall.to_numpy()
    he_after = SC.he3_after_wall.to_numpy()
    al_birth = SC.capsule_birth.to_numpy()
    al_med = float(np.interp(0.5, np.cumsum(al_birth * np.diff(IB.THETA_BINS)),
                             IB.THETA_MID))
    c_share = float(CS.loc[CS.species == '12C', 'share_of_gt109'].iloc[0])
    x17 = (IB.THETA_MID > 109) & (IB.THETA_MID < 145)
    sep = al_after[x17] / he_after[x17]
    n_scat = float(SCAT.loc[SCAT.quantity.str.startswith('the same, as the rate'),
                            'value'].iloc[0])

    return _number_sections(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>The internal-pair spectrum, and what the capsule adds to it</title>
{FONT_LINK}
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">IPC MODELLING</span>
    <span>n_TOF X17 &middot; sept26 preliminary</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>The internal-pair spectrum, and what the capsule adds to it</h1>
  <p class="sub">A Born multipole calculation instead of an ansatz &middot;
     the spectrum, not a threshold &middot; it does not move with arrival time
     &middot; and the aluminium, line by line</p>
</header>

<p class="lede"><b>The continuum is calculable, it is the same curve at
1&nbsp;ms and at 1&nbsp;s, and the aluminium capsule makes far more of it than
the gas does.</b> For a Z&nbsp;=&nbsp;2 nucleus at 20.6&nbsp;MeV the
one-photon-exchange calculation is essentially exact (&alpha;Z&nbsp;=&nbsp;0.015)
and gives one closed-form curve per multipole. Below 2&nbsp;eV &mdash; every
neutron this analysis sees &mdash; only two channels are open, and the
prediction is their sum: a continuum with a median opening angle of
<b>{med:.0f}&deg;</b> and <b>{100 * SUM['frac_gt109_mixed']:.1f}&nbsp;%</b> of
its pairs beyond 109&deg;. Nothing about it depends on when the neutron
arrives, to better than <b>{tv:.0e}</b> in total variation across the whole
window. What is <i>not</i> settled is how much of the E0 channel there is, and
&mdash; the bigger problem &mdash; that the capsule wall contributes
<b>{_ratio(ratio_lo)}&ndash;{_ratio(ratio_hi)}</b> as many wide-angle pairs as
the gas, in a spectrum of almost the same shape.</p>

<div class="cards">
  <div class="card"><div class="v">{med:.0f}&deg;</div>
    <div class="l">median opening angle of the predicted &sup3;He continuum
      below 2&nbsp;eV</div></div>
  <div class="card"><div class="v">{tv:.0e}</div>
    <div class="l">how much that spectrum changes between 1&nbsp;ms and
      1&nbsp;s of flight</div></div>
  <div class="card"><div class="v">{al_med:.0f}&deg;</div>
    <div class="l">median for the capsule-wall continuum &mdash; close enough
      that shape will not separate them</div></div>
  <div class="card"><div class="v">{_ratio(ratio_lo)}&ndash;{_ratio(ratio_hi)}</div>
    <div class="l">wide-angle capsule pairs per wide-angle gas pair, and the
      range is one unverified normalisation</div></div>
</div>

<h2><span class="n">1</span>What is being replaced, and why it is replaceable</h2>
<p><code>pair_physics.py</code> samples the virtual photon's invariant mass from
<code>dN/dM ~ 1/M</code> and decays it isotropically, because that is what the
Geant4 primary generator does, and it brackets that with three variants. The
module is honest about the ansatz being an ansatz. But the spread it produces
is not a physics uncertainty &mdash; it is the spread of four arbitrary curves,
and it is wider than anything the physics supports and centred in the wrong
place.</p>

<p>Internal pair creation is one-photon exchange. Write the transition energy
<code>W</code>, the pair mass <code>M</code>, the photon three-momentum
<code>k = &radic;(W&sup2; &minus; M&sup2;)</code> and the lepton angle
<code>&theta;*</code> in the pair rest frame. The QED half is exact:</p>
<div class="panel"><p style="max-width:none;margin:0"><code>
d&Gamma; / (dM&sup2; dcos&theta;*) = (1/M&#8308;) [ N<sub>T</sub>(k)&nbsp;S<sub>T</sub>
+ N<sub>L</sub>(k)&nbsp;S<sub>L</sub> ] &middot; k&nbsp;&beta;*
</code><br><code>
S<sub>T</sub> = 2M&sup2;[(1+cos&sup2;&theta;*) + (1&minus;&beta;*&sup2;)sin&sup2;&theta;*]
&nbsp;&nbsp;&nbsp; S<sub>L</sub> = 2M&sup2;[1 &minus; &beta;*&sup2;cos&sup2;&theta;*]
</code></p></div>
<p>and the nuclear half is two numbers per multipole, in the long-wavelength
limit:
<code>M&lambda;: N<sub>T</sub> = k<sup>2&lambda;</sup>, N<sub>L</sub> = 0</code>;
<code>E&lambda;: N<sub>T</sub> = ((&lambda;+1)/&lambda;)W&sup2;k<sup>2&lambda;&minus;2</sup>,
N<sub>L</sub> = M&sup2;k<sup>2&lambda;&minus;2</sup></code>;
<code>E0: N<sub>T</sub> = 0, N<sub>L</sub> = M&sup2;k&sup2;</code>.
Everything on this page follows from those five expressions.</p>

<h2><span class="n">2</span>The one picture the rest of the page rests on</h2>
<p>Before any of that means anything: <b>a pair is always emitted back-to-back
in the virtual photon's own rest frame.</b> What sets the angle you actually
measure is how fast that frame is moving, and its Lorentz factor is
<code>&gamma; = W/M</code>. A light virtual photon is fast and throws both
tracks forward; a heavy one is barely moving and the pair stays open.</p>
{DG.mechanism()}
<p>That gives a hard floor. For a symmetric pair,
<code>cos&theta;<sub>min</sub> = 1 &minus; 2M&sup2;/W&sup2;</code>, and no decay
angle can beat it. Read backwards it is the entire X17 argument: to reach
109&deg; from a 20.58&nbsp;MeV transition the pair must carry at least
16.8&nbsp;MeV of invariant mass, which is exactly the mass the anomaly is
claimed at.</p>
{figure('ipc_kinematics',
        'The kinematic floor, and the median an isotropic decay actually '
        'produces. Everything else on this page is a weighting of this curve '
        'by how often the transition supplies each mass &mdash; which is what '
        'a multipole is.',
        'minimum and median lab opening angle against virtual photon mass')}
<p class="note">So the question &ldquo;what does the IPC continuum look
like?&rdquo; is not a question about angles at all. It is the question
<i>which masses is the virtual photon allowed to have</i>, and that is fixed by
the multipole, in closed form, with no free parameters.</p>

<h2><span class="n">3</span>Three multipoles, three different curves</h2>
{figure('ipc_shapes',
        'The Born opening-angle law for each multipole against the band '
        'pair_physics.py currently carries. E0 is not a falling continuum at '
        'all &mdash; it peaks near 60&deg; and is the only curve that does not '
        'collapse in the X17 region. The four-ansatz band brackets these '
        'curves, but for none of the right reasons: its lower edge (1/M³) '
        'corresponds to no multipole and its upper edge over-predicts M1.',
        'opening-angle distributions per multipole against the ansatz band')}

{spectrum_table({'E0': y_e0, 'M1': y_m1, 'E1': y_e1},
                'Normalised dN/d&theta; in 1/deg, per 15&deg; bin, at '
                'W&nbsp;=&nbsp;20.58&nbsp;MeV. The last two columns are E0. '
                'The full 1&deg; spectrum for every curve is in the CSV '
                'beside each figure; nothing on this page is quoted from '
                'anywhere else.')}

<div class="scroll">{multipole_table(T)}</div>

<p class="note"><b>Why E0 is different in kind.</b> A 0&#8314;&rarr;0&#8314;
transition has no transverse amplitude, and its Coulomb amplitude is
<i>flat in momentum transfer</i>: the monopole charge matrix element goes as
<code>k&sup2;</code> and the Coulomb propagator as <code>1/k&sup2;</code>, so
the two cancel and the nucleus can absorb any recoil at no cost. The pair is
then unconstrained in opening angle &mdash; the exact lab law is
<code>(1 + &epsilon;&nbsp;cos&theta;)</code> with
<code>&epsilon; = p&#8314;p&#8315;/(E&#8314;E&#8315; &minus; m&sup2;) &le; 1</code>.
The same cancellation is stated in words by Viviani <i>et al.</i>
(PRC&nbsp;105, 014001): <i>&ldquo;this singularity poses no problem, since
|C0000(q)|&sup2; ~ q&#8308;&rdquo;</i>.</p>

{figure('ipc_mass',
        'The same disagreement in the variable that causes it. The Geant '
        'ansatz (flat in ln M) is the M1 answer with the endpoint phase-space '
        'suppression left out, which is exactly the region that maps onto '
        'large opening angles; the 1/M³ variant, described as a deliberate '
        'extreme, corresponds to no multipole at all. E0 is not a falling '
        'mass spectrum &mdash; it peaks at M ≈ 0.63 W.',
        'virtual photon mass spectra per multipole against the two ansatze')}

{figure('ipc_density',
        'The same three multipoles in the plane they live in. M1 hugs the low '
        'masses, so it is stuck below the X17 band; E1 reaches further because '
        'its strength is set by the transition energy rather than by the '
        'momentum transfer; E0 sits at HIGH mass and rides the kinematic floor '
        'straight through the X17 region. This is the mechanism the curves '
        'above are a projection of.',
        'two-dimensional pair density in mass and opening angle per multipole')}

<h2><span class="n">5</span>Why this is trustworthy: three exact checks</h2>
<p>The framework is not calibrated to anything, so it has to be checked against
results derived independently of it.</p>
<div class="scroll">{val_table(V)}</div>
<p class="note">The first rows compare the virtual-photon machinery against the
E0 distribution derived a completely different way &mdash; as a local
<code>&psi;&#772;&gamma;&#8304;&psi;</code> contact operator, which gives the
lab-frame <code>(1 + &epsilon; cos&theta;)</code> law directly &mdash; and then
integrate over angle and compare the positron-energy spectrum against
Wilkinson's published E0 pair integrand,
<code>p&#8314;p&#8315;(E&#8314;E&#8315; &minus; &gamma;&sup2;)F(Z,E&#8314;)F(Z,E&#8315;)</code>
(Nucl.&nbsp;Phys.&nbsp;A133 (1969) 1, reproduced as Eq.&nbsp;18 of Dowie
<i>et al.</i>, arXiv:1911.00031); at Z&nbsp;=&nbsp;2 the Fermi functions are 1
to 0.1&nbsp;%. The two new rows are the reason this page can quote curves at
all: the spectra are now computed by quadrature rather than sampled, and they
have to agree with the sampled ones over the <i>whole</i> distribution, in
total variation, not at one threshold.</p>
<div class="caution"><b>What is <i>not</i> validated.</b> Nothing here has been
compared against a measured IPC angular correlation, because no measurement of
this transition exists &mdash; that is the experiment. The nearest external
anchor is the E0 pair correlation of the 6.05&nbsp;MeV 0&#8314; state in
&sup1;&#8310;O, measured repeatedly since 1949 &mdash; but that is Z&nbsp;=&nbsp;8,
where the Coulomb distortion of the Born form is already visible, which is
precisely why ours at Z&nbsp;=&nbsp;2 is the easy case rather than a check on
it. <b>Aluminium is not the easy case</b>: &alpha;Z&nbsp;=&nbsp;0.095, and
&sect;6 carries a Born calculation there anyway, with a ~10&nbsp;% label on it.
The Siegert factor <code>(&lambda;+1)/&lambda;</code> in the E&lambda;
transverse strength is the one place the module leans on a long-wavelength
relation rather than on kinematics; it moves E1's transverse:longitudinal ratio
by two and nothing else.</div>

<h2><span class="n">4</span>The window decides which multipoles exist</h2>
<p>Nothing is recorded before 1&nbsp;ms, and over the 19.5&nbsp;m EAR2 flight
path 1&nbsp;ms is <b>{e_win:.1f}&nbsp;eV</b>. Every neutron in this analysis is
thermal or epithermal, and that has two consequences that are not small.</p>
<p><b>First, only s-wave survives.</b> A p-wave capture cross section falls as
<code>v</code> relative to the 1/v s-wave, so between the lowest energy Viviani
<i>et al.</i> tabulate (0.17&nbsp;MeV) and our window the 1&#8315; resonance
that dominates every number in their Table&nbsp;V is down by
<b>{IC.p_wave_suppression(e_win):.0e}</b>. It is not there.</p>
<p><b>Second, n&nbsp;+&nbsp;&sup3;He has two s-wave entrance channels and only
one of them can emit a photon.</b> Both particles are spin-&#189;, so
J&nbsp;=&nbsp;0&#8314; and 1&#8314;, both positive parity, and
&#8308;He's ground state is 0&#8314;:</p>
{DG.channels()}
<ul>
<li><b>1&#8314; (&sup3;S&#8321;) &rarr; 0&#8314;</b> is M1. This is the radiative
capture and its cross section is measured: 55&nbsp;&plusmn;&nbsp;3&nbsp;&mu;b at
thermal. Its pair yield follows from
&alpha;<sub>pair</sub>(M1)&nbsp;=&nbsp;{m1.alpha_pair:.2e}.</li>
<li><b>0&#8314; (&sup1;S&#8320;) &rarr; 0&#8314;</b> is E0. A 0&#8314;&rarr;0&#8314;
transition <i>cannot</i> emit a real photon, so this channel produces
e&#8314;e&#8315; and nothing else. Its yield is not bounded by, or even related
to, the measured (n,&gamma;) cross section &mdash; and it is not a rare corner
of the reaction. The 5333&nbsp;b of thermal &sup3;He(n,p)&sup3;H is the broad
J&#8317;&nbsp;=&nbsp;0&#8314; state of &#8308;He, and because that resonance is
open <i>only</i> in the 0&#8314; channel the absorption is strongly spin
dependent &mdash; which is the entire operating principle of a &sup3;He
<b>neutron spin filter</b>. So the channel with no photons is the one the
neutron overwhelmingly goes into.</li>
</ul>
{figure('ipc_thermal',
        'The prediction for the >1 ms window: M1 from the 1⁺ channel plus E0 '
        'from the 0⁺ channel. The shaded band is the E0 fraction swept over '
        'the full range the §5 sensitivity table allows (6–52 %), which is '
        'the only free parameter left in the ³He half of this page.',
        'thermal 3He IPC prediction decomposed into M1 and E0')}
<div class="scroll">{channel_table(CH)}</div>

{spectrum_table({'M1 alone': y_m1, 'E0 alone': y_e0, 'the prediction': tot},
                'The predicted &sup3;He spectrum below 2&nbsp;eV. Last two '
                'columns are the M1-alone curve; the CSV beside the figure '
                'carries all three at 1&deg;, weighted and unweighted.')}

<h2><span class="n">6</span>Does any of it depend on when the neutron arrives?</h2>
<p><b>No, and by a wide margin.</b> This matters more than it sounds: if the
expected spectrum were a function of time of flight, every fit would have to
track it and arrival time would stop being available as a handle against other
backgrounds. Four things could make it move, and all four are evaluated rather
than argued.</p>
<div class="scroll">{time_table(E)}</div>
<p class="note">The two middle columns are <i>exactly</i> 1, not
approximately: the E0:M1 mix is a ratio of two s-wave channels that both go as
1/v, and the Al-to-&sup3;He capture ratio is a ratio of two 1/v absorbers, so
the velocity cancels identically rather than to some order. That cancellation
survives until the first &sup2;&#8311;Al resonance at
{E.attrs['first_al_resonance_eV']:.0f}&nbsp;eV, which is 34&nbsp;&mu;s of
flight &mdash; three orders of magnitude earlier than the flash veto lets
anything through. The transition energy does move, by
<code>(3/4)E<sub>n</sub></code>, which at the top of the window is 1.5&nbsp;eV
on 20.58&nbsp;MeV. Folding that through the Born calculation moves the whole
spectrum by <b>{tv:.0e}</b> in total variation.</p>
{figure('ipc_time',
        'Left: the predicted spectrum at seven arrival times spanning three '
        'decades of neutron energy. There are seven curves in that panel. '
        'Right: the two effects that are not identically zero, against the '
        'dashed line at 1 where the two exact cancellations sit.',
        'the predicted opening angle spectrum at seven arrival times, identical')}
<div class="panel"><p style="margin:0"><b>One template covers the whole
window.</b> Any binning in time-of-flight can be used freely against
backgrounds that <i>do</i> vary with it &mdash; accidentals, beam-correlated
noise, the tail of the flash &mdash; without the signal model having to
change.</p></div>

<h2><span class="n">7</span>Where 2.1&times;10&#8315;&sup3; came from, and why it
does not apply here</h2>
<p>The rate calculation
(<code>calculation_tables/results_3He</code>) uses
<code>IPC/capture = 2.1&times;10&#8315;&sup3;</code>, and the INTC proposal's
7&times;10&#8308; expected IPC background is the same number. It is Viviani
<i>et al.</i>'s ratio of their two total cross sections, Table&nbsp;V of
PRC&nbsp;105, 014001 &mdash; and it is beautifully stable across their whole
tabulated range, which is exactly why it is easy to carry too far:</p>
<div class="scroll">{viviani_table(IC.VIVIANI_TABLE_V)}</div>
<p>Every row there is dominated by the p-wave 1&#8315; resonance, in both
numerator and denominator, which is why the ratio barely moves. Our window is
five decades below the first row and contains none of that. Redoing the same
ratio out of the two s-wave channels gives
<b>{SUM['ipc_per_gamma_capture']:.1e}</b> pairs per radiative capture &mdash;
about {SUM['ipc_per_gamma_capture'] / SUM['ipc_per_gamma_capture_table']:.1f}
times the number in use &mdash; and, more importantly, a different shape.</p>

<div class="caution"><b>The E0 number is an order of magnitude, and is labelled
as one.</b> It uses a single-level ratio
<code>&sigma;<sub>pair</sub>/&sigma;<sub>np</sub> = &Gamma;<sub>pair</sub>/&Gamma;<sub>tot</sub></code>
with three quoted inputs: the &#8308;He monopole transition matrix element
M(E0)&nbsp;=&nbsp;1.53&nbsp;&plusmn;&nbsp;0.05&nbsp;fm&sup2; measured in (e,e&prime;)
(arXiv:2306.07268, Table&nbsp;I), the 0.50&nbsp;MeV total width of the
20.21&nbsp;MeV 0&#8314; state, and 5333&nbsp;b of thermal (n,p) taken as
all-singlet. That gives
&Gamma;<sub>pair</sub>(E0)&nbsp;=&nbsp;{1e6 * SUM['e0_pair_width_eV']:.1f}&nbsp;&mu;eV.
The 20.21&nbsp;MeV state sits 0.37&nbsp;MeV <i>below</i> the n&nbsp;+&nbsp;&sup3;He
threshold, so at 2&nbsp;eV a single-level Breit&ndash;Wigner is being asked to do
real work &mdash; and the same paper that supplies M(E0) is about the
<b>&alpha;-particle monopole puzzle</b>: <i>ab initio</i> theory misses this
form factor, so the structure behind the number is itself an open problem. Read
the answer as &ldquo;tens of percent of the pair yield&rdquo;, not as a
prediction.</div>
<div class="scroll">{sens_table(S)}</div>

<h2><span class="n">8</span>The capsule, line by line</h2>
<p>The capsule is 0.5&nbsp;mm of aluminium and 1.2&nbsp;mm of carbon fibre, and
&sup2;&#8311;Al captures thermal neutrons at 0.231&nbsp;b against the gas's
55&nbsp;&mu;b radiative channel. The estimate this page used to carry took the
two hard primaries near 7.7&nbsp;MeV, assumed E1, and carried a factor-ten
bracket on their branching because nobody had pulled the numbers. The branchings
are now pulled &mdash; {len(LINES)} prompt lines from the IAEA PGAA catalogue
with the EGAF level scheme behind them (<code>data/nuclear/</code>) &mdash; and
they say the old estimate was looking at the wrong lines.</p>

<p><b>The hard primaries are M1, not E1.</b> &sup2;&#8311;Al is 5/2&#8314; and
s-wave capture makes the 7725&nbsp;keV state 2&#8314; or 3&#8314;. The
7724.0&nbsp;keV primary feeds the <b>3&#8314;</b> ground state of
&sup2;&#8312;Al and the 7693.4&nbsp;keV one feeds the <b>2&#8314;</b> level at
30.6&nbsp;keV. Same parity both times, so both are M1 (or E2, which differs by
under 4&nbsp;% in everything computed here). M1 has a smaller conversion
coefficient and a more collimated pair, so assuming E1 overstated their
wide-angle yield by 3.0&times;.</p>
<p><b>And the wide-angle yield is not theirs anyway.</b> The 2.3&ndash;4.3&nbsp;MeV
primaries feed the negative-parity levels &mdash; 3033.9&nbsp;keV to the
3&#8315; at 4691, 4259.5 to the 4&#8315; at 3465, 4133.4 to the 3&#8315; at
3591 &mdash; so those <i>are</i> E1, and a 3&nbsp;MeV pair is three times less
collimated than a 7.7&nbsp;MeV one.</p>
{DG.al_scheme(LINES)}
{figure('ipc_al_lines',
        'Every 27Al capture line above the pair threshold, weighted by what it '
        'contributes beyond 109°. Colour is the multipole assignment, taken '
        'from the parity of the level the primary feeds. The single tallest '
        'line is the 7724 keV ground-state primary, but the 2–5 MeV group '
        'together is nearly three times it. Aluminium only; the carbon fibre '
        'is added below.',
        'per-line wide-angle pair yield for 27Al thermal capture')}
<div class="scroll">{al_lines_table(TOP)}</div>
<p class="note">Two completeness numbers, both computed from the catalogues
rather than asserted: the placed prompt lines carry
<b>{100 * LINES.attrs['energy_completeness']:.0f}&nbsp;%</b> of
&sigma;&#8320;&times;S<sub>n</sub> in &gamma; energy, and the identified
primaries carry <b>{100 * LINES.attrs['primary_completeness']:.0f}&nbsp;%</b>
of the captures. The rest is weak and unplaced strength, spread over energy,
and it would add to the numbers below rather than subtract.</p>

<p><b>And the wall is not only aluminium.</b> &sup1;&sup2;C captures at
3.53&nbsp;mb against &sup2;&#8311;Al's 231&nbsp;mb, but the carbon fibre carries
7.8&times; the areal density, so it is
{100 * float(CS.loc[CS.species == '12C', 'share_of_captures'].iloc[0]):.0f}&nbsp;%
of the wall's captures &mdash; and per capture it is <i>worse</i>, because both
of its strong primaries (4945&nbsp;keV to the 1/2&#8315; ground state,
1262&nbsp;keV to the 3/2&#8315; at 3685) are E1 and both are soft. It supplies
{100 * c_share:.0f}&nbsp;% of the wall's wide-angle pairs from
{100 * float(CS.loc[CS.species == '12C', 'share_of_captures'].iloc[0]):.0f}&nbsp;%
of its captures.</p>
<div class="scroll">{capsule_table(CS)}</div>

<div class="panel"><p style="margin:0">Per capture anywhere in the wall the
capsule makes <b>{cap_per_cap:.2e}</b> pairs, of which
<b>{1e6 * cap_gt109:.0f}&times;10&#8315;&#8310;</b> land beyond 109&deg;
(aluminium alone would be
{1e6 * yld_m1.pairs_gt109_per_capture:.0f}&times;10&#8315;&#8310;, or
{1e6 * yld_e1.pairs_gt109_per_capture:.0f}&times;10&#8315;&#8310; if every
unassigned aluminium line is taken as E1). For comparison, one &sup3;He
<i>radiative</i> capture makes {he_per_radcap:.2e} pairs and
{1e6 * he_gt109_per_radcap:.0f}&times;10&#8315;&#8310; beyond 109&deg;.
<b>Per capture the two are the same problem.</b> The entire difficulty is that
there are {_ratio(ratio_lo)} to {_ratio(ratio_hi)} times more capsule
captures.</p></div>

<h2><span class="n">9</span>How many of each, and the one number nobody has
checked</h2>
{DG.capsule()}
<p>The &sup3;He cell is 500&nbsp;atm over 4&nbsp;cm. At thermal its optical
depth to &sup3;He(n,p) is <b>{IC.SIGMA_NP_B * AL.N_HE3_ATB * np.sqrt(0.0253 / BK.attrs['en_ev']):.0f}</b>,
so it absorbs <b>{100 * p_abs:.2f}&nbsp;%</b> of every neutron that enters and a
thin-target formula does not apply to it. The signal is the
55&nbsp;&mu;b/5333&nbsp;b radiative branch of those absorptions, which is
{RATE.attrs['he_rad_thin'] / (p_abs * 55e-6 / 5333):.0f}&times; smaller than
the thin-target answer. The rate table appears to use the thin-target answer.
That single choice moves the comparison by two orders of magnitude, so the
answer is given three ways rather than one:</p>
<div class="scroll">{rate_table(RATE)}</div>
<div class="caution"><b>This is the biggest open number on the page, and it is
not a nuclear one.</b> If <code>results_3He</code> really does compute
&sup3;He radiative captures without self-shielding, then every expected IPC and
X17 yield in that table &mdash; and in the INTC proposal that quotes it &mdash;
is high by roughly two orders of magnitude, and this page's aluminium ratio is
correspondingly worse. It needs confirming with whoever produced the table;
nothing here can tell whether the code applies the correction elsewhere. The
opposite bias is real too: the analytic row assumes a neutron crosses the wall
once and never scatters, and the table's own numbers say it scatters
{n_scat:.1f} times in the carbon fibre and aluminium before it gets
anywhere, which raises the capsule capture count by the 6&times; the same table
reports.</div>
<div class="scroll">{scat_table(SCAT)}</div>

<h2><span class="n">10</span>Can the shape tell them apart? Not on its own</h2>
<p>The capsule continuum is a sum over lines &mdash; every &sup2;&#8311;Al and
&sup1;&sup2;C line above the pair threshold, each with its own Born curve at its
own transition energy &mdash; and it comes out <i>wider</i> than the helium one
&mdash; median {al_med:.0f}&deg; against
{float(np.interp(0.5, np.cumsum(SC.he3_birth.to_numpy() * wdeg), IB.THETA_MID)):.0f}&deg;
&mdash; because its pairs are softer. Then it has to escape the wall it was
born in, which is 0.33&nbsp;g/cm&sup2;, and the scattering pushes it wider
still.</p>
{figure('ipc_al_shape',
        'The two continua, at birth and after the capsule wall. The capsule '
        'pair is born inside the wall and crosses on average half of it; the '
        'helium pair is born in the gas and crosses all of it, but at 20.6 MeV '
        'it barely notices. The two curves differ by about a factor of two in '
        'the X17 region and by nothing like enough to fit them apart.',
        'capsule and helium pair spectra at birth and after the capsule wall')}
{spectrum_table({'³He at birth': SC.he3_birth.to_numpy(),
                 '³He after the wall': he_after,
                 'capsule at birth': al_birth,
                 'capsule after the wall': al_after},
                f'Last two columns are the &sup3;He birth spectrum. After '
                f'the wall the capsule curve sits '
                f'{sep.min():.1f}&ndash;{sep.max():.1f}&times; above the '
                f'helium one across the X17 region &mdash; which is the whole '
                f'problem: under a factor of three in shape against a factor '
                f'of 10&#8308;&ndash;10&#8310; in normalisation.')}
<p class="note"><b>Multiple scattering here is a first-order filter, not
transport.</b> Each track is scattered by a Highland-width Gaussian over the
wall thickness remaining ahead of it, and dropped if its Katz&ndash;Penfold
range does not reach the outside &mdash; which removes only
{100 * (1 - SC.attrs['capsule_escape']):.1f}&nbsp;% of the capsule pairs, because
a pair sharing 3&nbsp;MeV rarely puts a track below the 0.8&nbsp;MeV escape
threshold. For {100 * SC.attrs['capsule_unreliable']:.0f}&nbsp;% of the surviving
weight the Highland angle exceeds 1&nbsp;rad, where a Gaussian is not a
description of anything; that fraction is reported rather than hidden, and it
is why this curve is an indication and the real answer is a Geant4 run.</p>

<div class="panel"><p style="margin:0"><b>What would actually separate them is
energy.</b> A wide-angle aluminium pair carries 2&ndash;4&nbsp;MeV between its
two tracks; a &sup3;He one carries 20.6&nbsp;MeV. Nothing in this setup measures
that &mdash; no magnet, no calorimetry. The n_TOF proposal's own detector
carries a 50&nbsp;mT coil for precisely this reason. Without it the handles
left are the vertex (the capsule wall is 2&nbsp;cm off the gas centre) and the
shape, and the shape is worth a factor of two.</p></div>

<h2><span class="n">11</span>What is missing</h2>
<p>Ordered by how much it could move the answer, not by how hard it is.</p>
<div class="scroll">{missing_table(MISS)}</div>

<h2><span class="n">0</span>And at a different beam energy</h2>
<p>Everything above is specific to a beam in which the neutron brings nothing.
Take the same cell to a 1&ndash;40&nbsp;MeV beam and the &#8308;He excitation
becomes <code>20.578 + 0.749 E<sub>n</sub></code>, which moves the X17 opening
angle from 109&deg; down to 39&deg;, raises the gas's radiative branch per
neutron by ~280&times;, and switches the capsule from capture to inelastic
scattering &mdash; where its two strongest lines are <i>below the pair
threshold</i>. There is a window below 2.3&nbsp;MeV in which the capsule
background nearly vanishes. That is a separate page:
<a href="../ganil/report.html">the same experiment at GANIL</a>.</p>

<h2><span class="n">12</span>What to do with this</h2>
<ul>
<li><b>Replace the four-ansatz band with the two-channel spectrum.</b> The
physics variable becomes the E0 fraction, which is a number about the reaction,
not a choice of curve. <code>ipc_channels.thermal_spectrum()</code> is the
curve and <code>ipc_born.mixture()</code> re-mixes it.</li>
<li><b>Use the whole time window with one template.</b> &sect;5 says it is the
same spectrum from 1&nbsp;ms to 1&nbsp;s, so time-of-flight binning is free to
be spent on backgrounds that do vary.</li>
<li><b>Check the &sup3;He self-shielding in the rate table.</b> One question to
one person, and it is worth two orders of magnitude on every expected yield
this experiment quotes.</li>
<li><b>Run the capsule through Geant4 with an Al-capture generator.</b> The
line list in <code>data/nuclear/</code> is exactly the input a primary
generator needs, and it settles the wall transport, the escape, and the
acceptance in one go &mdash; including the possibility that the vertex cuts
kill most of it, which is the most likely way this background shrinks.</li>
<li><b>The intra-chamber sample measures the E0 fraction.</b> E0 and M1 differ
most below 90&deg;, which is precisely the topology the opening-angle page
already treats as the IPC control region &mdash; but it is also where the
aluminium and any external conversion sit, so that measurement needs
&sect;7&ndash;9 folded in before it means anything.</li>
<li><b>Ask for the thermal point.</b> Viviani, Marcucci, Kievsky and Schiavilla
already have the C0000 (&sup1;S&#8320;&rarr;0&#8314;) and M1
(&sup3;S&#8321;&rarr;0&#8314;) reduced matrix elements in the code that produced
PRC&nbsp;105, 014001; they stopped at 0.17&nbsp;MeV because nobody asked for
lower. A run at E<sub>n</sub>&nbsp;&lt;&nbsp;10&nbsp;eV replaces every estimate
in &sect;4 and &sect;6 with an <i>ab initio</i> number.</li>
</ul>

<footer>
Generated by <code>make_ipc_report.py</code> from <code>ipc_born.py</code>,
<code>ipc_channels.py</code> and <code>ipc_aluminium.py</code> &middot; every
figure ships its CSV &middot; {dt.date.today().isoformat()}<br>
Sources: Rose, Phys. Rev. 76 (1949) 678 &middot; Wilkinson, Nucl. Phys. A133
(1969) 1 &middot; Dowie <i>et al.</i>, arXiv:1911.00031 &middot; Viviani
<i>et al.</i>, PRC 105 (2022) 014001 (arXiv:2104.07808) &middot; Kegel
<i>et al.</i> / arXiv:2306.07268 for M(E0) &middot; IAEA PGAA catalogue and
EGAF (Firestone, LBNL 2003) for the &sup2;&#8311;Al capture scheme &middot;
Gustavino, INTC 5&nbsp;Feb&nbsp;2025.
</footer>
</div>
</body>
</html>
""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=N)
    ap.add_argument('--assume', default='M1', choices=('M1', 'E1'))
    a = ap.parse_args()
    od = paths.out('ipc')

    V = IB.validate(min(2 * a.n, 4_000_000))
    T = IB.multipole_table(a.n)
    CH = IC.thermal_channels()
    S = IC.e0_sensitivity()
    SUM = IC.thermal_summary()
    E = IC.energy_invariance()
    P = IC.thermal_spectrum()

    LINES = AL.line_list()
    TOP = AL.top_lines(LINES, a.assume)
    YLD = AL.yield_summary(LINES)
    BK = AL.bookkeeping()
    SCAT = AL.scattering_check()
    RATE = AL.rate_comparison(a.assume)
    SC = AL.shape_comparison(a.assume)
    CS = AL.capsule_summary(a.assume)
    MISS = AL.missing()

    for name, df in (('ipc_born_validation', V), ('ipc_born_multipoles', T),
                     ('ipc_channels_thermal', CH),
                     ('ipc_channels_e0_sensitivity', S),
                     ('ipc_channels_spectrum', P),
                     ('ipc_channels_energy_invariance', E),
                     ('al_line_list', LINES), ('al_top_lines', TOP),
                     ('al_yield_summary', YLD), ('al_bookkeeping', BK),
                     ('al_scattering_check', SCAT),
                     ('al_rate_comparison', RATE),
                     ('al_shape_comparison', SC), ('al_capsule_summary', CS),
                     ('al_missing', MISS)):
        df.to_csv(od / f'{name}.csv', index=False)

    html = build_html(V, T, CH, S, SUM, E, P, LINES, TOP, YLD, BK, SCAT,
                      RATE, SC, CS, MISS)
    out = od / 'report.html'
    out.write_text(html, encoding='utf-8')
    # rerun_chain.sh publishes <dir>/index.html + <dir>/figures; keeping the
    # copy here means the page is publishable straight out of this script.
    (od / 'index.html').write_text(html, encoding='utf-8')
    print(f'wrote -> {out}')
    if not V['pass'].all():
        print('WARNING: a validation row failed; the page says so.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
