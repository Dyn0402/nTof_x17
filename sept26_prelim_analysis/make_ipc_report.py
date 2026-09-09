#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ipc_report.py -- build ``report.html`` for the IPC modelling deep dive.

Answers one question: can the expected internal-pair spectrum be pinned down
better than the four-ansatz band the opening-angle page carries?  Generated
from :mod:`ipc_born` and :mod:`ipc_channels`, never hand-written, so re-running
them moves the prose and the tables together.

    python -m sept26_prelim_analysis.make_ipc_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402
from sept26_prelim_analysis import ipc_channels as IC  # noqa: E402
from sept26_prelim_analysis.make_funnel_report import CSS, FONT_LINK  # noqa: E402

N = 2_000_000


def figure(name: str, caption: str, alt: str = '') -> str:
    import html as _h
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(alt or caption)}">'
            f'</a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


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


def multipole_table(T, A) -> str:
    rows = []
    for _, r in T.iterrows():
        ap = '&mdash;' if r.alpha_pair != r.alpha_pair else f'{r.alpha_pair:.2e}'
        rows.append(f'<tr><th class="s">{r.multipole}'
                    f'<span class="why">{r.what}</span></th>'
                    f'<td class="n">{ap}</td>'
                    f'<td class="n">{r.median_deg:.0f}&deg;</td>'
                    f'<td class="n">{100 * r.frac_gt90:.1f}%</td>'
                    f'<td class="n"><b>{100 * r.frac_gt109:.2f}%</b></td>'
                    f'<td class="n">{100 * r.frac_gt130:.2f}%</td></tr>')
    rows.append('<tr><td colspan="6" style="border-bottom:none;padding-top:18px">'
                '<b>and the four assumptions currently carried as the band</b>'
                '</td></tr>')
    for _, r in A.iterrows():
        rows.append(f'<tr><th class="s" style="color:var(--ink-3)">{r.variant}'
                    f'</th><td class="n">&mdash;</td>'
                    f'<td class="n">{r.median_deg:.0f}&deg;</td>'
                    f'<td class="n">{100 * r.frac_gt90:.1f}%</td>'
                    f'<td class="n">{100 * r.frac_gt109:.2f}%</td>'
                    f'<td class="n">{100 * r.frac_gt130:.2f}%</td></tr>')
    return ('<table class="t"><thead><tr><th>multipole</th>'
            '<th>pairs per photon<br><span class="u">&alpha;<sub>pair</sub>'
            '</span></th><th>median &theta;</th><th>above 90&deg;</th>'
            '<th>above 109&deg;</th><th>above 130&deg;</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def channel_table(T) -> str:
    rows = []
    for _, r in T.iterrows():
        rows.append(f'<tr><th class="s">{r.channel} &nbsp;<code>{r.entrance}</code>'
                    f'<span class="why">{r.note}</span></th>'
                    f'<td class="n">{r.sigma_pair_ub:.3f}</td>'
                    f'<td class="n">{100 * r.share_of_pairs:.0f}%</td>'
                    f'<td class="n">{100 * r.frac_gt109:.1f}%</td>'
                    f'<td class="n"><b>{100 * r.share_gt109:.0f}%</b></td></tr>')
    return ('<table class="t"><thead><tr><th>channel</th>'
            '<th>&sigma;<sub>pair</sub> [&mu;b]</th><th>share of all pairs</th>'
            '<th>its own fraction above 109&deg;</th>'
            '<th>share of the pairs above 109&deg;</th>'
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


def al_table(A) -> str:
    rows = []
    for _, r in A.iterrows():
        ap = '&mdash;' if r.alpha_pair != r.alpha_pair else f'{r.alpha_pair:.2e}'
        pp = ('&mdash;' if r.pairs_gt109_per_photon != r.pairs_gt109_per_photon
              else f'{1e4 * r.pairs_gt109_per_photon:.2f}')
        rows.append(f'<tr><th class="s">{r.source}</th>'
                    f'<td class="n">{r.w_MeV:.2f}</td><td class="n">{ap}</td>'
                    f'<td class="n">{r.median_deg:.0f}&deg;</td>'
                    f'<td class="n">{100 * r.frac_gt109:.1f}%</td>'
                    f'<td class="n"><b>{pp}</b></td></tr>')
    return ('<table class="t"><thead><tr><th>source</th><th>W [MeV]</th>'
            '<th>&alpha;<sub>pair</sub></th><th>median &theta;</th>'
            '<th>above 109&deg;</th>'
            '<th>wide-angle pairs per photon [&times;10&#8315;&#8308;]</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def build_html(V, T, A, CH, S, R, AL, SUM) -> str:
    e0 = T[T.multipole == 'E0'].iloc[0]
    m1 = T[T.multipole == 'M1'].iloc[0]
    e1 = T[T.multipole == 'E1'].iloc[0]
    geant = A[A.variant.str.startswith('geant')].iloc[0]
    spread = A.frac_gt109.max() / A.frac_gt109.min()
    e_win = IC.window_energy(1.0)
    ksup = IC.p_wave_suppression(e_win)

    return f"""<title>How well can the IPC continuum actually be predicted?</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">IPC MODELLING</span>
    <span>n_TOF X17 &middot; sept26 preliminary</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>How well can the internal-pair continuum actually be predicted?</h1>
  <p class="sub">Born multipoles instead of ansätze &middot; two exact
     validations &middot; the &gt;1&nbsp;ms window changes which channels are
     open &middot; and a first look at aluminium</p>
</header>

<p class="lede"><b>Yes, substantially better &mdash; and the remaining
uncertainty is a different kind of thing.</b> The opening-angle page carries
the IPC continuum as a band across four guessed shapes that span a factor of
<b>{spread:.0f}</b> in the fraction above 109&deg;. None of that spread is
irreducible. For a Z&nbsp;=&nbsp;2 nucleus at 20.6&nbsp;MeV the one-photon-exchange
calculation is essentially exact (&alpha;Z&nbsp;=&nbsp;0.015) and gives the
distribution in closed form, one curve per multipole, with
<b>{100 * m1.frac_gt109:.1f}&nbsp;%</b> above 109&deg; for M1,
<b>{100 * e1.frac_gt109:.1f}&nbsp;%</b> for E1 and
<b>{100 * e0.frac_gt109:.1f}&nbsp;%</b> for E0. What is left over is not a
modelling choice at all: it is <i>which multipole the reaction makes</i>, and
below 2&nbsp;eV that is a two-line question with one measured answer and one
missing one.</p>

<div class="cards">
  <div class="card"><div class="v">{100 * m1.frac_gt109:.1f}%</div>
    <div class="l">above 109&deg; for M1 &mdash; the channel that makes the
       20.58&nbsp;MeV photon</div></div>
  <div class="card"><div class="v">{100 * e0.frac_gt109:.1f}%</div>
    <div class="l">above 109&deg; for E0 &mdash; the channel that makes no
       photon at all</div></div>
  <div class="card"><div class="v">{100 * geant.frac_gt109:.1f}%</div>
    <div class="l">what the Geant generator's ansatz gives, for neither
       reason</div></div>
  <div class="card"><div class="v">{100 * SUM['e0_share_of_gt109']:.0f}%</div>
    <div class="l">of the wide-angle ³He continuum that E0 would supply, on the
       estimate in §4</div></div>
</div>

<h2><span class="n">1</span>What is being replaced, and why it is replaceable</h2>
<p><code>pair_physics.py</code> samples the virtual photon's invariant mass from
<code>dN/dM ~ 1/M</code> and decays it isotropically, because that is what the
Geant4 primary generator does, and it brackets that with three variants. The
module is honest about the ansatz being an ansatz. But the spread it produces
is not a physics uncertainty &mdash; it is the spread of four arbitrary curves,
and in the region the X17 search lives in it is a factor of {spread:.0f}, which
is wider than anything the physics supports and centred in the wrong place.</p>

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
Everything below follows from those five expressions.</p>

<h2><span class="n">2</span>Three multipoles, three different curves</h2>
{figure('ipc_shapes',
        'The Born opening-angle law for each multipole against the band '
        'pair_physics.py currently carries. E0 is not a falling continuum at '
        'all &mdash; it peaks near 60&deg; and is the only curve that does not '
        'collapse in the X17 region. The four-ansatz band brackets these '
        'curves, but for none of the right reasons: its lower edge (1/M³) '
        'corresponds to no multipole, and its upper edge over-predicts M1 by '
        f'{geant.frac_gt109 / m1.frac_gt109:.1f} times.',
        'opening-angle distributions per multipole against the ansatz band')}
<div class="scroll">{multipole_table(T, A)}</div>

<p class="note"><b>Why E0 is different in kind.</b> A 0&#8314;&rarr;0&#8314;
transition has no transverse amplitude, and its Coulomb amplitude is
<i>flat in momentum transfer</i>: the monopole charge matrix element goes as
<code>k&sup2;</code> and the Coulomb propagator as <code>1/k&sup2;</code>, so
the two cancel and the nucleus can absorb any recoil at no cost. The pair is
then unconstrained in opening angle &mdash; the exact lab law is
<code>(1 + &epsilon;&nbsp;cos&theta;)</code> with
<code>&epsilon; = p&#8314;p&#8315;/(E&#8314;E&#8315; &minus; m&sup2;) &le; 1</code>
&mdash; which is why {100 * e0.frac_gt109:.0f}&nbsp;% of E0 pairs come out
beyond 109&deg;. The same cancellation is stated in words by Viviani <i>et al.</i>
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

<h2><span class="n">3</span>Why this is trustworthy: two exact checks</h2>
<p>The framework is not calibrated to anything, so it has to be checked against
results derived independently of it. Two are available and both are exact
rather than eyeballed.</p>
<div class="scroll">{val_table(V)}</div>
<p class="note">The first pair of rows compares the virtual-photon machinery
against the E0 distribution derived a completely different way &mdash; as a
local <code>&psi;&#772;&gamma;&#8304;&psi;</code> contact operator, which gives
the lab-frame <code>(1 + &epsilon; cos&theta;)</code> law directly. The third row
integrates the machinery over angle and compares the positron-energy spectrum
against Wilkinson's published E0 pair integrand,
<code>p&#8314;p&#8315;(E&#8314;E&#8315; &minus; &gamma;&sup2;)F(Z,E&#8314;)F(Z,E&#8315;)</code>
(Nucl.&nbsp;Phys.&nbsp;A133 (1969) 1, reproduced as Eq.&nbsp;18 of Dowie
<i>et al.</i>, arXiv:1911.00031); at Z&nbsp;=&nbsp;2 the Fermi functions are 1
to 0.1&nbsp;%, so the comparison is to the bare form. The last two rows check
the normalisation of <code>&alpha;<sub>pair</sub></code> against the standard
soft-virtual-photon limit.</p>
<div class="caution"><b>What is <i>not</i> validated.</b> Nothing here has been
compared against a measured IPC angular correlation, because no measurement of
this transition exists &mdash; that is the experiment. The nearest external
anchor is the E0 pair correlation of the 6.05&nbsp;MeV 0&#8314; state in
&sup1;&#8310;O, measured repeatedly since 1949 &mdash; but that is Z&nbsp;=&nbsp;8,
where the Coulomb distortion of the Born form is already visible, which is
precisely why ours at Z&nbsp;=&nbsp;2 is the easy case rather than a check on
it. The Siegert
factor <code>(&lambda;+1)/&lambda;</code> in the E&lambda; transverse strength
is the one place the module leans on a long-wavelength relation rather than on
kinematics; it moves E1's transverse:longitudinal ratio by two and nothing
else.</div>

<h2><span class="n">4</span>The window decides which multipoles exist</h2>
<p>Nothing is recorded before 1&nbsp;ms, and over the 19.5&nbsp;m EAR2 flight
path 1&nbsp;ms is <b>{e_win:.1f}&nbsp;eV</b>. Every neutron in this analysis is
thermal or epithermal, and that has two consequences that are not small.</p>
<p><b>First, only s-wave survives.</b> A p-wave capture cross section falls as
<code>v</code> relative to the 1/v s-wave, so between the lowest energy Viviani
<i>et al.</i> tabulate (0.17&nbsp;MeV) and our window the 1&#8315; resonance
that dominates every number in their Table&nbsp;V is down by
<b>{ksup:.1e}</b>. It is not there.</p>
<p><b>Second, n&nbsp;+&nbsp;&sup3;He has two s-wave entrance channels and only
one of them can emit a photon.</b> Both particles are spin-&#189;, so
J&nbsp;=&nbsp;0&#8314; and 1&#8314;, both positive parity, and
&#8308;He's ground state is 0&#8314;:</p>
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
neutron overwhelmingly goes into, and the state it goes into is the same
20.21&nbsp;MeV 0&#8314; whose monopole matrix element to the ground state has
been measured in electron scattering. The channel we normalise the pair rate to
is the 55&nbsp;&mu;b one.</li>
</ul>
{figure('ipc_thermal',
        'The prediction for the >1 ms window: M1 from the 1⁺ channel plus E0 '
        'from the 0⁺ channel, in the proportion estimated below. The sum sits '
        'well inside the old band at small angles and well above its centre in '
        'the X17 region, because E0 &mdash; a fifth of the pairs &mdash; '
        'supplies two fifths of everything past 109&deg;.',
        'thermal 3He IPC prediction decomposed into M1 and E0')}
<div class="scroll">{channel_table(CH)}</div>

<h2><span class="n">5</span>Where 2.1&times;10&#8315;&sup3; came from, and why it
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
M(E0)&nbsp;=&nbsp;1.53&nbsp;&plusmn;&nbsp;0.05&nbsp;fm&sup2; measured in (e,e′)
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

<h2><span class="n">6</span>Aluminium &mdash; a first look, and it is not small</h2>
<p>Two facts, and their product is the problem.</p>
{figure('ipc_energy',
        'Wide-angle pairs per photon against transition energy. It is almost '
        'flat: a 7.73 MeV Al capture primary makes 4.0×10⁻⁴ pairs beyond 109° '
        'per photon, a 20.58 MeV ³He one makes 4.3×10⁻⁴. Losing two thirds of '
        'the transition energy costs 8 %, because the pair conversion '
        'coefficient falls but the pairs get less collimated, and the two '
        'effects nearly cancel.',
        'wide-angle pair yield per photon against transition energy')}
<div class="scroll">{al_table(AL)}</div>
<p>The second fact is the capture bookkeeping. In the thermal bin of the same
rate calculation, the capsule (0.5&nbsp;mm Al + carbon fibre) takes
<b>{IC.GC_CAPTURES_PER_PULSE:.2e}</b> captures per pulse against
<b>{IC.HE3_CAPTURES_PER_PULSE:.2f}</b> &sup3;He radiative captures &mdash; a
factor of {IC.GC_CAPTURES_PER_PULSE / IC.HE3_CAPTURES_PER_PULSE:.0e}, essentially
all of it &sup2;&#8311;Al at 0.231&nbsp;b against 55&nbsp;&mu;b. Folding in only
the hard primaries near 7.7&nbsp;MeV and nothing softer:</p>
<div class="panel"><p style="margin:0"><b>The capsule makes
{R.ratio.min():.0f}&ndash;{R.ratio.max():.0f} times as many pairs beyond
109&deg; as the gas does</b>, and this setup has no magnet and no calorimetry,
so the 20.6 vs 7.7&nbsp;MeV that separates them outright is not measured. The
range is the spread of the &sup2;&#8311;Al primary branching, which has not been
pulled from the IAEA PGAA database yet &mdash; the conclusion survives either
end of it.</p></div>
<p class="note"><b>What this does not say.</b> It is a production estimate, not
a prediction of what lands in the sample. Al capture pairs are born in the
capsule wall, so they pass the pointing and DCA cuts just as the gas ones do;
but they are ~7&nbsp;MeV total, so multiple scattering and the two-plastic
trigger treat them differently, and none of that acceptance is folded in here.
The n_TOF proposal's own detector carries a 50&nbsp;mT coil precisely so the
pair's total energy can be cut on; without it, the only handle left on this
background is the opening-angle shape &mdash; and §6's figure says the shapes
at 7.7 and 20.6&nbsp;MeV are <i>similar</i>, which is the bad news.</p>

<h2><span class="n">7</span>What to do with this</h2>
<ul>
<li><b>Replace the four-ansatz band with a two-channel one.</b> The physics
variable becomes the E0 fraction, which is a number about the reaction, not a
choice of curve. <code>ipc_born.mixture({{'E0': f, 'M1': 1-f}})</code> does it.</li>
<li><b>The intra-chamber sample measures the mix.</b> E0 and M1 differ most
below 90&deg; &mdash; E0 peaks near 60&deg;, M1 is falling monotonically &mdash;
which is precisely the topology the opening-angle page already treats as the
IPC control region. That turns the E0 fraction from an assumption into a fit
parameter with data behind it.</li>
<li><b>Ask for the thermal point.</b> Viviani, Marcucci, Kievsky and Schiavilla
already have the C0000 (&sup1;S&#8320;&rarr;0&#8314;) and M1 (&sup3;S&#8321;&rarr;0&#8314;)
reduced matrix elements in the code that produced PRC&nbsp;105, 014001; they
stopped at 0.17&nbsp;MeV because nobody asked for lower. A run at
E<sub>n</sub>&nbsp;&lt;&nbsp;10&nbsp;eV replaces every estimate in §4 and §5
with an <i>ab initio</i> number, and it is a phone call, not a project.</li>
<li><b>Cost the aluminium properly.</b> Pull the &sup2;&#8311;Al primary
branchings, run the capsule geometry through the existing Geant setup with an
Al-capture generator, and put the result on the opening-angle page as a second
component. Until then the IPC curve on that page is the smaller of the two
things it is being compared against.</li>
</ul>

<footer>
Generated by <code>make_ipc_report.py</code> from <code>ipc_born.py</code> and
<code>ipc_channels.py</code> &middot; every figure ships its CSV &middot;
{dt.date.today().isoformat()}<br>
Sources: Rose, Phys. Rev. 76 (1949) 678 &middot; Wilkinson, Nucl. Phys. A133
(1969) 1 &middot; Dowie <i>et al.</i>, arXiv:1911.00031 &middot; Viviani
<i>et al.</i>, PRC 105 (2022) 014001 (arXiv:2104.07808) &middot; Kegel
<i>et al.</i> / arXiv:2306.07268 for M(E0) &middot; Gustavino, INTC
5&nbsp;Feb&nbsp;2025.
</footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=N)
    a = ap.parse_args()
    od = paths.out('ipc')

    V = IB.validate(min(2 * a.n, 4_000_000))
    T = IB.multipole_table(a.n)
    A = IB.ansatz_table(a.n)
    CH = IC.thermal_channels()
    S = IC.e0_sensitivity()
    R = IC.aluminium_ratio()
    AL = IC.aluminium_comparison()
    SUM = IC.thermal_summary()

    for name, df in (('ipc_born_validation', V), ('ipc_born_multipoles', T),
                     ('ipc_born_ansatz', A), ('ipc_channels_thermal', CH),
                     ('ipc_channels_e0_sensitivity', S),
                     ('ipc_channels_al', AL), ('ipc_channels_al_ratio', R)):
        df.to_csv(od / f'{name}.csv', index=False)

    html = build_html(V, T, A, CH, S, R, AL, SUM)
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
