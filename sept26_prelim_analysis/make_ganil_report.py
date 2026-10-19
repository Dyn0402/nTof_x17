#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ganil_report.py -- ``report.html`` for the NFS/GANIL background study.

Answers one question: if this apparatus were taken to a 1-40 MeV neutron beam,
what would the 3He and capsule pair backgrounds look like, and would the
measurement be easier or harder?  Generated from :mod:`ganil_background`,
:mod:`endf` and :mod:`ipc_born`, so re-running them moves the prose and the
verdict together.

    python -m sept26_prelim_analysis.make_ganil_report
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
from sept26_prelim_analysis import ipc_diagrams as DG  # noqa: E402
from sept26_prelim_analysis import ganil_background as G  # noqa: E402
from sept26_prelim_analysis.make_funnel_report import CSS, FONT_LINK  # noqa: E402
from sept26_prelim_analysis.make_ipc_report import (  # noqa: E402
    figure, missing_table, _number_sections, _ratio)


def scan_table(S) -> str:
    rows = []
    for _, r in S.iterrows():
        band = (' style="background:color-mix(in srgb,var(--good) 8%,transparent)"'
                if r.En_MeV < G.quiet_band()[1] else '')
        ratio = ('&mdash;' if not np.isfinite(r.capsule_over_he3)
                 else _ratio(r.capsule_over_he3))
        rows.append(
            f'<tr{band}><th class="s">{r.En_MeV:g} MeV'
            f'<span class="why">E<sub>x</sub> = {r.Ex_MeV:.1f} MeV</span></th>'
            f'<td class="n">{r.theta_min_deg:.0f}&ndash;{r.window_hi_deg:.0f}&deg;</td>'
            f'<td class="n">{100 * r.x17_in_window:.0f}%</td>'
            f'<td class="n">{r.he3_in_window:.2e}</td>'
            f'<td class="n">{r.capsule_in_window:.2e}</td>'
            f'<td class="n"><b>{ratio}</b></td>'
            f'<td class="n">{100 * r.discrete_fraction_Al:.0f}%</td></tr>')
    return ('<table class="t"><thead><tr><th>neutron energy</th>'
            '<th>signal window</th>'
            '<th>X17 inside it<br><span class="u">if it exists</span></th>'
            '<th>&sup3;He pairs in the window<br><span class="u">per neutron'
            '</span></th>'
            '<th>capsule pairs in the window<br><span class="u">per neutron'
            '</span></th><th>ratio</th>'
            '<th>&sup2;&#8311;Al strength in named levels</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def gas_table(H) -> str:
    rows = []
    for _, r in H.iterrows():
        lab = ('thermal (n_TOF)' if r.En_MeV < 1e-3
               else f'{r.En_MeV:g} MeV')
        flag = (' <span style="color:var(--warn)">extrapolated</span>'
                if r.beyond_evaluation else '')
        rows.append(f'<tr><th class="s">{lab}{flag}</th>'
                    f'<td class="n">{1e6 * r.sigma_ngamma_b:.1f}</td>'
                    f'<td class="n">{r.sigma_np_b:.3g}</td>'
                    f'<td class="n">{r.tau_absorption:.3g}</td>'
                    f'<td class="n"><b>{r.radiative_per_neutron:.2e}</b></td>'
                    f'<td class="n">{_ratio(float(r.np_per_radiative))}</td></tr>')
    return ('<table class="t"><thead><tr><th>neutron energy</th>'
            '<th>&sigma;(n,&gamma;) [&mu;b]</th><th>&sigma;(n,p) [b]</th>'
            '<th>optical depth of the cell</th>'
            '<th>radiative captures per neutron entering</th>'
            '<th>(n,p) two-prongs per radiative capture</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def lines_table(C, n: int = 10) -> str:
    rows = []
    for _, r in C.head(n).iterrows():
        note = ('<span style="color:var(--warn)">below the pair threshold '
                '&mdash; makes no pairs at all</span>' if r.below_threshold
                else f'taken as {r.multipole}')
        rows.append(f'<tr><th class="s">{r.nuclide} {r.source}'
                    f'<span class="why">{note}</span></th>'
                    f'<td class="n">{r.e_gamma_MeV:.3f}</td>'
                    f'<td class="n">{1e3 * r.sigma_b:.1f}</td>'
                    f'<td class="n">{1e3 * r.per_neutron:.2f}</td></tr>')
    return ('<table class="t"><thead><tr><th>source</th>'
            '<th>E<sub>&gamma;</sub> [MeV]</th><th>&sigma; [mb]</th>'
            '<th>photons per 1000 neutrons</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def compare_table(N) -> str:
    rows = []
    for _, r in N.iterrows():
        rows.append(f'<tr><th class="s">{r.quantity}</th>'
                    f'<td>{r.ntof}</td><td>{r.nfs}</td></tr>')
    return ('<table class="t"><thead><tr><th></th><th>n_TOF EAR2, as run</th>'
            '<th>NFS / GANIL, 1&ndash;20 MeV</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def build_html(S, H, C, N, MISS, COMP) -> str:
    qlo, qhi = G.quiet_band()
    best = S[S.En_MeV < qhi]
    r_best = float(best.capsule_over_he3.max())
    r_worst = float(S.capsule_over_he3.max())
    th = H[H.En_MeV < 1e-3].iloc[0]
    at2 = H[np.isclose(H.En_MeV, 2.0)].iloc[0]
    gain = float(at2.radiative_per_neutron / th.radiative_per_neutron)
    prong = float(th.np_per_radiative / at2.np_per_radiative)
    t1, t40 = float(G.x17_min_angle(1.0)), float(G.x17_min_angle(40.0))
    alcomp = float(COMP.query('nuclide == "Al27" and En_MeV > 9')
                   .discrete_fraction.min())
    # missing_table wants the n_TOF page's column names
    miss_renamed = MISS.rename(columns={'what is assumed': 'what is missing'})

    return _number_sections(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>The same experiment at GANIL, where the neutron carries MeV</title>
{FONT_LINK}
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">NFS / GANIL</span>
    <span>X17 &middot; background feasibility</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>The same experiment at GANIL, where the neutron carries MeV</h1>
  <p class="sub">The excitation energy becomes a variable &middot; the signal
     angle moves and the capsule background does not &middot; and there is a
     window at 1&ndash;2.5&nbsp;MeV where the wall goes quiet</p>
</header>

<p class="lede"><b>Better, and by a lot &mdash; but only below about
{qhi:.1f}&nbsp;MeV.</b> Moving the same cell from thermal neutrons to MeV neutrons
changes three things at once. The gas converts <b>{gain:.0f}&times;</b> more of
its neutrons into radiative capture, because the 5333&nbsp;b (n,p) channel that
eats every thermal neutron and makes nothing has collapsed to under a barn. The
two-prong (n,p) load that a tracker has to reject per useful event falls by
<b>{prong:.0f}&times;</b>. And the capsule wall, which at n_TOF outnumbers the
gas by {_ratio(1e4)}&ndash;{_ratio(1e6)} in wide-angle pairs, <b>cannot make a
pair at all below {qhi:.1f}&nbsp;MeV</b> &mdash; its two strongest inelastic lines are
under the pair threshold and its first useful level has not opened. Above
3&nbsp;MeV that advantage is gone and the wall is back to
{_ratio(r_worst)}. The cost is that the X17 opening angle stops being a
constant: it slides from <b>{t1:.0f}&deg;</b> at 1&nbsp;MeV to
<b>{t40:.0f}&deg;</b> at 40&nbsp;MeV. That is a cost only if you cannot measure
the neutron energy, and at a time-of-flight facility you can.</p>

<div class="cards">
  <div class="card"><div class="v">{gain:.0f}&times;</div>
    <div class="l">more radiative captures per neutron entering the cell at
      2&nbsp;MeV than at thermal</div></div>
  <div class="card"><div class="v">{_ratio(r_best)}</div>
    <div class="l">capsule pairs per gas pair at {qlo:g}&ndash;{qhi:.1f}&nbsp;MeV, against
      {_ratio(1e4)}&ndash;{_ratio(1e6)} at n_TOF</div></div>
  <div class="card"><div class="v">{t1:.0f}&ndash;{t40:.0f}&deg;</div>
    <div class="l">where the X17 opening angle goes as E<sub>n</sub> runs
      1&nbsp;&rarr;&nbsp;40&nbsp;MeV</div></div>
  <div class="card"><div class="v">20 MeV</div>
    <div class="l">where every evaluation stops carrying what this page needs
      &mdash; from both directions</div></div>
</div>

<h2><span class="n">1</span>What actually changes</h2>
<p>The neutron's kinetic energy goes into the compound nucleus. With
<code>S<sub>n</sub>(&#8308;He) = {G.S_N_HE4:.4f}&nbsp;MeV</code> and the
centre-of-mass fraction <code>m(&sup3;He)/(m(&sup3;He)+m<sub>n</sub>) =
{G.CM_FRAC:.4f}</code>,</p>
<div class="panel"><p style="margin:0;text-align:center"><code>E<sub>x</sub>
= {G.S_N_HE4:.4f} + {G.CM_FRAC:.3f}&nbsp;E<sub>n</sub></code>
&nbsp;&nbsp;&mdash;&nbsp;&nbsp; {G.excitation(1.0):.1f} MeV at 1 MeV,
{G.excitation(40.0):.1f} MeV at 40 MeV</p></div>
<p>At n_TOF the second term is 1.5&nbsp;eV and the first is 20.58&nbsp;MeV, so
nothing moves and one template covers the whole run. Here the second term is
half the first.</p>
{DG.ganil_sweep()}

<h2><span class="n">2</span>The signal angle moves &mdash; and that is a handle,
not only a loss</h2>
<p>A 16.8&nbsp;MeV boson from a transition of energy <code>E<sub>x</sub></code>
has a hard minimum opening angle,
<code>cos&theta;<sub>min</sub> = 1 &minus; 2m&sup2;/E<sub>x</sub>&sup2;</code>.
The famous 109&deg; is not a property of the X17; it is a property of
20.58&nbsp;MeV. Raise the excitation and the signal walks down into the
continuum.</p>
{figure('ganil_kinematics',
        'The X17 minimum opening angle and a 36°-wide window on it, against '
        'neutron energy, with the n_TOF point at the left-hand edge. The ³He '
        'internal-pair continuum follows the signal down because it comes from '
        'the same excitation. The capsule continuum does not follow either of '
        'them: its photons are 27Al and 12C level energies, which do not care '
        'what the neutron did.',
        'X17 minimum opening angle against neutron energy')}
<p><b>The redeeming feature is that E<sub>n</sub> is measured.</b> At a
time-of-flight facility every event carries its own neutron energy, so the
signal is no longer &ldquo;a bump near 110&deg;&rdquo; but a
<i>correlation</i>: a peak whose position tracks a known function of a measured
quantity. A background that does not track it is rejected by that correlation
however large it is, and the capsule background is exactly such a background.
This discriminant does not exist at n_TOF, where every neutron gives the same
angle.</p>

<h2><span class="n">3</span>The gas: far more signal per neutron</h2>
<p>At thermal the cell is optically thick to &sup3;He(n,p) &mdash; an optical
depth of {th.tau_absorption:.0f} &mdash; so every neutron that enters is
absorbed, and one in {1 / (float(th.sigma_ngamma_b) / float(th.sigma_np_b)):.0e}
of those absorptions is radiative. At MeV energies (n,p) has fallen by three
and a half decades while (n,&gamma;) has fallen by less than one:</p>
<div class="scroll">{gas_table(H)}</div>
<p>Two things in that table matter more than the cross sections themselves. The
fourth column is the signal per neutron and it goes <b>up</b> by
{gain:.0f}&times; between thermal and 2&nbsp;MeV. The fifth is the number of
proton-triton pairs the tracker has to tell from an e&#8314;e&#8315; pair for
every useful event, and it falls from {_ratio(float(th.np_per_radiative))} to
{_ratio(float(at2.np_per_radiative))}. Both are consequences of the same fact: at thermal
the beam is consumed by a channel that produces nothing this experiment
wants.</p>
<div class="caution"><b>Above 20&nbsp;MeV there is no evaluated
&sup3;He(n,&gamma;) at all.</b> ENDF/B-VIII.0 and TENDL-2021 both stop there.
The rows above 20&nbsp;MeV in the figures are the evaluation's own trend
continued, marked as such, and nothing on this page is concluded from them. If
NFS running above 20&nbsp;MeV were the plan, that cross section would have to be
measured or calculated first &mdash; which is a piece of work, not a
footnote.</div>

<h2><span class="n">4</span>The capsule stops capturing and starts scattering</h2>
<p>At thermal the wall radiates by capture: 0.231&nbsp;b on &sup2;&#8311;Al, a
7.7&nbsp;MeV primary, and a cascade. At MeV energies capture is over
(0.66&nbsp;mb) and the wall radiates by <b>inelastic scattering</b>, at
cross sections of order a barn &mdash; four hundred times more reactions. And
yet, below 2.3&nbsp;MeV, it makes essentially no pairs. Two reasons, and both
are visible in the line table:</p>
<div class="scroll">{lines_table(C)}</div>
<ul>
<li><b>The two strongest lines cannot make a pair.</b> The 843.8 and
1014.5&nbsp;keV levels of &sup2;&#8311;Al carry most of the low-energy inelastic
strength, and both are below <b>2m<sub>e</sub> =
{1000 * G.PAIR_THRESHOLD_MEV:.0f}&nbsp;keV</b>. A photon that light has no pair
channel at all &mdash; not a small one, none.</li>
<li><b>The first level that can is at 2.211&nbsp;MeV</b>, which needs a
{qhi:.2f}&nbsp;MeV neutron to open. Below that the wall's only pair source is its
0.66&nbsp;mb radiative capture, which is three decades weaker than anything
else on this page.</li>
</ul>
<p>Then, above 4.8&nbsp;MeV, &sup1;&sup2;C's 4.44&nbsp;MeV level opens &mdash;
a 460&nbsp;mb E2 source of exactly the photons that make wide pairs, in the
carbon fibre, which carries eight times the areal density of the aluminium. That
is what ends the quiet window.</p>
{figure('ganil_lines',
        'The capsule photon inventory against neutron energy. The two dotted '
        'curves are the strongest lines in the whole capsule and neither can '
        'convert. The step at 2.3 MeV is the 27Al 2.21 MeV level opening; the '
        'one at 4.8 MeV is the 12C 4.44 MeV level, and it is the bigger of the '
        'two because the fibre outweighs the aluminium.',
        'capsule gamma production per neutron against neutron energy')}
<p class="note"><b>A completeness check, and it is honest about where it
fails.</b> Below 8&nbsp;MeV the named levels carry <b>100&nbsp;%</b> of the
evaluated inelastic cross section, so the line sum above is the whole of it. By
10&ndash;20&nbsp;MeV the evaluation has moved strength into the continuum
(MT&nbsp;=&nbsp;91), which this module cannot see, and the named levels hold
only <b>{100 * alcomp:.0f}&nbsp;%</b>. So the capsule numbers are complete
exactly where the recommendation lives and an <i>under</i>-estimate by up to a
factor of two where it does not.</p>

<h2><span class="n">5</span>Put the three together</h2>
{figure('ganil_rates',
        'Everything per neutron entering the cell. The gas curve is flat and '
        'high; the wall curve is three decades below it until 2.3 MeV and four '
        'decades above it by 10 MeV. The shaded strip is the window this page '
        'recommends.',
        'per-neutron rates for gas and capsule against neutron energy')}
{figure('ganil_spectra',
        'The same three components as absolute rates, at three neutron '
        'energies, with the signal window shaded. The shapes barely change '
        'between panels — what changes by four decades is the height of the '
        'capsule curve relative to the gas. The X17 curve is drawn at the '
        f'X17/IPC ratio the rate table assumes ({2.5e-2:g}); this page makes '
        'no claim about that number.',
        'absolute pair spectra at three neutron energies')}
<div class="scroll">{scan_table(S)}</div>

<h2><span class="n">6</span>The recommendation</h2>
{figure('ganil_window',
        f'Capsule pairs per gas pair inside the moving signal window. Below '
        f'{qhi:.2f} MeV the capsule background is comparable to the signal '
        f'continuum it sits on; by 10 MeV it is back where n_TOF was. The '
        f'shaded band at the top is where n_TOF sits.',
        'capsule to gas pair ratio in the signal window against neutron energy')}
<div class="panel"><p style="margin:0"><b>Run below {qhi:.1f}&nbsp;MeV.</b> In
that band the capsule makes {_ratio(r_best)} pairs per gas pair inside the
signal window, against {_ratio(1e4)}&ndash;{_ratio(1e6)} at n_TOF &mdash; four
to five orders of magnitude of background removed by choosing the beam energy,
with no change to the apparatus. The X17 opening angle there is
{float(G.x17_min_angle(qlo)):.0f}&ndash;{float(G.x17_min_angle(qhi)):.0f}&deg;,
close enough to the n_TOF geometry that the same detector acceptance applies.
The quasi-monoenergetic &#8311;Li(p,n) mode is the natural way to sit in it; a
white beam works too, because the neutron energy is measured per event and the
band is a cut rather than a beam property.</p></div>

<h2><span class="n">7</span>NFS against n_TOF, side by side</h2>
<div class="scroll">{compare_table(N)}</div>

<h2><span class="n">8</span>What is assumed</h2>
<p>Ordered by how much it could move the answer. Nothing here is fitted to
anything; the cross sections are read off ENDF/B-VIII.0 and the pair physics is
the same Born calculation the n_TOF page validates.</p>
<div class="scroll">{missing_table(miss_renamed)}</div>

<h2><span class="n">9</span>What would settle it</h2>
<ul>
<li><b>The NFS flux.</b> Every number here is per neutron entering the cell,
deliberately, because that is the part that is ours. Turning it into events per
day needs the facility's flux against energy at the intended flight path, and
that is the one input that has to come from GANIL.</li>
<li><b>Read MF&nbsp;=&nbsp;6 for the inelastic photons.</b> This page assumes
each excited level de-excites straight to the ground state. ENDF/B-VIII.0
carries the real photon production, and reading it would replace the assumption
with data and fix the 10&ndash;20&nbsp;MeV under-estimate at the same time.</li>
<li><b>Measure or calculate &sup3;He(n,&gamma;) above 20&nbsp;MeV</b> if running
there is ever the plan. Nothing evaluated exists.</li>
<li><b>Then run the capsule through Geant4</b> with a 1&ndash;3&nbsp;MeV beam.
Acceptance for wall-born pairs is still the largest unknown on both pages, and
it is the same simulation for both.</li>
<li><b>And note what this page does not do:</b> it quotes no signal rate. The
X17 branching at E<sub>x</sub>&nbsp;=&nbsp;21&ndash;51&nbsp;MeV, away from the
20.21 and 21.01&nbsp;MeV states the anomaly is reported for, is a model
statement. The background is calculable; the signal is not, and pretending
otherwise would be the easiest way to get this wrong.</li>
</ul>

<footer>
Generated by <code>make_ganil_report.py</code> from
<code>ganil_background.py</code>, <code>endf.py</code> and
<code>ipc_born.py</code> &middot; every figure ships its CSV &middot;
{dt.date.today().isoformat()}<br>
Cross sections: ENDF/B-VIII.0 neutron sublibrary (13-Al-27, 6-C-12, 2-He-3),
MF&nbsp;=&nbsp;3, staged in <code>data/nuclear/</code> &middot; pair physics as
on the <a href="../ipc/report.html">IPC continuum page</a>.
</footer>
</div>
</body>
</html>
""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--multipole', default='E1', choices=('E1', 'M1'))
    a = ap.parse_args()
    od = paths.out('ganil')

    S = G.energy_scan(he_multipole=a.multipole)
    H = G.he3_rates([0.0253e-6, 1, 1.5, 2, 3, 5, 10, 14, 20, 30])
    C = G.capsule_lines(5.0)
    N = G.ntof_comparison(a.multipole)
    MISS = G.missing()
    COMP = G.inelastic_completeness([1, 2, 3, 5, 8, 10, 14, 20])

    for name, df in (('ganil_energy_scan', S), ('ganil_he3_rates', H),
                     ('ganil_capsule_lines_5MeV', C),
                     ('ganil_ntof_comparison', N),
                     ('ganil_inelastic_completeness', COMP),
                     ('ganil_missing', MISS)):
        df.to_csv(od / f'{name}.csv', index=False)

    html = build_html(S, H, C, N, MISS, COMP)
    (od / 'report.html').write_text(html, encoding='utf-8')
    (od / 'index.html').write_text(html, encoding='utf-8')
    print(f'wrote -> {od / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
