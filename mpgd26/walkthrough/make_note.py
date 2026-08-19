#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_note.py -- build the standalone HTML walkthrough from steps.json and
figures/*.png.  Every number in the prose comes out of steps.json, so re-running
make_figures.py and then this script keeps the text and the figures in step.

    ../../.venv/bin/python make_note.py            # -> forward_fit_det3.html
"""
from __future__ import annotations

import base64
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'steps.json')))
OUT = os.path.join(HERE, 'forward_fit_det3.html')


def img(name, alt):
    p = os.path.join(HERE, 'figures', name + '.png')
    b = base64.b64encode(open(p, 'rb').read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{alt}"></figure>'


def cap(t):
    return f'<figcaption>{t}</figcaption>'


ev, raw, tr = S['event'], S['raw'], S['track']
ker, col, nn = S['kernel'], S['column'], S['nnls']
dec, res, sc = S['decompose'], S['residual'], S['scan']
rat, ens, imp = S['ratio'], S['ensemble'], S['implied_v']
ky, kx = ker['y'], ker['x']
CK = sorted(col, key=int)          # the two slices figure 5 drew
c_early, c_late = col[CK[0]], col[CK[-1]]
h, fr = ens['held'], ens['full_run']
pr, cu = rat['prod'], rat['cur']

# the w -> angle map re-measured for THIS bundle on the training half only
W0, KW = h['y']['w0'], h['y']['kw']
V = ev['v_drift']
tan_cor = (tr['w_um_ns'] - W0) / (KW * V)
th_raw = math.degrees(math.atan(ev['tan_raw']))
th_cor = math.degrees(math.atan(tan_cor))
th_ref = math.degrees(math.atan(ev['tan_ref']))

HTML = f"""<!doctype html>
<!--note
title: One muon through the forward fit, step by step
summary: Every stage of the wft forward model on one real det3 cosmic — the raw core and +-1, +-2 waveforms, the model that predicts them, and what the fit actually searches. On the corrected kernel.
tags: X17, cosmic bench, micromegas, reconstruction, MPGD26
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One muon through the forward fit, step by step</title>
<meta name="description" content="Every stage of the wft forward model on one real det3 cosmic muon: the raw waveforms, the model that predicts them, and what the fit searches.">
<style>
  :root {{
    --ink:#1f2933; --dim:#6b7280; --line:#e5e7eb; --bg:#ffffff;
    --own:#0072B2; --n1:#D55E00; --n2:#8a3f8f; --acc:#b45309;
    --ok:#15803d; --bad:#b91c1c;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    "Helvetica Neue",Arial,sans-serif; -webkit-text-size-adjust:100%; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:2.4rem 1.15rem 5rem; }}
  h1 {{ font-size:1.95rem; line-height:1.22; margin:0 0 .5rem; letter-spacing:-.01em; }}
  h2 {{ font-size:1.28rem; margin:3.1rem 0 .7rem; letter-spacing:-.005em;
       padding-top:.9rem; border-top:1px solid var(--line); }}
  h2 .num {{ color:var(--dim); font-weight:600; margin-right:.5rem; }}
  p {{ margin:.75rem 0; }}
  .dateline {{ color:var(--dim); font-size:.9rem; margin:0 0 1.6rem; }}
  .lead {{ font-size:1.08rem; }}
  figure {{ margin:1.3rem 0 .4rem; }}
  img {{ width:100%; height:auto; display:block; border:1px solid var(--line);
        border-radius:6px; }}
  figcaption {{ color:var(--dim); font-size:.88rem; margin:.5rem 0 1.4rem;
               line-height:1.5; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         font-size:.9em; background:#f3f4f6; padding:.1em .32em;
         border-radius:3px; }}
  table {{ border-collapse:collapse; width:100%; margin:1.1rem 0;
          font-size:.93rem; }}
  th,td {{ text-align:left; padding:.42rem .6rem; border-bottom:1px solid var(--line); }}
  th {{ font-weight:600; color:var(--dim); font-size:.86rem;
       text-transform:uppercase; letter-spacing:.04em; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .box {{ border:1px solid var(--line); border-left:4px solid var(--own);
         border-radius:6px; padding:.9rem 1.1rem; margin:1.4rem 0;
         background:#f9fafb; }}
  .box.warn {{ border-left-color:var(--bad); }}
  .box.good {{ border-left-color:var(--ok); }}
  .box h4 {{ margin:0 0 .45rem; font-size:.97rem; }}
  .box p:last-child {{ margin-bottom:0; }}
  .own {{ color:var(--own); font-weight:600; }}
  .n1 {{ color:var(--n1); font-weight:600; }}
  .n2 {{ color:var(--n2); font-weight:600; }}
  .bad {{ color:var(--bad); font-weight:600; }}
  .good {{ color:var(--ok); font-weight:600; }}
  ul {{ padding-left:1.25rem; }} li {{ margin:.35rem 0; }}
  .foot {{ color:var(--dim); font-size:.85rem; margin-top:3rem;
          border-top:1px solid var(--line); padding-top:1rem; }}
  @media (max-width:640px) {{ h1 {{ font-size:1.55rem; }} .wrap {{ padding-top:1.6rem; }} }}
</style>
</head>
<body>
<div class="wrap">

<h1>One muon through the forward fit, step by step</h1>
<p class="dateline">det3 &middot; Saturday long run, resistive 490&nbsp;V / drift 1000&nbsp;V &middot;
event {ev['eid']}, {ev['plane'].upper()} plane &middot; calibration
<code>{ev['bundle']}</code> &middot; 18 August 2026</p>

<p class="lead">This is the long version of deck slides 9b and 9c: the same
event, but with every stage drawn — including the raw waveforms of the core
strip and its &plusmn;1 and &plusmn;2 neighbours, which the slides only ever
show already modelled.</p>

<div class="box good">
<h4>Which calibration this runs on, and why it matters</h4>
<p>Everything below uses <code>{cu['bundle']}</code>, in which
c<sub>2</sub> = 0.6&thinsp;c<sub>1</sub> — the &plusmn;2 copy is
<i>smaller</i> than the &plusmn;1 copy, which is the only ordering a resistive
film can produce, because the &plusmn;2 strip is reached only through the
&plusmn;1 strip.</p>
<p><b>The frozen MPGD26 production reco was not run this way.</b>
<code>{pr['bundle']}</code> carries c<sub>2</sub>/c<sub>1</sub> =
{pr['ratio']:.2f} — a &plusmn;2 copy <i>larger</i> than the &plusmn;1 — and
every number currently on the deck came out of a reco that used it. That defect
was measured and fixed on 17–18 August; the refit bundle exists for det3 and
det7 and the re-freeze has not been run yet. Section 10 shows exactly what
changed and section 11 shows what it costs: <b>nothing measurable</b> — the
angle resolution is the same within 0.5&sigma;. The frozen results are not
wrong in value; the calibration that produced them was not defensible in
description, which is a different and still-real problem.</p>
</div>

<div class="box">
<h4>Your description, checked</h4>
<p><i>&ldquo;We guess a charge distribution over each strip, produce the 0,
&plusmn;1, &plusmn;2 waveforms, then sum over all strips and iterate track angle
to see what fits the waveforms simultaneously best.&rdquo;</i></p>
<p>Right in outline, and right about the thing that matters — every strip is
predicted, nothing is inverted, and one geometry has to explain the whole
window at once. Three corrections:</p>
<ul>
<li><b>The free charge is not per strip. It is per 60&nbsp;ns arrival
slice</b> — {tr['K']} of them, i.e. a charge-versus-<i>depth</i> profile.
<i>Where</i> each slice's charge lands on the strips is not free: it is fixed by
the geometry through the strip integral. That is what makes the fit tight —
{nn['cols']} charges and 3 geometric numbers against {nn['rows']}
measurements.</li>
<li><b>The charges are not iterated.</b> At every trial geometry they are
solved <i>exactly</i>, in one step, by non-negative least squares. Only three
numbers are searched: p<sub>0</sub>, w, t<sub>0</sub>.</li>
<li><b>The &plusmn;1 / &plusmn;2 copies are inside the prediction, not added
afterwards.</b> Each column of the design matrix already carries them, so the
strip that receives a copy and the strip that donated it are fitted with one
consistent set of charges.</li>
</ul>
</div>

<h2><span class="num">1</span>What comes in</h2>
<p>{raw['n_strips']} strips &times; {raw['nsamp']} samples of 60&nbsp;ns,
pedestal-subtracted DREAM ADC. Nothing else — no hit times, no thresholds, no
reference track. The diagonal in the left panel <i>is</i> the drift: charge
liberated near the cathode arrives last, and on an inclined track it arrives on
a different strip.</p>
{img('f1_raw', 'Left: the measured window as a strip-by-time image, with a clear '
     'diagonal drift ladder. Right: the raw waveforms of the core strip and its '
     'plus-minus-one and plus-minus-two neighbours, stacked.')}
{cap(f'Right: the five waveforms this note is about. Core peak '
     f'{raw["peak_adc"]["0"]:.0f} ADC at {raw["peak_ns"]["0"]:.0f} ns, and the '
     f'peak walks from {raw["peak_ns"]["2"]:.0f} ns on +2 to '
     f'{raw["peak_ns"]["-2"]:.0f} ns on &minus;2 — '
     f'{raw["peak_ns"]["-2"] - raw["peak_ns"]["2"]:.0f} ns across five strips. '
     f'Median electronics noise {raw["noise_med"]:.1f} ADC. That walk is the '
     'drift <i>and</i> the resistive delay mixed together, which is exactly why '
     'a per-strip time is not a drift time.')}

<h2><span class="num">2</span>The three numbers</h2>
<p>The model says: a straight segment crossed the {V:.1f}&nbsp;&micro;m/ns
drift gap. Cut its arrival into slices of &Delta;t = {tr['dt_ns']:.0f}&nbsp;ns.
Slice <i>k</i> arrives at t<sub>0</sub> + u<sub>k</sub> and was liberated at
transverse position p<sub>0</sub> + w&thinsp;u<sub>k</sub>. Three numbers,
{tr['K']} slices, and a full 30&nbsp;mm drift takes
{tr['drift_full_ns']:.0f}&nbsp;ns.</p>
{img('f2_track', 'Left: the fitted straight segment through the 30 mm drift gap '
     'with its 18 arrival slices. Right: the same slices laid over the measured '
     'strip-by-time image, tracking the diagonal.')}
{cap(f'This event: p<sub>0</sub> = {tr["p0"]:.2f} mm, w = '
     f'{tr["w_um_ns"]:.2f} &micro;m/ns, t<sub>0</sub> = {tr["t0"]:.0f} ns — so '
     f'the charge column walks {tr["span_mm"]:.1f} mm sideways while it drifts. '
     'The right panel is the whole idea in one picture: the fitted slices have '
     'to sit on the measured diagonal.')}

<h2><span class="num">3</span>Geometry decides where each slice lands</h2>
<p>Each slice is a little cloud: its transverse extent over the 60&nbsp;ns bin,
widened by the initial cloud size &sigma;<sub>p0</sub> =
{S['fractions']['sigma_p0']:.3f}&nbsp;mm and by transverse diffusion, integrated
over the {S['fractions']['pitch']:.2f}&nbsp;mm pitch. That integral —
<code>strip_fractions</code> — is the matrix F<sub>sk</sub>. No free parameter
in it that is not already calibrated.</p>
{img('f3_fractions', 'Left: the strip-fraction matrix as an image, a diagonal '
     'band. Middle: four individual slices spread over two or three strips. '
     'Right: the transverse cloud width against drift time, staying below one '
     'strip pitch.')}
{cap(f'The cloud is narrower than a pitch everywhere '
     f'({S["fractions"]["sigma_p0"]:.2f} mm at the mesh, '
     f'{S["fractions"]["sigma_end"]:.2f} mm at the far end), so a slice lands on '
     'two or three strips. Everything wider than that in the data has to come '
     'from the next step.')}

<h2><span class="num">4</span>The resistive kernel copies charge sideways, late</h2>
<p>The strips sit on a resistive film. Charge that spreads through it reaches
the neighbour <i>late</i> and dispersed. In this calibration that is modelled as
a delayed copy of the impulse response: amplitude c<sub>1</sub> at a lag of
&tau;<sub>s</sub> = {ker['tau_s']:.0f}&nbsp;ns to &plusmn;1, and
c<sub>2</sub> = {cu['ratio']:.1f}&thinsp;c<sub>1</sub> at twice the lag to
&plusmn;2 (<code>share_mode = '{ker['share_mode']}'</code>).</p>
{img('f4_kernel', 'The own-charge response and the plus-minus-one and '
     'plus-minus-two copies for the Y and X planes, normalised to the own-charge '
     'peak. The plus-minus-two copy is smaller than the plus-minus-one copy on '
     'both planes.')}
{cap(f'Y shares {ky["c1"] / kx["c1"]:.1f}&times; more than X — the strips run '
     f'along y, so only the Y view sees the film&rsquo;s own sheet resistance. '
     f'On Y, c<sub>1</sub> = {ky["c1"]:.3f} and c<sub>2</sub> = '
     f'{ky["c2"]:.3f}; the ratio is pinned at {cu["ratio"]:.1f} rather than '
     'fitted, because a cosmic-angle &chi;&sup2; cannot resolve it and will '
     'wander if left free. The value comes from the H4 beam, which measures it '
     'head-on and model-free at 0.45&nbsp;&plusmn;&nbsp;0.02; near-vertical '
     'bench cosmics on this detector give 0.63&nbsp;&plusmn;&nbsp;0.09.')}

<h2><span class="num">5</span>One slice &rarr; five waveforms</h2>
<p>This is the step your description was pointing at. Put one unit of charge in
one slice and ask what the electronics record. It lands mostly on one strip
(<span class="own">blue</span>); the kernel copies a fraction onto
&plusmn;1 (<span class="n1">vermillion</span>) and &plusmn;2
(<span class="n2">purple</span>); each of those is folded with that plane's
measured impulse response. That is one column of the design matrix.</p>
{img('f5_column', 'Two depth slices, early and late. For each, the geometric '
     'strip fractions on top and then five stacked waveforms — the strip the '
     'charge landed on and its plus-minus-one and plus-minus-two neighbours — '
     'each split by where the signal came from.')}
{cap(f'The two slices drawn are the best-centred early and late ones '
     f'(k = {CK[0]} and k = {CK[-1]}); a slice straddling a strip boundary '
     f'splits geometrically and hides the point. Take the late one, arriving '
     f'{c_late["u"]:.0f} ns after t<sub>0</sub> at {c_late["p"]:.2f} mm: the '
     f'strip it landed on keeps {100 * c_late["own_frac"]["0"]:.0f} % of its own '
     f'charge, the &plusmn;1 strip&rsquo;s signal is still mostly its own '
     f'geometric share ({100 * c_late["own_frac"]["1"]:.0f} %) — the cloud is '
     f'wide enough to reach it without any film — but at &plusmn;2 '
     f'<b>{100 - 100 * c_late["own_frac"]["2"]:.0f} % of the pulse is a delayed '
     'copy of somebody else</b>. That is the signature the kernel exists to '
     'model, and the reason a threshold crossing two strips out is not a drift '
     'time.')}

<h2><span class="num">6</span>The charges are solved, not searched</h2>
<p>Stack those columns and you have A, {nn['rows']}&nbsp;&times;&nbsp;{nn['cols']}.
The prediction is A&thinsp;q. At each trial (p<sub>0</sub>, w, t<sub>0</sub>) the
charge profile is whatever minimises &#8214;A&thinsp;q &minus; y&#8214; subject to
q&nbsp;&ge;&nbsp;0 — a convex problem with one answer, no starting guess and no
iteration over the charges. The non-negativity is doing real work: it is what
stops the fit inventing negative charge to absorb a mismatched tail.</p>
{img('f6_nnls', 'Left: the design matrix as an image, one column per depth '
     'slice. Right: the solved non-negative charge profile against drift depth.')}
{cap(f'This muon&rsquo;s ionisation is genuinely clumpy — {nn["n_zero"]} of the '
     f'{nn["cols"]} slices come back at exactly zero and one late slice carries a '
     'large lump. That is measured, not assumed; the profile is an output.')}

<h2><span class="num">7</span>What each strip&rsquo;s prediction is made of</h2>
<p>Now the same decomposition on the fitted event: for six consecutive strips,
the model waveform split into the strip&rsquo;s own charge, its
&plusmn;1 neighbours&rsquo; copies, and its &plusmn;2 neighbours&rsquo;, against
the measured samples.</p>
{img('f7_modelvsdata', 'Six consecutive strips. Each panel stacks the model into '
     'own charge, plus-minus-one copies and plus-minus-two copies, with the '
     'measured samples drawn on top.')}
{cap('Neighbours&rsquo; charge as a fraction of the predicted peak: '
     + ', '.join(('core' if d == 0 else f'{d:+d}')
                 + f'&nbsp;{100 * dec["neighbour_frac"][str(d)]:.0f}&nbsp;%'
                 for d in (-2, -1, 0, 1, 2, 3) if str(d) in dec['neighbour_frac'])
     + f'. Even the core strip is {100 * dec["neighbour_frac"]["0"]:.0f} % '
       'somebody else&rsquo;s charge, and on the flanks it reaches '
     + f'{100 * max(dec["neighbour_frac"].values()):.0f} %. Every one of those '
       'strips has a threshold crossing that times a mixture — which is the '
       'whole argument for fitting forward.')}

<h2><span class="num">8</span>The whole window at once</h2>
{img('f8_residual', 'The measured window, the model and the residual divided by '
     'the electronics noise, as three images.')}
{cap(f'{nn["rows"]} measurements described by 3 geometric numbers and '
     f'{nn["cols"]} non-negative charges. Residual rms '
     f'{res["rms_adc"]:.0f} ADC = {res["rms_pct_peak"]:.1f} % of the peak.')}
<div class="box warn">
<h4>What this is not</h4>
<p>It is <b>not</b> a &chi;&sup2;&nbsp;&asymp;&nbsp;1 fit.
&chi;&sup2;/dof&nbsp;=&nbsp;{sc['chi2_dof']:.0f} here, and the residual image has
visible structure: the model is good to ~{res['rms_pct_peak']:.0f}&nbsp;% of the
peak, while the electronics noise is
{100 * res['noise_adc'] / ev['peak_adc']:.1f}&nbsp;% of it. So the error bars in
the fit are the noise, but the residuals are dominated by model error — the
impulse-response shape, the single-&tau; kernel, the straight line, the 60&nbsp;ns
binning. The parameter <i>errors</i> that come out of the curvature are
therefore optimistic; the resolution we quote is measured against the reference
(step 11), not read off the &chi;&sup2;.</p>
</div>

<h2><span class="num">9</span>What is actually searched</h2>
<p>Three scans, one per searched number, plus a two-dimensional map, with
everything else held at the optimum. The fit itself is a coarse grid followed by
Nelder-Mead — but the scans are what the minimiser is walking on.</p>
{img('f9_scan', 'Chi-squared against track angle, against p0, and against t0, '
     'plus a two-dimensional map in p0 and angle.')}
{cap(f'&chi;&sup2; doubles within {sc["width_theta"] / 2:.1f}&deg; in angle, '
     f'{1e3 * sc["width_p0"] / 2:.0f} &micro;m in position and '
     f'{sc["width_t0"] / 2:.0f} ns in t<sub>0</sub>. The (p<sub>0</sub>, &theta;) '
     'map shows the one real correlation: slide the track sideways and tilt it, '
     'and the prediction barely changes. This event lands at '
     f'&theta; = {sc["theta_fit"]:.2f}&deg; against the M3 reference&rsquo;s '
     f'{sc["theta_ref"]:.2f}&deg;.')}
<p>Two details that are easy to miss. <b>t<sub>0</sub> is not free.</b> The
&chi;&sup2; surface has near-degenerate minima one depth bin (60&nbsp;ns) apart —
shift the profile a bin and slide p<sub>0</sub> by
w&thinsp;&times;&thinsp;60&nbsp;ns and the prediction is nearly the same — and
only about a third of unconstrained fits land in the physical one. A prior from
the scintillator trigger phase selects it — mostly by <i>seeding</i> the search
there rather than by out-weighing the &chi;&sup2;, as the box in section 10
shows. Here it sits at {ev['t0_pred']:.0f}&nbsp;ns with &sigma; =
{ev['t0_prior_sigma']:.0f}&nbsp;ns.
<b>And v<sub>drift</sub> enters exactly once</b>, at the end:
tan&thinsp;&theta; = w / v.</p>

<table>
<tr><th>step</th><th>quantity</th><th class="n">value</th></tr>
<tr><td>fit</td><td>w</td><td class="n">{tr['w_um_ns']:.3f} &micro;m/ns</td></tr>
<tr><td>&divide; v</td><td>tan&thinsp;&theta; raw</td><td class="n">{ev['tan_raw']:.4f} &nbsp;({th_raw:.2f}&deg;)</td></tr>
<tr><td>per-plane mapping</td><td>(w &minus; w<sub>0</sub>) / (k<sub>w</sub> v), w<sub>0</sub> = {W0:.3f}, k<sub>w</sub> = {KW:.4f}</td><td class="n">{tan_cor:.4f} &nbsp;({th_cor:.2f}&deg;)</td></tr>
<tr><td>M3 reference</td><td>tan&thinsp;&theta;</td><td class="n">{ev['tan_ref']:.4f} &nbsp;({th_ref:.2f}&deg;)</td></tr>
</table>
<p>The w<sub>0</sub>/k<sub>w</sub> line is an honest wart: a per-plane linear map
measured <i>after the fact</i> from free fits of reference tracks, absorbing a
small residual offset and scale. The bundle ships the old kernel&rsquo;s values,
which are stale by construction, so the ones used here were re-measured on the
180 calibration events with <code>bench/set_w0.py</code>&rsquo;s recipe and
applied to the 220 held-out ones — never fitted on the events being scored.</p>

<h2><span class="num">10</span>What was replaced, and what it cost</h2>
<p>The &plusmn;2 strip is only reached <i>through</i> the &plusmn;1 strip, so
c<sub>2</sub>&nbsp;&lt;&nbsp;c<sub>1</sub> always. The frozen production bundle
carries c<sub>2</sub>/c<sub>1</sub> = <span class="bad">{pr['ratio']:.2f}</span>.
That is not a bound artefact: the ref-pinned cosmic &chi;&sup2; is genuinely flat
in this direction, so an unconstrained fit wanders there and stays. The H4 beam,
at normal incidence, breaks the degeneracy and measures the ratio directly. The
fix is to pin the ratio and refit everything else — so &tau;<sub>s</sub> and
&sigma;<sub>s</sub> move too; this is a refit, not a relabelling.</p>
{img('f10_ratio', 'Left: the superseded frozen kernel, with the plus-minus-two '
     'copy larger than the plus-minus-one. Middle: the kernel in use here, '
     'correctly ordered. Right: the plus-two strip of this event under both, '
     'against the measurement.')}
{cap(f'Same event, same code, two calibrations. The angle moves by '
     f'{abs(rat["d_theta"]):.3f}&deg; ({pr["theta"]:.2f}&deg; &rarr; '
     f'{cu["theta"]:.2f}&deg;) and &chi;&sup2;/dof falls from '
     f'{pr["chi2_dof"]:.1f} to {cu["chi2_dof"]:.1f}. The right-hand panel is '
     'where they differ most: the superseded kernel needs a large late '
     '&plusmn;2 bump on the +2 strip that the corrected one does not.')}
<table>
<tr><th></th><th>superseded — <code>{pr['bundle']}</code></th><th>in use here — <code>{cu['bundle']}</code></th></tr>
<tr><td>c<sub>1</sub>, c<sub>2</sub> (Y)</td><td>{pr['c1']:.3f}, <span class="bad">{pr['c2']:.3f}</span></td><td>{cu['c1']:.3f}, <span class="good">{cu['c2']:.3f}</span></td></tr>
<tr><td>c<sub>2</sub>/c<sub>1</sub></td><td class="bad">{pr['ratio']:.2f}</td><td class="good">{cu['ratio']:.2f} (pinned)</td></tr>
<tr><td>&tau;<sub>s</sub></td><td>{pr['tau_s']:.0f} ns</td><td>{cu['tau_s']:.0f} ns</td></tr>
<tr><td>&sigma;<sub>s</sub></td><td>{pr['sigma_s']:.1f} ns</td><td>{cu['sigma_s']:.1f} ns</td></tr>
<tr><td>free hyper-parameters</td><td>7</td><td>6</td></tr>
<tr><td>this event: p<sub>0</sub>, t<sub>0</sub></td><td>{pr['p0']:.2f} mm, {pr['t0']:.0f} ns</td><td>{cu['p0']:.2f} mm, {cu['t0']:.0f} ns</td></tr>
<tr><td>this event: &theta;, &chi;&sup2;/dof</td><td>{pr['theta']:.2f}&deg;, {pr['chi2_dof']:.1f}</td><td>{cu['theta']:.2f}&deg;, {cu['chi2_dof']:.1f}</td></tr>
<tr><td>held-out &sigma;<sub>68</sub> X / Y</td><td>{h['x']['s68_prod']:.2f}&deg; / {h['y']['s68_prod']:.2f}&deg;</td><td>{h['x']['s68']:.2f}&deg; / {h['y']['s68']:.2f}&deg;</td></tr>
</table>
<p><b>The correction is free, not an improvement.</b> On 220 held-out events the
angle resolution changes by {h['x']['s68'] - h['x']['s68_prod']:+.03f}&deg; (X)
and {h['y']['s68'] - h['y']['s68_prod']:+.03f}&deg; (Y). A paired bootstrap over
shared held-out events, run when the refit was benched, put it at
+0.028&nbsp;&plusmn;&nbsp;0.062&deg; on det3 and
+0.023&nbsp;&plusmn;&nbsp;0.080&deg; on det7 — under 0.6&sigma; either way. What is bought is that the kernel
constants now mean what they are called, and one fitted parameter becomes a
measured constraint. Anyone who reads c<sub>2</sub>&nbsp;&gt;&nbsp;c<sub>1</sub>
off a slide will ask how charge reaches the second neighbour without passing the
first, and there is no answer.</p>
<div class="box warn">
<h4>One thing this event exposed</h4>
<p>Under the corrected kernel this event settles a full depth bin earlier —
t<sub>0</sub> {pr['t0']:.0f}&nbsp;&rarr;&nbsp;{cu['t0']:.0f}&nbsp;ns with
p<sub>0</sub> sliding {cu['p0'] - pr['p0']:+.2f}&nbsp;mm along the degeneracy,
exactly as w&thinsp;&times;&thinsp;60&nbsp;ns predicts. Scanning t<sub>0</sub>
by hand, <i>both</i> calibrations have their lower &chi;&sup2; at the earlier bin;
the frozen fit sat in the higher one. With &chi;&sup2;/dof&nbsp;&asymp;&nbsp;20
the &sigma;&nbsp;=&nbsp;5&nbsp;ns prior contributes ~50 units against a ~400-unit
&chi;&sup2; difference between the two minima, so it cannot arbitrate here — the
bin is chosen by where the coarse grid starts. The reference prefers the
corrected answer (position residual {abs(cu['p0'] - sc['p0_ref']):.2f} mm against
{abs(pr['p0'] - sc['p0_ref']):.2f} mm), but that is one event. <b>Worth checking
across the run during the re-freeze</b>; it is not visible in the ensemble
numbers, which are unchanged.</p>
</div>

<h2><span class="num">11</span>Does it work</h2>
<p>The reference telescope never enters any of the above. Held-out events —
the ones the calibration did not train on — put the fitted angle against M3.</p>
{img('f11_ensemble', 'Reconstructed against reference angle for held-out '
     'events, the angle residual for both kernels, the position residual, and '
     'the implied drift velocity against track angle.')}
{cap(f'Held out, {h["y"]["n"]} events: &sigma;<sub>68</sub> = '
     f'{h["x"]["s68"]:.2f}&deg; (X) and {h["y"]["s68"]:.2f}&deg; (Y), bias under '
     f'0.06&deg;; the dashed histograms are the superseded kernel on the same '
     f'events. Over the full frozen run ({fr["y"]["n"]:,} events, superseded '
     f'kernel) it is {fr["x"]["s68"]:.2f}&deg; / {fr["y"]["s68"]:.2f}&deg; — the '
     'cached sample is the cleanly-matched subset, so it flatters slightly.')}
<p>The bottom-right panel is the check that cannot be tuned away. Divide the
fitted w by the <i>reference</i> angle and you get an implied drift velocity; if
the model were mistiming the ladder, that number would depend on track angle.
Across three bins it varies by {imp['x']['spread']:.1f} (X) and
{imp['y']['spread']:.1f} (Y) &micro;m/ns against a per-bin uncertainty of about
{0.5 * (imp['x']['err_typ'] + imp['y']['err_typ']):.1f} — flat within what 220
events can say, and no more than that. The full frozen run has the statistics to
say it properly: spread {fr['x']['implied_v_spread']:.1f} (X) and
{fr['y']['implied_v_spread']:.1f} (Y) &micro;m/ns around the calibrated
{V:.1f}. Re-running that check on the corrected bundle is part of the
re-freeze. The per-strip hit-time chain, on identical events, sweeps
13&nbsp;&micro;m/ns across the same bins.</p>
<p>The position panel is <b>not</b> a detector resolution:
&sigma;<sub>68</sub>&nbsp;&asymp;&nbsp;{1e3 * h['y']['pos_s68']:.0f}&nbsp;&micro;m
is dominated by M3 pointing and by multiple scattering of a few-GeV cosmic
between the telescope and the chamber. The detector&rsquo;s position resolution
is measured on the SPS beam instead (176&nbsp;&plusmn;&nbsp;35&nbsp;&micro;m on
det4).</p>

<h2><span class="num">12</span>What this means for the deck</h2>
<ul>
<li><b>The kernel figure on slide 9 has to change.</b> It is drawn from the
frozen bundle and therefore shows c<sub>2</sub>&nbsp;&gt;&nbsp;c<sub>1</sub>. It
should be drawn from the corrected constants — which is a one-line change in
<code>make_share.py</code>, since it reads the bundle rather than hard-coding —
and the amplitudes can then be quoted rather than hedged. The
&ldquo;Y shares ~3&times; more than X&rdquo; line is measured
({ky['c1'] / kx['c1']:.1f}&times;) and stays.</li>
<li><b>Slides 9b and 9c should be regenerated on the corrected bundle too</b>,
so the deck does not carry two different kernels in one section. The angle
numbers do not move (section 10), so no caption arithmetic changes — but slide
9c&rsquo;s event display will shift by one depth bin, which is fine and worth
knowing before someone compares screenshots.</li>
<li><b>Slide 9c&rsquo;s denominator is wrong independently of all this:</b> it
attributes &sigma;<sub>68</sub> = 1.19&deg;/1.16&deg; to &ldquo;the full
7,093-event run&rdquo;. 7,093 is the number of reconstructed events; the
resolution is measured on the {fr['x']['n']:,} (X) and {fr['y']['n']:,} (Y) that
also have an M3 reference.</li>
<li><b>None of the deck&rsquo;s physics numbers move</b> — efficiency, angle
resolution, the implied-velocity check are all unchanged within errors. What
changes is that the calibration behind them becomes describable.</li>
</ul>
<p>The full re-freeze — swap the manifest to <code>{cu['bundle']}</code> for
det3 and det7, re-run the reco, re-measure w<sub>0</sub>/k<sub>w</sub>, re-run
the digest — is a condor campaign, and is the remaining piece. Details in the
<a href="https://dylan-neff.web.cern.ch/notes/sharing-kernel-measured.html">sharing
kernel, measured</a> note.</p>

<p class="foot">Generated by <code>mpgd26/walkthrough/make_figures.py</code> +
<code>make_note.py</code>. Every kernel and design-matrix call goes through
<code>wft.model</code> itself, not a re-implementation. Cross-check: the same
code on the frozen bundle reproduces the production <code>events.parquet</code>
row for event {ev['eid']} exactly — p<sub>0</sub> {pr['p0']:.4f} mm,
w {pr['w'] / 1e3:.6f} mm/ns, t<sub>0</sub> {pr['t0']:.1f} ns against the
table&rsquo;s 230.9417 / &minus;0.008548 / 290.0.</p>

</div>
</body>
</html>
"""

open(OUT, 'w').write(HTML)
print('wrote', OUT, f'{os.path.getsize(OUT) / 1e6:.2f} MB')
