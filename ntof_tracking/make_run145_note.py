#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_run145_note.py — build the run_145 target-imaging note for the CERN site.

One coherent, current page: every figure and number on the FULL-COVERAGE basis
(2026-08-13, head-on band included; the slope_reliable gate is gone except
where a quantity divides by tan). Rebuild after any re-run so figures, tables
and verdict text move together:

    python -m ntof_tracking.make_run145_note
    cd ~/PycharmProjects/dylan-cern-site && \
        python3 scripts/add-note.py pages/notes/run145-target-imaging.html \
        --force && ./scripts/deploy-eos.sh
"""
from __future__ import annotations

import base64
import json
import os

BASE = '/media/dylan/data/x17/beam_july/analysis/wft/run_145/stat090_0000'
FULLCOV = BASE + '/imaging_fullcov'
NOTE_FIGS = BASE + '/imaging/note_figs_fullcov'
DISPLAYS = BASE + '/mx17_A/displays'
WALL3D = FULLCOV + '/wall_3d'
OUT = os.path.expanduser(
    '~/PycharmProjects/dylan-cern-site/pages/notes/run145-target-imaging.html')


def img(path):
    with open(path, 'rb') as f:
        return ('<img src="data:image/png;base64,'
                + base64.b64encode(f.read()).decode() + '">')


def pct(x):
    return f'{100 * x:.0f}%'


S = json.load(open(FULLCOV + '/imaging_summary.json'))
R = {a['arm']: a for a in S['results']}
W3 = json.load(open(WALL3D + '/wall3d_summary.json'))
SUB1 = BASE.replace('stat090_0000', 'stat090_0001') + '/imaging_fullcov'
R1 = {a['arm']: a for a in
      json.load(open(SUB1 + '/imaging_summary.json'))['results']} \
    if os.path.exists(SUB1 + '/imaging_summary.json') else None

# Per-side k medians (u>0 / u<0, inclined wall-matched tracks, both
# sub-runs) — measured 2026-08-13 on the desktop working copy; the check
# script is inline in the session record. A's ~15% split is the u0-offset
# signature; D's ~60% split is its one-sided anomaly, reproducible.
SIDE_K = {'A': {'0000': (1.086, 1.272), '0001': (1.092, 1.292)},
          'D': {'0000': (1.182, 1.957), '0001': (1.243, 1.938)}}

A = R['A']
kA = A['k_track']['median']
vA = A['k_track']['v_insitu']
kcA = A['k_track_coincident']['median']
coinA, predA = A['n_pointing_coincident'], \
    A['pointing_coincidence']['n_predictable']

VERDICT_ARM = {'A': 'the reference arm',
               'D': ('consistent with A; its wall crossings are compressed '
                     'one-sidedly on +u — an open beam-side geometry anomaly '
                     '(occupancy and bench scale both check out)'),
               'C': ('qualitative only — its bench bundle is known-degenerate '
                     '(three boundary-ish hypers)'),
               'B': ('no pointing correlation (drift field not set by its '
                     'supply, degrador absent): positions usable, angles '
                     'not')}
V_CELL = {'A': '<strong>≈34–36</strong>', 'D': '<strong>≈37</strong>',
          'C': '(≈29)', 'B': '—'}

fleet_rows = []
for arm, det in (('A', 'det3'), ('D', 'det7'), ('C', 'det6'), ('B', 'det2')):
    r = R[arm]
    kc = (r.get('k_track_coincident') or {}).get('median')
    frac = r['n_pointing_coincident'] / r['pointing_coincidence'][
        'n_predictable']
    fleet_rows.append(
        f"<tr><td>{arm} ({det})</td><td>{r['n_2plane']:,}</td>"
        f"<td>{r['n_wall_matched']:,}</td>"
        f"<td>{r['n_pointing_coincident']:,} ({pct(frac)})</td>"
        f"<td>{kc:.2f}</td><td>{V_CELL[arm]}</td>"
        f"<td>{VERDICT_ARM[arm]}</td></tr>")

headon_rows = []
for arm in 'ADCB':
    r = R[arm]
    ir, ih = r.get('image_at_kphys_relonly'), r.get('image_at_kphys_headon')
    headon_rows.append(
        f"<tr><td>{arm}</td><td>{r['n_relonly']:,}</td>"
        f"<td>{r['n_headon']:,}</td>"
        f"<td>{ir['r_core']:.1f}</td><td>{ih['r_core']:.1f}</td></tr>")

w3_rows = []
for arm in 'ABCD':
    w = W3['arms'][arm]
    w3_rows.append(
        f"<tr><td>{arm}</td><td>{w['n']:,}</td>"
        f"<td>{w['convergence']['spread_wall_mm']:.0f}</td>"
        f"<td>{w['convergence']['spread_target_mm']:.0f}</td>"
        f"<td>{w['null']['spread_wall_mm']:.0f}</td></tr>")

sub1_section = ''
if R1:
    rows = []
    for arm in 'ADCB':
        r0, r1 = R[arm], R1[arm]
        c0, c1 = r0.get('k_track_coincident') or {}, \
            r1.get('k_track_coincident') or {}
        rows.append(
            f"<tr><td>{arm}</td>"
            f"<td>{c0.get('median', float('nan')):.2f} "
            f"(n={c0.get('n', 0):,})</td>"
            f"<td>{c1.get('median', float('nan')):.2f} "
            f"(n={c1.get('n', 0):,})</td>"
            f"<td>{r0['n_pointing_coincident']:,} / "
            f"{r1['n_pointing_coincident']:,}</td></tr>")
    side = []
    for arm in ('A', 'D'):
        s = SIDE_K[arm]
        side.append(
            f"<tr><td>{arm}</td><td>{s['0000'][0]:.2f} / {s['0000'][1]:.2f}"
            f"</td><td>{s['0001'][0]:.2f} / {s['0001'][1]:.2f}</td></tr>")
    sub1_section = f'''
<h2>Sub-run 0001 — an independent hour, a new code generation</h2>
<p>Sub-run 0001 was reconstructed with the post-restore <code>wft</code>
(w0/kw applied inside the fit, <code>angle_constants.applied</code> stamped in
the output metadata; the analysis detects the stamp and skips its post-hoc
correction). Same chain, independent data, different code generation:</p>
<table>
<tr><th>arm</th><th>k coincident, 0000</th><th>k coincident, 0001</th>
<th>confirmed tracks 0000 / 0001</th></tr>
{''.join(rows)}
</table>
<p>A and C reproduce to better than 3&nbsp;%. D's single-number k moves —
because D does not <em>have</em> a single angle scale. Splitting the
per-track estimator by which side of the plane the track crossed:</p>
<table>
<tr><th>arm</th><th>k (u&gt;0 / u&lt;0), sub-run 0000</th>
<th>k (u&gt;0 / u&lt;0), sub-run 0001</th></tr>
{''.join(side)}
</table>
<p>A's ~15&nbsp;% side asymmetry is stable and is the expected signature of
the un-surveyed in-plane offset u<sub>0</sub> (it biases the two sides in
opposite directions). D's ~60&nbsp;% split is far too large for any offset
and reproduces exactly across the two hours: its +u side agrees with the
fleet (k&nbsp;≈&nbsp;1.2) while its −u side wants k&nbsp;≈&nbsp;1.9 — the
same one-sided anomaly the wall crossings show, now measured as an angle
scale. The apparent 0000→0001 shift in D's headline k was only the mixture
weights moving with the coincident sample composition. B's k is unstable, as
expected for an arm with no usable angle information.</p>'''

html = f'''<!--note
title: Beam tracks image the He-3 target (run_145, arm A)
summary: First waveform-first tracking on a fully nTOF-matched beam run: arm A back-projects to a source at the origin, and the image sharpness gives an in-situ drift velocity. Full coverage incl. the head-on band; all four arms shown.
tags: X17, nTOF, tracking, imaging, campaign
date: 2026-08-13
-->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Beam tracks image the He-3 target (run_145, arm A)</title>
<meta name="description" content="First waveform-first tracking on a fully nTOF-matched beam run: arm A back-projects to a source at the origin; image sharpness gives an in-situ drift velocity.">
<style>
  :root {{ --ink:#1a1a2e; --muted:#5c5c6b; --accent:#c8433c; --rule:#e4e4ea;
          --card:#f6f6f9; }}
  html {{ color-scheme: light; }}
  body {{ margin:0 auto; max-width:860px; padding:2rem 1.2rem 4rem;
         font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
         color:var(--ink); background:#fff; }}
  h1 {{ font-size:1.7rem; line-height:1.25; margin:0.2rem 0 0.4rem; }}
  h2 {{ font-size:1.15rem; margin:2.2rem 0 0.6rem; border-bottom:1px solid var(--rule); padding-bottom:0.25rem; }}
  p.lede {{ font-size:1.05rem; color:var(--ink); }}
  .meta {{ color:var(--muted); font-size:0.88rem; margin-bottom:1.4rem; }}
  .verdict {{ background:var(--card); border-left:4px solid var(--accent);
             padding:0.8rem 1rem; margin:1.2rem 0; }}
  .tiles {{ display:flex; flex-wrap:wrap; gap:0.8rem; margin:1.2rem 0; }}
  .tile {{ flex:1 1 150px; background:var(--card); border-radius:8px;
          padding:0.7rem 0.9rem; min-width:140px; }}
  .tile .n {{ font-size:1.45rem; font-weight:650; letter-spacing:-0.01em; }}
  .tile .l {{ font-size:0.8rem; color:var(--muted); }}
  figure {{ margin:1.6rem 0; }}
  figure img {{ max-width:100%; height:auto; border:1px solid var(--rule);
               border-radius:6px; }}
  figcaption {{ font-size:0.88rem; color:var(--muted); margin-top:0.45rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.92rem; margin:0.8rem 0; }}
  th, td {{ text-align:left; padding:0.35rem 0.6rem; border-bottom:1px solid var(--rule); }}
  th {{ color:var(--muted); font-weight:600; font-size:0.84rem; }}
  code {{ background:var(--card); padding:0.08em 0.35em; border-radius:4px; font-size:0.88em; }}
  .warn {{ background:#fdf3e7; border-left:4px solid #d9922e; padding:0.7rem 1rem; margin:1.2rem 0; font-size:0.94rem; }}
  ul {{ padding-left:1.2rem; }}
  li {{ margin:0.3rem 0; }}
</style>
</head>
<body>

<h1>Beam tracks image the He-3 target</h1>
<p class="meta">run_145 · stat090_0000 · all four arms, A (det3) leading ·
nTOF partner 224670 · updated 2026-08-13 · PRELIMINARY</p>

<div class="verdict">
<strong>The waveform-first reconstruction, run on a fully nTOF-matched beam
run, back-projects arm&nbsp;A's tracks to a compact source at the origin — the
He-3 target is imaged.</strong> The reconstructed angle follows the
point-source expectation tan&thinsp;θ&nbsp;=&nbsp;u&nbsp;/&nbsp;235&nbsp;mm
across the plane, and the residual scale factor gives the first in-situ drift
velocity at beam conditions: <strong>v&nbsp;≈&nbsp;{vA:.1f}&nbsp;µm/ns</strong>,
15&nbsp;% below the clean-gas Magboltz prior of 42.6. A second step —
requiring an in-time hit in the SiPM wall segment the track points to
<em>and</em> in the plastic bar behind it — confirms {coinA:,} tracks
externally ({pct(coinA / predA)} of the predictable pool), with the
extrapolated tracks landing in the correct 100&nbsp;mm wall group and the
segment ordering reproduced exactly. Everything on this page is on the
<em>full-coverage</em> basis: the head-on band
(|tan&thinsp;θ|&nbsp;&lt;&nbsp;0.08), which the hits-era
<code>slope_reliable</code> gate used to discard, is measured natively by the
forward fit and is the middle of the image.
</div>

<div class="tiles">
  <div class="tile"><div class="n">{A['n_events']:,}</div><div class="l">events reconstructed (arm A, 1&nbsp;h sub-run)</div></div>
  <div class="tile"><div class="n">{A['n_wall_matched']:,}</div><div class="l">two-plane tracks, wall-matched (head-on incl.)</div></div>
  <div class="tile"><div class="n">{vA:.1f}&nbsp;µm/ns</div><div class="l">in-situ v_drift (k&nbsp;=&nbsp;{kA:.2f} vs prior)</div></div>
  <div class="tile"><div class="n">95&nbsp;%</div><div class="l">nTOF↔DREAM match efficiency of the run (slim)</div></div>
</div>

<h2>What ran</h2>
<p>run_145 (Aug&nbsp;5, production statistics at the run_67 optimum:
drift 700&nbsp;V on all four chambers, resist A540/B540/C525/D520&nbsp;V,
Ar/iC<sub>4</sub>H<sub>10</sub> 90/10, 20 samples × 60&nbsp;ns) is the only
last-week run with full nTOF matching: the slim pipeline joins DREAM events to
nTOF run 224670 on <code>eventId</code> at ~95&nbsp;% efficiency, so every
track knows which SiPM bars and plastics fired and when. The
<code>wft/</code> forward-model reconstruction ran over all four arms with
bench-transferred calibration bundles (template + sharing kernel as hardware;
v_drift seeded from Magboltz at 42.6&nbsp;µm/ns), per-plane angle-mapping
constants w0/kw applied, and the multi-track candidate sidecar on. Selection:
both planes fit OK,
|tan&thinsp;θ|&nbsp;&lt;&nbsp;1, and a same-arm in-time SiPM wall hit in the
slim record. No slope gate — the June bench re-analysis measured the forward
fit's head-on band <em>unbiased</em> (≤&nbsp;0.15°) at the same σ68 as the
inclined bands, and at nTOF that band is the tracks that point straight at
the target.</p>

<p><strong>Re-run 2026-08-19 on a corrected sharing kernel.</strong> The bench
bundles three of the four arms were seeded from carried an <em>inverted</em>
charge-sharing ladder — the ±2-strip copy larger than the ±1, which cannot
happen, since ±2 is reached only through ±1. They were refit with
c₂&nbsp;=&nbsp;0.6&nbsp;c₁ (arm&nbsp;A's det3 had carried 1.14, B's det2 1.53,
D's det7 1.75; C's det6 was always physical at 0.82 and is unchanged), and
everything on this page — numbers, figures and event displays — was
regenerated from that reconstruction. The effect on the result is small and
was measured, not assumed: on identical events the corrected kernel steepens
tan&thinsp;θ by 0.5&nbsp;% in x and 3.8&nbsp;% in y, but the angle scale here
is <em>fitted</em>, so k absorbs nearly all of it — arm&nbsp;A's in-situ
velocity moves 36.1&nbsp;→&nbsp;36.2&nbsp;µm/ns, the focus tightens slightly,
and the externally confirmed fraction is unchanged at 51&nbsp;%. Record:
<code>ntof_tracking/RUN145_R06_2026-08-19.md</code>.</p>

<p><strong>Corrected 2026-08-20: the in-plane sign, and the pinwheel.</strong>
Two geometry errors, both in this analysis rather than in the reconstruction,
and together they were the dominant defect in the image.
<em>(i)</em> The sign of the strip coordinate within the plane — flagged
provisional in <code>reco/geometry.py</code> and never verified — was applied
to the <em>angle</em> (by the fitted sign of the pointing slope) and not to
the <em>position</em>. That is a mirror about the strip-plane centre, and the
chambers are <strong>pinwheeled</strong>: each plane centre sits ~16&nbsp;mm
off the beam axis, so the mirror displaces the reconstructed source by
2&nbsp;×&nbsp;pinwheel, in opposite global directions for opposing arms.
<em>(ii)</em> The point-source relation used the distance to the plane
<em>centre</em> with the lever measured from the centre, instead of the
perpendicular distance with the lever measured from the perpendicular foot —
a spurious 0.07 offset in tan&thinsp;θ.</p>

<p>The fingerprint is unmistakable and it is scale-free. Where the pointing
band crosses tan&thinsp;θ&nbsp;=&nbsp;0 is the foot of the perpendicular from
the source; it involves no angle scale, no drift velocity and no part of the
bench transfer. Read that way, the four arms said:</p>

<table>
<tr><th>arm</th><th>measures</th><th>source, old convention</th>
    <th>source, corrected</th></tr>
<tr><td>A</td><td>global X</td><td>−21.8 mm</td><td><strong>−10.9 mm</strong></td></tr>
<tr><td>C</td><td>global X</td><td>+41.7 mm</td><td><strong>−7.1 mm</strong></td></tr>
<tr><td>B</td><td>global Z</td><td>−38.0 mm</td><td>+6.5 mm</td></tr>
<tr><td>D</td><td>global Z</td><td>+36.2 mm</td><td>−5.2 mm</td></tr>
</table>

<p>Opposing arms have to see the same source. Under the old convention A and C
disagreed by 64&nbsp;mm&nbsp;=&nbsp;2&nbsp;×&nbsp;(P<sub>A</sub>&nbsp;+&nbsp;P<sub>C</sub>),
which is the arithmetic of the mirror; corrected they agree to 4&nbsp;mm and
every arm lands within ~11&nbsp;mm of the beam axis. Three independent things
improved with it and one did not move, which is the right signature:
arm&nbsp;A's median axis-miss 34.6&nbsp;→&nbsp;14.8&nbsp;mm and its
r&nbsp;&lt;&nbsp;10&nbsp;mm fraction 18.5&nbsp;→&nbsp;31.4&nbsp;%; the external
SiPM/plastic confirmation rate 51.2&nbsp;→&nbsp;54.0&nbsp;%; the two
independent angle-scale estimators, which used to disagree badly, now agree
(image focus scan 0.75&nbsp;→&nbsp;1.18 against per-track 1.17). What did
<em>not</em> move is the in-situ drift velocity, 36.2&nbsp;→&nbsp;36.3&nbsp;µm/ns
— as it should not, because the defect was an offset and not a scale. Record:
<code>ntof_tracking/RUN145_ALIGNMENT_2026-08-20.md</code>.</p>

<h2>The pointing correlation</h2>
<figure>
{img(NOTE_FIGS + '/fig1_tan_vs_u.png')}
<figcaption><strong>Fig 1 — the money plot.</strong> Reconstructed
tan&thinsp;θ (x&nbsp;plane) vs track position u on the strip plane, for
two-plane, wall-matched tracks. The band lies on the point-source line
tan&thinsp;θ&nbsp;=&nbsp;u/L (dashed) across the plane. The horizontal row at
tan&thinsp;θ&nbsp;≈&nbsp;0 is the head-on band: at small |u| it sits
<em>on</em> the point-source line (tracks aimed straight at the target — real
signal); at large |u| it is the isochronous background, which the k
estimators exclude by construction. Shaded: |u|&nbsp;&gt;&nbsp;130&nbsp;mm,
where plane-edge acceptance and the 20-sample window truncation compress the
band — excluded from calibration.</figcaption>
</figure>

<h2>The image</h2>
<figure>
{img(NOTE_FIGS + '/fig2_image.png')}
<figcaption><strong>Fig 2 — top-down view of the back-projection.</strong>
Each track is extrapolated to its closest approach to the beam axis (the
capsule axis). Left: with the Magboltz-prior drift velocity. Right: with the
in-situ scale — the focal spot tightens onto the origin, inside the
r&nbsp;=&nbsp;10&nbsp;mm capsule. The X-shaped wings are the defocused fan;
the thin horizontal ridge is the head-on band painting arm&nbsp;A's line of
sight through the capsule (k-invariant, so it appears in both panels); the
ridge sits ~5&nbsp;mm off axis, consistent with the un-surveyed in-plane
strip offset.</figcaption>
</figure>

<h2>The in-situ calibration</h2>
<p>Positions never depend on v_drift; angles scale as 1/v. A ray from the
origin crossing the strip plane at u must have tan&thinsp;θ&nbsp;=&nbsp;u/L
exactly, so every inclined track measures the angle scale directly:
k<sub>i</sub>&nbsp;=&nbsp;(u<sub>i</sub>/L)/tan&thinsp;θ<sub>i</sub>. This is
the one place the head-on band cannot contribute — k divides by tan — so the
estimator keeps a |tan&thinsp;θ|&nbsp;&gt;&nbsp;0.1 floor.</p>
<figure>
{img(NOTE_FIGS + '/fig3_kdist.png')}
<figcaption><strong>Fig 3.</strong> Per-track angle scale for inclined tracks
(|tan&thinsp;θ|&nbsp;&gt;&nbsp;0.1, 40&nbsp;&lt;&nbsp;|u|&nbsp;&lt;&nbsp;130&nbsp;mm,
{A['k_track']['n']:,} tracks). Median k&nbsp;=&nbsp;{kA:.2f} →
v&nbsp;=&nbsp;42.6/{kA:.2f}&nbsp;=&nbsp;{vA:.1f}&nbsp;µm/ns (per-plane angle
constants w0/kw applied, a 0.6&nbsp;% effect on this arm). The spread
(MAD&nbsp;≈&nbsp;{A['k_track']['mad']:.1f}) is per-track angle resolution plus
real target extent, not scale uncertainty — the median's statistical error is
at the percent level. On the externally confirmed (pointing-coincident)
subset the median is {kcA:.2f}.</figcaption>
</figure>

<table>
<tr><th>quantity</th><th>value</th><th>comment</th></tr>
<tr><td>v prior (Magboltz, clean 90/10, 233&nbsp;V/cm)</td><td>42.6&nbsp;µm/ns</td><td>bundle as seeded</td></tr>
<tr><td>v in-situ, imaging (arm A)</td><td><strong>{vA:.1f}&nbsp;µm/ns</strong></td><td>k&nbsp;=&nbsp;{kA:.2f}, |u|&nbsp;&lt;&nbsp;130&nbsp;mm; coincident subset {kcA:.2f} → {42.6 / kcA:.1f}</td></tr>
<tr><td>bench wet-gas reference at this field</td><td>23.3&nbsp;µm/ns</td><td>Ar/iso 95/5 + H<sub>2</sub>O curve</td></tr>
</table>
<p>{vA:.1f} sits between the clean-gas prediction and the wet bench curve —
consistent with mildly wet gas after a month of flow, but deliberately quoted
as an <em>effective</em> angle scale: the 20-sample readout window truncates
the deepest columns and compresses large angles, and that compression is
partially absorbed into k. For pointing and imaging purposes the effective
scale is the right thing to use; separating true v from truncation needs the
per-angle treatment.</p>

<h2>What the head-on band adds</h2>
<p>Dropping the slope gate is not a compromise — the newly admitted band
images as sharply as the inclined tracks on every arm, at zero cost to any
calibration number (it is k-invariant, which is precisely why it can never
bias the scale):</p>
<table>
<tr><th>arm</th><th>inclined (old basis)</th><th>head-on added</th>
<th>r<sub>core</sub> inclined [mm]</th><th>r<sub>core</sub> head-on [mm]</th></tr>
{''.join(headon_rows)}
</table>
<figure>
{img(FULLCOV + '/image_A_headon_cmp.png')}
<figcaption><strong>Fig 4 — the band the gate used to discard (arm A), at the
physical k.</strong> Left: closest-approach distributions for the previously
kept inclined population and the head-on band — the head-on tracks peak at
the axis at least as sharply; the far peak at |u|&nbsp;≈&nbsp;175&nbsp;mm is
the plane-edge population, outside the core metric. Right: the head-on band
alone, top-down — the line-of-sight ridge with its brightest bin at the
origin. (On arm D the head-on band is the <em>cleanest</em> population,
r<sub>core</sub> 9.2 vs 18.8&nbsp;mm — consistent with D's +u anomaly living
in the inclined tracks' position–angle combination.)</figcaption>
</figure>

<h2>Step 2 — the wall + plastic pointing coincidence</h2>
<p>Every DREAM trigger is a same-arm SiPM-wall AND plastic coincidence, and
the slim record says <em>which</em> wall segment (4 groups of 100&nbsp;mm,
96&nbsp;mm past the strips) and which plastic bar (two 200&nbsp;mm bars,
~188&nbsp;mm past the strips) fired. The second step extrapolates each track
to both planes and requires an in-time hit
(−100&nbsp;&lt;&nbsp;Δt&nbsp;&lt;&nbsp;60&nbsp;ns) in the <em>predicted</em>
segment and bar: {coinA:,} of {predA:,} predictable arm-A tracks pass
({pct(coinA / predA)}). The head-on tracks take part — their prediction is
position-dominated and needs no angle lever.</p>
<figure>
{img(NOTE_FIGS + '/fig5_coincidence.png')}
<figcaption><strong>Fig 5 — the image before (left) and after (right) the
pointing-coincidence requirement.</strong> The cut removes background
preferentially and the focal spot fraction inside the capsule rises
(percentages printed in-panel).</figcaption>
</figure>
<figure>
{img(NOTE_FIGS + '/fig6_wall_pointing.png')}
<figcaption><strong>Fig 6 — external position truth.</strong> Track u
extrapolated to the wall plane, split by which SiPM segment pair actually
fired (colour), with each group's geometric span shaded. Full coverage:
medians +94&thinsp;/&thinsp;+41&thinsp;/&thinsp;−70&thinsp;/&thinsp;−125&nbsp;mm
in exactly the geometric order, 48–67&nbsp;% inside their 100&nbsp;mm group.
The wall knows nothing the chamber told it — this lever arm is
external.</figcaption>
</figure>

<h2>With and without the SiPM requirement</h2>
<p>The same-arm wall-hit requirement is the baseline event selection above;
these two comparisons show what it actually does. The DREAM trigger is an OR
over arms, so a reconstructed track without a same-arm wall hit is usually a
real track whose trigger came from elsewhere — the requirement removes about
a fifth of tracks and mostly cleans edges.</p>
<figure>
{img(NOTE_FIGS + '/fig7_cmp_tan.png')}
<figcaption><strong>Fig 7.</strong> The pointing correlation for all
reconstructed tracks (left) vs same-arm SiPM wall hit required (right). The
correlation exists either way; the requirement sharpens the band's contrast
against the diffuse background.</figcaption>
</figure>
<figure>
{img(NOTE_FIGS + '/fig8_cmp_image.png')}
<figcaption><strong>Fig 8.</strong> The back-projection under the same two
selections, at the in-situ scale. The focal spot is present in both; the
requirement mostly removes the off-axis haze.</figcaption>
</figure>

<h2>The event, from the waveforms up</h2>
<p>One golden two-plane track (evt&nbsp;36245), shown at every level the
chain sees it — the same event display set the run_79 preliminaries used,
regenerated on run_145 with the frozen reconstruction and slim
matching.</p>
<figure>
{img(DISPLAYS + '/evt36245_waveforms.png')}
<figcaption><strong>Fig 9 — the raw DREAM waveforms</strong> across both
strip planes: the inclined charge ladder in time-per-strip that the forward
model fits directly. Geometry comes from these, never from per-strip hit
times.</figcaption>
</figure>
<figure>
{img(DISPLAYS + '/evt36245_projections.png')}
<figcaption><strong>Fig 10 — the fitted track in both local projections</strong>,
with the per-strip model overlay and residuals.</figcaption>
</figure>
<figure>
{img(DISPLAYS + '/evt36245_3d.png')}
<figcaption><strong>Fig 11 — the same track in the 3D model</strong>: out of
the drift gap, through the SiPM wall segment that fired (solid), into the
plastic behind it.</figcaption>
</figure>
<figure>
{img(DISPLAYS + '/wall_segment_tour_run145.png')}
<figcaption><strong>Fig 12 — one confirmed track per wall segment</strong>
(projections view): four golden events, one landing on each of arm A's four
SiPM segments.</figcaption>
</figure>

<h2>The four fans — every arm, colored by its wall segment</h2>
<p>The run_79 closing visual, remade on run_145 for all four arms at once:
every wall-matched track, colored by the SiPM segment that triggered it,
drawn through the 3D model. Two things should be visible per arm and both
are, where the hardware allows: the four fans separate at the wall (that is
the matcher + geometry) and sweep back through the He-3 capsule (that is the
tracking). The number pair under each panel — the spread of the four bundle
medians at the wall vs at the target plane — is the same statement measured;
the shuffled-label null is the scale it has to beat.</p>
<figure>
{img(WALL3D + '/wall3d_run145_all_arms.png')}
<figcaption><strong>Fig 13 — all four arms, tracks colored by fired wall
segment</strong> (full coverage, per-arm in-situ angle scale).
A is textbook: 250&nbsp;mm apart at the wall, 13&nbsp;mm at the target
(null: 12&nbsp;mm — the convergence is real, not subsample noise). C
matches. B separates at the wall on positions alone. D's fans separate by
only 135&nbsp;mm — its one-sided +u compression, here visible directly as
the two dense crossing knots.</figcaption>
</figure>
<table>
<tr><th>arm</th><th>tracks drawn</th><th>wall spread [mm]</th>
<th>target spread [mm]</th><th>null at wall [mm]</th></tr>
{''.join(w3_rows)}
</table>

<h2>The other three arms</h2>
<p>The same chain on arms D, C, B (sub-run 0000). Where the with-coincidence
and all-inclined estimators disagree, the background fraction is high and the
<em>coincident</em> column is the one to read.</p>
<table>
<tr><th>arm</th><th>2-plane tracks</th><th>wall-matched</th><th>coincident</th>
<th>k (coincident)</th><th>v_insitu [µm/ns]</th><th>verdict</th></tr>
{''.join(fleet_rows)}
</table>
<p>D's high statistics (46k reconstructed events — twice the other arms)
match its known occupancy. The v ordering A&nbsp;≈&nbsp;D&nbsp;&gt;&nbsp;C is
not yet a gas statement: C's number is bundle-limited and B's field is not
nominal, so the clean gas-chain reading needs the C recalibration and the D
anomaly closed first.</p>
{sub1_section}

<h2>What this does not show (yet)</h2>
<ul>
<li><strong>The axial (Y) image is inconclusive.</strong> The y-plane pointing
correlation is flat and the Y-profile of near-axis tracks
(<a href="#yprof">below</a>) shows no clear capsule concentration — y-plane
angle quality and the in-plane offsets need work before the second imaging
coordinate is real.</li>
<li><strong>In-plane offsets are un-surveyed.</strong> The strip-map zero per
arm is provisional; the ~5&nbsp;mm off-axis ridge in Fig&nbsp;2 is its likely
signature. A naive (u<sub>0</sub>,&nbsp;k) grid scan rails at its edge — the
proper two-parameter likelihood fit is the open item.</li>
<li><strong>Sub-run 0002 is not pulled yet.</strong> Sub-runs 0000 and 0001
are reconstructed and imaged (all four arms); the run's third hour is staged
on EOS only.</li>
<li><strong>Arm D's +u compression is unexplained.</strong> Reflection and
map-mirroring are ruled out, the bench position scale is exact, and the wall
instrumentation is healthy on that side — the remaining suspects are the
beam-side geometry description of D's +u half or D's own reconstruction
there. The per-side angle scale (previous section) is the same anomaly as a
number: −u wants k&nbsp;≈&nbsp;1.9 while +u agrees with the fleet. The
multi-arm confirmed-track test is the queued instrument.</li>
<li><strong>Single-k calibration.</strong> First-order only; the
|u|&nbsp;&gt;&nbsp;130 compression is excluded, not corrected.</li>
</ul>
<figure id="yprof">
{img(NOTE_FIGS + '/fig4_yprofile.png')}
<figcaption><strong>Fig 14 — the axial profile.</strong> Global Y at closest
approach for near-axis tracks (r&nbsp;&lt;&nbsp;25&nbsp;mm): broad, with no
decisive concentration inside the He-3 gas extent. The second imaging
coordinate is open.</figcaption>
</figure>

<h2>Reproduce</h2>
<p><code>ntof_tracking/run145_target_imaging.py</code> (selection, pointing
fits, per-track k, pointing coincidence, imaging summary JSON) ·
<code>ntof_tracking/run145_displays.py</code> (event displays and the wall
tour from the slim file) ·
<code>ntof_tracking/run145_wall_segment_3d.py</code> (the four-arm colored
3D) · figures for this page from <code>figs_run145_A*.py</code> on the
desktop working copy; page built by
<code>ntof_tracking/make_run145_note.py</code>. Data: DREAM
<code>run_145/stat090_0000</code> waveforms, slim
<code>ntof_hits_run_145_stat090_0000_224670.root</code>, bench bundles per
arm as frozen for MPGD26. The run_79 preliminaries this page superseded live
in the July&nbsp;31 daily summary deck.</p>

</body>
</html>
'''

open(OUT, 'w').write(html)
print('wrote', OUT, f'{len(html) / 1e6:.2f} MB')
