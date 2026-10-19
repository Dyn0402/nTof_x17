#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_note.py -- the published, self-contained write-up.

Same numbers as report.html, read from the same JSONs, but written for someone
who was not in the room: every figure embedded as a data: URI so it works
offline, and the reasoning spelled out rather than assumed.

    .venv/bin/python -m ntof_active_area.make_note
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \\
        ntof_active_area/note_active_area.html --tags "X17, nTOF, micromegas, simulation, Geant4"
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from common import mx17_active_area as JUNE
from .clusters import BENCH_ALIAS, CHAMBERS, PITCH_MM
from .make_report import summarise

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'


def img(name: str, alt: str, caption: str, width: str = '100%') -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}" '
            f'style="width:{width}">\n<figcaption>{caption}</figcaption></figure>')


def _row(cells, tag='td'):
    return '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'


CSS = """
:root{
  --ink:#141413; --muted:#5c5b57; --faint:#8a8983;
  --rule:#e3e2dd; --surface:#ffffff; --panel:#f7f6f3;
  --beam:#2a78d6; --june:#c9541f; --good:#0f7a52; --bad:#b3261e;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#eceae4; --muted:#a9a79f; --faint:#807e77;
    --rule:#33322e; --surface:#17171a; --panel:#1f1f22;
    --beam:#5a9df0; --june:#e8834a; --good:#4fbf90; --bad:#e8776c;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font:16.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-text-size-adjust:100%}
.wrap{max-width:820px;margin:0 auto;padding:2.6rem 1.15rem 5rem}
h1{font-size:2.0rem;line-height:1.2;margin:0 0 .35rem;letter-spacing:-.02em}
h2{font-size:1.32rem;margin:3rem 0 .8rem;padding-top:1.1rem;border-top:1px solid var(--rule);letter-spacing:-.01em}
h3{font-size:1.06rem;margin:2rem 0 .5rem}
p{margin:0 0 1.05rem}
.dateline{color:var(--faint);font-size:.9rem;margin-bottom:1.8rem}
.lede{font-size:1.12rem;color:var(--ink)}
.answer{background:var(--panel);border-left:4px solid var(--beam);
  border-radius:0 8px 8px 0;padding:1.15rem 1.35rem;margin:1.8rem 0}
.answer p:last-child{margin-bottom:0}
.warn{background:var(--panel);border-left:4px solid var(--june);
  border-radius:0 8px 8px 0;padding:1rem 1.3rem;margin:1.6rem 0}
.warn p:last-child{margin-bottom:0}
.numbers{display:flex;gap:1rem;flex-wrap:wrap;margin:1.8rem 0}
.num{flex:1 1 210px;background:var(--panel);border-radius:10px;padding:1rem 1.15rem}
.num .big{font-size:1.85rem;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.num .was{color:var(--faint);text-decoration:line-through;font-size:1rem;font-weight:400}
.num .lab{color:var(--muted);font-size:.86rem;margin-top:.3rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1.2rem 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:460px}
th,td{border-bottom:1px solid var(--rule);padding:.46rem .6rem;text-align:right;
  vertical-align:top}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600;border-bottom:2px solid var(--rule);
  white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
figure{margin:2rem 0}
img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:8px;
  background:#fff}
figcaption{color:var(--muted);font-size:.87rem;margin-top:.55rem;line-height:1.5}
code{background:var(--panel);padding:.1rem .32rem;border-radius:4px;
  font-size:.87em;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--panel);padding:.85rem 1rem;border-radius:8px;overflow-x:auto;
  font-size:.83rem;line-height:1.5}
pre code{background:none;padding:0}
ul,ol{padding-left:1.25rem;margin:0 0 1.05rem}
li{margin-bottom:.42rem}
.beam{color:var(--beam);font-weight:600}
.june{color:var(--june);font-weight:600}
.ok{color:var(--good);font-weight:600}
.no{color:var(--bad);font-weight:600}
.small{font-size:.9rem;color:var(--muted)}
hr{border:none;border-top:1px solid var(--rule);margin:2.4rem 0}
"""


def build() -> str:
    mm = json.loads((OUT / 'results_mm.json').read_text())
    sc = json.loads((OUT / 'results_scint.json').read_text())
    s = summarise(mm)
    b = sc['plastic_lr_boundary']
    ws = sc['wall_segments']

    # per-chamber table
    rows = []
    for ch in CHAMBERS:
        e = mm['chambers'][ch]
        v = e['planes']['v']
        jb = JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]['y']
        if v['lo_determined'] and v['hi_determined']:
            beam = (f"<span class=\"beam\">{v['live_lo_mm']:.1f} – "
                    f"{v['live_hi_mm']:.1f}</span>")
            span = f"{v['live_hi_mm'] - v['live_lo_mm']:.1f}"
        else:
            beam, span = '<span class="no">not measurable</span>', '—'
        rows.append(_row([f'<b>{ch}</b> <span class="small">({e["bench_alias"]})</span>',
                          f"{e['n_pairs']:,}", beam, span,
                          f'<span class="june">{jb[0]:.1f} – {jb[1]:.1f}</span>',
                          f'{jb[1] - jb[0]:.1f}']))

    scint_rows = []
    for key, label, nominal in (('plastic_v', 'plastic bar, half-length along beam', 150.0),
                                ('plastic_u', 'plastic pair, half-width', 200.0),
                                ('wall_v', 'SiPM wall, half-length along beam', 250.0)):
        f = sc[key]
        scint_rows.append(_row([
            label, f'{nominal:.0f}', f"{f['half_mm']:.0f} ± {f['half_err_mm']:.0f}",
            f"{f['sigma_mm']:.0f}", f"{f['contrast']:.2f}",
            '<span class="no">not constrained</span>']))

    dead = [(k, i + 1) for k, v in mm['connector_health'].items()
            for i, f in enumerate(v) if f < 0.15]
    dead_txt = ', '.join(f'{k} connector {i} '
                         f'({(i-1)*64*PITCH_MM:.0f}–{(i*64-1)*PITCH_MM:.0f} mm)'
                         for k, i in dead) or 'none'

    return f"""<!--note
title: The chambers were bigger than we thought
summary: The MX17 active area in the Geant sim was an unsourced 38 x 34 cm guess. Beam data says 39.9 x 36.0 cm, and the 4 cm goes missing on the beam axis.
tags: X17, nTOF, micromegas, simulation, Geant4
-->
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The chambers were bigger than we thought</title>
<meta name="description" content="The MX17 active area in the Geant simulation was an unsourced 38 x 34 cm estimate. Beam data says 39.9 x 36.0 cm — 11 % more area — and the 4 cm goes missing on the beam axis, not the tangential one.">
<style>{CSS}</style>
</head><body><div class="wrap">

<h1>The chambers were bigger than we thought</h1>
<p class="dateline">2026-08-11 · MX17 / n_TOF EAR2 · run_79, 215 481 DREAM events</p>

<p class="lede">Every acceptance number the MX17 simulation has ever produced
used a micromegas active area of 38 × 34 cm. That figure was an estimate made
at the start of the project and never checked. The chambers are
<strong>39.9 × 36.0 cm</strong> — 11 % more area — and, more awkwardly, the
4 cm that <em>is</em> missing goes missing on a different axis than the
simulation assumed.</p>

<div class="numbers">
  <div class="num"><div class="big beam">39.9 cm <span class="was">38.0</span></div>
    <div class="lab">u — tangential. The full metallised strip region. No dead
    band at all on this axis.</div></div>
  <div class="num"><div class="big beam">36.0 cm <span class="was">34.0</span></div>
    <div class="lab">v — along the beam. 359.9 ± 1.8 mm, after ~19 mm of
    passivation at each end.</div></div>
  <div class="num"><div class="big">+11 %</div>
    <div class="lab">active area, and it is centred — nothing moved, only the
    size changed.</div></div>
</div>

<h2>How this came up</h2>

<p>The question was a broad one: the Geant geometry has active areas for the
chambers and for every scintillator, and most of them were guesses — could beam
data pin them down?</p>

<p>The first useful thing was to stop and read <code>SimConfig.hh</code>
properly. Almost every dimension in it carries a provenance comment and a survey
date: the SiPM wall measured 2026-07-15 and 07-17, the plastics 07-17 and 07-20,
the liquid scintillator vessel taken straight off a STEP export at
451.2 × 450.6 mm. Those are tape and CAD measurements, good to millimetres.</p>

<p>Two lines had no comment at all:</p>

<pre><code>double mm_size_u_cm    = 38.0;   // MM active area: u [cm]
double mm_size_v_cm    = 34.0;   // MM active area: v (along beam) [cm]</code></pre>

<p>So the job was much narrower than it first looked. Only the chambers needed
measuring — and, as it turns out, beam data cannot say anything useful about the
scintillators anyway, for reasons that are worth understanding and are in the
last section.</p>

<h2>What "active area" even means here</h2>

<p>An MX17 chamber reads out two planes of 512 strips each, at 0.78 mm pitch,
crossed at right angles. One plane measures the coordinate along the beam, the
other the tangential coordinate. 512 × 0.78 mm is 399.36 mm of metal, and that
is the same on both planes — the board is square.</p>

<p>But the chamber is not efficient over all of it. The strip plane is
<strong>passivated</strong> — covered — over a band about 19 mm wide at each end
of one coordinate. A particle landing in that band produces no collected charge
at all: not on one plane, not on the other. It is simply invisible.</p>

<div class="warn">
<p><strong>The band is on the beam axis.</strong> The passivated plane is the
chamber's FEU-Y plane, and detector-local Y is the coordinate along the beam —
what the simulation calls <code>v</code>. So the axis that loses 4 cm is
<code>v</code>, and <code>u</code> keeps its full width.</p>
<p>Measured against the 40 cm board, the old estimate took 2 cm off
<code>u</code>, where in fact nothing is lost, and 6 cm off <code>v</code>,
where 4 cm is. So it was wrong on both axes and wrong in different directions —
which is what an estimate that was never tied to a specific mechanism looks
like. Fixing the digits while leaving the axes muddled would reproduce the same
class of error, so this is the one thing to be careful about.</p>
</div>

{img('area_diagram.png', 'chamber active area diagram',
     'One chamber face. The grey dashed square is the metallised strip region — '
     'what the board actually is. The shaded bands are passivated, ~19 mm at '
     'each end of the beam axis. The blue rectangle is the measurement; the '
     'orange dashed one is what the simulation assumed.')}

<h2>The measurement</h2>

<h3>The trick: make the two planes agree</h3>

<p>The obvious thing — count hits per strip and see where they stop — does not
work, and fails in a way that would have been easy to miss. The outermost
channels on these boards are noisy. On three of the four chambers, the raw
occupancy <em>outside</em> the active area is <em>higher</em> than inside it:
3.6× on chamber B, 2.1× on C, 15× on D. A naive occupancy edge would put the
chamber boundary at the edge of the board, confidently and wrongly.</p>

<p>What fixes it is a physical fact about how these chambers work. An avalanche
in an MX17 splits its charge roughly 50/50 between the two strip planes. So a
real particle leaves <em>two</em> clusters — one on each plane, in the same
event, with matched charge. Noise does not: noise is uncorrelated between
planes, and when it does coincide, the charges do not balance.</p>

<p>So the observable used throughout is a <strong>paired track</strong>: exactly
one particle-like cluster on each plane, with the two charges within a factor
1.6 of each other. That single requirement is the whole measurement.</p>

{img('why_balance.png', 'raw occupancy versus paired tracks at the chamber edge',
     'Chamber B, at the far end of the beam axis, counted both ways and each '
     'normalised to its own level in the chamber interior. Counting every '
     'cluster (orange), the region beyond the chamber looks 3.6× busier than '
     'the chamber itself. Requiring a charge-balanced partner on the other '
     'plane (blue), tracks stop at 379 mm and there is nothing after it.')}

<h3>Why an edge is an edge and not a shadow</h3>

<p>The beam does not illuminate a chamber evenly — the source is the He-3 target
235 mm away plus the whole neutron flight path, so the rate varies by a factor
of five or so across the face. That could easily be mistaken for the detector
running out.</p>

<p>It cannot be, and the reason is scale. Illumination changes smoothly over
tens of millimetres. A physical boundary changes over one strip. So the analysis
looks for <em>steps</em>, and the big smooth gradient is not a nuisance to be
fitted away — it is the control that proves a step is a step.</p>

<p>In practice this meant abandoning the first approach. Fitting an
error-function turn-on to get a 50 % point kept driving the width parameter to
its lower bound, which is the fit saying the edge is sharper than it can
resolve. It is: the boundary is a <em>strip</em> boundary, and a strip is either
read or it is not. So the estimator became a per-strip live/dead question,
answered by walking outward from the middle of the chamber and comparing each
strip only to the strips just inside it — never a symmetric window, because that
lets the dead region set its own reference and the edge dissolves.</p>

{img('mm_maps.png', 'two-dimensional track maps for all four chambers',
     'Where the paired tracks land, chamber by chamber. Red is the June '
     'cosmic-bench active area, measured a month earlier with a completely '
     'different method; grey dashed is the metallised strip region. In every '
     'chamber the tracks stop at the red horizontal lines and run out to the '
     'grey vertical ones. The chamber-specific damage shows up too — see below.')}

<h2>The result</h2>

<div class="scroll"><table>
<thead>{_row(['chamber', 'paired tracks', 'beam: v active', 'span',
              'June telescope: v active', 'span'], 'th')}</thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
<p class="small">All values mm, detector-local, along the beam.</p>

<p>Three chambers give a hard edge at both ends, and they agree — with each
other, and with a measurement taken in June on the cosmic bench that shares
nothing with this one. That one used the M3 telescope as an external reference
and defined the edge as the 50 % efficiency point. This one has no external
reference at all and defines the edge as the outermost strip that ever takes
part in a track. Different beam, different reference, different definition.</p>

{img('two_methods.png', 'beam versus June telescope edge positions',
     'The same two edges, measured seven times. Blue is this analysis, orange '
     'the June cosmic bench against the M3 telescope. The grey line is the '
     'combined mean. Nothing is further than 2.4 mm from anything else, which '
     'is about as close as two methods that define "edge" differently can '
     'reasonably come.')}

<p>Combined over all seven measurements: the active span along the beam is
<strong>{s['v_span_mean']:.1f} ± {s['v_span_sd']:.1f} mm</strong>, centred at
{s['v_centre']:.1f} mm against a strip-plane centre of {s['strip_centre']:.1f} mm.
Centred to within 0.2 mm, so the size changes and the placement does not.</p>

<p>On the tangential axis there is <strong>no passivation</strong>, and the
argument is one line: chamber B is live at strip 0 <em>and</em> at strip 511.
The active width is the board.</p>

{img('mm_profiles.png', 'strip participation profiles',
     'How many paired tracks used each strip, for both planes of all four '
     'chambers. The broad hump is illumination. The cliffs are geometry. Black '
     'lines are the measured edges, red dotted the June telescope values.')}

{img('mm_edges_zoom.png', 'edge regions magnified',
     'Both ends of every plane, magnified, with the low-edge and high-edge '
     'windows placed side by side. The turn-offs on the v planes happen inside '
     'one or two strips — 1.5 mm — which is what makes them geometry rather '
     'than a fading illumination.')}

<h2>Three things that are broken, and are not geometry</h2>

<p>The same data shows real damage, and the whole point of separating it out is
that none of it belongs in a geometry constant. A simulation that quietly
absorbed a dead connector into its active area would be wrong in a way nobody
would ever find.</p>

<ul>
<li><strong>Chamber A lost a connector during the campaign.</strong> Its X-plane
connector 8 — strips 448–511, exactly one 64-channel connector — is completely
dead in run_79 on 26 July. It was <em>alive</em> on 18 July in run_55, at full
occupancy. So arm A read only 87.5 % of its tangential width for this run.
Dead in this run: {dead_txt}.</li>
<li><strong>Chamber D's tangential plane is mostly dark.</strong> Only two bands
produce tracks at all. D is not measurable in this run, and the analysis says so
rather than fitting a number to noise.</li>
<li><strong>Chamber C has a real interior dead stripe</strong> near u = 190 mm,
about 20 strips wide, visible as the vertical gap in its track map.</li>
</ul>

{img('mm_connectors.png', 'per-connector readout health',
     'Cluster occupancy per 64-strip connector, relative to each plane&rsquo;s '
     'interior. Chamber A&rsquo;s X connector 8 reads exactly 0.00.')}

<h2>The scintillators, and why they stayed as they were</h2>

<p>The original hope was to do all the detectors this way: take chamber tracks,
point them at the SiPM wall and the plastics, and read off where each detector
stops responding. The merged n_TOF ↔ DREAM sample for arm A has
{sc['n_tracks']:,} tracks with a full waveform reconstruction, angle included,
so the pointing is real.</p>

<p>It does not work, for two reasons that can both be measured rather than
argued about:</p>

<ul>
<li><strong>The pointing is blurry.</strong> Fitting the blur as a free parameter
gives <strong>σ = {b['sigma_mm']:.0f} ± {b['sigma_err_mm']:.0f} mm</strong> at the
plastic plane. The lever arm is 190 mm, the angle scale is about 0.8 of truth,
and not every particle that fired the trigger is the one the chamber
reconstructed.</li>
<li><strong>Roughly 40 % of the tags are accidental.</strong> The DREAM trigger
is an OR over all four arms, so an arm-A track routinely carries a tag that some
other particle in the same bunch produced. That is a flat pedestal under every
acceptance curve.</li>
</ul>

<p>With both floated, the fitted half-extents land on both sides of the survey —
which is the signature of a parameter the data does not constrain, not of a
disagreement:</p>

<div class="scroll"><table>
<thead>{_row(['quantity', 'survey [mm]', 'fit [mm]', 'blur σ [mm]', 'contrast',
              'verdict'], 'th')}</thead>
<tbody>{''.join(scint_rows)}</tbody>
</table></div>

<p>A tape measure beats this by a factor of about thirty. <strong>The surveyed
scintillator sizes stay.</strong> Knowing that, and knowing <em>why</em>, is
worth more than a number that would have looked like a measurement.</p>

<h3>What the pointing does establish</h3>

<p>Two things, and they are genuine checks on the geometry — just of placement
rather than size:</p>

<ul>
<li><strong>The plastic pair is centred on the chamber.</strong> The two 200 mm
bars abut on the chamber's centre line, so the L/R boundary is a prediction with
no free parameter: it should sit at 0. Measured, it sits at
<strong>{b['u0_mm']:+.1f} ± {b['u0_err_mm']:.1f} mm</strong>. This is the sharpest
statement the merge can make, because a boundary between two live detectors has
no acceptance falling off across it — unlike an edge.</li>
<li><strong>The wall segments map onto the chamber in the right order</strong>,
r = {ws['ordering_corr']:+.3f} across the four n_TOF channels. The slope of
{ws['slope_ratio']:.2f} against geometry is the accidental-tag dilution again,
and it matches the pedestal the acceptance fits find independently — two
different observables agreeing on the same contamination.</li>
</ul>

{img('scint_acceptance.png', 'scintillator acceptance curves',
     'Arm-A acceptance seen from chamber A. Top left is the one panel that '
     'constrains geometry: the L/R split of the plastic pair, landing where '
     'the survey says it should. The other three are what an unconstrained '
     'edge looks like — the model can put the boundary almost anywhere and '
     'still pass through the points.')}

{img('wall_segments.png', 'wall segment ordering', 'Mean chamber position of '
     'the tracks each wall segment tagged, against where that segment actually '
     'is. Ordered and monotonic; the compression towards the middle is the '
     'accidental tags, not a wrong segment pitch.', width='72%')}

<h2>A loose end worth pulling</h2>

<p>Why is the passivation on one axis and not the other? That asymmetry is odd
on its face, and the answer may already be written down elsewhere: the response
simulation's design notes record that the resistive strips contact copper bus
strips <strong>at both ends of one coordinate and nowhere in between</strong>.</p>

<p>A dead band of the same width, at the same two ends, on the same coordinate,
is what a covered bus termination would look like. Nobody has checked it — the
gerbers are in the repository and would settle it in an afternoon. If the bus
footprint measures ~19 mm, this stops being a coincidence and becomes an
explanation.</p>

<h2>What changed</h2>

<p><code>MX17_Full_Geant</code> now carries the measured values, with the
sources and the axis warning in the header comment. The mirrors in the plotting
scripts were updated too — one of them kept its own private copy of the width —
and every geometry figure was regenerated. The build is clean and a short run
checks 212 volumes for overlaps without a complaint.</p>

<p>The response simulation <code>MX17_Geant</code> never had the 38 × 34 guess:
it builds the 399.36 mm metallised window square, with real pad structure from
the gerbers, which is right for what it does. It does not model the passivation,
and that is now stated in its README, its CAD notes and the header where the
active width is declared, along with the warning that <em>any</em> sweep along
that axis, or any acceptance number, has to apply the band by hand.</p>

<p>One consequence worth stating plainly: <strong>every existing acceptance
number from the full simulation predates this.</strong> The chamber is 11 %
larger. Those runs need redoing.</p>

<h2>What this does not settle</h2>

<ul>
<li>The edge is measured on <strong>strip liveness</strong>, not efficiency. A
region that is live but inefficient would not show up here as an edge. June's
telescope measurement does carry efficiency information and puts the 50 % points
within 1–2 mm of these, so the two together bound it — but neither says the
efficiency <em>inside</em> the active area is uniform.</li>
<li><strong>Chamber D was never measured on beam data.</strong> Its readout was
too damaged in this run. The recommendation rests on three chambers from the
beam and all four from June.</li>
<li><strong>One run, two sub-runs.</strong> Nothing here tests whether the band
moves with time, though there is no mechanism by which it would.</li>
<li><strong>This is the sensitive area, not the gas volume.</strong> Ionisation
outside the collected region still happens. If a simulation needs the gas rather
than the readout, this is a lower bound.</li>
<li><strong>The scintillator sizes are not confirmed</strong>, only their
placement. A 10 cm error in the wall length would sit comfortably inside these
error bars.</li>
</ul>

<hr>
<p class="small">Analysis, figures and the reproducible chain:
<code>nTof_x17/ntof_active_area/</code> —
<code>.venv/bin/python -m ntof_active_area.run_all</code>. Inputs: run_79
sub-runs <code>stat090_0000</code> and <code>0001</code> (27 files, 215 481
events) and the arm-A n_TOF merge from run 224572. No hit times are used
anywhere, so the analysis stays inside the repository's reconstruction-basis
rule: strip identity and charge are detection quantities, and an occupancy edge
is exactly that.</p>

</div></body></html>
"""


def main():
    path = OUT / 'note_active_area.html'
    path.write_text(build())
    print('wrote', path, f'({path.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
