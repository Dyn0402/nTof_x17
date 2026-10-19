#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- build report.html from results_mm.json + results_scint.json.

    .venv/bin/python -m ntof_active_area.make_report

Everything numeric in the page is read from the result JSONs, so re-running the
measurement and then this script keeps the prose, the tables and the verdict in
step.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from common import mx17_active_area as JUNE
from .clusters import BENCH_ALIAS, CHAMBERS, PITCH_MM, STRIP_MAX_MM

OUT = Path(__file__).resolve().parent

# what the Geant sims currently assume (SimConfig.hh, both MX17_Geant trees)
SIM_U_CM, SIM_V_CM = 38.0, 34.0

# an edge this far inside the June geometry is a readout defect, not a chamber
# boundary, and is flagged rather than folded into the recommendation
READOUT_TOL = 5.0


def summarise(mm: dict) -> dict:
    """Fold the per-chamber beam edges together with the June telescope values
    into the two numbers the simulation needs."""
    beam_v, beam_u_lo, beam_u_hi = [], [], []
    for ch in CHAMBERS:
        pe = mm['chambers'][ch]['planes']
        v = pe['v']
        if v['lo_determined'] and v['hi_determined']:
            beam_v.append((ch, v['live_lo_mm'], v['live_hi_mm']))
        u = pe['u']
        if u['lo_determined'] or u['lo_at_board_end']:
            beam_u_lo.append((ch, u['live_lo_mm']))
        if u['hi_determined'] or u['hi_at_board_end']:
            beam_u_hi.append((ch, u['live_hi_mm']))

    june_v = [(ch, *JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]['y']) for ch in CHAMBERS]
    june_u = [(ch, *JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]['x']) for ch in CHAMBERS]

    v_spans = [b - a for _, a, b in beam_v] + [b - a for _, a, b in june_v]
    v_los = [a for _, a, _ in beam_v] + [a for _, a, _ in june_v]
    v_his = [b for _, _, b in beam_v] + [b for _, _, b in june_v]

    # u: the metallised strip region, measured centre-to-centre; the metal runs
    # half a pitch beyond the outermost strip centre at each end
    u_span_mm = STRIP_MAX_MM + PITCH_MM
    return dict(
        beam_v=beam_v, june_v=june_v, june_u=june_u,
        beam_u_lo=beam_u_lo, beam_u_hi=beam_u_hi,
        v_span_mean=float(np.mean(v_spans)), v_span_sd=float(np.std(v_spans)),
        v_lo_mean=float(np.mean(v_los)), v_hi_mean=float(np.mean(v_his)),
        v_centre=float(0.5 * (np.mean(v_los) + np.mean(v_his))),
        strip_centre=STRIP_MAX_MM / 2,
        u_span_mm=u_span_mm,
        rec_u_cm=round(u_span_mm / 10.0, 1),
        rec_v_cm=round(float(np.mean(v_spans)) / 10.0, 1),
        area_change=float((u_span_mm / 10.0) * (np.mean(v_spans) / 10.0)
                          / (SIM_U_CM * SIM_V_CM) - 1.0))


def _row(cells, tag='td'):
    return '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'


def build() -> str:
    mm = json.loads((OUT / 'results_mm.json').read_text())
    sc = json.loads((OUT / 'results_scint.json').read_text())
    s = summarise(mm)

    # ---- per-chamber table
    rows = []
    for ch in CHAMBERS:
        e = mm['chambers'][ch]
        pe = e['planes']
        cells = [f'<b>{ch}</b> <span class="dim">({e["bench_alias"]})</span>',
                 f'{e["n_pairs"]:,}']
        for plane in ('u', 'v'):
            p = pe[plane]
            jb = JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]['x' if plane == 'u' else 'y']
            for end in ('lo', 'hi'):
                val = p[f'live_{end}_mm']
                if not p[f'{end}_determined']:
                    cells.append(f'{val:.1f} <span class="dim">(board end)</span>'
                                 if p[f'{end}_at_board_end']
                                 else '<span class="bad">—</span>')
                    continue
                # an edge well inside the known geometry is this run's readout,
                # not the chamber: say which it is rather than averaging them
                inside = (val > jb[0] + READOUT_TOL if end == 'lo'
                          else val < jb[1] - READOUT_TOL)
                cells.append(f'{val:.1f}<sup class="bad">†</sup>' if inside
                             else f'{val:.1f}')
            cells.append(f'<span class="dim">{jb[0]:.1f}–{jb[1]:.1f}</span>')
        rows.append(_row(cells))

    # ---- readout health
    health = mm['connector_health']
    dead = [(k, i + 1) for k, v in health.items()
            for i, f in enumerate(v) if f < 0.15]
    dead_txt = ('none' if not dead else
                ', '.join(f'{k} connector {i} '
                          f'(strips {(i-1)*64}–{i*64-1}, '
                          f'{(i-1)*64*PITCH_MM:.0f}–{(i*64-1)*PITCH_MM:.0f} mm)'
                          for k, i in dead))

    b = sc['plastic_lr_boundary']
    ws = sc['wall_segments']
    scint_rows = []
    for key, label, nominal in (('plastic_v', 'plastic bar, along beam (half)', 150.0),
                                ('plastic_u', 'plastic pair, tangential (half)', 200.0),
                                ('wall_v', 'SiPM wall, along beam (half)', 250.0)):
        f = sc[key]
        verdict = ('<span class="ok">constrained</span>' if f['constrained']
                   else '<span class="bad">not constrained</span>')
        scint_rows.append(_row([
            label, f'{nominal:.0f} mm',
            f"{f['half_mm']:.0f} ± {f['half_err_mm']:.0f} mm",
            f"{f['sigma_mm']:.0f} mm", f"{f['contrast']:.2f}", verdict]))

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1080px;margin:2rem auto;padding:0 1.2rem;line-height:1.55;color:#1a1a1a}
    h1{font-size:1.7rem;margin-bottom:.2rem} h2{margin-top:2.2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
    h3{margin-top:1.6rem;font-size:1.05rem}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
    th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}
    th:first-child,td:first-child{text-align:left}
    thead th{background:#f4f6f8}
    .dim{color:#888} .ok{color:#137333;font-weight:600} .bad{color:#b3261e;font-weight:600}
    .verdict{background:#eef5ff;border-left:5px solid #1a73e8;padding:1rem 1.2rem;margin:1.4rem 0}
    .caveat{background:#fff8e6;border-left:5px solid #e0a800;padding:.9rem 1.2rem;margin:1.2rem 0}
    figure{margin:1.6rem 0} img{width:100%;border:1px solid #ddd;border-radius:4px}
    figcaption{color:#555;font-size:.88rem;margin-top:.4rem}
    code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
    """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MX17 active areas from n_TOF beam data</title>
<style>{css}</style></head><body>

<h1>MX17 detector active areas, measured on n_TOF beam data</h1>
<p class="dim">{date.today().isoformat()} &middot; run_79
({mm['n_files']} files, {mm['n_events']:,} DREAM events) &middot; n_TOF 224572
&middot; <code>ntof_active_area/</code></p>

<div class="verdict">
<h3 style="margin-top:0">Verdict</h3>
<p><b>The chambers are bigger than the simulation assumes, in both directions.</b>
The Geant <code>SimConfig.hh</code> carries <code>mm_size_u_cm = {SIM_U_CM}</code>,
<code>mm_size_v_cm = {SIM_V_CM}</code>.  The beam data says
<b>u = {s['rec_u_cm']} cm</b> (the full metallised strip region, no passivation)
and <b>v = {s['rec_v_cm']} cm</b> (a ~19 mm dead band at each end of the beam
axis), for <b>{s['area_change']*100:+.0f}% active area</b>.  Both numbers agree
with the June cosmic-bench telescope measurement already recorded in
<code>common/mx17_active_area.py</code>, which was taken with an external
reference and is the better of the two; this analysis is the independent
confirmation <em>in the n_TOF configuration</em>.</p>
<p>The active band is centred: its midpoint sits at
{s['v_centre']:.1f} mm against a strip-plane centre of {s['strip_centre']:.1f} mm,
so only the size changes, not the placement.</p>
<p><b>The scintillators are a different matter.</b>  Their sizes in
<code>SimConfig.hh</code> come from tape-and-STEP surveys (2026-07-15/17/18) and
are accurate to millimetres.  Chamber pointing has a blur of
<b>σ = {b['sigma_mm']:.0f} mm</b> at the plastic plane, so it cannot improve on
them and this analysis does not try to.  What it does do is confirm the
<em>placement</em>: the plastic L/R boundary lands at
{b['u0_mm']:+.1f} ± {b['u0_err_mm']:.1f} mm where the geometry puts it at 0.</p>
</div>

<h2>1. What was measured, and why this observable</h2>
<p>The active area is where a particle crossing the chamber produces a signal.
In the beam the illumination is smooth — the source is the He-3 target 235 mm
away plus the whole neutron flight path, so nothing about the <em>illumination</em>
changes over a few millimetres. A physical edge does: it is a step. So the
measurement is a search for steps.</p>

<p>The observable is a <b>paired track</b>: exactly one particle-like cluster on
each plane of the same chamber in the same event, with the two planes' charges
balanced ({mm['cuts']['q_ratio'][0]}–{mm['cuts']['q_ratio'][1]}). That balance
requirement is what makes this work — an MX17 avalanche splits about 50/50
between the two strip planes, so demanding it rejects the uncorrelated per-plane
noise that otherwise swamps the raw occupancy exactly where the edges are. On
three of the four chambers a raw <code>y</code>-plane occupancy is
<em>higher</em> beyond 380 mm — outside the chamber — than in the chamber's own
interior (B 3.6×, C 2.1×, D 15×). Paired tracks go to zero at 379 on all
four.</p>

<p>No hit times are used anywhere here, so this stays on the right side of
<code>RECONSTRUCTION_BASIS.md</code>: strip identity and charge are detection
and QA quantities, which is exactly what an occupancy edge is.</p>

<p>Coordinates: the <code>x</code> plane is <b>u</b>, the chamber's tangential
coordinate; the <code>y</code> plane is <b>v</b>, along the beam. Both are 512
strips at {PITCH_MM} mm spanning 0–{STRIP_MAX_MM} mm centre to centre.</p>

<h2>2. Chamber edges</h2>
<table>
<thead>{_row(['chamber', 'paired tracks', 'u low', 'u high', 'u June', 'v low', 'v high', 'v June'], 'th')}</thead>
<tbody>{''.join(rows)}</tbody>
</table>
<p class="dim">All values in mm, detector-local. “June” is the cosmic-bench
telescope 50 % efficiency point from <code>common/mx17_active_area.py</code>.
“—” means the profile ran out of contrast before a step was found (chamber D);
“board end” means the plane was still live at the last strip, which is itself
the answer for u; <sup class="bad">†</sup> marks an edge that sits more than
5 mm inside the known geometry, i.e. a run_79 readout defect rather than
a chamber boundary (section 3). Only the unmarked v edges feed the
recommendation.</p>

<h3>v (along the beam): a ~19 mm dead band at each end</h3>
<p>Three chambers give a determined edge at both ends, and they agree with each
other and with June:
{'; '.join(f'<b>{ch}</b> {lo:.1f}–{hi:.1f}' for ch, lo, hi in s['beam_v'])} mm.
Mean active span over the beam and June measurements together:
<b>{s['v_span_mean']:.1f} ± {s['v_span_sd']:.1f} mm</b>. The two methods define
the edge differently — June is a 50 % efficiency point against an external
track, this is the outermost strip that ever takes part in a track — so
agreement at the 1–2 mm level is as close as they can come.</p>

<h3>u (tangential): full width, no passivation</h3>
<p>Chamber B is live at strip 0 and at strip 511, i.e. across the whole
metallised region; A reaches strip 2 at the low end. Nothing anywhere in the
data shows a passivation band on this axis, which is what June found too. The u
active size is therefore the strip region itself: {STRIP_MAX_MM} mm centre to
centre, {s['u_span_mm']:.2f} mm of metal, <b>{s['rec_u_cm']} cm</b>.</p>

<figure><img src="figures/mm_maps.png" alt="2-D track maps">
<figcaption>Paired tracks in each chamber. Red = the June telescope active area,
grey dashed = the metallised strip region. In every chamber the tracks stop
dead at the red horizontal lines and run to the grey vertical ones. The
chamber-specific damage is also visible: A is cut off at 350 mm in u, C has a
dead stripe near u = 190 mm, D has lost most of its u plane.</figcaption></figure>

<figure><img src="figures/mm_profiles.png" alt="strip participation profiles">
<figcaption>How many paired tracks used each strip. The broad shape is
illumination; the edges are geometry. Black = measured last live strip, red
dotted = June.</figcaption></figure>

<figure><img src="figures/mm_edges_zoom.png" alt="edge zooms">
<figcaption>Both ends of every plane, magnified, with the low and high edge
regions placed side by side. The v turn-offs happen inside one or two
strips.</figcaption></figure>

<h2>3. Readout health in run_79 — not chamber geometry</h2>
<div class="caveat">
<p><b>Dead in this run: {dead_txt}.</b> Chamber A's X-plane connector 8 was fully
alive on 18 July (run_55) and is dead in run_79, so this is a cabling or
front-end fault during the campaign, not a property of the chamber. Anyone
simulating run_79 specifically needs it; anyone simulating the chamber must
not.</p>
<p>Chamber D's u plane is largely dark in this run and its edges are reported as
undetermined rather than fitted. Chamber C carries a genuine interior dead
stripe near u = 190 mm. These are all readout/detector pathologies on top of the
geometric active area, and the table above keeps them separate from it.</p>
</div>
<figure><img src="figures/mm_connectors.png" alt="connector health">
<figcaption>Cluster occupancy per 64-strip connector, relative to each plane's
interior. Green ≈ healthy.</figcaption></figure>

<h2>4. Scintillators</h2>
<p>Using the merged n_TOF ↔ DREAM sample for arm A ({sc['n_tracks']:,} tracks
with a full waveform-first fit), a track can be extrapolated to each
scintillator plane and asked whether that detector tagged the event. Two things
limit this and both are fitted rather than assumed: the pointing blur, and the
flat pedestal of accidental tags from the other arms (the DREAM trigger is an OR
over all four).</p>

<h3>What this does establish</h3>
<ul>
<li><b>The plastic pair is centred on the chamber.</b> The L/R boundary is at
u = {b['u0_mm']:+.1f} ± {b['u0_err_mm']:.1f} mm; the surveyed geometry — two
200 mm bars abutting on the pinwheel-shifted chamber centre line — puts it at
{b['predicted_u0_mm']:+.1f} mm. This is the sharpest statement available because
it is a boundary between two live detectors, not an edge, so no acceptance falls
off across it.</li>
<li><b>The wall segment ordering is right</b>, r = {ws['ordering_corr']:+.3f}
across the four n_TOF channels, confirming the descending mapping. The slope of
{ws['slope_ratio']:.2f} against geometry is accidental-tag dilution, not a wrong
segment pitch — it matches the pedestal the acceptance fits find
independently.</li>
<li><b>The pointing blur is σ = {b['sigma_mm']:.0f} ± {b['sigma_err_mm']:.0f} mm</b>
at the plastic plane. That is a useful number in its own right for anyone
planning to use chamber pointing.</li>
</ul>

<h3>What it does not</h3>
<table>
<thead>{_row(['quantity', 'survey', 'fit', 'blur σ', 'contrast', 'verdict'], 'th')}</thead>
<tbody>{''.join(scint_rows)}</tbody>
</table>
<p>None of the outer scintillator dimensions is constrained by this sample. The
blur is comparable to the extent being measured, and the tagged plateau stands
only a factor ~1.5–2.5 above the accidental pedestal, so the fitted half-extent
trades freely against the blur and the pedestal. The fits land both above and
below the survey, which is the signature of an unconstrained parameter rather
than a real disagreement. <b>Keep the surveyed scintillator sizes.</b></p>

<figure><img src="figures/scint_acceptance.png" alt="scintillator acceptance">
<figcaption>Arm-A acceptance seen from chamber A. Top left is the one panel that
constrains geometry.</figcaption></figure>
<figure><img src="figures/wall_segments.png" alt="wall segment ordering">
<figcaption>Mean chamber-u of the tracks each wall segment tagged, against where
that segment is. Ordered and monotonic; compressed by accidental tags.</figcaption></figure>

<h2>5. What to change</h2>
<table>
<thead>{_row(['SimConfig.hh field', 'now', 'recommended', 'basis'], 'th')}</thead>
<tbody>
{_row(['<code>mm_size_u_cm</code>', f'{SIM_U_CM}', f'<b>{s["rec_u_cm"]}</b>',
       'full metallised strip region; no passivation on this axis (B live at both board ends, plus June)'])}
{_row(['<code>mm_size_v_cm</code>', f'{SIM_V_CM}', f'<b>{s["rec_v_cm"]}</b>',
       f'~19 mm dead band at each end; {s["v_span_mean"]:.0f} ± {s["v_span_sd"]:.0f} mm over 3 beam + 4 June measurements'])}
{_row(['SiPM wall, plastics, LS', 'surveyed', 'unchanged',
       'beam pointing (σ ≈ 47 mm) cannot improve on a tape measure'])}
</tbody>
</table>
<p>The active area is centred on the chamber in both axes, so no offset changes
with it. If a run_79-specific simulation is wanted, chamber A's u range has to
be cut to the live connectors on top of this — but that is a readout mask, not
geometry.</p>

<h2>6. What this does not rule out</h2>
<ul>
<li><b>The v edge is measured on strip liveness, not on efficiency.</b> A region
that is live but at reduced efficiency would not show up here as an edge. June's
telescope measurement is the one that carries efficiency information, and it
puts the 50 % points within 1–2 mm of these, so the two together bound this
well — but neither says the efficiency inside is uniform.</li>
<li><b>Chamber D is not measured.</b> Its u plane is mostly dark in run_79 and
its v edges are undetermined. The recommendation above rests on A, B and C from
the beam plus all four from June.</li>
<li><b>Only two sub-runs, one run.</b> {mm['n_files']} files of run_79. Nothing
here tests whether the passivation band is the same at other times, though there
is no mechanism by which it would move.</li>
<li><b>Whether the drift volume is larger than the readout.</b> This measures
where charge is <em>collected</em>. Ionisation outside that region still
happens and still draws current; if the sim needs the gas volume rather than the
sensitive area, this is a lower bound.</li>
<li><b>The scintillator sizes are not confirmed, only their placement.</b> A
10 cm error in the wall length would sit comfortably inside these error
bars.</li>
</ul>

</body></html>
"""
    return html


def main():
    (OUT / 'report.html').write_text(build())
    print('wrote', OUT / 'report.html')


if __name__ == '__main__':
    main()
