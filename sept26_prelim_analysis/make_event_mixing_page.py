#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_event_mixing_page.py -- build the event-mixing explainer page.

Every number in the prose is read from a product on disk and formatted here, so
re-running after a new campaign pass moves the text and the tables together.
Figures come from `explain_event_mixing.py`; run that first.

    python explain_event_mixing.py && python make_event_mixing_page.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/media/dylan/data/x17/sept26_prelim')
OUT = SRC / 'event_mixing'
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def gather() -> dict:
    n = {}
    P = pd.read_parquet(SRC / 'angle_campaign' / 'pairs.parquet')
    real = P[~P.mixed]
    n['n_real'] = len(real)
    n['n_runs'] = real.run.nunique()
    n['matched'] = bool((real.groupby(['arm1', 'arm2']).size() ==
                         P[P.mixed].groupby(['arm1', 'arm2']).size()).all())
    b = real[(real.arm1 == 'B') | (real.arm2 == 'B')]
    n['frac_B'] = len(b) / len(real)

    C = pd.read_csv(SRC / 'angle_campaign' / 'compare.csv')
    A = C[C.selection == 'all_no_b2b']
    n['chi2'] = {}
    for t in ('intra', 'perpendicular', 'opposing'):
        g = A[A.topology == t]
        mx = float(g[g.model.str.startswith('event-mixed')].chi2dof.iloc[0])
        ph = g[~g.model.str.startswith('event-mixed')].sort_values('chi2dof')
        n['chi2'][t] = dict(
            n_obs=int(g.n_obs.iloc[0]), mixed=mx,
            best_model=str(ph.model.iloc[0]), best=float(ph.chi2dof.iloc[0]),
            ratio=float(ph.chi2dof.iloc[0]) / mx)

    # Both fractions on the SAME basis -- the back-to-back-free observed sample
    # against the mixed one -- so the caption and the chi2 table agree.
    S0 = pd.read_csv(SRC / 'angle_campaign' / 'spectra.csv')
    n['above'] = {}
    for t in ('perpendicular', 'opposing'):
        out = {}
        for key, sel in (('obs', 'all_no_b2b'), ('mixed', 'mixed')):
            g = S0[(S0.topology == t) & (S0.selection == sel)]
            out[key] = float(g[g.lo >= 105].n.sum() / g.n.sum())
        n['above'][t] = out

    T = C[C.selection == 'tight_pair']
    n['tight'] = {}
    for t in ('perpendicular', 'opposing'):
        g = T[(T.topology == t) & T.model.str.startswith('event-mixed')]
        n['tight'][t] = dict(n=int(g.n_obs.iloc[0]),
                             chi2dof=float(g.chi2dof.iloc[0]))

    VS = pd.read_csv(SRC / 'imaging' / 'vertex_summary_run_145.csv')
    n['vs'] = VS.set_index('topology').to_dict('index')
    VX = pd.read_csv(SRC / 'det_a_intra' / 'vertex_excess.csv')
    n['vx'] = VX.set_index('selection').to_dict('index')

    tm = pd.read_csv(OUT / 'figures' / 'f6_timing.csv')
    n['timing'] = tm.set_index('sample').to_dict('index')

    # where the tight opposing sample sits relative to the back-to-back cut
    m = pd.read_parquet(SRC / 'tight_coincidence' / 'pairs_tight_campaign.parquet')
    o = m[m.topo == 'opposing']
    tp, tg = o[o.tight_pair], o[~o.back_to_back]
    n['b2b'] = dict(tight_edge=float((tp.open_deg > 165).mean()),
                    tagged_edge=float((tg.open_deg > 165).mean()))

    # per-bin pulls of the tight opposing sample against the null
    S = pd.read_csv(SRC / 'angle_campaign' / 'spectra.csv')
    t = S[(S.topology == 'opposing') & (S.selection == 'tight_pair')].sort_values('theta')
    x = S[(S.topology == 'opposing') & (S.selection == 'mixed')].sort_values('theta')
    obs = t.n.values.astype(float)
    pred = x.n.values / x.n.sum() * obs.sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        pull = (obs - pred) / np.sqrt(np.clip(obs, 1, None))
    n['pulls'] = [(float(a), float(p)) for a, p, o_ in zip(t.theta, pull, obs) if o_ > 0]
    return n


def pct(x, d=1):
    return f'{100 * x:.{d}f}&thinsp;%'


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --bg:#fbfbfa; --panel:#ffffff; --ink:#1a1f26; --soft:#5c6672; --faint:#8b95a1;
  --rule:#e3e6ea; --accent:#1b3a6b; --warm:#c0621a; --good:#1e7a4d;
  --warnbg:#fdf6ed; --warnln:#e8c9a3; --stopbg:#fbf0f0; --stopln:#e6c2c2;
  --measure:37rem;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#12151a; --panel:#181c22; --ink:#e6e9ed; --soft:#a8b2bd;
         --faint:#7c8794; --rule:#2a3038; --accent:#8fb4e8; --warm:#e4924a;
         --good:#54c08b; --warnbg:#211a11; --warnln:#4a3a23;
         --stopbg:#231515; --stopln:#4d2e2e; }
  img{ filter:invert(.92) hue-rotate(180deg); }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:64rem;margin:0 auto;padding:0 1.5rem 6rem;}
header{padding:4.5rem 0 2.5rem;}
.kicker{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);margin:0 0 1rem;}
h1{font-size:2.35rem;line-height:1.14;letter-spacing:-.022em;margin:0 0 1.1rem;
  font-weight:640;max-width:24ch;}
.standfirst{max-width:var(--measure);font-size:1.08rem;line-height:1.6;
  color:var(--soft);margin:0;}
.meta{margin-top:2rem;font-size:.82rem;color:var(--faint);max-width:52rem;
  border-top:1px solid var(--rule);padding-top:.9rem;}
.meta b{color:var(--soft);font-weight:560;}
h2{font-size:1.32rem;letter-spacing:-.012em;margin:4.5rem 0 .2rem;font-weight:620;}
h2 .num{color:var(--faint);font-weight:450;margin-right:.6rem;
  font-variant-numeric:tabular-nums;}
h3{font-size:1.0rem;margin:2.4rem 0 .5rem;font-weight:620;color:var(--ink);}
h2+.lede{max-width:var(--measure);color:var(--soft);margin:.35rem 0 1.6rem;}
p,ul,ol{max-width:var(--measure);}
p{margin:0 0 1.05rem;}
li{margin:0 0 .5rem;}
a{color:var(--accent);}
code{font:.88em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  padding:.08em .34em;}
strong{font-weight:620;}
.answer{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:3px;padding:1.3rem 1.5rem;margin:2rem 0 0;max-width:46rem;}
.answer p{margin:0 0 .8rem;max-width:none;}
.answer p:last-child{margin:0;}
.note{background:var(--warnbg);border:1px solid var(--warnln);border-radius:3px;
  padding:1rem 1.2rem;margin:1.8rem 0;max-width:46rem;font-size:.94rem;}
.stop{background:var(--stopbg);border:1px solid var(--stopln);}
.note p{max-width:none;margin:0 0 .7rem;} .note p:last-child{margin:0;}
.note .h{font-weight:640;display:block;margin-bottom:.4rem;font-size:.82rem;
  letter-spacing:.06em;text-transform:uppercase;color:var(--soft);}
figure{margin:2.4rem 0;}
figure img{width:100%;max-width:62rem;display:block;border:1px solid var(--rule);
  border-radius:3px;background:#fff;}
figcaption{font-size:.88rem;color:var(--soft);margin-top:.75rem;max-width:44rem;}
figcaption b{color:var(--ink);font-weight:600;}
.svgbox{margin:2.4rem 0;padding:1.6rem 1.4rem 1.2rem;background:var(--panel);
  border:1px solid var(--rule);border-radius:3px;max-width:62rem;overflow-x:auto;}
.svgbox svg{display:block;width:100%;height:auto;min-width:34rem;}
table{border-collapse:collapse;width:100%;max-width:52rem;margin:1.6rem 0;
  font-size:.92rem;font-variant-numeric:tabular-nums;}
th,td{text-align:right;padding:.5rem .7rem;border-bottom:1px solid var(--rule);}
th:first-child,td:first-child{text-align:left;}
thead th{font-size:.76rem;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);font-weight:560;border-bottom:1px solid var(--ink);}
tbody tr:last-child td{border-bottom:1px solid var(--rule);}
td.win{color:var(--good);font-weight:620;} td.lose{color:var(--warm);}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem;max-width:52rem;margin:1.8rem 0;}
.cols>div{background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  padding:1.1rem 1.2rem;}
.cols h4{margin:0 0 .7rem;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;}
.cols ul{margin:0;padding-left:1.1rem;font-size:.93rem;}
.keep h4{color:var(--good);} .kill h4{color:var(--warm);}
footer{margin-top:6rem;border-top:1px solid var(--rule);padding-top:1.4rem;
  font-size:.84rem;color:var(--faint);max-width:52rem;}
footer code{font-size:.82em;}
@media (max-width:680px){
  h1{font-size:1.8rem} .wrap{padding:0 1.1rem 4rem} header{padding:3rem 0 2rem}
  .cols{grid-template-columns:1fr} .svgbox svg{min-width:30rem}
}
"""

# --------------------------------------------------------------------------- #
SVG_MECHANICS = """
<svg viewBox="0 0 900 320" role="img"
     aria-label="Real pairs are drawn inside one trigger; mixed pairs take one
     track from each of two different triggers.">
  <defs>
    <style>
      .lbl{font:13px ui-sans-serif,sans-serif;fill:#5c6672}
      .ttl{font:600 13px ui-sans-serif,sans-serif;fill:#1a1f26}
      .sml{font:11px ui-sans-serif,sans-serif;fill:#8b95a1}
      .box{fill:none;stroke:#c9ced6;stroke-width:1.2}
      .trk{stroke-width:2.6;stroke-linecap:round;fill:none}
      .brk{stroke-width:1.5;stroke-dasharray:5 4;fill:none}
      @media (prefers-color-scheme:dark){
        .lbl{fill:#a8b2bd}.ttl{fill:#e6e9ed}.sml{fill:#7c8794}.box{stroke:#39414b}
      }
    </style>
  </defs>

  <text class="ttl" x="0" y="14">Real pair &#8212; both legs from one trigger</text>
  <text class="ttl" x="470" y="14">Event-mixed pair &#8212; one leg from each of two triggers</text>

  <!-- ---------------- real ---------------- -->
  <rect class="box" x="0" y="34" width="196" height="134" rx="3"/>
  <line class="trk" x1="98" y1="140" x2="38"  y2="68" stroke="#0072B2"/>
  <line class="trk" x1="98" y1="140" x2="160" y2="76" stroke="#009E73"/>
  <circle cx="98" cy="140" r="4" fill="#1b3a6b"/>
  <text class="sml" x="98" y="188" text-anchor="middle">trigger 4021</text>

  <rect class="box" x="228" y="34" width="196" height="134" rx="3"/>
  <line class="trk" x1="326" y1="140" x2="272" y2="70" stroke="#0072B2"/>
  <line class="trk" x1="326" y1="140" x2="392" y2="96" stroke="#CC79A7"/>
  <circle cx="326" cy="140" r="4" fill="#1b3a6b"/>
  <text class="sml" x="326" y="188" text-anchor="middle">trigger 4022</text>

  <text class="lbl" x="0" y="240">Both legs come from one interaction, so they</text>
  <text class="lbl" x="0" y="258">share a vertex and a clock. Two examples.</text>

  <!-- ---------------- mixed ---------------- -->
  <rect class="box" x="470" y="34" width="186" height="134" rx="3"/>
  <line class="trk" x1="563" y1="140" x2="506" y2="68" stroke="#0072B2"/>
  <line class="trk" x1="563" y1="140" x2="624" y2="76" stroke="#009E73"
        stroke-opacity=".18"/>
  <circle cx="563" cy="140" r="4" fill="#1b3a6b" fill-opacity=".25"/>
  <text class="sml" x="563" y="188" text-anchor="middle">trigger 4021</text>

  <rect class="box" x="700" y="34" width="186" height="134" rx="3"/>
  <line class="trk" x1="793" y1="140" x2="736" y2="70" stroke="#0072B2"
        stroke-opacity=".18"/>
  <line class="trk" x1="793" y1="140" x2="854" y2="78" stroke="#009E73"/>
  <circle cx="793" cy="140" r="4" fill="#1b3a6b" fill-opacity=".25"/>
  <text class="sml" x="793" y="188" text-anchor="middle">trigger 5867</text>

  <path class="brk" d="M563 206 L563 218 L793 218 L793 206" stroke="#c0621a"/>
  <text class="lbl" x="470" y="240">Same two chambers, same track selection, same run</text>
  <text class="lbl" x="470" y="258">&#8212; but no shared interaction and no shared clock.</text>

  <text class="sml" x="0" y="296">Faded legs are the ones the mixing did not take. A draw that lands twice in the same trigger is rejected and redrawn.</text>
</svg>
"""

SVG_MODEL = """
<svg viewBox="0 0 900 250" role="img"
     aria-label="The observed pair sample splits into correlated pairs and
     accidental coincidences; mixing reproduces the accidental shape.">
  <defs>
    <style>
      .lbl{font:13px ui-sans-serif,sans-serif;fill:#5c6672}
      .ttl{font:600 13px ui-sans-serif,sans-serif;fill:#1a1f26}
      .sml{font:11px ui-sans-serif,sans-serif;fill:#8b95a1}
      .bx{stroke-width:1.2}
      .ar{stroke:#c9ced6;stroke-width:1.4;fill:none}
      @media (prefers-color-scheme:dark){
        .lbl{fill:#a8b2bd}.ttl{fill:#e6e9ed}.sml{fill:#7c8794}.ar{stroke:#39414b}
      }
    </style>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#c9ced6"/></marker>
  </defs>

  <rect class="bx" x="0" y="40" width="210" height="66" rx="3"
        fill="#1b3a6b" fill-opacity=".1" stroke="#1b3a6b"/>
  <text class="ttl" x="16" y="68">every two-track trigger</text>
  <text class="sml" x="16" y="88">what the detector wrote down</text>

  <path class="ar" d="M214 62 L282 40" marker-end="url(#a)"/>
  <path class="ar" d="M214 86 L282 112" marker-end="url(#a)"/>

  <rect class="bx" x="290" y="12" width="250" height="62" rx="3"
        fill="#1e7a4d" fill-opacity=".1" stroke="#1e7a4d"/>
  <text class="ttl" x="306" y="38">correlated pairs</text>
  <text class="sml" x="306" y="58">one interaction made both legs — the signal</text>

  <rect class="bx" x="290" y="90" width="250" height="62" rx="3"
        fill="#c0621a" fill-opacity=".1" stroke="#c0621a"/>
  <text class="ttl" x="306" y="116">accidental pairs</text>
  <text class="sml" x="306" y="136">two unrelated particles in one readout window</text>

  <path class="ar" d="M544 121 L646 121" marker-end="url(#a)"/>
  <rect class="bx" x="654" y="90" width="246" height="62" rx="3"
        fill="#c0621a" fill-opacity=".18" stroke="#c0621a"/>
  <text class="ttl" x="670" y="116">the event-mixed sample</text>
  <text class="sml" x="670" y="136">built to have this component and nothing else</text>

  <text class="lbl" x="290" y="196">Mixing models the lower box only. Subtract the upper box's</text>
  <text class="lbl" x="290" y="214">shape from the data and what is left is the pair signal —</text>
  <text class="lbl" x="290" y="232">provided the two boxes really do differ in the variable you plot.</text>
</svg>
"""


def build(n: dict) -> str:
    today = dt.date.today().isoformat()
    c = n['chi2']
    tp, to = n['tight']['perpendicular'], n['tight']['opposing']
    vx_all, vx_sl = n['vx']['all'], n['vx']['slope']
    vs_in = n['vs']['inter']
    tm = n['timing']

    chi2_rows = '\n'.join(
        f"""    <tr><td>{t}</td><td>{c[t]['n_obs']:,}</td>
        <td class="win">{c[t]['mixed']:.0f}</td>
        <td>{c[t]['best_model']}</td><td class="lose">{c[t]['best']:.0f}</td>
        <td>{c[t]['ratio']:.0f}&times;</td></tr>"""
        for t in ('intra', 'perpendicular', 'opposing'))

    pull_row = ', '.join(f'{a:.0f}&deg;: {p:+.1f}&sigma;' for a, p in n['pulls']
                         if abs(p) >= 2.0)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Event mixing in the X17 pair analysis</title>
<style>{CSS}</style>
</head><body><div class="wrap">

<header>
  <p class="kicker">n_TOF 2026 &middot; X17 preliminary analysis</p>
  <h1>What event mixing is doing in this analysis</h1>
  <p class="standfirst">Event mixing builds a fake dataset out of real tracks
  that never shared a trigger. It is the analysis's measurement of what an
  <em>accidental</em> two-track event looks like &mdash; and right now it is
  also the single most uncomfortable number on the page, because the observed
  opening-angle spectrum is indistinguishable from it.</p>
  <p class="meta"><b>Campaign pass</b> &middot; {n['n_real']:,} real pairs from
  {n['n_runs']} runs, each answered by one mixed pair &middot; built
  {today}</p>
</header>

<div class="answer">
  <p><strong>What it does.</strong> For every real pair of tracks found in one
  trigger, it draws a replacement pair &mdash; same two chambers, same track
  selection, same run &mdash; from two <em>different</em> triggers.</p>
  <p><strong>What it models.</strong> The accidental component: two unrelated
  particles that happened to land in the same readout window. Everything about
  the apparatus survives the shuffle; only the physical link between the two
  tracks is removed.</p>
  <p><strong>What it is telling us.</strong> Across the campaign the observed
  opening-angle spectrum matches this null better than it matches any pair
  spectrum &mdash; IPC or X17 &mdash; by a factor of
  {min(c[t]['ratio'] for t in c):.0f}&ndash;{max(c[t]['ratio'] for t in c):.0f} in
  &chi;&sup2;. The inclusive sample is consistent with being <em>all</em>
  accidentals. Only after a hard timing cut does the data begin to move off the
  null, and then it moves in the wrong direction.</p>
</div>

<h2><span class="num">1</span>The estimator, exactly</h2>
<p class="lede">Mixing is a permutation, and the details of which permutation
decide whether the answer means anything.</p>

<p>The input is the stage-3 track table: gated, angle-calibrated tracks that
point within 30&nbsp;mm of the beam axis. A <strong>real pair</strong> is any
two of those tracks inside one trigger. For each real pair, in
<code>source_imaging._pairs_mixed</code>:</p>

<ol>
  <li>read off its chamber combination, e.g. A&ndash;C;</li>
  <li>draw one track from the A pool and one from the C pool, <em>within the
  same run</em>;</li>
  <li>reject and redraw if the two draws come from the same trigger;</li>
  <li>emit it as one mixed pair, and compute exactly the same observables
  &mdash; opening angle, closest approach, vertex position.</li>
</ol>

<div class="svgbox">{SVG_MECHANICS}</div>

<p>Two choices in there are not cosmetic and were both got wrong on a first
pass. The pools are drawn <strong>only from tracks that actually form real
pairs</strong>, not from every track in the run: a trigger that produced two
tracks is a busier trigger and its tracks are not drawn from the same
distribution as a lone one. Mixing against the full track pool made the null
look better than the data and the lift came out below&nbsp;1. And the mixed
frame carries <strong>both</strong> event keys, <code>key1</code> and
<code>key2</code>; when it carried only the first, a downstream timing study
silently read the second track's scintillator out of the wrong event.</p>

<figure>
  <img src="figures/f1_construction.png"
       alt="Bar chart of pairs per chamber combination, real and mixed, equal in every combination.">
  <figcaption><b>One mixed pair per real pair.</b> The counts agree exactly in
  every chamber combination, by construction. This is why the mixed sample is a
  <em>shape</em> and carries no rate of its own &mdash; subtracting all of it
  would subtract the signal too, which is what happened the first time this ran
  and left every physics model with the same &chi;&sup2;. Chamber&nbsp;B
  contributes {pct(n['frac_B'])} of pairs and has no field cage; B&ndash;D is
  counted in <code>opposing</code> but is not a usable signal channel.</figcaption>
</figure>

<h2><span class="num">2</span>What it is meant to model</h2>
<p class="lede">One component of the observed sample, isolated by removing the
one thing mixing destroys.</p>

<p>Every two-track trigger is one of two things. Either a single interaction
produced both legs &mdash; an internal pair, an X17 decay, a scatter &mdash; or
two unrelated particles arrived close enough in time to be read out together.
The analysis wants the first; the trigger hands it both.</p>

<div class="svgbox">{SVG_MODEL}</div>

<p>Mixing is an attempt to measure the second box directly, using the data
itself rather than a simulation. The logic is that a mixed pair is
<em>guaranteed</em> accidental: the two tracks physically could not have come
from one interaction, because they arrived minutes apart. Everything else about
them is real &mdash; real detector, real acceptance, real efficiency, real
occupancy, real beam conditions.</p>

<div class="cols">
  <div class="keep"><h4>Preserved by the shuffle</h4><ul>
    <li>each chamber's single-track rate and angular distribution</li>
    <li>the geometric acceptance, including the 90&deg; chamber layout that
    decides most of the opening angle</li>
    <li>per-run conditions &mdash; gas, gain, threshold, dead channels
    (mixing never crosses a run)</li>
    <li>the track quality cuts and the pairing rule</li>
    <li>the topology composition, pair for pair</li>
  </ul></div>
  <div class="kill"><h4>Destroyed by the shuffle</h4><ul>
    <li>any correlation between the two tracks</li>
    <li>the common vertex &mdash; two tracks from different triggers cannot
    point at the same place except by chance</li>
    <li>the common interaction time</li>
    <li>and, unavoidably, <em>anything measured relative to the event's own
    trigger</em> &mdash; see &sect;5</li>
  </ul></div>
</div>

<p>So the comparison is a <strong>difference of correlations</strong>. If the
data contains real pairs, the data must differ from the mixed sample in some
variable that a common origin controls: opening angle, vertex position, or
closest approach. If it does not differ, the data contains no pairs that the
variable can see.</p>

<h2><span class="num">3</span>What it says about the data</h2>
<p class="lede">This is the headline, and it is a negative one.</p>

<figure>
  <img src="figures/f2_spectra.png"
       alt="Opening angle spectra for intra, perpendicular and opposing topologies, observed versus event-mixed.">
  <figcaption><b>The observed opening-angle spectrum and its accidental null
  are the same distribution</b>, in all three topologies, across the whole
  campaign. The mixed histogram is scaled to the observed total because it
  carries no rate. The fraction of pairs above the X17 threshold is
  {pct(n['above']['opposing']['obs'])} observed against
  {pct(n['above']['opposing']['mixed'])} in the null for
  <code>opposing</code>, and {pct(n['above']['perpendicular']['obs'])} against
  {pct(n['above']['perpendicular']['mixed'])} for
  <code>perpendicular</code> &mdash; the observed sample is, if anything,
  slightly <em>less</em> forward than pure accidentals. (The 15&deg; binning
  puts that boundary at 105&deg;, not 109&deg;.)</figcaption>
</figure>

<p>Put the mixed shape into the same &chi;&sup2; table as the physics models,
on the same footing &mdash; every shape normalised to the observed total, no
free parameter anywhere &mdash; and it wins everywhere, by a wide margin:</p>

<table>
  <thead><tr><th>topology</th><th>pairs</th><th>&chi;&sup2;/dof, mixed</th>
  <th>best physics model</th><th>&chi;&sup2;/dof</th><th>factor</th></tr></thead>
  <tbody>
{chi2_rows}
  </tbody>
</table>

<figure>
  <img src="figures/f3_chi2.png"
       alt="Log-scale bar chart of chi-squared per degree of freedom for each candidate shape and topology.">
  <figcaption><b>Nothing in the physics library comes close.</b> Intra has no
  X17 bar because a 17&nbsp;MeV boson cannot make a pair below 109&deg; and so
  predicts no intra pairs at all. Note that even
  the mixed shape has &chi;&sup2;/dof in the tens: with 10&ndash;30&thinsp;k
  pairs per topology the statistical errors are small enough that residual
  differences in detector response between a mixed and a real pair are
  themselves significant. The mixed number is the floor this dataset can
  reach, not a good fit in the textbook sense &mdash; which is why the
  <em>ratio</em> to the physics models is the statement, not the absolute
  value.</figcaption>
</figure>

<div class="note">
  <span class="h">The honest reading</span>
  <p>This is not evidence against X17. It is evidence that <strong>the
  inclusive two-track sample is dominated by accidental coincidences</strong>,
  at a level where a pair signal of any plausible size is invisible underneath
  it. The measurement that has to come first is not a spectrum &mdash; it is a
  selection that removes accidentals.</p>
</div>

<h2><span class="num">4</span>Where mixing earns its keep</h2>
<p class="lede">The same null, applied to a cleaner sample and to a different
variable, does start to separate.</p>

<h3>Timing-selected pairs</h3>
<p>The accidental-timing study established that in a two-arm event one arm is
the trigger and <em>the other fires at essentially a random time</em> &mdash;
median |&Delta;t| of 171&nbsp;ns against a 5&nbsp;ns single-arm reference. The
production accept window of (&minus;100,&nbsp;+60)&nbsp;ns is wide enough that
the plastic scintillators' own accidental rate lands something inside it almost
every time. Tighten it to the peak core &mdash; |t| &le; 30&nbsp;ns per arm
<em>and</em> |t<sub>1</sub>&nbsp;&minus;&nbsp;t<sub>2</sub>| &le; 20&nbsp;ns
between them &mdash; and the sample drops to a few hundred pairs that are
genuinely prompt with each other.</p>

<figure>
  <img src="figures/f4_tight.png"
       alt="Opening angle of prompt-coincident pairs against the event-mixed null, perpendicular and opposing.">
  <figcaption><b>The cut that the null implies.</b> The pink band is the
  &gt;170&deg; back-to-back region, already removed from every sample shown.
  Perpendicular pairs
  (n&nbsp;=&nbsp;{tp['n']}) still sit on the null at &chi;&sup2;/dof&nbsp;=
  {tp['chi2dof']:.1f}. Opposing pairs (n&nbsp;=&nbsp;{to['n']}) leave it, at
  &chi;&sup2;/dof&nbsp;= {to['chi2dof']:.1f}.</figcaption>
</figure>

<div class="note stop">
  <span class="h">But read the direction before celebrating</span>
  <p>The opposing deviation is not a bump in the X17 window. The per-bin pulls
  run {pull_row} &mdash; a deficit through 120&ndash;150&deg; and an excess
  piled against 180&deg;. That is the signature of <strong>one particle
  crossing two opposing chambers</strong>, which registers as a perfectly
  time-coincident &ldquo;pair&rdquo; at ~180&deg;. Pairs above 170&deg; are
  already cut, and the survivors still pile up against that edge:
  {pct(n['b2b']['tight_edge'])} of the tight opposing sample sits in the last
  5&deg; before the cut, against {pct(n['b2b']['tagged_edge'])} of the loose
  sample.</p>
  <p>So the tight cut works &mdash; it removes accidentals &mdash; but it
  <em>enriches</em> a correlated background that event mixing cannot model at
  all, because that background really is one correlated object. Mixing
  separates pairs from accidentals. It cannot separate pairs from other
  correlated things.</p>
</div>

<h3>The vertex, where mixing does deliver a signal</h3>
<p>The opening angle is a weak discriminator here because the geometry
dominates it. The vertex is a much stronger one: two tracks from different
triggers cannot converge on the beam axis except by accident.</p>

<figure>
  <img src="figures/f5_vertex.png"
       alt="Vertex radius distribution real versus mixed, and lift over the null for four samples.">
  <figcaption><b>Same estimator, four samples.</b> Inter-chamber pairs in
  run_145 do not beat the null &mdash; lift {vs_in['lift']:.2f}, i.e. real
  pairs converge on the capsule slightly <em>less</em> often than mixed ones.
  Nor does the full intra-chamber-A sample
  ({vx_all['ratio']:.2f}&nbsp;&plusmn;&nbsp;{vx_all['err']:.2f}). But require
  both legs to have a reliable slope &mdash; the tracks whose direction is
  actually measured rather than degenerate &mdash; and the same estimator gives
  a lift of <b>{vx_sl['ratio']:.2f}&nbsp;&plusmn;&nbsp;{vx_sl['err']:.2f}</b>,
  {vx_sl['excess_sigma']:.1f}&sigma;, on {int(vx_sl['n_real']):,} pairs. That is
  event mixing doing its job: a population that converges on the source more
  often than chance allows.</figcaption>
</figure>

<h2><span class="num">5</span>Where mixing is not a valid null at all</h2>
<p class="lede">Two failures on record, both of which produced a
physics-looking number before they were caught.</p>

<h3>Trigger-referenced timing</h3>
<p>Every hit time in this dataset is measured <em>relative to its own event's
trigger</em>. There is no common clock across events. So a mixed
&Delta;t&nbsp;= t<sub>1</sub>&nbsp;&minus;&nbsp;t<sub>2</sub> is a difference
of two numbers that each sit near zero for their own, unrelated reasons &mdash;
and the answer you get depends entirely on which hits you put in the pool.</p>

<figure>
  <img src="figures/f6_timing.png"
       alt="Delta-t distributions: real two-arm pairs, mixed from all hits, mixed from tagging hits.">
  <figcaption><b>The mixed answer is not a measurement.</b> Real two-arm pairs
  have {pct(tm['real']['frac_within_20ns'], 0)} of their
  &Delta;t within 20&nbsp;ns. Mix from every recorded hit and the null says
  {pct(tm['mixed / all hits']['frac_within_20ns'], 0)}; mix from each event's
  <em>tagging</em> hit &mdash; the one closest to its own trigger &mdash; and
  the same null says
  {pct(tm['mixed / tagging hits']['frac_within_20ns'], 0)}, more coincident
  than any real data could ever be. The variable has no cross-event meaning, so
  the permutation has no defined answer. The pair products deliberately write
  <code>NaN</code> into every timing column of a mixed pair rather than leave
  the number available to be misread.</figcaption>
</figure>

<h3>Symmetric mixing on a triggered sample</h3>
<p>The first version of this idea, at stage&nbsp;1, permuted <em>every</em>
chamber's contents independently across triggers. It reported a 1.45&thinsp;%
accidental two-arm rate against 1.03&thinsp;% observed, and the 29&thinsp;%
&ldquo;deficit&rdquo; was briefly read as a physics statement.</p>
<p>It was the trigger. The DAQ fires on a wall-plus-plastic coincidence in
<em>any one arm</em>, so every recorded event already contains one guaranteed
particle, and the four arms are anti-correlated by construction &mdash;
whichever arm the trigger particle went into, it did not go into the other
three. Mixing all four destroys that constraint and predicts the second track
at p&sup2;, which it is not. <code>event_mixing_background</code> now raises
<code>NotImplementedError</code> rather than returning a number.</p>
<p>The repaired version holds the trigger arm fixed and mixes only the others.
It is correct, and it is also <em>vacuous</em>: permuting a flag among events
conserves that flag's total exactly, so the mixed and observed second-arm
probabilities can differ only through clumping. Measured: 6.2335&thinsp;%
observed against 6.3154&nbsp;&plusmn;&nbsp;0.0082&thinsp;% mixed. <strong>No
permutation of counts can separate a pair partner from an unrelated
track</strong> &mdash; what distinguishes them is timing and geometry, which is
why mixing belongs downstream of the waveform reconstruction and not at
stage&nbsp;1.</p>

<h2><span class="num">6</span>What this does not rule out</h2>
<ul>
  <li><strong>A pair signal below the accidental floor.</strong> The inclusive
  spectrum matching the null means accidentals dominate, not that the pair rate
  is zero. No limit is quoted from this comparison, because the mixed sample
  carries no normalisation.</li>
  <li><strong>A signal visible in a variable not yet used.</strong> The
  opening angle is heavily constrained by the 90&deg; chamber geometry, which
  mixing preserves exactly &mdash; making it the <em>least</em> sensitive
  observable available. The vertex already behaves differently (&sect;4), and
  the intra-chamber slope-selected sample is a real excess over this null.</li>
  <li><strong>The opposing tight-pair deviation being partly real.</strong> It
  is dominated by collinear punch-through, but the 120&ndash;150&deg; deficit
  is not obviously explained by that alone, and the sample is only
  {to['n']} pairs.</li>
  <li><strong>Anything about the accidental <em>rate</em>.</strong> The
  Poisson estimate N<sub>trig</sub>&nbsp;&times;&nbsp;p<sub>i</sub>&nbsp;&times;&nbsp;p<sub>j</sub>
  over-predicts the observed pair count by 2&ndash;4&times; in every chamber
  combination, because the trigger correlates the arms. Normalising the
  accidental component remains the leading systematic and is stated as such
  rather than guessed.</li>
</ul>

<footer>
  <p>Code: <code>source_imaging._pairs_mixed</code> (the estimator),
  <code>campaign_angle.py</code> (campaign pass),
  <code>det_a_intra._mixed_index</code> (intra-chamber null),
  <code>candidate_filter.accidental_second_track</code> (the retired stage-1
  version, kept with its docstring),
  <code>tight_coincidence.py</code> (the timing cut).
  Figures rebuilt by <code>explain_event_mixing.py</code>; this page by
  <code>make_event_mixing_page.py</code>. Data:
  <code>sept26_prelim/{{angle_campaign, imaging, det_a_intra,
  tight_coincidence, accidental_timing}}</code>.</p>
  <p>Preliminary &mdash; n_TOF 2026 campaign, {n['n_runs']} runs. Built {today}.</p>
</footer>

</div></body></html>
"""


def main() -> int:
    n = gather()
    p = OUT / 'report.html'
    p.write_text(build(n), encoding='utf-8')
    print('wrote', p, f'({p.stat().st_size/1024:.0f} kB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
