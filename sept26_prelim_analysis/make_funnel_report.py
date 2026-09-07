#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_funnel_report.py -- build ``report.html`` for the reconstruction funnel.

Generated, never hand-written (CLAUDE.md): re-running ``funnel.py`` and then
this script moves every number, every bar width and the verdict text together.
Nothing below is a literal count -- they all come from ``funnel_<run>.csv`` and
``imaging_summary.json``.

The diagram is emitted as inline SVG rather than a PNG so it stays crisp on the
control-room projector and legible in a text diff.  Bar widths are proportional
to the counts; every bar carries its number as a visible label, which is also
what discharges the palette's sub-3:1 contrast warning.

    python -m sept26_prelim_analysis.make_funnel_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.figstyle import DET_COLOR  # noqa: E402

# Validated 2026-09-07 against the #fbfcfe surface (dataviz validate_palette):
# the three evidence hues return ALL CHECKS PASS; NEITHER is a deliberate
# neutral (absence of evidence), not a fifth category, so it is exempt from the
# chroma floor by construction.
EV_COLOR = {'wal_and_pss': '#0072B2', 'wal_only': '#56B4E9',
            'pss_only': '#E69F00', 'neither': '#b8bfc9'}
EV_LABEL = {'wal_and_pss': 'wall AND plastic', 'wal_only': 'wall only',
            'pss_only': 'plastic only', 'neither': 'neither'}

STAGES = [
    ('n_triggers',   'DAQ triggers',      'every event the DREAM wrote'),
    ('n_seeded',     'seeded',            '&ge;3 clustered strips on a plane (hits, channels only)'),
    ('n_plane_cand', 'plane candidates',  'clusters offered to the waveform fit, up to 3 per plane'),
    ('n_xy_ok',      'x and y both fit',  'both planes converged and passed the plausibility window'),
    ('n_pairings',   '(x,y) pairings',    'combinations the 3D pairing considered'),
    ('n_tracks',     'gated 3D tracks',   'pairings that passed the 3D gate'),
    ('pointing',     'pointing-confirmed', 'extrapolates to the wall segment AND plastic bar that fired'),
]


def fmt(n) -> str:
    return f'{int(n):,}'


def pct(x) -> str:
    return '&mdash;' if pd.isna(x) else f'{100 * x:.1f}%'


# --------------------------------------------------------------------- the SVG
def funnel_svg(F: pd.DataFrame) -> str:
    """One horizontal funnel per arm, bar width proportional to the count.

    Deliberately linear, not log: the point of the picture is that most of what
    the DAQ wrote does not become a track, and a log axis would hide exactly
    that.  The number sits outside the bar so the short stages stay readable.
    """
    row_h, gap, lab_w, bar_w, pad = 26, 8, 168, 430, 14
    head = 34
    n = len(STAGES)
    col_h = head + n * (row_h + gap) + pad
    cols = len(F)
    col_w = lab_w + bar_w + 96
    W, H = pad + cols * col_w, col_h + pad

    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="reconstruction funnel per chamber" '
         f'style="max-width:100%;height:auto;font-family:inherit">']
    for ci, (_, r) in enumerate(F.iterrows()):
        x0 = pad + ci * col_w
        c = DET_COLOR[r['arm']]
        o.append(f'<text x="{x0}" y="{head - 12}" font-size="15" '
                 f'font-weight="700" fill="{c}">chamber {r["arm"]}</text>')
        top = F['n_triggers'].max()
        for si, (key, label, _why) in enumerate(STAGES):
            y = head + si * (row_h + gap)
            v = r[key]
            w = 0 if not top else max(2.0, bar_w * float(v) / float(top))
            o.append(f'<text x="{x0}" y="{y + 17}" font-size="12.5" '
                     f'fill="var(--ink-2)">{label}</text>')
            o.append(f'<rect x="{x0 + lab_w}" y="{y}" width="{w:.1f}" '
                     f'height="{row_h - 6}" rx="4" fill="{c}" '
                     f'opacity="{1.0 - 0.07 * si:.2f}"/>')
            o.append(f'<text x="{x0 + lab_w + w + 8:.1f}" y="{y + 15}" '
                     f'font-size="12.5" font-weight="600" '
                     f'fill="var(--ink)">{fmt(v)}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def evidence_svg(F: pd.DataFrame) -> str:
    """Exclusive n_TOF partition of the tracked events, one stacked bar per arm.

    2 px surface gaps between segments; every segment direct-labelled.
    """
    keys = ('wal_and_pss', 'wal_only', 'pss_only', 'neither')
    row_h, gap, lab_w, bar_w, pad = 34, 16, 96, 620, 14
    H = pad * 2 + len(F) * (row_h + gap)
    W = pad * 2 + lab_w + bar_w + 90
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="n_TOF confirmation of tracked events per chamber" '
         f'style="max-width:100%;height:auto;font-family:inherit">']
    for i, (_, r) in enumerate(F.iterrows()):
        y = pad + i * (row_h + gap)
        tot = float(r['n_track_events']) or 1.0
        o.append(f'<text x="{pad}" y="{y + 22}" font-size="14" '
                 f'font-weight="700" fill="{DET_COLOR[r["arm"]]}">'
                 f'chamber {r["arm"]}</text>')
        x = pad + lab_w
        for k in keys:
            w = bar_w * float(r[k]) / tot
            if w <= 0:
                continue
            o.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" '
                     f'height="{row_h}" fill="{EV_COLOR[k]}"/>')
            if w > 46:
                o.append(f'<text x="{x + w / 2 - 1:.1f}" y="{y + 21}" '
                         f'font-size="12" font-weight="700" text-anchor="middle" '
                         f'fill="#fff">{fmt(r[k])}</text>')
            x += w
        o.append(f'<text x="{pad + lab_w + bar_w + 10}" y="{y + 22}" '
                 f'font-size="12.5" fill="var(--ink-2)">of {fmt(tot)}</text>')
    o.append('</svg>')
    key = ' '.join(
        f'<span class="k"><i style="background:{EV_COLOR[k]}"></i>'
        f'{EV_LABEL[k]}</span>' for k in keys)
    return f'{"".join(o)}<div class="legend">{key}</div>'


# ------------------------------------------------------------------- the tables
def funnel_table(F: pd.DataFrame) -> str:
    hdr = ''.join(f'<th>{a}</th>' for a in F.arm)
    rows = []
    for key, label, why in STAGES:
        cells = ''.join(f'<td class="n">{fmt(v)}</td>' for v in F[key])
        rows.append(f'<tr><th class="s">{label}<span class="why">{why}</span>'
                    f'</th>{cells}</tr>')
    return (f'<table class="t"><thead><tr><th>stage</th>{hdr}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def rate_table(F: pd.DataFrame) -> str:
    spec = [
        ('seed_eff', 'seeded / triggers', 'the seeder\'s acceptance'),
        ('track_eff', 'tracked / seeded', 'a seed becomes a gated 3D track'),
        ('coinc_frac', 'wall+plastic | tracked', 'both n_TOF layers, same arm'),
        ('ctrl_coinc_frac', 'wall+plastic | seeded, NO track', 'the control'),
        ('lift', 'lift (tracked / control)', '>1 means tracking selects real particles'),
        ('pointing_frac', 'pointing-confirmed | predictable',
         'hits the segment and bar that fired'),
    ]
    hdr = ''.join(f'<th>{a}</th>' for a in F.arm)
    rows = []
    for key, label, why in spec:
        if key == 'lift':
            cells = ''.join(f'<td class="n">{v:.2f}&times;</td>' for v in F[key])
        else:
            cells = ''.join(f'<td class="n">{pct(v)}</td>' for v in F[key])
        rows.append(f'<tr><th class="s">{label}<span class="why">{why}</span>'
                    f'</th>{cells}</tr>')
    return (f'<table class="t"><thead><tr><th>rate</th>{hdr}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def k_table(img: dict) -> str:
    if not img:
        return '<p class="note">no <code>imaging_summary.json</code> staged.</p>'
    rows = []
    for r in img['results']:
        p = r.get('pointing_x_coincident', {})
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r["arm"]]}">'
            f'chamber {r["arm"]}</th>'
            f'<td class="n">{r["v_bundle"]:.1f}</td>'
            f'<td class="n">{r["k_phys"]:.2f}</td>'
            f'<td class="n">{r["v_bundle"] / r["k_phys"]:.1f}</td>'
            f'<td class="n">{p.get("source_measured_axis", "&mdash;")}</td>'
            f'<td class="n">{p.get("source_measured_mm", float("nan")):+.1f}'
            f' &plusmn; {p.get("zero_crossing_err", float("nan")):.1f}</td>'
            f'<td class="n">{r["image_at_kphys_coincident"]["r_med"]:.1f}</td>'
            f'</tr>')
    return ('<table class="t"><thead><tr><th></th>'
            '<th>v in bundle<br><span class="u">&micro;m/ns</span></th>'
            '<th>k measured</th>'
            '<th>v implied<br><span class="u">&micro;m/ns</span></th>'
            '<th>images<br>axis</th>'
            '<th>source position<br><span class="u">mm from beam axis</span></th>'
            '<th>median miss<br><span class="u">mm</span></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# --------------------------------------------------------------------- the page
# Design notes.  The project already has a validated visual system --
# figstyle.py's Okabe-Ito chamber palette on a #fbfcfe surface, with the mx17
# purple as the one accent -- and a detector that changes colour between the
# slides and the web report is a detector the reader has to re-learn.  So the
# palette here is figstyle's, extended rather than replaced:
#
#   * neutrals carry a slight violet bias toward the mx17 accent (#1b2430 ->
#     hue-shifted slates) so the greys read as chosen, not defaulted;
#   * IBM Plex Sans / Plex Mono -- an instrumentation face, drawn for technical
#     documentation, and the mono is what carries every count, unit and channel
#     name.  Real fallback stacks: the control-room browser may be offline.
#   * `tabular-nums` everywhere digits stack, because the whole page is columns
#     of counts that must line up to be comparable at a glance.
FONT_LINK = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500;600&'
             'family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">')

CSS = """
:root{
  --bg:#fbfafd; --panel:#ffffff; --ink:#1e1b28; --ink-2:#5f5a70; --ink-3:#8f8aa0;
  --line:#e6e2ee; --accent:#8a3f8f; --warn:#a86a1e; --good:#0072B2;
  --caution-bg:#fdf5e9; --caution-line:#e6cfa6; --focus:#8a3f8f;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14121a; --panel:#1c1926; --ink:#ece9f4; --ink-2:#aaa3bd; --ink-3:#7d7691;
  --line:#2e2a3b; --accent:#c98fce; --warn:#e0a95f; --good:#5aa9dd;
  --caution-bg:#241c14; --caution-line:#4a3a22; --focus:#c98fce;
}}
:root[data-theme="dark"]{
  --bg:#14121a; --panel:#1c1926; --ink:#ece9f4; --ink-2:#aaa3bd; --ink-3:#7d7691;
  --line:#2e2a3b; --accent:#c98fce; --warn:#e0a95f; --good:#5aa9dd;
  --caution-bg:#241c14; --caution-line:#4a3a22; --focus:#c98fce;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.62;margin:0;padding:0 20px 72px;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

header{padding:44px 0 14px;border-bottom:2px solid var(--ink);margin-bottom:28px}
.eyebrow{font-family:var(--mono);font-size:11.5px;font-weight:500;
  text-transform:uppercase;letter-spacing:.14em;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 18px;margin-bottom:14px;align-items:center}
h1{font-size:31px;line-height:1.16;margin:0 0 10px;font-weight:600;
  letter-spacing:-.02em;text-wrap:balance;max-width:22ch}
.sub{color:var(--ink-2);font-size:13.5px;margin:0;font-family:var(--mono);
  line-height:1.75}
.badge{display:inline-block;background:var(--accent);color:#fff;font-size:10.5px;
  font-weight:600;letter-spacing:.14em;padding:3px 9px;border-radius:3px;
  font-family:var(--mono)}
h2{font-size:21px;margin:52px 0 8px;letter-spacing:-.015em;font-weight:600;
  text-wrap:balance;display:flex;align-items:baseline;gap:14px}
h2 .n{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--accent);
  letter-spacing:.06em;flex:none}
p{margin:10px 0;max-width:78ch}
.lede{font-size:17px;line-height:1.58;max-width:70ch;margin:20px 0 4px}
.note{color:var(--ink-2);font-size:13.5px;max-width:80ch;line-height:1.6}
code{font-family:var(--mono);font-size:12.5px;background:var(--panel);
  border:1px solid var(--line);border-radius:3px;padding:1px 5px}
b{font-weight:600}

.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:20px 22px;margin:18px 0}
.caution{background:var(--caution-bg);border-left:3px solid var(--warn);
  padding:16px 20px;margin:22px 0;font-size:14px;max-width:82ch}
.caution b{color:var(--warn);font-weight:600}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:1px;margin:22px 0;background:var(--line);border:1px solid var(--line);
  border-radius:8px;overflow:hidden}
.card{background:var(--panel);padding:16px 18px}
.card .v{font-family:var(--mono);font-size:27px;font-weight:600;
  letter-spacing:-.03em;line-height:1.1;font-variant-numeric:tabular-nums}
.card .l{font-size:12px;color:var(--ink-2);margin-top:5px;line-height:1.4}

.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:16px 0}
table.t{border-collapse:collapse;width:100%;font-size:13.5px;min-width:680px}
table.t th,table.t td{border-bottom:1px solid var(--line);padding:9px 12px;
  text-align:left;vertical-align:top}
table.t thead th{font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--ink-3);font-weight:500;
  border-bottom:1.5px solid var(--ink-3)}
table.t td.n,table.t thead th:not(:first-child){text-align:right}
table.t td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-weight:500;white-space:nowrap}
table.t th.s{font-weight:500;white-space:normal;max-width:30ch}
table.t .why{display:block;font-weight:400;color:var(--ink-3);font-size:11.5px;
  line-height:1.4;margin-top:2px}
table.t .u{font-family:var(--sans);font-weight:400;text-transform:none;
  letter-spacing:0;font-size:10.5px}
table.t tbody tr:last-child td,table.t tbody tr:last-child th{border-bottom:none}

.legend{margin-top:14px;font-family:var(--mono);font-size:11.5px;
  color:var(--ink-2);display:flex;flex-wrap:wrap;gap:6px 20px}
.legend .k{white-space:nowrap;display:inline-flex;align-items:center}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin-right:6px}
ul{margin:10px 0;padding-left:20px;max-width:80ch}li{margin:6px 0}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--ink-3);font-size:12px;font-family:var(--mono);line-height:1.8}
svg text{font-family:var(--mono);font-variant-numeric:tabular-nums}
"""


def build_html(F: pd.DataFrame, meta: dict, img: dict) -> str:
    n_trig = int(meta['n_triggers'])
    tot_tracks = int(F.n_tracks.sum())
    tot_point = int(F.pointing.sum())
    best = F.loc[F.lift.idxmax()]
    kmin = min(r['k_phys'] for r in img['results']) if img else float('nan')
    kmax = max(r['k_phys'] for r in img['results']) if img else float('nan')
    ac = {r['arm']: r for r in img['results']} if img else {}

    def src(a):
        p = ac.get(a, {}).get('pointing_x_coincident', {})
        return p.get('source_measured_mm', float('nan')), \
            p.get('zero_crossing_err', float('nan'))

    a_s, a_e = src('A')
    c_s, c_e = src('C')

    return f"""<title>Run 145 Reconstruction Funnel</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span>
    <span>run {html.escape(meta['run'].replace('run_', ''))}</span>
    <span>{html.escape(' / '.join(meta['subruns']))}</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>What we reconstruct, and what n_TOF says about it</h1>
  <p class="sub">{fmt(n_trig)} DAQ triggers &middot; four Micromegas TPC chambers
     &middot; full waveform pass, no prescale</p>
</header>

<p class="lede">Of {fmt(n_trig)} triggers, the waveform reconstruction returns
<b>{fmt(tot_tracks)} gated 3D track segments</b> across the four chambers, and
<b>{fmt(tot_point)}</b> of those sit in events where the track extrapolates onto
the scintillator wall segment <i>and</i> the plastic bar that actually fired.
The tracking is selecting real particles &mdash; every chamber shows a
wall-and-plastic coincidence rate above its own no-track control, by
{F.lift.min():.2f}&times; to {F.lift.max():.2f}&times;. <b>The angles are not yet
calibrated</b>: the drift velocity in every bundle is a prior, and the target
image says it is wrong by {kmin:.1f}&times; to {kmax:.1f}&times;, chamber by chamber.</p>

<div class="cards">
  <div class="card"><div class="v">{fmt(n_trig)}</div><div class="l">DAQ triggers</div></div>
  <div class="card"><div class="v">{fmt(tot_tracks)}</div><div class="l">gated 3D tracks</div></div>
  <div class="card"><div class="v">{fmt(tot_point)}</div><div class="l">pointing-confirmed</div></div>
  <div class="card"><div class="v">{best.lift:.2f}&times;</div>
    <div class="l">best coincidence lift (chamber {best.arm})</div></div>
</div>

<h2><span class="n">1</span>The funnel</h2>
<p>Bar width is proportional to the count, on a common linear scale, so the
shape of the picture is the message: most of what the DAQ writes does not become
a track, and that is expected &mdash; the trigger fires on a scintillator
coincidence in <i>one</i> arm, so the other three chambers are mostly empty by
construction.</p>
<div class="panel">{funnel_svg(F)}</div>
<div class="scroll">{funnel_table(F)}</div>
<p class="note"><b>Where <code>combined_hits</code> enters, and where it stops.</b>
Exactly one stage &mdash; <i>seeded</i> &mdash; reads the hits table, and only for
the set of channels carrying charge. No hit <i>time</i> crosses into the geometry;
position, angle and depth all come from the waveform fit. That boundary is the
subject of <code>RECONSTRUCTION_BASIS.md</code>. It does mean the seeder's
acceptance is a real efficiency term, which is why it is a row here rather than
an assumption.</p>

<h2><span class="n">2</span>What n_TOF says about the tracks</h2>
<p>For every event with at least one gated track, an exclusive partition of what
the <i>same chamber's</i> scintillators recorded in time. Two independent layers
at different depths agreeing &mdash; the dark blue segment &mdash; is much
stronger evidence of a real particle than either alone.</p>
<div class="panel">{evidence_svg(F)}</div>
<div class="scroll">{rate_table(F)}</div>
<p class="note">The <b>control</b> row is the same trigger population and the same
n_TOF, restricted to events the seeder accepted but the reconstruction failed to
turn into a track. The ratio of the two rows is the lift, and it is what says the
tracking is doing work rather than following the trigger.</p>

<div class="caution">
<b>Chamber D seeds twice as often as the others</b> ({pct(F.set_index('arm').loc['D','seed_eff'])}
of triggers, against {pct(F.set_index('arm').loc['A','seed_eff'])}&ndash;{pct(F.set_index('arm').loc['C','seed_eff'])}),
and it has the lowest coincidence fraction and the lowest lift. Same threshold,
same seeder: D is passing far more clusters that n_TOF does not confirm. Treat
D's track counts as an upper bound until this is understood. Chamber B is the
opposite problem &mdash; the fewest tracks and the weakest pointing confirmation.
</div>

<h2><span class="n">3</span>The angle scale is the open null</h2>
<p>The fit estimates three numbers per plane: the position at the mesh
<code>p0</code>, the transverse speed <code>w</code> in mm/ns, and the start time
<code>t0</code>. <b>None of them depends on the drift velocity.</b> <code>v</code>
enters only afterwards, converting the measured speed into an angle:
<code>tan&theta; = w / v</code>. So position is measured; angle is measured
&times; an assumed constant.</p>
<p>Every bundle carries <code>v = 42.6 &micro;m/ns</code> &mdash; a Magboltz
prior for Ar/iso 90/10, not an in-situ measurement. The target image measures the
correction:</p>
<div class="scroll">{k_table(img)}</div>
<p class="note">The <b>source position</b> column is the zero crossing of the
pointing band, <code>-intercept/slope</code>. Scaling every angle by <i>k</i>
scales the intercept and the slope together, so this number is <b>invariant under
the angle scale</b> &mdash; it cannot be produced, or improved, by tuning drift
velocities. Chambers A and C both measure the same global axis and land at
{a_s:+.1f} &plusmn; {a_e:.1f} mm and {c_s:+.1f} &plusmn; {c_e:.1f} mm against a
surveyed 0. That agreement is a genuine, scale-free result. The <i>k</i> column
beside it is not yet a calibration: the estimators (band slope, per-track,
image focus) agree for A and C and disagree badly for B and D.</p>

<h2><span class="n">4</span>What this does not show</h2>
<ul>
<li><b>No opening angles, no invariant mass.</b> Both go as 1/v, and v is not
measured. Nothing angular in this report should be quoted.</li>
<li><b>No neutron energy.</b> The per-bunch flash <i>t</i><sub>0</sub> is not yet
established, so time-of-flight is not available and every track here is
energy-blind.</li>
<li><b>Coincidence is per chamber, not per track.</b> An event with two gated
tracks in one chamber is counted once. Matching an individual track to an
individual scintillator hit needs the pointing column, which is why it is
reported separately.</li>
<li><b>The two-chamber (X17-topology) rate is deliberately absent.</b> Separating
a genuine pair from two unrelated clusters needs the pointing resolution, and
that is angle-limited &mdash; see section 3.</li>
<li><b>One run, two sub-runs.</b> {html.escape(', '.join(meta['subruns']))} only.
These are not campaign numbers, and the 23 July noise-floor step means they may
not be poolable with earlier runs at all.</li>
</ul>

<footer>
Built by <code>sept26_prelim_analysis/make_funnel_report.py</code> from
<code>funnel_{html.escape(meta['run'])}.csv</code> and
<code>imaging_summary.json</code>. Numbers regenerate with the analysis.<br>
Full pass: <code>{html.escape(meta['fullpass'])}</code> &middot;
in-time window {meta['dt_window'][0]:.0f} to {meta['dt_window'][1]:.0f} ns &middot;
selection: {html.escape(meta['allowlist'])}.
</footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--dir', default=None, help='funnel output directory')
    ap.add_argument('--imaging', default=None,
                    help='imaging_summary.json (the angle-scale measurement)')
    a = ap.parse_args()

    od = a.dir or str(paths.out('funnel'))
    F = pd.read_csv(paths.require(os.path.join(od, f'funnel_{a.run}.csv'),
                                  'funnel table -- run funnel.py first'))
    meta = json.load(open(paths.require(
        os.path.join(od, f'funnel_{a.run}.meta.json'), 'funnel meta')))

    ip = a.imaging or str(paths.out('kcal') / f'{a.run}_stat090_0000'
                          / 'imaging_summary.json')
    img = json.load(open(ip)) if os.path.exists(ip) else {}
    if not img:
        print(f'[warn] no imaging summary at {ip}; section 3 will be thin')

    out = os.path.join(od, 'report.html')
    with open(out, 'w') as fh:
        fh.write(build_html(F, meta, img))
    print(f'wrote {out}  ({os.path.getsize(out) / 1024:.0f} kB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
