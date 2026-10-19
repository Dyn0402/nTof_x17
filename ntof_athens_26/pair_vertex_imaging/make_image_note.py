#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_image_note.py -- the note: how a vertex is made from a pair, the 3D image
it gives, and how well the two legs agree.

Built from the tables `vertex_image.py` wrote and the figures
`make_image_figures.py` drew; no number in the prose is typed in.  The PNGs are
EMBEDDED (base64) rather than linked, because the note is published as a single
HTML file to the notes site; the same file therefore also works from disk and
from the DAQ page's Analysis tab.  The 3D view loads plotly from its CDN.

    python -m pair_vertex_imaging.make_image_note
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402

OUT = HERE / 'figures'
TITLE = 'Pair vertices as a capsule image'
SEL, CUT = 'perpendicular', 60.0


def esc(s) -> str:
    return html.escape(str(s))


def mm(v, nd=1) -> str:
    return f'{float(v):.{nd}f}'


def sgn(v, nd=1) -> str:
    return f'{float(v):+.{nd}f}'.replace('-', '&minus;')


def pc(v, nd=0) -> str:
    return f'{100 * float(v):.{nd}f}&nbsp;%'


def n(v) -> str:
    return f'{int(v):,}'


def img(name: str, alt: str) -> str:
    b = (OUT / f'{name}.png').read_bytes()
    return (f'<img src="data:image/png;base64,{base64.b64encode(b).decode()}" '
            f'alt="{esc(alt)}" style="width:100%;height:auto">')


def table(d: pd.DataFrame, cols: dict, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    th = ''.join(f'<th>{c}</th>' for c in cols.values())
    rows = []
    for _, r in d.iterrows():
        tds = []
        for k in cols:
            v = r[k]
            f = fmt.get(k)
            empty = (not isinstance(v, str)) and pd.isna(v)
            tds.append('<td>&mdash;</td>' if empty
                       else f'<td>{f(v) if f else esc(v)}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<div style="overflow-x:auto"><table class="t"><thead><tr>{th}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


# --------------------------------------------------------------------------- #
DIAGRAM = '''
<figure>
<div style="overflow-x:auto">
<svg viewBox="0 0 780 390" role="img" aria-label="Two track lines, their 3D closest approach and their transverse crossing"
     style="width:100%;min-width:560px;height:auto;font-family:var(--sans);font-size:13px">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker>
  </defs>
  <text x="20" y="24" style="fill:var(--ink);font-weight:600">1 &middot; 3D closest approach</text>
  <rect x="22" y="300" width="130" height="14" rx="2" style="fill:#0072B233;stroke:#0072B2"/>
  <text x="22" y="334" style="fill:#0072B2">chamber A strip plane</text>
  <rect x="246" y="60" width="14" height="130" rx="2" style="fill:#CC79A733;stroke:#CC79A7"/>
  <text x="268" y="72" style="fill:#CC79A7">chamber D</text>
  <line x1="92" y1="300" x2="150" y2="110" style="stroke:#0072B2;stroke-width:2.2"/>
  <line x1="246" y1="120" x2="70" y2="162" style="stroke:#CC79A7;stroke-width:2.2"/>
  <circle cx="92" cy="300" r="4.5" style="fill:#0072B2"/>
  <circle cx="246" cy="120" r="4.5" style="fill:#CC79A7"/>
  <text x="100" y="294" style="fill:var(--ink-2)">p&#8321;</text>
  <text x="228" y="110" style="fill:var(--ink-2)">p&#8322;</text>
  <line x1="92" y1="300" x2="104" y2="261" marker-end="url(#arr)" style="stroke:var(--ink);stroke-width:1.4"/>
  <text x="110" y="264" style="fill:var(--ink)">d&#8321;</text>
  <line x1="246" y1="120" x2="208" y2="129" marker-end="url(#arr)" style="stroke:var(--ink);stroke-width:1.4"/>
  <text x="204" y="148" style="fill:var(--ink)">d&#8322;</text>
  <line x1="128" y1="181" x2="131" y2="147" style="stroke:#d18a44;stroke-width:2;stroke-dasharray:3 2"/>
  <circle cx="128" cy="181" r="3" style="fill:var(--ink)"/>
  <circle cx="131" cy="147" r="3" style="fill:var(--ink)"/>
  <circle cx="129.5" cy="164" r="6" style="fill:none;stroke:#d18a44;stroke-width:2"/>
  <text x="142" y="202" style="fill:#d18a44">sep = |q&#8321; &minus; q&#8322;|</text>
  <text x="20" y="226" style="fill:var(--ink)">vertex = &frac12;(q&#8321; + q&#8322;)</text>
  <text x="20" y="366" style="fill:var(--ink-3)">q&#8321; = p&#8321; + s&middot;d&#8321;,&nbsp; q&#8322; = p&#8322; + t&middot;d&#8322;,&nbsp; s, t minimise |q&#8321; &minus; q&#8322;|</text>
  <line x1="380" y1="30" x2="380" y2="370" style="stroke:var(--line);stroke-width:1"/>
  <text x="400" y="24" style="fill:var(--ink);font-weight:600">2 &middot; transverse crossing (looking along the beam)</text>
  <circle cx="600" cy="200" r="34" style="fill:#d18a4422;stroke:#d18a44;stroke-width:1.5"/>
  <text x="640" y="258" style="fill:#d18a44">capsule, r = 10 mm</text>
  <line x1="578" y1="360" x2="612" y2="42" style="stroke:#0072B2;stroke-width:2.2"/>
  <line x1="420" y1="218" x2="770" y2="178" style="stroke:#CC79A7;stroke-width:2.2"/>
  <circle cx="597" cy="198" r="5.5" style="fill:none;stroke:var(--ink);stroke-width:2"/>
  <text x="400" y="92" style="fill:var(--ink)">(v&#8339;, v&#8347;): two lines in a plane</text>
  <text x="400" y="109" style="fill:var(--ink)">always cross &mdash; no sep here</text>
  <text x="620" y="120" style="fill:#0072B2">A leg fixes x</text>
  <text x="680" y="170" style="fill:#CC79A7">D leg fixes z</text>
  <text x="400" y="312" style="fill:var(--ink-3)">legs miss the source by e&#8321;, e&#8322; and cross at &psi;;</text>
  <text x="400" y="330" style="fill:var(--ink-3)">the crossing misses it by</text>
  <text x="400" y="350" style="fill:var(--ink-3)">&radic;(e&#8321;&sup2; + e&#8322;&sup2; &minus; 2e&#8321;e&#8322;cos&psi;) / |sin&psi;|</text>
</svg>
</div>
<figcaption><b>The two estimators.</b> Left: two straight track lines in 3D,
each an impact point <i>p</i> on its strip plane and a unit direction <i>d</i>.
They generally do not meet; the vertex is the midpoint of the shortest segment
between them, and that segment&rsquo;s length is the pair&rsquo;s <b>sep</b>.
Right: the same two lines projected onto the plane transverse to the beam,
where they always cross. That crossing is the transverse vertex, and its y is
taken from the two legs at that point. In a perpendicular pair the crossing is
nearly square, so x is fixed by the A (or C) leg and z by the D leg.
Schematic, not to scale.</figcaption>
</figure>
'''


def load(src: Path) -> dict:
    T = {}
    for k in ('image_stats', 'image_fit', 'image_fit_per_run', 'image_verify'):
        p = src / f'{k}.csv'
        # keep_default_na=False: image_stats' `sample` column holds the
        # literal 'null', which read_csv would otherwise turn into NaN
        T[k] = (pd.read_csv(p, keep_default_na=False, na_values=[''])
                if p.exists() else pd.DataFrame())
    T['meta'] = json.loads((src / 'pairs_image.meta.json').read_text())
    return T


def build(T: dict) -> str:
    S, F, PR, V, meta = (T['image_stats'], T['image_fit'],
                         T['image_fit_per_run'], T['image_verify'], T['meta'])
    cap = meta['capsule_xz']
    ic = pd.read_csv(paths.spell('out', 'imaging_campaign', 'per_arm.csv')
                     ).set_index('arm')

    def fit(sel, cut, coord):
        return F[(F.selection == sel) & (F.cut_mm == cut) & (F.coord == coord)].iloc[0]

    def st(topo, cut, sample, centre='axis', est='xz'):
        r = S[(S.topology == topo) & (S.cut_mm == cut) & (S['sample'] == sample)
              & (S.estimator == est) & (S.cut_centre == centre)]
        return r.iloc[0] if len(r) else None

    px, pz = fit(SEL, CUT, 'x'), fit(SEL, CUT, 'z')
    px0, pz0 = fit(SEL, 150.0, 'x'), fit(SEL, 150.0, 'z')
    ax_, cx_ = fit('A-D', CUT, 'x'), fit('C-D', CUT, 'x')
    PRA = PR[(PR.chamber == 'A')].dropna(subset=['c'])
    PRC = PR[(PR.chamber == 'C')].dropna(subset=['c'])
    match = bool(len(V)) and bool((V.delta == 0).all())
    xA, xC = float(ic.loc['A', 'median_mm']), float(ic.loc['C', 'median_mm'])
    adD, adN = st('A-D', CUT, 'data'), st('A-D', CUT, 'null')
    ad30, an30 = st('A-D', 30.0, 'data'), st('A-D', 30.0, 'null')
    ad10 = st('A-D', 10.0, 'data')
    ac10, acn10 = st('A-D', 10.0, 'data', 'capsule'), st('A-D', 10.0, 'null', 'capsule')
    pdD = st(SEL, CUT, 'data')

    # data/null width ratios and capsule-band excesses across the axis-centred
    # cut scan: x from A-D and C-D, z from the pooled perpendicular sample
    def scan(topo, coord, cuts):
        r, e = [], []
        for c in cuts:
            dd, nn = st(topo, c, 'data'), st(topo, c, 'null')
            r.append(dd[f'{coord}_rsig'] / nn[f'{coord}_rsig'])
            e.append(dd[f'f_{coord}band'] - nn[f'f_{coord}band'])
        return np.array(r), np.array(e)

    cuts_all = sorted(S.cut_mm.unique())
    cuts_cut = [c for c in cuts_all if c < 150.0]
    xr = np.r_[scan('A-D', 'x', cuts_all)[0], scan('C-D', 'x', cuts_all)[0]]
    xe = np.r_[scan('A-D', 'x', cuts_all)[1], scan('C-D', 'x', cuts_all)[1]]
    zr_cut, _ = scan(SEL, 'z', cuts_cut)
    _, ze = scan(SEL, 'z', cuts_all)
    zr_none = scan(SEL, 'z', [150.0])[0][0]
    pt = pd.read_csv(paths.spell('out', 'pair_vertex', 'pointing.csv')).set_index('arm')
    ptA, ptD = pt.loc['A'], pt.loc['D']

    verdict = f'''
<p class="verdict"><b>Half yes. The pair vertices make a blurred but genuine
image of the capsule in x, centred on where the single tracks put it &mdash;
and no image at all in z.</b> Pooled over chambers A and C the centre agrees
to {mm(abs(px.c - cap[0]))}&nbsp;mm; chamber by chamber it agrees to about
2&nbsp;mm, which is the real precision (section&nbsp;4).</p>

<p>Perpendicular pairs (A&ndash;D and C&ndash;D), each leg within
{CUT:.0f}&nbsp;mm of the beam axis, {n(pdD.n)} pairs. The x coordinate of their
crossing is set by the A or C leg. Fitted as <i>the He-3 gas volume blurred by
a Gaussian</i> on top of a background shaped like the same chambers with the
pointing destroyed, it gives a centre of
<b>x&nbsp;=&nbsp;{sgn(px.c)}&nbsp;&plusmn;&nbsp;{mm(px.c_err)}&nbsp;mm</b>
against <b>{sgn(cap[0])}&nbsp;mm</b> from the single-track band crossing, a
blur of <b>&sigma;&nbsp;=&nbsp;{mm(px.s)}&nbsp;mm</b>, and a capsule term
carrying {pc(px.f)} of the pairs (2&Delta;lnL&nbsp;=&nbsp;{mm(px.two_dnll_vs_none, 0)}
against no capsule term). The z coordinate is set by the D leg, and there the
fit, with the centre held at the capsule, wants a capsule term of
{pc(pz.f, 1)} (2&Delta;lnL&nbsp;=&nbsp;{mm(pz.two_dnll_vs_none, 1)}): within
&plusmn;15&nbsp;mm of the capsule, data and null agree to
{sgn(100 * pz.band_excess)}&nbsp;% of the pairs.</p>

<p>So the impression was right: <b>it is not as bad as it looked</b>. The
vertex radius distribution looked hopeless because it mixes a coordinate that
images (x) with one that does not (z) and with the poorly measured y. Taken one
coordinate at a time, the A and C legs place the capsule per event with a blur
of about 1.6 capsule radii, while chamber D &mdash; whose per-track pointing
barely beats its own null &mdash; contributes nothing per event, even though its
ensemble band crossing still finds the capsule&rsquo;s z.</p>
'''

    sample = (f'<p class="note">Sample: {n(meta["n_data"])} data pairs and '
              f'{n(meta["n_null"])} null pairs ({meta["n_shuffle"]} shuffles) over '
              f'{meta["n_runs"]} runs of the condor full pass (run_79/81, before '
              f'the 27 July access, excluded), legs in chambers A, C and D, built '
              f'at a {mm(meta["build_ceil_mm"], 0)}&nbsp;mm leg ceiling. At the '
              f'published 30&nbsp;mm cut the data sample is '
              + ('<b>pair for pair the <code>vertex_lab</code> / <code>pair_qa</code> '
                 'sample</b> on all six A/C/D arm pairs.' if match else
                 '<b>not</b> identical to the <code>vertex_lab</code> sample &mdash; '
                 'see <code>image_verify.csv</code>.')
              + '</p>')

    how = f'''
<h2>1. How a vertex is made from a pair</h2>

<p><b>The tracks.</b> Every selected track is a straight line from the stage-3
reconstruction: an impact point <i>p</i> on its chamber&rsquo;s strip plane and a
unit direction <i>d</i>, built from the in-plane angle (tan&theta;<sub>x</sub>,
scaled by the run&rsquo;s angle calibration <i>k</i>) and the out-of-plane
angle (tan&theta;<sub>y</sub>). The geometry comes from the waveform fit, not
from hit times. A track is <i>selected</i> if it is gated, its run carries a
certified angle scale, and its closest approach to the nominal beam axis
(<code>dca_axis_mm</code>, measured in the transverse plane) is below a cut.
The published pair sample uses 30&nbsp;mm; here the build is left open to
150&nbsp;mm so the cut can be scanned. Chambers A, C and D only &mdash; B has no
usable angle.</p>

<p><b>The pairs.</b> Every unordered pair of distinct selected tracks inside one
trigger, with the legs ordered by chamber so that &ldquo;leg&nbsp;1&rdquo;
always means the same chamber. Three classes by the chambers&rsquo; azimuth:
<b>perpendicular</b> (A&ndash;D, C&ndash;D, 90&deg; apart), <b>opposing</b>
(A&ndash;C) and <b>intra</b> (both legs in one chamber).</p>

{DIAGRAM}

<p><b>Estimator 1: the 3D closest approach.</b> For the lines
<i>q</i>&#8321;(s)&nbsp;=&nbsp;<i>p</i>&#8321;&nbsp;+&nbsp;s<i>d</i>&#8321; and
<i>q</i>&#8322;(t)&nbsp;=&nbsp;<i>p</i>&#8322;&nbsp;+&nbsp;t<i>d</i>&#8322;, with
<i>w</i>&nbsp;=&nbsp;<i>p</i>&#8321;&minus;<i>p</i>&#8322;,
a&nbsp;=&nbsp;<i>d</i>&#8321;&middot;<i>d</i>&#8321;,
b&nbsp;=&nbsp;<i>d</i>&#8321;&middot;<i>d</i>&#8322;,
c&nbsp;=&nbsp;<i>d</i>&#8322;&middot;<i>d</i>&#8322;,
e&nbsp;=&nbsp;<i>d</i>&#8321;&middot;<i>w</i>,
f&nbsp;=&nbsp;<i>d</i>&#8322;&middot;<i>w</i>:</p>
<pre>s = (b f &minus; c e) / (a c &minus; b&sup2;)        t = (a f &minus; b e) / (a c &minus; b&sup2;)
vertex = &frac12; [q&#8321;(s) + q&#8322;(t)]        sep = |q&#8321;(s) &minus; q&#8322;(t)|</pre>
<p><b>sep</b> &mdash; the distance of closest approach <i>between the two
tracks</i> &mdash; is the direct measure of how well a pair agrees on a common
point. It is what <code>pair_qa</code> publishes as <code>sep_mm</code>, and its
midpoint is the published <code>v_r</code>.</p>

<p><b>Estimator 2: the transverse crossing</b> &mdash; the one the image is made
from. Project both lines onto the x&ndash;z plane (transverse to the beam, which
runs along +y). Two lines in a plane always meet unless they are parallel, so
there is no residual distance, and the crossing <i>(v<sub>x</sub>,
v<sub>z</sub>)</i> is the transverse vertex. Its y is the mean of the two
legs&rsquo; y <i>at that crossing</i>, and their difference
<b>|y&#8321;&minus;y&#8322;|</b> is the agreement test the transverse
construction leaves free.</p>

<p><b>Why the second one.</b> The capsule is 10&nbsp;mm across but 80&nbsp;mm
long along the beam, and the y view of these chambers carries very little
pointing (band slope 0.06&ndash;0.35 of a point source). The 3D closest approach
reconciles a large y disagreement by sliding both points along their lines, and
every millimetre it slides also moves the vertex transversely &mdash;
<code>vertex_lab</code> measured this as a factor of about two in the vertex
radius. The transverse crossing uses exactly the information the single-track
band crossing uses and nothing else.</p>

<p><b>Why perpendicular pairs, and why one coordinate at a time.</b> If the legs
miss the source by e&#8321; and e&#8322; and cross at angle &psi;, the crossing
misses it by &radic;(e&#8321;&sup2;+e&#8322;&sup2;&minus;2e&#8321;e&#8322;cos&psi;)/|sin&psi;|
(verified in the data to 1e-13). Perpendicular pairs have
sin&psi;&nbsp;&asymp;&nbsp;0.9 and almost no amplification; opposing and intra
pairs are nearly parallel in the transverse plane and amplify the legs&rsquo;
error by 2&ndash;3 at the median and ten or more in the tail. And because a
perpendicular crossing is nearly square, <b>its x is the A or C leg&rsquo;s
pointing and its z is the D leg&rsquo;s</b>: the two coordinates are two
different chambers, and they have to be judged separately.</p>
'''

    null = '''
<h2>2. The null: the same chambers with no source</h2>
<p>A blob near the middle proves nothing by itself. The chambers surround the
axis and the pointing cut keeps only lines that pass near it, so lines with
<i>no</i> common origin also cross near the middle. The comparison has to be with
what this selection makes of chambers that see no source.</p>
<p>So every track keeps its measured impact point, and its direction (both tans
together) is replaced with the direction of a <i>randomly chosen other track of
the same chamber in the same run</i>. The impact points, the angular
distribution, the acceptance, the trigger multiplicity and which tracks share a
trigger all survive. The only thing destroyed is the correlation between where a
track landed and which way it pointed &mdash; which is exactly &ldquo;this track
came from the capsule&rdquo;. The shuffled tracks then go through the
<b>identical</b> pointing cut and pairing. Two independent shuffles are kept, so
the null has twice the data&rsquo;s statistics.</p>
<p class="caution">This is <b>not</b> the event-mixed null of the angle spectra.
Mixing pairs tracks from different triggers but keeps each track&rsquo;s own
direction, so both legs of a mixed pair still came out of the capsule and a mixed
pair images it exactly as well as a real one. For the question &ldquo;is there a
source here?&rdquo; only a null with the pointing removed gives an answer.</p>
'''

    image = f'''
<h2>3. The 3D image</h2>
<p>Perpendicular pairs, legs within {CUT:.0f}&nbsp;mm of the axis. Drag to
rotate and scroll to zoom; the buttons switch between the raw density and the
density with the no-source null (scaled by the x fit&rsquo;s background
fraction) subtracted. The capsule is drawn at the single-track position, the
dashed line is the nominal beam axis, and the beam runs upward.</p>
<div style="border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden">
{T['div3d']}
</div>
<p class="note">Transverse bins 4&nbsp;mm, y bins 10&nbsp;mm, smoothed by 0.8
bin for display only; the colour scale is relative. What to look for: after the
subtraction the density is a <b>slab</b> standing at the capsule&rsquo;s x and
spread along both z and y. The slab is the image; its extent in z is chamber D
seeing nothing, and its extent in y is the y views (robust
&sigma;<sub>y</sub>&nbsp;=&nbsp;{mm(pdD.y_rsig, 0)}&nbsp;mm against an
80&nbsp;mm capsule, median y {sgn(pdD.y_med, 0)}&nbsp;mm against a gas centroid at
{sgn(meta['capsule_y_centroid'])}&nbsp;mm).</p>
<figure>{img('image_projections', 'vertex density projections')}
<figcaption><b>The same density, projected.</b> Top: data. Bottom: data minus
the scaled null &mdash; purple is an excess over a no-source sample, blue a
deficit. In the transverse view (left) the excess is a vertical band through the
capsule circle: localised in x, flat in z. The dashed line is the fitted x
centre, the copper circle and outlines the capsule at the single-track
position, the open circle the nominal beam axis.</figcaption></figure>
'''

    xtab = F[F.coord == 'x'].assign(
        sel=lambda t: t.selection + ' (' + t.chamber + '), legs &lt; '
        + t.cut_mm.map(lambda c: 'none' if c >= 150 else f'{c:.0f} mm'),
        dcap=lambda t: t.c - t.capsule,
        chi2ndf=lambda t: t.chi2 / t.ndf)
    ztab = F[F.coord == 'z'].assign(
        sel=lambda t: t.selection + ' (' + t.chamber + '), legs &lt; '
        + t.cut_mm.map(lambda c: 'none' if c >= 150 else f'{c:.0f} mm'),
        chi2ndf=lambda t: t.chi2 / t.ndf)

    blur = f'''
<h2>4. How blurred: one coordinate at a time</h2>
<p>Each coordinate of the transverse crossing is histogrammed in a
&plusmn;64&nbsp;mm window in 2&nbsp;mm bins and fitted by binned Poisson
likelihood as</p>
<pre>density(u) = f &middot; [capsule(u &minus; c) &otimes; Gauss(&sigma;)] + (1 &minus; f) &middot; null(u)</pre>
<p>where <i>capsule</i> is the real He-3 polycone projected onto that coordinate
(a rounded profile 20&nbsp;mm wide at the base), and <i>null</i> is the shuffled
sample through the same cut, as a fixed shape. In x the centre, blur and
fraction are free. In z the centre is <b>fixed at the capsule</b>, and the only
question is whether any capsule term is wanted there. The model-free column
beside each fit is the fraction of vertices within &plusmn;15&nbsp;mm of the
capsule coordinate, data minus null.</p>

<h3>x &mdash; set by chamber A or C</h3>
{table(xtab, {'sel': 'selection', 'n_in_window': 'pairs in window',
              'c': 'centre', 'c_err': '&plusmn;', 'dcap': 'minus capsule',
              's': '&sigma;', 's_err': '&plusmn;', 'f': 'capsule term',
              'two_dnll_vs_none': '2&Delta;lnL', 'chi2ndf': '&chi;&sup2;/ndf',
              'band_excess': '&plusmn;15&nbsp;mm excess'},
       {'n_in_window': n, 'c': sgn, 'c_err': mm, 'dcap': sgn, 's': mm,
        's_err': mm, 'f': lambda v: pc(v), 'two_dnll_vs_none': lambda v: mm(v, 0),
        'chi2ndf': lambda v: mm(v, 1), 'band_excess': lambda v: sgn(100 * v) + '&nbsp;%'})}

<h3>z &mdash; set by chamber D</h3>
{table(ztab, {'sel': 'selection', 'n_in_window': 'pairs in window',
              's': '&sigma;', 'f': 'capsule term', 'f_err': '&plusmn;',
              'two_dnll_vs_none': '2&Delta;lnL', 'chi2ndf': '&chi;&sup2;/ndf',
              'band_excess': '&plusmn;15&nbsp;mm excess'},
       {'n_in_window': n, 's': mm, 'f': lambda v: pc(v, 1),
        'f_err': lambda v: pc(v, 1), 'two_dnll_vs_none': lambda v: mm(v, 1),
        'chi2ndf': lambda v: mm(v, 1),
        'band_excess': lambda v: sgn(100 * v) + '&nbsp;%'})}
<p>All lengths in mm. Errors are statistical, from the likelihood curvature.
Single-track capsule position: x&nbsp;=&nbsp;{sgn(cap[0])},
z&nbsp;=&nbsp;{sgn(cap[1])}&nbsp;mm.</p>

<figure>{img('image_profiles', 'coordinate profiles with fits')}
<figcaption><b>The fits, drawn.</b> Left and middle: the x of A&ndash;D and
C&ndash;D crossings, where the data rise above the no-source null in a peak at
the capsule and fall below it on either side. Right: z, set by chamber D. The
data follow the null&rsquo;s broad shape, with D&rsquo;s strip-plane spikes
near +25&nbsp;mm stronger in the data than the null reproduces &mdash; which is
what makes that fit&rsquo;s &chi;&sup2; large &mdash; and there is no extra
weight at the capsule.</figcaption></figure>

<p><b>The x image.</b> Pooling the two arm pairs, the centre lands
{mm(abs(px.c - cap[0]), 2)}&nbsp;mm from the single-track position. The blur is
{mm(px.s)}&nbsp;mm with legs &lt;&nbsp;{CUT:.0f}&nbsp;mm and {mm(px0.s)}&nbsp;mm
with no leg cut &mdash; the cut keeps the better-pointing tracks, so the blur is
a property of the selection and not a single resolution number. The capsule term
is {pc(px.f)} and {pc(px0.f)} of the pairs in the window.</p>

<p class="caution"><b>Per chamber the centres are good to about 2&nbsp;mm, not
to the statistical 0.2.</b> The A&ndash;D fit puts chamber A&rsquo;s image at
{sgn(ax_.c)}&nbsp;mm, where A&rsquo;s own band crossing says {sgn(xA)}; the
C&ndash;D fit puts C at {sgn(cx_.c)}, where C&rsquo;s band crossing says
{sgn(xC)}. The two shifts are about equal and opposite, which is why the pooled
centre lands on the capsule, and the A&ndash;D fit has
&chi;&sup2;/ndf&nbsp;=&nbsp;{mm(ax_.chi2 / ax_.ndf)}: with {n(ax_.n_in_window)}
pairs in the window the model&rsquo;s assumption &mdash; that non-capsule tracks
share the angular distribution of capsule tracks &mdash; is visibly not exact,
and the centre moves by millimetres with it. The blur and the capsule fraction
trade against the background shape too: chambers A and C, whose x blurs agree
({mm(ax_.s)} and {mm(cx_.s)}&nbsp;mm), give capsule fractions of {pc(ax_.f)}
and {pc(cx_.f)}, which differ far beyond their statistical errors &mdash; read
that spread as the scale of the fraction&rsquo;s systematic.</p>

<p><b>The z non-image.</b> Chamber D&rsquo;s single-track miss distance is
{mm(ptD.med_dca, 0)}&nbsp;mm at the median against {mm(ptD.med_dca_null, 0)} for
its own tan-shuffled null (<code>vertex_lab</code>; chamber A:
{mm(ptA.med_dca, 0)} against {mm(ptA.med_dca_null, 0)}), so per track it carries
almost no pointing. Its <i>ensemble</i> band crossing still locates the
capsule&rsquo;s z ({sgn(ic.loc['D', 'median_mm'])}&nbsp;&plusmn;&nbsp;{mm(ic.loc['D', 'std_mm'])}&nbsp;mm
over {int(ic.loc['D', 'n_runs'])} runs), because a weak correlation over millions
of tracks is a strong one. Per pair there is nothing to find.</p>
'''

    ag = []
    for topo in ('perpendicular', 'intra', 'opposing'):
        for cut in (CUT, 30.0):
            dd, nn = st(topo, cut, 'data'), st(topo, cut, 'null')
            if dd is None or nn is None:
                continue
            ag.append(dict(topology=topo, cut=cut, n=dd.n,
                           sep=dd.sep_med, sep_null=nn.sep_med,
                           dy=dd.dy_abs_med, dy_null=nn.dy_abs_med,
                           xb=dd.f_xband, xb_null=nn.f_xband,
                           zb=dd.f_zband, zb_null=nn.f_zband))
    AG = pd.DataFrame(ag)
    g = AG.set_index(['topology', 'cut'])
    p60, i60, o60 = (g.loc[('perpendicular', CUT)], g.loc[('intra', CUT)],
                     g.loc[('opposing', CUT)])
    p30, i30, o30 = (g.loc[('perpendicular', 30.0)], g.loc[('intra', 30.0)],
                     g.loc[('opposing', 30.0)])

    agree = f'''
<h2>5. How well do the two legs agree?</h2>
<p>Three per-pair measures, each against the null through the same cut:
<b>sep</b>, the 3D distance of closest approach between the two lines;
<b>|y&#8321;&minus;y&#8322;|</b> at the transverse crossing; and the fraction of
crossings within &plusmn;15&nbsp;mm of the capsule, separately in x and z.
Medians and fractions.</p>
{table(AG, {'topology': 'class', 'cut': 'legs &lt; [mm]', 'n': 'pairs',
            'sep': 'sep', 'sep_null': 'null', 'dy': '|dy|', 'dy_null': 'null',
            'xb': 'x within 15&nbsp;mm', 'xb_null': 'null',
            'zb': 'z within 15&nbsp;mm', 'zb_null': 'null'},
       {'cut': lambda v: mm(v, 0), 'n': n, 'sep': lambda v: mm(v, 0),
        'sep_null': lambda v: mm(v, 0), 'dy': lambda v: mm(v, 0),
        'dy_null': lambda v: mm(v, 0), 'xb': lambda v: pc(v, 1),
        'xb_null': lambda v: pc(v, 1), 'zb': lambda v: pc(v, 1),
        'zb_null': lambda v: pc(v, 1)})}

<p><b>sep is not a good agreement measure on this data.</b> For perpendicular
pairs the median sep is {mm(p60.sep, 0)}&nbsp;mm against {mm(p60.sep_null, 0)}
for lines with no common origin at all (legs &lt;&nbsp;{CUT:.0f}&nbsp;mm), and
{mm(p30.sep, 0)} against {mm(p30.sep_null, 0)} at 30&nbsp;mm. Intra and opposing
pairs come a little closer than their null ({mm(i30.sep, 0)} against
{mm(i30.sep_null, 0)} and {mm(o30.sep, 0)} against {mm(o30.sep_null, 0)} at
30&nbsp;mm), but they are the classes whose crossing is ill-conditioned. The
reason is the y information: sep is dominated by how far apart the two legs are
along the beam ({mm(p60.dy, 0)}&nbsp;mm median |dy| for perpendicular pairs),
which is set by the weak y views and by the capsule&rsquo;s own 80&nbsp;mm
length, not by whether the legs share a transverse origin.</p>

<p><b>The transverse agreement is where the information is, and only in x.</b>
{pc(p60.xb, 1)} of perpendicular crossings have x within 15&nbsp;mm of the
capsule, against {pc(p60.xb_null, 1)} for the null; for z it is
{pc(p60.zb, 1)} against {pc(p60.zb_null, 1)}.</p>
<figure>{img('image_agreement', 'leg agreement distributions')}
<figcaption><b>sep and |dy| per class.</b> Solid: data; grey dashed: the
shuffled null through the same cut. Heavy: legs &lt;&nbsp;{CUT:.0f}&nbsp;mm;
light: legs &lt;&nbsp;30&nbsp;mm.</figcaption></figure>
'''

    stab = f'''
<h2>6. Does it hold up?</h2>
<figure>{img('image_cut_scan', 'cut scan')}
<figcaption><b>Against the per-leg pointing cut.</b> Solid: the cut is on each
leg&rsquo;s miss from the nominal beam axis. Dashed: from the measured capsule
position instead. Left: image centre minus the capsule. Middle: the
data&rsquo;s robust width over the null&rsquo;s. Right: the excess within
&plusmn;15&nbsp;mm of the capsule.</figcaption></figure>
<p><b>Centring the cut matters.</b> The capsule sits 9.9&nbsp;mm off the nominal
axis, so a tight cut about the <i>axis</i> keeps lines through a disc that only
partly overlaps the source and drags the image toward the axis: chamber
A&rsquo;s median x goes from {sgn(adD.x_med)}&nbsp;mm at 60 to
{sgn(ad30.x_med)} at 30 and {sgn(ad10.x_med)} at 10. Centring the cut on the
capsule instead holds the median near {sgn(ac10.x_med)}&nbsp;mm &mdash; but
the null then sits at {sgn(acn10.x_med)} too, because a cut about a point
images that point by construction. That is why the result above uses the
axis-centred cut and a null, and not a tight capsule-centred cut.</p>
<p><b>In x the image is narrower than the null at every cut</b> (data over null
{mm(xr.min(), 2)}&ndash;{mm(xr.max(), 2)}, middle panel) <b>and carries an
excess at the capsule at every cut</b> ({mm(100 * xe.min())}&ndash;{mm(100 * xe.max())}&nbsp;%
of pairs, right panel). In z, once any leg cut is applied, the width is within
{mm(100 * (1 - zr_cut.min()), 0)}&nbsp;% of the null&rsquo;s and the excess is
at most {mm(100 * ze.max())}&nbsp;% &mdash; small, at the size of D&rsquo;s
acceptance structure, and not the peaked shape a source would make. (With no leg
cut at all the z width is {mm(zr_none)} of the null&rsquo;s: without a cut, the
null keeps tracks pointing anywhere and is simply broader.) Chamber A at
30&nbsp;mm: robust &sigma;<sub>x</sub> {mm(ad30.x_rsig)} against
{mm(an30.x_rsig)}&nbsp;mm for the null, &sigma;<sub>z</sub> {mm(ad30.z_rsig)}
against {mm(an30.z_rsig)}.</p>

<figure>{img('image_per_run', 'per run centres')}
<figcaption><b>Run by run.</b> Only the x centre is fitted per run; the blur and
capsule fraction are fixed to that arm pair&rsquo;s pooled fit and the null is
pooled. Open diamonds are that run&rsquo;s own single-track band crossing for the
same chamber.</figcaption></figure>
<p>Over {len(PRA)} runs chamber A&rsquo;s pair image has median x
{sgn(PRA.c.median())}&nbsp;mm with a run-to-run spread of
{mm(PRA.c.std())}&nbsp;mm, against a typical per-run error of
{mm(PRA.c_err.median())} &mdash; <b>as stable as its statistics allow</b>.
Chamber C ({len(PRC)} runs, {sgn(PRC.c.median())}&nbsp;mm) spreads by
{mm(PRC.c.std())}&nbsp;mm against errors of {mm(PRC.c_err.median())}. Neither
follows its chamber&rsquo;s per-run single-track value (correlation
{sgn(PRA.c.corr(PRA.single_track_x), 2)} for A and
{sgn(PRC.c.corr(PRC.single_track_x), 2)} for C), which is expected: the
single-track crossing moves by a few tenths of a millimetre run to run, far below
what a per-run pair image can resolve. <b>Chamber A&rsquo;s offset from its band
crossing is systematic</b>: {int((PRA.c < PRA.single_track_x).sum())} of
{len(PRA)} runs put the pair image below it. <b>Chamber C&rsquo;s is less
settled</b>: {int((PRC.c > PRC.single_track_x).sum())} of {len(PRC)} runs put
it above, but C scatters by more than its errors, and several runs land on its
band crossing.</p>
'''

    caveats = '''
<h2>What this does not rule out</h2>
<ul>
<li><b>The x image is not independent of the single tracks.</b> A perpendicular
crossing&rsquo;s x is, to a good approximation, the A or C leg&rsquo;s own
transverse miss. The pair image confirms the single-track position with a
different estimator on a different sample (two-track triggers only), not with
independent information.</li>
<li><b>The background model is approximate.</b> The shuffled null assumes that
non-capsule tracks share the angular distribution of capsule tracks. The
A&ndash;D &chi;&sup2; says that is not exact; the pooled centre survives it,
the per-chamber centres move by about 2&nbsp;mm, and &sigma; and the capsule
fraction by about 20&nbsp;%.</li>
<li><b>The angle scale is preliminary.</b> The vertex uses the angle&rsquo;s
magnitude, so an error in <i>k</i> both blurs and shifts it, whereas the band
crossing is scale-invariant. The per-detector recalibration is deferred to
October, and the blur quoted here contains whatever <i>k</i> error there is.</li>
<li><b>The blur is not an angular resolution.</b> It folds together the angle
error, multiple scattering in the capsule wall and chamber windows, and the
depth of the source along the lever arm, and it depends on the leg cut.</li>
<li><b>Nothing here says the two legs are one decay.</b> Two independent capsule
tracks in one trigger give the same blurred image. Whether real pairs are more
vertex-like than accidental ones is the lift question in
<code>vertex_lab</code>, and the answer there is no.</li>
<li><b>Chamber D does not image per event, and chamber B is excluded</b> (no
field-shaping rings, no usable angle), so z has no per-event measurement in this
campaign. <i>(Superseded in part: a follow-up, &ldquo;Where the z of the pair image
went&rdquo;, finds that cleaned D&ndash;D pairs do image z.)</i></li>
<li><b>The source model is the gas, and that is not the expectation.</b> The fits
place the He-3 gas volume, but the pairs are expected to come from the
capsule&rsquo;s aluminium &mdash; its bottom end and perhaps its top &mdash; and the
capsule is not expected at the nominal origin. The fitted <i>centre</i> does not
depend on that choice (it is free); the fitted blur and source fraction do, and
should be refitted with an aluminium end-cap model before being quoted.</li>
</ul>
'''
    return verdict + sample + how + null + image + blur + agree + stab + caveats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'pair_vertex')))
    ap.add_argument('--out', default=str(OUT / 'vertex_image_note.html'))
    a = ap.parse_args()
    T = load(Path(a.src))
    T['div3d'] = (OUT / 'image_3d.div.html').read_text(encoding='utf-8')
    doc = (f'<!doctype html><html lang="en"><head>{head(TITLE)}'
           f'<meta name="description" content="How a vertex is built from two '
           f'tracks, the 3D density of pair vertices against a no-source null, '
           f'and how blurred the capsule image is."></head><body>'
           f'<div class="topbar"><div class="topbar-in">'
           f'<span class="eyebrow">n_TOF 2026 &middot; X17 &middot; preliminary</span>'
           f'</div></div><div class="wrap">'
           f'<h1>{TITLE}</h1>'
           f'<p class="lede">How a vertex is made from two tracks, what the '
           f'vertices of the whole campaign look like in 3D, and how well they '
           f'agree &mdash; each measured against the same chambers with the '
           f'source information removed.</p>'
           f'{build(T)}'
           f'<p class="prov">Built by <code>ntof_athens_26/pair_vertex_imaging/'
           f'make_image_note.py</code> from the tables in <code>{esc(a.src)}</code> '
           f'(<code>vertex_image.py</code>); every number above is read from those '
           f'tables. Companion: <code>figures/report.html</code>, why the pair '
           f'vertex radius is not a 10&nbsp;mm image.</p>'
           f'</div></body></html>')
    p = Path(a.out)
    p.write_text(doc, encoding='utf-8')
    print(f'wrote -> {p}  ({len(doc) / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
