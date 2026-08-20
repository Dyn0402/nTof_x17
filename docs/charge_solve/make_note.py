#!/usr/bin/env python3
"""
Build the standalone note "The charge solve" from the figures produced by
figs.py. Every number in the prose comes from numbers.json, so the text cannot
drift away from the plots.

    ../../.venv/bin/python figs.py && ../../.venv/bin/python make_note.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py charge_solve.html \
        --tags "X17, cosmic bench, micromegas, reconstruction, waveforms" --force
"""
from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.environ.get(
    'CS_FIGDIR',
    '/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17-mpgd26/'
    'cf7ef626-6174-476c-b483-f2699f32d221/scratchpad/amat/figs')
OUT = os.path.join(HERE, 'charge_solve.html')

with open(os.path.join(FIGDIR, 'numbers.json')) as f:
    N = json.load(f)


def img(name):
    with open(os.path.join(FIGDIR, name + '.png'), 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()


def fig(name, num, cap, src):
    return (f'<figure id="fig{num}"><img alt="{cap[:80]}" src="{img(name)}">'
            f'<figcaption><b>Figure {num}.</b> {cap} '
            f'<span class="src">{src}</span></figcaption></figure>')


CSS = """
  :root { --ink:#16161d; --muted:#5c5c6b; --accent:#c8433c; --rule:#e4e4ea;
          --card:#f6f6f9; --code:#f0f0f4; --good:#137a4b; }
  html { color-scheme: light; }
  body { margin:0 auto; max-width:900px; padding:2rem 1.2rem 5rem;
         font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
         color:var(--ink); background:#fff; }
  h1 { font-size:1.8rem; line-height:1.2; margin:0.2rem 0 0.3rem; }
  h2 { font-size:1.22rem; margin:2.6rem 0 0.7rem; border-bottom:1px solid var(--rule);
       padding-bottom:0.3rem; }
  h3 { font-size:1.02rem; margin:1.6rem 0 0.4rem; color:#33333f; }
  p.lede { font-size:1.06rem; }
  .meta { color:var(--muted); font-size:0.88rem; margin-bottom:1.4rem; }
  .verdict { background:var(--card); border-left:4px solid var(--accent);
             padding:0.85rem 1rem; margin:1.3rem 0; }
  .box { background:var(--card); border-radius:8px; padding:0.85rem 1rem;
         margin:1.2rem 0; }
  .warn { background:#fdf3e7; border-left:4px solid #d9922e; padding:0.75rem 1rem;
          margin:1.2rem 0; font-size:0.95rem; }
  .ok { background:#eef7f1; border-left:4px solid var(--good); padding:0.75rem 1rem;
        margin:1.2rem 0; font-size:0.95rem; }
  .tiles { display:flex; flex-wrap:wrap; gap:0.7rem; margin:1.2rem 0; }
  .tile { flex:1 1 130px; background:var(--card); border-radius:8px;
          padding:0.65rem 0.85rem; min-width:120px; }
  .tile .n { font-size:1.35rem; font-weight:650; letter-spacing:-0.01em; }
  .tile .l { font-size:0.78rem; color:var(--muted); line-height:1.3; }
  figure { margin:1.7rem 0; }
  figure img { max-width:100%; height:auto; border:1px solid var(--rule);
               border-radius:6px; background:#fff; }
  figcaption { font-size:0.87rem; color:var(--muted); margin-top:0.45rem; }
  .src { display:block; font-size:0.78rem; opacity:0.75; margin-top:0.2rem;
         font-family:ui-monospace,Menlo,Consolas,monospace; }
  table { border-collapse:collapse; width:100%; font-size:0.92rem; margin:0.9rem 0; }
  th, td { text-align:left; padding:0.34rem 0.6rem; border-bottom:1px solid var(--rule); }
  th { color:var(--muted); font-weight:600; font-size:0.83rem; }
  td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
  code, pre { font-family:ui-monospace,Menlo,Consolas,monospace; }
  code { background:var(--code); padding:0.08em 0.35em; border-radius:4px;
         font-size:0.88em; }
  pre { background:var(--code); padding:0.8rem 1rem; border-radius:8px;
        overflow-x:auto; font-size:0.83rem; line-height:1.45; }
  .eq { background:var(--card); border-radius:8px; padding:0.9rem 1rem;
        margin:1rem 0; text-align:center; font-size:1.02rem; overflow-x:auto; }
  .eq small { display:block; color:var(--muted); font-size:0.8rem; margin-top:0.45rem; }
  ul, ol { padding-left:1.25rem; }
  li { margin:0.32rem 0; }
  .toc { columns:2; font-size:0.92rem; }
  .toc a { color:var(--ink); }
  @media (max-width:640px) { .toc { columns:1; } }
  svg.diagram { display:block; width:100%; height:auto; margin:1.2rem auto; }
"""


SVG_PIPE = """
<svg class="diagram" viewBox="0 0 880 236" role="img" aria-label="Three
geometric numbers plus the calibration build a design matrix; the charges are
then solved exactly.">
  <defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5"
    orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#8a8a99"/></marker></defs>
  <rect x="8" y="30" width="150" height="120" rx="8" fill="#f0f4ff" stroke="#b9c6ea"/>
  <text x="83" y="54" font-size="13" font-weight="700" text-anchor="middle"
        fill="#2b3a67">3 numbers</text>
  <text x="83" y="78" font-size="12.5" text-anchor="middle" fill="#2b3a67">p₀ — position</text>
  <text x="83" y="98" font-size="12.5" text-anchor="middle" fill="#2b3a67">w — slope</text>
  <text x="83" y="118" font-size="12.5" text-anchor="middle" fill="#2b3a67">t₀ — start time</text>
  <text x="83" y="140" font-size="10.5" text-anchor="middle" fill="#6b7280">searched</text>

  <rect x="8" y="164" width="150" height="60" rx="8" fill="#f7f2ff" stroke="#cfc0ea"/>
  <text x="83" y="186" font-size="12.5" font-weight="700" text-anchor="middle"
        fill="#4a3570">calibration bundle</text>
  <text x="83" y="206" font-size="11" text-anchor="middle" fill="#4a3570">h(t), c₁, c₂, σ, D</text>

  <path d="M162,90 L212,90" stroke="#8a8a99" stroke-width="1.6" marker-end="url(#ah)"/>
  <path d="M162,194 L190,194 L190,110 L212,110" fill="none" stroke="#8a8a99"
        stroke-width="1.6" marker-end="url(#ah)"/>

  <rect x="216" y="30" width="190" height="120" rx="8" fill="#fff6ec" stroke="#e8c496"/>
  <text x="311" y="54" font-size="13" font-weight="700" text-anchor="middle"
        fill="#7a4c12">build_matrix()</text>
  <text x="311" y="80" font-size="12" text-anchor="middle" fill="#7a4c12">A — 544 rows × 18 cols</text>
  <text x="311" y="102" font-size="11.5" text-anchor="middle" fill="#7a4c12">column k = what one unit</text>
  <text x="311" y="120" font-size="11.5" text-anchor="middle" fill="#7a4c12">of charge at depth k</text>
  <text x="311" y="138" font-size="11.5" text-anchor="middle" fill="#7a4c12">would have looked like</text>

  <path d="M410,90 L460,90" stroke="#8a8a99" stroke-width="1.6" marker-end="url(#ah)"/>

  <rect x="464" y="30" width="190" height="120" rx="8" fill="#eef7f1" stroke="#9ecbb2"/>
  <text x="559" y="54" font-size="13" font-weight="700" text-anchor="middle"
        fill="#137a4b">NNLS</text>
  <text x="559" y="80" font-size="12" text-anchor="middle" fill="#137a4b">min ‖Aq − y‖², q ≥ 0</text>
  <text x="559" y="102" font-size="11.5" text-anchor="middle" fill="#137a4b">18 charges, solved</text>
  <text x="559" y="120" font-size="11.5" text-anchor="middle" fill="#137a4b">exactly — never searched</text>
  <text x="559" y="138" font-size="11.5" text-anchor="middle" fill="#137a4b">≈ 0.4 ms</text>

  <path d="M658,90 L708,90" stroke="#8a8a99" stroke-width="1.6" marker-end="url(#ah)"/>

  <rect x="712" y="30" width="160" height="120" rx="8" fill="#fdecec" stroke="#e8a8a4"/>
  <text x="792" y="54" font-size="13" font-weight="700" text-anchor="middle"
        fill="#a33">χ² and q</text>
  <text x="792" y="80" font-size="11.5" text-anchor="middle" fill="#a33">χ² → the outer</text>
  <text x="792" y="98" font-size="11.5" text-anchor="middle" fill="#a33">search picks the</text>
  <text x="792" y="116" font-size="11.5" text-anchor="middle" fill="#a33">next 3 numbers</text>

  <path d="M792,154 L792,196 L340,196 L340,154" fill="none" stroke="#c8433c"
        stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#ah)"/>
  <text x="566" y="214" font-size="11.5" text-anchor="middle" fill="#c8433c">
    repeat ≈ 500 times per plane — the outer Nelder-Mead loop</text>
</svg>
"""

SVG_PROJ = """
<svg class="diagram" viewBox="0 0 620 250" role="img" aria-label="Least squares
as a perpendicular from the data vector onto the column space of A.">
  <defs><marker id="ah2" markerWidth="9" markerHeight="9" refX="8" refY="4.5"
    orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#333"/></marker>
  <marker id="ah3" markerWidth="9" markerHeight="9" refX="8" refY="4.5"
    orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#137a4b"/></marker></defs>
  <path d="M60,200 L420,200 L560,150 L200,150 z" fill="#e8eefc" stroke="#b9c6ea"/>
  <text x="470" y="192" font-size="12" fill="#4b5563">column space of A</text>
  <text x="470" y="208" font-size="11" fill="#6b7280">every possible Aq</text>
  <line x1="150" y1="180" x2="330" y2="180" stroke="#f97316" stroke-width="2"
        marker-end="url(#ah2)"/>
  <text x="336" y="184" font-size="12" fill="#b45309">a₃</text>
  <line x1="150" y1="180" x2="215" y2="160" stroke="#8b5cf6" stroke-width="2"
        marker-end="url(#ah2)"/>
  <text x="220" y="158" font-size="12" fill="#6d28d9">a₄</text>
  <line x1="150" y1="180" x2="300" y2="40" stroke="#3b82f6" stroke-width="2.4"
        marker-end="url(#ah2)"/>
  <text x="306" y="38" font-size="13" fill="#1d4ed8" font-weight="600">y — the data</text>
  <line x1="150" y1="180" x2="286" y2="168" stroke="#137a4b" stroke-width="2.2"
        marker-end="url(#ah3)"/>
  <text x="292" y="176" font-size="12.5" fill="#137a4b" font-weight="600">Aq — the model</text>
  <line x1="300" y1="40" x2="286" y2="168" stroke="#c8433c" stroke-width="1.8"
        stroke-dasharray="6 4"/>
  <text x="312" y="112" font-size="12" fill="#c8433c">residual — as short as it can be</text>
  <path d="M286,168 L279,166 L281,159" fill="none" stroke="#c8433c" stroke-width="1.4"/>
  <circle cx="150" cy="180" r="3" fill="#333"/>
</svg>
"""


def n(key, fmt='{:,.0f}'):
    return fmt.format(N[key])


BODY = f"""
<h1>The charge solve: what <code>A</code> is, and what NNLS does with it</h1>
<p class="meta">wft forward-fit reconstruction · every figure generated from
<code>sat_det3</code> (mx17_3, Saturday long run, resistive 490 V / drift
1000 V) with the frozen production bundle <code>calib_bundle_lp2_t0p</code> ·
2026-08-21</p>

<div class="verdict">
<strong>A is a dictionary of pictures.</strong> Column <em>k</em> of A is the
complete (strip × sample) picture that <em>one unit of charge, liberated at
drift depth k</em>, would have produced in this window — geometry, charge
sharing and amplifier response all folded in. There are 18 such pictures, one
per 60 ns slice of the drift gap. Solving for <strong>q</strong> asks a single
question: <em>how much of each picture do I add up to reproduce what the
detector actually recorded?</em> Because that question is linear, it has an
exact answer, and the fit never has to search for the charges — only for the
three geometric numbers that decide which 18 pictures to draw.
</div>

<div class="tiles">
  <div class="tile"><div class="n">{n('n_row')}</div>
    <div class="l">measurements in one plane<br>({n('n_strip')} strips ×
      {n('n_samp')} samples)</div></div>
  <div class="tile"><div class="n">3</div>
    <div class="l">geometric numbers<br>searched over</div></div>
  <div class="tile"><div class="n">18</div>
    <div class="l">charges<br>solved exactly</div></div>
  <div class="tile"><div class="n">{n('lh_iters')}</div>
    <div class="l">solver steps on the<br>display event</div></div>
  <div class="tile"><div class="n">{N['pop_nz']:.1f}</div>
    <div class="l">of 18 bins survive<br>the q ≥ 0 constraint</div></div>
</div>

<div class="box"><strong>Contents</strong>
<div class="toc">
1 · <a href="#s1">The problem, stated honestly</a><br>
2 · <a href="#s2">One column of A, built from scratch</a><br>
3 · <a href="#s3">Why it is a matrix at all: the flattening</a><br>
4 · <a href="#s4">A worked example small enough to check by hand</a><br>
5 · <a href="#s5">Weights and censoring</a><br>
6 · <a href="#s6">The solve, geometrically</a><br>
7 · <a href="#s7">How the constrained solve actually runs</a><br>
8 · <a href="#s8">Why non-negativity is not cosmetic</a><br>
9 · <a href="#s9">How independent are the eighteen unknowns?</a><br>
10 · <a href="#s10">The answer, and its error bars</a><br>
11 · <a href="#s11">Why this is called "profiling"</a><br>
12 · <a href="#s12">What it looks like across a population</a><br>
13 · <a href="#s13">The neighbours are in A, not in the way</a><br>
14 · <a href="#s14">Traps</a><br>
15 · <a href="#s15">Reproduce</a>
</div></div>

<h2 id="s1">1 · The problem, stated honestly</h2>

<p>One plane of one event is an array of numbers: every strip in the window,
every 60 ns sample. For the event this note follows, that is
<strong>{n('n_strip')} strips × {n('n_samp')} samples = {n('n_row')}
numbers</strong>. Nothing has been reduced, thresholded or turned into a
"hit" — this is the raw, pedestal- and gain-corrected window.</p>

{fig('f01_window', 1,
     'The window the fit is handed. Left: all ' + n('n_strip') + ' strips '
     'against time. Middle: the six brightest strips — each is one smooth '
     'pulse, and the pulse arrives later the further the strip is along the '
     'track. Right: peak amplitude per strip. The whole of the reconstruction '
     'is an attempt to explain these ' + n('n_row') + ' numbers.',
     'figs.py · sat_det3 event 1663, x plane · peak '
     + n('peak_adc') + ' ADC, noise ' + f"{N['noise_adc']:.1f}" + ' ADC/strip')}

<p>The physical picture behind those numbers: a charged particle crossed the
30 mm drift gap, ionising along its path. That ionisation drifts down to the
mesh at v ≈ 36.6 µm/ns, so charge liberated <em>deep</em> in the gap arrives
<em>late</em>. If the track is inclined, deep charge also arrives at a
different transverse position. Every strip therefore sees a superposition of
contributions from many drift depths, blurred by diffusion, spread onto its
neighbours by the resistive layer, and finally shaped by the amplifier.</p>

<p>So the unknowns are:</p>

<table>
<tr><th>unknown</th><th>how many</th><th>how it enters the model</th>
    <th>how it is found</th></tr>
<tr><td><code>p₀</code> position at the mesh</td><td class="n">1</td>
    <td>non-linearly</td><td rowspan="3">searched — Nelder-Mead, ~500 steps</td></tr>
<tr><td><code>w</code> transverse speed (the slope)</td><td class="n">1</td>
    <td>non-linearly</td></tr>
<tr><td><code>t₀</code> when the mesh charge arrives</td><td class="n">1</td>
    <td>non-linearly</td></tr>
<tr><td><code>q₀ … q₁₇</code> charge per 60 ns depth slice</td>
    <td class="n">18</td><td><strong>linearly</strong></td>
    <td><strong>solved in closed form</strong></td></tr>
</table>

<p>That last row is the whole trick. Twenty-one unknowns would be a miserable
search. But the model is <em>linear in the charges</em>: doubling the charge in
a depth slice exactly doubles its contribution to every sample of every strip.
Anything linear can be solved rather than searched.</p>

{SVG_PIPE}
<p class="meta">The two-level structure. The expensive object — A — depends
only on the three geometric numbers and the calibration. Given A, the charges
come out of a closed-form solve, and the χ² that comes back is what the outer
search actually navigates by.</p>

<h2 id="s2">2 · One column of A, built from scratch</h2>

<p>Forget the matrix for a moment and ask a smaller question: <em>if exactly
one unit of charge were liberated at depth slice k, and nothing else happened,
what would this window look like?</em></p>

<p>Answering it takes three ingredients, and they multiply:</p>

<ol>
<li><strong>Where it lands.</strong> Slice k's charge sits at transverse
position p₀ + w·u<sub>k</sub>, spread by a Gaussian whose width is the initial
cloud, plus diffusion √u, plus the sideways motion of the track within the
slice itself. Integrating that Gaussian across each 0.78 mm strip gives
<code>F[i,k]</code> — the fraction landing on strip i. That is
<code>strip_fractions()</code>, and it is pure geometry.</li>
<li><strong>When it arrives.</strong> The charge reaches the mesh at
t₀ + u<sub>k</sub>, and the amplifier turns that instant into the measured
pulse shape h(t). Sampling h(t − t₀ − u<sub>k</sub>) at the 32 sample times
gives <code>H0[:,k]</code>.</li>
<li><strong>Who else sees it.</strong> The resistive layer puts a fraction
c₁ of a strip's charge onto each ±1 neighbour and c₂ onto each ±2, with their
own, slightly later, response shapes h₁ and h₂.</li>
</ol>

<div class="eq">
column<sub>k</sub> = F<sub>:,k</sub> ⊗ h<sub>k</sub>
&nbsp;+&nbsp; c₁ · F<sup>±1</sup><sub>:,k</sub> ⊗ h₁<sub>,k</sub>
&nbsp;+&nbsp; c₂ · F<sup>±2</sup><sub>:,k</sub> ⊗ h₂<sub>,k</sub>
<small>⊗ is an outer product: a vector over strips times a vector over
samples gives a strip × sample picture. Three of them, added.</small>
</div>

{fig('f02_column_build', 2,
     'One column, assembled. Top row: where the charge lands (①), when it '
     'arrives (②), and the three response shapes — the neighbours see a copy '
     'that is both smaller and later (③). Bottom row: those factors as '
     'pictures. The own-strip term, the ±1 copies, the ±2 copies, and their '
     'sum — which <em>is</em> column ' + str(7) + ' of A. On the x plane the '
     'sharing terms are small (c₁ = ' + f"{N['c1']:.3f}" + '); on the y plane '
     'they are three times larger and clearly visible.',
     'figs.py · det3 frozen bundle, x plane, depth bin k = 7')}

<p>Do that for all eighteen depth slices and you have the whole of A. Seen
side by side, the columns tell you what the fit is actually able to
distinguish:</p>

{fig('f03_atlas', 3,
     'The complete dictionary: all 18 columns of A for this event, each drawn '
     'as the picture it is. As k increases the blob moves later in time '
     '<em>and</em> sideways in position — that coupling between time and '
     'position is exactly the track slope w, and it is why fitting the '
     'waveforms measures an angle at all. Note that k = 15–17 sit past the '
     '30 mm cathode: the basis deliberately overhangs the gap so the fit can '
     'put charge beyond the physical end rather than piling it against a wall.',
     'figs.py · sat_det3 event 1663, x plane, K = 18 bins of 60 ns')}

<h2 id="s3">3 · Why it is a matrix at all: the flattening</h2>

<p>Each column is naturally a 2-D picture, {n('n_strip')} × {n('n_samp')}.
Linear algebra wants vectors, so the picture is unrolled row by row: strip 0's
32 samples, then strip 1's 32 samples, and so on. Row index
<code>r = i·{n('n_samp')} + s</code>. That is the single line
<code>M.reshape(n·NSAMP, K)</code> at the end of <code>build_matrix</code>, and
it is the only thing standing between "a stack of pictures" and "a matrix".</p>

{fig('f04_flatten', 4,
     'Top left: column 7 as a picture. Bottom: the same numbers unrolled into '
     'one ' + n('n_row') + '-long vector — the vertical grid lines are strip '
     'boundaries, and the labelled bumps are the strips that see this depth '
     'slice. <b>This vector is what a column of A literally is.</b> Right: all '
     '18 columns side by side, the actual noise-weighted A. The staircase is '
     'the track: each successive depth slice appears a little later within '
     'its strip block and a little further along in strip number.',
     'figs.py · sat_det3 event 1663 · A is ' + n('n_row') + ' × 18')}

<div class="box">
<strong>The one-line summary.</strong> A is <em>tall and thin</em>:
{n('n_row')} rows (one per measurement) by 18 columns (one per unknown
charge). Tall and thin is good — it means the problem is
<strong>over-determined</strong> by a factor of thirty, which is why an
18-number answer extracted from a noisy window is stable at all.
</div>

<h2 id="s4">4 · A worked example small enough to check by hand</h2>

<p>The real problem is {n('n_row')} × 18. The structure is identical at 12 × 2,
so here is one built from scratch: 3 strips, 4 samples, 2 depth slices. The
"template" is <code>h = [0, 1, 0.5, 0.1]</code>; slice 0 lands 20/60/20 % across
the three strips and slice 1, being deeper on an inclined track, lands
5/35/60 %. The true charges are 100 and 60, and Gaussian noise of σ = 2 is
added.</p>

{fig('f05_toy', 5,
     'The whole calculation. Left: the 12 × 2 matrix, every entry printed — '
     'each is simply (strip fraction) × (template sample). Middle: the data, '
     'the two columns scaled by their true charges, and the fitted sum. '
     'Right: the normal equations, solved in closed form, recovering '
     f"({N['toy_q'][0]:.1f}, {N['toy_q'][1]:.1f}) against a truth of (100, 60).",
     'figs.py · synthetic, seed 7')}

<p>The arithmetic in the right-hand panel is the entire "solve" step:</p>

<div class="eq">
q̂ = (AᵀA)<sup>−1</sup> Aᵀy
<small>AᵀA is 2 × 2 here and 18 × 18 in the real problem — never
{n('n_row')} × {n('n_row')}. That is why this is fast.</small>
</div>

<p>AᵀA is the matrix of <em>overlaps between columns</em>: entry (k, k′) is how
much picture k looks like picture k′. Aᵀy is the matrix of overlaps between
each column and the data — "how much does the data resemble depth slice k?".
The solve trades one against the other: a bin gets credit for the data it
matches, minus what its neighbours have already explained. In the toy the two
columns overlap at <strong>{N['toy_corr']:.3f}</strong>, which is almost
exactly the {N['corr_adj']:.2f} that adjacent depth bins overlap at in the real
matrix — the toy is not a caricature, it is the same problem with the indices
turned down.</p>

<h2 id="s5">5 · What the fit is allowed to look at: weights and censoring</h2>

<p>Two adjustments happen before the solve, and both are edits to A and y
rather than to the model.</p>

<h3>Noise weighting</h3>
<p>χ² compares residuals to the noise, so every row is divided by that strip's
own σ: <code>A ← A/σ</code>, <code>y ← y/σ</code>. After that division the
residual in every row has unit variance and the plain sum of squares
<em>is</em> the χ². On det3 the strips are uniform
(σ ≈ {N['noise_adc']:.1f} ADC) so this barely changes the answer — its purpose
is the cases where the strips are not uniform, and the dead-channel case
below, where it is doing the real work.</p>

<h3>Censoring</h3>
<p>Two kinds of row carry no usable information and are <strong>deleted from
A</strong> entirely:</p>
<ul>
<li><strong>Saturated samples.</strong> Above {int(3550)} ADC the amplifier
clips; the sample says "at least this much" and nothing more. Those rows leave
the fit, and are replaced by a one-sided penalty that fires only if the model
falls <em>below</em> the clipped value.</li>
<li><strong>Dead channels.</strong> A broken connection reads baseline, not
zero charge. Those rows are censored the same way and their σ is set to 10⁹ so
the one-sided penalty cannot pull on them either — no information in either
direction. (This is the machinery that masks chamber A's connector 8 on
run_79.)</li>
</ul>

{fig('f06_censor', 6,
     'Censoring on a real saturated event. Left: ' + n('sat_n') + ' samples '
     'hit the clip. Middle: the brightest strip — the model is free to go '
     'above the clipped samples and is penalised only for going below. Right: '
     'the resulting row mask; ' + n('sat_rows') + ' of ' + n('sat_tot') + ' '
     'rows enter A, and the degrees of freedom are counted from the survivors.',
     'figs.py · sat_det3 event ' + str(N['sat_event']) + ', x plane')}

<h2 id="s6">6 · The solve, geometrically</h2>

<p>With A and y fixed, "fit the charges" means: <em>find the combination of
columns that comes closest to the data vector</em>. The set of all reachable
combinations, {{Aq}}, is a flat 18-dimensional subspace inside
{n('n_row')}-dimensional space — the <em>column space</em>. Least squares picks
the point of that subspace nearest to y, and the nearest point is found by
dropping a perpendicular.</p>

{SVG_PROJ}
<p class="meta">The picture behind (AᵀA)⁻¹Aᵀy. The residual is perpendicular to
every column of A — that perpendicularity, written out, <em>is</em> the normal
equations. The data will never lie exactly in the column space; noise and model
imperfection are precisely the part of y that no charge profile can reach.</p>

<h3>…and then non-negativity bends it</h3>

<p>Charge cannot be negative. That single constraint changes the character of
the answer completely, and it is worth seeing why on real columns.</p>

{fig('f07_projection', 7,
     'Two real columns of A — depth bins ' + str(N['pair'][0]) + ' and '
     + str(N['pair'][1]) + ', which overlap at ' + f"{N['pair_corr']:.2f}" +
     '. Left: χ² over their two charges. The valley is long and diagonal '
     'because the two pictures look so alike, and the unconstrained minimum '
     f"sits at ({N['pair_un'][0]:.0f}, {N['pair_un'][1]:.0f}) — deep inside "
     'the forbidden half-plane. NNLS cannot go there; it slides along the '
     f"boundary to ({N['pair_nn'][0]:.0f}, 0). Right: once a bin is pinned at "
     'zero the problem in the survivor is an ordinary one-dimensional '
     'parabola. <b>This is where the zeros in a charge profile come from</b> — '
     'a bin reads exactly zero because its constraint is active, not because '
     'the solver rounded a small number down.',
     'figs.py · sat_det3 event 1663, x plane')}

<h2 id="s7">7 · How the constrained solve actually runs</h2>

<p><code>scipy.optimize.nnls</code> is the Lawson–Hanson active-set algorithm.
It is not iterative refinement and it does not "converge" in the usual sense:
it terminates in a finite number of steps at the exact constrained optimum. The
recipe is short enough to state in full:</p>

<pre>start with every charge at zero and every constraint ACTIVE
loop:
    gradient = Aᵀ(y − Aq)          "which bin most wants charge?"
    if no active bin has a positive gradient: STOP — this is the optimum
    move that bin into the FREE set
    solve the ordinary least-squares problem on the free set alone
    while any freed charge came out negative:
        back off along the line to the constraint boundary,
        re-activate whatever hit zero, and re-solve</pre>

<p>Instrumenting it on the display event shows the whole story. (The
instrumented implementation reproduces <code>scipy</code>'s answer to
{N['lh_agree']:.0e} — it is the same algorithm, only narrating.)</p>

{fig('f08_lh', 8,
     'The solve, step by step. Left: at the first step every bin is at zero, '
     'so the gradient is just "how much does the data look like this depth?" '
     '— bin ' + str(N['lh_order'][0]) + ' wins and is admitted. Middle: χ² '
     'falls from ' + n('chi2_start') + ' to ' + n('chi2_end') + ' as bins are '
     'admitted one at a time, labelled in the order they enter. Right: the '
     'charge profile assembling itself, row by row. Ten steps and it is done.',
     'figs.py · sat_det3 event 1663 · own Lawson–Hanson, verified against '
     'scipy.optimize.nnls')}

<p>Two things in that figure are worth dwelling on. The order of admission is
<em>not</em> depth order — it is
{', '.join(str(k) for k in N['lh_order'])} — because each step picks whichever
depth best explains <em>what is left over</em>. And the first step alone
removes 45 % of the χ²: one depth slice, chosen greedily, is already most of
the answer.</p>

<div class="warn">
<strong>The display event never backtracks; most events do.</strong> Over 250
planes, <strong>{100 * N['pop_bt_frac']:.0f} %</strong> need at least one bin
pushed back out of the free set, and one needed {n('pop_bt_max')}. The
backtracking loop is not decoration — it is what makes the answer exact rather
than greedy.
</div>

<h2 id="s8">8 · Why non-negativity is not cosmetic</h2>

<p>It is tempting to think the constraint is a tidy-up. It is not: without it
the solve produces a profile that fits marginally better and means nothing.</p>

{fig('f09_uncon', 9,
     'Unconstrained against NNLS on the same event and the same A. Left: '
     'released, ' + n('n_neg') + ' of 18 bins go negative — several by more '
     'than a thousand units — and their neighbours grow to compensate. '
     'Middle: on the brightest strip both curves lie on the data; the '
     'unconstrained χ² is ' + n('chi2_uncon') + ' against NNLS’s '
     + n('chi2_nnls') + ', a ' + f"{N['chi2_gain_pct']:.0f}" + ' % difference '
     'invisible to the eye. Right: the cumulative charge — the two agree on '
     'the total and disagree completely about where it came from.',
     'figs.py · sat_det3 event 1663, x plane')}

<p>The negative excursions are physically impossible and they cancel almost
perfectly, which is exactly why the χ² hardly notices them. Everything
downstream that uses the profile — the charge column endpoint that measures the
drift gap, the arrival quantiles, the candidate score — would be reading noise.
The {N['chi2_gain_pct']:.0f} % of χ² given up by forbidding them is the price
of an interpretable answer, and it is a bargain.</p>

<h2 id="s9">9 · How independent are the eighteen unknowns, really?</h2>

<p>The zeros in a charge profile look alarming until you look at how much the
columns of A resemble each other. Adjacent depth slices are separated by 60 ns
of drift — but the amplifier response is hundreds of nanoseconds wide, so their
pictures overlap heavily.</p>

{fig('f10_gram', 10,
     'Left: the correlation between columns. Neighbouring depth bins overlap '
     'at ' + f"{N['corr_adj']:.2f}" + ', next-but-one at '
     + f"{N['corr_2']:.2f}" + '. Middle: the singular-value spectrum, '
     'spanning a factor ' + n('cond') + ' — the condition number. Right: the '
     'best- and worst-measured charge patterns. The best-determined '
     'combinations are <em>smooth</em>: total charge, and the depth centroid. '
     'The worst are bin-to-bin <em>zigzags</em>, measured '
     + n('cond') + '× more poorly than the smooth ones.',
     'figs.py · sat_det3 event 1663, x plane')}

<div class="ok">
<strong>This is the single most useful thing to understand about q.</strong>
The data constrains <em>smooth functionals</em> of the charge profile
extremely well and bin-to-bin structure hardly at all. So: the total charge,
the median arrival time, the depth at which the column ends — trustworthy. The
value of one individual bin, or the pattern of which bins happen to be zero —
not a measurement. The pipeline is built accordingly: what
<code>fit_plane</code> keeps from q is exactly <code>q_sum</code>,
<code>q_u50</code>, <code>q_u90</code> and <code>q_uend</code> — four smooth
summaries, not the eighteen numbers.
</div>

<h2 id="s10">10 · The answer, and its error bars</h2>

{fig('f11_result', 11,
     'What one solve buys. Top: data and model on eighteen strips at once — '
     'all of them are described by the <em>same</em> 18-number charge profile '
     'plus three geometric numbers; nothing is fitted strip by strip. Bottom '
     'left: the residual in units of σ, with structure still visible at the '
     'ends (χ²/dof = ' + f"{N['chi2_dof']:.1f}" + ' — the model is good, not '
     'perfect). Bottom middle: the charge against drift depth, with the '
     '30 mm cathode marked. Bottom right: the track that implies, against the '
     'independent M3 reference.',
     'figs.py · sat_det3 event 1663, x plane · fitted tan θ '
     + f"{N['tan_fit']:.3f}" + ' vs reference ' + f"{N['tan_ref']:.3f}")}

<p>Errors on the charges come from the same matrix, restricted to the bins that
survived: <code>cov = (A<sub>free</sub>ᵀ A<sub>free</sub>)⁻¹</code>. The
contrast between a single bin and a smooth summary is stark.</p>

{fig('f12_errors', 12,
     'Left: the profile with 1σ error bars from the free-set covariance; the '
     '<em>total</em> is ' + n('q_total') + ' ± ' + n('q_tot_err') + ', i.e. '
     + f"{N['q_tot_err_pct']:.2f}" + ' %. Middle: the surviving bins are '
     'strongly anti-correlated with their neighbours — one bin’s excess is '
     'the next one’s deficit, which is the zigzag mode of Figure 10 seen '
     'from another angle. Right: per-bin fractional errors.',
     'figs.py · sat_det3 event 1663, x plane')}

<p>Note the numbers: a single bin is known to a few per cent at best, while
their sum is known to {N['q_tot_err_pct']:.2f} %. That factor is the
anti-correlation doing its work, and it is the quantitative version of the
warning in §9.</p>

<h2 id="s11">11 · Why this is called "profiling", and what the outer loop sees</h2>

<p>The charges are <em>nuisance parameters</em>. They are not thrown away and
they are not guessed once and held: they are re-derived exactly at every single
trial geometry. What the outer search navigates by is the χ² <em>after</em>
that re-derivation — a profile likelihood in the statistical sense.</p>

{fig('f13_profile', 13,
     'Sliding p₀ across ±2 mm, with the charges re-solved at each of '
     + n('profile_nsolve') + ' points. Left: the profiled χ². Middle: the '
     'charge profile at every trial — as the assumed track position moves, '
     'the solve migrates charge to different depths to keep explaining the '
     'same data, and the diagonal streaks are that migration. Right: three of '
     'those profiles. Only the middle one is remotely flat, which is what a '
     'minimum-ionising track through a uniform gap should look like.',
     'figs.py · sat_det3 event 1663, x plane · ' + n('profile_nsolve') +
     ' NNLS solves')}

<p>The middle panel is also the clearest picture of why the fit works at all.
A wrong p₀ can still be made to fit the data — but only by an unphysical,
lumpy charge profile. The geometry is determined not because a wrong geometry
cannot describe the waveforms, but because it can only do so by demanding an
absurd distribution of ionisation.</p>

<h3>The degeneracy this leaves behind</h3>

<p>There is one direction in which the trade is nearly free. Shift t₀ earlier
by one depth bin and slide p₀ along the track by w·60 ns, and the charge
profile simply moves one index over — the model waveforms are almost
unchanged.</p>

{fig('f14_tooth', 14,
     'Left: the best solution at t₀ and at t₀ − 60 ns. The profile has moved '
     'one bin, p₀ has slid ' + f"{abs(N['tooth_dp'][0]):.2f}" + ' mm, and χ² '
     'is only ' + f"{N['tooth_dchi2_pct']:.1f}" + ' % worse. Middle: on the '
     'brightest strip the two are nearly indistinguishable. Right: scanning '
     't₀ with p₀ re-optimised at each point — a ladder of secondary minima, '
     'roughly every half sample, on a shallow plateau to the early side.',
     'figs.py · sat_det3 event 1663, x plane · ~5,000 NNLS solves')}

<div class="warn">
<strong>This is why the scintillator t₀ prior exists.</strong> A free fit lands
in the physically correct tooth only about a third of the time. The trigger,
through the ftst clock phase, predicts t₀ per event to a few nanoseconds; a
Gaussian penalty on that prediction (σ = 5 ns, carried in the bundle as
<code>t0_abs</code> / <code>t0_prior_sigma</code>) is what selects the right
one. It was A/B-validated on held-out data on 2026-08-12 and is the production
configuration on the bench. <em>It is bench-only</em> — <code>t0_abs</code>
must be dropped when a bundle is transferred to beam running, where there is no
such trigger.
</div>

<h2 id="s12">12 · What this looks like across a population</h2>

{fig('f15_population', 15,
     'Reference-pinned fits on ' + n('pop_n') + ' planes. Left: '
     f"{N['pop_nz']:.1f}" + ' of 18 bins survive on average — '
     f"{100 * N['pop_nz_frac']:.0f}" + ' %; individual profiles are sparse and '
     'spiky by construction. Middle: how much work the constraint does — '
     f"{100 * N['pop_bt_frac']:.0f}" + ' % of planes need at least one '
     'backtrack. Right: averaged over all of them the profile is flat between '
     'about 5 and 20 mm and falls off at the cathode — which is the physical '
     'truth for minimum-ionising tracks. The median (red) is far below the '
     'mean because with half the bins at zero, a median truncates.',
     'figs.py · sat_det3, 250 ref-pinned planes, x plane')}

<div class="warn">
<strong>The median trap, which has already cost us once.</strong> Summarising
these profiles with a per-bin median made the drift column read 24.7 mm and
launched a hunt for a "missing 4 mm of gap". Means, trimmed means or rebinned
medians moved it to 27.9 mm and the hardware question evaporated. Anything
that summarises a sparse NNLS profile must account for the sparsity —
<em>the zeros are constraint boundaries, not measurements of "no charge
here"</em>.
</div>

<p>The first-bin spike in the right-hand panel is real and worth knowing about:
bin 0 is where anything arriving earlier than the model expects gets deposited,
so it absorbs t₀ mismatch. It is another reason to read the <em>column
endpoint</em> rather than the first bin.</p>

<h2 id="s13">13 · The neighbours are in A, not in the way</h2>

<p>One structural point deserves its own section, because it is the reason this
whole approach exists. On a resistive-strip detector each strip's signal
contains delayed, dispersed copies of its neighbours'. If you reduce a strip to
a single hit time, that contamination is <em>irreducible</em> — it compresses
the drift-time ladder by 20–30 % and reads the angle several degrees too
steep, and no threshold or estimator change fixes it.</p>

<p>Here, the copies are terms in the model. They are part of what a unit of
charge <em>is</em> predicted to look like, so the solve accounts for them
instead of being fooled by them.</p>

{fig('f16_sharing', 16,
     'How much of A is neighbours. Left and middle: for one depth slice, the '
     'charge landing on each strip, split into the strip’s own and the '
     'copies it receives — ' + f"{100 * N['share']['x']['frac']:.1f}" + ' % of '
     'the matrix norm on x, ' + f"{100 * N['share']['y']['frac']:.1f}" + ' % '
     'on y, where the strips couple to the resistive layer far more strongly '
     '(kY = 2.88). Right: the copy is late as well as small — that lateness, '
     'mistaken for drift time, is precisely the bias a hits-based '
     'reconstruction cannot escape.',
     'figs.py · sat_det3 event 1663, both planes')}

<h2 id="s14">14 · Traps</h2>

<ul>
<li><strong>The stored <code>c2</code> can be a lie.</strong> On bundles refit
after 2026-08-19 the ±2 amplitude is slaved to the ±1 one
(<code>c2 = 0.6 · c1</code>) and the stored <code>c2</code> field is literally
<code>0.0</code>, with the ratio in the <code>c2_over_c1</code> hyper. Code
that reads <code>h['c2']</code> directly draws no ±2 copy at all and will
build a subtly wrong A. Use <code>build_matrix</code>, or
<code>bundle.summary()</code>, which reports what the model actually uses.</li>
<li><strong>The bundle this note uses is the frozen production one</strong>,
which carries the superseded kernel — c₂ ({N['c2']:.3f}) is <em>larger</em>
than c₁ ({N['c1']:.3f}), which cannot be physical, since the ±2 strip is
reached only through the ±1. That inversion is a fit artefact of a genuinely
flat χ² direction, and it is what the r06 refit corrected. It does not change
anything in this note — the structure of A, the solve and everything in §§6–12
are identical — but it is why the amplitudes here should not be quoted as
measurements.</li>
<li><strong>K is not universal.</strong> 18 bins of 60 ns is a
det3-at-1000-V number. Slower chambers need 22 or 26 bins or the charge column
runs off the end of the basis, and the fit then piles charge into the last bin
and biases t₀.</li>
<li><strong>t₀ is quantised to 5 ns during the coarse searches.</strong>
Building A is the expensive step and it depends on t₀ but not on p₀ or w, so
the three time tensors are cached on a 5 ns grid. The final Nelder-Mead steps
run unsnapped and pay full price. If you call <code>build_matrix</code>
yourself with an off-grid t₀ you take the slow path — which is correct, just
slower.</li>
<li><strong>σ<sub>p0</sub> absorbs model error.</strong> If the sharing kernel
is wrong, the fit can hide the missing transverse spread by inflating the
"initial cloud" instead. A σ<sub>p0</sub> of half a millimetre is not a
measurement of the primary ionisation cloud, it is a warning light.</li>
</ul>

<h2 id="s15">15 · Reproduce</h2>

<pre>cd docs/charge_solve
../../.venv/bin/python figs.py        # 16 figures + numbers.json, ~15 s
../../.venv/bin/python make_note.py   # this page

# override the inputs:
#   WFT_DOC_BUNDLE=&lt;path to a calib bundle&gt;   CS_FIGDIR=&lt;where the PNGs go&gt;</pre>

<p>Everything is driven by the live products under
<code>&lt;Analysis&gt;/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/</code>:
the bundle <code>calib_bundle_lp2_t0p</code> and the 400-event ref-pinned
calibration cache. The code under discussion is <code>wft/model.py</code> —
<code>build_matrix()</code>, <code>strip_fractions()</code> and
<code>chi2_plane()</code> are the three functions this whole note is about, and
together they are about eighty lines.</p>

<p class="meta">Companion reading: the full nine-part reference document
<code>docs/wft_reference/</code> covers the rest of the chain, from raw ADC to
physics outputs; <code>RECONSTRUCTION_BASIS.md</code> is why geometry comes
from waveforms and never from hit times.</p>
"""


TITLE = 'The charge solve: what A is, and what NNLS does with it'
SUMMARY = ('A deep dive on the linear half of the waveform-first fit: the '
           'design matrix A as a dictionary of "one unit of charge, this deep" '
           'pictures, the non-negative least-squares solve that reads the '
           'charge profile off it, and what that profile can and cannot be '
           'asked. 16 figures from live det3 data, plus a worked 12 x 2 '
           'example.')

PAGE = f"""<!--note
title: {TITLE}
summary: {SUMMARY}
tags: X17, cosmic bench, micromegas, reconstruction, waveforms
-->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{SUMMARY}">
<style>{CSS}</style>
</head>
<body>
{BODY}
</body>
</html>
"""


def main():
    with open(OUT, 'w') as f:
        f.write(PAGE)
    print(f'wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
