#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- build report.html from the figure set and the geometry module.

    ../.venv/bin/python make_report.py

Everything in the report (the geometry tables, the provenance list, the
assumptions, the figure inventory) is read from ``geometry.py`` and from what
is actually present in ``figures/``, so re-running after a re-render keeps the
numbers, the pictures and the caveats in step.  Figures are referenced with
relative links so the same file works from disk and through the DAQ web page's
``/analysis_file/<relpath>`` route.
"""
from __future__ import annotations

import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G          # noqa: E402
import scenes_x17 as X        # noqa: E402
from make_figures import FIGURES   # noqa: E402

FIG = os.path.join(HERE, 'figures')
OUT = os.path.join(HERE, 'report.html')

ANIM = os.path.join(HERE, 'animations')

ANIM_BLURB = {
    'turn_sps': 'Turntable of the H4 telescope.',
    'turn_bench': 'Turntable of the bench, both slots MX17.',
    'turn_bench_p2': 'Turntable of the bench, both slots P2 BASKET.',
    'turn_chamber': 'Turntable of the exploded chamber.',
    'build_sps': 'Build-up: table, then the uRWELL references, the three P2 '
                 'fans, and finally the beam.',
    'build_bench': 'Build-up: the rack, then the trigger paddles, the M3 '
                   'reference planes, the chambers under test, and finally '
                   'the muons.',
}

BLURB = {
    'x17_signature': 'The physics case in one figure: capture leaves '
                     '4He* with 20.58 MeV, three channels take it away, and '
                     'only one of them puts an e+e- pair at a large opening '
                     'angle.  The opening-angle curve is computed, not traced.',
    'x17_signature_bare': 'The same diagram without the title and caption '
                          'bands, cropped -- for a slide whose own title bar '
                          'already says it.',
    'chamber_exploded': 'One MX17 chamber with its layers separated along the '
                        'drift axis, and a muon whose ionisation drifts down '
                        'to the mesh -- the micro-TPC picture the whole '
                        'reconstruction rests on.',
    'sps_hero': 'The hero shot: three P2 BASKET fans between the two EIC '
                'uRWELL references, viewed from downstream-left so every '
                'readout face is turned towards the camera.',
    'sps_hero_mx17': 'The same view with MX17 "Detector E" added at '
                     'z = 1155 mm.  Its z is a placeholder in the run config, '
                     'and at 47 cm square it dominates the rail.',
    'sps_side': 'Near-elevation: the figure to use when the point is the '
                'spacing along the rail.',
    'sps_beam': "Beam's-eye view from downstream -- the stations nested "
                'inside one another, and the pad structure of the fan at its '
                'clearest.',
    'bench_hero': 'The bench as a rack: both test slots filled with MX17 '
                  'chambers, drift volumes showing.',
    'bench_side': 'Near-elevation with a slight lift, so every plane in the '
                  'stack is visible at once.  The stacking figure.',
    'bench_p2': 'The same bench with both slots carrying P2 BASKET fans.',
    'bench_p2_side': 'Elevation of the two-P2 configuration.',
    'bench_mixed': 'Mixed slots: a P2 BASKET fan lying flat in P1 and an MX17 '
                   'in P2 -- the configuration '
                   'mx17_det3_p2_det1_overnight_6-27-26 actually ran.  Not part '
                   'of the headline set, but it renders from the same code.',
}

CSS = """
:root { --ink:#16202b; --muted:#5d6874; --rule:#e3e8ee; --bg:#ffffff;
        --card:#f7f9fb; --accent:#1f6f8b; }
* { box-sizing:border-box; }
body { margin:0; padding:0 0 6rem; background:var(--bg); color:var(--ink);
       font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
            "Helvetica Neue",Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:0 1.5rem; }
header { border-bottom:1px solid var(--rule); padding:3rem 0 1.6rem;
         margin-bottom:2rem; }
h1 { font-size:2.0rem; margin:0 0 .4rem; letter-spacing:-.01em; }
h2 { font-size:1.32rem; margin:2.6rem 0 .8rem; padding-top:1.4rem;
     border-top:1px solid var(--rule); }
h3 { font-size:1.05rem; margin:1.8rem 0 .4rem; }
.lede { color:var(--muted); font-size:1.07rem; margin:0; }
.verdict { background:var(--card); border-left:4px solid var(--accent);
           padding:1rem 1.2rem; border-radius:0 6px 6px 0; margin:1.4rem 0; }
table { border-collapse:collapse; width:100%; margin:1rem 0 1.6rem;
        font-size:.94rem; }
th,td { text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--rule); }
th { font-weight:600; color:var(--muted); font-size:.82rem;
     text-transform:uppercase; letter-spacing:.04em; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
figure { margin:2rem 0; }
figure img { width:100%; height:auto; border:1px solid var(--rule);
             border-radius:6px; display:block; }
figcaption { color:var(--muted); font-size:.9rem; margin-top:.6rem; }
figcaption b { color:var(--ink); }
code { background:var(--card); padding:.12em .38em; border-radius:3px;
       font-size:.9em; }
pre { background:var(--card); padding:1rem; border-radius:6px; overflow-x:auto;
      font-size:.86rem; }
ul.tight li { margin:.3rem 0; }
.caveat { color:var(--muted); font-size:.94rem; }
.tag { display:inline-block; background:var(--card); border:1px solid var(--rule);
       border-radius:99px; padding:.1rem .6rem; font-size:.78rem;
       color:var(--muted); margin-right:.35rem; }
.scroll { overflow-x:auto; }
"""


def esc(s):
    return html.escape(str(s))


def fig_block(name, theme='light'):
    """One figure with its links.

    The 3-D figures come as a bare render plus a separately composed
    ``_labelled`` version; the X17 diagram is drawn with its type already in
    it, so there is nothing to compose and the plain file *is* the deliverable.
    Both layouts are handled here so the report does not need to know which is
    which.
    """
    base = f'figures/{name}_{theme}_labelled'
    plain = f'figures/{name}_{theme}.png'
    if os.path.exists(os.path.join(HERE, base + '.png')):
        links = [f'<a href="{base}.png">labelled PNG</a>',
                 f'<a href="{base}.pdf">labelled PDF</a>']
        if os.path.exists(os.path.join(HERE, plain)):
            links.append(f'<a href="{plain}">bare render</a>')
    elif os.path.exists(os.path.join(HERE, plain)):
        base = f'figures/{name}_{theme}'
        links = [f'<a href="{base}.png">PNG</a>',
                 f'<a href="{base}.pdf">vector PDF</a>']
    else:
        return ''
    return (f'<figure id="{esc(name)}">\n'
            f'  <img src="{base}.png" alt="{esc(name)}">\n'
            f'  <figcaption><b>{esc(name)}</b> &mdash; '
            f'{esc(BLURB.get(name, ""))}<br>{" &middot; ".join(links)}'
            f'</figcaption>\n</figure>\n')


def sps_table():
    rows = ''
    for n, z, kind, lab, yaw in G.SPS_STATIONS:
        note = 'placeholder z; yawed %.2f deg' % yaw if kind == 'mx17' else ''
        size = {'urwell': f'{G.URW_ACTIVE_MM:.1f} mm square active',
                'p2': 'fan, r 150.7-635.0 mm, 55.6 deg, 1280 pads',
                'mx17': f'{G.MX17_ACTIVE_MM:.1f} mm square active'}[kind]
        rows += (f'<tr><td>{esc(n)}</td><td class="num">{z:g}</td>'
                 f'<td>{esc(size)}</td><td class="caveat">{esc(note)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>detector</th>'
            '<th class="num">z [mm]</th><th>size</th><th>note</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def bench_table():
    planes = [('trigger scintillator (top)', G.BENCH_SCINT_Z['top'],
               '600 x 600 mm', 'not in any run config'),
              ('m3_top_top', G.BENCH_M3_Z['m3_top_top'], '500 x 500 mm', ''),
              ('m3_top_bot', G.BENCH_M3_Z['m3_top_bot'], '500 x 500 mm', ''),
              ('P2 test slot', G.BENCH_DUT_Z['P2'], 'MX17 400 x 400 mm or P2 fan',
               f'level p2_z = {G.BENCH_P2_Z:g} mm + board'),
              ('P1 test slot', G.BENCH_DUT_Z['P1'], 'MX17 400 x 400 mm or P2 fan',
               f'level p1_z = {G.BENCH_P1_Z:g} mm + board'),
              ('m3_bot_top', G.BENCH_M3_Z['m3_bot_top'], '500 x 500 mm', ''),
              ('m3_bot_bot', G.BENCH_M3_Z['m3_bot_bot'], '500 x 500 mm', ''),
              ('trigger scintillator (bottom)', G.BENCH_SCINT_Z['bottom'],
               '600 x 600 mm', 'not in any run config')]
    rows = ''.join(f'<tr><td>{esc(a)}</td><td class="num">{b:g}</td>'
                   f'<td>{esc(c)}</td><td class="caveat">{esc(d)}</td></tr>'
                   for a, b, c, d in planes)
    return ('<div class="scroll"><table><thead><tr><th>plane</th>'
            '<th class="num">z [mm]</th><th>size</th><th>note</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def anim_block(name):
    """One animation, as an inline video with a GIF/still fallback."""
    mp4 = f'animations/{name}.mp4'
    if not os.path.exists(os.path.join(HERE, mp4)):
        return ''
    gif = f'animations/{name}.gif'
    links = [f'<a href="{mp4}">MP4</a>']
    if os.path.exists(os.path.join(HERE, gif)):
        links.append(f'<a href="{gif}">GIF</a>')
    stills = sorted(f for f in os.listdir(ANIM)
                    if f.startswith(name + '_') and f.endswith('.png'))
    if stills:
        links.append(f'{len(stills)} numbered stills in <code>animations/</code>')
    return (f'<figure>\n  <video src="{mp4}" controls loop muted playsinline '
            f'style="width:100%;border:1px solid var(--rule);border-radius:6px">'
            f'</video>\n'
            f'  <figcaption><b>{esc(name)}</b> &mdash; '
            f'{esc(ANIM_BLURB.get(name, ""))}<br>{" &middot; ".join(links)}'
            f'</figcaption>\n</figure>\n')


def build(theme='light'):
    sps = [n for n, s in FIGURES.items() if s['kind'] == 'sps']
    bench = [n for n, s in FIGURES.items() if s['kind'] == 'bench']

    assumptions = ''.join(
        f'<li><b>{esc(k.replace("_", " "))}</b> &mdash; {esc(v)}</li>'
        for k, v in G.ASSUMPTIONS.items())

    body = f"""
<header><div class="wrap">
  <h1>MPGD conference visuals &mdash; setup renderings</h1>
  <p class="lede">Publication-grade 3-D views of the two test setups: the SPS
  H4 beam telescope in P2, and the Saclay cosmic bench.</p>
</div></header>
<div class="wrap">

<div class="verdict">
  <b>What this is.</b> Two rebuildable 3-D scenes, driven entirely from the run
  configs and the measured records &mdash; not from a drawing package. Every
  detector position, the P2 fan's exact shape and all 1280 of its pads, the
  measured beam spot, the muon angular acceptance and the trigger aperture come
  from files in this repository or on lxplus. Anything <i>not</i> measured is
  listed under &ldquo;What is drawn but not measured&rdquo; below and is stated
  in each figure's caption.
</div>

<p>
  <span class="tag">PyVista / VTK</span>
  <span class="tag">PBR metals + Phong dielectrics</span>
  <span class="tag">analytic cast shadows</span>
  <span class="tag">SSAA</span>
  <span class="tag">PNG + vector-text PDF</span>
</p>

<h2>The physics case</h2>
<p>The one diagram in this package that is not a render.  Neutron capture on
   <sup>3</sup>He leaves the compound nucleus with
   {X.X17['e_capture']:g} MeV of excitation, and the question the experiment
   asks is <i>how it gets rid of it</i>: a photon, a conventional internal
   pair conversion pair at small opening angle, or &mdash; if the ATOMKI
   anomaly is real &mdash; a {X.X17['m_x17']:g} MeV boson whose
   e<sup>+</sup>e<sup>-</sup> pair cannot open by less than
   <b>{X.opening_angle_pdf()[2]:.0f}&deg;</b>.</p>
<p>That number is not taken from anyone's plot: the curve in panel 3 is exact
   two-body decay kinematics evaluated in <code>scenes_x17.py</code> (isotropic
   in the boson rest frame, boosted to the lab, Gaussian-smeared by
   {X.X17['smear_deg']:g}&deg; so the Jacobian divergence has a width on
   paper), with nuclear recoil neglected.  The internal-pair-conversion curve
   next to it is a <i>shape</i> and is labelled as one in the figure &mdash; it
   is there to say where the known channel lives, not to predict a rate.  The
   two curves are each normalised to unit peak, so nothing in the panel implies
   a branching ratio.</p>
{fig_block('x17_signature', theme)}
{fig_block('x17_signature_bare', theme)}

<h2>SPS H4 beam telescope</h2>
<p>Six stations on one rail in the P2 zone, from <code>run_59</code>'s
   <code>det_center_coords</code>. The beam runs along +Z; heights are above the
   mechanical table top.</p>
{sps_table()}
<p>The P2 BASKET chambers are not drawn as rectangles: the fan is the annulus
   sector back-solved from the group's own Gerber-derived pad map (apex at pad
   ({G.P2_APEX_PAD[0]}, {G.P2_APEX_PAD[1]}) mm, bisector
   {G.P2_BISECTOR_DEG}&deg;), mounted bisector-vertical with the apex
   {G.P2_APEX_HEIGHT:g} mm above the table, and all 1280 pads are placed as
   their true rotated rectangles. Only sectors
   {G.P2_INSTRUMENTED_SECTORS[0]}&ndash;{G.P2_INSTRUMENTED_SECTORS[1]} of 10 are
   read out, and they are drawn brighter than the rest.</p>
<p>On <b>P2 MID</b> the pads carry the <b>measured</b> beam illumination
   (stage-22 <code>n_tag</code>, 15.1 M tagged tracks summed over the ten
   <code>eff_nominal_1</code> sub-runs). Pushed through the mounting transform it
   lands at {G.sps_beam_centre_lab()[1]:.0f} mm above the table with
   &sigma;<sub>h</sub> = {G.SPS_BEAM_SIGMA_H} mm &mdash; the documented numbers,
   reproduced by the scene rather than asserted by it.</p>
<p>Beam particles are drawn parallel (measured divergence
   &lt; {G.SPS_BEAM_DIVERGENCE_MRAD} mrad over 620 mm), uniform in height across
   the {G.SPS_TRIGGER_SLAB[0]:g}&ndash;{G.SPS_TRIGGER_SLAB[1]:g} mm trigger slab
   (that hard-edged 125 mm band is the external scintillator aperture, not the
   beam) and Gaussian horizontally.</p>
{''.join(fig_block(n, theme) for n in sps)}

<h2>Saclay cosmic test bench</h2>
<p>Four M3 reference Micromegas bracketing two test slots, from
   <code>bench_geometry</code> and <code>detectors</code> in
   <code>mx17_det2_det3_overnight_6-22-26/run_config.json</code>.</p>
{bench_table()}
<p>Either slot takes an MX17 chamber (drawn with its {G.MX17_DRIFT_GAP_MM:g} mm
   drift volume, the one real thickness in the scene) or a P2 BASKET fan lying
   flat; <code>--slots p2,mx17</code> reproduces the mixed configuration that
   <code>mx17_det3_p2_det1_overnight_6-27-26</code> ran. The uprights carry
   brackets at every rail level
   (<code>bottom_level_z</code> = {G.BENCH_BOTTOM_LEVEL_Z:g} mm, then every
   {G.BENCH_LEVEL_SPACING:g} mm), which is what makes P1 and P2 read as levels
   rather than free heights.</p>
<p>Muons are sampled from the sea-level cos<sup>2</sup>&theta; distribution and
   kept only if they cross <i>both</i> 60 &times; 60 cm paddles &mdash; the
   bench's actual trigger. Over the 1.53 m paddle separation that leaves nothing
   steeper than about 15&deg;, so the drawn tracks are near-vertical because the
   acceptance says so, not because they were drawn that way.</p>
{''.join(fig_block(n, theme) for n in bench)}

<h2>One chamber, exploded</h2>
<p>The layer stack of a single MX17: 512 readout strips per view at the strip
   map's own {(G.MX17_ACTIVE_MM / 512):.4f} mm pitch, crossed X and Y, the
   resistive strips over them, the micromesh
   {150:.0f} &micro;m above (garfield_sim/mm_config.py), and the
   {G.MX17_DRIFT_GAP_MM:g} mm drift volume with a muon in it.  Each primary
   cluster drifts straight down to the mesh, so its arrival time measures its
   depth &mdash; which is the micro-TPC measurement the whole reconstruction
   rests on.</p>
{fig_block('chamber_exploded', theme)}

<h2>Animations</h2>
<p>Turntables for the talk, and build-up sequences whose numbered stills can be
   dropped on successive slides so the setup assembles itself as you speak.
   Every frame comes from the same scene code as the stills above.</p>
{''.join(anim_block(n) for n in
         ['turn_sps', 'turn_bench', 'turn_bench_p2', 'turn_chamber',
          'build_sps', 'build_bench'])}

<h2>What is drawn but not measured</h2>
<ul class="tight">{assumptions}</ul>

<h2>Reproduce</h2>
<pre>cd mpgd26
../.venv/bin/python make_figures.py                # the whole still set
../.venv/bin/python make_chamber.py                # the exploded chamber
../.venv/bin/python make_x17.py --theme both       # the physics-case diagram
../.venv/bin/python make_x17.py --no-title         # ... without title/caption
../.venv/bin/python make_anim.py                   # turntables + build-ups
../.venv/bin/python make_report.py                 # this page

../.venv/bin/python make_figures.py --draft        # fast, for framing checks
../.venv/bin/python make_figures.py --theme both   # + dark theme

# individual scenes, with all the switches
../.venv/bin/python make_sps.py   --views hero,side,beam [--mx17] [--envelope]
../.venv/bin/python make_bench.py --views hero,side --slots p2,mx17
../.venv/bin/python make_anim.py  --only turn_bench --frames 120</pre>
<p class="caveat">Layout: <code>geometry.py</code> holds every number and its
   provenance; <code>meshes.py</code> turns geometry into meshes;
   <code>style.py</code> holds the palette, materials, light rig and render
   harness; <code>scenes_sps.py</code> / <code>scenes_bench.py</code> assemble
   each setup; <code>annotate.py</code> projects 3-D anchors to pixels and sets
   the type; <code>make_figures.py</code> drives the deliverable set.
   <code>scenes_x17.py</code> is the exception &mdash; a matplotlib diagram
   rather than a render, sharing only the palette.</p>

</div>"""

    with open(OUT, 'w') as f:
        f.write('<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width,'
                'initial-scale=1">\n'
                '<title>MPGD conference visuals</title>\n'
                f'<style>{CSS}</style>\n{body}\n')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    build()
