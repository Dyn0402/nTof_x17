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

ANIM_BLURB = {  # NTOF_BLURB is merged in below
    'turn_sps': 'Turntable of the H4 telescope.',
    'turn_bench': 'Turntable of the bench, both slots MX17.',
    'turn_bench_p2': 'Turntable of the bench, both slots P2 BASKET.',
    'turn_chamber': 'Turntable of the exploded chamber.',
    'build_sps': 'Build-up: table, then the uRWELL references, the three P2 '
                 'fans, and finally the beam.',
    'build_bench': 'Build-up: the rack, then the trigger paddles, the M3 '
                   'reference planes, the chambers under test, and finally '
                   'the muons.',
    'build_ntof': 'Build-up: the four detector layers going on around the '
                  'target, one at a time, with a real Geant4 e+e- pair '
                  'crossing them.  The close-up act of the still sequence is '
                  'not in the video -- it is at a different scale, and a cut '
                  'mid-way would read as an edit rather than as an assembly.',
    'turn_ntof': 'Turntable of the full n_TOF setup.',
}

NTOF_BLURB = {  # NTOF_BLURB is merged in below
    'ntof_build_1_capsule': 'The 3He capsule, cut open along the beam axis: '
        'a 20 mm bore of 500 bar 3He in a 0.6 mm Al vessel under a 0.9 mm '
        'CFRP wrap, mounted nose-first into the vertical EAR2 beam.',
    'ntof_build_2_neutron': 'A neutron arriving up the beam and captured in '
        'the gas.  The line is the beam axis, not a transported trajectory -- '
        'see the caveat below.',
    'ntof_build_3_pair': 'The e+e- pair leaving the vertex -- 10.7 and '
        '8.8 MeV, 110 degrees apart, both legs out through the barrel wall.  '
        'The neutron is drawn, not transported: see the caveat below.',
    'ntof_build_4_mm_near': 'The four micro-TPCs closing in around the vessel, '
        'at the scale where the 30 mm drift gap is still legible.  The camera '
        'looks through two of the arms, which are drawn as outlines.',
    'ntof_build_5_mm': 'The same four micro-TPCs at the scale of the '
        'apparatus, pinwheeled around the target: the only layer that measures '
        'a direction.  The pale trail inside each drift volume is the 13 keV '
        'the pair leaves there, drawn as the ionisation trail it is rather '
        'than as a dot.',
    'ntof_build_6_sipm': 'Plus the SiPM trigger walls, 16 of 20 bars read out.',
    'ntof_build_7_plastic': 'Plus the plastic scintillators, two wrapped bars '
        'per arm.',
    'ntof_build_8_plastic_top': 'The same, from almost straight down and with '
        'the structure dropped to a whisper.  The layers are stacked radially, '
        'so from a three-quarter view the trigger wall hides the plastics '
        'behind it; from here each leg can be followed chamber -> trigger wall '
        '-> plastic, in two different arms, and what keeps its colour is '
        'exactly the material that measures something.',
    'ntof_build_9_full': 'Plus the liquid-scintillator calorimeters -- the '
        'full setup, same camera as the frame before it.',
    'ntof_plan': 'The same setup and the same event as a PLAN: orthographic, '
        'looking down the beam, 1:1 in both axes.  Not a render -- a '
        'matplotlib drawing (make_ntof_plan.py) off the same geometry module, '
        'so the standoff, the layer radii and the size of the target '
        'can be measured off the page instead of being asserted in a caption.  '
        'A "_bare" version without the headline and the note is what the slide '
        'uses.',
}
# Every build frame names the layer it just added, with one leader per SOLID arm
# onto that layer, and drops the name again on the next frame -- so the label
# follows the build outward instead of accumulating on the finished apparatus.
# The anchors come from the geometry (scenes_ntof.layer_anchor), so a leader
# cannot end up pointing at where a layer used to be.

BLURB = {  # NTOF_BLURB is merged in below
    'x17_signature': 'The physics case in one figure: capture leaves '
                     '4He* with 20.58 MeV, three channels take it away, and '
                     'only one of them piles e+e- pairs up at a hard minimum '
                     'opening angle.  Both curves are sampled from the '
                     'MX17_Simulation generators, not traced from a paper.',
    'x17_signature_bare': 'The same diagram without the title and caption '
                          'bands, cropped -- for a slide whose own title bar '
                          'already says it.',
    'x17_story': 'The long version, five beats over two rows: beam on target, '
                 'capture, the level drop and its three channels, why the '
                 'parent mass fixes the opening angle, and the distribution '
                 'that falls out of it.  Panel 4 is the one the compact '
                 'layout has to assert instead of showing.  Re-flowed on '
                 '2026-08-18 onto a 124-unit canvas (scenes_x17.SW) so that '
                 'each row ALONE is 2.16:1, the shape of the figure hole on a '
                 'deck slide -- which is why the two-row compilation is now '
                 'portrait.  A slide figure is width-limited, so the units a '
                 'row spans are the only lever on how large its type comes '
                 'out; the beats were re-flowed rather than re-typed, and not '
                 'one font size changed.',
    'x17_story_bare': 'The five-beat layout without the title and caption '
                      'bands, cropped.',
    'x17_story_1of2': 'Slide 1 of the split version: beats 1-3, which set up '
                      'the physics and end on the reason the whole experiment '
                      'is a pair spectrometer.',
    'x17_story_2of2': 'Slide 2 of the split version: beats 4-5, which derive '
                      'the measurement from the boost.',
    'x17_beat1_beam_capsule': 'Beat 1 alone: the neutron beam arriving from '
                              'below on the real 3He vessel, with a zoom onto '
                              'one capture in the gas.',
    'x17_beat2_capture': 'Beat 2 alone: n + 3He -> 4He*, 20.58 MeV above the '
                         'ground state.',
    'x17_beat3_channels': 'Beat 3 alone: the level drop and the three ways it '
                          'can be taken away -- gamma, internal pair '
                          'conversion, and the X17 hypothesis.',
    'x17_beat4_boost': 'Beat 4 alone: the same pair cartoon for a heavy slow '
                       'X17 and a light fast IPC pair, boosted from five rest-'
                       'frame orientations. The X17 pair never closes below '
                       '109 deg; the IPC pair never opens above 11 -- which '
                       'the five angle numbers per row now have to say on '
                       'their own: the subtitle and the summary paragraph came '
                       'off on 2026-08-18, and the drawing grew 1.19x into the '
                       'space (Dylan: "remove the in the rest frame ... and '
                       'whatever the orientation ..."). The left block was '
                       'restacked to pay for it -- rest-frame icon over boost '
                       'arrow instead of beside it -- because the row is '
                       'width-limited and those units were the whole budget.',
    'x17_beat5_spectrum': 'Beat 5 alone: the opening-angle distributions the '
                          'boost produces -- a peak at the kinematic minimum '
                          'against a smooth IPC slope. The panel went 34 x 26 '
                          '-> 34 x 38 canvas units on 2026-08-18, into the '
                          'height the two paragraphs under it were using.',
    'x17_story_capsule': 'The same five beats, but with beat 1 drawing the '
                         'real 3He vessel from the Geant4 geometry instead of '
                         'a generic group of nuclei -- for later in a talk, '
                         'once the target hardware has been introduced.',
    'chamber_exploded': 'One MX17 chamber with its layers separated along the '
                        'drift axis, and a muon whose ionisation drifts down '
                        'to the mesh -- the micro-TPC picture the whole '
                        'reconstruction rests on. Landscape since '
                        '2026-08-17: a 44 x 34 mm window on the chamber '
                        'rather than a 30 mm square, with the labels on the '
                        'render beside their own layer instead of in a '
                        'gutter, so the layers get the width of the slide. '
                        'The readout side is the as-built board (L4 pads, '
                        'L5/L6 strips, the black ESL film on its own 0.80 mm '
                        'pitch) and the window went 120 x 30 -> 60 x 18 -> '
                        '60 x 34 -> 44 x 34 mm over that one day: in to '
                        'resolve the strip structure, then deeper along the '
                        'strips so the layers read as planes rather than '
                        'ribbons, then in again across them. The last step '
                        'came with the muon: its tube was drawn 0.9 mm across, '
                        'i.e. 1.2 strip pitches, at the scale of the structure '
                        'it is supposed to be crossing, and is now 0.30 with '
                        'drift lines to match. The frame width never changed '
                        '-- make_chamber.VIEW s view_angle tracks WIN_MM, so '
                        'these are magnifications and not crops.',
    'x17_story_bot_3_detect': 'Deck frame 6.3: the bottom row with the '
                        'micro-TPC cartoon standing where beat 4 was, and the '
                        'spectrum exactly where it was on 6.2 -- so the frame '
                        'changes the ARGUMENT beside the spectrum and not the '
                        'spectrum. Until 2026-08-18 the slide stacked two '
                        'full-width pictures in one figure box, which cost '
                        'both of them ~41 % of their width. One claim only -- '
                        'one gas gap gives a direction, two give the opening '
                        'angle. The OPENING ANGLE is drawn true (measure it '
                        'with a protractor); the standoff, the gap and the '
                        'chamber size are not to scale, and the real ones are '
                        '204 mm, 30 mm and 400 mm. The chamber, not the track, '
                        'carries the 21 deg tilt: a track square to the '
                        'readout plane deposits all its charge at one depth '
                        'and there is nothing for a micro-TPC to reconstruct.',
    'x17_detect_solo': 'The same micro-TPC cartoon on a canvas of its own, '
                       'for a slide or a poster that wants it without the '
                       'spectrum. scenes_x17.draw_detect; the version INSIDE '
                       'the story row is the same drawing at 0.87 of its '
                       'length scale, not a second implementation.',
    'share_cartoon': 'The sharing mechanism as a drawing: the avalanche lands '
                     'on the resistive layer, the charge that goes sideways '
                     'goes through the layer\'s own sheet resistance, and the '
                     'neighbours pick up copies that are LATE (166 ns to +-1, '
                     '333 to +-2) and dispersed. Deck slide 9.1.',
    'share_kernels': 'The kernels, per plane, out of the CORRECTED det3 '
                     'bundle calib_bundle_r06: the response to charge on the '
                     'strip itself, and the copies +-1 and +-2 see. '
                     'X 5 / 3 %, Y 15 / 9 % -- the layer\'s strips run along y, '
                     'so Y shares ~3x more (kY = 2.9). The +-2 copy is SMALLER '
                     'than the +-1 copy, which is the only ordering possible: '
                     'the +-2 strip is reached only through the +-1. The '
                     'frozen production bundle had it the other way round '
                     '(c2/c1 = 1.14); the ratio is now pinned at 0.6, from the '
                     'H4 beam\'s model-free 0.45 +- 0.02. The absolute c1 on a '
                     'cosmic fit is still a lower bound. Deck slide 9.2.',
    'share_build': 'What the model does, in four stages: the drift column in '
                   '60 ns slices with free non-negative charges, the '
                   'geometric strip integral, the kernel copies onto '
                   '+-1/+-2, and the fold with the measured impulse '
                   'response. Deck slide 10, left.',
    'share_decompose': 'The same split on REAL DATA -- four consecutive '
                       'strips of event 1663 (Y plane), each fitted waveform '
                       'stacked into own / +-1 / +-2 charge against the '
                       'measurement. Walk out from the core and 40 % of '
                       'the pulse stops being the strip\'s own charge (20 % on '
                       'the core, 42 % three strips out). Deck slide 10, '
                       'right.',
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


BLURB.update(NTOF_BLURB)


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
        links = [f'<a href="{base}.png">PNG</a>']
        # the X17 diagrams are matplotlib and ship a vector PDF next to the
        # PNG; the PyVista renders are rasters and have none, so the link is
        # offered only when the file is actually there
        if os.path.exists(os.path.join(HERE, base + '.pdf')):
            links.append(f'<a href="{base}.pdf">vector PDF</a>')
    else:
        return ''
    return (f'<figure id="{esc(name)}">\n'
            f'  <img src="{base}.png" alt="{esc(name)}">\n'
            f'  <figcaption><b>{esc(name)}</b> &mdash; '
            f'{esc(BLURB.get(name, ""))}<br>{" &middot; ".join(links)}'
            f'</figcaption>\n</figure>\n')


CAMPAIGN_BLURB = (
    'Nine months of the programme over the six weeks of beam, on one canvas: '
    'the mini timeline (four EAR2 exposures, names and dates only) with the '
    'daily event census expanded out of its last bar by a zoom wedge. The '
    'Saclay bench month is off the strip -- it is the one bar that is not a '
    "beam exposure. Events are entries in each sub-run's own decoded event "
    'tree, counted on EOS, so the census depends on neither the n_TOF stream '
    'nor any matching.')

RATE1_BLURB = (
    "Where the X17 rate is. Dylan's December 2025 rate calculation "
    '(data/x17_rate_3He.txt) on a neutron flight-time axis with energy on '
    'top, drawn as the interpolated point plot from '
    'neutron_energy_vs_flight_time.py: the markers carry the exact decade bin '
    'widths and the faint line is a log-log cubic spline through them. Two '
    'decades carry 79 % of the rate, and they arrive 0.45-4.5 us after the '
    'flash.')

RATE2_BLURB = (
    "The same drawing with the front end's measured dead time on it: firm to "
    '1 ms (no track has ever been reconstructed earlier, run_79) and fading '
    'to 9 ms (the slowest chamber at the production operating point, the '
    'run_57 recovery map). The whole MeV peak is inside it, so the accent '
    'moves to the thermal bin -- 10 % of the rate, and recordable. Same '
    'points, same limits, same annotation positions as frame 1.')


def plain_fig(name, blurb):
    """A figure that has no per-theme variant.

    make_campaign.py and make_x17_rate.py write one light-theme file each,
    without the ``_light`` suffix fig_block expects, because they are deck
    figures and the deck is light-only.  Rather than give them a theme axis
    they will never use, they get their own block here.
    """
    png = f'figures/{name}.png'
    if not os.path.exists(os.path.join(HERE, png)):
        return ''
    links = [f'<a href="{png}">PNG</a>']
    if os.path.exists(os.path.join(HERE, f'figures/{name}.pdf')):
        links.append(f'<a href="figures/{name}.pdf">vector PDF</a>')
    return (f'<figure id="{esc(name)}">\n'
            f'  <img src="{png}" alt="{esc(name)}">\n'
            f'  <figcaption><b>{esc(name)}</b> &mdash; {esc(blurb)}<br>'
            f'{" &middot; ".join(links)}</figcaption>\n</figure>\n')


def sps_table():
    rows = ''
    for st in G.SPS_STATIONS:
        note = ('placeholder z; yawed %.2f deg' % st.yaw
                if st.kind == 'mx17' else '')
        size = {'urwell': f'{G.URW_ACTIVE_MM:.1f} mm square active',
                'p2': 'fan, r 150.7-635.0 mm, 55.6 deg, 1280 pads',
                'mx17': f'{G.MX17_ACTIVE_MM:.1f} mm square active'}[st.kind]
        rows += (f'<tr><td>{esc(st.name)}</td><td class="num">{st.z:g}</td>'
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
<p>Neither curve in panel 3 is traced from anyone's plot.  Both are sampled from
   <code>MX17_Simulation/MX17_Simulator.py</code> &mdash;
   <code>X17PhysicsSpectrum</code> and <code>IPCPhysicsSpectrum</code>, the same
   generators the acceptance and significance studies use, and which that module
   documents as matching the Geant4 <code>X17PrimaryGenerator</code> event for
   event.  {X.SAMPLE_N:,} events per channel, Gaussian-smeared by
   {X.X17['smear_deg']:g}&deg; so the X17 Jacobian divergence has a width on
   paper.  The figure therefore cannot drift away from the simulation: change
   the generator and the figure follows.</p>
<p>The X17 channel is additionally solved analytically in
   <code>scenes_x17.py</code> (<code>opening_angle_pdf</code>), purely as a
   check on the sampler &mdash; <code>make_x17.py --validate</code> prints both
   minima, and they agree to better than 0.01&deg;.</p>
<p class="caveat"><b>Worth knowing before you quote this panel.</b> IPC is not
   confined to small opening angle the way a quick sketch suggests: because its
   pair invariant mass is drawn from dN/dM &prop; 1/M, it is a superposition of
   two-body decays over the whole mass range, with a median opening angle near
   30&deg; and roughly 30 % of its yield above 60&deg; &mdash; i.e. real
   background sitting underneath the X17 peak.  The two curves are each
   normalised to unit peak, so nothing in the panel implies a branching ratio;
   their relative rate is the measurement.</p>
{fig_block('x17_signature', theme)}
{fig_block('x17_signature_bare', theme)}

<h3>The long version</h3>
<p>Same physics, five beats over two rows, for when there is a whole slide to
   spend on it: the EAR2 beam coming up into the <sup>3</sup>He capsule, one
   capture, the level drop and its three channels, <b>what the boost does</b>,
   and only then the distribution.</p>
<p>Beat 1 is deliberately generic &mdash; a beam and some <sup>3</sup>He, no
   vessel &mdash; because early in a talk the target hardware has not been
   introduced yet.  <code>--capsule</code> swaps in the version that draws the
   real thing: the <code>He3Gas</code> / <code>He3Cap_Al</code> /
   <code>He3Cap_CFRP</code> polycones from
   <code>MX17_Full_Geant/src/DetectorConstruction.cc</code>, sectioned from the
   STEP solid <i>MASTINU X17 HPRV 00 01</i>, in true aspect and mounted
   nose-first as the simulation mounts it.  EAR2 is vertical either way, so the
   neutrons arrive from below.</p>
<p>The fourth beat is the reason this layout exists, and it turns on one fact:
   the pair is <i>always</i> back-to-back in the parent's rest frame, so
   everything you see in the lab is the boost.  Whether the parent outruns its
   own decay products decides the shape:</p>
<ul class="tight">
  <li><b>Heavy parent, slow</b> (X17 at {X.X17['m_x17']:g} MeV,
      &beta; = 0.58).  The backward lepton still goes backward, so the pair
      reaches 180&deg; and is bounded <i>below</i> &mdash; a hard edge at
      {X.opening_angle_pdf()[2]:.0f}&deg; with the yield piled against it.</li>
  <li><b>Light parent, ultra-relativistic</b> (a 2 MeV IPC pair,
      &beta; = 0.995).  Both leptons are swept forward into a cone, so the pair
      is bounded <i>above</i> &mdash; here at 11&deg; &mdash; and can close all
      the way to zero.</li>
</ul>
<p>The crossover is at <i>m</i> = &radic;(2m<sub>e</sub>E) &asymp; 4.6 MeV.
   Beat 4 shows this as three worked orientations per channel &mdash; the decay
   direction in the rest frame, and the lab opening angle it produces &mdash;
   so the reader can see that <i>no</i> orientation lets X17 close below
   109&deg;, and none lets a light IPC pair open past 11&deg;.  The boost
   arrows are drawn to length &beta;, which is why the X17 arrow is visibly
   stubby next to the IPC one.  Since IPC draws its pair mass from
   dN/dM &prop; 1/M it gets a different band for every mass, and those bands
   between them fill the whole axis &mdash; which is exactly the smooth slope
   panel 5 shows underneath the X17 peak.</p>
<p><b>Panel 5 is a stack, not an overlay</b> (2026-08-17). The compact
   <code>x17_signature</code> panel above compares two <i>shapes</i>, each
   normalised to unit peak; the story panel instead draws what the measurement
   will look like &mdash; the IPC background, with a small X17 yield sitting on
   top of it and the filled area between the two curves being the excess. That
   costs the figure its ratio-free honesty, so the ratio is a declared
   parameter: <code>scenes_x17.SIG_FRAC</code>
   ({X.SIG_FRAC * 100:.0f}&nbsp;% of the IPC yield over the plotted window)
   is printed on the panel in words, and is illustrative, not predicted. It
   puts the bump about 80&nbsp;% above the local background at its peak. The
   window starts at {X.SPEC_XLIM[0]:.0f}&deg; for the same reason ATOMKI plot
   from 40&deg;: the IPC forward peak is eight times the yield at 109&deg;, and
   including it flattens everything the panel is about &mdash; the forward
   sweep is beat 4's argument, made there as kinematics.</p>
{fig_block('x17_story', theme)}
{fig_block('x17_story_bare', theme)}
{fig_block('x17_story_capsule', theme)}

<h3>Split across two slides</h3>
<p>The two rows are a natural break: the top one sets up the physics and ends
   on <b>&ldquo;detect the e<sup>+</sup>e<sup>-</sup> pair&rdquo;</b>, the
   bottom one derives the measurement from the boost.
   <code>--layout split</code> writes them as two figures, each with its own
   title, subtitle and caption.</p>
<p class="caveat">They are not crops. Each part is the same drawing seen
   through a different canvas band, so a change to any beat lands in the
   combined figure and in its slide together &mdash; there is no second layout
   to keep in step.</p>
{fig_block('x17_story_1of2', theme)}
{fig_block('x17_story_2of2', theme)}
<p><b>The deck uses this split as of 2026-08-17</b> &mdash; two slides where
   there was one, on the <code>--no-title</code> variants, since the slide's own
   title bar carries what each row's headline said. The rows are wide and short
   (4.6:1 and 3.9:1), so on a 16:9 slide the figure is <b>width-bound and fills
   about 55&ndash;60&nbsp;% of the height</b>. Rearranging the beats inside a
   slide does not recover that &mdash; two stacked rows halve the height per row
   and give back what the extra width buys (measured: gains of &times;1.0 or
   worse for every 2-row arrangement). The only lever that would is a narrower
   set of beats, e.g. fewer than five orientation columns in beat 4.</p>

<h3>One beat per file</h3>
<p>The same five beats are also written one to a file
   (<code>--layout beats</code>, or <code>--layout beat3</code> for one), for
   building a slide up a beat at a time or for lifting a single picture into
   another deck. Same principle as the split above and the same guarantee: each
   is the story drawing cropped to its own beat, so nothing is redrawn and
   nothing can drift &mdash; adding them left
   <code>x17_story_capsule_light.png</code> byte-identical. Each keeps its
   row's full height by default so beats used in sequence stay in register;
   <code>--tight</code> trims to the ink instead.</p>
{fig_block('x17_beat1_beam_capsule', theme)}
{fig_block('x17_beat2_capture', theme)}
{fig_block('x17_beat3_channels', theme)}
{fig_block('x17_beat4_boost', theme)}
{fig_block('x17_beat5_spectrum', theme)}

<h3>&hellip;and the beat that hands over to the detector</h3>
<p>Added 2026-08-17.  By the end of beat 5 the audience has been told that the
   observable is an <b>angle</b>, and nothing has yet said what measures one.
   This is the bridge, and it is deliberately the smallest possible claim: a
   micro-TPC turns one gas gap into a direction, so two of them give the
   opening angle.  Since 2026-08-18 it is drawn <b>inside the story canvas</b>,
   in the box beat 4 was using (<code>--layout bot3</code>), so the deck's frame
   6.3 is one full-width picture: the spectrum neither moves nor resizes, and
   what changes beside it is the argument.  It sat <i>under</i> the spectrum
   until then, and two stacked full-width pictures can never be more than
   ~59&nbsp;% as wide as one.  <code>--layout detect_solo</code> still writes it
   on a canvas of its own.</p>
<p class="caveat">Drawn, and not to scale &mdash; except for the one thing that
   is.  The <b>opening angle is the real 110&deg;</b> and can be measured off
   the page; the standoff (204 mm from a 23 mm capsule), the 30 mm gap and the
   400 mm chamber are not, because at scale the gap would be a hairline.  What
   is tilted is the <b>chamber</b>, by 21&deg;: a track arriving square to the
   readout plane leaves all its charge at one depth, and then there is no
   drift-time ladder to reconstruct at all.</p>
{fig_block('x17_story_bot_3_detect', theme)}
{fig_block('x17_detect_solo', theme)}

<h2>The Status section&rsquo;s two arguments</h2>
<p>Rebuilt 2026-08-19.  The section used to be a list of results; it is now one
   argument, and these are the two figures that carry it.  Both are light-only
   and both are saved on their full canvas at the <b>measured</b> aspect of the
   slide hole they go into (2.95:1 with a stat row under it, 2.22:1 with only a
   caption) &mdash; a tight bounding box would crop to the ink and change the
   ratio, so the figure would arrive smaller than the hole.</p>
{plain_fig('campaign_overview', CAMPAIGN_BLURB)}
{plain_fig('x17_rate_1_physics', RATE1_BLURB)}
{plain_fig('x17_rate_2_window', RATE2_BLURB)}

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

<h2>Micro-TPC operation</h2>
<p>The measurement itself.  A muon crosses the
   {G.MX17_DRIFT_GAP_MM:g} mm gap at an angle and leaves primary ionisation
   clusters at ~30/cm; each drifts straight down at the MEASURED
   v = 36.6 &micro;m/ns, so its arrival time at the mesh <i>is</i> the depth it
   was created at &mdash; 820 ns across the full gap.  (The Magboltz value for
   the same mixture, ~34 &micro;m/ns, would give 882 ns; the two numbers agree
   within the measurement, but mixing them is how the deck's old caption came
   to read "36.6 &micro;m/ns &middot; 882 ns", which is neither.)  The slope of arrival time against strip position is
   then the track angle, from one plane.  v<sub>drift</sub> and &sigma;<sub>T</sub>
   are the Garfield++/Magboltz values for the mixture the bench actually runs
   (Ar/iso 95/5 + ~1&nbsp;% H<sub>2</sub>O at 333 V/cm, from
   <code>garfield_sim/results/water_grid.json</code>); the gap, the
   0.78 mm pitch and the 512 strips are the detector's own.</p>
<p>Two variants: <b>waveforms</b> (the raw per-strip signals, no fit) and
   <b>ladder</b> (first arrival per strip, with a straight-line fit).  The
   waveform pulse shape is the plane's own MEASURED single-electron response
   from the wft calibration bundle, and v<sub>drift</sub> = 36.6 &micro;m/ns is
   the value measured for det3 on that run &mdash; the Magboltz table's
   ~34 &micro;m/ns for the same mixture agrees with it.</p>
<p class="caveat">The ladder is fitted on the <b>first arrival per strip</b>, a
   deliberately simple estimator that carries a small bias &mdash; which is why
   the real reconstruction fits the waveforms forward (<code>wft/</code>)
   instead.  See <code>RECONSTRUCTION_BASIS.md</code>.</p>
{fig_block('microtpc_waveforms', theme)}
{fig_block('microtpc', theme)}

<h2>Charge sharing, and what a predicted waveform is made of</h2>
<p>Four figures from <code>make_share.py</code>, behind the two reconstruction
   slides.  They share one colour rule &mdash; blue for a strip's own charge,
   vermillion for &plusmn;1, purple for &plusmn;2 &mdash; in the cartoon, in the
   kernels, in the diagram and in the real-data decomposition, so a colour means
   one thing across the whole section.</p>
<p>Everything with a number on it comes from the <b>frozen production
   bundle</b> <code>calib_bundle_lp2_t0p</code> (det3, Saturday long run,
   resistive 490 V / drift 1000 V) and from event 1663 of its ref-pinned
   calibration cache &mdash; the same event the "One muon through the forward
   fit" slide uses.  The decomposition is exact rather than estimated: the model
   is a sum of an own-charge term and two neighbour terms, so rebuilding the
   design matrix with c<sub>1</sub> and c<sub>2</sub> zeroed and differencing
   gives the terms themselves.</p>
<p class="caveat"><b>The amplitudes are a floor, not a measurement.</b> On this
   bundle c<sub>1</sub> = 0.051 sits on the C1_MIN = 0.05 bound that
   <code>wft.calibrate</code> imposes: a cosmic-angle fit cannot separate
   sharing from a wider initial cloud plus a different v<sub>drift</sub>, and
   without the bound it walks c<sub>1</sub> to zero and hides the sharing in
   <code>sigma_p0</code>.  The H4 beam test at normal incidence breaks that
   degeneracy and measures ~0.28&ndash;0.30.  The <i>shapes</i> and the
   <i>delays</i> on these figures are the model's own and need no caveat; the
   X/Y ratio (kY = 2.9) is fitted and is the interesting number; the absolute
   level is floor-limited.  c<sub>2</sub> &gt; c<sub>1</sub> in this bundle,
   which is a fit correlation and not a ladder &mdash; what the model needs
   right is their sum and their delay.</p>
{''.join(fig_block(n, theme) for n in
         ('share_cartoon', 'share_kernels', 'share_build', 'share_decompose'))}

<h2>The n_TOF setup, built up</h2>
<p>The talk's setup section is one figure in eight states: the &sup3;He capsule,
   a neutron reaching it, the e&#8314;e&#8315; pair leaving it, and then the four
   detector layers going on one at a time at a fixed camera.  Unlike the two
   bench scenes, <b>the geometry here is not held in <code>geometry.py</code></b>
   &mdash; it is imported at run time from
   <code>MX17_Full_Geant/scripts/plot_geometry.py</code>, which is written
   against the simulation's own <code>SimConfig.hh</code>, so the figure and the
   Geant4 model cannot drift apart.</p>
<p>The subject changes scale by a factor of fifty between the capsule (23 mm)
   and the setup (1.2 m) and then changes what it has to show, so the sequence
   runs in four acts at four cameras: frames 1&ndash;3 on the vessel and the
   event in it, frame 4 as the chambers close in around it, frames 5&ndash;7 on
   the apparatus, and frames 8&ndash;9 on the same apparatus <b>from above</b>.
   The last cut is not decoration: the layers are stacked <b>radially</b>, so
   from any three-quarter view the trigger wall stands in front of the plastics
   and the liquid, and a leg arriving in them cannot be seen.  Within each act
   the frames share a camera and a size exactly, so the layers grow onto a still
   picture; frame 4 repeats frame 5's layer at a larger scale and is the first
   to cut.</p>
<p>The event is <b>selected, not hand-picked</b>.  On top of the legibility
   ranking, <code>extract_ntof_event.py</code> requires both legs to leave the
   vessel through the <b>barrel wall</b> &mdash; at the barrel radius, inside the
   straight section, and within ~33&deg; of transverse &mdash; rather than
   through a domed end, where a leg crosses several times the wall thickness at
   a glancing angle.  It then weights heavily towards the event in which
   <b>both legs cross all four layers</b>, so each can be followed chamber
   &rarr; trigger wall &rarr; plastic &rarr; liquid and the build-up has
   something to add at every step: exactly <b>one of the 400 simulated events
   does that</b>, and it is the one drawn.  (A leg can pass between the two
   plastic bars or out of the side of the stack, so this is rarer than it
   sounds.)  It also prefers events whose legs land in the two arms the figure
   draws solid, so no deposit ends up glowing inside a wireframe;
   <code>--prefer-arms</code>, <code>scenes_ntof.NEAR_ARMS</code> and the camera
   azimuth all have to be kept in step.</p>
<p>Only the <b>charged</b> pair is drawn.  The event's bremsstrahlung gammas are
   real, and they are in the JSON, but a neutral track leaves the frame in a
   straight line from wherever it was radiated, so on the picture they read as
   stray rays with no visible cause; <code>scenes_ntof.DRAWN_PARTICLES</code> is
   what suppresses them.</p>
<p>The legs <b>grow with the apparatus</b>: each is cut where it first reaches a
   layer that is not on the frame yet
   (<code>scenes_ntof.truncate_at_next_layer</code>), so a track only runs on to
   the trigger wall on the frame that puts the trigger wall there.  Drawn full
   length from the start, the tracks claim three frames early that all of it is
   measured, and the layer being added stops being the thing that changes.  The
   beam column stops at the vessel's <b>nose</b> as soon as any detector is
   placed: past the target it runs through the middle of the apparatus and reads
   as a pole holding it up, and drawn even alongside the capsule it sleeves a
   23 mm object in translucent grey, so the target goes hazy instead of crisp.
   The direction dart lies on the axis inside the column, on the close-up act
   only.</p>
<p>Each build frame <b>names the layer it just added</b>, and drops the name
   again on the next frame &mdash; so the label follows the build outward
   instead of piling four names onto the finished apparatus.  It carries
   <b>one leader per solid arm</b>: a layer is four objects and the frame draws
   two of them solid, so a single line would quietly imply the label is about
   that one &mdash; and it is one leader per drawn <i>object</i>, so the
   plastics, which are two separate bars per arm, carry four and their label
   drops the &quot;2 &times;&quot; the lines already say.
   The anchors are computed from the same geometry the meshes are
   built from (<code>scenes_ntof.layer_anchor</code>), so a leader cannot end up
   pointing at where a layer used to be, and the text is pinned to a corner of
   the frame rather than offset from its anchor, because a pinwheel's corners
   are the only reliably empty places &mdash; top-left on the build acts, and at
   the bottom on the close-up, where the chambers fill the frame and the only
   empty space left is the see-through one.  Sizes on the figures are in
   <b>centimetres</b>, since they are read from across a room; the slide bullets
   keep the millimetres.  <code>--no-labels</code> turns all in-image type off.</p>
<p>The overhead act draws the capsule <b>whole and solid</b>.  Everywhere else
   the vessel is sectioned on a plane through the beam axis with the near half
   removed &mdash; the only way to show 0.6 mm of wall and a lit gas core at
   once &mdash; but that plane <i>contains</i> the overhead view direction, so
   from up there it does not open the vessel, it deletes the half of it nearest
   the bottom of the frame (<code>VIEWS['over']</code> sets
   <code>cut=False</code>).  It is also the one part <code>BARE</code> does not
   whisper: everything else that drops to <code>BARE_ALPHA</code> is a box the
   eye can still infer from its neighbours, but the capsule is 23 mm on a 1.2 m
   frame and faded it is a smudge at the exact point the picture converges on.
   Solid, it is a small dark disc &mdash; the CFRP overwrap end-on, with the gas
   bore a speck at its centre &mdash; where the two legs meet.</p>
<p>The sequence closes on a <b>plan</b> (<code>make_ntof_plan.py</code>,
   <code>ntof_plan</code> below), which is the one figure here that is not a
   render.  Every frame above it has perspective: the four arms sit at four
   different distances from the lens, so every length on them is foreshortened
   by its own amount and the distances can only be <i>written</i> in a caption.
   The plan is orthographic and 1:1 in both axes &mdash; the beam is the view
   axis, so the drawing plane is the X-Z plane the apparatus is symmetric in.
   Three things become measurable rather than asserted: the <b>204.5 mm</b>
   standoff every arm's window sits on, drawn as the circle they are all
   tangent to; the layer radii out to the vessels (330 / 410 / 487 mm on arm B,
   the one arm no leg crosses and so the one the chain is drawn on &mdash; the
   outer two move by a few mm between arms); and the <b>size of the
   target</b>, which at 1:1 is a 23 mm dot in the middle of a 1.1 m apparatus.
   It is built from the same geometry module and the same event JSON as the
   renders, so it cannot drift away from them, and it makes two facts visible
   that a three-quarter camera cannot: the ~16 mm pinwheel offsets, and that
   two of the four liquid vessels are laid on their side with their PMTs
   pointing sideways into the plane.  What it gives up is the beam axis: the
   legs also rise ~135 mm along it, so the opening angle as drawn (122&deg;) is
   not the space angle (110&deg;), and the figure says both.</p>
<div class="verdict">
  <b>The e&#8314;e&#8315; pair is one real Geant4 event; the neutron that made
  it is drawn, not simulated alongside it</b>, and it has to be that way.  The neutron is transported
  from the measured EAR2 flux and does what the physics list says it does,
  which is &sup3;He(n,p)t; the radiative branch that forms the &#8308;He* this
  search lives on is ~10&#8315;&#8318; of it, so no neutron run will ever
  contain one.  The pair is therefore thrown by the generator from a vertex
  sampled in the gas, and the neutron history is translated so its interaction
  point coincides with that vertex.  <code>tools/extract_ntof_event.py</code>
  records the pairing and the shift in <code>data/ntof_event.json</code>; the
  talk says it on the slide.  What the <i>scene</i> draws for the neutron is a
  straight line up the beam axis to the pair vertex &mdash; the transported
  history is still selected and stored, but it belonged to a different event,
  so drawing it meant translating it onto this vertex and then showing its own
  in-gas scattering, which is a fact about that neutron rather than about this
  figure.  The neutron run is still needed either way: the beam envelope is
  measured from its sampled primaries.  The
  (n,p)t proton and triton the transported neutron actually made are kept in
  the JSON and deliberately <i>not</i> drawn &mdash; they belong to the other
  branch.
</div>
{''.join(fig_block(n, theme) for n in sorted(NTOF_BLURB))}
<p class="caveat">Drawn but not measured, in this scene only: the liquid
   scintillator's fill dome is extruded at constant height along the vessel's
   long axis (the real 6.5 L bulge falls away towards the edges), and the beam
   envelope is drawn at the radius the simulation's own sampled primaries
   occupy &mdash; 90&nbsp;% inside 8.8 mm &mdash; which is the beam profile, not
   the collimator; and the arriving neutron is a straight line up that axis
   rather than a trajectory.  The two arms nearest the camera &mdash; B in front
   of it and A on the right of the frame &mdash; are drawn as outlines only, so
   that the target and the two arms this pair actually crossed (D and C) stay
   visible, and on the overhead frames the structure of the other two drops to
   <code>BARE_ALPHA</code> so only the active volumes carry colour.</p>

<h2>Animations</h2>
<p>Turntables for the talk, and build-up sequences whose numbered stills can be
   dropped on successive slides so the setup assembles itself as you speak.
   Every frame comes from the same scene code as the stills above.</p>
{''.join(anim_block(n) for n in
         ['turn_sps', 'turn_bench', 'turn_bench_p2', 'turn_chamber',
          'build_sps', 'build_bench', 'build_ntof', 'turn_ntof'])}

<h2>What is drawn but not measured</h2>
<ul class="tight">{assumptions}</ul>

<h2>Reproduce</h2>
<pre>cd mpgd26
../.venv/bin/python make_figures.py                # the whole still set
../.venv/bin/python make_chamber.py                # the exploded chamber
../.venv/bin/python make_x17.py --theme both       # the physics-case diagram
../.venv/bin/python make_x17.py --no-title         # ... without title/caption
../.venv/bin/python make_ntof.py                   # the n_TOF build-up
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
