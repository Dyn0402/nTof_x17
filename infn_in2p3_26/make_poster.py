"""A0 portrait poster mock-up for the INFN-IN2P3 detector R&D conference (LNF, 2-4 Dec 2026).

Writes poster.html (sizes in mm, so the page is a true A0), and with --pdf prints poster.pdf through
headless Chrome plus a small poster_preview.png. Figures are copies in assets/ (see ASSETS below for where
each one comes from); the hero render is make_hero.py.

    ../.venv/bin/python make_poster.py --pdf
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

from PIL import Image, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
A = os.path.join(HERE, 'assets')

# asset name -> (source, how it is made)
ASSETS = {
    'hero_x17_box.png': 'make_hero.py',
    'chamber_exploded.png': 'mpgd26/figures/chamber_exploded_light_labelled.png (mpgd26/make_chamber.py), title cropped',
    'readout_stack_3d.png': 'papers/paper1_v2/figs/out/readout_stack_3d_render.png',
    'pointing.png': 'papers/paper1_v2/figures/pointing.pdf',
    'waterfall_7884.png': 'make_waterfall.py (papers/paper1_v2/display_lab/events.json)',
    'angle_resolution_v2.png': 'papers/paper1_v2/figures/angle_resolution_v2.pdf',
    'two_track_separation.png': 'papers/paper1_v2/figures/two_track_separation.pdf',
    'efficiency_map.png': 'papers/paper1_v2/figures/efficiency_map.pdf',
    'ntof_station.png': 'mpgd26: make_ntof.py --only full --view hero --no-labels --size 2400 2040',
    'ntof_pointing.png': 'ntof_athens_26/figures/sideview_run145.png',
    'chamber_bare.png': 'chamber_exploded.png without its label column (labels set in the HTML)',
    'ntof_fan.png': 'left part of ntof_pointing.png (the track fan)',
    'fig_air.svg': 'make_poster_figs.py (papers/paper1_v2/figs/out/pointing.json)',
    'fig_angres.svg': 'make_poster_figs.py (papers/paper1_v2/figs/out/angle_resolution_v2.csv)',
}


def trim(path, pad=10):
    """Crop a render to its content (transparent or white border)."""
    im = Image.open(path)
    rgb = Image.new('RGB', im.size, 'white')
    rgb.paste(im, mask=im.split()[3] if im.mode == 'RGBA' else None)
    box = ImageChops.difference(rgb, Image.new('RGB', im.size, 'white')).getbbox()
    if box:
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))
        im.crop(box).save(path)


def prepare_assets():
    os.makedirs(A, exist_ok=True)
    src = os.path.join(REPO, 'mpgd26/figures/chamber_exploded_light_labelled.png')
    im = Image.open(src)
    h = im.height
    im.crop((0, int(0.098 * h), im.width, int(0.851 * h))).save(os.path.join(A, 'chamber_exploded.png'))
    shutil.copy(os.path.join(REPO, 'papers/paper1_v2/figs/out/readout_stack_3d_render.png'),
                os.path.join(A, 'readout_stack_3d.png'))
    st = os.path.join(REPO, 'mpgd26/figures/ntof_build_9_full_hero_light.png')   # gitignored render
    if os.path.exists(st):
        shutil.copy(st, os.path.join(A, 'ntof_station.png'))
    shutil.copy(os.path.join(REPO, 'ntof_athens_26/figures/sideview_run145.png'), os.path.join(A, 'ntof_pointing.png'))
    for f in ('pointing', 'angle_resolution_v2', 'two_track_separation', 'efficiency_map'):
        subprocess.run(['pdftoppm', '-r', '250', '-png', '-singlefile',
                        os.path.join(REPO, 'papers/paper1_v2/figures', f + '.pdf'), os.path.join(A, f)], check=True)
    im = Image.open(os.path.join(A, 'ntof_pointing.png'))      # the track fan alone (left 55 %), for v3
    im.crop((0, int(0.09 * im.height), int(0.553 * im.width), int(0.89 * im.height))).save(os.path.join(A, 'ntof_fan.png'))
    im = Image.open(os.path.join(A, 'chamber_exploded.png'))   # render without its small label column (v3)
    im.crop((630, 0, im.width, im.height)).save(os.path.join(A, 'chamber_bare.png'))
    import make_poster_figs
    make_poster_figs.fig_air(); make_poster_figs.fig_angres()
    for f in ('hero_x17_box.png', 'ntof_station.png', 'readout_stack_3d.png', 'waterfall_7884.png'):
        if os.path.exists(os.path.join(A, f)):
            trim(os.path.join(A, f))


CSS = r"""
@page { size: 841mm 1189mm; margin: 0; }
:root { --ink:#16202e; --mute:#5b6675; --navy:#0b3d91; --navy2:#0a2a66; --accent:#d9631e; --bg:#f3f5f8;
        --panel:#ffffff; --tile:#eaf0f8; --A:#0072b2; --C:#009e73; --D:#cc79a7; }
* { box-sizing:border-box; }
html, body { margin:0; padding:0; background:#888; }
body { font-family:"Inter","Helvetica Neue",Helvetica,Arial,sans-serif; color:var(--ink); }
.poster { width:841mm; height:1189mm; margin:0 auto; background:var(--bg); display:grid; overflow:hidden;
          grid-template-rows: 100mm 236mm 1fr 15mm; position:relative; }

/* header: title, authors, logos */
header { display:grid; grid-template-columns:1fr 150mm; column-gap:12mm; align-items:center; padding:0 22mm;
         background:linear-gradient(100deg,var(--navy2) 0%,var(--navy) 60%,#1565c0 100%); color:#fff; }
h1 { font-size:20mm; line-height:1.05; margin:0 0 4.5mm; font-weight:800; letter-spacing:-0.3mm; }
.authors { font-size:8.4mm; } .affil { font-size:6.2mm; color:#c9d6ee; margin-top:1.5mm; }
.logos { display:grid; grid-template-columns:1fr 1fr 1fr; gap:4mm; }
.ph { border:0.6mm dashed #9aa6b5; color:#7b8797; font-size:5.2mm; display:flex; align-items:center;
      justify-content:center; text-align:center; border-radius:2mm; background:#fafbfc; padding:2mm; }
.logos .ph { height:40mm; }

/* hero band: the picture and the take-home message, full width */
.hero { display:grid; grid-template-columns: 395mm 1fr; column-gap:10mm; align-items:center; padding:6mm 22mm 4mm;
        background:#fff; border-bottom:1.6mm solid var(--accent); position:relative; }
.heroimg { position:relative; height:224mm; }
.heroimg img { height:100%; display:block; }
.lab { position:absolute; font-size:7mm; font-weight:800; background:rgba(255,255,255,.88); padding:.8mm 2.2mm;
       border-radius:1.5mm; white-space:nowrap; }
.msg .kicker { font-size:8mm; font-weight:800; color:var(--accent); text-transform:uppercase; letter-spacing:0.6mm;
               margin-bottom:4mm; }
.msg .big { font-size:15.2mm; line-height:1.17; font-weight:800; color:var(--navy2); letter-spacing:-0.2mm; }
.msg .big em { font-style:normal; color:var(--accent); }
.msg .sub { font-size:8.6mm; line-height:1.35; color:var(--ink); margin-top:7mm; }

/* two columns, read top to bottom */
.body { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); column-gap:13mm; padding:11mm 22mm 0; min-height:0; }
.col { display:grid; row-gap:11mm; min-height:0; min-width:0; }
.panel { background:var(--panel); border-radius:5mm; padding:8mm 10mm 7mm; display:flex; flex-direction:column;
         min-height:0; min-width:0; box-shadow:0 .6mm 2mm rgba(20,30,50,.10); position:relative; }
h2 { font-size:11.6mm; line-height:1.12; margin:0 0 5mm; font-weight:800; letter-spacing:-0.1mm; display:flex; gap:4mm; }
h2 .n { flex:none; color:#fff; background:var(--navy); border-radius:50%; width:14mm; height:14mm; line-height:14mm;
        text-align:center; font-size:9mm; margin-top:-0.6mm; }
p, li { font-size:8.6mm; line-height:1.32; margin:0 0 3mm; }
ul { margin:0; padding-left:8mm; } li { margin-bottom:2mm; }
b.k { color:var(--accent); }
.fig { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
.fig img { max-width:100%; max-height:100%; object-fit:contain; display:block; }
.cap { font-size:6.6mm; color:var(--mute); line-height:1.3; margin-top:2mm; }
.split { display:grid; column-gap:7mm; flex:1; min-height:0; }
.stack { display:flex; flex-direction:column; min-height:0; min-width:0; }
.tiles { display:grid; gap:4mm; }
.tile { background:var(--tile); border-radius:3mm; padding:3.5mm 5mm; }
.tile .v { font-size:14mm; font-weight:800; color:var(--navy); line-height:1.02; white-space:nowrap; }
.tile .l { font-size:6.2mm; color:var(--mute); line-height:1.22; margin-top:1.2mm; }
.tiles.big .tile .v { font-size:17mm; }
.take { margin-top:4mm; background:#fff4ec; border-left:2.4mm solid var(--accent); padding:3mm 5mm; font-size:8.4mm;
        font-weight:700; line-height:1.3; border-radius:0 2mm 2mm 0; }
.arm b { font-weight:800; }
.summary { background:var(--navy2); color:#fff; }
.summary h2 { color:#fff; } .summary h2 .n { background:var(--accent); }
.summary li { font-size:8.6mm; } .summary b.k { color:#ffb27d; }
.summary .ph { background:transparent; color:#c9d6ee; border-color:#7f93b8; }
.check { list-style:none; padding-left:0; } .check li { padding-left:10mm; position:relative; }
.check li::before { content:"\2713"; position:absolute; left:0; color:#ffb27d; font-weight:800; }
.bars { margin:2mm 0 0; } .bar { display:flex; align-items:center; height:11mm; margin-bottom:2mm; }
.bar span { display:block; height:100%; border-radius:1.5mm; flex:none; }
.bar b { font-size:8.6mm; margin:0 3mm; } .bar i { font-size:6.6mm; font-style:normal; color:#c9d6ee; }
.layers { position:relative; height:212mm; margin-left:64mm; align-self:center; }
.layers img { height:100%; display:block; }
.ll { position:absolute; right:100%; transform:translateY(-50%); font-size:7.4mm; white-space:nowrap; padding-right:2mm;
      display:flex; align-items:center; gap:2mm; }
.ll::after { content:""; display:block; width:10mm; border-bottom:0.4mm solid #8a94a3; }
.foot { display:flex; align-items:center; justify-content:space-between; padding:0 22mm; font-size:5.6mm; color:var(--mute); }
.qr { width:36mm; height:36mm; flex:none; }
"""

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MX17 poster mock-up</title>
<meta name="viewport" content="width=device-width, initial-scale=0.2">
<style>{css}</style></head><body>
<div class="poster">

<header>
  <div>
    <h1>MX17: a resistive Micromegas TPC that measures each lepton's direction to 1&deg;</h1>
    <div class="authors">D. Neff<sup>1</sup>, &hellip; <i>(author list to be set)</i>, for the MX17 team and the n_TOF Collaboration</div>
    <div class="affil"><sup>1</sup> CEA Saclay, Irfu &nbsp;&middot;&nbsp; INFN / n_TOF co-author affiliations to be added</div>
  </div>
  <div class="logos"><div class="ph">CEA&nbsp;Irfu logo</div><div class="ph">n_TOF / CERN logo</div><div class="ph">INFN logo</div></div>
</header>

<section class="hero">
  <div class="heroimg">
    <img src="assets/hero_x17_box.png" alt="Four TPCs boxed around a target, one e+e- pair">
  </div>
  <div class="msg">
    <div class="kicker">The detector for an X17 search</div>
    <div class="big">X17&nbsp;&rarr;&nbsp;e<sup>+</sup>e<sup>&minus;</sup> is an <em>opening-angle</em> measurement.
      One 30&nbsp;mm gas gap gives each lepton's direction to <em>1&deg;</em>, for 2&times;10<sup>&minus;4</sup>&nbsp;X<sub>0</sub>.</div>
    <div class="sub">Four chambers close a box around the target; each lepton leaves a short 3D segment in one of them.
      First use: the n_TOF neutron beam, summer 2026.</div>
  </div>
</section>

<div class="body">
 <div class="col" style="grid-template-rows: 262mm 318mm 1fr;">

  <div class="panel">
    <h2><span class="n">1</span><span>A few-MeV lepton allows only ~1&deg;: build for 1&deg; and no mass</span></h2>
    <div class="fig"><img src="assets/fig_air.svg" alt="Scattering in the air before the gas against lepton energy"></div>
    <div class="cap">Below &asymp;14&nbsp;MeV the air in front of the chamber already blurs the direction by more than the
      chamber's own 1&deg;: a better chamber would not be seen. Everything after the gas does not matter.</div>
  </div>

  <div class="panel">
    <h2><span class="n">2</span><span>One gas gap, one board: a 3D segment per chamber</span></h2>
    <div class="split" style="grid-template-columns: minmax(0,1fr) 108mm;">
    <div class="layers">
      <img src="assets/chamber_bare.png" alt="Exploded view of the chamber">
      <div class="ll" style="top:7.1%">Drift cathode</div>
      <div class="ll" style="top:28.6%">30&nbsp;mm drift gap</div>
      <div class="ll" style="top:59.2%">Micromesh</div>
      <div class="ll" style="top:65.6%"><b class="k">Resistive strips</b></div>
      <div class="ll" style="top:71.8%">Pads</div>
      <div class="ll" style="top:78.0%">Y strips</div>
      <div class="ll" style="top:84.1%">X strips</div>
      <div class="ll" style="top:94.6%">Readout board</div>
    </div>
    <div class="stack" style="justify-content:center; gap:5mm;">
    <div class="tiles">
      <div class="tile"><div class="v">40&times;36&nbsp;cm&sup2;</div><div class="l">active area</div></div>
      <div class="tile"><div class="v">0.78&nbsp;mm</div><div class="l">pitch, X and Y from one pad plane</div></div>
      <div class="tile"><div class="v">2 &times; 512</div><div class="l">strips, full waveforms (DREAM)</div></div>
    </div>
    <div class="cap" style="font-size:7mm;">Drift time gives the third coordinate. Resistive strips keep the chamber spark-tolerant at the gain a 0.78&nbsp;mm strip needs.</div>
    </div>
    </div>
  </div>

  <div class="panel">
    <h2><span class="n">3</span><span>Fit the waveforms, not the hits</span></h2>
    <div class="fig"><img src="assets/waterfall_7884.png" alt="One muon as waveforms, with the model at the fit"></div>
    <div class="cap">One cosmic muon, both views. <b style="color:#1d5fa8;">Filled:</b> recorded waveform of each strip.
      <b style="color:#111827;">Black:</b> forward model at the fit (track, drift, diffusion, resistive sharing, electronics).
      <b style="color:#d9631e;">Orange:</b> the fitted track.</div>
  </div>
 </div>

 <div class="col" style="grid-template-rows: 300mm 318mm 1fr;">

  <div class="panel">
    <h2><span class="n">4</span><span>1&deg; per chamber, flat over &plusmn;20&deg;</span></h2>
    <div class="fig"><img src="assets/fig_angres.svg" alt="Angular resolution against reference angle"></div>
    <div class="tiles big" style="grid-template-columns:1fr 1fr 1fr; margin-top:3mm;">
      <div class="tile"><div class="v">94.8&nbsp;%</div><div class="l">efficiency, single tracks, full active area</div></div>
      <div class="tile"><div class="v">~120&nbsp;&micro;m</div><div class="l">intrinsic position resolution</div></div>
      <div class="tile"><div class="v">95.6&nbsp;%</div><div class="l">two tracks in one chamber, separated in both views</div></div>
    </div>
    <div class="cap">Cosmic muons against a reference telescope, chamber A, 25&thinsp;422 tracks.</div>
  </div>

  <div class="panel">
    <h2><span class="n">5</span><span>In the n_TOF beam, the tracks find the <sup>3</sup>He target</span></h2>
    <div class="split" style="grid-template-columns: 0.62fr 1.38fr; align-items:center;">
      <div class="stack">
        <div class="fig"><img src="assets/ntof_station.png" alt="The n_TOF station, rendered"></div>
        <div class="cap">EAR2 station: TPCs, trigger wall and scintillators around a <sup>3</sup>He capsule.</div>
      </div>
      <div class="fig"><img src="assets/ntof_fan.png" alt="Tracks of two arms extrapolated to the beam axis"></div>
    </div>
    <div class="take">Three chambers put the capsule at the same height:
      <span class="arm"><b style="color:var(--A)">A</b> &minus;30.7&plusmn;0.8,
      <b style="color:var(--C)">C</b> &minus;32.6&plusmn;2.1,
      <b style="color:var(--D)">D</b> &minus;34.1&plusmn;2.2&nbsp;mm</span></div>
  </div>

  <div class="panel summary">
    <h2><span class="n">6</span><span>Summary and next step</span></h2>
    <ul class="check">
      <li>1&deg; per lepton from a single 30&nbsp;mm gas gap, below the scattering in front of it.</li>
      <li>Reconstruction fits the full waveform image.</li>
      <li>Running in a neutron beam: the tracks image the target.</li>
    </ul>
    <p style="margin-top:2mm;"><b class="k">Next: repeat ATOMKI on a proton beam</b> (<sup>7</sup>Li(p,e<sup>+</sup>e<sup>&minus;</sup>)). Pairs near 140&deg; in the acceptance:</p>
    <div class="bars">
      <div class="bar"><span style="width:200mm; background:#ffb27d;"></span><b>~40&nbsp;%</b><i>four-TPC box</i></div>
      <div class="bar"><span style="width:12.5mm; background:#9aa6b5;"></span><b>~2.5&nbsp;%</b><i>multi-telescope spectrometer</i></div>
    </div>
    <div style="display:flex; gap:5mm; margin-top:auto; align-items:center;">
      <div class="ph qr">QR<br>online version</div><div class="ph qr">QR<br>3D event display</div>
      <div style="font-size:6.4mm; color:#c9d6ee;">Two papers in preparation: the detector; the reconstruction.<br>Contact: <i>e-mail</i></div>
    </div>
  </div>
 </div>
</div>

<div class="foot"><span>INFN&ndash;IN2P3 Detector R&amp;D Conference &middot; LNF Frascati &middot; 2&ndash;4 December 2026</span>
  <span>MOCK-UP v3 &middot; 2026-11-18 &middot; numbers from the Paper I draft, to be reconciled before printing</span></div>
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', action='store_true', help='also print poster.pdf and poster_preview.png (needs google-chrome)')
    ap.add_argument('--no-assets', action='store_true', help='keep assets/ as it is')
    a = ap.parse_args()
    if not a.no_assets:
        prepare_assets()
    out = os.path.join(HERE, 'poster.html')
    with open(out, 'w') as f:
        f.write(HTML.replace('{css}', CSS))
    print('wrote', out)
    if a.pdf:
        pdf = os.path.join(HERE, 'poster.pdf')
        subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--print-to-pdf=' + pdf, 'file://' + out], check=True, capture_output=True)
        subprocess.run(['pdftoppm', '-r', '40', '-png', '-singlefile', pdf, os.path.join(HERE, 'poster_preview')],
                       check=True)
        print('wrote', pdf, 'and poster_preview.png')


if __name__ == '__main__':
    main()
