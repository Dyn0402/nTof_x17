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
    'muon_data_model.png': 'papers/paper1_v2/figures/muon_data_model.pdf',
    'angle_resolution_v2.png': 'papers/paper1_v2/figures/angle_resolution_v2.pdf',
    'two_track_separation.png': 'papers/paper1_v2/figures/two_track_separation.pdf',
    'efficiency_map.png': 'papers/paper1_v2/figures/efficiency_map.pdf',
    'ntof_photo.jpg': 'mpgd26/slides/assets/img/photo_station_topdown.jpg',
    'ntof_pointing.png': 'ntof_athens_26/figures/sideview_run145.png, left half',
    'hero_proton_box.png': 'make_hero.py --scene proton',
    'pair_event_24576.png': 'https://dylan-neff.web.cern.ch/files/x17_pair_24576.png (sept26_prelim_analysis/pair_event_display.py); pair_event_3d.png is its 3D part, cropped',
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
    shutil.copy(os.path.join(REPO, 'mpgd26/slides/assets/img/photo_station_topdown.jpg'),
                os.path.join(A, 'ntof_photo.jpg'))
    # the track fan only (left half); the height histogram beside it is too small to read in this panel
    im = Image.open(os.path.join(REPO, 'ntof_athens_26/figures/sideview_run145.png'))
    im.crop((0, int(0.10 * im.height), int(0.553 * im.width), int(0.90 * im.height))).save(os.path.join(A, 'ntof_pointing.png'))
    for f in ('pointing', 'muon_data_model', 'angle_resolution_v2', 'two_track_separation', 'efficiency_map'):
        subprocess.run(['pdftoppm', '-r', '250', '-png', '-singlefile',
                        os.path.join(REPO, 'papers/paper1_v2/figures', f + '.pdf'), os.path.join(A, f)], check=True)
    ev = os.path.join(A, 'pair_event_24576.png')
    if os.path.exists(ev):
        Image.open(ev).crop((0, 112, 1040, 855)).save(os.path.join(A, 'pair_event_3d.png'))
    for f in ('hero_x17_box.png', 'hero_proton_box.png', 'readout_stack_3d.png'):
        if os.path.exists(os.path.join(A, f)):
            trim(os.path.join(A, f))


CSS = r"""
@page { size: 841mm 1189mm; margin: 0; }
:root { --ink:#16202e; --mute:#5b6675; --navy:#0b3d91; --accent:#d9631e; --rule:#d5dbe3; --panel:#ffffff;
        --bg:#f3f5f8; --tile:#eaf0f8; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:#888; }
body { font-family: "Inter","Helvetica Neue",Helvetica,Arial,sans-serif; color:var(--ink); }
.poster { width:841mm; height:1189mm; background:var(--bg); margin:0 auto; padding:0 22mm 13mm; display:grid;
          grid-template-rows: 96mm 300mm 258mm 226mm 1fr; row-gap:10mm; overflow:hidden; position:relative; }
header { display:grid; grid-template-columns: 1fr 150mm; column-gap:14mm; align-items:center;
         background:linear-gradient(100deg,#0a2a66 0%,#0b3d91 55%,#1565c0 100%); margin:0 -22mm; padding:0 22mm;
         border-bottom:2.4mm solid var(--accent); }
h1 { font-size:21.5mm; line-height:1.06; margin:0 0 5mm; letter-spacing:-0.3mm; font-weight:800; color:#fff; }
.authors { font-size:8.6mm; color:#fff; } .affil { font-size:6.4mm; color:#c9d6ee; margin-top:1.5mm; }
.logos { display:grid; grid-template-columns:1fr 1fr 1fr; gap:4mm; }
.ph { border:0.6mm dashed #9aa6b5; color:#7b8797; font-size:5.4mm; display:flex; align-items:center;
      justify-content:center; text-align:center; border-radius:2mm; background:#fafbfc; padding:2mm; }
.logos .ph { height:42mm; }
.evt img { border-radius:3mm; }
.small li { font-size:7.5mm; margin-bottom:1.5mm; }
.prelim { position:absolute; right:3mm; bottom:3mm; font-size:5mm; font-weight:700; color:#5b6675; background:rgba(255,255,255,.8); padding:.5mm 2mm; border-radius:1mm; }
.row { display:grid; column-gap:11mm; min-height:0; }
.panel { background:var(--panel); border-radius:5mm; padding:9mm 10mm 8mm; display:flex; flex-direction:column;
         min-height:0; min-width:0; box-shadow:0 0.6mm 2mm rgba(20,30,50,.10); position:relative; }
h2 { font-size:12.4mm; margin:0 0 4mm; font-weight:800; letter-spacing:-0.1mm; }
h2 .n { display:inline-block; color:#fff; background:var(--navy); border-radius:50%; width:15mm; height:15mm;
        line-height:15mm; text-align:center; font-size:9.5mm; margin-right:3.5mm; vertical-align:1mm; }
p, li { font-size:8.2mm; line-height:1.33; margin:0 0 3mm; }
ul { margin:0; padding-left:8mm; } li { margin-bottom:1.8mm; }
b.k { color:var(--accent); }
.fig { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
.fig img { max-width:100%; max-height:100%; object-fit:contain; display:block; }
.cap { font-size:6.3mm; color:var(--mute); line-height:1.3; margin-top:2.5mm; }
.cols { display:grid; column-gap:8mm; flex:1; min-height:0; }
.col { display:flex; flex-direction:column; min-height:0; min-width:0; }
.tiles { display:grid; gap:4mm; }
.tile { background:var(--tile); border-radius:3mm; padding:4mm 5mm; }
.tile .v { font-size:14.5mm; font-weight:800; color:var(--navy); line-height:1.05; }
.tile .l { font-size:5.5mm; color:var(--mute); line-height:1.25; margin-top:1mm; }
.lab { position:absolute; font-size:6.6mm; font-weight:700; background:rgba(255,255,255,.88); padding:1mm 2.5mm;
       border-radius:1.5mm; white-space:nowrap; }
.hero { position:relative; flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
.heroimg { position:relative; height:100%; aspect-ratio:1759/1678; max-width:100%; }
.heroimg img { width:100%; height:100%; display:block; }
.tiles.t3 .tile .v { font-size:11.5mm; }
.photo img { object-fit:cover; width:100%; height:100%; border-radius:3mm; }
.mock { position:absolute; right:22mm; bottom:3.5mm; font-size:5mm; color:#8a94a3; }
.foot { font-size:5.6mm; color:var(--mute); }
.qr { width:44mm; height:44mm; flex:none; }
"""

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MX17 poster mock-up</title>
<meta name="viewport" content="width=device-width, initial-scale=0.2">
<style>{css}</style></head><body>
<div class="poster">

<header>
  <div>
    <h1>MX17: a low-mass resistive Micromegas TPC<br>to track e<sup>+</sup>e<sup>&minus;</sup> pairs in X17 searches</h1>
    <div class="authors">D. Neff<sup>1</sup>, &hellip; <i>(author list to be set)</i>, for the MX17 team and the n_TOF Collaboration</div>
    <div class="affil"><sup>1</sup> CEA Saclay, Irfu &nbsp;&middot;&nbsp; INFN / n_TOF co-author affiliations to be added</div>
  </div>
  <div class="logos"><div class="ph">CEA&nbsp;Irfu logo</div><div class="ph">n_TOF / CERN logo</div><div class="ph">INFN logo</div></div>
</header>

<!-- ROW 1: the search -->
<div class="row" style="grid-template-columns: 1.52fr 1fr;">
  <div class="panel">
    <h2><span class="n">1</span>An X17 search is an opening-angle measurement</h2>
    <div class="hero"><div class="heroimg">
      <img src="assets/hero_x17_box.png" alt="Four TPCs boxed around a target, one e+e- pair">
      <span class="lab" style="left:0%; top:4%; color:#c9772a;">scintillators (trigger, energy)</span>
      <span class="lab" style="left:20%; top:24%; color:#2c7a67;">TPC readout plane</span>
      <span class="lab" style="left:71%; top:50%; color:#3d86a1;">30&nbsp;mm drift gap</span>
      <span class="lab" style="left:52%; top:11%; color:#4b5563;">beam</span>
      <span class="lab" style="left:52%; top:47%; color:#1a1f2b;">target</span>
      <span class="lab" style="left:24%; top:33%; color:#7a3fb0;">e<sup>+</sup></span>
      <span class="lab" style="left:65%; top:23%; color:#d94f1e;">e<sup>&minus;</sup></span>
      <span class="lab" style="left:56%; top:90%; color:#6b7280; font-weight:500;">near arms drawn as outlines</span>
    </div></div>
    <div class="cap">A generalised search: a target in a beam, four TPCs 20&nbsp;cm away closing a box around it, scintillators
    behind. Each lepton leaves a short 3D track segment in the gas of one chamber. Idealised layout, drawn to scale.</div>
  </div>
  <div class="panel">
    <h2><span class="n">2</span>What the tracker must do</h2>
    <ul>
      <li>X17 would appear as an excess of e<sup>+</sup>e<sup>&minus;</sup> pairs at <b class="k">large opening angle</b>, on a
          falling internal-pair-creation continuum.</li>
      <li>The leptons carry a few MeV: they scatter in everything. The tracker has to be <b class="k">almost nothing</b>
          before the measurement is made.</li>
      <li>One chamber sees a short segment far from the target, so the quantity that matters is the
          <b class="k">track angle</b>, not the position.</li>
    </ul>
    <div class="fig"><img src="assets/pointing.png" alt="Error on the initial direction against lepton energy"></div>
    <div class="cap">A 1&deg; chamber is already below the scattering in the air and window in front of it for leptons
    under about 12&nbsp;MeV: a better angle would not be seen.</div>
  </div>
</div>

<!-- ROW 2: the chamber -->
<div class="row" style="grid-template-columns: 1fr;">
  <div class="panel">
    <h2><span class="n">3</span>The chamber: one gas gap, one board, a 3D segment</h2>
    <div class="cols" style="grid-template-columns: 1.25fr 0.95fr 0.8fr;">
      <div class="col">
        <div class="fig"><img src="assets/chamber_exploded.png" alt="Exploded view of the chamber"></div>
        <div class="cap">Layer stack, exploded along the drift axis, with a track and its drifting ionisation
        (in-plane dimensions real, thicknesses exaggerated).</div>
      </div>
      <div class="col">
        <div class="fig"><img src="assets/readout_stack_3d.png" alt="Readout stack, 16-pad patch"></div>
        <div class="cap">A 16-pad patch of the readout board from the gerbers: mesh, resistive strips, the pad
        plane, and the vias that take alternate pads to the X strips (orange) and the Y strips (green).</div>
      </div>
      <div class="col">
        <div class="tiles" style="grid-template-columns:1fr 1fr;">
          <div class="tile"><div class="v">40&times;36</div><div class="l">cm&sup2; active area</div></div>
          <div class="tile"><div class="v">30 mm</div><div class="l">drift gap, Ar/iC<sub>4</sub>H<sub>10</sub> 95/5</div></div>
          <div class="tile"><div class="v">0.78 mm</div><div class="l">pitch, 512 + 512 strips</div></div>
          <div class="tile"><div class="v">2.4&times;10<sup>&minus;4</sup></div><div class="l">X<sub>0</sub> of gas along the gap</div></div>
        </div>
        <ul style="margin-top:6mm;">
          <li>Bulk Micromegas with <b class="k">resistive strips</b>: spark-tolerant at the gain a 0.78&nbsp;mm strip needs.</li>
          <li>One plane of pads, shared between <b class="k">two strip views</b> through vias: X and Y see the same
              avalanche and carry equal charge.</li>
          <li>The drift time gives the third coordinate (&micro;TPC): a 3D segment from a single thin chamber.</li>
          <li>DREAM front end, full waveforms kept.</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<!-- ROW 3: reconstruction + performance -->
<div class="row" style="grid-template-columns: 1fr 1.45fr;">
  <div class="panel">
    <h2><span class="n">4</span>Fit the picture, not the hits</h2>
    <p>Each view is a strip&nbsp;&times;&nbsp;time image. The reconstruction fits a <b class="k">forward model</b> of that
    whole image: a straight track, drift and diffusion, charge sharing on the resistive layer, the electronics
    response.</p>
    <div class="fig"><img src="assets/muon_data_model.png" alt="One muon: data and model at the fit"></div>
    <div class="cap">One cosmic muon, both views: the recorded waveforms (left) and the model at the fitted track (right).</div>
  </div>
  <div class="panel">
    <h2><span class="n">5</span>Performance on cosmic muons and in the SPS beam</h2>
    <div class="cols" style="grid-template-columns: 1.15fr 1fr;">
      <div class="col">
        <div class="fig"><img src="assets/angle_resolution_v2.png" alt="Angular resolution"></div>
        <div class="cap">Fitted against reference angle, one chamber: flat over &plusmn;20&deg;.</div>
      </div>
      <div class="col">
        <div class="tiles t3" style="grid-template-columns:1fr 1fr 1fr;">
          <div class="tile"><div class="v">1.0&deg; / 0.9&deg;</div><div class="l">angle, X / Y view, single chamber</div></div>
          <div class="tile"><div class="v">94.8 %</div><div class="l">single tracks, full active area</div></div>
          <div class="tile"><div class="v">~120 &micro;m</div><div class="l">intrinsic position resolution</div></div>
          <div class="tile"><div class="v">40 ns</div><div class="l">event time</div></div>
          <div class="tile"><div class="v">97.7 %</div><div class="l">two particles paired correctly (equal X/Y charge)</div></div>
          <div class="tile"><div class="v">95.6 %</div><div class="l">second particle in the chamber, separate in both views</div></div>
        </div>
        <div class="fig" style="margin-top:4mm;"><img src="assets/two_track_separation.png" alt="Two-track separation"></div>
        <div class="cap">Two tracks in one chamber, against their separation in the closer view.</div>
      </div>
    </div>
  </div>
</div>

<!-- ROW 4: n_TOF + proton beams -->
<div class="row" style="grid-template-columns: 1.22fr 1fr;">
  <div class="panel">
    <h2><span class="n">6</span>First use: n_TOF EAR2, summer 2026</h2>
    <div class="cols" style="grid-template-columns: 0.46fr 1fr 1fr;">
      <div class="col photo"><div class="fig"><img src="assets/ntof_photo.jpg" alt="The four-arm station at n_TOF"></div></div>
      <div class="col evt">
        <div class="fig" style="position:relative;"><img src="assets/pair_event_3d.png" alt="3D display of a pair candidate">
          <span class="prelim">PRELIMINARY</span></div>
        <div class="cap">One event in 3D: two track segments in one chamber, 42&deg; apart, meeting within 2&nbsp;mm,
        each pointing on to a fired scintillator.</div>
      </div>
      <div class="col">
        <div class="fig"><img src="assets/ntof_pointing.png" alt="Tracks pointing back at the 3He capsule"></div>
        <div class="cap">Four chambers around a <sup>3</sup>He capsule in the neutron beam (photo: seen along the beam).
        6207 segments extrapolated to the beam axis image the capsule, to scale.</div>
      </div>
    </div>
  </div>
  <div class="panel">
    <h2><span class="n">7</span>Next: the ATOMKI measurement, on a proton beam</h2>
    <div class="cols" style="grid-template-columns: 0.82fr 1fr;">
      <div class="col">
        <div class="hero"><div class="heroimg" style="aspect-ratio:2400/2000;">
          <img src="assets/hero_proton_box.png" alt="The same box around a thin target on a proton beam">
          <span class="lab" style="left:50%; top:80%; color:#e0457b; font-size:5.6mm;">p, ~1 MeV</span>
          <span class="lab" style="left:36%; top:33%; color:#1a1f2b; font-size:5.6mm;">140&deg;</span>
          <span class="lab" style="left:58%; top:2%; color:#6b7280; font-size:5.6mm;">cosmic &mu;</span>
        </div></div>
        <div class="cap">The same box around a thin Li target at the end of a beam pipe.</div>
      </div>
      <div class="col small">
        <ul>
          <li>The anomaly was seen with protons: <sup>7</sup>Li(p,e<sup>+</sup>e<sup>&minus;</sup>)<sup>8</sup>Be, then
              <sup>3</sup>H and <sup>11</sup>B targets. The chambers need only a target and a ~1&nbsp;MeV beam.</li>
          <li><b class="k">~40 % of pairs near 140&deg;</b> fall in the box, against ~2.5 % for a multi-telescope
              spectrometer.</li>
          <li><b class="k">Every lepton is a 3D segment that points at the target</b>: a vertex, and cosmic muons,
              which can fake a bump near 140&deg;, are rejected.</li>
          <li><b class="k">Thin</b>: the angle is measured in gas, before any scintillator.</li>
          <li>Geant4 estimate, 1&nbsp;&micro;A on Li<sub>2</sub>O: the ATOMKI signal at 3&sigma; in about two weeks as
              built, about a day with larger scintillators.</li>
        </ul>
        <div style="display:flex; gap:5mm; margin-top:auto; align-items:center;">
          <div class="ph qr" style="width:34mm;height:34mm;">QR<br>3D event display</div>
          <div class="ph qr" style="width:34mm;height:34mm;">QR<br>3D readout board</div>
          <div class="foot">Two papers in preparation:<br>the detector; the reconstruction.<br>Contact: <i>e-mail</i></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="mock">MOCK-UP v1 &middot; 2026-11-18 &middot; numbers from the Paper I draft, to be reconciled before printing</div>
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
