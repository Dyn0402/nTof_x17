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
    for f in ('hero_x17_box.png', 'ntof_station.png', 'readout_stack_3d.png', 'waterfall_7884.png'):
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
.bars { margin:1mm 0 5mm; } .bl { font-size:6.4mm; color:var(--mute); margin-bottom:1.5mm; }
.bar { position:relative; height:13mm; margin-bottom:2mm; display:flex; align-items:center; }
.bar span { display:block; height:100%; border-radius:1.5mm; flex:none; max-width:46%; }
.bar b { font-size:9mm; margin:0 3mm; color:var(--navy); } .bar i { font-size:6.4mm; color:var(--mute); font-style:normal; }
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
.tiles.t3 .tile .v { font-size:9.6mm; white-space:nowrap; }
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
    <div class="fig"><img src="assets/hero_x17_box.png" alt="Four TPCs boxed around a target, one e+e- pair"></div>
    <div class="cap">A generalised search: a target in a beam, four TPCs 20&nbsp;cm away closing a box around it, scintillators
    behind. Each lepton leaves a short 3D track segment in the gas of one chamber. Idealised layout, drawn to scale; the two arms nearest the viewer are outlines.</div>
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
<div class="row" style="grid-template-columns: 1.12fr 1.3fr;">
  <div class="panel">
    <h2><span class="n">4</span>Fit the picture, not the hits</h2>
    <p>Each view is a strip&nbsp;&times;&nbsp;time image. The reconstruction fits a <b class="k">forward model</b> of that
    whole image: a straight track, drift and diffusion, charge sharing on the resistive layer, the electronics
    response.</p>
    <div class="fig"><img src="assets/waterfall_7884.png" alt="One muon as waveforms, with the model at the fit"></div>
    <div class="cap"><span style="color:#1d5fa8;font-weight:700;">Filled:</span> the recorded waveform of each strip, one cosmic
    muon, both views. <b style="color:#111827;">Black:</b> the model at the fit.
    <b style="color:#d9631e;">Orange:</b> the fitted track, strip position against arrival time.</div>
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
<div class="row" style="grid-template-columns: 1.55fr 1fr;">
  <div class="panel">
    <h2><span class="n">6</span>First use: n_TOF EAR2, summer 2026</h2>
    <div class="cols" style="grid-template-columns: 0.9fr 1.5fr;">
      <div class="col">
        <div class="fig"><img src="assets/ntof_station.png" alt="The n_TOF station, rendered"></div>
        <div class="cap">The station as built (two arms shown solid): TPC, trigger wall, plastic and liquid
        scintillators around a <sup>3</sup>He capsule; one simulated pair.</div>
      </div>
      <div class="col">
        <div class="fig"><img src="assets/ntof_pointing.png" alt="Tracks pointing back at the 3He capsule"></div>
        <div class="cap">Track segments extrapolated back to the beam axis image the capsule: three chambers agree
        on its height to a few mm.</div>
      </div>
    </div>
  </div>
  <div class="panel">
    <h2><span class="n">7</span>Next: repeat ATOMKI on a proton beam</h2>
    <div class="small" style="display:flex; flex-direction:column; flex:1; min-height:0;">
      <p style="font-size:7.5mm;">The anomaly was seen with protons: <sup>7</sup>Li(p,e<sup>+</sup>e<sup>&minus;</sup>)<sup>8</sup>Be,
      then <sup>3</sup>H and <sup>11</sup>B. The same box needs only a thin target and a ~1&nbsp;MeV beam.</p>
      <div class="bars">
        <div class="bl">Pairs near 140&deg; inside the acceptance</div>
        <div class="bar"><span style="width:100%; background:var(--navy);"></span><b>~40 %</b><i>four-TPC box</i></div>
        <div class="bar"><span style="width:6.25%; background:#9aa6b5;"></span><b>~2.5 %</b><i>multi-telescope spectrometer</i></div>
      </div>
      <ul>
        <li><b class="k">Every lepton points at the target</b>: a vertex for each pair, and cosmic muons, which can
            fake a bump near 140&deg;, are rejected.</li>
        <li><b class="k">Thin</b>: the angle is measured in gas, before any scintillator.</li>
        <li>Geant4 estimate, 1&nbsp;&micro;A on Li<sub>2</sub>O: the ATOMKI signal at 3&sigma; in about two weeks as built,
            about a day with larger scintillators.</li>
      </ul>
      <div style="display:flex; gap:5mm; margin-top:auto; align-items:center;">
        <div class="ph qr" style="width:32mm;height:32mm;">QR<br>3D event display</div>
        <div class="ph qr" style="width:32mm;height:32mm;">QR<br>3D readout board</div>
        <div class="foot">Two papers in preparation: the detector; the reconstruction.<br>Contact: <i>e-mail</i></div>
      </div>
    </div>
  </div>
</div>

<div class="mock">MOCK-UP v2 &middot; 2026-11-18 &middot; numbers from the Paper I draft, to be reconciled before printing</div>
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
