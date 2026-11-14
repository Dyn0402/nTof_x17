#!/usr/bin/env python3
"""make_readout_stack_3d.py -- Paper I v2, Fig. 3(c): a 16-pad patch of the readout stack in 3D (2026-11-14).

    .venv/bin/python papers/paper1_v2/figs/make_readout_stack_3d.py              # render (headless Chrome) + compose
    .venv/bin/python papers/paper1_v2/figs/make_readout_stack_3d.py --no-render  # compose from the cached render

The model is the "Peeled" view of the readout-stack 3D note (dylan-cern-site pages/notes/mx17-readout-stack-3d.html,
site commit 7a3a071, copied to render3d/ so this script does not depend on the site repo): a 3.1 x 3.1 mm patch at the
centre of the board, mesh to L7, positions from the production gerbers, z stretched 2.2x, orthographic, FR4 hidden,
each layer above the pads starting one pad row further back.  render3d/figmode.js strips the page down to the viewer
and reports where the note puts each layer label; headless Chrome renders the WebGL view at DSF x the CSS size (the
image is embedded at full resolution, ~600 dpi at print size).  Labels, legend and the panel letter are drawn here in
the paper's font, so the panel matches (a) and (b).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1])]
import paper_style as ps                                            # noqa: E402

R3D = HERE / 'render3d'
OUT = HERE / 'out' / 'readout_stack_3d'
RENDER = HERE / 'out' / 'readout_stack_3d_render.png'
LABELS = HERE / 'out' / 'readout_stack_3d_render.json'
VW, VH, DSF = 760, 560, 4                                           # viewer CSS px, device scale factor
CHROME = ['google-chrome', '--headless=new', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
          '--virtual-time-budget=8000', '--hide-scrollbars', f'--window-size={VW + 40},{VH + 140}']
YC, XC, INK, MUTED = '#009E73', ps.ACCENT, ps.INK, ps.MUTED
FS = ps.BASE_PT * 0.85


def light(c, f=0.5):                                                # the note's "land without a stub": half to white
    return tuple(x + (1 - x) * f for x in matplotlib.colors.to_rgb(c))


def render():
    src = (R3D / 'mx17-readout-stack-3d.html').read_text()
    cap = 'renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));'
    assert cap in src, 'note changed: pixel-ratio cap not found'
    src = src.replace(cap, 'renderer.setPixelRatio(window.devicePixelRatio || 1);')
    src = src.replace('</head>', '<script>' + (R3D / 'figmode.js').read_text() + '</script>\n</head>', 1)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / 'page.html'
        page.write_text(src)
        url = f'file://{page}?w={VW}&h={VH}'
        subprocess.run(CHROME + [f'--force-device-scale-factor={DSF}', f'--screenshot={RENDER}', url + '&shot=1'],
                       check=True, capture_output=True, timeout=180)
        dom = subprocess.run(CHROME + ['--dump-dom', url], check=True, capture_output=True, text=True, timeout=180).stdout
    m = re.search(r'<pre id="labjson">(\{.*?\})</pre>', dom)
    assert m, 'no label positions in the dumped page'
    LABELS.write_text(m.group(1))


if '--no-render' not in sys.argv:
    render()
lab = json.loads(LABELS.read_text())
img = plt.imread(RENDER)[..., :3]
x0v, y0v = round(lab['x0'] * DSF), round(lab['y0'] * DSF)          # the viewer inside the screenshot
img = img[y0v:y0v + VH * DSF, x0v:x0v + VW * DSF]
ink = np.where((img < 0.97).any(axis=2))
pad = 6
r0, r1 = max(ink[0].min() - pad, 0), ink[0].max() + pad
c0, c1 = max(ink[1].min() - pad, 0), ink[1].max() + pad
img = img[r0:r1, c0:c1]
ih, iw = img.shape[:2]

ps.use()
plt.rcParams['savefig.pad_inches'] = 0.01
W, H = 6.5, 2.95                                                     # text width (a4, 2.2 cm margins): (c) lines up with (a)
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
MH = H - 0.06                                                       # model height on the page (in)
MW = MH * iw / ih
MX0 = 1.85                                                          # model left edge
s = MH / ih                                                         # in per image pixel
ax.imshow(img, extent=[MX0, MX0 + MW, 0.03, 0.03 + MH], interpolation='none', zorder=1)
ax.set_xlim(0, W); ax.set_ylim(0, H)


def to_in(xc, yc):                                                  # note CSS px (viewer) -> figure inches
    return MX0 + (xc * DSF - c0) * s, 0.03 + MH - (yc * DSF - r0) * s


NAME = {'mesh (0 V)': ('mesh (0 V)', INK), 'amplification gap': ('amplification gap', INK),
        'resistive strips': ('resistive strips (+HV)', INK), 'kapton + glue': ('kapton + glue', INK),
        'L4  pads': ('pads (L4)', INK), 'L5  Y strips': ('Y strips (L5)', YC),
        'L6  X strips': ('X strips (L6)', XC), 'L7  via lands only': ('via lands (L7)', INK)}
pts = sorted(((to_in(l['x'], l['y']), *NAME[l['text']]) for l in lab['labels']), key=lambda t: -t[0][1])
RX = MX0 + MW + 0.22                                                # label column
GAP = 0.165                                                         # minimum spacing between label lines (in)
ys = []
for (x, y), _, _ in pts:                                            # push down where two labels would overlap
    ys.append(min(y, ys[-1] - GAP) if ys else y)
for ((x, y), text, col), yl in zip(pts, ys):
    ax.plot([x + 0.02, RX - 0.05], [y, yl], color=MUTED, lw=0.45, zorder=3)
    ax.plot(x + 0.02, y, 'o', ms=1.6, color=MUTED, mew=0, zorder=3)
    ax.text(RX, yl, text, ha='left', va='center', fontsize=FS, color=col, zorder=4)

# legend (left): pad colour = the strip its via feeds; lighter land = no stub on that layer
LX, LY = 0.02, 2.35
ent = [(YC, 'pad, via and land\non its Y strip (L5)'), (XC, 'pad, via and land\non its X strip (L6)'),
       (light(YC), None), (light(XC), 'land on the other\nlayers (no stub)')]
y = LY
for i, (c, t) in enumerate(ent):
    if i == 2:                                                      # the two light lands share one line
        ax.add_patch(Circle((LX + 0.06, y), 0.045, fc=c, ec='none', zorder=4))
        continue
    if i == 3:
        ax.add_patch(Circle((LX + 0.17, y), 0.045, fc=c, ec='none', zorder=4))
        ax.text(LX + 0.28, y, t, ha='left', va='center', fontsize=FS, color=INK, linespacing=1.05)
        continue
    ax.add_patch(Rectangle((LX + 0.015, y - 0.045), 0.09, 0.09, fc=c, ec='none', zorder=4))
    ax.text(LX + 0.17, y, t, ha='left', va='center', fontsize=FS, color=INK, linespacing=1.05)
    y -= 0.38
ax.text(LX, 0.30, 'centre of the board,\n4 $\\times$ 4 pads; $z$ stretched\n2.2$\\times$ against $x$, $y$',
        ha='left', va='center', fontsize=FS, color=MUTED, linespacing=1.1)
ax.text(0.0, H, '(c)', ha='left', va='top', fontsize=ps.BASE_PT, fontweight='bold', color=INK)
fig.canvas.draw()
ps.save(fig, OUT, dpi=600)
print('ok', OUT, 'image', iw, 'x', ih, 'px ->', round(MW, 2), 'x', round(MH, 2), 'in,', round(iw / MW), 'dpi')
