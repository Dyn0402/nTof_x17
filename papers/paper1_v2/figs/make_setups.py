#!/usr/bin/env python3
"""make_setups.py -- Paper I v2: the cosmic-bench drawing and the beam-test drawing in one figure (agent, 2026-11-10).

    .venv/bin/python papers/paper1_v2/figs/make_setups.py

Runs the two existing drawing scripts (papers/paper1/figs/make_bench_setup.py, make_sps_setup.py) unchanged except for
text patches applied to their source in memory: they draw into one shared figure (bench on top, beam below), their own
save calls are dropped, the bench labels read "reference planes" instead of "M3 reference planes", and (a)/(b) are added.
Scale bars, the z axes and the "to scale" notes are the originals'.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'papers' / 'paper1' / 'figs'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / 'papers')]
import paper_style as ps                                           # noqa: E402

H_B, H_S, GAP = 2.55, 1.85, 0.12
W, H = 6.1, H_B + H_S + GAP
fig = plt.figure(figsize=(W, H))


def load(name, yoff, hh):
    s = (SRC / name).read_text()
    s = s.replace("sys.path[:0] = [str(HERE.parents[2] / 'papers')]", "sys.path[:0] = [str(REPO_ROOT / 'papers')]")
    s = re.sub(r"fig = plt\.figure\(figsize=\(W, H\)\)", "W, H = W_ALL, H_ALL", s)
    s = s.replace("W, H = AX_W + 0.1, 2.55", "")
    s = s.replace("fig.add_axes([0.05 / W, 0.05 / H, AX_W / W, AX_H / H])",
                  f"fig.add_axes([0.05 / W, ({yoff} + 0.05) / H, AX_W / W, AX_H / H])")
    s = s.replace("'M3 reference planes", "'reference planes")
    # 2026-11-20 (Dylan): give the transverse sizes on the drawing. Every element is square; the slabs are already
    # drawn to scale in x, so only the labels change. The bench panel is widened to the right (into the spare width
    # of the shared figure) so that each size fits on its label's line.
    s = s.replace("XL, XR, YB, YT = -640, 700, -95, 560", "XL, XR, YB, YT = -640, 945, -95, 560")
    s = s.replace("'trigger scintillator'", "'trigger scintillator, 60 × 60 cm'")
    s = s.replace("'reference planes\\n(2 + 2)'", "'reference planes, 50 × 50 cm\\n(2 + 2)'")
    s = s.replace("((M3_Z[0], M3_Z[1]), 'reference planes')", "((M3_Z[0], M3_Z[1]), 'reference planes, 50 × 50 cm')")
    s = s.replace("'test chamber (top slot)'", "'test chamber (top slot), 40 × 40 cm'")
    s = s.replace("'test chamber (bottom slot)'", "'test chamber (bottom slot), 40 × 40 cm'")
    s = re.sub(r"^fig\.savefig\(.*$", "", s, flags=re.M)
    s = re.sub(r"^fig\.canvas\.draw\(\).*$", "", s, flags=re.M)
    s = re.sub(r"^ps\.save\(.*$", "", s, flags=re.M)
    s = s.replace("W, H = 6.1, 1.85", "")
    g = dict(__file__=str(SRC / name), __name__='drawing', fig=fig, W_ALL=W, H_ALL=H, REPO_ROOT=REPO)
    exec(compile(s, name, 'exec'), g)
    return g


# bench first (it sets W/H from its own axes width; both then use the shared W_ALL / H_ALL)
gb = load('make_bench_setup.py', H_S + GAP, H_B)
gs = load('make_sps_setup.py', 0.0, H_S)
fs = ps.BASE_PT
fig.text(0.0, 1.0, '(a)', fontsize=fs, fontweight='bold', color=ps.INK, ha='left', va='top')
fig.text(0.0, (H_S + 0.02) / H, '(b)', fontsize=fs, fontweight='bold', color=ps.INK, ha='left', va='top')
out = HERE / 'out' / 'setups'
ps.save(fig, out)
print('ok', out)
