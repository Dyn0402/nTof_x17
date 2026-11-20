#!/usr/bin/env python3
"""make_setups.py -- Paper I v2: the cosmic-bench drawing and the beam-test drawing in one figure (agent, 2026-11-10).

    .venv/bin/python papers/paper1_v2/figs/make_setups.py

Runs the two existing drawing scripts (papers/paper1/figs/make_bench_setup.py, make_sps_setup.py) unchanged except for
text patches applied to their source in memory: they draw into one shared figure (bench on top, beam below), their own
save calls are dropped, the bench labels read "reference planes" instead of "M3 reference planes", and (a)/(b) are added.
Scale bars, the z axes and the "to scale" notes are the originals'.
2026-11-20 (Dylan): the bench panel is widened to the figure width in x only (z compression 0.35 -> 0.24, printed at
run time; the caption quotes it), carries thin dimension lines (60/50/40 cm), and its muon label moves right of the track.
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
BENCH_W = 5.95                     # bench axes width, in (2026-11-20: the whole figure width)
fig = plt.figure(figsize=(W, H))


DIMS = '''
DIM_C = '#8a8f98'
def dim(w, ymid, lab, side, xl=0):
    """A thin dimension line of length w (plot units = mm), centred on x = 0 at height ymid; the label just outside
    it, on the side away from the detector (side = +1 above, -1 below), no background."""
    kw = dict(color=DIM_C, lw=0.45, zorder=4, solid_capstyle='butt')
    ax.plot([-w / 2, w / 2], [ymid, ymid], **kw)
    for xe in (-w / 2, w / 2):
        ax.plot([xe, xe], [ymid - 5, ymid + 5], **kw)
    ax.text(xl, ymid + side * 4, lab, ha='center', va='bottom' if side > 0 else 'top', fontsize=FS * 0.82,
            color=DIM_C, zorder=5)
DOFF = 22                                                            # line to slab edge, plot units
dim(SC_W, Y(SCINT_Z['bottom']) - SC_T / 2 - DOFF, '60 cm', -1)
dim(M3_W, Y(M3_Z[2]) - M3_T / 2 - DOFF, '50 cm', -1)
dim(MX_W, Y(SLOT_Z['top']) + MX_T / 2 + DOFF, '40 cm', +1, xl=-110)   # off the muon's path
'''


def load(name, yoff, hh):
    s = (SRC / name).read_text()
    s = s.replace("sys.path[:0] = [str(HERE.parents[2] / 'papers')]", "sys.path[:0] = [str(REPO_ROOT / 'papers')]")
    s = re.sub(r"fig = plt\.figure\(figsize=\(W, H\)\)", "W, H = W_ALL, H_ALL", s)
    s = s.replace("W, H = AX_W + 0.1, 2.55", "")
    s = s.replace("fig.add_axes([0.05 / W, 0.05 / H, AX_W / W, AX_H / H])",
                  f"fig.add_axes([0.05 / W, ({yoff} + 0.05) / H, AX_W / W, AX_H / H])")
    s = s.replace("'M3 reference planes", "'reference planes")
    # 2026-11-20 (Dylan): the transverse sizes as thin dimension lines, one per kind of element (every element is
    # square, said in the caption). The slabs are already drawn to scale in x. The panel gains a little room below
    # for the scintillator's line, and the muon arrow stops just above it.
    s = s.replace("XL, XR, YB, YT = -640, 700, -95, 560", "XL, XR, YB, YT = -455, 700, -128, 560")
    s = s.replace("ax.text(xat(950) - 45, Y(950), 'cosmic\\nmuon', color=ps.ACCENT, fontsize=FS, fontstyle='italic', ha='right',",
                  "ax.text(xat(950) + 40, Y(950), 'cosmic\\nmuon', color=ps.ACCENT, fontsize=FS, fontstyle='italic', ha='left',")
    s = s.replace("AX_W, AX_H = (XR - XL) / UPI, 2.40", "AX_W, AX_H = BENCH_W, 2.40")
    s = s.replace("x0, z0, x1, z1 = 150, 1510, -135, -150", "x0, z0, x1, z1 = 150, 1510, -126, -120")
    s = s.replace("# muon, going down", DIMS + "\n# muon, going down")
    s = re.sub(r"^fig\.savefig\(.*$", "", s, flags=re.M)
    s = re.sub(r"^fig\.canvas\.draw\(\).*$", "", s, flags=re.M)
    s = re.sub(r"^ps\.save\(.*$", "", s, flags=re.M)
    s = s.replace("W, H = 6.1, 1.85", "")
    g = dict(__file__=str(SRC / name), __name__='drawing', fig=fig, W_ALL=W, H_ALL=H, REPO_ROOT=REPO, BENCH_W=BENCH_W)
    exec(compile(s, name, 'exec'), g)
    return g


# bench first (it sets W/H from its own axes width; both then use the shared W_ALL / H_ALL)
gb = load('make_bench_setup.py', H_S + GAP, H_B)
sx = BENCH_W / (gb['XR'] - gb['XL']); sy = 2.40 / (gb['YT'] - gb['YB'])
print('bench: in/mm x %.5f, z %.5f, z compression %.3f of transverse' % (sx, sy * gb['ZC'], sy * gb['ZC'] / sx))
gs = load('make_sps_setup.py', 0.0, H_S)
fs = ps.BASE_PT
fig.text(0.0, 1.0, '(a)', fontsize=fs, fontweight='bold', color=ps.INK, ha='left', va='top')
fig.text(0.0, (H_S + 0.02) / H, '(b)', fontsize=fs, fontweight='bold', color=ps.INK, ha='left', va='top')
out = HERE / 'out' / 'setups'
ps.save(fig, out)
print('ok', out)
