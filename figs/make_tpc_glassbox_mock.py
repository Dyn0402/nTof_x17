#!/usr/bin/env python3
"""make_tpc_glassbox_mock.py -- MOCK-UP, not a paper figure (Dylan 2026-11-16: "mock up the glass box just so I can see").

    .venv/bin/python papers/paper1_v2/figs/make_tpc_glassbox_mock.py

A 3D alternative to Fig. 1 (b): the drift volume as a transparent box, one lepton, its ionisation drifting down to the
readout plane, and the track's two projections (X view, Y view) on the walls. Left: the 2D panel as it is in the paper
(make_experiment_layout.section); right: the glass box. Cabinet projection drawn by hand; schematic, no data.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
import paper_style as ps  # noqa: E402
from make_experiment_layout import section, RES_C  # noqa: E402,F401

YC, XC = '#009E73', ps.ACCENT        # Y and X strips, as in fig.chamber (c)
L, G = 30.0, 30.0                    # box side and gap [mm]
KY, ANG = 0.55, np.deg2rad(32)       # cabinet projection: depth (y) drawn shortened along ANG


def P(x, y, z):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    return x + KY * y * np.cos(ANG), z + KY * y * np.sin(ANG)


def glassbox(ax):
    import matplotlib.patches as mp
    ink, mut, blue = ps.INK, ps.MUTED, ps.DET_COLOR['det3']

    def line(a, b, **kw):
        ax.plot(*P(*zip(a, b)), **kw)

    def face(pts, **kw):
        ax.add_patch(mp.Polygon(np.column_stack(P(*zip(*pts))), closed=True, **kw))

    # readout plane (floor) with the two strip sets, the walls that carry the projections, the cathode
    face([(0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0)], fc='#f4efe2', ec=ink, lw=0.8, zorder=1)
    for s in np.arange(1.5, L, 3.0):
        line((s, 0, 0), (s, L, 0), color=XC, lw=0.6, alpha=0.55, zorder=2)      # X strips run along y
        line((0, s, 0), (L, s, 0), color=YC, lw=0.6, alpha=0.55, zorder=2)      # Y strips run along x
    face([(0, L, 0), (L, L, 0), (L, L, G), (0, L, G)], fc='#f3f6fb', ec=mut, lw=0.5, zorder=0)   # back wall (x, z)
    face([(0, 0, 0), (0, L, 0), (0, L, G), (0, 0, G)], fc='#eaf0f8', ec=mut, lw=0.5, zorder=0)   # left wall (y, z)
    face([(0, 0, G), (L, 0, G), (L, L, G), (0, L, G)], fc='none', ec=ink, lw=1.2, zorder=7)       # cathode
    for x, y in ((L, 0), (L, L)):
        line((x, y, 0), (x, y, G), color=mut, lw=0.5, zorder=7)
    # the lepton: in through the cathode, out through the readout plane
    a, b = np.array([8.0, 7.0, G]), np.array([23.0, 21.0, 0.0])
    d = (b - a) / G
    p0, p1 = a - 8.0 * d, b + 3.0 * d
    ax.annotate('', xy=P(*p1), xytext=P(*p0), arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.4,
                                                              mutation_scale=9, shrinkA=0, shrinkB=0), zorder=9)
    u, v = P(*p0)
    ax.text(float(u) + 1.0, float(v), 'lepton', color=ps.ACCENT, fontsize=7.5, ha='left', va='center')
    pts = np.array([a + f * (b - a) for f in np.linspace(0.1, 0.9, 6)])
    for q in pts:
        ax.plot(*P(*q), 'o', ms=2.6, color=blue, zorder=10)
        ax.annotate('', xy=P(q[0], q[1], 0.5), xytext=P(q[0], q[1], q[2] - 0.7),
                    arrowprops=dict(arrowstyle='-|>', color=blue, lw=0.6, mutation_scale=5, alpha=0.7), zorder=8)
        ax.plot(*P(q[0], q[1], 0), 's', ms=2.2, color=blue, zorder=8)
        # the two projections: back wall keeps (x, z), left wall keeps (y, z)
        ax.plot(*P(q[0], L, q[2]), 'o', ms=2.2, color=XC, zorder=5)
        ax.plot(*P(0, q[1], q[2]), 'o', ms=2.2, mfc='white', mec=YC, mew=0.8, zorder=5)
    line((a[0], L, G), (b[0], L, 0), color=XC, lw=0.9, zorder=4)
    line((0, a[1], G), (0, b[1], 0), color=YC, lw=0.9, ls=(0, (3, 1.5)), zorder=4)
    line((a[0], a[1], 0), (b[0], b[1], 0), color=blue, lw=0.6, alpha=0.6, zorder=3)
    u, v = P(L - 1.0, L, G - 3.0)
    ax.text(float(u), float(v), 'X view', color=XC, fontsize=7.5, ha='right', va='center', zorder=6)
    u, v = P(0, 21.0, 9.0)
    ax.text(float(u) + 0.8, float(v), 'Y view', color=YC, fontsize=7.5, ha='left', va='center', zorder=6)
    # labels
    u, v = P(L, L, G)
    ax.text(float(u) + 1.0, float(v), 'cathode', fontsize=7.5, color=ink, ha='left', va='center')
    u, v = P(L, 0.78 * L, 0)
    ax.text(float(u) + 1.0, float(v), 'mesh and\nreadout plane', fontsize=7.5, color=ink, ha='left', va='center')
    u, v = P(L, 0, 0)
    ax.text(float(u) - 4.0, float(v) - 1.0, 'X strips', fontsize=7, color=XC, ha='right', va='top')
    u, v = P(L, 3.0, 0)
    ax.text(float(u) + 1.0, float(v), 'Y strips', fontsize=7, color=YC, ha='left', va='center')
    ax.annotate('', xy=P(-2.0, 0, 0), xytext=P(-2.0, 0, G), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7,
                                                                           shrinkA=0, shrinkB=0))
    ax.text(-2.8, G / 2, r'gap $\sim$30 mm,  $t = t_0 + z/v_d$', rotation=90, fontsize=7.5, color=mut,
            ha='right', va='center')
    ax.set_xlim(-6, L + KY * L + 15)
    ax.set_ylim(-5, G + KY * L * np.sin(ANG) + 7)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ps.use()
    fig = plt.figure(figsize=(ps.FULL[0], 3.3))
    axl = fig.add_axes([0.0, 0.04, 0.47, 0.92])
    axr = fig.add_axes([0.50, 0.0, 0.50, 1.0])
    section(axl)
    glassbox(axr)
    axl.text(0.02, 1.0, '2D (in the paper)', transform=axl.transAxes, fontsize=8, va='top', ha='left')
    axr.text(0.02, 0.99, 'glass box (mock-up)', transform=axr.transAxes, fontsize=8, va='top', ha='left')
    out = os.path.join(HERE, 'out', 'tpc_glassbox_mock')
    ps.save(fig, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
