#!/usr/bin/env python3
"""make_experiment_layout.py -- Paper I, Fig. 1 (fig.experiment_layout), Dylan 2026-11-06.

    .venv/bin/python papers/paper1/figs/make_experiment_layout.py [--out ...]

Dylan (feedback note, 1 Introduction / Fig. 1): split the old TPC-principle figure. Fig. 1 becomes
(a) a top-down sketch of the measurement the chambers were designed for, "without all the gory details of what we
    actually did": a target, four chambers with their entrance windows at the nominal 20 cm, two layers of
    scintillator behind each, the beam into the page and one e+e- pair (opening angle 120 deg);
(b) one chamber in section, enlarged: the old panel (a) of tpc_principle (make_tpc_principle.sketch), drawn in mm with
    the nominal 30 mm gap ("since it's just a diagram").
The data and fit-model panels moved to fig.reco_example (make_tpc_principle.py --variant reco), in the
reconstruction subsection.

Distances: windows 20 cm from the target (as built); the scintillators are an idealised design (Dylan 11-06): a thin
first layer 10 cm behind the window, a thick second layer behind it, both projective and covering the acceptance. Chambers drawn 40 cm wide (active area) and 5 cm deep (window to back plate). No data are read.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
import paper_style as ps  # noqa: E402
from make_tpc_principle import sketch  # noqa: E402

L_WIN = 20.0          # window front face from the target [cm]
MM_DEPTH, MM_HALF = 5.0, 18.0
# Idealised stack (Dylan 11-06, "closer to our other facility plans"): a thin first (trigger) layer, then a thick
# second layer wide enough to cover the full acceptance of the chamber. Both are drawn projective (each face spans
# FAN x its distance on either side), so they cover the acceptance cone (MM_HALF / L_WIN = 0.9) and the four arms
# tile without overlapping at the corners.
S1 = (L_WIN + 10.0, 1.0)              # (front, depth) [cm]
S2 = (S1[0] + S1[1] + 5.0, 8.0)
FAN = 0.95
GAP_NOMINAL = 30.0    # mm
E_PLUS, E_MINUS = -55.0, 185.0        # pair directions [deg], opening angle 120 deg; e+ near the bottom chamber's edge


def layout(ax):
    import numpy as np
    import matplotlib.patches as mp
    ink, mut = ps.INK, ps.MUTED
    mm_c, sc1_c, sc2_c = ps.DET_COLOR['det3'], '#9ecae1', '#fdd0a2'

    def poly(pts, rot, **kw):
        c, s = np.cos(rot * np.pi / 2), np.sin(rot * np.pi / 2)
        R = np.array([[c, -s], [s, c]])
        ax.add_patch(mp.Polygon(np.asarray(pts) @ R.T, closed=True, **kw))

    def slab(front, depth, half, rot, **kw):            # rectangle facing the target (along +y for rot=0)
        poly([[-half, front], [half, front], [half, front + depth], [-half, front + depth]], rot, **kw)

    def wedge(front, depth, rot, **kw):                 # projective slab
        b = front + depth
        poly([[-FAN * front, front], [FAN * front, front], [FAN * b, b], [-FAN * b, b]], rot, **kw)

    for k in range(4):
        slab(L_WIN, MM_DEPTH, MM_HALF, k, fc='#e8eef8', ec=mm_c, lw=1.0, zorder=3)
        slab(L_WIN, 0.35, MM_HALF, k, fc=mm_c, ec='none', zorder=4)          # the entrance window side
        wedge(*S1, k, fc=sc1_c, ec='#3182bd', lw=0.6, zorder=3)
        wedge(*S2, k, fc=sc2_c, ec='#e6550d', lw=0.6, zorder=3)
    # target (beam into the page: the dot in the circle; said in the caption)
    ax.add_patch(mp.Circle((0, 0), 1.6, fc='white', ec=ink, lw=0.9, zorder=5))
    ax.plot(0, 0, 'o', ms=2.0, color=ink, zorder=6)
    ax.text(1.2, -2.4, 'target', fontsize=6.5, color=ink, ha='left', va='top')
    # the pair
    for ang in (E_PLUS, E_MINUS):
        a = np.deg2rad(ang)
        r1 = (S2[0] + S2[1] + 0.5) / max(abs(np.cos(a)), abs(np.sin(a)))
        ax.annotate('', xy=(r1 * np.cos(a), r1 * np.sin(a)), xytext=(1.7 * np.cos(a), 1.7 * np.sin(a)),
                    arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.1, mutation_scale=8), zorder=7)
    a = np.deg2rad(E_PLUS)
    ax.text(12.0 * np.cos(a) + 1.4, 12.0 * np.sin(a), '$e^+$', color=ps.ACCENT, fontsize=8, ha='left', va='center')
    a = np.deg2rad(E_MINUS)
    ax.text(11.0 * np.cos(a), 11.0 * np.sin(a) - 1.6, '$e^-$', color=ps.ACCENT, fontsize=8, ha='center', va='top')
    # 20 cm, horizontal, from the target to the left chamber's window, above the e- track
    ax.annotate('', xy=(-L_WIN, 3.0), xytext=(0, 3.0), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7,
                                                                       shrinkA=0, shrinkB=0))
    ax.text(-L_WIN / 2, 3.8, '20 cm', fontsize=7, color=mut, ha='center', va='bottom')
    # labels, horizontal, on the top stack
    for (y, lab, c) in ((L_WIN + MM_DEPTH / 2, 'TPC', mm_c), (S1[0] + S1[1] + 2.5, 'scint. 1 (thin)', '#3182bd'),
                        (S2[0] + S2[1] / 2, 'scint. 2', '#e6550d')):
        ax.text(0.0, y, lab, fontsize=7, color=c, ha='center', va='center', zorder=9)
    # the box that panel (b) enlarges: on the lower chamber (window up, towards the target, as the cathode in (b)),
    # where the e+ crosses it
    a = np.deg2rad(E_PLUS)
    bx = min(L_WIN / abs(np.tan(a)) + 0.5 * MM_DEPTH / abs(np.tan(a)), MM_HALF - 3.6)
    ax.add_patch(mp.Rectangle((bx - 3.5, -L_WIN - MM_DEPTH - 1.0), 7.0, MM_DEPTH + 2.0, fc='none', ec=ink, lw=0.7,
                              ls='--', zorder=8))
    lim = S2[0] + S2[1] + 1.0
    ax.set_xlim(-lim - 3.0, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    return (bx + 3.5, -L_WIN + 1.0), (bx + 3.5, -L_WIN - MM_DEPTH - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'experiment_layout'))
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    ps.use()
    fig = plt.figure(figsize=(ps.FULL[0], 3.0))
    axa = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    axb = fig.add_axes([0.52, 0.04, 0.48, 0.92])
    top, bot = layout(axa)
    g = GAP_NOMINAL
    sketch(axb, g, 1.0, 0.0, (-12.5, g + 12.5), tan=-0.45, particle='lepton', gap_label=f'gap = {g:.0f} mm', vsym='v_d')
    for p_a, p_b in ((top, (0.0, 0.98)), (bot, (0.0, 0.02))):
        # drawn with (a), under its detector components (zorder 1), not on top of the figure
        axa.add_artist(ConnectionPatch(xyA=p_a, coordsA=axa.transData, xyB=p_b, coordsB=axb.transAxes,
                                       color=ps.MUTED, lw=0.6, ls='--', zorder=1, clip_on=False))
    axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, fontsize=8, va='top', ha='left')
    axb.text(0.06, 1.0, '(b)', transform=axb.transAxes, fontsize=8, va='top', ha='left')
    ps.save(fig, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
