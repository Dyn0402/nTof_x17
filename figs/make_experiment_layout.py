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

Distances (schematic, rounded from MX17_Full_Geant/include/SimConfig.hh): windows 20 cm from the target; first
scintillator layer 11 cm behind the window, 3.5 cm deep, 50 cm wide; second about 6 cm behind it, 2 cm deep, 40 cm
wide. Chambers drawn 40 cm wide (active area) and 5 cm deep (window to back plate). No data are read.
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
S1 = (L_WIN + 11.0, 3.5, 22.0)       # (front, depth, half width) [cm]
S2 = (S1[0] + S1[1] + 6.0, 2.0, 18.0)
GAP_NOMINAL = 30.0    # mm


def layout(ax):
    import numpy as np
    import matplotlib.patches as mp
    ink, mut = ps.INK, ps.MUTED
    mm_c, sc1_c, sc2_c = ps.DET_COLOR['det3'], '#9ecae1', '#fdd0a2'

    def slab(front, depth, half, rot, **kw):
        # a slab facing the target: along +y for rot=0, then rotated by rot*90 deg
        x0, y0, w, h = -half, front, 2 * half, depth
        corners = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
        c, s = np.cos(rot * np.pi / 2), np.sin(rot * np.pi / 2)
        R = np.array([[c, -s], [s, c]])
        ax.add_patch(mp.Polygon(corners @ R.T, closed=True, **kw))

    for k in range(4):
        slab(L_WIN, MM_DEPTH, MM_HALF, k, fc='#e8eef8', ec=mm_c, lw=1.0, zorder=3)
        slab(L_WIN, 0.35, MM_HALF, k, fc=mm_c, ec='none', zorder=4)          # the entrance window side
        slab(*S1, k, fc=sc1_c, ec='#3182bd', lw=0.6, zorder=3)
        slab(*S2, k, fc=sc2_c, ec='#e6550d', lw=0.6, zorder=3)
    # target and beam (into the page)
    ax.add_patch(mp.Circle((0, 0), 1.6, fc='white', ec=ink, lw=0.9, zorder=5))
    ax.plot(0, 0, 'o', ms=2.0, color=ink, zorder=6)
    ax.text(2.4, 2.2, 'target\n(beam into page)', fontsize=6.5, color=ink, ha='left', va='bottom')
    # one pair: e- up-right, e+ to the left, opening angle ~140 deg
    for ang, lab in ((160.0, '$e^-$'), (-80.0, '$e^+$')):
        a = np.deg2rad(ang)
        r1 = 43.5 / max(abs(np.cos(a)), abs(np.sin(a)))
        ax.annotate('', xy=(r1 * np.cos(a), r1 * np.sin(a)), xytext=(1.7 * np.cos(a), 1.7 * np.sin(a)),
                    arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.1, mutation_scale=8), zorder=7)
        rl = 13.0
        ax.text(rl * np.cos(a) + (0.0 if ang > 0 else 1.6), rl * np.sin(a) + (2.4 if ang > 0 else 0.0), lab,
                color=ps.ACCENT, fontsize=8, ha='center' if ang > 0 else 'left', va='center')
    # 20 cm dimension (to the lower chamber's window)
    ax.annotate('', xy=(-10.0, -L_WIN), xytext=(-10.0, 0), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7,
                                                                            shrinkA=0, shrinkB=0))
    ax.text(-10.8, -L_WIN / 2, '20 cm', rotation=90, fontsize=7, color=mut, ha='right', va='center')
    # labels on the right-hand arm
    for (x, half, lab, c) in ((L_WIN + MM_DEPTH / 2, MM_HALF, 'TPC', mm_c), (S1[0] + S1[1] / 2, S1[2], 'scint. 1', '#3182bd'),
                              (S2[0] + S2[1] / 2, S2[2], 'scint. 2', '#e6550d')):
        ax.text(x, half + 1.0, lab, fontsize=7, color=c, ha='center', va='bottom', rotation=90)
    # the box that panel (b) enlarges: on the lower chamber (window up, towards the target, as the cathode in (b)),
    # where the e+ crosses it
    bx = L_WIN / np.tan(np.deg2rad(80.0))
    ax.add_patch(mp.Rectangle((bx - 3.5, -L_WIN - MM_DEPTH - 1.0), 7.0, MM_DEPTH + 2.0, fc='none', ec=ink, lw=0.7,
                              ls='--', zorder=8))
    ax.set_xlim(-46, 46)
    ax.set_ylim(-46, 46)
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
    sketch(axb, g, 1.0, 0.0, (-12.5, g + 12.5), particle='lepton', gap_label=f'$g$ = {g:.0f} mm')
    for p_a, p_b in ((top, (0.0, 0.98)), (bot, (0.0, 0.02))):
        fig.add_artist(ConnectionPatch(xyA=p_a, coordsA=axa.transData, xyB=p_b, coordsB=axb.transAxes,
                                       color=ps.MUTED, lw=0.6, ls='--'))
    axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, fontsize=8, va='top', ha='left')
    axb.text(0.06, 1.0, '(b)', transform=axb.transAxes, fontsize=8, va='top', ha='left')
    ps.save(fig, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
