#!/usr/bin/env python3
"""make_experiment_layout.py -- Paper I v2, Fig. 1 (fig.experiment_layout). Copy of papers/paper1/figs/ version.

    .venv/bin/python papers/paper1_v2/figs/make_experiment_layout.py [--out ...]

2026-11-11 (feedback note, Fig. 1): scintillators rectangular (realistic), scint. 2 ~30 % thinner; e+ turned
further into the page, away from the chamber edge (opening angle ~100 deg); "target" label off the e+ line.

Dylan (feedback note, 1 Introduction / Fig. 1): split the old TPC-principle figure. Fig. 1 becomes
(a) a top-down sketch of the measurement the chambers were designed for, "without all the gory details of what we
    actually did": a target, four chambers with their entrance windows at the nominal 20 cm, two layers of
    scintillator behind each, the beam into the page and one e+e- pair (opening angle 120 deg);
(b) one chamber in section, enlarged: the old panel (a) of tpc_principle (make_tpc_principle.sketch), drawn in mm with
    the nominal 30 mm gap ("since it's just a diagram").
2026-11-16 (Dylan): panel (b) is drawn here (section(), no longer make_tpc_principle.sketch): one ionisation cluster
above each strip the track crosses, a strip against arrival-time ladder under the strips (what is measured: the
inclination is a slope in time across strips), the stage below the mesh marked "enlarged" with the resistive strips
in the colour of fig.chamber (c), and the gap labelled "~30 mm" (not exact; chamber A is quoted at 28 mm later).
The data and fit-model panels moved to fig.reco_example (make_tpc_principle.py --variant reco), in the
reconstruction subsection.

Distances: windows 20 cm from the target (as built); the scintillators are an idealised design (Dylan 11-06): a thin
first layer 10 cm behind the window, a thicker second layer behind it, both rectangular and covering the acceptance. Chambers drawn 40 cm wide (active area) and 5 cm deep (window to back plate). No data are read.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..', 'paper1', 'figs'), os.path.join(HERE, '..', '..')]
import paper_style as ps  # noqa: E402

RES_C = '#3d4144'     # resistive strips, the colour they have in fig.chamber (c) (render3d material)

L_WIN = 20.0          # window front face from the target [cm]
MM_DEPTH, MM_HALF = 5.0, 18.0
# Idealised stack (Dylan 11-06, "closer to our other facility plans"): a thin first (trigger) layer, then a thicker
# second layer wide enough to cover the full acceptance of the chamber. Rectangular slabs (Dylan 11-11, "realistic"),
# scint. 2 ~30 % thinner than the 11-06 draft (8 -> 5.6 cm). Each slab's half-width is just under its distance from
# the target, so it covers the acceptance cone (MM_HALF / L_WIN = 0.9) and the four arms do not overlap at the corners.
S1 = (L_WIN + 10.0, 1.0)              # (front, depth) [cm]
S2 = (S1[0] + S1[1] + 5.0, 5.6)
CORNER_GAP = 0.4                      # [cm] between neighbouring arms
GAP_NOMINAL = 30.0    # mm
E_PLUS, E_MINUS = -75.0, 185.0        # pair directions [deg], opening angle 100 deg; e+ well inside the bottom chamber


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

    for k in range(4):
        slab(L_WIN, MM_DEPTH, MM_HALF, k, fc='#e8eef8', ec=mm_c, lw=1.0, zorder=3)
        slab(L_WIN, 0.35, MM_HALF, k, fc=mm_c, ec='none', zorder=4)          # the entrance window side
        slab(*S1, S1[0] - CORNER_GAP, k, fc=sc1_c, ec='#3182bd', lw=0.6, zorder=3)
        slab(*S2, S2[0] - CORNER_GAP, k, fc=sc2_c, ec='#e6550d', lw=0.6, zorder=3)
    # target (beam into the page: the dot in the circle; said in the caption)
    ax.add_patch(mp.Circle((0, 0), 1.6, fc='white', ec=ink, lw=0.9, zorder=5))
    ax.plot(0, 0, 'o', ms=2.0, color=ink, zorder=6)
    ax.text(1.8, 1.6, 'target', fontsize=6.5, color=ink, ha='left', va='bottom')   # above-right: clear of e+ and e-
    # the pair
    for ang in (E_PLUS, E_MINUS):
        a = np.deg2rad(ang)
        r1 = (S2[0] + S2[1] + 0.5) / max(abs(np.cos(a)), abs(np.sin(a)))
        ax.annotate('', xy=(r1 * np.cos(a), r1 * np.sin(a)), xytext=(1.7 * np.cos(a), 1.7 * np.sin(a)),
                    arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.1, mutation_scale=8), zorder=7)
    a = np.deg2rad(E_PLUS)
    ax.text(12.0 * np.cos(a) + 1.2, 12.0 * np.sin(a), '$e^+$', color=ps.ACCENT, fontsize=8, ha='left', va='center')
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


def section(ax, g=GAP_NOMINAL, tan=-0.45, xm=24.0):
    """Panel (b): one chamber in section, in mm (x across the strips, z above the mesh), and the time ladder."""
    import numpy as np
    import matplotlib.patches as mp
    ink, mut, blue = ps.INK, ps.MUTED, ps.DET_COLOR['det3']
    X0, X1, PITCH, HW = 0.0, 30.0, 2.1, 0.8                  # schematic strips: 14 at 2.1 mm
    xc = X0 + 1.2 + PITCH * np.arange(14)
    lab = dict(va='center', ha='left', fontsize=7.5, color=ink)
    ax.add_patch(mp.Rectangle((X0, 0), X1 - X0, g, fc='#f3f6fb', ec='none', zorder=0))
    ax.plot([X0, X1], [g, g], color=ink, lw=1.6, zorder=3)
    ax.text(X1 + 0.6, g, 'cathode', **lab)
    ax.plot([X0, X1], [0, 0], color=ink, lw=0.9, ls=(0, (1, 1.2)), zorder=3)
    ax.text(X1 + 0.6, 0, 'mesh', **lab)
    # below the mesh, enlarged: resistive strips and readout strips
    for x in xc:
        ax.add_patch(mp.Rectangle((x - HW, -5.2), 2 * HW, 1.8, fc=RES_C, ec='none', zorder=2))
        ax.add_patch(mp.Rectangle((x - HW, -10.4), 2 * HW, 1.8, fc='#c98b2b', ec='none', zorder=2))
    ax.text(X1 + 0.6, -4.3, 'resistive strips', **lab)
    ax.text(X1 + 0.6, -9.5, 'readout strips', **lab)
    ax.plot([X0 - 1.0] * 2, [-10.4, -1.2], color=mut, lw=0.7)
    ax.text(X0 - 1.7, -5.8, 'enlarged', rotation=90, va='center', ha='right', fontsize=7, color=mut)
    # the lepton
    xs = lambda zz: xm + tan * zz
    ax.annotate('', xy=(xs(-2.6), -2.6), xytext=(xs(g + 7.5), g + 7.5),
                arrowprops=dict(arrowstyle='-|>', color=ps.ACCENT, lw=1.3, mutation_scale=9), zorder=5)
    ax.text(xs(g + 6.0) + 0.9, g + 6.0, 'lepton', color=ps.ACCENT, fontsize=7.5, va='center', ha='left')
    # one ionisation cluster above each strip the track crosses, drifting straight down
    hit = xc[(xc > xs(g) + 0.5) & (xc < xs(0) - 0.5)]
    zh = (hit - xm) / tan
    for x, zz in zip(hit, zh):
        ax.plot(x, zz, 'o', ms=2.4, color=blue, zorder=6)
        ax.annotate('', xy=(x, 0.6), xytext=(x, zz - 0.6),
                    arrowprops=dict(arrowstyle='-|>', color=blue, lw=0.6, mutation_scale=5, alpha=0.7), zorder=4)
    ax.annotate('', xy=(X0 - 1.6, 0), xytext=(X0 - 1.6, g),
                arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(X0 - 2.3, g / 2, rf'gap $\sim${g:.0f} mm', rotation=90, va='center', ha='right', fontsize=7.5, color=mut)
    ax.text(xs(g * 0.62) + 5.0, g * 0.62, r'$t = t_0 + z/v_d$', fontsize=8, color=ink, ha='left', va='center',
            bbox=dict(fc='white', ec='none', pad=1.0))
    # what is measured: the arrival time on each strip (later for a higher start), a slope across the strips
    T0, T1 = -25.5, -14.0                                    # the ladder's time axis, in the sketch's z units
    tt = T0 + 1.5 + zh / g * (T1 - T0 - 3.0)
    for x in xc:
        ax.plot([x, x], [T0, T1], color=ps.GRID, lw=0.6, zorder=1)
    ax.plot([X0, X1], [T0, T0], color=mut, lw=0.7, zorder=2)
    ax.annotate('', xy=(X0, T1 + 1.2), xytext=(X0, T0), arrowprops=dict(arrowstyle='-|>', color=mut, lw=0.7,
                                                                        mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(X0 - 0.9, (T0 + T1) / 2, 'time', rotation=90, va='center', ha='right', fontsize=7.5, color=mut)
    ax.text((X0 + X1) / 2, T0 - 0.9, 'strip', va='top', ha='center', fontsize=7.5, color=mut)
    ax.plot(hit, tt, '-', color=blue, lw=0.6, alpha=0.6, zorder=3)
    ax.plot(hit, tt, 's', ms=3.2, color=blue, zorder=4)
    ax.text(X1 + 0.6, (T0 + T1) / 2, 'arrival time\non each strip', **lab)
    ax.set_xlim(X0 - 4.0, X1 + 19.5)
    ax.set_ylim(T0 - 4.5, g + 9.5)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'experiment_layout'))
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    ps.use()
    fig = plt.figure(figsize=(ps.FULL[0], 3.3))
    axa = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    axb = fig.add_axes([0.52, 0.04, 0.48, 0.92])
    top, bot = layout(axa)
    section(axb)
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
