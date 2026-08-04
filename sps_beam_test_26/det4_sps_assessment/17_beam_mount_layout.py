#!/usr/bin/env python3
"""
17_beam_mount_layout.py -- how det4 gets bolted into the H4 rail set, to scale.

A mechanical drawing, not an analysis. It takes the board geometry that
`14_board_map.py` measured out of the DFS3498A Gerbers (PCB outline, active
square, the 4.2 mm frame-hole grid) and the live band from
`DET4_SPS_ASSESSMENT.md` (local X 177-215 mm), and draws them into the beam-line
rail set as described on 2026-07-31:

  * three parallel extruded profiles running ALONG the beam, equally spaced,
    300 mm outer face to outer face, middle one on the centre-line;
  * ~250 mm of free space along the rails between the two neighbouring
    detectors -- the whole budget for det4 plus any rotation;
  * det4 ~500 x 500 x 50 mm, carried on a 600 mm cross bar.

Everything about the RAILS is an assumption (flagged ASSUMED in the figure) and
is exposed as a CLI flag. Everything about the BOARD is measured.

Four panels:
  A  plan view from above -- the rotation problem and the 250 mm budget
  B  elevation looking along the beam -- heights, the live band, the beam spot
  C  the two bracket joints, in the views that make the bolt directions obvious
  D  along-beam depth swept vs yaw angle

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/17_beam_mount_layout.py
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- MEASURED board geometry (14_board_map.py, DFS3498A Gerbers) ------------
PCB = (-20.32, 449.68)        # PCB outline, detector-local mm, both axes
ACTIVE = (0.0, 398.58)        # metallised strip region
BAND = (177.0, 215.0)         # the live band, local X (DET4_SPS_ASSESSMENT §3b)
SQUARE = (149.76, 248.82)     # X4+X5 / Y4+Y5 readout square, both axes
FRAME_HOLE = np.array([-214., -165., -110., -55., 0., 55., 110., 165., 214.])
BOTTOM = -PCB[0]              # 20.32: local -> height above the bare PCB edge

INK, MUTED = '#1b1b1b', '#8a8a8a'
BEAMC, DET4C, RAIL, BAR, BRK = '#0072b2', '#d55e00', '#7a7a7a', '#4c6b8a', '#1f7a4d'
BOX = dict(fc='white', ec='#dddddd', alpha=0.92, pad=2.4, boxstyle='round,pad=0.32')


def rect(x, y, w, h, **kw):
    return Rectangle((x, y), w, h, **kw)


def prof_section(ax, cx, cy, w, fc='#cfd6dc', ec=RAIL, z=3, lw=1.0):
    """Extruded-profile cross-section: square body with four T-slot mouths."""
    ax.add_patch(rect(cx - w / 2, cy - w / 2, w, w, fc=fc, ec=ec, lw=lw, zorder=z))
    m, d = w * 0.28, w * 0.20
    for sx, sy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if sy:
            ax.add_patch(rect(cx - m / 2, cy + sy * (w / 2 - d), m, d * sy,
                              fc='w', ec=ec, lw=lw * .8, zorder=z + 1))
        else:
            ax.add_patch(rect(cx + sx * (w / 2 - d), cy - m / 2, d * sx, m,
                              fc='w', ec=ec, lw=lw * .8, zorder=z + 1))
    ax.add_patch(Circle((cx, cy), w * 0.10, fc='w', ec=ec, lw=lw * .8, zorder=z + 1))


def dim(ax, p0, p1, txt, fs=8.0, color=INK, ha='center', va='center',
        dx=0.0, dy=0.0, rot=0):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='<->', color=color, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=20)
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    ax.text(mx + dx, my + dy, txt, fontsize=fs, color=color, ha=ha, va=va,
            rotation=rot, zorder=21, bbox=BOX)


def det_poly(theta, W=500.0, T=50.0):
    """det4 footprint in plan, columns (along-beam z, transverse x)."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    P = np.array([[-W / 2, -T / 2], [W / 2, -T / 2], [W / 2, T / 2], [-W / 2, T / 2]])
    Q = P @ np.array([[c, -s], [s, c]]).T
    return Q[:, ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rail-span', type=float, default=300.0,
                    help='ASSUMED outer face to outer face of the 3 rails [mm]')
    ap.add_argument('--profile', type=float, default=40.0,
                    help='ASSUMED profile cross-section [mm]')
    ap.add_argument('--gap', type=float, default=250.0,
                    help='ASSUMED free space along the rails between neighbours [mm]')
    ap.add_argument('--crossbar', type=float, default=600.0)
    ap.add_argument('--det-w', type=float, default=500.0,
                    help='ASSUMED overall chamber width [mm] (the PCB is 470)')
    ap.add_argument('--det-t', type=float, default=50.0,
                    help='ASSUMED overall chamber thickness [mm]')
    ap.add_argument('--beam-height', type=float, default=250.0,
                    help='ASSUMED beam height above the rail top face [mm] -- MEASURE IT')
    ap.add_argument('--frame-below-pcb', type=float, default=0.0,
                    help='chamber frame protruding below the bare PCB edge [mm]')
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()

    W, T, P = a.det_w, a.det_t, a.profile
    xr = np.array([-1., 0., 1.]) * (a.rail_span - P) / 2.0     # rail centres
    band_h = np.array(BAND) + BOTTOM + a.frame_below_pcb       # above chamber foot
    band_c = float(band_h.mean())
    foot = a.beam_height - band_c                              # mount height budget

    Rr, ph = math.hypot(W, T), math.atan2(T, W)
    th_max = math.degrees(math.asin(min(1., a.gap / Rr)) - ph)
    th_saf = math.degrees(math.asin(min(1., (a.gap - 40) / Rr)) - ph)
    th_show = round(th_saf)
    travel = math.tan(math.radians(th_show)) * abs(xr[0])
    pullin = abs(xr[0]) * (1 - math.cos(math.radians(th_show)))

    fig = plt.figure(figsize=(17.6, 13.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], width_ratios=[1.06, 1],
                          left=0.05, right=0.985, top=0.915, bottom=0.05,
                          hspace=0.30, wspace=0.16)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ================= A : PLAN VIEW ====================================== #
    zl = 300.0
    for xc in xr:
        axA.add_patch(rect(-zl, xc - P / 2, 2 * zl, P, fc='#e4e9ed', ec=RAIL,
                           lw=1.0, zorder=2))
        for f in (0.25, 0.75):
            axA.plot([-zl, zl], [xc - P / 2 + f * P] * 2, color=RAIL, lw=0.6, zorder=3)

    for s in (-1, 1):
        x0 = a.gap / 2 if s > 0 else -zl
        axA.add_patch(rect(x0, -340, zl - a.gap / 2, 680, fc='k', alpha=0.06,
                           lw=0, zorder=1))
        axA.text(s * (zl - 12), 300, 'neighbour\ndetector', fontsize=8.5,
                 color=MUTED, ha='right' if s > 0 else 'left', va='center')
        axA.axvline(s * a.gap / 2, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=4)

    for th in (-th_show, th_show):
        axA.add_patch(Polygon(det_poly(th, W, T), closed=True, fc=DET4C,
                              alpha=0.26, ec=DET4C, lw=1.0, ls=(0, (4, 2)), zorder=6))
    axA.add_patch(Polygon(det_poly(0, W, T), closed=True, fc=DET4C, alpha=0.6,
                          ec='#8f3f00', lw=1.6, zorder=8))
    axA.text(0, -160, u'det4   %g × %g mm' % (T, W), fontsize=8.8, color='w',
             ha='center', va='center', rotation=90, zorder=12, fontweight='bold')

    axA.add_patch(rect(-72, -a.crossbar / 2, 144, a.crossbar, fc=BAR, alpha=0.16,
                       ec=BAR, lw=1.2, zorder=5))
    axA.plot([0], [0], marker='o', ms=10, mfc='w', mec=BRK, mew=2.2, zorder=13)
    for xc in (xr[0], xr[2]):
        axA.add_patch(rect(-11, xc - 11, 22, 22, fc='w', ec=BRK, lw=1.5, zorder=13))
        axA.annotate('', xy=(travel, xc), xytext=(-travel, xc),
                     arrowprops=dict(arrowstyle='<->', color=BRK, lw=1.5), zorder=13)

    axA.text(-zl + 10, -318, u'● pivot  = 1 bolt into a T-nut in the '
             u'MIDDLE rail, directly under the beam spot\n'
             u'□ clamps = 1 bolt each into the OUTER rails; they run '
             u'±%.0f mm along the rail slot at ±%d°\n'
             u'      (Ø18 clearance holes in the sled absorb the %.0f mm '
             'transverse pull-in)' % (travel, th_show, pullin),
             fontsize=8.4, color=BRK, va='bottom', ha='left', zorder=15, bbox=BOX)
    axA.text(-zl + 10, 72, 'three rails,\nrunning along the beam\n'
             '(ASSUMED %g mm profile)' % P, fontsize=8.4, color=RAIL,
             va='center', ha='left', zorder=15, bbox=BOX)
    axA.text(-80, 250, 'sled', fontsize=8.8, color=BAR, ha='right',
             va='center', zorder=15)
    dim(axA, (-a.gap / 2, -250), (a.gap / 2, -250),
        'ASSUMED %g mm free\nalong the rails' % a.gap, dy=0)
    dim(axA, (zl - 55, xr[0] - P / 2), (zl - 55, xr[2] + P / 2),
        'ASSUMED\n%g mm' % a.rail_span, dx=0)
    axA.annotate('beam', xy=(-zl + 130, 195), xytext=(-zl + 20, 195),
                 arrowprops=dict(arrowstyle='-|>', color=BEAMC, lw=2.4),
                 fontsize=10.5, color=BEAMC, va='center')
    axA.plot([-zl, zl], [0, 0], color=BEAMC, lw=0.9, ls=(0, (7, 4)), zorder=10)

    axA.set_xlim(-zl, zl)
    axA.set_ylim(-345, 345)
    axA.set_aspect('equal')
    axA.set_xlabel('along the beam [mm]')
    axA.set_ylabel('transverse [mm]')
    axA.set_title(u'A   Plan view from above', fontsize=11, loc='left',
                  fontweight='bold', pad=24)
    axA.text(0.0, 1.008, u'yaw fits in the gap only to ±%.0f° '
             u'(hard) / ±%.0f° (with 20 mm clearance each side)'
             % (th_max, th_saf), transform=axA.transAxes, fontsize=9.2,
             color=INK, va='bottom')

    # ================= B : ELEVATION ====================================== #
    for xc in xr:
        prof_section(axB, xc, -P / 2, P)
    axB.plot([-430, 430], [0, 0], color=RAIL, lw=1.6, zorder=4)
    axB.text(-425, -P - 14, 'rail top face = datum', fontsize=8.4, color=RAIL)

    # cross bar: it lives BESIDE the chamber (up/downstream), seen edge-on here
    axB.add_patch(rect(-a.crossbar / 2, max(foot - P, -P / 2), a.crossbar, P,
                       fc='#dce4ec', ec=BAR, lw=1.1, alpha=0.55, zorder=5,
                       hatch='///'))
    axB.text(-a.crossbar / 2 + 8, max(foot - P, -P / 2) + P / 2,
             '600 mm cross bar (behind / in front of the chamber)', fontsize=7.8,
             color=BAR, va='center', zorder=6)

    axB.add_patch(rect(-W / 2, foot, W, W, fc='#f3f3f3', ec=INK, lw=1.5, zorder=6))
    axB.add_patch(rect(-235., foot, 470., 470., fc='w', ec='#555', lw=1.1, zorder=7))
    axB.add_patch(rect(-398.58 / 2, foot + BOTTOM, 398.58, 398.58, fill=False,
                       ec=MUTED, lw=1.0, ls='--', zorder=8))
    axB.add_patch(rect(-398.58 / 2, foot + band_h[0], 398.58, band_h[1] - band_h[0],
                       fc=DET4C, alpha=0.38, lw=0, zorder=9))
    axB.add_patch(rect(-99.06 / 2, foot + BOTTOM + SQUARE[0], 99.06, 99.06,
                       fill=False, ec=DET4C, lw=1.8, zorder=10))
    for gy in FRAME_HOLE:
        for gx in (-214., 214.):
            axB.add_patch(Circle((gx, foot + gy + 220.), 2.2, fc='w', ec='#aaa',
                                 lw=.7, zorder=11))
        if gy in (FRAME_HOLE[0], FRAME_HOLE[-1]):
            for gx in FRAME_HOLE:
                axB.add_patch(Circle((gx, foot + gy + 220.), 2.2, fc='w', ec='#aaa',
                                     lw=.7, zorder=11))

    axB.add_patch(Ellipse((0, a.beam_height), 2 * 28.6, 2 * 37.8, fill=False,
                          ec=BEAMC, lw=2.0, zorder=13))
    axB.add_patch(Ellipse((0, a.beam_height), 4 * 28.6, 4 * 37.8, fill=False,
                          ec=BEAMC, lw=1.1, zorder=13))
    axB.plot([0], [a.beam_height], '+', color=BEAMC, ms=13, mew=2.0, zorder=14)
    axB.axhline(a.beam_height, color=BEAMC, lw=0.9, ls=(0, (7, 4)), zorder=5)

    dim(axB, (-292, foot), (-292, foot + band_c),
        u'%.0f mm\nfoot → band' % band_c, ha='center', color=DET4C)
    dim(axB, (272, 0), (272, foot),
        'only %.0f mm of mount\nfits under the chamber' % foot, ha='left',
        color=BRK, dx=14)
    axB.text(0, foot + band_c, u'live band, 38 mm  (local X 177–215)',
             fontsize=8.6, color='#8f3f00', ha='center', va='center', zorder=12,
             bbox=BOX)
    axB.text(0, foot + BOTTOM + SQUARE[0] - 8, 'X4+X5 / Y4+Y5 readout square, 99 mm',
             fontsize=7.9, color=DET4C, ha='center', va='top', zorder=12)
    axB.text(-425, foot + W + 74,
             u'MEASURED: PCB 470 × 470, active 399 mm square,\n'
             u'Ø4.2 mm frame holes at 55 mm pitch on a 428 mm square',
             fontsize=8.2, color='#666', va='top', ha='left', bbox=BOX)
    axB.text(425, foot + W + 74, u'beam, ASSUMED %g mm above the rail top\n'
             u'$\\sigma$ = 28.6 mm horizontal / 37.8 mm vertical' % a.beam_height,
             fontsize=8.5, color=BEAMC, ha='right', va='top', bbox=BOX)

    axB.set_xlim(-430, 430)
    axB.set_ylim(-P - 34, foot + W + 110)
    axB.set_aspect('equal')
    axB.set_xlabel('transverse [mm]')
    axB.set_ylabel('height above the rail top face [mm]')
    axB.set_title('B   Elevation, looking along the beam', fontsize=11, loc='left',
                  fontweight='bold', pad=24)
    axB.text(0.0, 1.008, u'the band centre sits %.0f mm above the board’s bare '
             'edge — that number sets the whole mount' % band_c,
             transform=axB.transAxes, fontsize=9.2, color=INK, va='bottom')

    # ================= C : the two joints ================================= #
    axC.set_xlim(0, 100)
    axC.set_ylim(0, 52)
    axC.axis('off')
    axC.set_title(u'C   The two joints', fontsize=11, loc='left',
                  fontweight='bold', pad=24)
    axC.text(0.0, 1.008, u'an ordinary 90° angle bracket does work here — '
             'the two slots it bridges face different ways',
             transform=axC.transAxes, fontsize=9.2, color=INK, va='bottom')

    # --- C1: rail -> cross bar, elevation looking along the beam
    axC.text(0, 48, u'C1   rail → cross bar', fontsize=9.2, color=INK,
             fontweight='bold')
    axC.text(0, 44.5, 'elevation, looking along the beam', fontsize=8, color=MUTED)
    axC.add_patch(rect(3, 12, 22, 8, fc='#e4e9ed', ec=RAIL, lw=1.1))
    axC.text(14, 16, 'rail (into the page)', fontsize=7.0, ha='center',
             va='center', color=RAIL)
    axC.add_patch(rect(3, 20, 22, 1.6, fc='w', ec=RAIL, lw=.7))
    axC.text(2, 21, 'top slot,\nruns along\nthe beam', fontsize=6.8, ha='right',
             va='center', color=RAIL)
    axC.add_patch(rect(7, 22.6, 14, 9, fc='#dce4ec', ec=BAR, lw=1.1))
    axC.text(14, 27, 'cross bar\n(across the beam)', fontsize=7.0, ha='center',
             va='center', color=BAR)
    axC.add_patch(Polygon([(21, 21.4), (28.5, 21.4), (28.5, 24.2), (23.8, 24.2),
                           (23.8, 31.5), (21, 31.5)], closed=True, fc='#d9f0e2',
                          ec=BRK, lw=1.5))
    axC.plot([26], [22.8], 'o', ms=3.6, color=BRK)
    axC.plot([22.4], [28.5], 'o', ms=3.6, color=BRK)
    axC.annotate(u'horizontal leg → bolt DOWN into the rail’s top slot\n'
                 u'(its T-nut slides along the beam — free)',
                 xy=(26, 22.8), xytext=(30, 15.5), fontsize=7.6, color=BRK,
                 arrowprops=dict(arrowstyle='-', color=BRK, lw=.8), va='center')
    axC.annotate(u'vertical leg → bolt SIDEWAYS into the cross bar’s\n'
                 u'side slot (its T-nut slides across the beam — free)',
                 xy=(22.4, 28.5), xytext=(30, 33.5), fontsize=7.6, color=BRK,
                 arrowprops=dict(arrowstyle='-', color=BRK, lw=.8), va='center')
    axC.text(0, 10.0, u'The “Ls are for parallel profiles” feeling comes '
             u'from trying to\nbutt the cross bar into a rail END-ON. Don’t. Lay '
             u'it ACROSS the\nrails: the rail slot faces up, the cross-bar slot '
             u'faces sideways,\nand a plain L bridges them. Two Ls per rail, one '
             u'either side of\nthe bar — or a flat drilled plate, which is '
             u'stiffer and is what a\nproper cross-connector is.',
             fontsize=7.9, color=INK, va='top', ha='left')

    # --- C2: chamber -> post
    axC.text(53, 48, u'C2   chamber → mount', fontsize=9.2, color=INK,
             fontweight='bold')
    axC.text(53, 44.5, u'the “random” hole heights are a non-problem',
             fontsize=8, color=MUTED)
    axC.add_patch(rect(56, 14, 5, 27, fc='#f3f3f3', ec=INK, lw=1.3))
    axC.text(58.5, 42, 'chamber\nedge', fontsize=7.0, ha='center', va='bottom',
             color=INK)
    axC.add_patch(rect(70, 12, 6, 31, fc='#dce4ec', ec=BAR, lw=1.1))
    for f in (0.28, 0.72):
        axC.plot([70 + f * 6] * 2, [12, 43], color=BAR, lw=.6)
    axC.text(77.5, 41.5, 'vertical post', fontsize=7.4, ha='left', va='center',
             color=BAR)
    for yv in (19.0, 27.0, 36.0):
        axC.plot([60.4], [yv], 'o', ms=3.0, color=INK)
        axC.add_patch(rect(61, yv - 1.3, 8, 2.6, fc='#d9f0e2', ec=BRK, lw=1.1))
        axC.plot([71.7], [yv], 'o', ms=3.0, color=BRK)
    axC.text(55, 27, u'side holes at\n“random” heights', fontsize=7.4,
             ha='right', va='center', color=INK)
    axC.annotate(u'the post’s slot is CONTINUOUS — a T-nut\n'
                 u'goes to ANY height, so every bracket\n'
                 u'lands wherever its hole happens to be,\n'
                 u'and the whole chamber slides up and\n'
                 u'down for beam-height trim',
                 xy=(71.7, 27), xytext=(78.5, 28), fontsize=7.6, color=BRK,
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color=BRK, lw=.8))
    axC.text(53, 10.0, u'Two posts, one either side, bolted to the cross bar '
             u'with the same\nLs (vertical to horizontal, in plane — exactly '
             u'what they are made\nfor). Nothing has to line up with anything: '
             u'measure once at the\nbeam, slide, tighten. It is also what lets you '
             u're-roll the chamber\n90° between runs — the holes move, the slot '
             u'does not care.',
             fontsize=7.9, color=INK, va='top', ha='left')

    # ================= D : clearance vs yaw =============================== #
    th = np.linspace(0, 45, 400)
    axD.plot(th, W * np.sin(np.radians(th)) + T * np.cos(np.radians(th)),
             color=DET4C, lw=2.4, label=u'det4, %g × %g mm' % (W, T))
    axD.plot(th, 470 * np.sin(np.radians(th)) + T * np.cos(np.radians(th)),
             color=MUTED, lw=1.2, ls='--', label='PCB alone, 470 mm')
    axD.axhline(a.gap, color=INK, lw=1.4)
    axD.text(44.4, a.gap + 6, 'ASSUMED %g mm gap' % a.gap, fontsize=8.6, ha='right')
    axD.axhline(a.gap - 40, color=MUTED, lw=1.0, ls=':')
    axD.text(44.4, a.gap - 36, '20 mm clearance each side', fontsize=8, ha='right',
             color=MUTED)
    axD.axvspan(0, th_saf, color=BRK, alpha=0.09, lw=0)
    axD.axvline(th_max, color=INK, lw=1.0, ls=':')
    axD.plot([th_saf], [a.gap - 40], '*', ms=16, color=BRK, zorder=6)
    axD.annotate(u'usable yaw  ±%.0f°\n(|tanθ| ≤ %.2f)'
                 % (th_saf, math.tan(math.radians(th_saf))),
                 xy=(th_saf, a.gap - 40), xytext=(th_saf - 1.5, a.gap - 130),
                 fontsize=10, color=BRK, ha='right',
                 arrowprops=dict(arrowstyle='->', color=BRK, lw=1.1))
    for tt, lab in ((2.9, u'|tanθ| 0.05'), (6.8, '0.12'), (14.0, '0.25'),
                    (31.0, '0.60')):
        axD.axvline(tt, color=BEAMC, lw=0.8, ls=(0, (3, 3)), alpha=.7)
        axD.text(tt - 0.4, 8, lab, fontsize=7.6, color=BEAMC, rotation=90,
                 va='bottom', ha='right')
    axD.text(44.4, 108, u'dashed blue: the inclination bins §3b\nof the '
             'assessment measures the band in', fontsize=8.2, color=BEAMC,
             ha='right', va='bottom')
    axD.set_xlim(0, 45)
    axD.set_ylim(0, 460)
    axD.set_xlabel(u'yaw about the vertical axis [°]')
    axD.set_ylabel('along-beam depth swept [mm]')
    axD.set_title('D   How much of the gap the chamber eats as it turns',
                  fontsize=11, loc='left', fontweight='bold')
    axD.legend(fontsize=8.8, loc='upper left', frameon=False)
    axD.grid(alpha=0.18, lw=0.6)
    for ax in (axA, axB, axD):
        ax.tick_params(labelsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.suptitle('det4 into the SPS H4 rail set   —   board geometry MEASURED '
                 '(DFS3498A Gerbers + the June cosmic band);  rail set ASSUMED from '
                 'the 2026-07-31 description', fontsize=12.5, x=0.05, ha='left')
    png = os.path.join(a.out, 'beam_mount_layout.png')
    fig.savefig(png, dpi=160)
    print('wrote', png)

    print(f'\nband centre {band_c:.1f} mm above the chamber foot '
          f'(bare PCB edge + {a.frame_below_pcb:.0f} mm of frame)')
    print(f'beam ASSUMED {a.beam_height:.0f} mm above the rail top '
          f'-> {foot:+.1f} mm of mount height available under the chamber')
    print(f'yaw: hard limit {th_max:.1f} deg; with 20 mm clearance each side '
          f'{th_saf:.1f} deg (tan {math.tan(math.radians(th_saf)):.2f})')
    print(f'clamp bolts travel +-{travel:.0f} mm along the outer rails; '
          f'transverse pull-in {pullin:.1f} mm')


if __name__ == '__main__':
    main()
