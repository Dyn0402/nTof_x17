#!/usr/bin/env python3
"""
make_capsule_pose_deck.py -- the capsule-pose study as a figure-first slide
note: the station and the capsule to scale, how tracks locate a source, the
one compact source ~10 mm off the frame centre at every ToF, where neutrons
are captured on the capsule in each picture, and the two readings that remain.

Reads the analysis's own outputs (capsule_pose.py):
  <out>/capsule_pose/{data,early,image,alternatives,halo,bound,fit}.csv
  <out>/capsule_tof/{tof,beam_capture}.csv
  <out>/imaging_campaign/per_run.csv
and recomputes, through capsule_pose itself, the capture clouds it draws and a
sample of real track lines.

    python -m sept26_prelim_analysis.make_capsule_pose_deck [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug capsule-pose --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)
from sept26_prelim_analysis import paths                               # noqa: E402
from sept26_prelim_analysis import capsule_pose as CP                  # noqa: E402

POSE = paths.out('capsule_pose')
CTOF = paths.out('capsule_tof')
IMG = paths.out('imaging_campaign')
ARM_COL = dict(A=BLUE, C=ORANGE, B=GREEN, D=PURPLE)
#: colours of the pictures, fixed across slides
COL_ON = INK          # flux centred on the capsule
COL_103 = PURPLE      # beam at 0, capsule 10.3 mm off
COL_13 = ORANGE       # beam at 0, capsule 13 mm off
COL_TAIL = GREY       # beam at 0, capsule 15-20 mm off (tail-lit)
SRC = RED             # the measured source
BEAMC = GOLD          # the beam
#: chamber strip-plane centres (run_config.json, survey) and half-width
CENTRE = {'A': (-16.35, 234.6), 'B': (-234.1, -15.75), 'C': (17.3, -234.6), 'D': (234.1, 15.5)}  # (x, z)
HALF = CP.HALF


def rd(d, name):
    return pd.read_csv(d / f'{name}.csv')


def load():
    A = {n: rd(POSE, n) for n in ('data', 'early', 'image', 'alternatives', 'halo', 'bound', 'fit')}
    A['tof'] = rd(CTOF, 'tof')
    A['beamcap'] = rd(CTOF, 'beam_capture')
    A['per_run'] = rd(IMG, 'per_run')
    A['late'] = A['data'][A['data'].window == 'late'].set_index('arm')
    A['alt'] = pd.concat([A['alternatives'], A['halo']], ignore_index=True)
    al = A['alt']
    al = al[al.window == 'late'].copy()
    al['x'] = 0.5 * (al.x_cross_A + al.x_cross_C)
    al['y'] = 0.5 * (al.y_cross_A + al.y_cross_C)
    A['alt_late'] = al
    F0 = A['fit'][A['fit'].theta == 0].set_index('d')
    A['x_src'] = float(F0.loc[0, 'x_beam'])           # A/C mean, toy estimator offset removed
    A['z_src'] = float(F0.loc[0, 'z_beam'])
    A['y_src'] = float(0.5 * (A['late'].loc['A', 'y_cross'] + A['late'].loc['C', 'y_cross']))
    A['dy0'] = float(F0.loc[0, 'dy_c'])
    e, pdf = CP.radial_profile(CP.WINDOWS['late'])
    c = np.cumsum(pdf)
    r = e[1:]
    A['r50'], A['r90'], A['r99'] = (float(r[np.searchsorted(c, q)]) for q in (0.5, 0.9, 0.99))
    return A


def numbers(A):
    N = {}
    E = A['early']
    d = E[E.kind == 'data'].set_index(['arm', 'lo'])
    mix = E[(E.kind == 'mix') & (E.arm == 'A')].set_index('frac_at_plus10').peak_x
    N['slope'] = float((mix.loc[0.5] - mix.loc[0.0]) / 0.5)
    ref = {a: float(d.loc[(a, 30.0), 'peak_x']) for a in 'AC'}

    def lim(a, lo):
        sh = float(d.loc[(a, lo), 'peak_x']) - ref[a]
        return max(0.0, (sh + 2 * float(d.loc[(a, lo), 'peak_err'])) / N['slope'])
    N['f46'] = {a: lim(a, 4.0) for a in 'AC'}
    N['f612'] = max(lim(a, lo) for a in 'AC' for lo in (6.0, 8.0, 10.0))
    N['peak'] = ref
    al = A['alt_late'].copy()
    ys = A['y_src']
    al['dy'] = ys - al.y
    al['dx'] = al.x - 0.5 * (A['late'].loc['A', 'x_cross'] + A['late'].loc['C', 'x_cross'])
    off = al[~al.case.str.startswith('beam on')]
    near = off[off.dx.abs() < 2.5]
    N['dy_min'] = float(near.dy.abs().min())
    N['dy_max'] = float(near.dy.abs().max())
    N['gas_off_max'] = float(near.gas.max())
    N['gas_off_min'] = float(near.gas.min())
    B = A['bound'].set_index('height_prior_mm')
    N['b5'] = B.loc[5]
    PR = A['per_run']
    N['post'] = PR[PR.condition.str.startswith('post')].groupby('arm').mm.agg(['median', 'std'])
    N['pre'] = PR[PR.condition.str.startswith('pre')].groupby('arm').mm.median()
    return N


# --------------------------------------------------------------------------- #
# geometry drawing helpers (data units = mm)
# --------------------------------------------------------------------------- #
def circle(P, xc, yc, r, color, w=2.5, dash=None, fill='none', opacity=1.0, tip=None):
    """A circle of radius r (mm) in a Plot with equal scales."""
    cx, cy = P.X(xc), P.Y(yc)
    rx = abs(P.X(xc + r) - cx)
    ry = abs(P.Y(yc + r) - cy)
    ds = f' stroke-dasharray="{dash}"' if dash else ''
    P.fore.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" '
                  f'fill-opacity="{opacity}" stroke="{color}" stroke-width="{w}"{ds}{sd.tipattr(tip)}/>')


def star(P, x, y, color, r=14, tip=None):
    px, py = P.X(x), P.Y(y)
    pts = []
    for k in range(10):
        a = -np.pi / 2 + k * np.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append(f'{px + rr * np.cos(a):.1f},{py + rr * np.sin(a):.1f}')
    P.fore.append(f'<polygon points="{" ".join(pts)}" fill="{color}" stroke="white" stroke-width="1.5"'
                  f'{sd.tipattr(tip)}/>')


def cross(P, x, y, color=MUT, s=10, label=None, tip=None):
    px, py = P.X(x), P.Y(y)
    P.fore.append(sd.line(px - s, py, px + s, py, color, 2.5) + sd.line(px, py - s, px, py + s, color, 2.5))
    if tip:
        P.fore.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="16" fill="transparent"{sd.tipattr(tip)}/>')
    if label:
        P.fore.append(sd.T(px + 14, py - 12, label, 20, color, 'start'))


def capsule_side(P, x0, y0, color=INK, gas=True, tip=None):
    """The nose-first capsule in side view (x across, y up), axis at x0, local
    s = 0 at y0: CFRP, Al and the gas outlines."""
    for S, R, col, w in ((CP.S_AL, CP.R_CF, GREY, 1.5), (CP.S_AL, CP.R_AL, color, 2.5)) + \
            (((CP.S_GAS, CP.R_GAS, BLUE, 1.8),) if gas else ()):
        xs = list(x0 + np.r_[R, -R[::-1]])
        ys = list(y0 + np.r_[S, S[::-1]])
        P.fore.append(sd.poly([P.X(a) for a in xs], [P.Y(b) for b in ys], col, w, tip=tip))


def capsule_top(P, xc, zc, color=INK, label=None, tip=None, dash=None):
    """Top-down capsule outline (barrel OD 21.2 mm, CFRP 23 mm) at (x, z);
    the plot's horizontal is z and its vertical is x."""
    circle(P, zc, xc, 11.5, GREY, 1.5, dash=dash)
    circle(P, zc, xc, 10.6, color, 2.5, dash=dash, tip=tip)
    if label:
        P.text(zc + 12, xc + 12, label, 19, color)


def beam_top(P, xb, zb, r50, r90, label=None):
    circle(P, zb, xb, r90, BEAMC, 0, fill=BEAMC, opacity=0.18,
           tip=f'beam, 90 % of the late-window flux inside r = {r90:.0f} mm (Lambda2D)')
    circle(P, zb, xb, r50, BEAMC, 0, fill=BEAMC, opacity=0.45,
           tip=f'beam core, 50 % inside r = {r50:.1f} mm')
    if label:
        P.text(zb - r90 - 2, xb + r90 + 3, label, 19, '#9a6f0f', 'end')


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, A, N):
    b5 = N['b5']
    body = (sd.kicker('n_TOF 2026 · X17 · capsule pose in the beam')
            + sd.p('One compact source on the capsule nose sits ~10 mm off the frame centre, at every time of '
                   'flight. Either the beam or the capsule is not where we think.', 54, DINK, 600,
                   'line-height:1.15;letter-spacing:-0.5px')
            + sd.row(
                sd.bignum(f'{A["x_src"]:+.1f} mm', 'source x (toward B)', DRED,
                          f'z {A["z_src"]:+.1f} mm (toward C); A and C agree, every run, 2–75 ms',
                          tip='A/C late band crossings −10.1 / −10.7, toy estimator offset removed; z from B/D'),
                sd.bignum(f'{A["y_src"]:+.0f} mm', 'source height', DGREY,
                          'the nose: Geant4 puts it at −29 for the drawn capsule',
                          tip='late y band crossing, y mirrored: A −31.3, C −34.6'),
                sd.bignum(f'≥ {100 * b5.gas_min:.0f} %', 'of the beam on the ³He, IF the beam is on the capsule', DGREEN,
                          'reading 1: the beam / frame is ~10 mm off the alignment'),
                sd.bignum(f'{100 * N["gas_off_min"]:.0f}–{100 * N["gas_off_max"]:.0f} %',
                          'IF the beam is centred as aligned', DBLUE,
                          f'reading 2: capsule far off the beam and ≥ {N["dy_min"]:.0f} mm lower than drawn'),
                gap=48))
    notes = """
<p>This note is the visual companion of <code>&lt;out&gt;/capsule_pose/report.html</code>
(<code>capsule_pose.py</code>, <code>make_capsule_pose_report.py</code>). Everything uses the post-access runs,
chambers A and C on the is2_v2 chain at the capsule-focus k, with stage-3 y mirrored on load (stage 3 stores y
upside down; COORDINATES.md).</p>
<p>An earlier version of the analysis called the source "the beam at x −10.3". That was retracted: the source is where
the neutrons that illuminate the capsule are captured on aluminium, and it is the beam position only if the beam is the
illumination and actually hits the capsule. The slides show why the two readings on the right both survive.</p>"""
    D.slide('cover', body, notes, dark=True)


def s_setup(D, A, N):
    # station, top-down: horizontal z, vertical x (COORDINATES.md: +z right, +x up)
    P = sd.Plot(620, 640, x=(-300, 300), y=(-300, 300), xlabel='z (mm)', ylabel='x (mm)',
                margin=(24, 24, 92, 104), title='the station from above')
    P.xticks([(v, str(v)) for v in (-200, 0, 200)])
    P.yticks([(v, str(v)) for v in (-200, 0, 200)])
    for arm, (xc, zc) in CENTRE.items():
        tip = (f'chamber {arm}: strip plane centre at x {xc:+.2f}, z {zc:+.2f} mm (run_config.json, survey); '
               f'active ±{HALF:.0f} mm along its u; pinwheel shift {abs(xc if arm in "AC" else zc):.2f} mm')
        if arm in 'AC':
            P.fore.append(sd.line(P.X(zc), P.Y(xc - HALF), P.X(zc), P.Y(xc + HALF), ARM_COL[arm], 9))
            P.fore.append(f'<rect x="{P.X(zc) - 10:.1f}" y="{P.Y(xc + HALF):.1f}" width="20" '
                          f'height="{P.Y(xc - HALF) - P.Y(xc + HALF):.1f}" fill="transparent"{sd.tipattr(tip)}/>')
            P.fore.append(sd.line(P.X(zc) - 14, P.Y(xc), P.X(zc) + 14, P.Y(xc), ARM_COL[arm], 3))
            P.text(zc + (24 if zc > 0 else -24), xc + HALF - 20, arm, 30, ARM_COL[arm],
                   'start' if zc > 0 else 'end', 600)
        else:
            P.fore.append(sd.line(P.X(zc - HALF), P.Y(xc), P.X(zc + HALF), P.Y(xc), ARM_COL[arm], 9))
            P.fore.append(f'<rect x="{P.X(zc - HALF):.1f}" y="{P.Y(xc) - 10:.1f}" '
                          f'width="{P.X(zc + HALF) - P.X(zc - HALF):.1f}" height="20" fill="transparent"{sd.tipattr(tip)}/>')
            P.fore.append(sd.line(P.X(zc), P.Y(xc) - 14, P.X(zc), P.Y(xc) + 14, ARM_COL[arm], 3))
            P.text(zc + HALF - 10, xc + (22 if xc > 0 else -40), arm, 30, ARM_COL[arm], 'end', 600)
    P.rect(-30, 30, -30, 30, MUT, 1.5, dash='5 4', tip='the zoomed square on the right (±30 mm)')
    cross(P, 0, 0, INK, 8, tip='frame centre: the surveyed station centre; the beam was aligned to it')
    # zoom
    Z = sd.Plot(560, 640, x=(-30, 30), y=(-30, 30), xlabel='z (mm)', ylabel='x (mm)',
                margin=(24, 24, 92, 104), title='zoom: beam, capsule, source')
    Z.xticks([(v, str(v)) for v in (-20, -10, 0, 10, 20)])
    Z.yticks([(v, str(v)) for v in (-20, -10, 0, 10, 20)])
    beam_top(Z, 0, 0, A['r50'], A['r90'], label='beam, as aligned')
    capsule_top(Z, 0, 0, INK, label='capsule as drawn',
                tip='capsule barrel OD 21.2 mm (Al) + 0.9 mm CFRP, at the drawn (Geant4) position')
    cross(Z, 0, 0, INK, 7)
    star(Z, A['z_src'], A['x_src'], SRC, 16,
         tip=f'measured source: x {A["x_src"]:+.1f} (A/C), z {A["z_src"]:+.1f} (B/D) mm')
    Z.text(A['z_src'] - 2, A['x_src'] - 4, 'measured source', 20, SRC, 'end', 600)
    Z.raw(sd.arrow(Z.X(-22), Z.Y(-24), Z.X(-22), Z.Y(-29), GREEN, 2.5, 10))
    Z.text(-21, -27, 'toward B', 18, GREEN)
    Z.raw(sd.arrow(Z.X(-24), Z.Y(27), Z.X(-29), Z.Y(27), ORANGE, 2.5, 10))
    Z.text(-23, 26, 'toward C', 18, ORANGE)
    # side view
    S = sd.Plot(440, 640, x=(-30, 30), y=(-60, 70), xlabel='x (mm)', ylabel='height y (mm)',
                margin=(24, 24, 92, 104), title='side view')
    S.xticks([(v, str(v)) for v in (-20, 0, 20)])
    S.yticks([(v, str(v)) for v in (-50, -25, 0, 25, 50)])
    S.band([-A['r90'], A['r90']], [-60, -60], [70, 70], BEAMC, 0.15, tip='beam (90 %), travelling up')
    S.band([-A['r50'], A['r50']], [-60, -60], [70, 70], BEAMC, 0.35, tip='beam core (50 %)')
    capsule_side(S, 0, 0, tip='capsule, nose-first: Al tip at y −35, gas from −29.5, valve at +51')
    S.raw(sd.arrow(S.X(0), S.Y(-56), S.X(0), S.Y(-42), '#9a6f0f', 3, 12))
    S.hline(A['y_src'], SRC, '6 5', 2.5, tip=f'measured source height {A["y_src"]:+.1f} mm (A/C mean)')
    star(S, A['x_src'], A['y_src'], SRC, 14, tip='measured source (x, height)')
    S.text(-28, A['y_src'] - 9, 'source', 19, SRC, 'start', 600)
    S.text(12, 30, '³He 500 bar', 18, BLUE)
    S.text(12, -42, 'Al nose', 18, INK)
    body = (sd.title('The station, the capsule and the measured source, to scale',
                     'Hover the chambers, the capsule, the beam and the star. Dotted words and points everywhere carry tooltips.')
            + sd.row(P.svg('station'), Z.svg('zoom'), S.svg('side view'), gap=20))
    notes = f"""
<p><b>Frame.</b> +y is the beam, vertical and up; +z points right (A), +x up (D) in this top-down view
(COORDINATES.md). The origin is the surveyed station centre, which is also where the capsule is drawn (Geant4) and
what the beam was aligned to. Each chamber's strip plane is 234.6 mm (A, C) or 234.1 mm (B, D) out and is shifted
15.5–17.3 mm sideways (the "pinwheel"), so no chamber is centred on the axis.</p>
<p><b>Capsule.</b> The STEP polycones from MX17_Full_Geant: a 0.6 mm Al barrel (OD 21.2 mm) around 500 bar ³He, a
5.5 mm thick Al nose at the bottom (upstream), a 0.9 mm CFRP wrap. Beam neutrons that reach the ³He are absorbed
within ~0.15 mm, so the gas is black: Al captures (the 7.7 MeV γ that make the tracks) happen where the beam first
meets Al, before the gas.</p>
<p><b>Beam.</b> The late-window Lambda2D profile (EAR2 Phase-3 flux × the n_TOF transport radial profile per
energy, as the Geant4 gun samples it): 50 % inside r = {A['r50']:.1f} mm, 90 % inside {A['r90']:.0f} mm,
99 % inside {A['r99']:.0f} mm.</p>
<p><b>The star</b> is the measured source: x from chambers A and C, z from B and D, height from the y band crossing
(scale-free; stage-3 y mirrored back).</p>"""
    D.slide('setup', body, notes, foot='geometry: run_config.json (survey), DetectorConstruction.cc polycones · '
                                       'beam: lamda2DvsEn_EAR2.root × fluxEAR2-Ph3')


def s_lines(D, A, N):
    d = CP.load_data(arms=('A', 'C'))
    late = d[d.ms >= 20]
    rng = np.random.default_rng(4)
    P = sd.Plot(980, 630, x=(-260, 260), y=(-140, 120), xlabel='z (mm)', ylabel='x (mm)',
                margin=(24, 24, 92, 104), title='280 real tracks per chamber, ≥ 20 ms, extended to the axis plane')
    P.xticks([(v, str(v)) for v in (-234.6, -100, 0, 100, 234.6)])
    P.yticks([(v, str(v)) for v in (-100, -50, 0, 50, 100)])
    for arm, zc in (('A', 234.6), ('C', -234.6)):
        g = late[late.arm == arm]
        g = g[(g.pos.abs() < 125) & ((g.pos + g.disp).abs() < 110)]
        s = g.iloc[rng.choice(len(g), 280, replace=False)]
        segs = []
        for pos, disp in zip(s.pos, s.disp):
            x_axis = pos + disp
            segs.append(sd.line(P.X(zc), P.Y(pos), P.X(-0.04 * zc), P.Y(x_axis - 0.04 * (pos - x_axis)),
                                ARM_COL[arm], 1.1))
        P.back.append(f'<g stroke-opacity="0.22"{sd.tipattr(arm + ": track lines, chamber plane to beyond the axis")}>'
                      + ''.join(segs) + '</g>')
        xc = CENTRE[arm][0]
        P.fore.append(sd.line(P.X(zc), P.Y(xc - 130), P.X(zc), P.Y(xc + 120), ARM_COL[arm], 8))
        P.text(zc + (-16 if zc > 0 else 16), 108, arm, 30, ARM_COL[arm], 'end' if zc > 0 else 'start', 600)
    P.vline(0, MUT, '6 5', 1.5)
    cross(P, 0, 0, INK, 9, tip='frame centre')
    star(P, 0, A['x_src'], SRC, 13, tip=f'source x {A["x_src"]:+.1f} mm')
    P.text(4, -128, 'axis plane z = 0', 20, MUT)
    # right: the excess profile, data vs shuffled null
    g = late[late.arm == 'A'].sample(300_000, random_state=1)
    cx = (g.pos + g.disp).to_numpy()
    c0 = (g.pos + rng.permutation(g.disp.to_numpy())).to_numpy()
    edges = np.arange(-120, 121, 4.0)
    h, _ = np.histogram(cx, edges)
    h0, _ = np.histogram(c0, edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    R = sd.Plot(640, 630, x=(-120, 120), y=(0, 1.05 * h.max() / len(g) * 100), xlabel='x at the axis plane (mm)',
                ylabel='tracks per 4 mm (%)', title='chamber A: where the lines cross z = 0')
    R.xticks([(v, str(v)) for v in (-100, -50, 0, 50, 100)])
    ymax = 1.05 * h.max() / len(g) * 100
    R.yticks([(v, f'{v:g}') for v in np.arange(0, ymax, 2.0)])
    R.line(list(mid), list(h / len(g) * 100), BLUE, 3, markers=False,
           tip='data: x where each line crosses the axis plane')
    R.line(list(mid), list(h0 / len(g) * 100), GREY, 3, dash='8 6', markers=False,
           tip='the same tracks with their angles shuffled among each other: no pointing, same acceptance')
    R.vline(N['peak']['A'], SRC, '6 5', 2.5, tip=f'peak of data − shuffled: x {N["peak"]["A"]:+.1f} mm')
    R.vline(0, MUT, '4 6', 1.5)
    R.text(N['peak']['A'] - 4, ymax * 0.93, f'peak {N["peak"]["A"]:+.1f}', 21, SRC, 'end', 600)
    R.text(40, ymax * 0.55, 'shuffled angles', 20, GREY)
    R.text(40, ymax * 0.62, 'data', 20, BLUE)
    body = (sd.title('Tracks from A and C converge ~10 mm toward B, not on the centre',
                     'left: real lines from A and C in the top-down view · right: their crossing points against a no-pointing null')
            + sd.row(P.svg('track lines'), R.svg('crossing'), gap=44))
    notes = """
<p>Each line is one reconstructed track (gated, angle-calibrated, |tan| &lt; 0.6, ≥ 20 ms), drawn from its chamber's
strip plane through the axis plane z = 0 and a little beyond. Individual lines miss the source by ~25–40 mm: the γ
converts or Compton-scatters in the capsule wall and air, and the electron scatters (Geant4: p·β·θ ≈ 0.4 MeV·rad). The
ensemble still focuses, and the focus is the measurement.</p>
<p><b>Two estimators, same answer.</b> The <i>band crossing</i> fits displacement-to-the-axis against position on the
plane and takes the zero; it is scale-free (a wrong angle scale changes the slope, not the zero). The <i>pointing
excess</i> (right) is the histogram of crossing points minus the same tracks with shuffled angles, which keeps the
acceptance and destroys pointing; its peak survives background that ruins the band crossing before ~10 ms.</p>
<p>Chamber A's u axis points along +x and C's along −x, yet both find −10. A shift, a tan bias or a twist of one
chamber would move the two images in opposite directions.</p>"""
    D.slide('lines', body, notes, foot='is2_v2 cf, post-access runs, gated & angle-calibrated tracks · capsule_pose.load_data')


def s_image(D, A, N):
    I = A['image']
    panels = []
    for w, lab in (('beam', '4–12 ms: direct beam'), ('mid', '12–20 ms'), ('late', '30–75 ms: after the flux ends')):
        g = I[(I.arm == 'A') & (I.window == w)]
        vmax = float(g.excess.max())
        P = sd.Plot(540, 660, x=(-60, 40), y=(-160, 120), xlabel='x at the axis plane (mm)',
                    ylabel='height (mm)' if w == 'beam' else '', margin=(24, 16, 92, 104 if w == 'beam' else 40),
                    title=lab)
        P.xticks([(v, str(v)) for v in (-50, -25, 0, 25)])
        P.yticks([(v, str(v)) for v in (-150, -100, -50, 0, 50, 100)] if w == 'beam' else [])
        cells = []
        for r in g.itertuples():
            if -160 <= r.h <= 120:
                cells.append((r.x - 2, r.x + 2, r.h - 10, r.h + 10, max(r.excess, 0) / vmax,
                              f'x {r.x:+.0f}, height {r.h:+.0f}: {1000 * r.excess:+.2f} per 1000 tracks over the null'))
        P.cells(cells, 0, 1, sd.VIRIDIS)
        capsule_side(P, 0, 0, '#ffffff', gas=False, tip='capsule as drawn, at the frame centre')
        P.vline(0, '#ffffff', '6 5', 1.5)
        panels.append(P.svg(lab))
    cb = sd.colorbar(130, 560, 0, 1, [(0, '0'), (0.5, '½'), (1, 'max')], 'excess', sd.VIRIDIS)
    body = (sd.title('The pointing image is one compact spot left of the frame centre, at all times',
                     'chamber A, lines at the axis plane minus the shuffled-angle null, normalised per panel; white: the capsule as drawn')
            + sd.row(*panels, cb, gap=16))
    notes = """
<p>Per window, a 2D histogram of where each line crosses the axis plane (x, height) minus the same with the x and y
angles shuffled together among the tracks. The spot is elongated vertically because individual lines miss by
±25–40 mm in height (electron scattering, and the y angle scale is not calibrated); its <i>centre</i> is what the
crossing estimators measure.</p>
<p>What is <b>not</b> there: a vertical ridge at the beam's x from captures along the beam line (¹⁴N(n,γ) in the air
column gives 10.8 MeV γ at roughly the Al nose's rate per neutron), and a second spot near x = 0. Chamber C looks the
same (image.csv).</p>"""
    D.slide('image', body, notes, foot='capsule_pose image.csv (A); heights mirrored back to +y = up')


def s_stable(D, A, N):
    E = A['early']
    d = E[E.kind == 'data']
    P = sd.Plot(820, 600, x=(np.log10(2), np.log10(75)), y=(-14, -2), xlabel='time since the flash (ms)',
                ylabel='source x (mm)', title='against time of flight')
    P.xticks([(np.log10(v), str(v)) for v in (2, 5, 10, 20, 50)])
    P.yticks([(v, str(v)) for v in (-14, -12, -10, -8, -6, -4, -2)])
    mix = E[(E.kind == 'mix') & (E.arm == 'A')].set_index('frac_at_plus10').peak_x
    for f in (0.2, 0.5):
        P.hline(float(mix.loc[f]), GREY, '6 6', 1.5, label=f'{f:.0%} of tracks from a source 10 mm toward 0',
                tip='calibration: late data mixed with a copy of themselves moved +10 mm', anchor='end')
    for arm in 'AC':
        g = d[d.arm == arm]
        t = np.log10(np.sqrt(g.lo * g.hi))
        tips = [f'{arm}, {lo:g}–{hi:g} ms: x {v:+.1f} ± {e:.1f} mm ({n:,.0f} tracks)'
                for lo, hi, v, e, n in zip(g.lo, g.hi, g.peak_x, g.peak_err, g.n)]
        P.line(list(t), list(g.peak_x), ARM_COL[arm], tips=tips)
        for x, y, e in zip(t, g.peak_x, g.peak_err):
            P.fore.append(sd.line(P.X(x), P.Y(y - e), P.X(x), P.Y(y + e), ARM_COL[arm], 2.5))
    P.text(np.log10(2.2), -13.3, 'A', 24, BLUE, 'start', 600)
    P.text(np.log10(2.7), -13.3, 'C', 24, ORANGE, 'start', 600)
    PR = A['per_run']
    R = sd.Plot(800, 600, x=(76, 165), y=(-14, 0), xlabel='run', ylabel='source position (mm)',
                title='against run (campaign imaging, production chain)')
    R.xticks([(v, str(v)) for v in (80, 100, 120, 140, 160)])
    R.yticks([(v, str(v)) for v in (-14, -12, -10, -8, -6, -4, -2, 0)])
    R.band([76, 82.5], [-14, -14], [0, 0], GREY, 0.12, tip='pre-access runs 79, 81 (before the 27 July access)')
    for arm in 'ACDB':
        g = PR[PR.arm == arm].sort_values('num')
        tips = [f'{arm} {r}: {v:+.1f} mm ({"x" if arm in "AC" else "z"}), {n:,} tracks'
                for r, v, n in zip(g.run, g.mm, g.n)]
        R.line(list(g.num), list(g.mm), ARM_COL[arm], 2.5, r=5, tips=tips, dash=None if arm in 'AC' else '6 5')
    R.text(78, -1.3, 'pre-access', 20, MUT)
    R.text(160, -12.8, 'A, C (x)', 20, INK, 'end')
    R.text(160, -0.8, 'B, D (z)', 20, INK, 'end')
    f46 = N['f46']
    body = (sd.title('It does not move: the same x from 2 to 75 ms, and in every run',
                     'left: pointing-excess peak per ToF bin (band crossing unusable < 10 ms) · right: per-run crossings')
            + sd.row(P.svg('vs ToF'), R.svg('vs run'), gap=44))
    notes = f"""
<p><b>Left.</b> The excess peak is read with gain 1 on a point source (toy: −5 → −4.7, −10 → −9.7). Mixing the late
data with a copy of itself moved +10 mm shifts it by {N['slope'] / 10:.1f} mm × fraction × 10, linearly (the grey
lines). Diluting the late data with 1–2 ms tracks (mostly background) moves it by ≤ 2 mm. A beam centred on the frame
with the capsule partly in it would put its own captures near x ≈ 0: at 2σ, at most {100 * f46['A']:.0f} % (A) /
{100 * f46['C']:.0f} % (C) of the capsule tracks at 4–6 ms and {100 * N['f612']:.0f} % at 6–12 ms come from there.
A at 4–6 ms is the only bin that moves, and C in the same bin does not, so it is more likely background structure.</p>
<p><b>Right.</b> The campaign-imaging crossings per run (production chain): post-access A {N['post'].loc['A', 'median']:+.1f}
(s.d. {N['post'].loc['A', 'std']:.1f}), C {N['post'].loc['C', 'median']:+.1f} ({N['post'].loc['C', 'std']:.1f}), D
{N['post'].loc['D', 'median']:+.1f}; pre-access A {N['pre'].loc['A']:+.1f}, C {N['pre'].loc['C']:+.1f}. The access did not
move it. The production chain reads A ~2 mm closer to 0 than is2_v2; C agrees between chains.</p>"""
    D.slide('stable', body, notes, foot='capsule_pose early.csv · imaging_campaign per_run.csv')


def capture_panel(pose, ws, title, color, toy_xy, A, show_beam_at=0.0, w=540):
    C, fr = CP.beam_captures(pose, n=150_000, window=CP.WINDOWS['late'], width_scale=ws)
    C = C[C.mat == 2]
    rng = np.random.default_rng(0)
    k = rng.choice(len(C), min(2500, len(C)), p=(C.w / C.w.sum()).to_numpy())
    P = sd.Plot(w, 620, x=(-32, 22), y=(-48, 80), xlabel='x (mm)', ylabel='height (mm)',
                margin=(24, 16, 92, 96), title=title)
    P.xticks([(v, str(v)) for v in (-30, -20, -10, 0, 10, 20)])
    P.yticks([(v, str(v)) for v in (-40, -20, 0, 20, 40)])
    xb = pose['x_b']
    r50, r90 = A['r50'] * ws, A['r90'] * ws
    P.band([xb - min(r90, 30), xb + min(r90, 30)], [-48, -48], [80, 80], BEAMC, 0.12, tip=f'beam 90 % (r {r90:.0f} mm)')
    P.band([xb - r50, xb + r50], [-48, -48], [80, 80], BEAMC, 0.30, tip=f'beam core 50 % (r {r50:.1f} mm)')
    capsule_side(P, pose['x_c'], 0, tip='capsule outline (Al, CFRP, gas)')
    P.scatter(list(C.x.iloc[k]), list(C.y.iloc[k]), color, 2.2, 0.35,
              tip='Al capture points, drawn in proportion to the capture rate')
    tx, ty = toy_xy
    star(P, tx, ty, color, 13, tip=f'what the chambers would measure: x {tx:+.1f}, height {ty:+.1f} mm')
    star(P, A['x_src'], A['y_src'], SRC, 13, tip=f'measured: x {A["x_src"]:+.1f}, height {A["y_src"]:+.1f} mm')
    P.text(-31, 74, f'³He gets {100 * fr["gas"]:.0f} % of the beam', 20, INK, 'start', 600)
    P.text(-31, 66, f'Al captures: nose {100 * fr["al_nose"]:.0f} %, barrel {100 * fr["al_barrel"]:.0f} %', 19, MUT)
    return P.svg(title), fr


def s_captures(D, A, N):
    al = A['alt_late']

    def xy(case, ws):
        r = al[(al.case == case) & (al.width_scale == ws)].iloc[0]
        return float(r.x), float(r.y)
    xs = A['x_src'] - xy('beam on the capsule axis', 1)[0]
    on = xy('beam on the capsule axis', 1)
    p1, _ = capture_panel(dict(CP.NOMINAL, x_c=A['x_src'], x_b=A['x_src']), 1, '1 · beam on the capsule',
                          COL_ON, (on[0] + xs, on[1]), A)
    p2, _ = capture_panel(dict(CP.NOMINAL, x_c=-10.3), 1, '2 · beam at 0, capsule 10 mm off', COL_103,
                          xy('beam at 0, capsule at -10.3', 1), A)
    p3, _ = capture_panel(dict(CP.NOMINAL, x_c=-17.5), 1, '3 · beam at 0, capsule 17.5 mm off', COL_TAIL,
                          xy('beam at 0, capsule at -17.5 (tail-lit)', 1), A)
    body = (sd.title('Where the neutrons are captured decides where the source appears',
                     'side views, capsule at drawn height; gold: beam; dots: Al captures; stars: predicted (colour) vs measured (red)')
            + sd.row(p1, p2, p3, gap=22))
    notes = """
<p>A ray walks up through the capsule: in Al it is captured with probability Σ ds (0.8 % through the 5.5 mm nose),
neutron scattering in the CFRP and Al takes it off its path, and if it reaches the ³He it is gone. The capture point
is therefore where the beam first meets Al.</p>
<p><b>1.</b> Beam on the capsule axis: almost every capture is in the nose, so the source sits at the nose and at
the beam's x. Moved to x −10.3 here, it lands on the measured star.</p>
<p><b>2.</b> Beam centred as aligned, capsule 10 mm off: the core grazes the barrel wall, which it runs along for
~40 mm (5 % capture per ray), so the captures climb the wall at the beam's x. Predicted source: x ≈ −1, height ≈ −8.</p>
<p><b>3.</b> Capsule far enough off that only the beam's tail reaches it: the near-side wall is lit, so the source x
can reach −10, but it sits on the barrel (height ≈ −12) and only ~1 % of the beam reaches the ³He.</p>
<p>The stars are not the centroid of the dots: they are the chambers' band crossings on Geant4 tracks translated to
these capture points, which is what the data are compared with.</p>"""
    D.slide('captures', body, notes)


def s_pictures(D, A, N):
    al = A['alt_late']
    xd = 0.5 * (A['late'].loc['A', 'x_cross'] + A['late'].loc['C', 'x_cross'])
    yd = A['y_src']
    P = sd.Plot(1060, 640, x=(-14, 1), y=(-38, -4), xlabel='source x (mm), beam as aligned at 0',
                ylabel='source height (mm)', title='every picture against the data (late window)')
    P.xticks([(v, str(v)) for v in (-14, -12, -10, -8, -6, -4, -2, 0)])
    P.yticks([(v, str(v)) for v in (-35, -30, -25, -20, -15, -10, -5)])
    for case, g in al.groupby('case', sort=False):
        g = g.sort_values('width_scale')
        tail = 'tail-lit' in case
        onc = case.startswith('beam on')
        col = COL_ON if onc else (COL_TAIL if tail else (COL_103 if '-10.3' in case else COL_13))
        x = g.x + (xd - g.x.iloc[0] if onc else 0.0)
        tips = [f'{case}{"" if tail else f", flux ×{w:g} the Lambda2D width"}: x {a:+.1f}, height {b:+.1f}; '
                f'capsule must move {yd - b:+.0f} mm; ³He gets {100 * q:.0f} % of the beam'
                for w, a, b, q in zip(g.width_scale, x, g.y, g.gas)]
        P.line(list(x), list(g.y), col, 3, r=7, tips=tips, dash='8 6' if tail else None)
        if not tail:
            for w, a, b in zip(g.width_scale, x, g.y):
                P.text(a + 0.15, b + 0.7, f'×{w:g}', 17, col)
        else:
            for xc, a, b in zip(g.x_c, x, g.y):
                P.text(a + 0.15, b - 1.4, f'{xc:+.1f}', 17, col)
    for arm in 'AC':
        P.text(A['late'].loc[arm, 'x_cross'] - 0.25, A['late'].loc[arm, 'y_cross'] - 0.4, f'data {arm}', 20,
               ARM_COL[arm], 'end', 600)
        P.points([A['late'].loc[arm, 'x_cross']], [A['late'].loc[arm, 'y_cross']], ARM_COL[arm], 11,
                 tips=[f'data {arm}: x {A["late"].loc[arm, "x_cross"]:+.1f}, height {A["late"].loc[arm, "y_cross"]:+.1f}'])
    # arrow: what a capsule height change does
    P.raw(sd.arrow(P.X(-9.6), P.Y(-18.5), P.X(-9.6), P.Y(-30), RED, 3, 14))
    P.text(-9.4, -24, 'lower the capsule', 19, RED)
    side = sd.col(
        sd.callout('<b>Only the black curve reaches the data at the drawn height.</b> It is the beam on the capsule; '
                   'moved to x −10 it needs the capsule just ' + f'{-A["dy0"]:.0f}' + ' mm lower.', COL_ON, 24),
        sd.callout('<b>Purple / orange:</b> beam centred as aligned, capsule 10 or 13 mm off, flux ×1…×10 the '
                   'Lambda2D width. To reach x −10 the flux must be ×10 wider, and the source is still '
                   f'~{N["dy_min"]:.0f} mm too high.', COL_103, 24),
        sd.callout('<b>Grey:</b> capsule 15–20 mm off, lit by the tail: x −10 at 17.5 mm off, 20 mm too high, and '
                   '~1 % of the beam on the ³He.', COL_TAIL, 24),
        sd.callout('A capsule-height change moves a point straight down: the height is degenerate with how much '
                   'barrel is lit; the x is not.', RED, 24), w=560, gap=20)
    body = (sd.title('With the beam as aligned, nothing reaches the data at drawn height',
                     'predicted source (x, height) for each picture, against chambers A and C')
            + sd.row(P.svg('pictures'), side, gap=44))
    notes = """
<p>Each curve is one picture with the flux width scaled from the Lambda2D profile (×1, 2, 3, 5, 10), computed by the
full model (ray trace + Geant4 tracks + the data's estimators); hover the points for the ³He share and the height
change each needs. The black curve is computed on the frame centre and moved sideways to the data's x, which is
legitimate because the model is translation-equivariant (closure: a +5 mm shift reads +4.9).</p>
<p>The data stars sit 2–6 mm below the black ×1 point. The capsule's height in the chamber frame has never been
measured, so a few mm is unremarkable; 14–25 mm (what every other picture needs) would be a mounting error.</p>
<p>Not in the grid: tilts. They were scanned for the beam-on-capsule family (0–20°, both senses) and change the
height by at most a few mm while the beam stays on the nose.</p>"""
    D.slide('pictures', body, notes, foot='capsule_pose alternatives.csv + halo.csv (late window, A/C mean)')


def s_late(D, A, N):
    T = A['tof']
    bc = A['beamcap']
    P = sd.Plot(820, 600, x=(np.log10(2), np.log10(75)), y=(0, 1.05), xlabel='time since the flash (ms)',
                ylabel='per ms, normalised to the maximum', title='when capsule tracks arrive')
    P.xticks([(np.log10(v), str(v)) for v in (2, 5, 10, 20, 50)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    b = bc[bc.lo >= 2]
    t = np.log10(np.sqrt(b.lo * b.hi))
    P.line(list(t), list(b.per_ms / b.per_ms.max()), BEAMC, 3,
           tips=[f'{lo:g}–{hi:g} ms: {100 * s:.1f} % of the beam captures (flux × 1/v)' for lo, hi, s in
                 zip(b.lo, b.hi, b.capture_share)])
    for arm in 'AC':
        g = T[(T.arm == arm) & (T.lo >= 2)]
        t = np.log10(np.sqrt(g.lo * g.hi))
        v = g.focus_per_ms / g.focus_per_ms.max()
        P.line(list(t), list(v), ARM_COL[arm], 3,
               tips=[f'{arm} {lo:g}–{hi:g} ms: {f:,.0f} capsule-pointing tracks per ms (recorded; DAQ-limited)'
                     for lo, hi, f in zip(g.lo, g.hi, g.focus_per_ms)])
    P.text(np.log10(3.2), 0.93, 'beam captures expected (flux × 1/v)', 20, '#9a6f0f')
    P.text(np.log10(24), 0.98, 'capsule tracks recorded', 20, INK)
    TT = rd(POSE, 'tof')
    R = sd.Plot(800, 600, x=(np.log10(4), np.log10(75)), y=(-48, -26), xlabel='time since the flash (ms)',
                ylabel='source height (mm)', title='the same nose height after the flux ends')
    R.xticks([(np.log10(v), str(v)) for v in (5, 10, 20, 50)])
    R.yticks([(v, str(v)) for v in (-45, -40, -35, -30)])
    for arm in 'AC':
        g = TT[TT.arm == arm]
        t = np.log10(np.sqrt(g.lo * g.hi))
        R.line(list(t), list(g.y_bgsub), ARM_COL[arm], 3,
               tips=[f'{arm} {lo:g}–{hi:g} ms: {y:+.1f} mm (bg-subtracted), purity {p:.2f}' for lo, hi, y, p in
                     zip(g.lo, g.hi, g.y_bgsub, g.purity)])
    R.hline(-29.0, GREY, '8 6', 2, label='Geant4 nose, capsule as drawn', where='below',
            tip='toy at the Geant4 pose, A/C mean')
    R.vline(np.log10(28), MUT, '4 6', 1.5, label='flux ends', tip='the evaluated EAR2 flux has ~nothing below 2.5 meV (after ~28 ms)')
    body = (sd.title('Most capsule tracks come after the beam, yet they image the same nose spot',
                     'left: arrival time of capsule-pointing tracks vs expected beam captures · right: source height vs ToF')
            + sd.row(P.svg('arrival'), R.svg('height'), gap=44))
    notes = """
<p><b>Left.</b> 87 % of the beam's own captures (EAR2 flux × 1/v) fall before 12 ms, peaking at 5–6 ms; 92 % of the
tracks that point at the capsule are after 12 ms and half after 30 ms (recorded rates, DAQ-limited, so only the
shape means anything). The ≥ 30 ms trigger rate falls with T½ ≈ 24 ms, read in the tracking handoff (§10f) as the
die-away of thermalised neutrons in the EAR2 hall.</p>
<p><b>Right.</b> Those late tracks image the nose (−31 / −35) and the same x. An isotropic hall flux would light the
whole capsule body and put the source at y ≈ +5 to +8. So whatever illuminates the capsule late is narrow and
on the nose, like a beam, which is not understood; it is also why the direct-beam test cannot isolate the beam: the
4–12 ms capsule tracks are a mixture whose beam share is unknown.</p>
<p>Before 12 ms the height drops; that follows the purity drop and no pose moves the beam window down, so it is read
as an unsubtracted beam-line source below the capsule.</p>"""
    D.slide('late', body, notes, foot='capsule_tof tof.csv / beam_capture.csv · capsule_pose tof.csv')


def s_readings(D, A, N):
    b5 = N['b5']

    def zoom(title, beam_xz, cap_xz, cap_lab, note_lines, col):
        Z = sd.Plot(540, 600, x=(-34, 26), y=(-34, 26), xlabel='z (mm)', ylabel='x (mm)',
                    margin=(24, 16, 92, 96), title=title)
        Z.xticks([(v, str(v)) for v in (-30, -20, -10, 0, 10, 20)])
        Z.yticks([(v, str(v)) for v in (-30, -20, -10, 0, 10, 20)])
        beam_top(Z, beam_xz[0], beam_xz[1], A['r50'], A['r90'])
        capsule_top(Z, cap_xz[0], cap_xz[1], col, tip=cap_lab)
        cross(Z, 0, 0, INK, 8, label='frame centre', tip='surveyed station centre (what the beam was aligned to)')
        star(Z, A['z_src'], A['x_src'], SRC, 14, tip='measured source')
        for i, s in enumerate(note_lines):
            Z.text(-33, 23 - 4.2 * i, s, 19, INK if i == 0 else MUT, 'start', 600 if i == 0 else 400)
        return Z.svg(title)
    z1 = zoom('reading 1: beam on the capsule', (A['x_src'], A['z_src']), (A['x_src'], A['z_src']),
              'capsule centred on the beam, ~10 mm off the frame centre',
              ['beam + capsule 10 mm toward B', 'capsule ~4 mm below drawn'], COL_ON)
    z2 = zoom('reading 2: beam as aligned', (0.0, 0.0), (-17.5, -6.0),
              'capsule far off the beam (13–18 mm), and ≥ 14 mm lower than drawn',
              ['capsule 13–18 mm off the beam', 'and 14–25 mm below drawn'], COL_TAIL)
    c1 = sd.col(sd.p('<b>Reading 1</b>', 30, INK),
                sd.p('The beam is not where the frame says, or the frame is not where it was aligned.', 24, INK),
                sd.p(f'³He gets ≥ {100 * b5.gas_min:.0f} % of the beam (capsule within ±5 mm of drawn).', 24, GREEN, 600),
                sd.p('Needs one surprise: a ~10 mm beam/frame offset.', 22, MUT), w=300, gap=14)
    c2 = sd.col(sd.p('<b>Reading 2</b>', 30, INK),
                sd.p('The beam is centred; the capsule is far off it, lower than drawn, lit by its tail or a wider beam.', 24, INK),
                sd.p(f'³He gets {100 * N["gas_off_min"]:.0f}–{100 * N["gas_off_max"]:.0f} % of the beam: IPC and X17 yields cut ×7–100.',
                     24, RED, 600),
                sd.p('Needs two surprises: the offset and a ≥ 14 mm drop.', 22, MUT), w=300, gap=14)
    body = (sd.title('Two readings survive, and only an outside measurement separates them',
                     'top-down zooms to scale: beam (gold), capsule (outline), frame centre (+), measured source (star)')
            + sd.row(z1, c1, z2, c2, gap=26))
    notes = """
<p><b>What would decide it:</b> how the beam centring was done and against what (laser on the beam axis? the capsule
holder? the frame?); a survey or photo of the capsule against the beam axis; the capsule tip's height against the
chamber mid-plane and how the capsule is held (anything massive near the nose would also explain why late,
non-beam neutrons image the nose); the beam spot at the station (collimator in use, distance from the 19.5 m point,
halo beyond Lambda2D's 3 cm).</p>
<p><b>What the data already exclude:</b> a Lambda2D beam at the frame centre hitting the capsule anywhere at its
drawn height; a source at x ≈ 0 making more than ~5 % of the capsule tracks after 6 ms; a single misplaced or
twisted chamber (A and C would disagree).</p>
<p>Neither reading explains the 1 : 8 same-chamber pair rate on its own: in reading 1 the Al capture rate per neutron
is near nominal; in reading 2 it falls, but so does everything the capsule makes.</p>"""
    D.slide('readings', body, notes)


def s_checks(D, A, N):
    # diagram: a local shift in one chamber moves A's and C's images opposite ways
    W, H = 900, 560
    o = []
    cx, cy = W / 2, H / 2 + 10
    s = 0.85
    zA, zC = cx + 234.6 * s, cx - 234.6 * s
    o.append(sd.line(zA, cy - 200, zA, cy + 200, BLUE, 10) + sd.T(zA + 30, cy - 180, 'A', 34, BLUE, 'start', 600))
    o.append(sd.line(zC, cy - 200, zC, cy + 200, ORANGE, 10) + sd.T(zC - 30, cy - 180, 'C', 34, ORANGE, 'end', 600))
    o.append(sd.arrow(zA + 30, cy + 150, zA + 30, cy + 100, BLUE, 3) + sd.T(zA + 40, cy + 175, 'u_A = +x', 20, BLUE, 'start'))
    o.append(sd.arrow(zC - 30, cy + 100, zC - 30, cy + 150, ORANGE, 3) + sd.T(zC - 40, cy + 175, 'u_C = −x', 20, ORANGE, 'end'))
    o.append(sd.line(cx, cy - 210, cx, cy + 210, MUT, 1.5, '6 5') + sd.T(cx, cy + 240, 'axis plane', 20, MUT))
    # a true source at 0, chamber A shifted by +delta in its own u
    for col, sh, lab, anc, dx in ((BLUE, -34, 'A sees +δ', 'start', 16),
                                  (ORANGE, +34, 'C sees −δ', 'end', -16)):
        o.append(f'<circle cx="{cx:.1f}" cy="{cy + sh:.1f}" r="9" fill="{col}"/>')
        o.append(sd.T(cx + dx, cy + sh + 7, lab, 21, col, anc, 600))
    o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    o.append(sd.T(cx + 16, cy + 7, 'true source', 20, INK, 'start'))
    o.append(sd.T(cx, cy + 290 - 10, 'an error δ in each chamber’s own u: the images split by 2δ', 20, MUT))
    o.append(sd.T(cx, 34, 'the same error in both chambers’ own u (strip map, tan bias, twist):', 22, INK))
    dia = sd.svg(W, H, ''.join(o), 'local error splits A and C')
    side = sd.col(
        sd.callout(f'<b>A and C agree</b> ({A["late"].loc["A", "x_cross"]:+.1f} / {A["late"].loc["C", "x_cross"]:+.1f} mm), '
                   'so the offset is not a strip-map, angle or alignment error of one chamber. The same holds for '
                   'B and D in z (−4.1 / −3.5).', BLUE, 24),
        sd.callout('<b>What survives</b> is a shift of the whole A–C pair (and B–D pair) against the beam. That is '
                   'exactly the beam-vs-frame question.', RED, 24),
        sd.callout('<b>The scintillators cannot test it:</b> the SiPM wall sits within 1–6 mm of the survey, but wall and '
                   'chambers share one structure.', GREY, 24),
        sd.callout('<b>The x sign was fixed by this same split</b> (2026-08-20): a wrong sign splits A and C by twice the '
                   '~16 mm pinwheel.', GREEN, 24), w=700, gap=18)
    body = (sd.title('A misplaced chamber cannot fake this: it would split A and C',
                     'the A and C x views point opposite ways in global x')
            + sd.row(dia, side, gap=40))
    notes = """
<p>U_HAT is +x for A and −x for C (ntof_tracking/reco/geometry.py). Any error that lives in a chamber's own
coordinates — the strip-map centre (STRIP_MAP_HALF), a constant tan offset, a yaw about y, or a sign — therefore moves
the two images in opposite global directions. Their agreement to 0.5 mm (is2_v2) bounds those to well below the 10 mm
offset. The production chain reads A ~2 mm closer to 0 than is2_v2; that is the chain systematic.</p>
<p>The chamber centres in run_config.json are the survey's (pinwheel shifts 15.5 / 15.75 / 16.35 / 17.3 mm;
MX17_Full_Geant docs/HANDOFF_ARM_GEOMETRY.md). A common shift of both A and C (A's pinwheel 10 mm smaller, C's 10 mm
larger) is invisible to every internal check.</p>"""
    D.slide('checks', body, notes)


def s_close(D, A, N):
    li = lambda t: f'<li style="margin-bottom:14px">{t}</li>'
    left = ('<ul style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('The beam and the frame ~10 mm apart (reading 1), or the capsule far off a centred beam and well below '
                 'its drawn height (reading 2).')
            + li('A source that is not the capsule Al: a holder or fitting at the nose, lit by the hall flux.')
            + li('A beam spot unlike Lambda2D at the station: distance, collimator, halo beyond 3 cm.')
            + li('Why the after-the-flux population images the nose like a beam, not the whole capsule.')
            + li('A ~3.6σ hint of a second, higher source (x-vs-height step), Geant4-library-limited.') + '</ul>')
    right = ('<ol style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:32px">'
             + li('<b>How was the beam centred, and against what?</b> Was it checked against the capsule?')
             + li('<b>Capsule tip height</b> against the chamber mid-plane, and how the capsule is held.')
             + li('<b>The beam spot at the station</b> (n_TOF profile measurement, collimator in use).')
             + li('A photo or survey of the capsule against the beam axis settles both readings at once.') + '</ol>')
    body = (sd.title('<span style="color:#eef0f3">What this does not rule out, and what would settle it</span>')
            + sd.row(sd.col(sd.p('not ruled out', 26, DMUT), left, w=820),
                     sd.col(sd.p('questions for the hardware', 26, DMUT), right, w=780), gap=56))
    D.slide('close', body, '', dark=True)


def build(out: Path) -> Path:
    A = load()
    N = numbers(A)
    D = sd.Deck('Where the capsule sits in the beam',
                'one compact source on the capsule nose, ~10 mm off the frame centre: beam off the frame, or capsule off the beam?')
    for f in (s_cover, s_setup, s_lines, s_image, s_stable, s_captures, s_pictures, s_late, s_checks, s_readings,
              s_close):
        f(D, A, N)
        print('  slide', f.__name__, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.write(out, note_meta=dict(
        title='Where the capsule sits in the beam',
        summary='The tracks image one compact source on the capsule nose ~10 mm off the station centre at every time of '
                'flight; a beam centred as aligned cannot produce it at the drawn capsule height. Two readings remain.',
        tags='X17,capsule,geometry,beam', date=dt.date.today().isoformat()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    print('wrote', build(a.out or POSE / 'deck' / 'capsule-pose.html'))


if __name__ == '__main__':
    main()
