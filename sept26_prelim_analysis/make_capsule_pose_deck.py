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
           tip=f'beam, 90 % of the late-window flux inside r = {r90:.0f} mm (EAR2 transport)')
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
                          tip='late y band crossing, +y up: A −31.3, C −34.6'),
                sd.bignum(f'≥ {100 * b5.gas_min:.0f} %', 'of the beam on the ³He, IF the beam is on the capsule', DGREEN,
                          'reading 1: the beam / frame is ~10 mm off the alignment'),
                sd.bignum(f'{100 * N["gas_off_min"]:.0f}–{100 * N["gas_off_max"]:.0f} %',
                          'IF the beam is centred as aligned', DBLUE,
                          f'reading 2: capsule far off the beam and ≥ {N["dy_min"]:.0f} mm lower than drawn'),
                gap=48))
    notes = """
<p>This note is the visual companion of <code>&lt;out&gt;/capsule_pose/report.html</code>
(<code>capsule_pose.py</code>, <code>make_capsule_pose_report.py</code>). Everything uses the post-access runs,
chambers A and C on the is2_v2 chain at the capsule-focus k, in true y (+y up). Stage 3 stored y mirrored from 7 Sep until
the 14 Oct regeneration; this analysis already mirrored it back on load, so its numbers did not change
(COORDINATES.md, Y_FLIP_AUDIT.md).</p>
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
<p><b>Beam.</b> The EAR2 beam at 19.5 m (n_TOF phase-4 phase space through the EAR2 collimation, ear2_beam/),
late window, 1/v-weighted: 50 % inside r = {A['r50']:.1f} mm, 90 % inside {A['r90']:.0f} mm, 99 % inside
{A['r99']:.0f} mm. (Until 14 Oct this slide drew Lambda2D, the resolution function, ~4× too narrow.)</p>
<p><b>The star</b> is the measured source: x from chambers A and C, z from B and D, height from the y band crossing
(scale-free; +y up).</p>"""
    D.slide('setup', body, notes, foot='geometry: run_config.json (survey), DetectorConstruction.cc polycones · '
                                       'beam: ear2_beam (EAR2 transport) × Phase-4 flux')


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
    D.slide('image', body, notes, foot='capsule_pose image.csv (A); +y = up')


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

# --------------------------------------------------------------------------- #
# the real beam and the pose refit (2026-10-14/15): capsule_pose_ear2
# --------------------------------------------------------------------------- #
POSE2 = paths.out('capsule_pose_ear2')
#: the retired beam model: Lambda2D (the n_TOF resolution function) sampled as a
#: radius, 5-200 meV (HANDOFF_CAPSULE_POSE_FIT.md section 4, as tabulated then)
OLD_BEAM_Q = ((2.5, 0.50), (4.5, 0.68), (13.0, 0.90), (27.0, 0.99))
HC = 0.5 * (CP.H_EDGES[:-1] + CP.H_EDGES[1:])
PHI_COL = {135: BLUE, 180: GOLD, 225: GREEN}
COL_D = PURPLE        # D's z crossing
COL_AC = '#0f8a8a'    # A/C band-slope depth


def _curve(s):
    return np.array([float(v) for v in str(s).split()])


def load_pose2(A):
    """The real-beam pose grid (pose2_xz: A/C x shape + D z shape), the A/C
    depth fit (pose2d_*.csv, if run), and the data curves."""
    M = pd.read_csv(POSE2 / 'pose2_xz.csv')
    fd = sorted(POSE2.glob('pose2d*.csv'))
    if fd:
        Dd = pd.concat([pd.read_csv(f) for f in fd], ignore_index=True)
        Dd = Dd.drop_duplicates(['x_b', 'x_c', 'theta', 'phi'])
        M = M.merge(Dd[['x_b', 'x_c', 'theta', 'phi', 'chi2_d', 'n_d', 'd_off', 'depth']],
                    on=['x_b', 'x_c', 'theta', 'phi'], how='left')
    A['M'] = M
    A['has_d'] = 'chi2_d' in M and M.chi2_d.notna().any()
    A['best'] = M.sort_values('chi2_xz').iloc[0]
    A['best0'] = M[M.theta == 0].sort_values('chi2').iloc[0]
    d = rd(POSE2, 'data').set_index(['arm', 'window'])
    A['d2'] = d
    S = rd(POSE2, 'data_shear')
    A['dshear'] = S[S.window == 'late']
    A['dz'] = rd(POSE2, 'data_shear_z')
    A['ddepth'] = rd(POSE2, 'data_depth') if (POSE2 / 'data_depth.csv').exists() else None
    A['y_src2'] = float(0.5 * (d.loc[('A', 'late'), 'y_cross'] + d.loc[('C', 'late'), 'y_cross']))
    A['x_src2'] = float(0.5 * (d.loc[('A', 'late'), 'x_cross'] + d.loc[('C', 'late'), 'x_cross']))
    e, pdf = CP.radial_profile(CP.WINDOWS['late'])
    A['prof_r'], A['prof_c'] = e[1:], np.cumsum(pdf)
    # families: the best pose for each beam position
    fam = []
    for xb, g in M[M.theta > 0].groupby('x_b'):
        fam.append(g.sort_values('chi2_xz').iloc[0])
    A['fam'] = sorted(fam, key=lambda r: -r.x_b)
    # tilt azimuth preferences: min chi2 over everything else at each phi
    T = M[M.theta > 0]
    # the A/C depth enters as data (s_tilt_z), not here: its toy curve is
    # Geant4-library-limited (2-3 mm per bin) and its chi2 is flat in phi
    cols = [('chi2', 'A/C x (crossings + x-vs-height shape)', INK), ('chi2_z', 'D z vs height', COL_D)]
    if A['has_d']:
        dd = M[M.theta > 0].groupby('phi').chi2_d.min()
        A['d_flat'] = float(dd.max() - dd.min())
    A['az'] = [(c, lab, col, T.groupby('phi')[c].min(), float(M[M.theta == 0][c].min())) for c, lab, col in cols]
    return A


def ac_depth(A, sl='late'):
    """A/C depth curve (mm, mean over the fit bins removed) and its error."""
    dd = A['ddepth']
    s = dd[(dd.slice == sl)].set_index('h')
    m = s.index.to_series().between(*CP.DEPTH_FIT)
    z = s.z - s.z[m].mean()
    return z, s.z_err


def pose_axis(r):
    return CP.axis_of(float(r.theta), float(r.phi))


def pose_centre(r):
    return np.array([float(r.x_c), float(r.dy_c), float(r.z_c)])


def capsule_proj(P, centre, axis, view, color=INK, w=2.5, dash=None, gas=True, tip=None):
    """The posed capsule's outline in a side view: view 'xy' (horizontal x) or
    'zy' (horizontal z), vertical y.  Each polycone ring s is drawn at
    centre + s*axis, offset by +-R(s) perpendicular to the projected axis."""
    k = 0 if view == 'xy' else 2
    a = np.array([axis[k], axis[1]])
    n = np.array([a[1], -a[0]]) / np.hypot(*a)
    c = np.array([centre[k], centre[1]])
    for S, R, col, ww in ((CP.S_AL, CP.R_CF, GREY, 1.4), (CP.S_AL, CP.R_AL, color, w)) + \
            (((CP.S_GAS, CP.R_GAS, BLUE, 1.6),) if gas else ()):
        p = c[None, :] + S[:, None] * a[None, :]
        left = p + R[:, None] * n[None, :]
        right = p - R[:, None] * n[None, :]
        q = np.r_[left, right[::-1]]
        P.fore.append(sd.poly([P.X(u) for u in q[:, 0]], [P.Y(v) for v in q[:, 1]], col, ww, dash=dash, tip=tip))


def capsule_from_top(P, centre, axis, color=INK, dash=None, tip=None):
    """Top-down (horizontal z, vertical x): rings at the nose shoulder and the
    barrel's top joined by tangents, and an arrow nose -> top."""
    def at(s):
        q = centre + s * axis
        return q[2], q[0]
    z0, x0 = at(-20.0)
    z1, x1 = at(20.0)
    circle(P, z0, x0, 10.6, color, 2.5, dash=dash, tip=tip)
    circle(P, z1, x1, 10.6, color, 1.5, dash='4 4')
    d = np.array([z1 - z0, x1 - x0])
    if np.hypot(*d) > 0.5:
        u = d / np.hypot(*d)
        nrm = np.array([-u[1], u[0]]) * 10.6
        for sgn in (1, -1):
            P.fore.append(sd.line(P.X(z0 + sgn * nrm[0]), P.Y(x0 + sgn * nrm[1]),
                                  P.X(z1 + sgn * nrm[0]), P.Y(x1 + sgn * nrm[1]), color, 1.5, '4 4'))
        zt, xt = at(51.0)
        P.raw(sd.arrow(P.X(z0), P.Y(x0), P.X(zt), P.Y(xt), RED, 3.5, 14))


def capture_cloud(r, n=60_000, keep=2500, seed=5):
    """Capture points of the pose (x, y with the fitted dy_c, z), weight-sampled."""
    pose = dict(CP.NOMINAL, x_c=float(r.x_c), z_c=float(r.z_c), theta=float(r.theta), phi=float(r.phi),
                x_b=float(r.x_b))
    C, fr = CP.beam_captures(pose, n=n, window=CP.WINDOWS['late'])
    rng = np.random.default_rng(seed)
    i = rng.choice(len(C), min(keep, len(C)), p=C.w / C.w.sum())
    c = C.iloc[i]
    return c.x.to_numpy(), c.y.to_numpy() + float(r.dy_c), c.z.to_numpy(), fr


def beam_disc(P, xb, zb, r50, r90, edge=None):
    circle(P, zb, xb, r90, BEAMC, 0, fill=BEAMC, opacity=0.16,
           tip=f'beam: 90 % of the late-window, 1/v-weighted flux inside r = {r90:.0f} mm (EAR2 transport, 19.5 m)')
    circle(P, zb, xb, r50, BEAMC, 0, fill=BEAMC, opacity=0.38, tip=f'beam: 50 % inside r = {r50:.1f} mm')


def s_cover2(D, A, N):
    b, M = A['best'], A['M']
    good = M[(M.theta > 0) & (M.chi2_xz < b.chi2_xz + 10)]
    th_lo, th_hi = good.theta.min(), good.theta.max()
    dy = M.loc[M.chi2_xz < b.chi2_xz + 10, 'dy_c']
    gas_lo, gas_hi = (min(f.gas for f in A['fam']), max(f.gas for f in A['fam']))
    body = (sd.kicker('n_TOF 2026 · X17 · capsule pose in the beam · updated 15 Oct')
            + sd.p('On the real EAR2 beam the capsule leans ~10° with its top toward B and A, and sits ~1 cm '
                   'below its drawn height. Sideways, the data cannot tell the beam from the capsule.', 50, DINK, 600,
                   'line-height:1.15;letter-spacing:-0.5px')
            + sd.row(
                sd.bignum(f'{th_lo:g}–{th_hi:g}°', 'tilt from vertical', DRED,
                          f'untilted: χ² {A["best0"].chi2:.0f} vs {b.chi2:.0f} tilted (A/C x shape)',
                          tip='poses within Δχ²(x + D z) < 10 of the best'),
                sd.bignum('φ ≈ 135°', 'top toward −x and +z (B and A)', DBLUE,
                          'z part seen two independent ways: D, and A/C alone',
                          tip='phi from +x toward +z; 135 = halfway between -x (B) and +z (A)'),
                sd.bignum(f'{-dy.max():.0f}–{-dy.min():.0f} mm', 'below the drawn height', DGREY,
                          'every good pose needs it'),
                sd.bignum(f'{100 * gas_lo:.0f}–{100 * gas_hi:.0f} %', 'of the beam on the ³He', DGREEN,
                          'beam centred on the frame → low end; beam off with the capsule → high end'),
                gap=44))
    notes = """
<p><b>What changed since the 13 Oct version.</b> The beam model was wrong: <code>Lambda2D</code>
(<code>lamda2DvsEn_EAR2.root</code>) is the n_TOF <i>resolution function</i>, not a beam radius. The fit now uses
the EAR2 beam from n_TOF's phase-4 FLUKA phase space through the EAR2 collimation (<code>ear2_beam/</code>,
validated on the PPACMon beam-interception factors at 19.95 m to 1–2 %). The beam is about four times wider, so the
"two readings" of the old version (and its ³He fractions) are void; the slides that depended on the old beam were
removed. The data slides (track lines, pointing image, stability, late population, the A/C split) do not depend
on the beam and are unchanged.</p>
<p>Fit: <code>capsule_pose pose2</code> (A/C x crossings and x-vs-height shape), <code>pose2z</code> (D's z vs
height), <code>pose2d</code> (A/C band-slope depth vs height, new 15 Oct). Report:
<code>&lt;out&gt;/capsule_pose_ear2/report.html</code>.</p>"""
    D.slide('cover', body, notes, dark=True)


def s_beamfix(D, A, N):
    P = sd.Plot(860, 640, x=(0, 30), y=(0, 1.0), xlabel='radius from the beam centre (mm)',
                ylabel='fraction of the beam inside r', title='cumulative beam profile, late window (20–75 ms)')
    P.xticks([(v, str(v)) for v in (0, 5, 10, 15, 20, 25, 30)])
    P.yticks([(v, f'{v:g}') for v in (0, 0.25, 0.5, 0.75, 1.0)])
    r, c = A['prof_r'], A['prof_c']
    P.line(list(r), list(c), INK, 4, markers=False, tip='EAR2 transport (n_TOF phase-4 phase space, EAR2 collimation), 19.5 m')
    ro = [0.0] + [q[0] for q in OLD_BEAM_Q]
    co = [0.0] + [q[1] for q in OLD_BEAM_Q]
    P.line(ro, co, RED, 3, dash='8 6', tips=[None] + [f'old model (Lambda2D as a radius): {100 * q[1]:.0f} % inside {q[0]:g} mm'
                                                      for q in OLD_BEAM_Q])
    P.vline(10.6, BLUE, '4 4', 2, tip='Al barrel outer radius 10.6 mm')
    P.text(11.0, 0.04, 'capsule barrel r = 10.6', 20, BLUE)
    P.text(5.5, 0.47, 'old: Lambda2D', 22, RED, 'start', 600)
    P.text(17, 0.6, 'real EAR2 beam', 22, INK, 'start', 600)
    # to-scale top views, old and new
    def top(title, q50, q90):
        Z = sd.Plot(300, 300, x=(-26, 26), y=(-26, 26), margin=(10, 10, 10, 10), title=title)
        circle(Z, 0, 0, q90, BEAMC, 0, fill=BEAMC, opacity=0.18, tip=f'90 % inside r = {q90:.1f} mm')
        circle(Z, 0, 0, q50, BEAMC, 0, fill=BEAMC, opacity=0.45, tip=f'50 % inside r = {q50:.1f} mm')
        circle(Z, 0, 0, 10.6, INK, 2.5, tip='capsule barrel (OD 21.2 mm), centred')
        circle(Z, 0, 0, 10.0, BLUE, 1.5)
        return Z.svg(title)
    r50 = float(np.interp(0.5, c, r))
    r90 = float(np.interp(0.9, c, r))
    side = sd.col(top('old: 50 % in 2.5 mm', 2.5, 13.0), top(f'real: 50 % in {r50:.1f} mm', r50, r90), gap=8)
    gas0 = float(np.interp(10.0, r, c))
    body = (sd.title('Our old beam was the resolution function, ~4× too narrow',
                     'n_TOF\'s Lambda2D is λ (the moderation-length resolution), not a spot size; replaced on 14 Oct')
            + sd.row(P.svg('beam profile'), side,
                     sd.col(sd.callout(f'A centred 20 mm bore now sees <b>{100 * gas0:.0f} %</b> of the beam, not ~86 %.', GOLD, 26),
                            sd.callout('The beam has a sharp edge at r ≈ 22 mm (the collimator image) and sits on the pipe axis '
                                       'to 1–2 mm.', INK, 24),
                            sd.callout('Everything fitted on the old beam (the “two readings”) is void and has been refit.', RED, 24),
                            w=360, gap=18), gap=28))
    notes = f"""
<p>The histogram <code>Lambda2D</code> in <code>MX17_Full_Geant/data/lamda2DvsEn_EAR2.root</code> has its axis
titled "lambda (cm)", is <code>transport.cc</code>'s <code>HIST_LAMBDA</code> and sits beside its SAMMY resolution
export: it is the distribution of the effective moderation length, used to build the time-of-flight resolution.
The Geant4 gun and the first pose fit sampled it as a beam radius.</p>
<p>The real beam (<code>ear2_beam/</code>): n_TOF's phase-4 FLUKA phase space transported through the EAR2
collimation. Checked against the PPACMon beam-interception factors at 19.95 m to 1–2 %. At 19.5 m: r50
{r50:.1f} mm, r90 {r90:.1f} mm (late window, 1/v-weighted), sharp edge at ~22 mm. The thermal flux is the Phase-4 one
(×1.19). Geant4 was rerun with this gun (condor 4454431/2, 19.5 and 19.95 m): ³He 0.448 per neutron (was ~0.95),
Al 0.0028 (was 0.0045).</p>"""
    D.slide('beam', body, notes, foot='ear2_beam/ (n_TOF phase-4 phase space, EAR2 collimation) · old: lamda2DvsEn_EAR2.root',
            short='beam fixed')


def s_pose(D, A, N):
    """The best pose, to scale, from the front, the side and above, with its
    capture cloud, against the capsule as drawn."""
    b = A['best']
    ctr, ax = pose_centre(b), pose_axis(b)
    cx, cy, cz, fr = capture_cloud(b)
    r50 = float(np.interp(0.5, A['prof_c'], A['prof_r']))
    r90 = float(np.interp(0.9, A['prof_c'], A['prof_r']))
    xb, zb = float(b.x_b), 0.0
    cap_tip = (f'fitted pose: axis at x {b.x_c:+.0f}, z {b.z_c:+.1f} mm, {-b.dy_c:.0f} mm below drawn; '
               f'tilt θ {b.theta:g}° toward φ {b.phi:g}°')
    nom_tip = 'the capsule as drawn (Geant4): on the frame centre, untilted'

    def side(view, title, hlab):
        P = sd.Plot(390, 700, x=(-42, 28), y=(-85, 70), xlabel=hlab, ylabel='height y (mm)',
                    margin=(24, 14, 80, 92), title=title)
        P.xticks([(v, str(v)) for v in (-40, -20, 0, 20)])
        P.yticks([(v, str(v)) for v in (-80, -40, 0, 40)])
        c0 = xb if view == 'xy' else zb
        P.band([c0 - r90, c0 + r90], [-85, -85], [70, 70], BEAMC, 0.14, tip=f'beam, 90 % (r < {r90:.0f} mm), travelling up')
        P.band([c0 - r50, c0 + r50], [-85, -85], [70, 70], BEAMC, 0.30, tip=f'beam, 50 % (r < {r50:.1f} mm)')
        capsule_proj(P, np.zeros(3), np.array([0, 1.0, 0]), view, GREY, 1.6, dash='6 5', gas=False, tip=nom_tip)
        capsule_proj(P, ctr, ax, view, INK, 2.6, tip=cap_tip)
        k = 0 if view == 'xy' else 2
        a0, a1 = ctr + (-60) * ax, ctr + 75 * ax
        P.raw(sd.line(P.X(a0[k]), P.Y(a0[1]), P.X(a1[k]), P.Y(a1[1]), RED, 2, '10 4 2 4'))
        nose = ctr + (-35) * ax
        P.raw(sd.line(P.X(nose[k]), P.Y(nose[1]), P.X(nose[k]), P.Y(70), MUT, 1.5, '2 4'))
        h = cx if view == 'xy' else cz
        P.scatter(list(h), list(cy), RED, 2.2, 0.35, tip='where beam neutrons are captured on the Al (toy, this pose)')
        P.vline(0, MUT, '3 5', 1.2, tip='frame centre')
        P.raw(sd.arrow(P.X(c0), P.Y(-80), P.X(c0), P.Y(-66), '#9a6f0f', 3, 12))
        return P
    F = side('xy', 'front: seen from C', 'x (mm)   ← B · D →')
    star(F, A['x_src2'], A['y_src2'], SRC, 13, tip=f'measured source (A/C band crossings): x {A["x_src2"]:+.1f}, '
                                                    f'height {A["y_src2"]:+.1f} mm')
    S = side('zy', 'side: seen from B', 'z (mm)   ← C · A →')
    zsrc = float(A['dz'][A['dz'].arm == 'D'].z_all.iloc[0])
    star(S, zsrc, A['y_src2'], SRC, 13, tip=f'measured source z (D, wall-coincident, ≥ 25 ms): {zsrc:+.1f} mm')
    Tp = sd.Plot(560, 600, x=(-30, 30), y=(-36, 18), xlabel='z (mm)   ← C · A →', ylabel='x (mm)   ← B · D →',
                 margin=(24, 14, 80, 92), title='from above (looking down the beam)')
    Tp.xticks([(v, str(v)) for v in (-20, -10, 0, 10, 20)])
    Tp.yticks([(v, str(v)) for v in (-30, -20, -10, 0, 10)])
    beam_disc(Tp, xb, zb, r50, r90)
    capsule_top(Tp, 0, 0, GREY, dash='6 5', tip=nom_tip)
    capsule_from_top(Tp, ctr, ax, INK, tip=cap_tip)
    Tp.scatter(list(cz), list(cx), RED, 2.0, 0.3, tip='capture points seen from above')
    cross(Tp, 0, 0, INK, 7, tip='frame centre (surveyed station centre)')
    star(Tp, zsrc, A['x_src2'], SRC, 13, tip='measured source (x from A/C, z from D)')
    Tp.text(-29, 15, 'red arrow: nose → top of the capsule', 18, RED, 'start', 600)
    legend = sd.col(
        sd.p('<b>Best pose</b>', 26, INK),
        sd.p(f'tilt <b>{b.theta:g}°</b> toward φ {b.phi:g}° (top toward B and A)', 22, INK),
        sd.p(f'<b>{-b.dy_c:.0f} mm</b> below drawn', 22, INK),
        sd.p(f'capsule x {b.x_c:+.0f}, beam x {b.x_b:+.0f} mm', 22, INK),
        sd.p(f'³He sees <b>{100 * fr["gas"]:.0f} %</b> of the beam; Al captures {100 * fr["al_nose"]:.0f} % nose', 22, GREEN),
        sd.p(f'χ² {b.chi2:.0f} (A/C x, {int(b.n_shape) + 2} pts) + {b.chi2_z:.0f} (D z, {int(b.n_z)} pts)', 20, MUT),
        sd.p('Dashed grey: as drawn. Gold: beam. Red dots: captures. Star: measured source. Red dash-dot: the '
             'fitted axis; dotted: vertical through the nose.', 20, MUT), w=300, gap=10)
    body = (sd.title('The fitted pose, to scale: top toward B and A, 1 cm low',
                     'best pose of the real-beam grid; the same capsule from the front, the side and above')
            + sd.row(F.svg('front'), S.svg('side'), Tp.svg('top'), legend, gap=18))
    notes = f"""
<p><b>Frame</b> (COORDINATES.md): +y is the beam, up; A at +z, C at −z, D at +x, B at −x. The tilt θ is from +y,
toward the azimuth φ measured from +x toward +z, so φ = 135° puts the capsule top toward −x and +z: toward B and A.
"Seen from C" looks along +z (x across); "seen from B" looks along +x (z across). Both side views and the top view
share the scale in mm.</p>
<p><b>The pose shown</b> is the minimum of χ²(A/C x crossings + A/C x-vs-height shape + D z-vs-height shape) over
the grid: beam x ∈ {{0, −5, −10}}, capsule x −10…−20, θ 0–15°, φ in 45° steps; capsule z fixed at the B/D crossing
({b.z_c:+.1f} mm); height not fitted but set by the y crossing (dy_c = {b.dy_c:+.1f} mm). The beam/capsule split
sideways is degenerate (slide "sideways"): this picture is one member of that family.</p>
<p><b>The capture cloud</b> is the toy: rays from the EAR2 beam walked through the posed STEP polycones,
Al/CFRP capture ∝ 1/v with neutron scattering, the gas black. The measured source sits at the cloud's dense end on
the nose, as it should: the pointing estimators measure where the Al captures are, not the capsule axis.</p>"""
    D.slide('pose', body, notes, short='the pose, to scale')


def s_tilt_x(D, A, N):
    b, b0 = A['best'], A['best0']
    panels = []
    for arm in 'AC':
        P = sd.Plot(800, 620, x=(-95, 35), y=(-15, -4), xlabel='track height at the axis plane (mm, +y up)',
                    ylabel='source x (mm)', title=f'chamber {arm}: x crossing per height bin (20–75 ms)')
        P.xticks([(v, str(v)) for v in (-80, -60, -40, -20, 0, 20)])
        P.yticks([(v, str(v)) for v in (-14, -12, -10, -8, -6)])
        s = A['dshear'][(A['dshear'].arm == arm) & A['dshear'].h.between(-90, 30)]
        for h, x, e in zip(s.h, s.x_cross, s.x_err):
            P.raw(sd.line(P.X(h), P.Y(x - e), P.X(h), P.Y(x + e), INK, 2))
        P.points(list(s.h), list(s.x_cross), INK, 7,
                 tips=[f'{arm}, height {h:+.0f}: x {x:+.2f} ± {e:.2f} mm (n {int(n)})'
                       for h, x, e, n in zip(s.h, s.x_cross, s.x_err, s.n)])
        m = (HC >= -60) & (HC <= 20)
        for r, col, dash, lab in ((b0, GREY, '8 6', 'untilted'), (b, RED, None, f'θ {b.theta:g}°, φ {b.phi:g}°')):
            cu = _curve(r[f'shear_{arm}'])
            P.line(list(HC[m]), list(cu[m]), col, 3.2, dash=dash, markers=False,
                   tip=f'toy, {lab}: capsule x {r.x_c:+.0f}, beam x {r.x_b:+.0f}; χ²(A+C) {r.chi2:.0f}')
        P.vline(A['y_src2'], SRC, '4 5', 1.5, tip='source height (y band crossing)')
        P.text(-92, -4.8, 'top of the capsule ⟶ further toward B', 20, MUT)
        panels.append(P.svg(f'x vs height {arm}'))
    body = (sd.title('Higher up, the source sits further toward B: the top leans to −x',
                     f'data (points) vs the toy: untilted (grey, χ² {b0.chi2:.0f}) and the best tilted pose '
                     f'(red, χ² {b.chi2:.0f}); 10 shape points + 2 crossings')
            + sd.row(*panels, gap=44)
            + sd.legend([('data, is2_v2 cf, post-access', INK, 'dot'), ('toy, untilted best', GREY, 'dash'),
                         ('toy, best tilted pose', RED)], 22))
    notes = """
<p>Each point is the robust band crossing (where the tracks' lines cross the axis plane) of the tracks whose line
reaches the axis plane at that height. A point source gives a flat curve; Geant4's untilted capsule is flat within
±1 mm. The toy curves are moved onto the data's overall crossing (x and y), so only the <i>shape</i> is compared:
χ²_shape over the populated bins (−60…+20), with the Geant4-library bootstrap spread added to the data error.</p>
<p>The data's x moves ~4 mm toward −x from the bottom of the nose to above it, the same in A and C. A and C's u
axes point opposite ways in global x, so a chamber-local error (strip map, tan bias) would split them, not move
them together. What it cannot exclude by itself is a common in-plane rotation of the A–C pair; the z slides test
the tilt with chambers that do not share that.</p>"""
    D.slide('tilt-x', body, notes, foot='capsule_pose pose2 · data_shear.csv (A/C, 20–75 ms) · toy: EAR2 beam × Geant4 nose library',
            short='tilt: x')


def s_tilt_z(D, A, N):
    dz = A['dz'][(A['dz'].arm == 'D') & A['dz'].h.between(-90, 30)]
    mz = dz[dz.h.between(*CP.DEPTH_FIT)].z_cross.mean()
    P = sd.Plot(900, 640, x=(-95, 35), y=(-6, 4), xlabel='track height at the axis plane (mm, +y up)',
                ylabel='source z, mean removed (mm)   ← C · A →',
                title='two independent measurements of z against height')
    P.xticks([(v, str(v)) for v in (-80, -60, -40, -20, 0, 20)])
    P.yticks([(v, f'{v:+g}') for v in (-6, -4, -2, 0, 2, 4)])
    P.hline(0, MUT, '3 5', 1)
    for h, z, e in zip(dz.h, dz.z_cross - mz, dz.z_err):
        P.raw(sd.line(P.X(h - 1.5), P.Y(z - e), P.X(h - 1.5), P.Y(z + e), COL_D, 2))
    P.points(list(dz.h - 1.5), list(dz.z_cross - mz), COL_D, 8,
             tips=[f'D, height {h:+.0f}: z crossing {z:+.2f} ± {e:.2f} mm (n {int(n)})'
                   for h, z, e, n in zip(dz.h, dz.z_cross, dz.z_err, dz.n)])
    have = A['ddepth'] is not None
    if have:
        z, ze = ac_depth(A)
        z = z[(z.index >= -90) & (z.index <= 30)]
        for h, v in z.items():
            P.raw(sd.line(P.X(h + 1.5), P.Y(v - ze[h]), P.X(h + 1.5), P.Y(v + ze[h]), COL_AC, 2))
        P.points(list(z.index + 1.5), list(z.values), COL_AC, 8, marker='square',
                 tips=[f'A/C band slopes, height {h:+.0f}: z {v:+.2f} ± {ze[h]:.2f} mm' for h, v in z.items()])
        rise_ac = float(z.loc[-20] - z.loc[-80])
    rise_d = float(dz.set_index('h').z_cross.loc[-20] - dz.set_index('h').z_cross.loc[-80])
    P.text(-92, 3.3, 'D: where its tracks cross the x = 0 plane', 21, COL_D, 'start', 600)
    if have:
        P.text(-92, 2.6, 'A/C: how steep A’s and C’s bands are (no D)', 21, COL_AC, 'start', 600)
    P.vline(A['y_src2'], SRC, '4 5', 1.5, tip='source height')
    # stability of the A/C curve
    side = []
    if have:
        Q = sd.Plot(640, 420, x=(-95, 35), y=(-6, 3), xlabel='height (mm)', ylabel='A/C z (mm)',
                    title='A/C depth: every slice', margin=(24, 20, 80, 92))
        Q.xticks([(v, str(v)) for v in (-80, -40, 0)])
        Q.yticks([(v, f'{v:+g}') for v in (-6, -3, 0, 3)])
        sl_cols = {'20-30': BLUE, '30-45': GOLD, '45-75': GREEN, 'wall': PURPLE}
        for sl in A['ddepth'].slice.unique():
            if sl == 'late':
                continue
            zz, _ = ac_depth(A, sl)
            zz = zz[(zz.index >= -90) & (zz.index <= 30)]
            Q.line(list(zz.index), list(zz.values), sl_cols.get(sl, GREY), 2.2, r=4,
                   tips=[f'{sl}: height {h:+.0f}, z {v:+.2f} mm' for h, v in zz.items()])
        Q.hline(0, MUT, '3 5', 1)
        side.append(Q.svg('stability'))
        side.append(sd.legend([('20–30', BLUE), ('30–45', GOLD), ('45–75 ms', GREEN), ('wall', PURPLE),
                               ('run halves', GREY)], 19))
        side.append(sd.p('ToF 20–30 / 30–45 / 45–75 ms, both run halves, wall-coincident only: '
                         'all within ~0.4 mm.', 21, MUT))
        side.append(sd.callout(f'<b>Rise from −80 to −20 mm:</b> D {rise_d:+.1f} mm, A/C {rise_ac:+.1f} mm. '
                               'The top of the capsule is toward +z (A).', BLUE, 24))
    else:
        side.append(sd.callout('A/C depth: pending (<code>capsule_pose data_depth</code>).', GREY, 24))
    body = (sd.title('D and A/C agree: the source moves toward A going up',
                     'D: z band crossing (wall-coincident, 25–75 ms) · A/C: band-slope depth, no D (20–75 ms)')
            + sd.row(P.svg('z vs height'), sd.col(*side, w=660, gap=14), gap=40))
    notes = """
<p><b>D</b> (the 14 Oct measurement): D's x view measures global z. Its band crossing per height bin, wall-coincident
tracks only, ≥ 25 ms (below that D is junk). D's systematic: an in-plane rotation of D itself would tilt its curve;
a ±3° rotation in the fit does not reorder the azimuths, but it is the one chamber.</p>
<p><b>A/C band-slope depth</b> (new, 15 Oct; <code>capsule_pose data_depth</code>): A and C look along z, so they
cannot see the source's z as a <i>position</i>, but they see it as a <i>distance</i>. A line from a source at depth
w reaches a chamber at distance D_plane with displacement disp = −(pos − x_s)·D_plane/(D_plane − w): the band of
(pos, disp) is steeper the closer the source is to the chamber. A source moving toward +z steepens A's band and
flattens C's. So ln|slope_A| − ln|slope_C| per height bin is the source's z per height bin, up to a constant.
A tan scale k common to all heights cancels (it only sets the constant), as does any height dependence that A and C
share. The response, d(ln|a_A| − ln|a_C|)/dz = 0.0074 per mm (88 % of the ideal 2/234.6; the background dilutes it),
is calibrated in each bin by moving every line of the sample by a known ±5 mm in z.</p>
<p><b>The two have different systematics.</b> D's is a rotation of D; the A/C one's is a tan scale that depends on
height <i>differently</i> in A and C (e.g. through tan y). They share no chamber, no estimator and no chain.</p>
<p><b>Caveat: the z and x shapes do not match one rigid tilt.</b> The z rise happens <i>below</i> the source height
(bins −80…−20; flat above), while the x drift toward B happens <i>above</i> it (−40…+20; flat or reversed below). A
rigid tilt moves both along the whole capsule. So the lean's direction (its sign in x and in z) is well supported,
by three measurements on two chamber pairs, but its size and shape are not: the toy matches the sign of D's rise, not
its shape below the nose. A bent holder, a second source, or a height-dependent estimator bias are not excluded.</p>"""
    D.slide('tilt-z', body, notes, short='tilt: z, two ways')


def s_azimuth(D, A, N):
    P = sd.Plot(980, 640, x=(-20, 335), y=(0, 60), xlabel='tilt azimuth φ (deg): the direction the capsule top leans',
                ylabel='Δχ² (best over θ > 0, positions)', title='which way the top leans, per measurement')
    P.xticks([(v, f'{v}') for v in range(0, 360, 45)])
    P.yticks([(v, str(v)) for v in (0, 10, 20, 30, 40, 50, 60)])
    for v, t in ((0, '+x (D)'), (90, '+z (A)'), (180, '−x (B)'), (270, '−z (C)')):
        P.text(v, 56.5, t, 20, MUT, 'middle')
    M = A['M']
    T = M[M.theta > 0]
    comb = [('chi2_xz', 'combined: A/C x + D z', RED, T.groupby('phi').chi2_xz.min(), float(M[M.theta == 0].chi2_xz.min()))]
    rows = A['az'] + comb
    for c, lb, col, s_, c0 in rows:
        y = (s_ - s_.min()).clip(upper=54)
        dash = '10 6' if c == 'chi2_xz' else None
        P.line(list(s_.index), list(y.values), col, 3.4, dash=dash, r=7,
               tips=[f'{lb}: φ {ph:g}°, χ² {v:.1f} (Δ {v - s_.min():.1f})' for ph, v in s_.items()])
    P.vline(135, RED, '4 4', 1.5, tip='φ = 135°: top toward B and A')
    # compass: each measurement a ring, each 45° sector shaded by exp(-Δχ²/8)
    W = 560
    o = []
    cx, cy = W / 2, 300
    rings = A['az']
    R0, dr = 80, 70
    for i, (c, lb, col, s_, c0) in enumerate(rings):
        r1, r2 = R0 + i * dr, R0 + (i + 1) * dr - 6
        for ph, v in s_.items():
            op = max(0.06, float(np.exp(-(v - s_.min()) / 8.0)))
            a1, a2 = np.radians(ph - 22.5), np.radians(ph + 22.5)
            # top view: z right, x up -> screen (sin phi, -cos phi)
            pts = [(cx + r2 * np.sin(t), cy - r2 * np.cos(t)) for t in np.linspace(a1, a2, 8)] + \
                  [(cx + r1 * np.sin(t), cy - r1 * np.cos(t)) for t in np.linspace(a2, a1, 8)]
            o.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{col}" '
                     f'fill-opacity="{op:.2f}" stroke="white" stroke-width="1.5"'
                     f'{sd.tipattr(f"{lb}: φ {ph:g}°, Δχ² {v - s_.min():.1f}")}/>')
    Rout = R0 + len(rings) * dr
    for ang, t, col in ((0, 'D (+x)', PURPLE), (90, 'A (+z)', BLUE), (180, 'B (−x)', GREEN), (270, 'C (−z)', ORANGE)):
        a_ = np.radians(ang)
        o.append(sd.T(cx + (Rout + 40) * np.sin(a_), cy - (Rout + 30) * np.cos(a_) + 8, t, 23, col, 'middle', 600))
    b = A['best']
    a_ = np.radians(float(b.phi))
    o.append(sd.arrow(cx, cy, cx + (Rout - 10) * np.sin(a_), cy - (Rout - 10) * np.cos(a_), RED, 4, 18))
    o.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{INK}"/>')
    o.append(sd.T(cx, 24, 'seen from above; darker = preferred', 21, INK))
    o.append(sd.T(cx, 612, 'inner ring: A/C x · outer ring: D z · arrow: best combined', 20, MUT))
    comp = sd.svg(W, 625, ''.join(o), 'compass')
    zac = ''
    if A['ddepth'] is not None:
        zz, _ = ac_depth(A)
        zac = (f'A/C band-slope depth (data, no D): source +{zz.loc[-20] - zz.loc[-80]:.1f} mm toward A from −80 '
               'to −20 mm height: the +z half again.')
    leg = sd.legend([(lb, col) for c, lb, col, *_ in rows], 19)
    body = (sd.title('Each view fixes one half: the top leans toward B and A',
                     'Δχ² against the tilt azimuth φ (minimum over θ and the positions); dashed red: the two combined')
            + sd.row(P.svg('azimuth'), sd.col(comp, sd.p(zac, 21, COL_AC, 600), w=W, gap=4), gap=40) + leg)
    notes = f"""
<p>For each measurement and each φ, the minimum χ² over the tilt θ (2–15°) and the beam/capsule positions in the
grid. The x measurement (A/C crossings and x-vs-height shape) only knows the x component of the lean: it wants the
top toward −x (φ 135–225) and cannot tell +z from −z. D's z vs height only knows the z component: it wants +z (φ
45–135) and cannot tell +x from −x. Only φ ≈ 135° satisfies both, which is what the combined curve shows.</p>
<p>The A/C band-slope depth is not in this χ²: its toy prediction (<code>capsule_pose pose2d</code>) is limited by the
Geant4 library (≈10 k tracks per arm) to 2–3 mm per height bin, and its best χ² over φ varies by only
{A.get('d_flat', float('nan')):.1f}. Its direct reading of the data (the previous slide) is what it contributes: the
same +z half as D, from different chambers. Both z measurements will tighten with the EAR2-gun Geant4 library.</p>"""
    D.slide('azimuth', body, notes, short='tilt azimuth')


def s_sideways(D, A, N):
    r50 = float(np.interp(0.5, A['prof_c'], A['prof_r']))
    r90 = float(np.interp(0.9, A['prof_c'], A['prof_r']))
    panels = []
    for r in A['fam']:
        ctr, ax = pose_centre(r), pose_axis(r)
        cx, cy, cz, fr = capture_cloud(r, n=40_000, keep=1500)
        t = f'beam x {r.x_b:+.0f} · capsule x {r.x_c:+.0f}'
        Z = sd.Plot(530, 600, x=(-34, 26), y=(-38, 22), xlabel='z (mm)', ylabel='x (mm)',
                    margin=(24, 14, 80, 92), title=t)
        Z.xticks([(v, str(v)) for v in (-30, -20, -10, 0, 10, 20)])
        Z.yticks([(v, str(v)) for v in (-30, -20, -10, 0, 10, 20)])
        beam_disc(Z, float(r.x_b), 0.0, r50, r90)
        capsule_from_top(Z, ctr, ax, INK, tip=f'θ {r.theta:g}° φ {r.phi:g}°, {-r.dy_c:.0f} mm below drawn')
        Z.scatter(list(cz), list(cx), RED, 2.0, 0.3, tip='captures')
        cross(Z, 0, 0, INK, 7, tip='frame centre')
        star(Z, float(A['dz'][A['dz'].arm == 'D'].z_all.iloc[0]), A['x_src2'], SRC, 12, tip='measured source')
        Z.text(-32, 18.5, f'³He sees {100 * fr["gas"]:.0f} % of the beam', 22, GREEN if fr['gas'] > 0.4 else RED, 'start', 600)
        Z.text(-32, 14.5, f'χ² {r.chi2_xz:.0f}  (x {r.chi2:.0f} + D z {r.chi2_z:.0f})', 20, MUT)
        panels.append(Z.svg(t))
    gl, gh = min(f.gas for f in A['fam']), max(f.gas for f in A['fam'])
    body = (sd.title(f'Sideways it is degenerate: ³He sees {100 * gl:.0f}–{100 * gh:.0f} % of the beam',
                     'best tilted pose for each beam position, from above, to scale; every one puts the captures on the '
                     'measured source')
            + sd.row(*panels, gap=24))
    notes = """
<p>The x crossing measures where the Al captures are, which is where the beam meets the capsule: a beam on the
frame with the capsule 16–20 mm off, and a beam 10 mm off with the capsule on it, put the captures at the same spot.
The x-vs-height shape mildly prefers the beam off the frame (Δχ² ≈ 10–15), but that shape error is
toy-library-limited, and n_TOF puts the beam on the pipe axis to 1–2 mm. If the frame is centred on the pipe, the
first panel is the right one and the ³He sees ~0.2 of the beam at 19.5 m.</p>
<p>What would decide it: how the beam centring was done and against what; a photo or survey of the capsule against
the beam axis; the beam spot at the station. The 19.95 m beam gives the same picture (beam-centred family χ² 35,
³He 0.23; beam −5/−10: 29, 0.34–0.41).</p>"""
    D.slide('sideways', body, notes, short='sideways: degenerate')


def s_close2(D, A, N):
    li = lambda t: f'<li style="margin-bottom:14px">{t}</li>'
    left = ('<ul style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:28px">'
            + li('Where the beam is relative to the frame: the sideways split is degenerate, and it sets the ³He yield ×3.')
            + li('Faking the lean needs a common rotation of the A–C pair (x) <i>and</i>, for z, both a twist of D and an '
                 'A/C tan-scale asymmetry with height, of the right sizes.')
            + li('The tilt\'s size and shape: z moves below the source height, x above it; one rigid tilt fits neither shape '
                 'well (a bent holder or a second source would).')
            + li('The toy\'s shapes are Geant4-library-limited (≈10 k tracks per arm, July Lambda2D gun).')
            + li('Why the after-the-flux (20–75 ms) population images the nose like the beam does.') + '</ul>')
    right = ('<ol style="font-size:27px;line-height:1.35;color:#eef0f3;padding-left:32px">'
             + li('<b>Photo / survey of the capsule</b> against the beam axis: settles the sideways split and checks the lean.')
             + li('<b>Capsule tip height</b> against the chamber mid-plane, and its holder (we need ~1 cm lower).')
             + li('<b>EAR2-gun Geant4 library</b> (condor done; steps + digitisation on lxplus next) to tighten every shape χ².')
             + li('How the beam centring was done, and the collimator in use.') + '</ol>')
    body = (sd.title('<span style="color:#eef0f3">What this does not rule out, and what would settle it</span>')
            + sd.row(sd.col(sd.p('not ruled out', 26, DMUT), left, w=820),
                     sd.col(sd.p('next', 26, DMUT), right, w=780), gap=56))
    D.slide('close', body, '', dark=True)


def build(out: Path) -> Path:
    A = load_pose2(load())
    N = numbers(A)
    D = sd.Deck('Where the capsule sits in the beam',
                'the capsule leans ~10 deg toward B and A and sits ~1 cm low; sideways, beam and capsule are degenerate')
    # s_captures, s_pictures and s_readings drew the retired Lambda2D beam (13 Oct); kept for the record, not built
    for f in (s_cover2, s_setup, s_beamfix, s_lines, s_image, s_stable, s_pose, s_tilt_x, s_tilt_z, s_azimuth,
              s_sideways, s_late, s_checks, s_close2):
        f(D, A, N)
        print('  slide', f.__name__, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.write(out, note_meta=dict(
        title='Where the capsule sits in the beam',
        summary='On the real EAR2 beam the capsule leans ~10 deg with its top toward B and A (the z lean seen by D and, '
                'independently, by A/C band slopes) and sits ~1 cm below drawn; sideways, beam and capsule are degenerate.',
        tags='X17,capsule,geometry,beam', date=dt.date.today().isoformat()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    print('wrote', build(a.out or POSE / 'deck' / 'capsule-pose.html'))


if __name__ == '__main__':
    main()
