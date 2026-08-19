#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_share.py -- resistive charge sharing, and what a predicted waveform is
made of.  The figure set behind the two reconstruction slides.

    ../.venv/bin/python make_share.py [--only cartoon|kernels|build|decompose]
                                      [--no-slide]

Four figures (light theme only -- the deck defines one palette):

  share_cartoon     THE MECHANISM, as a drawing.  The avalanche lands on the
                    screen-printed resistive film; the film is a distributed RC
                    line, so the charge that goes sideways arrives at the
                    neighbours LATE and dispersed.  That is the whole reason the
                    reconstruction cannot use a per-strip threshold time.
  share_kernels     THE KERNELS PRODUCTION USES, per plane, straight out of the
                    frozen det3 calibration bundle: the response to charge on
                    the strip itself, on the +-1 neighbour, on the +-2.
  share_build       WHAT THE MODEL DOES, as a diagram: the drift column in
                    60 ns slices -> strips by geometry and diffusion ->
                    neighbours by the kernel -> folded with the measured
                    impulse response = one predicted (strip x sample) window.
  share_decompose   THE SAME THING ON REAL DATA: four strips of one real muon,
                    each strip's fitted waveform split into own / +-1 / +-2
                    charge, drawn against the measurement.

COLOUR RULE: own charge, +-1 and +-2 keep the same three colours in all four
figures and on both slides.  A colour means one thing across the section --
that is the whole point of drawing the set together (asked for 2026-08-17:
"the colors for different strip contributions are different").

PROVENANCE.  Nothing here is invented:

  kernel amplitudes     c1, c2, kY, tau_s, sigma_s of the FROZEN production
  and shapes            bundle calib_bundle_lp2_t0p (det3, Saturday long run,
                        resistive 490 V / drift 1000 V), and that bundle's own
                        measured per-plane impulse response.  share_mode is
                        'delay' -- see the caveat below.
  the worked event      event 1663 of the ref-pinned calibration cache, the
                        same event the document's model-vs-data figure and the
                        deck's "One muon through the forward fit" slide use.
                        Fitted here the way production fits: (p0, w, t0) free
                        with the bundle's absolute-t0 prior, charges by NNLS.
  the geometry drawn     0.78 mm readout pitch, 0.80 mm / 550 um resist pitch,
  in the cartoon        both as built (MX17_Geant gerbers).

THE AMPLITUDES ARE NOW QUOTABLE (2026-08-18).  They were not until this week,
and the caveat block that used to sit here said so at length.  What changed:

  * the H4 beam test, at NORMAL incidence, breaks the degeneracy a cosmic-angle
    fit cannot ("sharing" vs "a wider initial cloud plus a different v_drift",
    WAVEFORM_FIRST_THREADING.md 17.2).  Measured there, model-free, by the
    cross-relation: c2/c1 = 0.45 +- 0.02 over a 2.6x range of drift field
    (sps_beam_test_26/analysis/sharing_kernel).
  * det3 was refit with that ratio pinned (c2_over_c1 = 0.6, the bench-cosmic
    value, 0.63 +- 0.09 -- the three ratios 0.45/0.6/0.8 are indistinguishable
    in resolution, what matters is that it is BELOW 1).  Held-out angle
    resolution moves by less than 0.6 sigma; one fitted hyper becomes a
    measured constraint.
  * so the ORDERING is now structural, not fitted: c2 < c1 always, because the
    +-2 strip is reached only through the +-1 strip.  main() refuses to draw a
    kernel that violates it.

Still true, and still worth saying if asked: the ABSOLUTE level of c1 on a
cosmic fit is a lower bound (the H4 beam sees c1 ~ 0.28-0.30 head-on), and the
X/Y ratio kY is the number this figure defends best.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
           os.path.join(REPO, 'cosmic_bench_analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import plotstyle as PS                                   # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

# --------------------------------------------------------------------------- #
# the frozen production calibration, and the worked event
# --------------------------------------------------------------------------- #
ANALYSIS = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
            'mx17_det3_saturday_scan_6-27-26/'
            'long_run_resist_490V_drift_1000V/mx17_3')
# 2026-08-18: repointed from calib_bundle_lp2_t0p to the CORRECTED kernel.
# The frozen production bundle carries c2/c1 = 1.14 -- a +-2 copy larger than
# the +-1 copy, which cannot happen on a resistive film, since the +-2 strip is
# reached only through the +-1 strip.  calib_bundle_r06 is the same fit with
# the ratio pinned at 0.6 (measured head-on at H4: 0.45 +- 0.02; near-vertical
# bench cosmics on this detector: 0.63 +- 0.09).  See
# sps_beam_test_26/analysis/sharing_kernel and mpgd26/walkthrough.
BUNDLE = os.path.join(ANALYSIS, 'wft', 'calib_bundle_r06')
CALIB_CACHE = os.path.join(ANALYSIS, 'wft', 'calib_work', 'calib_cache.pkl')
WORKED_EID = 1663          # f_model.WORKED_EID -- clean on both planes

# the geometry the cartoon draws, as built (MX17_Geant)
PITCH_MM = 0.780
RESIST_PITCH_MM = 0.800
RESIST_WIDTH_MM = 0.550

# --------------------------------------------------------------------------- #
# THE colour rule
# --------------------------------------------------------------------------- #
# Okabe-Ito blue / vermillion plus the deck accent: three hues that stay
# distinct for deutan and protan viewers, and never used for anything else in
# this section.  Every one of them always carries a text label as well, so
# identity never rests on hue alone (plotstyle's palette rule).
OWN = '#0072B2'
N1 = '#D55E00'
N2 = '#8a3f8f'
FILM = '#1c1c1c'           # the ESL resist, same black as the board figures
COPPER = '#e09a55'         # the L4 pads / strips, same copper as the board


def save_both(fig, path):
    """Write the PNG the deck uses and the vector PDF the report links.

    plotstyle.save closes the figure, so the PDF has to go first.
    """
    fig.savefig(path[:-4] + '.pdf')
    PS.save(fig, path)


def _bundle():
    from wft.calib import CalibrationBundle
    from wft import model as wm
    cal = CalibrationBundle.load(BUNDLE)
    wm.use_calibration(cal)
    return cal, wm


def _amps(h, plane):
    """(c1, c2) as ``wft.model.build_matrix`` computes them for one plane.

    MUST mirror build_matrix: when the bundle carries ``c2_over_c1`` the stored
    ``c2`` is dead (it is 0.0 on those bundles) and c2 is slaved to c1 BEFORE
    the per-plane kY/cX scaling.  Reading h['c2'] directly, as this file used
    to, silently draws a kernel with no +-2 copy at all.
    """
    k = h.get('kY', 1.0) if plane == 'y' else h.get('cX', 1.0)
    c1 = h['c1'] * k
    r = h.get('c2_over_c1')
    c2 = float(r) * c1 if r is not None else h['c2'] * k
    return c1, c2


def _kernels(cal, wm, plane, t):
    """(own, +-1, +-2) responses on the time grid ``t``, at the amplitudes the
    model uses for ``plane`` -- i.e. c1/c2 already multiplied by kY on Y.

    This calls the model's own ``_copy_responses``, not a re-implementation of
    it: the figure has to move if the kernel form ever does.
    """
    h = dict(cal.hyper)
    H1, H2 = wm._copy_responses(plane, t, h)
    H0 = np.interp(t, np.asarray(cal.grid, float),
                   np.asarray(cal.tmpl[plane], float), left=0, right=0)
    c1, c2 = _amps(h, plane)
    return H0, c1 * H1, c2 * H2, c1, c2


def _trim(P, keep_frac=0.02, pad=3):
    """The strips that carry charge, plus a pad -- what production's window
    extraction hands the fit.  Same rule as wftdoc.trim_window."""
    W = np.asarray(P['W'], float)
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    amp = W.max(axis=1)
    live = np.where((amp > 5 * noise) & (amp > keep_frac * amp.max()))[0]
    if len(live) == 0:
        live = np.array([int(np.argmax(amp))])
    sl = slice(max(0, live.min() - pad),
               min(W.shape[0] - 1, live.max() + pad) + 1)
    return dict(W=W[sl], pos=np.asarray(P['pos'], float)[sl],
                noise=np.asarray(P['noise'], float)[sl],
                ch=np.asarray(P['ch'])[sl])


def worked_plane(cal, wm, plane='y', eid=WORKED_EID):
    """Fit one plane of the worked event and split the model three ways.

    Returns a dict with the data window, the fitted geometry, and the model
    decomposed into own-charge / +-1 / +-2 contributions.  The split is made by
    REBUILDING the design matrix with c1 and c2 zeroed, which is exact: the
    model is a sum of those three terms, so differences of the three builds are
    the terms themselves.
    """
    with open(CALIB_CACHE, 'rb') as f:
        evs = pickle.load(f)
    ev = evs[eid]
    P = _trim(ev[plane])
    wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, plane)

    h = dict(cal.hyper)
    p0_ref = ev[f'ref_mesh_{plane}']
    w_ref = ev[f'tan_{plane}'] * cal.v_drift * 1e-3
    t0_pred = cal.t0_abs[plane][ev[f'ftst_{plane}']]
    r = wm.fit_plane_raw(P, plane, p0_ref, w_ref, t0_pred, hyper=h,
                         t0_prior=(t0_pred, cal.t0_prior_sigma))
    q, t0, p0, w = r['q'], r['t0'], r['p0'], r['w']

    def build(hh):
        M = wm.build_matrix(plane, pos, p0, w, t0, hh)
        return (M @ q).reshape(len(pos), wm.NSAMP)

    # the three builds differ ONLY in which copies are switched on, so their
    # differences are the terms themselves.  c2_over_c1 has to be dropped, not
    # just zeroed, or build_matrix re-derives c2 from c1.
    h_no2 = dict(h, c2=0.0)
    h_no2.pop('c2_over_c1', None)
    h_none = dict(h_no2, c1=0.0)
    own = build(h_none)
    with1 = build(h_no2)
    full = build(h)
    return dict(W=W, pos=pos, noise=noise, t=np.arange(wm.NSAMP) * wm.SNS,
                own=own, sh1=with1 - own, sh2=full - with1, full=full,
                q=q, t0=t0, p0=p0, w=w, tan=w * 1e3 / cal.v_drift,
                chi2_dof=r['chi2'] / max(r['dof'], 1), plane=plane, eid=eid,
                tan_ref=ev[f'tan_{plane}'])


# --------------------------------------------------------------------------- #
# 1.  the mechanism, as a drawing
# --------------------------------------------------------------------------- #
def _zigzag(ax, x0, x1, y, amp=1.6, n=4, **kw):
    """A resistor symbol between two nodes."""
    xs = np.linspace(x0, x1, 2 * n + 3)
    ys = np.full_like(xs, float(y))
    ys[2:-2:2] = y + amp
    ys[3:-2:2] = y - amp
    ax.plot(xs, ys, solid_joinstyle='miter', **kw)


def fig_cartoon(out, c1x, c2x, c1y, c2y, tau):
    """The avalanche, the RC film, and the delayed copies the neighbours see.

    The flow is drawn the way it happens, and that order is the point: DOWN the
    avalanche onto the film, SIDEWAYS along the film through its own sheet
    resistance, and only then DOWN onto the strips.  The first version of this
    drawing had the sideways arrows going straight from the impact point to the
    neighbouring strips, which crosses the pick-up path and quietly says the
    charge is split at the moment it lands.  It is not: the sideways trip
    through the resistance is what costs the time.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(-19, 100)
    ax.set_ylim(4, 100)
    ax.axis('off')

    xs = np.array([14.0, 32.0, 50.0, 68.0, 86.0])       # strip centres, j-2..j+2
    names = ['j−2', 'j−1', 'j', 'j+1', 'j+2']
    sw = 14.0                                            # drawn strip width
    Z_FILM, Z_SPREAD, Z_STRIP = 71.0, 62.0, 40.0

    def margin(y, text):
        ax.text(-18, y, text, color=PS.MUTED, fontsize=11.5, ha='left',
                va='center', linespacing=1.35)

    # --- the micromesh, and the avalanche arriving on the film ---------------
    ax.plot([6, 94], [92, 92], color=PS.MUTED, lw=1.4, ls=(0, (5, 4)))
    margin(92, 'micromesh')
    ax.add_patch(FancyArrowPatch((50, 90.5), (50, Z_FILM + 8.4),
                                 arrowstyle='-|>', mutation_scale=22,
                                 lw=3.2, color=PS.TRACK, zorder=5))
    ax.text(53.0, 85.0, 'avalanche', color=PS.TRACK, fontsize=12.5,
            fontweight='bold', va='center')

    # --- the resistive film: one distributed RC line ------------------------
    ax.add_patch(FancyBboxPatch((6, Z_FILM), 88, 7.6,
                                boxstyle='round,pad=0.0,rounding_size=1.6',
                                facecolor=FILM, edgecolor='none', zorder=3))
    margin(Z_FILM + 3.8, 'resistive\nlayer')
    # kept SHORT and hard left: the avalanche arrow comes down at x = 50 and a
    # longer line runs straight into it
    ax.text(6, Z_FILM + 9.6,
            f'{RESIST_WIDTH_MM * 1000:.0f} µm / {RESIST_PITCH_MM:.2f} mm',
            color=PS.MUTED, fontsize=11, va='bottom')
    # the sheet resistance ALONG the film, drawn as what it is
    for a, b in zip(xs[:-1], xs[1:]):
        _zigzag(ax, a + 3.2, b - 3.2, Z_FILM + 3.8, amp=2.0, n=4,
                color='#f0d9b8', lw=2.0, zorder=4)
    for x in xs:
        ax.plot([x], [Z_FILM + 3.8], marker='o', ms=5.5, color='#f0d9b8',
                zorder=5)

    # --- 1. sideways, through the resistance --------------------------------
    # arrow WIDTH is the copy amplitude, so the drawing and the kernel figure
    # beside it are saying the same thing in two encodings.
    for dx, col, amp in ((1, N1, c1y), (2, N2, c2y)):
        for s in (-1, +1):
            ax.add_patch(FancyArrowPatch(
                (50 + s * 3.0, Z_SPREAD), (50 + s * dx * 18.0 - s * 2.0,
                                           Z_SPREAD),
                arrowstyle='-|>', mutation_scale=17 + 22 * amp,
                lw=1.6 + 16 * amp, color=col, zorder=6,
                connectionstyle='arc3,rad=%.2f' % (-s * 0.30)))
    # --- 2. down onto the strips -------------------------------------------
    for x, col, amp in zip(xs, (N2, N1, OWN, N1, N2),
                           (c2y, c1y, 1.0, c1y, c2y)):
        ax.add_patch(FancyArrowPatch((x, Z_SPREAD - 1.0), (x, Z_STRIP + 9.8),
                                     arrowstyle='-|>',
                                     mutation_scale=15 + 14 * amp,
                                     lw=1.4 + 5.2 * amp, color=col, zorder=6))

    # --- the readout strips underneath --------------------------------------
    for x, nm in zip(xs, names):
        ax.add_patch(Rectangle((x - sw / 2, Z_STRIP), sw, 9.0,
                               facecolor=COPPER, edgecolor='#b87333', lw=1.0,
                               zorder=3))
        ax.text(x, Z_STRIP - 2.4, nm, ha='center', va='top', fontsize=13,
                color=PS.INK)
    margin(Z_STRIP + 4.5, f'readout\nstrips\n{PITCH_MM:.2f} mm')

    # What each arrow means, as three lines under the left margin column.  They
    # were first written under the strip each one lands on, which reads better
    # but does not fit: at this type size a label is ~20 drawn units wide and
    # the strips are 18 apart, so any two of them touch.
    for k, (col, lab) in enumerate(
            ((OWN, 'own charge'),
             (N1, f'±1 neighbour — late by {tau:.0f} ns'),
             (N2, f'±2 — {2 * tau:.0f} ns'))):
        ax.text(-18, 24.0 - 7.4 * k, lab, color=col, fontsize=12,
                fontweight='bold', ha='left', va='center')

    # NO closing sentence: it used to say "the sideways trip is through
    # resistance, so the copy arrives LATE and dispersed".  That is the slide's
    # fig-label now -- set in the deck's own type, at the deck's own size.

    save_both(fig, out)


# --------------------------------------------------------------------------- #
# 2.  the kernels production uses
# --------------------------------------------------------------------------- #
def fig_kernels(out, cal, wm):
    """Own charge, +-1 and +-2 -- one panel per plane, at the model\'s own
    amplitudes, normalised to the own-charge peak because the number that
    matters here is the RATIO.

    Two layout decisions worth keeping:

    * the copies are drawn at their TRUE relative amplitude, not magnified.  On
      X that means a bump at 5 % of the strip\'s own pulse, and that is the
      honest picture -- the X strips are pitched across the film\'s strips.
    * the amplitudes go in a text block, the delays go on the curve.  Labelling
      the copies where they peak (the first attempt) collides: the +-1 and +-2
      peaks are 145 ns apart and the labels are wider than that.
    """
    # the template stops at +1.4 us; asking for more gets a step down to zero
    # at the grid edge, which reads as a feature of the response and is not one
    t = np.linspace(-150.0, 1340.0, 1400)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 6.4), sharex=True)

    h = dict(cal.hyper)
    for ax, plane, sub in zip(
            axs, ('x', 'y'),
            ('pitched ACROSS the layer’s strips — the charge barely crosses them',
             'pitched ALONG them, where the charge really spreads  '
             '(k$_Y$ = %.1f)' % h['kY'])):
        H0, K1, K2, c1, c2 = _kernels(cal, wm, plane, t)
        n = H0.max()
        H0, K1, K2 = H0 / n, K1 / n, K2 / n
        for v, col, lw in ((H0, OWN, 2.6), (K1, N1, 2.4), (K2, N2, 2.2)):
            ax.fill_between(t, 0, v, color=col, alpha=0.22, lw=0)
            ax.plot(t, v, color=col, lw=lw, zorder=4)
        ax.axhline(0, color=PS.LINE, lw=0.9)

        t0p, t1p, t2p = (t[int(np.argmax(v))] for v in (H0, K1, K2))

        # all three labels in the two top corners, where no curve goes: the
        # own-charge label used to sit on its own peak and ran straight into the
        # amplitude block
        ax.text(0.015, 0.99, 'charge on THIS strip', transform=ax.transAxes,
                ha='left', va='top', color=OWN, fontsize=12,
                fontweight='bold')
        ax.text(0.985, 0.93, f'±1 neighbour — {100 * c1:.0f} % of it',
                transform=ax.transAxes, ha='right', va='top', color=N1,
                fontsize=12, fontweight='bold')
        ax.text(0.985, 0.79, f'±2 — {100 * c2:.0f} %',
                transform=ax.transAxes, ha='right', va='top', color=N2,
                fontsize=12, fontweight='bold')

        # the delays, marked where they are: peak to peak
        for tp, y, col in ((t1p, 0.48, N1), (t2p, 0.31, N2)):
            ax.annotate('', xy=(tp, y), xytext=(t0p, y),
                        arrowprops=dict(arrowstyle='<|-|>', color=col,
                                        lw=1.2, shrinkA=0, shrinkB=0,
                                        alpha=0.85))
            ax.text(0.5 * (t0p + tp), y + 0.022, f'+{tp - t0p:.0f} ns',
                    ha='center', va='bottom', fontsize=11.5, color=col)

        ax.set_ylim(-0.06, 1.16)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel('response   (own peak = 1)')
        PS.strip(ax)
        PS.title(ax, f'{plane.upper()} plane', sub)
    axs[-1].set_xlabel('time after the charge reaches the layer  [ns]')
    fig.subplots_adjust(hspace=0.40)
    save_both(fig, out)


# --------------------------------------------------------------------------- #
# 3.  what the model does, as a diagram
# --------------------------------------------------------------------------- #
def fig_build(out, c1, c2, tau, c1_other, c2_other):
    """Four stages, top to bottom, each with its own little sketch.

    Deliberately a DRAWING and not a plot: its job is to say what happens, and
    the figure beside it on the slide does the same thing on real data.
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    rows = [86.0, 63.0, 40.0, 15.0]        # centre of each stage
    sk_x0, sk_w = 4.0, 30.0                # the sketch column
    tx = 39.0                              # the text column

    def stage(y, n, head, body):
        ax.text(tx, y + 5.6, f'{n}  {head}', fontsize=13, color=PS.INK,
                fontweight='bold', va='top')
        ax.text(tx, y - 0.6, body, fontsize=11.5, color=PS.MUTED, va='top',
                linespacing=1.5)
        if n != '④':
            ax.add_patch(FancyArrowPatch((sk_x0 + sk_w / 2, y - 8.5),
                                         (sk_x0 + sk_w / 2, y - 13.5),
                                         arrowstyle='-|>', mutation_scale=18,
                                         lw=1.8, color=PS.LINE))

    # -- (1) the drift column in 60 ns slices -------------------------------
    y = rows[0]
    ax.add_patch(Rectangle((sk_x0, y - 8), sk_w, 16, facecolor='#eef4fa',
                           edgecolor=PS.LINE, lw=1.0))
    ax.plot([sk_x0 + 4, sk_x0 + sk_w - 4], [y - 7.4, y + 7.4],
            color=PS.TRACK, lw=2.6, zorder=4)
    for f in np.linspace(0.08, 0.92, 9):
        xx = sk_x0 + 4 + f * (sk_w - 8)
        yy = y - 7.4 + f * 14.8
        ax.plot([sk_x0, sk_x0 + sk_w], [yy, yy], color=PS.LINE, lw=0.6,
                zorder=2)
        ax.plot([xx], [yy], marker='o', ms=4.2 + 3.0 * np.sin(3 * f),
                color=PS.ACCENT, zorder=5)
    stage(y, '①', 'the track, in 60 ns slices',
          'K = 18 slices of the drift column.  Each holds a\n'
          'charge q$_k$ ≥ 0 at transverse position p₀ + w·u$_k$.\n'
          'The charges are solved, not searched.')

    # -- (2) geometry: which strips a slice lands on -------------------------
    y = rows[1]
    ctr = sk_x0 + sk_w / 2
    bars = np.array([0.04, 0.13, 0.30, 1.00, 0.36, 0.11, 0.03])
    bx = ctr + (np.arange(7) - 3) * 3.9
    for xx, bb in zip(bx, bars):
        ax.add_patch(Rectangle((xx - 1.6, y - 7.5), 3.2, 13.0 * bb,
                               facecolor=OWN, alpha=0.30 + 0.55 * bb,
                               edgecolor='none'))
    ax.plot([sk_x0, sk_x0 + sk_w], [y - 7.5] * 2, color=PS.LINE, lw=1.0)
    stage(y, '②', 'geometry → strips',
          'The slice’s cloud, integrated over the 0.78 mm\n'
          'pitch: F$_{ik}$.  Its width is the initial cloud,\n'
          'diffusion, and the slice’s own sideways travel.')

    # -- (3) the kernel: neighbours get delayed copies -----------------------
    y = rows[2]
    ax.add_patch(Rectangle((ctr - 1.9, y - 7.5), 3.8, 13.0, facecolor=OWN,
                           edgecolor='none'))
    # the neighbour bars are at their TRUE fraction of the central one.  They
    # were drawn 3.2x bigger at first, which made the picture say something the
    # kernel figure beside it contradicts.
    for dx, col, amp in ((1, N1, c1), (2, N2, c2)):
        for s in (-1, +1):
            xx = ctr + s * dx * 4.6
            ax.add_patch(Rectangle((xx - 1.9, y - 7.5), 3.8,
                                   13.0 * amp, facecolor=col,
                                   edgecolor='none'))
            ax.add_patch(FancyArrowPatch((ctr + s * 1.9, y + 2.0),
                                         (xx - s * 1.9, y + 2.0),
                                         arrowstyle='-|>', mutation_scale=13,
                                         lw=1.4, color=col,
                                         connectionstyle='arc3,rad=%.2f'
                                                         % (-s * 0.45)))
    ax.plot([sk_x0, sk_x0 + sk_w], [y - 7.5] * 2, color=PS.LINE, lw=1.0)
    stage(y, '③', 'the kernel → neighbours',
          f'Every strip’s charge appears again on ±1 and\n'
          f'±2 — and LATE, by {tau:.0f} and {2 * tau:.0f} ns.\n'
          f'Per plane: Y {c1:.2f} / {c2:.2f},  X {c1_other:.2f} / '
          f'{c2_other:.2f}.')

    # -- (4) fold with the measured response --------------------------------
    y = rows[3]
    tt = np.linspace(0, 1, 200)
    pulse = (tt / 0.28) ** 2 * np.exp(-((tt - 0.28) / 0.34)) * np.exp(-tt * 1.6)
    pulse = pulse / pulse.max()
    px = sk_x0 + tt * sk_w
    ax.fill_between(px, y - 7.5, y - 7.5 + 13.0 * pulse, color=OWN,
                    alpha=0.22, lw=0)
    ax.plot(px, y - 7.5 + 13.0 * pulse, color=OWN, lw=2.2)
    ax.plot([sk_x0, sk_x0 + sk_w], [y - 7.5] * 2, color=PS.LINE, lw=1.0)
    stage(y, '④', 'fold with h(t)',
          'The measured single-electron response of this\n'
          'plane, sampled at 32 × 60 ns — one complete\n'
          'predicted (strip × sample) window.')

    save_both(fig, out)


# --------------------------------------------------------------------------- #
# 4.  the same split, on real data
# --------------------------------------------------------------------------- #
def fig_decompose(out, st, picks=None):
    """Four strips of one real muon: own / +-1 / +-2, against the data.

    WHICH STRIPS, and why it is not cherry-picking: the peak strip, and the
    three CONSECUTIVE strips next to it, on whichever side the model tracks
    better.  Nothing is skipped inside the run of four, so the panels cannot be
    a selection of flattering strips -- and walking outwards from the core is
    exactly the direction along which the neighbours' share grows, which is the
    point being made.

    The shared fraction quoted is by AREA, not by peak height: the shared
    copies peak later than the strip's own charge, so a peak-height ratio would
    understate them.  Area is what "how much of this pulse is not this strip's
    charge" actually means.
    """
    W, pos, t = st['W'], st['pos'], st['t']
    own, sh1, sh2, full = st['own'], st['sh1'], st['sh2'], st['full']
    area = full.sum(axis=1)
    shared = np.where(area > 0, (sh1 + sh2).sum(axis=1) / np.maximum(area, 1e-9),
                      0.0)
    rms = np.sqrt(np.mean((full - W) ** 2, axis=1)) / np.maximum(
        W.max(axis=1), 1.0)

    if picks is None:
        ic = int(np.argmax(W.max(axis=1)))
        n = len(pos)
        runs = [[ic + d * k for k in range(4)] for d in (+1, -1)]
        runs = [r for r in runs if 0 <= min(r) and max(r) < n]
        runs.sort(key=lambda r: float(np.mean(rms[r])))
        picks = sorted(runs[0]) if runs else list(range(min(4, n)))

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True)
    for k, (ax, i) in enumerate(zip(axs.ravel(), picks)):
        ax.fill_between(t, 0, own[i], color=OWN, alpha=0.34, lw=0)
        ax.fill_between(t, own[i], own[i] + sh1[i], color=N1, alpha=0.40, lw=0)
        ax.fill_between(t, own[i] + sh1[i], full[i], color=N2, alpha=0.40,
                        lw=0)
        ax.plot(t, full[i], color=PS.INK, lw=1.7, zorder=5)
        ax.plot(t, W[i], ls='none', marker='o', ms=4.6,
                markerfacecolor='none', markeredgecolor=PS.INK,
                markeredgewidth=1.3, zorder=6)
        ax.set_ylim(0, 1.30 * max(W[i].max(), full[i].max()))
        ax.set_xlim(t[0], t[-1])
        PS.strip(ax)
        # per-panel y scale (NOT shared): these strips differ by 5x in
        # amplitude, and a shared scale leaves three of the four panels empty
        # left and right titles rather than one long one: a single string
        # carrying both the position and the fraction is wider than the panel
        ax.set_title(f'{pos[i]:.1f} mm', loc='left', fontsize=12.5,
                     color=PS.INK, pad=6)
        ax.set_title(f'{100 * shared[i]:.0f} % from ±1, ±2', loc='right',
                     fontsize=11, color=PS.MUTED, fontweight='normal', pad=6)
        if k == 0:
            for f, col, lab in ((0.97, OWN, 'own charge'),
                                (0.84, N1, '±1 neighbours'),
                                (0.71, N2, '±2 neighbours'),
                                (0.58, PS.INK, 'measured  ○')):
                ax.text(0.03, f, lab, transform=ax.transAxes, ha='left',
                        va='top', color=col, fontsize=11.5,
                        fontweight='bold' if col != PS.INK else 'normal')
    for ax in axs[-1]:
        ax.set_xlabel('time  [ns]')
    for ax in axs[:, 0]:
        ax.set_ylabel('ADC')
    fig.subplots_adjust(hspace=0.42, wspace=0.20)
    save_both(fig, out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    choices=['cartoon', 'kernels', 'build', 'decompose'])
    ap.add_argument('--plane', default='y', choices=['x', 'y'],
                    help='which plane the real-data decomposition shows '
                         '(default y: the plane the film actually shares on)')
    ap.add_argument('--no-slide', action='store_true')
    args = ap.parse_args()

    PS.use()
    # plotstyle's 160 dpi puts these at ~1000 px across.  Each one lands in one
    # column of a two-column slide, which is ~960 px on a 1920-wide projector,
    # so 160 dpi is exactly at the limit and any scaling up shows it.  220 dpi
    # gives the projector something to throw away.
    matplotlib.rcParams['savefig.dpi'] = 220
    cal, wm = _bundle()
    print(f'  bundle: {cal.summary()}')
    h = dict(cal.hyper)
    c1x, c2x = _amps(h, 'x')
    c1y, c2y = _amps(h, 'y')
    tau = h['tau_s']
    print(f'  kernel: X c1={c1x:.3f} c2={c2x:.3f} | Y c1={c1y:.3f} '
          f'c2={c2y:.3f} | c2/c1={c2y / c1y:.2f} | tau={tau:.0f} ns')
    if c2y > c1y:
        raise SystemExit('REFUSING to draw a kernel with c2 > c1 -- the +-2 '
                         'strip is reached only through the +-1 strip. Check '
                         'the bundle.')

    made = {}
    if args.only in (None, 'cartoon'):
        made['share_cartoon'] = os.path.join(FIG, 'share_cartoon_light.png')
        fig_cartoon(made['share_cartoon'], c1x, c2x, c1y, c2y, tau)
    if args.only in (None, 'kernels'):
        made['share_kernels'] = os.path.join(FIG, 'share_kernels_light.png')
        fig_kernels(made['share_kernels'], cal, wm)
    if args.only in (None, 'build'):
        made['share_build'] = os.path.join(FIG, 'share_build_light.png')
        fig_build(made['share_build'], c1y, c2y, tau, c1x, c2x)
    if args.only in (None, 'decompose'):
        st = worked_plane(cal, wm, plane=args.plane)
        print(f'  event {st["eid"]} {st["plane"]} plane: '
              f'chi2/dof {st["chi2_dof"]:.1f}, tan {st["tan"]:+.3f} '
              f'(reference {st["tan_ref"]:+.3f}), t0 {st["t0"]:.0f} ns, '
              f'{len(st["pos"])} strips')
        made['share_decompose'] = os.path.join(FIG,
                                               'share_decompose_light.png')
        fig_decompose(made['share_decompose'], st)

    # the deck uses the same PNGs -- these figures are drawn for the slide in
    # the first place (no title band, no burned-in provenance), and the report
    # supplies its own caption through make_report.BLURB.
    if not args.no_slide:
        os.makedirs(SLIDE_IMG, exist_ok=True)
        for name, src in made.items():
            dst = os.path.join(SLIDE_IMG, name + '.png')
            shutil.copyfile(src, dst)
            print(f'  -> {dst}')


if __name__ == '__main__':
    main()
