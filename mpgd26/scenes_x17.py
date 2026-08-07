#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_x17.py -- the X17 physics case, as one conference figure.

This is the odd one out in the package: it is a *diagram*, not a render, so it
is drawn in matplotlib rather than PyVista.  It shares the package palette and
the "type is vector, so the PDF scales" rule with everything else.

The narrative runs left to right, which is the one thing the 3/24 version of
this slide did not do:

    n + 3He capture  ->  how 4He* gets rid of 20.58 MeV  ->  what we measure

Panel 3 carries no hand-drawn curves.  Both channels are sampled from the
generators in ``MX17_Simulation/MX17_Simulator.py`` -- the same ones the
acceptance and significance studies use, and which that module documents as
matching the Geant4 ``X17PrimaryGenerator`` event for event -- so the figure
tracks the simulation rather than paraphrasing it.

``opening_angle_pdf`` is kept alongside as an independent analytic solution for
the X17 channel: it is exact, so it both supplies the kinematic minimum the
figure annotates and checks the sampler (``validate``).

Everything numeric lives in ``X17`` so the figure and the caption can never
drift apart.
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch  # noqa: E402
import matplotlib.patheffects as pe                # noqa: E402

import style as S                                  # noqa: E402

FONT = {'family': 'DejaVu Sans'}

# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
X17 = dict(
    # n + 3He -> 4He* : the capture Q-value, i.e. the excitation energy the
    # compound nucleus has to shed.  This is the number on the 3/24 slide.
    e_capture=20.58,        # MeV
    m_x17=16.8,             # MeV, the ATOMKI mass for the proposed boson
    m_e=0.511,              # MeV
    # display-only smearing of the kinematic curve.  The Jacobian peak at the
    # minimum opening angle is a true divergence; something has to set its
    # width on paper, and a few degrees is the scale our own chambers work at
    # (2 deg per track on the Saclay bench).
    smear_deg=3.0,
)

SOURCES = ('ATOMKI anomaly:\n'
           'Krasznahorkay et al.,\n'
           'Phys. Rev. Lett. 116, 042501 (2016)')


def opening_angle_pdf(m_x=None, e_tot=None, smear_deg=None, n=400001,
                      grid=None):
    """Opening-angle distribution of e+e- from X -> e+e-, in the lab.

    Exact two-body kinematics: the decay is isotropic in the X rest frame, so
    cos(theta*) is uniform; each grid point is boosted into the lab and the
    opening angle histogrammed.  The distribution runs from a hard kinematic
    minimum -- reached at theta* = 90 deg, where it diverges as a Jacobian
    peak -- up to 180 deg, and that minimum is the whole signature.

    Returns ``(theta_deg, density, theta_min_deg)`` with the density
    normalised to unit peak.
    """
    m_x = X17['m_x17'] if m_x is None else m_x
    e_tot = X17['e_capture'] if e_tot is None else e_tot
    smear_deg = X17['smear_deg'] if smear_deg is None else smear_deg
    m_e = X17['m_e']

    e_star = m_x / 2.0
    p_star = np.sqrt(e_star ** 2 - m_e ** 2)
    gamma = e_tot / m_x
    beta_gamma = np.sqrt(max(gamma ** 2 - 1.0, 0.0))

    cos_star = np.linspace(-1.0, 1.0, n)              # isotropic -> uniform
    sin_star = np.sqrt(np.clip(1.0 - cos_star ** 2, 0.0, None))
    pt = p_star * sin_star
    pz_a = gamma * (p_star * cos_star) + beta_gamma * e_star
    pz_b = gamma * (-p_star * cos_star) + beta_gamma * e_star
    theta = np.degrees(np.arctan2(pt, pz_a) + np.arctan2(pt, pz_b))

    if grid is None:
        grid = np.linspace(0.0, 180.0, 1441)
    dens, edges = np.histogram(theta, bins=grid.size, range=(0.0, 180.0))
    centres = 0.5 * (edges[1:] + edges[:-1])
    dens = dens.astype(float)

    if smear_deg > 0:
        step = centres[1] - centres[0]
        half = int(round(4 * smear_deg / step))
        k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / smear_deg) ** 2)
        dens = np.convolve(dens, k / k.sum(), mode='same')

    return centres, dens / dens.max(), float(theta.min())


# --------------------------------------------------------------------------- #
# The modelled shapes, from the simulation package
# --------------------------------------------------------------------------- #
# Both curves in panel 3 come from MX17_Simulation/MX17_Simulator.py -- the same
# X17PhysicsSpectrum and IPCPhysicsSpectrum the acceptance and significance
# studies use, and which that module documents as matching the Geant4
# X17PrimaryGenerator event for event.  The figure therefore cannot drift away
# from the simulation: if the generator changes, this changes with it.
#
# IPC is the one that had to come from there rather than from a hand-drawn
# shape.  Its pair invariant mass is sampled from dN/dM ~ 1/M over
# [2 m_e, E_transition], so it is a *superposition* of two-body decays over the
# whole mass range, and it keeps far more yield at large opening angle than any
# single falling curve suggests -- median 30 deg, and ~30 % of it above 60 deg,
# i.e. right underneath the X17 peak.  Drawing that honestly is the difference
# between a figure that flatters the measurement and one that describes it.
SIM_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'MX17_Simulation'))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

SAMPLE_N = 400_000
SAMPLE_SEED = 20260807


def modelled_shapes(n=SAMPLE_N, seed=SAMPLE_SEED, bin_deg=0.5, smooth_deg=None):
    """Opening-angle shapes for both channels, from the simulation package.

    Sampling 2 x ``n`` events takes the better part of a minute, so the binned
    result is cached under ``.cache/``; delete it (or bump ``SAMPLE_SEED``) to
    force a resample.  Each curve is normalised to unit peak -- the relative
    rate of the two channels is exactly the thing the experiment is trying to
    measure, so the figure must not appear to assert it.

    Returns ``(theta_deg, x17, ipc)``.
    """
    smooth_deg = X17['smear_deg'] if smooth_deg is None else smooth_deg
    tag = (f'shapes_n{n}_s{seed}_b{bin_deg}_g{smooth_deg}'
           f'_m{X17["m_x17"]}_e{X17["e_capture"]}.npz')
    path = os.path.join(CACHE, tag)
    if os.path.exists(path):
        z = np.load(path)
        return z['theta'], z['x17'], z['ipc']

    if SIM_DIR not in sys.path:
        sys.path.insert(0, SIM_DIR)
    from MX17_Simulator import X17PhysicsSpectrum, IPCPhysicsSpectrum

    np.random.seed(seed)
    ang_x = X17PhysicsSpectrum(m_x17_mev=X17['m_x17'],
                               E_transition_mev=X17['e_capture']).sample(n)
    ang_i = IPCPhysicsSpectrum(E_transition_mev=X17['e_capture']).sample(n)

    edges = np.arange(0.0, 180.0 + bin_deg, bin_deg)
    centres = 0.5 * (edges[1:] + edges[:-1])

    def binned(a):
        h = np.histogram(a, bins=edges)[0].astype(float)
        if smooth_deg > 0:
            half = int(round(4 * smooth_deg / bin_deg))
            k = np.exp(-0.5 * (np.arange(-half, half + 1) * bin_deg
                               / smooth_deg) ** 2)
            h = np.convolve(h, k / k.sum(), mode='same')
        return h / h.max()

    x17, ipc = binned(ang_x), binned(ang_i)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, theta=centres, x17=x17, ipc=ipc,
                        x17_min=ang_x.min(), ipc_median=np.median(ang_i),
                        ipc_frac_gt60=(ang_i > 60).mean())
    return centres, x17, ipc


def validate(n=40_000, seed=SAMPLE_SEED):
    """Check the sampled X17 channel against the analytic solution.

    Two independent routes to the same number: ``opening_angle_pdf`` solves the
    boost on a grid here, ``X17PhysicsSpectrum`` samples it event by event over
    in the simulation package.  If the kinematic minimum ever stops agreeing,
    one of them has changed and the figure is no longer describing the
    simulation.  Returns ``(analytic_min, sampled_min, ipc_median,
    ipc_frac_above_60)``.
    """
    if SIM_DIR not in sys.path:
        sys.path.insert(0, SIM_DIR)
    from MX17_Simulator import X17PhysicsSpectrum, IPCPhysicsSpectrum

    np.random.seed(seed)
    ang_x = X17PhysicsSpectrum(m_x17_mev=X17['m_x17'],
                               E_transition_mev=X17['e_capture']).sample(n)
    ang_i = IPCPhysicsSpectrum(E_transition_mev=X17['e_capture']).sample(n)
    analytic = opening_angle_pdf()[2]
    return (analytic, float(ang_x.min()), float(np.median(ang_i)),
            float((ang_i > 60).mean()))


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
def palette(theme='light'):
    th = S.THEMES[theme]
    dark = theme == 'dark'
    return dict(
        page='#0a0d13' if dark else '#ffffff',
        ink=th['ink'] if dark else '#141b24',
        muted=th['muted'] if dark else '#5d6874',
        rule='#2b3543' if dark else '#dde2e9',
        halo='#0a0d13' if dark else '#ffffff',
        card='#141a24' if dark else '#f4f6f9',
        proton=S.COL['m3'] if not dark else '#5f9fd8',
        neutron='#8b96a3' if not dark else '#aab4c0',
        lepton='#d6402c' if not dark else '#ff7a63',
        gamma='#c1841c' if not dark else '#ffd166',
        # IPC gets its own orange so the channel is not confused with the red
        # of the e+e- arrows, which belong to the *particles* and are shared
        # with the X17 card
        ipc='#e8621f' if not dark else '#ff9152',
        x17='#a5308f' if not dark else '#e884d6',
    )


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
# Deterministic nucleon packings, in units of the nucleon radius.
_PACKING = {
    1: [(0.0, 0.0)],
    3: [(-0.92, -0.55), (0.92, -0.55), (0.0, 1.05)],
    4: [(-0.92, -0.92), (0.92, -0.92), (-0.92, 0.92), (0.92, 0.92)],
}


def nucleus(ax, x, y, n_p, n_n, r=1.35, P=None, zorder=6):
    """A little nucleus: ``n_p`` protons and ``n_n`` neutrons, packed by a fixed
    layout so the same species is drawn identically everywhere in the figure.

    Each nucleon gets an offset highlight rather than a gradient -- it reads as
    a sphere at slide size and stays a single vector path in the PDF.
    """
    slots = _PACKING[n_p + n_n]
    cols = [P['proton']] * n_p + [P['neutron']] * n_n
    for (dx, dy), c in zip(slots, cols):
        cx, cy = x + dx * r, y + dy * r
        ax.add_patch(Circle((cx, cy), r, facecolor=c, edgecolor='none',
                            zorder=zorder))
        ax.add_patch(Circle((cx - 0.30 * r, cy + 0.32 * r), r * 0.40,
                            facecolor='#ffffff', alpha=0.30, edgecolor='none',
                            zorder=zorder + 1))
    return x, y


def excitation_waves(ax, x, y, P, r=1.35, n=3, zorder=8):
    """Three short wavy arcs standing off a nucleus -- "this one is excited".

    Deliberately small and low-contrast: it is a state marker, not a channel,
    and it must not compete with the three de-excitation arrows for attention.
    """
    for ang in (128.0, 48.0, -36.0):
        a0, a1 = np.radians(ang) - 0.30, np.radians(ang) + 0.30
        t = np.linspace(0, 1, 120)
        a = a0 + (a1 - a0) * t
        # taper the wave to nothing at both ends so the arc doesn't finish on a
        # stray kink -- that is what makes small squiggles look scraggly
        rr = r * 2.85 + 0.16 * np.sin(2 * np.pi * 2.0 * t) * np.sin(np.pi * t)
        ax.plot(x + rr * np.cos(a), y + rr * np.sin(a), color=P['ink'],
                lw=1.0, alpha=0.38, solid_capstyle='round', zorder=zorder)


def squiggle(ax, x0, y0, x1, y1, color, n_wave=5, amp=0.85, lw=1.9,
             zorder=5):
    """A photon: sine along the segment, with an arrowhead on the end."""
    t = np.linspace(0, 1, 240)
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    off = amp * np.sin(2 * np.pi * n_wave * t) * np.clip(np.sin(np.pi * t) * 2, 0, 1)
    xs = x0 + ux * L * t - uy * off
    ys = y0 + uy * L * t + ux * off
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=zorder)
    ax.annotate('', xy=(x1, y1), xytext=(xs[-6], ys[-6]),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def lepton_fork(ax, x0, y0, length, half_angle_deg, P, lw=1.9, zorder=5,
                label=True, fs=8.5):
    """The e+e- pair: two arrows opening symmetrically about the horizontal."""
    a = np.radians(half_angle_deg)
    for sgn, lab in ((+1, r'$e^+$'), (-1, r'$e^-$')):
        x1 = x0 + length * np.cos(a)
        y1 = y0 + sgn * length * np.sin(a)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=P['lepton'], lw=lw,
                                    mutation_scale=11, shrinkA=1.0, shrinkB=0),
                    zorder=zorder)
        if label:
            ax.text(x1 + 0.9, y1 + sgn * 0.35, lab, color=P['lepton'],
                    fontsize=fs, ha='left', va='center', zorder=zorder,
                    **FONT)


def arrow(ax, p0, p1, color, lw=1.6, style='-|>', rad=0.0, ls='-',
          zorder=4, ms=12, alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                        linestyle=ls, mutation_scale=ms, alpha=alpha,
                        shrinkA=0, shrinkB=0, zorder=zorder,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    return a


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
W, H = 160.0, 90.0          # canvas units; 16:9, isotropic (circles stay round)


def draw(theme='light', dpi=300, title=True):
    """Build the whole figure and return it. Caller saves it."""
    P = palette(theme)
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig = plt.figure(figsize=(W / 10.0, H / 10.0), dpi=dpi, facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')

    halo = [pe.withStroke(linewidth=2.4, foreground=P['halo'], alpha=0.85)]

    # ---------------- header ------------------------------------------------
    if title:
        ax.text(8, 84.2, 'The X17 signature in n + $^{3}$He',
                fontsize=21, fontweight='bold', color=P['ink'],
                ha='left', va='center', **FONT)
        ax.text(8, 78.6,
                'Neutron capture leaves $^{4}$He$^{*}$ with 20.58 MeV.  '
                'One of the ways it could come back down is a 17 MeV boson.',
                fontsize=10.5, color=P['muted'], ha='left', va='center', **FONT)
        ax.plot([8, 152], [75.0, 75.0], color=P['rule'], lw=1.0, zorder=1)

    _panel_levels(ax, P, halo)
    _panel_channels(ax, P, halo)
    _panel_signature(fig, ax, P, halo)
    _footer(ax, P)
    return fig


# ---- panel 1: the level scheme -------------------------------------------- #
Y_EXC, Y_GND = 58.0, 24.0
X_LVL0, X_LVL1 = 15.0, 52.0
Y_HEAD = 72.0                # the three panel headings share a baseline


def _panel_levels(ax, P, halo):
    ax.text(8, Y_HEAD, '1.  Neutron capture', fontsize=11, fontweight='bold',
            color=P['ink'], ha='left', va='center', **FONT)

    # --- energy axis ---
    ax.plot([11.2, 11.2], [Y_GND, Y_EXC], color=P['muted'], lw=1.2, zorder=2)
    for y, lab in ((Y_GND, '0'), (Y_EXC, '20.58')):
        ax.plot([10.4, 11.2], [y, y], color=P['muted'], lw=1.2, zorder=2)
        ax.text(9.9, y, lab, fontsize=8.5, color=P['muted'], ha='right',
                va='center', **FONT)
    ax.text(6.4, (Y_GND + Y_EXC) / 2, 'excitation energy  (MeV)', fontsize=8.5,
            color=P['muted'], ha='center', va='center', rotation=90, **FONT)

    # --- the two levels ---
    for y in (Y_EXC, Y_GND):
        ax.plot([X_LVL0, X_LVL1], [y, y], color=P['ink'], lw=2.6,
                solid_capstyle='round', zorder=4)

    # --- entrance channel: n + 3He -> 4He* ---
    nucleus(ax, 17.0, 66.6, 0, 1, P=P)
    ax.text(20.8, 66.4, '+', fontsize=13, color=P['muted'], ha='center',
            va='center', **FONT)
    nucleus(ax, 25.4, 66.2, 2, 1, P=P)
    ax.text(17.0, 62.4, 'n', fontsize=10, color=P['ink'], ha='center',
            va='center', **FONT)
    ax.text(25.4, 62.4, '$^{3}$He', fontsize=10, color=P['ink'], ha='center',
            va='center', **FONT)
    arrow(ax, (29.6, 66.0), (37.4, 59.5), P['muted'], lw=1.5, rad=-0.22)
    ax.text(34.4, 64.9, 'capture', fontsize=9, color=P['muted'], ha='center',
            va='center', rotation=-22, **FONT)

    # --- the compound nucleus, on its level ---
    excitation_waves(ax, 44.0, 62.2, P)
    nucleus(ax, 44.0, 62.2, 2, 2, P=P)
    ax.text(49.4, 63.2, '$^{4}$He$^{*}$', fontsize=12, fontweight='bold',
            color=P['ink'], ha='left', va='center', **FONT)
    ax.text(49.4, 59.8, 'compound', fontsize=8.5, color=P['muted'],
            ha='left', va='center', **FONT)

    # --- the ground state, under its level ---
    nucleus(ax, 44.0, 19.8, 2, 2, P=P)
    ax.text(49.4, 20.8, '$^{4}$He', fontsize=12, fontweight='bold',
            color=P['ink'], ha='left', va='center', **FONT)
    ax.text(49.4, 17.4, 'ground state', fontsize=8.5, color=P['muted'],
            ha='left', va='center', **FONT)

    # --- the energy that has to go somewhere ---
    arrow(ax, (20.0, Y_EXC - 0.6), (20.0, Y_GND + 0.6), P['ink'], lw=1.7,
          style='<|-|>', ms=11, zorder=5)
    ax.text(22.0, (Y_GND + Y_EXC) / 2, '20.58 MeV', fontsize=13,
            fontweight='bold', color=P['ink'], ha='left', va='center',
            path_effects=halo, zorder=6, **FONT)
    ax.text(22.0, (Y_GND + Y_EXC) / 2 - 3.6, 'to be released',
            fontsize=9, color=P['muted'], ha='left', va='center',
            path_effects=halo, zorder=6, **FONT)

    # --- nucleon key ---
    for i, (col, lab) in enumerate(((P['proton'], 'proton'),
                                    (P['neutron'], 'neutron'))):
        x = 15.6 + i * 13.0
        ax.add_patch(Circle((x, 12.6), 1.0, facecolor=col, edgecolor='none',
                            zorder=5))
        ax.text(x + 1.9, 12.6, lab, fontsize=8.5, color=P['muted'],
                ha='left', va='center', **FONT)


# ---- panel 2: the three de-excitation channels ---------------------------- #
CARD_X0, CARD_X1 = 63.0, 104.0
CARD_Y = (57.5, 41.0, 24.5)      # centres, top to bottom
CARD_H = 14.0


def _panel_channels(ax, P, halo):
    ax.text(CARD_X0, Y_HEAD, '2.  Three ways down', fontsize=11,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)

    # fan from the end of the 4He* level into the three cards
    for y in CARD_Y:
        arrow(ax, (X_LVL1 + 6.6, Y_EXC), (CARD_X0 - 1.2, y), P['muted'],
              lw=1.1, rad=0.16, alpha=0.65, ms=10, zorder=2)

    cards = [
        dict(y=CARD_Y[0], accent=P['gamma'], hypothesis=False,
             name='Gamma emission',
             eq=r'$^{4}$He$^{*}\ \rightarrow\ ^{4}$He $+\ \gamma$',
             note='one 20.58 MeV photon'),
        dict(y=CARD_Y[1], accent=P['ipc'], hypothesis=False,
             name='Internal pair conversion',
             eq=r'$^{4}$He$^{*}\ \rightarrow\ ^{4}$He $+\ e^{+}e^{-}$',
             note='known QED process; pairs favour\nsmall opening angle'),
        dict(y=CARD_Y[2], accent=P['x17'], hypothesis=True,
             name='X17 emission and decay',
             eq=r'$^{4}$He$^{*} \rightarrow\ ^{4}$He $+\ X17,'
                r'\ \ X17 \rightarrow e^{+}e^{-}$',
             note='hypothetical boson, m $\\approx$ 17 MeV'),
    ]

    for c in cards:
        y = c['y']
        box = FancyBboxPatch(
            (CARD_X0, y - CARD_H / 2), CARD_X1 - CARD_X0, CARD_H,
            boxstyle='round,pad=0,rounding_size=1.6',
            facecolor=P['card'],
            edgecolor=c['accent'] if c['hypothesis'] else P['rule'],
            linestyle='--' if c['hypothesis'] else '-',
            lw=1.6 if c['hypothesis'] else 1.0, zorder=3)
        ax.add_patch(box)
        # accent spine
        ax.plot([CARD_X0 + 0.9, CARD_X0 + 0.9], [y - CARD_H / 2 + 1.4,
                                                 y + CARD_H / 2 - 1.4],
                color=c['accent'], lw=2.4, solid_capstyle='round', zorder=4)

        ax.text(CARD_X0 + 3.4, y + 4.6, c['name'], fontsize=10.5,
                fontweight='bold', color=P['ink'], ha='left', va='center',
                **FONT)
        ax.text(CARD_X0 + 3.4, y + 0.7, c['eq'], fontsize=9.5,
                color=P['ink'], ha='left', va='center', **FONT)
        ax.text(CARD_X0 + 3.4, y - 3.9, c['note'], fontsize=8.5,
                color=P['muted'], ha='left', va='center', linespacing=1.45,
                **FONT)
        if c['hypothesis']:
            chip = FancyBboxPatch((CARD_X1 - 12.6, y + CARD_H / 2 - 3.6), 10.6,
                                  3.0, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=c['accent'], edgecolor='none',
                                  alpha=0.16, zorder=4)
            ax.add_patch(chip)
            ax.text(CARD_X1 - 7.3, y + CARD_H / 2 - 2.1, 'HYPOTHESIS',
                    fontsize=7.2, fontweight='bold', color=c['accent'],
                    ha='center', va='center', **FONT)

    # --- the little process pictures, to the right of each card ------------ #
    xi = CARD_X1 + 2.2
    # gamma
    squiggle(ax, xi, CARD_Y[0], xi + 7.6, CARD_Y[0], P['gamma'], n_wave=4,
             amp=0.9)
    ax.text(xi + 9.6, CARD_Y[0], r'$\gamma$', fontsize=13, color=P['gamma'],
            ha='left', va='center', **FONT)
    # IPC: a tight fork
    lepton_fork(ax, xi + 0.6, CARD_Y[1], 7.2, 16.0, P)
    # X17: dashed line, then a wide fork
    arrow(ax, (xi - 0.4, CARD_Y[2]), (xi + 5.4, CARD_Y[2]), P['x17'], lw=2.0,
          ls=(0, (3.2, 2.0)), style='-', zorder=5)
    ax.text(xi + 2.4, CARD_Y[2] + 2.0, 'X17', fontsize=9, fontweight='bold',
            color=P['x17'], ha='center', va='center', **FONT)
    lepton_fork(ax, xi + 5.4, CARD_Y[2], 6.6, 52.0, P)


# ---- panel 3: what we actually measure ------------------------------------ #
def _panel_signature(fig, ax, P, halo):
    ax.text(122.0, Y_HEAD, '3.  What we measure', fontsize=11,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)

    th, x17, ipc = modelled_shapes()
    th_min = opening_angle_pdf()[2]

    # axes placed in canvas units so it lines up with the cards
    px = fig.add_axes([122.0 / W, 24.0 / H, 30.0 / W, 36.0 / H],
                      facecolor='none')
    for s in ('top', 'right'):
        px.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        px.spines[s].set_color(P['muted'])
        px.spines[s].set_linewidth(0.9)
    px.tick_params(colors=P['muted'], labelsize=8, width=0.9, length=3)
    for lab in px.get_xticklabels() + px.get_yticklabels():
        lab.set_fontfamily('DejaVu Sans')

    px.fill_between(th, 0, x17, color=P['x17'], alpha=0.16, lw=0, zorder=2)
    px.plot(th, x17, color=P['x17'], lw=2.2, zorder=4,
            label='X17 $\\rightarrow e^{+}e^{-}$')
    px.plot(th, ipc, color=P['ipc'], lw=2.0, zorder=3,
            label='internal pair conversion')

    px.axvline(th_min, color=P['x17'], lw=0.9, ls=':', alpha=0.8, zorder=1)
    # above and left of the peak, right-aligned: the lower left of the panel
    # belongs to the IPC curve now, and anything to the right of the peak runs
    # off the canvas
    px.annotate(f'kinematic minimum\n{th_min:.0f}°',
                xy=(th_min + 0.5, 1.00), xytext=(101, 1.05),
                fontsize=8.2, color=P['x17'], ha='right', va='center',
                arrowprops=dict(arrowstyle='-|>', color=P['x17'], lw=1.0,
                                mutation_scale=9,
                                connectionstyle='arc3,rad=-0.3'),
                **FONT)

    px.set_xlim(0, 180)
    px.set_ylim(0, 1.20)
    px.set_xticks([0, 45, 90, 135, 180])
    px.set_yticks([])
    px.set_xlabel('e$^{+}$e$^{-}$ opening angle  (deg)', fontsize=8.8,
                  color=P['muted'], labelpad=3, **FONT)
    px.set_ylabel('yield  (arb.)', fontsize=8.8, color=P['muted'], labelpad=4,
                  **FONT)
    leg = px.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.235),
                    frameon=False, fontsize=8.2, handlelength=1.9,
                    labelspacing=0.45)
    for t in leg.get_texts():
        t.set_color(P['muted'])
        t.set_fontfamily('DejaVu Sans')

    # --- the headline, under the plot ---
    ax.plot([122.0, 152.0], [20.6, 20.6], color=P['rule'], lw=1.0, zorder=2)
    ax.text(122.0, 17.4, 'Signal', fontsize=9, fontweight='bold',
            color=P['x17'], ha='left', va='center', **FONT)
    ax.text(122.0, 13.4,
            'an e$^{+}$e$^{-}$ pair at large opening\n'
            'angle, tagged by the neutron\ntime of flight',
            fontsize=9.2, color=P['ink'], ha='left', va='center',
            linespacing=1.5, **FONT)


def _footer(ax, P):
    ax.plot([8, 152], [8.6, 8.6], color=P['rule'], lw=1.0, zorder=1)
    cap = (
        'Both curves are the MX17_Simulation generators (X17PhysicsSpectrum, '
        'IPCPhysicsSpectrum), which track the Geant4 X17PrimaryGenerator: '
        'a %.1f MeV boson, and an IPC pair whose invariant mass is drawn from '
        'dN/dM ∝ 1/M, each carrying the %.2f MeV transition and decaying '
        'isotropically in its own rest frame. %s events per channel, '
        'Gaussian-smeared by %.0f° so the X17 Jacobian peak has a width on '
        'paper; nuclear recoil is neglected. Each curve is normalised to unit '
        'peak — their relative rate is what the experiment is trying to '
        'measure, so nothing here implies it.'
        % (X17['m_x17'], X17['e_capture'], f'{SAMPLE_N:,}'.replace(',', ' '),
           X17['smear_deg']))
    ax.text(8, 5.2, textwrap.fill(cap, 168), fontsize=7.6, color=P['muted'],
            ha='left', va='center', linespacing=1.7, **FONT)
    ax.text(152, 5.2, SOURCES, fontsize=7.6, color=P['muted'], ha='right',
            va='center', linespacing=1.7, **FONT)
