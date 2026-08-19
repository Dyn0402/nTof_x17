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

# --------------------------------------------------------------------------- #
# The 3He capsule, as built
# --------------------------------------------------------------------------- #
# Axial profiles lifted verbatim from the Geant4 geometry --
# MX17_Full_Geant/src/DetectorConstruction.cc, the three nested G4Polycones
# "He3Gas" / "He3Cap_Al" / "He3Cap_CFRP", themselves sectioned from the STEP
# solid MASTINU X17 HPRV 00 01.  Drawing them as a cross-section (r against z,
# mirrored) gives the real vessel rather than a drawn approximation of it.
#
# Mounting is NOSE-FIRST: local +z runs from the tip at z = -35 mm, which faces
# the beam, up to the valve at z = +51 mm.  The EAR2 beam is vertical and comes
# from below, so local +z is world "up" and the tip is the bottom of the figure.
CAPSULE_Z_VESSEL = np.array([
    -35.0, -34.0, -33.0, -31.0, -29.0, -27.0, -25.0, -23.0, -21.0, -20.0,
    -15.0, -5.0, 5.0, 15.0, 20.0, 21.0, 23.0, 25.0, 27.0, 29.0,
    31.0, 33.0, 35.0, 37.0, 39.0, 40.0, 45.0, 50.0, 51.0])
CAPSULE_R_AL = np.array([
    0.000, 3.803, 5.287, 7.206, 8.480, 9.375, 9.994, 10.386, 10.600, 10.600,
    10.600, 10.600, 10.600, 10.600, 10.600, 10.600, 10.386, 9.994, 9.375, 8.480,
    7.206, 5.747, 4.708, 4.015, 3.621, 3.500, 3.500, 3.500, 3.500])
CAPSULE_R_CFRP = np.array([
    0.000, 4.703, 6.187, 8.106, 9.380, 10.275, 10.894, 11.286, 11.500, 11.500,
    11.500, 11.500, 11.500, 11.500, 11.500, 11.500, 11.286, 10.894, 10.275,
    9.380, 8.106, 6.647, 5.608, 4.915, 4.521, 4.400, 4.400, 4.400, 4.400])
CAPSULE_Z_GAS = np.array([
    -29.5, -28.0, -26.0, -24.0, -22.0, -20.0, -15.0, -5.0, 5.0, 15.0,
    20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0,
    40.0, 44.0, 50.7])
CAPSULE_R_GAS = np.array([
    0.001, 6.000, 8.000, 9.165, 9.798, 10.000, 10.000, 10.000, 10.000, 10.000,
    10.000, 9.798, 9.165, 8.000, 6.299, 4.842, 3.660, 2.711, 1.967, 1.410,
    1.026, 0.750, 0.750])


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


def ipc_mass_weight(m, e_tot=None):
    """The IPC parent-mass spectrum, dN/dM ~ 1/M on [2 m_e, E], unit peak.

    Same law ``_sample_ipc_mass`` inverts in MX17_Simulator; written out here
    so the story panel can *draw* it rather than sample it.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    e_tot = X17['e_capture'] if e_tot is None else e_tot
    w = np.where((m >= 2 * X17['m_e']) & (m <= e_tot), 1.0 / np.maximum(m, 1e-9),
                 0.0)
    return w / w.max()


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
        # the pair reads as one family, but the two halves are told apart by
        # more than their label: warm red for e+, cooler wine for e-.  Kept
        # close together on purpose -- they are the same kind of object, and a
        # hard red/blue split would fight the proton blue in beats 1 and 2.
        lepton='#d6402c' if not dark else '#ff7a63',
        positron='#d6402c' if not dark else '#ff8a70',
        electron='#a02c52' if not dark else '#f07a9e',
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


def _profile_polygon(z, r):
    """(z, r) axial profile -> a closed cross-section outline, r mirrored."""
    zz = np.concatenate([z, z[::-1]])
    rr = np.concatenate([r, -r[::-1]])
    return rr, zz


def draw_capsule(ax, xc, y_at_z0, scale, P, zorder=4):
    """The 3He capsule in cross-section, drawn from the Geant4 polycones.

    ``scale`` is figure units per mm; the three shells are filled from the
    outside in, so each reads as a wall of the one outside it.  True aspect --
    the vessel really is this slender.
    """
    shells = [
        (CAPSULE_Z_VESSEL, CAPSULE_R_CFRP, '#33383f', 1.0, 'cfrp'),
        (CAPSULE_Z_VESSEL, CAPSULE_R_AL, P['neutron'], 1.0, 'al'),
        (CAPSULE_Z_GAS, CAPSULE_R_GAS, S.COL['gas'], 0.85, 'gas'),
    ]
    for i, (z, r, col, alpha, _tag) in enumerate(shells):
        rr, zz = _profile_polygon(z, r)
        ax.fill(xc + rr * scale, y_at_z0 + zz * scale, facecolor=col,
                edgecolor='none', alpha=alpha, zorder=zorder + i)
    # one outline over the top, so the silhouette stays crisp at slide size
    rr, zz = _profile_polygon(CAPSULE_Z_VESSEL, CAPSULE_R_CFRP)
    ax.plot(xc + rr * scale, y_at_z0 + zz * scale, color=P['ink'], lw=0.7,
            alpha=0.55, zorder=zorder + 4)


def magnifier(ax, c1, r1, c2, r2, P, zorder=3, lw=0.9):
    """Small circle on the object, big circle holding the zoom, and the two
    external tangents between them -- the standard callout."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = np.linalg.norm(c2 - c1)
    alpha = np.arctan2(*(c2 - c1)[::-1])
    phi = np.arccos(np.clip((r2 - r1) / d, -1, 1))
    for sgn in (+1, -1):
        a = alpha + sgn * phi
        u = np.array([np.cos(a), np.sin(a)])
        ax.plot(*zip(c1 + r1 * u, c2 + r2 * u), color=P['muted'], lw=lw,
                alpha=0.55, zorder=zorder)
    ax.add_patch(Circle(c1, r1, facecolor='none', edgecolor=P['muted'],
                        lw=lw, alpha=0.85, zorder=zorder + 1))
    ax.add_patch(Circle(c2, r2, facecolor=P['card'], edgecolor=P['muted'],
                        lw=lw, alpha=1.0, zorder=zorder + 1))


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
    for sgn, lab, col in ((+1, r'$e^+$', P['positron']),
                          (-1, r'$e^-$', P['electron'])):
        x1 = x0 + length * np.cos(a)
        y1 = y0 + sgn * length * np.sin(a)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=lw,
                                    mutation_scale=11, shrinkA=1.0, shrinkB=0),
                    zorder=zorder)
        if label:
            ax.text(x1 + 0.9, y1 + sgn * 0.35, lab, color=col,
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


# --------------------------------------------------------------------------- #
# The "story" layout -- the same physics in five beats over two rows
# --------------------------------------------------------------------------- #
# Where the compact layout asserts that X17 makes a peak at a large opening
# angle, this one derives it: beam -> capture -> level drop -> *why the parent
# mass sets the angle* -> the distribution.  Beat 4 is the one that earns the
# extra row; without it panel 5 is a picture you have to be told how to read.
#
# THE STORY CANVAS IS NARROWER THAN THE COMPACT ONE, and that is the whole
# mechanism behind how big this figure comes out on a slide (2026-08-18, Dylan:
# "make them taller such that they better fill the page").  On the deck each
# row is one slide's figure, so it is WIDTH-limited: whatever the canvas is, it
# is drawn across the same ~12.4 in of slide.  Type is set in POINTS and the
# drawing in canvas UNITS, so the only lever on how big both come out is how
# many units the row spans -- 160 units across 12.4 in renders 9 pt type at
# 7 pt, 124 units renders it at 9 pt.  Hence SW: the beats were re-flowed from
# three/two wide, flat boxes into narrower, taller ones, the row went from
# 160x35 to 124x57 (4.6:1 -> 2.2:1, which is the shape of the hole in the
# slide), and everything on it is ~29 % larger without one font size changing.
# Do not widen SW back out to W without re-flowing the beats: the row would go
# flat again and the type would shrink with it.
SW = 124.0                  # story-canvas width (the compact layout is W=160)
S_ROW1 = (66.0, 120.0)      # (bottom, top) of the upper row
S_ROW2 = (7.0, 63.0)
S_HEAD1, S_HEAD2 = 117.6, 60.6
S_A = (1.0, 37.0)           # 1 beam on target
S_B = (40.0, 68.0)          # 2 capture
S_C = (71.0, 122.0)         # 3 de-excitation
S_D = (3.0, 79.0)           # 4 why the pair opens the way it does
S_E = (82.0, 122.0)         # 5 what we measure


# Each part is the SAME drawing seen through a different window: the beats keep
# their absolute coordinates, and the canvas is cropped to the band that holds
# them.  That is why the split slides need no second layout to maintain -- a
# change to a beat lands in the combined figure and in its slide together.
#
# ``full`` is the band with the title and caption bands included, ``bare`` the
# band with just the drawing.  Header/footer y move with the band.
#
# The two ``bare`` bands are each ~2.16:1 BY DESIGN -- that is the shape of the
# figure hole on the deck slide (1186 x 547 px, measured, see slides/NOTES.md),
# so a row printed bare fills it top to bottom instead of floating in a band of
# white.  Change a beat's height and re-measure; a row that ends up flatter
# than ~2.2:1 is leaving slide height, and rendered size, on the table.
STORY_PARTS = {
    'all': dict(full=(-6.6, 136.0), bare=(4.6, 121.6),
                head=(132.6, 127.0, 124.0), foot=-1.8, beats='12345'),
    'top': dict(full=(56.0, 136.0), bare=(64.2, 121.6),
                head=(132.6, 127.0, 124.0), foot=60.2, beats='123'),
    # the BOTTOM bare band is 124 x 61.1 = 2.03:1 since 2026-08-18, not 2.16:1.
    # Its deck slide lost its .fig-label that day, and four lines of small type
    # under the picture were four lines of figure height: the hole re-measured
    # 2.028:1 with the red-box recipe, so the band follows it.  The top row keeps
    # its label and keeps 2.16.
    'bottom': dict(full=(-6.6, 78.0), bare=(0.9, 62.0),
                   head=(74.6, 69.0, 66.0), foot=-1.8, beats='45'),
}

STORY_TITLES = {
    'all': ('How a 17 MeV boson would show up in n + $^{3}$He',
            'The pair opening angle is set by the mass of whatever emitted it '
            '— which is why a single new mass would put a hard edge in a '
            'smooth background.'),
    'top': ('How a 17 MeV boson would show up in n + $^{3}$He',
            'Capture leaves $^{4}$He$^{*}$ with 20.58 MeV.  Two of the three '
            'ways it can shed that energy put an e$^{+}$e$^{-}$ pair in the '
            'detector.'),
    # deliberately does NOT repeat beat 4's own subtitle, which already says
    # the pair leaves back-to-back
    'bottom': ('The opening angle is set by the boost',
               'Everything the lab sees is what the boost did to the pair.  '
               'One new mass puts a hard edge in the spectrum; a spread of '
               'masses only makes a slope.'),
}

TOP_CAPTION = (
    'Nuclei, gas volume and beam are schematic — this row carries no measured '
    'quantity. The 20.58 MeV is the n + $^{3}$He capture Q-value, and it is '
    'what fixes every angle on the next slide.')


def draw_story(theme='light', dpi=300, title=True, capsule=False, part='all',
               upto=None, detect=False):
    """The five-beat layout, or one row of it, or one row part-drawn.

    ``part`` is 'all', 'top' (beats 1-3) or 'bottom' (beats 4-5).  The two rows
    split cleanly across two slides: the top one sets up the physics, the
    bottom one derives the measurement from it.

    ``detect`` (bottom row only) draws the micro-TPC cartoon in beat 4's box
    instead of the boost rows: the last frame of the deck's slide 6, where the
    spectrum stays exactly where it is and the argument beside it changes.

    ``upto`` keeps only the first N beats of that part, on the SAME canvas --
    which is how a row is turned into a build (2026-08-17).  The frames are
    strict subsets of one picture: identical canvas, identical coordinates, so
    a beat lands in its final position the moment it appears and nothing
    already on the slide moves.  Same discipline as the EAR2 build in the deck.
    """
    spec = STORY_PARTS[part]
    y0, y1 = spec['full'] if title else spec['bare']
    P = palette(theme)
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig = plt.figure(figsize=(SW / 10.0, (y1 - y0) / 10.0), dpi=dpi,
                     facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, SW)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = [pe.withStroke(linewidth=2.4, foreground=P['halo'], alpha=0.85)]

    if title:
        ty, sy, ry = spec['head']
        head, sub = STORY_TITLES[part]
        ax.text(3, ty, head, fontsize=19.5, fontweight='bold', color=P['ink'],
                ha='left', va='center', **FONT)
        # wrapped: the subtitle used to run as one line across a 160-unit
        # canvas, which is wider than this one
        ax.text(3, sy, textwrap.fill(sub, 96), fontsize=10, color=P['muted'],
                ha='left', va='center', linespacing=1.5, **FONT)
        ax.plot([3, SW - 3], [ry, ry], color=P['rule'], lw=1.0, zorder=1)

    beats = spec['beats'] if upto is None else spec['beats'][:upto]
    if '1' in beats:
        _story_beam(ax, P, capsule=capsule)
    if '2' in beats:
        _story_capture(ax, P)
    if '3' in beats:
        _story_levels(ax, P, halo)
    if '4' in beats:
        # ``detect`` swaps beat 4's boost rows for the micro-TPC cartoon in the
        # same box -- deck frame 6.3, see _story_detect
        (_story_detect(ax, P, halo) if detect
         else _story_mechanism(fig, ax, P, halo))
    if '5' in beats:
        _story_measure(fig, ax, P, halo, y0=y0, y1=y1)
    if title:
        _story_footer(ax, P, spec['foot'],
                      TOP_CAPTION if part == 'top' else None)
    return fig


# Each beat, on its own canvas -- added 2026-08-16 so the five pictures can be
# dropped into slides one at a time (a build, another deck, a poster) instead
# of only as the compilation.  Same principle as STORY_PARTS above and the same
# consequence: THERE IS NO SECOND DRAWING TO MAINTAIN.  A beat keeps its
# absolute coordinates and the canvas is cropped to the window that holds it,
# so an edit to a beat lands in the compilation and in its standalone file
# together, at the same size, in the same style.
#
# The windows are (x0, x1, y0, y1) in canvas units, padded off the S_A..S_E
# extents; the y bands are the STORY_PARTS 'bare' rows, so a beat printed alone
# is the same height it is in the compilation.
BEAT_WINDOWS = {
    '1': (0.0, 38.5, 64.2, 121.6),
    '2': (38.5, 69.0, 64.2, 121.6),
    '3': (69.0, 124.0, 64.2, 121.6),
    # beat 4 runs a little past its nominal S_D right edge -- the last
    # orientation column's "back-to-back" / "collinear" note. Cropping at
    # S_D[1] cut them off, so the window overlaps beat 5's; the two never share
    # a file, so the overlap costs nothing but a little whitespace.
    '4': (1.0, 86.5, 0.9, 62.0),
    '5': (84.0, 124.0, 0.9, 62.0),
}
BEAT_NAMES = {'1': 'beam', '2': 'capture', '3': 'channels',
              '4': 'boost', '5': 'spectrum'}


def draw_beat(beat, theme='light', dpi=300, capsule=False):
    """One beat of the story layout, alone on a canvas cropped to it.

    ``beat`` is '1'..'5'.  ``capsule`` applies to beat 1 only and swaps the
    generic gas volume for the real Geant4 vessel, exactly as in ``draw_story``.
    """
    beat = str(beat)
    x0, x1, y0, y1 = BEAT_WINDOWS[beat]
    P = palette(theme)
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig = plt.figure(figsize=((x1 - x0) / 10.0, (y1 - y0) / 10.0), dpi=dpi,
                     facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = [pe.withStroke(linewidth=2.4, foreground=P['halo'], alpha=0.85)]

    if beat == '1':
        _story_beam(ax, P, capsule=capsule)
    elif beat == '2':
        _story_capture(ax, P)
    elif beat == '3':
        _story_levels(ax, P, halo)
    elif beat == '4':
        _story_mechanism(fig, ax, P, halo)
    elif beat == '5':
        _story_measure(fig, ax, P, halo, y0=y0, y1=y1, x0f=x0, x1f=x1)
    return fig


def _head(ax, x, y, text, P):
    ax.text(x, y, text, fontsize=10.5, fontweight='bold', color=P['ink'],
            ha='left', va='center', **FONT)


# --------------------------------------------------------------------------- #
# ...and the beat that hands the story to the detector (added 2026-08-17)
# --------------------------------------------------------------------------- #
# Dylan: "slide 6.2 needs a transition to the Micromegas. Maybe we add a 6.3
# with a cartoon at the bottom of the page showing e+e- coming out at an angle
# and going through a cartoon of our Micromegas drift volume."
#
# By the end of beat 5 the audience has been told that the observable is an
# ANGLE.  Nothing on the slide has yet said what measures one, and the next
# slide opens on an exploded chamber -- which lands as a change of subject
# unless something bridges it.  This is the bridge, and it makes exactly one
# claim: a micro-TPC turns ONE gas gap into a DIRECTION, so two of them give
# the opening angle.  That is the argument for the whole detector half of the
# talk, in one picture.
#
# It is its own canvas rather than a sixth beat of the story layout: the story
# rows are full (S_A..S_E cover 8..152 on both), and this figure has to be a
# short wide band because it sits UNDER the spectrum on a slide that already
# has one figure on it.
#
# Drawn, not simulated -- and deliberately not to scale.  The real standoff is
# 204 mm from a 23 mm capsule, i.e. the chambers are nine capsule-diameters
# away and the drift gap is a tenth of that again; at scale the gas gap would
# be a hairline.  What IS honest here is the opening angle (a real 110 deg,
# the kinematic minimum beat 5 has just put on the screen) and the topology:
# the pair crosses the gas, the ionisation drifts to the mesh, and the fit
# reads a direction out of one gap.
# W is trimmed to the ink (2026-08-17): at 160 the drawing left 15 units of
# margin on each side, and on the slide that margin is height -- the figure is
# width-limited, so empty canvas at the sides shrinks the whole picture.
DETECT_W, DETECT_H = 142.0, 47.0
DETECT_OPENING_DEG = 110.0      # the X17 kinematic minimum, drawn true

# geometry of one drawn chamber, in canvas units along its own leg
_ARM = 24.0                     # vertex -> near face of the drift volume
_GAP = 13.0                     # drawn drift volume (the real 30 mm)
_BOARD = 3.0                    # readout board
_FACE = 34.0                    # how much of the 400 mm plane is drawn
_TILT = 21.0                    # track's angle of incidence on the chamber


def _utpc_arm(ax, vertex, ang_deg, colour, P, halo, flip=False, scale=1.0):
    """One leg of the pair crossing one Micromegas, drawn along ``ang_deg``.

    ``ang_deg`` is measured from +x and IS the lepton's direction, so the angle
    the two arms subtend on the page is the real opening angle and can be
    measured off the figure with a protractor.  The CHAMBER is what gets
    rotated: its readout plane is tilted ``_TILT`` off normal to the leg, which
    is the only reason a micro-TPC has anything to reconstruct -- a track
    arriving square to the plane deposits all its charge at one depth.

    ``scale`` multiplies every LENGTH and leaves every ANGLE alone, so the same
    drawing can stand on its own canvas (draw_detect) or inside beat 4's box on
    the story canvas (_story_detect) without a second copy of the geometry.
    """
    _ARM, _GAP, _BOARD, _FACE = (globals()['_ARM'] * scale,
                                 globals()['_GAP'] * scale,
                                 globals()['_BOARD'] * scale,
                                 globals()['_FACE'] * scale)
    th = np.radians(ang_deg)
    u = np.array([np.cos(th), np.sin(th)])          # along the leg, outward
    o = np.asarray(vertex, float)

    # the chamber's own frame: m out of the readout plane, p across it
    tilt = np.radians(-_TILT if flip else _TILT)
    m = np.array([np.cos(th + tilt), np.sin(th + tilt)])
    p = np.array([-m[1], m[0]])
    # centred so the leg crosses the middle of the gas
    c = o + u * (_ARM + _GAP / 2.0)

    def quad(a0, a1):
        return [c + m * a0 - p * _FACE / 2, c + m * a0 + p * _FACE / 2,
                c + m * a1 + p * _FACE / 2, c + m * a1 - p * _FACE / 2]

    ax.add_patch(plt.Polygon(quad(-_GAP / 2, _GAP / 2), closed=True,
                             facecolor=S.COL['gas'], alpha=0.20,
                             edgecolor=S.COL['gas'], lw=1.1, zorder=3))
    ax.add_patch(plt.Polygon(quad(_GAP / 2, _GAP / 2 + _BOARD), closed=True,
                             facecolor=S.COL['pcb'], edgecolor='none',
                             zorder=5))
    mesh = np.array([c + m * _GAP / 2 - p * _FACE / 2,
                     c + m * _GAP / 2 + p * _FACE / 2])
    ax.plot(mesh[:, 0], mesh[:, 1], color=S.COL['mesh'], lw=1.4, zorder=5)

    # the lepton, straight along its own direction, in and out of the gas
    reach = (_GAP / 2 + _BOARD + 5.0 * scale) / np.cos(tilt)
    arrow(ax, o + u * 5.5 * scale, c + u * reach, colour, lw=2.0 * scale ** 0.5,
          ms=13 * scale ** 0.5, zorder=7)

    # the ionisation it leaves in the gas, and its drift to the mesh.  The
    # drift is along the chamber normal; the track is not -- that difference IS
    # the depth-versus-position ladder the fit reads.
    for f in np.linspace(-0.40, 0.40, 6):
        q = c + u * (f * _GAP / np.cos(tilt))
        depth = _GAP / 2 - float(np.dot(q - c, m))      # distance to the mesh
        ax.plot(*np.array([q, q + m * depth]).T, color=S.COL['gas'], lw=1.0,
                alpha=0.95, zorder=4)
        ax.plot([q[0]], [q[1]], marker='o', ms=2.8 * scale ** 0.5,
                color='#e0a52f', mec='none', zorder=6)
    return u, m, p, c


def draw_detect(theme='light', dpi=300, title=False):
    """The transition figure: two micro-TPCs measure the opening angle.

    ``title`` is accepted and ignored -- this figure only ever appears under a
    slide that has one (make_x17 forces it bare).
    """
    P = palette(theme)
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    fig = plt.figure(figsize=(DETECT_W / 10.0, DETECT_H / 10.0), dpi=dpi,
                     facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, DETECT_W)
    ax.set_ylim(0, DETECT_H)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = [pe.withStroke(linewidth=2.6, foreground=P['halo'], alpha=0.9)]

    vx, vy = DETECT_W / 2, 7.0
    half = DETECT_OPENING_DEG / 2.0
    ang_e, ang_p = 90.0 + half, 90.0 - half      # symmetric about vertical

    _, m_e, p_e, c_e = _utpc_arm(ax, (vx, vy), ang_e, P['electron'], P, halo,
                                 flip=True)
    _, m_p, p_p, c_p = _utpc_arm(ax, (vx, vy), ang_p, P['positron'], P, halo)

    # the vertex, and the angle it subtends
    ax.plot([vx], [vy], marker='o', ms=7, color=P['gamma'], mec=P['ink'],
            mew=0.8, zorder=8)
    r = 15.0
    t = np.radians(np.linspace(ang_p, ang_e, 100))
    ax.plot(vx + r * np.cos(t), vy + r * np.sin(t), color=P['ink'], lw=1.2,
            ls='--', alpha=0.75, zorder=6)
    ax.text(vx, vy + r + 2.4, f'θ = {DETECT_OPENING_DEG:.0f}°', fontsize=12.5,
            fontweight='bold', color=P['ink'], ha='center', va='bottom',
            path_effects=halo, zorder=9, **FONT)
    ax.text(vx, vy - 3.4, 'e$^{+}$e$^{-}$ from $^{4}$He*', fontsize=9.5,
            color=P['muted'], ha='center', va='top', **FONT)
    ax.text(vx - 26.0, vy + 4.0, 'e$^{-}$', fontsize=12, fontweight='bold',
            color=P['electron'], ha='center', va='center', path_effects=halo,
            zorder=9, **FONT)
    ax.text(vx + 26.0, vy + 4.0, 'e$^{+}$', fontsize=12, fontweight='bold',
            color=P['positron'], ha='center', va='center', path_effects=halo,
            zorder=9, **FONT)

    # ONE sentence, and it is the whole argument for the rest of the talk.
    ax.text(vx, DETECT_H - 1.5,
            'One gas gap → a 3-D segment.   Two segments → the opening angle.',
            fontsize=12.5, fontweight='bold', color=P['ink'], ha='center',
            va='top', path_effects=halo, zorder=9, **FONT)
    # the object itself, named once, in the empty corner above the left
    # chamber (the only quadrant of this canvas with nothing in it)
    ax.text(6.0, DETECT_H - 10.0, 'Micromegas µTPC\n30 mm drift gap',
            fontsize=10.0, color=P['muted'], ha='left', va='center',
            linespacing=1.35, **FONT)
    return fig


# The contents of the gas volume.  Placed at random, but from a FIXED seed and
# with a rejection radius, so it reads as a gas and still comes out identical
# on every rebuild.  (A golden-angle spiral was the first attempt: perfectly
# even coverage, and it read as a manufactured lattice.)
HE3_N, HE3_SEED = 30, 11
NEUTRON_N, NEUTRON_SEED = 17, 5


def _poisson_points(rng, n, min_sep, sampler, max_tries=60000):
    """Dart-throwing with a rejection radius: up to n points, none closer to
    each other than ``min_sep``."""
    pts = []
    for _ in range(max_tries):
        if len(pts) >= n:
            break
        p = sampler(rng)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2
               for q in pts):
            pts.append(p)
    return np.array(pts)


def he3_sites(cx, cy, r_fill, n=HE3_N, min_sep=1.8, seed=HE3_SEED):
    rng = np.random.RandomState(seed)

    def sample(r):
        a = r.uniform(0.0, 2.0 * np.pi)
        rad = r_fill * np.sqrt(r.uniform())        # uniform over the disc
        return (cx + rad * np.cos(a), cy + rad * np.sin(a))

    return _poisson_points(rng, n, min_sep, sample)


def neutron_sites(cx, y_lo, y_hi, half_w, n=NEUTRON_N, min_sep=1.5,
                  seed=NEUTRON_SEED):
    rng = np.random.RandomState(seed)

    def sample(r):
        return (cx + r.uniform(-half_w, half_w), r.uniform(y_lo, y_hi))

    return _poisson_points(rng, n, min_sep, sample)


def _story_beam(ax, P, capsule=False):
    """EAR2 is a vertical beamline: the neutrons arrive from below.

    Deliberately generic -- a beam and some 3He, no vessel -- because at this
    point in the talk the target hardware has not been introduced yet.  Pass
    ``capsule=True`` for the version that draws the real Geant4 vessel, which
    is the right picture once it has been.
    """
    x0, x1 = S_A
    _head(ax, x0, S_HEAD1, '1.  A neutron beam on $^{3}$He', P)

    if capsule:
        return _story_beam_capsule(ax, P)

    # --- the gas volume: a light disc full of 3He ---
    cx, cy, rv = (x0 + x1) / 2, 96.0, 11.6
    ax.add_patch(Circle((cx, cy), rv, facecolor=S.COL['gas'], alpha=0.14,
                        edgecolor=P['rule'], lw=1.0, zorder=2))
    sites = he3_sites(cx, cy, rv - 2.0)
    for x, y in sites:
        nucleus(ax, x, y, 2, 1, r=0.65, P=P, zorder=6)
    ax.text(cx + rv - 1.4, cy + rv - 0.8, '$^{3}$He', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center',
            zorder=7, **FONT)

    # the one that captures: left of centre and a little low, so the beam
    # reaches it and the label can sit outside on the same side
    target = np.array([cx - 4.5, cy - 1.8])
    hit_x, hit_y = sites[int(np.argmin(((sites - target) ** 2).sum(axis=1)))]

    # --- the beam: an irregular column of neutrons, up from below ---
    for x, y in neutron_sites(cx, 71.0, 82.0, 4.4):
        nucleus(ax, x, y, 0, 1, r=0.7, P=P, zorder=6)
        arrow(ax, (x, y + 1.05), (x, y + 2.8), P['neutron'], lw=1.1, ms=7,
              alpha=0.5, zorder=3)

    # one of them carries on into the gas and captures
    arrow(ax, (cx - 2.0, 82.8), (hit_x + 0.8, hit_y - 1.8), P['neutron'],
          lw=1.4, ms=9, alpha=0.85, zorder=5)
    for r, a in ((3.0, 0.20), (1.9, 0.40)):
        ax.add_patch(Circle((hit_x, hit_y), r, facecolor=P['x17'], alpha=a,
                            edgecolor='none', zorder=6))
    # label on the same side as the struck nucleus, so the leader does not have
    # to cross the volume
    ax.annotate('capture', xy=(hit_x - 2.4, hit_y - 0.2),
                xytext=(cx - rv - 0.8, cy - 1.8), fontsize=8.2,
                color=P['x17'], fontweight='bold', ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color=P['x17'], lw=0.9,
                                alpha=0.8), zorder=8, **FONT)

    ax.text(cx, 67.2, 'neutrons up from EAR2\nenergy from time of flight',
            fontsize=8.0, color=P['muted'], ha='center', va='center',
            linespacing=1.45, **FONT)


def _story_beam_capsule(ax, P):
    """The same beat once the target hardware *has* been introduced: the real
    vessel from the Geant4 geometry, with a zoom onto the gas."""
    x0, x1 = S_A
    # scale 0.40 (was 0.245): the vessel is the one object in the row that is
    # naturally VERTICAL, so it is what pays for the taller canvas
    xc, scale = 15.6, 0.40
    y_z0 = 90.4                       # world y of capsule z = 0

    # --- the beam, from below ---
    for dx in (-6.0, 0.0, 6.0):
        x = xc + dx
        nucleus(ax, x, 70.4, 0, 1, r=1.0, P=P, zorder=6)
        arrow(ax, (x, 71.9), (x, 74.2 + (1.2 if dx == 0 else 0.0)),
              P['neutron'], lw=1.4, ms=9, alpha=0.85, zorder=4)
    ax.text(xc, 67.2, 'neutrons, from EAR2 below', fontsize=8.0,
            color=P['muted'], ha='center', va='center', **FONT)

    # --- the capsule, as built ---
    draw_capsule(ax, xc, y_z0, scale, P)
    # leader labels live in a right-aligned column clear of the silhouette
    # anchor every leader on the NEAR edge, so no line is drawn across the
    # silhouette
    lab_x = xc - 6.4
    for text, anchor in (('valve', (xc - 4.2 * scale, y_z0 + 45.0 * scale)),
                         ('Al + CFRP', (xc - 11.2 * scale, y_z0 + 8.0 * scale)),
                         ('$^{3}$He, 500 bar', (xc - 9.4 * scale,
                                                y_z0 - 12.0 * scale))):
        ax.annotate(text, xy=anchor, xytext=(lab_x, anchor[1]), fontsize=7.8,
                    color=P['muted'], ha='right', va='center',
                    arrowprops=dict(arrowstyle='-', color=P['muted'], lw=0.8,
                                    alpha=0.65), zorder=8, **FONT)

    # --- zoom on the gas, where the capture happens ---
    spot = (xc + 2.4 * scale, y_z0 - 4.0 * scale)
    zc, zr = (29.6, 96.4), 8.8
    magnifier(ax, spot, 2.2, zc, zr, P)

    # one neutron meeting one 3He inside the bubble
    nucleus(ax, zc[0] - 4.5, zc[1] + 3.7, 0, 1, r=1.1, P=P, zorder=7)
    arrow(ax, (zc[0] - 3.8, zc[1] + 2.3), (zc[0] - 1.5, zc[1] + 0.5),
          P['neutron'], lw=1.4, ms=9, zorder=6)
    nucleus(ax, zc[0] + 1.8, zc[1] - 1.7, 2, 1, r=1.2, P=P, zorder=7)
    ax.text(zc[0] - 6.4, zc[1] + 5.1, 'n', fontsize=8.4, color=P['ink'],
            ha='center', va='center', zorder=7, **FONT)
    ax.text(zc[0] + 1.8, zc[1] - 5.2, '$^{3}$He', fontsize=8.4,
            color=P['ink'], ha='center', va='center', zorder=7, **FONT)

    ax.text(zc[0], 82.6, 'neutron energy\nfrom time of flight', fontsize=8.0,
            color=P['muted'], ha='center', va='center', linespacing=1.45,
            **FONT)


def _story_capture(ax, P):
    """Beat 2, read DOWNWARDS since 2026-08-18.

    It used to run left to right -- n + 3He -> 4He* -- which made the shortest
    beat of the five the second-widest, on a row where width is the scarce
    thing (see SW).  Stacked, the same three objects are drawn half again as
    large in a box two-thirds as wide, and the reaction arrow now points the
    way the beam actually travels.
    """
    x0, x1 = S_B
    xc = (x0 + x1) / 2
    _head(ax, x0, S_HEAD1, '2.  Capture makes $^{4}$He$^{*}$', P)

    # --- the entrance channel, side by side ---
    nucleus(ax, xc - 7.4, 107.0, 0, 1, r=1.9, P=P)
    ax.text(xc - 2.6, 106.8, '+', fontsize=15, color=P['muted'], ha='center',
            va='center', **FONT)
    nucleus(ax, xc + 4.6, 106.6, 2, 1, r=1.9, P=P)
    ax.text(xc - 7.4, 101.4, 'n', fontsize=10.5, color=P['ink'], ha='center',
            va='center', **FONT)
    ax.text(xc + 4.6, 101.4, '$^{3}$He', fontsize=10.5, color=P['ink'],
            ha='center', va='center', **FONT)

    # --- and down into the compound nucleus ---
    arrow(ax, (xc, 98.0), (xc, 92.0), P['muted'], lw=1.7, ms=13)

    excitation_waves(ax, xc, 86.6, P, r=1.9)
    nucleus(ax, xc, 86.6, 2, 2, r=1.9, P=P)
    ax.text(xc, 79.8, '$^{4}$He$^{*}$', fontsize=13, fontweight='bold',
            color=P['ink'], ha='center', va='center', **FONT)

    ax.text(xc, 72.8,
            'the compound nucleus is left\n'
            '20.58 MeV above its ground state',
            fontsize=8.6, color=P['muted'], ha='center', va='center',
            linespacing=1.5, **FONT)


def _story_levels(ax, P, halo):
    x0, x1 = S_C
    _head(ax, x0, S_HEAD1, '3.  Three ways to shed it', P)

    # a narrow ladder: the level lines only have to carry the 20.58 MeV drop,
    # and the width freed up goes to the three processes on the right
    lx0, lx1 = x0 + 2.0, x0 + 12.0
    y_hi, y_lo = 105.4, 84.0
    for y in (y_hi, y_lo):
        ax.plot([lx0, lx1], [y, y], color=P['ink'], lw=2.4,
                solid_capstyle='round', zorder=4)

    # the nucleus itself, above its level and below the other
    lxc = (lx0 + lx1) / 2
    excitation_waves(ax, lxc, y_hi + 4.2, P, r=1.15)
    nucleus(ax, lxc, y_hi + 4.2, 2, 2, r=1.15, P=P, zorder=6)
    ax.text(lxc + 5.0, y_hi + 4.2, '$^{4}$He$^{*}$', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)
    nucleus(ax, lxc, y_lo - 4.0, 2, 2, r=1.15, P=P, zorder=6)
    ax.text(lxc + 3.6, y_lo - 4.0, '$^{4}$He', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)

    arrow(ax, (lx0 + 2.0, y_hi - 0.6), (lx0 + 2.0, y_lo + 0.6), P['ink'],
          lw=1.5, style='<|-|>', ms=10, zorder=5)
    ax.text(lx0 + 3.4, (y_hi + y_lo) / 2, '20.58\nMeV', fontsize=10.0,
            fontweight='bold', color=P['ink'], ha='left', va='center',
            linespacing=1.35, path_effects=halo, zorder=6, **FONT)

    # --- the three channels, each drawn as what it actually emits ---
    ix = x1 - 33.0                       # left edge of the process pictures
    tx = ix + 12.0                       # where the wording starts
    chans = [(103.4, P['gamma'], 'gamma', r'$\gamma$  emission',
              'no pair to see'),
             (93.0, P['ipc'], 'ipc', 'internal pair conversion',
              'pair mass anywhere in 1–20 MeV'),
             (82.6, P['x17'], 'x17', 'X17 $\\rightarrow e^{+}e^{-}$',
              'one fixed mass, $\\approx$ 17 MeV')]
    # The two channels that put a pair in the detector, boxed together: that is
    # the whole experimental handle, and the one thing to take away from this
    # beat.  Drawn in the lepton colour rather than either channel's own, since
    # it is the pair that is being called out, not the process.
    bx0, bx1_, by0, by1 = 87.4, 122.4, 76.8, 98.4
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bx1_ - bx0, by1 - by0,
        boxstyle='round,pad=0,rounding_size=1.6', facecolor=P['lepton'],
        alpha=0.07, edgecolor='none', zorder=2))
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bx1_ - bx0, by1 - by0,
        boxstyle='round,pad=0,rounding_size=1.6', facecolor='none',
        edgecolor=P['lepton'], lw=1.5, alpha=0.85, zorder=2))
    ax.text((bx0 + bx1_) / 2, 71.8, 'Detect the e$^{+}$e$^{-}$ pair!',
            fontsize=13.5, fontweight='bold', color=P['lepton'],
            ha='center', va='center', zorder=6, **FONT)

    for y, col, kind, name, note in chans:
        arrow(ax, (lx1 + 1.2, y_hi - 0.4), (ix - 1.4, y), col, lw=1.3,
              rad=0.16, ms=10, alpha=0.85, zorder=3)
        if kind == 'gamma':
            squiggle(ax, ix, y, ix + 7.4, y, col, n_wave=4, amp=0.75, lw=1.7)
            ax.text(ix + 9.0, y, r'$\gamma$', fontsize=11, color=col,
                    ha='left', va='center', **FONT)
        elif kind == 'ipc':
            lepton_fork(ax, ix + 0.4, y, 7.4, 17.0, P, lw=1.6, fs=7.4)
        else:
            arrow(ax, (ix - 0.4, y), (ix + 4.2, y), col, lw=1.8,
                  ls=(0, (3.0, 1.9)), style='-', zorder=5)
            ax.text(ix + 1.9, y + 1.9, 'X17', fontsize=7.6, fontweight='bold',
                    color=col, ha='center', va='center', **FONT)
            lepton_fork(ax, ix + 4.2, y, 5.6, 38.0, P, lw=1.6, fs=7.4)
        ax.text(tx, y + 1.35, name, fontsize=9, fontweight='bold',
                color=P['ink'], ha='left', va='center', **FONT)
        ax.text(tx, y - 1.95, note, fontsize=7.8, color=P['muted'],
                ha='left', va='center', **FONT)


def opening_band(m_parent, e_tot=None, n=40001):
    """(lowest, highest) lab opening angle reachable for a parent of mass m.

    Two regimes, and the difference between them is the whole point of the
    boost panel:

      * parent slower than the leptons are in its rest frame (heavy parent):
        the backward lepton still goes backward in the lab, so the pair can
        reach 180 deg and is bounded BELOW at theta_min -- a hard lower edge.
      * parent faster (light parent): both leptons are swept forward into a
        cone, so the pair is bounded ABOVE and can close all the way to 0.

    The crossover is at m = sqrt(2 m_e E) ~ 4.6 MeV for E = 20.58 MeV.
    """
    e_tot = X17['e_capture'] if e_tot is None else e_tot
    m_e = X17['m_e']
    e_star = m_parent / 2.0
    p_star = np.sqrt(max(e_star ** 2 - m_e ** 2, 0.0))
    gamma = e_tot / m_parent
    beta_gamma = np.sqrt(max(gamma ** 2 - 1.0, 0.0))

    c = np.linspace(-1.0, 1.0, n)
    s = np.sqrt(np.clip(1.0 - c ** 2, 0.0, None))
    a = (np.arctan2(p_star * s, gamma * p_star * c + beta_gamma * e_star)
         + np.arctan2(p_star * s, -gamma * p_star * c + beta_gamma * e_star))
    a = np.degrees(a)
    return float(a.min()), float(a.max())


def lab_angles(m_parent, theta_star_deg, e_tot=None):
    """Where each lepton actually points in the lab, measured from the boost
    axis: ``(positron_deg, electron_deg)``, positive above the axis.

    The pair is back-to-back in the parent rest frame, but the boost does NOT
    treat the two halves alike, so the lab pair is only symmetric about the
    boost axis in the one case theta* = 90 deg.  Drawing every example
    symmetric -- which is what a fork at +/- theta/2 does -- gets the shape
    wrong everywhere else: at theta* = 0 the pair is collinear *along* the
    axis, one lepton forward and one backward, not two arms splayed about it.
    """
    e_tot = X17['e_capture'] if e_tot is None else e_tot
    m_e = X17['m_e']
    e_star = m_parent / 2.0
    p_star = np.sqrt(max(e_star ** 2 - m_e ** 2, 0.0))
    gamma = e_tot / m_parent
    beta_gamma = np.sqrt(max(gamma ** 2 - 1.0, 0.0))

    c = np.cos(np.radians(theta_star_deg))
    s = np.sin(np.radians(theta_star_deg))
    pt = p_star * s
    a = np.degrees(np.arctan2(pt, gamma * p_star * c + beta_gamma * e_star))
    b = np.degrees(np.arctan2(pt, -gamma * p_star * c + beta_gamma * e_star))
    return float(a), float(-b)


def opening_at(m_parent, theta_star_deg, e_tot=None):
    """Lab opening angle for a decay emitted at ``theta_star`` to the boost
    axis in the parent rest frame.  This is the per-orientation version of
    ``opening_band``, and it is what the worked examples in beat 4 show."""
    e_tot = X17['e_capture'] if e_tot is None else e_tot
    m_e = X17['m_e']
    e_star = m_parent / 2.0
    p_star = np.sqrt(max(e_star ** 2 - m_e ** 2, 0.0))
    gamma = e_tot / m_parent
    beta_gamma = np.sqrt(max(gamma ** 2 - 1.0, 0.0))
    c = np.cos(np.radians(theta_star_deg))
    s = np.sin(np.radians(theta_star_deg))
    return float(np.degrees(
        np.arctan2(p_star * s, gamma * p_star * c + beta_gamma * e_star)
        + np.arctan2(p_star * s, -gamma * p_star * c + beta_gamma * e_star)))


EXAMPLE_THETA_STAR = (90.0, 67.5, 45.0, 22.5, 0.0)


def _rest_frame_pair(ax, cx, cy, r, P, theta_star=52.0, lw=1.4, ms=9,
                     label=False, fs=7.6):
    """The pair as it leaves the parent: back-to-back, at ``theta_star`` to the
    boost axis."""
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=P['muted'],
                        lw=0.9, ls=(0, (2.4, 2.0)), alpha=0.8, zorder=3))
    a = np.radians(theta_star)
    for sgn, lab, col in ((+1, r'$e^+$', P['positron']),
                          (-1, r'$e^-$', P['electron'])):
        ex = cx + sgn * r * 0.84 * np.cos(a)
        ey = cy + sgn * r * 0.84 * np.sin(a)
        arrow(ax, (cx, cy), (ex, ey), col, lw=lw, ms=ms, zorder=4)
        if label:
            ax.text(ex + sgn * 0.9, ey + sgn * 0.7, lab, color=col,
                    fontsize=fs, ha='center', va='center', zorder=5, **FONT)


def _orientation_example(ax, xc, yc, m_parent, theta_star, col, P, halo,
                         arm=5.2, note=None):
    """One worked example: the decay direction in the rest frame, and where the
    boost actually puts the two leptons in the lab.

    Both arms are drawn at their own computed lab angle, so the picture is only
    symmetric where the physics is.
    """
    theta_lab = opening_at(m_parent, theta_star)
    a_pos, a_ele = lab_angles(m_parent, theta_star)

    # --- rest frame: the pair, back-to-back along theta_star ---
    # theta* is labelled UNDER the icon rather than beside it: with five
    # examples in the row, a side label is what sets the column pitch
    icy = yc + 6.8
    _rest_frame_pair(ax, xc, icy, 2.0, P, theta_star=theta_star, lw=1.3, ms=7)
    ax.text(xc, yc + 3.2, f'$\\theta^{{*}}$ = {theta_star:g}°', fontsize=7.8,
            color=P['muted'], ha='center', va='center', **FONT)

    # --- lab: each lepton at its own angle to the boost axis ---
    vx, vy = xc - 3.8, yc - 3.2
    for ang, lcol in ((a_pos, P['positron']), (a_ele, P['electron'])):
        t = np.radians(ang)
        arrow(ax, (vx, vy), (vx + arm * np.cos(t), vy + arm * np.sin(t)),
              lcol, lw=1.9, ms=10, zorder=4)
    if theta_lab > 4.0:
        t = np.linspace(np.radians(a_ele), np.radians(a_pos), 60)
        ax.plot(vx + 2.4 * np.cos(t), vy + 2.4 * np.sin(t), color=col, lw=1.0,
                alpha=0.9, zorder=4)
    # +0.7 rather than +1.4 off the arm: on the tighter column pitch the number
    # has to clear the NEXT column's backward arm, which is what it would run
    # into, not its own fork
    # The number sits ABOVE the vertex line, not on it (2026-08-18).  At pitch
    # 15 the NEXT column's backward arm -- the theta* = 0 case, where an X17
    # pair really is back-to-back -- comes back along vy and arrives exactly
    # where this column's number was.  Lifting it 2.2 units clears that arm
    # without costing any width, and the row has the height to spend.
    ax.text(vx + arm + 0.8, vy + 2.2, f'{theta_lab:.0f}°',
            fontsize=10.4, fontweight='bold', color=col, ha='left',
            va='center', path_effects=halo, zorder=6, **FONT)
    # The note is CENTRED UNDER ITS OWN COLUMN, not hung off the number: it
    # only ever lands on the last column, and "back-to-back" set flush left at
    # pitch 15 runs straight into the spectrum panel's left spine.
    if note:
        ax.text(xc, vy - 5.6, note, fontsize=7.6, color=P['muted'],
                ha='center', va='center', path_effects=halo, zorder=6, **FONT)


def _boost_row(ax, x0, yc, m_parent, tag, col, P, halo=None):
    """One channel: how hard its parent is boosted, then three worked
    orientations.

    The arrow length is the parent's beta -- the X17 arrow is visibly stubby
    next to the IPC one, which is the entire mechanism in one glance.
    """
    e_tot = X17['e_capture']
    gamma = e_tot / m_parent
    beta = np.sqrt(max(1.0 - 1.0 / gamma ** 2, 0.0))

    ax.text(x0, yc + 12.4, tag, fontsize=9.8, fontweight='bold', color=col,
            ha='left', va='center', **FONT)

    # THE LEFT BLOCK IS STACKED, NOT INLINE (2026-08-18).  It used to run
    # rest-frame icon -> boost arrow -> five orientation columns across one
    # line, and those first two items ate 21.5 of the row's ~82 units of width.
    # This row is WIDTH-LIMITED like every other figure in the deck, so those
    # units were the whole budget for making the columns bigger: stacking the
    # icon over the arrow costs vertical space the row now has (the summary
    # paragraph under it came off the same day) and returns ~9 units of width,
    # which is what pays for pitch 12.6 -> 15.0 and everything drawn at 1.19x.
    _rest_frame_pair(ax, x0 + 5.0, yc + 5.4, 4.2, P, label=True)
    ax.text(x0 + 5.0, yc - 0.4, 'rest frame', fontsize=7.8, color=P['muted'],
            ha='center', va='center', **FONT)

    # --- the boost, as an arrow whose length is beta ---
    bx0, bmax = x0 + 0.6, 9.0
    arrow(ax, (bx0, yc - 6.2), (bx0 + bmax * beta, yc - 6.2), col, lw=2.8,
          ms=15, zorder=4)
    # 2 dp reads as a flat 1.00 once the parent is ultra-relativistic, which is
    # exactly the regime the row is about
    bstr = f'{beta:.2f}' if beta < 0.99 else f'{beta:.3f}'
    ax.text(bx0, yc - 3.6, 'boost', fontsize=7.4, color=P['muted'],
            ha='left', va='center', **FONT)
    ax.text(bx0, yc - 9.8, f'$\\beta$ = {bstr}   $\\gamma$ = {gamma:.1f}',
            fontsize=8.6, color=col, ha='left', va='center',
            fontweight='bold', **FONT)

    last = 'collinear' if m_parent < 4.6 else 'back-to-back'
    notes = [None] * (len(EXAMPLE_THETA_STAR) - 1) + [last]
    for i, ts in enumerate(EXAMPLE_THETA_STAR):
        # pitch 15.0, not less: at theta* = 0 the X17 pair is back-to-back, so
        # ONE ARM POINTS BACKWARDS 5.2 units from its vertex and would be drawn
        # through the previous column's angle number
        _orientation_example(ax, x0 + 13.0 + i * 15.0, yc, m_parent, ts, col,
                             P, halo, note=notes[i])
    return opening_band(m_parent)


def _story_mechanism(fig, ax, P, halo):
    """Beat 4, all visual: what the boost does, and what that leaves on the
    opening-angle axis.

    The left column is one cartoon run twice with only the parent mass changed.
    The right column carries the consequence onto the same 0-180 deg axis panel
    5 uses, so the reader arrives at the last panel already knowing what the
    two curves have to look like.
    """
    x0, x1 = S_D
    _head(ax, x0, S_HEAD2, '4.  The boost is what makes the difference', P)

    # NO SUBTITLE AND NO SUMMARY PARAGRAPH since 2026-08-18, on Dylan's call
    # ("keep the 4. and 5. titles, but remove the 'in the rest frame ...' and
    # 'whatever the orientation ...'  -- can add it on the html later if
    # needed").  Both said in words what the drawing under them shows: the
    # rest-frame icon IS the pair leaving back-to-back, and the five angle
    # numbers per row ARE the bound.  They cost ~14 canvas units of height,
    # which on a width-limited figure is 14 units the pictures now have.
    m_ipc = 2.0
    _boost_row(ax, x0 + 1.0, 45.0, X17['m_x17'],
               f'X17  —  one mass, {X17["m_x17"]:g} MeV,  heavy and slow',
               P['x17'], P, halo=halo)
    _boost_row(ax, x0 + 1.0, 17.0, m_ipc,
               f'IPC  —  any mass, here {m_ipc:g} MeV,  light and fast',
               P['ipc'], P, halo=halo)


# Beat 4's box, with the micro-TPC cartoon standing in it instead of the boost
# rows -- deck frame 6.3 (2026-08-18, Dylan: "for 6.3, remove the left diagram
# with the angles and replace it with the MMs, keeping the spectrum in place").
#
# It is the SAME drawing as draw_detect, at 0.78 of its length scale, placed on
# the story canvas: _utpc_arm grew a ``scale`` argument rather than a second
# implementation, so the 110 deg is still the kinematic minimum drawn true and
# still measures 110 with a protractor on the slide.  What this buys the frame
# is that the spectrum does not move and does not resize between 6.2 and 6.3 --
# only the argument beside it changes.  Frame 3 used to stack the cartoon UNDER
# the spectrum in the slide's figure box, which cost both pictures ~41 % of
# their width.
DETECT_INSET_SCALE = 0.87


def _story_detect(ax, P, halo):
    x0, x1 = S_D
    _head(ax, x0, S_HEAD2, '4.  ...and this is what measures it', P)

    sc = DETECT_INSET_SCALE
    vx, vy = (x0 + x1) / 2.0 + 2.0, 15.0
    half = DETECT_OPENING_DEG / 2.0
    ang_e, ang_p = 90.0 + half, 90.0 - half      # symmetric about vertical
    _utpc_arm(ax, (vx, vy), ang_e, P['electron'], P, halo, flip=True, scale=sc)
    _utpc_arm(ax, (vx, vy), ang_p, P['positron'], P, halo, scale=sc)

    ax.plot([vx], [vy], marker='o', ms=7, color=P['gamma'], mec=P['ink'],
            mew=0.8, zorder=8)
    r = 14.5
    t = np.radians(np.linspace(ang_p, ang_e, 100))
    ax.plot(vx + r * np.cos(t), vy + r * np.sin(t), color=P['ink'], lw=1.2,
            ls='--', alpha=0.75, zorder=6)
    ax.text(vx, vy + r + 2.0, f'θ = {DETECT_OPENING_DEG:.0f}°', fontsize=12.5,
            fontweight='bold', color=P['ink'], ha='center', va='bottom',
            path_effects=halo, zorder=9, **FONT)
    ax.text(vx, vy - 2.8, 'e$^{+}$e$^{-}$ from $^{4}$He*', fontsize=9.0,
            color=P['muted'], ha='center', va='top', **FONT)
    ax.text(vx - 25.0, vy + 3.4, 'e$^{-}$', fontsize=12, fontweight='bold',
            color=P['electron'], ha='center', va='center', path_effects=halo,
            zorder=9, **FONT)
    ax.text(vx + 25.0, vy + 3.4, 'e$^{+}$', fontsize=12, fontweight='bold',
            color=P['positron'], ha='center', va='center', path_effects=halo,
            zorder=9, **FONT)
    # the object, named once, in the corner of the box the drawing leaves empty
    ax.text(x0, S_HEAD2 - 6.0, 'Micromegas µTPC\n30 mm drift gap',
            fontsize=9.4, color=P['muted'], ha='left', va='top',
            linespacing=1.35, **FONT)


# The spectrum panel is a STACK: a small X17 yield sitting on top of the IPC
# background (2026-08-17, Dylan's call).  It used to be the two channels
# overlaid, each normalised to unit peak, which is the honest way to compare two
# *shapes* -- but it is not what a measurement looks like, and the slide's job
# is to show what we will be staring at: a smooth background with a bump on it.
#
# Stacking forces the figure to state a ratio it does not know -- the relative
# rate is exactly what the experiment is trying to measure.  So the ratio is a
# free, declared parameter: SIG_FRAC is the X17 yield as a fraction of the IPC
# yield in the plotted window, it is printed on the panel in words, and it is
# NOT a prediction.  0.04 puts the bump ~80 % above the local background at the
# peak: legible from the back of a room, and still visibly a bump ON something
# rather than a peak of its own.  Chosen by eye at 2 % / 4 % / 6 %.
#
# The window starts at SPEC_XLIM[0] rather than 0 for the same reason ATOMKI
# plot from 40 deg: the IPC forward peak is eight times the yield at 109 deg, so
# including it flattens everything the panel is about.  The forward sweep is
# already beat 4's argument, drawn there as kinematics -- it does not need to be
# argued twice, once as a spectrum the audience cannot read.
SIG_FRAC = 0.04
SPEC_XLIM = (40.0, 180.0)


def _story_measure(fig, ax, P, halo, y0=0.0, y1=136.0, x0f=0.0, x1f=SW):
    x0, x1 = S_E
    _head(ax, x0, S_HEAD2, '5.  So this is what we look for', P)

    th, x17, ipc = modelled_shapes()
    th_min = opening_angle_pdf()[2]
    # both arrive normalised to unit peak; SIG_FRAC is defined on the integrals
    # over the plotted window, so the number quoted on the panel is the one the
    # eye is actually being shown
    win = (th >= SPEC_XLIM[0]) & (th <= SPEC_XLIM[1])
    sig = x17 * (SIG_FRAC * ipc[win].sum() / x17[win].sum())
    tot = ipc + sig

    # The canvas may be a cropped band -- vertically for the split slides, and
    # since 2026-08-16 horizontally too for the one-beat-per-file renders --
    # so the axes is placed against the window rather than against a fixed
    # 160x90 page.  The defaults are the full page, i.e. what it always did.
    # 34 x 26 at y = 24 until 2026-08-18, when the two paragraphs under it came
    # off the figure (Dylan: "remove the text 'The edge at ...' and 'X17 drawn
    # at ...'").  The panel took the height they were using: same width, 26 ->
    # 38 tall, starting 12 units off the floor of the band.
    span, xspan = y1 - y0, x1f - x0f
    px = fig.add_axes([(x0 + 4.0 - x0f) / xspan, (12.0 - y0) / span,
                       34.0 / xspan, 38.0 / span], facecolor='none')
    for s in ('top', 'right'):
        px.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        px.spines[s].set_color(P['muted'])
        px.spines[s].set_linewidth(0.9)
    px.tick_params(colors=P['muted'], labelsize=8.6, width=0.9, length=3)
    for lab in px.get_xticklabels() + px.get_yticklabels():
        lab.set_fontfamily('DejaVu Sans')

    # background first, then the signal band on top of it: the filled area
    # between the two curves IS the excess, which is the whole point
    px.fill_between(th, 0, ipc, color=P['ipc'], alpha=0.14, lw=0, zorder=2)
    px.fill_between(th, ipc, tot, color=P['x17'], alpha=0.34, lw=0, zorder=3)
    px.plot(th, ipc, color=P['ipc'], lw=1.8, zorder=4,
            label='internal pair conversion')
    px.plot(th, tot, color=P['x17'], lw=2.2, zorder=5,
            label='+ X17 $\\rightarrow e^{+}e^{-}$')
    px.axvline(th_min, color=P['x17'], lw=0.9, ls=':', alpha=0.8, zorder=1)

    px.set_xlim(*SPEC_XLIM)
    px.set_ylim(0, 1.16 * tot[win].max())
    px.set_xticks([45, 90, 135, 180])
    px.set_yticks([])
    px.set_xlabel('e$^{+}$e$^{-}$ opening angle  (deg)', fontsize=9.0,
                  color=P['muted'], labelpad=2, **FONT)
    px.set_ylabel('yield  (arb.)', fontsize=9.0, color=P['muted'], labelpad=3,
                  **FONT)
    leg = px.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.16),
                    frameon=False, fontsize=8.6, handlelength=1.8,
                    labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(P['muted'])
        t.set_fontfamily('DejaVu Sans')


def _story_footer(ax, P, y=2.2, cap=None):
    cap = cap or ('Panel 5 samples the MX17_Simulation generators (X17PhysicsSpectrum, '
           'IPCPhysicsSpectrum) that track the Geant4 X17PrimaryGenerator: '
           '%s events per channel, smeared %.0f°, recoil neglected. The X17 '
           'yield is stacked on the IPC background at %.0f %% of it over the '
           'plotted range — a drawn ratio, not a predicted one: the relative '
           'rate is what the experiment measures. In panel 4 the boost arrow '
           'lengths (β) and the opening angles are to scale; lepton arm '
           'lengths are not.'
           % (f'{SAMPLE_N:,}'.replace(',', ' '), X17['smear_deg'],
              SIG_FRAC * 100))
    ax.text(3, y, textwrap.fill(cap, 132), fontsize=7.2, color=P['muted'],
            ha='left', va='center', linespacing=1.65, **FONT)
    ax.text(SW - 3, y, SOURCES, fontsize=7.2, color=P['muted'], ha='right',
            va='center', linespacing=1.65, **FONT)


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
