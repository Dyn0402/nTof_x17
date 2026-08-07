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
S_ROW1 = (47.0, 74.0)       # (bottom, top) of the upper row
S_ROW2 = (7.0, 40.0)
S_HEAD1, S_HEAD2 = 76.6, 42.6
S_A = (8.0, 46.0)           # 1 beam on target
S_B = (52.0, 90.0)          # 2 capture
S_C = (96.0, 152.0)         # 3 de-excitation
S_D = (8.0, 100.0)          # 4 why the pair opens the way it does
S_E = (104.0, 152.0)        # 5 what we measure


def draw_story(theme='light', dpi=300, title=True, capsule=False):
    """The five-beat layout.  Same numbers, same palette, more room."""
    P = palette(theme)
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig = plt.figure(figsize=(W / 10.0, H / 10.0), dpi=dpi, facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = [pe.withStroke(linewidth=2.4, foreground=P['halo'], alpha=0.85)]

    if title:
        ax.text(8, 85.4, 'How a 17 MeV boson would show up in n + $^{3}$He',
                fontsize=19.5, fontweight='bold', color=P['ink'],
                ha='left', va='center', **FONT)
        ax.text(8, 80.6,
                'The pair opening angle is set by the mass of whatever emitted '
                'it — which is why a single new mass would put a hard edge in '
                'a smooth background.',
                fontsize=10, color=P['muted'], ha='left', va='center', **FONT)
        ax.plot([8, 152], [78.2, 78.2], color=P['rule'], lw=1.0, zorder=1)

    _story_beam(ax, P, capsule=capsule)
    _story_capture(ax, P)
    _story_levels(ax, P, halo)
    _story_mechanism(fig, ax, P, halo)
    _story_measure(fig, ax, P, halo)
    _story_footer(ax, P)
    return fig


def _head(ax, x, y, text, P):
    ax.text(x, y, text, fontsize=10.5, fontweight='bold', color=P['ink'],
            ha='left', va='center', **FONT)


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
    cx, cy, rv = (x0 + x1) / 2, 65.8, 8.4
    ax.add_patch(Circle((cx, cy), rv, facecolor=S.COL['gas'], alpha=0.14,
                        edgecolor=P['rule'], lw=1.0, zorder=2))
    sites = he3_sites(cx, cy, rv - 1.5)
    for x, y in sites:
        nucleus(ax, x, y, 2, 1, r=0.5, P=P, zorder=6)
    ax.text(cx + rv - 1.0, cy + rv - 0.6, '$^{3}$He', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center',
            zorder=7, **FONT)

    # the one that captures: left of centre and a little low, so the beam
    # reaches it and the label can sit outside on the same side
    target = np.array([cx - 3.4, cy - 1.4])
    hit_x, hit_y = sites[int(np.argmin(((sites - target) ** 2).sum(axis=1)))]

    # --- the beam: an irregular column of neutrons, up from below ---
    for x, y in neutron_sites(cx, 46.9, 56.0, 3.4):
        nucleus(ax, x, y, 0, 1, r=0.55, P=P, zorder=6)
        arrow(ax, (x, y + 0.85), (x, y + 2.3), P['neutron'], lw=1.1, ms=7,
              alpha=0.5, zorder=3)

    # one of them carries on into the gas and captures
    arrow(ax, (cx - 1.6, 56.6), (hit_x + 0.6, hit_y - 1.4), P['neutron'],
          lw=1.4, ms=9, alpha=0.85, zorder=5)
    for r, a in ((2.4, 0.20), (1.5, 0.40)):
        ax.add_patch(Circle((hit_x, hit_y), r, facecolor=P['x17'], alpha=a,
                            edgecolor='none', zorder=6))
    # label on the same side as the struck nucleus, so the leader does not have
    # to cross the volume
    ax.annotate('capture', xy=(hit_x - 2.0, hit_y - 0.2),
                xytext=(cx - rv - 0.8, cy - 1.4), fontsize=8.2,
                color=P['x17'], fontweight='bold', ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color=P['x17'], lw=0.9,
                                alpha=0.8), zorder=8, **FONT)

    ax.text(cx, 45.4, 'neutrons up from EAR2 — energy from time of flight',
            fontsize=8.0, color=P['muted'], ha='center', va='center', **FONT)


def _story_beam_capsule(ax, P):
    """The same beat once the target hardware *has* been introduced: the real
    vessel from the Geant4 geometry, with a zoom onto the gas."""
    x0, x1 = S_A
    xc, scale = 19.6, 0.245
    y_z0 = 52.6 + 35.0 * scale        # world y of capsule z = 0

    # --- the beam, from below ---
    for dx in (-4.8, 0.0, 4.8):
        x = xc + dx
        nucleus(ax, x, 48.6, 0, 1, r=0.85, P=P, zorder=6)
        arrow(ax, (x, 49.9), (x, 51.7 + (1.0 if dx == 0 else 0.0)),
              P['neutron'], lw=1.4, ms=9, alpha=0.85, zorder=4)
    ax.text(xc, 45.9, 'neutrons, from EAR2 below', fontsize=8.0,
            color=P['muted'], ha='center', va='center', **FONT)

    # --- the capsule, as built ---
    draw_capsule(ax, xc, y_z0, scale, P)
    # leader labels live in a right-aligned column clear of the silhouette
    # anchor every leader on the NEAR edge, so no line is drawn across the
    # silhouette
    lab_x = xc - 3.6
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
    zc, zr = (36.4, 62.2), 7.6
    magnifier(ax, spot, 1.7, zc, zr, P)

    # one neutron meeting one 3He inside the bubble
    nucleus(ax, zc[0] - 3.9, zc[1] + 3.2, 0, 1, r=0.95, P=P, zorder=7)
    arrow(ax, (zc[0] - 3.3, zc[1] + 2.0), (zc[0] - 1.3, zc[1] + 0.4),
          P['neutron'], lw=1.4, ms=9, zorder=6)
    nucleus(ax, zc[0] + 1.6, zc[1] - 1.5, 2, 1, r=1.05, P=P, zorder=7)
    ax.text(zc[0] - 5.6, zc[1] + 4.4, 'n', fontsize=8.4, color=P['ink'],
            ha='center', va='center', zorder=7, **FONT)
    ax.text(zc[0] + 1.6, zc[1] - 4.6, '$^{3}$He', fontsize=8.4,
            color=P['ink'], ha='center', va='center', zorder=7, **FONT)

    ax.text(zc[0], 51.0, 'neutron energy from\ntime of flight', fontsize=8.0,
            color=P['muted'], ha='center', va='center', linespacing=1.45,
            **FONT)


def _story_capture(ax, P):
    x0, x1 = S_B
    _head(ax, x0, S_HEAD1, '2.  Capture makes $^{4}$He$^{*}$', P)

    nucleus(ax, x0 + 5.0, 65.0, 0, 1, P=P)
    ax.text(x0 + 9.0, 64.8, '+', fontsize=12, color=P['muted'], ha='center',
            va='center', **FONT)
    nucleus(ax, x0 + 13.6, 64.6, 2, 1, P=P)
    arrow(ax, (x0 + 18.6, 64.6), (x0 + 24.4, 64.6), P['muted'], lw=1.5, ms=11)
    excitation_waves(ax, x0 + 31.0, 64.6, P)
    nucleus(ax, x0 + 31.0, 64.6, 2, 2, P=P)

    ax.text(x0 + 5.0, 60.4, 'n', fontsize=9.5, color=P['ink'], ha='center',
            va='center', **FONT)
    ax.text(x0 + 13.6, 60.4, '$^{3}$He', fontsize=9.5, color=P['ink'],
            ha='center', va='center', **FONT)
    ax.text(x0 + 31.0, 59.6, '$^{4}$He$^{*}$', fontsize=11, fontweight='bold',
            color=P['ink'], ha='center', va='center', **FONT)
    ax.text(x0 + 19.0, 53.4,
            'the compound nucleus is left\n'
            '20.58 MeV above its ground state',
            fontsize=8.6, color=P['muted'], ha='center', va='center',
            linespacing=1.5, **FONT)


def _story_levels(ax, P, halo):
    x0, x1 = S_C
    _head(ax, x0, S_HEAD1, '3.  Three ways to shed it', P)

    # a narrow ladder: the level lines only have to carry the 20.58 MeV drop,
    # and the width freed up goes to the three processes on the right
    lx0, lx1 = x0 + 3.0, x0 + 16.0
    y_hi, y_lo = 68.2, 52.8
    for y in (y_hi, y_lo):
        ax.plot([lx0, lx1], [y, y], color=P['ink'], lw=2.4,
                solid_capstyle='round', zorder=4)

    # the nucleus itself, above its level and below the other
    lxc = (lx0 + lx1) / 2
    excitation_waves(ax, lxc, y_hi + 3.6, P, r=0.95)
    nucleus(ax, lxc, y_hi + 3.6, 2, 2, r=0.95, P=P, zorder=6)
    ax.text(lxc + 4.4, y_hi + 3.6, '$^{4}$He$^{*}$', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)
    nucleus(ax, lxc, y_lo - 3.4, 2, 2, r=0.95, P=P, zorder=6)
    ax.text(lxc + 3.0, y_lo - 3.4, '$^{4}$He', fontsize=9.5,
            fontweight='bold', color=P['ink'], ha='left', va='center', **FONT)

    arrow(ax, (lx0 + 2.4, y_hi - 0.6), (lx0 + 2.4, y_lo + 0.6), P['ink'],
          lw=1.5, style='<|-|>', ms=10, zorder=5)
    ax.text(lx0 + 3.8, (y_hi + y_lo) / 2, '20.58\nMeV', fontsize=10.0,
            fontweight='bold', color=P['ink'], ha='left', va='center',
            linespacing=1.35, path_effects=halo, zorder=6, **FONT)

    # --- the three channels, each drawn as what it actually emits ---
    ix = x1 - 33.0                       # left edge of the process pictures
    tx = ix + 12.0                       # where the wording starts
    chans = [(67.8, P['gamma'], 'gamma', r'$\gamma$  emission',
              'no pair to see'),
             (60.5, P['ipc'], 'ipc', 'internal pair conversion',
              'pair mass anywhere in 1–20 MeV'),
             (53.2, P['x17'], 'x17', 'X17 $\\rightarrow e^{+}e^{-}$',
              'one fixed mass, $\\approx$ 17 MeV')]
    # The two channels that put a pair in the detector, boxed together: that is
    # the whole experimental handle, and the one thing to take away from this
    # beat.  Drawn in the lepton colour rather than either channel's own, since
    # it is the pair that is being called out, not the process.
    bx0, bx1_, by0, by1 = 117.8, 150.6, 48.4, 64.2
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bx1_ - bx0, by1 - by0,
        boxstyle='round,pad=0,rounding_size=1.6', facecolor=P['lepton'],
        alpha=0.07, edgecolor='none', zorder=2))
    ax.add_patch(FancyBboxPatch(
        (bx0, by0), bx1_ - bx0, by1 - by0,
        boxstyle='round,pad=0,rounding_size=1.6', facecolor='none',
        edgecolor=P['lepton'], lw=1.5, alpha=0.85, zorder=2))
    ax.text((bx0 + bx1_) / 2, 45.9, 'Detect the e$^{+}$e$^{-}$ pair!',
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
            lepton_fork(ax, ix + 0.4, y, 7.0, 15.0, P, lw=1.6, fs=7.4)
        else:
            arrow(ax, (ix - 0.4, y), (ix + 4.2, y), col, lw=1.8,
                  ls=(0, (3.0, 1.9)), style='-', zorder=5)
            ax.text(ix + 1.9, y + 1.7, 'X17', fontsize=7.6, fontweight='bold',
                    color=col, ha='center', va='center', **FONT)
            # a shade narrower than it looks in beat 4: the rows are only 7
            # apart and the arms have to clear the IPC labels above
            lepton_fork(ax, ix + 4.2, y, 5.0, 38.0, P, lw=1.6, fs=7.4)
        ax.text(tx, y + 1.15, name, fontsize=9, fontweight='bold',
                color=P['ink'], ha='left', va='center', **FONT)
        ax.text(tx, y - 1.75, note, fontsize=7.8, color=P['muted'],
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
                         arm=4.4, note=None):
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
    icy = yc + 5.4
    _rest_frame_pair(ax, xc, icy, 1.6, P, theta_star=theta_star, lw=1.1, ms=6)
    ax.text(xc, yc + 2.6, f'$\\theta^{{*}}$ = {theta_star:g}°', fontsize=6.9,
            color=P['muted'], ha='center', va='center', **FONT)

    # --- lab: each lepton at its own angle to the boost axis ---
    vx, vy = xc - 3.2, yc - 2.6
    for ang, lcol in ((a_pos, P['positron']), (a_ele, P['electron'])):
        t = np.radians(ang)
        arrow(ax, (vx, vy), (vx + arm * np.cos(t), vy + arm * np.sin(t)),
              lcol, lw=1.6, ms=9, zorder=4)
    if theta_lab > 4.0:
        t = np.linspace(np.radians(a_ele), np.radians(a_pos), 60)
        ax.plot(vx + 2.0 * np.cos(t), vy + 2.0 * np.sin(t), color=col, lw=0.9,
                alpha=0.9, zorder=4)
    ax.text(vx + arm + 1.4, vy + (0.9 if note else 0.0), f'{theta_lab:.0f}°',
            fontsize=9.0, fontweight='bold', color=col, ha='left',
            va='center', path_effects=halo, zorder=6, **FONT)
    # the note belongs to the angle, so it sits under the number rather than
    # floating at the bottom of the row
    if note:
        ax.text(vx + arm + 1.4, vy - 1.4, note, fontsize=6.9,
                color=P['muted'], ha='left', va='center',
                path_effects=halo, zorder=6, **FONT)


def _boost_row(ax, x0, yc, m_parent, tag, col, P, halo=None):
    """One channel: how hard its parent is boosted, then three worked
    orientations.

    The arrow length is the parent's beta -- the X17 arrow is visibly stubby
    next to the IPC one, which is the entire mechanism in one glance.
    """
    e_tot = X17['e_capture']
    gamma = e_tot / m_parent
    beta = np.sqrt(max(1.0 - 1.0 / gamma ** 2, 0.0))

    ax.text(x0, yc + 8.6, tag, fontsize=8.8, fontweight='bold', color=col,
            ha='left', va='center', **FONT)

    # --- the pair as it leaves the parent, same for both rows ---
    _rest_frame_pair(ax, x0 + 5.0, yc, 4.2, P, label=True)
    ax.text(x0 + 5.0, yc - 5.8, 'rest frame', fontsize=7.4, color=P['muted'],
            ha='center', va='center', **FONT)

    # --- the boost, as an arrow whose length is beta ---
    bx0, bmax = x0 + 11.5, 9.5
    arrow(ax, (bx0, yc), (bx0 + bmax * beta, yc), col, lw=2.8, ms=15, zorder=4)
    # 2 dp reads as a flat 1.00 once the parent is ultra-relativistic, which is
    # exactly the regime the row is about
    bstr = f'{beta:.2f}' if beta < 0.99 else f'{beta:.3f}'
    ax.text(bx0, yc + 3.0, f'$\\beta$ = {bstr},   $\\gamma$ = {gamma:.1f}',
            fontsize=8.0, color=col, ha='left', va='center',
            fontweight='bold', **FONT)
    ax.text(bx0, yc - 3.2, 'boost', fontsize=7.2, color=P['muted'],
            ha='left', va='center', **FONT)

    last = 'collinear' if m_parent < 4.6 else 'back-to-back'
    notes = [None] * (len(EXAMPLE_THETA_STAR) - 1) + [last]
    for i, ts in enumerate(EXAMPLE_THETA_STAR):
        _orientation_example(ax, x0 + 27.0 + i * 15.0, yc, m_parent, ts, col,
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
    ax.text(x0 + 48.0, S_HEAD2,
            'in the rest frame the pair is always back-to-back',
            fontsize=8.0, color=P['muted'], ha='left', va='center', **FONT)

    m_ipc = 2.0
    lo_x, hi_x = _boost_row(ax, x0 + 1.0, 31.8, X17['m_x17'],
                            f'X17  —  one mass, {X17["m_x17"]:g} MeV,  '
                            'heavy and slow', P['x17'], P, halo=halo)
    lo_i, hi_i = _boost_row(ax, x0 + 1.0, 13.8, m_ipc,
                            f'IPC  —  any mass, here {m_ipc:g} MeV,  '
                            'light and fast', P['ipc'], P, halo=halo)

    ax.text(x0 + 1.0, 5.4,
            f'Whatever the orientation, X17 stays open — never below '
            f'{lo_x:.0f}°.  A light IPC pair is swept forward — never above '
            f'{hi_i:.0f}°.',
            fontsize=8.0, color=P['ink'], ha='left', va='center', **FONT)


def _story_measure(fig, ax, P, halo):
    x0, x1 = S_E
    _head(ax, x0, S_HEAD2, '5.  So this is what we look for', P)

    th, x17, ipc = modelled_shapes()
    th_min = opening_angle_pdf()[2]

    px = fig.add_axes([(x0 + 6.0) / W, 13.6 / H, 39.0 / W, 21.4 / H],
                      facecolor='none')
    for s in ('top', 'right'):
        px.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        px.spines[s].set_color(P['muted'])
        px.spines[s].set_linewidth(0.9)
    px.tick_params(colors=P['muted'], labelsize=7.6, width=0.9, length=3)
    for lab in px.get_xticklabels() + px.get_yticklabels():
        lab.set_fontfamily('DejaVu Sans')

    px.fill_between(th, 0, x17, color=P['x17'], alpha=0.16, lw=0, zorder=2)
    px.plot(th, x17, color=P['x17'], lw=2.2, zorder=4,
            label='X17 $\\rightarrow e^{+}e^{-}$')
    px.plot(th, ipc, color=P['ipc'], lw=2.0, zorder=3,
            label='internal pair conversion')
    px.axvline(th_min, color=P['x17'], lw=0.9, ls=':', alpha=0.8, zorder=1)

    px.set_xlim(0, 180)
    px.set_ylim(0, 1.16)
    px.set_xticks([0, 45, 90, 135, 180])
    px.set_yticks([])
    px.set_xlabel('e$^{+}$e$^{-}$ opening angle  (deg)', fontsize=8.0,
                  color=P['muted'], labelpad=2, **FONT)
    px.set_ylabel('yield  (arb.)', fontsize=8.0, color=P['muted'], labelpad=3,
                  **FONT)
    leg = px.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.24),
                    frameon=False, fontsize=7.8, handlelength=1.8,
                    labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(P['muted'])
        t.set_fontfamily('DejaVu Sans')

    ax.text(x0 + 6.0, 6.6,
            f'The edge at {th_min:.0f}° is the measurement — IPC has only a '
            'slope there.',
            fontsize=8.0, color=P['ink'], ha='left', va='center', **FONT)


def _story_footer(ax, P):
    cap = ('Panel 5 samples the MX17_Simulation generators (X17PhysicsSpectrum, '
           'IPCPhysicsSpectrum) that track the Geant4 X17PrimaryGenerator: '
           '%s events per channel, smeared %.0f°, recoil neglected, each curve '
           'normalised to unit peak — their relative rate is the measurement, '
           'so nothing here implies it. In panel 4 the boost arrow lengths '
           '(β) and the opening angles are to scale; lepton arm lengths are '
           'not.' % (f'{SAMPLE_N:,}'.replace(',', ' '), X17['smear_deg']))
    ax.text(8, 2.2, textwrap.fill(cap, 152), fontsize=7.2, color=P['muted'],
            ha='left', va='center', linespacing=1.65, **FONT)
    ax.text(152, 2.2, SOURCES, fontsize=7.2, color=P['muted'], ha='right',
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
