#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_conclusion_figure.py -- the Athens Conclusions slide's figure.

    python make_conclusion_figure.py            # -> figures/al_pair_outlook.{png,pdf}

The Athens counterpart of ``mpgd26/scenes_x17.py``'s ``draw_outlook`` (the
MPGD Summary figure: find the two-track events -> histogram the opening angle,
with an X17 bump drawn on).  By Athens we know the recorded thermal data holds
no X17 and no visible 3He IPC, so the spectrum panel no longer promises a bump.
It shows what the pair search on these data is actually going to look at: the
27Al-capture e+e- pairs, split into the three chamber topologies of the pair-
topology slide (intra / perpendicular / opposing, ``make_topology_figures.py``).

WHAT IS COMPUTED AND WHAT IS DRAWN.

  computed  the topology acceptance: which pairs of a given opening angle land
            in one chamber, two neighbouring chambers or two facing chambers.
            Straight ray tracing on the as-built station, the same geometry and
            method as ``draw_outlook``'s ``pair_acceptance``, with chamber B
            left out -- it has no field-shaping rings and measures no angle, so
            it enters no pairing (as on the topology slide).
  DRAWN     the underlying Al pair opening-angle shape.  A sketch: a narrow
            external-conversion peak broadened by the wall's multiple scattering,
            plus a broad internal-pair tail.  Not ``ipc_aluminium.spectrum`` and
            not Geant4, and labelled on the panel as a sketch.

Standalone matplotlib on purpose: ``scenes_x17`` imports ``mpgd26/style.py``,
which imports PyVista, and this figure is a diagram.  The station constants and
palette below are copied from ``scenes_x17`` -- keep them in step.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patheffects as pe                 # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
FONT = {'family': 'DejaVu Sans'}
_trapz = getattr(np, 'trapezoid', None) or np.trapz     # numpy 2 renamed it

# ---- palette: scenes_x17.palette('light') and style.COL ------------------- #
P = dict(page='#ffffff', ink='#141b24', muted='#5d6874', halo='#ffffff',
         gamma='#c1841c', x17='#a5308f', ipc='#e8621f',
         gas='#6fd0e8', pcb='#1a5b50', mesh='#9aa3ad')
# one colour per topology; intra keeps the one-chamber orange of draw_outlook
TOPO = {
    'intra':         dict(col='#e8621f', label='both legs in one chamber'),
    'perpendicular': dict(col='#1f8a70', label='neighbouring chambers'),
    'opposing':      dict(col='#2b5fa8', label='facing chambers'),
}

# ---- the station: scenes_x17.STATION / ARM_N / ARM_U ---------------------- #
STATION = dict(standoff=204.0, gap=30.0, board=8.0, half_u=199.5, half_v=180.0,
               pinwheel=16.0, n_arms=4, merge_mm=12.0)
ARM_N = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]], float)
ARM_U = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0]], float)
ARM_LETTER = ('D', 'B', 'A', 'C')
NO_ANGLE = {1}                   # chamber B: no drift field, enters no pairing
FACING = {frozenset((0, 1)), frozenset((2, 3))}

X17_MIN_DEG = 109.0              # the kinematic minimum the deck quotes
MERGE_DEG = 3.2                  # draw_outlook's one-cluster band
ACC_N, ACC_SEED = 60_000, 20260930
SMOOTH_BINS = 5                  # running mean over the acceptance, display only

W, H = 152.0, 63.0               # draw_outlook's canvas: the slide's hole
FS = 1.6                         # draw_outlook's OUTLOOK_FS
FS_SPEC = 1.18                   # ... and OUTLOOK_FS_SPEC


def ofs(pt):
    return pt * FS


def sfs(pt):
    return pt * FS * FS_SPEC


# --------------------------------------------------------------------------- #
# computed: the topology acceptance
# --------------------------------------------------------------------------- #
def _arm_hit(d):
    plane = STATION['standoff']
    n = len(d)
    arm = np.full(n, -1)
    uu, vv = np.zeros(n), np.zeros(n)
    for k in range(STATION['n_arms']):
        if k in NO_ANGLE:
            continue
        c = d @ ARM_N[k]
        ok = c > 1e-9
        t = np.where(ok, plane / np.where(ok, c, 1.0), 0.0)
        p = d * t[:, None]
        u = p @ ARM_U[k] + STATION['pinwheel']
        v = p[:, 1]
        good = ok & (np.abs(u) <= STATION['half_u']) & (np.abs(v) <= STATION['half_v'])
        arm = np.where(good, k, arm)
        uu = np.where(good, u, uu)
        vv = np.where(good, v, vv)
    return arm, uu, vv


def topology_acceptance(theta_deg):
    """Fraction of pairs at each opening angle landing intra / perp / opposing."""
    rng = np.random.default_rng(ACC_SEED)
    out = {k: np.empty(len(theta_deg)) for k in TOPO}
    n = ACC_N
    for i, t in enumerate(theta_deg):
        cz = rng.uniform(-1.0, 1.0, n)
        ph = rng.uniform(0.0, 2 * np.pi, n)
        sz = np.sqrt(1.0 - cz * cz)
        d1 = np.stack([sz * np.cos(ph), sz * np.sin(ph), cz], axis=1)
        a = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        a[np.abs(d1[:, 2]) > 0.9] = (1.0, 0.0, 0.0)
        e1 = np.cross(d1, a)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(d1, e1)
        psi = rng.uniform(0.0, 2 * np.pi, n)
        th = np.radians(t)
        d2 = np.cos(th) * d1 + np.sin(th) * (np.cos(psi)[:, None] * e1
                                             + np.sin(psi)[:, None] * e2)
        a1, u1, v1 = _arm_hit(d1)
        a2, u2, v2 = _arm_hit(d2)
        both = (a1 >= 0) & (a2 >= 0)
        same = both & (a1 == a2) & (np.hypot(u1 - u2, v1 - v2) >= STATION['merge_mm'])
        facing = both & np.array([frozenset((x, y)) in FACING for x, y in zip(a1, a2)])
        out['intra'][i] = same.mean()
        out['opposing'][i] = facing.mean()
        out['perpendicular'][i] = (both & (a1 != a2) & ~facing).mean()
    k = np.ones(SMOOTH_BINS) / SMOOTH_BINS
    for key, v in out.items():
        pad = np.pad(v, SMOOTH_BINS // 2, mode='edge')
        out[key] = np.convolve(pad, k, mode='valid')
    return out


# --------------------------------------------------------------------------- #
# drawn: the Al pair opening-angle shape (a sketch)
# --------------------------------------------------------------------------- #
def al_pair_shape(theta_deg):
    """dN/dtheta, arbitrary units: conversion peak + internal-pair tail."""
    t = np.asarray(theta_deg, float)
    conv = (t / 5.0 ** 2) * np.exp(-t / 5.0)                 # wall conversions
    th = np.radians(t)
    ipc = np.sin(th) / (1.04 - np.cos(th)) ** 1.3            # internal pairs
    ipc /= _trapz(ipc, t)
    return 0.85 * conv + 0.15 * ipc


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def arrow(ax, p0, p1, color, lw=1.6, ms=12, zorder=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0,
                                 zorder=zorder))


def head(ax, x, y, text):
    ax.text(x, y, text, fontsize=ofs(10.5), fontweight='bold', color=P['ink'],
            ha='left', va='center', **FONT)


def station_panel(ax, halo):
    head(ax, 2.0, 58.0, '1.  Find the two-track events')
    cx, cy = 22.0, 28.0
    sc = 0.083
    st = STATION
    r_gas = st['standoff']
    r_out = r_gas + st['gap'] + st['board']

    def mm(px_, pz):                    # seen from above: +X on the left
        return cx - px_ * sc, cy + pz * sc

    def ray(az_deg, r_mm):              # canvas azimuth, anticlockwise from right
        a = np.radians(az_deg)
        return cx + r_mm * sc * np.cos(a), cy + r_mm * sc * np.sin(a)

    for k in range(st['n_arms']):
        n2 = np.array([ARM_N[k][0], ARM_N[k][2]])
        u2 = np.array([ARM_U[k][0], ARM_U[k][2]])

        def quad(w0, w1, n2=n2, u2=u2):
            pts = []
            for w, sgn in ((w0, -1), (w0, +1), (w1, +1), (w1, -1)):
                q = n2 * w + u2 * (sgn * st['half_u'] - st['pinwheel'])
                pts.append(mm(q[0], q[1]))
            return pts

        dead = k in NO_ANGLE
        ax.add_patch(plt.Polygon(quad(r_gas, r_gas + st['gap']), closed=True,
                                 facecolor='none' if dead else P['gas'],
                                 alpha=0.6 if dead else 0.22,
                                 hatch='////' if dead else None,
                                 edgecolor=P['mesh'] if dead else P['gas'],
                                 lw=1.0, zorder=2))
        ax.add_patch(plt.Polygon(quad(r_gas + st['gap'], r_out), closed=True,
                                 facecolor=P['mesh'] if dead else P['pcb'],
                                 edgecolor='none', zorder=3))
        lp = n2 * (r_gas + st['gap'] / 2) + u2 * (0.80 * st['half_u'] - st['pinwheel'])
        ax.text(*mm(lp[0], lp[1]), ARM_LETTER[k], fontsize=ofs(10.0),
                fontweight='bold', color=P['muted'], ha='center', va='center',
                path_effects=halo, zorder=6, **FONT)

    # one example pair per topology, canvas azimuths (D left, A top, C bottom)
    legs = {'intra': (80.0, 104.0),           # both into A
            'perpendicular': (196.0, 286.0),  # D and C, 90 deg
            'opposing': (64.0, 244.0)}        # A and C, 180 deg
    for key, (az1, az2) in legs.items():
        for az in (az1, az2):
            ax.plot(*np.array([mm(0, 0), ray(az, r_out + 3.0)]).T,
                    color=TOPO[key]['col'], lw=2.4, zorder=6,
                    solid_capstyle='round')

    lab = dict(fontsize=ofs(7.6), fontweight='bold', ha='center', va='center',
               linespacing=1.2, path_effects=halo, zorder=8, **FONT)
    ax.text(cx + 1.5, cy + 12.5, 'intra', color=TOPO['intra']['col'], **lab)
    ax.text(cx - 11.0, cy - 10.5, 'perpen-\ndicular', color=TOPO['perpendicular']['col'], **lab)
    ax.text(*ray(38.0, 160.0), 'opposing', color=TOPO['opposing']['col'], **lab)

    ax.add_patch(Circle((cx, cy), 11.5 * sc, facecolor=P['gamma'],
                        edgecolor=P['ink'], lw=0.8, zorder=9))
    ax.text(cx + 1.6, cy - 1.9, '$^{27}$Al', fontsize=ofs(7.6), color=P['muted'],
            ha='left', va='top', path_effects=halo, zorder=9, **FONT)


def arrow_panel(ax, halo):
    xa, xb, y = 44.5, 55.5, 28.0
    arrow(ax, (xa, y), (xb, y), P['ink'], lw=3.0, ms=21, zorder=6)
    ax.text(0.5 * (xa + xb), y + 3.8, '41.8 M\nevents', fontsize=ofs(9.0),
            fontweight='bold', color=P['ink'], ha='center', va='bottom',
            linespacing=1.3, path_effects=halo, zorder=7, **FONT)
    ax.text(0.5 * (xa + xb), y - 3.4, 'one angle\nper pair', fontsize=ofs(8.4),
            color=P['muted'], ha='center', va='top', linespacing=1.3, zorder=7,
            **FONT)


def spectrum_panel(fig, ax, halo):
    x0, x1 = 60.0, 150.0
    head(ax, x0 - 3.0, 58.0, '2.  Histogram the opening angle')

    theta = np.concatenate([np.arange(0.5, 10.0, 0.5), np.arange(10.0, 180.01, 2.0)])
    acc = topology_acceptance(theta)
    shape = al_pair_shape(theta)
    comp = {k: shape * acc[k] for k in TOPO}
    total = sum(comp.values())
    scale = 1.0 / total.max()
    total = total * scale
    comp = {k: v * scale for k, v in comp.items()}

    px = fig.add_axes([x0 / W, 10.5 / H, (x1 - x0) / W, 41.0 / H], facecolor='none')
    for sp in ('top', 'right'):
        px.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        px.spines[sp].set_color(P['muted'])
        px.spines[sp].set_linewidth(1.1)
    px.tick_params(colors=P['muted'], labelsize=sfs(8.6), width=1.1, length=4)

    lo, hi = 1e-4, 4.0
    px.axvspan(0, MERGE_DEG, color=P['muted'], alpha=0.20, lw=0, zorder=1)
    px.axvspan(X17_MIN_DEG, 180, color=P['x17'], alpha=0.07, lw=0, zorder=1)
    px.axvline(X17_MIN_DEG, color=P['x17'], lw=1.1, ls=':', alpha=0.85, zorder=4)

    px.fill_between(theta, lo, np.maximum(total, lo), color=P['ink'], alpha=0.06,
                    lw=0, zorder=2)
    px.plot(theta, total, color=P['ink'], lw=3.0, zorder=5,
            label='$^{27}$Al e$^{+}$e$^{-}$ pairs  (sketch)')
    for k, d in TOPO.items():
        y = np.where(comp[k] > lo * 0.5, comp[k], np.nan)
        px.plot(theta, y, color=d['col'], lw=2.2, zorder=4,
                label=f'   … {d["label"]}')

    px.set_yscale('log')
    px.set_xlim(0, 180)
    px.set_ylim(lo, hi)
    px.set_xticks([0, 45, 90, 135, 180])
    px.set_yticks([])
    px.set_yticks([], minor=True)
    px.set_xlabel('e$^{+}$e$^{-}$ opening angle  (deg)', fontsize=sfs(9.4),
                  color=P['muted'], labelpad=3, **FONT)
    px.set_ylabel('pairs  (log, arb.)', fontsize=sfs(9.4), color=P['muted'],
                  labelpad=5, **FONT)
    leg = px.legend(loc='upper right', bbox_to_anchor=(1.03, 1.06), frameon=True,
                    facecolor=P['page'], edgecolor='none', framealpha=1.0,
                    borderpad=0.5, fontsize=sfs(8.0), handlelength=1.9,
                    labelspacing=0.42)
    leg.set_zorder(10)
    for t_, c in zip(leg.get_texts(), [P['ink']] + [d['col'] for d in TOPO.values()]):
        t_.set_color(c)
        t_.set_fontfamily('DejaVu Sans')

    note = dict(ha='center', va='center', zorder=9, path_effects=halo, **FONT)
    px.text(144.5, 0.075, 'where X17 would sit\n(θ ≥ %.0f°)' % X17_MIN_DEG,
            color=P['x17'], fontsize=sfs(8.2), fontweight='bold',
            linespacing=1.3, **note)
    px.annotate('below ~3° the two tracks\nare one cluster',
                xy=(3.6, 0.3), xytext=(34.0, 3.0e-4), color=P['muted'],
                fontsize=sfs(8.0), linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color=P['muted'], lw=1.2,
                                shrinkA=3, shrinkB=3), **note)
    return {k: float(_trapz(v, theta)) for k, v in comp.items()}


def draw():
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    fig = plt.figure(figsize=(W / 10.0, H / 10.0), dpi=300, facecolor=P['page'])
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = [pe.withStroke(linewidth=2.6, foreground=P['halo'], alpha=0.9)]
    station_panel(ax, halo)
    arrow_panel(ax, halo)
    shares = spectrum_panel(fig, ax, halo)
    return fig, shares


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    fig, shares = draw()
    base = os.path.join(FIG, 'al_pair_outlook')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', facecolor=P['page'])
    tot = sum(shares.values())
    for k, v in shares.items():
        print(f'  {k:14s} {v / tot:6.1%} of the sketched accepted pairs')
    print(f'  -> {base}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
