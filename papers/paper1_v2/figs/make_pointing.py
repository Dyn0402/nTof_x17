#!/usr/bin/env python3
"""make_pointing.py -- Paper I v2: why the angle, not the position, sets how well a chamber points at the target.

    .venv/bin/python papers/paper1_v2/figs/make_pointing.py [--out papers/paper1_v2/figs/out/pointing] [--n 200000]

Toy (no data are read). Geometry along the chamber normal (mcs_common): target at 0, window at 200 mm, 60 um Mylar,
5.0 mm gas gap, cathode (50 um Kapton + 9 um Cu, assumed), drift gas 28.2 mm (measured), readout plane at
d = 233.3 mm. d is the distance from the target to the plane where the position is measured (the readout plane);
position and slope errors are taken as independent there.

Tracks are thrown isotropically from a point target into the efficient area (399.4 x 359.9 mm^2) at the window.
One projection (x, z): the measured position at the readout plane is smeared by N(0, sigma_x), the measured projected
angle by N(0, sigma_theta); the measured line is extrapolated back and its miss distance is the distance of closest
approach to the target in that projection (impact parameter, b = x0 cos(alpha)). Pointing resolution = 68th
percentile of |b| (the s68 used for the measured angle resolution). For normal incidence this is
sqrt(sigma_x^2 + (d sigma_theta)^2); over the acceptance the path length is d/cos(alpha), so the toy sits a few %
above that curve.

Right-hand marks in (b): the pointing smear from multiple scattering in the material IN FRONT of the drift gas
(air, window, gas gap, cathode; Highland on the total, shared by x/X0 among the layers, each kick at distance s from
the target moving the back-extrapolation by s * kick; normal incidence). It is not part of the chamber resolution,
but it is the same kind of error (an angle) and a referee will ask.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
import paper_style as ps  # noqa: E402
import mcs_common as mc  # noqa: E402

D = mc.Z_READOUT                       # 233.3 mm
SEG = mc.G_DRIFT                       # 28.2 mm measured segment (normal incidence)
SIG_X = (0.1, 0.5, 1.0, 2.0)
SIG_TH = np.arange(0.0, 3.01, 0.25)    # deg
MEAS = dict(theta_x=0.99, theta_y=0.90, x_intr=0.12, x_core=0.5)
Q68 = 0.6826894921


def throw(n, rng):
    """Isotropic directions from the target that cross the efficient area at the window: projected tan in x."""
    cmax = np.cos(np.arctan(np.hypot(mc.ACT_X / 2, mc.ACT_Y / 2) / mc.D_WIN))
    out = []
    while sum(len(o) for o in out) < n:
        c = rng.uniform(cmax, 1.0, 2 * n)
        ph = rng.uniform(0, 2 * np.pi, 2 * n)
        s = np.sqrt(1 - c * c)
        tx, ty = s * np.cos(ph) / c, s * np.sin(ph) / c
        ok = (np.abs(tx * mc.D_WIN) < mc.ACT_X / 2) & (np.abs(ty * mc.D_WIN) < mc.ACT_Y / 2)
        out.append(tx[ok])
    return np.concatenate(out)[:n]


def pointing_toy(tx, sx, sth_deg, rng):
    a = np.arctan(tx)
    xm = D * tx + rng.normal(0, sx, tx.size)
    am = a + rng.normal(0, np.radians(sth_deg), tx.size)
    b = (xm - D * np.tan(am)) * np.cos(am)
    return float(np.quantile(np.abs(b), Q68))


def analytic(sx, sth_deg, d=D):
    return np.sqrt(sx ** 2 + (d * np.radians(sth_deg)) ** 2)


def scatter_pointing(T):
    """rms back-extrapolation miss [mm] and total theta0 [deg] from the material in front of the drift gas."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in mc.LAYERS])
    th = mc.highland(xs.sum(), T)
    var = 0.0
    for (_, _, m, t, z0, _), x in zip(mc.LAYERS, xs):
        z1 = z0 + t
        var += th ** 2 * x / xs.sum() * (z0 * z0 + z0 * z1 + z1 * z1) / 3.0
    return float(np.sqrt(var)), float(np.degrees(th))


def scatter_chord(T, zm=D):
    """rms error [rad] of the direction target -> measured position (origin known exactly) from the same material:
    a kick at distance s moves the position at zm by (zm - s) * kick, so the direction by (1 - s/zm) * kick."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in mc.LAYERS])
    th = mc.highland(xs.sum(), T)
    var = 0.0
    for (_, _, m, t, z0, _), x in zip(mc.LAYERS, xs):
        u0, u1 = zm - z0, zm - z0 - t
        var += th ** 2 * x / xs.sum() * (u0 * u0 + u0 * u1 + u1 * u1) / 3.0 / zm ** 2
    return float(np.sqrt(var))


def two_estimators(th_ch_deg=0.95, sx=0.12, m_x=16.8, e_tot=20.58):
    """Local angle against (position + known origin), per projection, and what each gives for a pair.
    th_ch_deg: chamber angle resolution (mean of the X and Y views of chamber A)."""
    out = {}
    th_open = 2 * np.arcsin(m_x / e_tot)
    dm_per_rad = m_x / (2 * np.tan(th_open / 2))
    for T in (4, 8, 16, round((e_tot - 2 * 0.511) / 2, 2)):
        _, th0 = scatter_pointing(T)
        local = float(np.hypot(th0, th_ch_deg))
        chord = float(np.hypot(np.degrees(scatter_chord(T)), np.degrees(sx / D)))
        out[f'{T}MeV'] = dict(
            local_deg=round(local, 2), chord_scatter_deg=round(chord, 2),
            origin_breakeven_mm=round(float(D * np.radians(np.sqrt(local ** 2 - chord ** 2))), 1),
            opening_local_deg=round(np.sqrt(2) * local, 2),
            dm_local_MeV=round(float(dm_per_rad * np.radians(np.sqrt(2) * local)), 2))
    out['chamber_only'] = dict(
        local_deg=th_ch_deg, origin_breakeven_mm=round(float(D * np.radians(th_ch_deg)), 1),
        opening_deg=round(np.sqrt(2) * th_ch_deg, 2),
        dm_MeV=round(float(dm_per_rad * np.radians(np.sqrt(2) * th_ch_deg)), 3))
    out['kinematics'] = dict(m_x_MeV=m_x, e_tot_MeV=e_tot, min_opening_deg=round(float(np.degrees(th_open)), 1),
                             dm_per_deg_MeV=round(float(dm_per_rad * np.radians(1.0)), 3))
    out['notes'] = ('per projection; local = scattering in front of the gas (+) chamber angle; chord = direction '
                    'from an exactly known origin to the position at the readout plane; breakeven = origin error at '
                    'which the two are equal; opening = sqrt(2) x one direction (two tracks, in-plane component); '
                    'the fourth energy is the kinetic energy of each lepton of a symmetric pair')
    return out


# ---------------------------------------------------------------------------------------------------------------
def side_view(ax, rng):
    import matplotlib.patches as mp
    ink, mut = ps.INK, ps.MUTED
    cA = ps.DET_COLOR['det3']
    tan_tr = 0.62                                   # the drawn track: 32 deg
    # chamber slab (one half, to scale)
    y0, y1 = -10.0, 200.0
    ax.add_patch(mp.Rectangle((mc.Z_DRIFT0, y0), mc.G_DRIFT, y1 - y0, fc='#e8eef8', ec='none', zorder=1))
    ax.add_patch(mp.Rectangle((mc.D_WIN, y0), mc.T_GAP5 + mc.T_WIN, y1 - y0, fc='#f4f6f9', ec='none', zorder=1))
    ax.plot([mc.D_WIN] * 2, [y0, y1], color=cA, lw=1.6, zorder=3)                  # window
    ax.plot([mc.Z_CATH] * 2, [y0, y1], color=ink, lw=0.9, zorder=3)                # cathode
    ax.plot([D] * 2, [y0, y1], color=ink, lw=0.9, ls=(0, (1, 1.2)), zorder=3)     # readout plane / mesh
    ax.add_patch(mp.Rectangle((D, y0), 1.7, y1 - y0, fc='#7f8a99', ec='none', zorder=2))   # readout board
    ax.text(mc.D_WIN - 2, 203, 'window', ha='right', va='bottom', fontsize=7, color=cA)
    ax.text(mc.Z_DRIFT0 + mc.G_DRIFT / 2, 203, 'drift\n28 mm', ha='center', va='bottom', fontsize=7, color=ink,
            linespacing=0.95)
    # target
    ax.add_patch(mp.Circle((0, 0), 3.0, fc='white', ec=ink, lw=0.9, zorder=6))
    ax.plot(0, 0, 'o', ms=1.5, color=ink, zorder=7)
    ax.text(-4, -9, 'target', fontsize=7, ha='left', va='top', color=ink)
    # the track: true path in light ink, measured segment heavy, back-extrapolation dashed
    zz = np.array([0, D + 8])
    ax.plot(zz, tan_tr * zz, color=ps.ACCENT, lw=0.8, alpha=0.6, zorder=4)
    zs = np.array([mc.Z_DRIFT0, D])
    ax.plot(zs, tan_tr * zs, color=ps.ACCENT, lw=3.2, solid_capstyle='butt', zorder=5)
    ax.text(mc.D_WIN - 4, tan_tr * D + 4, 'measured\nsegment', ha='right', va='bottom', fontsize=7, color=ps.ACCENT,
            linespacing=0.95)
    ax.text(105, tan_tr * 105 + 7, 'extrapolation', rotation=np.degrees(np.arctan(tan_tr)), rotation_mode='anchor',
            ha='center', va='bottom', fontsize=7, color=mut)
    # distance arrow
    ya = -24
    ax.annotate('', xy=(D, ya), xytext=(0, ya), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7, shrinkA=0,
                                                                shrinkB=0))
    ax.text(D / 2, ya - 3, f'd = {D:.0f} mm', ha='center', va='top', fontsize=7, color=mut)
    ax.set_xlim(-12, D + 10)
    ax.set_ylim(-40, 225)
    ax.set_aspect('equal')
    ax.axis('off')

    # zoomed inset at the target, to scale: angle error 1 deg (position perfect) vs position error 0.5 mm
    axi = ax.inset_axes([0.0, 0.50, 0.60, 0.43])
    W = 7.0
    n_l = 40
    a0 = np.arctan(tan_tr)
    nrm = np.array([-np.sin(a0), np.cos(a0)])       # unit normal to the track
    zp = D                                          # pivot: the measured point
    pz = np.array([zp, tan_tr * zp])
    zline = np.array([-W * 1.6, W * 1.6])
    for i in range(n_l):
        da = np.radians(rng.normal(0, 1.0))
        a = a0 + da
        # line through the measured point with angle a, drawn near the target in track-aligned coordinates
        u = np.array([np.cos(a), np.sin(a)])
        t_ = -pz @ u                                 # parameter of closest approach to the target
        pc = pz + t_ * u
        off = pc @ nrm
        along = np.array([-W * 2, W * 2])
        axi.plot(along, off + along * np.tan(da), color=cA, lw=0.5, alpha=0.6, zorder=3)
        dx = rng.normal(0, 0.5)
        axi.plot(along, np.full(2, dx * np.cos(a0)), color=ps.ACCENT, lw=0.5, alpha=0.55, zorder=4)
    s_ang = D * np.radians(1.0)
    axi.axhspan(-s_ang, s_ang, color=cA, alpha=0.08, lw=0, zorder=1)
    axi.add_patch(mp.Circle((0, 0), 0.35, fc='white', ec=ink, lw=0.7, zorder=6))
    axi.set_xlim(-W * 1.5, W * 1.5)
    axi.set_ylim(-W, W)
    axi.set_aspect('equal')
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_color(mut)
        s.set_linewidth(0.6)
    axi.grid(False)
    axi.plot([W * 1.5 - 1.0 - 5, W * 1.5 - 1.0], [-W + 1.0] * 2, color=ink, lw=1.0, zorder=7)
    axi.text(W * 1.5 - 3.5, -W + 1.5, '5 mm', ha='center', va='bottom', fontsize=6.5, color=ink, zorder=7,
             bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    axi.text(-W * 1.5 + 0.6, W - 0.6, r'$\sigma_\theta = 1°$', ha='left', va='top', fontsize=7, color=cA, zorder=7, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    axi.text(-W * 1.5 + 0.6, -W + 0.6, r'$\sigma_x = 0.5$ mm', ha='left', va='bottom', fontsize=7,
             color=ps.ACCENT, zorder=7, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    axi.text(W * 1.5 - 0.6, W - 0.6, 'at the target,\nalong the track', ha='right', va='top', fontsize=6.5,
             color=mut, linespacing=0.95, zorder=7, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    # link the inset to the target
    from matplotlib.patches import ConnectionPatch
    ax.add_artist(ConnectionPatch(xyA=(0, 3.5), coordsA=ax.transData, xyB=(0.08, 0.0), coordsB=axi.transAxes,
                                  color=mut, lw=0.6, ls='--', zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'pointing'))
    ap.add_argument('--n', type=int, default=200000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ps.use()
    rng = np.random.default_rng(20261110)
    tx = throw(a.n, rng)

    toy = {sx: [pointing_toy(tx, sx, s, rng) for s in SIG_TH] for sx in SIG_X}

    fig = plt.figure(figsize=(ps.FULL[0], 3.0))
    axa = fig.add_axes([0.0, 0.03, 0.47, 0.95])
    axb = fig.add_axes([0.58, 0.15, 0.33, 0.80])
    side_view(axa, np.random.default_rng(7))

    cols = ['#0b3d91', '#56B4E9', '#E69F00', '#CC79A7']
    mk = ['o', 's', '^', 'D']
    th_f = np.linspace(0, 3.0, 200)
    for sx, c, m in zip(SIG_X, cols, mk):
        axb.plot(th_f, analytic(sx, th_f), color=c, lw=1.1, zorder=3)
        axb.plot(SIG_TH, toy[sx], ls='none', marker=m, color=c, ms=3.6, zorder=4,
                 label=f'{sx:g} mm')
    # the measured chamber: theta 0.90-0.99 deg, sigma_x 0.12-0.5 mm
    lo, hi = analytic(MEAS['x_intr'], MEAS['theta_y']), analytic(MEAS['x_core'], MEAS['theta_x'])
    axb.add_patch(plt.Rectangle((MEAS['theta_y'], lo), MEAS['theta_x'] - MEAS['theta_y'], hi - lo,
                                fc='none', ec=ps.ACCENT, lw=1.2, zorder=6))
    axb.plot([MEAS['theta_y'], MEAS['theta_x']], [lo, hi], ls='none')
    axb.annotate('this chamber', xy=(MEAS['theta_x'] + 0.03, hi), xytext=(1.2, 2.0), fontsize=7,
                 color=ps.ACCENT, arrowprops=dict(arrowstyle='-', color=ps.ACCENT, lw=0.7), ha='left')
    # scattering in front of the gas (marks on the right edge)
    scat = {T: scatter_pointing(T) for T in (4, 8, 16)}
    ymax = 13.0
    for T, (s, th) in scat.items():
        if s < ymax:
            axb.annotate('', xy=(3.0, s), xytext=(3.22, s), annotation_clip=False,
                         arrowprops=dict(arrowstyle='-|>', color=ps.MUTED, lw=0.7, mutation_scale=6))
            axb.text(3.25, s, f'{T} MeV', fontsize=7, color=ps.MUTED, va='center', ha='left',
                     transform=axb.transData, clip_on=False)
        else:
            axb.text(3.25, ymax, f'{T} MeV:\n{s:.0f} mm', fontsize=7, color=ps.MUTED, va='top', ha='left',
                     clip_on=False, linespacing=0.95)
    axb.text(3.05, ymax + 0.25, 'scattering in\nfront of gas', fontsize=6.5, color=ps.MUTED, ha='left',
             va='bottom', clip_on=False, linespacing=0.95)
    axb.set_xlim(0, 3.0)
    axb.set_ylim(0, ymax)
    axb.set_xlabel(r'angular resolution $\sigma_\theta$ (°)')
    axb.set_ylabel('pointing resolution at target (mm)')
    axb.legend(title=r'$\sigma_x$', loc='upper left', fontsize=7, title_fontsize=7, handlelength=1.0,
               borderaxespad=0.2, labelspacing=0.25)
    axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, fontsize=ps.BASE_PT, fontweight='bold', va='top', ha='left')
    axb.text(-0.25, 1.04, '(b)', transform=axb.transAxes, fontsize=ps.BASE_PT, fontweight='bold', va='top',
             ha='left')

    ps.save(fig, a.out)

    # numbers
    num = dict(
        d_mm=round(D, 2),
        d_definition='target to the readout (mesh) plane, along the chamber normal: 200 window + 0.060 Mylar + '
                     '5.0 gas + 0.059 cathode + 28.2 drift',
        segment_mm=SEG,
        lever_over_segment=round(D / SEG, 2),
        lever_mid_over_segment=round((mc.Z_DRIFT0 + SEG / 2) / SEG, 2),
        pointing_mm={}, pointing_toy_acceptance_mm={},
        sigma_x_equiv_to_1deg_mm=round(D * np.tan(np.radians(1.0)), 2),
        theta_equiv_to_0p5mm_deg=round(float(np.degrees(0.5 / D)), 3),
        theta_equiv_to_0p12mm_deg=round(float(np.degrees(0.12 / D)), 3),
        position_share_of_pointing_variance={},
        scattering_in_front_of_gas={f'{T}MeV': dict(theta0_deg=round(th, 2), pointing_rms_mm=round(s, 1),
                                                    equiv_sigma_theta_deg=round(float(np.degrees(s / D)), 2))
                                    for T, (s, th) in scat.items()},
        two_estimators=two_estimators(),
        toy=dict(n_tracks=a.n, sigma_theta_deg=SIG_TH.round(3).tolist(),
                 s68_mm={f'{sx:g}': np.round(v, 3).tolist() for sx, v in toy.items()}),
        notes='pointing = s68 of the impact parameter in one projection; analytic = normal incidence '
              'sqrt(sx^2+(d sth)^2); toy averages isotropic tracks over the efficient area (path d/cos alpha).',
    )
    for th in (0.99, 0.90):
        for sx in (0.12, 0.5):
            k = f'theta{th:.2f}_x{sx:g}'
            num['pointing_mm'][k] = round(float(analytic(sx, th)), 2)
            num['pointing_toy_acceptance_mm'][k] = round(pointing_toy(tx, sx, th, rng), 2)
            num['position_share_of_pointing_variance'][k] = round(float(sx ** 2 / analytic(sx, th) ** 2), 4)
    with open(a.out + '.json', 'w') as f:
        json.dump(num, f, indent=1)
    print(json.dumps({k: v for k, v in num.items() if k != 'toy'}, indent=1))


if __name__ == '__main__':
    main()
