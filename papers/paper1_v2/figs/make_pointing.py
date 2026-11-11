#!/usr/bin/env python3
"""make_pointing.py -- Paper I v2, Fig. 2: what the angle of a short segment far from the target is worth.

    .venv/bin/python papers/paper1_v2/figs/make_pointing.py [--out papers/paper1_v2/figs/out/pointing] [--n 200000]

Toy (no data are read). Kept general on purpose (Dylan 2026-11-11): the argument must hold for any X17 search with a
chamber of this kind, so the figure uses only the air between target and chamber and a generic source size, not the
foils of this chamber or the n_TOF 3He capsule. Geometry (mcs_common): target at 0, window at 200 mm, readout plane
at d = 233.3 mm, the plane where the position is measured.

(a) One chamber and an extended target to scale, with one lepton whose deflection in the air is exaggerated (8 deg
    drawn). Its measured local angle, extrapolated back, misses the origin by b.
(b) Error of the measured angle as an estimate of the initial direction, one projection, against the kinetic energy of
    the lepton, for a chamber of TH_REQ = 1 deg and 200 mm of air. Line analytic (scatter_pointing on AIR), points
    scatter_toy.
Dylan 2026-11-12: the source is taken as extended, so the direction has to come from the chamber; the estimate from an
assumed origin and the measured position (compact source) is no longer drawn. Its numbers stay in the json
(air_only.source_breakeven_mm, toy_directions) for the one sentence the text keeps about a compact source.

The json also keeps the numbers that include the foils (scattering_in_front_of_gas, two_estimators: sections 3.3 and
6.2) and the pointing toy (tracks thrown isotropically into the efficient area, position smeared by sigma_x, angle by
sigma_theta, s68 of the impact parameter: sqrt(sigma_x^2 + (d sigma_theta)^2) at normal incidence, ~10 % more over
the acceptance).
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
TH_CH = 0.95                           # chamber angle resolution used for the pair numbers (mean of X and Y)
TH_REQ = 1.0                           # the requirement drawn in (b) [deg]
ORIGINS = (2.0, 10.0)                  # rms size of the source [mm] drawn in (b): a beam spot, an extended target
R_BORE = 10.0                          # radius of the target drawn in (a) [mm] (generic extended source)


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


AIR = mc.LAYERS[:1]                    # the air between target and window, alone


def scatter_pointing(T, layers=mc.LAYERS):
    """rms back-extrapolation miss [mm] and total theta0 [deg] from the material in front of the drift gas."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in layers])
    th = mc.highland(xs.sum(), T)
    var = 0.0
    for (_, _, m, t, z0, _), x in zip(layers, xs):
        z1 = z0 + t
        var += th ** 2 * x / xs.sum() * (z0 * z0 + z0 * z1 + z1 * z1) / 3.0
    return float(np.sqrt(var)), float(np.degrees(th))


def scatter_chord(T, zm=D, layers=mc.LAYERS):
    """rms error [rad] of the direction target -> measured position (origin known exactly) from the same material:
    a kick at distance s moves the position at zm by (zm - s) * kick, so the direction by (1 - s/zm) * kick."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in layers])
    th = mc.highland(xs.sum(), T)
    var = 0.0
    for (_, _, m, t, z0, _), x in zip(layers, xs):
        u0, u1 = zm - z0, zm - z0 - t
        var += th ** 2 * x / xs.sum() * (u0 * u0 + u0 * u1 + u1 * u1) / 3.0 / zm ** 2
    return float(np.sqrt(var))


def two_estimators(th_ch_deg=TH_CH, sx=0.12, m_x=16.8, e_tot=20.58):
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


def scatter_toy(T, n, rng, th_ch_deg=TH_CH, sx=MEAS['x_intr'], origins=ORIGINS, n_air=50, layers=mc.LAYERS):
    """Toy check of the two direction estimates, one projection, normal incidence. Each lepton starts on the axis,
    takes Gaussian kicks in the layers in front of the gas (air in n_air steps, each foil one kick; Highland on the
    total, shared by x/X0 as in scatter_pointing) and is measured at the readout plane.
    Returns s68 [deg] of (local angle - initial direction) and of (position - assumed origin)/d - initial direction,
    the second for each rms origin error in `origins`."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in layers])
    th = mc.highland(xs.sum(), T)
    ang = np.zeros(n)
    pos = np.zeros(n)
    z = 0.0
    for (_, _, m, t, z0, _), x in zip(layers, xs):
        k = n_air if t > 1.0 else 1
        for i in range(k):
            zk = z0 + (i + 0.5) * t / k
            pos += ang * (zk - z)
            z = zk
            ang += rng.normal(0, th * np.sqrt(x / xs.sum() / k), n)
    pos += ang * (D - z)
    local = ang + rng.normal(0, np.radians(th_ch_deg), n)
    xm = pos + rng.normal(0, sx, n)
    s68 = lambda v: float(np.degrees(np.quantile(np.abs(v), Q68)))
    return s68(local), {o: s68((xm - rng.normal(0, o, n)) / D) for o in origins}


# ---------------------------------------------------------------------------------------------------------------
def side_view(ax):
    """One chamber and the target to scale. The deflection of the drawn lepton is exaggerated (KINK in all)."""
    import matplotlib.patches as mp
    ink, mut = ps.INK, ps.MUTED
    cL = ps.DET_COLOR['det3']
    y0, y1 = -28.0, 150.0
    ax.add_patch(mp.Rectangle((mc.Z_DRIFT0, y0), mc.G_DRIFT, y1 - y0, fc='#e8eef8', ec='none', zorder=1))
    ax.plot([mc.D_WIN] * 2, [y0, y1], color=ink, lw=0.9, zorder=3)                 # window
    ax.add_patch(mp.Rectangle((D, y0), 1.7, y1 - y0, fc='#7f8a99', ec='none', zorder=2))   # readout board
    ax.text(mc.D_WIN + 2, y1 + 3, 'window', ha='right', va='bottom', fontsize=7, color=ink,
            linespacing=0.95)
    ax.text(mc.Z_DRIFT0 + mc.G_DRIFT / 2 + 2, y1 + 3, 'drift\n28 mm', ha='left', va='bottom', fontsize=7, color=ink,
            linespacing=0.95)
    # the 3He volume across the beam, to scale; the lepton starts off its centre
    ax.add_patch(mp.Circle((0, 0), R_BORE, fc='#f1f3f6', ec=ink, lw=0.8, zorder=2))
    org = np.array([0.0, 6.0])
    ax.plot(*org, 'o', ms=2.6, color=ink, zorder=8)
    ax.text(0, R_BORE + 3, 'target', fontsize=7, ha='center', va='bottom', color=ink)
    # true path: initial direction t0, a smooth bend in the air and a kink at the foils
    t0 = 0.36
    bend_air, kink = np.radians(8.0), np.radians(0.0)
    za = np.linspace(0, mc.D_WIN, 60)
    ang_a = np.arctan(t0) + bend_air * (za / mc.D_WIN) ** 1.0
    ya = org[1] + np.concatenate([[0], np.cumsum(np.tan(0.5 * (ang_a[1:] + ang_a[:-1])) * np.diff(za))])
    a_gas = ang_a[-1] + kink
    zg = np.array([mc.D_WIN, D])
    yg = ya[-1] + np.tan(a_gas) * (zg - mc.D_WIN)
    ax.plot(za, ya, color=ink, lw=0.9, zorder=5)
    ax.plot(zg, yg, color=ink, lw=0.9, zorder=5)
    zs = np.array([mc.Z_DRIFT0, D])
    ys = ya[-1] + np.tan(a_gas) * (zs - mc.D_WIN)
    ax.plot(zs, ys, color=cL, lw=3.4, solid_capstyle='butt', zorder=6)
    # initial direction, continued
    zz = np.array([0, D])
    ax.plot(zz, org[1] + t0 * zz, color=mut, lw=0.7, ls=(0, (1, 1.6)), zorder=4)
    # estimate 1: the local angle, extrapolated back to the target plane
    yb = ys[1] - np.tan(a_gas) * D
    ax.plot([0, zs[0]], [yb, ys[0]], color=cL, lw=0.9, ls=(0, (4, 2)), zorder=4)
    # the miss distance
    ax.annotate('', xy=(-18, yb), xytext=(-18, org[1]),
                arrowprops=dict(arrowstyle='<->', color=cL, lw=0.7, shrinkA=0, shrinkB=0))
    ax.plot([-21, 0], [yb, yb], color=cL, lw=0.5, zorder=3)
    ax.plot([-21, 0], [org[1]] * 2, color=cL, lw=0.5, zorder=3)
    ax.text(-21, (yb + org[1]) / 2, '$b$', ha='right', va='center', fontsize=8, color=cL)
    rot = lambda t: float(np.degrees(np.arctan(t)))
    kw = dict(rotation_mode='anchor', ha='center', fontsize=7)
    zt = 118
    ax.text(zt, org[1] + t0 * zt + 19, 'path, and its initial direction', rotation=rot(t0), va='bottom',
            color=ink, **kw)
    zt = 100
    ax.text(zt, yb + np.tan(a_gas) * zt - 7, 'measured angle, extrapolated', rotation=rot(np.tan(a_gas)), va='top',
            color=cL, **kw)
    ax.text(zs[0] - 5, ys[1] + 8, 'measured\nsegment', ha='right', va='bottom', fontsize=7, color=cL,
            linespacing=0.95)
    yd = yb - 12
    ax.annotate('', xy=(D, yd), xytext=(0, yd), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7, shrinkA=0,
                                                                shrinkB=0))
    ax.text(D / 2, yd - 3, f'air, d = {D:.0f} mm to the readout', ha='center', va='top', fontsize=7, color=mut)
    ax.set_xlim(-30, D + 6)
    ax.set_ylim(yd - 16, y1 + 26)
    ax.set_aspect('equal')
    ax.axis('off')
    return dict(drawn_deflection_deg=round(float(np.degrees(bend_air + kink)), 1), origin_offset_mm=float(org[1]))


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
    axa = fig.add_axes([0.0, 0.02, 0.48, 0.96])
    axb = fig.add_axes([0.595, 0.15, 0.275, 0.80])
    drawn = side_view(axa)

    # (b) error on the initial direction of one lepton, one projection, against its kinetic energy.
    # Only the air between target and window scatters here: the argument does not depend on the foils of a chamber.
    cL = ps.DET_COLOR['det3']
    scat = {T: scatter_pointing(T) for T in (4, 8, 16)}            # all layers, for the json (section 3.3 / 6.2)
    Tf = np.linspace(3.0, 16.5, 120)
    th0 = np.array([scatter_pointing(T, AIR)[1] for T in Tf])
    pos_deg = float(np.degrees(MEAS['x_intr'] / D))
    axb.axhline(TH_REQ, color=cL, lw=0.9, ls=(0, (4, 2)), zorder=3)
    axb.plot(Tf, th0, color=ps.MUTED, lw=1.2, zorder=3)
    axb.plot(Tf, np.hypot(th0, TH_REQ), color=cL, lw=1.5, zorder=5)
    T_TOY = (4, 6, 8, 10, 12, 14, 16)
    toy2 = {T: scatter_toy(T, a.n, rng, th_ch_deg=TH_REQ, layers=AIR) for T in T_TOY}
    axb.plot(T_TOY, [toy2[T][0] for T in T_TOY], ls='none', marker='o', ms=3.4, color=cL, zorder=6)
    lab = dict(fontsize=7, linespacing=0.95, ha='left', va='center', clip_on=False)
    xe = 16.75                                                     # curve labels, outside the right edge
    axb.text(xe, 1.42, 'measured\nangle', color=cL, **lab)
    axb.text(xe, 0.95, 'air alone', color=ps.MUTED, **lab)
    axb.text(3.25, 1.04, '1° chamber', color=cL, fontsize=7, ha='left', va='bottom')
    axb.set_xlim(3.0, 16.5)
    axb.set_ylim(0, 4.6)
    axb.set_xticks([4, 8, 12, 16])
    axb.set_xlabel('lepton kinetic energy (MeV)')
    axb.set_ylabel('error on the initial direction (°)')
    te = two_estimators()
    air = {f'{T}MeV': dict(theta_air_deg=round(scatter_pointing(T, AIR)[1], 2),
                           local_deg=round(float(np.hypot(scatter_pointing(T, AIR)[1], TH_REQ)), 2),
                           origin_scatter_deg=round(float(np.degrees(scatter_chord(T, layers=AIR))), 2),
                           source_breakeven_mm=round(float(D * np.radians(np.sqrt(
                               scatter_pointing(T, AIR)[1] ** 2 + TH_REQ ** 2
                               - np.degrees(scatter_chord(T, layers=AIR)) ** 2))), 1))
           for T in (4, 8, 12, 16)}
    from scipy.optimize import brentq
    air['T_where_air_equals_requirement_MeV'] = round(float(brentq(
        lambda T: scatter_pointing(T, AIR)[1] - TH_REQ, 5, 30)), 1)
    air['kinematics'] = {f'E{E:g}': dict(min_opening_deg=round(float(np.degrees(2 * np.arcsin(16.8 / E))), 1),
                                         dm_per_deg_MeV=round(float(16.8 / (2 * np.tan(np.arcsin(16.8 / E)))
                                                                    * np.radians(1.0)), 3))
                         for E in (20.58, 18.15)}
    axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, fontsize=ps.BASE_PT, fontweight='bold', va='top', ha='left')
    axb.text(-0.28, 1.04, '(b)', transform=axb.transAxes, fontsize=ps.BASE_PT, fontweight='bold', va='top',
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
        two_estimators=te,
        air_only=air,
        panel_a=drawn,
        toy_directions=dict(n=a.n, origins_mm=list(ORIGINS), layers='air only', chamber_deg=TH_REQ,
                            s68_deg={f'{T}MeV': dict(local=round(v[0], 3), **{f'origin{o:g}mm': round(w, 3) for o, w in v[1].items()}) for T, v in toy2.items()},
                            analytic_deg={f'{T}MeV': dict(local=round(float(np.hypot(scatter_pointing(T, AIR)[1], TH_REQ)), 3), **{f'origin{o:g}mm': round(float(np.sqrt(np.degrees(scatter_chord(T, layers=AIR)) ** 2 + pos_deg ** 2 + np.degrees(o / D) ** 2)), 3) for o in ORIGINS}) for T in T_TOY}),
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
