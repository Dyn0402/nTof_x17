#!/usr/bin/env python3
"""make_pointing.py -- Paper I v2, Fig. 2: what the angle of a short segment far from the target is worth.

    .venv/bin/python papers/paper1_v2/figs/make_pointing.py [--out papers/paper1_v2/figs/out/pointing] [--n 200000]

Toy (no data are read). Geometry along the chamber normal (mcs_common): target at 0, window at 200 mm, 60 um Mylar,
5.0 mm gas gap, cathode (50 um Kapton + 9 um Cu, assumed), drift gas 28.2 mm (measured), readout plane at
d = 233.3 mm, the plane where the position is measured.

(a) One chamber and the 3He volume to scale, with one lepton whose deflection is exaggerated (8 deg drawn). Two
    estimates of its initial direction: the measured local angle (extrapolated back, it misses the origin by b), and
    the line from an assumed origin (centre of the volume) to the measured position.
(b) Error of each estimate, one projection, against the kinetic energy of the lepton. Lines are analytic
    (scatter_pointing, scatter_chord); points are scatter_toy, which throws leptons through the layers. Right axis:
    the same error as a mass error for a pair at the X17 edge with both leptons alike (sqrt(2) x dm/dtheta).
    Redrawn 2026-11-11: the earlier (b), pointing against angular resolution for four position resolutions, showed
    only eq. (1); its numbers are still written to the json (pointing_mm, toy) because the text quotes them.

Pointing toy (json only): tracks thrown isotropically from a point target into the efficient area; position smeared
by sigma_x, projected angle by sigma_theta; miss distance = impact parameter in that projection, s68. At normal
incidence this is sqrt(sigma_x^2 + (d sigma_theta)^2); over the acceptance the path is d/cos(alpha), ~10 % more.
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
ORIGINS = (5.0, 15.0)                  # rms origin error [mm]: 3He bore across the beam, along the beam
R_BORE = 10.0                          # 3He bore radius [mm] (MX17_Full_Geant SimConfig.hh, per STEP)


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


def scatter_toy(T, n, rng, th_ch_deg=TH_CH, sx=MEAS['x_intr'], origins=ORIGINS, n_air=50):
    """Toy check of the two direction estimates, one projection, normal incidence. Each lepton starts on the axis,
    takes Gaussian kicks in the layers in front of the gas (air in n_air steps, each foil one kick; Highland on the
    total, shared by x/X0 as in scatter_pointing) and is measured at the readout plane.
    Returns s68 [deg] of (local angle - initial direction) and of (position - assumed origin)/d - initial direction,
    the second for each rms origin error in `origins`."""
    xs = np.array([mc.x_over_X0(m, t) for _, _, m, t, _, _ in mc.LAYERS])
    th = mc.highland(xs.sum(), T)
    ang = np.zeros(n)
    pos = np.zeros(n)
    z = 0.0
    for (_, _, m, t, z0, _), x in zip(mc.LAYERS, xs):
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
    cL, cO = ps.DET_COLOR['det3'], ps.ACCENT
    y0, y1 = -28.0, 150.0
    ax.add_patch(mp.Rectangle((mc.Z_DRIFT0, y0), mc.G_DRIFT, y1 - y0, fc='#e8eef8', ec='none', zorder=1))
    ax.plot([mc.D_WIN] * 2, [y0, y1], color=ink, lw=0.9, zorder=3)                 # window
    ax.plot([mc.Z_CATH] * 2, [y0, y1], color=ink, lw=0.9, zorder=3)                # cathode
    ax.add_patch(mp.Rectangle((D, y0), 1.7, y1 - y0, fc='#7f8a99', ec='none', zorder=2))   # readout board
    ax.text(mc.D_WIN + 2, y1 + 3, 'window,\ncathode', ha='right', va='bottom', fontsize=7, color=ink,
            linespacing=0.95)
    ax.text(mc.Z_DRIFT0 + mc.G_DRIFT / 2 + 2, y1 + 3, 'drift\n28 mm', ha='left', va='bottom', fontsize=7, color=ink,
            linespacing=0.95)
    # the 3He volume across the beam, to scale; the lepton starts off its centre
    ax.add_patch(mp.Circle((0, 0), R_BORE, fc='#f1f3f6', ec=ink, lw=0.8, zorder=2))
    ax.plot(0, 0, '+', ms=5, mew=0.8, color=cO, zorder=7)
    org = np.array([0.0, 6.0])
    ax.plot(*org, 'o', ms=2.6, color=ink, zorder=8)
    ax.text(0, R_BORE + 3, r'$^3$He, 20 mm', fontsize=7, ha='center', va='bottom', color=ink)
    # true path: initial direction t0, a smooth bend in the air and a kink at the foils
    t0 = 0.36
    bend_air, kink = np.radians(2.5), np.radians(5.5)
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
    # estimate 2: assumed origin (centre of the volume) to the measured position
    pm = np.array([zs.mean(), ys.mean()])
    ax.plot([0, pm[0]], [0, pm[1]], color=cO, lw=0.9, zorder=4)
    ax.plot(*pm, 'o', ms=3.0, mfc='white', mec=cO, mew=0.9, zorder=7)
    # the miss distance
    ax.annotate('', xy=(-18, yb), xytext=(-18, org[1]),
                arrowprops=dict(arrowstyle='<->', color=cL, lw=0.7, shrinkA=0, shrinkB=0))
    ax.plot([-21, 0], [yb, yb], color=cL, lw=0.5, zorder=3)
    ax.plot([-21, 0], [org[1]] * 2, color=cL, lw=0.5, zorder=3)
    ax.text(-21, (yb + org[1]) / 2, '$b$', ha='right', va='center', fontsize=8, color=cL)
    rot = lambda t: float(np.degrees(np.arctan(t)))
    kw = dict(rotation_mode='anchor', ha='center', fontsize=7)
    zt = 118
    ax.text(zt, org[1] + t0 * zt + 9, 'path, and its initial direction', rotation=rot(t0), va='bottom',
            color=ink, **kw)
    zt = 100
    ax.text(zt, yb + np.tan(a_gas) * zt - 5, 'measured angle, extrapolated', rotation=rot(np.tan(a_gas)), va='top',
            color=cL, **kw)
    ax.text(zs[0] - 5, ys[1] + 8, 'measured\nsegment', ha='right', va='bottom', fontsize=7, color=cL,
            linespacing=0.95)
    ax.annotate('assumed origin\nto measured position', xy=(150, 150 * pm[1] / pm[0]), xytext=(128, 2),
                fontsize=7, color=cO, ha='left', va='top', linespacing=0.95,
                arrowprops=dict(arrowstyle='-', color=cO, lw=0.6, shrinkA=1, shrinkB=1))
    yd = yb - 12
    ax.annotate('', xy=(D, yd), xytext=(0, yd), arrowprops=dict(arrowstyle='<->', color=mut, lw=0.7, shrinkA=0,
                                                                shrinkB=0))
    ax.text(D / 2, yd - 3, f'd = {D:.0f} mm', ha='center', va='top', fontsize=7, color=mut)
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
    axa = fig.add_axes([0.0, 0.02, 0.50, 0.96])
    axb = fig.add_axes([0.585, 0.15, 0.325, 0.80])
    drawn = side_view(axa)

    # (b) error on the initial direction of one lepton, one projection, against its kinetic energy
    cL, cO = ps.DET_COLOR['det3'], ps.ACCENT
    scat = {T: scatter_pointing(T) for T in (4, 8, 16)}
    Tf = np.linspace(3.0, 16.5, 120)
    th0 = np.array([scatter_pointing(T)[1] for T in Tf])
    chs = np.degrees([scatter_chord(T) for T in Tf])
    pos_deg = float(np.degrees(MEAS['x_intr'] / D))
    axb.axhspan(MEAS['theta_y'], MEAS['theta_x'], color=cL, alpha=0.22, lw=0, zorder=2)
    axb.plot(Tf, th0, color=ps.MUTED, lw=0.9, ls=(0, (4, 2)), zorder=3)
    axb.plot(Tf, np.hypot(th0, TH_CH), color=cL, lw=1.4, zorder=5)
    for o, ls in zip(ORIGINS, ('-', (0, (5, 2)))):
        axb.plot(Tf, np.sqrt(chs ** 2 + pos_deg ** 2 + np.degrees(o / D) ** 2), color=cO, lw=1.4, ls=ls, zorder=4)
    axb.axhline(pos_deg, color=ps.INK, lw=0.8, ls=(0, (1, 1.5)), zorder=3)
    T_TOY = (4, 6, 8, 10, 12, 14, 16)
    toy2 = {T: scatter_toy(T, a.n, rng) for T in T_TOY}
    axb.plot(T_TOY, [toy2[T][0] for T in T_TOY], ls='none', marker='o', ms=3.4, color=cL, zorder=6)
    for o, mk_ in zip(ORIGINS, ('s', 'D')):
        axb.plot(T_TOY, [toy2[T][1][o] for T in T_TOY], ls='none', marker=mk_, ms=3.0, mfc='white', mec=cO,
                 mew=0.9, zorder=6)
    lab = dict(fontsize=7, linespacing=0.95)
    axb.text(5.0, 5.0, 'measured angle', color=cL, ha='left', va='bottom', **lab)
    axb.text(16.3, 7.2, 'dashed grey: scattering alone', color=ps.MUTED, ha='right', va='top', **lab)
    axb.text(16.3, 3.95, 'origin to 15 mm', color=cO, ha='right', va='bottom', **lab)
    axb.text(3.3, 1.5, 'origin to 5 mm', color=cO, ha='left', va='top', **lab)
    axb.text(16.3, 0.52, 'chamber angle', color=cL, ha='right', va='center', **lab)
    axb.text(16.3, pos_deg + 0.08, 'chamber position', color=ps.INK, ha='right', va='bottom', **lab)
    axb.set_xlim(3.0, 16.5)
    axb.set_ylim(0, 7.5)
    axb.set_xticks([4, 8, 12, 16])
    axb.set_xlabel('lepton kinetic energy (MeV)')
    axb.set_ylabel('error on the initial direction (°)')
    te = two_estimators()
    k_m = np.sqrt(2) * te['kinematics']['dm_per_deg_MeV']          # MeV of mass per degree, both leptons alike
    axr = axb.secondary_yaxis('right', functions=(lambda v: v * k_m, lambda v: v / k_m))
    axr.set_ylabel('mass error of a pair (MeV)')
    axr.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
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
        two_estimators=te,
        panel_a=drawn,
        toy_directions=dict(n=a.n, origins_mm=list(ORIGINS), mass_MeV_per_deg=round(float(k_m), 4),
                            s68_deg={f'{T}MeV': dict(local=round(v[0], 3), **{f'origin{o:g}mm': round(w, 3) for o, w in v[1].items()}) for T, v in toy2.items()},
                            analytic_deg={f'{T}MeV': dict(local=round(float(np.hypot(scatter_pointing(T)[1], TH_CH)), 3), **{f'origin{o:g}mm': round(float(np.sqrt(np.degrees(scatter_chord(T)) ** 2 + pos_deg ** 2 + np.degrees(o / D) ** 2)), 3) for o in ORIGINS}) for T in T_TOY}),
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
