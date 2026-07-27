#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27g_pointing_geometry.py — realistic-geometry pointing display + before/after
calibration convergence on the beam axis, for run_55 micro-TPC tracks.

Inputs (already produced by 27b/27c):
  calib/27_tracks.npz  — 762 matched X/Y 3-D segments (det, xcen/ycen strip pos,
                         xslope/yslope = dt/du, headon flags, resist_v)
  calib/27_align.json  — per-plane source-hypothesis calibration:
                         u0 (alignment), scale_sR (charge-sharing angle scale),
                         dist_pos/dist_dtan (fringe-field distortion map), v_nom, R

Geometry (from MX17_Full_Geant DetectorConstruction.cc):
  Pinwheel of 4 single-micro-TPC chambers in the HORIZONTAL (X-Z) plane, beam
  VERTICAL (Y).  Arm order 0=D(+X) 1=B(-X) 2=A(+Z) 3=C(-Z).  Local frame
  u=transverse(in horizontal plane), v=beam-axis(vertical Y), w=radial drift.
  x-plane strips read u (transverse); y-plane strips read v (beam axis).
  He-3 capsule (source) at the origin, D20 x L40 mm.

Angle:  tan(theta) = 1000 / (slope * v_drift)   [theta from radial normal]
Source model (27c): tan(theta) = -(scale_sR/R) * (u - u0)   -> points at u0.

Two figures:
  figures/27_tracks/14_pointing_geometry.png  (3 panels)
  figures/27_tracks/15_beamaxis_convergence.png (before/after)

Run:  venv/bin/python mx_july_beam_qa/27g_pointing_geometry.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, 'calib')
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
os.makedirs(FIGDIR, exist_ok=True)

STRIP_CENTER = 199.3          # (512-1)/2 * 0.78 mm
CAP_D, CAP_L = 20.0, 40.0     # He-3 capsule diameter (transverse) x length (beam)

# global placement of each arm's readout plane in the horizontal (X,Z) plane.
# 'axis' = which world axis the readout plane sits along (radial),
# 'sign' = +/- side, 'uworld' = world axis the transverse strip coord maps to.
ARM = {
    'A': dict(radial='Z', rsign=+1, uworld='X'),   # +Z, u along X
    'C': dict(radial='Z', rsign=-1, uworld='X'),   # -Z, u along X
    'D': dict(radial='X', rsign=+1, uworld='Z'),   # +X, u along Z
    'B': dict(radial='X', rsign=-1, uworld='Z'),   # -X, u along Z
}
DETCOL = {'A': '#4e79a7', 'B': '#59a14f', 'C': '#e15759', 'D': '#b07aa1'}


def interp_dist(al_plane, u):
    """fringe-field distortion dtan(u) from the mapped nodes (meas-model residual)."""
    p = np.asarray(al_plane['dist_pos']); dt = np.asarray(al_plane['dist_dtan'])
    return np.interp(u, p, dt)


def main():
    d = np.load(os.path.join(CALIB, '27_tracks.npz'), allow_pickle=True)
    al = json.load(open(os.path.join(CALIB, '27_align.json')))
    detn = d['detn'].astype(str)
    xcen, ycen = d['xcen'], d['ycen']
    xsl, ysl = d['xslope'], d['yslope']
    xho, yho = d['x_headon'], d['y_headon']

    # ---- per-track transverse (x) and beam-axis (y) pointing ----
    # tan measured, model, calibrated (distortion-removed, de-compressed by scale).
    rows = []
    for i in range(len(detn)):
        dn = detn[i]
        for pn, cen, sl, ho in (('x', xcen[i], xsl[i], xho[i]),
                                ('y', ycen[i], ysl[i], yho[i])):
            key = dn + pn
            e = al.get(key, {})
            if 'u0_mm' not in e:              # Dy never fit
                rows.append((i, dn, pn, cen, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            if ho or not np.isfinite(sl) or abs(sl) < 1e-6:
                rows.append((i, dn, pn, cen, np.nan, np.nan, e['u0_mm'], e['scale_sR'], e['R']))
                continue
            v = e['v_nom']; R = e['R']; u0 = e['u0_mm']; sc = e['scale_sR']
            tan_raw = 1000.0 / (sl * v)
            tan_dc = tan_raw - interp_dist(e, cen)   # distortion-corrected
            rows.append((i, dn, pn, cen, tan_dc, tan_raw, u0, sc, R))
    R_ = {}
    for name, col in zip(('idx', 'det', 'pl', 'u', 'tandc', 'tanraw', 'u0', 'sc', 'R'),
                         zip(*rows)):
        R_[name] = np.array(col, dtype=object if name in ('det', 'pl') else float)

    # ---- lock the pointing sign empirically ----
    # endpoint transverse (relative to source) at r=0 for the CALIBRATED angle:
    #   E = (u - u0) + sigma * R * (tandc / scale)
    # correct sigma minimizes the spread of E (rays converge on the source).
    def endpoints(sigma, plane, use_scale=True, use_u0=True, use_dc=True):
        m = (R_['pl'] == plane) & np.isfinite(R_['tandc'].astype(float))
        u = R_['u'][m].astype(float); u0 = R_['u0'][m].astype(float)
        sc = R_['sc'][m].astype(float); Rr = R_['R'][m].astype(float)
        tan = R_['tandc'][m].astype(float)
        if not use_scale:
            sc = np.ones_like(sc)
        ref = u0 if use_u0 else STRIP_CENTER
        return m, (u - ref) + sigma * Rr * (tan / sc)

    stds = {}
    for sg in (+1, -1):
        _, E = endpoints(sg, 'x')
        stds[sg] = np.nanstd(E)
    SIGMA = min(stds, key=stds.get)
    print(f'pointing sign sigma={SIGMA:+d} (std x-endpoints +1:{stds[+1]:.1f} '
          f'-1:{stds[-1]:.1f} mm)')

    # =====================================================================
    # FIGURE 1 — realistic geometry + tracks, 3 panels
    # =====================================================================
    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, wspace=0.26)
    axT = fig.add_subplot(gs[0]); axS = fig.add_subplot(gs[1]); axR = fig.add_subplot(gs[2])

    # ---------- panel (a): TOP VIEW (horizontal X-Z), transverse pointing ----------
    Rnom = 234.3
    half = STRIP_CENTER  # ~200 mm half strip length for drawing the chamber bar
    for dn, a in ARM.items():
        rad, rs, uw = a['radial'], a['rsign'], a['uworld']
        # chamber readout bar: spans +-half along uworld at radial distance Rnom
        if rad == 'Z':
            axT.plot([-half, half], [rs*Rnom, rs*Rnom], color=DETCOL[dn], lw=4, solid_capstyle='butt')
            axT.text(0, rs*(Rnom+26), f'{dn}', color=DETCOL[dn], ha='center',
                     va='center', fontsize=13, fontweight='bold')
        else:
            axT.plot([rs*Rnom, rs*Rnom], [-half, half], color=DETCOL[dn], lw=4, solid_capstyle='butt')
            axT.text(rs*(Rnom+26), 0, f'{dn}', color=DETCOL[dn], ha='center',
                     va='center', fontsize=13, fontweight='bold')

    # draw a subsample of calibrated track rays per chamber
    m, Ex = endpoints(SIGMA, 'x')                    # transverse endpoint rel. source
    idxs = np.where(m)[0]
    rng = np.random.default_rng(0)
    for dn in 'ACDB':
        a = ARM[dn]; rad, rs, uw = a['radial'], a['rsign'], a['uworld']
        sel = [k for k in idxs if R_['det'][k] == dn]
        if len(sel) > 90:
            sel = list(rng.choice(sel, 90, replace=False))
        for k in sel:
            u = float(R_['u'][k]); u0 = float(R_['u0'][k]); sc = float(R_['sc'][k])
            Rr = float(R_['R'][k]); tan = float(R_['tandc'][k])
            p_ro = u - u0                              # transverse offset at readout
            p_src = p_ro + SIGMA * Rr * (tan / sc)     # transverse at r=0
            # world coords: readout at radius Rnom, source-cross at radius 0
            if rad == 'Z':
                x0, z0 = p_ro, rs*Rnom
                x1, z1 = p_src, 0.0
            else:
                x0, z0 = rs*Rnom, p_ro
                x1, z1 = 0.0, p_src
            axT.plot([x0, x1], [z0, z1], color=DETCOL[dn], lw=0.35, alpha=0.30)
    # He-3 capsule at origin (transverse footprint ~ D20)
    axT.add_patch(Circle((0, 0), CAP_D/2, color='k', zorder=5))
    axT.add_patch(Circle((0, 0), CAP_D/2, facecolor='gold', edgecolor='k', zorder=6))
    axT.text(0, -34, u'³He\ncapsule', ha='center', va='top', fontsize=8)
    axT.set_xlim(-290, 290); axT.set_ylim(-290, 290); axT.set_aspect('equal')
    axT.set_xlabel('world X  [mm]'); axT.set_ylabel('world Z  [mm]')
    axT.set_title('(a) top view — transverse pointing\ncalibrated track rays (beam ⊙ vertical)')
    axT.grid(alpha=0.15)

    # ---------- panel (b): SIDE VIEW (radial r vs beam Y), beam-axis pointing ----------
    # collapse all chambers onto one radial side; y-plane calibration (no Dy).
    my, Ey = endpoints(SIGMA, 'y')
    idy = np.where(my)[0]
    # chamber bar at r=Rnom spanning beam-axis strip length
    axS.plot([Rnom, Rnom], [-half, half], color='0.35', lw=4, solid_capstyle='butt')
    axS.text(Rnom+8, half*0.85, 'readout\n(all arms)', fontsize=8, va='top')
    ysrc = []
    for k in (idy if len(idy) <= 400 else rng.choice(idy, 400, replace=False)):
        u = float(R_['u'][k]); u0 = float(R_['u0'][k]); sc = float(R_['sc'][k])
        Rr = float(R_['R'][k]); tan = float(R_['tandc'][k]); dn = R_['det'][k]
        y_ro = u - STRIP_CENTER                        # beam-axis pos at readout (abs)
        y_src = (u - u0) + SIGMA * Rr * (tan / sc) + (u0 - STRIP_CENTER)
        ysrc.append(y_src)
        axS.plot([Rnom, 0.0], [y_ro, y_src], color=DETCOL[dn], lw=0.35, alpha=0.30)
    ysrc = np.array(ysrc)
    ymed = np.nanmedian(ysrc)
    # He-3 capsule side profile: L40 along beam (Y), D20 transverse(radial) at r~0
    axS.add_patch(Rectangle((-CAP_D/2, -CAP_L/2), CAP_D, CAP_L, facecolor='gold',
                            edgecolor='k', zorder=6))
    axS.axhline(0, color='k', lw=0.8, ls=':')
    axS.axhline(ymed, color='crimson', lw=1.4, ls='--',
                label=f'recon source Y = {ymed:+.0f} mm')
    axS.set_xlim(-70, Rnom+55); axS.set_ylim(-260, 260)
    axS.set_xlabel('radial distance from beam axis  r  [mm]')
    axS.set_ylabel('beam axis  Y  [mm]   (↑ toward neck)')
    axS.set_title('(b) side view — beam-axis pointing\nrays converge below centre (source low in Y)')
    axS.legend(loc='lower left', fontsize=8); axS.grid(alpha=0.15)

    # ---------- panel (c): pointing at the target (transverse vs beam-axis miss) ----------
    # per matched 3-D segment: transverse source coord and beam-axis source coord
    # (both relative to chamber centre), overlay capsule.
    seg_t, seg_y, seg_c = [], [], []
    for i in range(len(detn)):
        dn = detn[i]
        ex = al.get(dn+'x', {}); ey = al.get(dn+'y', {})
        if 'u0_mm' not in ex or 'u0_mm' not in ey:
            continue
        if xho[i] or yho[i] or not (np.isfinite(xsl[i]) and np.isfinite(ysl[i])):
            continue
        if abs(xsl[i]) < 1e-6 or abs(ysl[i]) < 1e-6:
            continue
        # transverse source coord (abs, rel chamber centre)
        tanx = 1000.0/(xsl[i]*ex['v_nom']) - interp_dist(ex, xcen[i])
        t_src = (xcen[i]-ex['u0_mm']) + SIGMA*ex['R']*(tanx/ex['scale_sR']) + (ex['u0_mm']-STRIP_CENTER)
        tany = 1000.0/(ysl[i]*ey['v_nom']) - interp_dist(ey, ycen[i])
        y_src = (ycen[i]-ey['u0_mm']) + SIGMA*ey['R']*(tany/ey['scale_sR']) + (ey['u0_mm']-STRIP_CENTER)
        seg_t.append(t_src); seg_y.append(y_src); seg_c.append(DETCOL[dn])
    seg_t = np.array(seg_t); seg_y = np.array(seg_y)
    axR.axhline(0, color='k', lw=0.6, ls=':'); axR.axvline(0, color='k', lw=0.6, ls=':')
    axR.add_patch(Rectangle((-CAP_D/2, -CAP_L/2), CAP_D, CAP_L, facecolor='gold',
                            edgecolor='k', alpha=0.85, zorder=5, label=u'³He capsule (⌀20×40)'))
    axR.scatter(seg_t, seg_y, s=10, c=seg_c, alpha=0.5, edgecolors='none')
    ct, cy = np.median(seg_t), np.median(seg_y)
    axR.plot(ct, cy, 'kx', ms=13, mew=3, zorder=8)
    axR.plot(ct, cy, 'wx', ms=9, mew=1.5, zorder=9,
             label=f'median ({ct:+.0f}, {cy:+.0f}) mm')
    axR.set_xlim(-160, 160); axR.set_ylim(-200, 160); axR.set_aspect('equal')
    axR.set_xlabel('transverse source coord  [mm]')
    axR.set_ylabel('beam-axis source coord  Y  [mm]')
    axR.set_title(f'(c) reconstructed origin at the target\n{len(seg_t)} matched 3-D segments')
    axR.legend(loc='lower left', fontsize=8); axR.grid(alpha=0.15)

    fig.suptitle('run_55 micro-TPC — realistic geometry & source pointing '
                 '(in-situ calibrated; b1/b2 windows)', y=1.02, fontsize=13)
    f1 = os.path.join(FIGDIR, '14_pointing_geometry.png')
    fig.savefig(f1, dpi=130, bbox_inches='tight'); print('wrote', f1)

    # =====================================================================
    # FIGURE 2 — before/after calibration on the beam axis.
    # Single-track pointing is resolution-limited (~11 deg -> ~80 mm at R), so
    # calibration does NOT shrink the width; what it fixes is (i) the BIAS
    # (centroid lands on the source) and (ii) the position-dependent fringe-field
    # DISTORTION (miss vs strip position goes flat).  Show both honestly.
    # =====================================================================
    def crossing(plane, u0mode):
        """beam-axis/transverse crossing rel. source per track + its strip pos u.
        u0mode 'raw'  -> nominal (u0=STRIP_CENTER, raw angle, no dist, no scale)
        u0mode 'cal'  -> fitted u0, distortion removed, de-compressed by scale."""
        m = (R_['pl'] == plane) & np.isfinite(R_['tandc'].astype(float))
        u = R_['u'][m].astype(float); u0 = R_['u0'][m].astype(float)
        sc = R_['sc'][m].astype(float); Rr = R_['R'][m].astype(float)
        tandc = R_['tandc'][m].astype(float)
        tanraw = R_['tanraw'][m].astype(float)
        if u0mode == 'raw':
            # nominal geometry: raw angle WITH distortion, no scale, no fitted u0
            ref = STRIP_CENTER
            return u, (u - ref) + SIGMA * Rr * tanraw, m
        ref = u0
        return u, (u - ref) + SIGMA * Rr * (tandc / sc), m

    def med_profile(u, e, edges):
        c = 0.5*(edges[:-1]+edges[1:]); med = np.full(c.size, np.nan)
        for i in range(c.size):
            s = (u >= edges[i]) & (u < edges[i+1])
            if s.sum() >= 6:
                med[i] = np.median(e[s])
        return c, med

    fig2, ax = plt.subplots(1, 2, figsize=(13, 4.9))
    edges = np.linspace(40, 360, 17)
    # ----- (a) beam-axis pointing miss vs strip position -----
    ub, eb, _ = crossing('y', 'raw'); ua, ea, _ = crossing('y', 'cal')
    ax[0].axhline(0, color='k', lw=0.8, ls=':')
    ax[0].axvspan(70, 330, color='0.9', zorder=0, label='fiducial 70–330')
    ax[0].scatter(ub, eb, s=6, color='#c0c0c0', alpha=0.4)
    ax[0].scatter(ua, ea, s=6, color='#e15759', alpha=0.35)
    cb, mb = med_profile(ub, eb, edges); ca, ma = med_profile(ua, ea, edges)
    ax[0].plot(cb, mb, '-o', color='0.35', lw=2, ms=4, label='before (raw): median')
    ax[0].plot(ca, ma, '-o', color='#a11', lw=2, ms=4, label='after (calib): median')
    ax[0].set_ylim(-260, 260)
    ax[0].set_xlabel('beam-axis strip position u  [mm]')
    ax[0].set_ylabel('beam-axis pointing miss  [mm]')
    ax[0].set_title('(a) miss vs position — calibration removes the\n'
                    'slope (angle scale) + edge fringe-field bending')
    ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.15)

    # ----- (b) beam-axis crossing distribution, before vs after -----
    bins = np.linspace(-260, 260, 53)
    ebf = eb[np.isfinite(eb)]; eaf = ea[np.isfinite(ea)]
    ax[1].hist(ebf, bins=bins, histtype='stepfilled', color='#b8b8b8', alpha=0.55,
               label=f'before: med {np.median(ebf):+.0f}, σ {np.std(ebf):.0f} mm')
    ax[1].hist(eaf, bins=bins, histtype='step', color='#a11', lw=2.2,
               label=f'after:  med {np.median(eaf):+.0f}, σ {np.std(eaf):.0f} mm')
    ax[1].axvline(0, color='k', ls=':', lw=1, label='source (beam axis)')
    ax[1].axvline(np.median(eaf), color='#a11', ls='--', lw=1.2)
    ax[1].set_xlabel('beam-axis crossing rel. source  [mm]')
    ax[1].set_ylabel('tracks'); ax[1].legend(fontsize=8, loc='upper left')
    ax[1].set_title('(b) beam-axis crossing — centroid moves onto source\n'
                    '(width is single-track resolution ~11°, not improved here)')
    ax[1].grid(alpha=0.15)
    fig2.suptitle('run_55 micro-TPC — beam-axis pointing, before/after in-situ '
                  'calibration  (alignment fit in-sample: shows bias/distortion '
                  'removal, not an independent validation)', y=1.02, fontsize=11)
    fig2.tight_layout()
    f2 = os.path.join(FIGDIR, '15_beamaxis_convergence.png')
    fig2.savefig(f2, dpi=130, bbox_inches='tight'); print('wrote', f2)

    # ---- console summary ----
    print(f'\nBeam-axis miss median: before {np.median(ebf):+.0f} -> after {np.median(eaf):+.0f} mm')
    print(f'Beam-axis miss σ:      before {np.std(ebf):.0f} -> after {np.std(eaf):.0f} mm '
          '(resolution-limited, not expected to shrink)')
    print(f'Target-plane median (transverse, beam-Y) = ({ct:+.0f}, {cy:+.0f}) mm '
          f'[capsule D20xL40 at origin]')


if __name__ == '__main__':
    main()
