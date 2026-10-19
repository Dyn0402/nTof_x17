#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_run145_pointing.py -- the two closing Status figures, both from run_145.

[1] run145_xangle.png   (slide "every track knows where the target is")
    A schematic of the measurement beside the measurement itself: a point
    source on the beam axis at distance L can only reach the strip plane at
    position u with tan(theta) = u / L, so a scatter of reconstructed
    (u, tan theta) from a real point source must lie on a LINE THROUGH THE
    ORIGIN OF KNOWN SLOPE.  Nothing is fitted -- the line is geometry.
    Arm A (det3), the chamber with the cleanest transferred calibration.

[2] run145_overhead_AC.png  (slide "the two opposing arms cross at the capsule")
    The same statement as a picture, and with the external confirmation
    applied: arms A and C sit on opposite sides of the beam (Z = +-234.6 mm),
    so an overhead view puts their two fans nose to nose.  Each track is drawn
    from where it crossed the strip plane back to its closest approach to the
    beam axis, and ONLY tracks whose extrapolation lands on a SiPM wall segment
    AND a plastic bar that both fired in time are kept -- so the fan you see is
    the fan the scintillators agree with.

Both figures are drawn AS RECONSTRUCTED: no angle rescaling, no in-situ
velocity fit, nothing tuned to make the image converge.

Provenance.  run_145 (2026-08-09, production configuration), sub-run
stat090_0000, n_TOF partner 224670.  Tracks: the waveform-first forward fit
(`ntof_tracking.wft_beam`), tables under WFT_BEAM_ANALYSIS.  The wall/plastic
coincidence and the geometry come straight from
`ntof_tracking.run145_target_imaging` -- imported, not re-implemented, so this
figure cannot drift away from the analysis it illustrates.

Usage:  python make_run145_pointing.py [--slides]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LogNorm                              # noqa: E402
from matplotlib.patches import Circle, Rectangle, Arc              # noqa: E402

import plotstyle as P                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RUN = 'run_145'
SUBRUNS = ('stat090_0000', 'stat090_0001')
SLIM = ('/media/dylan/data/x17/slim/out_224670/'
        'ntof_hits_run_145_{sub}_224670.root')
RUN_CFG = '/media/dylan/data/x17/beam_july/runs/run_145/run_config.json'

CAPSULE_R = 10.0            # He-3 bore radius [mm]
PLANE_HALF = 199.29         # strip plane half-width [mm]

# IN-PLANE SIGN: not here any more.  It is a property of the reconstruction,
# it was MEASURED on 2026-08-20 (run145_target_imaging.IN_PLANE_SIGN), and it
# now lives in TI.track_lines, which this figure calls.  What used to be here
# was a frozen `SIGN = -1` applied to the TAN -- which is the mirror of the
# right answer, because the plane centre is 16 mm off the beam axis.  See
# RUN145_ALIGNMENT_2026-08-20.md.

# In-situ angle scale per arm, read from the imaging summary (v_insitu =
# v_bundle / k).  One number per chamber, measured from the tracks; the place
# where the fans cross does NOT depend on it (that is the zero crossing of the
# pointing band, which is scale-free).
IMAGING = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/'
           '{sub}/imaging_fullcov/imaging_summary.json')


def k_insitu():
    """Per-arm k, averaged over the two sub-runs."""
    acc = {}
    for sub in SUBRUNS:
        for r in json.load(open(IMAGING.format(sub=sub)))['results']:
            if r.get('k_phys'):
                acc.setdefault(r['arm'], []).append(float(r['k_phys']))
    return {a: float(np.mean(v)) for a, v in acc.items()}


# --------------------------------------------------------------------- inputs
def _imaging():
    from ntof_tracking import run145_target_imaging as TI
    return TI


def arm_tracks(arm, coincidence=True, k=1.0):
    """One arm, both sub-runs concatenated, in the plotted sign convention.

    Sub-runs are read and joined to their OWN slim file separately (event_id is
    unique within a sub-run, not across), then stacked."""
    TI = _imaging()
    from ntof_tracking.reco import geometry as G

    cfg = json.load(open(RUN_CFG))
    tr = G.detector_transforms(cfg)[f'mx17_{arm}']

    d_perp, foot_x = TI.plane_geometry(tr)
    info = dict(arm=arm, sub_runs=list(SUBRUNS), n_events=0, n_2plane=0,
                n_sel=0, n_wall=0, n_predictable=0, n_coincident=0,
                k_applied=float(k), d_perp=float(d_perp),
                foot_x=float(foot_x),
                L=float(np.linalg.norm(tr.center[[0, 2]])), bundles=[])
    U, T, P0s, Ds = [], [], [], []

    for sub in SUBRUNS:
        df, meta = TI.load_tracks(RUN, sub, arm)
        df, _ = TI.apply_w0_kw(df, arm, meta['bundle']['v_drift'], meta)
        seeded = str((meta['bundle'].get('provenance') or {})
                     .get('seeded_from', ''))
        info['bundles'].append(os.path.basename(seeded))

        ok = (df['x_ok'] & df['y_ok']).to_numpy()
        sane = ((df['x_tan_theta'].abs() < TI.TAN_SANE)
                & (df['y_tan_theta'].abs() < TI.TAN_SANE)).to_numpy()
        sel = ok & sane
        info['n_events'] += int(len(df))
        info['n_2plane'] += int(ok.sum())
        info['n_sel'] += int(sel.sum())

        if coincidence:
            slim = SLIM.format(sub=sub)
            wal = TI.slim_wall_events(slim, arm)
            inwall = df['event_id'].isin(wal).to_numpy()
            # no ordering trap left: nothing is flipped here any more, the
            # in-plane sign is inside TI and the coincidence uses the same one.
            coin, ci = TI.pointing_coincidence(slim, arm, df, sel & inwall,
                                               foot_x=foot_x)
            info['n_wall'] += int((sel & inwall).sum())
            info['n_predictable'] += int(ci['n_predictable'])
            info['n_coincident'] += int(coin.sum())
            mask = coin
        else:
            mask = sel

        d2 = df.copy()
        d2['x_tan_theta'] = k * df['x_tan_theta']
        d2['y_tan_theta'] = k * df['y_tan_theta']
        p0, d = TI.track_lines(d2, tr, mask)
        P0s.append(p0)
        Ds.append(d)
        U.append(TI.local_x(df['x_p0'].to_numpy()[mask]) - foot_x)
        T.append(k * df['x_tan_theta'].to_numpy()[mask])

    return dict(u=np.concatenate(U), tan=np.concatenate(T),
                P0=np.concatenate(P0s), D=np.concatenate(Ds),
                center=tr.center, d_perp=d_perp, foot_x=foot_x,
                info=info)


# ------------------------------------------------------------------ figure 1
def _density_cmap():
    """Surface -> deck accent.  A single-hue ramp off the page colour so a
    one-track bin is barely there and the band is the only thing you see;
    magma/viridis put a saturated colour on the singles and the eye reads the
    background as structure."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'deck_density', [P.SURFACE, '#e6dcea', '#c4a8cc', '#a06aa8',
                         P.ACCENT, '#4a1f52'])


def draw_schematic(ax, L=235.2):
    """Overhead cartoon, drawn so that u runs left-right exactly as it does on
    the data panel beside it: the source sits below, the strip plane above."""
    ax.set_xlim(-252, 236)
    ax.set_ylim(-126, 316)
    ax.set_aspect('equal')
    ax.axis('off')

    U = 140.0                                   # the ray the labels hang on

    # the strip plane
    ax.add_patch(Rectangle((-PLANE_HALF, L), 2 * PLANE_HALF, 14,
                           facecolor=P.LINE, edgecolor=P.MUTED, lw=0.8,
                           zorder=3))
    ax.text(-PLANE_HALF, L + 22, 'Micromegas strip plane', ha='left',
            va='bottom', fontsize=10.5, color=P.MUTED, fontweight='bold')

    # the beam axis and the capsule on it
    ax.plot([0, 0], [-58, 318], color=P.MUTED, lw=1.0, ls=(0, (5, 4)),
            alpha=0.55, zorder=1)
    ax.add_patch(Circle((0, 0), 21.0, facecolor=P.TRACK, edgecolor='none',
                        alpha=0.9, zorder=4))
    ax.text(0, -32, '³He capsule\non the beam axis', ha='center',
            va='top', fontsize=10.5, color=P.INK, fontweight='bold',
            linespacing=1.35)

    # three rays out of it
    for u in (-U, 0.0, U):
        ax.annotate('', xy=(u * 0.985, L * 0.985), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=P.ACCENT, lw=1.7,
                                    shrinkA=8, shrinkB=0, mutation_scale=13),
                    zorder=5)

    # u, measured along the plane from the beam axis
    for x in (0.0, U):
        ax.plot([x, x], [L + 6, L + 58], color=P.MUTED, lw=0.9, ls=':',
                zorder=2)
    ax.annotate('', xy=(U, L + 46), xytext=(0, L + 46),
                arrowprops=dict(arrowstyle='<->', color=P.INK, lw=1.1))
    ax.text(U / 2, L + 53, 'u', ha='center', va='bottom', fontsize=13,
            color=P.INK, fontstyle='italic')

    # theta, between the plane normal and the ray, where the chamber sees it
    ax.plot([U, U], [L, L - 84], color=P.MUTED, lw=0.9, ls=':', zorder=2)
    a0 = np.degrees(np.arctan2(-L, -U)) % 360.0
    ax.add_patch(Arc((U, L), 116, 116, theta1=a0, theta2=270.0,
                     color=P.INK, lw=1.2, zorder=6))
    am = np.radians(0.5 * (a0 + 270.0))
    ax.text(U + 46 * np.cos(am), L + 46 * np.sin(am), r'$\theta$',
            fontsize=14, color=P.INK, ha='center', va='center')

    # L, the one number the line on the data panel is made of
    ax.annotate('', xy=(-236, L), xytext=(-236, 0),
                arrowprops=dict(arrowstyle='<->', color=P.INK, lw=1.1))
    ax.text(-228, L / 2, f'L = {L:.0f} mm', rotation=90, ha='left',
            va='center', fontsize=11.5, color=P.INK)

    ax.text(0, -108, 'a point source can only arrive as   '
                     r'$\tan\theta = u\,/\,L$',
            ha='center', va='center', fontsize=13.5, color=P.ACCENT,
            fontweight='bold')


def figure_xangle(out_base, arm='A'):
    d = arm_tracks(arm, coincidence=True)
    # u is the lever from the PERPENDICULAR FOOT and L the perpendicular
    # distance -- not the plane centre and |centre|, which is what this drew
    # before 2026-08-20 and which put a 16 mm pinwheel into the reference line.
    u, tan, L = d['u'], d['tan'], d['d_perp']

    fig = plt.figure(figsize=(13.35, 6.0))          # 2.225:1, the slide's hole
    ax_s = fig.add_axes([0.010, 0.115, 0.372, 0.845])
    ax_d = fig.add_axes([0.462, 0.150, 0.522, 0.800])

    draw_schematic(ax_s, L)

    # Confirmed tracks only: the scintillator coincidence is what separates a
    # track from the flash residue and the accidentals, and it is external --
    # the wall sits 96 mm BEHIND the strip plane and its geometry contains no
    # 235 mm, so it cannot manufacture the slope the line is drawn at.
    m = np.abs(tan) < 0.85
    ax_d.hist2d(u[m], tan[m], bins=[92, 84],
                range=[[-200, 200], [-0.85, 0.85]],
                cmap=_density_cmap(), norm=LogNorm(vmin=1, vmax=26), zorder=2)
    uu = np.array([-200.0, 200.0])
    ax_d.plot(uu, uu / L, color=P.TRACK, lw=2.2, ls=(0, (6, 4)), zorder=5,
              label=f'a point source at the capsule:   '
                    f'tan θ = u / {L:.0f} mm')
    ax_d.set_xlabel('track position on the strip plane  u  [mm]'
                    '   (from the beam-axis perpendicular)')
    ax_d.set_ylabel(r'reconstructed  tan $\theta$   (x plane)')
    ax_d.set_xlim(-200, 200)
    ax_d.set_ylim(-0.85, 0.85)
    ax_d.legend(loc='upper left', fontsize=12, framealpha=0.92,
                facecolor=P.SURFACE, frameon=True, edgecolor='none')
    ax_d.grid(alpha=0.3)
    P.strip(ax_d)

    n = int(m.sum())
    ax_d.text(0.984, 0.032,
              f'arm A · {n:,} tracks the scintillators confirm',
              transform=ax_d.transAxes, ha='right', va='bottom',
              fontsize=11, color=P.MUTED, fontweight='bold',
              bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=2.5))

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', bbox_inches=fig.bbox_inches,
                    pad_inches=0.0)
    plt.close(fig)
    print(f'  -> {out_base}.png   n={n:,}  L={L:.1f} mm')
    return d['info']


# ------------------------------------------------------------------ figure 2
def _fan(ax, d, color, lw, alpha):
    """Draw each track from the strip plane back to its closest approach."""
    P0, D = d['P0'], d['D']
    p, dd = P0[:, [0, 2]], D[:, [0, 2]]
    s = -np.einsum('ij,ij->i', p, dd) / np.einsum('ij,ij->i', dd, dd)
    s = np.clip(s, 0.0, None)
    seg = np.empty((len(P0), 2, 2))
    seg[:, 0, 0], seg[:, 0, 1] = P0[:, 2], P0[:, 0]
    end = P0 + s[:, None] * D
    seg[:, 1, 0], seg[:, 1, 1] = end[:, 2], end[:, 0]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=3))
    return end


def _x_at_target(d):
    """Global X where each track crosses the target plane Z = 0.

    This is the back-projection the whole figure is about, and it is what arms
    A and C measure: they sit on the +-Z axis, so they resolve X and are
    DEGENERATE in Z. Hence a 1-D X distribution here and not a 2-D image."""
    P0, D = d['P0'], d['D']
    t = -P0[:, 2] / D[:, 2]
    return P0[:, 0] + t * D[:, 0]


def _stage(ax, arms):
    """The capsule and the two full strip planes, in the overhead frame."""
    for arm, d in arms.items():
        cx, cz = d['center'][0], d['center'][2]
        sgn = 1.0 if cz > 0 else -1.0
        ax.add_patch(Rectangle((cz - sgn * 5.0, cx - PLANE_HALF),
                               sgn * 14.0, 2 * PLANE_HALF,
                               facecolor=P.LINE, edgecolor=P.MUTED, lw=0.8,
                               zorder=4))
    ax.add_patch(Circle((0, 0), CAPSULE_R, facecolor='none',
                        edgecolor=P.TRACK, lw=2.2, zorder=9))


def figure_overhead(out_base):
    """Left: the whole station from above, both planes end to end, the two
    opposing fans. Right: what those fans actually measure -- the back-
    projected X of every confirmed track, from each arm separately."""
    K = k_insitu()
    arms = {a: arm_tracks(a, coincidence=True, k=K.get(a, 1.0))
            for a in ('A', 'C')}
    col = {a: P.DET_COLOR[a] for a in arms}

    fig = plt.figure(figsize=(13.35, 6.0))          # 2.225:1, the slide's hole
    H = 0.885
    wl = H * 6.0 / 13.35 * (510.0 / 436.0)          # left panel, equal aspect
    wr = H * 6.0 / 13.35 * 1.02
    axL = fig.add_axes([0.012, 0.075, wl, H])
    axR = fig.add_axes([0.012 + wl + 0.088, 0.135, wr, H - 0.075])

    # ---------------------------------------------------------- left: overhead
    _stage(axL, arms)
    for a, d in arms.items():
        _fan(axL, d, col[a], 0.35, 0.030)
    axL.set_xlim(-255, 255)
    axL.set_ylim(-218, 218)
    axL.set_aspect('equal')
    axL.set_facecolor(P.SURFACE)
    axL.axis('off')
    for a, x, ha in (('C', -222, 'left'), ('A', 222, 'right')):
        axL.text(x, 210, f'arm {a}', ha=ha, va='top', fontsize=13,
                 fontweight='bold', color=col[a])
    axL.annotate('the ³He capsule, seen end-on', xy=(0, CAPSULE_R),
                 xytext=(0, 74), fontsize=11.5, color=P.INK,
                 fontweight='bold', ha='center', va='bottom',
                 arrowprops=dict(arrowstyle='-', color=P.INK, lw=1.0,
                                 shrinkB=4))
    axL.plot([-246, -146], [-212, -212], color=P.INK, lw=2.4, zorder=10,
             solid_capstyle='butt')
    axL.text(-196, -208, '100 mm', ha='center', va='bottom', fontsize=10,
             color=P.INK, zorder=10)
    n = {a: arms[a]['info']['n_coincident'] for a in arms}
    axL.text(246, -212, f"{n['A'] + n['C']:,} confirmed tracks · "
                        f"strip planes end to end",
             ha='right', va='bottom', fontsize=10.5, color=P.MUTED,
             fontweight='bold', zorder=10)

    # ------------------------------------------- right: what they measure in X
    axR.axvspan(-CAPSULE_R, CAPSULE_R, color=P.TRACK, alpha=0.13, zorder=1)
    axR.axvline(0, color=P.MUTED, lw=0.9, ls=(0, (5, 4)), zorder=2)
    stats = {}
    for a, d in arms.items():
        x = _x_at_target(d)
        h, e = np.histogram(x, bins=90, range=(-160, 160))
        c = 0.5 * (e[:-1] + e[1:])
        axR.step(c, h, where='mid', color=col[a], lw=2.0, zorder=4,
                 label=f'arm {a}  ({len(x):,})')
        core = x[np.abs(x) < 90]
        stats[a] = dict(peak=float(c[np.argmax(h)]), med=float(np.median(core)),
                        n=int(len(x)),
                        in_bore=float(np.mean(np.abs(x) < CAPSULE_R)))
    axR.set_xlim(-160, 160)
    axR.set_xlabel('back-projected position at the target,  global X  [mm]')
    axR.set_ylabel('confirmed tracks / 3.6 mm')
    axR.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.92,
               facecolor=P.SURFACE, edgecolor='none')
    axR.grid(alpha=0.3)
    P.strip(axR)
    axR.text(0.5, 1.028, 'the ³He bore, r = 10 mm', transform=axR.transAxes,
             ha='center', va='bottom', fontsize=10.5, color=P.TRACK,
             fontweight='bold')
    axR.text(0.985, 0.60,
             'two opposing arms,\nfitted independently:\n'
             f"A  {stats['A']['med']:+.1f} mm\nC  {stats['C']['med']:+.1f} mm",
             transform=axR.transAxes, ha='right', va='top', fontsize=11,
             color=P.INK, linespacing=1.5,
             bbox=dict(facecolor=P.SURFACE, edgecolor=P.LINE, pad=5.0))

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', bbox_inches=fig.bbox_inches,
                    pad_inches=0.0)
    plt.close(fig)
    print(f'  -> {out_base}.png   A={n["A"]:,}  C={n["C"]:,}  k={K}')
    print(f'     X at target: {stats}')
    out = {a: arms[a]['info'] for a in arms}
    out['x_at_target'] = stats
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--slides', action='store_true',
                    help='write into slides/assets/img/ as well')
    a = ap.parse_args()

    P.use()
    outs = [os.path.join(HERE, 'figures')]
    if a.slides:
        outs.append(os.path.join(HERE, 'slides', 'assets', 'img'))
    for o in outs:
        os.makedirs(o, exist_ok=True)

    info = {}
    info['xangle'] = figure_xangle(os.path.join(outs[0], 'run145_xangle'))
    info['overhead'] = figure_overhead(os.path.join(outs[0],
                                                    'run145_overhead_AC'))
    for o in outs[1:]:
        for n in ('run145_xangle', 'run145_overhead_AC'):
            for ext in ('png',):
                src = os.path.join(outs[0], f'{n}.{ext}')
                dst = os.path.join(o, f'{n}.{ext}')
                with open(src, 'rb') as f, open(dst, 'wb') as g:
                    g.write(f.read())
                print(f'  -> {dst}')

    with open(os.path.join(outs[0], 'run145_pointing.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))


if __name__ == '__main__':
    main()
