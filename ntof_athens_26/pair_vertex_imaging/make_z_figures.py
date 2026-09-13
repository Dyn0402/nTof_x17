#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_z_figures.py -- the figures for the z follow-up (`z_image.py`).

Draws from ``pairs_z[_ac_aligned].parquet``, ``z_band.npz``,
``z_pointing_hist.csv`` and ``z_fits[_ac_aligned].csv``; the fit model is
re-evaluated at the stored parameters, never refitted.  Written as
``z_<name>.png`` + ``.pdf`` into ``figures/``.

    python -m pair_vertex_imaging.make_z_figures
    python -m pair_vertex_imaging.make_z_figures --only bands,summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
import plotstyle as P  # noqa: E402
from pair_vertex_imaging import vertex_image as VI  # noqa: E402
from pair_vertex_imaging import z_image as Z  # noqa: E402
from pair_vertex_imaging.make_image_figures import DIV, SEQ, foot, save  # noqa: E402

OUT = HERE / 'figures'
FOOT = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full pass, '
        'stage-3 tracks, chambers A, C, D')
TIER_LABEL = {'all': 'all gated', 'reliable': 'slope measured (|tan| ≥ 0.08)',
              'clean': '+ not in a noisy column',
              'confirmed': '+ own scintillators fired'}
TIER_COLOR = {'all': '#9aa3ad', 'reliable': '#56B4E9', 'clean': '#0072B2',
              'confirmed': '#1b2430'}
CUT = 60.0
OTHER = 'clean'   # the A/C leg's tier when the D leg's is varied


def read_fits(od, tag=''):
    p = od / Z.names(tag)['fits']
    return pd.read_csv(p, keep_default_na=False, na_values=['']) if p.exists() else None


def frow(F, sel, tier, coord, cut=CUT, smin=0.0, tier_other=None):
    tier_other = tier if tier_other is None else tier_other
    r = F[(F.selection == sel) & (F.tier == tier) & (F.tier_other == tier_other)
          & (F.coord == coord) & (F.cut_mm == cut)
          & np.isclose(F.sin_psi_min.astype(float), smin)]
    return r.iloc[0] if len(r) else None


def has_fit(fr) -> bool:
    return fr is not None and 'c' in fr and np.isfinite(fr.c)


# --------------------------------------------------------------------------- #
def bands(od):
    B = np.load(od / 'z_band.npz')
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    (x0, x1), (t0, t1) = Z.BAND_RANGE
    for i, arm in enumerate(Z.ARMS):
        for j, tier in enumerate(Z.TIERS):
            H = sum((B[f'{arm}_{c}'] for c in range(8)
                     if f'{arm}_{c}' in B and Z.tier_ok(np.array([c]), tier)[0]),
                    np.zeros(Z.BAND_BINS))
            ax = axes[i, j]
            ax.imshow(H.T + 0.5, origin='lower', aspect='auto',
                      extent=(x0, x1, t0, t1), cmap=SEQ,
                      norm=LogNorm(vmin=0.5, vmax=max(H.max(), 2)),
                      interpolation='nearest')
            ax.set_title(f'{arm} · {TIER_LABEL[tier]}\n{int(H.sum()):,} tracks',
                         loc='left', fontsize=10.5, color=P.INK)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 2:
                ax.set_xlabel('impact position on the strip plane, x_local  [mm]')
            if j == 0:
                ax.set_ylabel(f'{arm}: in-plane tan θ (calibrated)')
    fig.suptitle('The pointing band, and what each cut removes', x=0.01,
                 ha='left', fontsize=15, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    foot(fig, 'Every gated, angle-calibrated track of the campaign, log colour. '
         'A track from the capsule lies on the diagonal band; the two lobes are '
         'the trigger’s plastic-bar gap.  Horizontal line at tan ≈ 0: slopes '
         'the drift timing could not measure (|tan| < 0.08).  Vertical stripes: '
         'noisy readout columns.  D’s x_local is −z, so D’s stripes are fixed z '
         'values.   ' + FOOT)
    save(fig, 'z_bands')


def pointing(od):
    PH = pd.read_csv(od / 'z_pointing_hist.csv')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    for ax, arm in zip(axes, Z.ARMS):
        g = PH[PH.arm == arm]
        for tier in Z.TIERS:
            h = g[Z.tier_ok(g.code.to_numpy(), tier)]
            hd = h.groupby('lo').data.sum()
            hn = h.groupby('lo').null.sum()
            lo = hd.index.to_numpy()
            hi = np.r_[lo[1:], Z.DCA_EDGES[len(lo)]]
            w = hi - lo
            col = TIER_COLOR[tier]
            ax.stairs(hd.to_numpy() / hd.sum() / w, np.r_[lo, hi[-1]], color=col,
                      lw=2.0, label=TIER_LABEL[tier])
            ax.stairs(hn.to_numpy() / hn.sum() / w, np.r_[lo, hi[-1]], color=col,
                      lw=1.0, ls=(0, (4, 2.5)))
        ax.set_xlim(0, 200)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('single-track miss distance from the beam axis  [mm]')
        ax.set_title(f'chamber {arm}', loc='left', fontsize=13,
                     fontweight='bold', color=P.INK)
        P.strip(ax)
    axes[0].set_ylabel('density  [per mm]')
    axes[0].plot([], [], color=P.MUTED, lw=1.0, ls=(0, (4, 2.5)),
                 label='dashed: same tracks, directions shuffled')
    axes[0].legend(fontsize=8.5, loc='upper right')
    fig.suptitle('How much each cut sharpens a single track’s pointing', x=0.01,
                 ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    foot(fig, 'A peak at small miss distance above its dashed curve is pointing '
         'information.  The shuffle keeps every impact point and every angle and '
         'destroys only which angle goes with which position, within the same '
         'cut.   ' + FOOT)
    save(fig, 'z_pointing')


# --------------------------------------------------------------------------- #
def transverse(d, cap, F):
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    e = np.arange(-64, 66, 2.0)
    for ax, tier in zip(axes, Z.TIERS):
        g = Z.select(d, 'perpendicular', CUT, tier, tier_other=OTHER)
        D, N = g[~g.null], g[g.null]
        fx = frow(F, 'perpendicular', tier, 'x', tier_other=OTHER)
        f = float(fx.f) if fx is not None and 'f' in fx and np.isfinite(fx.f) else 0.0
        scale = (1 - f) * len(D) / max(len(N), 1)
        H = (np.histogram2d(D.vx_xz, D.vz_xz, bins=(e, e))[0]
             - np.histogram2d(N.vx_xz, N.vz_xz, bins=(e, e))[0] * scale)
        v = max(np.nanpercentile(np.abs(H), 99.5), 1)
        im = ax.imshow(H.T, origin='lower', extent=(e[0], e[-1], e[0], e[-1]),
                       cmap=DIV, norm=TwoSlopeNorm(0, -v, v), interpolation='nearest')
        ax.add_patch(Circle(cap, 10, fill=False, color=P.COPPER, lw=1.6))
        ax.plot(0, 0, 'o', mfc='none', color=P.MUTED, ms=6)
        ax.set_aspect('equal')
        ax.set_title(f'D leg: {TIER_LABEL[tier]}\n{len(D):,} pairs', loc='left',
                     fontsize=10.5, color=P.INK)
        ax.set_xlabel('x  [mm]  (A or C leg)')
        if tier == 'all':
            ax.set_ylabel('z  [mm]  (D leg)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7.5, colors=P.MUTED)
        cb.outline.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f'Perpendicular pairs, data − null, as the D leg is cleaned  '
                 f'(A/C leg: slope measured, not noisy; legs within {CUT:.0f} mm)',
                 x=0.01, ha='left', fontsize=13.5, fontweight='bold', color=P.INK)
    # the footer wraps to four lines here; leave it room below the x labels
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    foot(fig, 'Transverse crossings minus the shuffled null through the same '
         'cuts, scaled by that selection’s x-fit background fraction.  Purple: '
         'excess; blue: deficit.  Copper: the capsule at the single-track '
         'position.  The excess is a vertical stripe at the capsule’s x that '
         'runs the length of z, strongest where D is live (its readout is dead '
         'from z ≈ −57 to 0 mm): the A/C leg points, the D leg contributes '
         'little more than where it hit D.  Confirming the D leg leaves too few '
         'pairs to see anything.   ' + FOOT)
    save(fig, 'z_transverse_tiers')


def _profile_panel(ax, g, coord, fr, hue, title):
    col = 'vx_xz' if coord == 'x' else 'vz_xz'
    x_ = g.loc[~g.null, col].to_numpy(float)
    n_ = g.loc[g.null, col].to_numpy(float)
    if len(x_) < 50 or len(n_) < 50:
        ax.text(0.5, 0.5, 'too few pairs', transform=ax.transAxes, ha='center',
                color=P.MUTED)
        ax.set_title(title, loc='left', fontsize=10, color=P.INK)
        P.strip(ax)
        return
    M = VI.ProfileModel(x_, n_)
    x = M.centres
    c0 = float(fr.capsule) if fr is not None else 0.0
    ax.axvspan(c0 - 10, c0 + 10, color=P.COPPER, alpha=0.12, lw=0)
    ax.axvline(c0, color=P.COPPER, lw=1.0)
    ax.errorbar(x, M.n, np.sqrt(M.n), fmt='o', ms=2.8, color=hue, lw=0.9,
                label='data')
    ax.plot(x, M.N * M.B, color=P.MUTED, lw=1.2, ls=(0, (4, 2.5)),
            label='null, same cuts')
    if has_fit(fr):
        S = M.N * fr.f * M.source(fr.c, fr.s)
        B = M.N * (1 - fr.f) * M.B
        ax.plot(x, S + B, color=P.INK, lw=1.6, label='fit, centre free')
        ax.fill_between(x, B, S + B, color=hue, alpha=0.18, lw=0)
        sub = (f'centre {fr.c:+.1f} ± {fr.c_err:.1f}   σ {fr.s:.0f}   '
               f'term {100 * fr.f:.0f} %   2ΔlnL {fr.two_dnll_vs_none:.0f}   '
               f'χ²/ndf {fr.chi2 / max(fr.ndf, 1):.1f}')
    elif fr is not None and 'f' in fr:
        sub = 'the fit wants no capsule term'
    else:
        sub = 'too few pairs to fit'
    ax.set_xlim(-64, 64)
    ax.set_ylim(bottom=0)
    ax.set_title(f'{title}\n{sub}', loc='left', fontsize=9.5, color=P.INK)
    P.strip(ax)


def perp_profiles(d, F):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for j, tier in enumerate(Z.TIERS):
        g = Z.select(d, 'perpendicular', CUT, tier, tier_other=OTHER)
        for i, coord in enumerate(('z', 'x')):
            fr = frow(F, 'perpendicular', tier, coord, tier_other=OTHER)
            _profile_panel(axes[i, j], g, coord,
                           fr, '#CC79A7' if coord == 'z' else '#0072B2',
                           f'{coord} · D leg: {TIER_LABEL[tier]}')
            axes[i, j].set_xlabel(f'{coord} of the crossing  [mm]')
    for i in range(2):
        axes[i, 0].set_ylabel('pairs per 2 mm')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Perpendicular pairs: z (top, from the D leg) and x (bottom, '
                 'from the A/C leg), as the D leg is cleaned', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, f'A/C leg: slope measured and not in a noisy column; both legs '
         f'within {CUT:.0f} mm.  Fit: the He-3 gas projected onto the coordinate '
         '⊗ a Gaussian + the null shape, centre, blur and fraction free.  Copper: '
         'the single-track capsule position ± 10 mm.   ' + FOOT)
    save(fig, 'z_perp_profiles')


def parallel_profiles(d, F, tier='clean'):
    sels = ('A-A', 'C-C', 'A-C', 'D-D')
    smins = (0.0, 0.5)
    fig, axes = plt.subplots(len(smins), len(sels), figsize=(18, 8.6))
    for i, smin in enumerate(smins):
        for j, sel in enumerate(sels):
            g = Z.select(d, sel, CUT, tier, smin)
            fr = frow(F, sel, tier, 'z', smin=smin)
            hue = '#CC79A7' if sel == 'D-D' else '#009E73'
            _profile_panel(axes[i, j], g, 'z', fr, hue,
                           f'{sel}, sin ψ ≥ {smin:.1f} · {int((~g.null).sum()):,} pairs')
            axes[i, j].set_xlabel('z of the crossing  [mm]')
        axes[i, 0].set_ylabel('pairs per 2 mm')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'z from pairs within A and C, and from D–D  '
                 f'({TIER_LABEL[tier]} on both legs, within {CUT:.0f} mm)',
                 x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, 'A–A and C–C lines both run along z, so their crossing locates x and '
         'hardly z.  A–C lines come from opposite sides and their crossing does '
         'locate z, through the difference of the two chambers’ x and slopes.  '
         'D–D lines run along x, so their crossing locates z and hardly x — '
         'that column is chamber D again.   ' + FOOT)
    save(fig, 'z_parallel_profiles')


# --------------------------------------------------------------------------- #
def aligned(d0, d1, F0, F1, cap):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    for ax, (d, F, lab) in zip(axes[:2], ((d0, F0, 'as reconstructed'),
                                          (d1, F1, 'A and C shifted onto their common x'))):
        g = Z.select(d, 'A-C', CUT, 'clean', 0.3)
        _profile_panel(ax, g, 'z', frow(F, 'A-C', 'clean', 'z', smin=0.3),
                       '#009E73', f'A–C, sin ψ ≥ 0.3, clean · {lab}')
        ax.set_xlabel('z of the crossing  [mm]')
    axes[0].set_ylabel('pairs per 2 mm')
    axes[0].legend(fontsize=8, loc='upper left')
    ax = axes[2]
    ks = []
    for k, (tier, smin) in enumerate([(t, s) for t in ('all', 'clean', 'confirmed')
                                      for s in Z.SIN_PSI_BINS]):
        ks.append(f'{tier} · sin ψ ≥ {smin:.1f}')
        for F, off, col, lab in ((F0, -0.15, P.MUTED, 'as reconstructed'),
                                 (F1, 0.15, '#009E73', 'A/C aligned')):
            fr = frow(F, 'A-C', tier, 'z', smin=smin)
            if has_fit(fr):
                ax.errorbar(fr.c, k + off, xerr=fr.c_err, fmt='o', ms=5, color=col,
                            label=lab if k == 0 else None)
    ax.axvline(cap[1], color=P.COPPER, lw=1.2)
    ax.axvspan(cap[1] - 10, cap[1] + 10, color=P.COPPER, alpha=0.08, lw=0)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-40, 30)
    ax.set_xlabel('fitted A–C z centre  [mm]   (copper: single tracks, chamber D)')
    ax.set_title('Every A–C z fit, both ways', loc='left', fontsize=11.5,
                 color=P.INK)
    ax.legend(fontsize=8.5, loc='lower right')
    P.strip(ax)
    fig.suptitle('Aligning A and C onto their common x does not move the A–C z '
                 'image', x=0.01, ha='left', fontsize=14, fontweight='bold',
                 color=P.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    foot(fig, 'Right-hand rebuild: A and C moved in x (each by ~1 mm, opposite '
         'directions) so that their single-track band crossings coincide.  That '
         'the 2 mm disagreement is a placement offset is a hypothesis being '
         'tested, not a correction.   ' + FOOT)
    save(fig, 'z_aligned')


# --------------------------------------------------------------------------- #
def summary(F, cap):
    rows = [('perp', sel, tier, OTHER, 0.0) for tier in Z.TIERS
            for sel in Z.PERP_SELECTIONS]
    rows += [('par', sel, tier, tier, smin) for tier in ('all', 'clean', 'confirmed')
             for sel in Z.PAR_SELECTIONS for smin in (0.0, 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 14), sharey=True)
    ylab = []
    for k, (kind, sel, tier, other, smin) in enumerate(rows):
        lab = (f'{sel} · D leg {tier}' if kind == 'perp'
               else f'{sel} · {tier}' + (f' · sin ψ ≥ {smin:.1f}' if smin else ''))
        ylab.append(lab)
        for ax, coord in zip(axes, ('z', 'x')):
            fr = frow(F, sel, tier, coord, smin=smin, tier_other=other)
            col = TIER_COLOR[tier]
            if not has_fit(fr):
                if fr is not None and 'f' in fr:
                    ax.text(-60, k, 'no capsule term wanted', va='center',
                            fontsize=7.5, color=P.MUTED, style='italic')
                continue
            filled = fr.two_dnll_vs_none > 25
            ax.errorbar(fr.c, k, xerr=fr.c_err if np.isfinite(fr.c_err) else None,
                        fmt='o', ms=6, color=col, mfc=col if filled else 'white',
                        lw=1.2)
            ax.text(50, k, f'σ{fr.s:4.0f} f{100 * fr.f:3.0f}% '
                           f'ΔL{fr.two_dnll_vs_none:6.0f} '
                           f'χ²{fr.chi2 / max(fr.ndf, 1):4.1f}',
                    va='center', fontsize=7, color=P.MUTED, family='monospace')
    for ax, coord, c in zip(axes, ('z', 'x'), (cap[1], cap[0])):
        ax.axvline(c, color=P.COPPER, lw=1.2)
        ax.axvspan(c - 10, c + 10, color=P.COPPER, alpha=0.08, lw=0)
        ax.set_xlim(-64, 90)
        ax.set_xlabel(f'fitted {coord} centre  [mm]   (copper: single tracks)')
        ax.set_title(coord, loc='left', fontsize=13, fontweight='bold', color=P.INK)
        P.strip(ax)
        ax.grid(axis='y', color=P.LINE, lw=0.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(ylab, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle(f'Every fitted image centre, centre free  (legs within '
                 f'{CUT:.0f} mm; filled: 2ΔlnL > 25)', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    foot(fig, 'Perpendicular rows: the D leg’s cut varies, the A/C leg is always '
         '"slope measured, not noisy".  ΔL = 2ΔlnL against no capsule term; '
         'χ² = χ²/ndf.  A fit far from the capsule with a large χ² is chasing '
         'structure the null does not describe.   ' + FOOT)
    save(fig, 'z_summary')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    want = set(x for x in a.only.split(',') if x)
    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    P.use()
    F0 = read_fits(od)
    F1 = read_fits(od, 'ac_aligned')
    wants = lambda k: not want or k in want  # noqa: E731
    d0 = Z.load(od) if any(wants(k) for k in ('transverse', 'perp', 'parallel', 'aligned')) else None
    if wants('bands'):
        bands(od)
    if wants('pointing'):
        pointing(od)
    if wants('transverse'):
        transverse(d0, cap, F0)
    if wants('perp'):
        perp_profiles(d0, F0)
    if wants('parallel'):
        parallel_profiles(d0, F0)
    if wants('aligned') and F1 is not None:
        d1 = Z.load(od, 'ac_aligned')
        aligned(d0, d1, F0, F1, cap)
        del d1
    if wants('summary'):
        summary(F0, cap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
