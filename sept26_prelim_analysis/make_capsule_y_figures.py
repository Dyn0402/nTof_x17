#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_capsule_y_figures.py -- the figures for the capsule-height ray trace.

Reads only what `capsule_y.py` wrote (``capsule_y_fits.csv``,
``capsule_y_separated.csv``, ``capsule_y_curves.npz``): no fitting happens here,
so a figure cannot disagree with the table it is drawn beside.

    python -m sept26_prelim_analysis.make_capsule_y_figures
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

from sept26_prelim_analysis import paths                           # noqa: E402
from sept26_prelim_analysis import figstyle as fs                  # noqa: E402
from sept26_prelim_analysis import capsule_y as CY                 # noqa: E402

ARMS = CY.ARMS


def load():
    od = paths.out('capsule_y')
    R = pd.read_csv(od / 'capsule_y_fits.csv')
    S = pd.read_csv(od / 'capsule_y_separated.csv')
    meta = json.loads((od / 'capsule_y.meta.json').read_text())
    Z = np.load(od / 'capsule_y_curves.npz') if (
        od / 'capsule_y_curves.npz').exists() else None
    return od, R, S, meta, Z


# --------------------------------------------------------------------------- #
def fig_profiles(od, R, meta, Z):
    """The measurement itself: the v profile, the fit, and the CAD prediction."""
    base = R[R.variant == 'baseline'].set_index('arm')
    run = meta.get('curve_run', '')
    rows = []
    fig, axes = plt.subplots(1, len(ARMS), figsize=(fs.FULL[0], 3.9),
                             sharex=True)
    for ax, arm in zip(np.atleast_1d(axes), ARMS):
        v = Z[f'{arm}|v']
        n = Z[f'{arm}|counts']
        mu = Z[f'{arm}|mu_best']
        mn = Z[f'{arm}|mu_nominal']
        use = Z[f'{arm}|use'].astype(bool)
        r = base.loc[arm]
        # the plateau is not fitted, and the figure has to say so rather than
        # quietly drawing a curve through bins the likelihood never saw
        if (~use).any():
            ax.axvspan(v[~use].min() - 5, v[~use].max() + 5, color=fs.GRID,
                       zorder=0)
        ax.errorbar(v, n, yerr=np.sqrt(np.maximum(n, 1)), fmt='o', ms=3.0,
                    color=fs.INK, lw=0, elinewidth=0.9, zorder=4,
                    label='trigger-matched tracks')
        ax.step(v, mn, where='mid', color=fs.MUTED, lw=1.5, ls=(0, (5, 3)),
                zorder=3, label=f'CAD source, +{CY.NOMINAL_Y0:.1f} mm')
        ax.step(v, mu, where='mid', color=fs.DET_COLOR[arm], lw=2.0, zorder=5,
                label=f'ray trace, fitted {r.y0:+.1f} mm')
        ax.set_xlim(-CY.V_FID, CY.V_FID)
        # headroom for the legend, which has nowhere else to go on a shared row
        ax.set_ylim(0, 1.32 * max(n.max(), mu.max(), mn.max()))
        if arm == ARMS[0]:
            ax.set_ylabel(f'tracks / {CY.V_BIN:.0f} mm')
        fs.strip(ax)
        ax.grid(alpha=0.3)
        bad = r.chi2_ndf > 3.0
        ax.set_title(f'chamber {arm}', color=fs.DET_COLOR[arm])
        ax.text(0.5, 0.035,
                f'$\\chi^2/\\nu$ = {r.chi2_ndf:.1f}' + ('  (poor)' if bad else '')
                + f'\n$\\sigma$ = {r.sigma:.0f} mm',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=fs.BASE_PT * 0.8,
                color=fs.TRACK if bad else fs.MUTED)
        for k, vv, nn, mm, uu in zip(range(len(v)), v, n, mu, use):
            rows.append(dict(arm=arm, v_mm=vv, counts=nn, model_fitted=mm,
                             model_cad=Z[f'{arm}|mu_nominal'][k], in_fit=bool(uu)))
    np.atleast_1d(axes)[0].legend(loc='upper left', fontsize=fs.BASE_PT * 0.75,
                                  frameon=False, handlelength=1.6,
                                  borderaxespad=0.2, labelspacing=0.3)
    # one x label under the row, not three colliding ones
    fig.supxlabel('v on the strip plane (along the beam) [mm]',
                  fontsize=fs.BASE_PT, color=fs.INK)
    fs.preliminary(np.atleast_1d(axes)[-1], loc='upper right')
    fs.fig_title(fig, 'The trigger clips v where a source 30 mm up the beam '
                      'puts it, not where the CAD capsule would',
                 f'{run}, full pass. Grey: the plateau, excluded from the fit '
                 f'-- it carries no acceptance edge and no y0 information. '
                 f'The dashed curve is the same ray trace with the source at '
                 f'the CAD centroid.')
    return fs.save(fig, od / 'figures' / 'capsule_y_profiles',
                   data=pd.DataFrame(rows))


def fig_likelihood(od, R, meta, Z):
    """The profile likelihood, with the CAD position and the band on it."""
    base = R[R.variant == 'baseline'].set_index('arm')
    rows = []
    fig, ax = fs.figure(figsize=(fs.FIG[0], 4.0))
    for arm in ARMS:
        P = Z[f'{arm}|profile']
        y0, q = P[:, 0], P[:, 1]
        d = q - np.nanmin(q)
        ax.plot(y0, d, lw=2.0, **{k: v for k, v in fs.det_style(arm).items()
                                  if k != 'marker'})
        r = base.loc[arm]
        ax.plot([r.band_mm], [np.interp(r.band_mm, y0, d)],
                marker=fs.DET_MARKER[arm], ms=9, mfc=fs.SURFACE, mew=1.8,
                mec=fs.DET_COLOR[arm], color=fs.DET_COLOR[arm], lw=0,
                zorder=8)
        for yy, dd in zip(y0, d):
            rows.append(dict(arm=arm, y0_mm=yy, delta_nll=dd))
    ax.axvline(CY.NOMINAL_Y0, color=fs.MUTED, lw=1.2, ls=(0, (5, 3)))
    ax.annotate('CAD gas\ncentroid', xy=(CY.NOMINAL_Y0, 2.0), xytext=(-26, 5.0),
                fontsize=fs.BASE_PT * 0.85, color=fs.MUTED, va='center',
                ha='left', linespacing=1.2,
                arrowprops=dict(arrowstyle='->', color=fs.MUTED, lw=0.9))
    ax.axhline(0.5, color=fs.LINE, lw=0.9)
    ax.text(88, 0.55, '1$\\sigma$', color=fs.MUTED, ha='right', va='bottom',
            fontsize=fs.BASE_PT * 0.8)
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_ylim(0, 400)
    ax.set_xlabel('source height along the beam, $y_0$ [mm]')
    ax.set_ylabel('$\\Delta$ negative log-likelihood')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(alpha=0.3)
    fs.preliminary(ax, loc='lower right')
    fs.fig_title(fig, 'Every chamber that measures anything excludes the CAD '
                      'position outright',
                 'Open markers: where the same tracks\' scale-free band '
                 'crossing sits, for comparison -- it is not a term in this '
                 'likelihood. Chamber D has no minimum worth reading.')
    return fs.save(fig, od / 'figures' / 'capsule_y_likelihood',
                   data=pd.DataFrame(rows))


def fig_separation(od, S):
    """The decisive figure: two estimators, two gains, one crossing point.

    Each chamber contributes two straight lines in the (capsule height, v-origin
    offset) plane -- one per estimator -- and they cross where both are
    satisfied.  The CAD hypothesis is the vertical at +0.8 mm, and the "it is all
    a frame error" hypothesis is the horizontal at the band's own value.
    """
    fig, ax = fs.figure(figsize=(fs.FIG[0], 4.2))
    d = np.linspace(-40, 40, 200)
    rows = []
    for r in S.itertuples():
        if r.arm not in CY.DECIDE or not np.isfinite(r.y_source):
            continue
        st = fs.det_style(r.arm)
        c = st['color']
        ax.plot(r.band_mm - r.g_band * d, d, lw=1.8, color=c,
                label=f'{r.arm}: band crossing')
        ax.plot(r.ray_mm - r.g_ray * d, d, lw=1.8, ls=(0, (4, 2.5)), color=c,
                label=f'{r.arm}: acceptance fit')
        ax.plot([r.y_source], [r.delta], marker=st['marker'], ms=9, color=c,
                mec=fs.SURFACE, mew=1.2, lw=0, zorder=6)
        rows.append(dict(arm=r.arm, y_source_mm=r.y_source, delta_mm=r.delta,
                         band_mm=r.band_mm, ray_mm=r.ray_mm,
                         g_band=r.g_band, g_ray=r.g_ray))
    ax.axvline(CY.NOMINAL_Y0, color=fs.MUTED, lw=1.2, ls=(0, (5, 3)))
    ax.axhline(0.0, color=fs.LINE, lw=1.0)
    ax.annotate('CAD capsule', xy=(CY.NOMINAL_Y0, -34), xytext=(9, -34),
                fontsize=fs.BASE_PT * 0.85, color=fs.MUTED, va='center',
                arrowprops=dict(arrowstyle='-', color=fs.MUTED, lw=0.9))
    ax.set_xlim(-20, 60)
    ax.set_ylim(-35, 35)
    ax.set_xlabel('capsule emission centroid along the beam, $y_s$ [mm]')
    ax.set_ylabel('v-origin offset $\\delta$ [mm]')
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.82, ncol=2,
              loc='upper right')
    ax.grid(alpha=0.3)
    fs.preliminary(ax, loc='lower left')
    fs.fig_title(fig, 'The 30 mm cannot be a common v-origin error: that needs '
                      '$\\delta \\approx +30$ in both chambers',
                 'Gain 1 for the band crossing, ~2 for the acceptance fit, so '
                 'the lines are not parallel. Read only what is common to the '
                 'chambers: each marker also absorbs that chamber\'s own eff(v), '
                 'so the gap between them is NOT an alignment (see the '
                 'sensitivity figure).')
    return fs.save(fig, od / 'figures' / 'capsule_y_separation',
                   data=pd.DataFrame(rows))


def fig_systematics(od, R):
    """Every knob, moved: the spread here is the error bar, not the fit error."""
    base = R[R.variant == 'baseline'].set_index('arm')
    piv = R.pivot_table(index='variant', columns='arm', values='y0',
                        aggfunc='mean')
    order = ['baseline'] + sorted(v for v in piv.index if v != 'baseline')
    piv = piv.loc[order]
    # the eff(v) tilt is the DOMINANT systematic and is a column, not a variant;
    # it has to appear here or the ladder understates the error
    for col, lab in (('y0_tilt_m10', 'eff(v) tilt &minus;10 %'),
                     ('y0_tilt_p10', 'eff(v) tilt +10 %')):
        if col in base:
            lab = lab.replace('&minus;', '−').replace('&nbsp;', ' ')
            piv.loc[lab] = base[col]
            order.append(lab)
    piv = piv.loc[order]
    fig, ax = fs.figure(figsize=(fs.FIG[0], 0.42 * len(order) + 2.1))
    y = np.arange(len(order))[::-1]
    for arm in ARMS:
        if arm not in piv:
            continue
        st = fs.det_style(arm)
        ax.plot(piv[arm].to_numpy(), y, lw=0, ms=7, **st)
        b = base.loc[arm]
        ax.axvline(b.y0, color=st['color'], lw=0.9, alpha=0.4)
        ax.errorbar([b.y0], [y[0]], xerr=[b.err_boot], fmt='none',
                    ecolor=st['color'], elinewidth=2.2, capsize=3)
    ax.axvline(CY.NOMINAL_Y0, color=fs.MUTED, lw=1.3, ls=(0, (5, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=fs.BASE_PT * 0.85)
    ax.set_xlabel('fitted source height $y_0$ [mm]')
    ax.legend(frameon=False, loc='lower right', fontsize=fs.BASE_PT * 0.85)
    ax.grid(alpha=0.3, axis='x')
    fs.preliminary(ax, loc='upper left')
    fs.fig_title(fig, 'On A and C, no knob brings the answer within 25 mm of '
                      'the CAD position',
                 'Error bars on the baseline row are the Poisson bootstrap; the '
                 'spread down each column is the systematic, and it is the '
                 'larger of the two. Dashed vertical: the CAD gas centroid. '
                 'Chamber D is on the far side of it and moves tens of '
                 'millimetres under every variant &mdash; it is shown to be '
                 'excluded, not to be averaged in.')
    return fs.save(fig, od / 'figures' / 'capsule_y_systematics', data=piv)




#: Runs with fewer tracks in the fit than this are left out of the campaign
#: figure and summary: run_126 (51) and run_128 (137) are the two below it.
MIN_TRACKS = 500


def fig_campaign(od, R):
    """Run to run: both estimators, per chamber.

    Deliberately NOT drawn: each chamber's own two-estimator "capsule height"
    (``capsule_y_separated.csv``'s ``y_source``).  An earlier version drew it and
    the two chambers' curves sat ~15 mm apart, which read as a failed consistency
    check or an A-C misalignment.  It was neither -- see :func:`fig_sensitivity`.
    """
    base = R[R.variant == 'baseline'].copy()
    base['num'] = base.run.str.extract(r'run_(\d+)').astype(int)
    if base.num.nunique() < 2:
        return None
    fig, ax = fs.figure(figsize=(fs.WIDE[0], 3.9))
    for arm in CY.DECIDE:
        st = fs.det_style(arm)
        g = base[(base.arm == arm) & (base.n_in_fit >= MIN_TRACKS)].sort_values('num')
        ax.plot(g.num, g.band_mm, lw=1.6, ms=5, color=st['color'],
                marker=st['marker'], label=f'{arm}: band crossing (the height)')
        ax.plot(g.num, g.y0, lw=1.0, ls=(0, (4, 2.5)), color=st['color'],
                alpha=0.6, label=f'{arm}: acceptance fit')
    ax.axhline(CY.NOMINAL_Y0, color=fs.MUTED, lw=1.3, ls=(0, (5, 3)))
    ax.set_xlabel('DREAM run number')
    ax.set_ylabel('source height [mm]')
    ax.set_ylim(-5, 55)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.78, loc='lower right',
              ncol=2)
    fs.preliminary(ax, loc='upper right')
    fs.fig_title(fig, 'Every run puts the source 30-40 mm up the beam, far from '
                      'the CAD centroid',
                 f'Runs with at least {MIN_TRACKS} tracks in the fit. Solid: the '
                 'band crossing, which quotes the height. Dashed: the acceptance '
                 'fit, whose chamber-to-chamber differences are eff(v), not '
                 'geometry. Dashed horizontal: the CAD gas centroid.')
    return fs.save(fig, od / 'figures' / 'capsule_y_campaign', data=base)


def fig_sensitivity(od, R, meta):
    """The evidence behind the rule: which estimator a chamber's eff(v) can move.

    Left: on the detailed run, both estimators refitted after thinning the SAME
    tracks by an imposed efficiency tilt.  Right: over the campaign, each
    chamber's response per unit tilt -- the band crossing's from thinning, the
    acceptance fit's from its pinned model tilt.
    """
    p = paths.out('capsule_y') / 'capsule_y_sensitivity.csv'
    if not p.exists():
        return None
    T = pd.read_csv(p)
    run = meta.get('curve_run', 'run_145')
    one = T[T.run == run].set_index('arm')
    tilts = [t for t in CY.TILTS]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL[0], 3.9),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    for a_ in (ax, bx):
        fs.strip(a_)
    rows = []
    for arm in CY.ARMS:
        if arm not in one.index:
            continue
        st = fs.det_style(arm)
        r = one.loc[arm]
        band = [r.band_mm if t == 0 else r[f'band_tilt_{t:+.1f}'] for t in tilts]
        ax.plot(np.array(tilts) * 100, np.array(band) - r.band_mm, lw=1.8,
                color=st['color'], marker=st['marker'], ms=5,
                label=f'{arm}: band crossing')
        if 'ray_mm' in r and np.isfinite(r.get('ray_mm', np.nan)):
            ray = [r.ray_mm if t == 0 else r[f'ray_tilt_{t:+.1f}'] for t in tilts]
            ax.plot(np.array(tilts) * 100, np.array(ray) - r.ray_mm, lw=1.2,
                    ls=(0, (4, 2.5)), color=st['color'], marker=st['marker'],
                    ms=5, mfc=fs.SURFACE, mew=1.2, label=f'{arm}: acceptance fit')
            rows.append(dict(arm=arm, tilt_pct=np.array(tilts) * 100,
                             band_shift=np.array(band) - r.band_mm,
                             ray_shift=np.array(ray) - r.ray_mm))
    ax.axhline(0, color=fs.LINE, lw=1.0)
    ax.set_xlabel('imposed eff(v) tilt across $\\pm$170 mm [%]')
    ax.set_ylabel('shift in fitted height [mm]')
    ax.set_title(f'{run}: the same tracks, thinned', loc='left',
                 fontsize=fs.BASE_PT)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.72, ncol=2,
              loc='upper left')

    # campaign: |mm per unit tilt| for both estimators
    base = R[R.variant == 'baseline'].copy()
    base['ray_per_tilt'] = (base.y0_tilt_p10 - base.y0_tilt_m10) / 0.2
    M = T.merge(base[['run', 'arm', 'ray_per_tilt', 'n_in_fit']],
                on=['run', 'arm'], suffixes=('_thin', ''))
    M = M[M.n_in_fit >= MIN_TRACKS]
    x = np.arange(len(CY.ARMS))
    for i, arm in enumerate(CY.ARMS):
        g = M[M.arm == arm]
        st = fs.det_style(arm)
        jit = np.linspace(-0.12, 0.12, max(len(g), 1))
        bx.plot(i - 0.2 + jit, np.abs(g.band_per_tilt), lw=0, marker=st['marker'], zorder=3,
                ms=4.5, color=st['color'], alpha=0.8)
        bx.plot(i + 0.2 + jit, np.abs(g.ray_per_tilt), lw=0, marker=st['marker'],
                ms=4.5, mfc=fs.SURFACE, mew=1.1, color=st['color'], alpha=0.8)
        for off, col in ((-0.2, 'band_per_tilt'), (0.2, 'ray_per_tilt')):
            bx.plot([i + off - 0.17, i + off + 0.17],
                    [np.abs(g[col]).median()] * 2, color=fs.INK, lw=2.0)
    bx.set_yscale('log')
    bx.set_xticks(x)
    bx.set_xticklabels([f'chamber {a}\nband  |  fit' for a in CY.ARMS],
                       fontsize=fs.BASE_PT * 0.85)
    bx.set_ylabel('|shift| per 100 % tilt [mm]')
    bx.set_title('every run: filled = band crossing, open = acceptance fit',
                 loc='left', fontsize=fs.BASE_PT)
    bx.grid(alpha=0.3, axis='y')
    fs.preliminary(bx, loc='upper right')
    fs.fig_title(fig, 'A chamber\'s own efficiency along v moves the acceptance '
                      'fit and leaves the band crossing alone',
                 'So the acceptance fit\'s chamber-to-chamber differences are '
                 'eff(v), and a per-chamber split of the two estimators is not an '
                 'alignment. Positions come from pointing (the band crossing), '
                 'not from the shape of a distribution. Bars: campaign medians.')
    data = M[['run', 'arm', 'band_per_tilt', 'ray_per_tilt', 'band_mm']]
    return fs.save(fig, od / 'figures' / 'capsule_y_sensitivity', data=data)


def main() -> int:
    fs.use()
    od, R, S, meta, Z = load()
    made = []
    # every figure but the campaign ones describes the detailed run alone: the
    # variant grid and the stored curves exist only for it, and a campaign-wide
    # table indexed by chamber would have one row per run per chamber
    run = meta.get('curve_run', 'run_145')
    R1, S1 = R[R.run == run], S[S.run == run]
    if Z is not None:
        made.append(fig_profiles(od, R1, meta, Z))
        made.append(fig_likelihood(od, R1, meta, Z))
    made.append(fig_separation(od, S1))
    made.append(fig_systematics(od, R1))
    for f in (fig_campaign(od, R), fig_sensitivity(od, R, meta)):
        if f is not None:
            made.append(f)
    for p in made:
        print(f'  -> {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
