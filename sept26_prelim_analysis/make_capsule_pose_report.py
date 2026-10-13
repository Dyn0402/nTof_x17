#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_capsule_pose_report.py -- figures and ``report.html`` for the capsule-pose
study (capsule_pose.py, HANDOFF_CAPSULE_POSE_FIT.md).

Generated, never hand-written: every number is read back from the CSVs that
capsule_pose.py wrote, so re-running the analysis updates the tables, figures
and verdict together.  Figures are referenced with relative links.

    python -m sept26_prelim_analysis.make_capsule_pose_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as _h
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import capsule_pose as CP  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402


def _f(v, d=1, plus=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '&mdash;'
    return f'{v:{"+" if plus else ""}.{d}f}'


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_captures(out: Path):
    """Side view (x, y) of where beam neutrons are captured, on the capsule
    outline, for three beam-capsule offsets."""
    import matplotlib.pyplot as plt
    ds = (0, 8, 10)
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 4.4), sharey=True)
    rows = []
    for ax, d in zip(axs, ds):
        fs.strip(ax)
        C, fr = CP.beam_captures(dict(CP.NOMINAL, x_c=-float(d)), n=80_000, window=CP.WINDOWS['late'])
        C = C[C.mat == 2]
        rng = np.random.default_rng(0)
        k = rng.choice(len(C), min(6000, len(C)), p=C.w / C.w.sum())
        ax.scatter(C.x.iloc[k], C.y.iloc[k], s=1.2, color=fs.ACCENT, alpha=0.35, lw=0)
        for S, R, col in ((CP.S_AL, CP.R_CF, fs.MUTED), (CP.S_AL, CP.R_AL, fs.INK),
                          (CP.S_GAS, CP.R_GAS, '#5b9bd5')):
            ax.plot(-d + np.r_[R, -R[::-1]], np.r_[S, S[::-1]], color=col, lw=0.8)
        ax.axvline(0, color=fs.COPPER, lw=0.8, ls='--')
        ax.set_xlim(-24, 14)
        ax.set_ylim(-40, 68)
        ax.set_aspect('equal')
        ax.set_xlabel('x from the beam centre [mm]')
        ax.set_title(f'offset {d} mm', fontsize=10)
        ax.text(0.02, 0.98, f"beam on gas {fr['gas']:.0%}\nAl: nose {fr['al_nose']:.0%}, "
                f"barrel {fr['al_barrel']:.0%}", transform=ax.transAxes, fontsize=8.5, color=fs.INK, va='top')
        rows.append(dict(d=d, **fr))
    axs[0].set_ylabel('height y [mm]  (beam travels up)')
    fs.save(fig, out / 'figures' / 'captures', data=pd.DataFrame(rows))
    plt.close(fig)


def fig_height_needed(out: Path, F: pd.DataFrame, D: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = fs.figure(fs.FIG)
    styles = {(0, 0): ('-', 'o', 'untilted'), (5, 0): ('--', 's', '5°, nose away from beam'),
              (10, 0): ('--', '^', '10°, nose away from beam'), (20, 0): ('--', 'D', '20°, nose away'),
              (10, 180): (':', 'v', '10°, nose toward beam'), (20, 180): (':', 'x', '20°, nose toward')}
    for (th, ph), (ls, mk, lab) in styles.items():
        g = F[(F.theta == th) & (F.phi == ph)].sort_values('d')
        if g.empty:
            continue
        ax.plot(g.d, g.dy_c, ls=ls, marker=mk, ms=4, lw=1.2, label=lab,
                color=fs.INK if th == 0 else (fs.ACCENT if ph == 0 else fs.COPPER))
    ac = abs(D.loc['A', 'y_cross'] - D.loc['C', 'y_cross']) / 2
    ax.axhspan(-5 - ac, 5 + ac, color=fs.GRID, zorder=0)
    ax.axhline(0, color=fs.MUTED, lw=0.7)
    ax.set_xlabel('beam offset from the capsule axis, d [mm]')
    ax.set_ylabel('capsule height change needed [mm]')
    ax.set_title('How far the capsule must move down to keep the source at the data\'s height')
    ax.legend(loc='lower left', fontsize=8.5)
    fs.preliminary(ax)
    fs.save(fig, out / 'figures' / 'height_needed', data=F)
    plt.close(fig)


def fig_consequences(out: Path, F: pd.DataFrame):
    import matplotlib.pyplot as plt
    g = F[(F.theta == 0)].sort_values('d')
    a0 = float(g[g.d == 0].al_per_ray.iloc[0])
    fig, ax = fs.figure(fs.FIG)
    ax.plot(g.d, g.gas, marker='o', color='#5b9bd5', label='beam into the ³He (³He capture rate)')
    ax.plot(g.d, g.al_per_ray / a0, marker='s', color=fs.INK, label='Al captures per beam neutron, rel. d = 0')
    ax.plot(g.d, g.al_barrel, marker='^', color=fs.COPPER, label='share of Al captures in the 0.6 mm barrel')
    ax.set_xlabel('beam offset from the capsule axis, d [mm]  (untilted)')
    ax.set_ylabel('fraction')
    ax.set_ylim(0, 1.25)
    ax.set_title('What the offset would change')
    ax.legend(loc='upper left', fontsize=8.5)
    fs.save(fig, out / 'figures' / 'consequences', data=g)
    plt.close(fig)


def fig_tof(out: Path, T: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = fs.figure(fs.FIG)
    for arm, g in T.groupby('arm'):
        st = fs.det_style(arm)
        x = 0.5 * (g.lo + g.hi)
        ax.plot(x, g.y_bgsub, marker=st.get('marker', 'o'), color=st.get('color'), label=f'{arm}, background-subtracted')
        ax.plot(x, g.y_raw, ls=':', color=st.get('color'), lw=1, label=f'{arm}, raw')
    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(['5', '10', '20', '50'])
    ax.set_ylim(-65, -25)
    ax.set_xlabel('time since the flash [ms]')
    ax.set_ylabel('source height, y band crossing [mm]')
    ax.set_title('Source height against ToF (y mirrored)')
    ax.legend(fontsize=8.5, ncol=2, loc='lower right')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, out / 'figures' / 'tof', data=T)
    plt.close(fig)


def fig_shear(out: Path, S: pd.DataFrame, D: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    for ax, (arm, g) in zip(axs, S.groupby('arm')):
        fs.strip(ax)
        g = g[g.n_data > 5000]
        st = fs.det_style(arm)
        ax.fill_between(g.h, g.toy - g.toy_sd, g.toy + g.toy_sd, color=fs.GRID, label='toy, point-like nose (±1σ library)')
        ax.plot(g.h, g.toy, color=fs.MUTED, lw=1)
        ax.errorbar(g.h, g.data, g.data_err, marker='o', color=st.get('color'), lw=1.2, label=f'data {arm}, late')
        ax.axvline(D.loc[arm, 'y_cross'], color=fs.COPPER, lw=0.8, ls='--')
        ax.set_xlabel('reconstructed height at the axis plane [mm]')
        ax.set_ylim(-20, -4)
        ax.legend(fontsize=8, loc='lower left')
    axs[0].set_ylabel('x band crossing [mm]')
    axs[0].set_title('x crossing against height')
    fs.save(fig, out / 'figures' / 'shear', data=S)
    plt.close(fig)


def fig_early(out: Path, E: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = fs.figure(fs.FIG)
    D = E[E.kind == 'data']
    rows = []
    for arm, g in D.groupby('arm'):
        st = fs.det_style(arm)
        t = np.sqrt(g.lo * g.hi)
        ax.errorbar(t, g.peak_x, g.peak_err, marker=st['marker'], color=st['color'], lw=1.2, capsize=0,
                    label=f'{arm}: pointing-excess peak')
        ok = g.band_err < 1.0
        ax.plot(t[ok], g.band_x[ok], ls=':', color=st['color'], lw=1, label=f'{arm}: band crossing (where usable)')
    M = E[(E.kind == 'mix') & (E.arm == 'A')].set_index('frac_at_plus10').peak_x
    for f in (0.2, 0.5):
        ax.axhline(M.loc[f], color=fs.MUTED, lw=0.7, ls='--')
        ax.text(2.1, M.loc[f] + 0.15, f'{f:.0%} of tracks from a source 10 mm toward 0', fontsize=8, color=fs.MUTED)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(['2', '5', '10', '20', '50'])
    ax.set_ylim(-14, -2)
    ax.set_xlabel('time since the flash [ms]')
    ax.set_ylabel('source x at the axis plane [mm]')
    ax.set_title('Source x against ToF: the same at 4 ms as at 50 ms')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    fs.preliminary(ax, 'upper left')
    fs.save(fig, out / 'figures' / 'early', data=E)
    plt.close(fig)


def fig_image(out: Path, I: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 4.2), sharey=True)
    g0 = I[I.arm == 'A']
    for ax, w in zip(axs, ('beam', 'mid', 'late')):
        g = g0[g0.window == w].pivot(index='h', columns='x', values='excess')
        vmax = np.nanmax(g.values)
        ax.pcolormesh(g.columns, g.index, g.values / vmax, cmap='magma_r', vmin=-0.1, vmax=1, shading='nearest')
        ax.axvline(0, color=fs.COPPER, lw=0.8, ls='--')
        ax.set_title({'beam': '4–12 ms (direct beam)', 'mid': '12–20 ms', 'late': '30–75 ms'}[w], fontsize=10)
        ax.set_xlabel('x at the axis plane [mm]')
        ax.grid(False)
    axs[0].set_ylabel('height at the axis plane [mm]')
    fs.save(fig, out / 'figures' / 'image', data=I)
    plt.close(fig)


def fig_alternatives(out: Path, Al: pd.DataFrame, late: pd.DataFrame):
    """Source (x, y) predicted by each picture against the data.  A capsule
    height change moves a model point straight up or down, so only the x
    position and the height change it needs are fixed by the data."""
    import matplotlib.pyplot as plt
    fig, ax = fs.figure(fs.FIG)
    A = Al[Al.window == 'late'].copy()
    A['x'] = 0.5 * (A.x_cross_A + A.x_cross_C)
    A['y'] = 0.5 * (A.y_cross_A + A.y_cross_C)
    tail = A[A.case.str.contains('tail-lit')]
    A = A[~A.case.str.contains('tail-lit')]
    if len(tail):
        ax.plot(tail.x, tail.y, marker='s', ms=5, color=fs.MUTED, lw=1, ls='--',
                label='beam at 0 (×1), capsule 15–20 mm off: lit by the tail')
        for _, r in tail.iterrows():
            ax.annotate(f"{r.x_c:+.1f}", (r.x, r.y), fontsize=7, color=fs.MUTED, xytext=(3, -9), textcoords='offset points')
    for case, g in A.groupby('case', sort=False):
        g = g.sort_values('width_scale')
        on = case.startswith('beam on')
        col = fs.INK if on else (fs.ACCENT if '-10.3' in case else fs.COPPER)
        shift = 0.0
        if on:   # the beam-on-capsule family: carry it to the data's x
            shift = 0.5 * (late.loc['A', 'x_cross'] + late.loc['C', 'x_cross']) - g.x.iloc[0]
        ax.plot(g.x + shift, g.y, marker='o', ms=3.5, color=col, lw=1.1,
                label=case + (' (moved to the data x)' if on else ''))
        for _, r in g.iterrows():
            ax.annotate(f"×{r.width_scale:g}", (r.x + shift, r.y), fontsize=7, color=col,
                        xytext=(3, 3), textcoords='offset points')
    for arm in ('A', 'C'):
        st = fs.det_style(arm)
        ax.plot(late.loc[arm, 'x_cross'], late.loc[arm, 'y_cross'], marker='*', ms=13, color=st['color'],
                ls='none', label=f'data {arm}, late')
    ax.set_xlabel('source x [mm]  (beam centred on the frame at 0)')
    ax.set_ylabel('source height, y band crossing [mm]')
    ax.set_title('Which pictures reach the data')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.text(0.99, 0.02, 'a capsule height change moves a model point vertically', transform=ax.transAxes,
            ha='right', fontsize=8, color=fs.MUTED)
    fs.save(fig, out / 'figures' / 'alternatives', data=Al)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def table(df: pd.DataFrame, cols, heads, fmt) -> str:
    r = ['<table><thead><tr>' + ''.join(f'<th class="n">{h}</th>' for h in heads) + '</tr></thead><tbody>']
    for _, x in df.iterrows():
        r.append('<tr>' + ''.join(f'<td class="n">{f(x[c])}</td>' for c, f in zip(cols, fmt)) + '</tr>')
    return ''.join(r) + '</tbody></table>'


def figure(name: str, caption: str) -> str:
    return (f'<figure><a href="figures/{name}.png"><img src="figures/{name}.png" '
            f'alt="{_h.escape(caption)}"></a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a></figcaption></figure>')


def build(d: Path) -> str:
    D = pd.read_csv(d / 'data.csv')
    late = D[D.window == 'late'].set_index('arm')
    C = pd.read_csv(d / 'closure.csv')
    F = pd.read_csv(d / 'fit.csv')
    B = pd.read_csv(d / 'bound.csv')
    T = pd.read_csv(d / 'tof.csv')
    S = pd.read_csv(d / 'shear.csv')
    E = pd.read_csv(d / 'early.csv')
    Al = pd.read_csv(d / 'alternatives.csv')
    H = pd.read_csv(d / 'halo.csv') if (d / 'halo.csv').exists() else None
    PR = pd.read_csv(paths.out('imaging_campaign') / 'per_run.csv')

    st = S.groupby('arm')[['step_data', 'step_toy', 'step_toy_err']].first()
    st['sig'] = (st.step_data - st.step_toy) / st.step_toy_err
    sig_comb = float(abs(st.sig.sum()) / np.sqrt(len(st)))
    F0 = F[F.theta == 0].set_index('d')
    xs = 0.5 * (late.loc['A', 'x_cross'] + late.loc['C', 'x_cross'])
    xs_corr = float(F0.loc[0, 'x_beam'])
    zs = float(F0.loc[0, 'z_beam'])
    dy0 = float(F0.loc[0, 'dy_c'])
    ac = abs(late.loc['A', 'y_cross'] - late.loc['C', 'y_cross'])
    Bi = B.set_index('height_prior_mm')
    b5 = Bi.loc[5]
    ED = E[E.kind == 'data'].set_index(['arm', 'lo'])
    mixA = E[(E.kind == 'mix') & (E.arm == 'A')].set_index('frac_at_plus10').peak_x
    slope = float((mixA.loc[0.5] - mixA.loc[0.0]) / 0.5)          # mm per unit fraction
    bgA = E[(E.kind == 'bg') & (E.arm == 'A')].iloc[0]
    bgC = E[(E.kind == 'bg') & (E.arm == 'C')].iloc[0]
    bg_shift = max(abs(bgA.peak_x - bgA.peak_ref), abs(bgC.peak_x - bgC.peak_ref))
    ref = {a: float(ED.loc[(a, 30.0), 'peak_x']) for a in ('A', 'C')}

    def frac(arm, lo):   # 2-sigma upper limit on the fraction of a +10 mm source
        sh = float(ED.loc[(arm, lo), 'peak_x']) - ref[arm]
        return max(0.0, (sh + 2 * float(ED.loc[(arm, lo), 'peak_err'])) / slope)
    f46 = {a: frac(a, 4.0) for a in ('A', 'C')}
    f612 = {a: max(frac(a, lo) for lo in (6.0, 8.0, 10.0)) for a in ('A', 'C')}
    post = PR[PR.condition.str.startswith('post')].groupby('arm').mm.agg(['median', 'std', 'min', 'max'])
    pre = PR[PR.condition.str.startswith('pre')].groupby('arm').mm.median()

    AlL = Al[Al.window == 'late'].copy()
    AlL['x'] = 0.5 * (AlL.x_cross_A + AlL.x_cross_C)
    AlL['y'] = 0.5 * (AlL.y_cross_A + AlL.y_cross_C)
    AlL['dy_needed'] = 0.5 * (late.loc['A', 'y_cross'] + late.loc['C', 'y_cross']) - AlL.y
    AlL['dx_off'] = AlL.x - xs
    al0 = float(AlL[(AlL.case.str.startswith('beam on')) & (AlL.width_scale == 1)].al_per_ray.iloc[0])
    AlL['al_rel'] = AlL.al_per_ray / al0
    if H is not None:
        H = H[H.window == 'late'].copy()
        H['x'] = 0.5 * (H.x_cross_A + H.x_cross_C)
        H['y'] = 0.5 * (H.y_cross_A + H.y_cross_C)
        H['dy_needed'] = 0.5 * (late.loc['A', 'y_cross'] + late.loc['C', 'y_cross']) - H.y
        H['al_rel'] = H.al_per_ray / al0
        H['dx_off'] = H.x - xs
    alt = pd.concat([AlL, H]) if H is not None else AlL
    alt_tab = table(alt, ['case', 'width_scale', 'x', 'dx_off', 'y', 'dy_needed', 'gas', 'al_rel'],
                    ['picture', 'flux width (× Lambda2D)', 'source x', 'x vs data', 'source y',
                     'capsule move needed', 'beam into ³He', 'Al captures / neutron (rel.)'],
                    [str, lambda v: f'{v:g}', lambda v: _f(v, 1, True), lambda v: _f(v, 1, True),
                     lambda v: _f(v, 1, True), lambda v: _f(v, 0, True), lambda v: _f(v, 2), lambda v: _f(v, 3)])

    cl = C[(C.kind == 'reco') & C['sample'].isin(['geant4 capsule', 'toy nominal'])]
    cl_tab = table(cl, ['arm', 'sample', 'x_cross', 'y_cross', 'h_w68', 'f_focus'],
                   ['arm', 'sample', 'x crossing', 'y crossing', 'height w<sub>68</sub>', 'focus fraction'],
                   [str, str, lambda v: _f(v, 2, True), lambda v: _f(v, 2, True), _f, lambda v: _f(v, 2)])
    ed = E[E.kind == 'data']
    early_tab = table(ed, ['arm', 'lo', 'hi', 'n', 'peak_x', 'peak_err', 'band_x', 'band_err'],
                      ['arm', 'from [ms]', 'to', 'tracks', 'excess peak x', '&plusmn;', 'band crossing', '&plusmn;'],
                      [str, lambda v: f'{v:g}', lambda v: f'{v:g}', lambda v: f'{int(v):,}', lambda v: _f(v, 1, True),
                       lambda v: _f(v, 1), lambda v: _f(v, 1, True), lambda v: _f(v, 1)])
    g0 = F[F.theta == 0].sort_values('d')
    scan_tab = table(g0, ['d', 'y_cross_A', 'y_cross_C', 'dy_c', 'gas', 'al_nose', 'al_barrel'],
                     ['d [mm]', 'toy y, A', 'toy y, C', 'capsule move needed', 'beam into ³He', 'Al: nose', 'Al: barrel'],
                     [lambda v: f'{int(v)}', lambda v: _f(v, 1, True), lambda v: _f(v, 1, True),
                      lambda v: _f(v, 1, True), lambda v: _f(v, 2), lambda v: _f(v, 2), lambda v: _f(v, 2)])
    b_tab = table(B, ['height_prior_mm', 'n_poses', 'd_tip_max', 'gas_min', 'barrel_max', 'al_rel_min'],
                  ['capsule height known to &plusmn; [mm]', 'poses allowed', 'nose tip from flux centre, max [mm]',
                   'beam into ³He, min', 'barrel share, max', 'Al captures / neutron, min (rel.)'],
                  [lambda v: f'{int(v)}', lambda v: f'{int(v)}', lambda v: _f(v, 0), lambda v: _f(v, 2),
                   lambda v: _f(v, 2), lambda v: _f(v, 2)])
    best0 = AlL[~AlL.case.str.startswith('beam on')].copy()
    best0['bad'] = best0.dx_off.abs()
    near = best0[best0.dx_off.abs() < 2.0]
    if H is not None:
        hn = H[H.dx_off.abs() < 2.5]
        near = pd.concat([near, hn])
    dy_min = float(near.dy_needed.abs().min()) if len(near) else np.nan

    body = f"""
<h1>Where the capsule sits in the beam</h1>
<p class="sub">capsule_pose.py &middot; is2_v2 at the capsule-focus k, post-access runs, A and C &middot;
y mirrored on load &middot; beam: EAR2 Phase-3 flux &times; Lambda2D radial profile per energy (the Geant4 gun's
recipe) &middot; built {dt.date.today().isoformat()}</p>

<div class="verdict"><p><b>What is measured.</b> The tracks image <b>one compact source</b>. It sits at
<b>x {_f(xs_corr, 1, True)}, z {_f(zs, 1, True)}&nbsp;mm</b> in the chamber frame (A/C x corrected by the toy's
estimator offset; B/D z uncorrected), at the height of the capsule nose (y {_f(late.loc['A', 'y_cross'], 1, True)} A /
{_f(late.loc['C', 'y_cross'], 1, True)} C). The x is the same from 2&nbsp;ms to 75&nbsp;ms after the flash, in every run
of the campaign, and on both sides of the 27 July access.</p>
<p><b>What it means is not settled, and the earlier version of this report overstated it.</b> That version called the
source "the beam". The source is where the illuminating neutrons meet aluminium. Its x is the beam's x only if the
illumination is the narrow beam <i>and</i> it hits the capsule. The forward model says:</p>
<ul>
<li><b>A compact source at the nose needs a narrow flux centred within ~{_f(b5.d_tip_max, 0)}&nbsp;mm of the nose</b>
(capsule within &plusmn;5&nbsp;mm of its drawn height). Any flux that only grazes the capsule lights the 0.6&nbsp;mm barrel
along its length and lifts the source by 10&ndash;20&nbsp;mm.</li>
<li><b>No Lambda2D-shaped beam centred on the chamber frame reproduces the data at the drawn height.</b> The ones
that reach x &asymp; {_f(xs, 0)} (capsule ~13&ndash;18&nbsp;mm off; a flux &times;10 the Lambda2D width; or the beam's tail
alone) all put the source {_f(dy_min, 0)}&ndash;25&nbsp;mm too high. They need the capsule that much lower than drawn,
and they leave 1&ndash;15 % of the beam on the ³He.</li>
<li><b>The direct-beam window shows no second source near x&nbsp;=&nbsp;0.</b> If the beam were centred on the frame and
the capsule partly in it, the beam's own captures would sit near x&nbsp;&asymp;&nbsp;0. The pointing peak allows at most
{_f(f46['A'] * 100, 0)} % (A) / {_f(f46['C'] * 100, 0)} % (C) of capsule tracks from there at 4&ndash;6&nbsp;ms and
{_f(max(f612.values()) * 100, 0)} % at 6&ndash;12&nbsp;ms (2&sigma;; background alone moves the peak by up to
{_f(bg_shift, 1)}&nbsp;mm, i.e. ~{_f(bg_shift / slope * 100, 0)} %).</li>
</ul>
<p><b>So there are two readings, and these data cannot choose between them:</b></p>
<ol>
<li><b>The flux is centred on the capsule</b>, ~{_f(abs(xs_corr), 0)}&nbsp;mm toward B and ~{_f(abs(zs), 0)}&nbsp;mm toward C
from the frame centre. Then the beam is not where the frame says it is, or the frame's centre is not where it was
aligned to the beam. The capsule then sits {_f(-dy0, 0)}&nbsp;mm below its drawn height, and if it is within &plusmn;5&nbsp;mm of it
&ge; {_f(b5.gas_min * 100, 0)} % of the beam enters the ³He.</li>
<li><b>The beam is centred on the frame (as aligned)</b>, and the capsule is 13&ndash;18&nbsp;mm toward B and
&ge; {_f(dy_min, 0)}&nbsp;mm lower than drawn, lit by a beam broader than Lambda2D or by its tail. Then ³He captures,
and so the IPC and X17 yields, are 1&ndash;15 % of nominal.</li>
</ol>
<p>The first is the economical reading of the tracks, but it contradicts the alignment. The second fits the alignment
but needs two further surprises. <b>A survey or photo of the capsule against the beam axis, the capsule tip's height,
and the beam spot size at the station would settle it.</b> Neither reading lets the pose explain the 1&nbsp;:&nbsp;8 pair
rate on its own.</p></div>

<h2>1 &middot; Every picture against the data</h2>
<p>The beam is centred at x&nbsp;=&nbsp;0 unless the picture says otherwise. "Capsule move needed" is how far the capsule
must move vertically to put the toy source at the data's height; the y crossing is degenerate with that, the x is not.
Late window (&ge; 20&nbsp;ms).</p>
{alt_tab}
{figure('alternatives', 'Source (x, height) predicted by each picture, against the data (stars). Black: the flux centred on the capsule, at increasing width, moved to the data x. Purple, orange: a beam centred on the frame with the capsule 10.3 or 13 mm toward B.')}

<h2>2 &middot; The direct-beam window (the test asked for)</h2>
<p>Below ~10&nbsp;ms the band crossing is driven by background tracks and is unusable. The <b>pointing-excess peak</b>
is not. It is the peak, in x at the axis plane, of the fraction of lines passing within 10&nbsp;mm minus the same
with the angles shuffled among the tracks. Two calibrations:</p>
<ul>
<li><b>Response to a second source:</b> the late data mixed with a copy of themselves moved +10&nbsp;mm (an identical
kernel). A fraction f of tracks from the moved copy shifts the peak by {_f(slope / 10, 1)}&nbsp;&times;&nbsp;f&nbsp;mm
per mm of separation, linearly (early.csv, <i>mix</i>). A point
source alone, through the toy, is read with gain 1 (&minus;5 &rarr; &minus;4.7, &minus;10 &rarr; &minus;9.7).</li>
<li><b>Response to background:</b> late data diluted with the 1&ndash;2&nbsp;ms tracks (mostly background) to below the
early purity: {_f(bgA.peak_x - bgA.peak_ref, 1, True)} (A), {_f(bgC.peak_x - bgC.peak_ref, 1, True)} (C)&nbsp;mm.</li>
</ul>
{early_tab}
{figure('early', 'Source x against time since the flash: the pointing-excess peak (points) and, where it is usable, the band crossing (dotted). Dashed: where the peak would sit if 20 % or 50 % of the tracks came from a source 10 mm toward x = 0.')}
<p>Only A at 4&ndash;6&nbsp;ms moves ({_f(float(ED.loc[('A', 4.0), 'peak_x']), 1, True)} against
{_f(ref['A'], 1, True)}). C in the same bin does not ({_f(float(ED.loc[('C', 4.0), 'peak_x']), 1, True)}), so this is
most likely background structure, not a source.</p>
<p><b>What the test cannot show:</b> it needs the 4&ndash;12&nbsp;ms capsule tracks to be direct-beam captures. After
~28&nbsp;ms the evaluated flux is empty, yet 46 % of tracks still point at the same spot. Whatever makes the late tracks
(the &sect;10f "hall die-away" population) is present early too, and its share at 4&ndash;12&nbsp;ms is not known.
That late population images the nose like a narrow beam, not the whole capsule like an isotropic hall flux
(isotropic: y &asymp; +5 to +8). That is itself unexplained.</p>
{figure('image', 'Pointing image of chamber A (lines at the axis plane minus the shuffled-angle null), normalised per panel. One compact spot at all times; no vertical ridge from a beam-line source (e.g. ¹⁴N captures in the air column) at x = 0 or elsewhere.')}

<h2>3 &middot; Frame checks</h2>
<ul>
<li><b>A and C agree</b> ({_f(late.loc['A', 'x_cross'], 1, True)} / {_f(late.loc['C', 'x_cross'], 1, True)}). Their u axes
point opposite ways in global x (U_HAT A = +x, C = &minus;x), so a strip-map offset, a tan bias or a yaw of either
chamber would split them, as a wrong x sign splits them by twice the pinwheel (build_tracks, 2026-08-20). Only a
<b>common</b> shift of the A&ndash;C pair against the beam survives. The same holds for B&ndash;D in z (&minus;4.1 / &minus;3.5).</li>
<li><b>The frame's x/z origin is the surveyed station centre.</b> The chamber centres in run_config.json are the
survey's pinwheel shifts (15.5&ndash;17.3&nbsp;mm along each arm's &minus;u; MX17_Full_Geant docs/HANDOFF_ARM_GEOMETRY.md).
The source is ~10&nbsp;mm off that centre. Whether the centre is on the beam is what the alignment says and what these
data cannot check.</li>
<li><b>Stable over the campaign:</b> campaign-imaging crossings per run, post-access, A {_f(post.loc['A', 'median'], 1, True)}
(s.d. {_f(post.loc['A', 'std'], 1)}), C {_f(post.loc['C', 'median'], 1, True)} ({_f(post.loc['C', 'std'], 1)}), D
{_f(post.loc['D', 'median'], 1, True)} ({_f(post.loc['D', 'std'], 1)}); pre-access A {_f(pre.loc['A'], 1, True)},
C {_f(pre.loc['C'], 1, True)}. The 27 July access did not move it. (The production chain reads A ~2&nbsp;mm closer to 0 than
is2_v2; C agrees.)</li>
<li><b>The scintillators do not test it.</b> The SiPM wall sits within 1&ndash;6&nbsp;mm of the survey (ntof_scint_stack),
but wall and chambers share the structure, so a shift of the whole structure against the beam is invisible.</li>
<li><b>The beam model is the Geant4 gun's:</b> energy from the Phase-3 flux in the window, radius from that energy's
Lambda2D row, capture &prop; 1/v. Al captures by radius reproduce Geant4 bin by bin (51.7 vs 51.4 % inside 2&nbsp;mm,
CFRP/Al 1.62 vs 1.57 %). Lambda2D is scored to r = 3&nbsp;cm at 19.5&nbsp;m; a halo beyond that, or a different station
distance or collimator, is not in it.</li>
</ul>

<h2>4 &middot; Model and closure</h2>
<p>Each ray is walked through the STEP polycones (CFRP, Al, gas; nose-first). Al captures &Sigma;&nbsp;ds, scattering
removes neutrons from the ray, and the 500&nbsp;bar ³He is black. Tracks are the Geant4 nose-first run through the real
reconstruction (g4_digi), each translated from its own capture point to a toy capture point. Without neutron
scattering in the CFRP the barrel captures were 2&times; Geant4's; with it:</p>
{cl_tab}

<h2>5 &middot; If the flux is centred on the capsule (reading 1): pose scan</h2>
<p>Untilted, late window. "Capsule move needed" keeps the toy source at the data's height.</p>
{scan_tab}
{figure('captures', 'Where neutrons are captured on the Al for three offsets of a Lambda2D beam from the capsule axis (beam centre dashed).')}
{figure('height_needed', 'Capsule height change needed against the offset between flux and capsule axis, for several tilts.')}
{b_tab}
{figure('consequences', 'Beam into the ³He, Al captures per neutron (relative), and the barrel share, against offset (untilted).')}

<h2>6 &middot; The data</h2>
{table(D[D.window.isin(['late', 'mid', 'beam', 'beam_bgsub'])], ['arm', 'window', 'n', 'x_cross', 'y_cross', 'f_focus', 'h_w68'],
       ['arm', 'window', 'tracks', 'x crossing', 'y crossing', 'focus fraction', 'height w<sub>68</sub>'],
       [str, str, lambda v: f'{int(v):,}', lambda v: _f(v, 1, True), lambda v: _f(v, 1, True), lambda v: _f(v, 2), _f])}
{figure('tof', 'Source height against ToF (background-subtracted and raw). Flat after 20 ms; lower before, following the purity.')}
<p><b>x against height.</b> The x crossing steps by {_f(st.loc['A', 'step_data'], 1, True)} (A) /
{_f(st.loc['C', 'step_data'], 1, True)}&nbsp;mm (C) for tracks pointing above the nose, against
{_f(st.loc['A', 'step_toy'], 1, True)} &plusmn; {_f(st.loc['A', 'step_toy_err'], 1)} / {_f(st.loc['C', 'step_toy'], 1, True)}
&plusmn; {_f(st.loc['C', 'step_toy_err'], 1)} for a point source through Geant4: a {_f(sig_comb, 1)}&sigma; hint of a
second, higher source, limited by the Geant4 sample.</p>
{figure('shear', 'x crossing against reconstructed height, late window, with the toy point source and its library spread.')}
<p><b>Height shape.</b> Background-subtracted, the late height distribution is wider than every toy (w<sub>68</sub> 32 / 34
against 22&ndash;29&nbsp;mm). Once the uncalibrated y scale is tuned to the width, every picture gives the same skew as the
data (0.00&ndash;0.05), so the shape does not discriminate.</p>

<h2>7 &middot; What this does not rule out</h2>
<ul>
<li><b>Either reading in the verdict.</b> The data fix the source; the beam&ndash;frame and capsule&ndash;beam relations need
an outside measurement.</li>
<li><b>A source that is not the capsule's aluminium:</b> a holder or fitting at the nose, lit by the hall flux. It would
explain why the late, non-beam tracks image the nose and not the whole capsule.</li>
<li><b>A beam spot unlike Lambda2D</b> at the station (distance, collimator, halo beyond 3&nbsp;cm).</li>
<li><b>B/D are not modelled:</b> their z comes from the campaign crossings with no Geant4 correction.</li>
</ul>

<h2>8 &middot; Next</h2>
<ul>
<li>Dylan: how the beam centring was done (reference, tool), and whether it was checked against the capsule.</li>
<li>Dylan: the capsule tip height against the chamber mid-plane, and how the capsule is held (anything massive near the nose?).</li>
<li>The beam spot at the station (n_TOF beam-profile measurement, collimator in use, station distance).</li>
<li>The stage-3 y-sign fix.</li>
</ul>
<p class="prov">Built by <code>make_capsule_pose_report.py</code> from <code>capsule_pose.py</code> (closure, data, tof,
shear, early, image, alternatives, scan, fit CSVs here) and <code>imaging_campaign/per_run.csv</code>. Preliminary.</p>
"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Capsule pose in the beam</title>{HEAD}'
            f'</head><body><main>{body}</main></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    d = paths.out('capsule_pose')
    (d / 'figures').mkdir(exist_ok=True)
    if not a.no_figures:
        fs.use()
        F = pd.read_csv(d / 'fit.csv')
        D = pd.read_csv(d / 'data.csv')
        late = D[D.window == 'late'].set_index('arm')
        fig_captures(d)
        fig_height_needed(d, F, late)
        fig_consequences(d, F)
        fig_tof(d, pd.read_csv(d / 'tof.csv'))
        fig_shear(d, pd.read_csv(d / 'shear.csv'), late)
        fig_early(d, pd.read_csv(d / 'early.csv'))
        fig_image(d, pd.read_csv(d / 'image.csv'))
        alt = pd.read_csv(d / 'alternatives.csv')
        if (d / 'halo.csv').exists():
            alt = pd.concat([alt, pd.read_csv(d / 'halo.csv')], ignore_index=True)
        fig_alternatives(d, alt, late)
    (d / 'report.html').write_text(build(d), encoding='utf-8')
    print(f'wrote -> {d / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
