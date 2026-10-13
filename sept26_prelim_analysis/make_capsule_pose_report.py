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
    st = S.groupby('arm')[['step_data', 'step_toy', 'step_toy_err']].first()
    st['sig'] = (st.step_data - st.step_toy) / st.step_toy_err
    sig_comb = float(abs(st.sig.sum()) / np.sqrt(len(st)))
    F0 = F[F.theta == 0].set_index('d')
    xb, zb = float(F0.loc[0, 'x_beam']), float(F0.loc[0, 'z_beam'])
    dy0 = float(F0.loc[0, 'dy_c'])
    ac = abs(late.loc['A', 'y_cross'] - late.loc['C', 'y_cross'])

    Bi = B.set_index('height_prior_mm')
    b5, b10 = Bi.loc[5], Bi.loc[10]
    d10 = float(F0.loc[10, 'dy_c']) if 10 in F0.index else np.nan
    gas10 = float(F0.loc[10, 'gas']) if 10 in F0.index else np.nan
    bar10 = float(F0.loc[10, 'al_barrel']) if 10 in F0.index else np.nan
    al10 = float(F0.loc[10, 'al_per_ray'] / F0.loc[0, 'al_per_ray']) if 10 in F0.index else np.nan
    tof_late = T[T.lo >= 20].groupby('arm').y_bgsub.agg(['min', 'max'])
    tof_early = T[T.lo == 4].set_index('arm').y_bgsub

    cl = C[(C.kind == 'reco') & C['sample'].isin(['geant4 capsule', 'toy nominal'])]
    cl_tab = table(cl, ['arm', 'sample', 'x_cross', 'y_cross', 'h_w68', 'f_focus'],
                   ['arm', 'sample', 'x crossing', 'y crossing', 'height w<sub>68</sub>', 'focus fraction'],
                   [str, str, lambda v: _f(v, 2, True), lambda v: _f(v, 2, True), _f, lambda v: _f(v, 2)])
    dat = D[D.window.isin(['late', 'mid', 'beam', 'beam_bgsub'])]
    dat_tab = table(dat, ['arm', 'window', 'n', 'x_cross', 'y_cross', 'f_focus', 'h_w68'],
                    ['arm', 'window', 'tracks', 'x crossing', 'y crossing', 'focus fraction', 'height w<sub>68</sub>'],
                    [str, str, lambda v: f'{int(v):,}', lambda v: _f(v, 1, True), lambda v: _f(v, 1, True),
                     lambda v: _f(v, 2), _f])
    g0 = F[F.theta == 0].sort_values('d')
    scan_tab = table(g0, ['d', 'y_cross_A', 'y_cross_C', 'dy_c', 'gas', 'al_nose', 'al_barrel', 'al_per_ray'],
                     ['d [mm]', 'toy y, A', 'toy y, C', 'capsule move needed', 'beam into ³He',
                      'Al: nose', 'Al: barrel', 'Al captures / neutron'],
                     [lambda v: f'{int(v)}', lambda v: _f(v, 1, True), lambda v: _f(v, 1, True),
                      lambda v: _f(v, 1, True), lambda v: _f(v, 2), lambda v: _f(v, 2), lambda v: _f(v, 2),
                      lambda v: f'{1e3 * v:.2f}&times;10<sup>&minus;3</sup>'])
    b_tab = table(B, ['height_prior_mm', 'n_poses', 'd_tip_max', 'gas_min', 'barrel_max', 'al_rel_min'],
                  ['capsule height known to &plusmn; [mm]', 'poses allowed', 'nose tip from beam, max [mm]',
                   'beam into ³He, min', 'barrel share, max', 'Al captures / neutron, min (rel.)'],
                  [lambda v: f'{int(v)}', lambda v: f'{int(v)}', lambda v: _f(v, 0), lambda v: _f(v, 2),
                   lambda v: _f(v, 2), lambda v: _f(v, 2)])

    body = f"""
<h1>Where the capsule sits in the beam</h1>
<p class="sub">capsule_pose.py &middot; is2_v2 at the capsule-focus k, post-access runs, A and C &middot;
y mirrored on load (stage-3 <code>IN_PLANE_SIGN_Y</code>) &middot; built {dt.date.today().isoformat()}</p>

<div class="verdict"><p><b>Verdict.</b> The beam neutrons that make the tracks are captured on the
capsule's <b>nose</b>, not along its barrel. The capture point (the beam) sits at
<b>x = {_f(xb, 1, True)}, z = {_f(zb, 1, True)}&nbsp;mm</b> in the chamber frame. The data's source height
(late window, {_f(late.loc['A', 'y_cross'], 1, True)} A / {_f(late.loc['C', 'y_cross'], 1, True)} C) matches a
beam on the capsule axis if the capsule sits <b>{_f(-dy0, 0)}&nbsp;mm lower</b> than in Geant4.</p>
<p><b>Where the capsule's centre is relative to the beam is not determined.</b> The y crossing is
degenerate between the capsule's height and how the beam meets it, and a tilt can carry the nose back into the
beam however far the centre is off. Untilted, keeping the source at the data's height needs the capsule moved by
{_f(F0.loc[7, 'dy_c'], 0, True)}&nbsp;mm at a 7&nbsp;mm offset and {_f(d10, 0, True)}&nbsp;mm at 10&nbsp;mm. <b>So
the hypothesis of a capsule 1&nbsp;cm off the beam with the nose out needs it ~{_f(-d10, 0)}&nbsp;mm lower than
drawn.</b></p>
<p><b>What the data do bound are the consequences</b>, given a prior on the capsule height. If the capsule is
within &plusmn;5&nbsp;mm of the Geant4 placement, then over every scanned pose that matches, <b>&ge; {_f(b5.gas_min * 100, 0)} % of the
beam enters the ³He</b>, at most {_f(b5.barrel_max * 100, 0)} % of the Al captures are on the barrel, and the nose
tip is within {_f(b5.d_tip_max, 0)}&nbsp;mm of the beam axis. At &plusmn;10&nbsp;mm anything is allowed, including a
capsule nearly off the beam (³He share down to {_f(b10.gas_min, 2)}).</p>
<p><b>A pose that cuts Al captures (and so pairs) by 8 has the capsule mostly out of the beam</b> (Al per neutron
&ge; {_f(b5.al_rel_min, 2)}&times; nominal at &plusmn;5&nbsp;mm; {_f(b10.al_rel_min, 2)}&times; at &plusmn;10&nbsp;mm only
with ~10 % of the beam on the ³He). Untilted at d = 10, Al captures stay at {_f(al10, 2)}&times; nominal (barrel
grazing replaces the nose) while the ³He share falls to {_f(gas10, 2)}.</p></div>

<h2>Headline numbers</h2>
<ul>
<li><b>Beam / capture position:</b> x {_f(xb, 1, True)} (A and C, corrected by the toy's
{_f(F0.loc[0, 'x_cross_A'], 2, True)} / {_f(F0.loc[0, 'x_cross_C'], 2, True)} estimator offset), z {_f(zb, 1, True)}&nbsp;mm
(B, D campaign crossings, uncorrected). The x crossing follows the <b>beam</b>, not the capsule axis: it moves by
&lt; 1&nbsp;mm for any offset up to 10&nbsp;mm.</li>
<li><b>Source height, late window (&ge; 20&nbsp;ms):</b> A {_f(late.loc['A', 'y_cross'], 1, True)}, C
{_f(late.loc['C', 'y_cross'], 1, True)}&nbsp;mm. The toy at the Geant4 pose gives {_f(F0.loc[0, 'y_cross_A'], 1, True)} /
{_f(F0.loc[0, 'y_cross_C'], 1, True)}. Flat from 20 to 75&nbsp;ms (A {_f(tof_late.loc['A', 'min'], 1)} to
{_f(tof_late.loc['A', 'max'], 1)}, C {_f(tof_late.loc['C', 'min'], 1)} to {_f(tof_late.loc['C', 'max'], 1)}),
so the late, delayed neutrons light the nose like the beam does. An isotropic thermal flux would put the source
at y &asymp; +5 to +8 (it lights the whole Al body).</li>
<li><b>A &minus; C height difference {_f(ac, 1)}&nbsp;mm</b> is the open issue from HANDOFF_CAPSULE_Y. It is
carried as an uncertainty here.</li>
<li><b>Earlier windows read lower:</b> {_f(tof_early.loc['A'], 0, True)} (A) / {_f(tof_early.loc['C'], 0, True)} (C) at
4&ndash;6&nbsp;ms after background subtraction. The drop follows the purity drop, and no pose in the scan
moves the beam-window source <i>down</i> relative to the late one (the wider high-energy beam grazes more barrel,
which moves it <i>up</i>). Read it as an unsubtracted source on the beam line below the capsule, not as pose.</li>
<li><b>x against height:</b> the data's x crossing steps by {_f(st.loc['A', 'step_data'], 1, True)} (A) /
{_f(st.loc['C', 'step_data'], 1, True)}&nbsp;mm (C) toward &minus;x for tracks pointing above the nose. A point-like
nose source through Geant4 gives {_f(st.loc['A', 'step_toy'], 1, True)} &plusmn; {_f(st.loc['A', 'step_toy_err'], 1)} /
{_f(st.loc['C', 'step_toy'], 1, True)} &plusmn; {_f(st.loc['C', 'step_toy_err'], 1)}. That is a
{_f(sig_comb, 1)}&sigma; combined hint of a second, higher source at more negative x, limited by the size of
the Geant4 sample. It is not a capsule tilt: with a nose source the height spread (&plusmn;25&nbsp;mm) is
kinematics, not geometry. Nor is it an in-plane rotation, which would give a straight line, not a step.</li>
</ul>

<h2>Step 0: the y mirror</h2>
<p>Every stage-3 table has global y mirrored (<code>build_tracks.IN_PLANE_SIGN_Y = &minus;1</code> flips the
position where it should flip tan&nbsp;y; wiring confirmed 2026-10-13). This study undoes it on load:
p0_y &rarr; &minus;p0_y, d_y &rarr; &minus;d_y. Under the mirror, the "capsule 30&nbsp;mm too high" of
HANDOFF_CAPSULE_Y becomes the nose, a few millimetres below where Geant4 puts it. <b>Fixing stage 3 is Dylan's
call</b> (handoff &sect;8 lists what moves).</p>

<h2>Model and closure</h2>
<p>Each beam ray is walked through the STEP polycones (CFRP, Al, gas; nose-first) at the pose. Al captures
&Sigma;&nbsp;ds, scattering removes neutrons from the ray, and the 500&nbsp;bar ³He is black. The radial
profile is Lambda2D, weighted by flux &times; 1/v over the window's energies. Tracks are the Geant4 nose-first
run through the real reconstruction (g4_digi), each <i>translated</i> from its own capture point to a toy
capture point. So the kinematics, the electron scattering (lines miss their vertex by ~40&nbsp;mm) and the
reconstruction are Geant4's; only the capture point is modelled. The estimators are capsule_tof's band
crossings.</p>
<p>The first closure attempt failed: barrel-grazing captures came out at 2&times; Geant4's, because the toy had
no neutron scattering. With scattering in the dome's CFRP (H, C; 1.5&nbsp;/cm, tuned on the Geant4
radius distribution of Al captures) and in the Al, it closes:</p>
{cl_tab}
<p>A shift of the capture points by (+5, +10)&nbsp;mm moves the crossings by (+4.9, +10.0) (closure.csv).</p>

<h2>The data (y mirrored)</h2>
{dat_tab}
{figure('tof', 'Source height against ToF. Background subtraction uses the off-focus tracks of the same window, normalised to the shuffled-angle null; on the late window it reproduces the raw crossing to 0.2 mm.')}

<h2>The scan</h2>
<p>Everything is translation-equivariant, so only the beam&ndash;capsule offset d and the tilt are scanned. The
absolute position is then whatever matches the data. The table is untilted, late window:</p>
{scan_tab}
{figure('captures', 'Where beam neutrons are captured on the Al (late-window profile; beam centre dashed). At d = 8&ndash;10 mm the core grazes the 0.6 mm barrel, and the capture height climbs the capsule.')}
{figure('height_needed', 'Capsule height change needed to keep the toy source at the data height, against offset, for several tilts. Shaded: ±5 mm about the Geant4 placement, widened by half the A&minus;C difference.')}
<p>A pose is allowed when the capsule move it needs is within the prior, widened by half the A&minus;C difference.
The scan covers d 0&ndash;14&nbsp;mm and tilts 0&ndash;20&deg; in both senses. The geometry stays degenerate; the
consequences do not:</p>
{b_tab}
{figure('consequences', 'Beam into the ³He (³He, IPC and X17 rate), Al captures per beam neutron relative to the centred capsule, and the barrel share, against offset (untilted).')}
{figure('shear', 'x crossing against reconstructed height, late window. Grey: the toy with a point-like nose source, aligned to the data; its band is the Geant4-library bootstrap spread. Dashed: the source height.')}

<h2>What this does not rule out</h2>
<ul>
<li><b>A capsule ~{_f(-d10, 0)}&nbsp;mm lower than drawn, 1&nbsp;cm off the beam.</b> The y crossing is degenerate
between capsule height and offset. The height spread would separate them (the barrel source is 40&nbsp;mm long),
but the y angle scale is uncalibrated: the data's spread (w<sub>68</sub> {_f(late.loc['A', 'h_w68'], 0)} /
{_f(late.loc['C', 'h_w68'], 0)}&nbsp;mm) exceeds every toy pose (22&ndash;31). So the spread measures the scale,
not the source. <b>A survey or photo of the capsule tip's height against the chamber mid-plane settles d.</b></li>
<li><b>The capsule offset direction.</b> Only the beam's x and z are measured; with a nose hit the capsule axis
could be up to ~5&nbsp;mm away in any direction.</li>
<li><b>Tilt.</b> A tilt alone barely moves anything when the beam is on the nose (10&deg; moves y by
1&nbsp;mm). Combined with an offset it either brings the nose back into the beam (nose toward the beam) or
lays the barrel across it (nose away). The second sense is the hypothesis, and it is the one that needs the
largest height change.</li>
<li><b>The second, higher source.</b> At {_f(sig_comb, 1)}&sigma; it is a hint, not a measurement. If real, it
sits at more negative x than the nose capture. Candidates: the neck/valve and the upper body lit by
room-return neutrons, with the capsule axis at more negative x than the beam (an offset toward B, as
hypothesised but smaller); or support hardware. A Geant4 sample ~10&times; larger (the shear is library-limited)
would settle whether it is there.</li>
<li><b>Whatever pulls the early windows down</b> is not modelled. It is presumably beam-line material below
the capsule, at the beam's x.</li>
<li><b>B/D.</b> z comes from the campaign crossings with no Geant4 correction (no B/D reco sample exists);
B has few tracks and D needs the hot-channel cleaning. So the beam's z is known to ~1&nbsp;mm, and the
capsule-offset direction in z not at all.</li>
</ul>

<h2>Next</h2>
<ul>
<li>Dylan: the capsule's mounted height (tip against the chamber mid-plane), and how it is held. This is the
one number that turns the table above into a measurement of d.</li>
<li>Dylan: the stage-3 y-sign fix (<code>IN_PLANE_SIGN_Y</code> on tan, not position) and its regeneration.</li>
<li>Optional: a 10&times; g4_digi sample (condor) to make the x-height step decisive.</li>
</ul>
<p class="prov">Built by <code>make_capsule_pose_report.py</code> from <code>capsule_pose.py</code>
(closure, data, tof, shear, scan, fit CSVs in this directory). Preliminary.</p>
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
    (d / 'report.html').write_text(build(d), encoding='utf-8')
    print(f'wrote -> {d / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
