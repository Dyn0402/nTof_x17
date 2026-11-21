#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_audit_report.py -- report.html for the n_TOF pass-1 tracking audit (rerun/ntof/improve/tracking_audit.py).

Reads the CSVs of `tracking_audit tables / bow / slim` in <out> and writes <out>/report.html with figures in
<out>/figures/ (each with its CSV).  Re-run after any of those steps.  The numbers of sections 1-6 are read from the
tables.  Sections 7 and 8 quote one-off checks of the code reviews (scripts kept in <out>/review_scripts/).

    python -m rerun.ntof.improve.make_audit_report [--copy]      # --copy: also into rerun/ntof/improve/audit_report/
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from rerun.ntof.improve.tracking_audit import out_dir, Q_MIN, N_MAX, CENS_MAX, CHI2_REL  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ARM_NAME = {'A': 'chamber A', 'C': 'chamber C', 'D': 'chamber D'}


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = ''.join(f'<th>{c}</th>' for c in df.columns)
    body = ''
    for _, r in df.iterrows():
        cells = ''
        for c in df.columns:
            v = r[c]
            if c in fmt and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                v = fmt[c].format(v)
            cells += f'<td>{v}</td>'
        body += f'<tr>{cells}</tr>'
    return f'<table><tr>{head}</tr>{body}</table>'


def step(ax, lo, y, width, **kw):
    x = np.r_[lo, lo[-1] + width]
    ax.step(x, np.r_[y, y[-1]], where='post', **kw)


def fig_pointing(out: Path, H: pd.DataFrame) -> str:
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, constrained_layout=True, sharey=True)
    for ax, a in zip(axs, 'AC'):
        for name, color, ls in ((f'data {a}', fs.DET_COLOR[a], '-'), (f'Geant4 truth {a}', fs.INK, '--'),
                                (f'Geant4 reconstructed {a}', fs.COPPER, ':')):
            g = H[H['sample'] == name]
            step(ax, g.lo.to_numpy(), g.frac.to_numpy() * 100, 5.0, color=color, ls=ls, lw=1.6, label=name)
        ax.set_xlim(-100, 100)
        ax.set_xlabel('distance of the track line from the source at the capsule [mm]')
        ax.set_title(ARM_NAME[a], fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
    axs[0].set_ylabel('tracks per 5 mm [%]')
    fs.save(fig, out / 'figures' / 'pointing_vs_g4.png', data=H)
    plt.close(fig)
    return 'figures/pointing_vs_g4.png'


def fig_charge(out: Path, Q: pd.DataFrame) -> str:
    fs.use()
    fig, axs = plt.subplots(1, 3, figsize=fs.WIDE, constrained_layout=True, sharey=True)
    for ax, a in zip(axs, 'ACD'):
        for (lo, hi), color in (((1, 8), fs.COPPER), ((20, 75), fs.DET_COLOR[a])):
            g = Q[(Q.arm == a) & (Q.ms_lo == lo)]
            x = np.arange(len(g))
            ax.bar(x + (-0.2 if lo == 1 else 0.2), g.frac * 100, width=0.4, color=color, label=f'{lo}-{hi} ms')
        lab = [f'{int(v)}' for v in g.q_lo]
        ax.set_xticks(x - 0.5)
        ax.set_xticklabels(lab, rotation=60, fontsize=7.5)
        k = int(np.where(g.q_lo.to_numpy() == Q_MIN)[0][0])
        ax.axvline(k - 0.5, color=fs.TRACK, lw=1.2)
        ax.set_title(ARM_NAME[a], fontsize=10)
        ax.set_xlabel('smaller view charge [ADC], bin lower edge')
        ax.legend(frameon=False, fontsize=8.5)
    axs[0].set_ylabel('gated tracks [%]')
    fs.save(fig, out / 'figures' / 'charge_spectrum.png', data=Q)
    plt.close(fig)
    return 'figures/charge_spectrum.png'


def fig_tof(out: Path, T: pd.DataFrame) -> str:
    fs.use()
    fig, axs = plt.subplots(2, 3, figsize=fs.FULL, constrained_layout=True, sharex=True)
    for col, a in enumerate('ACD'):
        for sel, ls, alpha in (('gated', '--', 0.55), ('clean', '-', 1.0)):
            g = T[(T.arm == a) & (T.sel == sel)]
            x = np.sqrt(g.ms_lo * g.ms_hi)
            axs[0, col].plot(x, g.rate_per_ms / 1e3, ls, marker='o', ms=3.5, color=fs.DET_COLOR[a], alpha=alpha, label=sel)
            axs[1, col].plot(x, g.f30, ls, marker='o', ms=3.5, color=fs.DET_COLOR[a], alpha=alpha, label=sel)
        axs[0, col].set_title(ARM_NAME[a], fontsize=10)
        axs[0, col].set_yscale('log')
        axs[0, col].set_xscale('log')
        axs[1, col].set_ylim(0, 1)
        axs[1, col].set_xlabel('time since the flash [ms]')
        axs[0, col].legend(frameon=False, fontsize=8.5)
    axs[0, 0].set_ylabel('tracks per ms [thousands]')
    axs[1, 0].set_ylabel('fraction within 30 mm of the source')
    fs.save(fig, out / 'figures' / 'tof_purity.png', data=T)
    plt.close(fig)
    return 'figures/tof_purity.png'


def fig_pairs(out: Path, P: pd.DataFrame) -> str:
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, constrained_layout=True, sharey=True)
    for ax, a in zip(axs, 'AC'):
        for sel, ls, alpha in (('gated', '--', 0.55), (f'charge >= {Q_MIN:.0f}', '-', 1.0), ('clean', ':', 1.0)):
            g = P[(P.arm == a) & (P.sel == sel)]
            ax.errorbar(np.sqrt(g.ms_lo * g.ms_hi), g.pair_pct, yerr=g.err, fmt='o', ls=ls, ms=3.5,
                        color=fs.DET_COLOR[a], alpha=alpha, label=sel)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('time since the flash [ms]')
        ax.set_title(ARM_NAME[a], fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
    axs[0].set_ylabel('events with two or more tracks [%]')
    fs.save(fig, out / 'figures' / 'pairs_tof.png', data=P)
    plt.close(fig)
    return 'figures/pairs_tof.png'


def fig_positions(out: Path, X: pd.DataFrame) -> str:
    fs.use()
    fig, axs = plt.subplots(2, 2, figsize=fs.FULL, constrained_layout=True)
    for r, a in enumerate('AD'):
        for c, coord in enumerate(('x_local', 'y_local')):
            g = X[(X.arm == a) & (X.coord == coord)]
            ax = axs[r, c]
            step(ax, g.lo.to_numpy(), g.n.to_numpy() / 1e3, 5.0, color=fs.DET_COLOR[a], lw=1.4, label='all gated tracks')
            step(ax, g.lo.to_numpy(), g.not_pointing.to_numpy() / 1e3, 5.0, color=fs.TRACK, lw=1.4,
                 label='do not point (more than 60 mm from the source)')
            step(ax, g.lo.to_numpy(), g.not_pointing_clean.to_numpy() / 1e3, 5.0, color=fs.INK, lw=1.2, ls=':',
                 label='do not point, after the clean selection')
            ax.set_title(f'{ARM_NAME[a]}, {coord.replace("_", " ")}', fontsize=10)
            ax.set_xlabel(f'{coord} [mm]')
            ax.set_ylabel('tracks per 5 mm [thousands]')
            if r == 1 and c == 0:
                ax.legend(frameon=False, fontsize=8, loc='upper right')
    fs.save(fig, out / 'figures' / 'positions.png', data=X)
    plt.close(fig)
    return 'figures/positions.png'


def fig_height(out: Path, Y: pd.DataFrame) -> str:
    fs.use()
    fig, ax = plt.subplots(figsize=fs.FIG, constrained_layout=True)
    ax.axvspan(-29.5, 50.7, color=fs.GRID, label='drawn gas volume of the capsule')
    for a in 'ACD':
        g = Y[Y['sample'] == f'data {a}']
        step(ax, g.lo.to_numpy(), g.frac.to_numpy() * 100, 10.0, color=fs.DET_COLOR[a], lw=1.6, label=f'data, {ARM_NAME[a]}')
    for name, ls, lab in (('Geant4 nose truth A', '--', 'Geant4, truth (chamber A)'),
                          ('Geant4 nose reconstructed A', ':', 'Geant4 through the reconstruction (chamber A)')):
        g = Y[Y['sample'] == name]
        step(ax, g.lo.to_numpy(), g.frac.to_numpy() * 100, 10.0, color=fs.INK, lw=1.4, ls=ls, label=lab)
    ax.set_xlim(-140, 140)
    ax.set_xlabel('height at which the track line meets the source plane [mm], +y up')
    ax.set_ylabel('tracks per 10 mm [%]')
    ax.legend(frameon=False, fontsize=8)
    fs.save(fig, out / 'figures' / 'height_profile.png', data=Y)
    plt.close(fig)
    return 'figures/height_profile.png'


def fig_untimed(out: Path, U: pd.DataFrame) -> str:
    fs.use()
    U = U.assign(r=U.run.str.replace('run_', '').astype(int)).sort_values('r')
    fig, ax = plt.subplots(figsize=fs.WIDE, constrained_layout=True)
    ax.bar(np.arange(len(U)), U.untimed_frac * 100, color=fs.ACCENT)
    ax.set_xticks(np.arange(len(U)))
    ax.set_xticklabels(U.r, rotation=90, fontsize=7.5)
    ax.set_xlabel('run')
    ax.set_ylabel('tracks without a time since the flash [%]')
    fs.save(fig, out / 'figures' / 'untimed_by_run.png', data=U)
    plt.close(fig)
    return 'figures/untimed_by_run.png'


def main() -> int:
    out = out_dir()
    (out / 'figures').mkdir(exist_ok=True)
    cuts = pd.read_csv(out / 'cuts.csv')
    T = pd.read_csv(out / 'tof.csv')
    Q = pd.read_csv(out / 'charge_hist.csv')
    P = pd.read_csv(out / 'pairs_tof.csv')
    X = pd.read_csv(out / 'positions.csv')
    G = pd.read_csv(out / 'pointing_data_vs_g4.csv')
    H = pd.read_csv(out / 'miss_hist.csv')
    Y = pd.read_csv(out / 'height_profile.csv')
    E = pd.read_csv(out / 'height_edges.csv')
    U = pd.read_csv(out / 'untimed_by_run.csv')
    B = pd.read_csv(out / 'bow_partner.csv') if (out / 'bow_partner.csv').exists() else None
    S = pd.read_csv(out / 'slim_files.csv') if (out / 'slim_files.csv').exists() else None

    f_point = fig_pointing(out, H)
    f_charge = fig_charge(out, Q)
    f_tof = fig_tof(out, T)
    f_pairs = fig_pairs(out, P)
    f_pos = fig_positions(out, X)
    f_height = fig_height(out, Y)
    f_unt = fig_untimed(out, U)

    def g(sample):
        return G[G['sample'] == sample].iloc[0]

    def t(arm, lo, sel):
        return T[(T.arm == arm) & (T.ms_lo == lo) & (T.sel == sel)].iloc[0]

    def c(arm, cut):
        return cuts[(cuts.arm == arm) & (cuts.cut == cut)].iloc[0]

    def p(arm, lo, sel):
        return P[(P.arm == arm) & (P.ms_lo == lo) & (P.sel == sel)].iloc[0]

    def e(sample):
        return E[E['sample'] == sample].iloc[0]

    qsel = f'charge >= {Q_MIN:.0f}'
    untimed = U.untimed.sum() / U.n.sum()
    early = {a: T[(T.arm == a) & (T.sel == 'gated') & (T.ms_hi <= 12)] for a in 'ACD'}
    low_early = {a: float((early[a].low_charge * early[a].rate_per_ms * (early[a].ms_hi - early[a].ms_lo)).sum()
                          / (early[a].rate_per_ms * (early[a].ms_hi - early[a].ms_lo)).sum()) for a in 'ACD'}
    late_np = {a: {s: float(np.average(T[(T.arm == a) & (T.sel == s) & (T.ms_lo >= 20)].not_pointing,
                                       weights=T[(T.arm == a) & (T.sel == s) & (T.ms_lo >= 20)].rate_per_ms
                                       * (T[(T.arm == a) & (T.sel == s) & (T.ms_lo >= 20)].ms_hi
                                          - T[(T.arm == a) & (T.sel == s) & (T.ms_lo >= 20)].ms_lo)))
                   for s in ('gated', 'clean')} for a in 'ACD'}

    gt = G.copy()
    gt = gt[['sample', 'n', 'f15', 'f30', 'f90', 'hw50']].rename(columns={
        'n': 'tracks', 'f15': 'within 15 mm', 'f30': 'within 30 mm', 'f90': 'within 90 mm', 'hw50': 'half of the tracks within [mm]'})
    tt = T[['arm', 'ms_lo', 'ms_hi', 'sel', 'rate_per_ms', 'f15', 'f30', 'f90', 'not_pointing', 'wall', 'low_charge',
            'q_med', 'chi2x_med', 'nstrips_med', 'crossing', 'crossing_err']].rename(columns={
        'ms_lo': 'from [ms]', 'ms_hi': 'to [ms]', 'sel': 'selection', 'rate_per_ms': 'tracks per ms', 'f15': 'within 15 mm',
        'f30': 'within 30 mm', 'f90': 'within 90 mm', 'not_pointing': 'more than 60 mm away', 'wall': 'wall fired',
        'low_charge': f'charge below {Q_MIN:.0f}', 'q_med': 'median charge', 'chi2x_med': 'median chi2/dof x',
        'nstrips_med': 'median strips x', 'crossing': 'band crossing [mm]', 'crossing_err': 'error'})
    tfmt = {'tracks per ms': '{:,.0f}', 'within 15 mm': '{:.3f}', 'within 30 mm': '{:.3f}', 'within 90 mm': '{:.3f}',
            'more than 60 mm away': '{:.3f}', 'wall fired': '{:.3f}', f'charge below {Q_MIN:.0f}': '{:.3f}',
            'median charge': '{:.0f}', 'median chi2/dof x': '{:.2f}', 'median strips x': '{:.0f}',
            'band crossing [mm]': '{:.2f}', 'error': '{:.2f}'}
    ct = cuts[['arm', 'cut', 'keeps_reference', 'removes_lt12', 'removes_12_20', 'removes_ge20']].rename(columns={
        'keeps_reference': 'keeps of the reference tracks', 'removes_lt12': 'removes before 12 ms',
        'removes_12_20': 'removes at 12-20 ms', 'removes_ge20': 'removes after 20 ms'})
    pt = P.pivot_table(index=['arm', 'ms_lo', 'ms_hi'], columns='sel', values='pair_pct').reset_index()
    pt = pt.merge(P[P.sel == 'gated'][['arm', 'ms_lo', 'low_charge_leg']], on=['arm', 'ms_lo'])
    pt = pt.rename(columns={'ms_lo': 'from [ms]', 'ms_hi': 'to [ms]', 'gated': 'gated [%]', qsel: f'{qsel} [%]',
                            'clean': 'clean [%]', 'low_charge_leg': 'gated pairs with a low-charge track'})
    pfmt = {'gated [%]': '{:.3f}', f'{qsel} [%]': '{:.3f}', 'clean [%]': '{:.3f}', 'gated pairs with a low-charge track': '{:.2f}'}

    slim_txt = ''
    if S is not None:
        multi = S[S.n_slim > 1]
        slim_txt = (f'On EOS, {len(multi)} of the {len(S)} production sub-runs have more than one slim file, and '
                    f'{1 - S.first_bytes.sum() / S.all_bytes.sum():.1%} of the slim bytes are in a file that is not the first. ')
    bow_txt = ''
    if B is not None:
        b2 = B[B['sample'].str.startswith('two')]
        bow_txt = table(B.rename(columns={'wrong_frac': 'tracks with a wrong tangent', 'dtan_median': 'median |Δtan|',
                                          'dtan_p90': '90th percentile', 'dtan_max': 'largest'}),
                        {'tracks with a wrong tangent': '{:.3f}', 'median |Δtan|': '{:.4f}', '90th percentile': '{:.4f}',
                         'largest': '{:.4f}'})
        bow_sum = (f'In run_145 this affects {b2.wrong_frac.min():.0%} to {b2.wrong_frac.max():.0%} of the tracks in events '
                   f'with two or more tracks in one chamber. The error is small: the median |Δtan| is below '
                   f'{b2.dtan_median.max():.4f} and the 90th percentile is {b2.dtan_p90.max():.3f} (chamber C).')
    else:
        bow_sum = ''

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>n_TOF pass 1: tracking audit</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2em auto;padding:0 16px;color:#1b2430;line-height:1.5}}
table{{border-collapse:collapse;margin:0.6em 0 1.2em;font-size:0.85em}}td,th{{border:1px solid #c9ced6;padding:3px 7px;text-align:right}}
th{{background:#f1f3f6}}img{{max-width:100%}}.verdict{{background:#f6f0f7;border-left:4px solid #8a3f8f;padding:0.6em 1em}}
figcaption{{font-size:0.9em;color:#4a5563;margin-bottom:1.5em}}code{{background:#f1f3f6;padding:0 3px}}
.scroll{{overflow-x:auto}}</style></head><body>
<h1>n_TOF pass 1: audit of the tracking and of the capsule imaging</h1>
<p>Written 2026-11-21. Code: <code>rerun/ntof/improve/tracking_audit.py</code> and <code>make_audit_report.py</code>.
Input: the pass-1 analysis view (v(t) angles, in-situ map, censored-window cut, <code>gated</code> = true |tan| &lt; 0.5),
production runs after the 27 July access, and Geant4 through the pass-1 reconstruction. Existing data only.</p>

<div class="verdict"><b>Verdict.</b> The width of the capsule image is not a tracking problem. In Geant4, the true
lines of capsule electrons point at the capsule no better than the data do, so the 45 to 60 % "in focus" fraction is
multiple scattering of MeV electrons. We also could not explain the 10 mm x offset of the source with a tracking effect.
However, the audit found three things that do bias the present analysis.
<ol>
<li><b>{untimed:.1%} of the production tracks have no time since the flash and no scintillator flags.</b> Sub-runs that
span two n_TOF run numbers have two slim files, and the code reads only the first one. Every study that cuts on the
time since the flash loses these tracks, and some of them were taken for beam-off cosmic events.</li>
<li><b>Before about 12 ms most gated tracks in chambers A and C are not particles.</b> They are fits to the disturbed
baseline after the flash. They have a quarter of the charge of a real track. They make {low_early['A']:.0%} (A) and
{low_early['C']:.0%} (C) of the gated tracks before 12 ms, and {p('A', 12, 'gated').low_charge_leg:.0%} (A) of the
same-chamber pairs at 12 to 15 ms. A minimum charge of {Q_MIN:.0f} ADC in each view removes them and keeps
{c('A', 'charge').keeps_reference:.1%} (A) and {c('C', 'charge').keeps_reference:.1%} (C) of the real tracks.</li>
<li><b>Chamber D is still heavily contaminated after the censored-window cut.</b> {late_np['D']['gated']:.0%} of its
tracks after 20 ms do not point at the capsule (A: {late_np['A']['gated']:.0%}, C: {late_np['C']['gated']:.0%}), and they
sit at a few strip positions. With a clean selection the fraction is {late_np['D']['clean']:.0%}, and the z position of
the source read through D is the same in every time window ({t('D', 20, 'clean').crossing:.1f} and
{t('D', 30, 'clean').crossing:.1f} mm). This explains why two estimators gave −5.5 and −1.7 mm.</li>
</ol>
Two code errors were also found: the scintillator-stack study extrapolates the tracks along the bars with the wrong sign
of the y slope, and the reconstruction maps the slope of some paired tracks with the other track's position.</div>

<h2>1. The capsule image width is electron scattering</h2>
<p>We measure how well a track points with the distance between the track line and the source, in the plane through the
capsule that is parallel to the chamber. The source position is the band crossing of the late tracks of that chamber.
We compare three samples in the central 200 mm of chambers A and C: the data after 20 ms, the true lines of Geant4
capsule electrons in the drift gap, and the same Geant4 events after the digitiser and the pass-1 reconstruction.</p>
<figure><img src="{f_point}"><figcaption><b>Figure 1.</b> Distribution of the distance between the track line and the
source for chamber A (left) and chamber C (right). The solid line is the data after 20 ms with the clean selection. The
dashed line is the Geant4 truth for the nose-first capsule library. The dotted line is the same Geant4 sample through the
pass-1 reconstruction.</figcaption></figure>
<div class="scroll">{table(gt, {'tracks': '{:,.0f}', 'within 15 mm': '{:.3f}', 'within 30 mm': '{:.3f}', 'within 90 mm': '{:.3f}',
                                'half of the tracks within [mm]': '{:.1f}'})}</div>
<p>Half of the data tracks pass within {g('data A, gated').hw50:.1f} mm (A) and {g('data C, gated').hw50:.1f} mm (C) of
the source. For the Geant4 truth the value is {g('Geant4 nose A, truth').hw50:.1f} mm, and the reconstruction changes it
to {g('Geant4 nose A, pass-1 reconstruction').hw50:.1f} mm. The Geant4 width depends strongly on the electron energy:
{g('Geant4 nose A, truth, 1-2 MeV').hw50:.0f} mm at 1 to 2 MeV and {g('Geant4 nose A, truth, 4-20 MeV').hw50:.0f} mm above
4 MeV. This means that the tails of the image are scattered low-energy electrons and not badly measured tracks. The data
are narrower than Geant4, so the electrons that give tracks in the data appear to be more energetic than in the Geant4
library. That is a question for the Geant4 source model and the trigger emulation, like the same-chamber pair rate.</p>
<p>We also looked for a track property that separates pointing from non-pointing tracks after 20 ms. There is none that
matters: the half-width stays between 10.3 and 10.7 mm (A) across χ², strip count, seed size, t0, the x to y charge
ratio, the column duration and the candidate rank. The tracks that miss the capsule by more than 60 mm are
{late_np['A']['gated']:.0%} (A) and {late_np['C']['gated']:.0%} (C) of the late sample. They are spread uniformly over
the chamber, the scintillator wall fires for 84 % (A) and 93 % (C) of them, and their charge is normal. They are real particles that do
not come from the capsule.</p>

<h2>2. Tracks without a time since the flash</h2>
<p>The time since the flash and the scintillator flags of a trigger come from the slim n_TOF file of its sub-run. A
sub-run whose DREAM acquisitions span two n_TOF run numbers has two slim files. Four readers take only the first file:
<code>trigger_time.slim_path</code>, <code>scintillators.read_slim</code> (and so the slim export),
<code>candidate_filter</code> (stage 1) and the slim inventory of <code>freeze_sample</code>. {slim_txt}For example,
run_116 stat090_0027 has the files 224639 (events 1 to 7231) and 224640 (events 8027 to 110526), and only the first is
used.</p>
<figure><img src="{f_unt}"><figcaption><b>Figure 2.</b> Fraction of the reconstructed tracks of chambers A, C and D
without a time since the flash, for each production run.</figcaption></figure>
<p>Over the production runs {untimed:.1%} of the tracks have no time. These tracks are dropped by every study that
selects on the time since the flash, so the capsule imaging, the pair studies and the scintillator maps use about 85 %
of the data they could use. In addition, <code>improve/line_residuals.py</code> selects beam-off events as "time is
null". For this reason 14,742 of the 46,171 A-C "cosmic" lines that the angle map was fitted on come from production
runs. A refit on the true beam-off runs changes the map by about 0.001 in tan, so the map stands, but finding 1 of the
improve study (13 to 14 % of views off the line) is 6 to 7 % on real cosmic events.</p>

<h2>3. The gated tracks before 12 ms are mostly not particles</h2>
<figure><img src="{f_charge}"><figcaption><b>Figure 3.</b> Charge of the gated tracks (the smaller of the x-view and
y-view charge) for chambers A, C and D. Orange bars are 1 to 8 ms after the flash, coloured bars are 20 to 75 ms. The red
line is the proposed minimum of {Q_MIN:.0f} ADC.</figcaption></figure>
<p>A track from the capsule after 20 ms leaves a median charge of about 1,600 ADC in each view, and fewer than 1.5 % of
these tracks have less than {Q_MIN:.0f} ADC. Before 8 ms most gated tracks have 100 to 400 ADC. These low-charge tracks
are 22 to 30 strips wide, their χ²/dof is 6 to 19, their fitted t0 stays at its starting value, the wall fires for a
quarter of them, and the fraction that points within 15 mm of the capsule is the chance value. The raw waveforms of such
events show about 80 strips above +5σ and 80 strips below −5σ, all peaking in the same sample. So these are fits to the
ringing baseline after the flash. The gate lets them through because it only asks for a column duration, |tan| &lt; 0.5
and an x to y time coincidence. It has no requirement on charge, χ² or significance.</p>
<figure><img src="{f_tof}"><figcaption><b>Figure 4.</b> Top: gated tracks per ms as a function of the time since the
flash, for the present selection (dashed) and the clean selection (solid). Bottom: fraction of the tracks in the central
200 mm that pass within 30 mm of the source.</figcaption></figure>
<div class="scroll">{table(tt, tfmt)}</div>
<p>The clean selection is: at least {Q_MIN:.0f} ADC in each view, a fit window narrower than {N_MAX} strips, less than
{CENS_MAX:.0%} of a window on censored strips, and a χ²/dof below {CHI2_REL:.0f} times the median of late, pointing,
wall-confirmed tracks of that chamber and view. The table below gives what each part removes. The reference tracks are
the tracks after 30 ms that point within 15 mm and have the wall fired.</p>
{table(ct, {'keeps of the reference tracks': '{:.3f}', 'removes before 12 ms': '{:.3f}', 'removes at 12-20 ms': '{:.3f}',
            'removes after 20 ms': '{:.3f}'})}
<p>The existing <code>quality_ok</code> flag does not do this job. Its threshold is 5 times the median χ²/dof of the
sub-run, and the ringing tracks themselves raise that median (to 7.4 on A x, where real tracks have 1.4).</p>

<h3>Effect on the same-chamber pairs</h3>
<figure><img src="{f_pairs}"><figcaption><b>Figure 5.</b> Fraction of tracked events with two or more gated tracks in
one chamber as a function of the time since the flash, for the present selection, with the charge minimum, and with the
clean selection.</figcaption></figure>
<div class="scroll">{table(pt, pfmt)}</div>
<p>The low-charge tracks often come in groups, so they make pairs. At 12 to 15 ms the pair fraction of chamber A falls
from {p('A', 12, 'gated').pair_pct:.2f} % to {p('A', 12, qsel).pair_pct:.2f} % with the charge minimum. The improve study
(finding 9) moved the start of the rate fit from 12 to 25 ms because 12 to 20 ms "carries a different population" with a
worst-leg χ²/dof of 11.9. That population is these tracks. With the charge minimum the pair fraction varies smoothly
from 12 ms on, so the rate fit can be tried again from 12 ms.</p>

<h2>4. Chamber D</h2>
<figure><img src="{f_pos}"><figcaption><b>Figure 6.</b> Position of the gated tracks after 20 ms on chamber A (top) and
chamber D (bottom), in x_local (left) and y_local (right). The coloured line is all gated tracks. The red line is the
tracks whose line passes more than 60 mm from the source. The dotted line is the same after the clean selection.</figcaption></figure>
<p>In chamber A the tracks that do not point are spread uniformly. In chamber D they pile up at a few positions
(x_local near −184, −136, −112 and +124 mm, and y_local near +2 to +30 and −140 mm). These are windows on hot or
partly censored strips that the present rule (an all-censored x window, or a y window at least half censored) does not
catch. Before 20 ms nearly all gated D tracks are of this kind: the clean selection keeps
{t('D', 15, 'clean').rate_per_ms / t('D', 15, 'gated').rate_per_ms:.1%} of them at 15 to 20 ms.</p>
<p>The z position of the source seen through D depends on this contamination. With the present selection the band
crossing is {t('D', 20, 'gated').crossing:.1f} mm at 20 to 30 ms and {t('D', 30, 'gated').crossing:.1f} mm at 30 to
75 ms. With the clean selection it is {t('D', 20, 'clean').crossing:.2f} ± {t('D', 20, 'clean').crossing_err:.2f} and
{t('D', 30, 'clean').crossing:.2f} ± {t('D', 30, 'clean').crossing_err:.2f} mm. The clean selection keeps only
{c('D', 'all (clean)').keeps_reference:.0%} of D's reference tracks, mostly because of the χ² part, so it is a
demonstration and not a final cut for D.</p>

<h2>5. The height of the source</h2>
<figure><img src="{f_height}"><figcaption><b>Figure 7.</b> Height at which the track line meets the source plane, for the
late tracks that point within 15 mm of the source in the other coordinate. Coloured lines: data, chambers A, C and D.
Black dashed: Geant4 truth. Black dotted: Geant4 through the reconstruction. The grey band is the drawn gas volume of the
capsule (−29.5 to +50.7 mm).</figcaption></figure>
{table(E.rename(columns={'edge_lo': 'lower half-maximum [mm]', 'edge_hi': 'upper half-maximum [mm]', 'length': 'length [mm]',
                         'centre': 'centre [mm]'}),
       {'lower half-maximum [mm]': '{:.1f}', 'upper half-maximum [mm]': '{:.1f}', 'length [mm]': '{:.1f}', 'centre [mm]': '{:.1f}'})}
<p>In all three chambers the source is a flat line in height, from {e('data A').edge_lo:.0f} mm to about
{e('data A').edge_hi:.0f} mm, with a sharp lower edge and a softer upper edge. Its length is {e('data A').length:.0f}
to {e('data D').length:.0f} mm, which is close to the length of the capsule. Geant4 with the centred, untilted capsule
gives a compact peak near −30 mm, and the reconstruction does not stretch it. So the flat line is not a tracking effect.
The band crossing (−30 mm) is the centre of this line. It agrees with the Geant4 nose position, but the shapes do not
agree: in the data the emitting length reaches about 40 mm below the drawn nose. This is a constraint for the capsule
pose question and it is not used in the pose fits.</p>

<h2>6. The x offset of the source is not a tracking effect</h2>
<p>The late band crossing is {t('A', 30, 'clean').crossing:.2f} mm in A and {t('C', 30, 'clean').crossing:.2f} mm in C,
with or without the clean selection, and the same from 15 ms on. A and C look at the capsule from opposite sides, so
their local x axes point in opposite global directions. A bias of the reconstruction that is the same in the local frame
of both chambers (an offset of the slope, of the strip coordinate, of t0, or of the reference depth) therefore moves the
two images in opposite global directions. It cannot produce a common offset. The A-C cosmic lines cannot see a common
shift of the chamber pair relative to the capsule either. Note that the in-situ map is fitted on the A-C lines. For this reason the A and C images agree almost by
construction, and half of any relative x misalignment of A and C goes into the common position. The difference A minus C
is therefore no longer an independent measurement that a pose fit can use.</p>
<p>Before 12 ms the crossings of A and C move apart in global x (A to {t('A', 4, 'clean').crossing:.0f} mm, C to
{t('C', 4, 'clean').crossing:+.0f} mm at 4 to 8 ms). This is the same direction in the local frame of each chamber, so it
is contamination of the estimator and not a displacement of the source.</p>

<h2>7. The imaging estimators</h2>
<p>We also reviewed the estimators themselves, with toy sources and by injecting early tracks into late samples. None
has a sign or frame error, and the x offset of the source is not produced by an estimator: four estimators give −10.4 to
−10.7 mm (A) and −10.5 to −11.4 mm (C) on the analysis view. However, several quoted numbers are inflated or are not the
same quantity.</p>
<ul>
<li><b>D's z from <code>capsule_tof tilt</code> (−1.7 mm) is not a measurement.</b> The step uses all D tracks after
12 ms, and 59 % of that sample are the non-pointing tracks of section 4. The robust line in <code>crossing</code>
clips at 3 times the MAD of the residuals. On D the MAD is 52 mm, so nothing is clipped and the fitted slope is −0.15
where a pointing sample gives −1. The value changes from −1.7 to +7.5 to −2.7 mm between the three views. With late,
wall-confirmed tracks four estimators give −3.8 to −4.8 mm, and the clean selection above gives −4.1 mm. We would quote
z = −4.5 ± 1 mm.</li>
<li><b><code>crossing</code> fails abruptly when the background grows.</b> Adding 20 % of early tracks to the late
samples moves A by −0.9 mm and C by −1.1 mm, and adding 40 % moves A by −7 mm. D's late sample moves by −6.6 mm with
only 10 %. A fit restricted to 100 mm around the crossing is much more robust (A moves by 0.2 mm at 40 %).</li>
<li><b><code>campaign_imaging</code> does not read the analysis view.</b> It uses the raw fit tangent of the reco tree
with no <code>gated</code>, no map, no censored-window cut and no cut on the time since the flash, so its output is
identical in the map and no-map chains. Its fit window (30 to 130 mm about the surveyed foot) is truncated on one side
by the plastic acceptance, which moves A by +0.5 and C by −0.4 mm. About half of the "A-C half difference 1.16 mm" is
this truncation. So the −9.1 mm of the imaging step and the −10.9 mm of the pose step are not the same quantity.</li>
<li><b>The focus scale split by the sign of tan is the nominal axis.</b> <code>focus_split</code> and
<code>capsule_focus</code> measure the miss about x = 0, and the source is at −10.9 mm. About the measured source the
split disappears (A 0.86 / 1.14 becomes 1.00 / 1.00, C 1.14 / 0.85 becomes 0.98 / 1.01). The pooled scale is also about
0.015 too low for the same reason, so the common scale is 0.994 and not 0.977.</li>
<li><b>The pose toy has a fixed A minus C offset.</b> <code>capsule_tof.FOOT_SIM['C']</code> is 16.4 mm and the Geant4
geometry has 17.3 mm (<code>SimConfig.hh</code>, <code>run_digi.FOOT</code>). This makes the toy's x in C 0.9 mm too
negative. The rest of the +2.3 to 2.7 mm that every pose shows is probably the size of the library (about 2,500 capsule
tracks per chamber, reused in every pose).</li>
<li><b>Smaller points.</b> The null of <code>track_quality</code> shuffles inside |tan| bins and keeps part of the
pointing, so its excess is about one third low. Its cache (<code>tracks_AC.parquet</code>) predates the map and is
returned whatever the view. <code>capsule_pose.Z_BD</code> is from the retired chain.</li>
</ul>

<h2>8. Code errors</h2>
<ul>
<li><b>Scintillator stack, y slope.</b> The analysis view stores <code>tan_raw_y</code> with the sign of
<code>tany</code>. <code>ntof_scint_stack/extract.py</code> multiplies it by <code>TAN_SIGN_Y</code> again, so
<code>tan_v_raw</code> = −<code>tany</code> for every track (checked on run_145, chambers A, C and D). The along-bar
prediction is therefore extrapolated in the wrong direction, and the fitted along-bar scale is 0. The x direction
(λ = 0.85) is not affected.</li>
<li><b>Bow mapping of paired tracks.</b> <code>wft/reco.py</code> (<code>_worker_fit</code>) maps the slopes of each
pair with the pair's own coordinates and then maps the two plane winners together. When the winners belong to different
pairs, each takes the coordinate of the other pair. {bow_sum}{bow_txt}</li>
<li><b>Smaller points.</b> Stage 3 places the fitted position at the nominal strip plane and does not apply the bow
mesh height (|tan·h| up to 0.9 mm for beam tracks). <code>dca_axis_mm</code> and <code>in_bore</code> are distances to
the nominal axis, 11 mm from the measured source. The stage-1 merge duplicates two rows (run_139). On chamber D in the
early runs the reconstruction's own slope cut ends at 0.48 to 0.50 for a quarter of the tracks, below the 0.5
acceptance. <code>map_view.py</code> has no guard against being applied to an already mapped view. Two-track children
ignore the t0 pin (31 % are more than 180 ns from it). The head-on rescue is not wired into the beam driver, and chamber
C has half the tracks of chamber A at |tan x| &lt; 0.08.</li>
</ul>

<h2>What this does not rule out</h2>
<ul>
<li>We did not test the efficiency of the reconstruction for real tracks before 12 ms. The clean selection removes the
ringing fits, but a real track on a ringing baseline may also fail it. Geant4 overlays on early triggers would measure
this.</li>
<li>The Geant4 comparison of the image width uses the capsule libraries as they are. The data are narrower than Geant4.
We did not find out whether this is the electron spectrum, the trigger (wall and plastic) or the source extent.</li>
<li>The x offset of 10 mm and the height profile are geometry or physics. Neither the tracks nor the cosmic lines can
tell whether the chambers or the capsule are displaced.</li>
<li>The clean selection was chosen by hand on the whole campaign. Its thresholds were not optimised, and for chamber D
it is too strict.</li>
<li>Chamber B was not audited (it has positions only).</li>
</ul>
</body></html>"""
    (out / 'report.html').write_text(page)
    print('wrote', out / 'report.html')
    if '--copy' in sys.argv:
        dst = REPO / 'rerun' / 'ntof' / 'improve' / 'audit_report'
        (dst / 'figures').mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / 'report.html', dst / 'report.html')
        for f in (out / 'figures').glob('*'):
            shutil.copy2(f, dst / 'figures' / f.name)
        for f in out.glob('*.csv'):
            shutil.copy2(f, dst / f.name)
        print('copied to', dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
