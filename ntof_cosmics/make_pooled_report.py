#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build results/tracking/pooled/report.html: the pooled run_149 through-goers
(`pool_tracking.py`) and the angle response measured on them and tested on
beam (`angle_response.py`).

Generated, not hand-written: re-run those two, then this, and the numbers,
figures and verdict move together.  Figures are referenced as figures/x.png
(relative), so the page works from disk, from the DAQ Analysis tab, or copied.

    .venv/bin/python ntof_cosmics/make_pooled_report.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'sept26_prelim_analysis'))
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

import cosmic_tracks as CT  # noqa: E402

P = CT.OUT / 'pooled'
FIG = P / 'figures'
KS = ('run_147', 'run_150')
VIEWS = [('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y')]


def load():
    d = dict(
        pooled=json.loads((P / 'pooled_run_149.json').read_text()),
        ar=json.loads((P / 'angle_response.json').read_text()),
        R=pd.read_csv(P / 'response.csv'),
        G=pd.read_csv(P / 'response_gradient.csv'),
        M=pd.read_csv(P / 'response_models.csv'),
        Rs=pd.read_csv(P / 'resolution.csv'),
        DS=pd.read_csv(P / 'drift_span.csv'),
        SF=pd.read_csv(P / 'slope_flag.csv'),
        KT=pd.read_csv(P / 'k_true_by_borrowed_k.csv'),
        BR=pd.read_csv(P / 'beam_response.csv'),
        BC=pd.read_csv(P / 'beam_closure.csv'),
        pairs={k: pd.read_parquet(P / f'pairs_run_149_k{k}.parquet') for k in KS},
        sample=pd.read_parquet(P / 'ac_sample_run_149_krun_150.parquet'),
    )
    return d


def k_applied(D, arm):
    """The beam k_arm values the per-track tables were built with."""
    return {k: float(D['KT'].query('k_from == @k and arm == @arm').k_borrowed.iloc[0])
            for k in KS}


# --------------------------------------------------------------------------- #
def figures(D) -> list[tuple[str, str]]:
    fs.use()
    out = []
    R = D['R']
    BR = D['BR']

    # 1 -- the response: raw/true against |true tan|
    fig, axs = fs.plt.subplots(2, 2, figsize=fs.FULL, sharex=True, sharey='row')
    for ax, (arm, view) in zip(axs.flat, VIEWS):
        col = fs.DET_COLOR[arm]
        for sel, ls in (('sep20', ':'), ('nosep', '--')):
            g = R.query('selection == @sel and arm == @arm and axis == @view')
            ax.plot(g.j_med, g.ratio_med, ls=ls, color=fs.MUTED, lw=0.9,
                    label={'sep20': 'cosmic, sep < 20 mm', 'nosep': 'cosmic, no sep cut'}[sel])
        g = R.query('selection == "primary" and arm == @arm and axis == @view')
        ax.errorbar(g.j_med, g.ratio_med, yerr=g.ratio_med_err, marker=fs.DET_MARKER[arm],
                    ms=5, color=col, lw=1.4, capsize=0, label='cosmic (primary)')
        kb = k_applied(D, arm)
        ax.axhspan(1 / max(kb.values()), 1 / min(kb.values()), color=fs.COPPER, alpha=0.18,
                   lw=0, label='1/k, beam k_arm 147–150')
        if view == 'x':
            b = BR[BR.arm == arm].groupby('lo').agg(j=('j_med', 'median'), r=('ratio_med', 'median'),
                                                    lo_=('ratio_med', 'min'), hi_=('ratio_med', 'max'))
            ax.errorbar(b.j, b.r, yerr=[b.r - b.lo_, b.hi_ - b.r], marker='s', ms=5,
                        mfc='white', mec=fs.INK, mew=1.2, color=fs.INK, lw=1.0, capsize=0,
                        label='beam, capsule-pointing (runs 145–152)')
        ax.set_title(f'chamber {arm}, {view} view', fontsize=10, color=fs.MUTED, loc='left')
        ax.set_xlim(0.08, 0.6)
    for ax in axs[1]:
        ax.set_xlabel('|true tan|  (cosmic: joined A–C line; beam: lever / 234.6 mm)')
    for ax in axs[:, 0]:
        ax.set_ylabel('raw tan ÷ true tan  (= 1/k)')
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    axs[0, 1].legend(h0, l0, frameon=False, fontsize=8, loc='lower left')
    m = D['M'].set_index(['arm', 'axis'])
    fs.fig_title(fig, 'One k per chamber does not describe the response: it falls with '
                      'angle, and faster on beam tracks',
                 f"run_149 through-goers, {D['ar']['n_selected']['primary']:,} A–C pairs "
                 '(primary: one gated track per arm, sep < 60 mm); band = 1/k the beam '
                 'tables use')
    fs.preliminary(axs[0, 1])
    fs.save(fig, FIG / 'response', data={'cosmic': R, 'beam': BR})
    out.append(('response.png',
                'Raw reconstructed tan divided by the true tan, in bins of the true tan. A single '
                'multiplicative k would be a horizontal line. Cosmic (coloured): the true slope is '
                'the line through both chambers, 469 mm lever arm, no capsule assumption; the '
                f"ratio falls by {abs(m.loc[('A', 'x')].ratio_trend_per_unit_tan):.2f} (A x), "
                f"{abs(m.loc[('C', 'x')].ratio_trend_per_unit_tan):.2f} (C x) and "
                f"{abs(m.loc[('C', 'y')].ratio_trend_per_unit_tan):.2f} (C y) per unit tan, "
                f"{m.loc[('A', 'x')].trend_signif:.0f}σ, {m.loc[('C', 'x')].trend_signif:.0f}σ "
                f"and {m.loc[('C', 'y')].trend_signif:.0f}σ (sub-run bootstrap); A y is nearly "
                'flat. The two grey variants (stricter and no lateral-miss cut) show it is not the '
                'selection. Beam (open squares, x only): k_arm\'s pointing-coincident sample, true '
                'tan from the capsule; median of the four runs, bars the run-to-run range. It agrees '
                'with the cosmics only below |tan| ≈ 0.15 and falls far more steeply. The orange band '
                'is 1/k as applied: it sits on neither curve except at one angle.'))

    # 2 -- resolution
    Rs = D['Rs']
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE)
    for arm, view in VIEWS:
        g = Rs.query('arm == @arm and axis == @view')
        x = 0.5 * (g.lo + g.hi)
        kw = dict(color=fs.DET_COLOR[arm], marker=fs.DET_MARKER[arm], ms=5,
                  ls='-' if view == 'x' else '--', lw=1.3, label=f'{arm} {view}')
        axs[0].plot(x, g.sigma, **kw)
        axs[1].plot(x, g.tail_frac * 100, **kw)
    for ax in axs:
        ax.axvspan(0, 0.08, color=fs.BAND_DEAD, alpha=0.10, lw=0)
        ax.set_xlabel('|true tan| (joined A–C line)')
        ax.set_xlim(0, 0.6)
    axs[0].set_yscale('log')
    axs[0].axhline(0.022, color=fs.MUTED, lw=0.8, ls=':')
    axs[0].text(0.59, 0.024, 'quoted tan_err (A)', ha='right', fontsize=8, color=fs.MUTED)
    axs[0].set_ylabel('σ of corrected tan − true tan (robust)')
    axs[1].set_ylabel('% of tracks off by > 0.15 in tan')
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    near = Rs[Rs.hi <= 0.08]
    core = Rs[(Rs.lo >= 0.12) & (Rs.hi <= 0.45)]
    fs.fig_title(fig, f'Below |tan| 0.08 the angle is not measured (σ {near.sigma.min():.2f}–'
                      f'{near.sigma.max():.2f}); at 0.12–0.45 it is good to {core.sigma.min():.3f}–'
                      f'{core.sigma.max():.3f}',
                 'per-track error against the joined line, after the cosmic response '
                 'correction; shaded: |tan| < 0.08')
    fs.preliminary(axs[1])
    fs.save(fig, FIG / 'resolution', data=Rs)
    out.append(('resolution.png',
                'Per-track angle error measured against an external truth (the A–C line), in '
                'true-tan units: the raw tan mapped through the cosmic response, minus the joined '
                f"slope. Below |tan| ≈ 0.08 the scatter is {near.sigma.min():.2f}–"
                f"{near.sigma.max():.2f}, i.e. the fit returns essentially any angle; at "
                f"0.12–0.45 it is {core.sigma.min():.3f}–{core.sigma.max():.3f}. The table's own "
                'tan_err is a constant (0.022 in A, 0.026 in C, dotted) and does not know about '
                'either regime: pulls are ~1.3 in the core and 9–16 near normal.'))

    # 3 -- scatter, raw vs true, x view
    S = D['sample']
    S = S[(S.n_A == 1) & (S.n_C == 1) & (S.sep_mm < 60)]
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE, sharex=True, sharey=True)
    rows = []
    for ax, arm in zip(axs, 'AC'):
        j, raw = S.jx.to_numpy(), S[f'{arm}_raw_x'].to_numpy()
        ax.plot(j, raw, '.', ms=2, color=fs.DET_COLOR[arm], alpha=0.35, rasterized=True)
        g = R.query('selection == "primary" and arm == @arm and axis == "x"')
        ax.plot(g.j_med, g.raw_med, 'o', color=fs.INK, ms=4, label='bin medians (folded)')
        ax.plot(-g.j_med, -g.raw_med, 'o', color=fs.INK, ms=4)
        xx = np.array([-0.65, 0.65])
        kb = k_applied(D, arm)['run_150']
        ax.plot(xx, xx / kb, color=fs.COPPER, lw=1.2, label=f'beam k_arm = {kb:.2f}')
        km = float(D['M'].query('arm == @arm and axis == "x"').k_mult.iloc[0])
        ax.plot(xx, xx / km, color=fs.MUTED, lw=1.0, ls='--', label=f'best single k = {km:.2f}')
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.65)
        ax.set_xlabel('joined A–C line, dx/dz')
        ax.set_title(f'chamber {arm}, x view', fontsize=10, color=fs.MUTED, loc='left')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        rows.append(pd.DataFrame(dict(arm=arm, j=j, raw=raw)))
    axs[0].set_ylabel('raw track tan (global sign)')
    fs.fig_title(fig, 'The bin medians bend away from every straight line, and the fit '
                      'avoids tans near zero', 'run_149 through-goers, primary selection')
    fs.save(fig, FIG / 'scatter', data=pd.concat(rows, ignore_index=True))
    out.append(('scatter.png',
                'Every primary-selection A–C pair: raw track tan against the joined-line slope. The '
                'bin medians bend away from any straight line through the origin. Few tracks '
                'reconstruct with |tan| < 0.1, and the true near-normal tracks that do exist scatter '
                'across the plot: the fit pushes them away from zero rather than measuring ~0.'))

    # 4 -- beam closure
    BC = D['BC'].groupby(['arm', 'run', 'tans'])[['band', 'track']].median().reset_index()
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    for ax, arm in zip(axs, 'AC'):
        g = BC[BC.arm == arm]
        runs = sorted(g.run.unique(), key=lambda r: int(r.split('_')[1]))
        x = np.arange(len(runs))
        for tans, mfc, off in (('raw', None, -0.08), ('cosmic_corrected', 'white', 0.08)):
            h = g[g.tans == tans].set_index('run').loc[runs]
            lab = 'raw tans' if tans == 'raw' else 'cosmic-corrected tans'
            ax.plot(x + off, h.band, 'o', ms=6, color=fs.DET_COLOR[arm], mfc=mfc,
                    mec=fs.DET_COLOR[arm], mew=1.3, label=f'band (gradient), {lab}')
            ax.plot(x + off, h.track, 's', ms=6, color=fs.INK, mfc=mfc, mec=fs.INK, mew=1.3,
                    label=f'track (median ratio), {lab}')
        ax.axhline(1.0, color=fs.MUTED, lw=0.8)
        ax.set_xticks(x, runs)
        ax.set_title(f'chamber {arm}, x view', fontsize=10, color=fs.MUTED, loc='left')
    axs[0].set_ylabel("k_arm estimator on the beam sample")
    axs[0].legend(frameon=False, fontsize=7.5, loc='upper right')
    fs.fig_title(fig, 'The cosmic response does not close on beam tracks',
                 "k_arm's band and track estimators, per run (median of sub-runs); "
                 'closure would put every open marker on 1')
    fs.preliminary(axs[1])
    fs.save(fig, FIG / 'beam_closure', data=D['BC'])
    out.append(('beam_closure.png',
                "k_arm's two non-scan estimators on its pointing-coincident beam sample (rebuilt "
                'from the campaign track table; reproduces k_arm to ~3 %), with raw tans (filled) and '
                'after mapping each tan through the cosmic response (open). If the cosmic response '
                'were the beam response, both open markers would sit on 1 and on each other. They '
                'do not: A still needs ~10 %, and in C the band and track estimators still disagree '
                'by ~15 %, the same gap as on raw tans.'))

    # 5 -- 170 deg on the pooled sample
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    bins = np.arange(90, 181, 2.5)
    rows = []
    for ax, k in zip(axs, KS):
        p = D['pairs'][k].sort_values('open_deg', ascending=False).drop_duplicates(['subrun', 'event_id'])
        p = p[p.topo == 'opposing']
        c = p[p.sep_mm < CT.CLEAN_SEP_MM]
        ax.hist(p.open_deg, bins=bins, histtype='stepfilled', color=fs.GRID, ec=fs.MUTED,
                label=f'all opposing ({len(p):,})')
        ax.hist(c.open_deg, bins=bins, histtype='step', color=fs.ACCENT, lw=1.4,
                label=f'lines meet < {CT.CLEAN_SEP_MM:g} mm ({len(c):,})')
        ax.axvline(CT.BACK_TO_BACK_DEG, color=fs.COPPER, lw=1, ls='--')
        ax.set_xlabel('opening angle [deg]')
        ax.set_title(f'k from {k}', fontsize=10, color=fs.MUTED, loc='left')
        rows.append(p[['subrun', 'event_id', 'open_deg', 'sep_mm']].assign(k_from=k))
    axs[0].set_ylabel('triggers per 2.5°')
    axs[0].legend(frameon=False, loc='upper left')
    po = D['pooled']
    fs.fig_title(fig, f"{po['run_150']['frac_clean_above_170'] * 100:.0f}–"
                      f"{po['run_147']['frac_clean_above_170'] * 100:.0f} % of clean "
                      'through-goers pass the 170° cut, under an angle scale known to be wrong',
                 'run_149, 87 sub-runs, most collinear opposing pair per trigger')
    fs.save(fig, FIG / 'opening_angle', data=pd.concat(rows, ignore_index=True))
    out.append(('opening_angle.png',
                'Pooled opening-angle distribution of the most collinear opposing pair per trigger. '
                'The 170° fraction depends on the angle scale, which this page shows is neither a '
                'single number nor the same for beam and cosmic tracks; it is therefore not yet a '
                'basis for moving BACK_TO_BACK_DEG.'))
    return out


# --------------------------------------------------------------------------- #
def table(df, cols, heads, fmts):
    th = ''.join(f'<th>{h}</th>' for h in heads)
    tr = ''
    for _, r in df.iterrows():
        tr += '<tr>' + ''.join(f'<td>{f.format(r[c]) if isinstance(f, str) else f(r[c])}</td>'
                               for c, f in zip(cols, fmts)) + '</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def main() -> None:
    D = load()
    figs = figures(D)
    po, ar, M, KT, Rs, DS = D['pooled'], D['ar'], D['M'], D['KT'], D['Rs'], D['DS']
    BC = D['BC'].groupby(['arm', 'run', 'tans'])[['band', 'track']].median().reset_index()
    m = M.set_index(['arm', 'axis'])
    kt = KT.groupby(['arm', 'axis']).k_true.agg(['min', 'max'])
    kb = {a: k_applied(D, a) for a in 'AC'}
    near = Rs[Rs.hi <= 0.08]
    near0, near1 = Rs[Rs.hi <= 0.04], Rs[(Rs.lo >= 0.04) & (Rs.hi <= 0.08)]
    core = Rs[(Rs.lo >= 0.12) & (Rs.hi <= 0.45)]
    ca = BC.query('tans == "cosmic_corrected"')
    mult = ar['multiplicity']
    BR = D['BR']
    br_lo = BR[BR.lo == 0.10].groupby('arm').ratio_med.median()
    br_hi = BR[BR.lo == 0.45].groupby('arm').ratio_med.median()
    Rp = D['R'].query('selection == "primary" and axis == "x"')
    cr_lo = Rp[Rp.lo == 0.10].set_index('arm').ratio_med
    cr_hi = Rp[Rp.lo == 0.45].set_index('arm').ratio_med

    fig_html = ''.join(f'<figure><img src="figures/{f}" alt=""><figcaption>{html.escape(c)}'
                       f'</figcaption></figure>' for f, c in figs)

    kt_tab = table(
        KT, ['k_from', 'arm', 'axis', 'k_borrowed', 'median_ratio', 'k_true'],
        ['k borrowed from', 'arm', 'view', 'k<sub>borrowed</sub>', 'median s/j',
         'k<sub>true</sub> = k<sub>b</sub> / ratio'],
        ['{}', '{}', '{}', '{:.3f}', '{:.3f}', '<b>{:.3f}</b>'])
    m_tab = table(
        M, ['arm', 'axis', 'n', 'k_mult', 'k_grad', 'offset', 'ratio_trend_per_unit_tan',
            'ratio_trend_err', 'trend_signif'],
        ['arm', 'view', 'n', 'single k', 'gradient k<sub>d</sub>', 'offset c (raw)',
         'd(ratio)/d|tan|', '± (sub-run boot.)', 'σ'],
        ['{}', '{}', '{:,}', '{:.3f}', '{:.3f}', '{:+.4f}', '{:+.3f}', '{:.3f}', '{:.1f}'])
    rp = D['R'].query('selection == "primary"').copy()
    rp['bin'] = [f'{a:.2f}–{b:.2f}' for a, b in zip(rp.lo, rp.hi)]
    r_tab = ''
    for arm, view in VIEWS:
        g = rp.query('arm == @arm and axis == @view')
        r_tab += f'<h3>{arm} {view}</h3>' + table(
            g, ['bin', 'n', 'j_med', 'raw_med', 'ratio_med', 'ratio_med_err', 'k_eff',
                'ratio_med_pos', 'ratio_med_neg'],
            ['|true tan|', 'n', 'median |j|', 'median raw', 'raw/true', '±', 'k_eff',
             'raw/true, j &gt; 0', 'raw/true, j &lt; 0'],
            ['{}', '{:,}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.3f}'])
    Rs2 = Rs.copy()
    Rs2['bin'] = [f'{a:.2f}–{b:.2f}' for a, b in zip(Rs2.lo, Rs2.hi)]
    rs_tab = table(Rs2, ['arm', 'axis', 'bin', 'n', 'bias', 'sigma', 'tail_frac', 'tan_err_quoted',
                         'pull_mad'],
                   ['arm', 'view', '|true tan|', 'n', 'median', 'σ (MAD)', '> 0.15',
                    'quoted tan_err', 'pull σ'],
                   ['{}', '{}', '{}', '{:,}', '{:+.3f}', '{:.3f}', lambda v: f'{v * 100:.0f} %',
                    '{:.3f}', '{:.1f}'])
    bc_tab = table(BC, ['arm', 'run', 'tans', 'band', 'track'],
                   ['arm', 'run', 'tans', 'band k', 'track k'],
                   ['{}', '{}', '{}', '{:.3f}', '{:.3f}'])
    ds_tab = table(DS, ['arm', 'axis', 'n', 'frac_railed', 'span_med_ns', 'v_um_ns', 'v_prior_over_v'],
                   ['arm', 'view', 'n (unrailed)', 'railed', 'median span [ns]',
                    'v = 30 mm / span [µm/ns]', '42.6 / v'],
                   ['{}', '{}', '{:,}', lambda v: f'{v * 100:.0f} %', '{:.0f}', '{:.1f}', '{:.2f}'])
    sr = pd.DataFrame(po['run_150']['slope_ratio'])
    SF = D['SF']
    sf_tab = table(SF, ['arm', 'axis', 'n_true_near', 'frac_true_near_flagged_reliable',
                        'frac_reliable_true_near', 'n_unreliable', 'unreliable_true_median',
                        'frac_unreliable_true_below_008'],
                   ['arm', 'view', 'true |tan| &lt; 0.06', '… flagged reliable',
                    'reliable tracks that are truly &lt; 0.06', 'flagged unreliable',
                    '… median true |tan|', '… truly &lt; 0.08'],
                   ['{}', '{}', '{:,}', lambda v: f'{v * 100:.0f} %', lambda v: f'{v * 100:.1f} %',
                    '{:,}', '{:.3f}', lambda v: f'{v * 100:.0f} %'])

    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmic angle response</title>{HEAD}</head><body><main>
<p class="eyebrow">n_TOF EAR2 · X17 · ntof_cosmics/angle_response</p>
<h1>The angle response of chambers A and C, from 87 sub-runs of beam-off cosmics</h1>

<div class="verdict"><p><b>Do not apply a cosmic k, and do not trust a single k.</b>
(1) The reconstructed tan is <b>not</b> a fixed multiple of the true tan: on
through-goers, with the line through both chambers as truth, raw/true falls with
angle in A x, C x and C y at {min(m.loc[v].trend_signif for v in VIEWS if v != ('A', 'y')):.0f}–{max(m.loc[v].trend_signif for v in VIEWS if v != ('A', 'y')):.0f}σ
(A y nearly flat). (2) Beam capsule tracks have a response of the same kind but
<b>much steeper</b>: in A x raw/true goes {br_lo['A']:.2f} → {br_hi['A']:.2f} on beam
against {cr_lo['A']:.2f} → {cr_hi['A']:.2f} on cosmics over |tan| 0.10 → 0.50, and
mapping the beam tans through the cosmic response leaves k_arm's estimators at
{ca[ca.arm == 'A'].band.median():.2f}/{ca[ca.arm == 'A'].track.median():.2f} (A) instead of
1. So the cosmic calibration does not transfer, and that disagreement is the
result. (3) <b>Near normal incidence there is no angle</b>: below |tan| 0.04 the
per-track scatter is {near0.sigma.min():.2f}–{near0.sigma.max():.2f}, and
{near1.sigma.min():.2f}–{near1.sigma.max():.2f} at 0.04–0.08. Neither the quoted
tan_err nor <code>slope_reliable</code> flags it: {SF.frac_true_near_flagged_reliable.min() * 100:.0f}–{SF.frac_true_near_flagged_reliable.max() * 100:.0f} %
of truly near-normal tracks are flagged reliable.</p>
<p>A correction to the handoff of 2026-10-06: the slope ratio is
s/j = k<sub>borrowed</sub>/k<sub>true</sub>, so a ratio above 1 means the tan reads
too <b>large</b>. The single-k summary of the cosmics is k<sub>A</sub> ≈
{kt.loc[('A', 'x')]['min']:.2f}, k<sub>C</sub> ≈ {kt.loc[('C', 'x')]['min']:.2f} (x) /
{kt.loc[('C', 'y')]['min']:.2f} (y), the same under either borrowed k, against
the beam's {min(kb['A'].values()):.2f}–{max(kb['A'].values()):.2f} and
{min(kb['C'].values()):.2f}–{max(kb['C'].values()):.2f}.</p></div>

<h2>Headline numbers</h2>
<table><thead><tr><th></th><th>value</th></tr></thead><tbody>
<tr><td>A–C pairs (gated, calibrated), run_149</td><td>{ar['n_ac_pairs']:,}</td></tr>
<tr><td>primary selection (one gated track per arm, lines miss &lt; 60 mm)</td><td>{ar['n_selected']['primary']:,}</td></tr>
<tr><td>single-k summary, k<sub>true</sub> = k<sub>b</sub>/ratio (A x, A y, C x, C y)</td>
<td>{kt.loc[('A', 'x')]['min']:.3f}, {kt.loc[('A', 'y')]['min']:.3f}, {kt.loc[('C', 'x')]['min']:.3f}, {kt.loc[('C', 'y')]['min']:.3f}
(spread over borrowed k ≤ {(kt['max'] - kt['min']).max():.3f})</td></tr>
<tr><td>beam k_arm applied (run_147 / run_150)</td><td>A {kb['A']['run_147']:.3f} / {kb['A']['run_150']:.3f};
C {kb['C']['run_147']:.3f} / {kb['C']['run_150']:.3f}</td></tr>
<tr><td>trend of raw/true per unit tan (A x, A y, C x, C y)</td>
<td>{', '.join(f"{m.loc[v].ratio_trend_per_unit_tan:+.2f} ± {m.loc[v].ratio_trend_err:.2f}" for v in VIEWS)}</td></tr>
<tr><td>per-track σ<sub>tan</sub>, |tan| 0.12–0.45</td><td>{core.sigma.min():.3f}–{core.sigma.max():.3f}</td></tr>
<tr><td>per-track σ<sub>tan</sub>, |tan| &lt; 0.04 / 0.04–0.08</td><td>{near0.sigma.min():.2f}–{near0.sigma.max():.2f} / {near1.sigma.min():.2f}–{near1.sigma.max():.2f}</td></tr>
<tr><td>clean single-muon crossings with a 2nd gated track in A / C</td>
<td>{mult['frac_A_ge2'] * 100:.1f} % / {mult['frac_C_ge2'] * 100:.1f} % of {mult['n_clean']:,}
(all ≥ 6 cm away, sharing no view: second particles, not splits)</td></tr>
<tr><td>clean opposing pairs &gt; 170° (k 147 / 150)</td>
<td>{po['run_147']['frac_clean_above_170'] * 100:.1f} ± {po['run_147']['frac_clean_above_170_err'] * 100:.1f} % /
{po['run_150']['frac_clean_above_170'] * 100:.1f} ± {po['run_150']['frac_clean_above_170_err'] * 100:.1f} %</td></tr>
</tbody></table>

<h2>What was compared</h2>
<ul>
<li><b>Truth, cosmic:</b> the line through chamber A's and chamber C's mesh-plane
crossings (z = ±234.6 mm, 469 mm apart). Positions do not depend on k, and a
0.5 mm position error is 0.002 in slope, so this slope is truth here. It is also
why the handoff's regression-dilution reading of the least-squares ratio does not
hold: the noise is in the track slope, which least squares handles. The
least-squares and median ratios differ because the ratio changes with angle.</li>
<li><b>Measured, cosmic:</b> raw = track slope ÷ borrowed k. Identical under
both borrowed k (max difference {max(v for k, v in ar['k_independence_max_abs_raw_diff'].items() if k != 'n_common'):.0e}
over {ar['k_independence_max_abs_raw_diff']['n_common']:,} pairs), so nothing below depends on which
k the tables were built with.</li>
<li><b>Beam:</b> k_arm's pointing-coincident sample (gated, own arm fired, x
charge 25–75 %, lever 30–130 mm), from the campaign track table, runs
145/147/150/152 (run_149's neighbours on both sides). True tan = lever/234.6 mm,
x only. The local <code>k_arm.coincident_tracks</code> path finds only ~10 % of
the sample condor used (the local slim export differs), so it was rebuilt from
<code>coinc_this_arm</code>; per sub-run it reproduces k_arm's band/track to ~3 %.</li>
<li><b>Errors</b> are a bootstrap over sub-runs, so statistical only.</li>
</ul>

<h2>Single-k summary under each borrowed k</h2>
<p>The handoff's sep &lt; 20 mm sample. It is shown so the corrected direction can be
checked against the old numbers; the angle dependence below is why one k is not
the right model.</p>
{kt_tab}

<h2>Response models</h2>
<p>Robust (Huber) fits of the folded raw tan on |true tan| in 0.1–0.6, primary
selection. A gradient plus a small outward offset describes the cosmics; it is
why k_arm's band (gradient) estimator exceeds its track (median ratio) estimator
in every arm. That excess is not evidence of a drift-velocity error. The trend
column is the slope of the per-track ratio against |tan|; zero would mean one
k is enough.</p>
{m_tab}

<h2>Response by angle (primary selection)</h2>
<p>raw/true and k_eff = 1/(raw/true), folded, with the two signs shown apart.</p>
{r_tab}

<h2>Per-track resolution against the joined line</h2>
{rs_tab}

<h2>Does <code>slope_reliable</code> catch near-normal tracks?</h2>
<p><code>wft</code> flags a view unreliable when |raw tan| &lt; 0.08
(<code>TAN_MIN_SLOPE</code>), and <code>det_a_intra</code>'s <code>slope</code>
selection relies on it. Against the joined line it mostly does not work: the flag
is set on the reconstructed tan, and the fit pushes near-normal tracks away from
zero, so {SF.frac_true_near_flagged_reliable.min() * 100:.0f}–{SF.frac_true_near_flagged_reliable.max() * 100:.0f} %
of truly near-normal gated tracks are called reliable, while most tracks it does
flag are really at |tan| ≈ 0.1. In this cosmic sample near-normal tracks are only
~2 % of the reliable ones; capsule tracks reach near normal more often, so the
contamination there is larger.</p>
{sf_tab}

<h2>Beam closure</h2>
{bc_tab}

<h2>Drift-time span of through-goers (indicative)</h2>
<p>Every through-goer crosses the full 30 mm gap, so its drift-time extent is a
drift-velocity handle independent of any angle. Only indicative: the end is the
last 60 ns bin above 5 % of the profile peak, which diffusion and shaping push
late, and 13–21 % rail at 1080 ns. A's span is near the 42.6 µm/ns prior and C's
is longer, which points the same way as the single-k summary (k<sub>C</sub> &gt;
k<sub>A</sub>) but cannot by itself account for A's beam k of 1.22.</p>
{ds_tab}

<h2>Figures</h2>
{fig_html}

<h2>What this does not rule out</h2>
<ul>
<li><b>Why beam and cosmic differ.</b> Not chamber position in A (at fixed angle
the cosmic ratio varies 3–7 % across the lever range, with no trend). In C, outward-going cosmic
tracks read ~9 % lower than inward ones, and every beam track is outward, which
covers part of C's gap. Charge moves the ratio only ~5 % from the lowest to the highest charge quartile, in both samples.
Left: the particle (beam tracks are low-energy Compton electrons, which scatter
and ionise unlike MIP muons, and have 2–3× the cosmic angle scatter); and the
beam truth itself (lever/234.6 mm assumes a line source on the axis at the
pinwheel foot). Neither is tested here.</li>
<li><b>Run condition.</b> run_149 only. The 128–147 k shift has not been
re-measured on cosmics (run_133, run_134).</li>
<li><b>B and D:</b> untested; B–D through-goers need a horizontal cosmic.</li>
<li><b>Two-particle coincidences</b> in the primary sample: one gated track per
arm and lines within 60 mm; the sep &lt; 20 and no-cut variants agree, so they
do not drive the trend.</li>
<li><b>Mesh-plane positions are assumed unbiased.</b> An angle-dependent position
bias of 1 mm per unit tan would move j by 0.4 %.</li>
<li><b>The 170° fraction</b> is quoted on angle scales this page shows to be
wrong, so it is not yet the basis for a cut.</li>
</ul>

<p class="muted">Pooled slope ratios as <code>pool_tracking.py</code> writes them
(k from run_150, sep &lt; 20 mm):
{', '.join(f"{r.arm}{r.axis} {r.median_ratio:.3f}" for r in sr.itertuples())}.
Generated by <code>ntof_cosmics/make_pooled_report.py</code> from
<code>angle_response.py</code> and <code>pool_tracking.py</code> outputs.</p>
</main></body></html>
"""
    (P / 'report.html').write_text(out)
    print(f'wrote {P / "report.html"}')


if __name__ == '__main__':
    main()
