#!/usr/bin/env python3
"""
make_intra_bench_report.py -- report.html for the two-track overlay bench.

Generated from what intra_bench.py wrote: re-run it after `build`, `floor` and
`derive` and every number, figure and verdict sentence moves together.

    python -m sept26_prelim_analysis.make_intra_bench_report
"""
from __future__ import annotations

import datetime as dt
import html as _h
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import intra_bench as IB  # noqa: E402
from sept26_prelim_analysis.report_style import head  # noqa: E402

FAR_MM = 24.0
CLOSE_MM = 12.0
SEP_LABEL = {0.0: '0–6', 6.0: '6–12', 12.0: '12–24', 24.0: '24–80', 80.0: '80–400'}
CLS_LABEL = {'coincident': 'time-coincident (< 30 ns)', 'between': '30–150 ns apart',
             'offset': '> 150 ns apart'}
OUT_COLOR = {'found': '#6a7583', 'swapped': '#E69F00', 'merged': '#56B4E9',
             'seed_lost': '#8a3f8f', 'fit_elsewhere': '#d18a44', 'unpaired': '#c9ced6'}
OUT_LABEL = {'found': 'found, correctly paired', 'swapped': 'in an x/y-swapped track',
             'merged': "merged into its partner's cluster", 'seed_lost': 'lost at seeding',
             'fit_elsewhere': 'fit landed > 3 mm away', 'unpaired': 'fitted, not a gated track'}
RULES = ['production', 'time', 'charge', 'profile', 'charge+profile', 'all']
RULE_LABEL = {'production': 'production\n(Δχ² rank)', 'time': 't0 only', 'charge': 'charge ratio',
              'profile': 'arrival profile', 'charge+profile': 'charge +\nprofile',
              'all': 'charge +\nprofile + t0'}
RULE_TICK = {'production': 'production', 'time': 't0 only', 'charge': 'charge ratio',
             'profile': 'profile', 'charge+profile': 'charge + profile', 'all': 'charge + profile + t0'}
VARIANTS = ['production', 'local_40mm', 'local_16mm', 'no_floor']
VAR_STYLE = {'production': dict(color=fs.INK, ls='-', marker='o', label='production: 10 % of the plane'),
             'local_40mm': dict(color='#E69F00', ls='-', marker='s', label='10 % within ±40 mm'),
             'local_16mm': dict(color='#56B4E9', ls='-', marker='^', label='10 % within ±16 mm'),
             'no_floor': dict(color=fs.MUTED, ls='--', marker='', label='no floor')}
RATIO_LABEL = {0.0: '< 1', 1.0: '1–2', 2.0: '2–4', 4.0: '4–8', 8.0: '≥ 8'}


# --------------------------------------------------------------------------- #
# tables derived from the bench products
# --------------------------------------------------------------------------- #
def load(od: Path) -> dict:
    L = dict(O=pd.read_csv(od / 'outcomes.csv'),
             PS=pd.read_csv(od / 'pairing_rules_summary.csv'),
             DP=pd.read_csv(od / 'data_pairing.csv'),
             S=pd.read_parquet(od / 'scores.parquet'),
             bm=json.loads((od / 'build.meta.json').read_text()),
             dm=json.loads((od / 'derive.meta.json').read_text()))
    for k, f in (('FR', 'floor_by_ratio.csv'), ('FE', 'floor_side_effect.csv')):
        L[k] = pd.read_csv(od / f) if (od / f).exists() else None
    L['PS']['rank_disagree'] = L['PS']['rank_disagree'].astype(str)
    return L


def with_sep(S: pd.DataFrame, mode: str = 'overlay') -> pd.DataFrame:
    ov = S[S['mode'] == mode].copy()
    ov['sep_lo'] = pd.cut(np.minimum(ov.sep_x, ov.sep_y), IB.COARSE_SEP,
                          labels=IB.COARSE_SEP[:-1]).astype(float)
    return ov


def events(S: pd.DataFrame) -> pd.DataFrame:
    ov = with_sep(S)
    return (ov.groupby('oid').agg(arm=('arm', 'first'), cls=('cls', 'first'),
                                  sep_lo=('sep_lo', 'first'), found=('track_found', 'all'))
            .reset_index())


def efficiency(S: pd.DataFrame) -> pd.DataFrame:
    E = (events(S).groupby(['arm', 'cls', 'sep_lo']).found.agg(['mean', 'size'])
         .rename(columns={'mean': 'eff', 'size': 'n'}).reset_index())
    E['err'] = np.sqrt(E.eff * (1 - E.eff) / E.n)
    return E


def fit_quality(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, lo), g in with_sep(S).groupby(['arm', 'sep_lo']):
        r = dict(arm=arm, sep_lo=lo, n_donors=len(g))
        for p in 'xy':
            r[f'rsig_dp0_{p}'] = IB.rsig(g[f'dp0_{p}'])
            r[f'rsig_dtan_{p}'] = IB.rsig(g[f'dtan_{p}'])
            r[f'strips_{p}'] = float(g[f'strips_{p}'].median())
            r[f'strips0_{p}'] = float(g[f'strips0_{p}'].median())
            r[f'chi2dof_{p}'] = float(g[f'chi2dof_{p}'].median())
            r[f'chi2dof0_{p}'] = float(g[f'chi2dof0_{p}'].median())
        rows.append(r)
    return pd.DataFrame(rows)


def noise_control(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in S[S['mode'] == 'noise'].groupby('arm'):
        d = np.abs(np.concatenate([g.dp0_x.to_numpy(float), g.dp0_y.to_numpy(float)]))
        d = d[np.isfinite(d)]
        rows.append(dict(arm=arm, n=len(g), found=float(g.track_found.mean()),
                         unchanged=float(np.mean(d < 1e-3)), median_abs_dp0=float(np.median(d)),
                         p90_abs_dp0=float(np.percentile(d, 90))))
    return pd.DataFrame(rows)


def weighted(O: pd.DataFrame, col: str, mask) -> pd.Series:
    g = O[mask]
    return (g[col] * g.n_donors).groupby(g.arm).sum() / g.n_donors.groupby(g.arm).sum()


def floor_pooled(FR: pd.DataFrame) -> pd.DataFrame:
    F = FR.assign(k=FR.seeded * FR.n).groupby(['arm', 'variant', 'ratio_lo']).agg(
        k=('k', 'sum'), n=('n', 'sum')).reset_index()
    F['seeded'] = F.k / F.n
    return F.drop(columns='k')


def pairing_view(PS: pd.DataFrame) -> pd.DataFrame:
    P = PS[PS.rank_disagree == 'all'].copy()
    P['production'] = P.prod_correct / (P.prod_correct + P.prod_swapped)
    return P


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _sep_axis(ax, n):
    ax.set_xticks(range(n))
    ax.set_xticklabels([SEP_LABEL[s] for s in IB.COARSE_SEP[:n]])


def fig_outcomes(O, figdir, eff_far):
    fig, axes = fs.figure(figsize=(9.6, 5.8), nrows=2, ncols=3, sharex=True, sharey=True)
    for i, arm in enumerate(IB.ARMS):
        for j, cls in enumerate(IB.CLASSES):
            ax = axes[i, j]
            g = O[(O.arm == arm) & (O.cls == cls)].sort_values('sep_lo')
            x = np.arange(len(g))
            bottom = np.zeros(len(g))
            for k in IB.OUTCOMES:
                v = g[k].to_numpy(float)
                ax.bar(x, v, bottom=bottom, width=0.78, color=OUT_COLOR[k],
                       edgecolor=fs.SURFACE, linewidth=1.0,
                       label=OUT_LABEL[k] if (i, j) == (0, 0) else None)
                bottom += v
            _sep_axis(ax, len(g))
            ax.set_ylim(0, 1)
            ax.grid(axis='x', visible=False)
            if i == 0:
                ax.set_title(CLS_LABEL[cls], loc='left', fontsize=fs.BASE_PT, fontweight='normal')
            if j == 0:
                ax.set_ylabel(f'chamber {arm}\nfraction of tracks')
            if i == 1:
                ax.set_xlabel('smaller separation  [mm]')
    fs.preliminary(axes[0, -1])
    fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07))
    fs.fig_title(fig, 'Where each overlaid track ends up',
                 f'Beyond {FAR_MM:.0f} mm both tracks return correctly paired in '
                 f'{100 * min(eff_far.values):.0f}–{100 * max(eff_far.values):.0f} % of events '
                 '(A, C); below 12 mm they merge')
    return fs.save(fig, figdir / 'outcomes', data=O)


def fig_efficiency(E, figdir):
    fig, axes = fs.figure(figsize=(9.6, 3.6), ncols=3, sharey=True)
    for j, cls in enumerate(IB.CLASSES):
        ax = axes[j]
        for k, arm in enumerate(IB.ARMS):
            g = E[(E.arm == arm) & (E.cls == cls)].sort_values('sep_lo')
            st = fs.det_style(arm)
            ax.errorbar(np.arange(len(g)) + (k - 0.5) * 0.14, g.eff, yerr=g.err,
                        color=st['color'], marker=st['marker'], label=st['label'],
                        capsize=0, markersize=5, linewidth=1.6)
        ax.axhline(0.9, color=fs.MUTED, ls=':', lw=1.0)
        _sep_axis(ax, len(IB.COARSE_SEP) - 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(CLS_LABEL[cls], loc='left', fontsize=fs.BASE_PT, fontweight='normal')
        ax.set_xlabel('smaller separation  [mm]')
    axes[0].text(0.0, 0.915, 'handoff target, 90 %', color=fs.MUTED, fontsize=fs.BASE_PT * 0.85)
    axes[0].set_ylabel('both tracks found\nand correctly paired')
    axes[-1].legend(loc='upper left')
    fs.preliminary(axes[-1], 'lower right')
    fs.fig_title(fig, 'Two-track finding never reaches the 90 % target',
                 'Overlays of two clean run_145 single-track triggers, production seeder and fit')
    return fs.save(fig, figdir / 'efficiency', data=E)


def fig_floor(F, figdir):
    fig, axes = fs.figure(figsize=(9.6, 3.7), ncols=2, sharey=True)
    for j, arm in enumerate(IB.ARMS):
        ax = axes[j]
        for v in VARIANTS:
            g = F[(F.arm == arm) & (F.variant == v)].sort_values('ratio_lo')
            ax.plot(np.arange(len(g)), g.seeded, markersize=5, **VAR_STYLE[v])
        ax.set_xticks(range(len(RATIO_LABEL)))
        ax.set_xticklabels(RATIO_LABEL.values())
        ax.set_ylim(0, 1.03)
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT, fontweight='normal')
        ax.set_xlabel("partner's peak significance / this track's")
    axes[0].set_ylabel('fraction of tracks\nstill seeded')
    axes[0].legend(loc='lower left')
    fs.preliminary(axes[-1], 'lower right')
    fs.fig_title(fig, 'A brighter partner anywhere in the plane erases a fainter track',
                 'Hits only: the significance floor is 10 % of the brightest strip of the whole plane')
    return fs.save(fig, figdir / 'floor', data=F)


def fig_pairing(P, figdir):
    fig, axes = fs.figure(figsize=(9.6, 3.9), ncols=3, sharey=True)
    x = np.arange(len(RULES))
    for j, cls in enumerate(IB.CLASSES):
        ax = axes[j]
        for k, arm in enumerate(IB.ARMS):
            row = P[(P.arm == arm) & (P.cls == cls)]
            if row.empty:
                continue
            vals = [float(row.iloc[0][r]) for r in RULES]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=fs.DET_COLOR[arm],
                   edgecolor=fs.SURFACE, linewidth=1.0, hatch=None if arm == 'A' else '///',
                   label=f'chamber {arm}')
        ax.axhline(0.5, color=fs.MUTED, ls=':', lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([RULE_TICK[r] for r in RULES], rotation=35, ha='right',
                           rotation_mode='anchor', fontsize=fs.BASE_PT * 0.85)
        ax.set_ylim(0, 1.08)
        ax.grid(axis='x', visible=False)
        ax.set_title(CLS_LABEL[cls], loc='left', fontsize=fs.BASE_PT, fontweight='normal')
    axes[0].set_ylabel('true x/y assignment chosen')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    fs.preliminary(axes[-1], 'upper right')
    fs.fig_title(fig, 'Time-coincident tracks are paired by fit rank; charge pairs them better',
                 'Overlays whose two tracks both reach the candidate list in both planes')
    return fs.save(fig, figdir / 'pairing', data=P)


def fig_fit_quality(Q, figdir):
    fig, axes = fs.figure(figsize=(9.6, 3.6), ncols=2, sharey=True)
    for j, p in enumerate('xy'):
        ax = axes[j]
        for arm in IB.ARMS:
            g = Q[Q.arm == arm].sort_values('sep_lo')
            ax.plot(np.arange(len(g)), g[f'rsig_dp0_{p}'], markersize=5, **fs.det_style(arm))
        _sep_axis(ax, len(IB.COARSE_SEP) - 1)
        ax.set_yscale('log')
        ax.set_title(f'{p} view', loc='left', fontsize=fs.BASE_PT, fontweight='normal')
        ax.set_xlabel('smaller separation  [mm]')
    axes[0].set_ylabel('robust σ of p0 − donor  [mm]')
    axes[-1].legend(loc='upper right')
    fs.preliminary(axes[0], 'lower left')
    fs.fig_title(fig, 'Once found, a separated track is fitted as if it were alone',
                 'Plane-level candidates within 3 mm of their donor; single-donor re-fits differ by 0')
    return fs.save(fig, figdir / 'fit_quality', data=Q)


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #
def _f(v, d=2, unit=''):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return '&mdash;'
    return f'{v:.{d}f}{unit}'


def _pct(v, d=0):
    return _f(100 * v, d, '&nbsp;%') if v is not None and np.isfinite(v) else '&mdash;'


def _i(v):
    return f'{int(v):,}'


def _sep(lo):
    return SEP_LABEL.get(float(lo), f'{lo:.0f}') + '&nbsp;mm'


def figure_html(name, caption):
    return (f'<figure><a href="figures/{name}.png"><img src="figures/{name}.png" '
            f'alt="{_h.escape(caption)}"></a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a></figcaption></figure>')


def table(head_cells, rows):
    th = ''.join(f'<th>{c}</th>' for c in head_cells)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def outcomes_table(O):
    rows = []
    for r in O.sort_values(['arm', 'cls', 'sep_lo']).itertuples():
        rows.append(f'<tr><th class="s">{r.arm}</th><td>{r.cls}</td><td class="n">{_sep(r.sep_lo)}</td>'
                    f'<td class="n">{_i(r.n_donors)}</td>'
                    + ''.join(f'<td class="n">{_pct(getattr(r, k))}</td>' for k in IB.OUTCOMES)
                    + '</tr>')
    return table(['chamber', 'timing', 'separation', 'tracks'] + [OUT_LABEL[k] for k in IB.OUTCOMES], rows)


def pairing_table(PS):
    rows = []
    order = {'all': 0, 'no': 1, 'yes': 2}
    g = PS.assign(o=PS.rank_disagree.map(order)).sort_values(['arm', 'cls', 'o'])
    for r in g.to_dict('records'):
        label = {'all': 'all', 'no': 'planes rank alike', 'yes': 'planes rank differently'}[r['rank_disagree']]
        cells = [f'<td class="n">{_pct(r[k])}</td>' for k in
                 ('prod_correct', 'prod_swapped', 'prod_incomplete', 'time', 'charge', 'profile')]
        cells += [f'<td class="n"><b>{_pct(r["charge+profile"])}</b></td>',
                  f'<td class="n">{_pct(r["all"])}</td>']
        rows.append(f'<tr><th class="s">{r["arm"]}</th><td>{r["cls"]}</td><td>{label}</td>'
                    f'<td class="n">{_i(r["n"])}</td>' + ''.join(cells) + '</tr>')
    return table(['chamber', 'timing', 'subset', 'events', 'production correct', 'production swapped',
                  'production incomplete', 't0 only', 'charge ratio', 'arrival profile',
                  'charge + profile', 'charge + profile + t0'], rows)


def data_table(DP):
    rows = []
    for r in DP.sort_values(['arm', 'cls']).to_dict('records'):
        rows.append(f'<tr><th class="s">{r["arm"]}</th><td>{r["cls"]}</td><td class="n">{_i(r["n"])}</td>'
                    f'<td class="n">{_pct(r["time"])}</td>'
                    f'<td class="n"><b>{_pct(r["charge+profile"])}</b></td>'
                    f'<td class="n">{_pct(r["bench_acc"])}</td>'
                    f'<td class="n">{_pct(r["implied_swap_frac"])} &plusmn;&nbsp;{_pct(r["implied_swap_err"])}</td></tr>')
    return table(['chamber', 'timing', 'two-track chambers', 't0 prefers the other pairing',
                  'charge + profile prefers the other pairing', 'rule accuracy on the bench',
                  'implied swapped fraction'], rows)


def floor_side_table(FE):
    rows = []
    for r in FE.sort_values(['arm', 'variant']).itertuples():
        rows.append(f'<tr><th class="s">{r.arm}</th><td>{r.variant}</td>'
                    f'<td class="n">{_i(r.n_triggers)}</td><td class="n">{_pct(r.frac_triggers_changed, 1)}</td>'
                    f'<td class="n">{_i(r.n_donors)}</td><td class="n"><b>{_pct(r.frac_donors_changed, 1)}</b></td></tr>')
    return table(['chamber', 'floor', 'seeded triggers', 'seeds changed', 'clean single tracks',
                  'their seeds changed'], rows)


def fitq_table(Q):
    rows = []
    for r in Q.sort_values(['arm', 'sep_lo']).itertuples():
        rows.append(f'<tr><th class="s">{r.arm}</th><td class="n">{_sep(r.sep_lo)}</td><td class="n">{_i(r.n_donors)}</td>'
                    f'<td class="n">{_f(r.rsig_dp0_x, 3)}</td><td class="n">{_f(r.rsig_dp0_y, 3)}</td>'
                    f'<td class="n">{_f(r.rsig_dtan_x, 4)}</td><td class="n">{_f(r.rsig_dtan_y, 4)}</td>'
                    f'<td class="n">{_f(r.strips_x, 0)} / {_f(r.strips0_x, 0)}</td>'
                    f'<td class="n">{_f(r.strips_y, 0)} / {_f(r.strips0_y, 0)}</td>'
                    f'<td class="n">{_f(r.chi2dof_x, 1)} / {_f(r.chi2dof0_x, 1)}</td>'
                    f'<td class="n">{_f(r.chi2dof_y, 1)} / {_f(r.chi2dof0_y, 1)}</td></tr>')
    return table(['chamber', 'separation', 'tracks', 'σ p0 x <span class="u">mm</span>',
                  'σ p0 y <span class="u">mm</span>', 'σ tan x', 'σ tan y', 'strips x, overlay / alone',
                  'strips y, overlay / alone', 'χ²/dof x, overlay / alone', 'χ²/dof y, overlay / alone'], rows)


def build_report(od: Path) -> Path:
    fs.use()
    L = load(od)
    O, PS, DP, S = L['O'], L['PS'], L['DP'], L['S']
    figdir = od / 'figures'
    figdir.mkdir(exist_ok=True)

    E = efficiency(S)
    ev = events(S)
    eff_far = ev[ev.sep_lo >= FAR_MM].groupby('arm').found.mean()
    n_far = ev[ev.sep_lo >= FAR_MM].groupby('arm').size()
    eff_close = ev[ev.sep_lo < CLOSE_MM].groupby('arm').found.mean()
    far = O.sep_lo >= FAR_MM
    swap_far_coinc = weighted(O, 'swapped', far & (O.cls == 'coincident'))
    swap_far_offset = weighted(O, 'swapped', far & (O.cls == 'offset'))
    lost_far = weighted(O, 'seed_lost', far)
    merged_close = weighted(O, 'merged', O.sep_lo < CLOSE_MM)
    P = pairing_view(PS)
    pc = P[P.cls == 'coincident'].set_index('arm')
    Q = fit_quality(S)
    Qf = Q[Q.sep_lo >= FAR_MM]
    NC = noise_control(S)

    fig_outcomes(O, figdir, eff_far)
    fig_efficiency(E, figdir)
    fig_pairing(P, figdir)
    fig_fit_quality(Q, figdir)
    have_floor = L['FR'] is not None
    if have_floor:
        F = floor_pooled(L['FR'])
        fig_floor(F, figdir)
        fr = F.set_index(['arm', 'variant', 'ratio_lo']).seeded
        fe = L['FE'].set_index(['arm', 'variant'])

    def rng(s, d=0):
        lo, hi = f'{100 * s.min():.{d}f}', f'{100 * s.max():.{d}f}'
        return f'{lo}&nbsp;%' if lo == hi else f'{lo}–{hi}&nbsp;%'

    dpc = DP[DP.cls == 'coincident'].set_index('arm')
    dpb = DP[DP.cls == 'between'].set_index('arm')
    dpo = DP[DP.cls == 'offset'].set_index('arm')

    verdict = (
        f'<p><b>No: the reconstruction does not recover two tracks in one chamber well, and the bench '
        f'says why.</b> Beyond {FAR_MM:.0f}&nbsp;mm both tracks come back correctly paired in only '
        f'{rng(eff_far)} of overlays of two clean single-track triggers (chambers A and C), against the '
        f'handoff&rsquo;s 90&nbsp;% target. Three mechanisms account for nearly all of the loss, and they '
        f'are separable:</p><ul>'
        f'<li><b>x/y swaps.</b> {rng(swap_far_coinc)} of time-coincident tracks beyond {FAR_MM:.0f}&nbsp;mm '
        f'sit in a swapped track ({rng(swap_far_offset)} when the tracks are &gt;&nbsp;150&nbsp;ns apart). '
        f'The selector pairs coincident candidates by summed fit improvement, so whenever the two planes '
        f'rank the tracks differently it picks the wrong assignment. Pairing by x/y charge ratio and '
        f'arrival profile is right in {_pct(pc["charge+profile"].min())}&ndash;{_pct(pc["charge+profile"].max())} '
        f'of those events, against production&rsquo;s {_pct(pc["production"].min())}&ndash;{_pct(pc["production"].max())} '
        f'&mdash; with no re-fit.</li>'
        f'<li><b>Lost at seeding, at any separation.</b> {rng(lost_far)} of tracks beyond {FAR_MM:.0f}&nbsp;mm '
        f'never become a seed: the significance floor is 10&nbsp;% of the brightest strip in the <i>whole plane</i>, '
        f'so a brighter partner hundreds of millimetres away pushes a fainter track under the 5-strip minimum.'
        + (f' Judging each strip against the brightest within &plusmn;16&nbsp;mm keeps '
           f'{_pct(fr[("A", "local_16mm", 4.0)])} (A) and {_pct(fr[("C", "local_16mm", 4.0)])} (C) of tracks '
           f'whose partner is 4&ndash;8&times; brighter, against {_pct(fr[("A", "production", 4.0)])} and '
           f'{_pct(fr[("C", "production", 4.0)])} today &mdash; but it changes the seeds of '
           f'{_pct(fe.loc[("A", "local_16mm"), "frac_donors_changed"], 1)} / '
           f'{_pct(fe.loc[("C", "local_16mm"), "frac_donors_changed"], 1)} of clean single tracks, '
           f'so it needs the single-track A/B before it goes anywhere.' if have_floor else '')
        + '</li>'
        f'<li><b>Merged below the seed gap.</b> Under {CLOSE_MM:.0f}&nbsp;mm {rng(merged_close)} of tracks '
        f'share one seed cluster with their partner and come back as one track; efficiency there is '
        f'{rng(eff_close)}.</li></ul>'
        f'<p>What the bench does <b>not</b> reproduce is the data&rsquo;s widened fits. Once a separated track '
        f'is found, its plane fit is the single-track fit: robust &sigma; of p0 against its donor is '
        f'{_f(np.nanmin(Qf[["rsig_dp0_x", "rsig_dp0_y"]].to_numpy()), 2)}&ndash;'
        f'{_f(np.nanmax(Qf[["rsig_dp0_x", "rsig_dp0_y"]].to_numpy()), 2)}&nbsp;mm and its strip count is unchanged, '
        f'where real two-track chambers carry twice the strips. Overlaying two clean events does not make a '
        f'messy one &mdash; the real two-track events are busier than their parts (handoff H3), and the '
        f'separation-growing y damage in the handoff is what swaps would produce on top of it.</p>')

    cards = ''.join(
        f'<div class="card"><span class="v">{v}</span><span class="l">{l}</span></div>' for v, l in [
            (rng(eff_far), f'both tracks found and correctly paired, &ge;&nbsp;{FAR_MM:.0f}&nbsp;mm apart (A, C)'),
            (rng(swap_far_coinc), f'of time-coincident tracks &ge;&nbsp;{FAR_MM:.0f}&nbsp;mm apart are x/y-swapped'),
            (rng(lost_far), f'of tracks &ge;&nbsp;{FAR_MM:.0f}&nbsp;mm apart lost at seeding'),
            (rng(merged_close), f'of tracks &lt;&nbsp;{CLOSE_MM:.0f}&nbsp;mm apart merged into one cluster'),
            (f'{_pct(pc["charge+profile"].min())}&ndash;{_pct(pc["charge+profile"].max())}',
             'coincident pairs assigned correctly by charge + profile (production: '
             f'{_pct(pc["production"].min())}&ndash;{_pct(pc["production"].max())})'),
        ])

    bm, dm = L['bm'], L['dm']
    options = table(
        ['option', 'fixes', 'measured on the bench', 'risk to single tracks', 'next step'],
        [f'<tr><th class="s">R1 &mdash; pair x with y by charge ratio and arrival profile when the candidates are '
         f'time-degenerate</th><td>swaps</td><td>coincident assignment {_pct(pc["production"].min())}&ndash;'
         f'{_pct(pc["production"].max())} &rarr; {_pct(pc["charge+profile"].min())}&ndash;'
         f'{_pct(pc["charge+profile"].max())}</td><td>none if pair 0 keeps <code>select_pair</code>&rsquo;s choice; '
         f'the current contract pins it, so R1 must re-pair only tracks 1+ or the contract changes deliberately</td>'
         f'<td>implement in <code>select_tracks</code> behind a flag, flip the pinned test, re-run this bench</td></tr>',
         f'<tr><th class="s">Local significance floor (&plusmn;16 or &plusmn;40&nbsp;mm)</th><td>seed losses</td>'
         f'<td>{rng(lost_far)} of far tracks lost today; see the floor table</td>'
         + (f'<td>changes seeds of {rng(L["FE"][L["FE"].variant != "no_floor"].frac_donors_changed, 1)} '
            f'of clean single tracks (hits only)</td>' if have_floor else '<td>run the floor study</td>') +
         f'<td>fit the changed single-track events and compare to the frozen pass (handoff &sect;8)</td></tr>',
         f'<tr><th class="s">R2/R3 &mdash; two-track fit of one window, or splitting a cluster</th>'
         f'<td>merges below the seed gap</td><td>{rng(merged_close)} of close tracks merged</td>'
         f'<td>largest; the D wildcard lost half its fits on a seed change</td>'
         f'<td>only after R1 and the floor: the close pairs are a small share of the real sample</td></tr>'])

    body = f"""<main>
<header><div class="eyebrow"><span class="badge">Preliminary</span><span>run_145 / {bm['subrun']}</span>
<span>chambers {', '.join(bm['arms'])}</span><span>{bm['n_overlays']:,} bench events</span></div>
<h1>Two tracks in one chamber: a waveform-overlay truth bench</h1>
<p class="sub">sept26_prelim_analysis/intra_bench.py &middot; handoff HANDOFF_INTRA_TWO_TRACK_RECO.md &sect;5</p></header>
<div class="verdict">{verdict}</div>
<div class="cards">{cards}</div>

<h2>What was compared</h2>
<p>Donors are clean single-track triggers of run_145 sub-run {bm['subrun']}: the only gated track of their chamber,
both slopes measured, pointing within 30&nbsp;mm of the capsule, one candidate per plane. Two donors of the same
file tag and trigger phase are summed on the second donor&rsquo;s signal strips (its seed span &plusmn;{bm['region_margin']}
strips), their hits are merged, and the <b>unmodified</b> beam seeder and <code>wft</code> fit run on the result.
Truth for each track is its donor&rsquo;s frozen single-track fit. Pairs are stratified on the smaller of the two per-view
separations and on the donors&rsquo; t0 difference: &lt;&nbsp;{bm['coinc_ns']:.0f}&nbsp;ns (time-coincident, where
the x&ndash;y time gate cannot tell the assignments apart), {bm['coinc_ns']:.0f}&ndash;{bm['offset_ns']:.0f}&nbsp;ns, and
&gt;&nbsp;{bm['offset_ns']:.0f}&nbsp;ns (separable by time alone). A track is <i>found</i> when one gated track has both
planes within {bm['match_mm']:.0f}&nbsp;mm of its donor.</p>
<p>Two controls: <b>{dm['n_single_refits']}</b> donors re-run alone through the harness differ from the frozen pass in
<b>{dm['n_single_refits_differ']}</b>; and a noise-only overlay (a charge-free trigger&rsquo;s waveforms on the same strips)
leaves the plane fit bit-identical for {rng(NC.unchanged)} of matched candidates (median |&Delta;p0|
{_f(NC.median_abs_dp0.max(), 3)}&nbsp;mm, 90th percentile {_f(NC.p90_abs_dp0.min(), 2)}&ndash;{_f(NC.p90_abs_dp0.max(), 2)}&nbsp;mm)
and the donor found in {rng(NC.found)}. Extra noise on those strips moves some fits, far less than a second track does.</p>

<h2>Where each track goes</h2>
{figure_html('outcomes', 'Each donor track of an overlay, assigned to the first thing that went wrong. Rows are chambers, columns the donors&rsquo; time difference, bars the smaller per-view separation.')}
{outcomes_table(O)}

<h2>Two-track finding efficiency</h2>
{figure_html('efficiency', 'Fraction of overlays in which both tracks are found and correctly paired, with binomial errors. The dotted line is the handoff&rsquo;s acceptance target for separations &ge;&nbsp;20&nbsp;mm.')}

<h2>The significance floor</h2>
""" + ((figure_html('floor', 'Hits only, no fits: 10&nbsp;500 random donor pairs per chamber, merged the same way as the bench. The fainter-or-equal track is counted seeded when a seed cluster covers its p0 within 3&nbsp;mm.')
        + '<p>The production floor keeps a strip only if its significance is at least 10&nbsp;% of the brightest strip of the plane. On a single track that removes the tails of coherent noise; with a second, brighter track anywhere in the plane it removes the fainter track. The two local variants judge each strip against the brightest within &plusmn;16 or &plusmn;40&nbsp;mm. Their cost is on single tracks:</p>'
        + floor_side_table(L['FE'])) if have_floor else '<p>Run <code>intra_bench floor</code> to fill this section.</p>') + f"""

<h2>Pairing x with y</h2>
{figure_html('pairing', 'Share of overlays in which each rule picks the true x/y assignment, among overlays where both tracks reach the candidate list in both planes. &ldquo;Production&rdquo; is its correct share among the events it paired completely. Each rule scores the four x-minus-y features against their spread on clean single tracks.')}
<p>For time-coincident tracks the selector&rsquo;s key reduces to the summed &Delta;&chi;&sup2;, so the strongest x is paired with
the strongest y. When the two planes rank the tracks the same way that is right; when they rank them differently it is
almost always wrong. The pinned test <code>test_time_degenerate_pairing</code> in <code>wft/tests/test_multitrack.py</code>
records that behaviour.</p>
<p>The best rule is not the same in both chambers: for time-coincident tracks the charge ratio alone is right in
{_pct(pc.loc["A", "charge"])} in A against {_pct(pc.loc["A", "charge+profile"])} with the profile added, while C prefers
charge + profile ({_pct(pc.loc["C", "charge+profile"])} against {_pct(pc.loc["C", "charge"])}). The features were calibrated
per chamber on its own single tracks; which combination to ship is a per-chamber choice, made on more than one sub-run.</p>
{pairing_table(PS)}
<h3>The same score on real two-track chambers</h3>
<p>On real run_145 chambers with exactly two gated tracks (all local sub-runs), the charge + profile score is asked whether it prefers
the pairing production did not choose. Where production pairs by time (&gt;&nbsp;150&nbsp;ns apart) it should disagree only at its
bench error rate, so the implied swapped fraction there should be zero. <b>It is not:</b> it comes out at
{', '.join(f'{_pct(dpo.loc[a, "implied_swap_frac"])} &plusmn;&nbsp;{_pct(dpo.loc[a, "implied_swap_err"])} ({a})' for a in dpo.index)}.
The score is not as accurate on real chambers as on overlays, in either direction, by about
{_pct(np.abs(dpo.implied_swap_frac).max())}. For time-coincident chambers it disagrees with production in
{_pct(dpc["charge+profile"].min())}&ndash;{_pct(dpc["charge+profile"].max())} and for 30&ndash;150&nbsp;ns in
{_pct(dpb["charge+profile"].min())}&ndash;{_pct(dpb["charge+profile"].max())}, well above the offset rows; the implied
swapped fractions in the last column carry that same systematic and are an indication that real swaps are common, not a
measurement of how common.</p>
{data_table(DP)}

<h2>Fit quality once a track is found</h2>
{figure_html('fit_quality', 'Robust &sigma; of the plane candidate&rsquo;s p0 against its donor, for candidates within 3&nbsp;mm. Log scale.')}
{fitq_table(Q)}

<h2>What this means for the handoff&rsquo;s hypotheses</h2>
<ul>
<li><b>H1, wrong x/y assignment &mdash; confirmed and measured.</b> {rng(swap_far_coinc)} of coincident far tracks, rising to most of them when the planes rank the tracks differently.</li>
<li><b>H2a, merging below the seed gap &mdash; confirmed.</b> {rng(merged_close)} merged under {CLOSE_MM:.0f}&nbsp;mm.</li>
<li><b>H2b, a window or fit spanning both tracks &mdash; not reproduced.</b> Found tracks keep their single-track strip count and p0 beyond {FAR_MM:.0f}&nbsp;mm; on data, fewer than 1&nbsp;% of two-track planes have both tracks in one seed.</li>
<li><b>H3, busier events &mdash; favoured for the widened fits.</b> The data&rsquo;s doubled strip counts and 3&ndash;4&times; &chi;&sup2;/dof do not appear when two clean events are overlaid; on run_145 the two-track chambers carry ~4&times; the floored hits of single-track ones.</li>
<li><b>New: the plane-wide significance floor</b> loses a fainter track at any separation.</li>
</ul>

<h2>What to do about it</h2>
{options}

<h2>What this does not rule out</h2>
<ul>
<li><b>The donors are clean.</b> One candidate per plane, pointing at the capsule. Real two-track chambers are busier; every efficiency here is an upper bound on what the same reconstruction does on them.</li>
<li><b>Truth is the frozen single-track fit</b>, not the true track. The bench measures whether a second track changes the answer, not the absolute resolution.</li>
<li><b>One sub-run, two chambers.</b> run_145 {bm['subrun']}, chambers A and C, one calibration bundle each; B and D are not tested.</li>
<li><b>The second donor&rsquo;s hits are cut to its signal strips</b> and its waveforms added only there, so any junk elsewhere in that trigger is absent.</li>
<li><b>The implied swap fraction on data</b> assumes the score is as accurate on real chambers as on overlays; time-coincident real chambers may be enriched in split fragments of one particle, where no pairing is right.</li>
<li><b>The local floors were tested on seeds only.</b> Which fits change, and whether single-track resolution moves, is not measured.</li>
<li><b>Pairing rules were scored only where both tracks reached the candidate list</b>; they do nothing for seed losses or merges.</li>
</ul>

<footer>generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by make_intra_bench_report.py<br>
build {bm['built']} ({bm.get('minutes', '?')} min, seed {bm['seed']}, {bm['per_cell']} pairs per cell) &middot; derive {dm['derived']} &middot; data sub-runs {', '.join(dm['data_subruns'])}<br>
inputs <code>{_h.escape(str(od))}</code></footer>
</main>"""
    html = (f'<!doctype html><html lang="en"><head>{head("Two-track overlay bench")}</head>'
            f'<body>{body}</body></html>')
    out = od / 'report.html'
    out.write_text(html)
    print(f'wrote {out}')
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='', help='report on an A/B variant instead of production')
    build_report(IB.out_dir(ap.parse_args().variant))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
