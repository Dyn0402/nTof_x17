#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pair_timing_report.py -- figures and ``report.html`` for `pair_timing.py`:
why the two arms of an inter-chamber pair are neither prompt nor flat.

Generated, never hand-written.

    python -m sept26_prelim_analysis.make_pair_timing_report            # campaign
    python -m sept26_prelim_analysis.make_pair_timing_report --stem run_145
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as _h
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import pair_timing as PT  # noqa: E402
from sept26_prelim_analysis.make_funnel_report import CSS, FONT_LINK, fmt  # noqa: E402

ARMS = PT.ARMS
NAMES = [f'{a}{b}' for i, a in enumerate(ARMS) for b in ARMS[i + 1:]]
CLS_ORDER = ['both legs prompt', 'a leg is a plastic after-pulse',
             'a leg is an accidental single', 'no wall reference in event']
CLS_COLOR = {'both legs prompt': fs.ACCENT,
             'a leg is a plastic after-pulse': fs.COPPER,
             'a leg is an accidental single': '#56B4E9',
             'no wall reference in event': '#b9bfc8'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


# --------------------------------------------------------------------------- #
# derived tables
# --------------------------------------------------------------------------- #
def load(stem: str) -> dict:
    od = paths.out('pair_timing')
    D = dict(
        align=pd.read_parquet(od / f'align_hist_{stem}.parquet'),
        pairs=pd.read_parquet(od / f'pairs_{stem}.parquet'),
        legs=pd.read_parquet(od / f'legs_{stem}.parquet'),
        other=pd.read_parquet(od / f'other_arm_wall_{stem}.parquet'),
        meta=json.load(open(od / f'pair_timing_{stem}.meta.json')))
    z = np.load(od / f'baseline_{stem}.npz')
    D.update(base=z['base'], base_n=z['base_n'], pss=z['pss'], edges=z['edges'])
    return D


def align_fit(A: pd.DataFrame) -> pd.DataFrame:
    """Gaussian + constant on +-80 ns, per arm pair."""
    from scipy.optimize import curve_fit
    rows = []
    for n in NAMES:
        s = A[A.dt_ns.abs() < 80]
        x, y = s.dt_ns.to_numpy(), s[n].to_numpy()
        ped = float(A.loc[A.dt_ns.abs() > 200, n].mean())
        f = lambda t, a, m, sg, c: a * np.exp(-0.5 * ((t - m) / sg) ** 2) + c  # noqa
        p, cov = curve_fit(f, x, y, p0=[y.max() - ped, 0, 5, ped])
        e = np.sqrt(np.diag(cov))
        n_peak = p[0] * abs(p[2]) * np.sqrt(2 * np.pi) / 2.0     # 2 ns bins
        rows.append(dict(arm_pair=f'{n[0]}−{n[1]}', mean_ns=p[1],
                         mean_err=e[1], sigma_ns=abs(p[2]), sigma_err=e[2],
                         n_peak=n_peak, pedestal_per_2ns=ped,
                         peak_over_ped=p[0] / ped))
    return pd.DataFrame(rows)


def pair_classes(L: pd.DataFrame) -> pd.DataFrame:
    """The published figure's pairs (both arms loose-tagged), one class each."""
    w = L.pivot_table(index='pid', columns='leg',
                      values=['dt_ns', 'cls', 'family'], aggfunc='first')
    w = w.dropna(subset=[('dt_ns', '1'), ('dt_ns', '2')])
    c1, c2 = w[('cls', '1')], w[('cls', '2')]
    anyc = lambda k: (c1 == k) | (c2 == k)                          # noqa
    out = pd.DataFrame(dict(
        delta_t=(w[('dt_ns', '1')] - w[('dt_ns', '2')]).astype(float),
        fam1=w[('family', '1')], fam2=w[('family', '2')]), index=w.index)
    out['cls'] = np.select(
        [anyc('no reference'), anyc('plastic after-pulse'),
         anyc('accidental single')],
        CLS_ORDER[3:4] + CLS_ORDER[1:3], default=CLS_ORDER[0])
    return out.reset_index()


def other_arm(D: dict) -> tuple:
    """Pairs whose reference (trigger) crossing is in one of the pair's arms:
    the OTHER arm's wall crossings vs t - t_ref, and the all-trigger baseline
    expectation for exactly that arm composition."""
    P = D['pairs']
    Q = P[P.other_arm.notna()].copy()
    idx = {a: i for i, a in enumerate(ARMS)}
    exp = np.zeros(D['edges'].size - 1)
    for (ra, oa), n in Q.groupby(['ref_arm', 'other_arm']).size().items():
        ri, oi = idx[ra], idx[oa]
        exp += n * D['base'][ri, oi] / max(D['base_n'][ri], 1)
    o = D['other'][D['other'].pid.isin(Q.pid)]
    obs = np.histogram(o.dt_other, D['edges'])[0]
    return Q, obs, exp


def accounting(D: dict, Q: pd.DataFrame, exp: np.ndarray) -> pd.DataFrame:
    P = D['pairs']
    c = 0.5 * (D['edges'][:-1] + D['edges'][1:])
    k = np.where(Q.ref_arm == Q.arm1, '2', '1')
    wp = np.where(k == '1', Q.wprompt1, Q.wprompt2)
    nw = np.where(k == '1', Q.nwall1, Q.nwall2)
    n = len(Q)
    exp_prompt = exp[np.abs(c) < PT.PROMPT_NS].sum() / n
    exp_any = 1 - np.exp(-exp.sum() / n)
    rows = [
        dict(group='trigger crossing in neither MM arm, or no wall reference',
             n=int(len(P) - n), frac=(len(P) - n) / len(P), all_trigger_expectation=np.nan),
        dict(group='other arm: prompt wall crossing (|t−t_ref| < 25 ns)',
             n=int(wp.sum()), frac=wp.sum() / len(P),
             all_trigger_expectation=exp_prompt * n / len(P)),
        dict(group='other arm: wall crossing only at an unrelated time (±1 µs)',
             n=int(((~wp.astype(bool)) & (nw > 0)).sum()),
             frac=((~wp.astype(bool)) & (nw > 0)).sum() / len(P),
             all_trigger_expectation=(exp_any - exp_prompt) * n / len(P)),
        dict(group='other arm: no wall crossing anywhere in ±1 µs',
             n=int((nw == 0).sum()), frac=(nw == 0).sum() / len(P),
             all_trigger_expectation=(1 - exp_any) * n / len(P)),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_timeline(D, fd):
    """The schematic: what each time window is, drawn to scale."""
    plt = _plt()
    P = D['pairs']
    Q = P[P.other_arm.notna()]
    t_oth = np.where(Q.ref_arm == Q.arm1, Q.t0_2, Q.t0_1)
    t_ref = np.where(Q.ref_arm == Q.arm1, Q.t0_1, Q.t0_2)
    lo, hi = np.nanpercentile(np.r_[t_oth, t_ref], [2, 98])
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    fs.strip(ax, left=False)
    rows = [
        ('Micromegas track t0 of the\npaired tracks (2–98 %)',
         lo, hi, '#9fb3c8'),
        ('n_TOF slim: scintillator hits kept', -1000, 1000, '#cfd5dd'),
        ('plastic after-pulse train\n(+20 ns … ~1 µs)', 20, 1000, fs.COPPER),
        ('per-arm tag window of the\npublished Δt (−100, +60)',
         PT.DT_WINDOW[0], PT.DT_WINDOW[1], '#56B4E9'),
        ('a true pair: both arms here\n(±25 ns)', -25, 25, fs.ACCENT),
    ]
    for i, (lab, a, b, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, b - a, left=a, height=0.56, color=col, alpha=0.85,
                edgecolor='white', linewidth=2)
        ax.text(-1520, y, lab, ha='left', va='center', fontsize=fs.BASE_PT * 0.86)
    ax.axvline(0, color=fs.INK, lw=0.9, ls=':')
    ax.plot([81, 81], [2 - 0.28, 2 + 0.28], color=fs.INK, lw=1.2)
    ax.text(90, 2.33, '81 ns echo', fontsize=fs.BASE_PT * 0.75, color=fs.INK)
    ax.text(8, -0.55, 'trigger crossing', fontsize=fs.BASE_PT * 0.8,
            color=fs.MUTED, va='center')
    ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.set_xlim(-1540, max(hi, 1000) + 60)
    ax.set_yticks([])
    ax.set_xticks(np.arange(-1000, 1601, 500))
    ax.set_xlabel('time relative to the trigger crossing  [ns]')
    ax.grid(axis='y', visible=False)
    fs.title(ax, 'The windows are not the same size',
             'an accidental second particle can be anywhere in the grey-blue band; '
             'the published Δt only ever sees the thin blue one')
    fs.save(fig, fd / 'timeline', data=pd.DataFrame(
        [dict(row=r[0].replace('\n', ' '), lo_ns=r[1], hi_ns=r[2]) for r in rows]))
    plt.close(fig)


def fig_alignment(D, F, fd):
    plt = _plt()
    A = D['align']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    for a in (a1, a2):
        fs.strip(a)
    tot = A[NAMES].sum(axis=1)
    x = A.dt_ns.to_numpy()
    xb = x.reshape(-1, 10).mean(1)
    a1.step(xb, tot.to_numpy().reshape(-1, 10).sum(1), where='mid',
            color=fs.INK, lw=1.2)
    a1.set_yscale('log')
    a1.set_xlim(-1000, 1000)
    a1.set_xlabel('t(arm 1) − t(arm 2), wall crossings  [ns]')
    a1.set_ylabel('crossing pairs / 20 ns')
    a1.set_title('every trigger, all six arm pairs', loc='left',
                 fontsize=fs.BASE_PT, fontweight='normal', color=fs.MUTED)
    cols = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    m = np.abs(x) <= 60
    for n, c in zip(NAMES, cols):
        ped = A.loc[A.dt_ns.abs() > 200, n].mean()
        y = A[n].to_numpy()[m] - ped
        a2.step(x[m], y / y.max(), where='mid', color=c, lw=1.3,
                label=f'{n[0]}−{n[1]}')
    a2.set_xlim(-60, 60)
    a2.set_xlabel('t(arm 1) − t(arm 2)  [ns]')
    a2.set_ylabel('pedestal-subtracted, peak = 1')
    a2.legend(ncol=2, fontsize=fs.BASE_PT * 0.8, loc='upper left')
    s = F.sigma_ns.median()
    a2.text(0.98, 0.97, f'median σ = {s:.1f} ns\nmeans {F.mean_ns.min():+.1f} '
            f'… {F.mean_ns.max():+.1f} ns', transform=a2.transAxes,
            ha='right', va='top', fontsize=fs.BASE_PT * 0.85)
    fs.fig_title(fig, 'The arms are timed in: a real two-arm coincidence is a '
                 f'{2.355 * s:.0f} ns FWHM spike',
                 'Wall crossings seen at both bar ends (ends averaged), no '
                 'Micromegas requirement. The flat floor is the accidental rate.')
    fs.save(fig, fd / 'arm_alignment', data=A)
    plt.close(fig)


def fig_plastic(D, fd):
    plt = _plt()
    e = D['edges']
    c = 0.5 * (e[:-1] + e[1:])
    base = D['base']
    wall = sum(base[i, i] for i in range(4))
    clean, ap = D['pss']
    rb = lambda y: y.reshape(-1, 5).sum(1)                        # noqa
    cb = c.reshape(-1, 5).mean(1)
    fig, ax = fs.figure(figsize=(9.6, 3.9))
    ax.axvspan(*PT.DT_WINDOW, color='#56B4E9', alpha=0.16, lw=0)
    ax.text(PT.DT_WINDOW[0] + 4, 0.97, 'tag window', transform=ax.get_xaxis_transform(),
            fontsize=fs.BASE_PT * 0.8, color=fs.INK, va='top')
    n = D['base_n'].sum()
    for y, lab, col in ((wall, 'wall (both ends)', fs.INK),
                        (clean, 'plastic, not flagged', fs.ACCENT),
                        (ap, 'plastic, flagged after-pulse', fs.COPPER)):
        ax.step(cb, rb(y) / n / 10, where='mid', color=col, lw=1.3, label=lab)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 0.3)
    ax.set_xlim(-300, 1000)
    ax.set_xlabel('hit time − trigger crossing, same arm  [ns]')
    ax.set_ylabel('hits per trigger per ns')
    ax.legend(loc='upper right')
    fs.title(ax, 'Behind every plastic pulse sits a train of after-pulses',
             'triggering arm only; the published tag picks one hit at random '
             'from the blue band, wall or plastic')
    fs.save(fig, fd / 'plastic_afterpulses', data=pd.DataFrame(
        dict(dt_ns=cb, wall=rb(wall), plastic_clean=rb(clean),
             plastic_afterpulse=rb(ap), n_triggers_with_ref=n)))
    plt.close(fig)


def fig_decomposed(C, fd):
    plt = _plt()
    edges = np.arange(-170, 171, 10)
    fig, ax = fs.figure(figsize=(9.6, 4.2))
    bottom = np.zeros(edges.size - 1)
    tab = {}
    for k in CLS_ORDER:
        h = np.histogram(C.delta_t[C.cls == k], edges)[0]
        tab[k] = h
        ax.bar(edges[:-1], h, width=10, bottom=bottom, align='edge',
               color=CLS_COLOR[k], edgecolor='white', linewidth=1.0,
               label=f'{k} ({(C.cls == k).sum():,})')
        bottom += h
    ax.set_xlim(-170, 170)
    ax.set_xlabel('t(arm 1) − t(arm 2), published loose tag  [ns]')
    ax.set_ylabel('pairs / 10 ns')
    ax.legend(loc='upper right', fontsize=fs.BASE_PT * 0.85)
    fs.title(ax, 'The published Δt, taken apart hit by hit',
             f'{len(C):,} inter-chamber pairs with a tag in both arms; each '
             'leg classed by what the random pick actually landed on')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, fd / 'published_decomposed',
            data=pd.DataFrame(dict(lo_ns=edges[:-1], **tab)))
    plt.close(fig)


def fig_full_range(D, Q, obs, exp, fd):
    plt = _plt()
    e = D['edges']
    c = 0.5 * (e[:-1] + e[1:])
    k = 10
    cb, ob, eb = c.reshape(-1, k).mean(1), obs.reshape(-1, k).sum(1), exp.reshape(-1, k).sum(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.6, 1]))
    for a in (a1, a2):
        fs.strip(a)
    a1.bar(cb, ob, width=2 * k, color=fs.ACCENT, alpha=0.75, lw=0,
           label='observed: other arm of an MM pair')
    a1.step(cb, eb, where='mid', color=fs.INK, lw=1.3,
            label='all triggers with the same trigger arm, scaled')
    a1.set_xlim(-1000, 1000)
    a1.set_xlabel('other-arm wall crossing − trigger crossing  [ns]')
    a1.set_ylabel('crossings / 20 ns')
    a1.legend(loc='upper left', fontsize=fs.BASE_PT * 0.82)
    m = np.abs(c) <= 100
    a2.bar(c[m], obs[m], width=2, color=fs.ACCENT, alpha=0.75, lw=0)
    a2.step(c[m], exp[m], where='mid', color=fs.INK, lw=1.3)
    a2.set_xlim(-100, 100)
    a2.set_xlabel('zoom  [ns]')
    a2.set_ylabel('crossings / 2 ns')
    far = np.abs(c) > 100
    xs_far = obs[far].sum() - exp[far].sum()
    pr = np.abs(c) < PT.PROMPT_NS
    xs_pr = obs[pr].sum() - exp[pr].sum()
    fs.fig_title(fig, 'Timed with the wall over the full ±1 µs: a prompt '
                 'spike and a raised flat floor',
                 f'{len(Q):,} pairs whose trigger crossing is in one MM arm. Excess '
                 f'over all triggers with the same trigger arm: {xs_pr:,.0f} prompt '
                 f'(|Δt| < 25 ns), {xs_far:,.0f} spread flat beyond ±100 ns.')
    fs.save(fig, fd / 'full_range', data=pd.DataFrame(
        dict(dt_ns=c, observed=obs, all_trigger_expectation=exp)))
    plt.close(fig)
    return xs_pr, xs_far


def t0_table(D) -> pd.DataFrame:
    """Per pair (trigger crossing in one MM arm): the other-arm wall class and
    the MM t0 difference, each chamber's t0 centred on its trigger-arm median."""
    P = D['pairs']
    Q = P[P.other_arm.notna()].copy()
    r1 = Q.ref_arm == Q.arm1
    Q['t0r'] = np.where(r1, Q.t0_1, Q.t0_2)
    Q['t0o'] = np.where(r1, Q.t0_2, Q.t0_1)
    wp = np.where(r1, Q.wprompt2, Q.wprompt1).astype(bool)
    nw = np.where(r1, Q.nwall2, Q.nwall1)
    Q['wall'] = np.select([wp, nw > 0], WALL_CLS[:2], WALL_CLS[2])
    off = Q.groupby('ref_arm').t0r.median()
    Q['dt0'] = (Q.t0o - Q.other_arm.map(off)) - (Q.t0r - Q.ref_arm.map(off))
    return Q


WALL_CLS = ['prompt wall crossing', 'wall crossing at an unrelated time',
            'no wall crossing in \u00b11 \u00b5s']


def fig_t0(Q, fd):
    plt = _plt()
    edges = np.arange(-600, 601, 30)
    fig, ax = fs.figure(figsize=(9.6, 3.9))
    tab = {}
    for k, col, ls in zip(WALL_CLS, (fs.ACCENT, fs.COPPER, fs.INK),
                          ('-', '-', '--')):
        y = Q.dt0[Q.wall == k]
        h = np.histogram(y, edges)[0]
        tab[k] = h
        ax.step(edges[:-1] + 15, h / max(len(y), 1), where='mid', color=col,
                lw=1.5, ls=ls,
                label=f'other arm: {k} ({len(y):,}; '
                      f'{100 * (y.abs() < 100).mean():.0f}% within \u00b1100 ns)')
    ax.set_xlabel('t0(other-arm track) \u2212 t0(trigger-arm track), '
                  'chamber offsets removed  [ns]')
    ax.set_ylabel('fraction / 30 ns')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=1, fontsize=fs.BASE_PT * 0.82)
    fs.title(ax, 'The Micromegas clock is too coarse to time the no-wall majority',
             'even with a prompt wall crossing in both arms only about half the '
             'pairs agree to 100 ns; the no-wall pairs sit between the two templates')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, fd / 'mm_t0', data=pd.DataFrame(dict(lo_ns=edges[:-1], **tab)))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def figure(name, caption):
    return (f'<figure><a href="figures/{name}.png"><img src="figures/{name}.png" '
            f'alt="{_h.escape(caption[:120])}"></a><figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            '</figcaption></figure>')


def table(df, cols, heads, fmts):
    th = ''.join(f'<th>{h}</th>' for h in heads)
    rows = []
    for _, r in df.iterrows():
        rows.append('<tr>' + ''.join(
            f'<td class="{"n" if f else ""}">{(f.format(r[c]) if f else _h.escape(str(r[c]))) if pd.notna(r[c]) else "&mdash;"}</td>'
            for c, f in zip(cols, fmts)) + '</tr>')
    return (f'<div class="scroll"><table class="t"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def build_html(D, F, C, acc, xs_pr, xs_far, stem, Q):
    P, meta = D['pairs'], D['meta']
    n_pub = len(C)
    frac = C.cls.value_counts(normalize=True)
    n_runs = len(meta['runs'])
    ap_share = frac.get(CLS_ORDER[1], 0)
    acc_share = frac.get(CLS_ORDER[2], 0)
    pr_share = frac.get(CLS_ORDER[0], 0)
    Pr = C[C.cls == CLS_ORDER[0]]
    pr_in20 = (Pr.delta_t.abs() < 20).mean() if len(Pr) else np.nan
    a = acc.set_index(acc.index)
    n_pr, n_late, n_none = acc.n.iloc[1], acc.n.iloc[2], acc.n.iloc[3]
    e_pr = acc.all_trigger_expectation.iloc[1] * len(P)
    s_med = F.sigma_ns.median()
    topo = Q.groupby('topo').wall.value_counts(normalize=True).unstack()
    opp_pr, perp_pr = topo.loc['opposing', WALL_CLS[0]], topo.loc['perpendicular', WALL_CLS[0]]
    opp_nw, perp_nw = topo.loc['opposing', WALL_CLS[2]], topo.loc['perpendicular', WALL_CLS[2]]
    oq = Q[Q.topo == 'opposing']
    b2b = oq.groupby('wall').open_deg.agg(
        med='median', f170=lambda x: (x > 170).mean())
    within = Q.groupby('wall').dt0.apply(lambda x: (x.abs() < 100).mean())
    P_nr = P[P.other_arm.isna()]
    n_noref = int(P_nr.ref_arm.isna().sum())
    n_refB = int((P_nr.ref_arm == 'B').sum())
    n_refelse = len(P_nr) - n_noref - n_refB
    return f"""<title>Two-Arm Pair Timing</title>
{FONT_LINK}
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow"><span class="badge">PRELIMINARY</span>
    <span>n_TOF EAR2 &middot; X17</span><span>{_h.escape(stem)}</span>
    <span>{n_runs} runs &middot; {fmt(meta['n_triggers'])} triggers</span>
    <span>{dt.date.today().isoformat()}</span></div>
  <h1>Why the two arms of a pair are neither prompt nor flat</h1>
  <p class="sub">{fmt(len(P))} real inter-chamber Micromegas pairs &middot;
  scintillator hits from the n_TOF slim, full &plusmn;1000 ns</p>
</header>

<p class="lede"><b>The scintillators are fine and your expectation is right.</b>
Across arms, a genuine coincidence between two wall crossings is a
<b>&sigma; &asymp; {s_med:.1f} ns</b> spike sitting within
{F.mean_ns.abs().max():.1f} ns of zero for all six arm pairs, on a flat floor.
The published two-arm &Delta;t is wide and triangular because of <b>how each
arm's time was picked</b>, not because of the detectors: one random hit, wall
<i>or</i> plastic, inside a (&minus;100,&nbsp;+60)&nbsp;ns window. Only
{100 * pr_share:.0f}% of the {fmt(n_pub)} plotted pairs have a genuinely prompt hit
on both legs (and {100 * pr_in20:.0f}% of those sit within 20 ns). In
{100 * ap_share:.0f}% a leg is a <b>plastic after-pulse</b>, which makes the
&plusmn;30&ndash;90 ns shoulders; in {100 * acc_share:.0f}% a leg is an
<b>accidental single</b>. The window caps |&Delta;t| at 160 ns, which is why the
accidentals make a triangle instead of the flat distribution you expected.
Looked at over the full range with the wall alone, the flat distribution is
there. <b>Two caveats on what the real coincidences are</b>: the prompt ones
are ten times more common in opposing chambers than perpendicular ones and
cluster near 180&deg;, the signature of a single particle crossing both; and for most pairs the second arm
has no wall crossing at all, so the scintillators cannot say whether they are
prompt.</p>

<div class="cards">
  <div class="card"><div class="v">{s_med:.1f} ns</div>
    <div class="l">&sigma; of a real arm-to-arm wall coincidence (median of 6 arm pairs)</div></div>
  <div class="card"><div class="v">{100 * pr_share:.0f}%</div>
    <div class="l">of the published &Delta;t pairs have a prompt hit on both legs</div></div>
  <div class="card"><div class="v">{100 * ap_share:.0f}%</div>
    <div class="l">have a plastic after-pulse on a leg</div></div>
  <div class="card"><div class="v">{100 * n_none / len(P):.0f}%</div>
    <div class="l">of all inter pairs: the second arm has no wall crossing
    anywhere in &plusmn;1 &micro;s</div></div>
</div>

<h2><span class="n">1</span>The picture: four windows of very different size</h2>
{figure('timeline', 'Drawn to scale. The Micromegas record a track from any '
        'particle inside roughly the grey-blue band (measured here from the '
        'track t0 of the pairs themselves). The slim keeps scintillator hits in '
        '&plusmn;1 &micro;s. The published tag keeps only hits in the thin blue '
        'band, and every large plastic pulse drags an after-pulse train into '
        'it. An uncorrelated second particle is spread over the grey-blue band; '
        'only the part of it that falls in the blue band is ever plotted, and '
        'the difference of two times confined to a 160 ns box is a triangle, '
        'not a flat line: uniform on each side, triangular in the difference.')}

<h2><span class="n">2</span>The premise checks out: the arms are timed in</h2>
<p>No Micromegas, every trigger: pair every wall crossing (seen at both bar
ends, ends averaged to remove the propagation along the bar) with every
crossing in a different arm of the same trigger.</p>
{figure('arm_alignment', 'Left: over &plusmn;1 &micro;s, a spike on an accidental '
        'floor &mdash; the two components you expected. The floor is not '
        'quite flat, and for the same reason as the published figure: both '
        'times are confined to the &plusmn;1 &micro;s slim, and a difference of '
        'two bounded times is a triangle. Most combinations include the '
        'trigger crossing itself at ~0, so here it is a gentle one. '
        'Right: each arm pair, pedestal-subtracted. The peaks agree to a few ns; '
        'the residual spread (A a few ns early, D a few ns late) is the scale of '
        'the flash-calibration residual, not a problem for a 20 ns cut.')}
{table(F, ['arm_pair', 'mean_ns', 'sigma_ns', 'n_peak', 'peak_over_ped'],
       ['arms', 'peak mean [ns]', '&sigma; [ns]', 'coincidences in peak',
        'peak / floor'], [None, '{:+.1f}', '{:.1f}', '{:,.0f}', '{:.1f}'])}
<p class="note"><b>One thing that does move between arms, and why it does not
matter for &Delta;t.</b> A single arm's <code>dt_ns</code> peak sits at
&minus;15 ns for A and +9 ns for B. That shift follows the arm that
<i>triggered</i>, not the arm that was hit: it is the DREAM trigger-path delay
(<code>a_arm</code> in <code>DREAM_NTOF_CALIBRATION.md</code>, &minus;16.8 to
+7.6 ns), and <code>dt_ns</code> is referenced to an arm-agnostic prediction.
It moves every hit of an event together, so it cancels in
t(arm&nbsp;1)&nbsp;&minus;&nbsp;t(arm&nbsp;2). It does not cancel in the per-arm
|t|&nbsp;&le;&nbsp;30&nbsp;ns cut of <code>tight_coincidence.py</code>, which
therefore sits off-centre for A- and B-triggered events.</p>

<h2><span class="n">3</span>What the published tag actually picks</h2>
{figure('plastic_afterpulses', 'Hits on the arm that triggered, relative to '
        'its own wall crossing. The wall has a clean spike and a flat floor. '
        'The plastic has the spike and then a train of after-pulses: the broad '
        'component at 30&ndash;40 ns, the 81 ns cable echo, and a tail that '
        'lasts hundreds of ns (<code>ntof_processing/pss_ringing</code>, '
        'measured on raw traces). Inside the tag window the flagged '
        'after-pulses are a large share of all plastic hits, and plastic hits '
        'outnumber wall hits &mdash; so a random pick lands on one often.')}
{figure('published_decomposed', 'The same pairs, the same kind of random pick '
        'as <code>accidental_timing.arm_tag_time(require_both=False)</code>, '
        'each leg classed against the event&rsquo;s own wall reference: '
        '<i>prompt</i> within 25 ns of it, <i>after-pulse</i> if a hit more than '
        '20&times; larger preceded it on the same channel within 1 &micro;s, '
        'otherwise <i>accidental single</i>. The spike is the prompt class; the '
        'shoulders are after-pulses; the triangle under everything is '
        'accidentals squeezed into the window.')}

<h2><span class="n">4</span>The same question, asked without the window</h2>
<p>Take the pairs whose trigger crossing is in one of the two Micromegas arms,
and list every wall crossing in the <i>other</i> arm over the full
&plusmn;1&nbsp;&micro;s. The comparison line is what an ordinary trigger shows: the
wall-crossing density in that arm, measured on all triggers with the same
trigger arm and summed over exactly these pairs. It is mostly accidental, plus
the small rate of genuine multi-arm events any trigger has; the excess above it
is what the second Micromegas track brought with it.</p>
{figure('full_range', 'This is the distribution you were expecting. A prompt '
        'spike &mdash; real coincidences &mdash; and, beyond it, a floor that is '
        'flat but raised above the accidental line: the second particle hit the '
        'wall, just not when the trigger did.')}
{table(acc, ['group', 'n', 'frac', 'all_trigger_expectation'],
       ['all inter pairs', 'pairs', 'fraction', 'same, all triggers with that trigger arm'],
       [None, '{:,.0f}', '{:.1%}', '{:.1%}'])}
<p>So the published &Delta;t is looking at a small corner of the sample.
{fmt(n_pr)} pairs have a prompt wall crossing in the other arm, against
{e_pr:.0f} on an ordinary trigger with the same trigger arm &mdash; about {fmt(round(xs_pr))} real prompt
coincidences. About {fmt(round(xs_far))} further crossings sit flat beyond
&plusmn;100 ns: the second particle arrived at an unrelated time.
{fmt(n_none)} pairs have no wall crossing in the other arm anywhere in
&plusmn;1 &micro;s; they cannot be timed by the scintillators at all, and they
never enter the published figure. (The {fmt(len(P_nr))} pairs outside this
table's timed rows: {fmt(n_noref)} have no wall crossing near the trigger at all,
{fmt(n_refB)} were triggered by chamber B and {fmt(n_refelse)} by another
chamber outside the pair.)</p>
<div class="caution"><b>Most of the prompt coincidences look like one particle,
not two.</b> A prompt wall crossing in the other
arm: <b>{100 * opp_pr:.1f}%</b> of opposing pairs against
<b>{100 * perp_pr:.1f}%</b> of perpendicular ones. No wall crossing at all:
{100 * opp_nw:.0f}% opposing, {100 * perp_nw:.0f}% perpendicular. Opposing is also where a
wide-angle pair lands, so topology alone would not decide it &mdash; but the
opening angle does. Opposing pairs with a prompt second wall have a median
opening angle of <b>{b2b.loc[WALL_CLS[0], 'med']:.0f}&deg;</b>, and
<b>{100 * b2b.loc[WALL_CLS[0], 'f170']:.0f}%</b> are above 170&deg;; the
opposing pairs whose second wall fired at an unrelated time, or not at all, sit
at {b2b.loc[WALL_CLS[1], 'med']:.0f}&deg; and {b2b.loc[WALL_CLS[2], 'med']:.0f}&deg;
with {100 * b2b.loc[WALL_CLS[1], 'f170']:.0f}% and
{100 * b2b.loc[WALL_CLS[2], 'f170']:.0f}% above 170&deg;. A straight particle
crossing the target and both chambers is prompt by construction. This is the
back-to-back population <code>tight_coincidence.py</code> flags, and it means a
prompt &Delta;t alone does not make a pair.</div>

<h2><span class="n">5</span>An independent clock: the Micromegas t0</h2>
<p>The Micromegas t0 exists for every pair, wall or no wall, so it is the
only handle on the {100 * n_none / len(P):.0f}% of pairs the scintillators
cannot time. It is not a good one.</p>
{figure('mm_t0', 'Difference of the two tracks&rsquo; t0, each chamber centred '
        'on its own trigger-arm median, split by what the other arm&rsquo;s '
        'wall saw. The prompt-wall class is the MM resolution function on '
        'a sample known to be prompt: only '
        f'{100 * within[WALL_CLS[0]]:.0f}% within &plusmn;100 ns. The '
        'unrelated-time class is the template for a second particle that is '
        f'known not to be prompt: {100 * within[WALL_CLS[1]]:.0f}%. The no-wall '
        f'majority sits between them at {100 * within[WALL_CLS[2]]:.0f}%. Read '
        'naively that would make a sizeable fraction of it prompt, but the '
        'MM acceptance is not flat in time (a track far from the trigger is '
        'truncated by the readout window), so neither template is clean '
        'enough to turn that into a number.')}

<h2><span class="n">6</span>What this does not rule out</h2>
<div class="caution"><ul>
<li><b>Prompt is not the same as a pair from the capsule.</b> A single particle
crossing both chambers, or a photon converting in one arm after a Compton in the
other, is prompt too (<code>tight_coincidence.py</code>'s back-to-back
flag).</li>
<li><b>No wall hit is not the same as no particle, or no coincidence.</b> A
low-energy electron can make a Micromegas track and stop before the wall, and a
perpendicular track may not point at a wall bar at all (not checked here:
<code>pred_sipm_bar</code> in the track table would separate &ldquo;missed the
wall&rdquo; from &ldquo;hit it and did not fire&rdquo;). These pairs are the
majority and are untimeable by the scintillators; section 5 shows the
Micromegas t0 cannot settle them either.</li>
<li><b>The after-pulse flag is re-derived</b> from the exported parquet, on
<code>amp</code> rather than <code>amp_0</code>; on run_145 it over-flags about
4 % of clean in-window plastic hits relative to the slim's stored flag. That
moves a few pairs between the after-pulse and prompt/accidental classes, not
the shape.</li>
<li><b>The random pick is re-drawn</b>, not bit-identical to the published
figure's (the sample is: {fmt(n_pub)} tagged pairs, the same count), and the
reference is &ldquo;largest wall crossing within (&minus;45,&nbsp;+35)
ns&rdquo;, not the DREAM trigger arm itself, which the exported slim does not
carry.</li>
<li>run_79 and run_81 are excluded, as in the published campaign figure.</li>
</ul></div>

<h2><span class="n">7</span>What to do with it</h2>
<p>Time each arm with the <b>wall</b> (both ends, largest crossing), not a random
wall-or-plastic hit; if a plastic is used, drop flagged after-pulses and take
the largest hit (the slim already stores <code>shadow_amp</code>; the export
should carry it). Measure &Delta;t over the full &plusmn;1&nbsp;&micro;s, where
the accidental floor is flat and can be subtracted from the sidebands directly,
instead of a window that folds it into a triangle. And centre any per-arm cut on
the trigger crossing, not on <code>dt_ns = 0</code>.</p>

<footer><p>Generated by <code>sept26_prelim_analysis/make_pair_timing_report.py</code>
from <code>pair_timing.py</code>. Figures carry their numbers as CSV.</p></footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--stem', default='campaign')
    a = ap.parse_args()
    D = load(a.stem)
    od = paths.out('pair_timing') / (a.stem if a.stem != 'campaign' else '')
    fd = od / 'figures'
    fd.mkdir(parents=True, exist_ok=True)
    F = align_fit(D['align'])
    C = pair_classes(D['legs'])
    Q, obs, exp = other_arm(D)
    acc = accounting(D, Q, exp)
    fig_timeline(D, fd)
    fig_alignment(D, F, fd)
    fig_plastic(D, fd)
    fig_decomposed(C, fd)
    xs_pr, xs_far = fig_full_range(D, Q, obs, exp, fd)
    Q0 = t0_table(D)
    fig_t0(Q0, fd)
    F.to_csv(od / 'arm_alignment_fit.csv', index=False)
    acc.to_csv(od / 'accounting.csv', index=False)
    body = build_html(D, F, C, acc, xs_pr, xs_far, a.stem, Q0)
    head, rest = body.split('<div class="wrap">', 1)
    (od / 'report.html').write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'{head}</head>\n<body>\n<div class="wrap">{rest}\n</body>\n</html>\n')
    print(F.round(2).to_string(index=False))
    print(C.cls.value_counts().to_string())
    print(acc.to_string(index=False))
    print(f'wrote {od}/report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
