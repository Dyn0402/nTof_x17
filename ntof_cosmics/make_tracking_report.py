#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build results/tracking/report.html from cosmic_tracks.py's outputs.

Generated, not hand-written: re-run `cosmic_tracks.py build` and `analyse`,
then this, and the numbers, figures and verdict move together.  One report per
sub-run for now (run_149/cos_0000, the one with a clock match).

    .venv/bin/python ntof_cosmics/make_tracking_report.py
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
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

import cosmic_tracks as CT  # noqa: E402

RES = CT.OUT
FIG = RES / 'figures'
KS = ('run_147', 'run_150')


def load(run, sub):
    S = {k: json.loads((RES / f'k_{k}' / f'summary_{run}_{sub}.json').read_text())
         for k in KS}
    P = {k: pd.read_parquet(RES / f'k_{k}' / f'pairs_{run}_{sub}.parquet') for k in KS}
    T = {k: pd.read_parquet(RES / f'k_{k}' / f'tracks_{run}_{sub}.parquet') for k in KS}
    return S, P, T


def figures(run, sub, S, P, T) -> list[tuple[str, str]]:
    fs.use()
    out = []
    bins = np.arange(90, 181, 5)

    # 1 -- opening angle, opposing pairs: all against clean
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    for ax, k in zip(axs, KS):
        p = P[k].sort_values('open_deg', ascending=False).drop_duplicates('event_id')
        p = p[p.topo == 'opposing']
        ax.hist(p.open_deg, bins=bins, histtype='stepfilled', color=fs.GRID,
                ec=fs.MUTED, label=f'all opposing ({len(p)})')
        c = p[p.sep_mm < CT.CLEAN_SEP_MM]
        ax.hist(c.open_deg, bins=bins, histtype='step', color=fs.ACCENT, lw=1.4,
                label=f'lines meet < {CT.CLEAN_SEP_MM:g} mm ({len(c)})')
        ax.axvline(CT.BACK_TO_BACK_DEG, color=fs.COPPER, lw=1, ls='--')
        ax.set_xlabel('opening angle [deg]')
        ax.set_title(f'k from {k}', fontsize=10, color=fs.MUTED, loc='left')
    axs[0].set_ylabel('triggers per 5 deg')
    axs[0].legend(frameon=False, loc='upper left')
    s = S['run_150']
    fs.fig_title(fig, f"Only {s['frac_clean_above_170'] * 100:.0f}–"
                      f"{S['run_147']['frac_clean_above_170'] * 100:.0f} % of clean "
                      f"through-goers pass the 170° cut",
                 f'{run}/{sub}, A–C pairs, most collinear pair per trigger; '
                 'dashed: BACK_TO_BACK_DEG')
    fs.save(fig, FIG / 'opening_angle',
            data=P['run_150'][['event_id', 'pair', 'open_deg', 'sep_mm']])
    out.append(('opening_angle.png',
                'Opening angle of the most collinear opposing pair in each '
                'trigger (all A–C: B–D needs a horizontal cosmic). Grey: every '
                'pair. Purple: the two chambers\' lines pass within '
                f'{CT.CLEAN_SEP_MM:g} mm of each other, i.e. plausibly one '
                'straight particle. The grey tail below ~160° is mostly pairs '
                'whose lines miss each other by centimetres: two particles, or '
                'one chamber mis-reconstructed. It is not resolution.'))

    # 2 -- separation vs opening angle
    p = P['run_150']
    p = p[p.topo == 'opposing']
    fig, ax = fs.figure(fs.FIG)
    ax.semilogy(p.open_deg, p.sep_mm.clip(lower=0.3), 'o', ms=3,
                color=fs.ACCENT, alpha=0.7)
    ax.axhline(CT.CLEAN_SEP_MM, color=fs.MUTED, lw=0.8, ls=':')
    ax.axvline(CT.BACK_TO_BACK_DEG, color=fs.COPPER, lw=1, ls='--')
    ax.set_xlabel('opening angle [deg]')
    ax.set_ylabel('closest approach of the two lines [mm]')
    fs.title(ax, 'Collinear pairs are the ones whose lines actually meet',
             'opposing pairs, k from run_150')
    fs.save(fig, FIG / 'sep_vs_open', data=p[['open_deg', 'sep_mm']])
    out.append(('sep_vs_open.png',
                'Every opposing pair. Pairs above ~170° nearly all have lines '
                'that meet within a few mm. The broad population at 120–160° '
                'mostly misses by > 20 mm.'))

    # 3 -- slope check
    t = T['run_150'].set_index(['event_id', 'arm', 'track_id'])
    c = p[(p.pair == 'AC') & (p.sep_mm < CT.CLEAN_SEP_MM)]
    a = t.loc[list(zip(c.event_id, c.arm1, c.track1))]
    b = t.loc[list(zip(c.event_id, c.arm2, c.track2))]
    J = {ax_: b[f'p0_{ax_}'].to_numpy() - a[f'p0_{ax_}'].to_numpy() for ax_ in 'xyz'}
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE, sharex=True, sharey=True)
    rows = []
    for ax, axis in zip(axs, 'xy'):
        j = J[axis] / J['z']
        for arm, tr in (('A', a), ('C', b)):
            s_ = tr[f'd_{axis}'].to_numpy() / tr.d_z.to_numpy()
            ax.plot(j, s_, fs.DET_MARKER[arm], ms=4, color=fs.DET_COLOR[arm],
                    alpha=0.8, label=arm)
            rows.append(pd.DataFrame(dict(axis=axis, arm=arm, joined=j, track=s_)))
        lim = 1.2
        ax.plot([-lim, lim], [-lim, lim], color=fs.MUTED, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'joined line, d{axis}/dz')
        ax.set_title(f'{axis} slope', fontsize=10, color=fs.MUTED, loc='left')
    axs[0].set_ylabel('chamber track, d/dz')
    axs[0].legend(frameon=False, loc='upper left')
    sc = pd.DataFrame(S['run_150']['slope_check'])
    r = sc.set_index(['arm', 'axis']).median_ratio
    fs.fig_title(fig, f"A reads ~{(r['A'].mean() - 1) * 100:+.0f} %, C "
                      f"~{(r['C'].mean() - 1) * 100:+.0f} % against the line "
                      'through both chambers',
                 f'{len(c)} clean A–C through-goers, k from run_150; '
                 'diagonal: borrowed k exactly right')
    fs.save(fig, FIG / 'slope_check', data=pd.concat(rows, ignore_index=True))
    out.append(('slope_check.png',
                'Each chamber\'s reconstructed track slope against the slope of '
                'the straight line through the two chambers\' track points: a '
                'capsule-free angle-scale check. A lies above the diagonal in '
                'both projections, so its tans read too large and the borrowed k '
                'is too LARGE for A (s/j = k_borrowed/k_true); C sits near or below '
                'it. The pooled run_149 analysis (pooled/report.html) shows the '
                'ratio also depends on angle. C\'s x slope also scatters (corr '
                f"{sc.set_index(['arm', 'axis'])['corr'][('C', 'x')]:.2f}), which "
                'the median ratio is robust to and the least-squares ratio is '
                'not.'))
    return out


def main() -> None:
    run, sub = 'run_149', 'cosbounce_cos_0000'
    S, P, T = load(run, sub)
    figs = figures(run, sub, S, P, T)
    a, b = S['run_147'], S['run_150']
    n = a['n_triggers']

    def pct(x):
        return f'{x / n * 100:.1f} %'

    pair_rows = ''.join(
        f'<tr><td>{pr}</td><td>{"opposing" if pr in ("AC", "BD") else "perpendicular"}'
        f'</td><td>{a["trig_by_pair"].get(pr, "—")}</td>'
        f'<td>{b["trig_by_pair"].get(pr, "—")}</td></tr>'
        for pr in ('AC', 'BD', 'AB', 'AD', 'BC', 'CD'))
    arm_rows = ''.join(
        f'<tr><td>{x}</td><td>{a["n_trig_per_arm"][x]}</td>'
        f'<td>{pct(a["n_trig_per_arm"][x])}</td>'
        f'<td>{"yes" if a["arms_calibrated"][x] else "<b>no</b>"}</td>'
        f'<td>{"yes" if b["arms_calibrated"][x] else "<b>no</b>"}</td></tr>'
        for x in CT.ARMS)
    sc = {k: pd.DataFrame(S[k]['slope_check']) for k in KS}
    slope_rows = ''
    for _, r in sc['run_150'].iterrows():
        if r.arm not in 'AC':
            continue
        r7 = sc['run_147'].set_index(['arm', 'axis']).loc[(r.arm, r.axis)]
        slope_rows += (f'<tr><td>{r.arm}</td><td>{r.axis}</td><td>{r.n}</td>'
                       f'<td>{r7.median_ratio:.3f}</td><td>{r.median_ratio:.3f}</td>'
                       f'<td>{r.lsq_ratio:.3f}</td><td>{r["corr"]:.2f}</td></tr>')
    q = {k: S[k]['open_deg_quantiles_clean'] for k in KS}
    fig_html = ''.join(
        f'<figure><img src="figures/{f}" alt=""><figcaption>{html.escape(c)}'
        f'</figcaption></figure>' for f, c in figs)

    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beam-off cosmic tracks</title>{HEAD}</head><body><main>
<p class="eyebrow">n_TOF EAR2 · X17 · ntof_cosmics/cosmic_tracks</p>
<h1>Beam-off cosmics through the campaign reco: {run}/{sub}</h1>

<div class="verdict"><p><b>The chain runs on cosmics unchanged, and the
first sub-run already says two things about the beam-on analysis.</b>
(1) <b>The 170° back-to-back cut keeps only
{b['frac_clean_above_170'] * 100:.0f}–{a['frac_clean_above_170'] * 100:.0f} % of
clean through-goers</b> (opposing pairs whose lines meet within
{CT.CLEAN_SEP_MM:g} mm; {b['n_opposing_clean']} of them): their opening angle
has a median of {q['run_150']['0.5']:.1f}° but a 16 % quantile of
{q['run_150']['0.16']:.0f}–{q['run_147']['0.16']:.0f}°. (2) <b>A capsule-free
angle-scale check disagrees with the borrowed k</b>: chamber A's track slopes
run ~{(sc['run_150'].query("arm=='A'").median_ratio.mean() - 1) * 100:+.0f} % and
chamber C's ~{(sc['run_150'].query("arm=='C'").median_ratio.mean() - 1) * 100:+.0f} %
against the line through both chambers, under either neighbour's k. Both
results come from ~40 events and are the reason to run the rest of
run_149, not conclusions.</p></div>

<h2>Headline numbers</h2>
<table><thead><tr><th></th><th>k from run_147</th><th>k from run_150</th></tr></thead><tbody>
<tr><td>DREAM triggers</td><td colspan="2">{n:,}</td></tr>
<tr><td>≥ 1 gated track (any arm)</td><td colspan="2">{a['n_trig_ge1_track']:,}
({pct(a['n_trig_ge1_track'])})</td></tr>
<tr><td>gated tracks in ≥ 2 arms</td><td colspan="2">{a['n_trig_ge2_arms']}
({pct(a['n_trig_ge2_arms'])})</td></tr>
<tr><td>two-arm pairs, both arms angle-calibrated</td><td>{a['n_pairs']}</td><td>{b['n_pairs']}</td></tr>
<tr><td>opposing, lines meet &lt; {CT.CLEAN_SEP_MM:g} mm</td><td>{a['n_opposing_clean']}</td><td>{b['n_opposing_clean']}</td></tr>
<tr><td>… of which &gt; 170°</td><td>{a['frac_clean_above_170'] * 100:.0f} %</td><td>{b['frac_clean_above_170'] * 100:.0f} %</td></tr>
<tr><td>clean opening angle, 5/16/50/84/95 %</td>
<td>{' / '.join(f'{v:.0f}' for v in q['run_147'].values())}°</td>
<td>{' / '.join(f'{v:.0f}' for v in q['run_150'].values())}°</td></tr>
<tr><td>pairs passing the beam-on pointing preselection (both tracks &lt; {CT.DCA_MAX:g} mm from the axis)</td>
<td>{a['n_beam_presel']} ({a['n_beam_presel_b2b']} &gt; 170°)</td><td>{b['n_beam_presel']} ({b['n_beam_presel_b2b']} &gt; 170°)</td></tr>
<tr><td>two-arm triggers with an n_TOF match (±50 ns)</td><td>{a['n_ge2arm_matched']}</td><td>{b['n_ge2arm_matched']}</td></tr>
<tr><td>… where the n_TOF arm is one of the two tracked arms</td>
<td>{a['frac_matched_ntof_arm_in_pair'] * 100:.0f} %</td><td>{b['frac_matched_ntof_arm_in_pair'] * 100:.0f} %</td></tr>
</tbody></table>

<h2>What was run</h2>
<ul>
<li><b>Reco:</b> the campaign full pass exactly — `make_stage2_campaign.py
--full-pass` with a new <code>--tags-json</code> (cosmic runs have no n_TOF
slim, so no stage 1 to take tags from), the campaign bundles (A/B/D r06, C lp),
v_drift pinned at 42.6 µm/ns, no hot-channel mask. Condor cluster 4354841, 12
jobs, none held; output in <code>/eos/user/d/dneff/x17/cosmics_fullpass</code>,
apart from the campaign's. Code changed since the campaign pass only by the
opt-in two-track switches, which are off.</li>
<li><b>Tracks:</b> <code>build_tracks.build</code>, no stage 1 or allowlist, k
<b>borrowed</b> from the neighbouring beam runs on each side (run_147, run_150) —
<code>k_arm</code> assumes capsule tracks and cannot run on cosmics. run_147 has
no certified k for B, so B pairs exist only under run_150's.</li>
<li><b>HV:</b> no trips in <code>hv_monitor.csv</code>; drift at set-point by
14:51:09, the first minute of the first file.</li>
<li><b>Pairs:</b> every pair of gated, calibrated tracks in different arms of
one trigger. <code>open_deg</code> is <code>source_imaging</code>'s definition
(the track directions all point inward), so the 170° cut means the same thing
as on the beam-on sample.</li>
</ul>

<h2>Per arm</h2>
<table><thead><tr><th>arm</th><th>triggers with a gated track</th><th>of all</th>
<th>calibrated (147)</th><th>calibrated (150)</th></tr></thead><tbody>{arm_rows}</tbody></table>

<h2>Two-arm triggers by arm pair</h2>
<p>Most collinear pair per trigger. B–D is empty: it needs a horizontal cosmic.</p>
<table><thead><tr><th>pair</th><th>topology</th><th>k from run_147</th><th>k from run_150</th></tr></thead>
<tbody>{pair_rows}</tbody></table>

<h2>Angle scale without the capsule</h2>
<p>Track slope ÷ slope of the line through both chambers' track points, on
clean A–C through-goers, for |joined slope| &gt; 0.1. A ratio of 1 means the
borrowed k is right; the ratio is k<sub>borrowed</sub> / k<sub>true</sub>, so
above 1 means the tan reads too large. <b>Superseded</b> by the pooled run_149
analysis, <code>pooled/report.html</code>: the ratio depends on angle, so no
single k<sub>true</sub> exists.</p>
<table><thead><tr><th>arm</th><th>axis</th><th>n</th><th>median (147)</th>
<th>median (150)</th><th>LSQ (150)</th><th>corr (150)</th></tr></thead>
<tbody>{slope_rows}</tbody></table>

<h2>Figures</h2>
{fig_html}

<h2>What this does not establish</h2>
<ul>
<li><b>~40 clean events.</b> Every percentage above has a ±7 % binomial error
or worse. The other 86 sub-runs of run_149 (cluster 4355060) take this to
~3 500.</li>
<li><b>The 170° efficiency includes the angle-scale error.</b> A's tans read
~10 % too large under the borrowed k, and the response is angle-dependent
(pooled/report.html), so a straight line's opening angle is smeared by that
alone. Do not move the cut until the angle response is settled.</li>
<li><b>"Clean" is a selection on line agreement</b>, so it favours pairs where
the two chambers agree and pulls the slope ratios toward 1. The departures
above are therefore if anything understated. The unclean tail (lines missing
by centimetres) is not yet understood: two particles, a δ-ray, or one chamber
mis-reconstructed.</li>
<li><b>The track yield is low</b>: only {pct(a['n_trig_ge1_track'])} of triggers
have a gated track, and only {a['n_trig_ge2_arms']} have two arms. Part of it is
the gate (arm A y-plausibility 55 %, D quality 76–79 %, against ~85 % on B/C),
which was tuned on capsule tracks near the normal; the rest is scintillator
coverage outside the chambers. Not yet split.</li>
<li><b>The clock join is thin here</b>: n_TOF sees ~16 % of the time, so only
{b['n_ge2arm_matched']} two-arm triggers are matched. That the n_TOF arm is one of
the two tracked chambers {b['frac_matched_ntof_arm_in_pair'] * 100:.0f} % of the time
is a consistency check of both the clock match and the reco, not a Δt result.</li>
<li>No time-since-flash (there is no flash); the <code>t_since_flash</code>
columns are empty by construction.</li>
</ul>
</main></body></html>
"""
    (RES / 'report.html').write_text(out)
    print(f'wrote {RES / "report.html"}')


if __name__ == '__main__':
    main()
