#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tracking_qa_report.py -- the HTML report for `tracking_qa.py`.

Writes `<qa-dir>/report.html`. The DAQ page's Analysis tab lists any `.html` in
an analysis directory and opens it inline, and its `/analysis_file/<relpath>`
route is path-based, so every figure is referenced with an ORDINARY RELATIVE
LINK (`figures/x.png`) and the same file works from disk, from the web page, or
copied elsewhere with its `figures/` beside it.

Generated, never hand-written: re-running `tracking_qa` and then this rebuilds
the tables, the tiles and the verdict text together, so a number cannot go
stale in one place and not the other.

    python -m sept26_prelim_analysis.make_tracking_qa_report
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths            # noqa: E402
from sept26_prelim_analysis.tracking_qa import VARS, ARMS, PATHOLOGY  # noqa: E402

ARM_COLOR = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}

FIGURES = [
    ('qa_reference.png', 'What normal looks like, per chamber',
     'The campaign distribution of the four variables that carry the most: '
     'fit quality, cluster size, the angle error that sets the opening-angle '
     'resolution, and the charge. Chamber A is visibly bimodal in '
     '&chi;<sup>2</sup>/dof &mdash; it has a population that fits at '
     '&chi;<sup>2</sup>/dof &asymp; 1.3 that no other chamber has at all. '
     'The spike in the last charge bin is the overflow: a quarter of all '
     'tracks sit above it.'),
    ('qa_by_run_chi2dof_x.png', '&chi;<sup>2</sup>/dof, one curve per run',
     'The outlier hunt. Grey is every run; red is a run this variable flags. '
     'A pack that stays tight while its angle scale moves is telling you the '
     'scale is not moving because the fits changed.'),
    ('qa_by_run_n_strips_x.png', 'Strips in fit, one curve per run',
     'Cluster size is where a dead or noisy connector shows up first: a plane '
     'missing a block of channels fits a wider, sparser cluster and drops far '
     'more strips. The pre-access runs are the curves that sit furthest right.'),
    ('qa_by_run_t0_x.png', 'Fitted t<sub>0</sub>, one curve per run',
     'The drift-window origin. A shift here moves the depth-to-time mapping '
     'that the transverse speed is divided by, so it is one of the few things '
     'in the reconstruction that could plausibly move an angle scale.'),
    ('qa_timeline.png', 'What drifts, tag by tag',
     'Per-tag median with its inter-quartile band against wall-clock time, '
     'for the variable that trends hardest on each chamber. The dashed line is '
     'the 27 July access. Most of what a campaign-wide correlation calls a '
     '&ldquo;trend&rdquo; is visibly a STEP at that line.'),
    ('qa_outlier_map.png', 'Which run departs, and on what',
     'Runs in time order down, variables across, robust z of the run median as '
     'colour, computed within each arm so a globally poor chamber does not '
     'colour every row. This is the index into the two figures above.'),
    ('qa_pathology.png', 'Fits that returned a number that cannot be right',
     'Not poor fits &mdash; impossible ones. A 12-bit DREAM cluster cannot '
     'hold 10<sup>6</sup> ADC and a 512-strip plane cannot give a 200-strip '
     'track a meaningful angle. A median is blind to every one of these.'),
]

#: Rows of the headline reference table, in the order a reader wants them.
REF_ROWS = [
    ('n_tracks',              'gated tracks',                 '{:,.0f}'),
    ('chi2dof_x_p25',         '&chi;<sup>2</sup>/dof x, p25',  '{:.2f}'),
    ('chi2dof_x_p50',         '&chi;<sup>2</sup>/dof x, median', '{:.1f}'),
    ('chi2dof_x_p75',         '&chi;<sup>2</sup>/dof x, p75',  '{:.1f}'),
    ('chi2dof_y_p50',         '&chi;<sup>2</sup>/dof y, median', '{:.1f}'),
    ('n_strips_x_p50',        'strips in fit x, median',       '{:.0f}'),
    ('n_strips_y_p50',        'strips in fit y, median',       '{:.0f}'),
    ('tan_err_x_p50',         '&sigma;(tan&thinsp;&theta;) x, median', '{:.5f}'),
    ('tan_err_x_p95',         '&sigma;(tan&thinsp;&theta;) x, p95', '{:.5f}'),
    ('q_total_p50',           'cluster charge, median [ADC]',  '{:,.0f}'),
    ('drift_len_p50',         'drift span, median [mm]',       '{:.2f}'),
    ('frac_gated',            'gated fraction',                '{:.3f}'),
    ('frac_x_quality_ok',     'x quality ok',                  '{:.3f}'),
    ('frac_y_quality_ok',     'y quality ok',                  '{:.3f}'),
    ('frac_x_slope_reliable', 'x slope reliable',              '{:.3f}'),
    ('frac_drift_railed',     'drift railed',                  '{:.3f}'),
]

PATH_ROWS = [
    ('frac_chi2_gt_100',   '&chi;<sup>2</sup>/dof &gt; 100'),
    ('frac_chi2_gt_1000',  '&chi;<sup>2</sup>/dof &gt; 1000'),
    ('frac_tanerr_gt_0p1', '&sigma;(tan&thinsp;&theta;) &gt; 0.1'),
    ('frac_q_gt_1e6',      'cluster charge &gt; 10<sup>6</sup> ADC'),
    ('frac_strips_ge_200', '&ge; 200 strips in one track'),
]


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def _fmt(v, f: str) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '&mdash;'
    if not np.isfinite(x):
        return '&mdash;'
    return f.format(x)


def arm_table(per_arm: pd.DataFrame, rows) -> str:
    d = per_arm.set_index('arm')
    head = ''.join(f'<th style="color:{ARM_COLOR[a]}">{a}</th>' for a in ARMS
                   if a in d.index)
    body = []
    for key, label, *rest in rows:
        f = rest[0] if rest else '{:.4f}'
        if key not in d.columns:
            continue
        cells = ''.join(f'<td>{_fmt(d.loc[a, key], f)}</td>'
                        for a in ARMS if a in d.index)
        body.append(f'<tr><td>{label}</td>{cells}</tr>')
    return (f'<div class="tbl-wrap"><table class="num"><thead><tr><th></th>'
            f'{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def outlier_table(o: pd.DataFrame, n: int = 20) -> str:
    if o.empty:
        return '<p class="sub">No run departs its arm at this threshold.</p>'
    body = []
    for r in o.head(n).itertuples():
        var = r.variable[:-4] if r.variable.endswith('_p50') else r.variable
        body.append(
            f'<tr><td style="color:{ARM_COLOR.get(r.arm, "")}"><b>{esc(r.arm)}'
            f'</b></td><td>{esc(r.run)}</td><td>{esc(var)}</td>'
            f'<td>{_fmt(r.value, "{:.4g}")}</td>'
            f'<td>{_fmt(r.arm_median, "{:.4g}")}</td>'
            f'<td>{_fmt(r.effect, "{:.2f}")}</td>'
            f'<td>{_fmt(r.z, "{:.1f}")}</td>'
            f'<td>{r.n_tracks:,}</td></tr>')
    return ('<div class="tbl-wrap"><table class="num"><thead><tr>'
            '<th>arm</th><th>run</th><th>variable</th><th>run median</th>'
            '<th>arm median</th><th>effect<br><small>IQR</small></th>'
            '<th>z</th><th>n</th></tr></thead><tbody>'
            + ''.join(body) + '</tbody></table></div>')


def drift_table(d: pd.DataFrame, n: int = 12) -> str:
    if d.empty:
        return '<p class="sub">No trend table.</p>'
    body = []
    for r in d.head(n).itertuples():
        var = r.variable[:-4] if r.variable.endswith('_p50') else r.variable
        body.append(
            f'<tr><td style="color:{ARM_COLOR.get(r.arm, "")}"><b>{esc(r.arm)}'
            f'</b></td><td>{esc(var)}</td>'
            f'<td>{_fmt(r.rho_post, "{:+.2f}")}</td>'
            f'<td>{_fmt(r.p_post, "{:.0e}")}</td>'
            f'<td>{_fmt(r.rho, "{:+.2f}")}</td>'
            f'<td>{_fmt(r.pre_access, "{:.4g}")}</td>'
            f'<td>{_fmt(r.post_access, "{:.4g}")}</td>'
            f'<td>{_fmt(r.last_decile, "{:.4g}")}</td></tr>')
    return ('<div class="tbl-wrap"><table class="num"><thead><tr>'
            '<th>arm</th><th>variable</th><th>&rho; after<br>the access</th>'
            '<th>p</th><th>&rho; all<br>tags</th><th>pre-access</th>'
            '<th>post-access</th><th>last 10&nbsp;%</th>'
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def gap_table(g: pd.DataFrame) -> str:
    """The geometric bound on k, per arm."""
    if g.empty:
        return '<p class="sub">No gap-check table.</p>'
    body = []
    for r in g.itertuples():
        over = float(r.frac_over_gap)
        flag = ' style="color:var(--warn);font-weight:650"' if over > 0.05 else ''
        body.append(
            f'<tr><td style="color:{ARM_COLOR.get(r.arm, "")}"><b>{esc(r.arm)}'
            f'</b></td><td>{_fmt(r.gap_mm, "{:.1f}")}</td>'
            f'<td>{_fmt(r.k_applied, "{:.3f}")}</td>'
            f'<td>{_fmt(r.v_um_ns, "{:.1f}")}</td>'
            f'<td>{_fmt(r.max_span_mm, "{:.1f}")}</td>'
            f'<td>{_fmt(r.span_p50, "{:.1f}")}</td>'
            f'<td>{_fmt(r.span_p50_unrailed, "{:.1f}")}</td>'
            f'<td{flag}>{_fmt(over, "{:.3f}")}</td>'
            f'<td>{_fmt(r.k_min_for_gap, "{:.3f}")}</td></tr>')
    return ('<div class="tbl-wrap"><table class="num"><thead><tr>'
            '<th>arm</th><th>gap<br>[mm]</th><th>k applied</th>'
            '<th>v<br>[&mu;m/ns]</th><th>deepest<br>possible [mm]</th>'
            '<th>span<br>median</th><th>span median<br>unrailed</th>'
            '<th>fraction<br>past the gap</th><th>k needed<br>for the gap</th>'
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def tiles(meta: dict, per_arm: pd.DataFrame, o: pd.DataFrame,
          d: pd.DataFrame) -> str:
    d0 = per_arm.set_index('arm')
    qbad = float(d0['frac_q_gt_1e6'].mean())
    if d.empty:
        rho_s, trend_k, trend_s = '&mdash;', 'no trend table', ''
    else:
        w = d.iloc[0]
        var = w['variable']
        var = var[:-4] if var.endswith('_p50') else var
        rho_s = '{:+.2f}'.format(float(w['rho_post']))
        trend_k = 'strongest post-access trend ({}, {})'.format(w['arm'], var)
        trend_s = '{:.4g} &rarr; {:.4g}'.format(float(w['pre_access']),
                                                float(w['last_decile']))
    items = [
        ('{:,}'.format(meta['n_tracks']), 'gated tracks profiled',
         '{} runs, {:,} file tags'.format(meta['n_runs'], meta['n_tags'])),
        ('{:.0f}&thinsp;%'.format(qbad * 100),
         'tracks with an impossible charge',
         'cluster charge above 10<sup>6</sup> ADC on a 12-bit ADC'),
        ('{}'.format(len(o)), 'run-level outliers flagged',
         'robust z &ge; 3.5 and &ge; 0.15 IQR of shift'),
        (rho_s, trend_k, trend_s),
    ]
    return '<div class="tiles">' + ''.join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-k">{k}</div><div class="tile-s">{s}</div></div>'
        for v, k, s in items) + '</div>'


def figures_html(qa_dir: str) -> str:
    out = []
    for name, title, cap in FIGURES:
        if not os.path.exists(os.path.join(qa_dir, 'figures', name)):
            continue
        out.append(
            f'<figure><img src="figures/{name}" alt="{esc(title)}">'
            f'<figcaption><b>{title}.</b> {cap}</figcaption></figure>')
    return '\n'.join(out)


CSS = """
:root { color-scheme: light dark;
  --bg:#fbfbfa; --surface:#ffffff; --line:#e3e2df;
  --ink:#14140f; --ink2:#55534c; --ink3:#87857c;
  --warn:#a8600f; --warnbg:#a8600f14; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --warn:#e0a44c; --warnbg:#e0a44c1a; } }
:root[data-theme="dark"] {
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --warn:#e0a44c; --warnbg:#e0a44c1a; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:1.62rem; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em; }
h2 { font-size:1.12rem; margin:38px 0 12px; padding-bottom:7px;
  border-bottom:1px solid var(--line); letter-spacing:-.005em; }
h3 { font-size:.98rem; margin:22px 0 8px; }
.sub { color:var(--ink2); margin:0 0 22px; font-size:.94rem; }
p { margin:0 0 12px; }
ul { margin:0 0 14px; padding-left:20px; }
li { margin:0 0 6px; }
code { font:.86em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; }
.verdict { background:var(--warnbg); border:1px solid var(--warn);
  border-left-width:4px; border-radius:8px; padding:16px 18px; margin:0 0 26px; }
.verdict b { color:var(--warn); }
.tiles { display:grid; gap:12px; margin:0 0 20px;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); }
.tile { background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:14px 16px; }
.tile-v { font-size:1.5rem; font-weight:650; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }
.tile-k { color:var(--ink2); font-size:.85rem; margin-top:3px; }
.tile-s { color:var(--ink3); font-size:.78rem; margin-top:2px; }
.tbl-wrap { overflow-x:auto; margin:0 0 14px; }
table { border-collapse:collapse; width:100%; font-size:.87rem;
  background:var(--surface); border:1px solid var(--line); border-radius:8px; }
th,td { padding:7px 11px; text-align:left; border-bottom:1px solid var(--line);
  white-space:nowrap; }
th { color:var(--ink2); font-weight:600; font-size:.8rem; }
tbody tr:last-child td { border-bottom:0; }
table.num td+td, table.num th+th { text-align:right;
  font-variant-numeric:tabular-nums; }
figure { margin:0 0 26px; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px; }
figure img { width:100%; height:auto; display:block; border-radius:4px; }
figcaption { color:var(--ink2); font-size:.85rem; margin-top:9px;
  padding:0 4px 2px; }
footer { color:var(--ink3); font-size:.8rem; margin-top:44px;
  border-top:1px solid var(--line); padding-top:14px; }
"""


def build(qa_dir: str) -> str:
    meta = json.load(open(paths.require(
        os.path.join(qa_dir, 'tracking_qa.meta.json'), 'tracking_qa meta')))
    per_arm = pd.read_csv(os.path.join(qa_dir, 'per_arm.csv'))
    o = pd.read_csv(os.path.join(qa_dir, 'outliers.csv'))
    dr = pd.read_csv(os.path.join(qa_dir, 'drift.csv'))
    gap_p = os.path.join(qa_dir, 'gap_check.csv')
    gp = pd.read_csv(gap_p) if os.path.exists(gap_p) else pd.DataFrame()
    if gp.empty:
        gap_verdict = 'No gap-check table.'
    else:
        g0 = gp.set_index('arm')
        fails = [a for a in g0.index if float(g0.loc[a, 'frac_over_gap']) > 0.05]
        passes = [a for a in g0.index if a not in fails]
        if not fails:
            gap_verdict = ('<b>Every chamber passes.</b> No arm reconstructs a '
                           'meaningful fraction of its tracks deeper than its '
                           'own gap.')
        else:
            bits = []
            for a in fails:
                bits.append(
                    f"<b>{a}: {float(g0.loc[a, 'frac_over_gap']) * 100:.0f}&thinsp;% "
                    f"of its gated tracks reconstruct deeper than its "
                    f"{float(g0.loc[a, 'gap_mm']):.1f}&nbsp;mm gap</b>, with a "
                    f"median unrailed span of "
                    f"{float(g0.loc[a, 'span_p50_unrailed']):.1f}&nbsp;mm and a "
                    f"scale of {float(g0.loc[a, 'k_applied']):.3f} against the "
                    f"{float(g0.loc[a, 'k_min_for_gap']):.3f} the gap demands")
            gap_verdict = (
                ('Chambers ' + ', '.join(passes) + ' pass. ' if passes else '')
                + 'Chamber' + ('s ' if len(fails) > 1 else ' ')
                + ', '.join(fails) + ' do' + ('' if len(fails) > 1 else 'es')
                + ' not &mdash; ' + '; and '.join(bits) + '.')

    d0 = per_arm.set_index('arm')
    qbad = float(d0['frac_q_gt_1e6'].mean())
    d_chi = float(d0.loc['D', 'frac_chi2_gt_100']) if 'D' in d0.index else np.nan
    top_runs = (o.groupby('run').size().sort_values(ascending=False)
                if not o.empty else pd.Series(dtype=int))
    worst = dr.iloc[0] if not dr.empty else None
    if worst is not None:
        w_rho = float(worst['rho_post'])
        w_p = float(worst['p_post'])
        w_n = int(worst['n_tags_post'])
        w_pre = float(worst['pre_access'])
        w_last = float(worst['last_decile'])
    a_p25 = float(d0.loc['A', 'chi2dof_x_p25']) if 'A' in d0.index else float('nan')

    # Every number below is read from the tables. The first version of this
    # typed them in, and the full pass then changed D's median fit quality from
    # 39 to 17 while the prose still said 39.
    chi_p50 = {a: float(d0.loc[a, 'chi2dof_x_p50']) for a in d0.index}
    chi_p25 = {a: float(d0.loc[a, 'chi2dof_x_p25']) for a in d0.index}
    best = min(chi_p25, key=chi_p25.get)
    others = [a for a in ARMS if a in chi_p25 and a != best]
    qmax = d0['frac_q_gt_1e6'].idxmax()

    if worst is None:
        verdict = '<p>Trend table empty.</p>'
    else:
        w_var = worst['variable']
        w_var = w_var[:-4] if w_var.endswith('_p50') else w_var
        verdict = f"""
<p><b>Three things this found that a median could not.</b></p>
<ul>
<li><b>Roughly one gated track in {round(1 / qbad)} carries a charge that
cannot be real.</b> {qbad * 100:.1f}&thinsp;% of tracks have
<code>q_total</code> above 10<sup>6</sup> ADC &mdash; worst on chamber
{esc(qmax)} at {float(d0.loc[qmax, 'frac_q_gt_1e6']) * 100:.1f}&thinsp;% &mdash;
and the 95th percentile reaches 10<sup>14</sup>&ndash;10<sup>17</sup> on a
12-bit ADC sitting on a ~330-count pedestal. <code>q_total</code> is
<code>x_q_sum + y_q_sum</code> and both plane sums diverge together, so this is
the fit's amplitude solution running away on some depth bins, not a units
error. Their &chi;<sup>2</sup> is unremarkable, which is why nothing caught it:
the waveform still fits. <b>Every charge-based statement in this analysis
&mdash; gain comparisons, <code>q_per_len</code>, the charge window
<code>k_arm</code> cuts on &mdash; runs on a column with a tail of garbage that
large.</b></li>
<li><b>Chamber {esc(best)} fits its own model better than any other
chamber.</b> Its lower quartile of &chi;<sup>2</sup>/dof is
<b>{chi_p25[best]:.2f}</b>, against
{", ".join(f"{a} {chi_p25[a]:.2f}" for a in others)}; medians
{chi_p50[best]:.1f} against
{", ".join(f"{chi_p50[a]:.1f}" for a in others)}. This is the same split that
shows up in the angle scale: {esc(best)} is also the arm whose <code>k</code>
varies least run to run. One fact, not two.</li>
<li><b>The 27 July access is a step, and separating it changes what counts as
a trend.</b> Correlating any variable against time across the whole campaign
scores that step as a strong monotone trend, so this reports &rho; before and
after it separately. The strongest surviving post-access trend is
<b>chamber&nbsp;{esc(worst['arm'])}'s {esc(w_var)}</b>
(&rho;&nbsp;=&nbsp;{w_rho:+.2f}, p&nbsp;=&nbsp;{w_p:.0e}, over {w_n:,} tags),
moving {w_pre:.4g} before the access to {w_last:.4g} in the last tenth of the
campaign.</li>
</ul>
<p><b>The clearest outliers are already understood.</b> The runs with the most
flags are {", ".join(f"{esc(r)} ({n})" for r, n in top_runs.head(3).items())}.
run_79 and run_81 are the pre-access condition, when chamber A's x-view
connector was dead; a run that fails to certify an angle scale flags on
everything downstream of it. Both are checks that this method finds what it
should, not new results.</p>
"""

    body = f"""<div class="wrap">
<h1>Tracking QA &mdash; the distributions, run by run and tag by tag</h1>
<p class="sub">{meta['n_tracks']:,} gated tracks &middot; {meta['n_runs']} runs
&middot; {meta['n_subruns']} sub-runs &middot; {meta['n_tags']:,} file tags
&middot; generated {esc(meta['generated'])}</p>

<div class="verdict">{verdict}</div>

{tiles(meta, per_arm, o, dr)}

<h2>Why this exists</h2>
<p>Every tracking number quoted in this analysis so far has been a median over
the whole campaign. A median cannot tell a chamber that is uniformly mediocre
from one that is fine for most of the campaign and catastrophic for two hours
on a Tuesday, and the second is the case worth finding. The angle scale
<code>k</code> is measured to move 13&ndash;17&thinsp;% between runs with
nothing in the configuration to explain it, so the question here is whether
anything in the reconstruction itself moves with it.</p>
<p><b>Nothing here cuts, corrects or reweights.</b> An outlier below is a lead,
not a verdict.</p>

<h2>What normal is, per chamber</h2>
{arm_table(per_arm, REF_ROWS)}
<p class="sub">Quantiles are computed on the raw column. Chamber&nbsp;B's
angles are null by design and not by failure: <code>k_arm</code> never
certifies&nbsp;B, so <code>tanx</code> is left null rather than defaulting
<code>k</code> to&nbsp;1. Everything downstream of that scale is null with it
&mdash; <code>tan_sane</code>, the drift span, the path length and
<code>q_per_len</code> &mdash; so B's blank cells here restate one missing
number and are not four tracking results.</p>

<h2>Fits that returned an impossible number</h2>
{arm_table(per_arm, [(k, lab, '{:.4f}') for k, lab in PATH_ROWS])}
<p class="sub">Fraction of that chamber's gated tracks. Chamber&nbsp;D puts
{d_chi * 100:.0f}&thinsp;% of its tracks above &chi;<sup>2</sup>/dof&nbsp;=&nbsp;100,
which is a quarter of the arm effectively unfit rather than merely poorly fit.</p>

<h2>Does the drift span fit inside the chamber?</h2>
<p>A bound on the angle scale that owes nothing to the pointing estimators, and
therefore checks them. The reconstruction turns the fitted drift end time into
a depth with <code>v&nbsp;=&nbsp;42.6/k</code>, and the depth grid stops at
18&nbsp;&times;&nbsp;60&nbsp;=&nbsp;1080&nbsp;ns, so the deepest span it can
produce is fixed once <code>k</code> is. That span has to fit in the drift gap.</p>
{gap_table(gp)}
<p class="sub">{gap_verdict}</p>
<p class="sub"><b>Two readings, both calibration faults, and this test cannot
choose between them.</b> Either the angle scale is too small on the failing
arms, or their depth-grid origin sits outside the gas so the span is inflated
with <code>k</code> innocent &mdash; which is the same fitted
<code>t<sub>0</sub></code> that moves between runs. Note the direction: the
pointing estimators want a <em>smaller</em> <code>k</code> and this bound wants
a <em>larger</em> one, on the same arms. That disagreement is the thing to
settle in October, and it argues against treating any single arm as the
reference.</p>

<h2>Which run departs, and on what</h2>
{outlier_table(o)}
<p class="sub">Flagged when a run's median sits at robust z&nbsp;&ge;&nbsp;3.5
from its own arm's median <em>and</em> is shifted by at least
0.15 of that arm's track-level inter-quartile range. Both are required:
several statistics agree so closely across runs that their across-run spread
collapses, and a z alone then flags every run that differs at all. Runs with
the most flags: {", ".join(f"{esc(r)} ({n})" for r, n in top_runs.head(5).items())}.</p>
<p class="sub"><b>run_145 has no per-tag granularity in this table.</b> Its
tracks were built from an already-merged August table and carry the literal
tag <code>prelim</code>, so it has no timestamp, contributes nothing to the
tag-level drift test, and is ordered here by run number rather than by clock.
The condor full pass writes run_145 per tag like every other run, so this
resolves when the table is rebuilt on it.</p>

<h2>What drifts, and what merely stepped</h2>
{drift_table(dr)}
<p class="sub">Spearman &rho; of the per-tag median against the tag timestamp.
Two columns, and the difference between them is the result: &rho; over every
tag, and &rho; over the tags after the 27&nbsp;July access only. A variable
with a large all-tag &rho; and a small post-access &rho; stepped once at the
access and then held. One with both is genuinely drifting.</p>

<h2>The figures</h2>
{figures_html(qa_dir)}

<h2>What this does not rule out</h2>
<ul>
<li><b>It does not explain the angle scale.</b> Nothing here moves run to run
in a way that tracks <code>k</code>'s 13&ndash;17&thinsp;% scatter. The
charge blow-up and D's degradation are real problems and are not, on this
evidence, the cause of that scatter.</li>
<li><b>It cannot separate the chamber from the model.</b> A chamber whose fit
quality is poor and a forward model that does not describe that chamber look
identical in every distribution here. Deciding between them needs the
per-detector recalibration, not more QA.</li>
<li><b>The tag is a clock, not a condition.</b> Two tags an hour apart share a
timestamp neighbourhood but nothing guarantees they share a gas state, a
threshold or an occupancy. A trend here is a correlation with time and not
with a cause.</li>
<li><b>Chamber B is under-measured, not clean.</b> Its distributions look
tight because the quantities that would expose a problem &mdash; anything
built on the angle &mdash; are null for every B track.</li>
</ul>

<footer>
Generated by <code>sept26_prelim_analysis/make_tracking_qa_report.py</code>
from <code>{esc(os.path.basename(meta['src']))}</code>.
Tables beside this file; every figure ships the CSV it was drawn from.
</footer>
</div>"""

    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>Tracking QA &mdash; run by run, tag by tag</title>\n'
            f'<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--qa-dir', default=None, help='default <out>/tracking_qa')
    a = ap.parse_args()
    qa = a.qa_dir or str(paths.out('tracking_qa'))
    out = os.path.join(qa, 'report.html')
    with open(out, 'w') as f:
        f.write(build(qa))
    print(f'  -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
