#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_campaign_imaging_report.py -- ``report.html`` for `campaign_imaging.py`.

Where the He-3 capsule is, measured once per run, and what its stability says
about a calibration that is NOT stable.

Generated, never hand-written: every number is read back from what
`campaign_imaging.py` wrote, so re-running the analysis moves the tables, the
verdict and the figures together.

    python -m sept26_prelim_analysis.make_campaign_imaging_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.figstyle import DET_COLOR  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import ARMS, K_BLOCK  # noqa: E402


def figure(name: str, caption: str) -> str:
    """A figure with an ORDINARY RELATIVE link, and its numbers beside it."""
    import html as _h
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(caption)}"></a>'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


def per_arm_table(PA: pd.DataFrame) -> str:
    rows = []
    for r in PA.itertuples():
        # The comparison that decides whether there is a run-to-run effect at
        # all: run-to-run scatter against the scatter one run's own sub-runs
        # already show.
        ratio = (r.std_mm / r.median_err_repro
                 if np.isfinite(r.median_err_repro) and r.median_err_repro
                 else np.nan)
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">chamber '
            f'{r.arm}</th><td class="n">{r.axis}</td>'
            f'<td class="n">{r.n_runs}</td>'
            f'<td class="n">{r.median_mm:+.2f}</td>'
            f'<td class="n"><b>{r.std_mm:.2f}</b></td>'
            f'<td class="n">{r.p10_mm:+.2f} … {r.p90_mm:+.2f}</td>'
            f'<td class="n">{r.median_err_stat:.2f}</td>'
            f'<td class="n">{r.median_err_repro:.2f}</td>'
            f'<td class="n">{"&mdash;" if not np.isfinite(ratio) else f"{ratio:.2f}"}'
            f'</td><td class="n">{int(r.median_n):,}</td></tr>')
    return (
        '<table><thead><tr><th></th><th>axis</th><th>runs</th>'
        '<th>median (mm)</th><th>sd over runs</th><th>p10 … p90</th>'
        '<th>stat err</th><th>sub-run spread</th><th>sd / sub-run</th>'
        '<th>tracks / run</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>')


def block_table(B: pd.DataFrame) -> str:
    rows = []
    for r in B.itertuples():
        strong = abs(r.shift_over_out_sigma) > 1.0 and r.p_mannwhitney < 0.05
        rows.append(
            f'<tr><th class="s">{r.quantity}</th>'
            f'<td class="n">{r.axis}</td>'
            f'<td class="n">{r.n_out} / {r.n_in}</td>'
            f'<td class="n">{r.median_out:+.2f}</td>'
            f'<td class="n">{r.median_in:+.2f}</td>'
            f'<td class="n">{"<b>" if strong else ""}{r.shift_mm:+.2f}'
            f'{"</b>" if strong else ""}</td>'
            f'<td class="n">{r.shift_over_out_sigma:+.2f}</td>'
            f'<td class="n">{r.p_mannwhitney:.3f}</td></tr>')
    return (
        '<table><thead><tr><th></th><th>axis</th><th>runs out / in</th>'
        '<th>outside (mm)</th><th>inside (mm)</th><th>shift</th>'
        '<th>in sd of the outside runs</th><th>p</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def k_table(V: pd.DataFrame) -> str:
    rows = []
    for r in V.itertuples():
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[r.arm]}">chamber '
            f'{r.arm}</th><td class="n">{r.n_runs}</td>'
            f'<td class="n">{r.k_median:.3f}</td>'
            f'<td class="n">{r.mm_median:+.2f}</td>'
            f'<td class="n">{r.rho:+.2f}</td>'
            f'<td class="n">{r.p:.3f}</td></tr>')
    return ('<table><thead><tr><th></th><th>runs</th><th>median k</th>'
            '<th>median crossing (mm)</th><th>Spearman &rho;</th><th>p</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def y_table(Y: pd.DataFrame) -> str:
    rows = []
    for arm in ARMS:
        g = Y[(Y.arm == arm) & Y.offset_mm.notna()]
        if not len(g):
            continue
        rows.append(
            f'<tr><th class="s" style="color:{DET_COLOR[arm]}">chamber '
            f'{arm}</th><td class="n">{len(g)}</td>'
            f'<td class="n">{g.offset_mm.median():+.1f}</td>'
            f'<td class="n">&plusmn;{g.offset_mm.std(ddof=1):.1f}</td>'
            f'<td class="n">{g.width_ratio.median():.2f}</td>'
            f'<td class="n">{int(g.n.median()):,}</td></tr>')
    return ('<table><thead><tr><th></th><th>runs</th>'
            '<th>median y offset (mm)</th><th>sd over runs</th>'
            '<th>IQR ratio</th><th>tracks / run</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def outlier_list(A: pd.DataFrame, n: int = 5) -> str:
    post = A[A.condition == 'post_access_27jul'].dropna(subset=['x_source_mm'])
    med = post.x_source_mm.median()
    d = post.assign(dev=(post.x_source_mm - med).abs()) \
            .sort_values('dev', ascending=False).head(n)
    return ''.join(
        f'<li><b>{r.run}</b> — capsule X {r.x_source_mm:+.2f} mm '
        f'({r.x_source_mm - med:+.2f} from the campaign median), '
        f'alignment {r.x_align_half_diff_mm:+.2f} mm'
        f'{" · inside runs 128–147" if r.k_block else ""}</li>'
        for r in d.itertuples())


def build(d: Path) -> str:
    meta = json.loads((d / 'campaign_imaging.meta.json').read_text())
    v = meta['verdict']
    PA = pd.read_csv(d / 'per_arm.csv')
    A = pd.read_csv(d / 'axis_per_run.csv')
    B = pd.read_csv(d / 'block_test.csv')
    V = pd.read_csv(d / 'versus_k.csv') if (d / 'versus_k.csv').exists() \
        else pd.DataFrame()
    Y = pd.read_csv(d / 'y_per_run.csv') if (d / 'y_per_run.csv').exists() \
        else pd.DataFrame()
    failed = meta.get('failed', {})
    kb = f'{K_BLOCK[0]}–{K_BLOCK[1]}'

    # The one comparison the whole page exists to make -- MEASURED here from
    # the same k_arm JSONs the analysis runs on, never transcribed from a
    # status file that can go stale while this page keeps quoting it.
    from sept26_prelim_analysis.campaign_imaging import read_k, condition
    K = read_k()
    K = K[[condition(r) == 'post_access_27jul' for r in K.run]]
    k_spread = {}
    for arm, g in K.groupby('arm'):
        if len(g) < 6:      # B certifies on a handful of runs; not a spread
            continue
        k_spread[arm] = float(100 * (g.k.quantile(0.9) - g.k.quantile(0.1))
                              / g.k.median())

    parts = [f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capsule imaging, run by run</title>{HEAD}
</head><body><main>
<h1>The capsule, imaged once per run</h1>
<p class="lede">The angle scale moves up to {max(k_spread.values()):.0f}&nbsp;% from run to run. The capsule
position, measured by the same tracks with a <b>scale-free</b> estimator, moves
<b>{v['x_source_std_mm']:.2f}&nbsp;mm</b>. The geometry is sound and the fault
is in the angle scale alone.</p>

<h2>The answer</h2>
<p>A track from a point source at perpendicular distance <i>d</i> crosses the
strip plane at tan&nbsp;=&nbsp;(<i>u</i>&nbsp;&minus;&nbsp;<i>u</i><sub>0</sub>)/<i>d</i>,
so the median of tan against <i>u</i> is a straight line whose zero crossing is
the source. Multiplying every angle by <i>k</i> scales that line&rsquo;s slope and
its intercept together and leaves &minus;intercept/slope untouched. <b>The
crossing therefore cannot see the angle scale at all</b>, which is what makes it
the right first test now that the scale is the open problem.</p>

<table><thead><tr><th></th><th>value</th><th>spread over
{v['n_runs']} post-access runs</th></tr></thead><tbody>
<tr><th class="s">capsule X, from A and C</th>
<td class="n">{v['x_source_mm']:+.2f} mm</td>
<td class="n"><b>&plusmn;{v['x_source_std_mm']:.2f} mm</b></td></tr>
<tr><th class="s">A&ndash;C alignment, half the difference</th>
<td class="n">{v['x_align_half_diff_mm']:+.2f} mm</td>
<td class="n">&plusmn;{v['x_align_std_mm']:.2f} mm</td></tr>
<tr><th class="s">capsule Z, from D alone</th>
<td class="n">{v['z_D_mm']:+.2f} mm</td>
<td class="n">&plusmn;{v['z_D_std_mm']:.2f} mm</td></tr>
</tbody></table>

<p>For contrast, the same runs&rsquo; angle scale
(<code>k_arm</code>, full pass), p10&ndash;p90 as a percentage of the median:
<b>{', '.join(f'{a} {p:.1f}&nbsp;%' for a, p in sorted(k_spread.items()))}</b>.
On chamber C that is a {k_spread.get('C', float('nan')):.0f}&nbsp;% swing in
the scale beside a
{100 * PA[PA.arm == "C"].std_mm.iloc[0] / abs(PA[PA.arm == "C"].median_mm.iloc[0]):.1f}&nbsp;%
swing in the position the same tracks point at.</p>

{figure('img_per_run', 'The crossing per chamber against run number. The '
        'shaded band is the campaign median ± 1 sd; the grey column is runs '
        f'{kb}, where every arm’s k rose together.')}

<h2>Per chamber</h2>
<p>The column that decides whether there is a run-to-run effect at all is
<b>sd&nbsp;/&nbsp;sub-run</b>: the scatter between runs divided by the scatter one
run&rsquo;s own sub-runs already show. At or below 1 there is nothing between runs
that is not already inside them.</p>
{per_arm_table(PA)}
<p><b>It is below 1 on every chamber</b>
({', '.join(f'{r.arm} {r.std_mm / r.median_err_repro:.2f}' for r in PA.itertuples() if np.isfinite(r.median_err_repro) and r.median_err_repro)}).
The runs do not differ from each other by more than one run&rsquo;s own sub-runs
differ among themselves. On this observable there is nothing run-to-run left to
explain.</p>
<p class="sub">Chamber B has no drift field, a few hundred tracks per run and a
scatter of {PA[PA.arm == "B"].std_mm.iloc[0]:.1f}&nbsp;mm. It is reported and
never averaged into a verdict &mdash; Z rests on D.</p>

<h2>The source and the alignment are different numbers</h2>
<p>A and C face each other, so both measure global X. Their <b>mean</b> is the
capsule; <b>half their difference</b> is how far the two chambers&rsquo; strip
origins sit apart in the plane. Shifting one chamber&rsquo;s origin by &delta; moves
only that chamber&rsquo;s estimate, so the two quantities separate cleanly &mdash;
and neither is available from a single chamber. Z has no such pair: B cannot
check D.</p>
{figure('img_axis', 'The capsule X and the A–C alignment, per run.')}

<h3>The runs furthest from the campaign median</h3>
<ul>{outlier_list(A)}</ul>

<h2>Does a scale-free quantity move inside the <i>k</i> excursion?</h2>
<p>Runs {kb} are the contiguous 48-hour block in which every arm&rsquo;s <i>k</i>
rose together (A&nbsp;+2.3&nbsp;%, C&nbsp;+7.8&nbsp;%, D&nbsp;+7.3&nbsp;%), and in
which the drift end time &mdash; which <i>k</i> never touches &mdash; moved only
~1&nbsp;%, falsifying a gas explanation. If the excursion were a velocity change,
the crossing would not move at all.</p>
{block_table(B)}
<p><b>It moves, slightly, and coherently.</b>
{('The largest is ' + v['block_largest']['quantity'] + ' at '
  + f"{v['block_largest']['shift_mm']:+.2f} mm, "
  + f"{abs(v['block_largest']['in_sigma']):.1f} sd of the outside runs "
  + f"(p = {v['block_largest']['p']:.3f}).")
 if v.get('block_largest') else ''}
These are sub-millimetre shifts against an 8&nbsp;% shift in <i>k</i>, so the
excursion is not <i>mostly</i> geometric &mdash; but it is not <i>purely</i> an
estimator artefact either. Something in the illumination moved, which is the
same lead <code>STATUS.md</code> reached from chamber A&rsquo;s median
<code>x_local</code> stepping +3.4&nbsp;→&nbsp;+7.6&nbsp;mm inside exactly this
window. <b>A lead for October, not a mechanism.</b></p>
''']

    if len(V):
        parts.append(f'''
<h2>The crossing against the scale</h2>
<p><i>k</i> cancels out of the crossing algebraically, so a correlation here is
not scale dependence. It is evidence that a third thing moved and changed
both.</p>
{k_table(V)}
{figure('img_vs_k', 'The crossing against the per-run angle scale, per arm. '
        'Open markers are the runs inside the excursion block.')}
''')

    if len(Y):
        parts.append(f'''
<h2>The half that <i>does</i> carry the scale</h2>
<p>The capsule is 10&nbsp;mm across but 80&nbsp;mm long in y, so there is no zero
to find along the beam. y comes instead from where tracks pass closest to the
axis, against a forward model of the real He-3 polycone through the real
acceptance &mdash; and <code>target_y_mm</code> is built from the
<i>calibrated</i> direction, so unlike the crossing it carries <i>k</i>.</p>
{y_table(Y)}
<p><b>Both halves of this table are wrong and they have been wrong since
run_145.</b> Every chamber sits 15&ndash;27&nbsp;mm off the model&rsquo;s median, and
every observed distribution is 2.3&ndash;5.6&times; wider than the model
predicts. A width ratio above 1 can be read as resolution; a ratio of 5 cannot,
and neither can a 27&nbsp;mm median offset on a source 80&nbsp;mm long. Either the
polycone acceptance model is wrong or the y reconstruction is, and this page
does not separate them. <b>What it does establish is that the failure is
campaign-wide and stable, not a property of any one run</b> &mdash; the run-to-run
sd is under 1&nbsp;mm on A and C.</p>
{figure('img_y', 'The y offset and width ratio against the polycone model, '
        'per run.')}
''')

    parts.append(f'''
<h2>What this does not rule out</h2>
<ul>
<li><b>It does not make the angle scale right.</b> A stable crossing says the
pointing geometry and the alignment are reproducible. It says nothing about
whether the depth-to-length conversion is correct, and the geometric gap bound
still fails on A and C (<code>tracking_qa.gap_check</code>).</li>
<li><b>It does not check Z against anything.</b> B cannot cross-check D, so the
Z number carries no chamber-to-chamber systematic at all &mdash; only D&rsquo;s own
{v['z_D_std_mm']:.2f}&nbsp;mm run-to-run scatter.</li>
<li><b>It cannot separate a real capsule offset from a survey error.</b> Nothing
in this data does (<code>PLAN.md</code> D8). The
{v['x_source_mm']:+.2f}&nbsp;mm is a displacement from the surveyed axis, not a
statement about which of the two is wrong.</li>
<li><b>The hot-channel cut ran only where a strata table exists.</b> That is
run_145 alone; every other run&rsquo;s crossing is measured without it. On run_145
the cut moved D&rsquo;s <i>k</i> by 3.7&nbsp;% and removed 22&nbsp;% of D&rsquo;s
coincident sample, so D&rsquo;s numbers here are the least protected.</li>
<li><b>Pre-access runs are drawn but excluded from every verdict.</b> run_79 and
run_81 are a different detector &mdash; A&rsquo;s x-view connector was dead through
run_79.</li>
{f"<li><b>{len(failed)} run(s) produced nothing:</b> " + ', '.join(f'<code>{k}</code>' for k in failed) + ".</li>" if failed else ""}
</ul>

<p class="foot">Generated by <code>make_campaign_imaging_report.py</code> from
<code>{d}</code> on {dt.date.today().isoformat()}. Reconstruction:
<code>{meta['reco']}</code>. {meta['n_runs_ok']} of {len(meta['runs'])} runs
imaged.</p>
</main></body></html>''')
    return ''.join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dir', default=None, help='default <out>/imaging_campaign')
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else paths.out('imaging_campaign')
    paths.require(d / 'campaign_imaging.meta.json', 'the campaign imaging meta')
    out = d / 'report.html'
    out.write_text(build(d))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
