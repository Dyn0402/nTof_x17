#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_qsum_runaway_report.py -- `<out>/qsum_runaway/report.html`.

Every number in the prose is read from the census and refit tables.

    python -m sept26_prelim_analysis.make_qsum_runaway_report
"""
from __future__ import annotations

import html
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import qsum_runaway as qr  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402


def esc(s) -> str:
    return html.escape(str(s))


def pct(v, d=1) -> str:
    return f'{100 * v:.{d}f}&thinsp;%'


def refit_summary(R: pd.DataFrame) -> pd.DataFrame:
    R = R.copy()
    R['late'] = R.t0_prod > qr.LATE_T0
    R['grp'] = np.select(
        [(R.cls == 'normal') & ~R.late, (R.cls == 'normal') & R.late,
         R.cls == 'space', R.cls == 'time'],
        ['normal, t0 &le; 300', 'normal, t0 &gt; 300', 'runaway: space-censored',
         'runaway: time-censored'], 'runaway: other')
    out = []
    for g, s in R.groupby('grp', sort=False):
        dtan = (s.tan_theta_grd - s.tan_theta_prod).abs()
        dp0 = (s.p0_grd - s.p0_prod).abs()
        out.append(dict(
            grp=g, n=len(s), late=s.late.mean(),
            dtan_p50=dtan.median(), dtan_gt05=(dtan > 0.05).mean(),
            dp0_p50=dp0.median(), dp0_gt1=(dp0 > 1).mean(),
            dt0_moved=((s.t0_grd - s.t0_prod).abs() > 5).mean(),
            flat_prod=(s.tan_theta_prod.abs() < qr.FLAT_TAN).mean(),
            flat_grd=(s.tan_theta_grd.abs() < qr.FLAT_TAN).mean(),
            dchi2dof=((s.chi2_grd - s.chi2_prod) / s.dof_prod).median(),
            qobs_ratio=(s.q_sum_grd / s.q_obs.where(s.q_obs > 0)).median(),
            qgrd_p99=s.q_sum_grd.quantile(0.99),
            plaus_prod=s.plaus_prod.mean(), plaus_grd=s.plaus_grd.mean()))
    order = ['normal, t0 &le; 300', 'normal, t0 &gt; 300', 'runaway: space-censored',
             'runaway: time-censored', 'runaway: other']
    return pd.DataFrame(out).set_index('grp').reindex([o for o in order if o in
                                                       set(R.grp)])


def table(df: pd.DataFrame, cols, head) -> str:
    th = ''.join(f'<th>{h}</th>' for h in head)
    rows = []
    for idx, r in df.iterrows():
        tds = ''.join(f'<td>{fmt(r[c])}</td>' for c, fmt in cols)
        rows.append(f'<tr><th>{idx}</th>{tds}</tr>')
    return f'<table><thead><tr><th></th>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


FIGS = [
    ('runaway_vs_t0.png', 'The runaway rate against fitted t0',
     'Fraction of gated tracks, whole campaign, with q_sum above 10<sup>6</sup> ADC, '
     'per plane, against the fitted t0 (top), and where the tracks are (bottom). The '
     'floor below 300&nbsp;ns is the space-censored class; the step above it is the '
     'time-censored class.'),
    ('example_time.png', 'A time-censored runaway',
     'Left: data, production model and guarded model on the same strip &times; sample '
     'window. Right: the fitted charge per depth bin against that bin&rsquo;s arrival '
     'time, with the column&rsquo;s peak response (dotted). The charge sits in bins whose '
     'response is 10<sup>-6</sup> or less inside the window; dropping them moves the '
     'track.'),
    ('example_space.png', 'A space-censored runaway',
     'Same layout. Here the runaway bin arrives inside the time window, but the fitted '
     'slope carries its centre past the edge of the strip window. <b>Note what the '
     'data are:</b> no track, but a coherent bipolar oscillation (&plusmn;200&nbsp;ADC, '
     '~350&nbsp;ns period) on every strip at once, which the model hardly describes. '
     'The picker chose the most extreme case, so this is not a typical member. It does '
     'show that the space class contains noise windows as well as steep tracks.'),
    ('geometry_shift.png', 'What dropping the unobservable bins does to the geometry',
     'Cumulative distributions of the change in tan&theta; and p0 between the production '
     'fit and the guarded fit, per class.'),
    ('flat_tracks.png', 'The flat tracks at late t0',
     'The |tan&theta;| distribution of late (t0 &gt; 300&nbsp;ns) tracks. Production '
     'time-censored runaways pile up at zero slope; without the unobservable bins they '
     'follow their normal neighbours.'),
    ('q_obs.png', 'A post-hoc charge',
     'The production charge restricted to observable bins (q_obs) against the guarded '
     'refit&rsquo;s q_sum.'),
]


def build() -> str:
    od = qr.out_dir()
    cm = json.loads((od / 'census.meta.json').read_text())
    rm = json.loads((od / 'refit.meta.json').read_text())
    pa = pd.read_csv(od / 'census_per_arm.csv').set_index('arm')
    R = pd.read_parquet(od / 'refit.parquet')
    S = refit_summary(R)
    run_ = R[R.cls != 'normal']
    share_time = (run_.cls == 'time').mean()
    share_space = (run_.cls == 'space').mean()
    t_unobs = run_.q_unobs_share.median()
    tm = S.loc['runaway: time-censored'] if 'runaway: time-censored' in S.index else None
    sp = S.loc['runaway: space-censored'] if 'runaway: space-censored' in S.index else None
    nl = S.loc['normal, t0 &le; 300']
    late_tc = R[(R.cls == 'time') & (R.t0_prod > qr.LATE_T0)]
    late_ok = R[(R.cls == 'normal') & (R.t0_prod > qr.LATE_T0)]
    med = lambda s: float(s.abs().median())  # noqa: E731

    verdict = f"""
<p><b>The large-q_sum tracks are not discharges or a units error. They come
from one defect in the fit: the NNLS charge profile is unregularised, so a depth
bin that the data window cannot see takes an arbitrarily large charge at no
&chi;<sup>2</sup> cost.</b> In the median runaway fit, all but a fraction
{max(1 - t_unobs, 1e-16):.0e} of q_sum sits in such bins. A bin becomes unobservable in two ways, and the two are
different populations with different consequences.</p>
<ul>
<li><b>Time-censored ({pct(share_time, 0)} of runaways in the refit sample).</b>
The window is 20&nbsp;&times;&nbsp;60&nbsp;ns, so the last sample is at 1140&nbsp;ns, while the depth grid
always spans 1080&nbsp;ns after t0. For a track that arrives late, the deep bins
peak after the readout has stopped. <b>These fits are wrong in geometry, not just
in charge:</b> without the unobservable bins, tan&theta; moves by a median
{tm['dtan_p50']:.2f}, p0 by {tm['dp0_p50']:.1f}&nbsp;mm and t0 in {pct(tm['dt0_moved'], 0)}
of them. {pct(tm['flat_prod'], 0)} of them are fitted flat (|tan|&nbsp;&lt;&nbsp;{qr.FLAT_TAN}) against
{pct(tm['flat_grd'], 0)} once the bins are gone.</li>
<li><b>Space-censored ({pct(share_space, 0)}).</b> The fitted slope carries the deep
bins' centres a few mm past the edge of the strip window. <b>The typical fit's
geometry is untouched</b> (median |&Delta;tan| {sp['dtan_p50']:.3f},
|&Delta;p0| {sp['dp0_p50']:.3f}&nbsp;mm), <b>but a tail is not:</b>
{pct(sp['dtan_gt05'], 0)} move by more than 0.05 in tan. The production charge
summed over the observable bins already matches the guarded refit (median ratio
{sp['qobs_ratio']:.2f}).</li>
<li><b>The q_sum threshold is not the class boundary; t0 is.</b> Late tracks that
never crossed 10<sup>6</sup> move almost as often
({pct(S.loc['normal, t0 &gt; 300', 'dtan_gt05'], 0)} beyond 0.05 in tan, against
{pct(tm['dtan_gt05'], 0)} for the time-censored runaways), because their deep
bins are just as unobservable.</li>
</ul>
<p><b>Campaign scale.</b> {pct(cm['big_any'])} of the {cm['n_gated']:,} gated
tracks have q_sum&nbsp;&gt;&nbsp;10<sup>6</sup> in at least one plane:
{pct(cm['late_big'])} of all gated tracks in the late, geometry-wrong class (t0 &gt;
{qr.LATE_T0:.0f}&nbsp;ns in the runaway plane) and {pct(cm['early_big'])} in
the early, mostly charge-only class. Among the {cm['coinc_n']:,} tracks coincident with
their own arm's scintillators, the corresponding numbers are
<b>{pct(cm['coinc_late_big'])}</b> and {pct(cm['coinc_early_big'])}.</p>
"""
    tiles = f"""<div class="tiles">
<div class="tile"><div class="tile-v">{pct(cm['big_any'], 0)}</div><div class="tile-k">gated tracks with a runaway plane</div></div>
<div class="tile"><div class="tile-v">{pct(cm['late_big'], 0)}</div><div class="tile-k">late class: geometry wrong</div></div>
<div class="tile"><div class="tile-v">{pct(cm['early_big'], 0)}</div><div class="tile-k">early class: mostly charge only</div></div>
<div class="tile"><div class="tile-v">{pct(cm['coinc_late_big'], 0)}</div><div class="tile-k">of scint-coincident tracks in the late class</div></div>
</div>"""

    arm_tbl = table(
        pa, [('n', lambda v: f'{int(v):,}'), ('big_any', lambda v: pct(v)),
             ('late_big', lambda v: pct(v)), ('early_big', lambda v: pct(v)),
             ('coinc_late_big', lambda v: pct(v)), ('coinc_early_big', lambda v: pct(v))],
        ['gated tracks', 'runaway (x or y)', 'late class', 'early class',
         'late, coincident', 'early, coincident'])
    f3 = lambda v: f'{v:.3f}'  # noqa: E731
    fit_tbl = table(
        S, [('n', lambda v: f'{int(v)}'), ('dtan_p50', f3), ('dtan_gt05', pct),
            ('dp0_p50', lambda v: f'{v:.2f}'), ('dp0_gt1', pct), ('dt0_moved', pct),
            ('flat_prod', pct), ('flat_grd', pct), ('dchi2dof', lambda v: f'{v:+.3f}'),
            ('plaus_prod', pct), ('plaus_grd', pct)],
        ['plane fits', 'med |&Delta;tan|', '|&Delta;tan| &gt; 0.05', 'med |&Delta;p0| mm',
         '|&Delta;p0| &gt; 1 mm', 't0 moved', 'flat, prod', 'flat, guarded',
         'med &Delta;&chi;<sup>2</sup>/dof', 'plausible, prod', 'plausible, guarded'])

    figs = '\n'.join(
        f'<figure><img src="figures/{n}" alt="{esc(t)}"><figcaption><b>{t}.</b> {c}'
        f'</figcaption></figure>'
        for n, t, c in FIGS if (od / 'figures' / n).exists())

    body = f"""<main>
<h1>The large-q_sum tracks: what they are</h1>
<p class="sub">Campaign census: {cm['n_gated']:,} gated tracks (stage3_fullpass).
Refit: {rm['n_fits']:,} plane fits on {esc(rm['run'])}/{esc(rm['sub_run'])} tag
{esc(rm['tag'])}, {rm['n_per_arm']} events per arm, arms {', '.join(rm['arms'])}.
Guard threshold {rm['rel']:g} of the largest column. Generated {esc(rm['generated'])}.</p>

<div class="verdict">{verdict}</div>
{tiles}

<h2>The mechanism</h2>
<p><code>wft.model.chi2_plane</code> solves for the charge in each of 18 depth
bins (60&nbsp;ns each) by non-negative least squares on the noise-weighted
waveforms. Nothing bounds a bin's charge except the data. A bin whose design-matrix
column is ~0 inside the window can therefore absorb any positive residual at a
negligible price: a column of 10<sup>-12</sup> reproduces a 1-ADC wiggle in the last
sample with q&nbsp;=&nbsp;10<sup>12</sup>. That is the whole tail, and why its
&chi;<sup>2</sup> looks normal. The rest of the fit sees only the product q&middot;column,
which stays finite.</p>
<p>The test is a <b>guarded fit</b>. It is identical to production, except that each NNLS solve
drops any column whose weighted norm is below {rm['rel']:g} of the largest. On normal
fits with t0&nbsp;&le;&nbsp;300&nbsp;ns it changes nothing (median |&Delta;tan|
{nl['dtan_p50']:.1g}, {pct(nl['dtan_gt05'])} beyond 0.05), so it is a clean probe.</p>

<h2>Per chamber, whole campaign</h2>
{arm_tbl}
<p class="sub">Fractions of gated tracks. &ldquo;Late&rdquo; = a plane with
q_sum &gt; 10<sup>6</sup> and t0 &gt; {qr.LATE_T0:.0f}&nbsp;ns; &ldquo;early&rdquo; = any other runaway.
&ldquo;Coincident&rdquo; = <code>coinc_this_arm</code>. The rate is the same
before and after the 27 July access
({', '.join(f'{esc(k)} {pct(v)}' for k, v in cm['by_condition'].items())}).</p>

<h2>Production against guarded, per class</h2>
{fit_tbl}
<p class="sub">One sample tag of run_145, all four arms. &ldquo;t0 moved&rdquo; =
|&Delta;t0| &gt; 5&nbsp;ns. Positive &Delta;&chi;<sup>2</sup>/dof means the
guarded fit describes the waveform worse.</p>
<p><b>The late normal tracks also move</b>
({pct(S.loc['normal, t0 &gt; 300', 'dtan_gt05']) if 'normal, t0 &gt; 300' in S.index else 'n/a'}
beyond |&Delta;tan|&nbsp;=&nbsp;0.05). Their deep bins are just as unobservable. They
were simply not filled far enough to cross the 10<sup>6</sup> threshold. <b>The threshold
is a symptom marker, not the class boundary. The class is &ldquo;the column runs
past the window&rdquo;</b>, which every track with t0 above ~0 has to some degree.</p>
<p>Late tracks, |tan&theta;| median: production runaways {med(late_tc.tan_theta_prod):.3f},
guarded {med(late_tc.tan_theta_grd):.3f}; their normal neighbours
{med(late_ok.tan_theta_prod):.3f} (production).</p>

<h2>Figures</h2>
{figs}

<h2>What this does not establish</h2>
<ul>
<li><b>The guarded fit is a probe, not the right answer.</b> It is worse on
&chi;<sup>2</sup> for the time-censored class (median
{tm['dchi2dof']:+.2f}/dof), because the truncated tail does carry real signal at the
end of the window. Dropping those bins removes the runaway, and with it some real
information. Which geometry is correct needs an external reference, such as the
scintillator pointing in <code>det_a_scint</code>, and that was not used here.</li>
<li><b>One tag of one run.</b> The mechanism is structural, and the campaign census
shows the same t0 dependence everywhere, but the per-class geometry shifts are measured
on run_145 only.</li>
<li><b>What the late tracks are physically</b> is not settled. They are less often
scintillator-coincident than the in-time ones, which suggests out-of-time
particles. Their geometry is the part this note shows to be unreliable.</li>
<li><b>How much of the space-censored class is coherent noise</b> rather than
steep tracks is not measured. The example shown is a ringing window. A
goodness-of-fit fraction does not separate the classes, because &chi;<sup>2</sup>
is dominated by noise across the whole window for every class. The space class
reaches the gated table less often than normal fits do.</li>
<li><b>Threshold {rm['rel']:g}</b> is a choice. Columns at 1&ndash;few&nbsp;% still
let the guarded q_sum reach {tm['qgrd_p99']:.2g} (99th percentile, time class).</li>
</ul>

<h2>What to do with it</h2>
<ul>
<li><b>Now, with no re-pass:</b> treat q_sum and q_total as unusable above
10<sup>6</sup>. Flag <code>t0&nbsp;&gt;&nbsp;{qr.LATE_T0:.0f}</code> tracks with a runaway plane as
geometry-unreliable before any opening-angle or pointing study. That is
{pct(cm['coinc_late_big'])} of the coincident sample.</li>
<li><b>For the October re-pass:</b> stop offering the NNLS bins it cannot see.
The options are to truncate the depth grid per fit to bins whose pulse peaks inside
the window, or to replace the free NNLS with one that carries a smoothness or
continuity prior across depth. Then store <code>q_obs</code> alongside q_sum. This
belongs on <code>OCTOBER_2026.md</code>, beside the depth-grid edge
(<code>q_uend</code> railing) item, because it is the same edge.</li>
</ul>
</main>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Large q_sum tracks</title>{HEAD}</head><body>{body}</body></html>')


def main() -> int:
    od = qr.out_dir()
    p = od / 'report.html'
    p.write_text(build())
    print(f'-> {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
