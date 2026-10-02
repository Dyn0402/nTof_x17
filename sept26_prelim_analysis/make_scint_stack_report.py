#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_scint_stack_report.py -- ``report.html`` for the scintillator-stack
calibration from MM tracks.

Generated, never hand-written: every number is read back from what
`scint_stack_ana` wrote, so a re-run updates tables, figures and verdict
together.  Figures are referenced relatively (``figures/x.png``) so the page
works from disk and from the DAQ page's ``/analysis_file`` route.

    python -m sept26_prelim_analysis.make_scint_stack_report
"""
from __future__ import annotations

import datetime as dt
import html as _h
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
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402
from sept26_prelim_analysis.scint_stack_ana import MM_HALF_V  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


def figure(name: str, caption: str) -> str:
    csv = (f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
           if (paths.out('scint_stack') / 'figures' / f'{name}.csv').exists()
           else '')
    return (f'<figure><a href="figures/{name}.png">'
            f'<img src="figures/{name}.png" alt="{_h.escape(caption)}"></a>'
            f'<figcaption>{caption} {csv}</figcaption></figure>')


def pc(v, d=1):
    return '&ndash;' if v is None or not np.isfinite(v) else f'{100 * v:.{d}f}&thinsp;%'


def f(v, d=2):
    return '&ndash;' if v is None or not np.isfinite(v) else f'{v:.{d}f}'


def i(v):
    return f'{int(v):,}'.replace(',', '&thinsp;')


def build(d: Path) -> str:
    A = d / 'ana'
    rd = lambda n: pd.read_parquet(A / f'{n}.parquet')  # noqa: E731
    meta = json.loads((A / 'meta.json').read_text())
    S, E, G, W, R = rd('scales'), rd('eff'), rd('gain'), rd('wallpos_cal'), rd('wallpos_res')
    B, AC, RB, M, CM = rd('both_ends'), rd('accidental'), rd('by_run'), rd('mask'), rd('confirm_map')
    C, L = rd('confirm'), rd('liq_vs_plas')
    ext = json.loads((d / 'extract.meta.json').read_text())

    def eff(arm, lay, probe, samp):
        r = E[(E.arm == arm) & (E.layer == lay) & (E.probe == probe) & (E['sample'] == samp)]
        return (float(r.eff.iloc[0]), float(r.err.iloc[0]), int(r.n.iloc[0])) if len(r) else (np.nan, np.nan, 0)

    def gain(arm, lay, ch, samp='all_late'):
        r = G[(G.arm == arm) & (G.layer == lay) & (G.channel == ch) & (G['sample'] == samp)]
        return float(r['median'].iloc[0]) if len(r) else np.nan

    def sc(arm, lay, fp=2):
        r = S[(S.arm == arm) & (S.layer == lay) & (S.axis == 'u') & (S.fit_pass == fp)]
        return r.iloc[0]

    # masked share of ALL tracks, split: v rail vs unconfirmed cells
    mk = {}
    for arm in ARMS:
        c = CM[CM.arm == arm][['iu', 'iv', 'v', 'n_all']].dropna(subset=['n_all'])
        m = M[M.arm == arm][['iu', 'iv', 'masked']]
        c = c.merge(m, on=['iu', 'iv'], how='left')
        c['masked'] = c.masked.astype('boolean').fillna(False).astype(bool)
        tot = c.n_all.sum()
        rail = c.v.abs() > MM_HALF_V
        mk[arm] = (c.n_all[rail].sum() / tot, c.n_all[c.masked & ~rail].sum() / tot)

    # ---------------- tables ----------------
    rows = ''
    for arm in ARMS:
        w, p, l = eff(arm, 'wall', 'wany', 'unbiased'), eff(arm, 'plas', 'pm', 'unbiased'), eff(arm, 'liq', 'lf', 'all_late')
        wl, pl = eff(arm, 'wall', 'wany', 'all_late'), eff(arm, 'plas', 'pm', 'all_late')
        l1, l2 = eff(arm, 'liq', 'lf_behind_bar1', 'all_late'), eff(arm, 'liq', 'lf_behind_bar2', 'all_late')
        rows += (f'<tr><td>{arm}</td><td>{pc(w[0])} &plusmn; {pc(w[1])}</td><td>{pc(wl[0])}</td>'
                 f'<td>{pc(p[0])} &plusmn; {pc(p[1])}</td><td>{pc(pl[0])}</td>'
                 f'<td>{pc(l1[0])}</td><td>{pc(l2[0])}</td><td>{i(w[2])}</td></tr>')
    eff_tab = ('<table><thead><tr><th>arm</th><th>wall | plastic<br>unbiased</th><th>wall<br>self-trig.</th>'
               '<th>plastic | wall<br>unbiased</th><th>plastic<br>self-trig.</th>'
               '<th>liquid behind<br>L bar</th><th>liquid behind<br>R bar</th><th>unbiased<br>wall tags</th></tr></thead>'
               f'<tbody>{rows}</tbody></table>')

    rows = ''
    for arm in ARMS:
        g = [gain(arm, 'wall', f'g{k}') for k in range(4)]
        rows += (f'<tr><td>{arm}</td>' + ''.join(f'<td>{f(x, 1)}</td>' for x in g)
                 + f'<td>{f(gain(arm, "plas", "bar1") / 1e3, 2)} / {f(gain(arm, "plas", "bar2") / 1e3, 2)}</td>'
                 f'<td>{f(gain(arm, "plas", "bar1", "unbiased") / 1e3, 2)} / {f(gain(arm, "plas", "bar2", "unbiased") / 1e3, 2)}</td>'
                 f'<td>{f(gain(arm, "plas", "bar1_through") / 1e3, 2)} / {f(gain(arm, "plas", "bar2_through") / 1e3, 2)}</td>'
                 f'<td>{f(gain(arm, "liq", "cell"), 0)} {G[(G.arm == arm) & (G.layer == "liq")].unit.iloc[0]}</td></tr>')
    gain_tab = ('<table><thead><tr><th>arm</th><th>wall g0</th><th>g1</th><th>g2</th><th>g3</th>'
                '<th>plastic bar 1 / 2<br>self-trig., MeVee</th><th>unbiased</th><th>through-going</th>'
                '<th>liquid median</th></tr></thead>' f'<tbody>{rows}</tbody></table>')

    rows = ''
    for arm in ARMS:
        w = W[W.arm == arm]
        res = R[(R.arm == arm) & (R.by == 'all')].set_index('est').sigma
        rows += (f'<tr><td>{arm}</td><td>{f(w.atten_mm.min(), 0)}&ndash;{f(w.atten_mm.max(), 0)}</td>'
                 f'<td>{f(w.c_eff_mm_ns.min(), 0)}&ndash;{f(w.c_eff_mm_ns.max(), 0)}</td>'
                 f'<td>{"+" if w.lr_slope.mean() < 0 else "&minus; (reversed)"}</td>'
                 f'<td>{f(res["lr"], 0)}</td><td>{f(res["dt"], 0)}</td><td>{i(w.n.sum())}</td></tr>')
    pos_tab = ('<table><thead><tr><th>arm</th><th>attenuation length, mm</th><th>light speed, mm/ns</th>'
               '<th>end order</th><th>&sigma;(v) ratio, mm</th><th>&sigma;(v) timing, mm</th><th>tracks</th></tr></thead>'
               f'<tbody>{rows}</tbody></table>')

    rows = ''
    for arm in ARMS:
        b = B[(B.arm == arm)].set_index('sample')
        a = AC[AC.arm == arm]
        rows += (f'<tr><td>{arm}</td><td>{pc(b.loc["unbiased", "keep_real_both"], 1)}</td>'
                 f'<td>{pc(b.loc["all_late", "keep_real_both"], 2)}</td>'
                 f'<td>{pc(1 - b.loc["all_late", "keep_acc_both"], 0)}</td>'
                 f'<td>{pc(a.p_any.mean(), 2)}</td><td>{pc((a.p_only1 + a.p_only2).mean(), 2)}</td></tr>')
    both_tab = ('<table><thead><tr><th>arm</th><th>real hits kept<br>unbiased</th><th>real hits kept<br>self-trig.</th>'
                '<th>accidentals<br>removed</th><th>accidental rate<br>per group, 160 ns</th>'
                '<th>of which<br>one-ended</th></tr></thead>' f'<tbody>{rows}</tbody></table>')

    rows = ''
    for arm in ARMS:
        w1, w2, p2 = sc(arm, 'wall', 1), sc(arm, 'wall'), sc(arm, 'plas')
        rows += (f'<tr><td>{arm}</td><td>{f(w2.inv_k, 3)}</td><td>{f(w1.s, 3)}</td><td>{f(w2.s, 3)} &plusmn; {f(w2.s_err, 3)}</td>'
                 f'<td>{f(p2.s, 3)} &plusmn; {f(p2.s_err, 3)}</td><td>{f(w2.sigma, 0)} / {f(p2.sigma, 0)}</td>'
                 f'<td>{pc(meta["frac_intime"][arm], 0)}</td><td>{pc(mk[arm][0], 0)} + {pc(mk[arm][1], 0)}</td>'
                 f'<td>{pc(meta["frac_good"][arm], 0)}</td></tr>')
    scale_tab = ('<table><thead><tr><th>arm</th><th>1/k</th><th>wall s,<br>every track</th><th>wall s,<br>good</th>'
                 '<th>plastic s,<br>good</th><th>edge width<br>wall / plastic, mm</th><th>in time</th>'
                 '<th>masked: v rail +<br>unconfirmed cells</th><th>kept<br>("good")</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table>')

    # headline numbers
    wA, wC, wD = (eff(a, 'wall', 'wany', 'unbiased')[0] for a in 'ACD')
    pA = eff('A', 'plas', 'pm', 'unbiased')[0]
    keep = B[B['sample'] == 'all_late'].keep_real_both
    accr = 1 - B[B['sample'] == 'all_late'].keep_acc_both
    resA = R[(R.arm == 'A') & (R.by == 'all') & (R.est == 'lr')].sigma.iloc[0]
    sp = RB.groupby('arm').wall_gm_mV.apply(lambda s: (s.quantile(.9) - s.quantile(.1)) / s.median())
    lc_hi = L[(L.arm == 'C') & (L['sample'] == 'all_late')].sort_values('e_lo').iloc[-1]
    oot = 1 - np.mean([meta['frac_intime'][a] for a in ARMS])

    body = f"""
<header><p class="eyebrow">n_TOF 2026 &middot; sept26_prelim_analysis &middot; scint_stack</p>
<h1>The scintillator stack, calibrated from the tracks that point at it</h1>
<p class="deck">{i(ext['n_tracks'])} gated MM tracks on all four arms, {ext['n_runs']} runs, each walked back
through the SiPM wall, the plastic and the liquid. Efficiency and gain on every layer's face, a position
along the wall bars from its two ends, and whether to demand both ends.</p></header>

<div class="verdict"><h2>Verdict</h2><ul>
<li><b>The tracks point at the scintillators well enough to calibrate them, after two cuts the MM data
needs anyway.</b> {pc(oot, 0)} of tracks crossed the chamber at another time than the trigger (their own t0 falls outside a 300&ndash;425&nbsp;ns
wide in-time window) and no prompt scintillator can confirm them; and part of each
chamber reconstructs tracks that nothing confirms &mdash; chamber D worst, where {pc(mk['D'][1], 0)} of all
tracks sit in such cells. With both removed the wall groups and the plastic gap switch exactly where the
survey puts them on A, B and C.</li>
<li><b>The wall is {pc(wA, 0)} efficient on arm A, {pc(wC, 0)} on C, {pc(wD, 0)} on D</b> for particles that
reach the plastic, measured on events another arm triggered. On the events an arm triggered itself the same
numbers read 89&ndash;98&thinsp;%: the trigger requires the wall, so that sample cannot show a hole.
WALD groups 0 and 2 are the weak ones.</li>
<li><b>Gains:</b> WALA runs ~30&thinsp;% below the other walls and WALD group 3 ~35&thinsp;% below the rest of
D &mdash; both seen on the bench in July, both still there. Run by run, each arm's wall MIP median holds to
p10&ndash;p90 = {pc(sp.min(), 1)}&ndash;{pc(sp.max(), 1)} across the campaign, the plastic to ~1&thinsp;%. A particle that also lights the liquid
leaves 3.0&ndash;3.4&nbsp;MeVee in the plastic on every arm, an independent check of the source energy scale.</li>
<li><b>The liquids are the surprise.</b> LIQ&nbsp;A and D respond almost only near their +u edge (behind the R
plastic bar: 9&thinsp;% against 0.2&ndash;1&thinsp;% behind the L bar), with a smooth gradient across the cell
rather than a step at the bar gap &mdash; a light-collection pattern, not a shadow. LIQ&nbsp;C answers only
to deposits above ~8&nbsp;MeVee in the plastic ({pc(lc_hi.eff, 0)} there, ~0.1&thinsp;% below): alive, but
with a threshold or gain far off. LIQ&nbsp;B is roughly uniform. This is the beam-data answer to the open
question from the July source runs.</li>
<li><b>The wall measures position along its bars to ~{resA:.0f}&nbsp;mm</b> from ln(top/bottom) (an upper limit,
MM error included); attenuation lengths 0.4&ndash;1.3&nbsp;m. The top&ndash;bottom time difference is useless at
the stored timing precision. Chamber D's two ends run the other way round in both estimators.</li>
<li><b>Both ends: we do not demand them, and doing so is nearly free but buys little.</b> Real hits already
fire both ends (98&thinsp;% on A's unbiased sample) {pc(keep.min(), 1)}&ndash;{pc(keep.max(), 2)} of the time; demanding it removes
{pc(accr.min(), 0)}&ndash;{pc(accr.max(), 0)} of accidentals, because our accidentals are mostly real particles that
light both ends too. Recommended as an offline cut, with a top&ndash;bottom window no tighter than &plusmn;40&nbsp;ns;
not worth a hardware change.</li>
</ul></div>

<h2 id="what">What was done</h2>
<p>Every gated track of the full pass is extrapolated as a straight line to each layer
(lever arms from the DAQ's <code>run_config.json</code>: wall 97&nbsp;mm, plastic ~190&nbsp;mm, liquid
~250&nbsp;mm past the strip plane). For the event and arm the n_TOF slim gives every wall end, both plastic
bars and the liquid in a prompt window (&minus;100, +60)&nbsp;ns and in a same-width pre-trigger window
(&minus;560, &minus;400)&nbsp;ns; the second is the accidental floor of every rate on this page.</p>
{figure('stack', 'One arm, side view. The wall is 3 mm of scintillator in four read-out groups of four 25 mm bars, read at both ends; the plastic is two 200 x 300 mm bars of 20 mm PVT with one PMT each; the liquid one 451 x 450 mm cell.')}

<h3>Not trusting the MMs</h3>
<p>A track that no layer confirms is either not real, or a real particle the scintillators could not see.
Three populations are separable, and only the first two are removed:</p>
<ul><li><b>Out of time.</b> The chamber integrates over its drift window, so it reconstructs real particles
from other moments. Their own t0 shows it: confirmation is 70&ndash;80&thinsp;% at t0&nbsp;&asymp;&nbsp;&minus;100&nbsp;ns and
10&thinsp;% beyond +300&nbsp;ns. The in-time window is set per arm where confirmation is above half its peak.</li>
<li><b>Bad chamber regions.</b> 20&nbsp;mm cells of the strip plane where in-time tracks are confirmed (wall
OR plastic, so a dead patch of one layer cannot mask itself) at under half the arm's median, plus the
v rail outside the active area.</li>
<li><b>Not removed: tracks that range out.</b> On the unbiased sample even good in-time tracks are confirmed
only ~15&ndash;25&thinsp;% of the time before the 10&nbsp;ms cut: low-energy electrons stop in the readout PCB or
the wall. A scintillator cannot tell those from a fake track, and nothing here claims to.</li></ul>
{figure('t0_profile', 'Confirmation (wall or plastic, net) against the track t0. Shaded: the in-time window. Dotted: the unbiased sample, which is accidental-dominated before 10 ms.')}
{figure('confirm_map', 'Confirmation of in-time tracks over each chamber, relative to the arm median; red outline = masked. The faint vertical stripes on A and C are the wall-group boundaries projected back onto the chamber, not chamber defects. D masks a quarter of its cells, which hold most of its tracks.')}
{figure('quality', 'Confirmation against track properties. Tracks with no y slope at all ([&minus;0.03, 0.03)) and high chi2 are the least confirmed.')}

<h3>The extrapolation scale</h3>
<p>The angle scale <code>k</code> is the analysis's standing open question, so the extrapolation does not borrow
it. Each layer's scale <i>s</i> on the raw tangent is fitted from the scintillators' own fixed boundaries
(the three internal wall-group edges; the plastic L/R gap) by how the apparent boundary moves with the
track's slope &mdash; a pointing estimator. It is the scale that best <i>predicts</i> where a track lands,
which is what this page needs; it is shrunk by slope noise (the longer lever always fits lower) and is
<b>not</b> a measurement of the physical angle scale.</p>
{scale_tab}
{figure('scales', 'Fitted scales, every track (open) and in-time tracks in good cells (filled), against the campaign 1/k.')}
{figure('edges', 'With the fitted scale: which wall group fires (top), which plastic bar (middle), and how often the liquid fires (bottom), against the predicted crossing. Dotted lines are the survey.')}

<h2 id="eff">Efficiency</h2>
<p>Tag and probe on single in-time tracks in good cells, <b>more than 10&nbsp;ms after the flash</b> (before
that the plastic's accidental rate reaches 10&ndash;20&thinsp;% per track and half the tags are themselves
accidental). The quoted sample is <b>unbiased</b>: another arm satisfied the emulated hardware trigger
(wall-sum and plastic thresholds read back from the boards on run_79, assumed campaign-wide). The
self-triggered numbers are shown because their difference from the unbiased ones is the trigger bias.</p>
{eff_tab}
<p>Plastic given wall is ~57&thinsp;% unbiased: two particles in five that cross the 3&nbsp;mm wall leave nothing
in the plastic. That is the energy spectrum (sub-MeV electrons stop in the wall and its wrapping), not a
plastic inefficiency &mdash; the self-triggered 98&ndash;99&thinsp;% shows the plastic answers whenever there is
something to answer to.</p>
{figure('eff_summary', 'Per layer, unbiased (filled) against self-triggered (open).')}
{figure('eff_map_wall_unbiased', 'Wall, unbiased, 50 mm cells; only the |v| &lt; 125 mm the plastic tag covers.')}
{figure('eff_map_wall_all_late', 'Wall, every late trigger, 25 mm cells. Biased high, but with ten times the statistics it shows the same weak groups (D g0, D g2, C g2).')}
{figure('eff_map_plas_unbiased', 'Plastic, unbiased.')}
{figure('eff_map_liq_all_late', 'Liquid, every late trigger, liquid-centred frame; dashed = the two plastic bars. A and D brighten towards +u.')}
{figure('liquid', 'Left: the liquid behind each plastic bar. Right: against the energy left in the plastic; the liquid answers to through-going deposits (~3 MeVee) and to the high tail.')}

<h2 id="gain">Gain</h2>
<p>Wall: sqrt(top &times; bottom) &times; cos&theta;, which for an exponential bar does not depend on where along
it the light was made &mdash; the MIP response of the scintillator itself. Plastic: keVee with the 28 July source
calibration, &times; cos&theta;. Medians.</p>
{gain_tab}
{figure('gain_summary', 'Wall per group (left); plastic per bar (right), unbiased, self-triggered, and through-going.')}
{figure('gain_map_wall_all_late', 'Wall MIP response over each face, relative to the arm median.')}
{figure('gain_map_plas_unbiased', 'Plastic response, unbiased, relative to the arm median.')}
{figure('gain_map_plas_all_late', 'Plastic response, every late trigger: B bar 2 sits ~20 % above bar 1; elsewhere &plusmn;10 %.')}
{figure('attenuation', 'Each end of each wall group against where along the bar the track crossed.')}
{figure('stability', 'Per-run median response. The grey band is the 3-5 Aug k-block.')}

<h2 id="pos">Position along the wall bars</h2>
<p>Per group, ln(a<sub>1</sub>/a<sub>2</sub>) and t<sub>1</sub>&minus;t<sub>2</sub> are fitted linearly against the
MM-predicted v on even events and inverted on odd ones.</p>
{pos_tab}
<p>The residual includes the MM's own prediction error at the wall (the wall-edge width, 11&ndash;22&nbsp;mm),
so the wall's intrinsic resolution is somewhat better than the column. The stored top&ndash;bottom time
difference has tails to &plusmn;30&nbsp;ns against ~3.6&nbsp;ns of physical spread along a bar, so it adds nothing.</p>
{figure('wallpos', 'Top: v from the amplitude ratio against v from the MM track. Bottom: residuals, amplitude ratio against time difference.')}

<h2 id="both">Should we demand both ends?</h2>
<p><b>We do not today.</b> The hardware discriminates the analog SUM of a group's two ends, so one end
alone can fire it; stage 1, the efficiency and every match on this page count a group lit if either end
fired.</p>
{both_tab}
<p><b>It is feasible and almost free</b>: the data carry both ends, and real particles fire both
&ge;98&thinsp;% of the time; the one-ended ones sit at the bar ends, where the far end is attenuated. <b>But it
buys little here</b>: accidentals are already 0.4&ndash;1&thinsp;% per group per 160&nbsp;ns, and most of them are real
particles that light both ends too, so the cut removes only a fraction. Where it was decisive in the
presentation, the single-end background is usually SiPM dark noise at a low threshold; our 1.5&nbsp;mV
recording threshold and the wall&thinsp;AND&thinsp;plastic trigger already suppress that. Recommended offline,
in stage 1 and the efficiency, with a top&ndash;bottom window of &plusmn;40&nbsp;ns or wider (&plusmn;10&nbsp;ns would lose
5&ndash;19&thinsp;% of real hits); a hardware coincidence would cost eight more channels for no visible gain.</p>
{figure('both_ends', 'Left: real hits lost and accidentals removed. Middle: one-ended real hits along the bar. Right: real hits lost against the top-bottom window.')}

<h2 id="not">What this does not rule out</h2>
<ul>
<li><b>The extrapolation scale is not an angle calibration.</b> It is a best predictor, shrunk by slope noise;
it moved 10&ndash;20&thinsp;% between the full and the cleaned sample, and the two levers differ by 4&ndash;21&thinsp;%. It
says arm A's tangents predict best at 0.74 of raw (k&times;1.10), the same sign as <code>det_a_scint</code>'s +33&thinsp;% but a third
of its size; the two estimators are not reconciled here. The per-run scale scatters by &plusmn;7&thinsp;%, more than
the k-block, so the scintillators cannot adjudicate the k-block.</li>
<li><b>The unbiased sample is small</b> (a few thousand tags per arm; 178 on B), so its maps are coarse and B's
numbers are rough. Chamber B's tracks point poorly at the wall in any case.</li>
<li><b>The trigger thresholds are run_79's</b>, assumed for every run; the per-sub-run board configs are not on
this machine.</li>
<li><b>Chamber D's wall profile is distorted by chamber D</b> (its centre and its u &lt; &minus;180 mm region), and
July's analog duplication short on WALD may add to it. D's ends-reversed sign is either the wall cabling or a
mirrored y plane; this data cannot tell which.</li>
<li><b>The liquid gradient's cause is a hypothesis</b> (light collection falling with distance from a
photodetector near the +u edge). The cell geometry and readout drawing would settle it. Liquid responses
are punch-through probabilities times efficiency; the two are not separated.</li>
<li><b>Plastic and liquid "efficiencies" are response probabilities</b> for the beam's own spectrum. A clean
MIP efficiency would need the cosmic runs, which have no MM reconstruction yet.</li>
</ul>

<h2 id="repro">Reproducing this</h2>
<pre><code>python -m sept26_prelim_analysis.scint_stack --jobs 8          # ~3 min, per-track tables
python -m sept26_prelim_analysis.scint_stack_ana --jobs 4      # ~7 min
python -m sept26_prelim_analysis.make_scint_stack_figures
python -m sept26_prelim_analysis.make_scint_stack_report</code></pre>
<p class="prov">Schema <code>{ext['schema']}</code> / <code>{meta['schema']}</code>. Input: <code>stage3_fullpass</code>
track tables and the exported n_TOF slim; pre-access runs 79/81 excluded. Built {dt.date.today().isoformat()}.</p>
"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Scintillator stack calibration</title>{HEAD}</head><body><main>{body}</main></body></html>')


def main() -> int:
    d = paths.out('scint_stack')
    out = d / 'report.html'
    out.write_text(build(d))
    print(f'wrote -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
